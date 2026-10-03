import argparse
import json
import time
import csv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from cs336_alignment.MMLU import parse_mmlu_response


MMLU_DIR = Path("data/mmlu/test")

MMLU_PROMPT_PATH = Path(
    "cs336_alignment/prompts_safety/mmlu_zero_shot.prompt"
)

ALPACA_PROMPT_PATH = Path(
    "cs336_alignment/prompts_safety/alpaca_sft.prompt"
)

# 改成你训练完成后的 SFT 模型目录
MODEL_PATH = ""

# 如果自己租服务器，可以直接这样
RESULTS_VOLUME_MOUNT_PATH = "results"


def build_task_prompt(
    template,
    subject,
    question,
    options,
):
    return template.format(
        subject=subject.replace("_", " "),
        question=question,
        options=options,
    )


def build_full_prompt(
    alpaca_template,
    instruction,
):
    return alpaca_template.format(
        instruction=instruction,
        response="",
    )


def build_prompt(
    example: dict,
    mmlu_template: str,
    alpaca_template: str,
):
    task_prompt = build_task_prompt(
        mmlu_template,
        example["subject"],
        example["question"],
        example["options"],
    )

    return build_full_prompt(
        alpaca_template,
        task_prompt,
    )


def load_mmlu() -> list[dict]:
    examples = []

    for file_path in sorted(
        MMLU_DIR.glob("*_test.csv")
    ):
        subject = (
            file_path.stem
            .removesuffix("_test")
            .replace("_", " ")
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            reader = csv.reader(f)

            for row in reader:
                examples.append({
                    "subject": subject,
                    "question": row[0],
                    "options": row[1:5],
                    "answer": row[5],
                })

    return examples


@torch.inference_mode()
def evaluate(
    limit=None,
    batch_size=8,
):
    # 1. load MMLU
    examples = load_mmlu()

    if limit is not None:
        examples = examples[:limit]

    print(f"Loaded {len(examples)} examples")

    # 2. load prompt templates
    mmlu_template = MMLU_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    alpaca_template = ALPACA_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    prompts = [
        build_prompt(
            example,
            mmlu_template,
            alpaca_template,
        )
        for example in examples
    ]

    # 3. load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. load SFT model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    model.eval()

    results = []

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    # 5. generation
    for start in range(
        0,
        len(examples),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(examples),
        )

        batch_examples = examples[start:end]
        batch_prompts = prompts[start:end]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        input_length = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_tokens = outputs[:, input_length:]

        generations = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        for example, model_output in zip(
            batch_examples,
            generations,
        ):
            # 保留这个也没问题，但 SFT 模型一般不会生成 # Query:
            if "# Query:" in model_output:
                model_output = model_output.split(
                    "# Query:",
                    1,
                )[0]

            model_output = model_output.strip()

            prediction = parse_mmlu_response(
                example,
                model_output,
            )

            results.append({
                **example,
                "model_output": model_output,
                "prediction": prediction,
                "correct": (
                    prediction == example["answer"]
                ),
            })

        print(f"{end}/{len(examples)}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # 6. metrics
    total = len(results)

    correct = sum(
        result["correct"]
        for result in results
    )

    failed_parses = sum(
        result["prediction"] is None
        for result in results
    )

    accuracy = correct / total
    throughput = total / elapsed

    print()
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Failed parses: {failed_parses}")
    print(f"Generation time: {elapsed:.2f}s")
    print(
        f"Throughput: "
        f"{throughput:.3f} examples/s"
    )

    # 7. print parse failures
    failures = [
        result
        for result in results
        if result["prediction"] is None
    ]

    for result in failures[:5]:
        print()
        print("Parse failure:")
        print(result["question"])
        print(result["model_output"])

    # 8. save results
    output_dir = (
        Path(RESULTS_VOLUME_MOUNT_PATH)
        / "mmlu_sft"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir / "results.jsonl",
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "failed_parses": failed_parses,
        "generation_seconds": elapsed,
        "examples_per_second": throughput,
    }

    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    evaluate(
        limit=args.limit,
        batch_size=args.batch_size,
    )