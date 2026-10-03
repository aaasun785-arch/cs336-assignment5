import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# 改成训练完成后的 SFT 模型路径
MODEL_PATH = "/path/to/your/sft_model"

DATA_PATH = Path(
    "data/simple_safety_tests/simple_safety_tests.csv"
)

ALPACA_PROMPT_PATH = Path(
    "cs336_alignment/prompts_safety/alpaca_sft.prompt"
)


def load_sst():
    examples = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            examples.append({
                "prompts_final": row["prompts_final"],
            })

    return examples


def build_prompt(
    example,
    alpaca_template,
):
    return alpaca_template.format(
        instruction=example["prompts_final"],
        response="",
    )


@torch.inference_mode()
def evaluate(
    limit=None,
    batch_size=8,
):
    # 1. load SimpleSafetyTests
    examples = load_sst()

    if limit is not None:
        examples = examples[:limit]

    print(f"Loaded {len(examples)} examples")

    # 2. load Alpaca SFT prompt
    alpaca_template = ALPACA_PROMPT_PATH.read_text(
        encoding="utf-8"
    )

    prompts = [
        build_prompt(
            example,
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
            max_new_tokens=512,
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
            model_output = model_output.strip()

            results.append({
                "prompts_final": example["prompts_final"],
                "output": model_output,
            })

        print(f"{end}/{len(examples)}")

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    # 6. throughput
    total = len(results)
    throughput = total / elapsed

    print()
    print(f"Generation time: {elapsed:.2f}s")
    print(
        f"Throughput: "
        f"{throughput:.3f} examples/s"
    )

    # 7. save JSONL
    output_dir = Path(
        "results/sst_sft"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "results.jsonl"

    with open(
        output_path,
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

    print(f"Saved to {output_path}")


def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()

    evaluate(
        limit=args.limit,
        batch_size=args.batch_size,
    )