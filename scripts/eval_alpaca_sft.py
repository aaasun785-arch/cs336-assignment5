import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DATA_PATH = Path(
    "data/alpaca_eval/alpaca_eval_gpt4_turbo.json"
)

ALPACA_PROMPT_PATH = Path(
    "cs336_alignment/prompts_safety/alpaca_sft.prompt"
)

# 改成你训练完成后的 SFT 模型路径
MODEL_PATH = "/path/to/your/sft_model"

# 自己租服务器时可以直接设为 results
RESULTS_VOLUME_MOUNT_PATH = "results"


def build_prompt(
    example: dict,
    alpaca_template: str,
):
    return alpaca_template.format(
        instruction=example["instruction"],
        response="",
    )


def load_alpaca() -> list[dict]:
    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    examples = []

    for example in data:
        examples.append({
            "instruction": example["instruction"],
            "dataset": example["dataset"],
        })

    return examples


@torch.inference_mode()
def evaluate(
    limit=None,
    batch_size=8,
):
    # 1. load AlpacaEval
    examples = load_alpaca()

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

    # 5. generate
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
                "instruction": example["instruction"],
                "output": model_output,
                "generator": "llama-3.1-8b-sft",
                "dataset": example["dataset"],
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

    # 7. save as JSON array
    output_dir = (
        Path(RESULTS_VOLUME_MOUNT_PATH)
        / "alpaca_sft"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "results.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved to {output_path}")


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