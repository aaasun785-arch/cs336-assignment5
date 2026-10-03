import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from cs336_alignment.SFT import SFT_dataset


def evaluate(model, dataloader, device):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids).logits

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

            total_loss += loss.item()
            num_batches += 1

    model.train()

    return total_loss / num_batches


def train(args):
    device = "cuda"

    # 1. tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path
    )

    # 2. datasets
    train_dataset = SFT_dataset(
        tokenizer=tokenizer,
        dataset_path=args.train_path,
        seq_length=args.seq_length,
        shuffle=True,
    )

    val_dataset = SFT_dataset(
        tokenizer=tokenizer,
        dataset_path=args.val_path,
        seq_length=args.seq_length,
        shuffle=False,
    )

    # 3. dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # 4. model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device)

    model.train()

    # 5. optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 一个 optimizer step 包含多个 microbatch
    num_update_steps = math.ceil(
        len(train_loader)
        / args.gradient_accumulation_steps
    )

    warmup_steps = int(
        args.warmup_ratio * num_update_steps
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    optimizer.zero_grad()

    optimizer_step = 0

    # 6. 一个 epoch
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # forward
        logits = model(input_ids).logits

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),#[B,T,V]->[B*T,V]
            labels.reshape(-1),
        )

        # gradient accumulation
        scaled_loss = (
            loss
            / args.gradient_accumulation_steps
        )

        scaled_loss.backward()

        if (
            (batch_idx + 1)
            % args.gradient_accumulation_steps
            == 0
        ):
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            optimizer_step += 1

            if optimizer_step % args.log_every == 0:
                print(
                    f"step={optimizer_step} "
                    f"train_loss={loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            if optimizer_step % args.eval_every == 0:
                val_loss = evaluate(
                    model,
                    val_loader,
                    device,
                )

                print(
                    f"step={optimizer_step} "
                    f"val_loss={val_loss:.4f}"
                )

    # 7. 如果最后剩余一些 microbatch
    if (
        len(train_loader)
        % args.gradient_accumulation_steps
        != 0
    ):
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_grad_norm,
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # 8. final validation
    val_loss = evaluate(
        model,
        val_loader,
        device,
    )

    print(f"final_val_loss={val_loss:.4f}")

    # 9. save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved model to {output_dir}")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=100,
    )
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    train(args)