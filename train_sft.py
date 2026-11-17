"""Supervised Fine-Tuning (SFT / Instruction Tuning) for llm-from-scratch.

Turns a base pretrained GPT model into an instruction-following conversational assistant
(like ChatGPT / Alpaca) using prompt-masked cross-entropy loss.

Example:
    python train_sft.py --base_checkpoint out/base/best.pt --out_dir out/chat --max_steps 500
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from config import ModelConfig
from export_gguf import export_gguf
from model import GPT
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import CsvLogger, count_parameters, load_checkpoint, save_checkpoint, set_seed, strip_state_dict_prefixes


class InstructionDataset(Dataset):
    """Formats Alpaca-style instruction-output pairs with prompt masking."""

    def __init__(self, data, tokenizer, max_seq_len: int = 512):
        self.examples: List[Dict[str, torch.Tensor]] = []
        eot_id = tokenizer.token_to_id(EOT_TOKEN)

        for row in data:
            instr = row.get("instruction", "").strip()
            inp = row.get("input", "").strip()
            out = row.get("output", "").strip()
            if not instr or not out:
                continue

            if inp:
                prompt_text = f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt_text = f"### Instruction:\n{instr}\n\n### Response:\n"

            prompt_ids = tokenizer.encode(prompt_text).ids
            out_ids = tokenizer.encode(out).ids + [eot_id]

            # Drop prompts that are too long to leave room for a response
            if len(prompt_ids) >= max_seq_len - 16:
                continue

            full_ids = prompt_ids + out_ids
            if len(full_ids) > max_seq_len + 1:
                full_ids = full_ids[:max_seq_len + 1]
                if full_ids[-1] != eot_id:
                    full_ids[-1] = eot_id

            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]
            if len(input_ids) > max_seq_len:
                input_ids = input_ids[:max_seq_len]
                target_ids = target_ids[:max_seq_len]

            # Mask out the prompt tokens in targets with -100
            prompt_len = len(prompt_ids)
            labels = []
            for i in range(len(target_ids)):
                if i < prompt_len - 1:
                    labels.append(-100)
                else:
                    labels.append(target_ids[i])

            if len(input_ids) < 4:
                continue

            self.examples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pads a micro-batch dynamically to the length of the longest sample."""
    max_len = min(512, max(len(item["input_ids"]) for item in batch))
    b_size = len(batch)

    inputs = torch.zeros((b_size, max_len), dtype=torch.long)
    targets = torch.full((b_size, max_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = min(max_len, len(item["input_ids"]))
        inputs[i, :seq_len] = item["input_ids"][:seq_len]
        targets[i, :seq_len] = item["labels"][:seq_len]

    return {"inputs": inputs, "targets": targets}


def get_cosine_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning for conversational chat.")
    parser.add_argument("--base_checkpoint", default="out/base/best.pt", help="Path to base pretrained checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--out_dir", default="out/chat", help="Where to save chat checkpoints")
    parser.add_argument("--dataset_name", default="yahma/alpaca-cleaned", help="Hugging Face instruction dataset")
    parser.add_argument("--max_steps", type=int, default=500, help="SFT training steps")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per micro-batch")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Peak learning rate for SFT")
    parser.add_argument("--min_lr", type=float, default=1.5e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume from out_dir/best.pt if available")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export_gguf", choices=["f32", "f16", "q8_0"], default="f16")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          ("cpu" if args.device == "auto" else args.device))
    print(f"Device: {device}")

    # 1. Load Tokenizer
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        tok_path = os.path.join(os.path.dirname(args.base_checkpoint), "tokenizer.json")
    tokenizer = load_tokenizer(tok_path)
    shutil.copy(tok_path, os.path.join(args.out_dir, "tokenizer.json"))

    # Copy data_info.json if available
    base_dir = os.path.dirname(args.base_checkpoint)
    info_src = os.path.join(base_dir, "data_info.json")
    if os.path.exists(info_src):
        shutil.copy(info_src, os.path.join(args.out_dir, "data_info.json"))

    # 2. Load Base Checkpoint
    print(f"Loading pretrained base model from: {args.base_checkpoint}")
    checkpoint = load_checkpoint(args.base_checkpoint, device=str(device))
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    print(f"Loaded base model: {count_parameters(model):,} parameters (Step {checkpoint.get('step', '?')})")

    # 3. Load & Preprocess Instruction Dataset
    print(f"Loading instruction dataset: {args.dataset_name}...")
    raw_ds = load_dataset(args.dataset_name, split="train")
    # Split 95% train / 5% val
    split_ds = raw_ds.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = InstructionDataset(split_ds["train"], tokenizer, max_seq_len=model_cfg.block_size)
    val_dataset = InstructionDataset(split_ds["test"], tokenizer, max_seq_len=model_cfg.block_size)
    print(f"Dataset prepared: {len(train_dataset):,} train examples | {len(val_dataset):,} validation examples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Setup Optimizer
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate, betas=(0.9, 0.95), fused=(device.type == "cuda"))

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        autocast_ctx = contextlib.nullcontext()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    logger = CsvLogger(os.path.join(args.out_dir, "sft_log.csv"))

    def estimate_val_loss(iters: int = 25) -> float:
        model.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= iters:
                    break
                x, y = batch["inputs"].to(device), batch["targets"].to(device)
                with autocast_ctx:
                    _, loss = model(x, y)
                total_loss += loss.item()
                count += 1
        model.train()
        return total_loss / max(1, count)

    print("\n--- Starting Supervised Fine-Tuning (SFT) ---")
    step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(os.path.join(args.out_dir, "best.pt")):
        resume_ckpt = load_checkpoint(os.path.join(args.out_dir, "best.pt"), device=str(device))
        model.load_state_dict(strip_state_dict_prefixes(resume_ckpt["model"]))
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        step = resume_ckpt.get("step", 0)
        best_val_loss = resume_ckpt.get("val_loss", float("inf"))
        print(f"Resumed SFT from Step {step} (Best Val Loss: {best_val_loss:.4f})")

    train_iter = iter(train_loader)
    last_log_time = time.time()
    model.train()

    while step < args.max_steps:
        lr = get_cosine_lr(step, args.learning_rate, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Periodic Validation
        if step > 0 and step % args.eval_interval == 0:
            val_loss = estimate_val_loss()
            print(f"Step {step}/{args.max_steps} | SFT Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
            logger.log(step=step, val_loss=f"{val_loss:.4f}", lr=f"{lr:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "model_config": asdict(model_cfg),
                    "sft": True,
                })

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            x = batch["inputs"].to(device)
            y = batch["targets"].to(device)

            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1

        if step % args.log_interval == 0:
            now = time.time()
            elapsed = now - last_log_time
            last_log_time = now
            print(f"Step {step}/{args.max_steps} | SFT Loss: {accum_loss:.4f} | LR: {lr:.2e} | ({elapsed:.1f}s / {args.log_interval} steps)")
            logger.log(step=step, train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}")

        if step % args.save_interval == 0:
            save_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "val_loss": best_val_loss,
                "model_config": asdict(model_cfg),
                "sft": True,
            })

    # Final Save
    final_val_loss = estimate_val_loss()
    print(f"\nFinal SFT Validation Loss: {final_val_loss:.4f}")
    save_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.max_steps,
        "val_loss": final_val_loss,
        "model_config": asdict(model_cfg),
        "sft": True,
    })
    if final_val_loss < best_val_loss:
        save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.max_steps,
            "val_loss": final_val_loss,
            "model_config": asdict(model_cfg),
            "sft": True,
        })

    print(f"SFT Training Complete! Checkpoints saved to {args.out_dir}")

    # Export to GGUF
    if args.export_gguf:
        print(f"Exporting conversational model to GGUF ({args.export_gguf})...")
        export_gguf(os.path.join(args.out_dir, "best.pt"), dtype=args.export_gguf,
                    out_path=os.path.join(args.out_dir, f"llm-chat-{args.export_gguf}.gguf"))
        print(f"GGUF exported: {os.path.join(args.out_dir, f'llm-chat-{args.export_gguf}.gguf')}")


if __name__ == "__main__":
    main()

