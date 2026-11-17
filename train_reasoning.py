"""Reasoning & Chain-of-Thought (CoT) Fine-Tuning for llm-from-scratch.

Trains the model to think step by step on mathematical, logical, and scientific problems
using a blend of GSM8k, ARC-Challenge, and deductive instruction data.

Example:
    python train_reasoning.py --base_checkpoint out/chat/best.pt --out_dir out/reasoning --max_steps 2000
"""
import argparse
import contextlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import asdict
from typing import Dict, List

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


def build_reasoning_data() -> List[Dict[str, str]]:
    """Loads and formats a diverse blend of math, scientific, and logical reasoning pairs."""
    dataset = []

    # 1. GSM8k - Grade School Math with Step-by-Step Chain of Thought
    print("Loading GSM8k mathematical reasoning dataset...")
    try:
        gsm = load_dataset("openai/gsm8k", "main", split="train")
        for row in gsm:
            q = row.get("question", "").strip()
            raw_a = row.get("answer", "").strip()
            if not q or not raw_a:
                continue
            # Strip <<calc>> internal tags
            clean_a = re.sub(r"<<[^>]+>>", "", raw_a)
            parts = clean_a.split("####")
            steps = parts[0].strip()
            final_ans = parts[1].strip() if len(parts) > 1 else ""

            resp = f"Let's think step by step:\n{steps}\n\nTherefore, the final answer is: {final_ans}"
            dataset.append({"instruction": q, "input": "", "output": resp})
        print(f"  Added {len(gsm):,} GSM8k math reasoning examples.")
    except Exception as e:
        print(f"  Warning: could not load gsm8k ({e})")

    # 2. ARC - Science & Physical Reasoning
    print("Loading ARC Science Reasoning dataset...")
    try:
        arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        for row in arc:
            q = row.get("question", "").strip()
            choices = row.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            ans_key = row.get("answerKey", "").strip()
            if not q or not labels or not texts or not ans_key:
                continue

            options_str = "\n".join(f"{lbl}) {txt}" for lbl, txt in zip(labels, texts))
            instr = f"{q}\n\nOptions:\n{options_str}"

            # Find text of answer
            correct_text = ""
            for lbl, txt in zip(labels, texts):
                if lbl == ans_key:
                    correct_text = txt
                    break

            resp = f"Let's analyze the options step by step:\nLooking at the options, {ans_key}) '{correct_text}' accurately satisfies the scientific principle.\n\nTherefore, the correct answer is: {ans_key}) {correct_text}"
            dataset.append({"instruction": instr, "input": "", "output": resp})
        print(f"  Added {len(arc):,} ARC science reasoning examples.")
    except Exception as e:
        print(f"  Warning: could not load ai2_arc ({e})")

    # 3. Alpaca Cleaned - Logical, Analytical & General Instructions
    print("Loading curated logic & reasoning instructions from Alpaca...")
    try:
        alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
        # Filter for reasoning, analytical, math, or explanation keywords
        reasoning_keywords = ("why", "how", "explain", "calculate", "solve", "compare", "analyze", "reason", "prove", "difference")
        count = 0
        for row in alpaca:
            instr = row.get("instruction", "").strip().lower()
            if any(kw in instr for kw in reasoning_keywords) or count < 5000:
                dataset.append(row)
                count += 1
                if count >= 15000:
                    break
        print(f"  Added {count:,} curated logic & instruction examples.")
    except Exception as e:
        print(f"  Warning: could not load alpaca ({e})")

    random.seed(42)
    random.shuffle(dataset)
    print(f"Total unified reasoning dataset: {len(dataset):,} examples.")
    return dataset


class ReasoningDataset(Dataset):
    """Formats reasoning pairs with prompt-masked loss."""

    def __init__(self, data: List[Dict[str, str]], tokenizer, max_seq_len: int = 512):
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

            # Discard prompts that are too long to fit any answer
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

            # Mask out prompt tokens with -100
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
    parser = argparse.ArgumentParser(description="Fine-tune LLM for Chain-of-Thought reasoning.")
    parser.add_argument("--base_checkpoint", default="out/chat/best.pt", help="Path to starting checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer.json")
    parser.add_argument("--out_dir", default="out/reasoning")
    parser.add_argument("--max_steps", type=int, default=2000, help="Steps to train (~2 hours on RTX 4050)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--min_lr", type=float, default=1.0e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="auto")
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

    # 1. Setup Tokenizer
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        tok_path = os.path.join(os.path.dirname(args.base_checkpoint), "tokenizer.json")
    tokenizer = load_tokenizer(tok_path)
    shutil.copy(tok_path, os.path.join(args.out_dir, "tokenizer.json"))

    base_dir = os.path.dirname(args.base_checkpoint)
    info_src = os.path.join(base_dir, "data_info.json")
    if os.path.exists(info_src):
        shutil.copy(info_src, os.path.join(args.out_dir, "data_info.json"))

    # 2. Load Checkpoint
    ckpt_path = args.base_checkpoint
    if not os.path.exists(ckpt_path):
        if os.path.exists("out/chat/best.pt"):
            ckpt_path = "out/chat/best.pt"
        elif os.path.exists("out/base/best.pt"):
            ckpt_path = "out/base/best.pt"
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = load_checkpoint(ckpt_path, device=str(device))
    model_cfg = ModelConfig(**checkpoint["model_config"])
    model = GPT(model_cfg).to(device)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    print(f"Model loaded: {count_parameters(model):,} parameters (Step {checkpoint.get('step', '?')})")

    # 3. Build & Tokenize Reasoning Dataset
    raw_data = build_reasoning_data()
    split_idx = int(len(raw_data) * 0.95)
    train_data, val_data = raw_data[:split_idx], raw_data[split_idx:]

    print("Tokenizing training and validation sets...")
    train_dataset = ReasoningDataset(train_data, tokenizer, max_seq_len=model_cfg.block_size)
    val_dataset = ReasoningDataset(val_data, tokenizer, max_seq_len=model_cfg.block_size)
    print(f"Dataset ready: {len(train_dataset):,} train | {len(val_dataset):,} validation examples")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 4. Optimizer
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

    logger = CsvLogger(os.path.join(args.out_dir, "reasoning_log.csv"))

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

    print("\n--- Starting Reasoning (CoT) Training ---")
    start_step = 0
    best_val_loss = float("inf")

    if args.resume and os.path.exists(os.path.join(args.out_dir, "best.pt")):
        resume_ckpt = load_checkpoint(os.path.join(args.out_dir, "best.pt"), device=str(device))
        model.load_state_dict(strip_state_dict_prefixes(resume_ckpt["model"]))
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_step = resume_ckpt.get("step", 0)
        best_val_loss = resume_ckpt.get("val_loss", float("inf"))
        print(f"Resumed Reasoning training from Step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    train_iter = iter(train_loader)
    last_log_time = time.time()
    model.train()

    step = start_step
    while step < args.max_steps:
        lr = get_cosine_lr(step, args.learning_rate, args.min_lr, args.warmup_steps, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step > start_step and step % args.eval_interval == 0:
            val_loss = estimate_val_loss()
            print(f"Step {step}/{args.max_steps} | Reasoning Val Loss: {val_loss:.4f} | LR: {lr:.2e}")
            logger.log(step=step, val_loss=f"{val_loss:.4f}", lr=f"{lr:.6f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "val_loss": best_val_loss,
                    "model_config": asdict(model_cfg),
                    "reasoning": True,
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
            print(f"Step {step}/{args.max_steps} | Loss: {accum_loss:.4f} | LR: {lr:.2e} | ({elapsed:.1f}s / {args.log_interval} steps)")
            logger.log(step=step, train_loss=f"{accum_loss:.4f}", lr=f"{lr:.6f}")

        if step % args.save_interval == 0:
            save_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "val_loss": best_val_loss,
                "model_config": asdict(model_cfg),
                "reasoning": True,
            })

    # Final Save
    final_val_loss = estimate_val_loss()
    print(f"\nFinal Reasoning Validation Loss: {final_val_loss:.4f}")
    save_checkpoint(os.path.join(args.out_dir, "ckpt.pt"), {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.max_steps,
        "val_loss": final_val_loss,
        "model_config": asdict(model_cfg),
        "reasoning": True,
    })
    if final_val_loss < best_val_loss:
        save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": args.max_steps,
            "val_loss": final_val_loss,
            "model_config": asdict(model_cfg),
            "reasoning": True,
        })

    print(f"Reasoning Training Complete! Checkpoints saved to {args.out_dir}")

    if args.export_gguf:
        print(f"Exporting reasoning model to GGUF ({args.export_gguf})...")
        export_gguf(os.path.join(args.out_dir, "best.pt"), dtype=args.export_gguf,
                    out_path=os.path.join(args.out_dir, f"llm-reasoning-{args.export_gguf}.gguf"))
        print(f"GGUF exported: {os.path.join(args.out_dir, f'llm-reasoning-{args.export_gguf}.gguf')}")


if __name__ == "__main__":
    main()

