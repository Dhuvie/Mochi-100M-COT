"""Train the GPT model on the tokenized data.

Example:
    python train.py --preset tiny --data_dir data --out_dir out/tiny
    python train.py --preset base --data_dir data --out_dir out/base --resume
"""
import argparse
import contextlib
import json
import os
import shutil
import time
from dataclasses import asdict

import numpy as np
import torch

from config import get_preset, with_overrides
from model import GPT
from utils import (CsvLogger, count_parameters, get_lr, load_checkpoint, plot_loss_curve,
                   save_checkpoint, set_seed, strip_state_dict_prefixes)


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    """Sample random windows; targets are the inputs shifted by one token."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()
    x = torch.stack([torch.from_numpy(data[s:s + block_size].astype(np.int64)) for s in starts])
    y = torch.stack([torch.from_numpy(data[s + 1:s + 1 + block_size].astype(np.int64)) for s in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, datasets, batch_size, block_size, device, eval_iters, autocast_ctx):
    model.eval()
    losses = {}
    for split, data in datasets.items():
        values = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
            values[i] = loss.item()
        losses[split] = values.mean().item()
    model.train()
    return losses


def build_optimizer(model, train_cfg, device):
    """AdamW with weight decay on matrices only (not biases or LayerNorm)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": train_cfg.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=train_cfg.learning_rate,
                             betas=(train_cfg.beta1, train_cfg.beta2),
                             fused=(device.type == "cuda"))


def parse_args():
    parser = argparse.ArgumentParser(description="Train the GPT model.")
    parser.add_argument("--preset", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--out_dir", default=None, help="default: out/<preset>")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="continue from out_dir/ckpt.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--export_gguf", choices=["f32", "f16", "q8_0"], default=None,
                        help="convert the final checkpoint to GGUF when training ends")
    return parser.parse_args()


def main():
    args = parse_args()
    model_cfg, train_cfg = get_preset(args.preset)
    train_cfg = with_overrides(train_cfg, max_steps=args.max_steps, batch_size=args.batch_size,
                               grad_accum_steps=args.grad_accum_steps, learning_rate=args.learning_rate)
    out_dir = args.out_dir or os.path.join("out", args.preset)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, "data_info.json")) as f:
        data_info = json.load(f)
    model_cfg = with_overrides(model_cfg, vocab_size=data_info["vocab_size"])

    # keep the tokenizer and data description next to the checkpoints
    shutil.copy(os.path.join(args.data_dir, "tokenizer.json"), os.path.join(out_dir, "tokenizer.json"))
    shutil.copy(os.path.join(args.data_dir, "data_info.json"), os.path.join(out_dir, "data_info.json"))

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          ("cpu" if args.device == "auto" else args.device))
    set_seed(train_cfg.seed)

    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        autocast_ctx = contextlib.nullcontext()
        scaler = torch.amp.GradScaler("cpu", enabled=False)

    train_data = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    if len(val_data) <= model_cfg.block_size + 1:
        raise SystemExit("Validation split is smaller than one block; collect more data.")
    datasets = {"train": train_data, "val": val_data}

    model = GPT(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg, device)
    print(f"device: {device} | parameters: {count_parameters(model):,} "
          f"| train tokens: {len(train_data):,} | val tokens: {len(val_data):,}")

    start_step, best_val_loss = 0, float("inf")
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = load_checkpoint(ckpt_path, device=str(device))
        model.load_state_dict(strip_state_dict_prefixes(ckpt["model"]))
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val_loss = ckpt["step"], ckpt["best_val_loss"]
        print(f"resumed from step {start_step}")
    elif args.resume:
        print("no checkpoint found, starting from scratch")

    logger = CsvLogger(os.path.join(out_dir, "log.csv"))
    tokens_per_step = train_cfg.batch_size * train_cfg.grad_accum_steps * model_cfg.block_size

    def save(step: int, name: str) -> None:
        save_checkpoint(os.path.join(out_dir, name), {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
        })

    model.train()
    last_log_time = time.time()
    for step in range(start_step, train_cfg.max_steps):
        lr = get_lr(step, train_cfg.learning_rate, train_cfg.min_lr,
                    train_cfg.warmup_steps, train_cfg.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step > start_step and step % train_cfg.eval_interval == 0:
            losses = estimate_loss(model, datasets, train_cfg.batch_size, model_cfg.block_size,
                                   device, train_cfg.eval_iters, autocast_ctx)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            logger.log(step=step, train_loss=f"{losses['train']:.4f}", val_loss=f"{losses['val']:.4f}", lr=f"{lr:.6f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save(step, "best.pt")

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(train_cfg.grad_accum_steps):
            x, y = get_batch(train_data, train_cfg.batch_size, model_cfg.block_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / train_cfg.grad_accum_steps
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if (step + 1) % train_cfg.log_interval == 0:
            now = time.time()
            tokens_per_sec = tokens_per_step * train_cfg.log_interval / (now - last_log_time)
            last_log_time = now
            print(f"step {step + 1}/{train_cfg.max_steps} | loss {step_loss:.4f} | lr {lr:.2e} "
                  f"| {tokens_per_sec:,.0f} tokens/s")
            logger.log(step=step + 1, train_loss=f"{step_loss:.4f}", lr=f"{lr:.6f}",
                       tokens_per_sec=f"{tokens_per_sec:.0f}")
        if (step + 1) % train_cfg.checkpoint_interval == 0:
            save(step + 1, "ckpt.pt")

    losses = estimate_loss(model, datasets, train_cfg.batch_size, model_cfg.block_size,
                           device, train_cfg.eval_iters, autocast_ctx)
    print(f"final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    logger.log(step=train_cfg.max_steps, train_loss=f"{losses['train']:.4f}",
               val_loss=f"{losses['val']:.4f}", lr=f"{lr:.6f}")
    if losses["val"] < best_val_loss:
        best_val_loss = losses["val"]
        save(train_cfg.max_steps, "best.pt")
    save(train_cfg.max_steps, "ckpt.pt")
    plot_loss_curve(os.path.join(out_dir, "log.csv"), os.path.join(out_dir, "loss_curve.png"))
    print(f"saved checkpoints and loss curve to {out_dir}")

    if args.export_gguf:
        from export_gguf import export_gguf
        export_gguf(os.path.join(out_dir, "ckpt.pt"), dtype=args.export_gguf)


if __name__ == "__main__":
    main()
