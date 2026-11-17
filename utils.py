"""Small helpers shared by the training, generation and export scripts."""
import csv
import math
import os
import random

import numpy as np
import torch

CHECKPOINT_PREFIXES = ("_orig_mod.", "module.")  # added by torch.compile / DataParallel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of unique parameters (tied weights are counted once)."""
    return sum(p.numel() for p in model.parameters())


def get_lr(step: int, learning_rate: float, min_lr: float,
           warmup_steps: int, max_steps: int) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (learning_rate - min_lr)


def strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Remove '_orig_mod.' / 'module.' prefixes from parameter names."""
    cleaned = {}
    for key, value in state_dict.items():
        changed = True
        while changed:
            changed = False
            for prefix in CHECKPOINT_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def save_checkpoint(path: str, payload: dict) -> None:
    """Save to a temporary file first so an interrupted save never corrupts the checkpoint."""
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: str, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


class CsvLogger:
    """Appends one row per call to a CSV file (creates the header if the file is new)."""

    FIELDS = ["step", "train_loss", "val_loss", "lr", "tokens_per_sec"]

    def __init__(self, path: str):
        self.path = path
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(self.FIELDS)

    def log(self, **row) -> None:
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(name, "") for name in self.FIELDS])


def plot_loss_curve(csv_path: str, png_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["train_loss"]:
                steps.append(int(row["step"]))
                train_loss.append(float(row["train_loss"]))
            if row["val_loss"]:
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["val_loss"]))

    plt.figure(figsize=(7, 4))
    plt.plot(steps, train_loss, label="train loss", alpha=0.7)
    plt.plot(val_steps, val_loss, label="val loss", marker="o")
    plt.xlabel("optimizer step")
    plt.ylabel("cross-entropy loss")
    plt.title("Training curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()


def gguf_field_value(reader, key: str):
    """Read a metadata value from a gguf.GGUFReader (returns None if missing)."""
    field = reader.fields.get(key)
    if field is None:
        return None
    return field.contents()
