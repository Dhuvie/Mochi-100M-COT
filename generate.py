"""Generate text from a trained checkpoint.

Example:
    python generate.py --checkpoint out/tiny/ckpt.pt --prompt "The history of" --max_new_tokens 100
"""
import argparse
import json
import os

import torch

from config import ModelConfig
from model import GPT
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import load_checkpoint, strip_state_dict_prefixes


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with a trained model.")
    parser.add_argument("--checkpoint", default="out/base/ckpt.pt")
    parser.add_argument("--tokenizer", default=None, help="default: tokenizer.json next to the checkpoint")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8, help="0 means greedy")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))

    info_path = os.path.join(ckpt_dir, "data_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        print(f"Data cutoff date: {info.get('data_cutoff_date')} "
              f"(newest source snapshot: {info.get('newest_source_snapshot')})")

    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    config = ModelConfig(**checkpoint["model_config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    model.eval()

    tokenizer = load_tokenizer(args.tokenizer or os.path.join(ckpt_dir, "tokenizer.json"))
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    prompt_ids = tokenizer.encode(args.prompt).ids if args.prompt else [eot_id]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    output = model.generate(idx, args.max_new_tokens, temperature=args.temperature,
                            top_k=args.top_k, top_p=args.top_p, eos_token_id=eot_id)
    new_ids = [t for t in output[0].tolist() if t != eot_id]
    print(tokenizer.decode(new_ids))


if __name__ == "__main__":
    main()
