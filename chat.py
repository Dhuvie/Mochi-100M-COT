"""Interactive Terminal Chat interface for the fine-tuned LLM assistant.

Allows conversational back-and-forth like ChatGPT / Bard directly in your terminal.

Example:
    python chat.py --checkpoint out/chat/best.pt
    python chat.py --prompt "What is quantum computing?"
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import GPT
from rag_engine import LocalRAGRetriever
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import load_checkpoint, strip_state_dict_prefixes


def parse_args():
    parser = argparse.ArgumentParser(description="Chat interactively with your trained LLM.")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (e.g. out/reasoning/best.pt, out/chat/best.pt, or out/base/best.pt)")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = deterministic/accurate, 0.5 = creative)")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.85, help="Top-p nucleus sampling")
    parser.add_argument("--min_p", type=float, default=0.08, help="Min-p dynamic sampling threshold (relative to top token)")
    parser.add_argument("--repetition_penalty", type=float, default=1.15, help="Penalty for repeating tokens (> 1.0 reduces repetition)")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="Block repeating n-grams (e.g. 3 prevents cyclic loops)")
    parser.add_argument("--use_cache", action="store_true", default=True, help="Enable Key-Value (KV) cache for fast inference")
    parser.add_argument("--rag", action="store_true", default=True, help="Enable local factual retrieval (RAG) grounding")
    parser.add_argument("--no_rag", action="store_false", dest="rag", help="Disable RAG knowledge retrieval")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max response length")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt", default=None, help="Single query mode (non-interactive)")
    return parser.parse_args()


@torch.no_grad()
def stream_response(model, tokenizer, prompt_text: str, device: torch.device,
                    max_new_tokens: int = 256, temperature: float = 0.5,
                    top_k: int = 25, top_p: float = 0.85, min_p: float = 0.08,
                    repetition_penalty: float = 1.25, no_repeat_ngram_size: int = 3,
                    use_cache: bool = True) -> str:
    """Streams tokens one by one to stdout as they are generated with KV caching and Min-P sampling."""
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    input_ids = tokenizer.encode(prompt_text).ids
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    block_size = model.config.block_size

    generated_ids = []
    generated_text = ""
    past_kv = None

    for step in range(max_new_tokens):
        if use_cache:
            if step == 0:
                logits, past_kv = model(idx[:, -block_size:], use_cache=True)
            else:
                logits, past_kv = model(next_token, past_kv=past_kv, use_cache=True)
        else:
            context = idx[:, -block_size:]
            logits, _ = model(context)

        logits = logits[:, -1, :]

        # 1. N-gram repetition blocking (completely prevents infinite cyclic loops)
        if no_repeat_ngram_size > 0 and len(generated_ids) >= no_repeat_ngram_size:
            ngram_prefix = tuple(generated_ids[-(no_repeat_ngram_size - 1):])
            for i in range(len(generated_ids) - no_repeat_ngram_size + 1):
                if tuple(generated_ids[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned_token = generated_ids[i + no_repeat_ngram_size - 1]
                    logits[0, banned_token] = float("-inf")

        # 2. Frequency-scaled exponential repetition penalty (only penalize actual repeats count >= 2)
        if repetition_penalty != 1.0 and generated_ids:
            counts = {}
            for tid in generated_ids:
                counts[tid] = counts.get(tid, 0) + 1
            for tid, count in counts.items():
                if count >= 2:
                    penalty = repetition_penalty ** min(count - 1, 3)
                    if logits[0, tid] > 0:
                        logits[0, tid] /= penalty
                    else:
                        logits[0, tid] *= penalty

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                kth_val = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1, None]
                logits = logits.masked_fill(logits < kth_val, float("-inf"))
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
            if min_p is not None and min_p > 0.0:
                probs_all = F.softmax(logits, dim=-1)
                p_max = torch.max(probs_all, dim=-1, keepdim=True)[0]
                p_thresh = p_max * min_p
                logits = logits.masked_fill(probs_all < p_thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id == eot_id:
            break

        token_str = tokenizer.decode([token_id])

        # Stop conditions: section markers or multi-turn leaks
        if ("###" in token_str or "\n\nUser:" in token_str) and len(generated_text) > 10:
            break

        sys.stdout.write(token_str)
        sys.stdout.flush()

        generated_ids.append(token_id)
        generated_text += token_str
        idx = torch.cat([idx, next_token], dim=1)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return generated_text.strip()


def main():
    args = parse_args()

    # Resolve checkpoint: user-specified, or priority search (reasoning -> chat -> base)
    ckpt_path = args.checkpoint
    if not ckpt_path or not os.path.exists(ckpt_path):
        if os.path.exists("out/reasoning/best.pt"):
            ckpt_path = "out/reasoning/best.pt"
        elif os.path.exists("out/chat/best.pt"):
            ckpt_path = "out/chat/best.pt"
        elif os.path.exists("out/base/best.pt"):
            ckpt_path = "out/base/best.pt"
        elif os.path.exists("out/base/ckpt.pt"):
            ckpt_path = "out/base/ckpt.pt"
        else:
            raise FileNotFoundError(f"Checkpoint not found at: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "auto" else
                          ("cpu" if args.device == "auto" else args.device))

    # Tokenizer
    tok_dir = os.path.dirname(os.path.abspath(ckpt_path))
    tok_path = args.tokenizer or os.path.join(tok_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        tok_path = "data/tokenizer.json"
    tokenizer = load_tokenizer(tok_path)

    # Model
    checkpoint = load_checkpoint(ckpt_path, device=str(device))
    config = ModelConfig(**checkpoint["model_config"])
    model = GPT(config).to(device)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    model.eval()

    is_reasoning = checkpoint.get("reasoning", False)
    is_sft = checkpoint.get("sft", False)
    is_instruct = is_reasoning or is_sft
    if is_reasoning:
        model_type = "Mochi-Reason (Chain-of-Thought / CoT)"
    elif is_sft:
        model_type = "Mochi-Chat (Conversational Assistant)"
    else:
        model_type = "Mochi-Base (Foundational Pretrained Model)"

    print("=" * 65)
    print(f"  Mochi-103M Terminal Interface ({model_type})")
    print(f"  Checkpoint: {ckpt_path} | Device: {device}")
    print("  Commands: 'exit' or 'quit' to end | 'clear' to reset")
    print("=" * 65 + "\n")

    retriever = LocalRAGRetriever() if args.rag else None

    # Single-prompt mode
    if args.prompt:
        if is_instruct:
            if retriever:
                matched_doc = retriever.query(args.prompt)
                if matched_doc:
                    print(f"You: {args.prompt}")
                    print("Assistant: ", end="")
                    # Stream the verified factual answer smoothly
                    for word in matched_doc["content"].split(" "):
                        sys.stdout.write(word + " ")
                        sys.stdout.flush()
                        time.sleep(0.015)
                    sys.stdout.write("\n")
                    return
            formatted = f"### Instruction:\n{args.prompt.strip()}\n\n### Response:\n"
        else:
            formatted = args.prompt.strip() + " "
        print(f"You: {args.prompt}")
        print("Assistant: ", end="")
        stream_response(model, tokenizer, formatted, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        min_p=args.min_p,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        use_cache=args.use_cache)
        return

    # Interactive mode
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            print("[History cleared]\n")
            continue

        if is_instruct:
            if retriever:
                matched_doc = retriever.query(user_input)
                if matched_doc:
                    print("Assistant: ", end="")
                    for word in matched_doc["content"].split(" "):
                        sys.stdout.write(word + " ")
                        sys.stdout.flush()
                        time.sleep(0.015)
                    sys.stdout.write("\n\n")
                    continue
            formatted_prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        else:
            # Base model completion prompt
            formatted_prompt = f"Question: {user_input}\nAnswer:"

        print("Assistant: ", end="")
        stream_response(model, tokenizer, formatted_prompt, device,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_k=args.top_k, top_p=args.top_p,
                        min_p=args.min_p,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                        use_cache=args.use_cache)
        print()


if __name__ == "__main__":
    main()

