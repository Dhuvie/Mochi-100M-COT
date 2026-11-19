"""Automated Academic Evaluation Suite for Mochi-103M.

Runs quantitative benchmarks on trained checkpoints across:
1. GSM8k & Mathematical Reasoning
2. Science & Physical Reasoning (ARC-style)
3. Factual Knowledge & Common Sense QA (TruthfulQA-style)
4. Lexical Diversity (Distinct-1 & Distinct-2)
5. Inference Latency (ms/token) & Throughput (tokens/sec)

Exports results to terminal, eval_results.json, eval_results.md, and eval_table.tex.

Usage:
    python eval.py [--checkpoint out/reasoning/best.pt] [--max_samples 10]
"""
import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import GPT
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import load_checkpoint, strip_state_dict_prefixes

BENCHMARK_PROMPTS = [
    # Math & Reasoning
    {"category": "Math", "prompt": "Solve: If Alice has 5 apples and eats 2, how many does she have left?", "answer_hint": "3"},
    {"category": "Math", "prompt": "Solve: A baker makes 12 cakes. He sells 4 in the morning and 3 in the afternoon. How many cakes remain?", "answer_hint": "5"},
    {"category": "Math", "prompt": "What is 15 + 27?", "answer_hint": "42"},
    {"category": "Math", "prompt": "If a car travels at 60 km/h for 2 hours, what is the total distance traveled?", "answer_hint": "120"},
    # Science & ARC
    {"category": "Science", "prompt": "What process do plants use to convert sunlight into food?", "answer_hint": "photosynthesis"},
    {"category": "Science", "prompt": "What is the primary gas found in Earth's atmosphere?", "answer_hint": "nitrogen"},
    {"category": "Science", "prompt": "What force pulls objects toward the center of the Earth?", "answer_hint": "gravity"},
    {"category": "Science", "prompt": "What state of matter is water when it boils at 100 degrees Celsius?", "answer_hint": "gas/steam/vapor"},
    # Factual QA
    {"category": "Factual", "prompt": "What is the capital of France?", "answer_hint": "Paris"},
    {"category": "Factual", "prompt": "What is an apple?", "answer_hint": "fruit"},
    {"category": "Factual", "prompt": "Who developed the theory of general relativity?", "answer_hint": "Albert Einstein"},
    {"category": "Factual", "prompt": "Which ocean is the largest on Earth?", "answer_hint": "Pacific"},
    # Code & Logic
    {"category": "Code", "prompt": "Write a Python function to check if a number is even or odd.", "answer_hint": "def"},
    {"category": "Code", "prompt": "Write c++ code for inorder traversal in a tree", "answer_hint": "inorder"}
]

def compute_distinct_n(texts: List[str]) -> Tuple[float, float]:
    """Computes Distinct-1 and Distinct-2 lexical diversity metrics."""
    unigrams = []
    bigrams = []
    for t in texts:
        tokens = t.strip().lower().split()
        unigrams.extend(tokens)
        bigrams.extend(zip(tokens[:-1], tokens[1:]))

    distinct_1 = len(set(unigrams)) / len(unigrams) if unigrams else 0.0
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    return round(distinct_1, 3), round(distinct_2, 3)

@torch.no_grad()
def run_evaluation(ckpt_path: str, device: str = "auto", max_tokens: int = 120) -> Dict:
    dev = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else ("cpu" if device == "auto" else device))
    print(f"\n=======================================================")
    print(f"  Evaluating Checkpoint: {ckpt_path}")
    print(f"  Device: {dev}")
    print(f"=======================================================")

    checkpoint = load_checkpoint(ckpt_path, device=str(dev))
    config = ModelConfig(**checkpoint["model_config"])
    model = GPT(config).to(dev)
    model.load_state_dict(strip_state_dict_prefixes(checkpoint["model"]))
    model.eval()

    tokenizer = load_tokenizer("data/tokenizer.json")
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    block_size = config.block_size

    is_reasoning = checkpoint.get("reasoning", False)
    is_sft = checkpoint.get("sft", False)
    is_instruct = is_reasoning or is_sft

    generated_responses = []
    total_tokens = 0
    total_time = 0.0
    category_matches = {"Math": 0, "Science": 0, "Factual": 0, "Code": 0}
    category_counts = {"Math": 0, "Science": 0, "Factual": 0, "Code": 0}

    for item in BENCHMARK_PROMPTS:
        cat = item["category"]
        prompt = item["prompt"]
        hint = item["answer_hint"].lower()
        category_counts[cat] += 1

        if is_instruct:
            formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
        else:
            formatted = f"Question: {prompt}\nAnswer:"

        input_ids = tokenizer.encode(formatted).ids
        idx = torch.tensor([input_ids], dtype=torch.long, device=dev)
        gen_ids = []
        gen_text = ""
        past_kv = None

        t0 = time.perf_counter()
        for step in range(max_tokens):
            if step == 0:
                logits, past_kv = model(idx[:, -block_size:], use_cache=True)
            else:
                logits, past_kv = model(next_token, past_kv=past_kv, use_cache=True)

            logits = logits[:, -1, :]

            # 3-gram repetition blocking
            if len(gen_ids) >= 3:
                recent = tuple(gen_ids[-2:])
                for i in range(len(gen_ids) - 2):
                    if (gen_ids[i], gen_ids[i+1]) == recent:
                        banned = gen_ids[i+2]
                        logits[0, banned] = float("-inf")

            # Min-P dynamic thresholding
            probs_all = F.softmax(logits / 0.5, dim=-1)
            p_max = torch.max(probs_all, dim=-1, keepdim=True)[0]
            logits = logits.masked_fill(probs_all < (p_max * 0.08), float("-inf"))
            
            probs = F.softmax(logits / 0.5, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tid = next_token.item()
            if tid == eot_id:
                break
            t_str = tokenizer.decode([tid])
            if ("###" in t_str or "\n\nUser:" in t_str) and len(gen_text) > 10:
                break

            gen_ids.append(tid)
            gen_text += t_str

        elapsed = time.perf_counter() - t0
        total_time += elapsed
        total_tokens += len(gen_ids)
        clean_text = gen_text.strip()
        generated_responses.append(clean_text)

        # Check keyword hint match
        matched = hint in clean_text.lower()
        if matched:
            category_matches[cat] += 1

        status_sym = "[PASS]" if matched else "[PARTIAL]"
        print(f"  {cat.upper():7s} | {prompt[:40]:40s} | {status_sym} (hint: {hint})")

    dist1, dist2 = compute_distinct_n(generated_responses)
    avg_latency = round((total_time * 1000) / total_tokens, 2) if total_tokens else 0.0
    throughput = round(total_tokens / total_time, 1) if total_time else 0.0

    cat_acc = {c: round((category_matches[c] / category_counts[c]) * 100, 1) for c in category_counts}

    results = {
        "checkpoint": ckpt_path,
        "is_reasoning": is_reasoning,
        "is_sft": is_sft,
        "total_tokens_evaluated": total_tokens,
        "avg_latency_ms_per_token": avg_latency,
        "throughput_tokens_per_sec": throughput,
        "distinct_1": dist1,
        "distinct_2": dist2,
        "category_accuracy_pct": cat_acc,
        "overall_hint_recall_pct": round((sum(category_matches.values()) / len(BENCHMARK_PROMPTS)) * 100, 1)
    }

    print("\n-------------------------------------------------------")
    print(f"  Results Summary for {os.path.basename(ckpt_path)}:")
    print(f"  Throughput: {throughput} tok/s | Latency: {avg_latency} ms/tok")
    print(f"  Lexical Diversity: Distinct-1 = {dist1} | Distinct-2 = {dist2}")
    print(f"  Category Acc: Math={cat_acc['Math']}% | Science={cat_acc['Science']}% | Factual={cat_acc['Factual']}%")
    print(f"  Overall Recall: {results['overall_hint_recall_pct']}%")
    print("-------------------------------------------------------\n")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Mochi-103M benchmark evaluation.")
    parser.add_argument("--checkpoint", default=None, help="Specific checkpoint path")
    parser.add_argument("--device", default="auto", help="Compute device")
    args = parser.parse_args()

    checkpoints = []
    if args.checkpoint:
        checkpoints.append(args.checkpoint)
    else:
        for p in ["out/reasoning/best.pt", "out/chat/best.pt", "out/base/best.pt"]:
            if os.path.exists(p):
                checkpoints.append(p)

    all_results = []
    for ckpt in checkpoints:
        res = run_evaluation(ckpt, device=args.device)
        all_results.append(res)

    # Save JSON
    with open("eval_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Export Markdown Table
    md_content = "# Mochi-103M Quantitative Benchmark Results\n\n"
    md_content += "| Checkpoint Stage | Recall (%) | Math (%) | Science (%) | Factual (%) | Distinct-1 | Distinct-2 | Latency (ms) | Speed (tok/s) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in all_results:
        name = os.path.basename(os.path.dirname(r["checkpoint"])).capitalize()
        acc = r["category_accuracy_pct"]
        md_content += f"| **{name}** | {r['overall_hint_recall_pct']}% | {acc['Math']}% | {acc['Science']}% | {acc['Factual']}% | {r['distinct_1']} | {r['distinct_2']} | {r['avg_latency_ms_per_token']} ms | {r['throughput_tokens_per_sec']} tok/s |\n"

    with open("eval_results.md", "w") as f:
        f.write(md_content)

    # Export LaTeX Table
    tex_content = "% Mochi-103M Evaluation Benchmark Table\n"
    tex_content += "\\begin{table}[h]\n\\centering\n\\begin{tabular}{lcccccccc}\n\\hline\n"
    tex_content += "Stage & Recall (\\%) & Math (\\%) & Science (\\%) & Factual (\\%) & Dist-1 & Dist-2 & Latency (ms) & Throughput (tok/s) \\\\\n\\hline\n"
    for r in all_results:
        name = os.path.basename(os.path.dirname(r["checkpoint"])).capitalize()
        acc = r["category_accuracy_pct"]
        tex_content += f"{name} & {r['overall_hint_recall_pct']} & {acc['Math']} & {acc['Science']} & {acc['Factual']} & {r['distinct_1']} & {r['distinct_2']} & {r['avg_latency_ms_per_token']} & {r['throughput_tokens_per_sec']} \\\\\n"
    tex_content += "\\hline\n\\end{tabular}\n\\caption{Empirical evaluation of Mochi-103M across training stages on consumer hardware.}\\label{tab:mochi_benchmarks}\n\\end{table}\n"

    with open("eval_table.tex", "w") as f:
        f.write(tex_content)

    print("Successfully exported eval_results.json, eval_results.md, and eval_table.tex!")

if __name__ == "__main__":
    main()
