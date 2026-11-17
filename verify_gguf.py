"""Check that a GGUF file matches the PyTorch checkpoint it was exported from.

1. Every tensor in the GGUF file is compared with the checkpoint
   (exact for f32, small tolerance for f16 and q8_0).
2. The metadata and tokenizer sizes are checked against the model config.
3. If llama-cpp-python is installed, the same prompt is run through llama.cpp and
   PyTorch: tokenization, next-token logits and greedy generation are compared.

Example:
    python verify_gguf.py --checkpoint out/base/ckpt.pt --gguf out/base/llm-from-scratch-f16.gguf
"""
import argparse
import os

import gguf
import numpy as np
import torch

from config import ModelConfig
from export_gguf import ARCH, expected_tensors
from model import GPT
from tokenizer import EOT_TOKEN, load_tokenizer
from utils import gguf_field_value, load_checkpoint, strip_state_dict_prefixes

TEST_PROMPTS = ["The", "In the beginning, the", "Python is a programming language that"]


def compare_tensors(reader: gguf.GGUFReader, state: dict, tensor_map: dict) -> int:
    """Return the number of mismatching tensors."""
    by_name = {t.name: t for t in reader.tensors}
    failures = 0
    for gguf_name, (torch_name, shape) in tensor_map.items():
        if gguf_name not in by_name:
            print(f"  MISSING  {gguf_name}")
            failures += 1
            continue
        tensor = by_name[gguf_name]
        original = state[torch_name].detach().float().cpu().numpy()
        qtype = tensor.tensor_type
        if qtype == gguf.GGMLQuantizationType.F32:
            stored, ok = tensor.data, None
            ok = np.array_equal(stored.reshape(shape), original)
        elif qtype == gguf.GGMLQuantizationType.F16:
            stored = tensor.data.astype(np.float32).reshape(shape)
            ok = np.allclose(stored, original, rtol=1e-3, atol=1e-4)
        elif qtype == gguf.GGMLQuantizationType.Q8_0:
            stored = gguf.quants.dequantize(tensor.data, qtype).reshape(shape)
            bound = np.abs(original).max() / 254 * 1.05 + 1e-8   # half a quantization step
            ok = np.abs(stored - original).max() <= bound
        else:
            print(f"  UNKNOWN TYPE {qtype.name} for {gguf_name}")
            failures += 1
            continue
        if not ok:
            print(f"  MISMATCH {gguf_name} ({qtype.name})")
            failures += 1
    return failures


def check_metadata(reader: gguf.GGUFReader, config: ModelConfig) -> int:
    expected = {
        "general.architecture": ARCH,
        f"{ARCH}.context_length": config.block_size,
        f"{ARCH}.embedding_length": config.n_embd,
        f"{ARCH}.block_count": config.n_layer,
        f"{ARCH}.feed_forward_length": config.ffn_dim,
        f"{ARCH}.attention.head_count": config.n_head,
    }
    failures = 0
    for key, want in expected.items():
        got = gguf_field_value(reader, key)
        if got != want:
            print(f"  METADATA {key}: expected {want}, found {got}")
            failures += 1
    n_tokens = len(gguf_field_value(reader, "tokenizer.ggml.tokens") or [])
    if n_tokens != config.vocab_size:
        print(f"  METADATA tokenizer has {n_tokens} tokens, model vocab_size is {config.vocab_size}")
        failures += 1
    return failures


def compare_with_llama_cpp(gguf_path: str, model: GPT, tokenizer, config: ModelConfig, n_tokens: int) -> int:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python is not installed, skipping the llama.cpp comparison "
              "(pip install llama-cpp-python to enable it).")
        return 0

    llm = Llama(model_path=gguf_path, n_ctx=config.block_size, logits_all=True, verbose=False)
    failures = 0
    for prompt in TEST_PROMPTS:
        ids = tokenizer.encode(prompt).ids
        cpp_ids = llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=False)
        if cpp_ids != ids:
            print(f"  TOKENIZER mismatch for {prompt!r}: tokenizers={ids} llama.cpp={cpp_ids}")
            failures += 1
            continue

        with torch.no_grad():
            torch_logits, _ = model(torch.tensor([ids]))
        torch_logits = torch_logits[0, -1].numpy()
        llm.reset()
        llm.eval(ids)
        cpp_logits = np.array(llm.scores[len(ids) - 1])
        max_diff = float(np.abs(torch_logits - cpp_logits).max())
        same_top = int(torch_logits.argmax()) == int(cpp_logits.argmax())

        torch_out = model.generate(torch.tensor([ids]), n_tokens, temperature=0)[0].tolist()[len(ids):]
        llm.reset()
        cpp_out = []
        for token in llm.generate(ids, top_k=1, top_p=1.0, temp=0.0, repeat_penalty=1.0):
            cpp_out.append(token)
            if len(cpp_out) >= n_tokens:
                break
        match = torch_out == cpp_out
        print(f"  {prompt!r}: max logit diff {max_diff:.4f}, top-1 {'same' if same_top else 'DIFFERENT'}, "
              f"first {n_tokens} greedy tokens {'match' if match else 'DIFFER'}")
        if not same_top:
            failures += 1
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a GGUF export against its checkpoint.")
    parser.add_argument("--checkpoint", default="out/base/ckpt.pt")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--n_tokens", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    checkpoint = load_checkpoint(args.checkpoint)
    config = ModelConfig(**checkpoint["model_config"])
    state = strip_state_dict_prefixes(checkpoint["model"])
    reader = gguf.GGUFReader(args.gguf)

    print("Comparing tensors")
    failures = compare_tensors(reader, state, expected_tensors(config))
    print("Checking metadata")
    failures += check_metadata(reader, config)

    model = GPT(config)
    model.load_state_dict(state)
    model.eval()
    tokenizer = load_tokenizer(args.tokenizer or os.path.join(ckpt_dir, "tokenizer.json"))
    print("Comparing with llama.cpp")
    failures += compare_with_llama_cpp(args.gguf, model, tokenizer, config, args.n_tokens)

    if failures:
        raise SystemExit(f"Verification FAILED with {failures} problem(s).")
    print("Verification passed.")


if __name__ == "__main__":
    main()
