"""Convert a trained checkpoint into a GGUF file that llama.cpp, Ollama and LM Studio can load.

The model uses the GPT-2 layout, so the weights are written under llama.cpp's built-in
"gpt2" architecture and no custom C++ code is needed.

Example:
    python export_gguf.py --checkpoint out/base/ckpt.pt --dtype f16
    python export_gguf.py --checkpoint out/base/ckpt.pt --dtype q8_0 --out model-q8_0.gguf
"""
import argparse
import json
import os
from typing import Dict, Tuple

import gguf
import numpy as np
import torch

from config import ModelConfig
from tokenizer import EOT_TOKEN, get_vocab_and_merges, load_tokenizer
from utils import load_checkpoint, strip_state_dict_prefixes

ARCH = "gpt2"
PROJECT_NAME = "llm-from-scratch"
FILE_TYPES = {"f32": gguf.LlamaFileType.ALL_F32,
              "f16": gguf.LlamaFileType.MOSTLY_F16,
              "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}


class ExportError(Exception):
    pass


def expected_tensors(config: ModelConfig) -> Dict[str, Tuple[str, tuple]]:
    """Map GGUF tensor name -> (PyTorch parameter name, expected shape)."""
    c, f, v, t = config.n_embd, config.ffn_dim, config.vocab_size, config.block_size
    tensors = {
        "token_embd.weight": ("transformer.wte.weight", (v, c)),
        "position_embd.weight": ("transformer.wpe.weight", (t, c)),
        "output_norm.weight": ("transformer.ln_f.weight", (c,)),
        "output_norm.bias": ("transformer.ln_f.bias", (c,)),
    }
    for i in range(config.n_layer):
        torch_prefix, gguf_prefix = f"transformer.h.{i}", f"blk.{i}"
        layer = {
            "attn_norm.weight": ("ln_1.weight", (c,)),
            "attn_norm.bias": ("ln_1.bias", (c,)),
            "attn_qkv.weight": ("attn.c_attn.weight", (3 * c, c)),
            "attn_qkv.bias": ("attn.c_attn.bias", (3 * c,)),
            "attn_output.weight": ("attn.c_proj.weight", (c, c)),
            "attn_output.bias": ("attn.c_proj.bias", (c,)),
            "ffn_norm.weight": ("ln_2.weight", (c,)),
            "ffn_norm.bias": ("ln_2.bias", (c,)),
            "ffn_up.weight": ("mlp.c_fc.weight", (f, c)),
            "ffn_up.bias": ("mlp.c_fc.bias", (f,)),
            "ffn_down.weight": ("mlp.c_proj.weight", (c, f)),
            "ffn_down.bias": ("mlp.c_proj.bias", (c,)),
        }
        for gguf_suffix, (torch_suffix, shape) in layer.items():
            tensors[f"{gguf_prefix}.{gguf_suffix}"] = (f"{torch_prefix}.{torch_suffix}", shape)
    return tensors


def validate_state_dict(state: dict, config: ModelConfig, tensor_map: dict) -> None:
    """Check that every tensor exists, has the right shape and holds finite numbers."""
    if not config.bias:
        raise ExportError("The GGUF 'gpt2' architecture needs biases; this checkpoint was trained with bias=False.")
    problems = []
    for gguf_name, (torch_name, shape) in tensor_map.items():
        if torch_name not in state:
            problems.append(f"missing tensor: {torch_name}")
            continue
        tensor = state[torch_name]
        if tuple(tensor.shape) != shape:
            problems.append(f"wrong shape for {torch_name}: got {tuple(tensor.shape)}, expected {shape}")
        elif not torch.isfinite(tensor.float()).all():
            problems.append(f"NaN or inf values in {torch_name}")
    head = state.get("lm_head.weight")
    wte = state.get("transformer.wte.weight")
    if head is not None and wte is not None and not torch.equal(head, wte):
        problems.append("lm_head.weight differs from transformer.wte.weight (output head is not tied)")
    if problems:
        raise ExportError("Checkpoint validation failed:\n  - " + "\n  - ".join(problems))


def convert_tensor(name: str, tensor: torch.Tensor, dtype: str):
    """Return (numpy data, raw GGML type or None). Only large weight matrices are reduced in precision."""
    array = tensor.detach().to(torch.float32).cpu().numpy()
    keep_f32 = dtype == "f32" or array.ndim != 2 or name == "position_embd.weight"
    if keep_f32:
        return array.astype(np.float32), None
    if dtype == "q8_0" and array.shape[-1] % 32 == 0:
        return gguf.quants.quantize(array, gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0
    if dtype == "q8_0":
        print(f"warning: {name} has row length {array.shape[-1]} (not a multiple of 32), storing as f16")
    half = array.astype(np.float16)
    if not np.isfinite(half).all():
        raise ExportError(f"{name} overflows float16; export with --dtype f32")
    return half, None


def add_tokenizer(writer: gguf.GGUFWriter, tokenizer_path: str, expected_vocab: int) -> int:
    tokenizer = load_tokenizer(tokenizer_path)
    tokens, merges = get_vocab_and_merges(tokenizer)
    if len(tokens) != expected_vocab:
        raise ExportError(f"Tokenizer has {len(tokens)} tokens but the model has vocab_size={expected_vocab}.")
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    token_types = [int(gguf.TokenType.CONTROL) if i == eot_id else int(gguf.TokenType.NORMAL)
                   for i in range(len(tokens))]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(eot_id)
    writer.add_eos_token_id(eot_id)
    writer.add_add_bos_token(False)
    return eot_id


def export_gguf(checkpoint_path: str, out_path: str = None, dtype: str = "f16",
                tokenizer_path: str = None, data_info_path: str = None) -> str:
    """Convert a checkpoint to GGUF and return the output path."""
    if dtype not in FILE_TYPES:
        raise ExportError(f"Unknown dtype '{dtype}'. Choose from {sorted(FILE_TYPES)}.")
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    tokenizer_path = tokenizer_path or os.path.join(ckpt_dir, "tokenizer.json")
    data_info_path = data_info_path or os.path.join(ckpt_dir, "data_info.json")
    out_path = out_path or os.path.join(ckpt_dir, f"{PROJECT_NAME}-{dtype}.gguf")
    for path in (checkpoint_path, tokenizer_path):
        if not os.path.exists(path):
            raise ExportError(f"File not found: {path}")

    checkpoint = load_checkpoint(checkpoint_path, device="cpu")   # CPU only, no GPU needed
    config = ModelConfig(**checkpoint["model_config"])
    state = strip_state_dict_prefixes(checkpoint["model"])
    tensor_map = expected_tensors(config)
    validate_state_dict(state, config, tensor_map)

    cutoff, collected, newest = "unknown", "unknown", "unknown"
    if os.path.exists(data_info_path):
        with open(data_info_path) as f:
            info = json.load(f)
        cutoff = str(info.get("data_cutoff_date", "unknown"))
        collected = str(info.get("data_collection_date", "unknown"))
        newest = str(info.get("newest_source_snapshot") or "unknown")

    tmp_path = out_path + ".tmp"          # written first, renamed at the end
    try:
        writer = gguf.GGUFWriter(tmp_path, ARCH)
        writer.add_name(PROJECT_NAME)
        writer.add_description(f"Small GPT-2 style language model trained from scratch. Data cutoff: {cutoff}.")
        writer.add_context_length(config.block_size)
        writer.add_embedding_length(config.n_embd)
        writer.add_block_count(config.n_layer)
        writer.add_feed_forward_length(config.ffn_dim)
        writer.add_head_count(config.n_head)
        writer.add_layer_norm_eps(config.layer_norm_eps)
        writer.add_file_type(FILE_TYPES[dtype])
        writer.add_string(f"{PROJECT_NAME}.data_cutoff_date", cutoff)
        writer.add_string(f"{PROJECT_NAME}.data_collection_date", collected)
        writer.add_string(f"{PROJECT_NAME}.newest_source_snapshot", newest)
        writer.add_string(f"{PROJECT_NAME}.project", PROJECT_NAME)
        add_tokenizer(writer, tokenizer_path, config.vocab_size)

        for gguf_name, (torch_name, _) in tensor_map.items():
            data, raw_dtype = convert_tensor(gguf_name, state[torch_name], dtype)
            writer.add_tensor(gguf_name, data, raw_dtype=raw_dtype)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    reader = gguf.GGUFReader(out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"wrote {out_path}")
    print(f"  size: {size_mb:.1f} MB | tensors: {len(reader.tensors)} | dtype: {dtype}")
    print(f"  metadata keys: {', '.join(sorted(reader.fields.keys()))}")
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export a checkpoint to GGUF.")
    parser.add_argument("--checkpoint", default="out/base/ckpt.pt")
    parser.add_argument("--out", default=None)
    parser.add_argument("--dtype", choices=sorted(FILE_TYPES), default="f16")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--data_info", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        export_gguf(args.checkpoint, args.out, args.dtype, args.tokenizer, args.data_info)
    except ExportError as err:
        raise SystemExit(f"Export failed: {err}")


if __name__ == "__main__":
    main()
