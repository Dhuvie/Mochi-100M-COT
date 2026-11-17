"""Upload a GGUF file and an auto-generated model card to the Hugging Face Hub.

The access token is read from the HF_TOKEN environment variable (never hard-code it).

Example:
    export HF_TOKEN=hf_xxx
    python upload_to_hub.py --gguf out/base/llm-from-scratch-f16.gguf --repo_id yourname/llm-from-scratch-100m
"""
import argparse
import json
import os
import tempfile

import gguf

from utils import gguf_field_value

CARD_TEMPLATE = """---
license: mit
library_name: gguf
tags:
- gguf
- gpt2
- llama.cpp
- from-scratch
language:
- en
---

# {repo_name}

A small GPT-2 style language model (about {params_m:.0f}M parameters) trained from scratch as a
college project. It is a base model: it continues text and does not follow instructions.

## Model details

| | |
|---|---|
| Architecture | decoder-only transformer, llama.cpp `gpt2` layout |
| Parameters | {params:,} |
| Layers / heads / width | {n_layer} / {n_head} / {n_embd} |
| Context length | {context} tokens |
| Vocabulary | {vocab:,} byte-level BPE tokens |
| File | `{filename}` ({size_mb:.0f} MB) |

## Training data and knowledge cutoff

- Data collection date: **{collected}**
- Data cutoff date recorded in the file: **{cutoff}**
- Sources: English Wikipedia and FineWeb-Edu (see the project README for exact snapshots).

The model only knows what was in its training snapshots. Web crawls can be older than the
collection date, so recent events may be missing.

## Run it

llama.cpp:
```
llama-cli -m {filename} -p "The history of" -n 100 --temp 0.8
```

Ollama (`Modelfile`):
```
FROM ./{filename}
PARAMETER temperature 0.8
```
then `ollama create {repo_name} -f Modelfile` and `ollama run {repo_name}`. Ollama's support
for the `gpt2` architecture depends on the Ollama version, so if it refuses the file use
llama.cpp instead.

LM Studio: download the `.gguf` file and place it in LM Studio's models folder
(LM Studio is built on llama.cpp, which supports this architecture).

## Limitations

A model this small writes fluent text that is often factually wrong. It has no safety
tuning and no instruction following. Use it for learning and experiments.
"""


def build_model_card(gguf_path: str) -> str:
    reader = gguf.GGUFReader(gguf_path)
    params = sum(int(t.n_elements) for t in reader.tensors)
    arch = gguf_field_value(reader, "general.architecture")
    repo_name = gguf_field_value(reader, "general.name") or "llm-from-scratch"
    return CARD_TEMPLATE.format(
        repo_name=repo_name, params=params, params_m=params / 1e6,
        n_layer=gguf_field_value(reader, f"{arch}.block_count"),
        n_head=gguf_field_value(reader, f"{arch}.attention.head_count"),
        n_embd=gguf_field_value(reader, f"{arch}.embedding_length"),
        context=gguf_field_value(reader, f"{arch}.context_length"),
        vocab=len(gguf_field_value(reader, "tokenizer.ggml.tokens") or []),
        filename=os.path.basename(gguf_path), size_mb=os.path.getsize(gguf_path) / 1e6,
        cutoff=gguf_field_value(reader, "llm-from-scratch.data_cutoff_date") or "unknown",
        collected=gguf_field_value(reader, "llm-from-scratch.data_collection_date") or "unknown",
    )


def main():
    parser = argparse.ArgumentParser(description="Upload a GGUF model to the Hugging Face Hub.")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--repo_id", required=True, help="e.g. yourname/llm-from-scratch-100m")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="only print the model card")
    args = parser.parse_args()

    card = build_model_card(args.gguf)
    if args.dry_run:
        print(card)
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the HF_TOKEN environment variable to a Hugging Face write token.")

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    api.upload_file(path_or_fileobj=args.gguf, path_in_repo=os.path.basename(args.gguf), repo_id=args.repo_id)
    with tempfile.TemporaryDirectory() as tmp:
        card_path = os.path.join(tmp, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card)
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md", repo_id=args.repo_id)
    print(f"uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
