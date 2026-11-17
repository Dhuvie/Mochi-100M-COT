"""Download, clean, tokenize and shard the training data.

Stages (run all with --stage all):
  collect   download documents, clean them, drop short/duplicate ones -> data/raw/documents.jsonl
  tokenizer train a byte-level BPE tokenizer on a sample of the documents -> data/tokenizer.json
  tokenize  encode every document and write data/train.bin, data/val.bin and data/data_info.json

Sources:
  --source hf     latest English Wikipedia snapshot + FineWeb-Edu (needs internet and `datasets`)
  --source local  every .txt file under --input_dir (useful offline and for quick tests)
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import re
from typing import Dict, Iterator, List

import numpy as np

from config import DATA_PRESETS
from tokenizer import EOT_TOKEN, load_tokenizer, train_tokenizer

MIN_CHARS = 200          # documents shorter than this are dropped
VAL_EVERY = 100          # every 100th document goes to the validation set (1%)
HTML_TAG = re.compile(r"<[^>]+>")
MANY_NEWLINES = re.compile(r"\n{3,}")
MANY_SPACES = re.compile(r"[ \t]{2,}")


def clean_text(text: str) -> str:
    """Strip leftover markup and normalise whitespace."""
    text = HTML_TAG.sub(" ", text)
    text = text.replace("\r\n", "\n").replace(" ", " ")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MANY_SPACES.sub(" ", text)
    text = MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def text_hash(text: str) -> int:
    normalised = " ".join(text.lower().split())
    return int.from_bytes(hashlib.md5(normalised.encode("utf-8")).digest()[:8], "big")


def latest_wikipedia_config() -> str:
    """Pick the newest 'YYYYMMDD.en' snapshot published in the wikimedia/wikipedia dataset."""
    from datasets import get_dataset_config_names
    names = [n for n in get_dataset_config_names("wikimedia/wikipedia") if re.fullmatch(r"\d{8}\.en", n)]
    if not names:
        raise RuntimeError("No English Wikipedia snapshots found on the Hub.")
    return max(names)


def iter_wikipedia(config_name: str, max_docs: int) -> Iterator[str]:
    from datasets import load_dataset
    stream = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    for i, row in enumerate(stream):
        if i >= max_docs:
            break
        yield row["text"]


def iter_fineweb_edu(max_docs: int, newest_date: Dict[str, str]) -> Iterator[str]:
    from datasets import load_dataset
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(stream):
        if i >= max_docs:
            break
        crawl_date = str(row.get("date") or "")[:10]
        if crawl_date > newest_date.get("value", ""):
            newest_date["value"] = crawl_date
        yield row["text"]


def iter_local(input_dir: str, chunk_chars: int = 4000) -> Iterator[str]:
    """Each .txt file is split into ~4000 character documents at paragraph boundaries."""
    for root, _, files in os.walk(input_dir):
        for name in sorted(files):
            if not name.endswith(".txt"):
                continue
            with open(os.path.join(root, name), encoding="utf-8", errors="ignore") as f:
                paragraphs = f.read().split("\n\n")
            chunk: List[str] = []
            size = 0
            for paragraph in paragraphs:
                chunk.append(paragraph)
                size += len(paragraph)
                if size >= chunk_chars:
                    yield "\n\n".join(chunk)
                    chunk, size = [], 0
            if chunk:
                yield "\n\n".join(chunk)


def write_documents(source_name: str, texts: Iterator[str], out_file, seen: set) -> Dict[str, int]:
    seen_docs = kept = too_short = duplicates = 0
    for text in texts:
        seen_docs += 1
        text = clean_text(text)
        if len(text) < MIN_CHARS:
            too_short += 1
            continue
        digest = text_hash(text)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        out_file.write(json.dumps({"text": text, "source": source_name}) + "\n")
        kept += 1
        if kept % 10000 == 0:
            print(f"  [{source_name}] kept {kept:,} documents")
    return {"seen": seen_docs, "kept": kept, "too_short": too_short, "duplicates": duplicates}


def stage_collect(args) -> None:
    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    seen: set = set()
    sources = []
    with open(os.path.join(raw_dir, "documents.jsonl"), "w", encoding="utf-8") as out_file:
        if args.source == "hf":
            wiki_config = args.wikipedia_config or latest_wikipedia_config()
            print(f"Wikipedia snapshot: {wiki_config}")
            stats = write_documents("wikipedia", iter_wikipedia(wiki_config, args.wiki_docs), out_file, seen)
            snapshot = datetime.datetime.strptime(wiki_config.split(".")[0], "%Y%m%d").date().isoformat()
            sources.append({"name": "wikipedia", "dataset": "wikimedia/wikipedia", "config": wiki_config,
                            "snapshot_date": snapshot, **stats})

            newest = {"value": ""}
            print("Streaming FineWeb-Edu (sample-10BT)")
            stats = write_documents("fineweb-edu", iter_fineweb_edu(args.web_docs, newest), out_file, seen)
            sources.append({"name": "fineweb-edu", "dataset": "HuggingFaceFW/fineweb-edu",
                            "config": "sample-10BT", "snapshot_date": newest["value"] or None, **stats})
        else:
            if not args.input_dir:
                raise SystemExit("--source local needs --input_dir")
            stats = write_documents("local", iter_local(args.input_dir), out_file, seen)
            sources.append({"name": "local", "dataset": os.path.abspath(args.input_dir),
                            "config": None, "snapshot_date": None, **stats})

    with open(os.path.join(raw_dir, "sources.json"), "w") as f:
        json.dump(sources, f, indent=2)
    for s in sources:
        print(f"{s['name']}: kept {s['kept']:,} of {s['seen']:,} documents "
              f"({s['too_short']:,} too short, {s['duplicates']:,} duplicates)")


def iter_jsonl(path: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def stage_tokenizer(args) -> None:
    documents = os.path.join(args.data_dir, "raw", "documents.jsonl")
    total = sum(1 for _ in open(documents, encoding="utf-8"))
    keep_prob = min(1.0, args.tokenizer_docs / max(1, total))
    rng = random.Random(args.seed)
    sample = [text for text in iter_jsonl(documents) if rng.random() < keep_prob]
    print(f"Training tokenizer on {len(sample):,} of {total:,} documents (vocab {args.vocab_size})")
    tokenizer = train_tokenizer(sample, args.vocab_size, os.path.join(args.data_dir, "tokenizer.json"))
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")


def stage_tokenize(args) -> None:
    tokenizer = load_tokenizer(os.path.join(args.data_dir, "tokenizer.json"))
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size >= 2 ** 16:
        raise SystemExit("vocab size must be below 65536 to store token ids as uint16")
    eot_id = tokenizer.token_to_id(EOT_TOKEN)

    documents = os.path.join(args.data_dir, "raw", "documents.jsonl")
    counts = {"train_docs": 0, "val_docs": 0, "train_tokens": 0, "val_tokens": 0}
    batch: List[str] = []
    doc_index = 0
    with open(os.path.join(args.data_dir, "train.bin"), "wb") as train_f, \
            open(os.path.join(args.data_dir, "val.bin"), "wb") as val_f:

        def flush(texts: List[str], start_index: int) -> None:
            for offset, enc in enumerate(tokenizer.encode_batch(texts)):
                ids = np.array(enc.ids + [eot_id], dtype=np.uint16)
                is_val = (start_index + offset) % VAL_EVERY == 0
                (val_f if is_val else train_f).write(ids.tobytes())
                key = "val" if is_val else "train"
                counts[f"{key}_docs"] += 1
                counts[f"{key}_tokens"] += len(ids)

        batch_start = 0
        for text in iter_jsonl(documents):
            batch.append(text)
            doc_index += 1
            if len(batch) == 512:
                flush(batch, batch_start)
                batch_start, batch = doc_index, []
        if batch:
            flush(batch, batch_start)

    with open(os.path.join(args.data_dir, "raw", "sources.json")) as f:
        sources = json.load(f)
    snapshot_dates = [s["snapshot_date"] for s in sources if s.get("snapshot_date")]
    today = datetime.date.today().isoformat()
    info = {
        "project": "llm-from-scratch",
        "data_collection_date": today,
        "data_cutoff_date": args.cutoff_date or today,
        "newest_source_snapshot": max(snapshot_dates) if snapshot_dates else None,
        "note": ("The model only knows what is in the snapshots listed under 'sources'. "
                 "Web crawls can be older than the collection date."),
        "sources": sources,
        "vocab_size": vocab_size,
        "eot_token_id": eot_id,
        "dtype": "uint16",
        **counts,
    }
    with open(os.path.join(args.data_dir, "data_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"train: {counts['train_tokens']:,} tokens in {counts['train_docs']:,} docs | "
          f"val: {counts['val_tokens']:,} tokens in {counts['val_docs']:,} docs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--stage", choices=["all", "collect", "tokenizer", "tokenize"], default="all")
    parser.add_argument("--source", choices=["hf", "local"], default="hf")
    parser.add_argument("--input_dir", default=None, help="folder of .txt files for --source local")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--wikipedia_config", default=None, help="e.g. 20231101.en (default: newest available)")
    parser.add_argument("--wiki_docs", type=int, default=None)
    parser.add_argument("--web_docs", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--tokenizer_docs", type=int, default=None)
    parser.add_argument("--cutoff_date", default=None, help="YYYY-MM-DD, defaults to today")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    defaults = DATA_PRESETS[args.preset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    if args.stage in ("all", "collect"):
        stage_collect(args)
    if args.stage in ("all", "tokenizer"):
        stage_tokenizer(args)
    if args.stage in ("all", "tokenize"):
        stage_tokenize(args)


if __name__ == "__main__":
    main()
