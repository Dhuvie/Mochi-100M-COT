"""Byte-level BPE tokenizer (GPT-2 style) trained with the HuggingFace `tokenizers` library."""
import json
from typing import Iterable, List, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT_TOKEN = "<|endoftext|>"


def train_tokenizer(texts: Iterable[str], vocab_size: int, save_path: str) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 byte symbols
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)


def get_vocab_and_merges(tokenizer: Tokenizer) -> Tuple[List[str], List[str]]:
    """Return (tokens ordered by id, merges as 'left right' strings) for the GGUF export."""
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    tokens = [None] * len(vocab)
    for token, idx in vocab.items():
        tokens[idx] = token
    if any(t is None for t in tokens):
        raise ValueError("Tokenizer ids are not contiguous from 0 to vocab_size - 1.")

    state = json.loads(tokenizer.to_str())
    merges = [m if isinstance(m, str) else " ".join(m) for m in state["model"]["merges"]]
    return tokens, merges
