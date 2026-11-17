"""Model and training settings for the two presets: "tiny" and "base"."""
from dataclasses import dataclass, replace


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    block_size: int = 512      # maximum context length
    n_layer: int = 11
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True          # GPT-2 uses biases in Linear and LayerNorm
    layer_norm_eps: float = 1e-5

    @property
    def ffn_dim(self) -> int:
        return 4 * self.n_embd


@dataclass
class TrainConfig:
    batch_size: int = 8            # sequences per micro-batch
    grad_accum_steps: int = 8      # micro-batches per optimizer step
    max_steps: int = 60000
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 1000
    seed: int = 1337


# Data settings used by data.py for each preset.
DATA_PRESETS = {
    "tiny": dict(wiki_docs=3000, web_docs=3000, vocab_size=4096, tokenizer_docs=3000),
    "base": dict(wiki_docs=2_000_000, web_docs=1_500_000, vocab_size=32000, tokenizer_docs=100_000),
}


def get_preset(name: str):
    """Return fresh (ModelConfig, TrainConfig) objects for a preset name."""
    if name == "tiny":
        model_cfg = ModelConfig(vocab_size=4096, block_size=128, n_layer=4, n_head=4,
                                n_embd=256, dropout=0.0)
        train_cfg = TrainConfig(batch_size=16, grad_accum_steps=1, max_steps=300,
                                learning_rate=1e-3, min_lr=1e-4, warmup_steps=20,
                                eval_interval=50, eval_iters=20, log_interval=10,
                                checkpoint_interval=100)
    elif name == "base":
        model_cfg = ModelConfig()
        train_cfg = TrainConfig()
    else:
        raise ValueError(f"Unknown preset '{name}'. Choose 'tiny' or 'base'.")
    return model_cfg, train_cfg


def with_overrides(cfg, **overrides):
    """Return a copy of a dataclass config with non-None overrides applied."""
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean)
