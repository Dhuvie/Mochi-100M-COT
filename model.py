"""A decoder-only transformer written from scratch with plain torch ops.

The layout follows GPT-2 (learned position embeddings, pre-LayerNorm, GELU, a fused
QKV projection and tied input/output embeddings). Keeping the same layout means the
trained weights map directly onto llama.cpp's built-in "gpt2" architecture, which is
what export_gguf.py relies on.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        # one Linear produces queries, keys and values for all heads at once
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        batch, seq_len, n_embd = x.shape                          # (B, T, C)
        q, k, v = self.c_attn(x).split(n_embd, dim=2)             # each (B, T, C)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        total_k_len = k.shape[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, total_k_len)
        if seq_len > 1:
            # During prompt processing, a token may only attend to itself and earlier tokens
            scores = scores.masked_fill(self.causal_mask[:, :, :seq_len, :total_k_len] == 0, float("-inf"))

        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        out = weights @ v                                              # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        res = self.resid_dropout(self.c_proj(out))
        if use_cache:
            return res, present_kv
        return res


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.ffn_dim, bias=config.bias)
        self.act = nn.GELU(approximate="tanh")  # same GELU variant GPT-2 and llama.cpp use
        self.c_proj = nn.Linear(config.ffn_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, past_kv=None, use_cache: bool = False):
        if use_cache:
            attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=True)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, present_kv
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight tying

        self.apply(self._init_weights)
        # scale down the output projections of each residual branch
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, past_kv=None, use_cache: bool = False):
        """idx: (B, T) token ids. Returns (logits, loss) or (logits, present_kv).

        With targets, logits cover every position. Without targets only the last
        position is computed, which is all that generation needs.
        """
        batch, seq_len = idx.shape
        if past_kv is None:
            assert seq_len <= self.config.block_size, "sequence longer than block_size"
            positions = torch.arange(seq_len, device=idx.device)
        else:
            past_len = past_kv[0][0].shape[2]
            assert past_len + seq_len <= self.config.block_size, "sequence + cache exceeds block_size"
            positions = torch.arange(past_len, past_len + seq_len, device=idx.device)

        x = self.transformer.wte(idx) + self.transformer.wpe(positions)  # (B, T, C)
        x = self.transformer.drop(x)

        if use_cache:
            present_kvs = []
            for i, block in enumerate(self.transformer.h):
                layer_past = past_kv[i] if past_kv is not None else None
                x, layer_present = block(x, past_kv=layer_past, use_cache=True)
                present_kvs.append(layer_present)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x[:, -1:, :])
            return logits, present_kvs

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)                                      # (B, T, V)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss
        return self.lm_head(x[:, -1:, :]), None                           # (B, 1, V)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, top_p: float = None, min_p: float = None,
                 eos_token_id: int = None, use_cache: bool = True) -> torch.Tensor:
        """Autoregressive sampling with optional KV-cache and Min-P sampling."""
        was_training = self.training
        self.eval()

        past_kv = None
        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    logits, past_kv = self(idx[:, -self.config.block_size:], use_cache=True)
                else:
                    logits, past_kv = self(idx[:, -1:], past_kv=past_kv, use_cache=True)
            else:
                context = idx[:, -self.config.block_size:]
                logits, _ = self(context)

            logits = logits[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth_value = torch.topk(logits, min(top_k, logits.size(-1)))[0][:, -1, None]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))
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

            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        self.train(was_training)
        return idx


if __name__ == "__main__":
    from utils import count_parameters
    for cfg in (ModelConfig(), ModelConfig(n_layer=4, n_head=4, n_embd=256, block_size=128, vocab_size=4096)):
        print(f"{cfg}: {count_parameters(GPT(cfg)):,} parameters")
