# 2.4 — GPT: Decoder-Only Transformers

## Intuition

GPT (*Generative Pre-trained Transformer*) takes one half of the original Transformer — the **decoder** — and uses it for everything. The idea is disarmingly simple: train a model to predict the next token, and it learns language, reasoning, and world knowledge as a side effect. This **decoder-only, autoregressive** design powers GPT-2, GPT-3, GPT-4, LLaMA, Mistral, and most production LLMs.

---

## Core concepts

### Architecture

A GPT model is a stack of Transformer blocks with **causal self-attention** (no encoder, no cross-attention):

```
tokens → Embedding + Position → [Block × N] → LayerNorm → Linear → logits
```

Each block:

1. **Causal MHA** — lower-triangular mask prevents attending to future tokens.
2. **FFN** — position-wise, usually SwiGLU in modern variants.
3. **Residual + Norm** — Pre-Norm is standard since GPT-2.

The output linear layer (the **LM head**) maps hidden states to vocabulary logits. Often this layer **shares weights** with the input embedding matrix (weight tying), reducing parameter count by \(V \times d_{\text{model}}\).

### Training objective: next-token prediction

\[
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(w_t \mid w_1, \ldots, w_{t-1})
\]

This is the cross-entropy loss from Part 1.5. Training is entirely **self-supervised** — no labels, no task structure, just raw text.

### Autoregressive generation

At inference, tokens are generated one at a time:

1. Feed the prompt through the model.
2. Get logits for the last position.
3. Sample or argmax the next token.
4. Append to the sequence and repeat.

The **KV cache** (Part 4.2) avoids recomputing attention for already-generated tokens.

### GPT evolution

| Model | Year | Params | Context | Key innovation |
| --- | --- | --- | --- | --- |
| GPT-1 | 2018 | 117M | 512 | Pre-train + fine-tune paradigm |
| GPT-2 | 2019 | 1.5B | 1024 | Zero-shot via prompting, Pre-Norm |
| GPT-3 | 2020 | 175B | 2048 | In-context learning, few-shot |
| GPT-4 | 2023 | ~1.8T (rumored MoE) | 8K–128K | Multimodal, RLHF, massive scale |

### What GPT-2 changed architecturally

GPT-2 introduced practices now universal in decoder-only LLMs:

- **Pre-Norm** (LayerNorm before attention/FFN, not after).
- **Modified initialization** — scale residual path weights by \(1/\sqrt{N}\) where \(N\) is the number of layers.
- Weight tying (LM head = embedding transpose).

### In-context learning (ICL)

GPT-3's breakthrough: provide examples in the prompt and the model performs the task without any gradient updates:

```
Translate English to French:
sea otter => loutre de mer
cheese => fromage
hello =>
```

This is **few-shot prompting** — the model uses attention over the prompt to "program" itself at inference. No fine-tuning required.

---

## Code — Minimal GPT (nanoGPT-style)

```python
"""
Minimal GPT — a decoder-only Transformer language model.
Implements: causal attention, weight tying, generation loop.
Follows nanoGPT conventions for clarity.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 256
    block_size: int = 128     # max context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data *= 1.0 / math.sqrt(2 * config.n_layer)
        self.attn_drop = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x).view(B, T, 3, self.n_head, self.d_k)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data *= 1.0 / math.sqrt(2 * config.n_layer)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    config = GPTConfig(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_embd=128)
    model = GPT(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT model: {n_params:,} parameters")

    # Train on a tiny string (character-level)
    text = "hello world " * 50
    data = torch.tensor([ord(c) for c in text], dtype=torch.long)
    x = data[:-1].unsqueeze(0)[:, :config.block_size]
    y = data[1:].unsqueeze(0)[:, :config.block_size]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    for step in range(200):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step:4d}  loss={loss.item():.4f}")

    # Generate
    prompt = torch.tensor([[ord("h"), ord("e")]], dtype=torch.long)
    generated = model.generate(prompt, max_new_tokens=40, temperature=0.8)
    text_out = "".join(chr(c) for c in generated[0].tolist())
    print(f"\nGenerated: {text_out}")
```

---

## Interview takeaways

1. **Decoder-only = causal LM** — the model sees only past tokens via the triangular mask. This is why it generates text autoregressively. Know the difference from BERT's bidirectional attention.
2. **Weight tying** — LM head shares parameters with the embedding matrix, saving \(V \times d\) parameters. Be able to explain why this works (input and output are both token representations).
3. **In-context learning** — GPT-3 showed that scaling enables the model to perform new tasks from examples in the prompt. No weight updates needed. This is fundamentally different from fine-tuning.
4. **Scaling laws** — loss decreases as a power law with compute, data, and parameters. Chinchilla (Part 6) refined this to show data matters as much as params.
5. **Pre-Norm + residual scaling** — GPT-2 moved LayerNorm before attention/FFN and scaled residual weights by \(1/\sqrt{N}\). Without this, deep models don't train stably.
6. **Generation loop** — token-by-token, feeding each new token back. This sequential dependency is the inference bottleneck that KV cache (Part 4.2) and speculative decoding (Part 4.3) address.

---

## References

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1)
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3)
- Karpathy (2022), [nanoGPT](https://github.com/karpathy/nanoGPT)
