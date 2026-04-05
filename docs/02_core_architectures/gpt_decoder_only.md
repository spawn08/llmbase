# GPT: Decoder-Only Transformers

## Why This Matters for LLMs

GPT-style **decoder-only** Transformers are the default architecture for frontier assistants and open-weight stacks alike. When an interviewer asks how ChatGPT, Claude, LLaMA, or Gemini produce text, the honest answer is: a deep stack of **causal** self-attention blocks trained to predict the **next token**, then sampled autoregressively at inference. That single pipeline governs pre-training data curation, alignment via preference optimization, inference serving with KV caches, and research on long-context methods. Roles touching training, evaluation, or systems engineering all assume you can walk through one forward pass and explain why each position only attends to the **past**.

A second reason this page matters is **mechanical**: decoder-only models are where **autoregressive generation** meets **GPU reality**. Training parallelizes across sequence positions because causal masking still allows packed attention kernels over the lower triangle. Inference, by contrast, becomes sequential along the generated length unless you add speculative decoding or other tricks. The **KV cache** exists precisely because recomputing keys and values for earlier tokens on every new step would waste memory bandwidth and compute. Interviewers reward candidates who connect the causal mask, the next-token loss, and the inference loop without hand-waving.

Third, GPT-family models are the home of **in-context learning**: the phenomenon that scaling makes few-shot prompts work without weight updates. That behavior is not magic; it emerges from attention patterns that route information from demonstration tokens to the prediction site, combined with pre-training that rewards flexible conditional prediction. Understanding decoder-only stacks is therefore prerequisite for discussing retrieval-augmented generation, tool use, and prompt engineering at a level that goes beyond memorized buzzwords.

---

## Core Concepts

### Architecture

A GPT model is a stack of Transformer **decoder** blocks with **causal** (unidirectional) self-attention. There is **no encoder** and **no cross-attention** to another sequence in the original language-modeling formulation:

```
token IDs → Token embedding + Positional embedding → Dropout → [TransformerBlock × L] → Final LayerNorm → Linear (LM head) → logits over vocabulary
```

Each **Transformer block** (GPT-2 style **Pre-Norm**) typically contains:

1. **LayerNorm** on the residual stream, then **causal multi-head self-attention**, then residual add.
2. **LayerNorm**, then position-wise **feed-forward network** (often GELU or SwiGLU in modern variants), then residual add.

The final **language modeling head** maps each position’s hidden vector \(\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}\) to a vector of logits \(\boldsymbol{\ell}_t \in \mathbb{R}^{V}\) where \(V\) is the vocabulary size.

\[
\boldsymbol{\ell}_t = \mathbf{W}_{\text{head}} \, \mathrm{LN}(\mathbf{h}_t^{\text{final}}) + \mathbf{b}_{\text{head}} \quad \text{(bias optional; often omitted when tying weights)}
\]

!!! math-intuition "In Plain English"
    Each position carries a hidden vector summarizing “what the model believes so far.” The LM head is a linear classifier from that vector to **vocabulary scores**. Higher logit means the model assigns more mass to that token as the next symbol. No softmax yet: the softmax appears inside the loss during training or inside the sampling step during generation.

**Weight tying** (popularized in GPT-2) reuses the **input token embedding matrix** \(\mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}\) as the **output projection** (up to transpose). Parameter savings are on the order of \(V \cdot d_{\text{model}}\) weights, which is large when \(V\) is fifty thousand or more.

---

### Training Objective: Next-Token Prediction

Given a token sequence \(w_1, w_2, \ldots, w_T\), the model defines a conditional distribution at each position using only **previous** tokens. The standard training loss is **average negative log-likelihood** (cross-entropy with the true next token):

\[
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(w_t \mid w_1, \ldots, w_{t-1})
\]

!!! math-intuition "In Plain English"
    For each position \(t\), the model outputs a probability vector over the whole vocabulary. You look up the probability assigned to the **actual** next token \(w_t\) and take the logarithm. The negative sign turns “maximize log probability” into a **loss** to minimize. Averaging over \(T\) positions normalizes for sequence length so short and long sequences are comparable in gradient magnitude at the macro level (though modern batches pack many sequences and use careful masking).

Equivalently, for each position \(t\) let \(\mathbf{p}_t = \mathrm{softmax}(\boldsymbol{\ell}_t)\) be the predicted distribution and let \(y_t\) be the one-hot target for the true token. Then the per-position cross-entropy is \(H(\mathbf{y}_t, \mathbf{p}_t) = -\sum_{v} y_{t,v} \log p_{t,v}\), which collapses to \(-\log p_{t,w_t}\) because the target is one-hot.

\[
H(\mathbf{y}_t, \mathbf{p}_t) = -\log p_{t,w_t} = -\log \frac{\exp(\ell_{t,w_t})}{\sum_{v'=1}^{V}\exp(\ell_{t,v'})}
\]

!!! math-intuition "In Plain English"
    Cross-entropy measures how surprised the model is by the correct answer. If the model puts all mass on the right token, \(-\log 1 = 0\). If the model spreads probability thinly, the true token’s probability can be tiny and the loss spikes. The softmax denominator forces **competition** across the vocabulary: raising the logit for the correct token automatically lowers total probability mass available for others.

!!! example "Worked Example: Three-Position Sequence"
    Consider the token sequence `["The", "cat", "sat"]`. A causal model is trained to predict:
    
    - At position \(1\) (after reading `The`): predict the token `"cat"`.
    - At position \(2\) (after reading `The cat`): predict the token `"sat"`.
    - At position \(3\) (after reading `The cat sat`): predict whatever comes next in the corpus (for example `"on"` or end-of-sequence), depending on the training data.
    
    **Toy vocabulary** \(\{\text{The}, \text{cat}, \text{sat}, \text{UNK}\}\) with indices \(0,1,2,3\). Suppose at position \(1\) the logits are \(\boldsymbol{\ell}_1 = (2.0,\, 1.0,\, 0.5,\, 0.0)\) for the four entries respectively, and the true next token is `"cat"` (index \(1\)).
    
    **Softmax:**  
    \(\exp(2.0)=7.389\), \(\exp(1.0)=2.718\), \(\exp(0.5)=1.649\), \(\exp(0.0)=1.000\). Sum \(= 12.756\).  
    \(P(\text{cat}) = 2.718 / 12.756 \approx 0.213\).
    
    **Cross-entropy:** \(-\log(0.213) \approx 1.53\) nats (or use \(\log_{10}\) or \(\log_2\) consistently; deep learning frameworks default to natural log).
    
    At position \(2\) suppose logits for the true token `"sat"` (index \(2\)) yield \(P(\text{sat}) = 0.60\). Then CE \(= -\log(0.60) \approx 0.51\).
    
    At position \(3\) suppose the model is less confident: \(P(\text{next true token}) = 0.15\). Then CE \(= -\log(0.15) \approx 1.90\).
    
    **Average loss** over these three positions (if the sequence ended there): \((1.53 + 0.51 + 1.90)/3 \approx 1.31\). Optimization pushes logits so that the true token’s probability moves toward \(1\) at each step.

---

### Causal Attention in One Line of Math

For single-head attention (omitting batch and head dimensions), queries, keys, and values are linear projections of hidden states: \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\). Pre-softmax scores for position \(i\) attending to position \(j\) are:

\[
\text{score}_{i,j} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d_k}}
\]

The **causal mask** sets \(\text{score}_{i,j} = -\infty\) for all \(j > i\) so that \(\mathrm{softmax}\) assigns zero weight to future positions.

\[
\alpha_{i,j} = \frac{\exp(\text{score}_{i,j}) \cdot M_{i,j}}{\sum_{j' \le i}\exp(\text{score}_{i,j'})},
\qquad M_{i,j} = \begin{cases} 1 & j \le i \\ 0 & j > i \end{cases}
\]

!!! math-intuition "In Plain English"
    Position \(i\) is only allowed to **look backward** along the sequence. The triangular mask enforces an ordering: when predicting token \(i\), the model may not peek at token \(i+1\). That matches the generative story “predict the next token given the past.” Without this mask, the representation at every position could depend on future words and the next-token training objective would be contaminated by leakage.

---

### Autoregressive Generation

At inference, text is produced **one token at a time**:

1. Encode the prompt into token IDs and run a **forward pass** through the model.
2. Read logits **only at the last position** (or the position where you want the next token).
3. Apply a **decoding rule**: argmax (greedy), top-\(k\), top-\(p\) (nucleus), or temperature scaling.
4. **Append** the sampled token to the sequence and repeat until a stop criterion (maximum length, end-of-sequence token, or custom rule).

This loop is inherently **sequential** along the generated length: step \(s\) depends on the token chosen at step \(s-1\). That dependency is why naive inference spends a large fraction of time on memory-bound **small matrix multiplies** and why systems engineers care about batching, KV caching, quantization, and speculative decoding.

\[
w_{t+1} \sim \mathrm{Sample}\Big(\mathrm{softmax}\big(\boldsymbol{\ell}_t / \tau\big)\Big)
\]

where \(\tau\) is **temperature**. \(\tau < 1\) sharpens the distribution (less random), \(\tau > 1\) flattens it (more random).

!!! math-intuition "In Plain English"
    Temperature divides logits before softmax. Lower temperature amplifies differences between the largest logit and the rest, so sampling often chases the single best token. Higher temperature makes probabilities more uniform, increasing diversity at the risk of hallucinations or incoherence. This is a **runtime knob**, not a training change.

---

### KV Cache

During autoregressive decoding, the hidden states for earlier tokens would require recomputing their keys and values in every layer at every new step unless you **store** them. A **KV cache** holds, for each layer, the key tensor \(\mathbf{K}\) and value tensor \(\mathbf{V}\) for all **past** positions so that a new forward pass only computes \(\mathbf{Q}, \mathbf{K}, \mathbf{V}\) for the **latest** token (and extends the cache).

For a model with \(L\) layers, \(H\) attention heads, head dimension \(d_{\text{head}}\), sequence length \(T\), and \(B\) batch size, a common first-order accounting for the cache tensor elements (keys plus values) is:

\[
\text{KV bytes} \approx B \cdot L \cdot 2 \cdot H \cdot T \cdot d_{\text{head}} \cdot \texttt{bytes\_per\_elem}
\]

The factor \(2\) is **keys plus values**. Some implementations store additional tensors (for example past hidden states in certain APIs), but the dominant memory for long contexts is often the KV cache in half precision.

!!! math-intuition "In Plain English"
    Attention compares queries to keys and mixes values. For tokens you already processed, those keys and values do not change when you append a new token at the end. Caching them avoids repeating identical linear projections and attention bookkeeping. The cost is **memory**: the cache grows linearly with context length and linearly with layer count and head count.

!!! example "Worked Example: KV Cache Memory"
    Take \(L=32\) layers, \(H=32\) heads, \(d_{\text{head}}=128\), \(T=2048\), **FP16** with \(2\) bytes per element, **batch size \(B=1\)**. Ignore optimizer states and main weights; focus only on KV tensors.
    
    \[
    \text{Elements} = B \cdot L \cdot 2 \cdot H \cdot T \cdot d_{\text{head}} = 1 \cdot 32 \cdot 2 \cdot 32 \cdot 2048 \cdot 128
    \]
    
    Compute stepwise: \(32 \times 2 = 64\). Then \(64 \times 32 = 2048\). Then \(2048 \times 2048 = 4{,}194{,}304\). Then \(4{,}194{,}304 \times 128 = 536{,}870{,}912\) elements.
    
    **Bytes** \(= 536{,}870{,}912 \times 2 = 1{,}073{,}741{,}824\) bytes, which is **1024 MiB** (approximately **1 GiB**).
    
    Doubling batch size doubles this footprint. Longer contexts (for example \(T=8192\)) scale KV memory linearly in \(T\) unless you use **grouped-query attention**, **multi-query attention**, **quantized KV**, or **offloading** to CPU or disk.

---

### GPT Evolution

| Model | Year | Parameters (approx.) | Context (typical) | Training data (order of magnitude) | Key innovation |
|-------|------|----------------------|-------------------|--------------------------------------|----------------|
| GPT-1 | 2018 | 117M | 512 | BookCorpus | Generative pre-training then supervised fine-tuning |
| GPT-2 | 2019 | 1.5B (largest public) | 1024 | WebText | Zero-shot task transfer from LM; Pre-Norm; weight tying |
| GPT-3 | 2020 | 175B | 2048 | Hundreds of billions of tokens | Few-shot in-context learning at scale |
| GPT-4 | 2023 | Not disclosed (public estimates vary widely) | 8k–128k product tiers | Not fully public | Alignment (RLHF), multimodal, tooling ecosystem |

Rumored mixture-of-expert routing appears in some frontier models; treat parameter counts for closed models as **uncertain** in interviews unless you qualify sources.

---

### In-Context Learning

**In-context learning** means the model adapts its behavior based on **examples placed in the prompt** without **gradient updates** to weights. At a high level, the transformer can attend from the answer position back to demonstration tokens, copying formats, styles, and shallow task structure.

\[
P(w \mid \underbrace{x_1, \ldots, x_k}_{\text{prompt and demonstrations}}, \underbrace{w_1, \ldots, w_{t-1}}_{\text{generated so far}})
\]

!!! math-intuition "In Plain English"
    The probability is still a single next-token distribution conditioned on **everything to the left**, including few-shot examples. There is no separate “learning rate” inside inference. What changes with scale is that the learned attention patterns and representations make this conditional distribution **useful** for tasks that resemble training mixtures.

---

### GPT-2 Architectural Details That Became Standard

- **Pre-Norm**: Apply LayerNorm **before** attention and before the feed-forward sublayer inside each block. This stabilizes gradient flow in deep stacks compared to post-norm formulations in the original Transformer decoder.
- **Residual scaling**: GPT-2 scales projection weights in residual branches (implementation detail in code) to keep variance stable as depth grows.
- **Larger vocabulary and byte-pair encoding**: Subword tokenization balances open-vocabulary coverage with manageable \(V\).
- **Weight tying**: Shares input embeddings with the output classifier, reducing parameters and often improving perplexity.

---

## Deep Dive

??? deep-dive "Deep Dive: Why Decoder-Only Won for General-Purpose Assistants"
    Encoder-only models (BERT-style) excel at **understanding** tasks where the full input is visible at once. Encoder–decoder models (T5-style) excel at **conditional generation** where a clear input–output split exists (translation, summarization with a provided passage). Decoder-only models won the **open-ended assistant** race for several practical reasons:
    
    **Unified pre-training:** A single next-token objective on broad internet-scale text produces a general conditional distribution \(P(\text{continuation} \mid \text{prompt})\). That same interface covers chat, code completion, and tool-formatted outputs without architectural branching.
    
    **Instruction tuning alignment:** Methods like supervised fine-tuning on demonstrations and preference optimization (for example RLHF or DPO) are naturally expressed as **next-token prediction** on curated dialogues. The model remains one forward pass per autoregressive step.
    
    **Ecosystem effects:** Research, tooling, and hardware kernels optimized for decoder-only inference (KV-cache-friendly attention, speculative decoding) concentrate on one dominant pattern. That concentration accelerates iteration even when hybrid architectures might be theoretically attractive for specific workloads.

??? deep-dive "Deep Dive: Scaling Laws, Chinchilla, and What Interviewers Expect"
    Early empirical studies showed that language modeling loss improves as a **power law** in model size, dataset size, and compute when training is done reasonably well. The **Chinchilla** work emphasized that many models were **undertrained**: for a fixed compute budget, smaller models trained on **more tokens** often outperform larger models trained on fewer tokens. The takeaway is not a single universal recipe—data quality, mixture composition, and curriculum all matter—but the conceptual point is stable: **parameters and tokens jointly determine quality**, not parameter count alone.
    
    In interviews, connect scaling laws to **engineering**: more tokens require storage and preprocessing pipelines; larger models require parallelism strategies (tensor, pipeline, expert parallelism). Mention that scaling laws are **empirical** and can shift with architecture changes (for example mixture-of-experts, retrieval augmentation, or improved optimizers).

---

## Code

The following program is **self-contained**: it defines a small GPT, runs a short training loop on repeating character-level text, and generates continuation tokens. It follows **nanoGPT**-style structure: causal attention with a triangular mask, Pre-Norm blocks, GELU MLP, and weight tying between token embedding and LM head.

```python
"""
Minimal GPT (decoder-only) language model — educational, nanoGPT-style.
Dependencies: PyTorch. Run: python gpt_decoder_only.py (or paste into a notebook).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Hyperparameters for a small GPT suitable for CPU or GPU demo."""

    vocab_size: int = 256  # byte-level or ASCII-sized toy vocab
    block_size: int = 128  # maximum context length (positional slots)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention: Q, K, V projections packed in one linear
    for efficiency. Causal mask prevents attending to future positions.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.d_k = config.n_embd // config.n_head
        # fused QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # GPT-2-style: scale residual branch init by depth
        self.c_proj.weight.data.mul_(1.0 / math.sqrt(2 * config.n_layer))
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # register causal mask buffer: (1, 1, T, T) lower-triangular ones
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(c, dim=2)
        # (b, t, nh, dk) -> (b, nh, t, dk)
        q = q.view(b, t, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_k).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward with GELU activation (GPT-2 default)."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        inner = 4 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, inner, bias=False)
        self.c_proj = nn.Linear(inner, config.n_embd, bias=False)
        self.c_proj.weight.data.mul_(1.0 / math.sqrt(2 * config.n_layer))
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Pre-Norm transformer block: LN -> Attn -> residual; LN -> MLP -> residual."""

    def __init__(self, config: GPTConfig) -> None:
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
    """Decoder-only LM: token + position embeddings, stack of blocks, LM head."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: same matrix for token embed and lm_head
        self.wte.weight = self.lm_head.weight

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling from the model distribution."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = GPTConfig(
        vocab_size=256,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter count: {n_params:,}")

    # Character-level language modeling on a repeated string
    training_text = "hello world " * 80
    ids = torch.tensor([ord(ch) for ch in training_text], dtype=torch.long)
    # Create one batch: inputs are all but last token, targets shifted by one
    chunk = ids[: cfg.block_size]
    x = chunk[:-1].unsqueeze(0)
    y = chunk[1:].unsqueeze(0)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(300):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    model.eval()
    prompt_ids = torch.tensor([[ord("h"), ord("e"), ord("l")]], dtype=torch.long)
    out = model.generate(prompt_ids, max_new_tokens=48, temperature=0.9, top_k=32)
    decoded = "".join(chr(int(i)) for i in out[0].tolist())
    print("Sample:", repr(decoded))
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What is the difference between decoder-only, encoder-only, and encoder–decoder Transformers, and which tasks favor each?**
    2. **Write or describe the next-token cross-entropy loss and explain what would happen if you removed the causal mask during training.**
    3. **Explain weight tying between the embedding matrix and the LM head: why is it reasonable and how many parameters does it save?**
    4. **Walk through autoregressive inference step by step. Why is it not fully parallelizable along the generated sequence?**
    5. **What is a KV cache, what tensors does it store, and how does memory scale with batch size, layers, heads, head dimension, and context length?**
    6. **What is in-context learning, and why is it different from fine-tuning?**
    7. **What did GPT-2 change compared to the original Transformer decoder block (Pre-Norm, initialization, activations)?**
    8. **How does temperature affect sampling, and what failure modes appear when temperature is too high or too low?**
    9. **Why do scaling laws matter for budgeting a pre-training run, and what did Chinchilla emphasize about data versus parameters?**
    10. **How would you debug a sudden increase in validation perplexity mid-training: what hypotheses and what measurements would you list first?**

!!! interview "Follow-up Probes"
    - **If the model attends to future positions during training, which positions become ill-posed for next-token prediction?**
    - **Why might grouped-query attention reduce KV memory, and what might you trade off?**
    - **When would you prefer nucleus sampling over greedy decoding for user-facing text?**
    - **How does packing multiple short sequences in one batch interact with causal attention and padding masks?**
    - **Why is softmax attention \(O(T^2)\) in sequence length, and what mitigations exist at long context?**

!!! key-phrases "Key Phrases to Use in Interviews"
    - **“Causal self-attention with a lower-triangular mask enforces an autoregressive factorization.”**
    - **“Training maximizes log-likelihood of each next token given the past.”**
    - **“Autoregressive generation is sequential: each new token conditions on all prior tokens.”**
    - **“The KV cache stores past keys and values to avoid recomputing attention for earlier positions.”**
    - **“Weight tying shares the embedding and output projection, saving \(V \times d\) parameters.”**
    - **“Pre-Norm and residual scaling stabilize optimization in deep decoder stacks.”**
    - **“In-context learning uses attention over demonstration tokens without gradient updates.”**

---

## References

- Radford et al. (2018), *Improving Language Understanding by Generative Pre-Training* (GPT-1)
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2)
- Brown et al. (2020), *Language Models are Few-Shot Learners* (GPT-3)
- Hoffmann et al. (2022), *Training Compute-Optimal Large Language Models* (Chinchilla)
- Vaswani et al. (2017), *Attention Is All You Need*
- Karpathy (2022), [nanoGPT](https://github.com/karpathy/nanoGPT)
