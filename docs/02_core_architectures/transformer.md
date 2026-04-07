# The Transformer

## Why This Matters for LLMs

The Transformer is not merely one architecture among many: it is the computational backbone of virtually every large language model in production. When you study GPT-style decoder stacks, BERT-style encoders, T5-style encoder–decoder systems, or open-weight stacks such as LLaMA and Mistral, you are studying variations on the same residual backbone: alternating attention and feed-forward sublayers, normalization, and a trainable map from tokens to logits. Interviewers treat fluency in this forward pass as a proxy for whether you can reason about scaling laws, debugging, and inference optimizations, because those topics all assume you can anchor discussion in tensors flowing through attention and MLP blocks.

A second reason this chapter matters is that the Transformer replaced recurrence with **parallelizable** pairwise interactions. That design choice governs training throughput (matrix multiplications dominate on GPUs), governs inference memory (the KV cache grows with sequence length and head configuration), and governs research direction (anything that touches long context, sparse attention, or linear attention still compares itself to the full attention baseline). You cannot articulate trade-offs in modern LLM systems without knowing what multi-head attention actually computes and how residuals carry information forward.

Third, the Transformer is where **inductive biases** meet **scale**. Convolutional networks bake in locality; recurrent networks bake in sequential updates. Attention is closer to a relational database query: each position asks every other position for information, weighted by learned compatibility. That flexibility is powerful but data-hungry. Understanding the residual stream, layer normalization choices, and gated feed-forward networks gives you vocabulary for how models both memorize facts in MLP parameters and route information between tokens in attention. Those distinctions show up constantly in mechanistic interpretability and in engineering decisions about pruning, distillation, and quantization.

---

## Architecture Walkthrough

### High-Level Structure

Picture data flowing left to right through a stack of identical **blocks**. Each block contains two trainable sublayers: multi-head self-attention (routing information between tokens) and a position-wise feed-forward network (processing each token independently but with shared weights across positions). Around each sublayer you find residual connections and normalization. The word **block** matters: research and codebases speak in terms of “32-layer Transformer” because depth is measured in these repeated units, not in raw matrix multiplies in isolation.

```
Input token IDs
       │
       ▼
┌──────────────────┐
│ Token Embedding  │
│        +         │
│ Position signal  │  (additive: same width d_model)
└────────┬─────────┘
         ▼
┌────────────────────────────────────────────────────────────┐
│                  Transformer Block × N                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Norm → Multi-Head Self-Attention → Add(residual)   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Norm → Feed-Forward Network → Add(residual)       │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
         │
         ▼
   Final Norm (often)
         │
         ▼
   Linear projection to vocabulary logits (for language modeling)
```

The diagram is schematic: some implementations swap the order of Norm and sublayer (Pre-Norm versus Post-Norm), and decoder-only models apply a causal mask inside attention. The **block** abstraction still holds: attention mixes tokens; the feed-forward sublayer transforms each mixed token vector in isolation.

### Input Embeddings

Each token identifier \(w_i\) indexes a row of an embedding matrix \(E \in \mathbb{R}^{V \times d_{\text{model}}}\), where \(V\) is vocabulary size. Positional information is added so that the model can distinguish order (attention alone is permutation-equivariant; see the positional encoding chapter). The conventional construction is **sum**, not concatenation:

\[
\mathbf{x}_i = E[w_i] + \mathbf{p}_i
\]

where \(\mathbf{p}_i\) is either a sinusoidal vector, a learned vector for index \(i\), or an implicit position signal injected later (as with RoPE, which acts on queries and keys inside attention).

!!! math-intuition "In Plain English"
    Summation keeps the vector width fixed at \(d_{\text{model}}\). Concatenating token and position embeddings would double the width going into the first attention layer unless you add a projection, which would mix dimensions in a less interpretable way. Addition forces the network to use the **same** channel budget for both lexical identity and location: every dimension can participate in carrying both kinds of information, and the downstream layers learn how to read the combined code.

If the batch has shape \((B, T)\) for batch size \(B\) and sequence length \(T\), after embedding lookup and position addition the tensor has shape \((B, T, d_{\text{model}})\).

### Multi-Head Self-Attention

The dedicated page develops attention in full detail. Here is the **dimension contract** you should memorize for interviews. Let hidden size be \(d_{\text{model}}\), batch \(B\), sequence length \(T\). Input \(X\) has shape \((B, T, d_{\text{model}})\). Learned projections produce queries, keys, and values:

\[
Q = X W^Q,\quad K = X W^K,\quad V = X W^V
\]

with \(W^Q, W^K, W^V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) in the single big-matrix layout (implementation detail: one fused linear for QKV is common). For \(h\) heads, each head uses a slice of dimension \(d_k = d_{\text{model}} / h\). Reshape \(Q\) to \((B, h, T, d_k)\), same for \(K\) and \(V\). Attention scores are \(\frac{Q K^\top}{\sqrt{d_k}}\), softmax over the key position, multiply by \(V\), reshape back to \((B, T, d_{\text{model}})\), then apply output projection \(W^O\).

!!! math-intuition "In Plain English"
    Think of each head as running a **small** attention with its own geometry: the \(\sqrt{d_k}\) scaling keeps dot products from growing so large that softmax saturates. The output projection \(W^O\) mixes head outputs back into the shared hidden width. The net effect is still a map from \((B, T, d_{\text{model}})\) to \((B, T, d_{\text{model}})\), but internally the model can implement several relationship patterns in parallel.

### Feed-Forward Network

Each token vector \(x \in \mathbb{R}^{d_{\text{model}}}\) passes through the same two-layer network. The original Transformer used ReLU:

\[
\text{FFN}(x) = W_2 \, \text{ReLU}(W_1 x + b_1) + b_2
\]

with \(W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}\), \(W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}\), and biases matching inner and outer shapes.

!!! math-intuition "In Plain English"
    This sublayer is **position-wise**: it does not mix tokens. The same weights apply at every time step. Intuitively, after attention has gathered context into each position’s vector, the FFN decides **what to do with that information** at that position—often described informally as where factual lookups and nonlinear transforms live. The inner width \(d_{\text{ff}}\) is typically several times larger than \(d_{\text{model}}\) to give the model enough capacity for elementwise nonlinear computation.

#### The Math: SwiGLU Variant

Modern LLaMA-class models replace the single hidden activation with a **gated** linear unit. SwiGLU uses three projections and the SiLU (sigmoid-linear) activation on one branch:

\[
\text{SwiGLU}(x) = W_2 \bigl( \text{SiLU}(W_1 x) \odot W_3 x \bigr)
\]

where \(\odot\) is elementwise product, \(\text{SiLU}(t) = t \cdot \sigma(t)\), and bias terms may be omitted (as in many open-weight checkpoints).

!!! math-intuition "In Plain English"
    You can read SwiGLU as **two** pathways into the inner width: one passes through a smooth gate (SiLU), one acts as a raw linear source. The elementwise product lets the gate suppress or amplify channels before the final down-projection. Compared with ReLU, the pathway is strictly more expressive per parameter at the cost of an extra matrix multiply. Engineering practice often chooses an inner dimension so total FLOPs remain comparable to the older \(4 d_{\text{model}}\) ReLU FFN.

!!! example "Worked Example: FFN Dimensions"
    Take \(d_{\text{model}} = 512\) and a ReLU-style inner width \(d_{\text{ff}} = 2048\). A token vector \(x\) has shape \((512,)\). The weight \(W_1\) has shape \((2048, 512)\): multiplying \(W_1 x\) yields a vector of length \(2048\). Applying ReLU keeps shape \((2048,)\). The matrix \(W_2\) has shape \((512, 2048)\), so \(W_2 h\) returns to \((512,)\). Parameter counts for the two matrices ignoring biases: \(W_1\) has \(2048 \times 512 = 1{,}048{,}576\) parameters, \(W_2\) has \(512 \times 2048 = 1{,}048{,}576\) parameters, totaling **2,097,152** parameters for the classical FFN. If you add biases, add \(2048 + 512\) more. For SwiGLU with the same inner width, you introduce a third up-projection \(W_3\) of the same shape as \(W_1\), which adds another **1,048,576** weights before accounting for optional biases, because the gated layer needs both an “up” and a “gate” pathway into the inner dimension.

### Residual Connections

Each sublayer is wrapped with an additive skip:

\[
x_{\ell+1} = x_\ell + f_\ell(x_\ell)
\]

in the Post-Norm formulation, or in Pre-Norm:

\[
x_{\ell+1} = x_\ell + f_\ell(\text{Norm}(x_\ell))
\]

!!! math-intuition "In Plain English"
    The residual path means the gradient can flow along the **identity** route. Even if \(f_\ell\) is poorly initialized or temporarily unhelpful, the layer can behave like the identity mapping plus a small correction. That stability matters when stacks reach dozens or hundreds of layers. In conversation, saying “the network can always fall back to passing the input forward” is exactly this identity pathway.

!!! example "Worked Example: Residual Add"
    Let \(x = [1.0, 0.0]\) and let a sublayer return \(f(x) = [0.5, -0.25]\). The residual output is \([1.5, -0.25]\). If later training drives \(f(x)\) toward zero, the output approaches \([1.0, 0.0]\): the block can delete its own influence without destabilizing the scale of activations along the main path.

### Layer Normalization

LayerNorm normalizes across features for each token:

\[
\text{LN}(x)_j = \gamma_j \cdot \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta_j
\]

where \(\mu\) and \(\sigma^2\) are mean and variance of \(x\) across its \(d_{\text{model}}\) coordinates for that token, and \(\gamma\), \(\beta\) are learnable.

RMSNorm drops centering and keeps root-mean-square scaling:

\[
\text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i,\quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}
\]

Pre-Norm places normalization **before** each sublayer; Post-Norm places it **after** the residual add. Most large decoder-only models use Pre-Norm with RMSNorm.

!!! math-intuition "In Plain English"
    LayerNorm fixes the scale of vectors entering each sublayer, which reduces internal covariate shift inside depth. RMSNorm is cheaper (no mean) and often works as well in Transformers because the residual stream already centers information differently than batch-normalized CNNs. When you say “Pre-Norm,” you mean “normalize before the attention and FFN so those modules always see well-scaled inputs.”

!!! example "Worked Example: LayerNorm"
    Take \(x = [1, 3, 5, 7]\). The mean is \(\mu = (1+3+5+7)/4 = 4\). The variance is \(\sigma^2 = \frac{1}{4}((1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2) = \frac{9+1+1+9}{4} = 5\). The standard deviation is \(\sigma = \sqrt{5} \approx 2.2360679775\). Subtract the mean: \([-3, -1, 1, 3]\). Divide by \(\sigma\): approximately \([-1.3416407865, -0.4472135955, 0.4472135955, 1.3416407865]\). With \(\gamma = [1,1,1,1]\) and \(\beta = [0,0,0,0]\), that normalized vector is the LayerNorm output. If you set \(\gamma = [2,2,2,2]\) and \(\beta = [0.5,0.5,0.5,0.5]\), you double then shift: approximately \([-2.183281573, -0.394427191, 1.394427191, 3.183281573]\).

### The Residual Stream View

Modern analysis treats depth as successive **writes** into a running sum. The embedding initializes the stream; each attention head and each FFN MLP adds a vector into that sum. Superposition is the observation that many features can be encoded across the same dimensions because linear structures approximate sparse interactions at scale. This mental model helps when you read about **logit lens** techniques or circuit-style analyses: layers move representations along directions that partially overlap.

### Encoder vs Decoder vs Encoder–Decoder

- **Encoder-only** (BERT family): bidirectional self-attention over the whole sequence; good for classification and embedding tasks; training often uses masked language modeling.
- **Decoder-only** (GPT family, LLaMA): causal self-attention so position \(i\) cannot attend to positions \(j > i\); autoregressive language modeling is the dominant pretraining objective.
- **Encoder–decoder** (T5, BART): encoder sees the source with bidirectional attention; decoder is causal and includes **cross-attention** where queries come from the decoder and keys and values come from encoder states.

### Key Hyperparameters Table

| Parameter | GPT-2 Small | GPT-3 175B | LLaMA-2 7B | LLaMA-2 70B |
| --- | --- | --- | --- | --- |
| Layers \(N\) | 12 | 96 | 32 | 80 |
| Hidden size \(d_{\text{model}}\) | 768 | 12288 | 4096 | 8192 |
| Attention heads | 12 | 96 | 32 | 64 |
| Head dim \(d_k\) | 64 | 128 | 128 | 128 |
| FFN inner (typical) | 3072 (4×) | \(4 \times d_{\text{model}}\) | 11008 (SwiGLU) | 28672 (SwiGLU) |
| KV sharing | None | None | GQA (8 KV heads) | GQA (8 KV heads) |
| Context length (reported) | 1024 | 2048 | 4096 | 4096 |

Figures for GPT-2 Small and GPT-3 follow OpenAI’s model card conventions; LLaMA-2 numbers follow Meta’s technical report. Exact checkpoint details can differ slightly by release and fine-tune, but the magnitudes are what you need for design intuition.

---

## Anatomy of a Forward Pass

!!! example "Traced Forward Pass"
    Suppose vocabulary index \(42\) maps to the word “Hello” at sequence position \(0\), with batch size \(1\), \(d_{\text{model}} = 256\), and \(4\) heads so \(d_k = 64\). The embedding lookup reads row \(42\) of \(E\), yielding \(x_0 \in \mathbb{R}^{256}\). Add positional vector \(p_0 \in \mathbb{R}^{256}\). The tensor entering block 1 has shape \((1, 1, 256)\) if we isolate that token; in a longer sentence it is \((1, T, 256)\). Inside block 1, Pre-Norm RMSNorm scales the vector. Attention computes \(Q, K, V\) each of width \(256\) before reshaping to heads. Attention probabilities have shape \((1, 4, T, T)\) without causal masking, or a triangular mask for decoder models. The attention output merges heads and projects, producing a residual update of shape \((1, T, 256)\) added back to the stream. The FFN expands to inner width (for example \(704\) or \(2048\) depending on architecture), applies SwiGLU nonlinearity, projects down to \(256\), and adds another residual. After \(N\) blocks, a final norm stabilizes scale, and the language modeling head multiplies by \(W_{\text{lm}} \in \mathbb{R}^{V \times 256}\) to produce logits of shape \((1, T, V)\). The row for position \(0\) gives a distribution over the next token after “Hello” in a causal model, or masked-token predictions in encoder-only training.

---

??? deep-dive "Deep Dive: Parameter Accounting vs. Inference Memory"
    Parameter count answers how many weights you store on disk and load into GPU high-bandwidth memory. Inference memory also includes activations, optimizer states during training, and KV caches during autoregressive generation. A lean mental model: parameters scale roughly with layers times width squared for attention projections and with layers times width times inner width for MLPs. KV cache scales with batch size, number of layers, sequence length, and the number of key-value heads times head dimension. Grouped-query attention reduces KV head count without changing query head count, which is why it appears in large deployed models.

---

??? deep-dive "Deep Dive: Why Pre-Norm Dominates at Depth"
    Post-Norm places normalization after the residual addition. Gradient paths still exist, but early training can be more brittle because the normalization sits on the main residual highway. Pre-Norm routes each sublayer through normalization **before** the heavy operation, which empirically improves optimization for deep stacks. When you read a model card that says “RMSNorm + SwiGLU + RoPE,” you are seeing the modern recipe for stable depth.

---

## Code

The listing below implements a compact decoder-style stack: RMSNorm, multi-head self-attention with optional causal mask, SwiGLU feed-forward, Pre-Norm residuals, learned token embeddings, and learned absolute positions for clarity. Production models often swap learned positions for RoPE.

```python
"""
Transformer encoder stack (decoder-style with causal mask) in PyTorch.
Educational layout: explicit modules, shape comments, deterministic demo.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — scales by RMS without mean centering."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable per-dimension scale (gain)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim) — normalize over last dimension
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV projection."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Single linear for Q, K, V stacked — matches common fused kernels
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, d_model)
        b, t, d_model = x.shape
        qkv = self.W_qkv(x)  # (B, T, 3*d_model)
        qkv = qkv.reshape(b, t, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).reshape(b, t, d_model)
        return self.W_o(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: SiLU(W1 x) ⊙ W3 x, then W2 down."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        if d_ff is None:
            # Common heuristic: ~8/3 * d_model then round up for tensor cores
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Pre-Norm block: attention sublayer then FFN sublayer, both residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerEncoder(nn.Module):
    """Token + learned positional embeddings, stack of blocks, logits head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_len: int = 512,
        d_ff: int | None = None,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return self.head(x)


if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 1000
    d_model = 256
    n_heads = 4
    n_layers = 4
    batch = 2
    seq_len = 32

    model = TransformerEncoder(vocab_size, d_model, n_heads, n_layers)
    tokens = torch.randint(0, vocab_size, (batch, seq_len))
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(
        1, 1, seq_len, seq_len
    )

    logits = model(tokens, mask=causal)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"tokens {tuple(tokens.shape)} -> logits {tuple(logits.shape)}")
    print(f"parameters: {n_params:,}")

    with torch.no_grad():
        baseline = model(tokens, mask=causal)
        for blk in model.layers:
            blk.ffn.w1.weight.zero_()
            blk.ffn.w3.weight.zero_()
        degraded = model(tokens, mask=causal)
        mean_abs_diff = (baseline - degraded).abs().mean().item()
        print(f"mean |logit diff| after zeroing SwiGLU up-projs: {mean_abs_diff:.6f}")
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Walk the forward pass**: Starting from token IDs, narrate shapes through embedding, one Transformer block, and the language modeling head. Interviewers want batch-major tensors, head reshaping, and where softmax sits.
    *Answer:* Token IDs of shape `(B, T)` embed to `(B, T, d_model)` and combine with positional signal; inside one block, Pre-Norm scales the stream, attention forms `Q, K, V` each `(B, T, d_model)`, reshapes to `(B, H, T, d_k)`, computes scores `(B, H, T, T)`, applies softmax **over the key dimension** (last axis), mixes `V`, merges heads, and adds a residual; the FFN applies the same MLP at each position (e.g. SwiGLU up-project then down) and adds again. After `N` blocks, final norm and the LM head map `(B, T, d_model)` → `(B, T, V)` logits; softmax appears in the loss or at sampling, not necessarily inside the block output.
    2. **Pre-Norm versus Post-Norm**: Give the residual formulas and explain which training stability story you believe for deep models.
    *Answer:* Post-Norm: \(x_{\ell+1} = \mathrm{LN}(x_\ell + f_\ell(x_\ell))\); Pre-Norm: \(x_{\ell+1} = x_\ell + f_\ell(\mathrm{LN}(x_\ell))\). Pre-Norm keeps a cleaner identity path through \(x_\ell\) into the next layer, so gradients and activations tend to behave more predictably in stacks of 32–80+ layers, which is why most large LMs use it; Post-Norm can work but often demands more careful init and was more common in the original paper-era setups.
    3. **Why attention is \(O(T^2)\)**: Tie it to the attention matrix over positions and mention implications for long-context systems.
    *Answer:* For each head you form an \(T \times T\) score matrix (each query position against every key position), so FLOPs and memory for that matrix scale as \(O(T^2 d_k)\) before the multiply with \(V\). At \(T=128\text{k}\) this dominates HBM and latency, which is why production systems use chunked attention, sparse patterns, or alternative layers (SSMs) for very long contexts.
    4. **SwiGLU versus ReLU FFN**: Count matrices, describe the gating intuition, and mention parameter overhead.
    *Answer:* Classic FFN uses two matrices (up and down); SwiGLU uses **three** up-projections: \(\mathrm{SiLU}(W_1 x) \odot W_3 x\) then \(W_2\), so you pay roughly **one extra** large matrix versus ReLU-FFN at the same inner width. The gate lets the model **suppress or amplify** channels per token before the down-projection, which empirically improves quality per FLOP on many LLM recipes (LLaMA family).
    5. **RMSNorm versus LayerNorm**: State the formulas and why RMSNorm saves compute.
    *Answer:* LayerNorm subtracts the mean over features and divides by standard deviation, then applies \(\gamma, \beta\). RMSNorm skips centering and only scales by the root-mean-square of features: fewer reductions (no mean), slightly cheaper, and in Transformers often matches LayerNorm quality because the residual stream already mixes statistics differently than batchnorm in CNNs.
    6. **Encoder-only versus decoder-only**: Masking, objectives, and a product example for each (BERT versus GPT).
    *Answer:* Encoder-only uses **full** self-attention (every token sees all others) and objectives like MLM—**BERT** for classification/embedding. Decoder-only uses a **causal** mask (position \(i\) only attends to \(j \le i\)) and next-token LM—**GPT**/ChatGPT-style generation. You cannot swap them blindly: bidirectional encoders are not trained for a left-to-right generative factorization.
    7. **Residual stream**: Explain additive updates and why identity pathways help optimization.
    *Answer:* Each sublayer adds its output to the running hidden state: \(h \leftarrow h + f(h)\), so the default “signal” is passed through unchanged and \(f\) learns a **delta**. Backpropagation gets short paths through the identity that avoid every layer’s Jacobian chaining, which stabilizes deep networks and lets early layers be skipped in effect if \(f \to 0\).
    8. **Where factual knowledge lives**: Give the nuanced view that MLP layers store and transform information after attention routes context.
    *Answer:* Attention **routes** information between positions (who talks to whom); position-wise FFNs apply large-capacity nonlinear maps and are often linked to **key–value** style storage in interpretability work. In practice both sublayers interact: attention gathers relevant context, then MLPs integrate and transform it—neither alone explains all “facts.”
    9. **Scaling width versus depth**: Discuss how both change parameter count and activation memory differently.
    *Answer:* **Width** (\(d_{\text{model}}\)) grows attention projections roughly as \(O(d^2)\) per layer and linearly increases activations `(B,T,d)` and KV cache per head dimension. **Depth** (more layers) stacks more such blocks—linear in layer count for params and activations, but deeper stacks improve representational power while making optimization and inference latency scale with `L`. Wider models often utilize GPU tensor cores more efficiently per step than extremely deep thin models at fixed budget.
    10. **KV cache**: Define what is cached per layer during autoregressive decoding and why grouped-query attention reduces memory.
    *Answer:* After processing a prefix, each layer stores **key and value** tensors for all past time steps (per batch, per KV head): without caching you would recompute them for every new token. GQA uses **fewer** distinct KV heads than query heads (e.g. 8 KV vs 32 Q) and repeats/broadcasts KV to match Q heads, cutting KV tensor footprint by the head ratio (often ~4×) with modest quality impact.

!!! interview "Follow-up Probes"
    - “How does gradient flow through Pre-Norm blocks?” Expect discussion of identity shortcuts and softmax saturation.
    - “Why multiply by \(\sqrt{d_k}\)?” Expect variance stabilization argument.
    - “How would you modify the block for cross-attention?” Expect Q from decoder, K and V from encoder, and mask shapes \((B,1,T_{\text{dec}},T_{\text{enc}})\).
    - “What changes in inference if you use weight tying between embeddings and logits?” Expect parameter reduction and training regularity discussion.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Pre-Norm stabilizes optimization in deep stacks because sublayers see normalized inputs.”
    - “Attention mixes tokens; the FFN processes each position with shared MLP weights.”
    - “SwiGLU adds a gated branch, improving expressivity per inner dimension.”
    - “The residual stream carries an additive backbone; layers write corrections.”
    - “Decoder-only models use causal masking to prevent peeking at future tokens.”

---

## References

- Vaswani et al., *Attention Is All You Need* (NeurIPS 2017)
- Shazeer, *GLU Variants Improve Transformer* (2020)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
- Elhage et al., *A Mathematical Framework for Transformer Circuits* (2021)
- The Illustrated Transformer — Jay Alammar: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
