# Mathematics of Attention

## Why This Matters for LLMs

Attention is the single most important idea in modern AI. It answers the question: "When processing one word, which other words should I pay attention to?" In the sentence "The cat sat on the mat because it was tired", attention helps the model figure out that "it" refers to "cat" — by giving "cat" a high attention weight when processing "it".

Every decoder-only LLM (GPT-class), encoder-only model (BERT-class), and encoder–decoder (T5, translation) **is** attention stacks plus MLPs plus norms. **Scaled dot-product attention** \\(\\text{softmax}(QK^\\top/\\sqrt{d_k})V\\) is the **atomic** operation interviewers whiteboard first. Understanding **scaling**, **masks**, **multi-head** splitting, and **\\(O(T^2)\\)** cost is table stakes for systems roles (KV cache, FlashAttention) and research roles (linear attention, state-space layers). This page ties **Bahdanau → dot product → scaled softmax → Transformer** into one quantitative thread.

!!! tip "Notation Help"
    - \\(Q\\), \\(K\\), \\(V\\) are **Query**, **Key**, **Value** matrices (learned projections of input)
    - \\(d_k\\) is the **dimension of key vectors** (often 64 or 128 per head)
    - \\(T\\) is the **sequence length** (number of tokens)
    - **MACs** = **Multiply-Accumulate operations** (one MAC = one multiply + one add)

---

## Core Concepts

### From Bahdanau to Dot-Product

Bahdanau (additive) attention:

\\[
e_{t,j} = \\mathbf{v}^\\top \\tanh(W_1 \\mathbf{h}_j^{\\text{enc}} + W_2 \\mathbf{s}_t)
\\]

**Dot-product** attention (Luong / Transformer style) replaces the small MLP score with **compatibility** between **query** and **key** vectors:

\\[
e_{i,j} = \\mathbf{q}_i^\\top \\mathbf{k}_j
\\]

!!! math-intuition "In Plain English"
    - **Query** \\(\\mathbf{q}_i\\): "what I am looking for at position \\(i\\)."
    - **Key** \\(\\mathbf{k}_j\\): "what is offered at position \\(j\\)."
    - **Dot product**: similarity if vectors are unit-norm—large positive → **align**; negative → anti-align.

### Scaled Dot-Product Attention

??? tip "🎯 Think of it like… A Library Search"
    Imagine you walk into a library looking for books about **machine learning** (that's your **Query**). Every book on the shelf has a label on its spine — "Statistics", "Programming", "Machine Learning", "Cooking" (those are the **Keys**). You compare your query against each label (the **dot product**):

    - "Machine Learning" label → HIGH match score ✅
    - "Statistics" label → medium match score
    - "Cooking" label → LOW match score ❌

    Then you **blend** the contents (**Values**) of all books, weighted by match quality: mostly the ML book, a bit of the statistics book, almost none of the cooking book. The result is a **custom-mixed summary** tailored to your query.

    **That's attention!** Query = what you want. Key = what each source offers. Value = the actual content. Score = how well they match. The weighted blend = the output.

The attention formula looks intimidating but is just THREE steps:

1. **SCORE:** For each word, compare its "query" (what am I looking for?) with every other word's "key" (what do I offer?) using dot products → gives a score matrix
2. **NORMALIZE:** Turn scores into weights using softmax (so they sum to 1) and scale by \\(\\sqrt{d_k}\\) to prevent extreme values
3. **BLEND:** Use the weights to take a weighted average of "value" vectors → each word gets a blended representation that incorporates relevant context

\\[
\\text{Attention}(Q, K, V) = \\text{softmax}\\!\\Bigl(\\frac{QK^\\top}{\\sqrt{d_k}}\\Bigr) V
\\]

<div class="equation-breakdown" markdown>
#### 📝 Building the Equation Piece by Piece

| Step | Math | What it does | Shape |
|------|------|-------------|-------|
| **1. Score** | \\(QK^\\top\\) | Each query row dot-products with each key row → similarity matrix | \\(T \\times T\\) |
| **2. Scale** | \\(\\div \\sqrt{d_k}\\) | Divide by square root of key dimension to prevent huge numbers | \\(T \\times T\\) |
| **3. Normalize** | \\(\\text{softmax}(\\cdots)\\) | Convert each row to weights that sum to 1 (probabilities) | \\(T \\times T\\) |
| **4. Blend** | \\(\\cdots \\times V\\) | Each row of weights blends the value vectors | \\(T \\times d_v\\) |

**Input shapes:** \\(Q \\in \\mathbb{R}^{T \\times d_k}\\), \\(K \\in \\mathbb{R}^{T \\times d_k}\\), \\(V \\in \\mathbb{R}^{T \\times d_v}\\). **Output:** \\(\\mathbb{R}^{T \\times d_v}\\).

**In one sentence:** "For each position, compute a weighted average of all value vectors, where the weights come from how well that position's query matches each position's key."
</div>

!!! math-intuition "In Plain English"
    - \\(QK^\\top\\): **all-pairs** similarity scores between queries (rows of \\(Q\\)) and keys (rows of \\(K\\)).
    - **Softmax** turns each query row into **weights** over positions.
    - **Multiply by \\(V\\)**: **blend** value vectors—**differentiable** weighted sum.

### Why \\(\\sqrt{d_k}\\)? — Numerical Stabilization

??? tip "🎯 Think of it like… Volume Control on a Speaker"
    Imagine 512 people each shouting a random number between -1 and 1. If you **add up** all their shouts, the total could easily be ±20 or more (even though individual shouts are small). That loud total would "blow out" a speaker (softmax) — it would only hear the loudest voice and ignore everyone else.

    **Dividing by \\(\\sqrt{512} ≈ 22.6\\)** is like turning down the volume so the speaker can hear **all** the voices, not just the loudest one. Without this volume knob, softmax becomes essentially **argmax** — only one word gets attention, and gradients to all other positions vanish.

Why divide by \\(\\sqrt{d_k}\\)? Here's the intuition: imagine adding up 512 random numbers. The sum can be huge! Before softmax, huge numbers lead to outputs like \\([0.999, 0.001, 0.000, 0.000]\\) — almost one-hot. That means the model can only "pay attention" to ONE word, which is too rigid. Dividing by \\(\\sqrt{512} \\approx 22.6\\) brings the numbers back to a reasonable range, so softmax can spread attention across multiple words.

If components of \\(\\mathbf{q}, \\mathbf{k}\\) are i.i.d. with variance 1 and mean 0, then

\\[
\\mathbb{E}[\\mathbf{q}^\\top \\mathbf{k}] = 0,\\quad \\mathrm{Var}(\\mathbf{q}^\\top \\mathbf{k}) = d_k
\\]

<div class="equation-breakdown" markdown>
#### 📝 Why Variance = \\(d_k\\) (The Core Statistical Argument)

Let's work through this step by step. The dot product \\(\\mathbf{q}^\\top\\mathbf{k} = \\sum_{i=1}^{d_k} q_i k_i\\) is a sum of \\(d_k\\) terms.

1. **Each term** \\(q_i k_i\\) is a product of two independent random variables, each with mean 0 and variance 1
2. **Mean of a product:** \\(\\mathbb{E}[q_i k_i] = \\mathbb{E}[q_i] \\cdot \\mathbb{E}[k_i] = 0 \\times 0 = 0\\)
3. **Variance of a product:** \\(\\text{Var}(q_i k_i) = \\mathbb{E}[q_i^2]\\mathbb{E}[k_i^2] = 1 \\times 1 = 1\\)
4. **Sum of \\(d_k\\) independent terms:** variances add! \\(\\text{Var}(\\sum_i q_i k_i) = d_k \\times 1 = d_k\\)
5. **Standard deviation** of the dot product \\(= \\sqrt{d_k}\\)

**So the typical magnitude** of dot products grows like \\(\\sqrt{d_k}\\). For \\(d_k = 512\\), that's \\(\\sqrt{512} ≈ 22.6\\). Dividing by \\(\\sqrt{d_k}\\) rescales the variance back to \\(\\approx 1\\), keeping softmax in a healthy gradient regime.
</div>

!!! math-intuition "In Plain English"
    Each product \\(q_i k_i\\) has variance 1 under the i.i.d. unit-variance assumption; summing \\(d_k\\) **independent** terms gives variance **\\(d_k\\)** (not \\(d_k^2\\)). So typical dot-product magnitudes scale with **\\(\\sqrt{d_k}\\)**—the **scale** of logits before softmax.

Thus dot products **grow** like \\(\\sqrt{d_k}\\) in typical magnitude. **Softmax** of huge logits \\(\\to\\) **nearly one-hot** \\(\\to\\) **vanishing gradients** through other positions. Dividing by \\(\\sqrt{d_k}\\) **re-scales** logits to \\(\\mathrm{Var} \\approx 1\\).

!!! example "Numerical Demo: \\(d_k = 512\\)"
    Rough i.i.d. heuristic: **unscaled** dot \\(\\mathbf{q}^\\top\\mathbf{k}\\) has **standard deviation** \\(\\approx \\sqrt{d_k} = \\sqrt{512} \\approx 22.6\\). Softmax on five logits around **22** vs **0** is effectively **argmax**—gradients through non-argmax positions \\(\\approx 0\\).

    **Scaled** logits use \\(\\mathbf{q}^\\top\\mathbf{k}/\\sqrt{512}\\): typical std \\(\\approx 22.6 / 22.6 = 1\\). Softmax is **smoother**; training signal reaches multiple positions.

    (Real networks learn non-i.i.d. statistics; **learned** LayerNorm and projections matter—but the **variance argument** is the textbook reason for the scale.)

### Softmax and Temperature

\\[
\\text{softmax}(z_i; \\tau) = \\frac{\\exp(z_i / \\tau)}{\\sum_j \\exp(z_j / \\tau)}
\\]

!!! math-intuition "In Plain English"
    - \\(\\tau \\to 0^+\\): **sharper** distribution (approaches one-hot).
    - \\(\\tau \\to \\infty\\): **uniform** mixing.
    - **Decoding temperature** in LMs applies the same idea to **next-token** logits—not identical to attention temperature, but the **same function family**.

#### 🔬 Try It Yourself: Interactive Softmax Temperature

<div class="viz-container" id="softmax-viz" markdown>
*Loading interactive softmax visualizer…*
</div>

---

## Worked Example: Four Tokens, \\(d_k = 4\\)

**Tokens:** `["The", "cat", "sat", "."]` — indices \\(0..3\\).

### 1. Toy \\(Q\\), \\(K\\), \\(V\\) (each \\(4 \\times 4\\))

Use **simple integers** (pedagogical, not trained weights):

\\[
Q = 2 I_4 = \\begin{bmatrix}
2&0&0&0\\\\0&2&0&0\\\\0&0&2&0\\\\0&0&0&2
\\end{bmatrix}
\\]

\\[
K = \\begin{bmatrix}
1&1&0&0\\\\
1&0&1&0\\\\
0&1&1&0\\\\
0&0&1&1
\\end{bmatrix},
\\quad
V = \\begin{bmatrix}
1&0&0&0\\\\
0&1&0&0\\\\
0&0&1&0\\\\
0&0&0&1
\\end{bmatrix}
\\]

(Here \\(V=I\\) so **output row** is literally the softmax **weight vector** over positions—easy to read.)

### 2. Compute \\(S = Q K^\\top\\) (every cell)

\\(K^\\top\\) is:

\\[
K^\\top = \\begin{bmatrix}
1&1&0&0\\\\
1&0&1&0\\\\
0&1&1&0\\\\
0&0&1&1
\\end{bmatrix}
\\]

Since \\(Q = 2I\\), **\\(S = Q K^\\top = 2 K^\\top\\)**:

\\[
S = \\begin{bmatrix}
2&2&0&0\\\\
2&0&2&0\\\\
0&2&2&0\\\\
0&0&2&2
\\end{bmatrix}
\\]

**Cell check (row 0 · col 2):** Row0 of \\(Q\\) is \\([2,0,0,0]\\), col2 of \\(K^\\top\\) is \\([0,1,1,0]^\\top\\), dot \\(=0\\). Matches \\(S_{0,2}=0\\).

### 3. Scale: \\(S / \\sqrt{d_k} = S / 2\\)

\\[
\\frac{S}{2} = \\begin{bmatrix}
1&1&0&0\\\\
1&0&1&0\\\\
0&1&1&0\\\\
0&0&1&1
\\end{bmatrix}
\\]

### 4. Softmax **row-wise** — show **row 0** fully

Row 0 logits: \\([1, 1, 0, 0]\\).

\\[
e^1 \\approx 2.718,\\quad e^0 = 1
\\]

Numerators: \\([2.718,\\, 2.718,\\, 1,\\, 1]\\). Sum \\(\\approx 7.436\\).

\\[
w_{0,0} \\approx 2.718/7.436 \\approx 0.366,\\quad
w_{0,1} \\approx 0.366,\\quad
w_{0,2} \\approx 0.134,\\quad
w_{0,3} \\approx 0.134
\\]

(Weights sum to 1; positions 0 and 1 tie for **highest** mass because logits tied at 1.)

### 5. Output row 0: \\(w_0 V\\)

Because \\(V=I\\), **output row 0** \\(\\approx [0.366,\\, 0.366,\\, 0.134,\\, 0.134]\\)—the **context vector** for token "The" is a **blend** of positional value vectors with those weights.

!!! math-intuition "In Plain English"
    - Row \\(i\\) of the attention output is: "**re-read** all positions, mixing their **values** by how well **keys** match **query** \\(i\\)."
    - With \\(V=I\\), you literally see the **attention distribution** as the row vector.

#### 🔬 Try It Yourself: Interactive Attention Heatmap

<div class="viz-container" id="attention-heatmap" markdown>
*Loading interactive attention heatmap…*
</div>

---

## Causal (Autoregressive) Masking

??? tip "🎯 Think of it like… A No-Spoilers Rule"
    Imagine you're writing a mystery novel. When writing Chapter 5, you can reference anything from Chapters 1–4 (you already wrote those). But you can **NOT** reference the ending in Chapter 10 — that would be cheating!

    **Causal masking** enforces this rule in attention: position 5 can "attend to" (look at) positions 1–4, but NOT positions 6, 7, 8… The model can't peek at future words because at generation time, those words don't exist yet. In the attention score matrix, future positions are set to \\(-\\infty\\) before softmax, which becomes exactly **0 weight** — making future words effectively invisible.

Masking prevents the model from "cheating" by looking at future words. In a causal (GPT-style) model, when predicting word 5, the model should only see words 1–4. The mask sets future positions to \\(-\\infty\\) before softmax, which converts to 0 weight — effectively making future words invisible.

For **decoder** self-attention, position \\(i\\) must **not** depend on \\(j > i\\). Take the **scaled** score matrix \\(Z = S/\\sqrt{d_k}\\). **Causal mask** sets \\(Z_{i,j} = -\\infty\\) for \\(j > i\\) **before** softmax.

!!! example "Mask Walkthrough (same \\(Z\\) as above)"
    For **row 3** (token "."), without mask, logits were \\([0,0,1,1]\\). With **causal** constraint, positions \\(j>3\\) do not exist; row 3 only has \\(j \\le 3\\). For **row 0**, mask out \\(j>0\\): keep only column 0 → softmax over a **single** finite logit → weight \\(1\\) on self (often combined with **causal** + **additive** pos encodings in real models).

    **Typical 4×4 causal \\(Z'\\)** (set upper triangle to \\(-\\infty\\); shown symbolically):

    \\[
    Z'_{i,j} = \\begin{cases}
    Z_{i,j} & j \\le i \\\\
    -\\infty & j > i
    \\end{cases}
    \\]

    After softmax, **masked** positions have **weight 0**—no information flows from the future.

!!! math-intuition "In Plain English"
    - **\\(-\\infty\\)** + softmax = **0** probability—clean masking without "almost zero" numerical hacks (implementation uses large negative floats).

### Multi-Head Intuition

??? tip "🎯 Think of it like… A Team of Analysts"
    Imagine you're a CEO reading a company report. You ask **four** different analysts to read it:

    - **Analyst 1 (Syntax):** "Who does what to whom?" — focuses on subject-verb-object relationships
    - **Analyst 2 (Semantics):** "What words mean similar things?" — finds synonym clusters
    - **Analyst 3 (Local Context):** "What's happening nearby?" — looks at adjacent words
    - **Analyst 4 (Long-Range):** "What does this connect to from earlier?" — finds distant references

    Each analyst writes their own summary. Then a secretary (**output projection**) combines their summaries into one cohesive brief.

    **That's multi-head attention!** Each "head" is an analyst with its own perspective (learned Q, K, V projections). They run in parallel, then their outputs are concatenated and projected back.

Multi-head attention is like having multiple "perspectives". One head might focus on syntactic relationships (subject–verb), another on semantic similarity (synonyms), another on positional proximity (nearby words). Each head has its own \\(Q\\), \\(K\\), \\(V\\) projections, runs attention independently, and the results are concatenated and projected back. Think of it as a team of analysts, each looking at the data through a different lens.

\\[
\\text{head}_h = \\text{softmax}\\!\\Bigl(\\frac{Q_h K_h^\\top}{\\sqrt{d_k}}\\Bigr) V_h, \\quad Q_h = X W_h^Q,\\ \\ldots
\\]

<div class="equation-breakdown" markdown>
#### 📝 Multi-Head: How It Works

| Step | What happens | Math |
|------|-------------|------|
| **Project** | Input \\(X\\) is projected into \\(h\\) separate subspaces | \\(Q_h = XW_h^Q\\), \\(K_h = XW_h^K\\), \\(V_h = XW_h^V\\) |
| **Attend** | Each head runs full attention **independently** | \\(\\text{head}_h = \\text{Attention}(Q_h, K_h, V_h)\\) |
| **Concat** | All head outputs are stacked side by side | \\([\\text{head}_1; \\ldots; \\text{head}_h]\\) |
| **Project back** | A final linear layer mixes the heads | \\(W^O \\cdot \\text{Concat}\\) |

**Key insight:** Total parameter count is the same as one big head (\\(d_{\\text{model}}\\) = \\(h \\times d_k\\)). Multi-head doesn't add parameters — it restructures them into parallel subspaces.
</div>

!!! math-intuition "In Plain English"
    - Each **head** projects into a **subspace** where a different similarity makes sense.
    - **Possible specialization (story, not guaranteed):** Head A attends to **local** neighbors (syntax / n-grams); Head B attends to **distant** coreferent mentions. In practice, heads are **mixed**, but **multi-head** increases **capacity** vs. one attention pool.

---

## Complexity Analysis

For sequence length \\(T\\), head dimension \\(d_k\\), value dimension \\(d_v\\), **one** attention layer (single head):

- **Form \\(QK^\\top\\):** \\(O(T^2 d_k)\\) multiply–accumulates (each of \\(T^2\\) scores needs \\(d_k\\) ops).
- **Softmax:** \\(O(T^2)\\).
- **Multiply by \\(V\\):** \\(O(T^2 d_v)\\).

**Dominant** term is often \\(O(T^2 \\cdot \\max(d_k, d_v))\\). **Memory** for full scores: \\(O(T^2)\\)—the **KV cache** and **FlashAttention** story for long contexts.

!!! example "Plug in Numbers: \\(T=2048\\), \\(d_k=64\\)"
    - Rough MACs for \\(QK^\\top\\): \\(T^2 d_k = 2048^2 \\times 64 = 4{,}194{,}304 \\times 64 \\approx 2.68 \\times 10^8\\) MACs **per head per layer** (order-of-magnitude; constants omitted).
    - This quadratic **\\(T^2\\)** term is why **long-context** inference stresses **memory bandwidth** and why **subquadratic** alternatives (linear attention, state-space models, sliding windows) matter.

---

## Code (existing implementation, with inline comments)

```python
"""
Scaled Dot-Product Attention from scratch in PyTorch.
Includes: basic attention, causal masking, multi-head attention,
and an attention weight heatmap.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        Q: (B, ..., n, d_k)
        K: (B, ..., m, d_k)
        V: (B, ..., m, d_v)
        mask: broadcastable to (B, ..., n, m), True = keep, False = mask out
    Returns:
        output: (B, ..., n, d_v)
        weights: (B, ..., n, m)
    """
    d_k = Q.size(-1)
    # Raw affinities: each query row vs all key rows
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B,...,n,m)

    if mask is not None:
        # Set masked positions to -inf so softmax -> 0 there
        scores = scores.masked_fill(~mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)  # (B,...,n,m) — convex combo per query row
    output = torch.matmul(weights, V)    # (B,...,n,d_v)
    return output, weights


def causal_mask(seq_len: int) -> torch.Tensor:
    """Lower-triangular boolean mask: position i can attend to j <= i."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n, _ = Q.shape
        m = K.size(1)

        # Split last dim into heads: (B, n, H, d_k) -> (B, H, n, d_k) for batched matmuls
        q = self.W_q(Q).view(B, n, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, m, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, m, self.n_heads, self.d_k).transpose(1, 2)

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, n, m)

        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, n, -1)
        return self.W_o(out), weights


def plot_attention(weights: torch.Tensor, tokens: list[str]) -> None:
    """Visualize a single-head attention weight matrix as a heatmap."""
    w = weights.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(w, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("Attention Weights")
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print("Saved: attention_heatmap.png")


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    T, D = len(tokens), 16
    N_HEADS = 2

    x = torch.randn(1, T, D)

    # 1) Basic self-attention (no mask)
    out, w = scaled_dot_product_attention(x, x, x)
    print(f"Self-attention output shape: {out.shape}")
    print(f"Weights shape: {w.shape}")
    print(f"Weights sum per query row: {w.sum(dim=-1)}")

    # 2) Causal self-attention
    mask = causal_mask(T)
    out_causal, w_causal = scaled_dot_product_attention(x, x, x, mask=mask)
    print(f"\nCausal attention — last query attends to all? "
          f"{(w_causal[0, -1] > 0).all().item()}")
    print(f"Causal attention — first query attends only to itself? "
          f"{(w_causal[0, 0, 1:] == 0).all().item()}")

    # 3) Multi-head attention
    mha = MultiHeadAttention(D, N_HEADS)
    mha_out, mha_w = mha(x, x, x, mask=mask)
    print(f"\nMulti-head output shape: {mha_out.shape}")
    print(f"MHA weights shape: {mha_w.shape}  (B, heads, n, m)")

    # 4) Visualize head 0
    plot_attention(mha_w[0, 0], tokens)

    # 5) Show scaling effect: unscaled vs scaled dot products -> softmax saturation
    d_k_large = 512
    q = torch.randn(1, 1, d_k_large)
    k = torch.randn(1, 5, d_k_large)
    raw_scores = torch.matmul(q, k.transpose(-2, -1))
    scaled_scores = raw_scores / math.sqrt(d_k_large)
    print(f"\nUnscaled score std: {raw_scores.std().item():.2f}")
    print(f"Scaled score std:   {scaled_scores.std().item():.2f}")
    print(f"Softmax(unscaled):  {F.softmax(raw_scores, -1).detach().numpy().round(3)}")
    print(f"Softmax(scaled):    {F.softmax(scaled_scores, -1).detach().numpy().round(3)}")
```

---

## Deep Dive

??? deep-dive "Attention as Low-Rank Kernel Approximation (sketch)"
    Writing \\(A = \\text{softmax}(QK^\\top/\\sqrt{d_k})\\) is **not** linear, but **before softmax**, scores are **rank-\\(\\le d_k\\)** bilinear forms. Some **linear attention** methods replace softmax with feature maps \\(\\phi(Q)\\phi(K)^\\top\\) to get **subquadratic** or recurrent forms—useful context for "**alternatives to softmax attention**" questions.

??? deep-dive "FlashAttention — What Problem It Solves"
    **Standard** attention materializes \\(T \\times T\\) scores in **HBM**; **FlashAttention** tiles computation to use **SRAM**, reducing **memory traffic**. Know: **asymptotic** \\(O(T^2)\\) unchanged, **wall-clock** and **memory footprint** improved—critical at **long** \\(T\\).

---

## Interview Guide

!!! interview "FAANG-Level Questions"

    **1. Write scaled dot-product attention and identify \\(Q\\), \\(K\\), \\(V\\).**

    **Full Answer:** Scaled dot-product attention is:

    \\(\\text{Attention}(Q, K, V) = \\text{softmax}\\big(\\frac{QK^T}{\\sqrt{d_k}}\\big)V\\)

    - **Q (Query):** "What am I looking for?" — each row represents one token's search criteria, produced by projecting the input \\(X\\) through learned weights \\(W^Q\\)
    - **K (Key):** "What do I offer?" — each row describes what information one token provides, produced by \\(XW^K\\)
    - **V (Value):** "What is my actual content?" — the information that gets blended together, produced by \\(XW^V\\)

    The computation: (1) \\(QK^T\\) computes all-pairs similarity scores — a \\(T \\times T\\) matrix where entry \\((i,j)\\) says how well query \\(i\\) matches key \\(j\\). (2) Dividing by \\(\\sqrt{d_k}\\) prevents softmax saturation. (3) Softmax normalizes each row to weights summing to 1. (4) Multiplying by \\(V\\) produces a weighted blend of value vectors.

    In **self-attention**, \\(Q\\), \\(K\\), \\(V\\) all come from the same input (different projections). In **cross-attention**, \\(Q\\) comes from the decoder while \\(K\\) and \\(V\\) come from the encoder.

    ---

    **2. Why divide by \\(\\sqrt{d_k}\\)?**

    **Full Answer:** Variance stabilization. If query and key components are i.i.d. with mean 0 and variance 1, then each element of the dot product \\(q_ik_i\\) has variance 1, and the full dot product (sum of \\(d_k\\) such terms) has variance \\(d_k\\). The standard deviation grows as \\(\\sqrt{d_k}\\).

    For \\(d_k = 512\\), unscaled dot products have standard deviation \\(\\approx 22.6\\). Softmax on logits of magnitude 20+ produces near-one-hot distributions (e.g., \\([0.9999, 0.0001, \\ldots]\\)). This means: (a) the model can only attend to ONE word, which is too rigid, and (b) gradients through the non-selected positions are nearly zero, killing training signal.

    Dividing by \\(\\sqrt{d_k}\\) rescales the variance back to \\(\\approx 1\\), keeping softmax in its "informative" regime where gradients flow to multiple positions. This is the "textbook" reason. In practice, learned LayerNorm and projections mean the i.i.d. assumption doesn't hold exactly, but the scaling remains essential for stable training.

    ---

    **3. What is causal masking and where is it applied?**

    **Full Answer:** Causal masking sets the upper triangle of the attention score matrix to \\(-\\infty\\) before softmax, so that position \\(i\\) can only attend to positions \\(j \\le i\\). After softmax, \\(e^{-\\infty} = 0\\), giving exactly zero weight to future positions.

    It is applied in **decoder self-attention** of autoregressive models (GPT, LLaMA, the decoder side of T5). During training, the model processes the entire sequence at once (for parallelism), but causal masking prevents each position from "seeing" future tokens — otherwise, the model could just copy the next token from the future, making training trivial and generation impossible.

    At **inference time**, causal masking is implicit because we generate tokens one at a time, so future tokens literally don't exist yet. The mask is necessary during training to simulate this sequential generation within a parallel forward pass.

    Implementation: `scores.masked_fill(mask == 0, float('-inf'))` where mask is a lower-triangular boolean matrix. The \\(-\\infty\\) is implemented as a large negative float (e.g., -1e9) for numerical safety.

    ---

    **4. Self-attention vs. cross-attention?**

    **Full Answer:** Mathematically identical operation. The difference is **where Q, K, V come from**:

    - **Self-attention:** Q, K, V are all projections of the **same** input sequence. Token \\(i\\) attends to all other tokens in the same layer. Used in GPT (causal self-attention), BERT (bidirectional self-attention), and both encoder and decoder of T5.
    - **Cross-attention:** Q comes from one sequence (the decoder), K and V come from a **different** sequence (the encoder). This is how the decoder "reads" the encoder output. Used in encoder-decoder models (T5, BART, original Transformer for translation).

    In cross-attention, the decoder asks "what do I need?" (query from decoder state), and the encoder answers "here's what I have" (keys and values from encoder output). This is the mechanism by which a translation model connects source language representations to target language generation.

    ---

    **5. Compute complexity of attention?**

    **Full Answer:** For sequence length \\(T\\) and head dimension \\(d\\):

    - **\\(QK^T\\):** \\(O(T^2 \\cdot d)\\) — for each of \\(T^2\\) pairs, compute a \\(d\\)-dimensional dot product
    - **Softmax:** \\(O(T^2)\\) — exponentiate and normalize each row
    - **Weights × V:** \\(O(T^2 \\cdot d)\\) — apply weights to values
    - **Memory:** \\(O(T^2)\\) just for the score matrix

    The \\(T^2\\) term is the bottleneck. For \\(T = 128K\\) tokens, \\(T^2 = 16.4\\) billion entries PER HEAD PER LAYER. This is why long-context inference is expensive and why FlashAttention (IO-aware, avoids materializing the full score matrix), sliding window attention (Mistral), and linear attention / SSMs (Mamba) were developed.

    **With \\(h\\) heads and \\(L\\) layers:** total attention FLOPs \\(\\approx O(L \\cdot h \\cdot T^2 \\cdot d_k)\\). For a 32-layer, 32-head model with \\(d_k = 128\\) processing 8K tokens: \\(32 \\times 32 \\times 8192^2 \\times 128 \\approx 8.8 \\times 10^{12}\\) FLOPs just for attention.

    ---

    **6. What does multi-head buy?**

    **Full Answer:** Multiple attention heads allow the model to learn **different attention patterns simultaneously** — parallel "relationship detectors" operating in separate subspaces. Head 1 might learn to attend to the syntactic subject, head 2 might track coreference, head 3 might focus on nearby context.

    Crucially, multi-head attention has the **same parameter count** as a single head with the same \\(d_{\\text{model}}\\). If \\(d_{\\text{model}} = 512\\) and \\(h = 8\\), each head operates on \\(d_k = 64\\) dimensions. The projection matrices \\(W^Q, W^K, W^V\\) each have shape \\(512 \\times 512\\) regardless of \\(h\\).

    The benefit is rank: a single attention head produces a rank-1 update per position (one weighted combination of values). Multiple heads can produce higher-rank updates, increasing the model's representational capacity. Empirically, 8–128 heads significantly outperform 1 head at the same parameter count. Some heads can be pruned post-training (20-40% pruning often has minimal quality impact), suggesting redundancy that aids optimization.

    ---

    **7. Temperature vs. attention sharpness?**

    **Full Answer:** Both use the same softmax function family. Temperature \\(\\tau\\) scales logits before softmax: \\(\\text{softmax}(z/\\tau)\\). Low \\(\\tau\\) → peaked (sharp) distribution; high \\(\\tau\\) → flat (uniform) distribution.

    In **attention**, the scaling factor \\(1/\\sqrt{d_k}\\) acts like an implicit temperature — it controls how peaked the attention weights are. If you removed the scaling, attention would be "colder" (more peaked), attending to fewer positions.

    In **decoding**, temperature controls generation diversity. \\(\\tau = 0.7\\) produces focused, deterministic text; \\(\\tau = 1.5\\) produces creative, diverse text. They're different applications of the same mathematical mechanism: softmax sensitivity to logit magnitude.

    ---

    **8. Why not additive Bahdanau everywhere?**

    **Full Answer:** Dot-product attention is **hardware-friendly**. The core operation \\(QK^T\\) is a matrix multiplication — the single most optimized operation on GPUs and TPUs (hitting near-peak FLOPS). Bahdanau attention requires per-pair feed-forward computation: \\(\\tanh(W_1h_j + W_2s_t)\\) for every \\((t,j)\\) pair, which cannot be expressed as a single GEMM call. For \\(T=2048\\), this means 4 million separate small computations vs. one large matrix multiply. The throughput difference can be 10-100x on modern accelerators.

    Additionally, dot-product attention's gradient computation is also a matrix multiply, keeping the entire training pipeline within BLAS-optimized code paths.

    ---

    **9. Gradient flow: what happens if softmax is almost one-hot?**

    **Full Answer:** If softmax produces \\([0.999, 0.001, 0.000, 0.000]\\), then during backpropagation, the gradient signal almost entirely flows through position 0 (the one with weight 0.999). Positions 1, 2, 3 receive near-zero gradient — the model cannot learn to adjust its attention toward those positions. This is the "attention collapse" failure mode.

    Mathematically, the Jacobian of softmax at a near-one-hot point has very small off-diagonal entries. The derivative \\(\\partial \\text{softmax}(z)_i / \\partial z_j = \\text{softmax}_i(\\delta_{ij} - \\text{softmax}_j)\\). When \\(\\text{softmax}_i \\approx 1\\) and \\(\\text{softmax}_j \\approx 0\\) for \\(j \\neq i\\), the off-diagonal derivatives are \\(\\approx 0\\). This is precisely why the \\(\\sqrt{d_k}\\) scaling exists — to keep softmax in a regime where gradients flow meaningfully to multiple positions.

    ---

    **10. KV cache in autoregressive decoding?**

    **Full Answer:** During autoregressive generation, at step \\(t\\) we generate one new token. We need its Q, K, V vectors. The K and V vectors for ALL previous tokens \\(1, \\ldots, t-1\\) are **unchanged** because causal masking means they don't see the new token. So we **cache** them instead of recomputing.

    The KV cache stores all past K and V vectors: size = \\(2 \\times L \\times T \\times n_{\\text{kv\\_heads}} \\times d_k \\times \\text{bytes}\\). For LLaMA-70B with GQA-8 (8 KV heads), \\(d_k = 128\\), 80 layers, 4K context in FP16: \\(2 \\times 80 \\times 4096 \\times 8 \\times 128 \\times 2 = 1.34\\) GB per request.

    Without KV cache, generating token \\(t\\) requires recomputing all \\(t-1\\) K and V vectors — \\(O(t \\cdot d)\\) work. With cache, only the new token's K and V need computation — \\(O(d)\\). The trade-off is memory: cache grows linearly with context length, which is why memory-efficient techniques (GQA, MLA, PagedAttention) are critical for serving.

!!! interview "Follow-up Probes"
    - "**Relative position** encodings—where do they enter?" — often **bias** to \\(QK^\\top\\) or alternate RPE layers (Transformer-XL, T5 biases).
    - "**AliBi** vs. rotary?" — both inject position info; know **high-level** tradeoffs only if you claim expertise.

!!! key-phrases "Key Phrases to Use in Interviews"
    - "**Scaled dot-product** keeps logits \\(O(1)\\) so **softmax** doesn't **saturate**."
    - "**Causal mask** enforces **autoregressive** factorization—no **future** tokens."
    - "**Attention is \\(O(T^2)\\)** in sequence length—that's the **long-context** bottleneck."
    - "**Multi-head** learns **multiple** compatibility functions in **parallel subspaces**."

---

## References

- Vaswani et al. (2017), *Attention Is All You Need* — Section 3.2
- Luong et al. (2015), *Effective Approaches to Attention-Based NMT*
- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- Dao et al. — FlashAttention (IO-aware exact attention)
