# LLaMA: Open and Efficient Foundation Language Models

**Authors:** Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière  
**Year:** 2023 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)

---

## TL;DR

LLaMA trains **openly released** foundation models (7B–65B parameters) using **Chinchilla-optimal** token budgets — significantly more training tokens relative to model size. Architecture tweaks include **RMSNorm** (replacing LayerNorm), **SwiGLU** activations, **Rotary Position Embeddings (RoPE)**, and efficient attention. The goal: **maximum performance per parameter** for research and local deployment.

---

## Why This Paper Matters

LLaMA triggered an explosion of open-source LLM development:

1. **Open weights** enabled an entire ecosystem: Alpaca, Vicuna, Llama 2, Llama 3, Llama 4
2. Proved that **smaller, well-trained models** can match much larger ones
3. Established **RMSNorm + SwiGLU + RoPE** as the standard modern architecture
4. Made LLM research accessible without massive compute budgets
5. **7B/13B** models run on consumer GPUs, enabling local AI deployment

---

## Key Concepts Explained Simply

### RMSNorm (Root Mean Square Normalization)

LayerNorm normalizes by subtracting the mean and dividing by the standard deviation. RMSNorm is simpler — it **skips the mean subtraction** and only divides by the root mean square:

- **LayerNorm:** Subtract mean, divide by std, scale and shift
- **RMSNorm:** Just divide by RMS, then scale

Why? Removing the mean-centering step reduces computation with minimal impact on performance. At scale, this saving adds up.

### SwiGLU Activation

The feed-forward network in a Transformer typically uses ReLU or GELU. SwiGLU uses a **gated** structure:

- Standard FFN: \(\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)\)
- SwiGLU: \(\text{FFN}(x) = (W_1 x \odot \text{Swish}(W_3 x)) W_2\)

The gate \(\text{Swish}(W_3 x)\) controls what information flows through, giving the network more expressivity.

### Rotary Position Embeddings (RoPE)

Instead of adding position information to the input (like sinusoidal encodings), RoPE **rotates** the query and key vectors based on position. The rotation angle is proportional to position, so the dot product between two rotated vectors depends on their **relative position**.

Key advantage: RoPE naturally encodes **relative** positions, and its extrapolation properties allow processing longer sequences than seen during training (with some quality loss).

---

## The Math — Explained Step by Step

### RMSNorm

\[
\bar{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\|\mathbf{x}\|^2 + \epsilon}} \odot \boldsymbol{\gamma}
\]

**Breaking it down:**

1. \(\|\mathbf{x}\|^2\): Sum of squared elements of the vector
2. \(\frac{1}{d}\|\mathbf{x}\|^2\): Mean of squared elements (the "root mean square" before the sqrt)
3. Divide by \(\sqrt{\text{mean of squares}}\): Normalizes the scale
4. \(\odot \gamma\): Learnable element-wise scaling (no bias term)

### SwiGLU

\[
\text{SwiGLU}(x) = \left(\text{Swish}_\beta(x W_1) \odot (x W_3)\right) W_2
\]

where \(\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)\) and \(\sigma\) is the sigmoid function.

The \(\odot\) is element-wise multiplication — the Swish output **gates** the linear transformation.

### RoPE

For position \(m\) and dimension pair \((2i, 2i+1)\):

\[
\begin{pmatrix} q_{2i}' \\ q_{2i+1}' \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}
\]

where \(\theta_i = 10000^{-2i/d}\).

**Key property:** \(q_m^\top k_n = f(q, k, m-n)\) — the dot product depends on the **relative distance** \(m - n\), not absolute positions.

---

## Python Implementation

```python
import numpy as np


def rmsnorm(x, gamma, eps=1e-6):
    """
    Root Mean Square Layer Normalization.
    x: [seq_len, d_model]
    gamma: [d_model] — learnable scale
    """
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * gamma


def layernorm(x, gamma, beta, eps=1e-6):
    """Standard LayerNorm for comparison."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def swish(x, beta=1.0):
    """Swish activation: x * sigmoid(beta * x)."""
    return x * (1 / (1 + np.exp(-beta * x)))


def swiglu(x, W1, W3, W2):
    """
    SwiGLU feed-forward network.
    x: [seq_len, d_model]
    W1: [d_model, d_ff] — gate projection
    W3: [d_model, d_ff] — value projection
    W2: [d_ff, d_model] — output projection
    """
    gate = swish(x @ W1)
    value = x @ W3
    return (gate * value) @ W2


def rope_freqs(d_model, seq_len, base=10000):
    """Compute RoPE frequency pairs."""
    freqs = 1.0 / (base ** (np.arange(0, d_model, 2).astype(float) / d_model))
    positions = np.arange(seq_len)
    angles = np.outer(positions, freqs)
    return np.cos(angles), np.sin(angles)


def apply_rope(x, cos, sin):
    """
    Apply Rotary Position Embeddings to query/key vectors.
    x: [seq_len, d_model]
    cos, sin: [seq_len, d_model//2]
    """
    d = x.shape[-1]
    x1 = x[:, :d // 2]
    x2 = x[:, d // 2:]

    x_rotated = np.concatenate([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    return x_rotated


def rope_attention(Q, K, V, cos, sin):
    """Self-attention with RoPE applied to Q and K."""
    Q_rot = apply_rope(Q, cos, sin)
    K_rot = apply_rope(K, cos, sin)

    d_k = Q.shape[-1]
    scores = (Q_rot @ K_rot.T) / np.sqrt(d_k)

    # Causal mask
    seq_len = Q.shape[0]
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    scores = scores + mask

    attn = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn = attn / np.sum(attn, axis=-1, keepdims=True)
    return attn @ V


class LLaMABlock:
    """Simplified LLaMA Transformer block."""

    def __init__(self, d_model, d_ff, n_heads):
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02

        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.W3 = np.random.randn(d_model, d_ff) * 0.02
        self.W2 = np.random.randn(d_ff, d_model) * 0.02

        self.gamma_attn = np.ones(d_model)
        self.gamma_ffn = np.ones(d_model)

    def forward(self, x, cos, sin):
        # Pre-RMSNorm + Attention + Residual
        h = rmsnorm(x, self.gamma_attn)
        Q, K, V = h @ self.W_q, h @ self.W_k, h @ self.W_v
        attn_out = rope_attention(Q, K, V, cos, sin)
        attn_out = attn_out @ self.W_o
        x = x + attn_out

        # Pre-RMSNorm + SwiGLU FFN + Residual
        h = rmsnorm(x, self.gamma_ffn)
        ffn_out = swiglu(h, self.W1, self.W3, self.W2)
        x = x + ffn_out

        return x


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    d_model, d_ff, n_heads, seq_len = 64, 128, 4, 10

    # RMSNorm vs LayerNorm
    x = np.random.randn(seq_len, d_model)
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)

    rms_out = rmsnorm(x, gamma)
    ln_out = layernorm(x, gamma, beta)
    print("RMSNorm output stats: mean={:.4f}, std={:.4f}".format(
        rms_out.mean(), rms_out.std()))
    print("LayerNorm output stats: mean={:.4f}, std={:.4f}".format(
        ln_out.mean(), ln_out.std()))

    # SwiGLU demo
    W1 = np.random.randn(d_model, d_ff) * 0.02
    W3 = np.random.randn(d_model, d_ff) * 0.02
    W2 = np.random.randn(d_ff, d_model) * 0.02
    ffn_out = swiglu(x, W1, W3, W2)
    print(f"\nSwiGLU: input {x.shape} → output {ffn_out.shape}")

    # RoPE demo
    cos, sin = rope_freqs(d_model, seq_len)
    Q = np.random.randn(seq_len, d_model)
    Q_rot = apply_rope(Q, cos, sin)
    print(f"RoPE: Q shape {Q.shape} → rotated {Q_rot.shape}")

    # Full block
    block = LLaMABlock(d_model, d_ff, n_heads)
    out = block.forward(x, cos, sin)
    print(f"\nLLaMA block: input {x.shape} → output {out.shape}")

    # Model size comparison
    print("\n--- LLaMA Model Sizes ---")
    sizes = [
        ("LLaMA-7B",  7e9,  1.0e12, "1x A100 (with quantization)"),
        ("LLaMA-13B", 13e9, 1.0e12, "2x A100 or 1x A100 (quantized)"),
        ("LLaMA-33B", 33e9, 1.4e12, "4x A100"),
        ("LLaMA-65B", 65e9, 1.4e12, "8x A100"),
    ]
    for name, params, tokens, hw in sizes:
        ratio = tokens / params
        print(f"  {name}: {params/1e9:.0f}B params, {tokens/1e12:.1f}T tokens, "
              f"{ratio:.0f} tok/param, {hw}")
```

---

## Interview Importance

LLaMA is a **top-5 most important** model to know. Its architecture (RMSNorm + SwiGLU + RoPE) is the modern standard, and understanding why these choices were made demonstrates practical LLM knowledge.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Name three architectural choices in LLaMA vs. original GPT-2/3.

**Answer:**
1. **RMSNorm instead of LayerNorm:** Removes mean-centering for efficiency. Same stabilization effect with fewer operations.
2. **SwiGLU instead of GELU:** Gated activation in the FFN provides more expressivity through element-wise gating.
3. **RoPE instead of learned/sinusoidal position embeddings:** Encodes relative positions through rotation, enabling better length generalization.

Additional changes: Pre-normalization (normalize before sublayer, not after), no bias terms in linear layers, and grouped-query attention in Llama 2.

### Q2: Connect LLaMA training to Chinchilla-optimal thinking.

**Answer:** LLaMA directly applied Chinchilla's insight that smaller models trained on more data outperform larger undertrained models:
- LLaMA-7B: 1T tokens (~143 tokens/param) — overtrained beyond Chinchilla-optimal (~20 tok/param), intentionally, because smaller models are cheaper to serve
- LLaMA-65B: 1.4T tokens (~21.5 tokens/param) — close to Chinchilla-optimal
- Result: LLaMA-13B matched GPT-3 (175B) on benchmarks, validating the Chinchilla thesis for open models
- Key insight: For **inference-time efficiency**, it's worth "overtraining" small models beyond the compute-optimal point

### Q3: What deployment constraints push teams toward 7B/13B rather than 70B?

**Answer:**
1. **GPU memory:** 70B needs ~140GB in float16 (multiple high-end GPUs). 7B needs ~14GB (single consumer GPU with quantization)
2. **Latency:** Larger models have higher per-token latency due to more layers and wider matrices
3. **Cost:** Serving 70B at scale requires 8-10× more GPU resources than 7B
4. **On-device deployment:** Mobile/edge devices can run quantized 7B but not 70B
5. **Fine-tuning cost:** LoRA on 7B is fast and cheap; full fine-tuning on 70B requires a cluster
6. **Quality is "good enough":** For many tasks (classification, retrieval, simple generation), 7B with good fine-tuning matches 70B

### Q4: Explain RoPE and why it's better than learned positional embeddings.

**Answer:** RoPE encodes positions by **rotating** query and key vectors. The dot product between a rotated query at position \(m\) and a rotated key at position \(n\) depends only on their **relative distance** \(m-n\). 

**Advantages over learned embeddings:**
1. **Relative positions:** Naturally captures relative distance, not just absolute position
2. **Length generalization:** Can extrapolate to positions beyond the training length (though with quality degradation)
3. **No extra parameters:** Position information is encoded through rotation, not learned vectors
4. **Theoretical grounding:** Based on Fourier analysis of position-dependent signals

---

## Connections to Other Papers

- **Chinchilla** → LLaMA uses Chinchilla-optimal data ratios
- **Transformer** → Base architecture with modern improvements
- **GPT-2/3** → LLaMA modernizes the GPT decoder-only design
- **LoRA** → Commonly fine-tuned with LoRA (parameter-efficient)
- **Mistral** → Further innovations (GQA, sliding window) on LLaMA-style architecture

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Architecture | Decoder-only with RMSNorm, SwiGLU, RoPE |
| Sizes | 7B, 13B, 33B, 65B |
| Training tokens | 1T-1.4T (Chinchilla-optimal or beyond) |
| Key result | 13B matches GPT-3 (175B) on many benchmarks |
| RMSNorm | LayerNorm without mean subtraction |
| SwiGLU | Gated FFN activation for expressivity |
| RoPE | Rotary embeddings for relative position encoding |
| Impact | Triggered open-source LLM ecosystem |
