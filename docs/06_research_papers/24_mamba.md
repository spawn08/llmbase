# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

**Authors:** Albert Gu, Tri Dao  
**Year:** 2023 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

---

## TL;DR

Mamba advances **structured state-space models (SSMs)** with **input-dependent (selective)** parameters, letting the model **choose what to remember** at each timestep. Unlike fixed-parameter SSMs (S4), Mamba's state transitions depend on the input. It achieves **linear time** in sequence length via **parallel scan** during training, competing with Transformers on quality while being much more efficient for long sequences.

---

## Why This Paper Matters

Mamba is the leading **alternative to attention** for sequence modeling:

1. **Linear complexity:** \(O(N)\) vs. Transformer's \(O(N^2)\) attention
2. **No KV cache:** Constant memory per token during inference (fixed state size)
3. **Long sequences:** Naturally handles very long sequences without sliding windows
4. **Hybrid models:** Mamba + attention hybrids (Jamba, Zamba) combine benefits
5. **Edge deployment:** Linear complexity enables efficient inference on resource-constrained devices

---

## Key Concepts Explained Simply

### State Space Models: The Core Idea

An SSM is like a **learned RNN** with a structured state:

1. **State \(h_t\):** A fixed-size vector that summarizes all history up to time \(t\)
2. **Update rule:** At each step, update the state based on the new input
3. **Output:** Generate the output from the current state

Unlike attention (which looks at all previous tokens), an SSM compresses history into a fixed-size state. The question is: can this compression be good enough?

### What Makes Mamba Special: Selectivity

Previous SSMs (S4) used **fixed** state transition matrices — the same A, B, C parameters for every input token. This limits the model's ability to filter information.

Mamba makes A, B, C **depend on the input**:
- For important tokens: "Remember this" (large B, slow-decaying A)
- For irrelevant tokens: "Forget this" (small B, fast-decaying A)

This **selectivity** is analogous to the gating in LSTMs but with structured state spaces that can be computed in parallel.

### Parallel Training via Associative Scan

An RNN processes tokens one by one — slow for training. Mamba uses the **associative scan** (parallel prefix sum) algorithm:
- The state update \(h_t = A_t h_{t-1} + B_t x_t\) is **associative**
- Associative operations can be computed in \(O(\log N)\) parallel steps instead of \(O(N)\) sequential steps
- This gives Mamba RNN-like inference with near-parallel training speed

---

## The Math — Explained Step by Step

### Discrete State Space Model

\[
h_t = \bar{A}_t \, h_{t-1} + \bar{B}_t \, x_t
\]
\[
y_t = C_t \, h_t
\]

**Breaking it down:**

1. \(h_t \in \mathbb{R}^{n}\): Hidden state at time \(t\) (fixed size, e.g., n=16)
2. \(\bar{A}_t \in \mathbb{R}^{n \times n}\): State transition matrix — how much of the old state to keep
3. \(\bar{B}_t \in \mathbb{R}^{n \times 1}\): Input projection — how much of the new input to incorporate
4. \(C_t \in \mathbb{R}^{1 \times n}\): Output projection — how to read from the state
5. **Selective:** \(\bar{A}_t, \bar{B}_t, C_t\) all depend on the current input \(x_t\)

### Discretization

The continuous parameters (A, B) are discretized using a learned step size \(\Delta_t\):

\[
\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = (\Delta_t A)^{-1}(\bar{A}_t - I) \cdot \Delta_t B_t
\]

In practice, a simplified discretization is used:
\[
\bar{A}_t = \exp(-\Delta_t \cdot \text{softplus}(A)), \quad \bar{B}_t = \Delta_t B_t
\]

### Selectivity Mechanism

The key innovation — \(\Delta_t, B_t, C_t\) are functions of the input:

\[
\Delta_t = \text{softplus}(\text{Linear}(x_t))
\]
\[
B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t)
\]

**What selectivity achieves:**
- **Large \(\Delta_t\):** Focus more on current input (reset-like behavior)
- **Small \(\Delta_t\):** Retain state from previous steps (memory behavior)
- The model learns **when** to read, write, or forget

### Complexity Comparison

| Model | Training | Inference (per token) | State Size |
|---|---|---|---|
| Transformer | \(O(N^2 d)\) | \(O(Nd)\) with KV cache | \(O(N \cdot d)\) growing |
| RNN/LSTM | \(O(Nd)\) sequential | \(O(d)\) | \(O(d)\) fixed |
| Mamba (SSM) | \(O(N d)\) parallel | \(O(d)\) | \(O(d)\) fixed |

---

## Python Implementation

```python
import numpy as np


def ssm_step(h, x, A_bar, B_bar, C):
    """
    Single SSM step (recurrent mode — used during inference).
    h: [state_dim] — hidden state
    x: scalar input
    A_bar: [state_dim] — diagonal state transition (simplified)
    B_bar: [state_dim] — input projection
    C: [state_dim] — output projection
    """
    h_new = A_bar * h + B_bar * x
    y = np.dot(C, h_new)
    return h_new, y


def ssm_sequential(x_seq, A, B, C, delta):
    """
    Run SSM sequentially over a sequence (inference mode).
    x_seq: [seq_len]
    """
    state_dim = len(A)
    h = np.zeros(state_dim)
    outputs = []

    for t in range(len(x_seq)):
        A_bar = np.exp(-delta[t] * np.abs(A))
        B_bar = delta[t] * B[t]

        h, y = ssm_step(h, x_seq[t], A_bar, B_bar, C[t])
        outputs.append(y)

    return np.array(outputs)


def selective_params(x, W_delta, W_B, W_C, state_dim):
    """
    Compute input-dependent (selective) SSM parameters.
    x: [seq_len, d_model]
    """
    delta = np.log(1 + np.exp(x @ W_delta))  # softplus
    B = x @ W_B  # [seq_len, state_dim]
    C = x @ W_C  # [seq_len, state_dim]
    return delta.flatten(), B, C


def parallel_scan(A_bars, B_bar_x):
    """
    Associative scan for parallel SSM computation.
    A_bars: [seq_len, state_dim] — discretized A per timestep
    B_bar_x: [seq_len, state_dim] — B * x per timestep

    This computes the recurrence h_t = A_t * h_{t-1} + B_t * x_t
    in O(log N) parallel steps instead of O(N) sequential steps.
    """
    N = len(A_bars)
    states = B_bar_x.copy()
    a_accum = A_bars.copy()

    # Up-sweep (reduce)
    stride = 1
    while stride < N:
        for i in range(stride, N, stride * 2):
            if i - stride >= 0:
                states[i] = a_accum[i] * states[i - stride] + states[i]
                a_accum[i] = a_accum[i] * a_accum[i - stride]
        stride *= 2

    return states


class MambaBlock:
    """Simplified Mamba block."""

    def __init__(self, d_model, state_dim=16, d_conv=4):
        self.d_model = d_model
        self.state_dim = state_dim
        self.d_conv = d_conv

        # Projections
        self.W_in = np.random.randn(d_model, d_model * 2) * 0.02
        self.W_delta = np.random.randn(d_model, 1) * 0.02
        self.W_B = np.random.randn(d_model, state_dim) * 0.02
        self.W_C = np.random.randn(d_model, state_dim) * 0.02
        self.W_out = np.random.randn(d_model, d_model) * 0.02

        # Fixed A (learned but input-independent)
        self.A = -np.arange(1, state_dim + 1, dtype=float)

        # 1D convolution
        self.conv_weight = np.random.randn(d_conv) * 0.1

    def forward(self, x):
        """
        x: [seq_len, d_model]
        Returns: [seq_len, d_model]
        """
        seq_len = x.shape[0]

        # Input projection and split
        projected = x @ self.W_in
        z = projected[:, :self.d_model]
        x_proj = projected[:, self.d_model:]

        # 1D convolution (simplified)
        x_conv = np.copy(x_proj)
        for t in range(seq_len):
            start = max(0, t - self.d_conv + 1)
            window = x_proj[start:t+1]
            weights = self.conv_weight[-(t-start+1):]
            if len(weights) > 0:
                x_conv[t] = np.sum(window * weights[:, np.newaxis], axis=0)

        # Selective SSM parameters
        delta, B, C = selective_params(
            x_conv, self.W_delta, self.W_B, self.W_C, self.state_dim
        )

        # Run SSM (sequential for simplicity; parallel scan in practice)
        h = np.zeros(self.state_dim)
        ssm_out = np.zeros((seq_len, self.d_model))

        for t in range(seq_len):
            A_bar = np.exp(delta[t] * self.A)
            B_bar = delta[t] * B[t]

            for d in range(self.d_model):
                h_d = A_bar * h + B_bar * x_conv[t, d % self.state_dim]
                ssm_out[t, d] = np.dot(C[t], h_d)

        # Gate and output
        y = ssm_out * (1 / (1 + np.exp(-z)))  # SiLU gating
        return y @ self.W_out


def compare_complexities():
    """Compare computational complexity of different architectures."""
    print("--- Complexity Comparison ---")
    seq_lengths = [512, 2048, 8192, 32768, 131072]
    d = 4096

    print(f"{'Seq Len':>10} {'Attention':>15} {'Mamba/SSM':>15} {'Speedup':>10}")
    print("-" * 55)
    for N in seq_lengths:
        attn_flops = N * N * d
        ssm_flops = N * d * 16  # state_dim=16
        speedup = attn_flops / ssm_flops
        print(f"{N:>10,} {attn_flops:>15,.0f} {ssm_flops:>15,.0f} {speedup:>10.0f}×")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Sequential SSM
    print("--- Sequential SSM ---")
    seq_len, state_dim = 20, 4
    x_seq = np.random.randn(seq_len)
    A = -np.arange(1, state_dim + 1, dtype=float)
    delta = np.ones(seq_len) * 0.1
    B = np.random.randn(seq_len, state_dim) * 0.1
    C = np.random.randn(seq_len, state_dim) * 0.1

    outputs = ssm_sequential(x_seq, A, B, C, delta)
    print(f"Input shape: ({seq_len},)")
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    # Mamba block
    print("\n--- Mamba Block ---")
    d_model = 32
    block = MambaBlock(d_model, state_dim=8)
    x = np.random.randn(seq_len, d_model)
    out = block.forward(x)
    print(f"Mamba block: input {x.shape} → output {out.shape}")

    # Complexity comparison
    print()
    compare_complexities()
```

---

## Interview Importance

Mamba represents the **future direction** of sequence modeling. Understanding SSMs vs. attention shows you're thinking beyond the current paradigm.

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: Contrast Mamba/SSM complexity vs. self-attention vs. RNNs.

**Answer:**
- **Self-attention:** \(O(N^2 d)\) training, \(O(Nd)\) inference per token. Quadratic in sequence length, but highly parallelizable. KV cache grows with context.
- **RNN/LSTM:** \(O(Nd)\) per step, but strictly sequential — can't parallelize across time steps. Fixed state size. Fast inference.
- **Mamba/SSM:** \(O(Nd)\) training via parallel scan — near-parallel despite being recurrent. \(O(d)\) inference per token. Fixed state size. Best of both: parallel training + efficient inference.

### Q2: What does selectivity buy you that fixed S4 parameters lack?

**Answer:** Fixed parameters apply the same state update to every token — they can't distinguish important from irrelevant tokens. Selectivity lets the model:
1. **Remember selectively:** Increase B (input projection) for important tokens
2. **Forget selectively:** Adjust A (transition) to decay old information when new important data arrives
3. **Read selectively:** Adjust C (output) based on the query

This is analogous to LSTM gates but with the parallel training advantages of structured SSMs. Without selectivity, S4 struggles on tasks requiring content-aware filtering (e.g., selective copying, where the model must copy only specific tokens).

### Q3: Where might you still want attention (vs. Mamba)?

**Answer:**
1. **In-context learning:** Attention excels at retrieving specific tokens from context (e.g., few-shot examples). SSMs compress history into a fixed state, losing precise token-level retrieval.
2. **Global retrieval:** Tasks requiring comparing distant tokens (e.g., "find the first occurrence of X") favor attention.
3. **Hybrid models:** Jamba/Zamba use attention for a few layers (for retrieval) and Mamba for most (for efficiency) — getting benefits of both.
4. **Short sequences:** For sequences under ~2K tokens, attention is efficient enough and has proven quality.
5. **Proven reliability:** Attention has years of engineering optimization (FlashAttention, KV cache); SSMs are newer.

---

## Connections to Other Papers

- **Transformer** → Mamba is an alternative to the attention mechanism
- **FlashAttention** → Both optimize memory access patterns; FlashAttention for attention, Mamba for SSMs
- **LLaMA** → Hybrid Mamba+attention models are emerging alternatives
- **Mistral** → Sliding window attention is another approach to long-context efficiency

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | SSM with input-dependent (selective) parameters |
| Complexity | \(O(N \cdot d)\) — linear in sequence length |
| State update | \(h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t\) |
| Selectivity | A, B, C depend on input → content-aware filtering |
| Training | Parallel scan (associative operation) |
| Inference | Recurrent mode — constant memory per token |
| vs. Attention | No KV cache, linear complexity, but weaker retrieval |
| Hybrid trend | Mamba + attention layers in models like Jamba |
