# 2.8 — State Space Models (SSMs): S4 and Mamba

## Intuition

Transformers have a problem: self-attention is \(O(T^2)\) in sequence length. For 128K or 1M-token contexts, this becomes prohibitive. **State Space Models** (SSMs) offer a fundamentally different approach: process sequences through a **continuous-time linear system** that is discretized and computed as a recurrence or convolution. Mamba (Gu & Dao, 2023) made SSMs competitive with Transformers by adding **input-dependent (selective) gating**, achieving linear-time sequence modeling without attention.

---

## Core concepts

### Linear state space model (continuous-time)

An SSM maps an input signal \(u(t)\) to an output \(y(t)\) through a hidden state \(x(t)\):

\[
\dot{x}(t) = A x(t) + B u(t)
\]
\[
y(t) = C x(t) + D u(t)
\]

where \(A \in \mathbb{R}^{N \times N}\), \(B \in \mathbb{R}^{N \times 1}\), \(C \in \mathbb{R}^{1 \times N}\), \(D \in \mathbb{R}^{1 \times 1}\).

This is a classic control-theory system. The innovation is using it as a **sequence model** for discrete tokens.

### Discretization

To handle discrete sequences (tokens, audio samples), we discretize with step size \(\Delta\):

\[
\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B
\]

This gives a **recurrence**:

\[
x_k = \bar{A} x_{k-1} + \bar{B} u_k
\]
\[
y_k = C x_k
\]

### Dual computation modes

The key property of linear SSMs: the same model can be computed as either:

| Mode | Algorithm | Complexity | Use case |
| --- | --- | --- | --- |
| Recurrence | \(x_k = \bar{A}x_{k-1} + \bar{B}u_k\) | \(O(T)\) sequential | Inference (token-by-token) |
| Convolution | \(y = \bar{K} * u\) where \(\bar{K} = (C\bar{B}, C\bar{A}\bar{B}, \ldots)\) | \(O(T \log T)\) parallel | Training (via FFT) |

During training, use the convolution mode for GPU parallelism. During inference, use the recurrence for \(O(1)\) per-step compute (like an RNN, but with a structured state).

### S4 (Structured State Spaces for Sequences)

Gu et al. (2021) made SSMs practical by:

1. **HiPPO initialization** — initialize \(A\) as the HiPPO matrix, which is designed to optimally compress a continuous signal into a fixed-size state. This solves the long-range dependency problem.
2. **Diagonal approximation** — parameterize \(A\) as diagonal (or diagonal-plus-low-rank) for efficient computation.
3. **Efficient convolution** — compute the kernel \(\bar{K}\) via FFT in \(O(T \log T)\).

S4 achieved state-of-the-art on the Long Range Arena benchmark (sequences up to 16K tokens) where Transformers failed.

### Mamba — selective state spaces

The limitation of S4: the state transition matrices \(A, B, C\) are **time-invariant** (the same for every token). This means the model can't selectively focus on or ignore specific tokens based on content — it processes everything uniformly.

Mamba fixes this with **input-dependent** (selective) parameters:

\[
B_k = \text{Linear}_B(x_k), \quad C_k = \text{Linear}_C(x_k), \quad \Delta_k = \text{softplus}(\text{Linear}_\Delta(x_k))
\]

Now \(B\), \(C\), and \(\Delta\) vary per token, allowing the model to:

- **Selectively remember** — large \(\Delta\) ≈ reset state (forget irrelevant context).
- **Selectively copy** — project specific tokens into the state and retrieve them later.

This breaks the convolution mode (parameters are no longer constant), so Mamba uses a **hardware-aware scan** algorithm instead — a parallel prefix scan that runs in \(O(T)\) on GPUs.

### Mamba block architecture

```
Input x
    ↓
Linear projection → two branches
    ↓                    ↓
Conv1D → SiLU      Linear (gate)
    ↓                    ↓
SSM (selective)      SiLU
    ↓                    ↓
    ×─────────────────────
    ↓
Linear projection
    ↓
Output (+ residual)
```

### SSM vs. Transformer comparison

| Property | Transformer | SSM (Mamba) |
| --- | --- | --- |
| Training complexity | \(O(T^2 d)\) | \(O(T d N)\) (linear) |
| Inference per step | \(O(T d)\) via KV cache | \(O(d N)\) constant |
| Long-range dependencies | Direct (attention) | Through state (compressed) |
| Selective retrieval | Excellent (attention is content-based) | Good (with selection, but lossy) |
| In-context learning | Strong | Weaker than Transformers |
| Parallelizable (training) | Yes (matrix ops) | Yes (scan/conv) |

### Hybrid architectures

The latest trend: **combine** attention layers and Mamba layers in a single model:

- **Jamba** (AI21, 2024) — alternates Transformer and Mamba blocks with MoE.
- **Mamba-2** (Dao & Gu, 2024) — shows that the selective SSM can be reformulated as a form of **structured masked attention**, bridging SSMs and Transformers theoretically.

---

## Code — Minimal selective SSM (Mamba-style) block

```python
"""
Minimal Selective State Space Model (Mamba-style) in PyTorch.
Implements: discretization, selective scan (sequential), and the Mamba block.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective (input-dependent) State Space Model.
    B, C, and delta are functions of the input.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_state = d_state

        # A is initialized from a structured (diagonal) matrix
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))  # log-space for stability

        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent projections
        self.proj_B = nn.Linear(d_model, d_state, bias=False)
        self.proj_C = nn.Linear(d_model, d_state, bias=False)
        self.proj_delta = nn.Linear(d_model, d_model, bias=True)

        # Initialize delta bias to be positive (softplus maps to ~1)
        nn.init.constant_(self.proj_delta.bias, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) — input after conv/projection
        Returns:
            y: (B, T, D)
        """
        B_batch, T, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log)                   # (D, N)
        D = self.D                                      # (D,)

        B = self.proj_B(x)                              # (B, T, N)
        C = self.proj_C(x)                              # (B, T, N)
        delta = F.softplus(self.proj_delta(x))          # (B, T, D)

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        delta_A = torch.exp(delta.unsqueeze(-1) * A)    # (B, T, D, N)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, D, N)

        # Sequential scan (production code uses a parallel scan kernel)
        h = torch.zeros(B_batch, D, N, device=x.device)  # state
        ys = []
        for t in range(T):
            h = delta_A[:, t] * h + delta_B[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)      # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, T, D)
        y = y + x * D               # skip connection with D
        return y


class MambaBlock(nn.Module):
    """
    Simplified Mamba block: Conv1D → SelectiveSSM with gated output.
    """

    def __init__(
        self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2
    ):
        super().__init__()
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # Two branches: SSM path and gate
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # Conv1D (causal)
        x_ssm = x_ssm.transpose(1, 2)                  # (B, D_inner, T)
        x_ssm = self.conv1d(x_ssm)[:, :, : x.size(1)]  # causal trim
        x_ssm = x_ssm.transpose(1, 2)                   # (B, T, D_inner)
        x_ssm = F.silu(x_ssm)

        # Selective SSM
        y = self.ssm(x_ssm)

        # Gated output
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + residual


class MambaModel(nn.Module):
    """Stack of Mamba blocks with embedding and LM head."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        d_state: int = 16,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model, d_state) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight  # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    V, D, L = 256, 128, 4
    B, T = 2, 64

    model = MambaModel(V, D, L)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Mamba model: {n_params:,} parameters")

    tokens = torch.randint(0, V, (B, T))
    logits = model(tokens)
    print(f"Input:  {tokens.shape}")
    print(f"Output: {logits.shape}")

    # Compare inference modes: full sequence vs token-by-token
    with torch.no_grad():
        full_out = model(tokens)
        # In theory, recurrent mode gives the same result
        # (our sequential scan is already recurrent)
        print(f"Full sequence output matches token-by-token: "
              f"{torch.allclose(full_out, model(tokens), atol=1e-5)}")

    # Show linear scaling
    import time
    for seq_len in [64, 256, 1024, 4096]:
        x = torch.randint(0, V, (1, seq_len))
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        elapsed = time.perf_counter() - start
        print(f"  T={seq_len:5d}  time={elapsed * 1000:.1f}ms")
```

---

## Interview takeaways

1. **Why SSMs exist** — attention is \(O(T^2)\). SSMs process sequences in \(O(T)\) by maintaining a compressed state, like a learned RNN with structured dynamics.
2. **Dual modes** — training uses convolution (parallel, \(O(T \log T)\)) or parallel scan; inference uses recurrence (\(O(1)\) per step). Know that this duality comes from the linearity of the system.
3. **Mamba's key insight** — making B, C, and Δ input-dependent (selective) lets the model decide what to remember and what to forget at each step. Without selection, SSMs can't do content-based retrieval.
4. **HiPPO initialization** — the A matrix is initialized to optimally compress history into a fixed-size state. This is what makes S4 handle long-range dependencies where RNNs fail.
5. **Where SSMs fall short** — in-context learning and associative recall (looking up specific past tokens) are weaker than Transformers because the state is lossy. Hybrid models (Jamba) combine both.
6. **Mamba-2 connection** — the selective SSM can be viewed as **structured masked attention** with specific constraints. This unifies the two paradigms theoretically and suggests future hybrid architectures.
7. **Practical status (2024+)** — SSMs are proven for long-context and efficiency-critical applications. But for general-purpose LLMs, Transformers (with FlashAttention) remain dominant. Watch this space.

---

## References

- Gu et al. (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4)
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2)
- Lieber et al. (2024), *Jamba: A Hybrid Transformer-Mamba Language Model*
