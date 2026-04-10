# State Space Models (SSMs) and Mamba

## Why This Matters for LLMs

Transformer decoders scale attention with sequence length \(T\) as \(O(T^2)\) in the naive attention matrix formulation, which creates a **quadratic memory and compute wall** for very long documents, high-resolution sequences, or multimodal timelines. **State space models** treat sequence modeling as evolving a hidden state through time with structured linear dynamics, enabling **linear scaling in \(T\)** for the core recurrence when formulated carefully. That matters for LLM discussions because it is no longer theoretical: Mamba-class blocks appear in open models and hybrid stacks aimed at long context without paying full attention cost everywhere.

The second paragraph of relevance is **quality versus efficiency**. Early SSMs like S4 showed strong long-range benchmarks, but the community still asked whether purely linear dynamics could rival Transformers on language. Mamba’s selective mechanism argued that **time-invariant** linear systems are too rigid for language: the model must decide **what to remember and what to discard** based on token content. That moves SSMs closer to the intuitive behavior people associate with attention, while retaining subquadratic structure. Interviewers increasingly ask candidates to compare **attention, RNN-like recurrence, and selective SSMs** on the same axes: complexity, state size, associative recall, and hardware utilization.

Third, **hybrid architectures** (for example interleaving attention layers with Mamba layers) are a practical compromise: attention handles content-based lookup in selective windows, while SSM layers propagate information cheaply across long spans. Understanding SSMs is therefore not an academic detour; it is background for reading system cards, research releases, and serving discussions where **context length** and **throughput per GPU** are first-class requirements.

---

## Continuous-Time Linear State Space Model

A continuous-time linear SSM defines a hidden state \(\mathbf{x}(t) \in \mathbb{R}^N\) driven by an input \(u(t)\) and producing an output \(y(t)\):

\[
\frac{d}{dt}\mathbf{x}(t) = A\mathbf{x}(t) + B u(t)
\]

!!! math-intuition "In Plain English"
    The derivative \(\frac{d}{dt}\mathbf{x}(t)\) is the **instantaneous change** of the hidden state. The term \(A\mathbf{x}(t)\) makes the state evolve even when the input is zero: internal dynamics can decay, oscillate, or mix coordinates depending on \(A\). The term \(B u(t)\) **injects** the current input into the state. Together they describe how memory updates as new observations arrive.

\[
y(t) = C\mathbf{x}(t) + D u(t)
\]

!!! math-intuition "In Plain English"
    The output \(y(t)\) is a **linear projection** of the hidden state through \(C\), plus an optional **direct path** from the current input through \(D\). In sequence modeling, \(C\) selects which aspects of memory become visible at the output, while \(D\) allows immediate passthrough without waiting for state dynamics.

Here \(A \in \mathbb{R}^{N \times N}\), \(B \in \mathbb{R}^{N \times 1}\), \(C \in \mathbb{R}^{1 \times N}\), and \(D\) is a scalar feedthrough term in the single-input single-output presentation. Multi-dimensional inputs and outputs generalize \(B\) and \(C\) to matrices.

??? deep-dive "Deep Dive: Why Continuous Time Appears in Sequence Modeling Papers"
    Continuous-time formulations let authors connect discrete recurrences to principled discretization schemes (zero-order hold, bilinear transforms, exponential integrators). In interviews, you can say: **discrete tokens are samples of an underlying process**, and discretization maps continuous parameters to step updates suitable for GPU scans.

---

## Discretization: From \(dt\) to Token Steps

For token index \(k\), choose a step size \(\Delta_k > 0\) (possibly learned or input-dependent in advanced models). A common exponential discretization uses:

\[
\bar{A}_k = \exp(\Delta_k A)
\]

!!! math-intuition "In Plain English"
    The discrete-time state transition \(\bar{A}_k\) is an **exponential of the continuous generator** scaled by the step \(\Delta_k\). Intuitively, it packages “how much one continuous-time step of length \(\Delta_k\)” advances the autonomous dynamics governed by \(A\). For stable systems, eigenvalues of \(\bar{A}_k\) often have magnitude below one so old information decays.

\[
\bar{B}_k = \int_{0}^{\Delta_k} \exp(A\tau)\, d\tau \, B
\]

!!! math-intuition "In Plain English"
    The integral expression for \(\bar{B}_k\) is the continuous-time impulse response **accumulated over one sampling interval**. It answers how much input influence lands in the state when \(u\) is held constant between ticks. Libraries implement this with numerically stable exponentials instead of naive matrix series at runtime.

For diagonal \(A\) and practical implementations, \(\bar{B}_k\) can be computed in stable closed forms without explicit numerical integration in the hot path.

The discrete recurrence becomes:

\[
\mathbf{x}_k = \bar{A}_k \mathbf{x}_{k-1} + \bar{B}_k u_k
\]

!!! math-intuition "In Plain English"
    This is the **Markov update**: new state equals transformed old state plus an input injection. Compare to an RNN: same structure, but \(\bar{A}_k\) and \(\bar{B}_k\) have structured parameterizations in SSM work rather than fully general learned matrices in every cell.

\[
y_k = C \mathbf{x}_k + D u_k
\]

!!! math-intuition "In Plain English"
    The discrete observation equation matches the continuous one: read out through \(C\), add direct feedthrough through \(D\). At token \(k\), the output depends on the updated state after incorporating \(u_k\). **Discretization maps continuous dynamics to token-indexed updates.** Smaller \(\Delta_k\) tends to make each step a small perturbation of the previous state; larger \(\Delta_k\) can move the state farther in one jump, which is one knob selective models learn per token in Mamba-style formulations.

!!! example "Worked Example: Scalar SSM with One-Dimensional State"
    Let \(N = 1\), \(A = -1\), \(B = 1\), \(C = 1\), \(D = 0\). Let \(\Delta = 1\) fixed. Then \(\bar{A} = \exp(-1) \approx 0.368\) and \(\bar{B}\) simplifies in the scalar case to a value on the order of \((1 - \bar{A}) B\) under common ZOH assumptions, here roughly \(1 - 0.368 = 0.632\).

    If \(u_0 = 1.0\), \(u_1 = 0.0\), \(u_2 = 2.0\), starting from \(x_{-1} = 0\):

    - \(x_0 = 0.368 \cdot 0 + 0.632 \cdot 1.0 = 0.632\), \(y_0 = x_0\)
    - \(x_1 = 0.368 \cdot 0.632 + 0.632 \cdot 0.0 \approx 0.233\)
    - \(x_2 = 0.368 \cdot 0.233 + 0.632 \cdot 2.0 \approx 0.086 + 1.264 = 1.350\)

    The state **decays** old evidence through \(\bar{A}\) and **adds** new evidence through \(\bar{B} u_k\).

---

## Dual Computation Modes: Recurrence and Convolution

For **time-invariant** linear SSMs (where \(\bar{A}\) and \(\bar{B}\) do not change with \(k\)), the mapping from inputs \(u_k\) to outputs \(y_k\) is linear time-invariant. Then the sequence can be computed either by recurrence or as a convolution with a structured kernel.

**Recurrence (sequential):**

\[
\mathbf{x}_k = \bar{A} \mathbf{x}_{k-1} + \bar{B} u_k
\]

!!! math-intuition "In Plain English"
    Recurrence updates the state one time step at a time. This is the default mental model for **online** inference: you keep \(\mathbf{x}_k\) in memory and update it when the next token arrives.

\[
y_k = C\mathbf{x}_k
\]

!!! math-intuition "In Plain English"
    The output at step \(k\) is emitted **after** the state absorbs \(u_k\). In implementations, \(y_k\) may include \(D u_k\) as well; the table below focuses on the recurrent core.

**Convolution (parallel):** there exists an impulse response kernel \(\bar{K}\) such that \(y = \bar{K} * u\) in the appropriate sense over finite horizons.

!!! math-intuition "In Plain English"
    Convolution says the entire output sequence is a **linear map** from the entire input sequence when the system is time-invariant. FFT-based methods compute this map quickly for long training sequences. When parameters vary with \(k\), the clean convolution picture breaks and you fall back to scans or sequential loops unless you introduce structured approximations.

| Mode | Typical complexity | Strength | Weakness |
| --- | --- | --- | --- |
| Recurrence | \(O(T)\) sequential steps | Constant memory per step for inference | Harder to saturate GPU parallelism |
| Convolution / FFT | \(O(T \log T)\) in many implementations | Parallel across time in training | Requires time-invariance for fixed kernel |

??? deep-dive "Deep Dive: Why Mamba Uses a Scan"
    When \(\Delta_k\), \(B_k\), and \(C_k\) depend on the input, kernels change per position. A parallel **prefix scan** can still compute the recurrence in \(O(T)\) work with careful engineering, but it is not the same as one static FFT convolution kernel for the entire sequence.

---

## S4 and HiPPO Initialization in Plain English

S4 (Structured State Spaces for Sequences) made long-range modeling practical by combining:

- Structured parameterizations of \(A\) (often diagonal or diagonal-plus-low-rank) for speed and stability.
- **HiPPO** initializations for \(A\) matrices designed so the hidden state acts like a **compressed memory** of recent history with mathematically motivated decay structure.
- Fast kernel computation for training using FFT-based methods in the time-invariant setting.

!!! math-intuition "In Plain English"
    HiPPO is often described informally as: **choose the internal dynamics so that the state stores a useful summary of the past**, not random oscillations. Random initialization of recurrent dynamics frequently forgets or explodes. HiPPO-derived structure gives a strong inductive bias for remembering signals over long horizons, which is exactly what long-sequence benchmarks punish when models behave like weak RNNs.

---

## Mamba: Selective Gating on \(B\), \(C\), and \(\Delta\)

Mamba makes key parameters **input-dependent** (exact projection shapes differ by implementation; the core idea is **functions of \(x_k\)**, not global constants):

\[
B_k = W_B x_k
\]

!!! math-intuition "In Plain English"
    \(B_k\) controls **how strongly the current input writes into the state** for step \(k\). When \(B_k\) depends on \(x_k\), the model can route different tokens into memory with different strengths, similar in spirit to deciding which facts deserve storage.

\[
C_k = W_C x_k
\]

!!! math-intuition "In Plain English"
    \(C_k\) controls **how the state is read out** at step \(k\). Input-dependent readouts let the model emphasize different memory coordinates depending on content, rather than using one fixed projection for all tokens.

\[
\Delta_k = \text{softplus}(W_\Delta x_k)
\]

!!! math-intuition "In Plain English"
    \(\Delta_k\) sets the **effective step size** of the discrete update. Larger \(\Delta_k\) often behaves like a stronger update or a faster integration step, which can **overwrite** older information; smaller \(\Delta_k\) makes the state evolve more slowly, preserving context longer. Softplus keeps \(\Delta_k\) positive and numerically stable.

!!! math-intuition "In Plain English"
    Together, input-dependent \(B_k\), \(C_k\), and \(\Delta_k\) let the system vary **write strength**, **read direction**, and **forgetting speed** per token. That is the selective behavior people want from attention-like mechanisms without full pairwise attention cost.

---

## Worked Example: Two-Dimensional State Over Five Tokens

This example uses **made-up but numerically consistent** 2D vectors to show accumulation and forgetting. The goal is pedagogy, not a production parameterization.

**State:** \(\mathbf{h}_k \in \mathbb{R}^2\). Initialize \(\mathbf{h}_0 = [0, 0]^\top\).

**Simplified discrete update (per channel diagonal decay for readability):**

\[
\mathbf{h}_k = \alpha_k \odot \mathbf{h}_{k-1} + \beta_k \odot \mathbf{e}_k
\]

!!! math-intuition "In Plain English"
    This toy recurrence separates **carry** from **new write**. The vector \(\alpha_k\) scales down the previous state per coordinate (forgetting). The vector \(\beta_k\) scales how much of the current embedding \(\mathbf{e}_k\) is written into memory. Elementwise multiplication makes each coordinate evolve independently in this pedagogical sketch.

where \(\odot\) is elementwise multiplication, \(\mathbf{e}_k \in \mathbb{R}^2\) is an embedding vector for token \(k\), and \(\alpha_k, \beta_k \in \mathbb{R}^2\) are decay and input gates in \([0,1]\).

Let the five tokens produce:

| Step \(k\) | \(\mathbf{e}_k\) | \(\alpha_k\) | \(\beta_k\) |
| --- | --- | --- | --- |
| 1 | \([1.0,\ 0.0]\) | \([0.5,\ 0.5]\) | \([1.0,\ 0.8]\) |
| 2 | \([0.0,\ 2.0]\) | \([0.8,\ 0.7]\) | \([0.6,\ 1.0]\) |
| 3 | \([0.5,\ 0.5]\) | \([0.9,\ 0.6]\) | \([0.5,\ 0.5]\) |
| 4 | \([2.0,\ 0.0]\) | \([0.7,\ 0.8]\) | \([1.0,\ 0.2]\) |
| 5 | \([0.0,\ 0.0]\) | \([0.6,\ 0.6]\) | \([0.4,\ 0.4]\) |

**Compute \(\mathbf{h}_1\):**

\[
\mathbf{h}_1 = \alpha_1 \odot \mathbf{h}_0 + \beta_1 \odot \mathbf{e}_1
= [0.5\cdot 0,\ 0.5\cdot 0] + [1.0\cdot 1.0,\ 0.8\cdot 0.0]
= [1.0,\ 0.0]
\]

**Compute \(\mathbf{h}_2\):**

\[
\alpha_2 \odot \mathbf{h}_1 = [0.8\cdot 1.0,\ 0.7\cdot 0.0] = [0.8,\ 0.0]
\]
\[
\beta_2 \odot \mathbf{e}_2 = [0.6\cdot 0.0,\ 1.0\cdot 2.0] = [0.0,\ 2.0]
\]
\[
\mathbf{h}_2 = [0.8,\ 2.0]
\]

**Compute \(\mathbf{h}_3\):**

\[
\alpha_3 \odot \mathbf{h}_2 = [0.9\cdot 0.8,\ 0.6\cdot 2.0] = [0.72,\ 1.20]
\]
\[
\beta_3 \odot \mathbf{e}_3 = [0.5\cdot 0.5,\ 0.5\cdot 0.5] = [0.25,\ 0.25]
\]
\[
\mathbf{h}_3 = [0.97,\ 1.45]
\]

**Compute \(\mathbf{h}_4\):**

\[
\alpha_4 \odot \mathbf{h}_3 = [0.7\cdot 0.97,\ 0.8\cdot 1.45] = [0.679,\ 1.160]
\]
\[
\beta_4 \odot \mathbf{e}_4 = [1.0\cdot 2.0,\ 0.2\cdot 0.0] = [2.0,\ 0.0]
\]
\[
\mathbf{h}_4 = [2.679,\ 1.160]
\]

**Compute \(\mathbf{h}_5\):**

\[
\alpha_5 \odot \mathbf{h}_4 = [0.6\cdot 2.679,\ 0.6\cdot 1.160] = [1.6074,\ 0.696]
\]
\[
\beta_5 \odot \mathbf{e}_5 = [0.4\cdot 0.0,\ 0.4\cdot 0.0] = [0.0,\ 0.0]
\]
\[
\mathbf{h}_5 \approx [1.607,\ 0.696]
\]

**Interpretation:** early steps write directional evidence into the two channels. Later steps scale down prior state through \(\alpha\) and add new evidence through \(\beta \odot \mathbf{e}\). The fifth token carries zero embedding, yet the state still decays and retains partial history: forgetting is gradual rather than instantaneous.

!!! example "Worked Example"
    The table above is the worked example in compact form: you can trace \(\mathbf{h}_1\) through \(\mathbf{h}_5\) and point to which coordinates remember earlier tokens and which coordinates are overwritten when new evidence arrives.

---

## SSM versus Transformer: Comparison Table

| Topic | Transformer (causal self-attention) | SSM / Mamba-style |
| --- | --- | --- |
| **Training complexity (attention layer)** | \(O(T^2 d)\) for full attention | \(O(T \cdot \text{state work})\) for recurrence or scan |
| **Inference per new token** | Needs KV cache scaling with past length | Recurrent state size often fixed (does not grow with \(T\) like KV) |
| **Content-based pairwise interactions** | Direct via attention weights | Approximated via selective gates and state dynamics |
| **Parallelism** | Excellent with GPUs | Good with scans; different kernel patterns |
| **Associative recall of specific past tokens** | Strong when attention can attend back | Can be weaker unless hybridized |

---

## Hybrid Architectures: Jamba and Attention Interleaving

**Jamba** (Lieber et al., 2024) combines Transformer attention blocks with Mamba blocks and can incorporate MoE for capacity. The engineering motivation is simple: **attention handles local and content-sharp interactions**, while **SSM layers propagate information across long spans at lower cost** than full attention everywhere.

!!! math-intuition "In Plain English"
    A hybrid stack is an admission that no single primitive wins every metric. Attention is flexible but expensive. SSMs are efficient carriers of state but can be lossy compared to full attention for certain retrieval-style tasks. Interleaving trades off complexity for a better total package.

---

## Full Code: Selective SSM Block with Inference Timing

The following PyTorch module implements a **minimal selective SSM path** with per-token \(\Delta\), and input-dependent \(B\) and \(C\) projections. Variable names avoid shadowing dimensions. The `__main__` block runs a short timing loop across sequence lengths.

```python
"""
Minimal selective state space block (Mamba-style) with sequential scan and timing demo.

Dependencies: torch
Educational code: not a byte-for-byte reproduction of official CUDA kernels.
"""
from __future__ import annotations

import time
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Input-dependent B, C, and delta; diagonal A in log-space per channel.
    State shape: (batch, d_inner, d_state)
    """

    def __init__(self, d_inner: int, d_state: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        a = torch.arange(1, d_state + 1, dtype=torch.float32).view(1, d_state).repeat(d_inner, 1)
        self.a_log = nn.Parameter(torch.log(a))

        self.proj_b = nn.Linear(d_inner, d_state, bias=False)
        self.proj_c = nn.Linear(d_inner, d_state, bias=False)
        self.proj_delta = nn.Linear(d_inner, d_inner, bias=True)
        nn.init.constant_(self.proj_delta.bias, 1.0)

        self.d_out = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, d_inner)
        returns y: (batch, time, d_inner)
        """
        b_batch, time_steps, d_in = x.shape
        assert d_in == self.d_inner

        a_neg = -torch.exp(self.a_log)
        b_mat = self.proj_b(x)
        c_mat = self.proj_c(x)
        delta = F.softplus(self.proj_delta(x))

        outputs: list[torch.Tensor] = []
        state = torch.zeros(b_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(time_steps):
            delta_t = delta[:, t, :]
            b_t = b_mat[:, t, :]
            c_t = c_mat[:, t, :]
            x_t = x[:, t, :]

            decay = torch.exp(delta_t.unsqueeze(-1) * a_neg)
            delta_b = delta_t.unsqueeze(-1) * b_t.unsqueeze(1)
            state = decay * state + delta_b * x_t.unsqueeze(-1)
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1)
            y_t = y_t + x_t * self.d_out
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaLikeBlock(nn.Module):
    """Conv1d local mixing + selective SSM + gating."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        x_c = x_branch.transpose(1, 2)
        x_c = self.conv1d(x_c)[:, :, : x_branch.size(1)]
        x_c = x_c.transpose(1, 2)
        x_c = F.silu(x_c)

        y = self.ssm(x_c)
        y = y * F.silu(z_branch)
        y = self.out_proj(y)
        return residual + y


class TinyModel(nn.Module):
    """Small stack for timing; not trained."""

    def __init__(self, vocab_size: int, d_model: int, layers: int, d_state: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(MambaLikeBlock(d_model, d_state=d_state) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(ids)
        for block in self.blocks:
            h = block(h)
        return self.lm_head(self.norm(h))


def benchmark_forward(model: nn.Module, lengths: Sequence[int], device: torch.device) -> None:
    model.eval()
    vocab_size = model.lm_head.out_features
    for seq_len in lengths:
        x = torch.randint(0, vocab_size, (1, seq_len), device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"seq_len={seq_len:5d}  seconds={t1 - t0:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = 4096
    d_model = 256
    layers = 4
    d_state = 16
    model = TinyModel(vocab, d_model, layers, d_state).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters={params:,}  device={device}")
    sample = torch.randint(0, vocab, (2, 128), device=device)
    logits = model(sample)
    print(f"logits shape={tuple(logits.shape)}")
    benchmark_forward(model, (64, 256, 1024, 4096), device)
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Why is naive self-attention \(O(T^2)\), and what bottleneck does that create?
    *Answer:* Full attention materializes **pairwise** scores for all \((i,j)\), giving \(O(T^2)\) work and memory per layer/head. For long \(T\), this hits **HBM bandwidth** and **latency** limits— the “quadratic wall” motivating subquadratic layers.
    2. Write the continuous-time SSM and explain each of \(A\), \(B\), \(C\), and \(D\).
    *Answer:* \(\dot{x}=Ax+Bu\), \(y=Cx+Du\). **\(A\)**: autonomous dynamics (how state decays/mixes). **\(B\)**: input injection. **\(C\)**: readout from state. **\(D\)**: **direct** feedthrough from current input to output (skip).
    3. What changes when you discretize with step \(\Delta\)?
    *Answer:* Continuous \((A,B)\) map to discrete **\(\bar{A}_\Delta\approx \exp(\Delta A)\)** and **\(\bar{B}_\Delta\)** (input over one step), giving **\(x_k=\bar{A}_k x_{k-1}+\bar{B}_k u_k\)** at token times. \(\Delta\) can be **global, learned, or input-dependent** (Mamba).
    4. Explain recurrence versus convolution for time-invariant linear SSMs.
    *Answer:* **Time-invariant** linear SSMs are **LTI**: outputs equal **convolution** of inputs with a fixed impulse response—train with **FFT** in \(O(T\log T)\). **Recurrence** is \(O(T)\) sequential but **online** with **O(1)** state per step—preferred for **streaming** inference.
    5. What problem does HiPPO initialization aim to solve?
    *Answer:* Random recurrent dynamics often **forget** or **explode**. HiPPO structures **\(A\)** so the state approximates a **compressed memory** of past inputs (polynomial/exponential bases)—better **long-range** behavior before large-scale training.
    6. What does Mamba change versus time-invariant S4-style models?
    *Answer:* Mamba makes **\(B_k, C_k, \Delta_k\)** **input-dependent** (functions of \(x_k\)): **selective** memory—what to write, read, and how fast to integrate—vs S4’s fixed dynamics per layer where the same kernel applies at every position.
    7. Why does selectivity break the simplest fixed-kernel FFT training picture?
    *Answer:* When \(\Delta_k,B_k,C_k\) **vary with \(k\)**, the map is **not** a single LTI convolution—there is **one kernel per step** or a **varying** recurrence. You cannot use **one** global FFT multiply; you use **scans** or sequential steps (though parallel scan algorithms still help).
    8. Compare inference memory for Transformer KV cache versus fixed-size SSM state.
    *Answer:* Transformer **KV** grows **linearly with \(T\)** (per layer, per head). SSM recurrence keeps a **fixed-size** state \(x_k\in\mathbb{R}^{N}\) (per layer/channel)—**\(O(1)\)** extra memory in \(T\) for the recurrence itself, attractive for **long** streams (attention may still be added in hybrids).
    9. Name a hybrid architecture pattern and justify why teams use it.
    *Answer:* **Jamba**-style: **interleave** Transformer **attention** blocks with **Mamba/SSM** blocks—attention gives **sharp content-based** routing and local precision; SSM **cheaply propagates** information over long spans—balancing quality vs cost.
    10. How does Mamba-2 connect SSMs and attention at a high level?
    *Answer:* **Mamba-2** (SSM duality) shows structured SSMs and **linear attention**-like mechanisms can be **unified** in one framework with **fast algorithms** (chunkwise states)—conceptually bridging **recurrent** updates and **attention-like** tensor programs.

!!! interview "Follow-up Probes"
    - "What kinds of tasks stress associative recall versus long-horizon compression?"
    - "How would you benchmark a hybrid model fairly against a dense Transformer?"
    - "What hardware metrics matter for scan kernels on GPUs?"
    - "When would you still pay for attention despite quadratic cost?"

!!! key-phrases "Key Phrases to Use in Interviews"
    - "Linear-time sequence modeling via structured state updates"
    - "Discretization maps continuous dynamics to token steps"
    - "Training parallelism via convolution or scan; inference via recurrence"
    - "Selective parameters make memory input-dependent"
    - "Hybrids interleave attention for sharp lookup with SSM for efficient propagation"

---

## References

- Gu et al. (2021), *Efficiently Modeling Long Sequences with Structured State Spaces* (S4)
- Gu & Dao (2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*
- Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality* (Mamba-2)
- Lieber et al. (2024), *Jamba: A Hybrid Transformer-Mamba Language Model*
