# FlashAttention: Fast and Memory-Efficient Exact Attention

**Original Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Timeline:** FlashAttention (2022) → FlashAttention-2 (2023) → FlashAttention-3 (2024) → FlashAttention-4 (2026)  
**Links:** [FA-1 arXiv:2205.14135](https://arxiv.org/abs/2205.14135) &nbsp;|&nbsp; [FA-2 arXiv:2307.08691](https://arxiv.org/abs/2307.08691) &nbsp;|&nbsp; [FA-3 arXiv:2407.08608](https://arxiv.org/abs/2407.08608)

---

## TL;DR

FlashAttention computes **exact** self-attention while avoiding materializing the full \(N \times N\) attention matrix in GPU high-bandwidth memory (HBM). It uses **tiling** and **kernel fusion** to keep intermediate results in fast SRAM, reducing memory reads/writes by orders of magnitude. The result: faster training, longer sequences, and lower memory usage — all without approximating the attention output. Four generations of improvements have pushed utilization from ~25% (FA-1) to **71% on B200** (FA-4).

---

## Evolution at a Glance

| Version | Year | GPU Target | Peak Throughput | Key Innovation |
|---------|------|-----------|----------------|----------------|
| FlashAttention | 2022 (NeurIPS) | A100 | ~25–40% utilization | IO-aware tiling + online softmax |
| FlashAttention-2 | Jul 2023 | A100 | 50–73% utilization (~230 TFLOPs/s) | Better parallelism + fewer non-matmul FLOPs |
| FlashAttention-3 | Jul 2024 | H100 (Hopper) | 75% utilization (~740 TFLOPs/s) | Asynchronous pipelining + FP8 support |
| FlashAttention-4 | Mar 2026 | B200 (Blackwell) | 71% utilization (~1605 TFLOPs/s) | Algorithm-kernel co-design for asymmetric HW |

---

## Why This Paper Series Matters

FlashAttention is now the **default attention implementation** in PyTorch 2, vLLM, and virtually every LLM serving framework:

1. **Enables long context:** Process 128K+ tokens without running out of memory
2. **Faster training:** 2-4× speedup on attention computation
3. **Exact, not approximate:** Unlike sparse or linear attention, output is mathematically identical
4. **IO-awareness:** Introduced the concept of designing algorithms around GPU memory hierarchy
5. **Universal adoption:** Used by every major model and framework
6. **Hardware co-evolution:** Each generation tracks new GPU architectures (Ampere → Hopper → Blackwell)

---

## Key Concepts Explained Simply

### The GPU Memory Problem

A modern GPU (A100) has two types of memory:
- **SRAM (on-chip):** ~20 MB, extremely fast (~19 TB/s bandwidth)
- **HBM (off-chip):** ~80 GB, much slower (~2 TB/s bandwidth)

Standard attention computes \(S = QK^\top\) (an \(N \times N\) matrix), stores it in HBM, applies softmax, then multiplies by \(V\). For \(N = 8192\) and 32 heads, that's \(32 \times 8192^2 \times 4 = 8.6\) GB just for the attention matrices — and this must be read/written multiple times.

**The bottleneck isn't FLOPs — it's memory bandwidth.**

### FlashAttention's Solution: Tiling

Instead of computing the full \(N \times N\) matrix:

1. **Split** \(Q, K, V\) into small blocks that fit in SRAM
2. **Compute** attention for each block pair in SRAM
3. **Accumulate** results using an online softmax trick
4. **Never** write the full \(N \times N\) matrix to HBM

The online softmax algorithm computes exact softmax across blocks by tracking running maximum and sum statistics.

### Online Softmax: The Key Trick

Normal softmax requires two passes: first to compute the max (for numerical stability), then to compute exponentials and normalize. FlashAttention does it in **one streaming pass** by maintaining running statistics that can be composed across blocks:

1. Process block 1: compute local max \(m_1\), local sum \(l_1\)
2. Process block 2: update global max \(m = \max(m_1, m_2)\), rescale previous sums
3. Continue for all blocks — final result is identical to full softmax

---

## The Math — Explained Step by Step

### Standard Attention (What We're Optimizing)

\[
S = QK^\top / \sqrt{d_k}, \quad P = \text{softmax}(S), \quad O = PV
\]

The output \(O\) is exact — FlashAttention produces the **same** result.

### IO Complexity Analysis

**Standard attention memory accesses:**
- Read \(Q, K, V\) from HBM: \(O(Nd)\)
- Write \(S = QK^\top\) to HBM: \(O(N^2)\)
- Read \(S\), write \(P = \text{softmax}(S)\) to HBM: \(O(N^2)\)
- Read \(P, V\), write \(O = PV\) to HBM: \(O(N^2 + Nd)\)
- **Total HBM accesses:** \(O(N^2 + Nd)\)

**FlashAttention memory accesses:**
- Read \(Q, K, V\) from HBM: \(O(Nd)\)
- Write final \(O\) to HBM: \(O(Nd)\)
- **Total HBM accesses:** \(O(N^2 d^2 / M)\) where \(M\) is SRAM size
- When \(M > d^2\) (typical), this is \(O(Nd)\) — linear!

### Online Softmax

For blocks \(B_1, B_2\) of the key dimension:

**Block 1:**

\[
m_1 = \max(S_{B_1}), \quad l_1 = \sum \exp(S_{B_1} - m_1), \quad O_1 = \frac{\exp(S_{B_1} - m_1)}{l_1} V_{B_1}
\]

**After Block 2:**

\[
m = \max(m_1, m_2), \quad l = l_1 e^{m_1 - m} + l_2 e^{m_2 - m}
\]

\[
O = \frac{l_1 e^{m_1 - m}}{l} O_1 + \frac{l_2 e^{m_2 - m}}{l} O_2
\]

This is mathematically equivalent to computing softmax over the entire sequence at once.

---

## Python Implementation

```python
import numpy as np


def standard_attention(Q, K, V):
    """
    Standard attention — materializes full N×N matrix.
    Q, K: [N, d], V: [N, d]
    """
    d_k = Q.shape[-1]
    S = (Q @ K.T) / np.sqrt(d_k)
    S_max = np.max(S, axis=-1, keepdims=True)
    P = np.exp(S - S_max)
    P = P / np.sum(P, axis=-1, keepdims=True)
    O = P @ V
    return O


def flash_attention(Q, K, V, block_size=4):
    """
    FlashAttention — tiled computation without materializing full N×N matrix.
    Produces the EXACT same output as standard_attention.
    """
    N, d = Q.shape
    O = np.zeros_like(Q)
    l = np.zeros((N, 1))  # Running sum of exponentials
    m = np.full((N, 1), -np.inf)  # Running max

    n_blocks = (N + block_size - 1) // block_size

    for j in range(n_blocks):
        j_start = j * block_size
        j_end = min(j_start + block_size, N)

        K_block = K[j_start:j_end]
        V_block = V[j_start:j_end]

        # Compute attention scores for this key block
        S_block = (Q @ K_block.T) / np.sqrt(d)

        # Online softmax update
        m_block = np.max(S_block, axis=-1, keepdims=True)
        m_new = np.maximum(m, m_block)

        # Rescale previous accumulation
        exp_old = np.exp(m - m_new)
        exp_new = np.exp(S_block - m_new)

        l = l * exp_old + np.sum(exp_new, axis=-1, keepdims=True)
        O = O * exp_old + exp_new @ V_block

        m = m_new

    O = O / l
    return O


def memory_comparison(seq_lengths, d=128, dtype_bytes=4):
    """Compare memory usage of standard vs flash attention."""
    print(f"{'Seq Len':>10} {'Standard':>15} {'Flash':>15} {'Savings':>10}")
    print("-" * 55)
    for N in seq_lengths:
        # Standard: stores N×N attention matrix
        std_bytes = N * N * dtype_bytes
        # Flash: only stores block-sized intermediates
        block_size = min(256, N)
        flash_bytes = 2 * N * d * dtype_bytes + N * block_size * dtype_bytes
        savings = std_bytes / flash_bytes
        print(f"{N:>10,} {std_bytes/1e9:>13.2f}GB {flash_bytes/1e9:>13.4f}GB {savings:>9.1f}×")


def verify_correctness(N=32, d=16, block_size=8):
    """Verify FlashAttention produces exact same output."""
    np.random.seed(42)
    Q = np.random.randn(N, d)
    K = np.random.randn(N, d)
    V = np.random.randn(N, d)

    out_standard = standard_attention(Q, K, V)
    out_flash = flash_attention(Q, K, V, block_size=block_size)

    max_diff = np.max(np.abs(out_standard - out_flash))
    return max_diff


def io_analysis(N, d, M_sram):
    """
    Compare HBM memory accesses.
    N: sequence length, d: head dimension, M_sram: SRAM size in elements
    """
    standard_io = 3 * N * d + 3 * N * N  # Read QKV + write S, P, read for O
    flash_io = 4 * N * d * (N * d / M_sram)  # Tiled reads

    return {
        "standard_io": standard_io,
        "flash_io": flash_io,
        "ratio": standard_io / flash_io
    }


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Correctness verification
    max_diff = verify_correctness(N=64, d=32, block_size=8)
    print(f"Correctness check — max difference: {max_diff:.2e} (should be ~1e-15)")

    # Memory comparison
    print("\n--- Memory Usage Comparison ---")
    memory_comparison([512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])

    # IO analysis
    print("\n--- IO Analysis (HBM accesses) ---")
    for N in [1024, 4096, 16384]:
        result = io_analysis(N, d=128, M_sram=100_000)
        print(f"  N={N:>6}: Standard={result['standard_io']:>12,.0f}, "
              f"Flash={result['flash_io']:>12,.0f}, "
              f"Ratio={result['ratio']:.1f}×")

    # Demo with different block sizes
    print("\n--- Block Size Effect ---")
    N, d = 64, 16
    Q = np.random.randn(N, d)
    K = np.random.randn(N, d)
    V = np.random.randn(N, d)
    ref = standard_attention(Q, K, V)

    for bs in [4, 8, 16, 32, 64]:
        out = flash_attention(Q, K, V, block_size=bs)
        err = np.max(np.abs(out - ref))
        print(f"  Block size {bs:>3}: max error = {err:.2e}")
```

---

## FlashAttention-2 (July 2023): Better Parallelism

FlashAttention-1 achieved only 25–40% of theoretical peak FLOPs on A100. FlashAttention-2 closes this gap with three algorithmic changes:

### What Changed

1. **Fewer non-matmul FLOPs.** GPU Tensor Cores execute matmul at 16× the throughput of other operations. FA-2 restructures the online softmax to minimize scalar rescaling, moving work into matmul form.

2. **Parallelism across sequence length.** FA-1 parallelized over batch and heads only. FA-2 additionally parallelizes over the **sequence-length dimension** (outer loop over Q blocks), increasing occupancy on modern GPUs with 108+ SMs.

3. **Better warp-level work partitioning.** Within each thread block, FA-2 splits work across warps to avoid redundant shared memory reads/writes, reducing the "4-warp → shared-memory → 4-warp" synchronization pattern from FA-1.

### Result

~2× speedup over FA-1, reaching **230 TFLOPs/s** (73% utilization) on A100. End-to-end GPT training is 2.2× faster than a baseline without FlashAttention.

---

## FlashAttention-3 (July 2024): Hopper Asynchrony + FP8

The H100 (Hopper) introduced hardware features — **TMA** (Tensor Memory Accelerator) and **warp-group MMA** — that FA-2's kernel structure couldn't exploit.

### Key Techniques

1. **Warp-specialization with producer–consumer overlap.** Dedicated warp groups issue TMA loads (producer) while other warp groups run Tensor Core matmuls (consumer). Data movement and compute are **fully overlapped**.

2. **Ping-pong scheduling.** Two warp groups alternate: while one computes softmax + rescaling on block \(j\), the other runs matmul on block \(j+1\). This hides the softmax latency.

3. **FP8 with block quantization.** Leverages Hopper's native FP8 Tensor Cores. Per-block dynamic quantization + **incoherent processing** (random orthogonal rotation of Q/K) reduces quantization error by 2.6× vs. naive FP8.

### Result

- **FP16:** Up to 740 TFLOPs/s (75% utilization on H100) — 1.5–2× faster than FA-2.
- **FP8:** Close to 1.2 PFLOPs/s with 2.6× lower error than baseline FP8.

---

## FlashAttention-4 (March 2026): Blackwell Co-Design

Blackwell GPUs (B200/GB200) have **4× more Tensor Core throughput** than Hopper, but shared memory bandwidth and special function units (SFUs) grew only ~1.5×. This **asymmetric scaling** means the exponential function in softmax becomes the new bottleneck.

### Key Innovations

1. **Exponential function emulation.** Instead of calling the hardware SFU `exp()`, FA-4 approximates \(\exp(x)\) with a polynomial computed entirely on Tensor Cores — converting a serial SFU bottleneck into a parallel matmul.

2. **Forward pass pipeline.** New 5-stage software pipeline exploits Blackwell's fully asynchronous MMA instructions and larger tile sizes. The pipeline interleaves: TMA load → matmul (QK) → exp emulation → matmul (PV) → writeback.

3. **Backward pass: tensor memory + 2-CTA MMA.** Intermediate results are cached in Blackwell's new **tensor memory** (per-SM scratchpad separate from shared memory), relieving shared-memory bandwidth. Uses 2-CTA cooperative MMA for larger tiles.

4. **Deterministic execution.** Full support for reproducible forward and backward passes — critical for debugging and compliance.

### Result

Up to **1605 TFLOPs/s** (71% utilization on B200), 1.3× faster than cuDNN 9.13, and 2.7× faster than Triton. FlexAttention integration enables custom attention variants (causal, sliding window, etc.) with FA-4 as the backend.

---

## Interview Importance

FlashAttention is a **top-5 systems topic** in LLM interviews. It tests whether you understand GPU architecture and the distinction between compute-bound and memory-bound operations.

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: Why is attention often memory-bound, not compute-bound, on GPUs?

**Answer:** Modern GPUs have enormous compute throughput (e.g., A100: 312 TFLOPS for bfloat16) but limited memory bandwidth (~2 TB/s for HBM). Standard attention materializes an \(N \times N\) matrix that must be written to and read from HBM multiple times. For typical head dimensions (\(d = 128\)), the ratio of memory accesses to compute operations means the GPU spends most of its time **waiting for data** rather than computing. FlashAttention reduces HBM traffic by keeping intermediates in SRAM.

### Q2: What does "exact" mean vs. sparse or linear attention?

**Answer:**
- **Exact (FlashAttention):** Produces mathematically identical output to standard attention. No information loss. Just a different computation order.
- **Sparse attention:** Only computes attention for a subset of positions (e.g., local window + global tokens). Faster but **loses** information from dropped positions.
- **Linear attention:** Approximates softmax attention with kernel functions, reducing complexity from \(O(N^2)\) to \(O(N)\). Approximation error means outputs differ.

FlashAttention is preferred because you get the speed benefit without sacrificing quality.

### Q3: How does FlashAttention interact with long context and batch size in serving?

**Answer:**
- **Long context:** FlashAttention makes long context practical by avoiding the \(O(N^2)\) memory bottleneck. Without it, 128K context requires ~64GB just for attention matrices (per layer, per head). With it, memory scales linearly with sequence length.
- **Batch size:** Lower memory per sequence means you can fit more sequences in a batch, improving GPU utilization and throughput
- **KV cache:** FlashAttention-2 also optimizes the KV cache access pattern during inference, which is critical for serving
- **Trade-off:** Longer sequences still have quadratic compute (FLOPs), but FlashAttention removes the memory wall that previously prevented reaching that compute

### Q4: Explain the online softmax trick.

**Answer:** Standard softmax requires knowing the global maximum across all elements for numerical stability. The online softmax algorithm maintains running statistics (\(m\) = running max, \(l\) = running sum of exponentials) that can be updated block by block. When processing a new block, you update: (1) new global max, (2) rescale previous sums by \(e^{m_{\text{old}} - m_{\text{new}}}\), (3) add new block's contribution. The final result is identical to computing softmax over the full sequence at once.

---

## Connections to Other Papers

- **Transformer** → FlashAttention optimizes the core attention computation
- **Mistral** → Uses FlashAttention-2 with sliding window attention
- **Mamba/SSMs** → Alternative approach: avoid quadratic attention entirely
- **LLaMA** → Training and inference use FlashAttention
- **DeepSeek-V3** → Uses FlashAttention-3 FP8 for cost-efficient training
- **FlexAttention** → PyTorch API that now uses FA-4 as backend for custom attention masks

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Tiled attention computation in SRAM, avoid HBM writes |
| Output | Mathematically exact (not approximate) |
| Key trick | Online softmax — composable running max + sum statistics |
| Standard memory | \(O(N^2)\) for attention matrix |
| Flash memory | \(O(N)\) — never materializes full matrix |
| FA-1 (2022) | IO-aware tiling; 25–40% A100 utilization |
| FA-2 (2023) | Sequence-parallel + warp partitioning; 73% A100 utilization |
| FA-3 (2024) | Hopper async pipelining + FP8; 75% H100 utilization |
| FA-4 (2026) | Blackwell co-design, exp emulation on Tensor Cores; 71% B200, 1605 TFLOPs/s |
| Adoption | Default in PyTorch 2, vLLM, every major framework |
| GPU insight | Each generation tracks the hardware bottleneck: HBM bandwidth → occupancy → async overlap → SFU throughput |
