# FlashAttention: Fast and Memory-Efficient Exact Attention

**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** NeurIPS  
**Link:** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

---

## TL;DR

FlashAttention computes **exact** self-attention while avoiding materializing the full \(N \times N\) attention matrix in GPU high-bandwidth memory (HBM). It uses **tiling** and **kernel fusion** to keep intermediate results in fast SRAM, reducing memory reads/writes by orders of magnitude. The result: faster training, longer sequences, and lower memory usage — all without approximating the attention output.

---

## Why This Paper Matters

FlashAttention is now the **default attention implementation** in PyTorch 2, vLLM, and virtually every LLM serving framework:

1. **Enables long context:** Process 128K+ tokens without running out of memory
2. **Faster training:** 2-4× speedup on attention computation
3. **Exact, not approximate:** Unlike sparse or linear attention, output is mathematically identical
4. **IO-awareness:** Introduced the concept of designing algorithms around GPU memory hierarchy
5. **Universal adoption:** Used by every major model and framework

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

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Tiled attention computation in SRAM, avoid HBM writes |
| Output | Mathematically exact (not approximate) |
| Key trick | Online softmax — composable running max + sum statistics |
| Standard memory | \(O(N^2)\) for attention matrix |
| Flash memory | \(O(N)\) — never materializes full matrix |
| Speedup | 2-4× on training, enables 4-16× longer sequences |
| Adoption | Default in PyTorch 2, vLLM, every major framework |
| GPU insight | Attention is memory-bound, not compute-bound |
