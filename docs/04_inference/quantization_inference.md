# Quantization for Inference

Low-bit inference is the **default** path to run **open** models on **single** GPUs and **edge** CPUs—this page focuses on **post-training** weight quantization, **formats**, and **error** **analysis**.

## Why This Matters for LLMs

Frontier models have **billions** of parameters; **FP16** weights alone need **2 bytes** each—**7B** \(\approx 14\) **GiB** before **activations**, **optimizer** states (training), and **KV** caches. **Post-training quantization (PTQ)** compresses **already-trained** weights to **INT8** or **INT4** with **minimal** **accuracy** loss so consumer GPUs and edge devices can run **open-weight** models. **GPTQ** and **AWQ** are the **poster children** for **one-shot** weight-only PTQ; **GGUF** + **llama.cpp** ship **CPU** and **Apple Silicon** deployments with **block**-quantized tensors.

Interviewers want **quantitative** literacy: **affine** maps \(q = \mathrm{round}(x/s) + z\), **per-channel** vs **per-tensor** scales, **activation-aware** clipping, and why **perplexity** is the **first** **sanity** metric but not the **only** one.

A second axis is **kernels**: INT4/INT8 **GEMM** on NVIDIA **Tensor Cores** requires **specific** layouts (**Marlin**, **CUTLASS**)—compression without **fast** **dequant** **fused** into matmul **does not** **accelerate** **wall-clock**.

---

## Core Concepts

### Affine Quantization

For floating vector \(x\), **integer** code \(q\) with **scale** \(s>0\) and **zero-point** \(z\):

\[
q = \mathrm{clip}\Bigl(\Bigl\lfloor \frac{x}{s} \Bigr\rceil + z,\, q_{\min},\, q_{\max}\Bigr).
\]

**Dequantize**:

\[
\hat{x} = s \cdot (q - z).
\]

For **symmetric** **INT8** often \(z=0\), \(q_{\min}=-128\), \(q_{\max}=127\).

!!! math-intuition "In Plain English"
    - **Scale** maps **one** **integer** **tick** to **real** **magnitude**.
    - **Zero-point** aligns **integer** **zero** with **real** **zero** when **asymmetric**—helps **biased** activations.

### Quantization Error (Worst-Case Bound)

For **round-to-nearest** with step \(s\), **per-element** error \(|\hat{x}_i - x_i| \le s/2\) (ignoring **clip** saturation). **Saturation** when \(|x_i/s|\) exceeds **representable** range causes **large** errors—**outliers** in activations or weights dominate **quality**.

\[
\|\hat{x} - x\|_\infty \le \frac{s}{2} \quad \text{(no saturation).}
\]

!!! math-intuition "In Plain English"
    - **Smaller** \(s\) → **finer** **grid** but **smaller** dynamic range—**tradeoff** set by **calibration**.

### GPTQ: One-Shot Weight Quantization

**GPTQ** (Frantar et al., 2023) quantizes weights **layer-wise** using **second-order** information (approximate **Hessian**) to **minimize** **layer output** **error** under **quantized** weights. Objective sketch for weight matrix \(W\) and layer input \(X\):

\[
\min_{\hat{W} \in \mathcal{Q}} \| X W - X \hat{W} \|_2^2
\]

over **quantized** **search** space \(\mathcal{Q}\) with **greedy** **column**-wise updates and **Hessian** **approximation** \(H \approx 2 X^\top X\) for linear layers.

!!! math-intuition "In Plain English"
    - **GPTQ** tries to keep **layer outputs** **close** to **FP16**—**not** just **minimize weight MSE**.
    - **One-shot**: **no** **fine-tuning** **dataset** **loop**—uses **calibration** **activations** from a **small** **sample**.

### AWQ: Activation-Aware Weight Quantization

**AWQ** (Lin et al., 2023) observes **salient** weights are those multiplied by **large-magnitude** **activations**. It learns **per-channel** **scaling** to **protect** **important** weights before **rounding**, improving **perplexity** vs naive **min-max** PTQ.

\[
W' = W \odot s,\quad \text{then quantize } W' \text{ with adjusted scales.}
\]

!!! math-intuition "In Plain English"
    - **Where activations are tiny**, **quant noise** matters less—**AWQ** reallocates **precision** **budget** using **activation** statistics.

### GGUF and llama.cpp

**GGUF** is a **file container** for **single-file** models with **metadata** and **multiple** **quantization** **variants** (**Q4_K_M**, **Q5_K_S**, …). **llama.cpp** implements **CPU**/**GPU** **kernels** with **K-quant** **block** formats—**block** size typically **32** or **64** elements sharing **scale**/**min**.

!!! math-intuition "In Plain English"
    - **GGUF** is **not** a **quant algorithm**—it is a **packaging** + **tensor layout** standard enabling **interoperable** **inference** **runtimes**.

### Perplexity vs Compression Tradeoff

**Perplexity** on a **held-out** corpus:

\[
\mathrm{PPL} = \exp\Bigl(-\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(w_i \mid w_{<i})\Bigr).
\]

Quantization usually **increases** PPL monotonically as **bitwidth** drops—**compare** at **fixed** tokenizer and **context**.

!!! math-intuition "In Plain English"
    - **PPL** misses **downstream** **task** **accuracy**—always **spot-check** **benchmarks** relevant to **your** **product**.

### INT4 / INT8 Kernels

**INT8** **Tensor Core** **GEMM** requires **operand** **layouts** (**row-major**/**col** **major** constraints) and **accumulation** in **INT32**. **INT4** packs **two** **weights** per **byte**—**decode** **inside** **kernel** or **prepacked** **Marlin** **tiles**.

\[
\text{GEMM: } C = \mathrm{dequant}(W_q) \cdot X \approx W X.
\]

!!! math-intuition "In Plain English"
    - **Weight-only** **quant** often **keeps** **activations** **FP16**—**speedup** depends on **memory** **bandwidth** reduction and **fused** **ops**.

---

## Calibration Snippet (PyTorch, INT8 Fake Quant)

Educational **fake-quant** (simulation) for a **linear** layer—**not** production **CUTLASS** kernels.

```python
"""
Fake-quantization of a Linear layer for perplexity experiments (CPU/GPU).
Uses per-channel symmetric quantization of weights.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quantize_symmetric_per_channel(
    w: torch.Tensor,
    n_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    w: (out_features, in_features) — quantize each output channel independently.
    Returns quantized int tensor (float container) and scales (out_features,).
    """
    assert w.dim() == 2
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -(2 ** (n_bits - 1))
    # per-output-channel max
    scales = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / float(qmax)
    w_q = (w / scales).round().clamp(qmin, qmax)
    w_hat = w_q * scales
    return w_hat, scales.squeeze()


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int = 8) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        w_hat, _ = fake_quantize_symmetric_per_channel(linear.weight.data, n_bits=bits)
        self.register_buffer("weight_hat", w_hat)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_hat, self.bias)


def kl_divergence_discrete(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p || q) for 1D distributions."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum().item())


if __name__ == "__main__":
    torch.manual_seed(0)
    lin = nn.Linear(128, 64)
    x = torch.randn(4, 128)
    y_fp = lin(x)
    fq = FakeQuantLinear(lin, bits=4)
    y_q = fq(x)
    print("relative error:", (y_fp - y_q).abs().mean().item())
```

!!! math-intuition "In Plain English"
    - **Fake-quant** **round-trips** weights through **integer** **grid**—measures **signal** loss without **custom** **CUDA**.
    - **Real** **GPTQ/AWQ** optimize **non-uniform** **bit allocations** and **Hessian**-aware **ordering**.

### Error Propagation Across Layers

Let **layer** maps \(f_\ell\) and **quant** maps \(Q_\ell\). **End-to-end** error composes **nonlinearly**:

\[
y_L = f_L \circ \cdots \circ f_1(x),\quad
\hat{y}_L = Q_L \circ f_L \circ \cdots \circ Q_1 \circ f_1(x).
\]

**Amplification** happens when later layers are **ill-conditioned** w.r.t. **earlier** **noise**—why **layer-wise** objectives (GPTQ) help.

---

## Quantization Error Bounds (Simplified)

For **linear** \(\mathbf{y} = W \mathbf{x}\), **quantized** \(\hat{W} = W + \Delta W\) with \(\|\Delta W\|_2 \le \epsilon_W\):

\[
\|\hat{\mathbf{y}} - \mathbf{y}\|_2 = \|\Delta W \mathbf{x}\|_2 \le \|\Delta W\|_2 \|\mathbf{x}\|_2.
\]

**Operator norm** bounds link **weight** **perturbation** to **output** **change**.

!!! math-intuition "In Plain English"
    - **Outlier** **activations** \(\mathbf{x}\) **blow up** **output** **error** even when \(\Delta W\) is **small**—motivates **SmoothQuant**-style **activation** **scaling** (related family).

### Mixed Precision Strategies

- **FP16** weights + **INT8** **activations** (W8A8) needs **calibration** for **activation** scales.
- **INT4** weights + **FP16** activations (W4A16) common in **consumer** **inference**.

### Normal Float (NF4) and QLoRA (Pointer)

**QLoRA** trains **adapters** in **NF4** **normal** **float** **quant** for **weights** while keeping **optimizer** states **small**—**inference** **quant** borrows **similar** **non-uniform** **codebooks** for **4-bit** **weights** (**bitsandbytes**). **NF4** assigns **more** **levels** near **zero** where **probability** **mass** concentrates for **Gaussian-like** weights.

!!! math-intuition "In Plain English"
    - **Uniform** **INT4** wastes **levels** in **tails**; **NF4** is **variance**-aware—**better** **PPL** at **same** **bit budget**.

### Signal-to-Noise Ratio Heuristic

Treat **quantization** as **additive** noise \(\Delta\) on weights. **SNR** (power):

\[
\mathrm{SNR} = \frac{\mathbb{E}[\|W\|_F^2]}{\mathbb{E}[\|\Delta W\|_F^2]}.
\]

Higher **SNR** correlates with **lower** **PPL** **degradation**—useful **sanity** **check** when **comparing** **two** **PTQ** **recipes**.

### Kernel Backends: Marlin, ExLlama, TensorRT-LLM

| Backend | Role |
|---------|------|
| **Marlin** | INT4 **weight** **GEMM** **kernels** for **NVIDIA** |
| **ExLlama** | CUDA **kernels** for **GPTQ**-style **4-bit** |
| **TensorRT-LLM** | End-to-end **fused** **graph** with **KV** **cache** **quant** options |
| **llama.cpp** | **CPU**/**Metal**/**CUDA** **block** **dot** products |

!!! math-intuition "In Plain English"
    - **Pick** **backend** **first**, then **quantize** into **supported** **layouts**—**incompatible** **packing** **wastes** **weeks**.

### Per-Channel vs Per-Tensor Scales

**Per-tensor** scale uses **one** \(s\) for **entire** **matrix**—**cheap** but **poor** if **row** **magnitudes** **vary** **wildly**.

**Per-channel** (per **output** row or column) uses **vector** \(\mathbf{s}\):

\[
\hat{W}_{i,:} = s_i \cdot Q_{i,:}.
\]

**GPTQ** often **operates** **per-column** **within** **Hessian** **blocks**.

### Dynamic Quantization of Activations (Runtime)

At inference, **activation** scales can be **computed** **per** **tensor** **per** **forward** from **running** **max**—**no** **calibration** **set** needed but **higher** **kernel** **overhead**:

\[
s_t = \frac{\max(|X_t|)}{127},\quad X_{q,t} = \mathrm{round}(X_t / s_t).
\]

!!! math-intuition "In Plain English"
    - **Dynamic** **acts** **adapt** to **input** **outliers**—helps **W8A8** **accuracy** at **cost** of **latency** **variance**.

### Compression Ratio Table (Illustrative)

| Format | Bytes/param (order-of-magnitude) |
|--------|----------------------------------|
| FP32 | 4 |
| FP16/BF16 | 2 |
| INT8 | 1 |
| INT4 | 0.5 |

**KV cache** **quant** **saves** **memory** **bandwidth**—orthogonal to **weight** **quant**.

### Safety Note on Quantized Models

**Quantization** can **shift** **model** **behavior** on **safety** **refusals** or **bias** benchmarks—**re-run** **evaluation** **suites** after **aggressive** **INT4** **PTQ**.

### Quantization-Aware Training (Contrast)

**QAT** inserts **fake-quant** **ops** during **fine-tuning** so **weights** **adapt** to **low-bit** **noise**—**higher** **quality** than **PTQ** at same **bits**, **costly** for **already** **massive** **models**. **PTQ** is **default** for **inference** **compression** of **frozen** **weights**.

### Blockwise Quantization (GGUF K-Quants)

For **block** size \(B\), partition **weights** into **blocks** \(\{b_1,\ldots,b_K\}\). Each block has **own** **scale** / **min**:

\[
\hat{w}_{i} = s_k \cdot q_i,\quad i \in b_k.
\]

**K-quants** mix **different** **bit** **widths** per **super-block**—**improves** **PPL** vs **uniform** **INT4** at **same** **average** **bpp**.

### Quantizing KV Cache (Inference)

**Attention** **inputs** remain **FP16**, but **stored** **\(K,V\)** can be **INT8** with **per-tensor** **scales**:

\[
K_{\text{int8}} = \mathrm{round}(K / s_K),\quad \hat{K} = s_K K_{\text{int8}}.
\]

**Error** in **attention** **output** scales with **\(\|Q\|\|\Delta K\|\)** terms—**long** **contexts** **amplify** **misfit**—**validate** **perplexity** **and** **needle** **tests**.

### Evaluation Protocol (Practical)

1. **WikiText** / **C4** **slice** **PPL** at **fixed** **seqlen**.
2. **MMLU** / **GSM8K** **task** **accuracy** for **reasoning** **regression**.
3. **Latency** **p50/p95** on **target** **hardware** with **chosen** **backend**.

### Rounding Modes

**Stochastic** **rounding** can **improve** **accuracy** by **unbiased** **error**—**harder** to **reproduce** **bit**-exact outputs—**production** **usually** **RN** (round-nearest).

### Bits-per-Parameter Budgeting

For **total** model bytes \(M\) and parameters \(N\), **average** **bpp** \(= 8M/N\). Moving **FP16** \(\to\) **INT4** saves **\(\approx 4\times\)** **weight** **storage**—**KV** **cache** **still** **FP16** unless **separately** **compressed**.

### Connection to FlashAttention

**FlashAttention** reduces **HBM** **traffic** for **attention**; **quantized** **weights** reduce **HBM** **traffic** for **GEMMs**—**orthogonal** **levers** on **different** **bottlenecks**.

---

## Interview Takeaways

- **Affine** **quant** maps floats to **low-bit** integers with **scale/zero-point**; **saturation** **dominates** **worst-case** **error**.
- **GPTQ** uses **Hessian**-aware **layer-wise** **fitting**; **AWQ** uses **activation-aware** **salience**—both target **output fidelity**, not **weight MSE**.
- **GGUF** is a **container** + **layout**; **llama.cpp** provides **CPU**/**Metal**/**CUDA** **kernels** for **block** formats.
- **Perplexity** is a **fast** **regression** test; **task** **benchmarks** still **required** for **deployment** **sign-off**.
- **INT4/INT8** **speedups** require **fused** **kernels**—**compressed** weights alone are **not** enough if **dequant** is **slow**.
- **Operator-norm** **bounds** explain why **outliers** **hurt** **quantized** **matmuls**.
- **NF4** / **K-quants** **non-uniformly** **allocate** **levels**—**better** **PPL** than **uniform** **INT4** at **same** **storage**.
- **W8A8** needs **activation** **calibration**; **W4A16** is **common** **GPU** **default** for **open** **weights**.

!!! math-intuition "In Plain English"
    - If you **only** remember **one** **formula**: \(\hat{x} = s(q-z)\) **links** **integer** **codes** to **float** **reconstruction**—every **engine** is a **variant** of this.
    - **PTQ** vs **QAT**: **PTQ** is **cheap** **deployment**; **QAT** is **expensive** **training-time** **adaptation**—pick based on **budget** and **accuracy** **SLO**.

---

## References

- Frantar et al. (2023), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

- Lin et al. (2023), *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- ggerganov et al., *llama.cpp* — [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- Dettmers et al. (2022), *LLM.int8()* — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339) — outlier handling
- Xiao et al. (2023), *SmoothQuant* — [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- NVIDIA *TensorRT-LLM* quantization docs — [https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- Dettmers & Zettlemoyer (2023), *QLoRA* — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) — NF4 training context
- *GGUF specification* — [https://github.com/ggerganov/ggml/blob/master/docs/gguf.md](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- Hubara et al. (2016), *Binarized Neural Networks* — historical fixed-point neural nets (context)
