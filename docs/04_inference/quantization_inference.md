# Quantization for Inference

## Why This Matters for LLMs

Inference quantization is the primary lever that lets you run multi-billion-parameter language models on a single consumer GPU, fit larger batches into the same HBM for higher throughput, or even execute compact models on CPUs and phones. Without post-training quantization (PTQ), weights in FP16 or BF16 dominate memory: a 7B parameter model needs on the order of fourteen gigabytes for weights alone, before KV cache, activations, and framework overhead. INT8 and INT4 schemes shrink that footprint by four to eight times, which directly translates into dollars saved on cloud accelerators and into feasibility on edge devices.

The tooling landscape is fragmented but converging on a few families: **GPTQ** and **AWQ** for GPU-friendly weight-only PTQ with calibration data; **GGUF** with **llama.cpp** for portable CPU and Apple Silicon deployments; **SmoothQuant** and related methods when activations must also be quantized for W8A8; and **FP8** on Hopper-and-newer hardware where tensor cores natively support narrow formats. Interviewers expect you to reason about **quality** (perplexity, downstream accuracy), **latency** (fused kernels vs dequantization overhead), and **compatibility** (which runtime consumes which layout).

Third, quantization interacts with **outliers** in weights and activations: a tiny fraction of channels can dominate L2 error if you use naive min–max scaling. Methods that use Hessian-aware updates (GPTQ) or activation-aware salience (AWQ) exist precisely because uniform rounding is blind to layer dynamics. The mathematics below ties affine maps, error bounds, and compression ratios to what you will actually see in deployment reports and ablation tables.

---

## Core Concepts

### Post-Training Quantization Landscape

| Method | Bits | Calibration | Speed | Quality | Format |
|--------|------|-------------|-------|---------|--------|
| GPTQ | 4 / 3 / 2 | Yes (e.g. 128 samples) | Fast on GPU with fused kernels | Good at 4-bit | Safetensors |
| AWQ | 4 | Yes (calibration activations) | Fast on GPU | Often best among 4-bit PTQ | Safetensors |
| GGUF | 2–8 | Often RTN / block scales | Fast on CPU; GPU backends vary | Varies by quant | GGUF binary |
| SmoothQuant | 8 | Yes | Fast when fused | Good W8A8 | Framework-specific |
| FP8 | 8 | Optional | Fastest on H100+ | Near FP16 with scaling | Native |

!!! math-intuition "In Plain English"
    - **Calibration** means you show the network real inputs to set scales—better than blind min–max on weights alone.
    - **Format** matters: a compressed tensor nobody can matmul is useless for latency.

### SmoothQuant (Pointer for W8A8)

SmoothQuant migrates quantization difficulty from **activations** to **weights** using per-tensor scaling so both can sit in INT8 without clipping salient activation outliers. A simplified relation (conceptual) ties activation scale \(s_X\) and weight scale \(s_W\):

\[
Y = X W \approx \left( \frac{X}{s_X} \right) \left( s_X s_W \cdot W_{\mathrm{int}} \right),
\]

where integer matrices approximate the scaled terms. Implementations fuse the scales into kernels.

!!! math-intuition "In Plain English"
    - **Outlier activations** are hard to quantize; **rescaling** \(X\) and **absorbing** factors into **weights** keeps matmuls in low-bit **without** exploding dynamic range.
    - Interview answer: “SmoothQuant makes activations easier to quantize by mathematically shifting the problem.”

### Affine Quantization Maps

The standard affine mapping from floating-point vector \(x\) to integers \(q\) uses scale \(s>0\) and zero-point \(z\):

\[
q = \mathrm{clip}\left( \mathrm{round}\left(\frac{x}{s}\right) + z,\ q_{\min},\ q_{\max} \right).
\]

Reconstruction (dequantization) is:

\[
\hat{x} = s \cdot (q - z).
\]

!!! math-intuition "In Plain English"
    - **Scale** sets the size of one quant step in float space; **zero-point** aligns integer zero with a real value when distributions are asymmetric (common for activations).
    - For symmetric INT8 weight quant you often set \(z=0\) and use signed range \([-127,127]\) or similar.

Symmetric per-channel quantization for row \(i\) of weight matrix \(W\) can be written as:

\[
\hat{W}_{i,:} = s_i \cdot \mathrm{clip}(\mathrm{round}(W_{i,:}/s_i),\ q_{\min},\ q_{\max}).
\]

!!! math-intuition "In Plain English"
    - **Per-channel** scales track large magnitude differences across output channels—critical for linear layers where row norms vary widely.

### GPTQ (GPT Quantization)

GPTQ builds on **Optimal Brain Quantization** ideas: quantize weights column-wise (or block-wise) while using second-order information to update unquantized weights to compensate for error. Conceptually, for weight matrix \(W\) and calibration inputs \(X\) (activations feeding the layer), GPTQ seeks quantized \(\hat{W}\) in a discrete set \(\mathcal{Q}\):

\[
\min_{\hat{W} \in \mathcal{Q}} \| X W - X \hat{W} \|_2^2.
\]

!!! math-intuition "In Plain English"
    - The objective cares about **layer outputs** \(XW\), not raw weight MSE—matching the functional behavior transformers need.
    - The **Hessian** (or its approximation) tells you which directions in weight space hurt the loss most when perturbed.

A local quadratic picture: for small perturbation \(\Delta w\) at a single weight, change in output energy is related to \(X^\top X\) structure—practitioners approximate:

\[
H \approx 2 X^\top X
\]

for linear layers (sketch used in implementations).

!!! math-intuition "In Plain English"
    - **\(X^\top X\)** captures which weight directions align with high-variance input directions—errors there hurt more.
    - GPTQ’s greedy procedures exploit this to **order** quant updates and **apply** **error feedback** to remaining weights.

### AWQ (Activation-Aware Weight Quantization)

AWQ starts from the observation that a small subset of weights is **salient** because they interact with **large-magnitude activations**. Protecting those weights (via per-channel scaling before rounding) preserves accuracy better than treating all weights uniformly at the same bit width.

Let \(s\) denote learned scales; a simplified view:

\[
W' = W \odot s, \qquad \hat{W} = \mathrm{Quant}(W'),
\]

with calibration to choose \(s\) so that \(\| X W - X \hat{W} \|\) is minimized.

!!! math-intuition "In Plain English"
    - **Large activations** amplify any quantization noise on the weights they multiply—AWQ moves precision budget toward those interactions.
    - At **4-bit**, AWQ often beats generic RTN GPTQ on perplexity for the same memory.

### GGUF and llama.cpp

GGUF is a **file container** for single-file models with metadata and multiple tensor quantizations. **llama.cpp** implements efficient kernels for block-wise formats (including **K-quants**) on CPU and GPU backends. Block size \(B\) with per-block scale \(s_k\) and integer codes \(q_i\) gives:

\[
\hat{w}_i = s_k \cdot q_i, \quad i \in \text{block } k.
\]

!!! math-intuition "In Plain English"
    - **GGUF** is not one algorithm—it is **packing** + **layout** so many runtimes can consume the same file.
    - **K-quants** mix bit allocations within super-blocks to preserve outliers better than uniform 4-bit.

### Perplexity as a Quality Proxy

For token sequence \(w_1,\ldots,w_N\), perplexity is:

\[
\mathrm{PPL} = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(w_i \mid w_{<i}) \right).
\]

!!! math-intuition "In Plain English"
    - **Lower PPL** means the model is less surprised by held-out text—useful for **fast** regression after quant.
    - PPL does not replace task benchmarks (math, coding, safety).

!!! example "Worked Example: Quality vs Bit-Width"
    The table below is **illustrative** of typical trends (exact numbers vary by model and calibration).

    | Setting | WikiText-2 PPL (example) | Relative vs FP16 |
    |---------|---------------------------|------------------|
    | FP16 | 5.47 | baseline |
    | INT8 | 5.48 | +0.2% |
    | GPTQ 4-bit | 5.63 | +2.9% |
    | GPTQ 3-bit | 6.12 | +11.9% |
    | GPTQ 2-bit | 8.34 | +52.5% |

    **Step-by-step reading:** Moving FP16 \(\to\) INT8 often adds negligible PPL if scales are calibrated—dynamic range is sufficient. At **4-bit**, a few percent PPL increase is common and often acceptable for chat. **3-bit** begins to show double-digit **relative** degradation unless methods like AWQ or additional tricks are used. **2-bit** is usually research-only for generative models at scale.

    **Conclusion:** INT8 is nearly lossless for many models; **4-bit** is the production sweet spot for size; **3-bit** is situational; **2-bit** is rarely production for general assistants.

### Quantization Error (Operator View)

For \(\mathbf{y} = W \mathbf{x}\) and quantized \(\hat{W} = W + \Delta W\):

\[
\|\hat{\mathbf{y}} - \mathbf{y}\|_2 = \|(\Delta W)\mathbf{x}\|_2 \le \|\Delta W\|_2 \|\mathbf{x}\|_2.
\]

!!! math-intuition "In Plain English"
    - **Outlier activations** \(\mathbf{x}\) blow up output error even when \(\Delta W\) is small in norm—motivates activation-aware clipping and smoothing methods.

### Bits per Parameter

Average **bytes per parameter** for weights only:

\[
\text{BPP}_{\text{weight}} = \frac{\text{total weight bytes}}{\text{number of parameters}}.
\]

!!! math-intuition "In Plain English"
    - FP16 \(\approx 2\) bytes/param; INT4 \(\approx 0.5\); **KV cache** and **activations** are separate budget lines at inference.

??? deep-dive "Deep Dive: Marlin, ExLlama, and Tensor Core Layouts"
    INT4 weights must be packed into layouts that **Tensor Cores** can consume—**Marlin** and similar kernels fuse dequant with GEMM. If your PTQ export format does not match the kernel, you may get **smaller** disk but **no** speedup. Always validate **wall-clock** tokens/sec, not just **model size**.

Storing \(K,V\) in INT8 or FP8 reduces memory bandwidth during long-context decode:

\[
\hat{K} = s_K \cdot \mathrm{round}(K / s_K).
\]

!!! math-intuition "In Plain English"
    - **Attention** quality depends on \(Q K^\top\); **error** in \(K\) propagates through softmax—validate long-context **needle** tests after aggressive KV quant.

??? deep-dive "Deep Dive: KV Cache Quantization (Separate from Weight PTQ)"
    KV quant is **orthogonal** to weight PTQ: you can run **W4A16** weights with FP16 KV, or compress both. Production stacks expose flags for **KV dtype** once kernels exist; always regression-test **long** **needle** **retrieval** when shrinking KV.

---

## Code

The following script runs **without downloading checkpoints**: it demonstrates **fake quantization** in PyTorch, **perplexity-style loss** on random logits (sanity only), and optional imports for **Hugging Face** / **llama.cpp** if installed.

```python
"""
Quantization for inference — educational runnable examples.

- fake_quant_linear(): symmetric per-channel fake quant (always runs).
- loss_on_random_lm(): toy scalar loss (always runs).
- gptq_style_stub(): shows API shape without real GPTQ (always runs).
- try_transformers_gptq(): optional — needs transformers + bitsandbytes/GPU.
- try_llama_cpp(): optional — needs llama-cpp-python + a GGUF path.

Run: python thisfile.py
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_quant_symmetric_per_channel(w: torch.Tensor, n_bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """w: (out, in); returns fake-quantized w_hat and scales (out, 1)."""
    assert w.dim() == 2
    qmax = float(2 ** (n_bits - 1) - 1)
    qmin = float(-(2 ** (n_bits - 1)))
    scales = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    w_q = (w / scales).round().clamp(qmin, qmax)
    w_hat = w_q * scales
    return w_hat, scales


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int) -> None:
        super().__init__()
        w_hat, _ = fake_quant_symmetric_per_channel(linear.weight.data, bits)
        self.register_buffer("weight_hat", w_hat)
        if linear.bias is not None:
            self.register_buffer("lin_bias", linear.bias.data.clone())
        else:
            self.lin_bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_hat, self.lin_bias)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / (b.detach().norm() + 1e-12))


def loss_on_random_lm(vocab: int = 32000, batch: int = 4, seq: int = 32, dim: int = 256) -> float:
    """Cross-entropy on random logits vs random targets — numerical smoke test only."""
    torch.manual_seed(0)
    logits = torch.randn(batch, seq, vocab)
    targets = torch.randint(0, vocab, (batch, seq))
    return float(F.cross_entropy(logits.view(-1, vocab), targets.view(-1)))


def gptq_style_stub() -> None:
    """
    Sketch of GPTQ objective: minimize ||XW - X W_hat|| on calibration batch.
    Uses fake quant instead of discrete search.
    """
    torch.manual_seed(1)
    out_f, in_f = 256, 256
    x = torch.randn(64, in_f)
    w = torch.randn(out_f, in_f)
    y_ref = x @ w.t()
    w_hat, _ = fake_quant_symmetric_per_channel(w, n_bits=4)
    y_q = x @ w_hat.t()
    print("GPTQ-style stub relative error:", relative_l2(y_q, y_ref))


def try_transformers_tiny_forward() -> None:
    """Float forward on HF tiny test model (no extra deps). Production GPTQ: load pre-quantized safetensors + compatible kernels."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers not installed; skip HF example.")
        return
    model_id = os.environ.get("HF_QUANT_DEMO_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
    except Exception as exc:
        print("HF optional demo skipped:", exc)
        return
    inputs = tok("Hello", return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    print("HF tiny model logits shape:", out.logits.shape)


def try_llama_cpp(gguf_path: Optional[str] = None) -> None:
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed; skip GGUF example.")
        return
    path = gguf_path or os.environ.get("GGUF_PATH")
    if not path or not os.path.isfile(path):
        print("Set GGUF_PATH to a local .gguf file to run llama_cpp demo.")
        return
    llm = Llama(model_path=path, n_ctx=512, verbose=False)
    out = llm("Quantization saves memory.", max_tokens=16, temperature=0.0)
    print("llama.cpp output:", out["choices"][0]["text"])


if __name__ == "__main__":
    torch.manual_seed(0)
    lin = nn.Linear(128, 64)
    x = torch.randn(8, 128)
    y_fp = lin(x)
    fq = FakeQuantLinear(lin, bits=4)
    y_q = fq(x)
    print("FakeQuant 4-bit relative output error:", relative_l2(y_q, y_fp))
    print("Toy CE loss:", loss_on_random_lm())
    gptq_style_stub()
    try_transformers_tiny_forward()
    try_llama_cpp()
```

!!! math-intuition "In Plain English"
    - **Fake quant** measures how much a layer’s **outputs** move under rounding—faster than end-to-end kernel bring-up.
    - **BitsAndBytes** `load_in_4bit` is **not** identical to GPTQ export, but interviewers group them under “4-bit inference paths on GPU.”

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Write the affine quant map and explain the role of scale vs zero-point for activations vs weights.
    2. Why does GPTQ optimize \(\|XW - X\hat{W}\|\) rather than \(\|W-\hat{W}\|\)?
    3. What activation behavior makes AWQ’s salience heuristic effective?
    4. Compare **GPTQ** vs **GGUF** deployment: when is CPU-first inference preferable?
    5. Why might INT4 weights not speed up inference if kernels are unfused?
    6. How does **perplexity** track quantization quality, and where does it fail?
    7. Explain **outlier** channels in activations and how SmoothQuant mitigates them at a high level.
    8. What is **W4A16** vs **W8A8**, and which is more common for open LLMs on consumer GPUs?
    9. How would you validate a quantized model for **safety** regressions, not just PPL?
    10. Why does **KV cache quantization** help long-context decode independently of weight PTQ?

!!! interview "Follow-up Probes"
    - “We quantized to INT4 and latency improved 0%—what diagnostics would you run?” (kernel packing, batch size, memory bound vs compute bound, dequant overhead.)
    - “PPL barely moved but GSM8K collapsed—what happened?” (reasoning-sensitive metrics, calibration mismatch, outliers.)
    - “When would you choose QAT over PTQ?” (budget, accuracy SLO, domain shift.)

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Affine mapping with per-channel scales preserves dynamic range across output channels.”
    - “GPTQ uses Hessian-aware greedy fitting to preserve layer outputs on calibration data.”
    - “AWQ protects salient weights that align with large activations.”
    - “GGUF is a container format; llama.cpp provides block-quant kernels for CPU/GPU.”
    - “Validate latency with fused Tensor Core kernels—compression alone is not speedup.”

---

## References

- Frantar et al. (2023), *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al. (2023), *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- Dettmers et al. (2022), *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- Xiao et al. (2023), *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* — [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- ggml / GGUF documentation — [https://github.com/ggerganov/ggml/blob/master/docs/gguf.md](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- ggerganov, *llama.cpp* — [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)
- NVIDIA, *TensorRT-LLM Quantization* — [https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- Micikevicius et al. (2017), *Mixed Precision Training* — context for FP16/BF16 baselines — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Hubara et al. (2016), *Binarized Neural Networks* — historical fixed-point NN context
