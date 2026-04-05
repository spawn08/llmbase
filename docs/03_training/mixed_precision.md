# Mixed Precision & Quantization

## Why This Matters for LLMs

Training and serving LLMs are **memory-bandwidth** and **capacity** limited. **Mixed precision** (FP16/BF16/FP8) lets you fit wider batches during training and increase **tokens/sec** during inference. Interviewers expect you to explain **dynamic loss scaling** for FP16**, why **BF16** often needs no scaling**, and how **quantization** maps high-precision floats to low-bit integers for **weights** and **activations**. Without this, answers about “INT4 weights” vs “INT8 dynamic” stay vague.

A second reason is **numerical analysis**: affine quantization uses **scale** and **zero-point** so that integer matrix multiplies approximate real-valued ops. Understanding **symmetric vs asymmetric** schemes separates candidates who memorized API names from those who can debug **accuracy regressions** after quantization.

Third, **PTQ vs QAT**: post-training quantization is cheap but brittle for outliers; quantization-aware training inserts **fake quant** nodes so the model adapts. Production stacks (GPTQ, AWQ, GGUF loaders) make different assumptions about **calibration data** and **weight grouping**—systems interviews often probe these trade-offs.

---

## Core Concepts

### Floating-Point Formats: FP32, FP16, BF16, FP8

**IEEE-754 FP32**: 1 sign, 8 exponent, 23 mantissa bits.

**FP16**: 1 sign, 5 exponent, 10 mantissa — **narrow** dynamic range; gradients can **underflow** to zero without scaling.

**BF16** (bfloat16): 1 sign, **8 exponent**, 7 mantissa — **same exponent width as FP32**, smaller mantissa. Often **loss scaling–free** in training because range matches FP32 better than FP16.

**FP8** (NVIDIA H100 / Transformer Engine): multiple layouts (E4M3, E5M2); used with **automatic scaling** policies in specialized kernels.

!!! math-intuition "In Plain English"
    Think **mantissa** = precision of steps between numbers; **exponent** = how large/small you can go before overflow/underflow. FP16 has **tiny** mantissa **and** small exponent range → **fragile** for deep networks unless you scale carefully. BF16 sacrifices mantissa but keeps **fat** exponent range like FP32 → friendlier for training.

### Bit Layout Summary (Conceptual)

| Format | Sign | Exponent bits | Mantissa bits | Notes |
|--------|------|----------------|----------------|-------|
| FP32 | 1 | 8 | 23 | Reference |
| FP16 | 1 | 5 | 10 | Needs loss scaling often |
| BF16 | 1 | 8 | 7 | Range like FP32 |
| FP8 E4M3 | 1 | 4 | 3 | Example FP8 variant |

### Loss Scaling for FP16 Training

Multiply the loss by a scale factor \(s\) before backward so small gradients gain magnitude above FP16 **subnormal** / **underflow** region; **unscale** weights before optimizer step.

Algorithm sketch:

1. Forward in FP16 (or mixed), compute loss \(L\).
2. \(\tilde{L} = s \cdot L\); backward yields scaled gradients \(\tilde{g} = s g\).
3. If **any** \(\tilde{g}\) is **inf/nan**, skip step, **reduce** \(s\).
4. Else **unscale**: \(g \leftarrow \tilde{g} / s\); optimizer step in **FP32** master weights.

**Dynamic loss scaling** adjusts \(s\) based on overflow frequency.

!!! math-intuition "In Plain English"
    You are **temporarily measuring gradients in a louder unit** so FP16 can hear them—then converting back to true units before updating **FP32** master weights. BF16 often skips this dance because its dynamic range is wide enough.

### Master Weights (FP32 Copy)

Store **primary weights** in FP32; cast to reduced precision for matmuls. Update:

\[
w_{\text{fp32}} \leftarrow w_{\text{fp32}} - \eta \cdot \widehat{g}
\]

where \(\widehat{g}\) is the **unscaled** gradient in FP32 after any loss scaling pipeline.

---

## Post-Training Quantization (PTQ)

Map FP weights \(w\) to integers \(q\) with **affine** mapping (per-tensor or per-channel):

### Symmetric Quantization

\[
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{s}\right), -2^{b-1}, 2^{b-1}-1\right)
\]

**Dequantization**:

\[
\hat{w} = s \cdot q
\]

No **zero-point**; zero maps exactly when \(w=0\) after scaling.

### Asymmetric Quantization (Affine)

A common **unsigned** INT8 mapping uses:

\[
q = \mathrm{clip}\left(\mathrm{round}\left(\frac{w}{s}\right) + z, 0, 255\right)
\]

**Dequantization**:

\[
\hat{w} = s (q - z)
\]

Here \(z\) is the **zero-point** (integer) such that \(\hat{w} \approx 0\) when \(q = z\). For **signed** INT8, clip range shifts (e.g., \([-128,127]\)).

Given real range \([w_{\min}, w_{\max}]\) and integer range \([q_{\min}, q_{\max}]\):

\[
s = \frac{w_{\max} - w_{\min}}{q_{\max} - q_{\min}},\quad
z = \mathrm{round}\left(q_{\min} - \frac{w_{\min}}{s}\right)
\]

!!! math-intuition "In Plain English"
    **Asymmetric** maps uses the **full** integer bucket range even when \(w_{\min} \neq -w_{\max}\)—common for **biased** activations after ReLU-like ops. **Symmetric** wastes half the codes if the distribution is one-sided, but hardware may be simpler. **Scale** sets the **step size** between integer ticks; **zero-point** slides the integer grid so **real zero** maps to a representable code where needed.

### INT8 and INT4

**INT8** uses 256 levels; **INT4** uses 16—far fewer, needs **grouping** (blocks of columns/rows) each with its own scale (GPTQ, AWQ). **Outlier** dimensions dominate error; methods may **protect** a few dimensions in FP16.

### GPTQ (Post-Training, Layer-wise)

**GPTQ** (Frantar et al.) quantizes weights **one layer at a time** using **Hessian**-informed errors to choose quantization order and updates—approximating optimal rounding under squared error.

### AWQ (Activation-Aware)

**AWQ** protects **salient** weights identified by activation magnitudes—scales channels to reduce quantization error where activations are large.

!!! math-intuition "In Plain English"
    GPTQ is **math-heavy** rounding with second-order awareness. AWQ is **cheaper heuristics** guided by **which weights actually move activations**—often strong accuracy at INT4 for LMs.

### SmoothQuant (Activations + Weights)

**SmoothQuant** migrates quantization difficulty from **activation outliers** to **weights** via a per-channel **mathematical smoothing** that preserves linear layer outputs—enabling **INT8** W/A for transformers with better accuracy than naive PTQ when outliers dominate.

### GGUF / Runtime Loaders (Inference)

**GGUF** files store **quantized tensors** with type tags (`Q4_K_M`, `Q5_K_S`, …) and metadata for **llama.cpp**-compatible runtimes. Interview angle: **disk format** vs **algorithm**—GPTQ/AWQ are **how** weights become INT4; GGUF is **how** they are **packaged** for inference engines.

### bitsandbytes 4-bit (Training / PEFT)

**NF4** / **FP4** quant types in **bitsandbytes** enable **4-bit** linear layers with **double quant** of scales—core to **QLoRA** (train adapters while base is quantized). Distinct from GPTQ’s **post-hoc** weight rounding.

---

## QAT vs PTQ

| Aspect | PTQ | QAT |
|--------|-----|-----|
| Cost | Low — no retraining loop | Higher — train with fake quant |
| Accuracy | Good for INT8 often; INT4 harder | Better for INT4 / tough models |
| Workflow | Calibrate on sample batch | Insert quant/dequant modules |

**Fake quant** forward:

\[
y = s \cdot \mathrm{round\_clip}\left(\frac{x}{s}\right)
\]

Straight-through estimator on backward for \(\mathrm{round}\).

---

## Quantization Math: Per-Channel Scales

For weight matrix \(W \in \mathbb{R}^{m \times n}\), **per-output-channel** scales \(s_i\) improve conv/linear quality:

\[
\hat{W}_{ij} = s_i \cdot Q_{ij}
\]

Error \(\|W - \hat{W}\|\) decreases vs single global \(s\).

---

## Python: Affine Quantization Round-Trip

```python
"""
Symmetric INT8 quantization round-trip for a 1-D tensor — educational.
"""
from __future__ import annotations

import torch


def quantize_symmetric_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns int8 q and scalar scale s (fp32)."""
    max_val = w.abs().max().clamp(min=1e-8)
    s = max_val / 127.0
    q = torch.round(w / s).clamp(-128, 127).to(torch.int8)
    return q, s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.float() * s


if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.randn(1000)
    q, s = quantize_symmetric_int8(w)
    w_hat = dequantize(q, s)
    print("relative error", (w - w_hat).norm() / w.norm())
```

---

## Python: Simulate FP16 Underflow Without Scaling

```python
"""
Show tiny gradients vanish in FP16 without loss scaling (toy).
"""
from __future__ import annotations

import torch


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(50, requires_grad=True)
    # Tiny loss ~ 1e-8 scale
    loss = (x * 1e-9).sum()
    loss.backward()
    g32 = x.grad.clone()
    x16 = torch.randn(50, requires_grad=True, dtype=torch.float16)
    loss16 = (x16 * 1e-9).sum()
    loss16.backward()
    g16 = x16.grad.clone()
    print("fp32 grad max abs:", g32.abs().max().item())
    print("fp16 grad max abs:", g16.abs().max().item())


if __name__ == "__main__":
    main()
```

---

## Mixed Precision Training Stack (Conceptual)

1. **Autocast** regions mark ops that tolerate FP16/BF16.
2. **GradScaler** (FP16) manages loss scaling.
3. **FP32** master params for optimizer.

```python
"""
Minimal pattern: autocast + GradScaler (FP16) — structure only.
"""
from __future__ import annotations

import torch
from torch.cuda.amp import GradScaler, autocast


def train_step(model: torch.nn.Module, batch: torch.Tensor, opt: torch.optim.Optimizer) -> float:
    scaler = GradScaler()
    opt.zero_grad(set_to_none=True)
    with autocast(dtype=torch.float16):
        logits = model(batch)
        loss = logits.pow(2).mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return float(loss.detach())


if __name__ == "__main__":
    if torch.cuda.is_available():
        m = torch.nn.Linear(32, 32).cuda()
        x = torch.randn(8, 32, device="cuda")
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        print(train_step(m, x, opt))
```

---

## KV Cache and Inference Quantization

Autoregressive decoding stores **KV activations** per layer; **INT8 KV** reduces memory bandwidth. **Weight-only INT4** (e.g., loaded from disk) keeps activations FP16—different kernels, different bottlenecks.

---

## Interview Takeaways

- **Formats**: FP16 narrow range → **loss scaling**; BF16 **wide exponent** → often **no scaling**; FP8 needs **hardware + TE** awareness.
- **Affine quant**: \(w \approx s(q-z)\); know **symmetric** as special case \(z=0\) after centering assumptions.
- **PTQ vs QAT**: cost vs accuracy; **INT4** usually wants **group scales** or **QAT**.
- **GPTQ vs AWQ**: Hessian-guided **rounding** vs **activation-aware** saliency—different inductive biases.
- **Master weights**: optimizer in **FP32**; forward matmuls in **low precision**.
- **Serving**: separate **weight-only** vs **activation** quantization; **KV cache** quantization saves **memory** on long contexts.

---

## References

- Micikevicius et al., *Mixed Precision Training* — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)
- Kahan (IEEE FP formats); **bfloat16** — Google Brain usage in TPUs
- Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* — CVPR 2018 (QAT, symmetric/asymmetric)
- Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* — [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- NVIDIA — **Transformer Engine** / FP8 training documentation
- Dettmers et al., *LLM.int8()* — mixed-precision for outliers — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
