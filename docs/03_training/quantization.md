# Mixed Precision and Quantization

## Why This Matters for LLMs

Quantization and reduced-precision arithmetic are the primary levers that make large language models deployable outside training clusters. A 70 billion parameter model at FP32 requires hundreds of gigabytes of storage for weights alone; at INT4, the same parameter count occupies eight times less memory in a simple bit-counting story, enabling single-GPU or CPU inference scenarios that would be impossible otherwise. Interviewers who focus on inference optimization, on-device assistants, or cost-aware serving expect fluency in symmetric versus affine quantization, per-channel schemes, and the trade-offs of GPTQ, AWQ, GGUF, and bitsandbytes integrations.

Training also relies on mixed precision: BF16 and FP16 matrix multiplies with FP32 master weights reduce memory bandwidth pressure and increase throughput on Tensor Cores, while loss scaling preserves small gradient components in FP16 regimes. Understanding dynamic range differences between formats prevents naive conclusions like “16-bit is always enough” without examining activation outliers in attention layers. The same awareness explains why INT8 training or inference sometimes needs mixed schemes such as LLM.int8() outlier channels.

Finally, quantization interacts with accuracy, latency, and hardware. INT4 weights may degrade reasoning on fringe tasks unless calibration is careful. FP8 on Hopper GPUs changes the frontier for training throughput. GGUF files bundle metadata and quantized tensors for llama.cpp ecosystems. A strong candidate articulates when post-training quantization suffices, when quantization-aware retraining is justified, and how to measure regressions beyond perplexity on a handful of prompts.

---

## Core Concepts

### Number Formats

| Format | Bits | Exponent | Mantissa | Approximate range | Notes |
|--------|------|----------|----------|-------------------|-------|
| FP32 | 32 | 8 | 23 | \(\pm 3.4 \times 10^{38}\) | Standard reference for master weights |
| FP16 | 16 | 5 | 10 | \(\pm 65504\) | Fast but limited range |
| BF16 | 16 | 8 | 7 | \(\pm 3.4 \times 10^{38}\) | Same exponent range as FP32, less mantissa precision |
| INT8 | 8 | 0 | 7 signed | \(-128\) to \(127\) | 4× compression vs 32-bit weights |
| INT4 | 4 | 0 | 3 signed | \(-8\) to \(7\) | 8× compression vs 32-bit weights in idealized packing |
| FP8 (E4M3) | 8 | 4 | 3 | \(\pm 448\) (typical finite) | Hopper GPUs |
| FP8 (E5M2) | 8 | 5 | 2 | larger magnitude, coarser mantissa | Gradient-oriented variant |

The **machine epsilon** order-of-magnitude for FP16 (relative to 1.0) is about \(2^{-10} \approx 10^{-3}\), while BF16 has coarser mantissa steps near unity because it has fewer mantissa bits than FP16.

!!! math-intuition "In Plain English"
    Exponent bits decide how large or small a number can be before overflow or underflow. Mantissa bits decide how finely you can distinguish nearby values. BF16 sacrifices mantissa precision versus FP32 but keeps exponent width, which is why it often tracks FP32 magnitudes more safely than FP16 in training. Integers have no exponent; their grid is uniform after scaling, which is why affine maps use scale and zero point.

### Quantization Basics

#### Affine quantization

Given real values \(x\) in a tensor, affine mapping to integers uses scale \(s > 0\) and zero point \(z\):

\[
x_q = \operatorname{round}\left( \frac{x}{s} + z \right).
\]

Dequantization reconstructs:

\[
\hat{x} = s \cdot (x_q - z).
\]

!!! math-intuition "In Plain English"
    You choose \(s\) so the integer grid covers the observed min and max of \(x\). The zero point \(z\) shifts the grid so exact zero in floats can map to an integer when the distribution is asymmetric. Rounding introduces error; the quantizer designer tries to keep that error small relative to signal magnitude.

#### Symmetric quantization

Symmetric schemes fix \(z = 0\) and use:

\[
x_q = \operatorname{round}\left( \frac{x}{s} \right), \qquad s = \frac{\max |x|}{q_{\max}},
\]

where \(q_{\max}\) is the maximum positive integer representable (127 for signed INT8).

!!! math-intuition "In Plain English"
    Symmetric quantization is simpler: one scale per tensor or channel, no zero-point arithmetic. It works well when weights are roughly zero-centered. Activations can be skewed positive after ReLU-like nonlinearities; some layers then benefit from affine codes.

#### Granularity

- **Per-tensor:** one \((s, z)\) for the entire tensor; simplest, often fastest.
- **Per-channel:** separate scales for each output channel of a weight matrix; better accuracy for linear layers.
- **Per-group:** partition a tensor into groups of \(g\) consecutive elements sharing one scale; common in GPTQ and AWQ at INT4.

The **quantization noise variance** scales roughly with \(s^2\) for uniform rounding in a local linearization sense, which motivates **smaller scales** via **per-channel** granularity when a few channels have much larger magnitude than others.

!!! math-intuition "In Plain English"
    One scale for the whole tensor forces **large** \(s\) when **any** element is large, which makes **small** elements land in a few coarse quantization levels. **Per-channel** scales let **each** channel use its own **ruler**, reducing distortion for channels with smaller dynamic range.

!!! example "Worked Example: INT8 Quantization"
    Weight vector (floating):

    \[
    x = [0.3,\ -0.5,\ 1.2,\ -0.1,\ 0.8].
    \]

    Maximum absolute value is **1.2**. For signed INT8, \(q_{\max} = 127\). Symmetric scale:

    \[
    s = \frac{1.2}{127} \approx 0.0094488.
    \]

    Quantize each entry with \(x_q = \operatorname{round}(x / s)\):

    - \(0.3 / s \approx 31.75 \rightarrow 32\)
    - \(-0.5 / s \approx -52.92 \rightarrow -53\)
    - \(1.2 / s = 127 \rightarrow 127\)
    - \(-0.1 / s \approx -10.58 \rightarrow -11\)
    - \(0.8 / s \approx 84.66 \rightarrow 85\)

    Integer codes: \([32, -53, 127, -11, 85]\).

    Dequantize \(\hat{x}_i = s \cdot x_{q,i}\):

    - \(32s \approx 0.302\)
    - \(-53s \approx -0.501\)
    - \(127s = 1.200\)
    - \(-11s \approx -0.104\)
    - \(85s \approx 0.803\)

    Absolute errors are \(0.002, 0.001, 0.000, 0.004, 0.003\), illustrating bounded reconstruction error when the scale matches the tensor range.

### Post-Training Quantization (PTQ) Methods

- **LLM.int8():** mixed-precision matmuls where outlier dimensions remain in FP16 while bulk operations use INT8.
- **GPTQ:** one-shot weight quantization using approximate Hessian information to minimize layer output error after quantizing weights.
- **AWQ (Activation-Aware Weight Quantization):** identifies salient weight regions using activation statistics and applies per-channel scaling to protect important weights under low-bit formats.
- **GGUF:** file format used by llama.cpp consumers; stores quantized tensors with metadata for CPU and GPU runtimes.
- **SmoothQuant:** migrates quantization difficulty from activations to weights via mathematically equivalent linear transforms that re-scale paired weight and activation tensors before quantization.

### Quantization-Aware Training (QAT)

During QAT, forward passes use quantized weights and activations while backward passes approximate gradients with a straight-through estimator (STE). A common idealization writes:

\[
q = \operatorname{clip}(x, -t, t),\qquad \hat{q} = \operatorname{round}(q) \approx q \text{ in backward.}
\]

!!! math-intuition "In Plain English"
    The forward pass uses discrete levels; the backward pass pretends the round function is identity inside a clip region so gradients flow. This biased estimator works when training can adapt parameters to live near good quantization levels. PTQ cannot recover from bad outliers as flexibly; QAT costs more compute but wins at very low bit widths or small networks.

### Memory Savings

Let \(P\) be parameter count. Uncompressed memory for weights only follows:

\[
M_{\text{FP32}} = 4P,\quad M_{\text{FP16}} = 2P,\quad M_{\text{INT8}} = P,\quad M_{\text{INT4}} = \frac{P}{2}
\]

in bytes when using one byte per INT8 and packed nibbles for INT4 in storage layouts that achieve half-byte per weight.

!!! math-intuition "In Plain English"
    Halve bytes each time you halve bits per parameter in a simple story. Real runtimes add alignment, metadata, and mixed formats, but order-of-magnitude VRAM for weight-only serving tracks bit-width linearly.

!!! example "Worked Example: Memory at Different Precisions"
    Take \(P = 70 \times 10^9\) parameters.

    - FP32: \(4 \times 70\) GB \(= 280\) GB.
    - FP16: \(2 \times 70\) GB \(= 140\) GB.
    - INT8: about 70 GB.
    - INT4: about 35 GB.

    A single A100 80GB can store INT8 weights for this parameter count with limited headroom for KV cache and runtime buffers; FP16 weights alone at 140 GB do not fit on one 80 GB card without sharding or offloading. This back-of-envelope explains why quantization is central for single-GPU 70B deployment in common hardware tiers.

### Mixed Precision Training and Loss Scaling

When training uses FP16 activations and gradients, the loss is often scaled before backward:

\[
\tilde{\mathcal{L}} = s \cdot \mathcal{L}.
\]

!!! math-intuition "In Plain English"
    Scaling the loss scales all gradients uniformly, helping tiny gradients stay representable in FP16. BF16 shares exponent width with FP32, so many transformer training recipes use BF16 without aggressive dynamic loss scaling. Always verify against the framework version you ship.

### NF4 and Groupwise Quantization (Inference)

Many INT4 schemes for LLM weights use **normal-float** codebooks or **groupwise** scales. For a group of \(g\) consecutive weights \(w \in \mathbb{R}^g\), a simple groupwise symmetric quantization picks one scale per group:

\[
s_g = \frac{\max_{i \in \text{group } g} |w_i|}{q_{\max}}.
\]

!!! math-intuition "In Plain English"
    **Groupwise** scales mean **nearby** weights in parameter memory share one **ruler**. Smaller groups track **local** magnitude variation more closely (better accuracy, more metadata bytes). Larger groups amortize overhead (fewer scale bytes) but increase distortion when a single outlier within the group inflates \(s_g\).

!!! example "Worked Example: Affine Quantization with Zero Point"
    Suppose activations on a channel are **non-negative** after ReLU, with values in \([0, 6.0]\). For UINT8 (0–255), affine mapping uses:

    \[
    s = \frac{6.0 - 0.0}{255 - 0} \approx 0.02353,\qquad z = 0.
    \]

    For a value \(x = 1.0\):

    \[
    x_q = \operatorname{round}\left(\frac{1.0}{s} + 0\right) = \operatorname{round}(42.5) = 43.
    \]

    Reconstruction \(\hat{x} = s \cdot x_q \approx 1.012\). The **zero point** \(z\) would be non-zero if the observed minimum were not exactly at an integer boundary; the workflow is identical with the generalized affine formulas above.

### KV Cache Precision

Autoregressive inference stores **key** and **value** tensors for past tokens. The memory for KV activations scales with **batch**, **heads**, **layer count**, and **sequence length**. Quantizing KV cache to INT8 or FP8 reduces bandwidth pressure in long-context serving:

\[
M_{\text{KV}} \propto L \cdot H \cdot D_{\text{head}} \cdot T \cdot \text{bytes per element}.
\]

!!! math-intuition "In Plain English"
    **Weights** are static; **KV cache** grows with **context length**. For long prompts, KV can dominate VRAM even when weights are INT4. Interview answers should mention **PagedAttention**-style memory management and **KV dtype** choices alongside **weight** quantization.

### Calibration Datasets for PTQ

Post-training quantization often uses a small **calibration** set of representative prompts to estimate **min**, **max**, or **percentile** ranges for activations. The calibration loss is not trained end-to-end; instead, histograms or running statistics set **\(s\)** and **\(z\)**. Larger calibration sets track production traffic better but cost more time before deployment.

\[
\text{range}_c = \text{percentile}_{99.9}\bigl(|a_c|\bigr)\ \text{per channel } c.
\]

!!! math-intuition "In Plain English"
    Using a **high percentile** instead of the true maximum **ignores** rare spikes that would explode the scale and **wreck** quantization resolution for typical values. The percentile is a **robust** heuristic, not a theorem—always validate on **downstream** tasks after changing calibration.

---

## Deep Dive

??? deep-dive "Deep Dive: Outliers in Activations and Weight Matrices"
    Transformer activations are not always well-behaved Gaussians. A small subset of channels can exhibit large magnitudes driven by attention patterns or layer norm interactions. Linear layers multiply activations by weights; outliers in either operand amplify errors after quantization. Methods like LLM.int8() explicitly route outlier dimensions through higher precision paths. SmoothQuant mathematically redistributes scale factors between weights and activations so both become easier to quantize jointly.

    In interviews, connect outliers to hardware: INT8 Tensor Cores expect bounded inputs; CPU kernels may use different accumulation dtypes. Show you would measure activation histograms per layer before choosing granularity.

??? deep-dive "Deep Dive: Evaluation Methodology Beyond Perplexity"
    Perplexity on a held-out corpus can improve while downstream reasoning degrades after aggressive INT4 schemes. Robust evaluation mixes closed-book QA, math word problems, coding generations with unit tests, and safety probes. Regression budgets often specify maximum allowed drop on business-critical tasks.

    Deployment teams also track latency and throughput: INT4 reduces memory bandwidth pressure but may increase dequantization overhead or kernel dispatch costs on some GPUs. Always pair accuracy tables with tokens per second at batch size representative of production.

---

## Code

```python
"""
Load a causal LM with bitsandbytes 8-bit and 4-bit quantization (transformers).
Requires: pip install transformers accelerate bitsandbytes torch
Run on a GPU machine; CPU-only path prints a skip message.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def build_quantized_model(
    model_id: str, mode: str
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    if mode == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
    elif mode == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError("mode must be 'int8' or 'int4'")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Public tiny model suitable for CI-style smoke tests (downloads on first run).
    MODEL_ID = "sshleifer/tiny-gpt2"
    if torch.cuda.is_available():
        mdl, tok = build_quantized_model(MODEL_ID, "int4")
        print(greedy_generate(mdl, tok, "Quantization reduces memory bandwidth pressure."))
    else:
        print("CUDA not available: skipping GPU quantized load in this demo.")
```

```python
"""
Manual symmetric INT8 quantize/dequantize for a NumPy vector (no torch required).
"""
from __future__ import annotations

import numpy as np


def symmetric_int8_quantize(x: np.ndarray) -> tuple[np.ndarray, float]:
    assert x.ndim == 1
    max_abs = float(np.max(np.abs(x)))
    if max_abs == 0.0:
        return np.zeros_like(x, dtype=np.int8), 1.0
    scale = max_abs / 127.0
    q = np.clip(np.round(x / scale), -128, 127).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return (q.astype(np.float32) * scale).astype(np.float32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    vec = rng.normal(size=8).astype(np.float32)
    q, s = symmetric_int8_quantize(vec)
    recon = dequantize(q, s)
    mse = float(np.mean((recon - vec) ** 2))
    print("original", vec)
    print("codes", q)
    print("scale", s)
    print("recon", recon)
    print("mse", mse)
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Symmetric vs affine:** When do you need a zero point in quantization, and when is symmetric quantization sufficient?
    2. **Per-tensor vs per-channel:** What failure mode does per-channel quantization mitigate in linear layers?
    3. **GPTQ vs AWQ:** Compare the objectives and inductive biases of these post-training methods at a high level.
    4. **Outliers:** Why do a few large-magnitude activation channels break naive INT8 matmuls, and how do practical systems mitigate that?
    5. **BF16 vs FP16 training:** Explain dynamic range differences and implications for loss scaling.
    6. **QAT vs PTQ:** When is QAT worth the extra training cost?
    7. **Memory planning:** Given parameter count \(P\), estimate FP16 versus INT4 weight memory and discuss single-GPU feasibility on 80GB cards.
    8. **SmoothQuant:** What problem does smoothing solve in activation quantization?
    9. **GGUF:** What role does the format play in llama.cpp ecosystems?
    10. **Evaluation:** Name three non-perplexity evaluations you would run after quantizing a chat model.

!!! interview "Follow-up Probes"
    - “How does double quantization in NF4 reduce overhead for scales?”
    - “What is group size in GPTQ, and how does it affect accuracy versus memory?”
    - “Why might INT4 kernels be slower than INT8 on some hardware despite fewer bits?”
    - “Describe straight-through estimators and their bias.”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Affine quantization maps floats to integers with scale and zero point.”
    - “Per-channel scales track outlier channels in weight tensors.”
    - “BF16 matches FP32 exponent range, often reducing loss scaling needs versus FP16.”
    - “PTQ is cheap; QAT recovers accuracy at very low bit width.”
    - “Memory bandwidth often dominates LLM inference; quantization cuts bytes moved.”

---

## References

1. Micikevicius, P., Narang, S., Alben, J., et al. **Mixed Precision Training.** ICLR (2018).
2. Jacob, B., Kligys, S., Chen, B., et al. **Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.** CVPR (2018).
3. Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.** NeurIPS (2022).
4. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** ICLR (2023).
5. Lin, J., Tang, J., Tang, H., et al. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** MLSys (2024).
6. Xiao, G., Lin, J., Seznec, M., et al. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.** ICML (2023).
7. NVIDIA **Hopper FP8 Training** documentation — E4M3 and E5M2 formats (versioned manuals).
8. Kahan, W. **Further Remarks on Reducing Truncation Errors.** Communications of the ACM (1965).
