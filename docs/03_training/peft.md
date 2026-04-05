# Parameter-Efficient Fine-Tuning

## Why This Matters for LLMs

Full fine-tuning updates **every** weight in a multi-billion-parameter model—expensive in **GPU memory** (optimizer states, activations) and risky for **catastrophic forgetting**. **Parameter-efficient fine-tuning (PEFT)** freezes most of the base model and injects **small trainable modules** (LoRA adapters, prefixes, prompts) so practitioners can specialize models on **private** data with consumer hardware. Interviewers expect you to derive **LoRA’s** low-rank structure, compare **adapters** vs **prompt tuning**, and explain **QLoRA** (quantized base + LoRA).

A second reason is **serving**: adapter weights are **megabytes** vs full checkpoints—multi-tenant systems can swap **LoRA** heads per customer while sharing one **base** model in GPU memory.

Third, **math clarity**: LoRA is not magic—it is **constrained optimization** in a **low-rank** subspace of weight updates, related to classical **matrix factorization** and **intrinsic dimension** results in deep learning.

---

## Core Concepts

### Full Fine-Tuning vs PEFT

**Full FT** updates \(\theta \in \mathbb{R}^{d}\) for **all** parameters. **PEFT** sets \(\theta = \theta_0 \oplus \Delta \phi\) where \(\theta_0\) is **frozen** and \(\phi\) is **small** (e.g., \(|\phi| \ll |\theta_0|\)).

### LoRA: Low-Rank Adaptation

For a frozen weight \(W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), LoRA learns:

\[
W = W_0 + \Delta W,\quad \Delta W = B A
\]

with \(B \in \mathbb{R}^{d_{\text{out}} \times r}\), \(A \in \mathbb{R}^{r \times d_{\text{in}}}\), rank \(r \ll \min(d_{\text{out}}, d_{\text{in}})\).

**Forward** (ignoring biases):

\[
h = W_0 x + B A x
\]

**Scaling** \(\alpha/r\) is common:

\[
h = W_0 x + \frac{\alpha}{r} B A x
\]

!!! math-intuition "In Plain English"
    Instead of learning a **full** update matrix (millions of entries), LoRA learns **two thin** matrices whose product is **low rank**. Most directions in weight space are **frozen**; you only move along a **small** subspace—often enough for **domain adaptation**.

### Parameter Savings

Trainable parameters for one layer:

\[
|\phi| = r (d_{\text{out}} + d_{\text{in}})
\]

vs full:

\[
|W| = d_{\text{out}} d_{\text{in}}
\]

**Savings factor** (order-of-magnitude):

\[
\frac{d_{\text{out}} d_{\text{in}}}{r(d_{\text{out}} + d_{\text{in}})} \approx \frac{\min(d_{\text{out}}, d_{\text{in}})}{2r}
\]

for comparable orders when \(d_{\text{out}} \approx d_{\text{in}} \approx d\): \(\approx d/(2r)\).

### Where to Apply LoRA

Typical practice targets **attention** projections \(W_q, W_k, W_v, W_o\) and sometimes **MLP** layers. **Not** every layer must be adapted—**rank** and **layer subset** are hyperparameters.

### QLoRA: Quantized Base + LoRA

**QLoRA** (Dettmers et al.) keeps **base weights** in **4-bit NF4** (or similar) via **bitsandbytes**, computes **forward** with **dequantization** or **fused kernels**, and trains **LoRA adapters** in **BF16/FP16**. This slashes **memory** for \(W_0\) while retaining **stable** adapter optimization.

!!! math-intuition "In Plain English"
    The **big** frozen matrix is **small on disk** (4-bit); the **tiny** adapter stays high precision so **gradients** don’t collapse. You pay **compute** for unpack/dequant, but **VRAM** drops massively.

### Adapters (Houlsby / Parallel)

Early **adapter** modules insert small **bottleneck** FFNs after attention/MLP:

\[
h' = h + f_{\text{adapter}}(h)
\]

**Serial** adapters chain in the residual stream; **parallel** adapters branch. LoRA is often **preferred** today for simplicity and **weight merging**.

### Prefix Tuning

**Prefix tuning** prepends **learned** continuous vectors (“virtual tokens”) to keys and values in attention:

\[
[K'; V'] = [\text{learned prefix } K_p, K],\ [\text{learned prefix } V_p, V]
\]

No change to **layer weights**—only **soft prompts** at each layer (variants exist).

### Prompt Tuning (Soft Prompts)

Learn an embedding tensor \(P \in \mathbb{R}^{p \times d}\) prepended to inputs—**shallow** (input only) vs **deep** (per layer). **Fewer** parameters than LoRA for small \(p\), but **less expressive** for hard tasks.

---

## Math: Rank and Expressivity

A matrix of **rank** \(r\) has **\(r\)** degrees of freedom per singular value. LoRA restricts \(\Delta W\) to **rank \(\le r\)**, implicitly **regularizing** toward **simple** updates—aligned with empirical observation that **intrinsic task dimension** is low for many fine-tunes.

### Relation to SVD (Conceptual)

If ideal update \(\Delta W = U \Sigma V^\top\), a rank-\(r\) approximation **truncates** to top singular values. LoRA’s \(BA\) is **not** explicitly SVD but **parameterizes** a rank-\(r\) subspace—similar **expressivity** class for small \(r\).

### Initialization Conventions

Common practice: **initialize \(A\) with random** (e.g., Kaiming), **\(B\) zero** so \(\Delta W = BA\) starts at **zero**—training begins from **base** model behavior and gradually **injects** task-specific shifts.

---

## Multi-Adapter Serving and Routing

**Multi-LoRA** serving loads **one** base and **many** small adapter checkpoints—swap per **tenant** or **task** with minimal VRAM overhead. **Router** models (MoE-style) can select adapters—advanced but **interview-flashy** when discussing **platform** ML.

---

## Python: LoRA Linear Layer (Educational)

```python
"""
Educational LoRA wrapped linear: y = W0(x) + (alpha/r) * B(A(x)).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor, base_out: torch.Tensor) -> torch.Tensor:
        scale = self.alpha / float(self.rank)
        lora_out = self.lora_b(self.lora_a(x))
        return base_out + scale * lora_out


if __name__ == "__main__":
    torch.manual_seed(0)
    in_f, out_f, r = 256, 512, 8
    x = torch.randn(4, 10, in_f)
    w0 = nn.Linear(in_f, out_f, bias=False)
    lora = LoRALinear(in_f, out_f, rank=r, alpha=16.0)
    with torch.no_grad():
        base = w0(x)
    y = lora(x, base)
    print(y.shape)
    trainable = sum(p.numel() for p in lora.parameters())
    frozen = sum(p.numel() for p in w0.parameters())
    print("trainable LoRA params:", trainable, "frozen base:", frozen)
```

---

## Python: Hugging Face PEFT Sketch

```python
"""
PEFT LoRA config sketch — requires transformers + peft installed.
"""
from __future__ import annotations

# from peft import LoraConfig, get_peft_model
# from transformers import AutoModelForCausalLM


def describe_lora_config() -> None:
    """
    Typical fields:
      r: rank
      lora_alpha: scaling alpha
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
      lora_dropout: 0.05
      bias: "none"
    """
    print("Uncomment imports and wrap model with get_peft_model when env has peft.")


if __name__ == "__main__":
    describe_lora_config()
```

---

## Merging LoRA into Base Weights

At deployment, **merge**:

\[
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
\]

for inference-only **speed** (no extra matmul). **Caveat**: merging **quantized** bases requires **dequant** first—workflow depends on stack.

---

## Worked Example: Parameter Count for One Layer

Take \(d_{\text{in}} = d_{\text{out}} = 4096\), rank \(r = 16\).

- **Full** weight count: \(4096^2 = 16{,}777{,}216\).
- **LoRA** trainable: \(r(d_{\text{in}} + d_{\text{out}}) = 16 \cdot 8192 = 131{,}072\).

**Ratio** \(\approx 128\times\) fewer trainable parameters for that layer’s update.

!!! math-intuition "In Plain English"
    Multiply by **dozens** of layers if you adapt **all** projections—still often **\(\ll\)** full model count, especially at **70B** scale.

---

## DoRA, AdaLoRA, VeRA (Pointers)

- **DoRA** (Weight-Decomposed Low-Rank Adaptation): decomposes updates into **direction** and **magnitude** components—sometimes better **fine-grained** control.
- **AdaLoRA**: prunes **rank** adaptively across layers—budget-aware allocation.
- **VeRA** (Vector-based Random Matrix Adaptation): shares **random** projections with **small** learned vectors—**extreme** parameter reduction.

Mention as **follow-on** methods; **LoRA/QLoRA** remain default **baseline** answers.

---

## Gradient Checkpointing + PEFT

PEFT reduces **optimizer** memory for **trainable** params; **activations** still dominate for long sequences. Combine with **gradient checkpointing** for long-context fine-tunes.

---

## PEFT Library: Typical Training Loop Pattern

1. Load **base** model in eval mode for frozen weights (except adapter paths).
2. Attach **LoRA** via `get_peft_model` with `LoraConfig`.
3. **Print** trainable parameters (`print_trainable_parameters()` in HF PEFT) to verify **\(\ll\)** base count.
4. Use standard **AdamW** on **adapter** params only; LR often **higher** than full FT (e.g., \(10^{-4}\)–\(10^{-3}\)) but **task-dependent**.
5. **Save** adapter-only checkpoint (`save_pretrained` on PEFT wrapper)—small **disk** artifact.

```python
"""
Conceptual HF PEFT usage — uncomment when dependencies available.
"""
from __future__ import annotations


def peft_training_skeleton() -> None:
    # from peft import LoraConfig, get_peft_model
    # from transformers import AutoModelForCausalLM
    # base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    # cfg = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"])
    # model = get_peft_model(base, cfg)
    # model.print_trainable_parameters()
    pass


if __name__ == "__main__":
    peft_training_skeleton()
```

---

## Orthogonal Subspaces and Overfitting

Because \(\Delta W\) is **low rank**, capacity to **memorize** idiosyncratic training noise is **reduced** vs full FT—often better **generalization** on small datasets. Conversely, **underfitting** can occur if rank \(r\) is too small for the task—**sweep** \(r \in \{8,16,32,64\}\) in practice.

---

## Interview Takeaways

- **LoRA**: \(\Delta W = BA\), rank \(r\), scaling \(\alpha/r\); huge **param savings** on attention/MLP projections.
- **QLoRA**: **4-bit** base + **LoRA** adapters in higher precision; **memory** win for fine-tuning on **single** GPUs.
- **Adapters / Prefix / Prompt**: different **injection points**—LoRA dominates **open** recipes today.
- **Merging**: optional **single-weight** deployment after \(W_{\text{merged}} = W_0 + BA\).
- **Savings**: trainable \(\approx r(d_{\text{in}} + d_{\text{out}})\) per adapted layer vs \(d_{\text{in}} d_{\text{out}}}\).

---

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- Houlsby et al., *Parameter-Efficient Transfer Learning for NLP* — adapters
- Li & Liang, *Prefix-Tuning: Optimizing Continuous Prompts for Generation* — [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- Lester et al., *The Power of Scale for Parameter-Efficient Prompt Tuning* — [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
- PEFT library — Hugging Face `peft` documentation
