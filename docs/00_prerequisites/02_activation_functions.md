# Activation Functions

## Why This Matters for LLMs

Activation functions are the nonlinearities that let deep networks approximate arbitrary functions. Without them, stacking linear layers would collapse to a single linear map, and a Transformer would be unable to represent the rich conditional structure of language. Modern LLMs do not all use the same choice: GPT-style models and BERT family networks typically use **GELU** in feed-forward blocks; **SiLU/Swish** appears in LLaMA and related architectures; **softmax** normalizes attention scores and produces probability distributions over vocabulary at the output layer. The specific nonlinearity affects gradient flow during training, how quickly units saturate, and how smoothly optimization landscapes behave.

Interviewers often probe whether you understand **saturation** (when activations sit in flat regions where gradients vanish), **gradient flow** through many layers, and **why** ReLU dominated for years while today's LLMs favor smooth, probabilistically motivated activations like GELU. You should be able to name what each major architecture uses, sketch the shape of the curve, and connect that choice to training stability and empirical performance—not just recite definitions.

Finally, activations do not exist in isolation: they interact with normalization (LayerNorm), residual connections, and gating (e.g., SwiGLU in LLaMA FFNs). A crisp mental model of sigmoid, tanh, ReLU, GELU, SiLU, and softmax is table stakes for ML systems and LLM roles at top companies.

!!! tip "Notation Help"
    If you're unfamiliar with derivatives or the notation \(\frac{d}{dz}\), see the [Math Prerequisites](00_math_prerequisites.md#2-functions-and-derivatives) section. It explains derivatives intuition-first with worked examples.

## Core Concepts

### Sigmoid

The sigmoid maps any real input to the open interval \( (0, 1) \):

\[
\sigma(z) = \frac{1}{1 + e^{-z}}.
\]

!!! math-intuition "In Plain English"
    The sigmoid squashes large positive \( z \) toward 1 and large negative \( z \) toward 0. It is interpretable as a soft "probability" when used in a single output, but hidden layers rarely use it today because of vanishing gradients.

The derivative is

\[
\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr).
\]

!!! math-intuition "In Plain English"
    The slope is largest near \( z = 0 \) and approaches zero when \( z \) is very positive or very negative. That is **saturation**: in those regions, backpropagation multiplies many small derivatives through the network, causing **vanishing gradients** in deep stacks.

!!! example "Worked Example: sigmoid at \( z = 2.0 \)"
    \[
    \sigma(2) = \frac{1}{1 + e^{-2}} \approx \frac{1}{1 + 0.1353} \approx 0.8808.
    \]
    \[
    \sigma'(2) = 0.8808 \times (1 - 0.8808) \approx 0.105.
    \]
    The neuron is already in a mildly saturated regime: the output is close to 1 and the local gradient is modest compared to the maximum \( \sigma'(0) = 0.25 \).

### Tanh

Hyperbolic tangent maps \( \mathbb{R} \to (-1, 1) \):

\[
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}.
\]

!!! math-intuition "In Plain English"
    Tanh is an S-shaped squashing function like sigmoid, but its outputs are **zero-centered** (negative inputs yield negative outputs). That often makes optimization easier than sigmoid hidden activations, which output only positive values.

It can be written in terms of sigmoid as \( \tanh(z) = 2\sigma(2z) - 1 \). The derivative is

\[
\frac{d}{dz}\tanh(z) = 1 - \tanh^2(z).
\]

!!! math-intuition "In Plain English"
    Like sigmoid, tanh **saturates** at large \( |z| \), so vanishing gradients remain a concern in very deep unmitigated stacks—though centering still helps compared to sigmoid in many classical setups.

!!! example "Worked Example: tanh at \( z = 1.0 \)"
    Using a calculator, \( \tanh(1) \approx 0.7616 \). Then
    \[
    \frac{d}{dz}\tanh(1) = 1 - (0.7616)^2 \approx 0.4200.
    \]
    The unit is still in a reasonably active region compared to \( |z| \gg 1 \), where the derivative would be nearly zero.

### ReLU

Rectified Linear Unit is piecewise linear:

\[
\mathrm{ReLU}(z) = \max(0, z).
\]

!!! math-intuition "In Plain English"
    Negative inputs are clipped to zero; positive inputs pass through unchanged. That sparsifies activations and avoids the saturation problem on the positive side that plagues sigmoid/tanh.

The derivative (where it exists) is

\[
\frac{d}{dz}\mathrm{ReLU}(z) =
\begin{cases}
1 & z > 0, \\
0 & z < 0,
\end{cases}
\]

with the undefined point at \( z = 0 \) handled in practice by convention (e.g., 0 or 1).

!!! math-intuition "In Plain English"
    If a neuron's pre-activation stays negative for all training examples, its gradient is always zero—the **dying ReLU** problem. On the positive side, the gradient is 1, which helps gradient flow but can also contribute to exploding activations without careful initialization and normalization.

Computationally, ReLU is cheap: comparison and max versus exponentials in tanh/sigmoid.

!!! example "Worked Example: ReLU in active and dead regions"
    - For \( z = 3.0 \): \( \mathrm{ReLU}(3) = 3 \), derivative \( 1 \) (active).
    - For \( z = -2.0 \): \( \mathrm{ReLU}(-2) = 0 \), derivative \( 0 \) (dead for backprop through this unit).

### Leaky ReLU and Parametric ReLU (PReLU)

**Leaky ReLU** uses a small slope \( \alpha > 0 \) for negative inputs:

\[
f(z) = \max(\alpha z, z) =
\begin{cases}
z & z \ge 0, \\
\alpha z & z < 0.
\end{cases}
\]

!!! math-intuition "In Plain English"
    A tiny leak (e.g., \( \alpha = 0.01 \)) prevents the gradient from being exactly zero for all negative \( z \), reducing the severity of dying ReLU at the cost of a small negative activation mass.

**Parametric ReLU (PReLU)** treats \( \alpha \) as a **learned** parameter (per channel or shared), so the network can adapt how much negative information to retain.

\[
f(z) = \max(\alpha z, z), \quad \alpha \text{ trainable.}
\]

!!! math-intuition "In Plain English"
    Instead of fixing the leak, the model learns how "linear" the negative half should be—useful when the optimal nonlinearity is data-dependent.

### GELU (Gaussian Error Linear Unit)

GELU is defined using the Gaussian cumulative distribution function \( \Phi \):

\[
\mathrm{GELU}(x) = x \cdot \Phi(x).
\]

!!! math-intuition "In Plain English"
    You scale the identity by the probability that a draw from a standard Gaussian is less than \( x \). Large positive \( x \) behave like identity; large negative \( x \) are softly suppressed—smoother than ReLU's hard zero.

A common **approximation** used in implementations (e.g., in many Transformer FFNs) is

\[
\mathrm{GELU}(x) \approx 0.5\, x \left( 1 + \tanh\left( \sqrt{\tfrac{2}{\pi}} \left( x + 0.044715\, x^{3} \right) \right) \right).
\]

!!! math-intuition "In Plain English"
    This is a smooth, differentiable function that looks like a softened ReLU: near zero it is curved rather than kinked, which can make optimization landscapes nicer for very deep networks and when combined with LayerNorm and Adam-style optimizers.

GELU is the activation used in **GPT-2**, **GPT-3**, and **BERT**-style Transformers for feed-forward sublayers (exact definition vs. approximation varies by codebase).

!!! example "Worked Example: approximate GELU at \( x = 1.0 \)"
    Let \( t = \sqrt{2/\pi}(1 + 0.044715) \approx 0.7979 \times 1.044715 \approx 0.8336 \). Then \( \tanh(t) \approx 0.6826 \), and
    \[
    \mathrm{GELU}(1) \approx 0.5 \times 1 \times (1 + 0.6826) \approx 0.8413.
    \]
    (Exact \( \mathrm{GELU}(1) = \Phi(1) \approx 0.8413 \)—the approximation matches closely at this point.)

### SiLU and Swish

The **Sigmoid Linear Unit (SiLU)** is also called **Swish** (with unit parameters):

\[
\mathrm{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}.
\]

!!! math-intuition "In Plain English"
    For negative \( x \), sigmoid damps the output toward zero but allows small negative values—unlike ReLU's hard cutoff. For large positive \( x \), \( \sigma(x) \approx 1 \), so SiLU behaves like identity.

SiLU/Swish is used in **LLaMA** and **PaLM** family models (often inside SwiGLU blocks—see Deep Dive). **Connection to GELU**: both are smooth, non-monotonic-ish adjustments of the identity; GELU uses the Gaussian CDF while SiLU uses the logistic sigmoid as a gate. They are distinct functions but play a similar "smooth ReLU family" role in modern architectures.

!!! example "Worked Example: SiLU at \( x = -1.0 \) and \( x = 2.0 \)"
    - \( x = -1 \): \( \sigma(-1) \approx 0.2689 \), so \( \mathrm{SiLU}(-1) \approx -0.2689 \) (small negative, not hard-clamped to 0).
    - \( x = 2 \): \( \sigma(2) \approx 0.8808 \), so \( \mathrm{SiLU}(2) \approx 1.7616 \) (close to 2).

### Softmax

Given a vector \( \mathbf{x} = (x_1, \ldots, x_K)^\top \), softmax produces a probability vector:

\[
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}, \quad i = 1, \ldots, K.
\]

!!! math-intuition "In Plain English"
    Softmax exponentiates each score and normalizes so entries are positive and sum to 1. The largest logit gets the most mass, but other classes still receive some probability unless gaps are huge.

**Temperature** \( T > 0 \) scales logits before softmax:

\[
\mathrm{softmax}_T(x_i) = \frac{e^{x_i / T}}{\sum_{j} e^{x_j / T}}.
\]

!!! math-intuition "In Plain English"
    \( T > 1 \) softens the distribution (more uniform); \( T < 1 \) sharpens it (more peaked). In some distillation and sampling setups, temperature controls randomness versus confidence.

In **attention**, softmax normalizes compatibility scores across keys so weights sum to 1 per query position. In the **output layer** of classifiers or language models, softmax maps logits to token probabilities.

!!! example "Worked Example: softmax on a 3-element vector"
    Let \( \mathbf{x} = (2.0,\, 1.0,\, 0.0)^\top \). Then \( e^2 \approx 7.389 \), \( e^1 \approx 2.718 \), \( e^0 = 1 \); sum \( \approx 11.107 \).
    \[
    \mathrm{softmax}(\mathbf{x}) \approx (0.665,\, 0.245,\, 0.090).
    \]
    The first class dominates but does not take all mass.

## Deep Dive

??? deep-dive "GLU variants, GELU/SwiGLU trend, and softmax stability"

    **Gated Linear Units (GLU)** and variants pair linear projections with multiplicative gating. A common FFN formulation in modern LLMs is **SwiGLU**:

    \[
    \mathrm{FFN}(\mathbf{x}) = \bigl( (\mathbf{x} W_1) \odot \mathrm{Swish}(\mathbf{x} V) \bigr) W_2,
    \]

    where \( W_1, V, W_2 \) are learned matrices, \( \odot \) is element-wise product, and \( \mathrm{Swish}(t) = t \cdot \sigma(t) \). Intuition: one branch computes **values**, the other a **sigmoid gate** that controls which dimensions pass through—richer than a single activation after one matrix multiply.

    **Why move from ReLU to GELU/SwiGLU in large Transformers?** Empirically, smooth activations (GELU, SiLU) often yield better perplexity and downstream metrics at scale; the inductive bias interacts favorably with residual connections, LayerNorm, and Adam. SwiGLU increases parameter and compute in the FFN (two up-projections) but has become a standard quality-efficiency tradeoff in LLaMA-class models.

    **Softmax numerical stability**: for any constant \( c \),

    \[
    \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}.
    \]

    Setting \( c = \max_j x_j \) prevents overflow in \( e^{x_i} \) and improves floating-point behavior. Libraries always use the stabilized form in production.

## Code

```python
"""
Plot common activation functions and derivatives; implement SwiGLU FFN in PyTorch.
Requires: pip install torch matplotlib numpy
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu_approx(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def gelu_approx_derivative(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return (gelu_approx(x + eps) - gelu_approx(x - eps)) / (2 * eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -500, 500)))


def silu_derivative(x: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    return s + x * s * (1.0 - s)


def plot_activations_and_derivatives() -> None:
    x = np.linspace(-4, 4, 800)

    def dsigmoid(z):
        s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        return s * (1.0 - s)

    def dtanh(z):
        t = np.tanh(z)
        return 1.0 - t**2

    def drelu(z):
        return (z > 0).astype(float)

    activations = {
        r"$\sigma$ (sigmoid)": (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))), dsigmoid(x)),
        r"$\tanh$": (np.tanh(x), dtanh(x)),
        r"ReLU": (np.maximum(0, x), drelu(x)),
        r"GELU (approx)": (gelu_approx(x), gelu_approx_derivative(x)),
        r"SiLU / Swish": (silu(x), silu_derivative(x)),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for name, (y, _) in activations.items():
        axes[0].plot(x, y, label=name, linewidth=2)
    axes[0].set_title("Activation functions")
    axes[0].set_xlabel(r"$z$ or $x$")
    axes[0].set_ylabel(r"$f(z)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8)

    for name, (_, dy) in activations.items():
        axes[1].plot(x, dy, label=name, linewidth=2)
    axes[1].set_title("Derivatives (GELU via central difference)")
    axes[1].set_xlabel(r"$z$ or $x$")
    axes[1].set_ylabel(r"$f'(z)$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


class SwiGLUFFN(nn.Module):
    """Feed-forward block: (x @ w1) * Swish(x @ v) @ w2 + bias pattern."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(a) = a * sigmoid(a); gate = Swish(x @ v)
        gate = F.silu(self.v(x))
        return self.w2(self.w1(x) * gate)


def _demo_swiglu() -> None:
    torch.manual_seed(0)
    b, t, d = 2, 4, 16
    x = torch.randn(b, t, d)
    ffn = SwiGLUFFN(d_model=d, d_ff=64)
    y = ffn(x)
    assert y.shape == x.shape
    print("SwiGLU output shape:", tuple(y.shape))


if __name__ == "__main__":
    plot_activations_and_derivatives()
    _demo_swiglu()
```

## Interview Guide

!!! interview "FAANG-Level Questions"

    1. **Why can't a deep network without nonlinear activations represent arbitrary decision boundaries?** Expect: composition of linear maps is linear; need nonlinearity for universality and feature hierarchies.

    2. **Compare vanishing gradients in sigmoid/tanh vs. ReLU.** Expect: sigmoid/tanh saturate on both sides; ReLU has zero gradient only on the negative side (dying ReLU) but unit gradient on positive side.

    3. **What is the derivative of ReLU and where is it undefined?** Expect: 1 for \( z > 0 \), 0 for \( z < 0 \); convention at 0; mention subgradient.

    4. **Write softmax and explain why we subtract the max before exponentiating.** Expect: numerical stability; mathematical equivalence.

    5. **What activation does BERT use in its FFN, and why might GELU beat ReLU there?** Expect: GELU; smoother than ReLU; empirical gains with Adam/LayerNorm/residuals (high level).

    6. **Define SiLU/Swish and name one major open-weight LLM family that uses SwiGLU.** Expect: \( x\sigma(x) \); LLaMA / many descendants.

    7. **How does temperature affect softmax at the output of a language model?** Expect: sharp vs. flat distributions; sampling vs. greedy behavior.

    8. **What is SwiGLU in a Transformer FFN?** Expect: two up-projections, multiply by Swish gate, down-project; parameter overhead vs. quality.

    9. **Why might zero-centered outputs (tanh) help compared to sigmoid in older networks?** Expect: nicer gradient dynamics when inputs to next layer have mean near zero (historical context).

    10. **When is softmax used inside attention?** Expect: over scores per query across keys; weights sum to 1; scaled dot-product attention.

!!! interview "Follow-up Probes"

    1. **If all ReLU units in a layer are dead, what happens in backprop?** Gradients w.r.t. inputs to that layer are zero for that path; may need different init, Leaky ReLU, or learning rate adjustment.

    2. **Is GELU monotonic?** Yes, GELU is monotonically increasing (unlike SiLU which is non-monotonic).

    3. **How does SwiGLU differ from a standard ReLU FFN \( \mathrm{ReLU}(x W_1) W_2 \)?** Extra gate projection and multiplicative interaction; Swish instead of ReLU.

    4. **What is the relationship between logits and cross-entropy at the output?** Softmax + log + NLL; stable fused implementation in frameworks.

    5. **Could you use softmax hidden activations?** Impractical: full vector normalization per unit; not used as element-wise hidden activation; softmax is for vectors to probabilities.

!!! key-phrases "Key Phrases to Use in Interviews"

    - "Saturation kills gradient flow when \( \sigma' \) or \( \tanh' \) is near zero."
    - "ReLU sparsifies and avoids bilateral saturation but can suffer from dying ReLU."
    - "GELU is a smooth, probabilistically motivated nonlinearity standard in GPT/BERT FFNs."
    - "SiLU and Swish are self-gated: \( x \cdot \sigma(x) \); SwiGLU pairs them with GLU-style FFNs in LLaMA."
    - "Always stabilize softmax with \( z_i - \max_j z_j \) before exponentiating."
    - "Attention uses softmax over scores so weights are positive and sum to one per query."
    - "Modern LLMs trade extra FFN parameters (SwiGLU) for better quality at scale."

## References

- Hendrycks, D., & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415)
- Ramachandran, P., Zoph, B., & Le, Q. V. (2017). *Searching for Activation Functions*. (Swish). [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)
- Dauphin, Y. N., et al. (2017). *Language Modeling with Gated Convolutional Networks*. (GLU formulation). [arXiv:1612.08083](https://arxiv.org/abs/1612.08083)
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
