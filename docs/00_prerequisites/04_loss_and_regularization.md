# Loss Functions and Regularization

## Why This Matters for LLMs

**Cross-entropy** is the training objective for essentially **all** autoregressive language models: at each position, the model outputs a distribution over the vocabulary, and you minimize the negative log-probability of the true next token. That is not a stylistic choice—it is the negative log-likelihood of a categorical model under maximum likelihood estimation (MLE). If you cannot connect softmax outputs to log-probabilities, to NLL, and to the KL-divergence story, you will not be able to reason cleanly about instruction tuning, distillation, or why certain auxiliary losses are shaped the way they are.

**Regularization** is not “extra math”—it is how large models avoid pathological solutions in finite data regimes. **Dropout**, **weight decay**, and **normalization** (especially **LayerNorm** and variants like **RMSNorm**) are structural fixtures of Transformers. Interviewers expect you to explain not just *what* these do, but *why* they change optimization dynamics and generalization, and how train-time randomness (dropout) differs from penalty terms (L1/L2).

Finally, the information-theoretic bridge—**cross-entropy versus KL divergence versus entropy**—is the clean language for comparing a model distribution to a target distribution. That vocabulary shows up when you discuss alignment losses, knowledge distillation, and probabilistic interpretations of calibration. Part 1 of LLMBase leans on this background; making it automatic here saves you from hand-waving later.

!!! tip "Notation Help"
    The notation \(\log\) in ML papers usually means the **natural logarithm** (base \(e\)), often written as \(\ln\) in calculus courses. The \(\sum\) symbol means "sum up." See [Math Prerequisites](00_math_prerequisites.md#4-exponentials-and-logarithms) for a refresher on logs and exponentials.

## Core Concepts

### Mean Squared Error (MSE)

For targets \(y_i\) and predictions \(\hat{y}_i\) over \(n\) examples, **mean squared error** is:

\[
L_{\mathrm{MSE}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2.
\]

!!! math-intuition "In Plain English"
    MSE measures **squared distance** between predictions and targets. Squaring penalizes large errors heavily and makes the loss smooth everywhere, which is convenient for gradient-based optimization in regression settings.

MSE is the standard objective for **regression** (real-valued outputs): predicting prices, scores, or any continuous quantity where Gaussian noise assumptions are a reasonable modeling shorthand.

**Why MSE is usually wrong for classification:** class labels are discrete; treating them as real numbers and minimizing squared error does not correspond to maximizing the likelihood of a Bernoulli or categorical distribution. Probabilistic classification wants a **calibrated probability** \(\hat{y}\) for each class, and the right scoring rule is typically **log-loss** (cross-entropy), not squared error on raw labels. (You *can* use MSE on **one-hot vectors** in toy setups, but it is not the principled likelihood objective and behaves differently under model misspecification.)

!!! example "Worked Example: MSE with three predictions"
    Suppose targets \(y = [1,\, 0,\, 2]\) and predictions \(\hat{y} = [0.5,\, 0.5,\, 2.5]\).

    \[
    \begin{aligned}
    L_{\mathrm{MSE}} &= \frac{1}{3}\Big[(0.5-1)^2 + (0.5-0)^2 + (2.5-2)^2\Big] \\
    &= \frac{1}{3}\Big[0.25 + 0.25 + 0.25\Big] = \frac{0.75}{3} = 0.25.
    \end{aligned}
    \]

### Cross-Entropy Loss (Binary)

For a single example with label \(y \in \{0,1\}\) and predicted probability \(\hat{y} \in (0,1)\) for class 1, **binary cross-entropy** is:

\[
L = -\Big[y\log \hat{y} + (1-y)\log(1-\hat{y})\Big].
\]

!!! math-intuition "In Plain English"
    If the true label is 1, you pay \(-\log \hat{y}\): confident correct predictions (\(\hat{y}\approx 1\)) cost almost nothing; confident wrong predictions cost a lot. The \((1-y)\) branch handles label 0 symmetrically.

**Derive from maximum likelihood (Bernoulli).** Assume \(Y \sim \mathrm{Bernoulli}(p)\) with \(p=\hat{y}\) parameterized by your model. For a single observation \(y\), the likelihood is \(p^{y}(1-p)^{1-y}\). Take the negative log:

\[
-\log p^{y}(1-p)^{1-y} = -y\log p - (1-y)\log(1-p),
\]

!!! math-intuition "In Plain English"
    Taking \(-\log\) turns “probability of what we saw” into a **cost**: high likelihood \(\Rightarrow\) small cost. The exponents \(y\) and \(1-y\) pick exactly one of the two branches—whichever label you observed contributes its log-probability term.

which is exactly binary cross-entropy. Minimizing NLL **is** MLE for this probabilistic model.

!!! example "Worked Example: binary cross-entropy"
    Let \(y=1\) and \(\hat{y}=0.8\).

    \[
    L = -\big[1\cdot \log(0.8) + 0\cdot \log(0.2)\big] = -\log(0.8) \approx 0.223.
    \]

    Let \(y=0\) and \(\hat{y}=0.3\) (so the model assigns \(0.7\) to class 1).

    \[
    L = -\big[0\cdot \log(0.3) + 1\cdot \log(0.7)\big] = -\log(0.7) \approx 0.357.
    \]

### Cross-Entropy Loss (Multi-class)

For \(K\) classes, represent the true label as a **one-hot** vector \(\mathbf{y} \in \{0,1\}^K\) (exactly one entry is 1). If model probabilities are \(\hat{\mathbf{y}}\) with \(\sum_{i=1}^{K}\hat{y}_i=1\), the **multi-class cross-entropy** for one example is:

\[
L = -\sum_{i=1}^{K} y_i \log(\hat{y}_i).
\]

Because only one \(y_i\) is 1, this collapses to \(-\log(\hat{y}_{c^\*})\) where \(c^\*\) is the correct class—**negative log-likelihood** under a categorical model.

In practice, \(\hat{\mathbf{y}}\) comes from a **softmax** on logits \(\mathbf{z}\):

\[
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}.
\]

!!! math-intuition "In Plain English"
    Softmax turns unconstrained scores into a **probability vector**: bigger logit \(\Rightarrow\) larger mass. The sum in the denominator forces \(\sum_i \hat{y}_i = 1\). Libraries typically combine `log_softmax` with NLL for numerical stability (the “log-sum-exp trick”). Once you have \(\hat{\mathbf{y}}\), cross-entropy asks: **how surprised are you by the correct token** under that distribution? In LLMs, \(K\) is the vocabulary size (often tens of thousands to hundreds of thousands), and you sum this loss over positions (and batch elements).

**This is THE LLM training loss** when targets are one-hot tokens: you maximize the log-probability of the observed next token (equivalently minimize its negative log).

For a length-\(T\) sequence with logits \(\mathbf{z}_{t}\) and integer targets \(y_t\), a typical **token-averaged** training loss is:

\[
L = \frac{1}{T}\sum_{t=1}^{T} \Big[-\log \mathrm{softmax}(\mathbf{z}_{t})_{y_t}\Big].
\]

!!! math-intuition "In Plain English"
    This is the same per-position categorical NLL, just **summed across time** and normalized by sequence length so shorter and longer sequences contribute comparably when you average over examples.

!!! example "Worked Example: vocabulary size \(K=4\)"
    True token is class 3 (one-hot \(\mathbf{y}=[0,0,1,0]\)). Suppose predicted probabilities are:

    \[
    \hat{\mathbf{y}} = [0.10,\, 0.05,\, 0.70,\, 0.15].
    \]

    Then

    \[
    L = -\sum_{i=1}^{4} y_i \log \hat{y}_i = -\log(0.70) \approx 0.357.
    \]

    If instead the model is confidently wrong, e.g. \(\hat{\mathbf{y}}=[0.70,\,0.10,\,0.10,\,0.10]\) while \(y_3=1\), then \(L=-\log(0.10)\approx 2.303\), a much larger penalty.

### Connection to KL Divergence

Let \(p\) be the **true** label distribution (often one-hot for supervised classification) and \(q\) be the model distribution. The **cross-entropy** can be written \(H(p,q)\), and the **KL divergence** is:

\[
\mathrm{KL}(p\,\|\,q) = \sum_i p_i \log\frac{p_i}{q_i} = \underbrace{\sum_i p_i \log p_i}_{=-H(p)} - \sum_i p_i \log q_i.
\]

!!! math-intuition "In Plain English"
    KL is an **asymmetric** measure of how far \(q\) is from \(p\): it weighs log-ratios \(\log(p_i/q_i)\) by how often \(p_i\) fires. Writing it as \(-H(p) - \sum_i p_i \log q_i\) exposes the **entropy of \(p\)** plus the cross-entropy term you already recognize.

Rearranging,

\[
\underbrace{-\sum_i p_i \log q_i}_{\text{cross-entropy } H(p,q)} = \mathrm{KL}(p\,\|\,q) + H(p).
\]

!!! math-intuition "In Plain English"
    This identity is the whole trick: **cross-entropy splits cleanly** into “how far \(q\) is from \(p\)” (KL) plus “how compressible \(p\) is by itself” (entropy). Optimize the model through \(q\), and the \(H(p)\) term is just background noise when \(p\) is fixed.

!!! math-intuition "In Plain English"
    When \(p\) is fixed (as with a one-hot label), \(H(p)\) is a **constant** independent of the model. So **minimizing cross-entropy** is equivalent to **minimizing KL divergence** from the data distribution to the model distribution—up to that constant. This is the bridge to information theory from Part 1: KL measures “extra bits” needed to encode samples from \(p\) using a code optimized for \(q\).

### L2 Regularization (Weight Decay)

Given a data loss \(L(\mathbf{w})\), **L2 regularization** adds a squared penalty:

\[
L_{\mathrm{total}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \sum_{i} w_i^2.
\]

The gradient gains an extra term:

\[
\nabla_{\mathbf{w}} L_{\mathrm{total}} = \nabla_{\mathbf{w}} L + 2\lambda \mathbf{w}.
\]

!!! math-intuition "In Plain English"
    The penalty contributes **\(2\lambda w_i\)** to each coordinate’s derivative: the larger the weight, the stronger the push back toward zero. In gradient descent, that becomes an extra **\(-2\eta\lambda w_i\)** term in the weight update—pure shrinkage toward the origin on top of the data-driven gradient.

In a vanilla gradient descent step \(\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla L_{\mathrm{total}}\), this behaves like **shrinking** weights toward zero by an amount proportional to \(\mathbf{w}\).

!!! math-intuition "In Plain English"
    Large weights often mean the model is leaning hard on particular features; L2 discourages extreme magnitudes, which tends to improve generalization in many settings and can make optimization smoother. In Transformers, **weight decay** is commonly applied in a way that interacts subtly with adaptive optimizers (see Deep Dive).

### L1 Regularization

\[
L_{\mathrm{total}}(\mathbf{w}) = L(\mathbf{w}) + \lambda \sum_i |w_i|.
\]

Near zero, the penalty is **linear** rather than quadratic in \(|w_i|\), which encourages many coefficients to become **exactly** zero (sparse solutions) under suitable conditions—unlike L2, which typically shrinks everything but rarely drives weights to exact zeros.

!!! math-intuition "In Plain English"
    L1 is a classic sparsity tool. LLMs are rarely trained “to sparsity” via plain L1 alone at full scale, but sparsity ideas matter for pruning, structured sparsity, and some compression workflows.

### Dropout

During training, **dropout** independently sets a fraction of activations to zero with probability \(p\) (per retained unit, the “keep probability” is \(1-p\)). At **test time**, standard dropout scales outputs to match expected magnitude—often by using **inverted dropout**: during training, surviving activations are scaled by \(1/(1-p)\) so that **evaluation can be inference without extra scaling**.

!!! math-intuition "In Plain English"
    Dropout prevents units from **co-adapting** (fitting elaborate mutual corrections that do not generalize). A useful mental model is an implicit **ensemble** over many thinned subnetworks sampled at training time; at test time you approximate averaging that ensemble.

### Batch Normalization

For a batch of activations, **batch normalization** standardizes each feature dimension using batch statistics. For a feature \(x\) across a batch \(B\):

\[
\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}}, \qquad y = \gamma \hat{x} + \beta,
\]

where \(\mu_B\) and \(\sigma_B^2\) are the batch mean and variance, and \(\gamma,\beta\) are learnable scale/shift parameters.

!!! math-intuition "In Plain English"
    BatchNorm re-centers and rescales activations so optimization is less sensitive to initialization and scale drift. But it introduces **batch dependence** and can behave oddly for small or non-i.i.d. batches—problematic for variable-length sequences and large distributed training.

**Why LLMs prefer LayerNorm:** sequence models want stable normalization **per position** without coupling across unrelated examples in a batch. LayerNorm normalizes across features for each token independently, avoiding batch statistics for that normalization step.

### Layer Normalization

**Layer normalization** normalizes across the **feature dimension** for each token (each position), not across the batch. For a feature vector \(\mathbf{x} \in \mathbb{R}^d\) (one position):

\[
\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \varepsilon}}, \qquad \mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta,
\]

where \(\mu\) and \(\sigma^2\) are computed across the \(d\) features of \(\mathbf{x}\), and \(\gamma,\beta \in \mathbb{R}^d\) are learnable (often per-dimension).

!!! math-intuition "In Plain English"
    Each token gets its own mean/variance summary across channels, stabilizing training for deep stacks of attention and MLP blocks. This is why **every mainstream Transformer block** includes LayerNorm (or a close variant) around sublayers.

## Deep Dive

??? note "Expand: RMSNorm, pre/post-norm, AdamW, label smoothing"
    ### RMSNorm (used in LLaMA) vs LayerNorm

    **LayerNorm** subtracts mean and divides by standard deviation across features, then applies affine \(\gamma,\beta\). **RMSNorm** often removes mean-centering and scales by the root-mean-square (RMS) of features, then applies a learned gain (and sometimes bias):

    \[
    \mathrm{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}, \qquad \tilde{x}_j = \frac{x_j}{\mathrm{RMS}(\mathbf{x})},
    \]

    !!! math-intuition "In Plain English"
        RMSNorm keeps a **scale-stabilizing** normalization (divide by a scalar RMS) without subtracting the mean across features. In practice this can be slightly cheaper and still controls activation scale in deep residual stacks; LLaMA-class models popularized it.

    ### Pre-norm vs post-norm Transformer (and why pre-norm won)

    In a **post-norm** block, you apply: \(\mathbf{x} + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}))\)—normalization inside the residual path is less common in the original formulation; the classic “original Transformer” uses LayerNorm placements that are harder to optimize at depth.

    In **pre-norm**, you compute:

    \[
    \mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf{x}_\ell)),
    \]

    !!! math-intuition "In Plain English"
        The residual stream \(\mathbf{x}_\ell\) is fed through LayerNorm **before** each sublayer (attention/FFN), so gradients through the residual path behave more like “identity + small perturbation” deep in the network—often easier to optimize than post-norm placements at large depth. Empirically, pre-norm became the default in many modern LLM stacks.

    ### Weight decay vs L2 in Adam (they differ!)

    For plain SGD, adding an L2 penalty to the loss is closely analogous to **weight decay** in the update. But in **Adam**, the two are **not equivalent** because Adam scales updates by adaptive second-moment estimates. **AdamW** decouples weight decay: you apply decay directly on weights,

    \[
    \mathbf{w} \leftarrow \mathbf{w} - \eta\left(\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\varepsilon} + \lambda \mathbf{w}\right),
    \]

    !!! math-intuition "In Plain English"
        Weight decay is applied **additively in the update**, not as if it were another gradient term that also gets divided by \(\sqrt{\hat{\mathbf{v}}}\). That matters because Adam’s adaptive scaling would otherwise **weaken or distort** L2-style penalties in a way that does not match classical weight decay—so **AdamW** is the standard fix. For Transformers, **AdamW + weight decay** is the common baseline.

    ### Label smoothing in LLM training

    Instead of a hard one-hot target vector \(p\) with a 1 on the true class, use a mixture:

    \[
    p'_i = (1-\varepsilon)\,p_i + \frac{\varepsilon}{K},
    \]

    !!! math-intuition "In Plain English"
        You are training against a **slightly blurred** target: still mostly the true token, but with a little probability smeared across the whole vocabulary. That discourages the model from pushing logits to absurd extremes just to drive training loss to zero on memorized patterns. It changes the effective objective from hard KL to a smoother target distribution—useful, but not a free lunch if \(\varepsilon\) is chosen poorly.

## Code

```python
"""
LLMBase — prerequisites demo: cross-entropy, dropout, LayerNorm vs BatchNorm.
Requires: pip install torch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last dimension."""
    return F.softmax(x, dim=-1)


def cross_entropy_from_scratch_nll(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Multi-class CE for integer class labels (same convention as nn.CrossEntropyLoss).

    logits: (N, C) unnormalized scores
    targets: (N,) with values in {0..C-1}
    Returns scalar mean loss.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs[torch.arange(targets.shape[0], device=logits.device), targets]
    return nll.mean()


def compare_with_pytorch() -> None:
    torch.manual_seed(0)
    n, c = 4, 5
    logits = torch.randn(n, c)
    targets = torch.randint(0, c, (n,))

    manual = cross_entropy_from_scratch_nll(logits, targets)
    ce = nn.CrossEntropyLoss()
    ref = ce(logits, targets)

    assert torch.allclose(manual, ref, atol=1e-6, rtol=1e-5)

    print("=== Cross-entropy: manual vs nn.CrossEntropyLoss ===")
    print(f"manual: {manual.item():.8f}")
    print(f"pytorch: {ref.item():.8f}")
    print(f"abs diff: {abs(manual - ref).item():.2e}")


def demo_dropout_train_vs_eval() -> None:
    torch.manual_seed(1)
    p = 0.5
    drop = nn.Dropout(p=p)
    x = torch.ones(2, 6)

    drop.train()
    y_train = drop(x)
    drop.eval()
    y_eval = drop(x)

    print("\n=== Dropout: train vs eval ===")
    print("input (ones):\n", x)
    print("train mode output (random zeros + scale 1/(1-p)):\n", y_train)
    print("eval mode output (identity; dropout inactive):\n", y_eval)


def demo_layer_norm_vs_batch_norm() -> None:
    torch.manual_seed(2)
    # (batch, features) — tiny toy tensor
    x = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 0.0, -1.0], [2.0, 2.0, 2.0]], dtype=torch.float32
    )

    ln = nn.LayerNorm(3, elementwise_affine=True)
    bn = nn.BatchNorm1d(3)

    ln.eval()
    # Use training mode for BatchNorm so statistics come **from this batch**
    # (in eval mode, running means/vars from training would be used instead).
    bn.train()

    with torch.no_grad():
        y_ln = ln(x)
        # BatchNorm1d expects (N, C) for 2D input in PyTorch
        y_bn = bn(x)

    print("\n=== LayerNorm vs BatchNorm (BN in train mode: batch stats) ===")
    print("x:\n", x)
    print("LayerNorm(x) (per-row normalize across features):\n", y_ln)
    print("BatchNorm1d(x) (normalize per feature across batch rows):\n", y_bn)


if __name__ == "__main__":
    compare_with_pytorch()
    demo_dropout_train_vs_eval()
    demo_layer_norm_vs_batch_norm()
```

Running the script should print near-zero difference between manual CE and `nn.CrossEntropyLoss`, show dropout stochasticity in train mode versus identity in eval mode, and contrast LayerNorm’s row-wise normalization with BatchNorm’s column-wise normalization on the same tensor.

**Implementation note (LLM training):** frameworks typically report a **mean** NLL over tokens (and possibly batch), sometimes with **masked** positions excluded from the denominator. Whether you sum or mean is a constant scale shift for fixed sequence length, but it matters when comparing reported losses across codebases—always check reduction (`mean` vs `sum`) and masking.

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Why is cross-entropy the right classification loss?** Expect: MLE / Bernoulli or categorical likelihood → negative log-likelihood; squared loss on labels is not a generative model for discrete outcomes.
    2. **What does minimizing cross-entropy accomplish in distribution space?** Expect: equivalence (for fixed targets) to minimizing \(\mathrm{KL}(p\|q)\) up to an entropy constant; interpretable in bits.
    3. **What is the gradient of binary cross-entropy with respect to logits if you include sigmoid vs if you use logits + log-sum-exp?** Expect: discussion of numerical stability; stable implementations use log-softmax / log-sigmoid formulations.
    4. **Why can MSE be inappropriate for classification even if outputs are in \([0,1]\)?** Expect: mismatch to probabilistic assumptions; vanishing gradients for very wrong confident predictions differ from log-loss behavior.
    5. **Explain dropout’s train/test difference and inverted dropout.** Expect: scaling by \(1/(1-p)\) in training so inference needs no rescaling; ensemble / co-adaptation story.
    6. **BatchNorm vs LayerNorm: what dimensions are normalized, and why LayerNorm in Transformers?** Expect: batch vs feature axes; sequence modeling + batch independence + stable per-token normalization.
    7. **What is the difference between L2 penalty as “weight decay” in SGD versus AdamW?** Expect: decoupling in adaptive methods; why AdamW exists.
    8. **Pre-norm vs post-norm: tradeoffs?** Expect: optimization stability vs representation; empirical preference for pre-norm in deep models.
    9. **What is label smoothing, and what problem does it target?** Expect: overconfidence; calibration; softened targets as KL to a mixture distribution.
    10. **RMSNorm vs LayerNorm—what changes in the normalization computation?** Expect: RMS scaling without mean subtraction; motivation (speed/simplicity) and empirical usage in LLaMA-class models.

!!! interview "Follow-up Probes"
    1. If the vocabulary is 128k, what exactly is the dimension of the softmax vector at one position—and what is minimized for that position?
    2. How does masking (padding) change the effective loss when averaging over tokens?
    3. Why can BatchNorm’s batch statistics be unstable for small batch sizes or distributed training?
    4. What happens to L1 vs L2 near zero weights, and why does that matter for sparsity?
    5. How does weight decay interact with learning rate and gradient clipping in large-scale LM training?

!!! key-phrases "Key Phrases to Use in Interviews"
    1. “**Negative log-likelihood** under a categorical model—MLE for next-token prediction.”
    2. “**Minimizing cross-entropy is minimizing KL** to the target distribution, up to a constant entropy term.”
    3. “**LayerNorm normalizes across features per token**—no batch coupling, ideal for sequences.”
    4. “**AdamW decouples weight decay** from the adaptive preconditioning—this is not the same as naive L2 with Adam.”
    5. “**Dropout is train-time stochastic depth / implicit ensembling**; inverted dropout keeps inference clean.”
    6. “**Pre-norm improves trainability of deep Transformers** by stabilizing residual stream gradients.”
    7. “**Label smoothing** reduces overconfident logits by pulling probability mass off the true class slightly.”

## References

- Goodfellow, Bengio, and Courville, *Deep Learning* (MIT Press) — Part II on regularization; maximum likelihood and cross-entropy as the standard classification objective.
- Murphy, *Probabilistic Machine Learning: An Introduction* — exponential families, NLL, and generative vs discriminative framing.
- Srivastava et al., “Dropout: A Simple Way to Prevent Neural Networks from Overfitting,” *JMLR* (2014).
- Ioffe and Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” *ICML* (2015).
- Ba, Kiros, and Hinton, “Layer Normalization,” *arXiv:1607.06450* (2016).
- Vaswani et al., “Attention Is All You Need,” *NeurIPS* (2017) — original Transformer; compare LayerNorm placement variants in follow-on work.
- Loshchilov and Hutter, “Decoupled Weight Decay Regularization,” *ICLR* (2019) — AdamW.
- Zhang and Sennrich, “Root Mean Square Layer Normalization,” *NeurIPS* (2019) — RMSNorm.
- Szegedy et al., “Rethinking the Inception Architecture for Computer Vision,” *CVPR* (2016) — label smoothing (widely reused in LM training recipes).
- Müller et al., “When Does Label Smoothing Help?” *NeurIPS* (2019) — calibration and teacher-student effects.
