# 1.5 — Information Theory for LLMs

## Intuition

Information theory gives us the **language** to talk about uncertainty, surprise, and model quality. When we say "GPT-4 has lower perplexity," we are making a statement rooted in Shannon's entropy. Three quantities — **entropy**, **cross-entropy**, and **KL divergence** — appear everywhere: in loss functions, in evaluation metrics, and in alignment objectives like DPO.

---

## Core concepts

### Entropy — how much surprise is in a distribution?

For a discrete random variable \(X\) with probability mass function \(p\):

\[
H(X) = -\sum_{x} p(x) \log p(x)
\]

- **Units:** bits if \(\log_2\), nats if \(\ln\).
- **Minimum:** \(H = 0\) when one outcome has probability 1 (no surprise).
- **Maximum:** \(H = \log |\mathcal{X}|\) for a uniform distribution (maximum surprise).

**Example — fair vs. biased coin:**

| Coin | \(P(\text{heads})\) | \(H\) (bits) |
| --- | --- | --- |
| Fair | 0.5 | 1.0 |
| Biased | 0.9 | 0.47 |
| Deterministic | 1.0 | 0.0 |

The biased coin is more predictable, so it has lower entropy.

### Cross-entropy — how well does \(q\) model \(p\)?

If the true distribution is \(p\) but we use model \(q\) to encode symbols:

\[
H(p, q) = -\sum_{x} p(x) \log q(x)
\]

**Key properties:**

- \(H(p, q) \ge H(p)\) — using the wrong distribution always costs extra bits.
- \(H(p, q) = H(p)\) only when \(q = p\).
- **This is the standard LLM training loss.** For next-token prediction with one-hot targets \(p\), cross-entropy reduces to \(-\log q(w^*)\) — the negative log-likelihood of the correct token.

### KL divergence — the gap between two distributions

**Kullback–Leibler divergence** measures how much extra information \(q\) wastes relative to the true \(p\):

\[
D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
\]

**Key properties:**

- \(D_{\text{KL}} \ge 0\) (Gibbs' inequality).
- \(D_{\text{KL}} = 0 \iff p = q\).
- **Not symmetric:** \(D_{\text{KL}}(p \| q) \ne D_{\text{KL}}(q \| p)\) in general.
- **Forward KL** (\(D_{\text{KL}}(p \| q)\)) — mode-covering: penalizes \(q\) for assigning 0 probability where \(p > 0\). Used in maximum likelihood / cross-entropy training.
- **Reverse KL** (\(D_{\text{KL}}(q \| p)\)) — mode-seeking: \(q\) may ignore some modes of \(p\). Used in variational inference and KL penalty in RLHF.

### Connecting the pieces: training ≡ minimizing cross-entropy ≡ minimizing KL

Minimizing cross-entropy \(H(p, q)\) over model parameters \(\theta\) is equivalent to minimizing \(D_{\text{KL}}(p \| q_\theta)\), because \(H(p)\) is a constant with respect to \(\theta\):

\[
\arg\min_\theta H(p, q_\theta) = \arg\min_\theta D_{\text{KL}}(p \| q_\theta)
\]

This is why cross-entropy loss and maximum likelihood estimation (MLE) are the same thing for LM training.

### Perplexity revisited

Perplexity (from Part 1.1) is just the exponentiated cross-entropy:

\[
\text{PPL} = \exp\!\bigl(H(p, q)\bigr) = \exp\!\Bigl(-\frac{1}{T}\sum_{t=1}^{T} \log q(w_t \mid w_{<t})\Bigr)
\]

| Model | PPL on WikiText-103 |
| --- | --- |
| Kneser–Ney 5-gram | ~68 |
| LSTM (Merity, 2018) | ~33 |
| GPT-2 (small) | ~30 |
| GPT-2 (large) | ~18 |

### Mutual information (bonus)

\[
I(X; Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y)) = H(X) - H(X \mid Y)
\]

Measures how much knowing \(Y\) reduces uncertainty about \(X\). Appears in information-theoretic analyses of attention, representation learning, and the information bottleneck.

---

## Code — Computing entropy, cross-entropy, and KL divergence

```python
"""
Entropy, cross-entropy, and KL divergence — computed from scratch and
verified against PyTorch / scipy implementations.
"""
import numpy as np
import torch
import torch.nn.functional as F


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in nats."""
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy H(p, q) in nats. q must be > 0 wherever p > 0."""
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats."""
    return cross_entropy(p, q) - entropy(p)


# ── True distribution vs. two models ──────────────────────────────────
p = np.array([0.4, 0.3, 0.2, 0.1])       # true next-token distribution
q_good = np.array([0.35, 0.30, 0.20, 0.15])  # close model
q_bad = np.array([0.1, 0.1, 0.1, 0.7])       # poor model

print("=== NumPy from-scratch ===")
print(f"H(p)          = {entropy(p):.4f} nats")
print(f"H(p, q_good)  = {cross_entropy(p, q_good):.4f} nats")
print(f"H(p, q_bad)   = {cross_entropy(p, q_bad):.4f} nats")
print(f"KL(p || q_good) = {kl_divergence(p, q_good):.4f} nats")
print(f"KL(p || q_bad)  = {kl_divergence(p, q_bad):.4f} nats")

# ── Verify with PyTorch ────────────────────────────────────────────────
p_t = torch.tensor(p, dtype=torch.float32)
q_good_t = torch.tensor(q_good, dtype=torch.float32)
q_bad_t = torch.tensor(q_bad, dtype=torch.float32)

kl_good_pt = F.kl_div(q_good_t.log(), p_t, reduction="sum").item()
kl_bad_pt = F.kl_div(q_bad_t.log(), p_t, reduction="sum").item()

print("\n=== PyTorch verification ===")
print(f"KL(p || q_good) = {kl_good_pt:.4f}")
print(f"KL(p || q_bad)  = {kl_bad_pt:.4f}")

# ── Cross-entropy as LLM loss ─────────────────────────────────────────
logits = torch.tensor([[2.0, 1.5, 0.8, 0.2]])  # raw model outputs
target = torch.tensor([0])                       # correct token index

ce_loss = F.cross_entropy(logits, target)
ppl = torch.exp(ce_loss)
print(f"\nCross-entropy loss for token 0: {ce_loss.item():.4f}")
print(f"Per-token perplexity: {ppl.item():.2f}")
```

---

## Interview takeaways

1. **Cross-entropy IS the LLM loss** — every time someone says "we trained with cross-entropy," they mean \(-\log q(w^*)\) summed over tokens. Know this cold.
2. **Minimizing cross-entropy = minimizing KL** — because the true entropy \(H(p)\) doesn't depend on model parameters. This equivalence is fundamental.
3. **Forward vs. reverse KL** — forward KL (standard training) is mode-covering; reverse KL (used in RLHF's KL penalty) is mode-seeking. Be able to sketch why and what each behavior looks like for a bimodal target.
4. **Perplexity = exp(cross-entropy)** — a perplexity of 30 means the model is as uncertain as choosing uniformly among 30 tokens. Know rough PPL numbers for key models.
5. **KL is not symmetric** — a common interview question. \(D_{\text{KL}}(p \| q) \ne D_{\text{KL}}(q \| p)\). If pressed, mention Jensen–Shannon divergence as a symmetric alternative.
6. **Information bottleneck** — advanced question. The idea that intermediate representations should compress input while preserving task-relevant information. Ties to representation learning theory.

---

## References

- Shannon (1948), *A Mathematical Theory of Communication*
- Cover & Thomas, *Elements of Information Theory*
- Jurafsky & Martin, SLP Ch. 3 — Language Modeling and Entropy
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (GPT-2 perplexity results)
