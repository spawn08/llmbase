# Decoding Strategies

## Why This Matters for LLMs

Language models define a **conditional distribution** \(P(w_{t+1} \mid w_{\le t})\) over a vocabulary at every step. Training minimizes cross-entropy against **human-written** continuations, but at inference **you choose a policy** that maps logits to the next token: greedy argmax, beam search, or a stochastic sampler (top-\(k\), top-\(p\), temperature). That choice changes **fluency**, **factuality**, **diversity**, and **latency**. Interviewers expect you to explain not only *what* each method does, but *why* it corresponds to manipulating the same softmax family—often with a one-line formula on the whiteboard.

A second reason decoding matters is **evaluation**. Perplexity is computed with teacher forcing and often **greedy** or **full-vocab** softmax; human preference and downstream task scores respond to **sampling temperature** and **truncation** (top-\(p\)). Comparing two models without fixing decoding hyperparameters is an apples-to-oranges mistake that shows up in both ML research and production A/B tests.

Third, **systems** people care because beam search multiplies compute by beam width, while top-\(p\) sampling needs **sorting** or **partial top-\(k\)** tricks on huge vocabularies. Repetition penalties touch logits **before** softmax and interact badly with **EOS** tokens if implemented naively. Understanding the math keeps you from shipping a “fast” sampler that silently biases rare tokens or breaks determinism when you need it.

---

## Core Concepts

### Autoregressive Factorization and Logits

Let vocabulary size be \(V\). At step \(t\), the model outputs logits \(\boldsymbol{\ell}_t \in \mathbb{R}^V\). The **next-token distribution** is

\[
p_i = \frac{\exp(\ell_i)}{\sum_{j=1}^{V}\exp(\ell_j)} = \mathrm{softmax}(\boldsymbol{\ell})_i.
\]

!!! math-intuition "In Plain English"
    - **Logits** are unnormalized scores—bigger means “more plausible” before normalization.
    - **Softmax** forces a **probability vector**: nonnegative entries summing to \(1\).
    - Decoding = pick an index \(i\) according to some rule on \(\boldsymbol{\ell}\) or on a **modified** \(\boldsymbol{\ell}'\).

### Greedy Decoding

**Greedy** decoding selects the single most likely token each step:

\[
w_{t+1} = \arg\max_{i} \, p_i = \arg\max_{i} \, \ell_i.
\]

!!! math-intuition "In Plain English"
    - Locally optimal, **not** globally optimal over full sequences—beam search tries to fix that for small search spaces.
    - Deterministic (given fixed model and no numerical nondeterminism): good for reproducible baselines, often **repetitive** for open-ended generation.

### Beam Search

Beam search keeps the top \(B\) **partial hypotheses** (beams) ranked by **aggregated log-probability**. For length-\(t\) prefix \(w_{1:t}^{(b)}\) in beam \(b\),

\[
\log P(w_{1:t}^{(b)}) = \sum_{s=1}^{t} \log P\big(w_s^{(b)} \mid w_{<s}^{(b)}\big).
\]

Each step expands each beam by all (or top-\(K\) locally pruned) vocabulary entries, scores new prefixes, and keeps the best \(B\) distinct continuations.

**Length normalization** avoids favoring short strings when raw log-probs are compared. A common scoring objective for a completed hypothesis of length \(|y|\) is

\[
\text{score}(y) = \frac{1}{|y|^{\alpha}} \sum_{s=1}^{|y|} \log P(y_s \mid y_{<s}),
\]

with **length penalty** exponent \(\alpha \in [0,1]\) (implementation variants exist; the key idea is **normalize** cumulative log-prob by length).

!!! math-intuition "In Plain English"
    - **Beam width** \(B\): more hypotheses explored—better NMT-style quality sometimes, **much** more compute.
    - **Length penalty**: without it, beams that **end early** (EOS) can look artificially probable because fewer negative log terms are summed.

### Temperature Scaling

Apply temperature \(\tau > 0\) **before** softmax:

\[
p_i(\tau) = \frac{\exp(\ell_i / \tau)}{\sum_j \exp(\ell_j / \tau)}.
\]

As \(\tau \to 0^+\), \(p(\tau)\) concentrates on \(\arg\max_i \ell_i\) (greedy in the limit). As \(\tau \to \infty\), \(p(\tau)\) approaches **uniform** over the support.

!!! math-intuition "In Plain English"
    - **Low** \(\tau\): sharper, more **deterministic** samples—often coherent but repetitive.
    - **High** \(\tau\): **flat** distribution—more diverse, higher risk of nonsense or hallucinations.

### Top-\(k\) Sampling

Let \(\pi\) be the sorted probabilities in **descending** order \(p_{(1)} \ge p_{(2)} \ge \cdots\). **Top-\(k\)** keeps indices \(\{ (1), \ldots, (k) \}\), **zeroes** the rest, and **renormalizes**:

\[
q_i \propto \begin{cases}
p_i & \text{if } i \text{ among top-}k \\
0 & \text{otherwise.}
\end{cases}
\]

Sample \(w \sim q\).

!!! math-intuition "In Plain English"
    - **Never** samples tokens outside the \(k\) most likely—cuts off the long tail.
    - If the model is **overconfident**, top-\(k\) still allows only a small set—may feel **stiff** unless \(k\) is large.

### Top-\(p\) (Nucleus) Sampling

(Holtzman et al., 2020.) Choose smallest set \(V_p \subseteq \{1,\ldots,V\}\) such that

\[
\sum_{i \in V_p} p_{(i)} \ge p_{\text{nucleus}},
\]

where \(p_{(i)}\) are probabilities sorted descending. Renormalize over \(V_p\) and sample.

!!! math-intuition "In Plain English"
    - **Adaptive** width: easy steps (peaked \(p\)) use a **small** nucleus; uncertain steps automatically include **more** tokens.
    - Typical \(p_{\text{nucleus}} \in [0.9, 0.98]\) in practice.

### Repetition and Frequency Penalties

**Repetition penalty** (Keskar-style; common in HF) scales logits of **already generated** tokens. Let \(S\) be the set of token ids that appeared in the context. A simple multiplicative form:

\[
\ell'_i = \begin{cases}
\ell_i / \theta & \text{if } i \in S \\
\ell_i & \text{otherwise,}
\end{cases}
\quad \theta > 1.
\]

**Frequency penalty** subtracts a term proportional to how often token \(i\) occurred (count \(c_i\)):

\[
\ell'_i = \ell_i - \beta \cdot c_i, \quad \beta \ge 0.
\]

!!! math-intuition "In Plain English"
    - Both **push down** logits for tokens the model already overused—reduces **degenerate loops**.
    - Too aggressive penalties can block **necessary** repeated function words or proper nouns; **EOS** handling must be careful so the model can still stop.

### Probability View: Sampling as Inverse Transform (Conceptual)

For discrete distributions, sampling can be implemented via **Gumbel-max** trick or **categorical** sampling from \(q\). In libraries, **reparameterization** is not used for discrete choices; instead **inverse CDF** or **alias** methods matter for speed at large \(V\).

---

## From-Scratch Implementations (PyTorch)

The following module implements **log-softmax stabilization**, **greedy**, **beam** (with optional length normalization), **top-\(k\)**, **top-\(p\)**, **temperature**, and **repetition penalty**, using only PyTorch—suitable for reading alongside production `transformers` `LogitsProcessor` stacks.

```python
"""
Decoding strategies from scratch on GPU tensors.
Requires: torch >= 2.0
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError("temperature must be positive")
    return logits / tau


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits
    out = logits.clone()
    # Unique tokens only (common variant)
    for tid in set(generated_ids):
        if out[tid] > 0:
            out[tid] /= penalty
        else:
            out[tid] *= penalty
    return out


def greedy_step(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits).item())


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return logits
    v, _ = torch.topk(logits, k)
    cutoff = v[-1]
    filtered = logits.clone()
    filtered[filtered < cutoff] = float("-inf")
    return filtered


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) on 1D logits."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    # Remove tokens with cumulative mass > p (keep first that exceeds)
    mask = cumsum > p
    if mask.any():
        # shift: keep at least one token
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    # Scatter back
    out = torch.full_like(logits, float("-inf"))
    out.scatter_(0, sorted_idx, sorted_logits)
    return out


def sample_from_logits(logits: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@dataclass
class BeamHypothesis:
    tokens: List[int]
    log_prob: float
    finished: bool


def beam_search_step(
    logits: torch.Tensor,
    beams: List[BeamHypothesis],
    beam_width: int,
    eos_id: Optional[int],
    length_penalty_alpha: float = 0.6,
) -> List[BeamHypothesis]:
    """
    One expansion step: each beam x vocab candidate, keep top `beam_width` by
    normalized log-prob (length_penalty_alpha > 0 applies (1/L)^alpha factor on log-prob).
    """
    vocab = logits.numel()
    log_probs = F.log_softmax(logits, dim=-1)  # (V,)
    candidates: List[BeamHypothesis] = []

    for b in beams:
        if b.finished:
            candidates.append(b)
            continue
        # For simplicity full vocab — production uses top-k local pruning
        for tok in range(vocab):
            lp = b.log_prob + float(log_probs[tok].item())
            new_toks = b.tokens + [tok]
            finished = eos_id is not None and tok == eos_id
            L = len(new_toks)
            if length_penalty_alpha > 0:
                score = lp / (L ** length_penalty_alpha)
            else:
                score = lp
            candidates.append(BeamHypothesis(new_toks, lp, finished))

    candidates.sort(key=lambda h: -(h.log_prob / (len(h.tokens) ** length_penalty_alpha)))
    out: List[BeamHypothesis] = []
    seen = set()
    for h in candidates:
        key = tuple(h.tokens)
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
        if len(out) >= beam_width:
            break
    return out


def decode_autoregressive_mock(
    max_steps: int = 20,
    vocab_size: int = 256,
    seed: int = 0,
) -> None:
    """Toy loop: random 'logits' to exercise APIs (replace with real LM)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    generated: List[int] = []

    for _ in range(max_steps):
        logits = torch.randn(vocab_size, generator=g)
        logits = apply_temperature(logits, tau=0.8)
        logits = apply_repetition_penalty(logits, generated, penalty=1.15)
        logits = top_k_filter(logits, k=40)
        logits = top_p_filter(logits, p=0.95)
        tok = sample_from_logits(logits, generator=g)
        generated.append(tok)
        if tok == 0:  # pretend EOS
            break
    print("sampled ids:", generated[:10], "...")


if __name__ == "__main__":
    decode_autoregressive_mock()
```

!!! math-intuition "In Plain English"
    - Production code uses **batched** logits, **FP16**, **vocab slicing**, and **FlashAttention**—the **math** above is unchanged.
    - **Beam** on full vocabulary is \(O(B \cdot V)\) per step—real systems **prune** expansions heavily.

### Entropy and “Randomness Budget”

The **Shannon entropy** of the next-token distribution is

\[
H(p) = -\sum_{i=1}^{V} p_i \log p_i \quad \text{(nats).}
\]

Temperature **monotonically** affects typical entropy: very peaked distributions have **low** \(H\); near-uniform have \(H \approx \log V\).

!!! math-intuition "In Plain English"
    - Interview framing: **temperature** controls how much of the **entropy budget** you spend per step—high entropy → diverse but **risky**; low entropy → safe but **boring**.
    - **Top-\(p\)** dynamically **caps** the support so you do not sample from the **junk tail** where cumulative model mass is tiny but raw vocabulary is enormous.

### Worked Micro-Example: Three Tokens

Suppose at some step \(V=3\) and logits are \(\boldsymbol{\ell}=(2.0,\, 1.0,\, 0.0)\). Then

\[
p = \mathrm{softmax}(\boldsymbol{\ell}) \approx (0.665,\, 0.245,\, 0.090).
\]

- **Greedy** picks token \(1\) (index \(0\)).
- **Temperature \(\tau=0.5\)** sharpens: \(\mathrm{softmax}((4,2,0))\) puts **more** mass on token \(1\).
- **Temperature \(\tau=2\)** flattens toward uniform.
- **Top-\(k=1\)** is **identical** to greedy here (unless ties).
- **Top-\(p=0.9\)** includes tokens \(\{1,2\}\) because \(0.665 < 0.9\) but \(0.665+0.245 \ge 0.9\); renormalize over two tokens.

This toy case shows **why** nucleus sampling is **adaptive**: if the first mass were \(0.96\), nucleus might keep **only** one token.

### Stochastic Beam and Diversity (Pointers)

Classical beam keeps **deterministic** best partial hypotheses; research systems sometimes inject **stochastic** beams or **diverse** beam objectives (penalize n-gram overlap between beams). You do not need to memorize full objectives, but know **why** plain beam **under-diversifies** for open-ended chat: all beams collapse to **similar** high-probability continuations.

### Logit Processing Order (Production Pitfall)

Typical **order** on each step:

1. **Gather** logits for last position (shape \([V]\) or \([1,1,V]\)).
2. **Processors**: forbid bad tokens, force grammar, **down-weight** repeats (`LogitsProcessor` list).
3. **Temperature** scaling.
4. **Top-\(k\) / top-\(p\)** masking.
5. **Softmax** + **sample** (or **argmax**).

Changing the order changes semantics—**temperature before vs after** repetition penalty is **not** interchangeable.

---

## Interview Takeaways

- **Softmax + temperature** is one family; decoding is **policy design** on top of the same logits.
- **Greedy** is deterministic and fast; **beam** improves local search at **linear-in-\(B\)** cost; **stochastic** methods trade **diversity** for **risk**.
- **Top-\(p\)** adapts cutoff mass to **uncertainty**; **top-\(k\)** uses a **fixed** width—know when each misbehaves (flat vs peaked distributions).
- **Length normalization** in beam avoids **short-sentence bias**; exact formula varies by toolkit—know the **purpose**.
- **Repetition penalty** alters logits **nonlinearly** through softmax—too strong hurts **natural** repetition and **EOS**.
- At **large \(V\)**, partial sorts and **approximate** top-\(k\) matter for **latency**—connect algorithms to **serving** SLAs.

## References

- Holtzman et al. (2020), *The Curious Case of Neural Text Degeneration* — [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
- Keskar et al. (2019), *CTRL: A Conditional Transformer Language Model* — repetition control
- Fan et al. (2018), *Hierarchical Neural Story Generation* — top-\(k\) sampling in context
- Hochreiter & Bengio (discussion) — softmax temperature in RNN LMs (historical)
- [Hugging Face: GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation) — practical hyperparameter defaults
