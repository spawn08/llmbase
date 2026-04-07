# Decoding Strategies

## Why This Matters for LLMs

Decoding strategies are the **policy layer** that sits between a trained language model’s next-token distribution and the **text users actually see**. A 70B-parameter model can be brilliant at estimating probabilities, but greedy decoding may collapse into repetition, beam search may favor bland continuations, and poorly tuned sampling can inject incoherence or unsafe content. In production, **latency**, **determinism**, **diversity**, and **safety** requirements often conflict; the same backbone model may need greedy decoding for structured outputs, nucleus sampling for chat, and constrained decoding for tool calls. Interviewers probe this topic because it is where **probability theory meets systems engineering**: you must reason about logits, temperatures, caches of partial hypotheses, and batching implications without hand-waving.

Second, decoding is not a single knob. **Temperature** rescales logits before softmax; **top-k** and **top-p (nucleus)** truncate or reweight the support of the distribution; **repetition penalties** and **min-p** reshape logits based on history or the shape of the current distribution. These mechanisms **interact**: a high temperature plus a wide top-p can explode variance, while a low temperature plus aggressive repetition penalty can over-suppress frequent words. Understanding these interactions is what separates “I called `model.generate`” from “I can tune a serving stack for a product’s tone and latency budget.” Teams shipping LLM APIs need engineers who can **predict failure modes**—degenerate loops, subtle toxicity spikes, or JSON-invalid outputs—before they reach users.

Third, decoding choices affect **downstream evaluation** and **reproducibility**. Benchmark scores are not comparable unless decoding hyperparameters are fixed and reported; research papers that omit temperature and top-p make replication difficult. For RLHF-aligned models, decoding can shift the effective policy away from the training objective (e.g., excessive randomness undermining refusal behavior). For system-design interviews, you are expected to connect decoding to **KV-cache residency** (long beams), **batching** (diverse temperatures per request), and **hardware** (sampling is cheap, beam search multiplies memory). This page builds the vocabulary and math you need to defend those design decisions crisply.

---

## Core Concepts

### From Logits to Text

At autoregressive step \(t\), the model outputs a **logit vector** \(\mathbf{z}_t \in \mathbb{R}^V\) over vocabulary size \(V\). Temperature-scaled probabilities are

\[
P_t(i) = \frac{\exp(z_{t,i}/T)}{\sum_{j=1}^{V}\exp(z_{t,j}/T)}.
\]

!!! math-intuition "In Plain English"
    **Logits** are unnormalized scores: higher means “more plausible next token” before normalization. **Softmax** turns them into a **probability vector** that sums to 1. **Temperature \(T\)** stretches or compresses differences: dividing logits by \(T\) before the exponential is the same as making the distribution **peaked** (\(T<1\)) or **flat** (\(T>1\)) before sampling.

**Decoding** means turning the sequence of distributions \(\{P_t\}\) into a discrete token sequence \(\{x_t\}\) by **argmax** (deterministic), **sampling** (stochastic), or **search** (maintaining multiple hypotheses).

!!! example "Worked Example: Logits to Probabilities"
    Suppose \(V=4\) and raw logits are \(\mathbf{z} = [2.1, 1.5, 0.3, -0.2]\) with **temperature \(T=1\)**.

    **Step 1 — exponentials** (use natural exp):

    \[
    e^{2.1} \approx 8.166,\quad e^{1.5} \approx 4.482,\quad e^{0.3} \approx 1.350,\quad e^{-0.2} \approx 0.819.
    \]

    **Step 2 — sum**:

    \[
    Z = 8.166 + 4.482 + 1.350 + 0.819 \approx 14.817.
    \]

    **Step 3 — softmax**:

    \[
    P \approx [0.551,\ 0.302,\ 0.091,\ 0.055].
    \]

    **Step 4 — greedy choice**: \(\arg\max_i P(i) = 0\) (first token), probability about **55%**.

    If we instead set **\(T=0.5\)**, we divide logits by 0.5 before softmax (equivalent to **doubling** them): \(\mathbf{z}/T = [4.2, 3.0, 0.6, -0.4]\). The distribution becomes **sharper**—mass concentrates more on token 0—illustrating how \(T<1\) pushes toward **argmax-like** behavior.

### Greedy Decoding

Greedy decoding selects

\[
x_t = \arg\max_{i} P_t(i).
\]

!!! math-intuition "In Plain English"
    Greedy means **no exploration**: at every step you pick the single most likely token. It is **fast** and **deterministic** (given fixed logits), but it can **lock in** suboptimal prefixes because language modeling is **locally** greedy, not **globally** optimal under long-horizon quality.

!!! example "Worked Example: Greedy on Toy Logits"
    **Vocabulary indices**: 0, 1, 2, 3. **One step** logits: \([2.1, 1.5, 0.3, -0.2]\).

    From the previous example, softmax probabilities are approximately \([0.551, 0.302, 0.091, 0.055]\). **Greedy** picks token **0**.

    **Why this can loop**: if token 0 strongly predicts token 0 again (self-loops in bigrams), greedy never “backs up.” Sampling or penalties are common mitigations.

### Beam Search

Beam search maintains \(k\) **partial sequences** (beams). At each step, each beam is expanded by the top candidates from the next-token distribution; the **best \(k\)** overall sequences by **cumulative log-probability** are kept.

A common scoring objective (unnormalized length) maximizes \(\sum_{t} \log P(x_t \mid x_{<t})\). **Length normalization** (dividing by \(|x|^\alpha\)) reduces a bias toward short sequences.

!!! math-intuition "In Plain English"
    Beam search is **limited lookahead**: it explores multiple plausible prefixes in parallel. Larger \(k\) can find **globally** better strings than greedy, but cost grows with **KV-cache footprint** and **memory** because you must keep multiple hypotheses alive.

!!! example "Worked Example: Beam Search with \(k=2\) over 3 Steps"
    **Vocabulary**: \(\{A,B,C,D\}\) indexed \(0..3\). **Beam width** \(k=2\). Assume **deterministic** transition logits so we can compute by hand at a toy scale.

    We specify **log-probabilities** \(\log P(\text{next} \mid \text{prefix})\) for expansions (more convenient than raw logits):

    **Start**: empty prefix `""`.

    - Step 1 candidates (log-probabilities from `""`):  
      \(A: -0.10,\ B: -0.20,\ C: -1.00,\ D: -1.50\).  
      Keep top-2 beams by **total log-prob** (here same as step log-prob): **`A`** (-0.10), **`B`** (-0.20).

    - Step 2 expansions:

      From `A`:  
      \(AA: -0.10 + (-0.05) = -0.15\)  
      \(AB: -0.10 + (-0.30) = -0.40\)  
      \(AC: -0.10 + (-2.00) = -2.10\)  
      \(AD: -0.10 + (-2.50) = -2.60\)

      From `B`:  
      \(BA: -0.20 + (-0.12) = -0.32\)  
      \(BB: -0.20 + (-0.18) = -0.38\)  
      \(BC: -0.20 + (-1.80) = -2.00\)  
      \(BD: -0.20 + (-2.20) = -2.40\)

      Overall top-2 **two-token** prefixes by total log-prob: **`AA`** (-0.15), **`AB`** (-0.40).

    - Step 3 expansions (only from surviving beams `AA` and `AB`):

      From `AA`: best child is \(AAA\) with incremental \(-0.04\) → total \(-0.19\).  
      From `AB`: best child is \(ABA\) with incremental \(-0.06\) → total \(-0.46\).

      Top-2 after step 3: **`AAA`** (-0.19), **`AAB`** if it beats second (suppose \(AAB\) totals \(-0.41\)) → **`AAA`**, **`AAB`**.

    This walkthrough shows the bookkeeping: **each step** merges expansions from **all** beams, sorts by **cumulative score**, and **prunes** to \(k\).

### Temperature

Temperature \(T>0\) scales logits as \(\mathbf{z}/T\) before softmax.

- \(T < 1\): **sharper** distribution (more deterministic).
- \(T = 1\): **unchanged** (matches training if training used \(T=1\) at softmax).
- \(T > 1\): **flatter** distribution (more diverse).

The limit \(T \to 0^+\) behaves like **argmax** in standard implementations (with tie-breaking rules).

!!! math-intuition "In Plain English"
    Temperature controls **how much randomness** you inject **after** the model has spoken. Low \(T\) makes the model act like a **nearly deterministic** policy; high \(T\) increases **entropy** of the next-token choice.

!!! example "Worked Example: Temperature Effect"
    Logits \(\mathbf{z} = [2.0, 1.0, 0.5, 0.3]\). Compute softmax at **\(T \in \{0.5, 1.0, 2.0\}\)**.

    **Case \(T=0.5\)**: scaled logits \([4.0, 2.0, 1.0, 0.6]\).  
    Exps: \(\approx [54.60, 7.39, 2.72, 1.82]\). Sum \(\approx 66.53\).  
    Probs: \(\approx [0.820, 0.111, 0.041, 0.027]\) — **very peaked** on token 0.

    **Case \(T=1.0\)**: logits unchanged.  
    Exps: \(\approx [7.39, 2.72, 1.65, 1.35]\). Sum \(\approx 13.11\).  
    Probs: \(\approx [0.564, 0.208, 0.126, 0.103]\).

    **Case \(T=2.0\)**: scaled logits \([1.0, 0.5, 0.25, 0.15]\).  
    Exps: \(\approx [2.72, 1.65, 1.28, 1.16]\). Sum \(\approx 6.81\).  
    Probs: \(\approx [0.399, 0.242, 0.188, 0.171]\) — **more uniform**.

    **Takeaway**: raising temperature from 0.5 to 2.0 **reduces** the lead of token 0 from ~0.82 to ~0.40.

### Top-k Sampling

Top-k sampling restricts sampling to the **\(k\)** largest probabilities, **renormalizing** over that set.

!!! math-intuition "In Plain English"
    Top-k **caps** how many tokens can be chosen. If \(k\) is small, you avoid **rare** tokens; if \(k\) is large, you allow more diversity. The weakness is **fixed \(k\)** regardless of whether the distribution is **peaked** or **flat**.

!!! example "Worked Example: Top-k with \(k=2\)"
    Probabilities (already softmaxed): \([0.50, 0.30, 0.15, 0.05]\).

    Keep top-2: tokens 0 and 1 with masses 0.50 and 0.30. Renormalize:

    \[
    P'(0)=\frac{0.50}{0.80}=0.625,\quad P'(1)=\frac{0.30}{0.80}=0.375.
    \]

    **In Plain English:** After **masking** the tail, you **re-scale** so the kept entries still form a **valid probability vector**—the two surviving masses **0.50** and **0.30** were **80%** of the original **100%**, so divide by **0.80**.

    Tokens 2 and 3 have **zero** probability under top-2 sampling.

!!! math-intuition "In Plain English"
    The **renormalized** pair \((0.625, 0.375)\) is what you feed to `torch.multinomial` after top-**k** masking: **only** the **two** largest **original** probabilities survive, scaled to sum to **1**.

### Top-p (Nucleus) Sampling

Let \(p \in (0,1]\). Sort tokens by probability **descending**: \(p_{(1)} \ge p_{(2)} \ge \dots\). Choose the **smallest** set \(S\) such that \(\sum_{i \in S} p_{(i)} \ge p\). Renormalize over \(S\).

!!! math-intuition "In Plain English"
    Top-p adapts the **width** of the sampling support. When the model is **confident**, a small set already reaches cumulative mass \(p\). When the model is **uncertain**, you keep **more** tokens automatically—unlike fixed top-k.

!!! example "Worked Example: Top-p with \(p=0.9\)"
    Sorted probabilities:

    \[
    [0.45, 0.25, 0.15, 0.08, 0.04, 0.03].
    \]

    **In Plain English:** Sorting **largest-first** is what makes top-**p** **adaptive**: you peel off **mass** until you cross the **\(p\)** threshold—**not** a fixed **k** slots.

    Cumulative sums:

    \[
    [0.45, 0.70, 0.85, 0.93, 0.97, 1.00].
    \]

    **In Plain English:** **Cumulative** sums answer “how much probability have I **included** so far?” The **first** time you meet or exceed **\(p=0.9\)**, you have **enough** nucleus mass—here the **fourth** token pushes cumulative from **0.85** to **0.93**.

    For \(p=0.9\), stop at cumulative **first** \(\ge 0.9\): that happens at the **fourth** token (0.93). Keep **four** tokens, renormalize dividing by 0.93:

    \[
    [0.45/0.93,\ 0.25/0.93,\ 0.15/0.93,\ 0.08/0.93] \approx [0.484,\ 0.269,\ 0.161,\ 0.086].
    \]

    **In Plain English:** The **0.93** denominator removes the **truncated** tail mass (last two tokens) so the **four** kept probabilities again sum to **1** on the **restricted** support.

!!! math-intuition "In Plain English"
    Nucleus sampling **adapts** the number of kept tokens: here **four** tokens are needed to capture **90%** of the mass; a **sharper** distribution might have stopped after **one** token.

### Repetition Penalty

A common implementation (GPT-2 style) **down-weights** logits for tokens that already appeared. For logits \(\mathbf{z}\), for each previously generated token index \(i\) in a set \(R\), replace

\[
z_i \leftarrow \frac{z_i}{\rho} \quad \text{if } z_i > 0, \qquad z_i \leftarrow z_i \cdot \rho \quad \text{if } z_i \le 0,
\]

with \(\rho > 1\) (e.g., 1.1–1.3).

!!! math-intuition "In Plain English"
    If a token’s logit was **positive** (encouraging), dividing by \(\rho>1\) **pulls it down**. If it was **negative** (discouraging), multiplying by \(\rho\) makes it **more negative**—stronger avoidance. Net effect: **reduce** the chance of re-selecting tokens already used.

!!! example "Worked Example: One-Step Repetition Penalty"
    Vocabulary \(\{a,b,c\}\). Logits \(\mathbf{z}=[1.0, 0.2, -0.5]\). Already generated \(\{a\}\) so apply penalty to index 0 with \(\rho=1.2\).

    Since \(z_0>0\), new \(z_0 = 1.0/1.2 \approx 0.833\). Others unchanged: \([0.833, 0.2, -0.5]\).

    Softmax shifts probability mass away from `a` compared to the unpenalized case.

### Min-p Sampling

Given current distribution \(P\), let \(p_{\max} = \max_i P(i)\). Choose a threshold \(\tau = \texttt{min\_p} \times p_{\max}\) and keep tokens with \(P(i) \ge \tau\), then renormalize.

!!! math-intuition "In Plain English"
    Min-p is **adaptive** like top-p but uses the **peak probability** as a ruler. When the model is very confident (high \(p_{\max}\)), the threshold rises and you may keep **fewer** tail tokens; when flat, you keep more.

!!! example "Worked Example: Min-p"
    Probabilities: \([0.40, 0.25, 0.20, 0.10, 0.05]\). Here \(p_{\max}=0.40\). Let `min_p=0.05`.

    Threshold \(\tau = 0.05 \times 0.40 = 0.02\). All listed probs \(\ge 0.02\), so **no tokens** removed. (Min-p shines when \(p_{\max}\) is tiny—then \(\tau\) is tiny—still typically used with top-p/top-k in pipelines.)

### Combining Strategies

A common **chat** recipe is: **temperature** \(T \approx 0.7\), **top-p** \( \approx 0.9\), **repetition penalty** \(\approx 1.1\). Coding assistants may use **lower temperature** for JSON and **constrained decoding** for grammars (not covered here, but interacts with sampling).

??? deep-dive "Deep Dive: Why Beam Search Can Feel Worse for Open-Ended Text"
    Beam search optimizes **product of token probabilities** (under Markov factorization). Human text under **maximum likelihood** often looks **generic** because high-probability continuations cluster around **safe** n-grams. For translation or summarization with clear targets, beams help; for creative writing, **stochastic** decoding is often preferred.

??? deep-dive "Deep Dive: Entropy and the Effective Support"
    The **entropy** \(H(P)=-\sum_i P(i)\log P(i)\) summarizes how “spread out” the next-token distribution is. Temperature increases \(H\) (usually), while top-k/top-p **truncate** tails, reducing \(H\) but controlling **rare-token risk** (hallucinated facts, unicode glitches).

??? deep-dive "Deep Dive: Batched Requests with Different Decoding"
    Serving systems may pack requests with **different** \(T\), top-p, and penalties. Implementations must ensure **per-request** RNG streams and **per-request** logits post-processing **before** sampling—otherwise batches are not independent.

---

## Code

The following script is **self-contained**: it implements **greedy**, **beam search** (with simple length normalization), **temperature**, **top-k**, **top-p**, **min-p**, and **repetition penalty**, and runs a **toy generation loop** on random logits for demonstration.

```python
"""
decoding_strategies_demo.py — self-contained decoding utilities (PyTorch).
Run: python decoding_strategies_demo.py
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return F.softmax(logits / temperature, dim=-1)


def greedy_step(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def apply_repetition_penalty(
    logits: torch.Tensor, generated: List[int], penalty: float
) -> torch.Tensor:
    if penalty <= 0:
        raise ValueError("penalty must be > 0")
    if not generated:
        return logits
    out = logits.clone()
    for idx in set(generated):
        if out[idx] > 0:
            out[idx] = out[idx] / penalty
        else:
            out[idx] = out[idx] * penalty
    return out


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= logits.numel():
        return logits
    values, _ = torch.topk(logits, k)
    min_keep = values[-1]
    filtered = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)
    return filtered


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum <= p
    if not mask.any():
        return logits
    last_true = int(torch.nonzero(mask, as_tuple=False)[-1].item())
    cutoff_logit = sorted_logits[last_true]
    filtered = torch.where(logits < cutoff_logit, torch.full_like(logits, float("-inf")), logits)
    return filtered


def min_p_filter(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    if not (0.0 <= min_p <= 1.0):
        raise ValueError("min_p must be in [0, 1]")
    pmax = float(torch.max(probs).item())
    thresh = min_p * pmax
    filtered = torch.where(probs < thresh, torch.zeros_like(probs), probs)
    s = float(torch.sum(filtered).item())
    if s <= 0:
        return probs
    return filtered / s


def sample_from_probs(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> int:
    return int(torch.multinomial(probs, 1, replacement=True, generator=generator).item())


def beam_search_step(
    log_probs_history: List[float],
    last_tokens: List[int],
    next_logits: torch.Tensor,
    beam_width: int,
    length_norm_alpha: float = 0.6,
) -> List[Tuple[float, List[int]]]:
    """
    Expand each beam by all vocab tokens (small-V demo only). Returns new beams scored by
    length-normalized log-prob.
    """
    logp_next = F.log_softmax(next_logits, dim=-1)
    vocab = int(logp_next.numel())
    candidates: List[Tuple[float, List[int]]] = []
    for lp_hist, seq in zip(log_probs_history, last_tokens):
        for v in range(vocab):
            new_lp = lp_hist + float(logp_next[v].item())
            new_seq = seq + [v]
            length = len(new_seq)
            score = new_lp / (length**length_norm_alpha)
            candidates.append((score, new_seq))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:beam_width]


def demo() -> None:
    torch.manual_seed(0)
    vocab = 8
    logits = torch.randn(vocab)
    print("Raw logits:", logits.tolist())
    print("Greedy:", greedy_step(logits))

    for T in (0.5, 1.0, 2.0):
        p = softmax_with_temperature(logits, T)
        print(f"Temperature {T}: probs={['%.3f' % x for x in p.tolist()]}")

    k_logits = top_k_filter(logits, k=3)
    print("Top-k filtered logits:", k_logits.tolist())

    p_logits = top_p_filter(logits, p=0.85)
    print("Top-p filtered logits:", p_logits.tolist())

    probs = F.softmax(logits, dim=-1)
    mp = min_p_filter(probs, min_p=0.05)
    print("Min-p renorm probs:", mp.tolist())

    rep = apply_repetition_penalty(logits, generated=[0, 3], penalty=1.15)
    print("Repetition penalty logits:", rep.tolist())

    gen = torch.Generator()
    gen.manual_seed(123)
    for _ in range(3):
        tok = sample_from_probs(F.softmax(logits / 0.8, dim=-1), generator=gen)
        print("Sampled token:", tok)

    # Toy beam on tiny vocab
    tiny = torch.tensor([2.0, 1.0, 0.5, 0.1])
    beams_lp = [0.0, 0.0]
    beams_seq = [0, 1]
    out = beam_search_step(beams_lp, beams_seq, tiny, beam_width=2)
    print("Beam demo:", out)


if __name__ == "__main__":
    demo()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Explain** how temperature changes the softmax distribution and what failure mode very high temperature causes in chat models.
        *Answer:* Dividing logits by \(T\) before softmax is equivalent to sharpening (\(T<1\)) or flattening (\(T>1\)) the distribution: at \(T=0.5\) the top token might hold ~80% mass versus ~56% at \(T=1\), while \(T=2\) spreads mass toward a near-uniform draw. Very high temperature inflates entropy so rare tokens gain probability mass—often producing incoherent hops, formatting glitches, or safety-policy violations because the model samples from a much wider tail than it was tuned for at moderate \(T\).
    2. **Contrast** greedy decoding vs nucleus (top-p) sampling for creative writing vs code generation.
        *Answer:* Greedy is deterministic and fast (one argmax per step) but can lock into repetitive or locally optimal prefixes; it is often preferred for structured outputs where validity beats diversity. Nucleus sampling adapts the candidate set to the current distribution—when the model is confident, only a few tokens matter; when uncertain, it keeps a wider set—so creative writing gets diversity without always fixing \(k\), while code generation often pairs lower temperature or constrained/greedy-style decoding to preserve syntax and API tokens.
    3. **Why** can beam search produce bland text even when it improves perplexity-style scores?
        *Answer:* Beam search maximizes (length-normalized) product of conditional probabilities, which favors high-frequency, “safe” n-grams that sit on the main mass of the training distribution—human-judged quality is not the same objective. Under MLE-trained LMs, the globally high-probability path often reads generic or repetitive because rare but interesting continuations lose in cumulative log-prob to boring ones, even when local perplexity looks good.
    4. **Describe** top-p in terms of cumulative mass on sorted probabilities; what happens as \(p \to 1\)?
        *Answer:* Sort tokens by descending probability, walk the list accumulating mass until the cumulative first reaches or exceeds \(p\), then mask out everything after that cut and renormalize over the kept set—so the sampling support width tracks uncertainty step by step. As \(p \to 1^-\), you eventually include essentially the full vocabulary (up to numerical ties), so behavior approaches full softmax sampling over all tokens unless another filter (top-k, temperature) limits the tail.
    5. **How** does repetition penalty modify logits differently for positive vs negative entries, and why?
        *Answer:* For tokens already in the generated set, positive logits are divided by \(\rho>1\) (pulling mass down) and negative logits are multiplied by \(\rho\) (pushing them more negative), asymmetrically discouraging re-selection whether the model was eager or reluctant. That split avoids a single rule that would wrongly boost inhibited tokens and directly targets the “loop” failure mode where the same high-logit token keeps winning after softmax.
    6. **When** would you prefer top-k alone vs top-p alone in production, and why is top-p often more adaptive?
        *Answer:* Fixed top-k is simple to implement and debug and can cap worst-case tail risk when \(k\) is small (e.g., \(k=50\)), but it keeps \(k\) tokens even when the distribution is extremely peaked—wasting probability mass on junk—or too few when the distribution is flat. Top-p adjusts the support size to the shape of the distribution: a sharp distribution might need only 2–3 tokens to reach \(p=0.9\), while a flat one might need dozens, which is why chat APIs commonly default to top-p (often \(0.9\)–\(0.95\)) with optional top-k as a hard ceiling.
    7. **What** system-level costs does beam width \(k\) impose beyond extra compute?
        *Answer:* Each beam is a separate hypothesis with its own KV cache state (or duplicated prefixes until shared-prefix optimizations), so memory scales roughly with \(k\) times per-sequence KV for divergent suffixes—e.g., going from \(k=1\) to \(k=4\) can mean multiple gigabytes extra HBM for long contexts. Scheduling and batching also worsen: variable-length beams complicate kernel fusion, and worst-case bookkeeping grows with hypothesis count even when FLOPs per step are only ~\(k\times\) forward passes in the naive form.
    8. **How** does min-p relate to the peak probability, and what problem is it trying to mitigate in tail sampling?
        *Answer:* Min-p sets a floor proportional to the current max probability \(p_{\max}\): keep token \(i\) only if \(P(i) \ge \texttt{min\_p} \times p_{\max}\), then renormalize—so the threshold scales with confidence. It trims ultra-low-probability tail tokens that top-p might still admit when the distribution is flat, reducing “random unicode” or rare-token hallucinations without hard-coding a fixed \(k\).
    9. **Why** must decoding hyperparameters be reported alongside benchmark results for reproducibility?
        *Answer:* The reported metric is a joint function of weights and the decoding policy: the same checkpoint can swing GSM8K or human-eval scores by several points when temperature, top-p, top-k, or penalties differ, so numbers are not comparable across papers or products without the full recipe. For stochastic sampling, you also need seed handling and version notes; otherwise “SOTA” claims are not reproducible and A/B tests between vendors are meaningless.
    10. **How** would you configure decoding differently for JSON tool outputs vs open-ended assistant chat?
        *Answer:* JSON and tool schemas benefit from low temperature (e.g., \(0\)–\(0.3\)), greedy or constrained/grammar decoding when available, and sometimes repetition penalty off or mild—maximizing valid syntax and stable key ordering. Open-ended chat typically uses \(T \approx 0.6\)–\(0.9\), top-p \(\approx 0.9\), and tuned repetition penalty to balance warmth and coherence, accepting more entropy because format validity is secondary to natural language quality.

!!! interview "Follow-up Probes"
    - **Probe A**: “If users report ‘the model repeats phrases,’ which knobs do you tune first and why?”
    - **Probe B**: “How do you keep determinism for regression tests while using sampling in production?”
    - **Probe C**: “What interactions do you expect between low temperature and aggressive repetition penalty?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Logits are unnormalized scores; softmax produces a categorical distribution.**”
    - “**Temperature scales logits before softmax; it controls entropy of the next-token draw.**”
    - “**Top-p adapts the support set to uncertainty; top-k uses a fixed cutoff.**”
    - “**Beam search searches multiple hypotheses but can favor generic high-probability text.**”
    - “**Repetition penalty reshapes logits for tokens already generated to reduce loops.**”

---

## References

1. **Holtzman et al., “The Curious Case of Neural Text Degeneration” (ICLR 2020)** — nucleus sampling and degeneration analysis: [arXiv:1904.09751](https://arxiv.org/abs/1904.09751).
2. **Fan et al., “Hierarchical Neural Story Generation” (ACL 2018)** — top-k sampling in neural story generation: [arXiv:1805.04833](https://arxiv.org/abs/1805.04833).
3. **Brown et al., “Language Models are Few-Shot Learners” (NeurIPS 2020)** — GPT-3 decoding and evaluation practices: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165).
4. **Vaswani et al., “Attention Is All You Need” (NeurIPS 2017)** — autoregressive factorization and softmax basics in Transformers: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
5. **Hugging Face `generate` documentation** — practical parameter interactions: [https://huggingface.co/docs/transformers/main_classes/model#transformers.generation.utils.GenerationMixin.generate](https://huggingface.co/docs/transformers/main_classes/model#transformers.generation.utils.GenerationMixin.generate).
