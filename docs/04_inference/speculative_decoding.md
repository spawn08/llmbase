# Speculative Decoding

## Why This Matters for LLMs

Large autoregressive models spend inference time in **matrix multiplies** that are often **memory-bandwidth bound** at small batch sizes: each forward pass for one new token moves **billions** of weights through HBM for only **one** additional token of output. **Speculative decoding** asks: can a **small, fast** “draft” model propose **several** candidate tokens, and a **large** “target” model verify them in **parallel**, accepting a **prefix** that matches what the large model would have sampled anyway? When acceptance rates are high, you **amortize** the cost of the big model across **multiple** emitted tokens per **batched** verification step.

This matters in interviews because it is the clean bridge between **probabilistic sampling** and **systems**: you must explain **exact sampling** equivalence (no change to the **marginal output distribution** under standard schemes) versus **heuristic** speculative methods that **do** bias outputs. The classic **draft + rejection** construction (Leviathan, 2023; Chen et al., 2023) uses **coupled** sampling so the **target** distribution is preserved.

A second thread is **tree** and **parallel head** speculation (**Medusa**, **EAGLE**): instead of a second model, add **heads** that predict **multiple future tokens** from the same hidden state, then verify. The math shifts toward **expected accepted length** per round and **parallel** verification batches.

Third, **latency** SLOs in chat often cap **time-to-first-token** separately from **tokens/sec**—speculation helps **throughput** more than **TTFT** unless draft is nearly free (e.g., tiny model on same GPU).

---

## Core Concepts

### Draft + Verifier Paradigm

Let **target** model \(p\) (large) and **draft** model \(q\) (small). At state \(x\) (prompt + generated prefix), naive sampling draws \(y \sim p(\cdot \mid x)\). Speculative decoding instead:

1. Draft model samples **\(K\)** tokens **i.i.d.** from \(q(\cdot \mid x)\) autoregressively (often \(K \in [2,8]\)).
2. Target model **evaluates** conditional probabilities \(p(y_i \mid x, y_{<i})\) for proposed prefix **in parallel** (single batched forward on the concatenated prefix).
3. **Accept/reject** procedure decides how many draft tokens to keep; may **resample** one token from an adjusted distribution when rejecting.

!!! math-intuition "In Plain English"
    - The **draft** proposes a **guess path** cheaply.
    - The **target** checks “would I have gone there?” using **true** \(p\)—in parallel over positions—then statistics decide **match length**.

### Acceptance–Rejection for One Step (Building Block)

Consider sampling a single discrete draw with probabilities \(p_i\) but you can only **evaluate** \(p\) and **propose** from \(q\). A standard **accept–reject** step:

1. Sample \(y \sim q\).
2. Compute acceptance probability \(a = \min\bigl(1,\, p_y / (M q_y)\bigr)\) with envelope \(M \ge \sup_i p_i/q_i\) (discrete case uses specialized bounds).

For **autoregressive** extensions, the **coupled** algorithms in speculative decoding avoid naive global \(M\) by using **token-wise** ratios along the prefix.

### Leviathan / Chen–etal Algorithm (High-Level)

For **greedy** target decoding, the procedure simplifies: **accept** draft tokens while they **match** the argmax of \(p\) at each step; on first mismatch, take the **target’s** argmax instead—**deterministic** speedups without changing **greedy** output (under exact arithmetic).

For **stochastic** target sampling, preserve \(p\) by:

- Defining **acceptance probabilities** using **min** of ratios \(p(y_i)/q(y_i)\) along the chain with careful **renormalization**.
- Drawing **uniform** random numbers to decide acceptance; on failure, **resample** from a **modified** distribution over the vocabulary for **one** step.

!!! math-intuition "In Plain English"
    - **Math property**: output token at each **real** step is **exactly** distributed as **one** draw from \(p(\cdot \mid \text{context})\) when the algorithm is implemented correctly—**no** “approximate ChatGPT.”
    - **Engineering property**: parallel \(p\) evaluations over a **candidate continuation** reduce **wall-clock** vs **serial** \(K\) forwards of the big model.

### Expected Tokens per Target Forward

Let **\(A\)** be the event that the first draft token is accepted (stochastic case). A rough **mean-field** model: if each position **independently** accepts with probability \(\alpha\), the **number** of accepted tokens before failure behaves like a **geometric** stopping time with success parameter related to **mismatch** probability.

Denote **\(L\)** as accepted length per **round**. Then

\[
\mathbb{E}[L] \approx \sum_{k=1}^{K} P(\text{first } k \text{ draft tokens all accepted}).
\]

Under simplifying assumptions (and **greedy** alignment), if draft matches target with probability \(\alpha\) **per token**, then \(\mathbb{E}[L] \approx \frac{1}{1-\alpha}\) **truncated** at \(K\).

!!! math-intuition "In Plain English"
    - **Higher** alignment between \(q\) and \(p\) → **longer** accepted runs → **higher** speedup.
    - If \(q\) is **bad**, most rounds reject immediately → overhead **worse** than baseline.

### Speedup Analysis (Sketch)

Let **\(C_p\)** = cost of one **target** forward (large), **\(C_q\)** = cost of drafting **\(K\)** tokens (small). **Idealized** speedup factor:

\[
S \approx \frac{\mathbb{E}[L] \cdot C_p}{C_q + C_p}
\]

(very rough—real systems count **kernels**, **batching**, **KV** appends). When \(\mathbb{E}[L] \gg 1\) and \(C_q \ll C_p\), \(S\) can approach **\(\mathbb{E}[L]\)** in the best case.

!!! math-intuition "In Plain English"
    - **Draft must be cheap** relative to target—often **2–3 orders** fewer parameters.
    - **Verification** still needs **target** forward on **extended** context—**KV cache** management matters.

### Medusa: Parallel Heads on One Model

**Medusa** adds **multiple decoding heads** on top of standard LM hidden states to predict **\(y_{t+1}, y_{t+2}, \ldots\)** simultaneously. Tree-based verification compares **candidate continuations** against **base** model probabilities.

\[
\hat{y}_{t+j}^{(j)} = \text{head}_j(h_t)
\]

Verification still uses **target** \(p\) to **approve** branches—**no free lunch** if you demand **exact** sampling.

!!! math-intuition "In Plain English"
    - **Pros**: no separate **weights** for a second model—**heads** are small.
    - **Cons**: **training** Medusa heads requires **preference** data or **distillation** so proposals **align** with \(p\).

### Rejection Probability

For a **single** proposed token \(y\) from \(q\), define **acceptance**:

\[
a(y) = \min\left(1,\, \frac{p(y)}{q(y)}\right)
\]

in the **independent** Metropolis–Hastings style (simplified illustration—full AR spec uses **vector** corrections). The **rejection** probability is \(1 - \mathbb{E}_q[a(y)]\).

!!! math-intuition "In Plain English"
    - When \(q\) **underestimates** \(p\) on high-probability tokens, **acceptance** drops—**adaptive** \(q\) (learned draft) helps.

---

## Simulation: Acceptance Length (Toy)

The script below does **not** implement full coupled sampling—it **simulates** **greedy agreement** between draft and target **next-token argmax** on random logits to illustrate **\(\mathbb{E}[L]\)** vs mismatch rate.

```python
"""
Toy simulation: greedy agreement between independent 'draft' and 'target' argmax.
Not full speculative decoding — illustrates acceptance length sensitivity.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def expected_accepted_length(
    vocab_size: int = 32000,
    rounds: int = 2000,
    max_k: int = 8,
    seed: int = 0,
) -> float:
    g = torch.Generator()
    g.manual_seed(seed)
    total = 0
    for _ in range(rounds):
        # random independent logits — unrelated draft/target => low agreement
        accepted = 0
        for _i in range(max_k):
            d_logits = torch.randn(vocab_size, generator=g)
            t_logits = torch.randn(vocab_size, generator=g)
            d_tok = int(torch.argmax(d_logits).item())
            t_tok = int(torch.argmax(t_logits).item())
            if d_tok == t_tok:
                accepted += 1
            else:
                break
        total += accepted
    return total / rounds


def softmax_min_ratio_acceptance(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Single-step acceptance prob min(1, p_i / q_i) under independent sampling — demo."""
    p = F.softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    # expected acceptance if y ~ q: E_q [min(1, p_y / q_y)]
    ratio = torch.clamp(p / (q + 1e-12), max=1.0)
    return float((q * ratio).sum().item())


if __name__ == "__main__":
    e_len = expected_accepted_length()
    print(f"Mean accepted greedy matches (random logits): {e_len:.3f}")

    g = torch.Generator()
    g.manual_seed(1)
    p_l = torch.randn(1000, generator=g)
    q_l = torch.randn(1000, generator=g)
    print("Single-step avg acceptance (toy ratio):", softmax_min_ratio_acceptance(p_l, q_l))
```

!!! math-intuition "In Plain English"
    - With **random** unrelated logits, agreement is **tiny**—real draft models are **correlated** with target (**distillation**, shared tokenizer, similar data).

### Stochastic Acceptance Math (One-Round Sketch)

Let draft propose \(y \sim q(\cdot \mid x)\). A correct **Metropolis–Hastings** style step uses acceptance

\[
\alpha(y) = \min\left(1,\, \frac{p(y\mid x)}{q(y\mid x)} \cdot \frac{q(x \mid \ldots)}{p(x \mid \ldots)}\right)
\]

for **reverse** proposals; the full **autoregressive** coupled sampler in Chen et al. chains **multiple** tokens with **careful** joint ratios so the **stationary** distribution remains \(p\). Interview takeaway: **ratios** \(p/q\) are the **sufficient statistics** for **whether** draft is **trusted**.

### Blockwise Parallel Decoding (Historical Context)

**Stern et al. (2018)** observe that certain **autoregressive** models admit **parallel** prediction of **blocks** when auxiliary networks predict **multiple** steps; verification still ties back to **teacher forcing** objectives. Modern speculative decoding can be seen as **learned** block proposals with **exact** \(p\) **checks**.

### EAGLE and Feature-Space Drafting

Recent **EAGLE**-class methods draft in **hidden feature** space (predicting **next** hidden states) then map through the **LM head**, improving **alignment** between draft and target compared to **token-only** small LMs. The **acceptance** analysis still reduces to **how often** the **large** model’s **next-token** distribution agrees with the **proposal path**—but **feature** drafting raises **\(\mathbb{E}[L]\)** empirically.

### Parallel Verification and Batching

When verifying **\(K\)** draft tokens, the target forward can **concatenate** positions so **attention** is **one** batched matmul over the **extended** sequence (subject to **causal** masks). **GPU utilization** improves vs **\(K\)** serial forwards. The **cost model**:

\[
T_{\text{verify}} \approx T_{\text{forward}}(L + K) - T_{\text{forward}}(L)
\]

which is **sublinear** in \(K\) compared to \(K\) **independent** forwards from scratch **without** KV reuse—**amortized** prefix in cache.

### Failure Modes

- **Draft too weak**: \(\mathbb{E}[L] \approx 0\) → extra **\(q\)** passes + **same** \(p\) → **slower** than baseline.
- **Draft misaligned tokenizer**: **spurious** rejections even when semantics match—**shared** BPE is mandatory.
- **Numerical** mismatch: FP16 **argmax** differences between machines can break **greedy** equivalence—production stacks sometimes **force** BF16 for verification parity.

### Relationship to Contrastive Decoding (Clarification)

**Contrastive decoding** subtracts **amateur** logits from **expert** logits to **sharpen** reasoning—it is **not** speculative decoding. Do **not** conflate **distribution shaping** with **compute** acceleration.

### Expected Tokens per Round — Geometric Heuristic

Assume **independent** Bernoulli trials with **accept** probability \(\alpha\) for each draft token **conditional** on all previous accepts. Let **\(L\)** be the **number** of leading accepted tokens before first failure (cap at \(K\)). Then \(\mathbb{P}(L \ge j) = \alpha^j\) for \(j \le K\), and

\[
\mathbb{E}[L] = \sum_{j=1}^{K} \mathbb{P}(L \ge j) = \sum_{j=1}^{K} \alpha^j = \frac{\alpha(1-\alpha^{K})}{1-\alpha}.
\]

As \(K \to \infty\), \(\mathbb{E}[L] \to \frac{\alpha}{1-\alpha}\) (truncated geometric series).

!!! math-intuition "In Plain English"
    - **\(\alpha = 0.8\)** → mean accepted prefix \(\approx 4\) if \(K\) large—if **target** verification is **cheap** enough vs **serial**, wall-clock wins.
    - Real drafts have **Markov** dependence—this formula is **pedagogical**, not calibrated.

### Draft Model Training Strategies

- **Distillation**: train \(q_\phi\) to minimize \(\mathrm{KL}\bigl(p(\cdot \mid x) \,\|\, q_\phi(\cdot \mid x)\bigr)\) on prompts from the **deployment** distribution.
- **Shared tokenizer + vocabulary**: ensures **token-level** comparisons are **well-defined**.
- **Depth–width trade**: tiny Transformers (few layers, same \(d_{\text{model}}\)) often beat **RNN** drafts for **\(\alpha\)**.

### Verification Latency vs Throughput

**Latency-sensitive** chat APIs may **disable** speculation under high **concurrency** because **KV** memory **fragmentation** and **queueing** dominate—**speedups** are **workload dependent**. Always report **p50/p95** latency separately from **tokens/sec**.

### Pseudocode: Greedy Speculative Chain (Educational)

```text
Given: draft q, target p (greedy / argmax), prompt x
Initialize: accepted prefix = x
loop until stop:
    # 1) Draft proposes K tokens greedily under q
    y[1:K] = greedy_chain_q(x, K)
    # 2) Target evaluates p(y[i] | x, y[<i]) for all i in parallel
    logits_stack = p.forward_parallel(x concatenated with draft positions)
    # 3) Find first index where argmax_p != y[i]
    i_star = first_mismatch(logits_stack, y)
    if i_star is None:
        append y[1:K] to output; continue
    else:
        append y[1:i_star]  # accepted shared prefix
        append argmax_p at position i_star  # correction token
        # KV caches must roll back speculative tail after i_star in real engines
```

!!! math-intuition "In Plain English"
    - Real engines **reuse** **KV** for **shared** prefix and **discard** **wrong** tail—**cache rollback** is subtle engineering.

### Alternative: N-gram / Retrieval Drafts

Some systems use **static** **n-gram** tables or **retrieval** over corpus to propose continuations—**acceptance** still uses **\(p\)**. This avoids a **second neural** draft but only helps when **text** is **memorized** or **template**-like.

### Summary Formula Sheet

| Quantity | Symbol | Role |
|----------|--------|------|
| Draft tokens per round | \(K\) | Max speculation depth |
| Per-token accept prob | \(\alpha\) | Drives \(\mathbb{E}[L]\) |
| Target cost | \(C_p\) | Big model forward |
| Draft cost | \(C_q\) | Small model chain |
| Idealized speedup | \(S\) | \(\approx \mathbb{E}[L] \cdot C_p / (C_q + C_p)\) (rough) |

### Worked Micro-Example: Greedy Agreement Length

Suppose (unrealistically) each draft token matches the greedy **target** token with probability **0.7**, **independently**. Then \(\mathbb{P}(L \ge j) = 0.7^j\) until cap \(K\). With \(K=8\):

\[
\mathbb{E}[L] = \sum_{j=1}^{8} 0.7^j = 0.7 \frac{1 - 0.7^8}{1 - 0.7} \approx 1.96.
\]

So even **decent** 70% **per-token** match yields **~2** accepted tokens per round on average—speedups require **stronger** alignment than intuition suggests, or **higher** \(K\) with **cheap** draft.

### Lookahead Decoding (Name Collision Warning)

Some vendors say **“lookahead decoding”** for **speculative** stacks; others reserve it for **different** parallel schemes. In interviews, **define terms**: **speculative sampling** (exact \(p\)) vs **heuristic** **draft** methods.

### Hardware Co-design

**Tensor cores** favor **batched** verification (large \(B\))—speculative decoding increases **micro-batch** **tensor** sizes on the **target** forward, improving **utilization** vs many **single-token** forwards. This **secondary** speedup is **not** captured by **FLOP-only** models.

### Safety and Alignment Note

**Exact** sampling preservation applies to the **target** distribution \(p\). If \(p\) itself is **unsafe**, speculative decoding **does not** fix **harm**—it only preserves **statistics**. Some **filtered** decoding pipelines **change** \(p\) with **constraints**; speculative layers must be **re-derived** against the **constrained** distribution.

### Token Trees and Multiple Candidates

Advanced implementations (**SpecInfer**, **Sequoia**) maintain **trees** of **draft** continuations and **verify** **multiple** branches with **batched** attention—**math** is still **accept/reject** on **paths**, but **bookkeeping** explodes. Interview level: **know** trees improve **\(\mathbb{E}[L]\)** when **draft** uncertainty is **multi-modal**.

### Coupling View (Informal)

Let **\(X \sim p\)** be the **ideal** next token. Speculative algorithms construct a **coupling** between a **proposal** chain and **\(p\)** so that **marginals** match **exactly** while **joint** processes **share randomness** to maximize **acceptance**. The **maximal coupling** intuition: if \(p\) and \(q\) overlap a lot, you can **often** pick the **same** \(X\) from both—speculative decoding operationalizes this at **sequence** level.

### Practical Hyperparameters

| Hyperparameter | Typical range | Effect |
|----------------|---------------|--------|
| \(K\) | 2–8 | Larger \(K\) raises **verify** cost linearly |
| Draft size | 7M–1B params | Smaller → faster **\(C_q\)** but lower \(\alpha\) |
| Precision | FP16/BF16 | Must **match** target for **greedy** parity |
| Batch verify | Enabled | Improves **GPU** occupancy |

### Why Independent Draft Tokens Are Easier to Analyze

If draft tokens were **independent** of context (absurd), acceptance at each step would factorize. Real **autoregressive** \(q(y_{1:K}\mid x) = \prod_i q(y_i \mid x, y_{<i})\) introduces **Markov** structure: **mistakes** early **invalidate** later **positions** even if local \(q\) is **sharp**. This is why **teacher-forced** draft **chains** are standard—**joint** \(q\) over the **prefix** matches **how** models **score** continuations.

### Reading the Original Papers

When preparing for **onsite** loops, re-derive **Algorithm 1** from Chen et al. on paper: identify where **uniform** random variables enter, what happens on **reject**, and how **resampling** preserves **\(p\)**. The exercise takes **20 minutes** and **locks in** the **sampling** story.

!!! math-intuition "In Plain English"
    - If you remember **only** one sentence: **speculative decoding trades cheap sequential proposals for fewer expensive target forwards, while rejection sampling keeps the target distribution exact.**
    - If you remember **two** numbers: **draft** cost \(C_q\) and **accept** probability \(\alpha\)—they dominate **real** speedups.
    - **Medusa** swaps a **second model** for **extra heads**—verification still uses **\(p\)**, but **proposals** come from **parallel** classifiers on **\(h_t\)**.
    - **EAGLE**-style **feature** drafting targets **higher** \(\alpha\) by aligning **hidden** dynamics—not just **surface** tokens.

---

*The next page on continuous batching explains how serving engines schedule many sequences so that **verification** batches stay **large** without wasting KV memory.*

---

## Interview Takeaways

- **Speculative decoding** uses a **cheap draft** + **parallel verification** by **target** \(p\) to emit **multiple** tokens per **target** forward while preserving **exact** sampling under correct algorithms.
- **Expected speedup** grows with **acceptance rate** and **cheap** \(C_q\); **bad** drafts add **overhead**.
- **Greedy** speculative variants compare **argmax** chains—simpler to reason about, **no** distribution guarantee needed beyond **matching** greedy behavior.
- **Medusa / tree** methods trade **extra heads** and **verification** logic for **avoiding** a second full model.
- **KV caches** must support **batched** verification forwards over **candidate** prefixes—**memory** spikes are a common **production** footgun.
- Always distinguish **exact** algorithms from **heuristic** speedups that **change** distributions.

## References

- Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding* — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Stern et al. (2018), *Blockwise Parallel Decoding for Deep Autoregressive Models* — historical parallel decoding
- Cai et al. (2024), *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Miao et al., *SpecInfer* — tree-based speculative inference
- Sun et al., *EAGLE* — [arXiv:2408.10188](https://arxiv.org/abs/2408.10188) — feature-space speculative sampling
