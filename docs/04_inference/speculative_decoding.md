# Speculative Decoding

**Speculative decoding** is the standard **draft–verify** framework for faster **autoregressive** generation without (in the **stochastic** case) changing the **target** distribution.

## Why This Matters for LLMs

**Speculative decoding** is one of the few inference optimizations that can deliver **multiplicative** wall-clock speedups **without changing** the **marginal output distribution** of the target model—when implemented correctly—because it is **not** an approximation to the forward pass; it is a **reorganization** of work between a **cheap proposal** mechanism and a **parallel verification** pass by the **large** model. In bandwidth-bound regimes, each target forward pass moves **billions** of parameters through HBM to emit **one** token; speculative methods ask whether a **small draft model** (or **auxiliary heads**) can propose **\(k\)** tokens such that the **large** model can **verify** all \(k\) **dependencies** in a **single** batched forward that **reuses** prefix **KV** state. Interviewers love this topic because it joins **probability** (acceptance sampling), **algorithms** (draft/verify), and **systems** (KV layout, batching).

A second reason this matters is **quality preservation**. Many “fast decoding” tricks **change** the distribution (heuristic merging, early exit without correction). **Exact** speculative sampling (Chen et al., 2023; Leviathan et al., 2023) ensures that—if you follow the acceptance and resampling rules—the **next accepted token** is drawn **exactly** from the target model’s conditional distribution **\(p(\cdot \mid \text{context})\)**. That distinction is career-relevant: teams can adopt speculation for **latency** without **re-tuning** safety filters that assume **\(p\)**. When a method only preserves **greedy** behavior (common in simpler presentations), it is still useful, but you must label it **greedy–greedy** alignment, not **full distributional** equivalence.

Third, **throughput** gains depend on **acceptance rate** \(\alpha\), **draft cost** \(C_q\), and **target cost** \(C_p\). A **weak** draft that disagrees with the target **wastes** work: you pay for **draft** forwards **plus** **target** verification **without** accepting long prefixes. Production systems therefore **distill** drafts toward **\(p\)**, share **tokenizers**, match **context windows**, and sometimes use **feature-space** drafting (**EAGLE**) to raise \(\alpha\). Understanding **expected accepted length** per round and how it interacts with **GPU occupancy** (parallel verification improves **tensor core** utilization) is how you move from “I heard about speculative decoding” to “here is how we’d prototype it on our stack.”

---

## Core Concepts

### The Core Idea

A **draft** model \(q_\phi\) proposes a **block** of tokens \(\gamma_1,\ldots,\gamma_K\) cheaply (often **autoregressive** under \(q\)). A **target** model \(p_\theta\) evaluates the **true** next-token distributions \(p_\theta(\cdot \mid x, \gamma_{<i})\) for **each** position \(i\) in **one** parallel forward over the **concatenated** sequence (thanks to causal masking and **KV** reuse). An **acceptance** procedure decides how many proposed tokens to keep; on the first **reject**, a **corrected** sample is drawn so that **marginals** match **\(p_\theta\)**.

!!! math-intuition "In Plain English"
    Think **proposal + verify**: the draft is a **guess**; the target is the **judge**. The cleverness is **batching** the judge’s work across **multiple** future positions **simultaneously**, then using **randomized** acceptance so the judge’s final decision is **statistically identical** to having run the target alone.

### Acceptance Probability for a Proposed Token

Consider a **single** proposed token \(y\) drawn from **\(q(\cdot \mid x)\)**. A standard building block uses acceptance probability

\[
a(y) = \min\left(1,\, \frac{p(y \mid x)}{q(y \mid x)}\right).
\]

!!! math-intuition "In Plain English"
    If the draft **overweights** \(y\) relative to the target (\(q > p\)), you **accept** only **sometimes**, with probability \(p/q\). If the draft is **pessimistic** about a token the target likes (\(p \ge q\)), you **always** accept that draw—there is no need to **down-weight** accepted proposals when \(p \ge q\) because the **min** clips at 1.

!!! example "Worked Example: Speculative Decoding Steps (Stochastic)"
    Draft proposes tokens for the continuation: **["The", "cat", "sat", "on"]**. For each position \(i\), compare **target** probability \(p_i = p_\theta(y_i \mid \text{prefix})\) with **draft** probability \(q_i = q_\phi(y_i \mid \text{prefix})\). Acceptance uses **\(r_i = \min(1, p_i/q_i)\)** and a uniform draw \(u_i \sim \mathrm{Unif}(0,1)\); accept if \(u_i \le r_i\) (implementation details chain these carefully; this example illustrates **magnitudes**).

    - **Token "The"**: \(p=0.9\), \(q=0.8\) → \(r=\min(1, 0.9/0.8)=1.0\) → **always accept** this proposal if it was sampled under \(q\) and passes the **joint** procedure’s checks (single-step intuition: **no downscaling**).

    - **Token "cat"**: \(p=0.3\), \(q=0.4\) → \(r=\min(1, 0.3/0.4)=0.75\) → accept with probability **0.75**.

    - **Token "sat"**: \(p=0.05\), \(q=0.3\) → \(r=\min(1, 0.05/0.3)\approx 0.167\) → usually **reject**; when rejecting, the algorithm draws a **replacement** token from a **residual** distribution derived from **\(p\)** and **\(q\)** so the **overall** next-token marginal remains **\(p\)**.

    **Effective throughput intuition**: if average accepted length per round is **~2.2** tokens for the target model **amortized** over verification cost, you approach **~2×** target-limited decoding **when** draft cost is **negligible** and verification is **batched efficiently**.

### Greedy Speculative Decoding (Deterministic)

If **both** draft and target use **greedy** decoding (argmax), verification simplifies: **accept** draft tokens while **each** matches the **target argmax** at that position; on the **first** mismatch, **emit** the **target** argmax instead. This preserves **greedy** target outputs (under exact tie-breaking assumptions).

\[
y_i^{\text{greedy}} = \arg\max_{w} p_\theta(w \mid x, y_{<i}^{\text{accepted}}).
\]

!!! math-intuition "In Plain English"
    **Greedy** speculation is easier to implement and reason about: you are asking whether the draft’s **top-1** choices match the target’s **top-1** choices along a prefix. It does **not**, by itself, guarantee **full** distributional equivalence—only **greedy** alignment.

### Expected Accepted Length (Geometric Heuristic)

Let **independent** Bernoulli approximations suggest each draft token **survives** with probability \(\alpha\). The **expected** number of **leading** accepts **before** failure (capped at \(K\)) satisfies:

\[
\mathbb{E}[L] \approx \sum_{j=1}^{K} \alpha^{j} = \frac{\alpha(1-\alpha^{K})}{1-\alpha}\quad (\alpha<1).
\]

!!! math-intuition "In Plain English"
    **Higher** \(\alpha\) (draft **agrees** with target) **lengthens** accepted runs. Real drafts have **Markov** dependence—this formula is a **pedagogical** guide, not a calibrated production predictor.

!!! example "Worked Example: \(\alpha=0.7\), \(K=8\)"
    \[
    \mathbb{E}[L] \approx \sum_{j=1}^{8} 0.7^{j} = 0.7 \frac{1-0.7^{8}}{1-0.7} \approx 1.96.
    \]

    Even **70%** per-step **agreement** yields **~2** accepted tokens per round on average under this **independent** toy—strong **alignment** or **larger** effective \(K\) helps **amortize** verification.

### Speedup (Rough Accounting)

Let **\(C_p\)** be the target cost for a **verification** forward that processes **\(K\)** draft tokens in parallel (amortizing prefix), and **\(C_q\)** the cost to **draft** those \(K\) tokens. A **rough** speedup factor:

\[
S \approx \frac{\mathbb{E}[L] \cdot \tau_{\text{serial}}}{\tau_{\text{draft}} + \tau_{\text{verify}}},
\]

where \(\tau\) denotes wall-clock time per **round** and \(\mathbb{E}[L]\) is **accepted target tokens per target verification** in expectation.

!!! math-intuition "In Plain English"
    If you **accept** ~3 tokens per target forward and **draft** is **10× cheaper** than **target**, net wins are plausible—but **constants** (attention kernels, **KV** bandwidth) dominate in practice.

### Draft Model Selection

- **Smaller** transformer in the **same family** (e.g., 7B draft vs 70B target).
- **Distillation** toward **\(p_\theta\)** on the **deployment prompt** distribution.
- **Self-speculation**: early layers/heads propose tokens; target verifies (**Medusa**-style).

??? deep-dive "Deep Dive: Medusa and Parallel Prediction Heads"
    **Medusa** attaches **multiple** shallow heads to a **single** backbone to predict **several** future tokens from the **same** hidden state. Verification still uses **\(p_\theta\)** to **approve** candidates—training aligns heads so proposals **match** the target’s **likely** continuations. The **systems** benefit is **avoiding** a **second** full model while keeping a **structured** proposal set.

??? deep-dive "Deep Dive: Feature-Space Drafting (EAGLE-Class)"
    **EAGLE**-style methods draft **features** (next hidden states) rather than **only** discrete tokens, improving **alignment** between proposal and target dynamics. Acceptance analysis still hinges on **how often** the target’s **token-level** tests accept the **proposed** path.

### Mathematical Guarantee (Informal Statement)

Correct **speculative sampling** algorithms ensure that the **sequence** of tokens produced by the **combined** procedure is **distributed identically** to running **target-only** sampling under **\(p_\theta\)**, **assuming** exact arithmetic and exact **\(p,q\)** evaluation. The proof uses **coupling** and **detailed balance**-style arguments at the token level; the practical **invariant** you cite in interviews is: **“no quality loss relative to the target distribution.”**

\[
\text{Law}(\text{output} \mid \text{speculative pipeline}) = \text{Law}(\text{output} \mid \text{target-only pipeline}).
\]

!!! math-intuition "In Plain English"
    This is **not** claiming the draft is **accurate**—it claims the **final** procedure **does not introduce bias** relative to **\(p\)** when implemented **exactly**. **Numerical** differences (FP16 **argmax** ties) can break **greedy** parity in real stacks—teams sometimes standardize on **BF16** for verification.

### Verification Forward: One Matmul, Many Positions

Let **context** length be \(n\) and draft **block** length \(K\). A **single** target forward on length \(n+K\) computes **all** next-token logits needed to **score** each draft token **in parallel**:

\[
\mathbf{z}_{n+i-1} = \mathrm{logits}_\theta\bigl(\cdot \mid x_{\le n+i-1}\bigr),\quad i=1..K.
\]

!!! math-intuition "In Plain English"
    **Causal** attention lets the model **fill** the lower triangle in **one** go: each position’s hidden state depends **only** on **left** context, which includes **draft** tokens when they are **present** in the input sequence. That is why **verification** is **not** the same as **\(K\)** separate forwards from **scratch** without **KV**—the **implementation** reuses **prefix** computation.

!!! example "Worked Example: Indexing Logits for Draft Checks"
    Suppose **context** token IDs have length **\(n=4\)** and draft proposes **\(K=3\)** tokens \(\gamma_1,\gamma_2,\gamma_3\). The **model input** is the **concatenation**:

    \[
    [x_1,x_2,x_3,x_4,\gamma_1,\gamma_2,\gamma_3]\quad(\text{length }7).
    \]

    **In Plain English:** The **concatenation** is what the **target** model sees in **one** forward: **causal** masking ensures \(\gamma_2\) was **not** available when predicting \(\gamma_1\), matching **autoregressive** semantics.

    **Causal LM convention**: at input position **\(j\)** (0-based), logits predict **token at position \(j+1\)**. The **first** draft token \(\gamma_1\) is the **prediction** after consuming **\(x_1..x_4\)**—that is **logits at position \(n-1=3\)**. The **second** draft \(\gamma_2\) is checked at **position \(4\)** after prefix \(\ldots,x_4,\gamma_1\), and so on: **draft \(\gamma_i\)** uses **position \(n-2+i\)** in 0-based indexing (equivalently **\(n-1+i-1\)**). **Sanity**: if **\(i=1\)**, index \(n-1=3\) predicts the **5th** token \(\gamma_1\). **Implementation** tip: **unit-test** your indexing against **teacher-forced** loss on short sequences.

### Rejection and Residual Sampling (Conceptual)

When a proposed token **fails** the randomized test, algorithms **do not** simply fall back to **argmax**—that would **bias** the chain. Instead, one samples from a **residual** distribution constructed from **\(p\)** and **\(q\)** so that **overall** the next-token marginal remains **\(p\)**. A **common** illustration uses **positive** mass:

\[
p'(w) \propto \max\bigl(0,\, p(w) - q(w)\bigr)
\]

(on **support** where needed), followed by **renormalization**—the **exact** discrete construction in papers includes **careful** handling of the **accepted** token case and **coupling** with **uniform** draws.

!!! math-intuition "In Plain English"
    Think of **reject sampling** as **subtracting** the **draft’s** mass from the **target** where the draft **overclaimed** probability for the **proposed** token—then **renormalize** what remains. The **full** coupled chain is longer to implement than **greedy** verification, which is why **production** stacks often ship **greedy** or **approximate** variants first, then add **exact** sampling.

??? deep-dive "Deep Dive: KV Cache Rollback After Partial Accept"
    When **only** a **prefix** \(\gamma_{1..i-1}\) is accepted and position \(i\) **rejects**, the **speculative tail** \(\gamma_{i..K}\) must be **discarded** from **KV** state. Engines **checkpoint** cache **blocks** at the **start** of the round or **recompute** a short **suffix**—the **cost** of **rollback** influences **optimal** \(K\) under **memory** pressure.

??? deep-dive "Deep Dive: Tree Speculation (SpecInfer / Sequoia)"
    **Multiple** draft **branches** can be verified with **batched** attention over **tree**-structured prefixes. **Acceptance** generalizes from **chains** to **paths** in a tree; **throughput** rises when **draft** uncertainty is **multi-modal** (several plausible continuations), but **bookkeeping** and **memory** grow with **branching factor**.

### Comparison to Contrastive Decoding

**Contrastive decoding** forms **logit differences** between **strong** and **weak** models to **shape** outputs—it **changes** the **effective** distribution. **Speculative decoding** **preserves** **\(p\)** when done **exactly**. Do **not** conflate **distribution shaping** with **compute** acceleration.

\[
z_{\text{contrast}} = z_{\text{large}} - \alpha\, z_{\text{small}}.
\]

!!! math-intuition "In Plain English"
    **Contrastive** modifies **logits** to **steer** behavior; **speculative** uses a **small** model as a **proposal** for **faster** **evaluation** of **\(p\)**—orthogonal goals.

---

## Code

The script below is **self-contained** (**PyTorch** only). It defines **two** small causal LMs (**draft** and **target**) over a **tiny** vocabulary, runs **greedy** speculative decoding with **parallel** verification, and demonstrates **one-step** acceptance probabilities **\(\min(1,p/q)\)** using explicit softmax distributions.

```python
"""
speculative_decoding_demo.py — toy draft/target models + greedy speculative decode.
Run: python speculative_decoding_demo.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ToyLMConfig:
    vocab_size: int = 64
    d_model: int = 32
    n_layers: int = 1


class TinyCausalLM(nn.Module):
    """Minimal unidirectional LM: token embeddings + 1-layer causal Transformer block + lm_head."""

    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ln = nn.LayerNorm(cfg.d_model)
        self.wq = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.wo = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (B, L) int64
        returns logits: (B, L, V) — logits at position i predict token at i+1
        """
        x = self.embed(token_ids)
        h = self.ln(x)
        q, k, v = self.wq(h), self.wk(h), self.wv(h)
        L = h.size(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.cfg.d_model**0.5)
        mask = torch.triu(torch.ones(L, L, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = self.wo(torch.matmul(attn, v))
        h2 = h + ctx
        return self.lm_head(h2)


def softmax_row(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)


def greedy_next(logits_last: torch.Tensor) -> int:
    return int(torch.argmax(logits_last, dim=-1).item())


def draft_greedy_chain(model: TinyCausalLM, context: List[int], k: int) -> List[int]:
    out: List[int] = []
    cur = context.copy()
    for _ in range(k):
        ids = torch.tensor([cur], dtype=torch.long)
        logits = model(ids)  # (1, L, V)
        last = logits[0, -1]
        nxt = greedy_next(last)
        out.append(nxt)
        cur.append(nxt)
    return out


def greedy_speculative_round(
    draft: TinyCausalLM,
    target: TinyCausalLM,
    context: List[int],
    k: int,
) -> Tuple[List[int], List[int]]:
    """
    Returns (accepted_prefix, final_context) where final_context extends context by accepted
    tokens plus possibly one correction token when mismatch occurs.
    """
    proposal = draft_greedy_chain(draft, context, k)
    seq = context + proposal
    ids = torch.tensor([seq], dtype=torch.long)
    t_logits = target(ids)[0]  # (Ltot, V)
    n = len(context)
    accepted: List[int] = []
    for i, y in enumerate(proposal):
        pos = n - 1 + i  # logits at pos predict y (next token after prefix)
        p_last = t_logits[pos]
        targ_tok = greedy_next(p_last)
        if y == targ_tok:
            accepted.append(y)
        else:
            # emit target greedy token at mismatch
            return accepted, context + accepted + [targ_tok]
    # all draft tokens matched target greedy — take one more greedy step from full sequence
    last_pos = len(seq) - 1
    extra = greedy_next(t_logits[last_pos])
    return proposal, context + proposal + [extra]


def min_ratio_acceptance(p: torch.Tensor, q: torch.Tensor) -> float:
    """Expected acceptance E_{y~q}[min(1, p_y/q_y)] for finite support (demo)."""
    pq = torch.clamp(p / (q + 1e-12), max=1.0)
    return float((q * pq).sum().item())


def demo() -> None:
    torch.manual_seed(0)
    cfg_d = ToyLMConfig(vocab_size=128, d_model=32, n_layers=1)
    cfg_t = ToyLMConfig(vocab_size=128, d_model=48, n_layers=1)
    draft = TinyCausalLM(cfg_d)
    target = TinyCausalLM(cfg_t)

    context = [1, 2, 3, 4]
    acc, new_ctx = greedy_speculative_round(draft, target, context, k=5)
    print("Accepted draft prefix:", acc)
    print("New context length:", len(new_ctx))

    # Single-step acceptance rate demo
    g = torch.Generator()
    g.manual_seed(1)
    p_logits = torch.randn(256, generator=g)
    q_logits = torch.randn(256, generator=g)
    p = softmax_row(p_logits)
    q = softmax_row(q_logits)
    print("Toy E[min(1,p/q)] under q:", min_ratio_acceptance(p, q))


if __name__ == "__main__":
    demo()
```

!!! math-intuition "In Plain English"
    The **toy** `TinyCausalLM` is **not** a production model—it exists to show **tensor shapes** and **parallel** verification. Real stacks fuse kernels, use **KV caches**, and implement **stochastic** acceptance with **exact** residual sampling per **Chen et al.**

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What problem** does speculative decoding solve, and **why** is the target model often **memory-bandwidth bound** at batch size 1?
    2. **Explain** the **draft/verify** pattern in **one** sentence suitable for a PM.
    3. **How** does **parallel** verification differ from **\(K\)** **serial** target forwards for **\(K\)** proposed tokens?
    4. **What** is **\(\min(1, p/q)\)** trying to enforce in **stochastic** acceptance?
    5. **Why** can a **weak** draft **slow** the system down versus **baseline** decoding?
    6. **Contrast** **greedy** speculative decoding with **exact sampling** preservation—what is each **good** for?
    7. **How** does **Medusa** avoid a **second** full model—what is verified at inference time?
    8. **Name** three **production** footguns: **numerical** parity, **tokenizer** mismatch, **KV** rollback after reject.
    9. **Write** a **rough** expected-accepted-length expression under **independent** per-token acceptance \(\alpha\) and cap \(K\).
    10. **Why** does **distillation** of \(q\) toward **\(p\)** on **realistic** prompts raise **acceptance** \(\alpha\)—often the **highest-ROI** knob before **custom** kernels?

!!! interview "Follow-up Probes"
    - **Probe A**: “If verification batches **long** candidate sequences, what **memory** component spikes first?”
    - **Probe B**: “How does **speculative decoding** interact with **structured** decoding / tool-call grammars?”
    - **Probe C**: “Does speculative decoding **fix** hallucinations? Why or why not?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Speculative decoding amortizes expensive target forwards across multiple tokens per round.**”
    - “**Acceptance sampling can preserve the target distribution exactly—unlike heuristic shortcuts.**”
    - “**Parallel verification evaluates \(p(y_i \mid \text{prefix})\) for all draft positions in one forward.**”
    - “**Acceptance rate and draft cost dominate real-world speedups; weak drafts add overhead.**”
    - “**Greedy speculative matches argmax chains; stochastic variants preserve full sampling.**”

### Whiteboard Summary

\[
\text{Round work} \approx \underbrace{C_q}_{\text{draft chain}} + \underbrace{C_p}_{\text{verify forward on } n+K} \quad\Rightarrow\quad
S \approx \frac{\mathbb{E}[L]}{C_q + C_p} \cdot C_p^{\text{baseline}}
\]

(only a **mnemonic**—constants absorb **KV** **append**, **kernel** **launches**, **batch** **shape**.)

!!! math-intuition "In Plain English"
    - **Draft** proposes a **path**; **target** **batched** forward is the **judge**; **acceptance** keeps **statistics** honest.
    - If **verification** is **not** **much** **cheaper** than **\(K\)** **serial** **targets**, **check** **KV** **reuse** and **kernel** **batching**—those are the **real** **wins**.

---

## References

- Leviathan, Kalman & Matias (2023), *Fast Inference from Transformers via Speculative Decoding* — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- Chen et al. (2023), *Accelerating Large Language Model Decoding with Speculative Sampling* — [arXiv:2302.01318](https://arxiv.org/abs/2302.01318)
- Stern et al. (2018), *Blockwise Parallel Decoding for Deep Autoregressive Models*
- Cai et al. (2024), *Medusa* — [arXiv:2401.10774](https://arxiv.org/abs/2401.10774)
- Li et al. (2024), *EAGLE* — [arXiv:2408.10188](https://arxiv.org/abs/2408.10188)
- Kwon et al. (2023), *PagedAttention* — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- Miao et al., *SpecInfer* — tree-based speculative inference
- Spector & Re, *Accelerating LLM Inference with Staged Speculative Decoding* — staged variants (ecosystem)
- Sun et al., *Speculative Decoding via Early Exiting* — related early-exit ideas (compare/contrast)
- NVIDIA blog — *Speculative Decoding in TensorRT-LLM* — implementation notes (vendor)
- vLLM documentation — *Speculative Decoding* user guide — [https://docs.vllm.ai](https://docs.vllm.ai)

