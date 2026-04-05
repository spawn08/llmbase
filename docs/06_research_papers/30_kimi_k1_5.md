# Kimi k1.5: Scaling Reinforcement Learning with LLMs
**Authors:** Moonshot AI (Kimi Team) &nbsp;|&nbsp; **Year:** 2025 &nbsp;|&nbsp; **Venue:** arXiv &nbsp;|&nbsp; **Link:** [arXiv:2501.12599](https://arxiv.org/abs/2501.12599)
---
## TL;DR
Kimi k1.5 matches **OpenAI o1** on **AIME (77.5)** and **MATH-500 (96.2)** **without** Monte Carlo Tree Search (MCTS), **without** a learned value function, and **without** process reward models. The central idea is to **scale the RL policy’s context window to 128K tokens** so that planning, reflection, and self-correction can unfold **inside one trajectory**—earlier reasoning stays in context for re-reading and revision. Three practical ingredients: **long-context scaling**, **KL-regularized policy optimization via online mirror descent** (more stable than naive long-sequence updates), and **partial rollout reuse** (cache long prefixes, only generate novel suffixes) to make 128K-token training affordable.

**Benchmarks (reported):** **AIME ≈ 77.5**, **MATH-500 ≈ 96.2**—numbers you can cite as **o1-class** ballpark when discussing **open vs closed** reasoning stacks. The headline is not only **accuracy** but **how**: **single-trajectory RL** at **long horizon** rather than **explicit search** or **dense process supervision**.

**One-sentence pitch:** Treat **context length** as a **first-class scaling axis** for **RL reasoning**, not only **model width/depth**.

**Scope note:** Details of **data mix**, **reward design**, and **infrastructure** are in the paper; this page distills **concepts** for **study** and **interviews**.

## Why This Paper Matters
- **Reframes “test-time reasoning” as in-context RL:** Instead of explicit search trees and value heads, the model performs a form of **implicit search** by **keeping its own work visible** and updating it across many tokens.
- **Challenges the MCTS-first mental model:** Shows that **plain RL + very long context** can reach **o1-class** math benchmarks, which sharpens interview answers about **when tree search is worth the engineering cost**.
- **Systems angle:** **Partial rollout reuse** is a concrete **training efficiency** idea—critical when rollouts are **hundreds of thousands of tokens** long.
- **Optimization angle:** Positions **online mirror descent** as a **stability** tool for **KL-regularized** LLM RL, complementing the **PPO/GRPO** family that dominates recent open recipes.
- **Generalization signal:** Strong **short-CoT** gains (reported up to **+550%** vs strong baselines in some settings) suggest that **reasoning-centric RL** can transfer beyond the exact long-CoT training regime.
- **Multimodal:** Joint training on **text and vision** aligns with product reality (documents, diagrams) and broadens where “long-context reasoning RL” applies.
- **Credit assignment at long horizon:** Standard seq2seq training often **does not** allocate gradient cleanly across **tens of thousands** of intermediate tokens; RL with **outcome** rewards plus **stable** KL updates is a different **optimization regime** than SFT on short answers.
- **Interview narrative:** Pairs naturally with **“test-time compute”** discussions: here, **compute** is spent as **longer generations in one context**, not necessarily **more parallel API calls** or **tree width**.

## Key Concepts Explained Simply
1. **Long context as a reasoning axis**  
   At **128K tokens**, the policy can **leave breadcrumbs**: intermediate steps, drafts, checks, and corrections remain **in the same forward context**. The model can **revisit** prior text—similar in spirit to **re-reading scratch work**—instead of relying on an external memory or a separate value model.

   - **Why interviewers care:** It reframes “reasoning budget” from **number of API calls** to **tokens visible to the policy**—a **single** rollout can still contain **many** internal edits.

2. **Partial rollout reuse**  
   A trajectory can be written as a concatenation \(\tau = [\tau_{\mathrm{prefix}};\, \tau_{\mathrm{new}}]\). If \(\tau_{\mathrm{prefix}}\) was already produced (and cached), training can **reuse logits / KV states** for that prefix and only **sample and backprop through** \(\tau_{\mathrm{new}}\). That slashes redundant compute when prefixes are long and shared across attempts.

   - **Operational pattern:** Think **“branch from checkpoint”**: keep a **hash** of \(\tau_{\mathrm{prefix}}\) in a **rollout cache**; only **extend** when exploration requires new tokens.

3. **Online mirror descent (OMD) for KL-regularized RL**  
   Mirror descent updates respect a **geometry** induced by a **Bregman divergence**—here commonly **KL** to the previous policy. Intuitively, it **regularizes** each step so the policy does not **lurch** on extremely long sequences where gradients can be noisy.

   - **Stability story:** Long sequences amplify **per-token noise**; anchoring updates in **KL** helps keep **sampling** distributions usable between steps.

4. **No MCTS required**  
   MCTS typically needs **many rollouts per decision**, **value estimation**, and careful branching—expensive at LLM scale. Kimi k1.5 argues that **long single trajectories** under RL can **match** strong search-based recipes **without** that machinery, trading **explicit branching** for **in-context revision**.

5. **Short-CoT transfer**  
   Even when evaluated under **short chain-of-thought** constraints, the model can **outperform** strong baselines like **GPT-4o** and **Claude Sonnet 3.5**, suggesting the **skills** learned during long reasoning training **compress** into shorter inference-time behavior.

6. **Multimodal joint training**  
   Reasoning is not text-only: training on **vision + language** encourages **grounded** reasoning (charts, figures) using the same **long-context RL** loop.

**Optional intuition (process vs outcome):** A **process reward model** scores intermediate steps; Kimi k1.5 leans on **outcome** verification (e.g., final answer correctness) while letting the **policy** allocate its own **internal** steps across **128K** tokens—reducing reliance on **human/PRM** labels for every intermediate token.

## The Math — Explained Step by Step

### 1. Online mirror descent (KL-regularized policy update)
Let \(\pi_t\) be the policy at iteration \(t\), \(R(\tau)\) an outcome reward for trajectory \(\tau\), and \(\eta > 0\) a step size. A canonical **mirror descent** update for the **KL-regularized** objective can be written in **closed form** as the **solution** to:
\[
\pi_{t+1} = \arg\min_{\pi} \Bigl[ -\mathbb{E}_{\tau \sim \pi}[R(\tau)] + \frac{1}{\eta} D_{\mathrm{KL}}(\pi \,\|\, \pi_t) \Bigr].
\]
**Intuition:** maximize expected return while staying **close** to \(\pi_t\) in **KL**—a **trust-region** in **distribution space**. Compared to **raw gradient ascent** on logits, this **geometry-aware** update tends to be **more stable** when sequences are long and gradients are high-variance.

### 2. Partial rollout decomposition
Split a sampled trajectory into a **cached prefix** and a **fresh suffix**:
\[
\tau = [\tau_{\mathrm{prefix}};\, \tau_{\mathrm{new}}],\qquad
\tau_{\mathrm{prefix}} \in \mathcal{C}\ \text{(cache hits)},\quad
\tau_{\mathrm{new}}\ \text{newly generated}.
\]
Let \(|\cdot|\) denote token length. The **compute** saved is roughly proportional to **not re-forwarding** \(\tau_{\mathrm{prefix}}\) through the network for every variant—especially when \(|\tau_{\mathrm{prefix}}| \gg |\tau_{\mathrm{new}}|\).

### 3. Length penalty (typical shaping form)
To discourage **runaway generation**, training may add a **penalty** increasing with length. A simple abstract form:
\[
R_{\mathrm{total}}(\tau) = R_{\mathrm{task}}(\tau) - \lambda \cdot f\bigl(|\tau|\bigr),
\]
where \(f\) might be **linear** or **piecewise** in tokens and \(\lambda\) trades **brevity** vs **exploration**. The exact \(f\) is a **design choice**; the key interview point is **reward shaping** interacts with **long-context** training—without penalties, policies may **pad** reasoning to exploit loopholes.

### 4. Where mirror descent sits vs GRPO / PPO
- **PPO:** **clipped** importance ratios + often a **critic** / GAE; **trust region** via clip.
- **GRPO:** **group-relative** advantages, **no critic**, still often **PPO-like** ratio machinery in practice.
- **OMD / KL mirror steps:** **explicit KL** anchoring to \(\pi_t\) (or a reference) via the **Bregman** structure; emphasizes **stable** iterative improvement under **noisy** long-sequence gradients.

**Mnemonic:** PPO clips **ratios**, GRPO redefines **advantages**, OMD shapes **where** the next policy lives in **probability space**.

### 5. GRPO-style advantage (for contrast)
For prompt \(q\), sample \(G\) completions with rewards \(r_i\). With mean \(\mu = \frac{1}{G}\sum_j r_j\) and std \(\sigma\) (with stabilizer \(\epsilon\)), GRPO uses
\[
\hat{A}_i = \frac{r_i - \mu}{\sigma + \epsilon}.
\]
**Contrast:** GRPO answers **“which samples are better than their siblings?”** for the same \(q\). OMD answers **“how do we update \(\pi\) without collapsing under KL?”** The two are **orthogonal**: one can imagine **group advantages** feeding the **reward** side while **mirror steps** govern **policy geometry**.

### 6. Suffix-only policy gradient (conceptual)
Let \(\log \pi_\theta(\tau)=\sum_t \log \pi_\theta(y_t\mid y_{<t})\). If \(\tau_{\mathrm{prefix}}\) is **fixed** under the current update (cached), then **only** tokens \(t\) in the suffix contribute **new** gradients:
\[
\nabla_\theta \mathbb{E}[R(\tau)] \approx \mathbb{E}\Bigl[R(\tau)\sum_{t \in \mathrm{suffix}} \nabla_\theta \log \pi_\theta(y_t\mid y_{<t})\Bigr],
\]
with variance reduction via **baselines** / **advantages** as usual. **Key point:** **stop-grad** or **reuse** on the prefix **does not** mean the prefix is irrelevant—it still **conditions** the suffix distribution through **attention**.

### 7. Reference policies and KL anchors (vocabulary alignment)
Many LLM RL pipelines anchor to a **reference** \(\pi_{\mathrm{ref}}\) (SFT model or earlier checkpoint) via a penalty \(D_{\mathrm{KL}}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})\). **Mirror descent** naturally pairs with this **because** each step already solves a **KL-regularized** proximal problem to \(\pi_t\); \(\pi_{\mathrm{ref}}\) can appear as an **additional** term or as the **initialization** trajectory of \(\pi_t\). Interview tip: clarify whether **KL** is to **last iterate** \(\pi_t\) (**mirror step**) or to a **fixed** ref (**RLHF-style** anchor)—implementations differ.

## Python Implementation
Below is a **minimal** illustration of **partial rollout reuse**: a long trajectory is represented as **cached prefix tokens** plus a **new suffix**. We compute **log-probabilities and a surrogate loss only on the suffix**, while **reusing** prefix **length** for bookkeeping. This is **educational**, not a full trainer.

**What to pretend:** `logits_suffix` is produced by a forward pass that **loaded KV cache** from \(\tau_{\mathrm{prefix}}\) and only **runs layers** for new positions—so **FLOPs** scale with **suffix length**, not full **128K** every time.

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PartialRollout:
    prefix_token_ids: torch.Tensor  # (Lp,) cached, detached from autograd for prefix
    suffix_token_ids: torch.Tensor  # (Ls,) targets for new tokens
    suffix_logp_old: torch.Tensor  # (Ls,) behavior policy log-probs (detached)


def forward_logprobs_on_suffix(
    logits_suffix: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """logits_suffix: (Ls, vocab); target_ids: (Ls,) next-token targets for suffix."""
    logp_all = F.log_softmax(logits_suffix, dim=-1)
    return logp_all.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def clipped_suffix_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantage: float,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_old)
    adv = torch.full_like(ratio, advantage)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.mean(torch.minimum(unclipped, clipped))


def token_level_reinforce_suffix(
    logp_new: torch.Tensor,
    advantages_per_token: torch.Tensor,
) -> torch.Tensor:
    """Unclipped REINFORCE-style loss on suffix tokens only: -E[A_t * log pi]."""
    return -(advantages_per_token * logp_new).mean()


def naive_vs_cached_forward_units(prefix_len: int, suffix_len: int) -> tuple[int, int]:
    """
    Toy linear cost model: units ~ sequence length for a full forward.
    Returns (full_sequence_units, suffix_only_with_cache_units).
    """
    full = prefix_len + suffix_len
    return full, suffix_len


def example_training_step() -> None:
    # Fake dimensions: prefix length 4096 cached; suffix length 256 trained
    Ls, vocab = 256, 32000
    logits_suffix = torch.randn(Ls, vocab, requires_grad=True)
    targets = torch.randint(0, vocab, (Ls,))

    old = PartialRollout(
        prefix_token_ids=torch.zeros(4096, dtype=torch.long),
        suffix_token_ids=targets,
        suffix_logp_old=torch.randn(Ls).detach(),
    )

    logp_new = forward_logprobs_on_suffix(logits_suffix, targets)
    adv = 1.0  # e.g., from outcome reward - baseline; only suffix gets gradients
    loss = clipped_suffix_loss(logp_new, old.suffix_logp_old, adv)
    loss.backward()
    # In a real system: logits_suffix comes from a forward pass that *reuses* KV cache
    # for prefix_token_ids and only computes suffix blocks.
    full_u, cached_u = naive_vs_cached_forward_units(4096, Ls)
    _ = full_u, cached_u  # e.g., log savings ratio full / cached_u for dashboards


if __name__ == "__main__":
    example_training_step()
```

**Reading guide:** In production, **KV-cache reuse** makes the **forward** on \(\tau_{\mathrm{prefix}}\) **once per reuse**, and autograd flows **only through** layers that affect \(\tau_{\mathrm{new}}\) (depending on checkpointing choices). **Gradient checkpointing** across the prefix can still affect **memory**—teams often **freeze** prefix layers or **detach** activations when acceptable.

## Interview Importance
Expect comparisons to **DeepSeek-R1 (GRPO)**, **OpenAI o1**, and **“search vs long context.”** Be ready to explain **why 128K matters**, **how partial rollouts save FLOPs**, and **why mirror descent is a stability narrative** alongside **PPO/GRPO**. Short-CoT transfer is a common **“does RL overfit?”** follow-up.

**Drill themes:** (1) **Implicit vs explicit search**—when is a **tree** worth it? (2) **Variance**—long sequences amplify **noisy** gradients; **KL** matters. (3) **Reward hacking**—length penalties and **verifiers**. (4) **Inference cost**—long CoT vs **distilled** students. (5) **Multimodal**—vision tokens increase **sequence** pressure—**cache reuse** becomes even more attractive.

## Interview Questions & Answers

### Q1: How does Kimi k1.5’s long-context RL differ from an MCTS-style reasoning stack?
**Answer:** **MCTS** expands a **search tree**, typically needs **many partial rollouts**, **branching**, and often a **value estimate** per node—high **wall-clock** and **implementation** cost at LLM scale. **Kimi k1.5** keeps **one** autoregressive trajectory but makes it **very long**, so the model can **revise** by **re-reading** prior text within the same context—**implicit** exploration instead of **explicit** node expansion. Trade-off: **less structured** search, but **far simpler** inference-time machinery.

**Follow-up:** MCTS shines when **action spaces** are small and **simulation** is cheap; autoregressive **text** has **huge** branching factor, so **neural policies** + **long CoT** are often the pragmatic path unless you invest heavily in **latent action** abstractions.

### Q2: Where does online mirror descent sit relative to GRPO?
**Answer:** **GRPO** is primarily an **advantage / baseline** construction (**group-relative** normalization across samples for the same prompt) and is often paired with **PPO-like** ratio clipping. **Online mirror descent** addresses **how** the next policy is computed under a **KL** proximity constraint to \(\pi_t\). You can think **GRPO** as **what signal** you optimize and **OMD** as **how you step** in policy space without **overshooting** on long sequences.

**Follow-up:** In interviews, avoid **false dichotomy**—a system might use **group baselines** *and* **mirror-style** updates; the paper’s contribution is highlighting **stability** at **128K** scale, not claiming **GRPO is obsolete**.

### Q3: Why does partial rollout reuse save so much compute in 128K-token training?
**Answer:** Autoregressive models cost roughly **linear** in sequence length per full forward **without** caching reuse. If many training examples share a **long common prefix** (or you **reuse** rejected partial samples), **skipping** re-computation of \(\tau_{\mathrm{prefix}}\) **amortizes** cost. Savings scale with **how often** prefixes repeat and **how long** they are—critical when **\(|\tau_{\mathrm{prefix}}|\)** dominates.

**Follow-up:** The win is **training throughput** and **experiment iteration**, not a free lunch—you still need **memory** for **KV** and must ensure **correct** conditioning when **resuming** from a cached prefix.

### Q4: Why would increasing context length improve reasoning at all?
**Answer:** Reasoning traces need **space** for **backtracking**, **checking**, and **alternatives**. Short contexts **force** compression or **forgetting** of intermediate steps. With **128K**, the policy can **condition** on its **entire** scratch path, improving **credit assignment** for which earlier mistakes led to failure—more **information** per gradient step **about its own process**.

**Follow-up:** This is **not** “more context always helps”—**data**, **verifiers**, and **optimization** must align; otherwise the model may **ramble** unless **length penalties** and **format** constraints are tuned.

### Q5: What does strong short-CoT performance suggest about generalization?
**Answer:** It suggests the model does not only memorize **verbose** patterns—it **internalizes** skills (e.g., verification habits) that **survive** when the evaluation regime **limits** CoT length. That supports **deployment** stories where you **cannot** afford **massive** generation at inference.

**Follow-up:** Teams may still **distill** long-CoT teachers to **short** students for **latency**; Kimi’s short-CoT result is evidence that **some** behaviors **compress**, not that **distillation** is automatic.

### Q6: How should I compare Kimi k1.5 to DeepSeek-R1 in an interview?
**Answer:** **DeepSeek-R1** popularized **GRPO + verifiable rewards** and **cold-start** choices on a **strong base**. **Kimi k1.5** emphasizes **128K in-trajectory** reasoning and **mirror-descent-style** stability, plus **partial rollout** engineering. Both target **o1-class** math; the **debate** is **recipe** (group baselines vs long-context geometry) and **systems** tricks (cache reuse vs more samples).

**Follow-up:** Mention **evaluation** caveats—**benchmarks** shift, **contamination** debates exist, and **product** quality includes **format**, **safety**, and **latency**, not only **AIME** points.

## Connections to Other Papers
**How to use this map:** In interviews, place Kimi k1.5 as **“long-horizon, KL-stable RL + systems (cache)”** in the same **neighborhood** as **GRPO-centric** recipes—same **goal** (strong math reasoning), different **emphasis** on **context length** and **mirror updates**.

| Paper / line of work | Connection |
|----------------------|------------|
| **DeepSeek-R1 (GRPO)** | Alternative **RL for reasoning** stack: **group-relative** advantages and **verifiable** rewards; compare **critic-free** training vs **long-context** mirror-descent emphasis. |
| **Chain-of-Thought (CoT)** | **CoT** is the **interface**; Kimi k1.5 scales **trainable** **long** reasoning traces via **RL**, not only **prompting**. |
| **InstructGPT (RLHF)** | Shared theme: **fine-tune with human/useful signals**; Kimi focuses on **task verifiers** and **long trajectories** rather than **preference-only** labels for core math capability. |
| **Constitutional AI** | Both touch **KL-to-reference** / **controlled** policy updates; Kimi stresses **math-system** stability and **length** regimes. |
| **DeepSeek-V3** | Illustrates how **strong base models** change **sample efficiency** of **downstream RL**; base quality sets the **ceiling** for reasoning RL. |

**Synthesis:** If **R1** answers **“how do we get advantages without a critic?”**, Kimi k1.5 answers **“how do we train when trajectories are **extremely** long?”**—together they cover **algorithm** and **systems** halves of modern **reasoning RL**.

## Key Takeaways for Quick Review

| Topic | One-liner |
|-------|-----------|
| **128K context** | **In-trajectory** planning/reflection—**implicit search** via **re-reading** long scratch work. |
| **No MCTS / no PRM** | **o1-class** results without **tree search** or **process reward models**—**systems** simplicity at inference. |
| **Partial rollouts** | \(\tau=[\tau_{\mathrm{prefix}};\tau_{\mathrm{new}}]\); **reuse** prefix **compute**, train on **suffix**. |
| **Online mirror descent** | KL-regularized update: \(\pi_{t+1}=\arg\min_\pi[-\mathbb{E}_\pi R + \frac{1}{\eta}D_{\mathrm{KL}}(\pi\|\pi_t)]\) for **stability**. |
| **Length penalty** | Shape \(R_{\mathrm{total}}=R_{\mathrm{task}}-\lambda f(|\tau|)\) to limit **runaway** generations. |
| **Short-CoT transfer** | Reasoning training **generalizes** beyond **verbose** evaluation—**useful** for **latency**-bounded deployment. |
| **Multimodal** | **Text + vision** joint training aligns with **real** reasoning over **documents and images**. |
| **GRPO contrast** | \(\hat{A}_i=(r_i-\mu)/(\sigma+\epsilon)\): **group-relative** signal—**orthogonal** to **OMD** geometry (can be combined conceptually). |
| **Suffix-only grads** | Prefix may be **cached/detached**; suffix tokens carry **policy gradient** while prefix **conditions** the distribution. |
| **Interview frame** | **Explicit search** vs **long single trace**; **cache** vs **more samples**; **verifiers** vs **PRMs**. |

**Last check before the interview:** State **one** systems win (**KV reuse**), **one** algorithmic win (**KL mirror step**), and **one** evaluation caveat (**benchmark scope**)—three sentences, high signal.
