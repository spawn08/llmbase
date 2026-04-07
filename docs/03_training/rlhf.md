# RLHF and Direct Preference Optimization

## Why This Matters for LLMs

**Reinforcement Learning from Human Feedback (RLHF)** turns pairwise **preferences** (“answer A is better than B”) into a **policy** that maximizes a **proxy reward** while staying near a **reference** model. Interviewers expect you to explain **reward model training** (Bradley–Terry), **PPO** with **KL control**, and why **DPO** replaces an explicit RL loop with a **classification** loss on preferences. This chapter is central to “how ChatGPT-style alignment works” questions.

A second reason is **optimization stability**: vanilla policy gradients on language models are **high-variance** and can **collapse** into gibberish without KL penalties. Understanding **clipped objectives** and **advantage** estimation separates API-level explanations from **implementable** ones.

Third, **DPO** (Direct Preference Optimization) shows the **reward model can be implicit**: the same objective rewrites preference likelihood under the **Bradley–Terry** model with an analytic optimum relating **policy** and **reference**—enabling **offline** training from preference datasets without online sampling.

---

## Core Concepts

### Preference Data and Bradley–Terry Model

Given prompt \(x\) and two completions \((y_w, y_l)\) where **\(y_w\)** is **preferred** (winner) over **\(y_l\)** (loser), define a **latent reward** \(r_\phi(x, y)\). The **Bradley–Terry** pairwise model:

\[
P_\phi(y_w \succ y_l \mid x) = \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)
\]

!!! math-intuition "In Plain English"
    The **difference** in reward scores between winner and loser maps through a logistic curve into a **probability** of preferring the winner. If \(r_\phi(x,y_w)\) is even slightly larger than \(r_\phi(x,y_l)\), the model predicts preference probability above one-half; large gaps push probability toward one.

with \(\sigma(z) = 1/(1+e^{-z})\).

**Reward model training** (regression on preferences) minimizes negative log-likelihood:

\[
\mathcal{L}_{\text{RM}} = - \mathbb{E}\bigl[\log \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\bigr]
\]

!!! math-intuition "In Plain English"
    The reward model only needs to **rank** pairs correctly—**scalar** scores are meaningful **up to monotonic transform**. What matters is **margin** between preferred and dispreferred answers.

### Policy Gradient (REINFORCE Sketch)

Let \(\pi_\theta\) be the language model policy. **Expected reward** objective:

\[
J(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\bigl[r_\phi(x, y)\bigr]
\]

!!! math-intuition "In Plain English"
    Sample prompts from the data distribution, sample full answers from the **current** policy, score them with the reward model, and try to **increase** average reward. There is no supervised target string—only scalar feedback—so exploration and variance reduction dominate engineering effort.

**Policy gradient** theorem:

\[
\nabla_\theta J = \mathbb{E}\bigl[r_\phi(x,y) \nabla_\theta \log \pi_\theta(y \mid x)\bigr]
\]

!!! math-intuition "In Plain English"
    The classic score-function gradient: push up log-probability of **entire** sampled sequences in proportion to their reward. For LMs, \(\log \pi_\theta(y\mid x)\) decomposes into a **sum** of per-token log-probs along the generated continuation.

(subterms for full sequence expand to **sum over tokens** with **causal** log-probs).

!!! math-intuition "In Plain English"
    Increase probability of sequences that get **high reward**, decrease probability of **low reward**—but vanilla REINFORCE is **noisy**; **baselines**, **advantages**, and **PPO clipping** reduce variance.

### PPO for Language Models

**Proximal Policy Optimization** constrains updates so the new policy \(\pi_\theta\) stays near **behavior** policy via a **ratio**:

\[
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
\]

!!! math-intuition "In Plain English"
    Compare **new** policy probability to the **old** behavior policy on the **same** sampled action. If ratio \(\gg 1\), the new policy would assign much more mass to that token—PPO will **clip** updates so this does not blow up in one step.

For autoregressive LMs, the **token** \(a_t\) is the next token; **state** \(s_t\) is prefix.

**Clipped surrogate**:

\[
L^{\text{CLIP}}(\theta) = \mathbb{E}\bigl[\min\bigl(r_t(\theta)\hat{A}_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\bigr)\bigr]
\]

!!! math-intuition "In Plain English"
    Take the **minimum** of two terms: an unclipped surrogate and a clipped one. When advantage \(\hat{A}_t\) is positive, you increase probability of token \(a_t\), but only until the ratio leaves \([1-\epsilon, 1+\epsilon]\)—preventing **destructively large** policy jumps.

where \(\hat{A}_t\) is an **advantage** estimate (e.g., from a learned **value** head or **GAE**).

### KL Penalty to Reference Policy

RLHF adds a penalty keeping \(\pi_\theta\) close to **SFT reference** \(\pi_{\text{ref}}\):

\[
J_{\text{PPO}} = \mathbb{E}\bigl[r_\phi(x,y)\bigr] - \beta \, \mathrm{KL}\bigl(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\bigr)
\]

!!! math-intuition "In Plain English"
    Maximize expected **reward model** score while paying \(\beta\) times KL divergence from the **SFT reference** on the same prompt. This is the same **anchor** idea as in DPO’s \(\beta\) on log-ratios—keep the policy from drifting into regions where the reward model is **miscalibrated**.

Practically, **KL** is estimated from sampled sequences; \(\beta\) trades **reward** vs **drift**.

!!! math-intuition "In Plain English"
    Without KL, the optimizer finds **adversarial** strings that fool the **reward model**—**Goodhart**’s law. KL anchors to **human-like** text from SFT.

---

## DPO: Implicit Reward Model

Rafailov et al. show that under the **Bradley–Terry** model and a **closed-form** optimal policy for a KL-regularized reward objective, preferences can be optimized by minimizing:

\[
\mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}\Bigl[\log \sigma\Bigl(
\beta \bigl(
\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\bigr)
\Bigr)\Bigr]
\]

**No explicit reward network**; only **policy** vs **reference** log-likelihood ratios on **preferred** and **dispreferred** completions.

!!! math-intuition "In Plain English"
    DPO asks: “Increase **relative** probability of \(y_w\) over \(y_l\) vs reference,” with **temperature** \(\beta\) controlling **how aggressively** to shift.

### Derivation Sketch (High Level)

Start from KL-regularized RL objective whose optimum satisfies:

\[
\pi^\star(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp\bigl(\tfrac{1}{\beta} r(x,y)\bigr)
\]

!!! math-intuition "In Plain English"
    The **optimal** policy under KL-regularized reward maximization reweights the reference by an exponential of true reward—**high-reward** strings get boosted exponentially; \(\beta\) controls how **peaked** that reweighting is.

Rearranging gives an **implicit reward**:

\[
r(x,y) = \beta \log \frac{\pi^\star(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
\]

!!! math-intuition "In Plain English"
    Solve for reward: it is **proportional** to the log-density ratio between optimal policy and reference, plus a **partition** term \(Z(x)\) that is constant for fixed \(x\). When you take **differences** \(r(x,y_w)-r(x,y_l)\), the \(Z(x)\) term **cancels**—so you never need to estimate it explicitly in DPO.

Substitute into Bradley–Terry; **\(Z(x)\)** cancels in **\(r(x,y_w) - r(x,y_l)\)**, yielding the DPO loss with \(\pi_\theta\) replacing \(\pi^\star\).

### Effect of \(\beta\) in DPO

- **Large \(\beta\)**: strong pressure to **separate** \(y_w\) vs \(y_l\) relative to reference—can **overfit** small preference sets or **collapse** diversity if paired with aggressive optimization.
- **Small \(\beta\)**: gentler shifts—policy stays **near** reference; may **underfit** preferences.

Tuning \(\beta\) is analogous to tuning **KL coefficient** in RLHF.

### Worked Micro-Example (Scalars)

Let \(\beta = 1\). Suppose for a fixed prompt, sequence log-probs satisfy:

| Model | \(y_w\) | \(y_l\) |
|-------|---------|---------|
| \(\pi_\theta\) | \(-1.0\) | \(-2.0\) |
| \(\pi_{\text{ref}}\) | \(-1.2\) | \(-1.5\) |

DPO logit (inside \(\sigma\)):

\[
( -1.0 - (-1.2) ) - ( -2.0 - (-1.5) ) = 0.2 - (-0.5) = 0.7
\]

\(\log \sigma(0.7) \approx -0.44\) contribution—**positive margin** encourages the preference.

---

## RLHF vs DPO Comparison

| Aspect | RLHF (PPO + RM) | DPO |
|--------|------------------|-----|
| **Reward model** | Explicit network \(r_\phi\) | Implicit |
| **Optimization** | Online sampling from \(\pi_\theta\) | Offline on preference pairs |
| **Stability** | Sensitive to RM exploits; needs KL | Often more stable; still needs \(\beta\) tuning |
| **Compute** | Higher (rollouts, value heads) | Lower (forward passes on pairs) |
| **Flexibility** | Can plug arbitrary **scalar** rewards | Tied to **preference** likelihood model |

---

## Python: Bradley–Terry Reward Loss (Toy)

```python
"""
Bradley-Terry preference loss for scalar rewards — toy vectors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def bt_loss(r_w: torch.Tensor, r_l: torch.Tensor) -> torch.Tensor:
    """r_w, r_l: shape (B,) — higher is better for winner."""
    return -F.logsigmoid(r_w - r_l).mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    b = 16
    r_w = torch.randn(b)
    r_l = torch.randn(b)
    print(bt_loss(r_w, r_l))
```

---

## Python: DPO Loss (Sequence Log-Probs Placeholder)

```python
"""
DPO loss given per-example logpi_theta and logpi_ref for winner/loser sequences.
Real code must sum token logprobs with attention to padding masks.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    logpi_w: torch.Tensor,
    logpi_l: torch.Tensor,
    logpiref_w: torch.Tensor,
    logpiref_l: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    logpi_* : total sequence log-probs for policy / ref on winner or loser.
    Shapes: (B,)
    """
    logits = beta * (
        (logpi_w - logpiref_w) - (logpi_l - logpiref_l)
    )
    return -F.logsigmoid(logits).mean()


if __name__ == "__main__":
    b = 8
    # Example batch: policy slightly prefers winner vs reference
    logpi_w = torch.tensor([-10.0, -9.0, -8.5, -11.0, -9.2, -8.0, -10.5, -9.1])
    logpi_l = torch.tensor([-12.0, -11.0, -10.0, -12.0, -11.5, -10.0, -12.0, -11.0])
    logpiref_w = torch.tensor([-10.5, -9.5, -9.0, -11.5, -9.8, -8.5, -11.0, -9.5])
    logpiref_l = torch.tensor([-11.8, -10.8, -10.5, -11.8, -11.2, -10.2, -11.9, -10.9])
    print("DPO batch loss:", dpo_loss(logpi_w, logpi_l, logpiref_w, logpiref_l, beta=0.1).item())
```

---

## Python: TRL-Style DPO Training (Skeleton)

```python
"""
Offline DPO with Hugging Face TRL — use for production pipelines.
pip install trl transformers datasets accelerate torch
"""
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def main() -> None:
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(model_id)
    ref = AutoModelForCausalLM.from_pretrained(model_id)

    # Minimal preference pair: prompt + chosen + rejected (TRL expects column names)
    ds = Dataset.from_dict(
        {
            "prompt": ["Explain gravity in one sentence."],
            "chosen": ["Gravity attracts masses; near Earth we feel it as weight."],
            "rejected": ["Stuff falls down."],
        }
    )

    args = DPOConfig(
        output_dir="./dpo_out",
        per_device_train_batch_size=1,
        beta=0.1,
        max_length=256,
        max_prompt_length=128,
        num_train_epochs=1,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

??? deep-dive "Deep Dive: `ref_model` and LoRA in DPO"
    Often the **policy** is LoRA-wrapped while **reference** stays full-precision frozen. TRL supports this pattern to save memory; `ref_model` may be `None` with custom log-prob computation in advanced configs—read current TRL docs for your version.

---

## Reward Hacking and Mitigations

**Hacking**: policy finds **high reward**, low human quality strings. Mitigations:

- **KL** to reference (RLHF, DPO \(\beta\)).
- **Ensemble** reward models; **Bradley–Terry** on multiple heads.
- **Periodic** human audits; **filter** train rollouts.
- **Constraint** decoding during RL sampling.

### Reward Model Training Details

- **Data**: pairwise comparisons \((x, y_w, y_l)\); sometimes **K-wise** rankings.
- **Architecture**: often **same backbone** as LM with **scalar head** on final token or **pooled** hidden state.
- **Regularization**: **LM auxiliary loss** on high-quality text keeps \(r_\phi\) from forgetting **language modeling**.
- **Normalization**: rewards may be **standardized** batchwise for stable PPO **advantages**.

---

## PPO Mechanics for Autoregressive LMs

1. **Rollout**: sample completions \(y \sim \pi_{\text{old}}(\cdot \mid x)\).
2. **Score**: \(R = r_\phi(x,y) - \beta \log \frac{\pi_{\text{old}}(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\) (KL term variants exist).
3. **Advantage**: subtract **baseline** \(V_\psi\) if using **actor–critic**; else **Monte Carlo** return minus mean.
4. **Update** \(\pi_\theta\) with **multiple epochs** on the same batch using **clipped** objective (trust region).
5. **Sync** \(\pi_{\text{old}} \leftarrow \pi_\theta\) periodically.

!!! math-intuition "In Plain English"
    PPO is **trust region**: don’t move the policy so far that linearizations fail. **Clip** enforces that **implicitly** without second-order Hessians.

### Advantage Estimation: GAE Sketch

Generalized Advantage Estimation combines **TD** errors with **eligibility** weighting:

\[
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l},\quad
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

For **single-shot** sequence reward \(R\) (no per-token reward), some LM stacks assign **one** advantage to **all** tokens in the completion; others use **reward at EOS** only—design choice affects **credit assignment**.

### Value Baseline for Variance Reduction

Using **learned** \(V_\psi(s_t)\) to **center** advantages reduces gradient variance:

\[
\hat{A}_t \approx \Bigl(\sum_{k=t}^{T} r_k\Bigr) - V_\psi(s_t)
\]

**States** \(s_t\) are **prefix hidden states**; value heads are **optional** when **sequence-level** returns suffice.

### IPO / cDPO Variants (Pointer)

**Identity Preference Optimization (IPO)** generalizes DPO with a different **link** function to reduce **length biases**—when **winners** are systematically longer, naive preference objectives may **prefer verbosity**. **Conservative** variants add **explicit** regularizers. Mention in interviews as “**follow-on** work addressing DPO pathologies.”

---

## Policy Gradient: Sequence-Level Expansion

For sequence \(y = (y_1,\dots,y_T)\):

\[
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})
\]

!!! math-intuition "In Plain English"
    Autoregressive models factor the joint probability of a string as a **product** of next-token probabilities; the **log** turns that product into a **sum** over positions—this is what you backprop through in practice.

**Gradient**:

\[
\nabla_\theta \log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})
\]

!!! math-intuition "In Plain English"
    Gradients **accumulate** across tokens: changing early-layer weights can improve later-token predictions. RL methods differ in whether one **scalar** reward is credited to all tokens or **shaped** per position.

REINFORCE uses **sum of token logprob grads** weighted by **total reward** (or **per-token** advantage variants in practice).

### ORPO and Odds-Ratio Objectives (Pointer)

**ORPO** combines **SFT** and preference optimization in one stage by adding an **odds ratio** term to the standard LM loss—useful when you **lack** a strong reference checkpoint. Interview framing: “**Single-stage** alignment vs **SFT then DPO** pipeline trade-offs.”

### SLiC / Sequence Likelihood Calibration (Pointer)

Some methods **rank** sequences using **calibrated** likelihoods with **margin** losses—related spirit to DPO but different **parameterization**. Useful keyword if comparing **families** of offline preference losses.

---

## Practical Engineering Notes

- **Reference model**: often **frozen SFT** weights in **FP16**; policy may be same arch with **EMA** for stability.
- **Batch construction**: each GPU may hold **different** prompts; **gather** preference pairs **stratified** by length to reduce **bias**.
- **Tokenizer edge cases**: **logprob** sums must **exclude** padding consistently between winner/loser sequences.
- **Evaluation**: **Arena**-style pairwise eval with **Elo** mirrors training objective better than single-score benchmarks.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Write the Bradley–Terry preference probability and explain what the reward model is learning.
        *Answer:* The standard form is \(P(y_w \succ y_l \mid x) = \sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr)\) with logistic \(\sigma\). The reward model learns a **scalar score** such that **margins** between preferred and rejected completions match observed pairwise labels—it is a **ranking** surrogate, not a literal “utility in human units,” and is trained by maximizing log-likelihood of those comparisons (often with auxiliary LM loss to preserve language quality).
    2. Why does RLHF include a KL penalty to the SFT reference, and what failure mode does it address?
        *Answer:* Without a KL anchor, policy optimization **overfits** the reward model: it finds **adversarial** strings that spike \(r_\phi\) but look like gibberish or exploit RM blind spots (**Goodhart**’s law). Penalizing \(\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})\) keeps updates near **human-demonstrated** text from SFT while still increasing reward, improving **robustness** when \(r_\phi\) is imperfect.
    3. Compare PPO-based RLHF with DPO—data flow, compute, and when you would choose each.
        *Answer:* **PPO+RM** needs **online** sampling from the current policy, reward scoring, advantage estimation, and often a **value** head—high GPU time and engineering complexity but flexible **arbitrary** scalar rewards (including non-preference signals). **DPO** is **offline**: only forward passes on **fixed** \((x, y_w, y_l)\) pairs to optimize implicit preferences—lower compute and simpler ops, best when you already have **logged** preferences and no custom reward. Choose PPO when you need **iterative** exploration or composite rewards; choose DPO for **stability** and **data-only** alignment at scale.
    4. Derive the intuition that DPO reweights the policy using log-ratios to the reference.
        *Answer:* Under KL-regularized reward maximization, the **optimal** policy satisfies \(\pi^\*(y|x) \propto \pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)\), so **implicit** reward differences are \(\beta \log \frac{\pi^\*(y|x)}{\pi_{\text{ref}}(y|x)}\) up to a prompt-dependent constant that **cancels** in pairwise comparisons. DPO plugs \(\pi_\theta\) into that structure: it pushes **relative** likelihood of \(y_w\) versus \(y_l\) **above** the reference odds, with \(\beta\) setting how aggressively that reweighting happens—**no separate** \(r_\phi\) network.
    5. What is reward hacking, and name three mitigations beyond increasing \(\beta\) or KL weight.
        *Answer:* **Reward hacking** is when the policy maximizes \(r_\phi\) with outputs that are **not** actually better for humans—e.g., excessive lists, keyword stuffing, or nonsense that exploits RM artifacts. Mitigations include **ensemble** or **multi-head** reward models, **periodic human audits** and **filtered** rollout data, **constraint** decoding or **format** penalties, **red-team** suites with **held-out** evaluators, and **mixing** a small LM auxiliary loss on the reward net so it stays calibrated to language.
    6. How are sequence log-probabilities aggregated for variable-length completions in DPO?
        *Answer:* For causal LMs, \(\log \pi(y\mid x)\) is the **sum** of per-token log-probabilities \(\sum_t \log \pi(y_t \mid x, y_{<t})\) over the **completion** tokens (excluding prompt positions), with **padding** tokens masked out consistently for winner and loser. Implementations must align **tokenization** boundaries so compared sequences refer to the **same** prompt length and EOS handling—length bias is a known pitfall if aggregation is sloppy.
    7. Why can reward models misgeneralize, and how does that hurt online RL fine-tuning?
        *Answer:* RMs are trained on **finite** comparison data and can rely on **spurious** cues (length, politeness markers) that do not transfer to new prompts or **long-horizon** rollouts. In **online** RL, the policy **actively** searches for RM blind spots, so small calibration errors become **large** behavioral failures—this is why KL penalties, diverse comparison data, and **ongoing** human monitoring matter for PPO-style loops.
    8. What is the clipped PPO objective doing in plain engineering terms?
        *Answer:* PPO forms a **ratio** of new to old policy probabilities per token and **clips** updates when that ratio leaves \([1-\epsilon, 1+\epsilon]\), so a single minibatch cannot **explode** policy mass on one action. In engineering terms: “take a **conservative** trust-region step that improves the objective when advantage is positive, but **cap** how far you move in one shot”—stabilizing RL on long autoregressive chains where raw policy gradients are high-variance.
    9. How does \(\beta\) in DPO relate to the KL coefficient in RLHF?
        *Answer:* Both control **how far** the tuned policy may deviate from \(\pi_{\text{ref}}\): larger \(\beta\) in DPO acts like a **stronger** preference signal relative to the reference (analogous to a **tighter** implicit KL budget in the derivation), while smaller \(\beta\) keeps the policy **near** SFT. In RLHF, the KL coefficient \(\beta_{\text{KL}}\) in \(J = \mathbb{E}[r] - \beta_{\text{KL}}\mathrm{KL}\) plays the same **trade-off** knob—higher KL weight / lower temperature \(\Rightarrow\) less deviation from reference behavior.
    10. What evaluations would you run before shipping a policy after preference optimization?
        *Answer:* Run **pairwise** or **Elo** human/LLM-judge evals on representative prompts, **safety** red-team suites (jailbreaks, toxicity), **regression** task benchmarks (MMLU, coding, math) for capability **tax**, **length/format** checks for verbosity hacks, and **online** A/B metrics if available (user satisfaction, task success). Ship only when **win-rate**, safety, and capability floors all meet thresholds—not when a single RM proxy improves.

!!! interview "Follow-up Probes"
    - “Walk through token-level vs sequence-level rewards in LM PPO.”
    - “What happens if the reference model is stale relative to SFT?”
    - “How do length biases show up in DPO, and what variants address them?”
    - “When would you still train an explicit reward model if DPO is available?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Bradley–Terry likelihood—reward model learns rankings, not absolute utilities.”
    - “KL anchor prevents Goodharting the reward model.”
    - “DPO is offline preference optimization via implicit reward from log-ratios.”
    - “PPO clips probability ratios for stable trust-region updates.”

---

## References

- Christiano et al., *Deep Reinforcement Learning from Human Preferences* — foundational human preference RL
- Stiennon et al., *Learning to Summarize with Human Feedback* — [arXiv:2009.01325](https://arxiv.org/abs/2009.01325)
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Schulman et al., *Proximal Policy Optimization Algorithms* — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Touvron et al., *LLaMA 2: Open Foundation and Fine-Tuned Chat Models* — RLHF details
- Welleck et al., *Natural Language Generation with Neural Likelihood Models* — (context on decoding)
