# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning
**Authors:** DeepSeek-AI &nbsp;|&nbsp; **Year:** 2025 &nbsp;|&nbsp; **Venue:** arXiv &nbsp;|&nbsp; **Link:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
---
## TL;DR
DeepSeek-R1-Zero is the first model shown to develop strong reasoning behavior **purely through reinforcement learning**, without any supervised fine-tuning (SFT) on reasoning traces. Starting from the DeepSeek-V3 base model, **Group Relative Policy Optimization (GRPO)** with **verifiable rewards** encourages self-verification and long chain-of-thought. **DeepSeek-R1** adds a small **cold-start SFT** phase with thousands of long-CoT examples for training stability. The system matches **OpenAI o1-class** performance on benchmarks such as **AIME** and **MATH-500**. **Distilled** variants (R1-Distill) transfer reasoning into smaller **7B–70B** student models via SFT on teacher-generated traces, often **without additional RL** for the student.

In one sentence for interviews: **verifiable rewards + multi-sample group baselines (GRPO) + strong base model** is enough to elicit **o1-like** math reasoning in an open pipeline, with **distillation** carrying that behavior down to smaller models.

## Why This Paper Matters
- **Proof of concept for “reasoning from RL alone”:** R1-Zero shows that reasoning-like behavior can emerge from RL on a strong base, supporting the view that capability is partly **latent in pretrained weights** and can be **elicited** by the right optimization signal.
- **Practical recipe:** GRPO + verifiable rewards + (optional) cold-start data + distillation gives a **reproducible pipeline** that does not rely on proprietary human preference data for the core math reasoning loop.
- **Benchmark relevance:** Strong results on **math** benchmarks make this a standard reference when discussing **test-time compute**, **long CoT**, and **open-weight** alternatives to closed “reasoning models.”
- **Engineering transparency:** The paper popularized concrete choices—**group size \(G\)**, **KL to reference**, **reward definition** on math/code—that teams can debate and replicate more easily than black-box “reasoning API” behavior.
- **Downstream product shape:** The **distillation** story clarifies a common deployment path: train a **large** reasoning teacher with RL, then ship **smaller** students that mimic traces for latency/cost.

## Key Concepts Explained Simply
1. **R1-Zero**  
   RL is applied **directly on the base model** with **no SFT** on reasoning demonstrations first. The claim is that **reasoning can be learned as a policy improvement problem** when rewards are clear and the base model is capable enough.

2. **GRPO (Group Relative Policy Optimization)**  
   For each prompt, sample **\(G\)** completions. Each completion gets a scalar reward **\(r_i\)**. Advantages are **normalized within the group** (subtract mean, divide by std), so the model learns **which of several attempts is relatively better** for the same question—without a separate **value network** (critic).  
   Increasing **\(G\)** makes the group mean a **lower-variance** baseline but costs **more rollouts** per optimizer step—the same fundamental **variance vs compute** tradeoff as using more samples in any Monte Carlo estimator.

3. **Verifiable rewards**  
   For math, correctness can be checked against a **ground-truth answer**; for code, **unit tests** pass or fail. These signals are **less ambiguous** than human preference labels and tend to yield **more stable** RL for reasoning tasks.

4. **Cold-start SFT (DeepSeek-R1 vs R1-Zero)**  
   **R1** adds **thousands** of long chain-of-thought examples **before** RL to reduce pathologies such as **language mixing**, **format collapse**, and **repetition** that pure RL can exhibit early on.

5. **“Aha moment”**  
   Training curves sometimes show a **sudden jump** in reasoning quality—an **emergent** transition rather than smooth linear improvement—highlighting that RL on LLMs can produce **phase-like** behavior.

6. **Distillation pipeline**  
   **R1-Distill** models are trained with **SFT on outputs (traces) from R1**. The **student** may not need RL if it mainly learns to **imitate** the teacher’s reasoning style and steps at a smaller scale.  
   Quality hinges on **teacher trace diversity** and **task coverage**: students can **overfit** to teacher quirks (length, phrasing) unless data is **filtered** and **mixed** with normal instruction tuning.

## The Math — Explained Step by Step
### 1. Group-relative advantage
For a fixed prompt \(q\), suppose we sample \(G\) completions indexed \(i = 1,\ldots,G\), each with reward \(r_i\). Let
\[
\mu = \frac{1}{G}\sum_{j=1}^{G} r_j,\qquad
\sigma = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu)^2 + \epsilon}
\]
with a small \(\epsilon > 0\) for numerical stability. The **group-relative advantage** is
\[
\hat{A}_i = \frac{r_i - \mu}{\sigma}.
\]
**Intuition:** we only need **ranking pressure within the group**: who beat the average, and by how many “standard deviations,” for **this** prompt.

### 2. GRPO objective (clipped policy gradient + KL)
Let \(\pi_\theta\) be the policy and \(\pi_{\mathrm{ref}}\) a reference (often the initial or supervised policy). For completion \(i\), let \(\rho_i = \frac{\pi_\theta(a_i \mid q)}{\pi_{\mathrm{old}}(a_i \mid q)}\) be the **probability ratio** to a behavior policy \(\pi_{\mathrm{old}}\) (e.g., rollout policy). A typical GRPO-style objective maximizes
\[
J(\theta) = \mathbb{E}_{q}\left[
\frac{1}{G}\sum_{i=1}^{G}
\min\bigl(\rho_i \hat{A}_i,\ \mathrm{clip}(\rho_i,\,1-\varepsilon,\,1+\varepsilon)\,\hat{A}_i\bigr)
\right]
- \beta\, D_{\mathrm{KL}}\bigl(\pi_\theta \,\|\, \pi_{\mathrm{ref}}\bigr).
\]
- The **min/clip** term mirrors **PPO’s** trust-region idea: large policy updates are **clipped** when \(\rho_i\) leaves \([1-\varepsilon, 1+\varepsilon]\).
- **Averaging over \(G\)** ties the update to **multiple sampled rollouts** per prompt.

### 3. Comparison to PPO
- **PPO** typically uses a **learned baseline** \(V(s)\) (advantage \(\approx r - V\) plus GAE, etc.) to **reduce variance**.
- **GRPO** uses the **group mean** (and scaling by std) as a **baseline substitute** derived from **multiple completions for the same input**, and **does not require a critic**—saving parameters and simplifying training when rewards are **sparse but verifiable**.

### 4. KL penalty and its role
The term \(\beta\, D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\) penalizes drifting too far from a **stable reference**. In practice it:
- **Prevents collapse** into degenerate text or exploit-the-reward hacks.
- **Keeps sampling** near a known-good policy so **GRPO updates** remain meaningful.

### 5. Why subtracting the group mean is a baseline
Write the centered reward \(\tilde{r}_i = r_i - \mu\). For fixed \(q\), \(\sum_i \tilde{r}_i = 0\), so positive \(\tilde{r}_i\) are **exactly offset** by negatives in the same group. This is the same **centering** idea as using a **baseline** to lower variance: the learning signal for completion \(i\) is measured **relative to what was typical for this prompt** under the current sampling process. Dividing by \(\sigma\) makes the scale comparable across prompts with different reward spreads.

### 6. Interpreting the clipped surrogate
For each \(i\), define \(f_i(\theta) = \min(\rho_i \hat{A}_i,\ \mathrm{clip}(\rho_i)\,\hat{A}_i)\). When \(\hat{A}_i > 0\), increasing \(\rho_i\) is “good” only until \(\rho_i\) hits \(1+\varepsilon\); beyond that, the **clipped** branch stops the gradient from pushing harder—reducing **destructively large** policy steps. When \(\hat{A}_i < 0\), the clip symmetrically limits how much we penalize ratio shrinkage. This is the same **trust-region flavor** as PPO, but \(\hat{A}_i\) comes from **group-relative** rewards instead of **GAE** on token rewards.

### 7. KL in practice (sketch)
Implementations often approximate \(D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})\) with **token-level** terms averaged over the sequence, e.g. \(\mathbb{E}_t[\log \pi_{\mathrm{ref}}(x_t\mid x_{<t}) - \log \pi_\theta(x_t\mid x_{<t})]\) under the **student** distribution, or use **k3/k1** estimators as in common RLHF codebases. The key interview point: **match your codebase’s KL estimator** to what the paper/report claims, because small differences matter for stability.

### 8. Worked micro-example (group advantages)
Suppose \(G=4\) and rewards are \((r_1,r_2,r_3,r_4) = (1, 0, 0, 1)\) for a binary correctness task. Then \(\mu = 0.5\) and (population-style) variance is \(\frac{1}{4}\sum (r_i-\mu)^2 = 0.25\), so \(\sigma = 0.5\) (with \(\epsilon\) negligible). Thus \(\hat{A}_1 = (1-0.5)/0.5 = 1\) and \(\hat{A}_2 = (0-0.5)/0.5 = -1\). The two wrong answers receive **negative** advantage even though their raw reward is not “below zero”—they are below the **group average**. This is exactly the **relative** learning signal GRPO encodes.

## Notation quick reference

| Symbol | Meaning |
|--------|---------|
| \(G\) | Number of sampled completions per prompt |
| \(r_i\) | Scalar **verifiable** reward for completion \(i\) |
| \(\mu,\sigma\) | Group mean and (stabilized) std of rewards |
| \(\hat{A}_i\) | Group-relative advantage \((r_i-\mu)/\sigma\) |
| \(\rho_i\) | Policy ratio \(\pi_\theta/\pi_{\mathrm{old}}\) on the trajectory |
| \(\varepsilon\) | PPO-style clip radius |
| \(\beta\) | KL penalty weight to \(\pi_{\mathrm{ref}}\) |

## Python Implementation
Below is a **simplified** training loop illustrating **GRPO-style** group advantages and **clipped** policy loss. It uses **dummy** log-probs and rewards; plug in your tokenizer, model, and environment.

```python
"""
Minimal educational sketch: GRPO-style advantages + PPO-style clipped objective.
Not production-RLHF; wire in your model, logprob sums, and reward function.
"""
import torch
import torch.nn as nn

def group_relative_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """rewards: shape (G,) — one scalar reward per sampled completion."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False).clamp_min(eps)
    return (rewards - mean) / std

def clipped_policy_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> torch.Tensor:
    """Minimize negative clipped surrogate (same sign convention as PPO implementations)."""
    ratio = torch.exp(logp_new - logp_old)  # (G,) — one scalar per completion
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.mean(torch.minimum(unclipped, clipped))

def kl_penalty(logp_new: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Token-level surrogate: mean (log pi_theta - log pi_ref) on sampled tokens."""
    return (logp_new - logp_ref).mean()

def grpo_step_dummy(
    rewards: torch.Tensor,
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    logp_ref: torch.Tensor,
    clip_eps: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    adv = group_relative_advantages(rewards)
    pol = clipped_policy_loss(logp_new, logp_old, adv, clip_eps)
    kl = kl_penalty(logp_new, logp_ref)
    return pol + beta * kl

class ValueBaseline(nn.Module):
    """PPO-style critic: maps a state vector to V(s). Not used in GRPO advantages."""

    def __init__(self, dim: int):
        super().__init__()
        self.v = nn.Linear(dim, 1)

    def forward(self, state_vec: torch.Tensor) -> torch.Tensor:
        return self.v(state_vec).squeeze(-1)

def ppo_style_advantage(
    reward: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Single-step MC advantage for illustration: A = r - V(s)."""
    return reward - value

def fake_grpo_training_step(
    g: int = 4,
    dim: int = 32,
    clip_eps: float = 0.2,
    beta: float = 0.01,
) -> torch.Tensor:
    """Toy tensors: pretend we already computed log-probs and rewards."""
    torch.manual_seed(0)
    rewards = torch.randn(g)
    logp_new = torch.randn(g) * 0.1
    logp_old = logp_new.detach() + torch.randn(g) * 0.05
    logp_ref = logp_new.detach() + torch.randn(g) * 0.02
    return grpo_step_dummy(rewards, logp_new, logp_old, logp_ref, clip_eps, beta)

def fake_ppo_contrast_step(
    state_dim: int = 32,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """One completion: advantage uses critic, not group normalization."""
    torch.manual_seed(1)
    s = torch.randn(1, state_dim)
    critic = ValueBaseline(state_dim)
    reward = torch.tensor([1.0])
    v = critic(s)
    adv = ppo_style_advantage(reward, v)
    logp_new = torch.tensor([0.0], requires_grad=True)
    logp_old = torch.tensor([-0.1])
    return clipped_policy_loss(logp_new, logp_old, adv, clip_eps)

# GRPO: advantages from (r_i - mean)/std across G completions for the *same* prompt.
# PPO: advantages from returns minus value (+ GAE across long trajectories), not group stats.
```

**Difference from PPO (summary):** PPO usually needs a **critic** \(V_\phi(s)\) and **advantage estimation** from trajectories (often **GAE**); GRPO uses **multiple answers per question** and **group-relative** scaling to obtain **\(\hat{A}_i\)** without \(V_\phi\). **Implementation note:** real LLM training sums **token** log-probs per sequence; the tensors above are **per-completion** scalars for clarity.

**Training loop (pseudocode):**

```
for batch of prompts:
    for each prompt:
        sample G completions
        score each with verifiable reward r_i
        compute A_hat from group (r_i)
        compute policy loss + KL to reference
    optimizer.step()
```

**PPO contrast (same pseudocode spirit):**

```
for each trajectory:
    compute returns R_t along tokens
    fit V(s) (critic) toward returns
    advantage = GAE(R, V) or R - V
    clipped policy loss with advantage
```

GRPO swaps “**fit critic + GAE**” for “**draw G answers; center rewards**,” which is why it shines when **scalar outcome rewards** are cheap to compute but **token-level** value estimation is noisy or expensive relative to **multi-sampling**.

**Pitfalls (implementation, still on-topic):** if **all** rewards in a group are identical, \(\sigma \to 0\) and you must **skip** the step or rely on **\(\epsilon\)**—otherwise advantages explode. If **\(G\)** is too small, group baselines are **high variance**. If **reward** is too sparse (always zero), learning stalls unless **exploration** or **curriculum** improves hit rate.

## Interview Importance
Expect **GRPO vs PPO**, **why verifiable rewards matter**, **what R1-Zero proves vs what R1 adds**, and **when distillation replaces RL**. This paper is a frequent anchor for **open reasoning models**, **long CoT**, and **RL without human preference labels** for math-like tasks.

Interviewers often probe whether you understand **where the advantage comes from** (group vs critic), whether you can **trade off compute** (\(G\) rollouts per prompt) versus **sample efficiency**, and how **KL** interacts with **reward hacking** on long outputs. Be ready to connect **emergent “Aha”** behavior to **monitoring** (entropy, length, language mix) and to explain why **cold-start** data is a **product-quality** choice, not a concession that RL “failed.”

## Interview Questions & Answers

### Q1: How does GRPO differ from PPO in terms of baseline and model components?
**Answer:** PPO typically trains a **value network** \(V(s)\) (or uses GAE) to estimate advantages and reduce variance. GRPO samples **\(G\)** completions per prompt and sets advantages from **group-relative** statistics: \(\hat{A}_i = (r_i - \mu)/\sigma\). That **replaces the critic’s baseline** with a **multi-sample baseline** for the same input, which fits settings where **verifiable scalar rewards** exist and multiple rollouts are affordable. You still keep **PPO-like clipping** on the policy ratio for stability; the main structural difference is **no critic head** and **per-prompt Monte Carlo groups** instead of **temporal credit assignment** via GAE.

### Q2: Why might R1-Zero’s “no SFT” setup be scientifically important?
**Answer:** It supports the claim that **reasoning-like behavior** can emerge from **RL alone** on a strong base, implying such behavior is **not only** taught by imitation data—it can be **incentivized** once rewards and exploration are right. It reframes “reasoning” partly as **policy optimization** over latent capabilities. Practically, it also shows which **failure modes** appear without demonstration data (format drift, mixing), which motivates **R1’s** cold-start phase.

### Q3: What is the role of verifiable rewards in stabilizing RL for LLMs?
**Answer:** Verifiable rewards (e.g., **correct/incorrect** math, **pass/fail** tests) give **objective**, low-disagreement feedback compared to human rankings. That reduces **reward hacking ambiguity** and **label noise**, which helps **credit assignment** when optimizing long generations. The interviewer may follow up: **preference RLHF** can disagree on “better reasoning,” whereas **binary correctness** is crisp—at the cost of **only** rewarding tasks that can be checked automatically.

### Q4: When would you distill from R1 instead of running RL on the student?
**Answer:** Distillation (SFT on teacher traces) is attractive when you want **cheaper training**, **smaller models**, or **faster iteration**, and when **imitation** of high-quality reasoning traces suffices. RL on the student may still help if there is **distribution shift**, **new domains**, or **reward shaping** not captured by static traces—but many distill setups **skip student RL** if SFT quality is high. A strong follow-up: **distillation** can copy **style and steps** but may not transfer **robustness** unless the student data is **diverse** or augmented with **hard negatives**.

### Q5: What is the “Aha moment” and why does it matter for debugging RL runs?
**Answer:** It refers to **sudden** improvements in reasoning metrics during training—**non-smooth** jumps. For debugging, it warns against assuming **linear progress**; monitoring **emergent phases**, **entropy**, and **format health** matters, and early instability may **precede** rapid gains. Operationally, you might **checkpoint** around suspicious valleys and **ablate** whether reward definition or **KL \(\beta\)** caused the jump.

### Q6: Why does DeepSeek-R1 use cold-start SFT if R1-Zero works without it?
**Answer:** Pure RL can produce **undesirable artifacts** (e.g., **language mixing**, **repetition**, **incoherent structure**) before the policy becomes useful. A **small** set of **long CoT** demonstrations **anchors** output format and language, improving **stability** and **usability** while GRPO refines reasoning. In interviews, emphasize **R1-Zero** as existence proof and **R1** as **production-friendly** refinement—both can be true.

## Connections to Other Papers

**InstructGPT** established **RLHF** with **PPO** on human preferences; DeepSeek-R1 inherits the **clipped surrogate** but swaps **preference models** for **verifiable task rewards** and **group-relative** advantages—better aligned with math/code where **ground truth** exists.

**Chain-of-Thought** work showed that **eliciting** intermediate steps helps at inference; R1 trains models to **produce** long reasoning **as a policy** and then **distills** that behavior to smaller models—moving from **prompting** to **training**.

**DeepSeek-V3** is the **base**: R1’s sample efficiency and ceiling depend on **pretraining** quality; weak bases may need more **SFT** or **curriculum** before RL pays off.

**Kimi k1.5** and similar projects explore **alternative RL recipes** (data mix, reward shaping, rollout strategies); compare **stability**, **cost**, and **evaluation** when discussing “reasoning RL” landscape.

**Constitutional AI** and related methods emphasize **principles** and **preference optimization**; R1 complements that thread by showing **objective rewards** can dominate when tasks are **checkable**—different safety and capability tradeoffs.

For **system-design** interviews, use this paper when discussing **why** you might prefer **automated evaluators** (unit tests, symbolic checkers) in RL loops, how **test-time scaling** (sampling **\(G\)** solutions) interacts with **latency budgets**, and how **teacher–student** deployment cuts cost for **on-device** assistants.

| Paper / line of work | Connection |
|----------------------|------------|
| **InstructGPT (RLHF / PPO)** | Shared **clipped policy** DNA; DeepSeek-R1 shifts emphasis to **verifiable task rewards** and **group-relative** advantages instead of **human preference** modeling. |
| **Chain-of-Thought prompting & CoT literature** | R1 operationalizes **long CoT** as a **trainable behavior** via RL and distillation, not only as a **prompting** trick. |
| **DeepSeek-V3 (base model)** | R1 builds on a **strong base**; capability floor and **pretraining quality** matter for RL sample efficiency. |
| **Kimi k1.5 (and related RL reasoning work)** | Alternative **RL stacks** and data recipes; compare **reward design**, **rollout budgets**, and **stability** tradeoffs. |
| **Constitutional AI / preference optimization** | Overlaps in **KL-to-reference** and **safety** framing; R1 highlights **objective verifiable** rewards for **math/code** versus **principle-based** preference training. |
| **[GLM-5](32_glm5.md) (Slime async RL)** | Extends RL-for-reasoning to **agentic coding**; Slime's async architecture solves GPU idle-time problems that synchronous GRPO shares when rollouts involve tool calls. |
| **[Kimi K2.5](33_kimi_k2_5.md) (PARL)** | Multi-agent RL with speedup rewards; contrasts with R1's single-agent reasoning — both use RL but optimize different capability profiles. |

## Key Takeaways for Quick Review

| Topic | One-liner |
|-------|-----------|
| **R1-Zero** | RL-only reasoning elicitation **without** SFT on CoT; shows latent capability can be **drawn out** by RL. |
| **GRPO** | Sample **\(G\)** answers; advantages **\((r_i-\mu)/\sigma\)**; **clipped** ratios like PPO; **no critic**. |
| **Rewards** | Prefer **verifiable** signals (math/code) for **stable** learning vs noisy preferences. |
| **R1 vs Zero** | **Cold-start SFT** mitigates **degenerate** outputs and stabilizes long CoT before RL. |
| **Distillation** | Students often trained on **R1 traces** (SFT); RL **optional** depending on goals. |
| **Emergence** | Watch for **“Aha”** phase transitions—**nonlinear** quality jumps during training. |
| **KL penalty** | Keeps **\(\pi_\theta\)** near **\(\pi_{\mathrm{ref}}\)** to reduce collapse and weird exploits. |
| **Compute tradeoff** | Larger **\(G\)** improves group baseline quality but costs **inference** per step. |
| **Evaluators** | Automate **correctness** checks where possible—reduces preference noise and **reward hacking** surface. |
| **Deployment** | **Distill** to small models for **latency**; keep **teacher** for **hard** queries or **verification**. |

