# InstructGPT: Training Language Models to Follow Instructions with Human Feedback

**Authors:** Long Ouyang, Jeff Wu, Xu Jiang, and 15 more  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** NeurIPS  
**Link:** [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)

---

## TL;DR

InstructGPT aligns GPT-3 with human intent using a three-stage pipeline: (1) **Supervised Fine-Tuning (SFT)** on human-written demonstrations, (2) **Reward Model (RM)** training from human preference rankings, and (3) **Proximal Policy Optimization (PPO)** to optimize the policy against the reward while penalizing deviation from the reference model via a KL penalty. The result: smaller InstructGPT models (1.3B) are preferred by humans over the much larger GPT-3 (175B).

---

## Why This Paper Matters

This paper defined the **RLHF pipeline** that powers ChatGPT, Claude, and most production LLMs. It's the canonical reference for:

1. **The alignment problem:** Why base LMs optimize token prediction, not user utility
2. **The three-stage pipeline:** SFT → RM → PPO (or variants like DPO)
3. **Reward hacking:** Why naive RL on reward models can go wrong
4. **The KL penalty:** How to stay close to the base model's capabilities

Understanding this paper is essentially mandatory for any LLM interview.

---

## Key Concepts Explained Simply

### Stage 1: Supervised Fine-Tuning (SFT)

Human labelers write high-quality responses to prompts. The model is fine-tuned on these demonstrations using standard next-token prediction loss. This gives the model a basic ability to follow instructions.

**Think of it as:** Teaching by example — "Here's what a good response looks like."

### Stage 2: Reward Model (RM)

For each prompt, the SFT model generates multiple responses. Human labelers **rank** these responses from best to worst. A reward model is trained to predict these rankings — it takes (prompt, response) and outputs a scalar reward score.

**Think of it as:** Training a "judge" that can score responses, so you don't need humans for every evaluation.

### Stage 3: PPO (Reinforcement Learning)

The SFT model is further trained using PPO to maximize the reward model's score, while staying close to the original SFT model (KL penalty). This is where the model learns to produce responses that humans prefer.

**Think of it as:** The model practices generating responses, gets scored by the judge, and improves — but isn't allowed to deviate too far from its original behavior.

### Why KL Penalty?

Without KL penalty, the model would find **degenerate** high-reward outputs — repetitive, sycophantic, or adversarial text that exploits weaknesses in the reward model. The KL penalty keeps the model close to the reference policy, preserving its general language abilities.

---

## The Math — Explained Step by Step

### Stage 1: SFT Loss

\[
\mathcal{L}_{\text{SFT}} = -\sum_{t} \log P_\theta(y_t \mid x, y_{<t})
\]

Standard next-token prediction on human demonstrations \((x, y)\).

### Stage 2: Reward Model Loss

Given a prompt \(x\) and two responses \(y_w\) (preferred) and \(y_l\) (rejected):

\[
\mathcal{L}_{\text{RM}} = -\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
\]

This is the **Bradley-Terry** model: the probability that \(y_w\) is preferred equals \(\sigma(r_w - r_l)\). The reward model learns to assign higher scores to preferred responses.

### Stage 3: PPO Objective

\[
\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_\theta(\cdot|x)} \left[r_\phi(x, y) - \beta \cdot \text{KL}\big(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\big)\right]
\]

**Breaking it down:**

1. **\(r_\phi(x, y)\):** Reward model score — "how good is this response?"
2. **\(\beta \cdot \text{KL}(\cdot)\):** Penalty for diverging from the reference model — prevents reward hacking
3. **\(\pi_\theta\):** The policy (current model) being optimized
4. **\(\pi_{\text{ref}}\):** The reference model (SFT checkpoint) — the anchor

The KL divergence per token:

\[
\text{KL} = \sum_t \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}
\]

### PPO Clipped Objective

Within each update step, PPO clips the policy ratio to prevent large updates:

\[
L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t \cdot A_t, \; \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\right)\right]
\]

where \(r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)\) is the probability ratio and \(A_t\) is the advantage.

---

## Python Implementation

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sft_loss(logits, target_ids):
    """Stage 1: Supervised fine-tuning loss (next-token prediction)."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = z - np.log(np.sum(np.exp(z), axis=-1, keepdims=True))

    loss = 0.0
    for t, tid in enumerate(target_ids):
        loss -= log_probs[t, tid]
    return loss / len(target_ids)


def reward_model_loss(reward_chosen, reward_rejected):
    """
    Stage 2: Bradley-Terry preference loss.
    Train reward model so r(chosen) > r(rejected).
    """
    return -np.log(sigmoid(reward_chosen - reward_rejected) + 1e-12)


def train_reward_model(comparisons, reward_fn):
    """
    comparisons: list of (prompt, chosen_response, rejected_response)
    reward_fn: function that returns a scalar reward
    """
    total_loss = 0.0
    correct = 0
    for prompt, chosen, rejected in comparisons:
        r_w = reward_fn(prompt, chosen)
        r_l = reward_fn(prompt, rejected)
        total_loss += reward_model_loss(r_w, r_l)
        if r_w > r_l:
            correct += 1
    accuracy = correct / len(comparisons)
    avg_loss = total_loss / len(comparisons)
    return avg_loss, accuracy


def kl_penalty(logp_policy, logp_reference):
    """
    Per-token KL divergence between policy and reference model.
    logp_policy: log probs from current policy [seq_len]
    logp_reference: log probs from reference/SFT model [seq_len]
    """
    return np.sum(logp_policy - logp_reference)


def rlhf_objective(reward, logp_policy, logp_reference, beta=0.1):
    """
    Stage 3: RLHF objective = reward - beta * KL.
    We MAXIMIZE this (so negate for loss).
    """
    kl = kl_penalty(logp_policy, logp_reference)
    return reward - beta * kl


def ppo_clip_loss(advantages, ratios, epsilon=0.2):
    """
    PPO clipped surrogate loss.
    advantages: [batch_size] — estimated advantage per sample
    ratios: [batch_size] — pi_new / pi_old probability ratios
    """
    clipped = np.clip(ratios, 1 - epsilon, 1 + epsilon)
    loss = -np.mean(np.minimum(ratios * advantages, clipped * advantages))
    return loss


class SimpleRLHFPipeline:
    """End-to-end demonstration of the three-stage pipeline."""

    def __init__(self, beta=0.1):
        self.beta = beta

    def stage1_sft(self, demonstrations):
        """Supervised fine-tuning on human demonstrations."""
        print("Stage 1: SFT")
        print(f"  Training on {len(demonstrations)} demonstrations")
        for i, (prompt, response) in enumerate(demonstrations[:3]):
            print(f"  Example {i+1}: '{prompt}' → '{response}'")
        return "sft_model"

    def stage2_reward_model(self, comparisons):
        """Train reward model on human preferences."""
        print("\nStage 2: Reward Model")
        print(f"  Training on {len(comparisons)} comparisons")
        for prompt, chosen, rejected in comparisons[:2]:
            print(f"  Prompt: '{prompt}'")
            print(f"    Chosen: '{chosen}'")
            print(f"    Rejected: '{rejected}'")
        return "reward_model"

    def stage3_ppo(self, n_steps=5):
        """PPO optimization against reward model."""
        print(f"\nStage 3: PPO ({n_steps} steps)")
        for step in range(n_steps):
            reward = np.random.uniform(0.3, 0.9) + step * 0.05
            kl = np.random.uniform(0.01, 0.1) + step * 0.02
            objective = reward - self.beta * kl
            print(f"  Step {step+1}: reward={reward:.3f}, "
                  f"KL={kl:.3f}, objective={objective:.3f}")
        return "aligned_model"


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    pipeline = SimpleRLHFPipeline(beta=0.1)

    demos = [
        ("What is Python?", "Python is a high-level programming language..."),
        ("Explain gravity.", "Gravity is a fundamental force..."),
        ("Write a haiku.", "Autumn moonlight—\na worm digs silently\ninto the chestnut."),
    ]
    pipeline.stage1_sft(demos)

    comparisons = [
        ("What is 2+2?", "2+2 equals 4.", "The answer is fish."),
        ("Explain ML.", "ML is a subset of AI...", "ML stands for something."),
    ]
    pipeline.stage2_reward_model(comparisons)
    pipeline.stage3_ppo(n_steps=5)

    # Reward model loss example
    print("\n--- Reward Model Math ---")
    r_chosen, r_rejected = 2.5, 1.0
    loss = reward_model_loss(r_chosen, r_rejected)
    prob = sigmoid(r_chosen - r_rejected)
    print(f"r(chosen)={r_chosen}, r(rejected)={r_rejected}")
    print(f"P(chosen > rejected) = {prob:.4f}")
    print(f"Loss = {loss:.4f}")

    # KL penalty example
    print("\n--- KL Penalty ---")
    logp_policy = np.array([-2.1, -1.5, -3.0, -0.8])
    logp_ref = np.array([-2.0, -1.6, -2.8, -0.9])
    kl = kl_penalty(logp_policy, logp_ref)
    print(f"KL divergence: {kl:.4f}")
```

---

## Interview Importance

InstructGPT/RLHF is a **top-3 most important** topic in LLM interviews. You must be able to explain the pipeline, draw it on a whiteboard, and discuss failure modes.

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: Draw the three-stage pipeline and name failure modes at each stage.

**Answer:**

**Pipeline:** Base LM → SFT → RM → PPO → Aligned Model

**Failure modes:**
1. **SFT stage:** If demonstrations are low-quality, biased, or inconsistent, the model inherits those problems. Labeler disagreement introduces noise.
2. **RM stage:** The reward model can learn **spurious correlations** (e.g., longer responses = higher reward). It may not generalize to out-of-distribution prompts. Labeler biases get baked in.
3. **PPO stage:** **Reward hacking** — the model finds adversarial outputs that score high on the RM but are low quality (e.g., repeating positive phrases). **KL collapse** if beta is too low. **Mode collapse** if beta is too high (model barely moves from SFT).

### Q2: Why include a KL penalty — what goes wrong without it?

**Answer:** Without KL penalty, the model drifts far from the reference policy to maximize the (imperfect) reward model. This causes:
- **Reward hacking:** Exploiting RM weaknesses with degenerate outputs
- **Loss of capabilities:** The model forgets general language abilities
- **Sycophancy:** Learning to say what the RM rewards (often agreement) rather than being truthful
- **Repetition/degeneracy:** Finding high-reward fixed points that don't generalize

The KL penalty acts as a **trust region** — the model can improve within a neighborhood of the reference policy but can't go too far.

### Q3: Compare RLHF to DPO at a high level.

**Answer:**
- **RLHF (PPO):** Three separate components — SFT model, reward model, PPO optimizer. Requires sampling from the policy, estimating advantages, and clipping updates. Complex to implement and tune.
- **DPO (Direct Preference Optimization):** Eliminates the reward model entirely. Derives a **closed-form** objective that directly optimizes the policy from preference pairs. The implicit reward is \(r(x,y) = \beta \log[\pi_\theta(y|x)/\pi_{\text{ref}}(y|x)]\).
- **Trade-offs:** DPO is simpler and more stable, but RLHF with PPO can be more flexible (e.g., online data collection, reward model reuse). DPO's quality depends on having good offline preference data.

### Q4: What is reward hacking and how do you detect it?

**Answer:** Reward hacking occurs when the policy finds outputs that score highly on the reward model but are actually poor quality. Detection methods:
1. **Monitor KL divergence:** If KL grows rapidly, the model is drifting too far
2. **Human eval:** Periodically evaluate samples with humans — if RM score increases but human preference doesn't, there's hacking
3. **Reward model consistency:** Check if the RM's rankings agree with held-out human preferences
4. **Diversity metrics:** Monitor output diversity — reward hacking often reduces it

### Q5: Why can't you just do more SFT instead of RLHF?

**Answer:** SFT teaches the model to **imitate** demonstrations, but:
1. Demonstrations only cover a fraction of possible prompts
2. The model can't learn from its own mistakes — it only sees "good" examples
3. SFT doesn't teach **preferences** (which of two good responses is better)
4. RLHF can optimize for outcomes that are hard to demonstrate but easy to evaluate
5. However, recent work shows that **high-quality SFT** with enough data can approach RLHF quality, especially with rejection sampling

---

## Connections to Other Papers

- **GPT-3** → InstructGPT aligns GPT-3 with human preferences
- **Constitutional AI** → Alternative: use AI feedback instead of human feedback
- **FLAN** → SFT-only approach to instruction following
- **DPO** → Simplifies RLHF by eliminating the separate reward model
- **Chain-of-Thought** → RLHF can be applied to reasoning traces

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Pipeline | SFT → Reward Model → PPO (three stages) |
| SFT | Fine-tune on human demonstrations |
| RM | Bradley-Terry model on preference pairs |
| PPO | Maximize reward - β × KL(policy ∥ reference) |
| KL penalty | Prevents reward hacking and capability loss |
| Key insight | 1.3B InstructGPT preferred over 175B GPT-3 |
| Evolution | RLHF → DPO → RLAIF (Constitutional AI) |
