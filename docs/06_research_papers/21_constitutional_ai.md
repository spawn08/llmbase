# Constitutional AI: Harmlessness from AI Feedback

**Authors:** Yuntao Bai, Saurav Kadavath, Sandipan Kundu, and 15 more  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)

---

## TL;DR

Constitutional AI (CAI) trains models using a set of **principles** (a "constitution") to critique and revise their own responses, enabling **scalable alignment** without as much human labeling. The pipeline: (1) Generate responses, (2) Use principles to critique and revise them, (3) Train a preference model on these AI-generated comparisons (**RLAIF**), (4) Optimize with RL. The result: improved harmlessness with reduced human oversight cost.

---

## Why This Paper Matters

CAI offers an alternative to pure RLHF that scales better:

1. **Reduces human labeling cost:** AI generates preference data instead of human labelers
2. **Explicit principles:** The constitution makes alignment goals transparent and auditable
3. **RLAIF:** RL from AI feedback as a complement or alternative to human feedback
4. **Addresses over-refusal:** Principles can include "be helpful" alongside "be harmless"
5. **Foundation for Claude:** CAI is Anthropic's core alignment approach

---

## Key Concepts Explained Simply

### The Constitution

A set of natural language principles that define desired behavior:

- "Choose the response that is most helpful, while being harmless and honest."
- "Choose the response that sounds most similar to what a thoughtful, senior employee at a company would say."
- "Choose the response that is least likely to be harmful or offensive."

These principles guide the AI in generating preference pairs — no need for humans to rank every response.

### The Pipeline

1. **Red-teaming:** Generate adversarial prompts designed to elicit harmful responses
2. **Initial response:** The model generates a (potentially harmful) response
3. **Critique:** The model critiques its own response using constitutional principles
4. **Revision:** The model revises its response to address the critique
5. **SL-CAI:** Fine-tune on (prompt, revised response) pairs
6. **RL-CAI:** Train a preference model on (revised, original) pairs, then do RL

### RLAIF (RL from AI Feedback)

Instead of humans ranking responses, the **model itself** ranks them using the constitution:
- Given two responses, ask the model: "According to principle X, which response is better?"
- Use these AI-generated preferences to train a reward model
- Run RL (PPO) against this reward model

---

## The Math — Explained Step by Step

### Bradley-Terry Preference Model

Given preferred response \(y_w\) and rejected \(y_l\):

\[
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)
\]

where \(\sigma\) is the sigmoid function. Same as InstructGPT, but the preference labels come from **AI feedback** guided by constitutional principles.

### Critique-Revision Process

For response \(y_0\) to prompt \(x\):

\[
\text{critique } c = P_\theta(\cdot \mid x, y_0, \text{principle}_k)
\]
\[
\text{revision } y_1 = P_\theta(\cdot \mid x, y_0, c, \text{"Please revise..."})
\]

Multiple critique-revision rounds can be applied iteratively.

### RLAIF Objective

Same as RLHF but with AI-labeled preferences:

\[
\max_{\pi_\theta} \; \mathbb{E}\left[r_\phi^{\text{AI}}(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})\right]
\]

where \(r_\phi^{\text{AI}}\) is trained on AI-generated preference pairs.

---

## Python Implementation

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


CONSTITUTION = [
    "Choose the response that is most helpful while being harmless.",
    "Choose the response that is most respectful and honest.",
    "Choose the response that avoids harmful stereotypes or biases.",
    "Choose the response that a thoughtful, senior employee would give.",
    "Choose the response that is least likely to cause real-world harm.",
]


def critique_response(prompt, response, principle):
    """
    Simulate the model critiquing its own response using a principle.
    In practice, this is another LLM call.
    """
    critique_prompt = (
        f"Prompt: {prompt}\n"
        f"Response: {response}\n\n"
        f"Principle: {principle}\n\n"
        f"Critique this response according to the principle above. "
        f"Identify any issues:"
    )
    return critique_prompt


def revise_response(prompt, response, critique):
    """
    Simulate the model revising its response based on the critique.
    """
    revision_prompt = (
        f"Prompt: {prompt}\n"
        f"Original response: {response}\n"
        f"Critique: {critique}\n\n"
        f"Please revise the response to address the critique "
        f"while remaining helpful:"
    )
    return revision_prompt


def generate_ai_preference(response_a, response_b, principle):
    """
    Simulate AI-generated preference using a constitutional principle.
    Returns which response is preferred.
    """
    comparison_prompt = (
        f"Principle: {principle}\n\n"
        f"Response A: {response_a}\n"
        f"Response B: {response_b}\n\n"
        f"According to the principle, which response is better? "
        f"Answer A or B."
    )
    # Simulate: revised responses are usually preferred
    return comparison_prompt


class ConstitutionalAIPipeline:
    """Simplified CAI training pipeline."""

    def __init__(self, constitution=None):
        self.constitution = constitution or CONSTITUTION

    def critique_and_revise(self, prompt, response, n_rounds=2):
        """Apply iterative critique-revision."""
        current_response = response
        history = [{"round": 0, "response": current_response}]

        for round_num in range(1, n_rounds + 1):
            principle = self.constitution[round_num % len(self.constitution)]

            critique = f"[Critique using: '{principle[:50]}...']"
            revised = f"[Revised version of round {round_num}]"

            history.append({
                "round": round_num,
                "principle": principle,
                "critique": critique,
                "response": revised,
            })
            current_response = revised

        return history

    def generate_preferences(self, prompt, original, revised):
        """Generate AI preference pair for reward model training."""
        principle = np.random.choice(self.constitution)
        # In practice, an LLM evaluates both responses
        # Revised responses are typically preferred
        return {
            "prompt": prompt,
            "chosen": revised,
            "rejected": original,
            "principle_used": principle,
        }

    def train_reward_model(self, preferences):
        """Train reward model on AI-generated preferences."""
        print(f"Training reward model on {len(preferences)} AI-labeled pairs")
        # Bradley-Terry loss on AI preferences
        losses = []
        for pref in preferences:
            r_chosen = np.random.uniform(0.5, 2.0)
            r_rejected = np.random.uniform(-1.0, 0.5)
            loss = -np.log(sigmoid(r_chosen - r_rejected) + 1e-12)
            losses.append(loss)
        return np.mean(losses)


def compare_alignment_approaches():
    """Compare RLHF, CAI/RLAIF, and SFT-only approaches."""
    approaches = [
        {
            "name": "Pure SFT",
            "human_labels": "High (demonstrations)",
            "ai_labels": "None",
            "scalability": "Limited by human demonstration quality",
            "transparency": "Low (implicit in demos)",
        },
        {
            "name": "RLHF",
            "human_labels": "High (preferences)",
            "ai_labels": "None",
            "scalability": "Limited by human labeler availability",
            "transparency": "Low (implicit in preferences)",
        },
        {
            "name": "CAI/RLAIF",
            "human_labels": "Low (only constitution)",
            "ai_labels": "High (AI-generated preferences)",
            "scalability": "High (AI generates labels)",
            "transparency": "High (explicit constitutional principles)",
        },
    ]

    print("--- Alignment Approach Comparison ---")
    for a in approaches:
        print(f"\n  {a['name']}:")
        for k, v in a.items():
            if k != "name":
                print(f"    {k}: {v}")


def audit_constitution(constitution):
    """Framework for auditing a constitution for bias or gaps."""
    checks = {
        "helpfulness": any("helpful" in p.lower() for p in constitution),
        "harmlessness": any("harm" in p.lower() for p in constitution),
        "honesty": any("honest" in p.lower() for p in constitution),
        "bias": any("bias" in p.lower() or "stereotype" in p.lower() for p in constitution),
        "respect": any("respect" in p.lower() for p in constitution),
    }

    print("--- Constitution Audit ---")
    for dimension, present in checks.items():
        status = "✓ Covered" if present else "✗ Missing"
        print(f"  {dimension}: {status}")

    missing = [k for k, v in checks.items() if not v]
    if missing:
        print(f"\n  ⚠ Consider adding principles for: {', '.join(missing)}")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Constitution audit
    audit_constitution(CONSTITUTION)

    # Critique-revision pipeline
    print("\n--- Critique-Revision Pipeline ---")
    pipeline = ConstitutionalAIPipeline()

    prompt = "How do I pick a lock?"
    initial_response = "Here are the steps to pick a lock: ..."

    history = pipeline.critique_and_revise(prompt, initial_response, n_rounds=3)
    for entry in history:
        print(f"  Round {entry['round']}: {entry['response'][:60]}")
        if 'principle' in entry:
            print(f"    Principle: {entry['principle'][:60]}...")

    # Generate preferences
    print("\n--- AI-Generated Preferences ---")
    preferences = []
    for i in range(5):
        pref = pipeline.generate_preferences(
            f"Prompt {i}", f"Original {i}", f"Revised {i}"
        )
        preferences.append(pref)
        print(f"  Pair {i+1}: chosen='{pref['chosen']}', "
              f"rejected='{pref['rejected']}'")

    # Reward model training
    print()
    avg_loss = pipeline.train_reward_model(preferences)
    print(f"  Average reward model loss: {avg_loss:.4f}")

    # Comparison
    print()
    compare_alignment_approaches()
```

---

## Interview Importance

CAI is important for understanding **scalable alignment** and the debate between human feedback vs. AI feedback.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Compare RLHF, RLAIF, and constitutional critique steps.

**Answer:**
- **RLHF:** Humans rank model outputs → train reward model → RL. Gold standard but expensive and slow.
- **RLAIF:** AI ranks model outputs using principles → train reward model → RL. Cheaper, scalable, but depends on AI judgment quality.
- **Constitutional critique:** Model critiques and revises its own outputs using principles → creates (original, revised) pairs for SFT and RL. Most automated approach.

The progression: human labels (expensive) → AI labels (cheap) → self-improvement (cheapest).

### Q2: What failure modes appear from AI-only feedback?

**Answer:**
1. **Circular preferences:** The AI might prefer responses that are similar to its own style, not objectively better
2. **Sycophancy amplification:** If the base model is sycophantic, AI feedback may reinforce this
3. **Principle gaps:** If the constitution doesn't cover a failure mode, it won't be corrected
4. **Overoptimization:** Training against AI preferences too aggressively can lead to responses that "game" the AI evaluator
5. **Capability ceiling:** AI feedback can't teach the model to be better than the model generating the feedback

### Q3: How would you audit a constitution for bias or over-blocking?

**Answer:**
1. **Dimension coverage:** Check if the constitution addresses helpfulness, harmlessness, honesty, fairness, and domain-specific concerns
2. **Red-teaming:** Test edge cases where principles conflict (e.g., helpful instruction that could be misused)
3. **Over-refusal analysis:** Measure how often the model refuses benign requests due to overly cautious principles
4. **Demographic analysis:** Test whether the constitution causes different treatment of different user groups
5. **A/B testing:** Compare models trained with different constitutions on real user interactions
6. **Principle interaction:** Check if principles can contradict each other and define a priority order

---

## Connections to Other Papers

- **InstructGPT** → CAI extends RLHF with AI-generated preferences
- **FLAN** → SFT component similar to FLAN-style instruction tuning
- **Chain-of-Thought** → Critique-revision uses CoT-like reasoning
- **GPT-3** → Base model capabilities enable the self-critique process

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Use principles (constitution) to generate alignment data |
| Pipeline | Generate → Critique → Revise → AI Preferences → RL |
| RLAIF | RL from AI Feedback — AI labels preferences, not humans |
| Constitution | Explicit natural-language principles for desired behavior |
| Key advantage | Scalable alignment without massive human labeling |
| Risk | Circular preferences, principle gaps, overoptimization |
