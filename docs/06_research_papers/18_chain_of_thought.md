# Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

**Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** NeurIPS  
**Link:** [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)

---

## TL;DR

**Chain-of-thought (CoT)** prompting appends intermediate reasoning steps before the final answer — either via few-shot examples with rationales or the zero-shot trigger "Let's think step by step." Large models show dramatic gains on math, commonsense, and symbolic reasoning tasks where direct answers fail. The effect is **emergent** — it helps large models significantly but barely affects small ones.

---

## Why This Paper Matters

CoT is the default approach for **reasoning** tasks with LLMs:

1. **Simple yet powerful:** Just adding "Let's think step by step" can improve accuracy by 20-40% on math tasks
2. **Foundation for agents:** CoT traces are the "Thought" component in ReAct agents
3. **Self-consistency:** Sampling multiple CoT paths and voting on the answer further improves reliability
4. **Verifiers and RL:** CoT traces can be scored and used for reward-based training
5. **Universal adoption:** Every major LLM provider uses CoT internally and in prompts

---

## Key Concepts Explained Simply

### Why Direct Answers Fail

Consider: "Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many does he have now?"

- **Direct answer:** The model might say "11" (wrong — it's 5 + 2×3 = 11, actually correct, but for harder problems, models fail without intermediate steps)
- **With CoT:** "Roger starts with 5 balls. He buys 2 cans with 3 balls each, so 2 × 3 = 6 new balls. Total: 5 + 6 = 11."

For complex problems, the model needs to "show its work" — decompose the problem into manageable steps.

### Few-Shot CoT

Provide examples that include reasoning steps:

```
Q: Tom has 3 apples. He gives away 1 and buys 5 more. How many?
A: Tom starts with 3 apples. He gives away 1, leaving 3-1=2. He buys 5 more: 2+5=7. The answer is 7.

Q: [Your question]
A:
```

### Zero-Shot CoT

Just append "Let's think step by step." to the prompt — no examples needed. Surprisingly effective.

### Self-Consistency

Instead of generating one CoT path, generate **multiple** (e.g., 40) with temperature > 0, extract the final answer from each, and take the **majority vote**. Different reasoning paths may make different errors, but the correct answer appears most often.

---

## The Math — Explained Step by Step

### CoT as Latent Variable

A chain-of-thought \(z\) is a latent intermediate computation between question \(x\) and answer \(a\):

\[
P_\theta(a \mid x) = \sum_z P_\theta(z \mid x) \cdot P_\theta(a \mid x, z)
\]

**Breaking it down:**

1. \(P_\theta(z \mid x)\): Probability of generating reasoning trace \(z\) given the question
2. \(P_\theta(a \mid x, z)\): Probability of the final answer given both the question and the reasoning
3. Marginalizing over all possible \(z\) is intractable — we sample instead

### Self-Consistency

Sample \(K\) reasoning chains \(z_1, \ldots, z_K\) and extract answers \(a_1, \ldots, a_K\):

\[
\hat{a} = \arg\max_a \sum_{k=1}^{K} \mathbf{1}[a_k = a]
\]

Majority vote over sampled answers. This approximates marginalizing over reasoning paths.

### Why CoT Helps (Computational Perspective)

From a complexity standpoint, Transformer forward passes compute a fixed-depth circuit. CoT effectively increases the computational depth:

- **Without CoT:** \(O(L)\) serial computation (fixed number of layers)
- **With CoT of length \(T\):** \(O(L \times T)\) effective computation — each generated token adds another full forward pass

This lets the model "think for longer" on harder problems.

---

## Python Implementation

```python
import numpy as np
from collections import Counter


def zero_shot_cot(question):
    """Zero-shot CoT: append the magic phrase."""
    return f"Q: {question}\nA: Let's think step by step.\n"


def few_shot_cot(question, examples):
    """
    Few-shot CoT: provide examples with reasoning traces.
    examples: list of (question, reasoning, answer) tuples
    """
    prompt = ""
    for q, reasoning, ans in examples:
        prompt += f"Q: {q}\n"
        prompt += f"A: {reasoning} The answer is {ans}.\n\n"
    prompt += f"Q: {question}\nA:"
    return prompt


def extract_answer(cot_response):
    """Extract the final answer from a CoT response."""
    markers = ["the answer is", "therefore", "so the answer is", "= "]
    response_lower = cot_response.lower()

    for marker in markers:
        if marker in response_lower:
            idx = response_lower.rfind(marker) + len(marker)
            answer = cot_response[idx:].strip().rstrip(".")
            try:
                return float(answer)
            except ValueError:
                return answer
    return cot_response.strip().split()[-1]


def self_consistency(question, generate_fn, n_samples=10, temperature=0.7):
    """
    Self-consistency: sample multiple CoT paths, majority vote on answer.
    generate_fn: function that takes a prompt and returns a response
    """
    answers = []
    traces = []

    for _ in range(n_samples):
        prompt = zero_shot_cot(question)
        response = generate_fn(prompt, temperature=temperature)
        answer = extract_answer(response)
        answers.append(answer)
        traces.append(response)

    # Majority vote
    counter = Counter(answers)
    best_answer, best_count = counter.most_common(1)[0]
    confidence = best_count / n_samples

    return {
        "answer": best_answer,
        "confidence": confidence,
        "vote_distribution": dict(counter),
        "n_samples": n_samples,
    }


def simulate_cot_accuracy(model_size_B, use_cot=False, task="math"):
    """
    Simulate the emergence of CoT: large models benefit, small ones don't.
    """
    base_accuracies = {
        "math": {0.3: 0.05, 1: 0.08, 7: 0.15, 13: 0.25, 70: 0.40, 175: 0.50, 540: 0.58},
        "commonsense": {0.3: 0.30, 1: 0.40, 7: 0.55, 13: 0.62, 70: 0.72, 175: 0.78, 540: 0.82},
    }

    cot_boosts = {
        "math": {0.3: 0.00, 1: 0.01, 7: 0.05, 13: 0.12, 70: 0.25, 175: 0.30, 540: 0.35},
        "commonsense": {0.3: 0.00, 1: 0.01, 7: 0.03, 13: 0.08, 70: 0.12, 175: 0.14, 540: 0.15},
    }

    sizes = list(base_accuracies[task].keys())
    closest = min(sizes, key=lambda s: abs(s - model_size_B))

    base = base_accuracies[task][closest]
    if use_cot:
        return base + cot_boosts[task][closest]
    return base


def cot_cost_analysis(prompt_tokens, answer_tokens_direct, answer_tokens_cot,
                      cost_per_1k_tokens=0.002):
    """Compare token costs of direct vs CoT responses."""
    direct_total = prompt_tokens + answer_tokens_direct
    cot_total = prompt_tokens + answer_tokens_cot

    return {
        "direct_tokens": direct_total,
        "cot_tokens": cot_total,
        "overhead_ratio": cot_total / direct_total,
        "direct_cost": direct_total / 1000 * cost_per_1k_tokens,
        "cot_cost": cot_total / 1000 * cost_per_1k_tokens,
    }


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Zero-shot CoT
    print("--- Zero-Shot CoT ---")
    print(zero_shot_cot("If a shirt costs $25 and is on 20% sale, what do you pay?"))

    # Few-shot CoT
    print("\n--- Few-Shot CoT ---")
    examples = [
        ("A book costs $15. Tax is 10%. What's the total?",
         "The book is $15. Tax is 10% of 15 = $1.50. Total = 15 + 1.50 = $16.50.",
         "$16.50"),
        ("You have 3 bags with 4 marbles each, plus 2 loose marbles. How many total?",
         "3 bags × 4 marbles = 12 marbles. Plus 2 loose = 12 + 2 = 14.",
         "14"),
    ]
    print(few_shot_cot(
        "A restaurant bill is $80. You tip 15%. What's the total?",
        examples
    ))

    # Emergence across model sizes
    print("\n--- CoT Emergence (Math Accuracy by Model Size) ---")
    print(f"{'Model Size':>12} {'Direct':>10} {'With CoT':>10} {'Boost':>10}")
    print("-" * 45)
    for size in [0.3, 1, 7, 13, 70, 175, 540]:
        direct = simulate_cot_accuracy(size, use_cot=False, task="math")
        cot = simulate_cot_accuracy(size, use_cot=True, task="math")
        boost = cot - direct
        print(f"{size:>10.1f}B {direct:>10.0%} {cot:>10.0%} {'+' + f'{boost:.0%}':>10}")

    # Cost analysis
    print("\n--- CoT Cost Overhead ---")
    cost = cot_cost_analysis(
        prompt_tokens=200,
        answer_tokens_direct=10,
        answer_tokens_cot=150,
        cost_per_1k_tokens=0.002
    )
    print(f"Direct: {cost['direct_tokens']} tokens (${cost['direct_cost']:.4f})")
    print(f"CoT: {cost['cot_tokens']} tokens (${cost['cot_cost']:.4f})")
    print(f"Overhead: {cost['overhead_ratio']:.1f}× more tokens")

    # Self-consistency simulation
    print("\n--- Self-Consistency Simulation ---")
    def mock_generate(prompt, temperature=0.7):
        correct_prob = 0.6
        if np.random.random() < correct_prob:
            return "Step 1: ... Step 2: ... The answer is 42."
        else:
            return f"Step 1: ... The answer is {np.random.choice([40, 41, 43, 44])}."

    result = self_consistency("What is 6×7?", mock_generate, n_samples=20)
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"Vote distribution: {result['vote_distribution']}")
```

---

## Interview Importance

CoT is a **must-know** for any role involving LLM applications. It's the standard approach for reasoning tasks and the foundation for agent architectures.

### Difficulty Level: ⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: When does CoT hurt?

**Answer:**
1. **Simple tasks:** CoT adds verbosity without accuracy gain on easy questions (e.g., sentiment classification)
2. **Hallucinated reasoning:** The model may generate plausible-sounding but incorrect reasoning steps, leading to wrong answers with false confidence
3. **Latency and cost:** CoT generates 5-20× more tokens, increasing response time and API costs
4. **Small models:** Models below ~10B parameters often don't benefit from CoT and may even perform worse
5. **Unfaithful chains:** The reasoning may not reflect the model's actual computation — it might arrive at the answer through different internal mechanisms

### Q2: Explain self-consistency decoding at a high level.

**Answer:** Self-consistency samples **multiple** diverse reasoning paths (e.g., 40 chains) using temperature sampling, extracts the final answer from each path, and takes the **majority vote**. The intuition: there may be many different valid reasoning paths to the correct answer, but fewer paths to any specific wrong answer. Sampling diversity ensures different errors cancel out while correct answers reinforce each other. It typically improves accuracy by 5-15% over single-sample CoT.

### Q3: How would you evaluate reasoning besides final-answer accuracy?

**Answer:**
1. **Step-by-step verification:** Check each reasoning step for correctness (process reward models)
2. **Faithfulness:** Compare the reasoning trace to the model's attention patterns — does the trace reflect actual computation?
3. **Counterfactual testing:** If you change a number in the problem, does the reasoning correctly propagate the change?
4. **Error categorization:** Classify errors as arithmetic, logical, misunderstanding, or hallucination
5. **Human judgment:** Experts rate reasoning quality independent of final answer correctness

### Q4: How does CoT relate to verifiers and reward models?

**Answer:** CoT traces provide **inspectable intermediate work** that can be scored:
1. **Process Reward Models (PRMs):** Score each step individually, rewarding correct reasoning steps and penalizing incorrect ones
2. **Outcome Reward Models (ORMs):** Score only the final answer — simpler but less informative
3. **RL on reasoning:** Use step-level rewards to train models that produce better reasoning traces
4. **Best-of-N:** Generate N CoT traces, score each with a verifier, return the highest-scoring one (better than majority vote when the verifier is good)

---

## Connections to Other Papers

- **GPT-3** → CoT enhances GPT-3's few-shot capabilities with reasoning
- **ReAct** → CoT + tool use = agent reasoning
- **FLAN** → CoT data included in instruction tuning mixes
- **InstructGPT** → RLHF can be applied to CoT traces
- **Toolformer** → CoT reasoning decides when to call tools

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Add reasoning steps before the final answer |
| Zero-shot | "Let's think step by step." |
| Few-shot | Provide examples with reasoning traces |
| Self-consistency | Sample multiple paths, majority vote on answer |
| Emergence | Works well only for large models (>10B) |
| Cost | 5-20× more tokens than direct answers |
| Limitation | Can hallucinate plausible but wrong reasoning |
| Foundation for | Agents (ReAct), verifiers (PRM), reasoning RL |
