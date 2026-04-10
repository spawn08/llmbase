# Codex: Evaluating Large Language Models Trained on Code

**Authors:** Mark Chen, Jerry Tworek, Heewoo Jun, and 20 more  
**Year:** 2021 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)

---

## TL;DR

Codex fine-tunes GPT models on **GitHub code**, creating systems that generate functional code from docstrings and natural language descriptions. The paper introduces **HumanEval**, a benchmark of 164 hand-written Python problems with unit tests, and the **pass@k** metric for evaluating code generation. Codex powers **GitHub Copilot** and established code LMs as a major application category.

---

## Why This Paper Matters

Codex defined how we evaluate and deploy code LLMs:

1. **HumanEval benchmark:** Still the standard (alongside MBPP, SWE-bench) for code evaluation
2. **pass@k metric:** Statistically rigorous way to evaluate code generation
3. **Copilot lineage:** Powered GitHub Copilot, the first widely-adopted AI coding tool
4. **Code-specific fine-tuning:** Showed that domain-specific fine-tuning dramatically improves performance
5. **Fill-in-the-middle (FIM):** Later models added FIM capability for infilling, based on Codex insights

---

## Key Concepts Explained Simply

### From Language Model to Code Model

GPT-3 can generate some code, but it's trained primarily on natural language. Codex fine-tunes GPT-3 on ~159GB of Python code from GitHub. This specialization dramatically improves:
- Function generation from docstrings
- Code completion
- Understanding of programming patterns and libraries

### HumanEval

164 hand-crafted Python programming problems, each with:
- A function signature and docstring
- Multiple unit tests that verify correctness
- Varying difficulty from simple string manipulation to dynamic programming

Example:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers are closer than the threshold."""
    # Model generates the implementation
```

The model passes if its generated code passes **all** unit tests.

### pass@k: The Right Way to Evaluate Code Generation

Simple accuracy (does the first generation work?) doesn't capture the full picture. With temperature sampling, the model might produce 10 different solutions — if **any one** is correct, that's useful. pass@k measures the probability that at least one of k samples is correct.

---

## The Math — Explained Step by Step

### pass@k

Given \(n\) total samples per problem and \(c\) correct samples:

\[
\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}
\]

**Breaking it down:**

1. \(\binom{n}{k}\): Total ways to choose \(k\) samples from \(n\)
2. \(\binom{n-c}{k}\): Ways to choose \(k\) samples that are **all wrong** (from the \(n-c\) incorrect samples)
3. The ratio is the probability that all \(k\) chosen samples are wrong
4. 1 minus that = probability that **at least one** of \(k\) is correct
5. This is an **unbiased estimator** — no need for infinite samples

**Example:** 100 samples, 30 correct, k=10
\(\text{pass@}10 = 1 - \binom{70}{10}/\binom{100}{10} = 1 - 0.023 = 97.7\%\)

### Why Not Just Use pass@1?

pass@1 with greedy decoding measures the **single best** generation. But in practice:
- Users can generate multiple solutions and pick the best
- Temperature sampling produces diverse solutions
- pass@k with k=10 or k=100 shows how much **potential** the model has
- The gap between pass@1 and pass@100 indicates how much **selection** matters

### Temperature Trade-off

Higher temperature → more diverse samples → higher pass@k (for large k) but lower pass@1.
Lower temperature → more consistent samples → higher pass@1 but lower pass@k.

Optimal temperature depends on the use case:
- **Autocomplete (Copilot):** Low temperature, high pass@1
- **Solution search:** Higher temperature, rely on pass@k with selection

---

## Python Implementation

```python
import numpy as np
from math import comb, factorial


def pass_at_k(n, c, k):
    """
    Unbiased estimator of pass@k.
    n: total samples, c: correct samples, k: samples to consider
    """
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k_problems(results, k):
    """
    Compute pass@k across multiple problems.
    results: list of (n_samples, n_correct) per problem
    """
    scores = []
    for n, c in results:
        scores.append(pass_at_k(n, c, k))
    return np.mean(scores)


def simulate_code_generation(n_problems=164, n_samples=100, base_accuracy=0.3,
                              temperature=0.8):
    """Simulate code generation results at a given temperature."""
    results = []
    for _ in range(n_problems):
        # Accuracy varies by problem difficulty
        difficulty = np.random.uniform(0.1, 0.9)
        prob_correct = base_accuracy * (1 - difficulty)

        # Higher temperature → more variance → some problems get more correct
        if temperature > 0.5:
            prob_correct *= np.random.uniform(0.5, 1.5)
            prob_correct = np.clip(prob_correct, 0, 0.95)

        n_correct = np.random.binomial(n_samples, prob_correct)
        results.append((n_samples, n_correct))

    return results


def humaneval_sample():
    """Example HumanEval-style problem."""
    return {
        "task_id": "HumanEval/0",
        "prompt": '''def has_close_elements(numbers: list, threshold: float) -> bool:
    """Check if in given list of numbers, are any two numbers
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
''',
        "canonical_solution": '''    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True
    return False
''',
        "test": '''
def check(candidate):
    assert candidate([1.0, 2.0, 3.0], 0.5) == False
    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True
    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False
    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0], 2.0) == True
''',
    }


def evaluate_code_solution(solution_code, test_code, function_name):
    """
    Execute a code solution against test cases.
    Returns True if all tests pass.
    """
    try:
        namespace = {}
        exec(solution_code, namespace)
        exec(test_code.replace("candidate", function_name), namespace)
        return True
    except Exception:
        return False


def fill_in_the_middle(prefix, suffix, model_completion="..."):
    """
    FIM (Fill-in-the-Middle) format used in later code models.
    """
    return {
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        "expected_structure": f"{prefix}{model_completion}{suffix}",
    }


def temperature_analysis():
    """Show how temperature affects pass@k."""
    print("--- Temperature vs. pass@k Trade-off ---")
    temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]

    for temp in temperatures:
        results = simulate_code_generation(
            n_problems=164, n_samples=100,
            base_accuracy=0.3, temperature=temp
        )
        p1 = pass_at_k_problems(results, k=1)
        p10 = pass_at_k_problems(results, k=10)
        p100 = pass_at_k_problems(results, k=100)
        print(f"  T={temp:.1f}: pass@1={p1:.1%}, pass@10={p10:.1%}, pass@100={p100:.1%}")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # pass@k calculation
    print("--- pass@k Examples ---")
    examples = [(100, 30, 1), (100, 30, 10), (100, 30, 100),
                (200, 10, 1), (200, 10, 10), (200, 10, 100)]
    for n, c, k in examples:
        score = pass_at_k(n, c, k)
        print(f"  n={n}, c={c}, k={k}: pass@{k} = {score:.4f}")

    # HumanEval example
    print("\n--- HumanEval Sample Problem ---")
    problem = humaneval_sample()
    print(f"Task: {problem['task_id']}")
    print(f"Prompt:\n{problem['prompt']}")

    # Solution evaluation
    full_code = problem['prompt'] + problem['canonical_solution']
    passed = evaluate_code_solution(full_code, problem['test'], 'has_close_elements')
    print(f"Canonical solution passes: {passed}")

    # Temperature analysis
    print()
    temperature_analysis()

    # FIM example
    print("\n--- Fill-in-the-Middle ---")
    fim = fill_in_the_middle(
        prefix="def factorial(n):\n    if n <= 1:\n        return 1\n",
        suffix="\n\nprint(factorial(5))",
        model_completion="    return n * factorial(n - 1)"
    )
    print(f"  FIM Prompt: {fim['prompt'][:80]}...")
```

---

## Interview Importance

Codex/HumanEval is important for roles involving **code generation**, **AI-assisted development**, and **evaluation methodology**.

### Difficulty Level: ⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Why is HumanEval limited, and what complements it?

**Answer:** HumanEval has only **164 problems**, all in Python, mostly self-contained functions without dependencies. It doesn't test:
1. **Multi-file projects:** Real code spans multiple files with imports
2. **Debugging:** Finding and fixing bugs in existing code
3. **Other languages:** Only Python
4. **Real-world context:** No repository context, no documentation reading

Complements: **MBPP** (974 simpler problems), **SWE-bench** (real GitHub issues requiring multi-file changes), **LiveCodeBench** (competition problems updated continuously), **DS-1000** (data science problems).

### Q2: Explain pass@k versus pass@1 for temperature-sampled models.

**Answer:** 
- **pass@1:** Probability that a single greedy/low-temperature generation is correct. Measures the model's "best guess" quality. Most relevant for autocomplete.
- **pass@k:** Probability that at least one of k sampled generations is correct. Measures the model's ability to produce correct solutions **somewhere** in its distribution. Relevant when you can generate multiple candidates and select the best.
- **Trade-off:** Optimizing pass@1 (low temperature) may hurt pass@k (less diversity), and vice versa.

### Q3: What security issues arise from training on public GitHub?

**Answer:**
1. **License violations:** Generated code may reproduce copyrighted code
2. **Vulnerability injection:** Training data contains buggy/vulnerable code that the model may reproduce
3. **Secret leakage:** Training data may contain API keys, passwords, or secrets committed accidentally
4. **Malicious code:** Some training data may contain intentionally malicious code patterns
5. **Attribution:** Users may inadvertently use generated code that should be attributed to original authors
6. **Supply chain attacks:** Model might suggest importing packages with known vulnerabilities

---

## Connections to Other Papers

- **GPT-3** → Codex fine-tunes GPT-3 on code
- **LLaMA** → Code LLaMA extends LLaMA for code
- **Chain-of-Thought** → CoT reasoning in code (step-by-step problem decomposition)
- **InstructGPT** → Copilot X adds instruction-following to code generation
- **ReAct** → Code agents use ReAct-style loops with code execution tools

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| What | GPT fine-tuned on 159GB of GitHub Python code |
| Benchmark | HumanEval: 164 Python problems with unit tests |
| Metric | pass@k: P(≥1 correct in k samples) |
| Formula | pass@k = 1 - C(n-c,k)/C(n,k) |
| Product | GitHub Copilot |
| Temperature | Low → better pass@1; High → better pass@k |
| Limitations | Python only, single functions, no repo context |
