# Chinchilla: Training Compute-Optimal Large Language Models

**Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, and 8 more  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

---

## TL;DR

Chinchilla revisits scaling laws and demonstrates that most large language models are **undertrained**: for a fixed compute budget, **smaller models trained on more data** consistently outperform larger models trained on less data. The paper fits empirical laws relating loss to parameters \(N\) and tokens \(D\), and derives **compute-optimal** allocation ratios — showing that parameters and data should scale roughly equally.

---

## Why This Paper Matters

Chinchilla fundamentally changed how the industry thinks about training budgets:

1. **Before Chinchilla:** "Make the model as large as possible" (GPT-3, PaLM)
2. **After Chinchilla:** "Balance model size with data" (LLaMA, Mistral)
3. The **70B** model trained on 1.4T tokens (Chinchilla) outperformed the **280B** Gopher trained on 300B tokens
4. Directly shaped open-source LLM training (LLaMA used Chinchilla-optimal ratios)
5. "Chinchilla-optimal" became a standard term in model training discussions

---

## Key Concepts Explained Simply

### The Core Insight

If you have a fixed compute budget (say, \$10M worth of GPU time), should you:
- (A) Train a **huge** 500B model on 100B tokens, or
- (B) Train a **smaller** 70B model on 1.4T tokens?

Chinchilla showed that **(B) is better**. The larger model is undertrained — it hasn't seen enough data to fully utilize its capacity. The smaller model, having seen more diverse data, generalizes better.

### The Rule of Thumb

For compute-optimal training, **tokens should scale linearly with parameters**:

- **~20 tokens per parameter** (Chinchilla's finding)
- GPT-3 (175B params, 300B tokens) → only ~1.7 tokens per parameter → **severely undertrained**
- Chinchilla (70B params, 1.4T tokens) → 20 tokens per parameter → **compute-optimal**

### When Chinchilla Breaks Down

The scaling laws assume:
- **Fresh, unique data** for each token (no repeated epochs)
- **Data quality** is consistent as you scale
- **Task distribution** at evaluation matches training
- In practice, running out of high-quality data means you may need to choose larger models

---

## The Math — Explained Step by Step

### Parametric Loss Function

\[
L(N, D) = \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}} + L_{\infty}
\]

**Breaking it down:**

1. **\(A / N^{\alpha}\):** Loss from **insufficient model capacity** — decreases with more parameters
2. **\(B / D^{\beta}\):** Loss from **insufficient data** — decreases with more training tokens
3. **\(L_{\infty}\):** **Irreducible loss** — the entropy of natural language, can't be reduced by any model
4. **\(\alpha \approx 0.34, \beta \approx 0.28\):** Fitted exponents — both show diminishing returns

### Compute Constraint

For a Transformer with \(N\) parameters trained on \(D\) tokens:

\[
C \approx 6 \cdot N \cdot D
\]

This is the approximate FLOPs budget. Given fixed \(C\), we want to find the \(N^*\) and \(D^*\) that minimize \(L(N, D)\) subject to \(C = 6ND\).

### Optimal Allocation

Taking the Lagrangian and solving:

\[
N^* \propto C^{a}, \quad D^* \propto C^{b}
\]

where \(a \approx 0.5\) and \(b \approx 0.5\). This means:
- **Parameters and tokens should scale equally with compute**
- Double your budget → increase both N and D by ~\(\sqrt{2}\)

### The 20× Rule

From the empirical fits:

\[
D^* \approx 20 \cdot N^*
\]

For compute-optimal training, you need roughly 20 tokens per parameter.

---

## Python Implementation

```python
import numpy as np
from scipy.optimize import minimize_scalar


def chinchilla_loss(N, D, A=406.4, alpha=0.34, B=410.7, beta=0.28, L_inf=1.69):
    """
    Parametric loss model from Chinchilla.
    N: number of parameters
    D: number of training tokens
    """
    return A / (N ** alpha) + B / (D ** beta) + L_inf


def compute_flops(N, D):
    """Approximate training FLOPs: C ≈ 6ND."""
    return 6 * N * D


def optimal_allocation(C, A=406.4, alpha=0.34, B=410.7, beta=0.28):
    """
    Find the optimal N and D for a given compute budget C.
    Minimizes L(N, D) subject to 6*N*D = C.
    """
    def loss_for_N(log_N):
        N = np.exp(log_N)
        D = C / (6 * N)
        if D <= 0:
            return 1e10
        return chinchilla_loss(N, D, A, alpha, B, beta)

    # Search over reasonable model sizes
    min_N = 1e6    # 1M params
    max_N = C / 6  # Maximum N (1 token)

    result = minimize_scalar(
        loss_for_N,
        bounds=(np.log(min_N), np.log(max_N)),
        method='bounded'
    )

    N_opt = np.exp(result.x)
    D_opt = C / (6 * N_opt)
    return N_opt, D_opt, result.fun


def tokens_per_parameter(N, D):
    """Compute the data-to-parameter ratio."""
    return D / N


def is_chinchilla_optimal(N, D, target_ratio=20, tolerance=0.5):
    """Check if a model is approximately Chinchilla-optimal."""
    ratio = D / N
    return abs(ratio - target_ratio) / target_ratio < tolerance


def compare_models():
    """Compare real models against Chinchilla-optimal ratios."""
    models = [
        ("GPT-3",       175e9,  300e9),
        ("Gopher",      280e9,  300e9),
        ("Chinchilla",  70e9,   1.4e12),
        ("LLaMA-7B",    7e9,    1.0e12),
        ("LLaMA-13B",   13e9,   1.0e12),
        ("LLaMA-65B",   65e9,   1.4e12),
        ("Mistral-7B",  7e9,    8e12),   # estimated
    ]

    print(f"{'Model':<15} {'Params':>10} {'Tokens':>10} {'Tok/Param':>10} {'Optimal?':>10}")
    print("-" * 60)
    for name, N, D in models:
        ratio = tokens_per_parameter(N, D)
        optimal = is_chinchilla_optimal(N, D)
        flops = compute_flops(N, D)
        print(f"{name:<15} {N/1e9:>8.0f}B {D/1e9:>8.0f}B {ratio:>10.1f} {'✓' if optimal else '✗':>10}")


def scaling_curve(compute_budgets):
    """Show optimal N and D for various compute budgets."""
    print(f"\n{'Compute (FLOPs)':>18} {'Optimal N':>12} {'Optimal D':>12} {'Tok/Param':>10} {'Loss':>8}")
    print("-" * 65)
    for C in compute_budgets:
        N_opt, D_opt, loss = optimal_allocation(C)
        ratio = D_opt / N_opt
        print(f"{C:>18.2e} {N_opt/1e9:>10.1f}B {D_opt/1e9:>10.1f}B {ratio:>10.1f} {loss:>8.3f}")


def data_constrained_analysis():
    """What happens when you run out of unique data?"""
    print("\n--- Data-Constrained Scenario ---")
    N = 70e9
    unique_tokens = 1e12

    for epochs in [1, 2, 4, 8]:
        effective_D = unique_tokens * epochs
        loss = chinchilla_loss(N, effective_D)
        # Diminishing returns from repeated data
        penalty = 1 + 0.05 * np.log(epochs) if epochs > 1 else 0
        adjusted_loss = loss + penalty
        print(f"  Epochs: {epochs}, Effective D: {effective_D/1e12:.0f}T, "
              f"Loss: {loss:.3f}, Adjusted (repeat penalty): {adjusted_loss:.3f}")


# --- Demo ---
if __name__ == "__main__":
    compare_models()
    scaling_curve([1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24])
    data_constrained_analysis()

    # Visualize the trade-off
    print("\n--- Fixed Compute Trade-off (C = 6e21 FLOPs) ---")
    C = 6e21
    print(f"{'N (params)':>15} {'D (tokens)':>15} {'Loss':>8}")
    print("-" * 40)
    for N in [1e9, 5e9, 10e9, 50e9, 100e9, 500e9]:
        D = C / (6 * N)
        if D > 0:
            loss = chinchilla_loss(N, D)
            print(f"{N/1e9:>13.1f}B {D/1e9:>13.1f}B {loss:>8.3f}")
```

---

## Interview Importance

Chinchilla is a **must-know** paper. "Chinchilla-optimal" comes up constantly when discussing model training decisions.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: State the Chinchilla insight in one sentence.

**Answer:** For a fixed compute budget, you should train a **smaller model on more data** rather than a larger model on less data, because parameters and training tokens should scale roughly equally with compute (approximately 20 tokens per parameter for optimal training).

### Q2: What assumptions break scaling laws?

**Answer:**
1. **Data saturation:** When you run out of unique, high-quality data and must repeat epochs, the effective value of additional tokens diminishes
2. **Repeated epochs:** Scaling laws assume each token is seen once; repeated data provides diminishing returns
3. **Data quality variation:** Scaling laws assume uniform data quality; in practice, scraping more data means lower quality
4. **Task-specific evaluation:** Scaling laws predict average loss, but specific task performance may not follow smooth trends
5. **Distribution shift:** Training on web text may not improve performance on specialized domains proportionally

### Q3: How would you decide whether to increase data vs. model width under a fixed budget?

**Answer:**
1. Compute the current **tokens-per-parameter ratio**
2. If ratio < 20: You're undertrained → invest in more data
3. If ratio > 20: You might benefit from a larger model
4. Check **data availability:** If you've exhausted high-quality data, a larger model on the same data may still help (but with diminishing returns)
5. Consider **inference cost:** A smaller, well-trained model is cheaper to serve than a larger undertrained one
6. Run **small-scale experiments:** Train models at 1/100th scale with different N/D ratios and extrapolate

### Q4: How does Chinchilla relate to LLaMA's training strategy?

**Answer:** LLaMA directly applied Chinchilla's insights:
- LLaMA-7B was trained on **1T tokens** (~143 tokens per parameter — far beyond Chinchilla-optimal)
- LLaMA-65B was trained on **1.4T tokens** (~21.5 tokens per parameter — close to optimal)
- The result: LLaMA-13B matched GPT-3 (175B) on many benchmarks because it was properly trained despite being 13× smaller
- LLaMA showed you can "overtrain" small models (go beyond 20 tokens/param) if you want smaller, cheaper-to-serve models

---

## Connections to Other Papers

- **GPT-3** → Chinchilla showed GPT-3 was undertrained
- **PaLM** → Also likely undertrained (540B params, ~780B tokens)
- **LLaMA** → Applied Chinchilla insights for efficient open models
- **Mistral** → Pushed even further: 7B model trained on massive data

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core insight | Smaller models + more data > larger models + less data |
| Optimal ratio | ~20 tokens per parameter |
| Loss formula | \(L(N,D) = A/N^α + B/D^β + L_∞\) |
| FLOPs rule | C ≈ 6ND |
| Equal scaling | N and D should scale equally with compute |
| Chinchilla model | 70B params on 1.4T tokens beat 280B Gopher |
| Practical impact | Shaped LLaMA, Mistral, and open-model training |
