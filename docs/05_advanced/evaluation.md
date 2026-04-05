# Evaluation and Benchmarking

## Why This Matters for LLMs

Evaluation is how you know whether a model is **fit for deployment** and whether a **new checkpoint** actually improves the behaviors you care about. Unlike classic supervised learning with a single held-out label distribution, LLMs are judged on **open-ended generation**, **multi-turn dialogue**, **tool use**, and **subjective** qualities like helpfulness. Without disciplined benchmarks, teams ship models that ace **proxy metrics** while failing **real users**—or regress silently when data mixtures change.

Standard suites such as **MMLU**, **HumanEval**, and **GSM8K** provide **comparable** numbers across labs, but each has **blind spots**: contamination (test snippets appearing in training corpora), **format sensitivity** (models that excel only with a specific prompt template), and **narrow** skill coverage. Understanding **automated** metrics (perplexity, pass rates), **human** preference protocols (Elo from pairwise battles), and **LLM-as-judge** biases (position bias, self-preference) is essential for any ML or applied research role—interviewers ask “**how would you evaluate your model?**” in almost every loop.

Finally, **evaluation design** is a systems problem: you must balance **statistical power** (enough items per slice), **latency** (how quickly CI runs), **cost** (human labels, API calls to judges), and **governance** (PII in prompts, reproducibility). A mature answer connects **offline** regression gates to **online** A/B tests and **incident** playbooks when metrics and user satisfaction diverge.

---

## Core Concepts

### Why Is LLM Evaluation Hard?

Unlike image classification with a fixed label set, many LLM outputs are **partially correct**, **stylistically** constrained, or **subjective**. Let \(y\) be a reference string and \(\hat{y}\) a model output on input \(x\). A naive exact-match score is:

\[
s_{\text{exact}}(y, \hat{y}) = \mathbf{1}[y = \hat{y}] .
\]

!!! math-intuition "In Plain English"
    **Exact match** is a **hammer**: it ignores paraphrases, formatting, and multiple valid code styles. It is still used where tasks demand it (e.g. short numeric answers on GSM8K) because it is **unambiguous**—but it **underrates** partially correct reasoning.

**Memorization** means the model may answer correctly for **wrong** reasons—having seen the benchmark in training. **Benchmark contamination** inflates scores; mitigations include **n-gram overlap** checks, **canary** strings, and **dynamic** benchmarks.

A **calibration** view: even when average accuracy rises, **slice** metrics (languages, domains, difficulty bins) may fall—always report **variance** and **worst-group** performance.

### Standard Benchmarks (Overview)

| Benchmark | What it measures | Format | Primary metric |
|-----------|------------------|--------|------------------|
| **MMLU** | Broad knowledge (57 subjects) | Multiple choice | Accuracy |
| **HellaSwag** | Commonsense continuation | Sentence completion | Accuracy |
| **ARC** | Science reasoning (Challenge/Easy) | Multiple choice | Accuracy |
| **HumanEval** | Python function synthesis | Code completion | pass@\(k\) |
| **GSM8K** | Grade-school math word problems | Free-form final number | Exact match |
| **MATH** | Competition mathematics | Free-form | Exact match |
| **TruthfulQA** | Resistance to false popular beliefs | MC / generation | MC accuracy, BLEURT, etc. |
| **WinoGrande** | Coreference / commonsense | Fill-in | Accuracy |
| **MT-Bench** | Multi-turn chat quality | Open-ended | LLM judge score |

Rows are **illustrative**—always check the **official** task definition and prompt template for the version you run.

### Perplexity and Cross-Model Comparisons

Perplexity is \(\mathrm{PPL} = \exp(L)\) where \(L\) is average **negative log-likelihood** (in nats or bits, depending on convention) on a test corpus. For token sequence \(x_{1:T}\):

\[
L = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) .
\]

!!! math-intuition "In Plain English"
    **Lower** perplexity means the model is **less surprised** by each next token. Comparing perplexity across models **only** makes sense with the **same tokenizer** and **same evaluation text**—different tokenizations change \(T\) and the probability mass split.

### pass@\(k\) for Code (HumanEval-style)

Following the **unbiased** estimator from **Codex** evaluation (Chen et al.), for each problem generate \(n \ge k\) samples, count \(c\) correct, and estimate:

\[
\text{pass@}k = \mathbb{E}\left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right].
\]

!!! math-intuition "In Plain English"
    If you drew \(k\) samples **without** looking, the chance **at least one** passes is **not** \(1-(1-p)^k\) unless you know the per-sample pass rate **and** independence. The combinatorial formula accounts for **finite** \(n\) and **empirical** \(c\) per task—avoid the naive bootstrap that **double-counts** easy problems.

!!! example "Worked Example: pass@\(k\) from \(n=5\) samples"
    Suppose for one HumanEval problem you generate **5** completions and **2** pass the unit tests (\(n=5\), \(c=2\)). For \(k=1\):
    \[
    1 - \frac{\binom{5-2}{1}}{\binom{5}{1}} = 1 - \frac{\binom{3}{1}}{5} = 1 - \frac{3}{5} = 0.4 .
    \]
    For \(k=2\):
    \[
    1 - \frac{\binom{3}{2}}{\binom{5}{2}} = 1 - \frac{3}{10} = 0.7 .
    \]
    Intuition: with **two** correct in five, drawing **two** without replacement has high odds to catch **at least one** correct. Reporting **pass@100** from only \(n=5\) samples is **not** valid—you need \(n \ge k\) and typically **much** larger \(n\) for high-\(k\) estimates.

### BLEU, ROUGE, and n-Gram Overlap

**BLEU** compares **n-gram precision** between hypothesis and references with brevity penalty. A simplified unigram precision is:

\[
p_1 = \frac{\sum_{w} \min(h_w, r_w)}{\sum_w h_w}
\]

where \(h_w\) is hypothesis unigram count and \(r_w\) reference count (with **clipping**).

!!! math-intuition "In Plain English"
    BLEU rewards **overlapping words**; it correlates weakly with **human** judgment on creative tasks. It can still help **sanity-check** summarization or translation **regressions** when used **within** a fixed pipeline.

**ROUGE-L** measures longest common subsequence—better for **fluency** overlap than BLEU in some settings. Neither should be the **sole** chat metric.

### BERTScore and Embedding Similarity

**BERTScore** compares token embeddings from a pretrained encoder:

\[
F_{\text{BERT}} = \frac{1}{|x|}\sum_{i}\max_j \cos(e_{x_i}, e_{\hat{y}_j}) .
\]

!!! math-intuition "In Plain English"
    Each token in the reference finds its **best semantic match** in the hypothesis (recall-oriented component); precision swaps roles. It captures **paraphrase** better than n-grams but depends on the **encoder** biases and is **slow** at scale.

### Automated Harness: lm-evaluation-harness

**EleutherAI’s `lm-evaluation-harness`** standardizes dataset loading, prompt formatting, and metric aggregation. Conceptually, for task set \(\mathcal{T}\), a run produces scores \(\{s_t\}_{t\in\mathcal{T}}\) with optional **normalization** to a **higher-is-better** scale.

\[
s_{\text{macro}} = \frac{1}{|\mathcal{T}|}\sum_{t\in\mathcal{T}} s_t .
\]

!!! math-intuition "In Plain English"
    **Macro** averaging treats each benchmark equally; **micro** averaging pools all items—choose based on whether you care about **per-task** balance or **per-example** balance.

### Human Evaluation and Elo from Pairwise Battles

**Chatbot Arena** (LMSYS) collects **blind** pairwise comparisons. If model A beats B, update strengths \(R_A, R_B\) (Elo-style):

\[
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad
R_A \leftarrow R_A + K (S_A - E_A)
\]

where \(S_A \in \{0, 0.5, 1\}\) is the outcome score and \(K\) is a step size.

!!! math-intuition "In Plain English"
    **Elo** turns **head-to-head** wins into a **single** number line—useful for **relative** ranking, but **not** an absolute “intelligence” score. Non-transitive user preferences can distort rankings.

### Inter-Annotator Agreement

**Krippendorff’s \(\alpha\)** generalizes agreement beyond chance for multiple raters and missing data. A simplified **Cohen’s \(\kappa\)** for two raters is:

\[
\kappa = \frac{p_o - p_e}{1 - p_e}
\]

where \(p_o\) is observed agreement and \(p_e\) expected by chance.

!!! math-intuition "In Plain English"
    High benchmark accuracy is **meaningless** if humans **disagree** on labels—report **agreement** to show your **human eval** is measuring something **stable**.

### LLM-as-Judge

A strong model **\(J\)** scores a candidate answer \(\hat{y}\) for prompt \(x\) on rubric dimensions (helpfulness, correctness, verbosity). A linear scoring template:

\[
\text{score}(x,\hat{y}) = w^\top \phi(J(x,\hat{y}))
\]

where \(\phi\) extracts judge logits or Likert outputs.

!!! math-intuition "In Plain English"
    Judges are **fast** and **scalable** but biased: **position bias** (preferring the first answer), **verbosity bias**, **self-bias** (favoring own style). Mitigations: **swap** positions, **mask** model identities, use **reference** answers.

### Evaluation Pipeline Design

A robust pipeline combines: (1) **task-specific** automatic metrics, (2) **slice** dashboards, (3) **human** spot checks, (4) **online** experiments. The following worked example ties steps to numbers.

!!! example "Worked Example: Building an Eval Pipeline"
    1. **Define metrics:** For a **customer-support** bot, track **resolution rate** (human), **citation** accuracy against KB (automatic), and **toxicity** score (classifier). Set **thresholds**: e.g. toxicity **< 0.05** prevalence at 95th percentile.
    2. **Golden set:** Create **300** curated dialogs with **approved** answers and **disallowed** behaviors. Stratify by **intent** (refund, bug, account) with **100** each.
    3. **Regression suite:** Weekly run **MMLU** subset (STEM only) + **HumanEval** + **in-house** 300-dialog set. Example: baseline **MMLU-STEM** **0.62**, candidate **0.63** (Δ **+0.01**)—within **sampling noise** if CI width ±**0.02**; do **not** ship on that alone.
    4. **LLM judge:** Sample **50** dialogs; judge **pairwise** (A vs B) with **position swap**—if A wins **60%** on original order but **45%** after swap, investigate **position bias** before trusting **55%** net win.
    5. **Online:** Run **5%** traffic A/B for **two** weeks; primary metric **user thumbs-up** rate (**+1.2%** lift) with **no** increase in **escalation** rate to humans.

??? deep-dive "Deep Dive: Contamination Audits"
    Contamination detection blends **n-gram** overlap, **embedding** similarity, and **manual** review of nearest neighbors in training corpora. There is no perfect test—**dynamic** benchmarks (fresh questions, live APIs) reduce **static** leakage concerns.

??? deep-dive "Deep Dive: LLM Judge Calibration"
    Align judge scores with human labels via **Platt scaling** or **isotonic** regression on a **pilot** set. Report **calibration** curves—judges can be **sharp** but **wrong** on out-of-domain styles.

### MT-Bench and Multi-Turn Scoring

**MT-Bench** evaluates **multi-turn** instruction following with **turn-specific** questions. A simplified aggregate score averages turn scores \(s_1,\ldots,s_T\) for a dialog:

\[
S = \frac{1}{T}\sum_{t=1}^{T} s_t .
\]

!!! math-intuition "In Plain English"
    **Multi-turn** tasks punish models that **drift** or **contradict** earlier turns—single-turn benchmarks miss this failure mode. When judges are LLMs, **pairwise** comparisons per turn can reduce **absolute** score drift.

### GSM8K and Exact-Match Extraction

**GSM8K** often scores answers by **extracting** the final numeric result and comparing to ground truth \(a^\star\). Let \(\text{extract}(\hat{y})\) be a deterministic parser:

\[
s_{\text{GSM}} = \mathbf{1}[\text{extract}(\hat{y}) = a^\star].
\]

!!! math-intuition "In Plain English"
    A model can show **correct reasoning** yet lose the point if the **final line** is misformatted—teams often add **regex** normalization and **sympy** parsing for robustness.

!!! example "Worked Example: Parsing Risk on GSM8K-Style Items"
    Ground truth: **42**. Model output: “The answer is **42.0** dollars.” A strict string match fails; a **numeric** parse succeeds. If three models output **41**, **42**, **43** under majority vote, the **vote** is wrong—showing **aggregation** must happen on **parsed** numbers, not raw strings. Always log **failure modes** (division error, wrong unit) separately from **parse** errors.

### Statistical Significance at Benchmark Scale

Let \(\hat{p}\) be the empirical accuracy on \(n\) i.i.d. items with true rate \(p\). A **normal** approximation to the **95%** confidence interval is:

\[
\hat{p} \pm 1.96 \sqrt{\frac{\hat{p}(1-\hat{p})}{n}} .
\]

!!! math-intuition "In Plain English"
    On **500** MMLU items, if accuracy moves from **0.70** to **0.73**, the interval width is roughly \(1.96\sqrt{0.7\cdot 0.3/500} \approx 0.04\). The lift may be **noise**—report **CIs** whenever you compare checkpoints.

---

## Code

Install dependencies as needed:

```bash
pip install numpy torch transformers bert-score tqdm
```

The script below demonstrates **pass@\(k\)** estimation, a **toy** LLM-as-judge prompt skeleton (local small models), and **BERTScore** when available.

```python
"""
evaluation_examples.py — self-contained demos for pass@k, judge template, BERTScore.
Requires: pip install numpy torch transformers bert-score tqdm
"""
from __future__ import annotations

import math
import numpy as np

# Optional: BERTScore pulls models on first use
try:
    from bert_score import score as bert_score
except ImportError:
    bert_score = None


def comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimate for one problem: Chen et al. Codex paper."""
    if n < k:
        raise ValueError("need n >= k")
    if comb(n, k) == 0:
        return 0.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def estimate_pass_at_k_aggregate(results: list[tuple[int, int]], k: int) -> float:
    """
    results: list of (n_samples, c_correct) per problem.
    Macro average of per-problem pass@k.
    """
    vals = []
    for n, c in results:
        vals.append(pass_at_k(n, c, k))
    return float(np.mean(vals))


def judge_prompt(task: str, response_a: str, response_b: str) -> str:
    """Template for pairwise LLM-as-judge (swap A/B between calls)."""
    return f"""You are an impartial evaluator. Pick which response better satisfies the task.
Task: {task}

Response A:
{response_a}

Response B:
{response_b}

Answer with a single letter: A or B."""


def run_bertscore_demo(cands: list[str], refs: list[str]) -> None:
    if bert_score is None:
        print("bert-score not installed; skip BERTScore demo.")
        return
    p, r, f1 = bert_score(cands, refs, lang="en", verbose=False)
    print("BERTScore F1 (per candidate):", f1.numpy())


def main() -> None:
    rng = np.random.default_rng(7)

    # --- pass@k toy aggregation: 3 problems ---
    scenarios = [
        (10, 3),  # n, c
        (10, 1),
        (10, 6),
    ]
    for k in (1, 5):
        agg = estimate_pass_at_k_aggregate(scenarios, k)
        print(f"Macro pass@{k} on toy scenarios: {agg:.4f}")

    # Show single-problem numbers
    print("Single problem n=10,c=3:", pass_at_k(10, 3, 5))

    # --- BERTScore ---
    cands = ["The capital of France is Paris.", "Paris is the capital city of France."]
    refs = ["Paris is the capital of France."]
    run_bertscore_demo(cands, refs * 2)

    # --- Judge template ---
    print(judge_prompt("Explain chain rule.", "dy/dx = dy/du * du/dx.", "Use nested functions."))

    # --- Simulate Elo update (one step) ---
    r_a, r_b = 1000.0, 1000.0
    k_elo = 32.0
    s_a = 1.0  # A wins
    e_a = 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))
    r_a += k_elo * (s_a - e_a)
    r_b += k_elo * ((1.0 - s_a) - (1.0 - e_a))
    print(f"Elo after one match: A={r_a:.2f}, B={r_b:.2f}")


if __name__ == "__main__":
    main()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Why is perplexity not comparable across different tokenizers?
    2. Define pass@\(k\) and explain why the naive \(1-(1-p)^k\) formula can be wrong for code benchmarks.
    3. What is benchmark contamination and how would you detect it?
    4. Compare macro versus micro averaging over tasks—when does each matter?
    5. Name two failure modes of BLEU/ROUGE for chat evaluation.
    6. How does Chatbot Arena estimate model strength, and what is a limitation of Elo here?
    7. What is position bias in LLM-as-judge, and how do you mitigate it?
    8. When would you trust human evaluation over automatic metrics for a release decision?
    9. What is Krippendorff’s alpha used for in evaluation pipelines?
    10. Sketch an end-to-end regression gate for a coding assistant (metrics + cadence).

!!! interview "Follow-up Probes"
    - “Your MMLU went up but users complain—what do you check first?”
    - “How do you evaluate **tool-calling** agents differently from raw chat?”
    - “What’s wrong with evaluating on the **training** chat logs?”
    - “How would you detect **reward hacking** on a learned metric?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Slice metrics** and **worst-group** performance, not just averages.”
    - “**Unbiased pass@\(k\)** with \(n \ge k\) samples per task.”
    - “**Contamination audits** plus **dynamic** benchmarks for trust.”
    - “**Pairwise** comparisons with **position swaps** for judge fairness.”
    - “**Offline** regression gates aligned with **online** KPIs.”

---

## References

1. Hendrycks, D., et al. (2021). *Measuring Massive Multitask Language Understanding (MMLU).* ICLR.
2. Chen, M., et al. (2021). *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374.
3. Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168.
4. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic Evaluation of Summaries.*
5. Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation.* ACL.
6. Zhang, T., et al. (2020). *BERTScore: Evaluating Text Generation with BERT.* ICLR.
7. EleutherAI. *lm-evaluation-harness* (GitHub). Standardized LM benchmarking framework.
8. Chiang, W.-L., et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* arXiv:2403.04132.
9. Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS Datasets & Benchmarks.
10. Krippendorff, K. (2018). *Content Analysis: An Introduction to Its Methodology.*
