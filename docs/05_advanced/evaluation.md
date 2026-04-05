# Evaluation & Benchmarking

## Why This Matters for LLMs

You cannot improve what you do not measure—but **wrong metrics** optimize toward **the wrong behaviors**. **Perplexity** tracks language modeling quality but misses instruction-following, safety, and tool use. **Benchmark suites** (MMLU, HumanEval, MT-Bench) operationalize capabilities for research and marketing, yet **contamination** (test data leaked into training corpora) and **gaming** (overfitting to prompts) undermine trust.

For engineers, evaluation is how you **gate releases**: regression tests on model updates, A/B tests in production, and **offline** correlation with human ratings. Interviewers expect you to name standard benchmarks, explain **perplexity limitations**, and discuss **lm-evaluation-harness** workflows.

Finally, **chat** evaluation differs from **base model** evaluation: instruction-tuned models need **pairwise preference** metrics (Elo from arena battles) and **task-specific** rubrics (code unit tests). A single number never captures deployment readiness.

---

## Core Concepts

### Perplexity and Cross-Entropy

For token sequence \(w_1,\ldots,w_T\) and model \(P_\theta\):

\[
\text{PPL}(w) = \exp\left(-\frac{1}{T}\sum_{i=1}^{T} \log P_\theta(w_i \mid w_{<i})\right)
\]

Lower perplexity implies better **average** next-token prediction. **Limitations**:

- Sensitive to **tokenizer** and **domain** (Wikipedia vs code).
- **Poor proxy** for reasoning, factual correctness, or safety.
- Can improve via **memorization** without generalization.

!!! math-intuition "In Plain English"
    Perplexity answers: “How surprised is the model by each token?” It is useful for **comparing** models on a **clean** held-out pile, not for user satisfaction.

### Knowledge and Reasoning: MMLU, HellaSwag, ARC, WinoGrande

| Benchmark | What it tests | Notes |
|-----------|----------------|-------|
| **MMLU** | 57 tasks (STEM, humanities, law) | Multiple-choice; **few-shot** or **zero-shot** variants |
| **HellaSwag** | Commonsense **next sentence** | Adversarially filtered negatives |
| **ARC** | Science exam questions (Easy/Challenge) | Requires multi-hop reasoning |
| **WinoGrande** | Coreference / commonsense | Winograd schema at scale |

**Accuracy** is standard; **calibration** (confidence vs correctness) matters for production.

### Code: HumanEval and MBPP

**HumanEval** (164 hand-written Python problems) measures **pass@k**:

\[
\text{pass@k} = \mathbb{E}_{\text{problems}} \left[ 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \right]
\]

where \(n\) samples are drawn per problem, \(c\) is the number of samples passing **unit tests**. This accounts for **stochastic** decoding.

**MBPP** (mostly basic Python problems) tests simpler patterns. Both are **saturated** for top models—**LiveCodeBench** and **contamination-aware** splits are increasingly used.

### Chat and Preference: MT-Bench, Chatbot Arena

**MT-Bench** uses GPT-4 as judge on multi-turn dialogues across categories (writing, reasoning, math). **LLM-as-judge** biases exist (position bias, verbosity bias)—mitigate with **swap** ordering and **human** spot checks.

**Chatbot Arena** collects **human pairwise preferences** and fits **Bradley–Terry** models to estimate **Elo** ratings:

\[
P(i \succ j) = \frac{1}{1 + 10^{(R_j - R_i)/400}}
\]

Elo provides **relative** strength—not absolute correctness.

### Contamination and Benchmark Gaming

**Contamination** occurs when benchmark strings appear in pretraining data. Mitigations:

- **N-gram overlap** scans between train and eval.
- **Canary** strings (BIG-bench canary).
- **Dynamic** benchmarks (fresh problems, private held-out sets).

**Gaming** includes training on test data, **prompt hacking**, and **teaching to the test** with synthetic fine-tunes.

### lm-evaluation-harness

**EleutherAI’s lm-evaluation-harness** standardizes task loading, few-shot formatting, and metric computation. Typical workflow:

1. Pin **task** versions and **model** tokenizer.
2. Run **GPU** servers with deterministic seeds where possible.
3. Aggregate **per-task** and **macro** scores.

Use `--tasks` lists, `--num_fewshot` controls, and record **git SHAs** for reproducibility.

### Code: Computing Perplexity on a Tiny Model

The following uses **Hugging Face** `transformers` with **GPT-2** small—use your own checkpoint in production evals.

```python
from __future__ import annotations

import math
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str = "cpu",
) -> float:
    """
    HF causal LM loss is mean negative log-likelihood over predicted token positions.
    Aggregate as weighted average across snippets for a crude corpus PPL.
    """
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
            input_ids = enc["input_ids"].to(device)
            outputs = model(input_ids, labels=input_ids)
            # loss is mean over supervised positions (model-internal shift)
            n_tokens = max(input_ids.numel() - 1, 1)
            total_nll += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    return math.exp(total_nll / total_tokens)


if __name__ == "__main__":
    name = "gpt2"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForCausalLM.from_pretrained(name)
    samples = [
        "The capital of France is Paris.",
        "def add(a, b): return a + b",
    ]
    print("PPL:", perplexity(mdl, tok, samples))
```

### Evaluation Design Principles

- **Match the deployment distribution**: coding evals for coding assistants, **multilingual** sets for global users.
- **Report variance** across seeds and prompts.
- **Combine** automatic metrics with **human eval** for high-stakes releases.
- **Track** regression **per capability** (retrieval, math, safety) rather than one headline number.

### BIG-bench and Beyond the Imitation Game

**BIG-bench** aggregates **200+** tasks (linguistics, math, commonsense, code). **Emergent** behavior appears on some tasks only at scale—interpret with **task clusters** and **confidence intervals**.

### HELM: Holistic Evaluation

**HELM** (Stanford) emphasizes **multimetric** reporting: accuracy, **calibration**, **robustness**, **fairness**, **efficiency**. The lesson: **leaderboard** single numbers hide **trade-offs**—publish **slices** (domains, demographics) where possible.

### GSM8K and Math Reasoning

**GSM8K** (grade-school math word problems) tests **multi-step** arithmetic reasoning. **Chain-of-thought** prompting dramatically improves scores—evaluate **with** and **without** CoT when comparing models.

### LiveCodeBench and Dynamic Evaluation

Static benchmarks **leak** into training corpora over time. **LiveCodeBench** and **private** eval sets reduce **contamination**—treat them as **gold** for major releases.

### Leaderboard Hygiene

- Disclose **prompt** templates and **few-shot** counts.
- Report **seeds** and **hardware** (GPU type, batch size) for timing.
- Separate **base** vs **instruct** checkpoints.

### Calibration and Expected Calibration Error (ECE)

For **classification** benchmarks (multiple-choice), **calibration** measures whether **confidence** matches **accuracy**. Partition predictions into bins \(B_m\) by confidence \(p\); **ECE** is:

\[
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \overline{p}_m \right|
\]

Well-calibrated models matter for **selective prediction** (abstain when \(p\) is low).

### BLEU and ROUGE (When They Still Appear)

For **summarization** and **translation**, **BLEU/ROUGE** remain common **cheap** metrics despite **semantic** blindness. Pair them with **LLM judges** or **human** ratings on **faithfulness**—especially for **news** and **clinical** summaries.

### Toxicity and Bias Metrics

**Perspective API**-style classifiers and **holistic** bias suites (e.g., **BBQ**, **Winogender**) provide **signals** but **not** guarantees. Report **per-group** error rates and **confidence** intervals; avoid **single** fairness numbers.

### Statistical Testing for Benchmarks

When comparing models **A** vs **B** on \(n\) items, use **McNemar’s test** for paired **binary** correctness or **bootstrap** confidence intervals on mean score differences—**avoid** claiming wins within **noise**.

### Reproducibility Checklist

- Pin **library** versions (`transformers`, `torch`, `lm-eval`).
- Save **generation** configs (`temperature`, `top_p`, `max_tokens`).
- Record **hardware** and **dtype** (BF16 vs FP16)—numerics can shift **exact** match tasks slightly.

### Cost-Aware Evaluation

**Dollar-per-1000-examples** matters for **continuous** eval in CI. Cache **deterministic** generations; use **smaller** proxy models for **smoke** tests and **large** models nightly.

### Pass@k Worked Example (Combinatorial)

With \(n=10\) samples and \(c=3\) correct programs passing tests:

\[
\text{pass@5} = 1 - \frac{\binom{10-3}{5}}{\binom{10}{5}} = 1 - \frac{21}{252} \approx 0.917
\]

Intuition: if **few** samples are correct, **larger** \(k\) helps—but **cost** grows linearly with \(n\) for generation.

### Winogender and Ambiguous Coreference

**Winogender** tests **gender bias** in **occupation** coreference—models should not **systematically** resolve **ambiguous** pronouns using **stereotypes**. Pair with **per-template** accuracy to locate **failure modes**.

### Domain-Specific Suites

- **MedQA / PubMedQA** for biomedical QA (verify **license** and **clinical** oversight).
- **LegalBench**-style tasks for **attorney** workflows—**never** substitute for **lawyers**.

### Throughput Metrics for Serving

Report **tokens/s** at **fixed** batch and **sequence** lengths; separate **prefill** (compute-heavy) vs **decode** (memory-bandwidth-heavy) phases—**different** optimizations apply.

### MMLU: Per-Domain Reporting

**MMLU** aggregates **57** tasks across STEM, humanities, and social sciences. **Macro** average treats each task equally; **micro** average weights by **example** count. A model can look strong on **macro** while failing **law** or **moral scenarios**—publish **per-group** tables.

### ARC: Easy vs Challenge

**ARC-Challenge** requires **reasoning** beyond **retrieval**; **ARC-Easy** is closer to **lookup**. Compare models on **Challenge** when discussing **reasoning** claims.

### HellaSwag: Adversarial Filtering

**HellaSwag** uses **adversarial filtering** to create **hard** negatives—models cannot rely on **n-gram** overlap alone. Strong scores indicate **commonsense** completion beyond **surface** cues.

### Chatbot Arena Limitations

**Elo** from **pairwise** battles reflects **user** preference—not **factual** correctness. **Popularity** and **verbosity** bias can **inflate** scores for **flashy** answers.

### Regression Testing in CI

Maintain a **golden** set of **prompts** with **expected** properties (JSON schema, **banned** substrings). **Fail** builds on **regressions**—cheap **smoke** tests complement **heavy** nightly harness runs.

### WinoGrande: Commonsense at Scale

**WinoGrande** scales **Winograd**-style **coreference** with **adversarial** filtering—strong performance suggests **commonsense** reasoning beyond **n-gram** association.

### TruthfulQA vs MMLU

**MMLU** probes **knowledge** breadth; **TruthfulQA** probes **avoiding** falsehoods. A model can score **high** on MMLU yet **low** on TruthfulQA if it **parrots** misconceptions—evaluate **both**.

### Efficiency Metrics: FLOPs per Token

When comparing **architectures**, **FLOPs/token** and **memory** footprint matter for **serving**—**accuracy** alone is incomplete for **engineering** decisions.

### Leaderboard Dynamics

Public leaderboards **change** as **test** sets **leak** or **prompts** **standardize**. Treat **rankings** as **snapshots**—**reproduce** locally before **betting** product decisions.

### Academic vs Production Eval

Academic **benchmarks** are **sanitized** and **static**; **production** traffic is **long-tailed** and **adversarial**. **Offline** scores are **necessary** but not **sufficient**.

---

## Interview Takeaways

- **Perplexity** measures **next-token** fit, not user-visible task success.
- **MMLU / ARC / HellaSwag** cover broad knowledge and reasoning; **HumanEval/MBPP** cover code generation with **pass@k**.
- **MT-Bench** and **Arena Elo** evaluate **chat** quality; **LLM judges** need bias controls.
- **Contamination** undermines leaderboard claims—**inspect** training data overlap.
- **lm-evaluation-harness** is the de facto **open** harness—pin versions and document seeds.
- Always pair **offline** metrics with **online** outcomes (retention, task completion).

## References

- Hendrycks et al., *Measuring Massive Multitask Language Understanding* (MMLU): [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- Zellers et al., *HellaSwag: Can a Machine Really Finish Your Sentence?* (2019): [arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
- Clark et al., *Think you have Solved Question Answering? Try ARC* (2018): [arXiv:1803.05457](https://arxiv.org/abs/1803.05457)
- Chen et al., *Evaluating Large Language Models Trained on Code* (HumanEval): [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- Austin et al., *Program Synthesis with Large Language Models* (MBPP): [arXiv:2108.07732](https://arxiv.org/abs/2108.07732)
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023): [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Gao et al., *The EleutherAI LM Evaluation Harness* (GitHub): [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
