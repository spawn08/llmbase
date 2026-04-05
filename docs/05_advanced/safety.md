# Hallucination & Safety

## Why This Matters for LLMs

**Hallucination**—confident outputs unsupported by facts or user context—undermines trust in medical, legal, and enterprise assistants. It is not a single bug: **factual** errors (wrong dates), **faithfulness** failures (ignoring retrieved context), and **reasoning** leaps (invalid implications) require different mitigations. **Detection** (self-consistency, retrieval grounding) and **prevention** (RLHF, constitutional rules) stack together.

**Safety** spans **harmful content** (violence, self-harm), **misuse** (malware assistance), **privacy** leaks, and **bias**. **Red-teaming** adversarially probes models; **guardrails** filter inputs and outputs; **benchmarks** like TruthfulQA and BBQ quantify progress. Interviewers expect you to separate **model** behavior from **system** controls (filters, tool policies, human review).

Finally, **Responsible AI** practices—documentation, incident response, **monitoring** for drift—are shipping requirements under emerging regulations and internal risk frameworks.

---

## Core Concepts

### Types of Hallucination

| Type | Description | Example mitigation |
|------|-------------|--------------------|
| **Factual** | Wrong claim about world | Retrieval grounding, citations, fact-check tools |
| **Faithfulness** | Answer not supported by provided context | Constrained decoding, extractive bias |
| **Reasoning** | Invalid logical step | Verifiers, chain-of-thought critique, formal tools |

Formally, let \(k\) be **known facts** from retrieval. A **faithful** answer \(a\) should satisfy:

\[
\text{Support}(a) \subseteq \text{Entail}(k)
\]

in some entailment semantics—approximated in practice by **NLI models** or LLM judges.

!!! math-intuition "In Plain English"
    Hallucination is **confident BS**. You reduce it by **shrinking freedom**: give evidence, demand citations, or force the model to say “I don’t know” when evidence is missing.

### Detection: Self-Consistency and Retrieval Grounding

**Self-consistency** samples multiple reasoning paths; **disagreement** signals uncertainty. Majority vote or confidence thresholds flag unreliable answers.

**Retrieval grounding** checks whether each sentence is supported by retrieved passages—**sentence-level NLI** or **attribution** models score alignment:

\[
\text{score}(s, p) = P_\phi(\text{entailment} \mid s, p)
\]

### Red-Teaming and Adversarial Testing

**Red-teaming** systematically elicits harmful or policy-violating outputs. Methods include:

- **Manual** probes by domain experts.
- **Automated** attacks (GCG-style suffixes, **prompt injection**).
- **Multi-turn** jailbreaks that gradually shift topic.

Outputs are logged, **triage**d, and fed into **RLHF** / **preference** data collection.

### Guardrails and Content Filtering

**Input filters** block PII, toxic prompts, or jailbreak patterns. **Output filters** run classifiers on generations; **blocked** responses trigger safe fallbacks. **Structured** tool use can enforce **allowlisted** actions only.

Latency stacks: lightweight filters first, **heavy** LLM judges only on borderline cases.

### Responsible AI Practices

- **Model cards** and **system cards** document scope, limits, and metrics.
- **Incident response** when a harmful output ships—rollback, patch prompts, retrain rewards.
- **Bias monitoring** across demographic groups; **human oversight** for high-stakes domains.

### Safety Benchmarks: TruthfulQA, BBQ

**TruthfulQA** measures **truthfulness** on questions humans often answer **wrong** due to misconceptions—models should **avoid** mimicking false popular beliefs.

**BBQ** (Bias Benchmark for QA) tests **social bias** via carefully constructed contexts where **ambiguous** vs **disambiguated** settings reveal whether models rely on **stereotypes**.

### Combining Mitigations

```
User query → Input filter → RAG retrieval → Model generation →
Grounding check → Output filter → Log & audit
```

Each stage can **fail**—defense in depth matters.

### Code: Self-Consistency Vote (Synthetic)

```python
from __future__ import annotations

import collections
import random
from typing import Callable, List, Sequence


def self_consistency_vote(
    sampler: Callable[[], str], n: int = 5
) -> tuple[str, float]:
    """sampler returns one stochastically generated answer string."""
    answers: List[str] = [sampler() for _ in range(n)]
    counter = collections.Counter(answers)
    best, count = counter.most_common(1)[0]
    agreement = count / n
    return best, agreement


def toy_sampler() -> str:
    # pretend model outputs one of two answers with noise
    return random.choice(["Paris", "Paris", "Lyon", "Paris"])


if __name__ == "__main__":
    random.seed(0)
    ans, conf = self_consistency_vote(toy_sampler, n=8)
    print("answer:", ans, "agreement:", conf)
```

### Code: Simple Entailment Stub

Use a real NLI checkpoint (e.g., DeBERTa fine-tuned on MNLI) in production; here is a **placeholder** interface:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class EntailmentResult:
    label: Literal["entailment", "neutral", "contradiction"]
    score: float


def mock_entailment(sentence: str, passage: str) -> EntailmentResult:
    # Replace with transformers pipeline("text-classification", model=...)
    overlap = len(set(sentence.lower().split()) & set(passage.lower().split()))
    if overlap >= 3:
        return EntailmentResult("entailment", 0.9)
    if overlap == 0:
        return EntailmentResult("contradiction", 0.6)
    return EntailmentResult("neutral", 0.5)


def grounded_sentence(sentence: str, passages: list[str]) -> bool:
    return any(
        mock_entailment(sentence, p).label == "entailment" for p in passages
    )
```

### RLHF, DPO, and Safety Alignment

**RLHF** optimizes a policy against a **reward model** that encodes human preferences (helpfulness, harmlessness). **DPO** directly fits preferences without explicit RL loops. Both can **reduce** toxic outputs but may introduce **over-refusal** or **sycophancy**—monitor **false refusals** on benign queries.

### Jailbreaks and Prompt Injection

**Jailbreaks** manipulate the model into **policy-violating** outputs via **roleplay**, **encoding tricks**, or **adversarial suffixes**. **Prompt injection** feeds **untrusted** text (web pages, emails) that instruct the model to **ignore** system policies. Mitigations: **delimiter** hardening, **tool** permission boundaries, **separate** channels for **trusted** vs **untrusted** content.

### Privacy and Data Minimization

**PII** in prompts and tool outputs should be **masked** or **dropped** before logging. **Retention** windows and **region** constraints (GDPR) apply to **fine-tuning** data as well as **inference** logs.

### OWASP LLM Top 10 (Mapping)

Map failures to categories: **Prompt Injection**, **Insecure Output Handling**, **Training Data Poisoning**, **Model Denial of Service**, **Supply Chain** vulnerabilities in **fine-tunes** and **plugins**. Security reviews should **trace** data flows across **tools** and **retrieval** layers.

### Human Oversight and Escalation

For **high-stakes** domains (clinical, legal), route **low-confidence** or **high-risk** topics to **human** review. Confidence can be estimated via **verifier** models, **self-consistency**, or **calibrated** probabilities.

### Factuality vs Plausibility

LLMs optimize **fluency**; **plausible** false statements are **especially** dangerous. Mitigations include **retrieval**, **calibration** training, and **uncertainty** elicitation prompts (“List unknowns before answering”).

### Attribution and Citations

For **RAG** systems, require **citations** to retrieved passages. **Attribution** evaluation checks whether each **sentence** is supported by a **specific** chunk—**not** merely nearby in cosine space.

### Content Policy Layers

Typical stack: **regex** and **keyword** filters → **classifier** (toxicity/PII) → **LLM** policy model → **output** formatter. **False positives** frustrate users—tune thresholds with **human** review of **borderline** cases.

### Adversarial Robustness Limits

**Provable** robustness against **all** prompt injections is **unrealistic** for general LMs—**assume breach** and **limit** blast radius via **tool** sandboxes and **least privilege**.

### Safety vs Capability Benchmarks

Improving **MMLU** does not imply improving **TruthfulQA**. Track **both** capability and **honesty** metrics; watch for **inverse** correlations after **RLHF** (over-refusal).

### Organizational Practices

- **Red-team** before major launches; **periodic** retests after **updates**.
- **Bug bounty** / **internal** reporting channels for harmful outputs.
- **Incident** postmortems with **public** summaries when appropriate.

### TruthfulQA Categories (Conceptual)

**TruthfulQA** spans **common misconceptions** (physics, law, health). Models should **abstain** or **contradict** myths—not **reproduce** them with confident prose. **Fine-tuning** on **helpful** but **false** demonstrations can **hurt** TruthfulQA—watch **data** quality.

### BBQ: Ambiguity and Disambiguation Contexts

**BBQ** contrasts **ambiguous** questions (no stereotype needed) with **disambiguated** contexts (only one **correct** answer). **High** error on ambiguous but **low** on disambiguated suggests **defaulting** to **bias** under uncertainty.

### Faithfulness in Summarization

**Summarization hallucination** introduces **unsupported** details. Evaluate with **QAG** (question-answer consistency) or **NLI** between **summary** sentences and **source** document—**not** ROUGE alone.

### Supply Chain for Models

**Fine-tunes** and **LoRA adapters** can **poison** downstream behavior—verify **checksums**, **signing**, and **provenance** for **weights** loaded at runtime.

### Monitoring in Production

Track **refusal rates**, **toxicity** classifier scores, and **user** reports by **locale** and **product** surface. **Spikes** may indicate **prompt** attacks or **data drift**.

### Decomposition of Harm Categories

Safety incidents split into **malicious use** (user intent), **accidental harm** (wrong medical advice), and **structural bias** (demographic disparities). **Mitigations** differ: **policy** vs **disclaimers** vs **rebalancing** data.

### Evaluation: Automatic vs Human

Automatic **toxicity** classifiers have **false positives** on **minority** dialects—**human** review remains necessary for **fairness** validation.

### Secure Development Lifecycle for LLM Features

Threat-model **untrusted** inputs (web, email), **tool** outputs, and **third-party** plugins. **Assume** prompts and tool returns are **hostile** unless **authenticated**.

### Documentation for Users

Clear **capability** boundaries (“not a lawyer/doctor”) reduce **misplaced** trust—**documentation** is a **safety** intervention.

### Internationalization

Safety policies must account for **multilingual** jailbreaks and **locale-specific** harms—**English-only** filters miss **cross-lingual** attacks.

### Alignment Tax: Capability Trade-offs

**RLHF** can **reduce** raw **capability** on some **benchmarks** while improving **helpfulness/harmlessness** rankings—**report** both **base** and **aligned** checkpoints where possible.

### Verifiable Rewards for Safety

Where **objective** checks exist (unit tests, **compiler** success), **reward** models can be **supplemented** with **verifiable** signals—reducing **reward hacking** surface.

### Child Safety and CSAM Policy

**Zero-tolerance** areas require **hash-matching** pipelines and **legal** escalation paths—**not** model **discretion** alone.

### Enterprise Policy Engines

**Feature flags** can **route** sensitive intents to **blocked** responses or **human** workflows—**defense** in depth beyond the **LM**.

### Audit Trails

Store **hashed** prompts and **policy** decisions for **regulatory** review—**avoid** raw **PII** retention without **consent**.

---

## Interview Takeaways

- Distinguish **factual**, **faithfulness**, and **reasoning** hallucinations—different tools apply.
- **Self-consistency** and **NLI grounding** are **offline** quality signals; **RAG** reduces factual drift when retrieval is correct.
- **Red-teaming** finds **policy failures**; **guardrails** reduce blast radius but can **over-refuse**.
- **TruthfulQA** and **BBQ** probe **truthfulness** and **bias**—not a substitute for domain-specific clinical/legal evals.
- **Safety** is **system-level**: model + filters + tools + monitoring + governance.
- Document **known failure modes** and **escalation paths** for production assistants.

## References

- Ji et al., *Survey of Hallucination in Natural Language Generation* (2023): [arXiv:2202.03629](https://arxiv.org/abs/2202.03629)
- Lin et al., *TruthfulQA: Measuring How Models Mimic Human Falsehoods* (2022): [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
- Parrish et al., *BBQ: A Hand-Built Bias Benchmark for Question Answering* (2022): [arXiv:2110.08193](https://arxiv.org/abs/2110.08193)
- Ganguli et al., *Red Teaming Language Models to Reduce Harms* (2022): [arXiv:2202.03286](https://arxiv.org/abs/2202.03286)
- Anthropic, *Constitutional AI: Harmlessness from AI Feedback* (2022): [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- OWASP Top 10 for LLM Applications: [https://owasp.org/www-project-top-10-for-large-language-model-applications/](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
