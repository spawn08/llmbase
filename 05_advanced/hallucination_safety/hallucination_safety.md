# Hallucination and Safety

## Why This Matters for LLMs

**Hallucination**—generating fluent, confident content that is **unsupported** or **false**—is the leading barrier to deploying LLMs in **medicine**, **law**, **finance**, and **enterprise** knowledge work. Unlike a traditional database that returns “no rows,” language models **always** produce a plausible continuation unless constrained; that optimizes **likelihood** on internet-scale text, where false statements appear next to true ones with similar surface patterns. Production teams must therefore combine **model** improvements with **retrieval**, **verification**, and **policy** layers.

**Safety** extends beyond factuality to **harmful instructions**, **toxicity**, **privacy** leakage from memorization, **jailbreaks** that bypass guardrails, and **dual-use** content (accurate but dangerous). Regulators and enterprise risk teams increasingly expect **red-teaming** evidence, **incident** response plans, and **monitoring** for prompt-injection and data exfiltration. Interviewers probe whether you can separate **intrinsic** model limits from **system** mitigations and articulate **defense in depth**.

Finally, **alignment trade-offs** matter: aggressive refusals reduce **harm** but cause **over-refusal** (blocking benign tasks). **Calibration**—teaching models to say “I don’t know”—interacts with **user trust** and **legal** liability. A strong answer connects **benchmarks** (TruthfulQA, toxicity classifiers) to **real** failure taxonomy and **operational** controls.

---

## Core Concepts

### What Is Hallucination?

**Intrinsic** hallucinations contradict **provided** evidence (critical in **RAG**). **Extrinsic** hallucinations invent facts **not** grounded in sources or user context. Let \(c\) denote retrieved or user-supplied context and \(a\) a generated answer. A **faithfulness** predicate can be written as:

\[
\text{Faithful}(a, c) \iff \text{Entail}(c, a) \ \text{(approximated in practice)} .
\]

!!! math-intuition "In Plain English"
    If your **KB** says “Policy updated 2024,” but the model asserts “2023,” that’s **intrinsic** to the session. If it invents a **nonexistent** citation, that’s often **extrinsic** relative to retrieval—both are bad, but **detection** hooks differ.

**Closed-domain** hallucination: drift from **given** passages. **Open-domain** hallucination: false claims about the **world** without a fixed corpus—harder to check without external tools.

### Why Do LLMs Hallucinate?

Training minimizes **cross-entropy** over next-token prediction:

\[
\mathcal{L}(\theta) = -\sum_{t} \log p_\theta(x_t \mid x_{<t}) .
\]

!!! math-intuition "In Plain English"
    The objective rewards **fluent** continuation, not **truth**. If “CEO is Jane Doe” appears often in training (or in-context), the model may emit it even when outdated—**likelihood** ≠ **factuality**.

Additional drivers: **stale** training data (knowledge cutoff), **parametric** memorization of errors, **reasoning** shortcuts on math (pattern matching without calculation), and **instruction** tuning that rewards **helpful** tone over **cautious** refusal.

### Detection: Confidence from Token Probabilities

A **heuristic** uncertainty signal uses average **negative log-likelihood** of generated tokens \(\hat{x}_{1:T}\):

\[
\bar{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(\hat{x}_t \mid \hat{x}_{<t}, x) .
\]

!!! math-intuition "In Plain English"
    **Higher** average surprise can indicate the model is **wandering**—but confident **wrong** answers can still have **high** probability under the model. Token probabilities are **necessary** but **not sufficient** for hallucination detection.

### Self-Consistency

Sample \(K\) answers \(\{a^{(k)}\}_{k=1}^{K}\). **Agreement rate**:

\[
r = \frac{1}{K(K-1)} \sum_{i\neq j} \mathbf{1}[a^{(i)} = a^{(j)}]
\]

(for exact match; softer metrics compare **semantic** equivalence).

!!! math-intuition "In Plain English"
    If **five** independent samples disagree on a **numeric** fact, **distrust** the model—unless the task is legitimately ambiguous. Self-consistency **costs** \(K\times\) latency.

!!! example "Worked Example: Self-Consistency Vote on a Date"
    Ask: “When was the Eiffel Tower completed?” Sample **4** answers: **1889**, **1889**, **1890**, **1889**. Majority **1889** with high agreement—**low** hallucination risk on the **year** (still verify with a tool). If answers are **1889**, **1936**, **1889**, **2001**, **agreement** collapses—**trigger** retrieval or refusal even if each answer **sounds** confident.

### NLI-Based Grounding

A **natural language inference** model assigns \(\{\text{entailment}, \text{neutral}, \text{contradiction}\}\) between a **sentence** \(s_i\) from the answer and **passage** \(p\):

\[
\ell_i = \arg\max_{y} P_\phi(y \mid s_i, p) .
\]

!!! math-intuition "In Plain English"
    If **many** sentences **contradict** or are **neutral** when they should **entail**, the answer is **not grounded**—flag for **regeneration** or **human** review.

### SelfCheckGPT-Style Consistency Without Reference

**SelfCheckGPT** compares multiple **sampled** answers and measures **consistency**—low consistency implies higher **hallucination** risk **without** external facts. A stylized score:

\[
\text{score} = 1 - \frac{1}{K}\sum_{k=1}^{K} d\big(a^{(k)}, \bar{a}\big)
\]

where \(d\) is an embedding distance and \(\bar{a}\) a **centroid** representation.

!!! math-intuition "In Plain English"
    If the model **cannot** agree with itself across samples, it may be **making things up**—especially for **rare** facts. This catches some failures **reference-free** but can **false-alarm** on creative tasks.

### Mitigation: RAG and Constrained Decoding

**RAG** conditions generation on retrieved documents \(d_1,\ldots,d_m\):

\[
P(a \mid x) \approx P_\theta(a \mid x, d_1,\ldots,d_m) .
\]

!!! math-intuition "In Plain English"
    Pulling **evidence** into context **anchors** tokens to real text—still not perfect if retrieval is wrong or the model **ignores** docs.

**Constrained decoding** restricts tokens to a **grammar** or **JSON schema**, shrinking the output space:

\[
\hat{a} = \arg\max_{a \in \mathcal{G}} P_\theta(a \mid x)
\]

where \(\mathcal{G}\) is the set of valid strings.

!!! math-intuition "In Plain English"
    If the answer **must** be a **date** field from an API, the model **cannot** invent a new century—**structure** reduces **creativity** where it hurts.

### Calibration and Abstention

Train or prompt the model to emit **abstain** token or **low-confidence** flags when evidence is weak. A **cost-sensitive** decision rule compares expected loss of **wrong** answer vs **abstain**:

\[
\text{answer if } \mathbb{E}[\text{loss}_{\text{wrong}}] \cdot (1-p) < \text{loss}_{\text{abstain}} .
\]

!!! math-intuition "In Plain English"
    If **wrong** answers are **expensive** (medical), lower the **threshold** for “I don’t know”—**utility** shapes **risk**.

### Safety: Jailbreaks and Prompt Injection

**Jailbreaks** elicit policy-violating outputs via **roleplay**, **encoding** tricks, or **multi-turn** priming. **Prompt injection** hides instructions inside **untrusted** text (emails, web pages) that **overrides** developer prompts.

Let **system** prompt \(S\) and **untrusted** content \(u\) concatenate into model input. Attacks seek \(\hat{y}\) that violates policy \(\mathcal{P}\):

\[
\exists u \ \text{s.t.}\ \neg \mathcal{P}(\hat{y}) \ \text{and high } P_\theta(\hat{y} \mid S, u) .
\]

!!! math-intuition "In Plain English"
    If the model **obeys** embedded instructions in \(u\), your **system** prompt lost—**separate** trusted and untrusted channels where possible, and **never** trust retrieved HTML as **instructions**.

### Guardrails: Layered Defense

**Input filters** scan for PII, toxicity, injection patterns. **Output filters** run classifiers and blocklists. **Policy engines** enforce tenant rules. Combined:

\[
\text{release}(y) = \mathbf{1}[\text{In}(x)\land \text{Out}(y) \land \text{Pol}(y)] .
\]

!!! math-intuition "In Plain English"
    **No** single filter catches everything—**layer** lightweight checks first, **expensive** LLM judges on **borderline** cases only.

### Alignment Tax and Over-Refusal

**Safety training** (RLHF, refusal tuning) can **reduce** helpfulness on **edge** tasks. Let \(U_h\) be helpfulness utility and \(U_s\) safety. A Pareto tension:

\[
\max_\theta\ U_h(\theta)\quad \text{s.t.}\quad U_s(\theta) \ge \tau .
\]

!!! math-intuition "In Plain English"
    Pushing **safety** \(\tau\) higher shrinks the **feasible** region—users see more **refusals**. Product teams tune \(\tau\) per **surface** (consumer vs internal tools).

### Memorization and Extraction Risk

Let \(x\) be a training example. The **memorization** probability can be modeled coarsely as increasing with **exposure count** and **model capacity**. A stylized bound might track the chance that a **prefix** of length \(m\) **uniquely** identifies a sensitive string:

\[
P(\text{emit } x \mid \text{prefix}) \approx \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t}) .
\]

!!! math-intuition "In Plain English"
    Rare **long** sequences with **high** per-token probability under the model are **extractable**—especially if they appeared **many** times in training. **Deduplication** and **privacy-preserving** training reduce this mass.

### Red-Teaming Workflow (Operational)

**Red-teaming** pairs **domain experts** and **adversarial** testers to elicit failures. Outputs feed a **risk register**: severity, likelihood, mitigation, owner. A **risk score** might combine impact \(I\) and exploitability \(E\):

\[
R = I \cdot E .
\]

!!! math-intuition "In Plain English"
    Not every jailbreak is **P0**—prioritize by **user harm** and **ease** of exploitation in your **actual** UI (e.g. file upload enabling **indirect** injection).

!!! example "Worked Example: Scoring a RAG Answer with NLI"
    **Context:** “Our SLA is **99.9%** uptime measured monthly.” **Model:** “Your SLA guarantees **99.99%** uptime.” NLI between **claim** and **context** → likely **contradiction** or **neutral** (numbers differ)—**block** or **revise**. **Numerical** checks beat prose-only NLI for **digits**—combine **regex** extraction with **compare**.

??? deep-dive "Deep Dive: Process Supervision vs Outcome Supervision"
    **Outcome supervision** rewards only final answers; **process supervision** rewards **correct intermediate steps** (e.g. in math). Process rewards can reduce **lucky guesses** but require **expensive** labels or verifiable **step** objects.

??? deep-dive "Deep Dive: Memorization and Privacy Attacks"
    Models can **emit** training data verbatim (**extraction** attacks). Mitigations: **differential privacy** (expensive), **deduplication**, **canaries**, **output** filters comparing against known PII patterns, and **minimizing** sensitive data in fine-tuning sets.

---

## Code

The script below uses **transformers** pipelines for **text-classification** (toxicity proxy) and **NLI** (entailment). Install:

```bash
pip install torch transformers numpy scipy
```

First run downloads model weights.

```python
"""
hallucination_safety_examples.py — SelfCheck-style consistency, guardrail sketch, NLI grounding.
pip install torch transformers numpy scipy
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def sample_selfcheck_consistency(
    gen_pipe,
    embed_pipe,
    prompt: str,
    k: int = 4,
    max_new_tokens: int = 64,
) -> tuple[list[str], float]:
    """Sample K completions; mean pairwise cosine distance on MiniLM embeddings."""
    outs: list[str] = []
    tok = gen_pipe.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for _ in range(k):
        out = gen_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=tok.pad_token_id,
        )
        text = out[0]["generated_text"]
        outs.append(text[len(prompt) :].strip())
    vecs = embed_pipe(outs)
    pooled = [v.numpy().mean(axis=0) for v in vecs]
    dists: list[float] = []
    for i in range(len(pooled)):
        for j in range(i + 1, len(pooled)):
            dists.append(float(cosine(pooled[i], pooled[j])))
    mean_dist = float(np.mean(dists)) if dists else 0.0
    return outs, mean_dist


@dataclass
class GuardrailResult:
    blocked: bool
    reason: str


def toxicity_guard(text: str, clf) -> GuardrailResult:
    """Block when toxic label beats non-toxic (unitary/toxic-bert: LABEL_0 ~ non-toxic)."""
    raw = clf(text)[0]
    by_lab = {str(x["label"]): float(x["score"]) for x in raw}
    toxic = by_lab.get("toxic", by_lab.get("LABEL_1", 0.0))
    nontoxic = by_lab.get("non-toxic", by_lab.get("non_toxic", by_lab.get("LABEL_0", 0.0)))
    if toxic > 0.55 and toxic >= nontoxic:
        return GuardrailResult(True, f"toxic={toxic:.3f}, nontoxic={nontoxic:.3f}")
    return GuardrailResult(False, "ok")


def nli_entailment_label(premise: str, hypothesis: str) -> tuple[str, np.ndarray]:
    """3-way NLI with a small cross-encoder (contradiction / entailment / neutral)."""
    name = "cross-encoder/nli-distilroberta-base"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    mdl.eval()
    inp = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = mdl(**inp).logits
    probs = torch.softmax(logits, dim=-1).numpy().ravel()
    id2label = mdl.config.id2label
    pred = int(probs.argmax())
    return str(id2label[pred]), probs


def extract_numbers(s: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", s)


def main() -> None:
    # --- Toxicity ---
    tox = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    print("Toxicity raw:", tox("I love this product!"))

    # --- NLI grounding (premise = evidence, hypothesis = claim) ---
    ev = "The company SLA promises 99.9% uptime measured monthly."
    ans = "The SLA guarantees 99.99% uptime."
    label, probs = nli_entailment_label(ev, ans)
    print("NLI label:", label, "probs:", probs)

    print("Guard benign:", toxicity_guard("You are wonderful!", tox))
    print("Guard stressed:", toxicity_guard("That is the worst idea I have ever heard.", tox))

    nums_ans = extract_numbers(ans)
    nums_ev = extract_numbers(ev)
    print("Numbers evidence vs answer:", nums_ev, nums_ans)

    # --- Self-consistency (downloads gpt2 + MiniLM on first run) ---
    try:
        gen = pipeline("text-generation", model="gpt2")
        emb = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        prompt = "The capital of France is"
        outs, dist = sample_selfcheck_consistency(gen, emb, prompt, k=3, max_new_tokens=16)
        print("Self-check samples:", outs)
        print("Mean pairwise cosine distance:", dist)
    except Exception as exc:  # noqa: BLE001
        print("Skipping self-check demo:", exc)


if __name__ == "__main__":
    main()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Define intrinsic versus extrinsic hallucination and give a RAG example of each.
    *Answer:* **Intrinsic** hallucination contradicts **provided** context (e.g., KB says “2024 policy” but the model answers “2023”). **Extrinsic** hallucination invents facts **not supported** by any supplied source—e.g., fabricated citation IDs or numbers when retrieval was empty. In RAG, intrinsic failures break **faithfulness** to passages; extrinsic failures often mean the model’s **parametric** prior overrode or filled gaps without evidence.
    2. Why does minimizing cross-entropy not guarantee factual correctness?
    *Answer:* Training maximizes likelihood of **next tokens** in web-scale text where **false** and **true** sentences share fluent patterns—objective rewards **plausibility**, not **truth**. The model also memorizes **stale** or **biased** statistics. Without retrieval, tools, or RLHF/verification objectives, low loss can coexist with confident fabrication on rare facts.
    3. How does self-consistency help detect hallucinations, and when does it fail?
    *Answer:* Sampling \(K\) answers and measuring **agreement** flags **instability** on facts—if the model gives different dates, distrust. It fails when errors are **systematic** (same wrong prior every sample), on tasks where **diversity** is legitimate (creative writing), or when samples are **too correlated** (low temperature)—then agreement is misleadingly high.
    4. Explain NLI-based grounding and one limitation on numerical claims.
    *Answer:* Split the answer into **claims** and run an NLI model between each claim and **evidence** passages—entailment supports grounding; contradiction flags hallucination. **Limitation:** NLI is **fuzzy** on numbers (“99.9%” vs “99.99%”)—treat digits with **structured** compare or regex extraction, not prose entailment alone.
    5. What is prompt injection, and how does it differ from a classic jailbreak?
    *Answer:* **Prompt injection** hides **instructions inside untrusted content** (email, webpage, retrieved doc) that the model **obeys**, hijacking app behavior—an **integrity** attack on the system boundary. **Jailbreaks** elicit **policy-violating** outputs from the model itself (roleplay, encoding tricks)—often about **safety** bypass, not necessarily untrusted third-party data channels. Defenses differ: channel separation and output sandboxing for injection; robust alignment and monitors for jailbreaks.
    6. Name three layers of a defense-in-depth guardrail stack for a customer chatbot.
    *Answer:* (1) **Input** filters: PII/toxicity/injection scanners; (2) **Runtime** controls: RAG with ACLs, schema-constrained tools, citation requirements; (3) **Output** filters: policy classifiers, blocklists, escalation to human on high-risk topics. Add **monitoring** and **kill switches** as operational layers—no single gate suffices.
    7. What is the alignment tax, and how might over-refusal show up in metrics?
    *Answer:* The **alignment tax** is the **helpfulness** you sacrifice to meet **safety** constraints—stricter refusals shrink valid task coverage. **Over-refusal** appears as rising **false refusal rate** on benign prompts, lower **task completion**, user frustration signals, and **disparate** impact on dialects or domains flagged incorrectly. Track slice metrics—not just aggregate harm rate.
    8. How would you combine RAG with citation verification?
    *Answer:* Require **inline citations** to chunk ids, then **post-check** that cited spans **support** each sentence (NLI or entailment model) and that chunk ids exist in the retrieved set. Reject or regenerate when citations are missing or mismatched—**structure** (numbered passages) makes automated verification feasible. Log verification failures for index tuning.
    9. What privacy risks arise from memorization, and one mitigation?
    *Answer:* Models can **verbatim-extract** training data—emails, phone numbers, secrets—especially for **high-exposure** rare sequences (**memorization** / extraction attacks). **Mitigation:** **deduplication**, **canary** monitoring, **minimize** sensitive data in fine-tunes, **output** filters for PII patterns, and **DP** training when budgets allow—operational retention limits matter too.
    10. Describe red-teaming: who participates, what artifacts are produced, and how outputs feed the next training cycle?
    *Answer:* **Participants:** safety researchers, domain experts (legal/medical), and adversarial testers simulating creative misuse. **Artifacts:** structured **failure reports**, severity-ranked **risk register**, reproduction prompts, and suggested mitigations. **Feedback loop:** curate **RLHF/DPO** preference data and **SFT** corrections from confirmed issues, update **filters** and **policies**, and **regress** with expanded attack suites—closing the loop without only one-off fixes.

!!! interview "Follow-up Probes"
    - “Users want **creative** writing—how do you avoid false positives from consistency checks?”
    - “Your toxicity model flags **medical** discussion—what now?”
    - “How do you evaluate **faithfulness** separately from **fluency**?”
    - “What breaks if **all** traffic goes through a single LLM judge?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Faithfulness** is entailment between **claims** and **evidence**, not fluency.”
    - “**Defense in depth**: retrieval, structured outputs, filters, and human escalation.”
    - “**Self-consistency** flags instability; **NLI** checks grounding to context.”
    - “**Prompt injection** treats untrusted content as data, never as instructions.”
    - “**Calibration** and **abstention** trade coverage for reliability in high-stakes domains.”

---

## References

1. Ji, Z., et al. (2023). *Survey of Hallucination in Natural Language Generation.* ACM Computing Surveys.
2. Maynez, J., et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization.* ACL.
3. Manakul, P., et al. (2023). *SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models.* arXiv:2303.08896.
4. Christiano, P., et al. (2017). *Deep Reinforcement Learning from Human Preferences.* NeurIPS.
5. Bai, Y., et al. (2022). *Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback.* arXiv:2204.05862.
6. Greshake, K., et al. (2023). *Not What You’ve Signed Up For: Compromising Real-World LLM-Integrated Applications With Indirect Prompt Injection.* arXiv:2302.12173.
7. Zou, A., et al. (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
8. Lin, S., et al. (2022). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* ACL.
9. Parrish, A., et al. (2022). *BBQ: A Hand-Built Bias Benchmark for Question Answering.* ACL.
10. OWASP *LLM Top 10* (project documentation) — systematic taxonomy for LLM application risks.
