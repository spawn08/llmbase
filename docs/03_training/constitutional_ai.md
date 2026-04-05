# Constitutional AI

## Why This Matters for LLMs

**Constitutional AI (CAI)** is Anthropic’s framework for training models using **written principles** (“constitution”) and **scalable supervision**—often **AI-generated critiques and revisions** instead of only human labels. Interviewers probing **alignment at scale** expect you to contrast **RLAIF** (AI feedback) with **RLHF** (human feedback), and to explain **critique → revise → supervised learning** loops versus **reward modeling + PPO**.

A second reason is **governance**: explicit constitutions make **value trade-offs** discussable in natural language (“be helpful, harmless, honest”) and enable **transparent** iteration—useful when comparing to opaque reward models trained only on **scalar** preferences.

Third, **red-teaming automation** pairs CAI with **classifiers** and **policy-staged** generation to find failure modes **before** deployment. Systems questions often connect CAI-style pipelines to **evaluation harnesses** and **continuous** monitoring.

---

## Core Concepts

### Principles and the Constitution

A **constitution** is a set of **rules** \(C = \{c_1,\dots,c_m\}\) expressed in **natural language** (e.g., “choose the response that is least harmful”). Training uses these rules to **synthesize** labels: an **AI critic** judges outputs against principles, producing **preference** signals or **revision** targets.

!!! math-intuition "In Plain English"
    Instead of only asking humans “which answer is better?”, you first ask “**which answer better satisfies these principles?**” The principles act as a **portable supervision interface**—you can edit them without re-collecting all human demonstrations.

### RLAIF: Reinforcement Learning from AI Feedback

**RLAIF** replaces **human** preference labels with **model-generated** preferences—often from a **stronger** or **specialized** model, or the same model in a **self-critique** setup with careful **calibration**. The **RL outer loop** can mirror RLHF (train reward model → PPO) but the **label source** differs.

| Aspect | RLHF | RLAIF |
|--------|------|-------|
| Preference source | Humans | AI model(s) |
| Cost | High latency / \$ | Lower marginal cost |
| Risks | Human inconsistency | **Model bias** amplification |
| Mitigation | Multiple raters, adjudication | **Ensemble** critics, constitutions, **human spot checks |

### Critique–Revision Loop

1. **Generate** candidate answer \(a\) to prompt \(x\).
2. **Critique** \(a\) with respect to constitution \(C\) → structured feedback \(z\).
3. **Revise** to produce \(a'\) incorporating critique.
4. **Supervise** the model on \((x, a')\) or **preference** pairs \((a', a)\).

This is **not** necessarily RL—it can be pure **SFT** on revised text, or **preference** training if you keep pairs.

!!! math-intuition "In Plain English"
    You are building a **tiny editorial process** inside the data factory: draft → editor notes → second draft. The model learns not only **final answers** but sometimes **rationales** if you train on critiques too.

### Red-Teaming Automation

**Red-teaming** probes harmful outputs with adversarial prompts. Automation uses **mutation** operators, **LLM adversaries**, and **coverage** metrics across **harm categories**. Outputs feed **classifier** training, **policy** updates, or **blocked** prompt lists.

### Comparison with RLHF

- **RLHF**: **scalar reward** from human rankings—flexible but **opaque**; **PPO** optimization.
- **CAI**: often **principle-grounded** **self-supervision** + **SFT/DPO**-style losses; may still use RL for some stages in hybrid systems.
- **Data efficiency**: CAI can **generate** many **principle-labeled** examples cheaply; quality depends on **critic** capability.

Anthropic’s public materials emphasize **helpful–harmless** trade-offs encoded via **multi-objective** principles and **chain-of-thought**-style critique models (internal details evolve).

### Process Supervision vs Outcome Supervision

**Outcome supervision** scores only final answers. **Process supervision** (related to CAI-style critique) rewards **intermediate reasoning**—when critiques identify **specific** flawed steps, gradients can target **where** errors arise. Not identical to CAI, but **interview-linked**: “**Show your work** supervision reduces **spurious** correct answers.”

### Self-Critique and Constitutional Chain-of-Thought

A generic pattern:

1. Sample answer \(a \sim \pi(\cdot \mid x)\).
2. Sample critique \(z \sim \pi(\cdot \mid x, a, \text{“critique per principles”})\).
3. Sample revision \(a' \sim \pi(\cdot \mid x, a, z)\).

Train \(\pi\) on **demonstrations** of \((x, z, a')\) or **distill** a **student** from a **teacher** that executes longer pipelines—**budget** trades quality vs latency.

!!! math-intuition "In Plain English"
    You are **augmenting trajectories** with **editorial metadata** \(z\). Even if deployment **does not show** critiques to users, training on them can **shape** internal computations (depending on architecture and training target).

### Multi-Objective Principles as Lagrangian (Sketch)

Suppose principles induce **proxy losses** \(\ell_k\) (e.g., toxicity classifier). A **Lagrangian** training objective:

\[
\mathcal{L} = \mathcal{L}_{\text{LM}} + \sum_k \lambda_k \, \ell_k
\]

with **dual** updates on \(\lambda_k\) to meet **constraint** targets (e.g., toxicity rate \(\le \epsilon\)). CAI papers may not present it this way, but **interviews** connect to **constrained RL** and **reward shaping**.

---

## Supervised vs Preference Stages (Qualitative)

1. **Harmless SFT** on curated demonstrations.
2. **Self-critique / revision** data augmentation.
3. **Preference modeling** (human and/or AI labels).
4. **RL** (optional) with KL to reference—similar to InstructGPT **stage 2–3**.

Exact **stitching** varies by product; interviews reward **clear separation** of **data stage** vs **optimization** stage.

---

## Anthropic CAI Pipeline (Public Narrative, Qualitative)

While internal recipes evolve, the **public CAI story** follows a **two-stage** arc:

### Stage A — Supervised Learning from AI-Generated Harmlessness Data

- Start from a **base** model.
- Use **constitutional principles** to **automatically** produce **critiques** and **revisions** of harmful/undesirable outputs.
- **Fine-tune** on **revised** responses (and optionally on **chain-of-thought** critiques if included in training targets).

**Goal**: inject **harmlessness** behaviors without proportional growth in **human** red-team labor.

### Stage B — RL from AI-Generated Preference Sets

- Build **preference** pairs using **AI** judges guided by the same constitution.
- Train a **preference model** or directly optimize with **RLAIF**-style objectives (implementation may mirror RLHF with different label sources).

**Goal**: generalize **beyond** demonstration coverage using **exploration** in policy space—similar to RLHF’s **PPO** stage but with **AI-labeled** preferences.

!!! math-intuition "In Plain English"
    Stage A is **imitation** of **good behavior** after editorial critique. Stage B is **optimization** against a **learned preference** landscape—where the labels came from **AI**, not crowdsourced humans.

---

## Debate and Multi-Agent Oversight (Context)

**Debate** proposals (Irving et al.) pit **two** models against each other with **human** or **strong judge** oversight—related **spirit** to critique loops but different **game structure**. CAI’s **single-model** critique is simpler operationally; debate systems aim for **scalable** oversight when **tasks** are **verifiable**.

---

## Python: Synthetic Preference from Rules (Toy)

```python
"""
Toy: choose higher score under hand-coded 'principle' weights — not real CAI.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Response:
    text: str
    toxicity: float  # lower better
    length: int


def principle_score(r: Response) -> float:
    return -1.0 * r.toxicity - 0.001 * r.length


def choose(a: Response, b: Response) -> Response:
    return a if principle_score(a) >= principle_score(b) else b


if __name__ == "__main__":
    r1 = Response("Sure, I can help.", toxicity=0.1, length=20)
    r2 = Response("lol sure", toxicity=0.6, length=8)
    print(choose(r1, r2).text)
```

---

## Python: Critique–Revision Simulation (Strings)

```python
"""
Non-ML simulation of critique/revision steps for data generation pipelines.
"""
from __future__ import annotations


def critique(answer: str, principles: list[str]) -> str:
    issues = []
    if "hack" in answer.lower():
        issues.append("violates safety principle: no illicit instructions")
    if len(answer) > 200:
        issues.append("violates conciseness principle")
    if not issues:
        issues.append("no major issues detected")
    header = "Principles:\n- " + "\n- ".join(principles)
    return header + "\nFindings:\n- " + "\n- ".join(issues)


def revise(prompt: str, draft: str, critique_text: str) -> str:
    # Placeholder: real system would call an LM
    if "illicit" in critique_text:
        return "I can’t help with that request."
    return draft.strip()


if __name__ == "__main__":
    principles = ["Be helpful", "Be harmless", "Be honest"]
    draft = "Sure, I can help you hack that."
    cr = critique(draft, principles)
    print(cr)
    print("--- revised ---")
    print(revise("user asks for hacking", draft, cr))
```

---

## Evaluation Metrics for Harmlessness

- **Violation rate** on **red-team** suites (per category: violence, self-harm, etc.).
- **False refusal rate** on **benign** prompts (over-refusal is a product failure).
- **Win-rate** against **baselines** in **pairwise** human evals.
- **Robustness** to **paraphrase** attacks and **multi-turn** escalation.

!!! math-intuition "In Plain English"
    Harmlessness is **not** one number—it's a **vector** of failure modes. Constitutional training tries to shift the whole vector, but **evaluation** must cover **each** coordinate.

---

## Constitutional AI vs Instruction Tuning Alone

**Instruction tuning** teaches **format** and **task following**. **CAI** targets **normative** behavior (harmlessness) using **principles** and often **self-supervision**. They **compose**: you might **SFT** on instructions, then **CAI**-style **revision** data, then **DPO** on preferences.

---

## Failure Modes

- **Critic–generator agreement**: weak critics **approve** harmful outputs.
- **Principle conflicts**: “be concise” vs “explain thoroughly”—needs **priority** rules.
- **Self-preference bias**: model prefers its **own** style when used as judge.
- **Coverage**: constitutions may **under-specify** edge cases found in deployment.

### Human-in-the-Loop Spot Checks

Even with **AI feedback**, production systems **sample** trajectories for **human** review—especially near **policy** updates or **novel** domains (medical, legal). **Active learning** prioritizes **uncertain** or **high-impact** prompts for human labels.

---

## Hybrid Pipelines: CAI + RLHF/DPO

Modern stacks rarely choose **only one** supervision type:

- **SFT** on human + AI-revised demonstrations.
- **Preference** training on **mixed** human and AI labels (AI for **cheap** breadth, humans for **high-stakes**).
- **Optional RL** with **KL** to **reference** to optimize **long-horizon** rewards.

Interview framing: “**CAI is a data-generation philosophy**; **RLHF/DPO are optimization shells** around preference data.”

---

## Red-Teaming: Operational Loop

1. **Generate** candidate harmful prompts (manual + automated).
2. **Execute** model policies with **temperature** sweeps.
3. **Label** outcomes with **harm rubrics** (policy violation categories).
4. **Feed** into **blocked** prompt classifiers, **SFT** negatives, or **reward** penalties.
5. **Regression-test** after each model release.

---

## Governance and Transparency

Constitutions are **human-readable**, enabling **policy teams** to **diff** changes between releases. Contrast with **end-to-end** reward models where failure analysis may require **probing** latent spaces. Trade-off: **natural-language rules** can be **gamed** by **sycophantic** models that **appear** compliant.

---

## Specifying Principles: Style vs Substance

Principles can target **tone** (“be concise”) or **substance** (“refuse illegal instructions”). **Substance** rules need **grounding** in policy and law; **style** rules need **examples** of acceptable length and format. Poorly scoped principles produce **inconsistent** AI labels—**iterate** constitutions like **prompts**, with regression suites after each edit.

### Example Principle Blocks (Illustrative, Not Prescriptive)

1. **Harmlessness**: “Choose the response that **minimizes** risk of enabling illegal acts, even if less detailed.”
2. **Honesty**: “If uncertain, **say so** and avoid fabricated citations.”
3. **Helpfulness**: “Prefer answers that **directly** address the user’s stated goal.”

Real systems contain **dozens** of principles with **priority** rules; public papers often show **short** excerpts only.

### Recording and Versioning Constitutions

Treat constitutions like **code**: store in **git**, tag with **model releases**, and run **diff** reviews when **policy** changes. Teams often pair **principle IDs** (e.g., `HARM-01`) with **eval** cases so regressions are **traceable** to a rule change.

### Auditing AI-Labeled Data

When **critics** generate labels, sample **stratified** subsets for **human** audit: focus on **high-stakes** domains (medical, legal), **ambiguous** principles, and **low-confidence** model judgments (if a **calibrated** score exists). **Inter-annotator** agreement between **AI** and **human** judges should be tracked over time—**drift** signals **constitution** or **base model** updates gone wrong.

### Multilingual and Jurisdictional Constitutions

Principles written in **English** may not **transfer** culturally or legally across locales. Production teams maintain **locale-specific** addenda (e.g., EU privacy emphasis) and test **translated** prompts for **consistent** refusals. Interview angle: “**Constitutional AI is not one-size-fits-all** across languages and jurisdictions.”

---

## Clarifying Questions (Interview Style)

**Q: Is Constitutional AI the same as RLAIF?**  
A: **RLAIF** names **where preferences come from** (AI vs human). **Constitutional AI** names **how** supervision is structured (principles + critiques + revisions). Anthropic’s CAI work **uses** AI feedback—so overlap is large, but **not every** RLAIF system uses a **written constitution**.

**Q: Do you still need humans?**  
A: Yes—for **audits**, **edge cases**, **value judgments** not captured in rules, and **benchmark** validation. CAI **reduces** marginal human labeling cost; it does not **eliminate** accountability.

**Q: Can constitutions conflict?**  
A: Yes. Production systems define **priority** orders (safety over verbosity) or use **multi-objective** optimization with **tunable** weights. **Interview tip**: discuss **explicit trade-offs** rather than pretending principles are always compatible.

**Q: How does this interact with jailbreak research?**  
A: **Jailbreaks** exploit **mismatches** between training and deployment prompts. Constitutional supervision must be paired with **inference-time** mitigations (classifiers, system prompts, tool policies) and **continuous** red-teaming.

---

## Benchmarks and Safety Suites (Context)

**Holistic** evaluation stacks (HELM-style) include **accuracy**, **calibration**, **robustness**, **fairness**, and **toxicity**. **Harm-specific** suites (e.g., adversarial prompts, toxicity classifiers on outputs) complement **general** knowledge benchmarks. Constitutional training should improve **harm** metrics without **catastrophically** dropping **capability** metrics—**Pareto** thinking matters in interviews.

---

## Extended Takeaway: When to Recommend CAI-Style Pipelines

Recommend **constitution + critique** when:

- You need **scalable** harmlessness labels **without** proportional human labeling.
- You want **auditable** rules for **policy** iteration.
- You can invest in **strong** automated **critics** and **human spot checks**.

Be cautious when:

- The **base** model cannot **understand** principles reliably (critiques become **noise**).
- **Stakeholders** cannot agree on **written** rules—**ambiguity** transfers into **training noise**.

---

## Interview Takeaways

- **Constitutional AI**: train using **explicit principles** + **scalable** oversight (critiques, AI labels).
- **RLAIF**: **AI** preferences replace/augment human labels—watch **bias loops**.
- **Critique–revision**: **data flywheel** for harmless, helpful outputs.
- **vs RLHF**: CAI emphasizes **interpretable rules** and **process** supervision; RLHF emphasizes **human scalar** rewards.
- **Red-teaming**: pair **automated** probing with **human** review for unknown unknowns.

---

## References

- Bai et al., *Constitutional AI: Harmlessness from AI Feedback* — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) — core CAI + RLAIF empirical results
- Lee et al., *RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback* — [arXiv:2309.00267](https://arxiv.org/abs/2309.00267) — AI vs human preference labels at scale
- Anthropic — public blog posts on **helpful–harmless** training and **red-teaming** — qualitative process narrative
- Perez et al., *Discovering Language Model Behaviors with Model-Written Evaluations* — automated eval generation
- Ganguli et al., *Red Teaming Language Models to Reduce Harms* — methods context for adversarial probing
- Kirk et al., *Understanding Bias in Language Models* — caution on feedback loops
- Irving et al., *AI Safety via Debate* — [arXiv:1805.00899](https://arxiv.org/abs/1805.00899) — related oversight framing
- Ziegler et al., *Fine-Tuning Language Models from Human Preferences* — precursor to large-scale preference learning
- Ouyang et al., *InstructGPT* — human baseline RLHF pipeline for comparison
