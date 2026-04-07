# Instruction Tuning (SFT)

## Why This Matters for LLMs

**Supervised fine-tuning (SFT)** on instruction–response pairs is the standard bridge between a **base** language model (trained on raw web text) and an **assistant** that follows user intent. Interviewers expect you to describe **dataset formats**, **loss masking** (train only on assistant tokens), and failure modes like **catastrophic forgetting** when SFT narrows the distribution too aggressively. This is also the layer where **safety** and **tone** are easiest to inject—before RLHF/DPO preference optimization.

A second reason is **evaluation**: SFT quality is measured both by **perplexity on completions** (weak signal) and by **task success** on held-out instructions. Understanding **why** we mask prompts connects to **training efficiency** (don’t waste gradients predicting user text we already conditioned on) and to **API parity** with inference-time prompt templates.

Third, **FLAN** and **InstructGPT** established that **multi-task instruction mixtures** plus **human-written demonstrations** unlock zero-shot generalization across benchmarks. Modern stacks (OpenAI, Anthropic, open recipes) blend **synthetic** and **human** data—SFT is no longer “just CSV files,” it is **curriculum design**.

---

## Core Concepts

### FLAN and Task Mixtures

**FLAN** (Wei et al.) fine-tuned on **phrased tasks** (“translate English to French: …”) across many NLP datasets, showing **instruction tuning** improves **held-out task** generalization when tasks are **described** at training time. The key idea: **describe the task in natural language** so the model learns to **read instructions**, not only patterns in a single benchmark’s format.

### InstructGPT Motivation

**InstructGPT** (Ouyang et al.) starts from a **base LM**, performs **SFT** on human demonstrations, then **RLHF**—but **SFT alone** already yields a large behavior upgrade: following explicit instructions, refusing some unsafe requests per policy, matching **desired style**. SFT is the **anchor** that RL later **refines**.

!!! math-intuition "In Plain English"
    **Base models** complete text like the internet (continue paragraphs). **Instruction-tuned models** treat the user message as a **command channel** and the assistant channel as **what to optimize**. Same weights, different **conditional distribution** emphasis.

### Supervised Fine-Tuning Pipeline (Typical)

1. **Collect** prompt–completion pairs \((u_i, a_i)\) with optional **system** message \(s_i\).
2. **Template** into a **chat format** (ChatML, ShareGPT, Alpaca, etc.).
3. **Tokenize** end-to-end as one sequence.
4. **Mask** loss on non-assistant tokens (details below).
5. Train with **AdamW**, smaller LR than pre-training, **few epochs** (often 1–3 passes).

### Dataset Formats: Instruction / Input / Output

**Alpaca-style**:

```json
{
  "instruction": "Explain what a transformer block does.",
  "input": "",
  "output": "A transformer block applies self-attention..."
}
```

**Chat-style** (multi-turn):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2 equals 4."}
  ]
}
```

Templates insert **special tokens** (`<|user|>`, `<|assistant|>`) so the model learns **role boundaries**.

### Chat Templates vs. Raw Concatenation

| Style | Pros | Cons |
|-------|------|------|
| **Plain concat** (Alpaca) | Simple, portable | Easy to **leak** delimiter tokens if not escaped |
| **ChatML / role tokens** | Clear **role** boundaries | Must match **inference** tokenizer additions |
| **Tokenized masks** | Exact **alignment** for loss | Template changes **invalidate** old checkpoints’ chat heads |

### Synthetic Data and Self-Instruct

**Self-Instruct** bootstraps instruction data from a seed set by generating **new instructions + instances** with the model itself, then **filtering** (heuristics, ROUGE dedup, optional human review). Quality hinges on **filters**—otherwise the model **amplifies** its own biases.

### Rejection Sampling and SFT Augmentation

Some pipelines **sample** \(N\) completions per prompt, **score** them with a reward model or rule-based checks, and keep the **best** for SFT. This is **not** RL yet—still **supervised** on chosen completions—but shifts the distribution toward **higher reward** regions cheaply.

---

## Loss Masking: Only on Completion Tokens

Let tokenized sequence be \(x_{1:T}\). Partition indices into **prompt** set \(\mathcal{P}\) and **completion** set \(\mathcal{C}\) (assistant). **Masked** cross-entropy:

\[
\mathcal{L}_{\text{SFT}} = - \sum_{t \in \mathcal{C}} \log p_\theta(x_t \mid x_{<t})
\]

Positions \(t \in \mathcal{P}\) contribute **zero** loss—gradients do not push the model to **reconstruct** the user’s prompt (we already observed it).

!!! math-intuition "In Plain English"
    You are **not** trying to compress the user message; you are learning **what the assistant should say next** given that message. Masking avoids wasting capacity on **easy** prompt reconstruction and prevents trivial **copy** solutions.

### Multi-Turn Masking

For conversations, typically **all** prior user + assistant turns are in context, but loss applies only to **assistant** segments (sometimes excluding **tool** outputs depending on recipe). **System** prompts are masked out as well.

---

## Catastrophic Forgetting and Mitigation

**Forgetting**: after SFT, the model may **lose** base capabilities (e.g., Python syntax, rare facts) because the fine-tune distribution is **narrow**. Mitigations:

- **Mix** instruction data with **small fraction** of raw pre-training–style text (continued LM loss).
- **LoRA / adapters** (see PEFT chapter) to **limit** weight drift.
- **Lower LR**, **few epochs**, **early stopping** on validation suites.
- **Replay** examples from diverse tasks (not only chat).

\[
\mathcal{L} = \mathcal{L}_{\text{SFT}} + \lambda \, \mathcal{L}_{\text{LM-aux}}
\]

where \(\mathcal{L}_{\text{LM-aux}}\) is standard CLM on **mixed** corpus.

!!! math-intuition "In Plain English"
    Fine-tuning is **distribution shift**. If you only show **chat**, the model **specializes** and may **erase** breadth. A little **LM replay** reminds it it is still a **general** model.

---

## Python: Masked SFT Loss (Toy)

```python
"""
Toy masked cross-entropy for SFT: only assistant tokens contribute.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T) with -100 ignored
    mask: (B, T) float {0,1} — 1 where loss applies
    """
    b, t, v = logits.shape
    ce = F.cross_entropy(
        logits.reshape(b * t, v),
        labels.reshape(b * t),
        reduction="none",
        ignore_index=-100,
    )
    m = mask.reshape(b * t)
    denom = m.sum().clamp(min=1.0)
    return (ce * m).sum() / denom


if __name__ == "__main__":
    torch.manual_seed(0)
    b, t, v = 2, 8, 50
    logits = torch.randn(b, t, v)
    labels = torch.randint(0, v, (b, t))
    mask = torch.zeros(b, t)
    mask[:, 4:] = 1.0  # assistant starts at 4
    labels[:, :4] = -100
    print(masked_ce_loss(logits, labels, mask))
```

---

## Python: Build Alpaca Template (String Level)

```python
"""
Minimal Alpaca-style prompt wrapping — no tokenizer calls.
"""
from __future__ import annotations


def format_alpaca(instruction: str, input_text: str, response: str) -> str:
    if input_text.strip():
        prompt = f"Below is an instruction...\\n### Instruction:\\n{instruction}\\n### Input:\\n{input_text}\\n### Response:\\n"
    else:
        prompt = f"Below is an instruction...\\n### Instruction:\\n{instruction}\\n### Response:\\n"
    return prompt + response


if __name__ == "__main__":
    s = format_alpaca("Say hi.", "", "Hello!")
    print(s)
```

---

## Hyperparameters and Stopping

- **Learning rate**: often \(10^{-5}\) to \(10^{-4}\) range for full fine-tunes; smaller for larger models.
- **Epochs**: **overfitting** to small chat sets hurts generalization—watch **validation** task suites, not only train loss.
- **Sequence length**: truncate or pack; **left-truncate** prompts if needed to preserve **recent** instructions.

### Worked Example: Token-Level Mask Bitmap

Suppose tokenized sequence length \(T=6\): `[BOS] [SYS] [USR] tok tok [ASST]`. Train loss on last two tokens only (assistant start). Indicator \(m_t = \mathbb{1}\{t \in \mathcal{C}\}\):

\[
\mathcal{L} = \frac{\sum_{t=1}^{T} m_t \cdot \bigl(-\log p_\theta(x_t \mid x_{<t})\bigr)}{\sum_{t=1}^{T} m_t}
\]

!!! math-intuition "In Plain English"
    This is the **average** negative log-likelihood over **only** the positions where \(m_t=1\). Dividing by \(\sum m_t\) makes the loss comparable across examples with different assistant lengths—otherwise longer answers would dominate the sum.

If \(m_t=0\) for \(t \le 4\), gradients **only** flow through assistant head computations that condition on the **full** prefix—earlier layers still receive **credit** for representations used to predict assistant tokens.

### Tool-Use and Function-Calling SFT

Modern assistants **emit** structured **tool calls** (JSON) inside assistant channels. SFT data includes **gold** tool traces; masking may **include** assistant segments that contain **valid** tool syntax while **excluding** raw tool-return payloads from loss if policy dictates (recipes vary). Interview line: “**Tool SFT** is still next-token prediction—just over a **grammar** of allowed tokens.”

---

## Evaluation Beyond Perplexity

Instruction-tuned models are judged on **behavior**, not only **loss**:

- **Task accuracy** on held-out instructions (MMLU-style multiple choice, GSM8K math).
- **Format adherence** — valid JSON, correct tool schema, citation markers if required.
- **Safety** — refusal rates on disallowed prompts (with **low** false refusals on benign).
- **Robustness** — paraphrased instructions, multilingual probes, **typos**.
- **Length control** — does the model follow “answer in one sentence” constraints.

!!! math-intuition "In Plain English"
    Train loss can **decrease** while **task** metrics stall—especially if the model **memorizes** templates. Always pair **offline** metrics with **human** or **LLM-judge** evals for open-ended quality.

---

## Common Pitfalls in SFT Pipelines

1. **Template mismatch** between train and serve → sudden quality collapse at deploy.
2. **Overlong** contexts silently **truncated** — user sees incomplete instructions; model sees wrong conditioning.
3. **Duplicate** conversations in data → **overfitting** to narrow phrasings.
4. **Imbalanced** domains (too much coding, too little reasoning) → skewed **capability profile**.
5. **No EOS handling** — assistant forgets to emit stop token; decoding heuristics mask training issues.

---

## Relation to Preference Optimization

SFT provides a **warm start** policy \(\pi_{\text{SFT}}\) close to human demonstrations. **RLHF/DPO** then optimizes **preferences** not fully captured by imitation. In ablations, **skipping** SFT and going straight to preference learning from base models is often **unstable**—SFT **reduces** exploration space to **reasonable** completions.

---

## Worked Example: Multi-Turn Masking by Index

Consider a toy chat after tokenization (numbers are **positions**):

| Index \(t\) | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|-------------|---|---|---|---|---|---|---|
| Segment | BOS | SYS | USR | USR | ASST | ASST | ASST |
| Loss mask | 0 | 0 | 0 | 0 | 1 | 1 | 1 |

The model always **computes** logits at every position, but **loss** accumulates only where mask \(=1\). Gradients **vanish** on masked positions in the **output** layer, but **earlier layers** still train because assistant predictions **condition** on hidden states built using **all** prior tokens—including user tokens.

!!! math-intuition "In Plain English"
    Masking is applied **after** logits; back-prop still flows through **shared** representations. You are **not** freezing encoder-like layers—there is no separate encoder—**all** transformer blocks co-adapt.

---

## Objective Comparison: CLM Pre-training vs SFT

**Pre-training CLM** maximizes \(\sum_t \log p(x_t \mid x_{<t})\) on **all** tokens—model must predict **everything** in the document.

**SFT** maximizes the **same** autoregressive likelihood but **restricts** the sum to assistant positions:

\[
\mathcal{L}_{\text{SFT}} = \sum_{t \in \mathcal{C}} \log p_\theta(x_t \mid x_{<t})
\]

!!! math-intuition "In Plain English"
    This is the **positive** log-likelihood form (equivalent to minimizing negative log-likelihood). You sum log-probabilities only over assistant tokens—each term rewards assigning high probability to the **demonstrated** next token given all prior context.

Equivalently, multiply each term by **indicator** \(m_t \in \{0,1\}\).

### KL-to-Base (Optional Regularizer)

Some recipes add a **KL penalty** to the base model \(\pi_{\text{base}}\) to limit drift:

\[
\mathcal{L} = \mathcal{L}_{\text{SFT}} - \beta \, \mathrm{KL}\bigl(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{base}}(\cdot \mid x)\bigr)
\]

!!! math-intuition "In Plain English"
    The KL term punishes drifting too far from the **frozen** base policy on the **same** prompts—similar spirit to KL in RLHF but applied during SFT. Larger \(\beta\) keeps outputs closer to the base model’s “shape,” which can **preserve** capabilities at the cost of slower instruction adaptation.

This resembles **trust-region** ideas used later in RLHF—optional at SFT stage.

??? deep-dive "Deep Dive: When KL-regularized SFT helps"
    Teams use this when a narrow instruction set risks **forgetting** open-domain fluency. You pay extra compute (second forward through base model or cached logits) and must tune \(\beta\) so you do not **underfit** instructions.

---

## Data Hygiene Checklist

- **Deduplicate** near-identical instructions (MinHash on normalized text).
- **Strip PII** consistently; avoid **leaking** eval benchmark strings verbatim.
- **Balance** languages if multilingual product requirements exist.
- **Version** templates in git; **pin** tokenizer merges with model card.

### Human vs. Synthetic Mix Ratios

Practitioners often blend **human-written** gold demonstrations (high **alignment** signal) with **model-generated** data scaled by **filters**. There is no universal ratio—**safety-critical** applications bias human; **prototyping** may accept more synthetic with **strong** automated verification. Document **provenance** per shard.

---

## Python: Assistant Span Mask from Chat String (Sketch)

```python
"""
Sketch: find assistant substring start in token space using a HF tokenizer.
Requires that tokenizer.decode(encode(x)) ~= x for substring search (verify).
"""
from __future__ import annotations

from transformers import AutoTokenizer


def assistant_mask_from_text(
    full_text: str, assistant_prefix: str, tokenizer: AutoTokenizer
) -> list[int]:
    if assistant_prefix not in full_text:
        raise ValueError("assistant_prefix not found")
    prefix = full_text.split(assistant_prefix, 1)[0] + assistant_prefix
    ids_full = tokenizer.encode(full_text, add_special_tokens=False)
    ids_pref = tokenizer.encode(prefix, add_special_tokens=False)
    start = len(ids_pref)  # first assistant content token index
    mask = [0] * len(ids_full)
    for i in range(start, len(ids_full)):
        mask[i] = 1
    return mask


if __name__ == "__main__":
    tok = AutoTokenizer.from_pretrained("gpt2")
    user = "### User:\\nHi\\n### Assistant:\\n"
    reply = "Hello there!"
    full = user + reply
    m = assistant_mask_from_text(full, "### Assistant:\\n", tok)
    print("mask:", m)
```

---

## Python: SFT with Hugging Face TRL (`SFTTrainer`)

Full imports, chat-template-friendly pattern: dataset supplies a **text** column that already includes the supervised completion; TRL masks prompt portions when configured via `SFTConfig`. Adjust `dataset_text_field` and formatting to your schema.

```python
"""
Instruction SFT with TRL SFTTrainer — template-compatible supervised fine-tuning.
Requires: pip install torch transformers datasets trl accelerate

Run on GPU for real jobs; CPU will work only for tiny smoke tests.
"""
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def build_text_dataset(tok: AutoTokenizer) -> Dataset:
    """Single-turn examples as one string: prompt + completion (model-specific template)."""
    rows = [
        {
            "text": (
                "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
                "The capital of France is Paris."
            )
        },
        {
            "text": (
                "<|user|>\nName a prime number.\n<|assistant|>\n"
                "The number 7 is prime."
            )
        },
    ]
    return Dataset.from_list(rows)


def main() -> None:
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    ds = build_text_dataset(tok)

    args = SFTConfig(
        output_dir="./sft_trl_out",
        max_seq_length=512,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-5,
        logging_steps=1,
        report_to=[],
        dataset_text_field="text",
    )

    # TRL ≥0.9 may use `processing_class=tok`; older versions use `tokenizer=tok`.
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds, tokenizer=tok)
    trainer.train()
    print("Training complete; checkpoints in ./sft_trl_out")


if __name__ == "__main__":
    main()
```

??? deep-dive "Deep Dive: `packing` and `max_seq_length` in TRL"
    Packing concatenates examples to fill context windows; it needs correct attention masks so tokens do not attend across unrelated conversations. Raise `max_seq_length` carefully—memory grows linearly with sequence length and quadratically with attention in standard implementations.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Why do we mask loss on assistant tokens only, and how does teacher forcing interact with that mask?
        *Answer:* We only want to maximize likelihood of **what the assistant should say**, conditioned on the user/system prefix; predicting the user’s tokens wastes capacity and encourages trivial **copy** of the prompt. **Teacher forcing** still feeds the **ground-truth** prefix tokens when computing hidden states for assistant positions; gradients flow into layers below masked positions because assistant logits **condition on** the full prefix, but the loss is **zeroed** on prompt tokens so there is no direct next-token target there.
    2. Compare FLAN-style task mixtures with modern multi-turn chat SFT—what changes in evaluation?
        *Answer:* FLAN-style training optimizes **single-turn** task phrasing (“translate: …”) and is often scored on **held-out task clusters** and NLP benchmarks. Modern **chat SFT** optimizes multi-turn **roles**, tool boundaries, and stop tokens, so evaluation shifts toward **dialogue** metrics (MT-Bench, Arena-style pairwise), **format** adherence, and **multi-turn** robustness—not only single-shot accuracy. You report both **skill** benchmarks and **interaction** quality because distribution shift from FLAN to product chat is large.
    3. What is the alignment tax, and how would you measure whether SFT is worth the regression on raw LM benchmarks?
        *Answer:* The **alignment tax** is the drop in **raw LM** scores (perplexity on web corpora, MMLU, coding benchmarks) after fine-tuning narrows the distribution toward assistant behavior. You measure it with **pre/post** suites on **diverse** tasks—not just chat—and weigh against **alignment** wins (instruction following, safety, user satisfaction); if the tax is large, you add **replay** of pretraining-style text, KL to base, or smaller LR/epochs.
    4. Walk through a multi-turn example and identify exactly which token positions receive gradients.
        *Answer:* Tokenize the full conversation with special role tokens; mark **assistant** spans with label IDs and set **ignore_index** (or mask 0) on system, user, and prior-turn tokens. Only positions where the model is trained to predict **the next assistant token**—typically every token inside each assistant message, sometimes excluding leading role delimiters depending on recipe—accumulate cross-entropy; earlier layers still receive gradients **through** assistant predictions that attend to user tokens, but user token positions have **no** CE loss.
    5. How does LIMA challenge the assumption that more SFT data is always better?
        *Answer:* LIMA showed that a **small**, **high-quality**, curated instruction set can match or beat much larger noisy mixtures for instruction following—emphasizing **diversity and quality** over raw count. The takeaway is not “always use 1k examples,” but that **marginal examples must earn their place**; beyond a point, duplicate or low-signal rows hurt more than they help, and careful curation beats scaling low-quality SFT soup.
    6. Describe three mitigations for catastrophic forgetting during SFT.
        *Answer:* (1) **Mix** a small fraction of **continued pretraining** or domain-general text with standard CLM loss so the model retains broad fluency. (2) Use **PEFT** (LoRA) or lower LR / fewer epochs to limit **global** weight drift. (3) **Replay** diverse tasks from the base model’s strengths (code, math, multilingual) in the SFT blend and monitor **regression** suites—not only chat loss.
    7. Why must chat templates match between training and inference?
        *Answer:* The model learns **conditional** distributions over tokens that follow specific **delimiters** and role tags; at inference, a different template changes token boundaries and conditioning, so the model may see **out-of-distribution** prefixes (wrong stops, missing `<|assistant|>`), harming quality and safety. `apply_chat_template` parity ensures **train/serve** alignment—the same string the user would send is what the model was trained to continue after the same markers.
    8. How would you structure SFT data for tool-calling or JSON-only assistant outputs?
        *Answer:* Represent **tool calls** as assistant-channel text (or dedicated tool role) with a **fixed grammar** (JSON schema, XML tags) and train with loss on those tokens so the model learns valid syntax; include **gold** tool arguments and optionally **tool return** messages in context with masking rules per your policy (sometimes tool returns are **context-only** with zero loss). For **JSON-only** outputs, constrain decoding at inference and train on **valid** examples only—invalid JSON in SFT teaches the wrong distribution.
    9. What is the difference between rejection sampling for SFT augmentation and RLHF?
        *Answer:* **Rejection sampling** for SFT **selects** a single high-scoring completion (from \(N\) samples or a filter) and **supervises** on it—still **behavioral cloning** with a better target distribution. **RLHF** optimizes a **policy** against a **learned reward** with exploration (PPO) or preference gradients (DPO), updating **probabilities** over many rollouts—not just picking one static label. RS+SFT is cheaper and more stable; RLHF addresses preferences not captured by a single best demo.
    10. When would you add KL regularization toward the base model during SFT?
        *Answer:* Add KL when the instruction set is **narrow** and risks **washing out** base capabilities—KL penalizes large shifts from \(\pi_{\text{base}}\) on the same contexts so outputs stay **fluent** and **fact-shaped** like the pretrained model. You pay extra compute (second forward or cached logits) and must tune \(\beta\) so instructions still fit; it is most common when catastrophic forgetting appears in offline evals before any RL stage.

!!! interview "Follow-up Probes"
    - “If validation loss improves but MT-Bench scores drop, what do you check first?”
    - “How do you implement masking for ChatML: string-level or token-level?”
    - “What’s your stance on training on refusals—how do you balance safety and false refusals?”
    - “How does packing interact with BOS/EOS tokens in your tokenizer?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Masked causal LM on assistant completions; prompts condition but are not predicted.”
    - “Train/serve parity—`apply_chat_template` must match production.”
    - “SFT is the imitation-learning anchor before preference optimization.”
    - “Quality and diversity often beat raw scale; cite LIMA for curated-data wins.”

---

## References

- Wei et al., *Finetuned Language Models Are Zero-Shot Learners* (FLAN) — [arXiv:2109.01652](https://arxiv.org/abs/2109.01652)
- Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- Wang et al., *Self-Instruct: Aligning Language Models with Self-Generated Instructions* — [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
- Taori et al., *Stanford Alpaca* — instruction-following demo dataset / pipeline (blog / repo)
- Chung et al., *Scaling Instruction-Finetuned Language Models* (FLAN-T5) — [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)
- Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* — EWC (context)
