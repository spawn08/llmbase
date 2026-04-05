# FLAN: Scaling Instruction-Finetuned Language Models

**Authors:** Hyung Won Chung, Le Hou, Shayne Longpre, and 10 more  
**Year:** 2022 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2210.11416](https://arxiv.org/abs/2210.11416)

---

## TL;DR

FLAN **instruction fine-tunes** a pretrained LM on a large mixture of **1,800+ tasks phrased as natural language instructions**, improving zero-shot generalization to unseen tasks. Scaling three dimensions — **number of tasks**, **model size**, and **chain-of-thought data** — together yields FLAN-T5 and FLAN-PaLM variants that outperform base models on held-out task clusters.

---

## Why This Paper Matters

FLAN is the bridge between raw pre-trained models and the instruction-following chat models we use today:

1. **Instruction tuning as data engineering:** The key insight is that formatting tasks as instructions teaches the model a general "instruction-following" capability
2. **Precursor to chat models:** FLAN-style SFT is Stage 1 of the RLHF pipeline
3. **Task diversity scales:** More diverse instruction formats → better generalization
4. **Chain-of-thought integration:** Including CoT examples in the instruction mix teaches reasoning

---

## Key Concepts Explained Simply

### What is Instruction Tuning?

Regular fine-tuning trains on labeled examples directly: input → output. Instruction tuning wraps them in **natural language instructions**:

- Regular: "The movie was great" → "positive"
- Instruction-tuned: "Classify the sentiment of the following review as positive or negative.\n\nReview: The movie was great\n\nSentiment:" → "positive"

The model learns to **parse the instruction** and generalize to new instructions it hasn't seen.

### Why More Tasks Help

With 10 tasks, the model memorizes each instruction format. With 1,800 tasks, the model learns the **meta-skill** of following instructions generally. This transfers to novel tasks described at inference time.

### Chain-of-Thought in FLAN

Some tasks include step-by-step reasoning in the target:
- "What is 47 + 83? Let's think step by step. 47 + 83: first, 7 + 3 = 10, carry 1. Then 4 + 8 + 1 = 13. So the answer is 130."

Including these CoT examples in the instruction mix teaches the model **when and how** to reason step-by-step.

---

## The Math — Explained Step by Step

### Instruction Tuning Loss

The loss is the same as standard supervised fine-tuning — just on instruction-formatted data:

\[
\mathcal{L}_{\text{SFT}} = -\sum_{t=1}^{|y|} \log P_\theta(y_t \mid \text{instruction}, \text{input}, y_{<t})
\]

**Key difference from pre-training:** The data distribution shifts from "web text" to "instruction-response pairs curated from diverse tasks."

### Task Scaling

The paper shows that performance scales as a function of task count:

\[
\text{Quality}(\text{held-out tasks}) \propto \log(\text{number of fine-tuning tasks})
\]

More tasks → better generalization, with logarithmic diminishing returns.

### Multitask Mixing

Tasks are mixed with different sampling strategies:
- **Proportional sampling:** Sample proportional to dataset size
- **Examples-proportional:** Cap maximum examples per task to prevent large datasets from dominating
- **Temperature-based:** Use temperature \(\tau\) to flatten the distribution: \(p_i \propto n_i^{1/\tau}\) where \(n_i\) is dataset size

---

## Python Implementation

```python
import numpy as np
from collections import defaultdict


def format_instruction(task_type, input_text, template_idx=0):
    """
    Format input as an instruction. Multiple templates per task
    for diversity.
    """
    templates = {
        "sentiment": [
            f"Classify the sentiment of the following text as positive or negative.\n\nText: {input_text}\n\nSentiment:",
            f"Is the following review positive or negative?\n\n{input_text}\n\nAnswer:",
            f"What is the sentiment expressed in this text?\n\n\"{input_text}\"\n\nThe sentiment is",
        ],
        "summarize": [
            f"Summarize the following text in one sentence.\n\n{input_text}\n\nSummary:",
            f"Write a brief summary of this passage.\n\n{input_text}\n\nBrief summary:",
            f"TL;DR of the following:\n\n{input_text}\n\nTL;DR:",
        ],
        "translate": [
            f"Translate the following English text to French.\n\n{input_text}\n\nFrench:",
            f"What is the French translation of: {input_text}",
            f"English: {input_text}\nFrench:",
        ],
        "qa": [
            f"Answer the following question.\n\nQuestion: {input_text}\n\nAnswer:",
            f"Q: {input_text}\nA:",
            f"Please answer this question: {input_text}",
        ],
    }
    task_templates = templates.get(task_type, templates["qa"])
    return task_templates[template_idx % len(task_templates)]


def format_cot_instruction(question, steps, answer):
    """Format an instruction with chain-of-thought reasoning."""
    return (
        f"Answer the following question. Show your reasoning step by step.\n\n"
        f"Question: {question}\n\n"
        f"Let's think step by step.\n"
        + "\n".join(f"Step {i+1}: {s}" for i, s in enumerate(steps))
        + f"\n\nTherefore, the answer is {answer}."
    )


def temperature_sampling(dataset_sizes, temperature=2.0):
    """
    Temperature-based task mixing.
    Higher temperature → more uniform sampling across tasks.
    """
    sizes = np.array(dataset_sizes, dtype=float)
    probs = sizes ** (1.0 / temperature)
    probs = probs / probs.sum()
    return probs


def examples_proportional_mixing(dataset_sizes, max_examples=50000):
    """Cap examples per task, then sample proportionally."""
    capped = np.minimum(dataset_sizes, max_examples)
    return capped / capped.sum()


class InstructionTuningDataset:
    """Simulated instruction tuning dataset with diverse tasks."""

    def __init__(self):
        self.tasks = {
            "sentiment_sst2": 67000,
            "sentiment_imdb": 25000,
            "nli_mnli": 393000,
            "nli_snli": 550000,
            "qa_squad": 87000,
            "qa_triviaqa": 95000,
            "summarize_cnn": 287000,
            "translate_wmt_en_de": 4500000,
            "translate_wmt_en_fr": 36000000,
            "cot_gsm8k": 7500,
            "cot_aqua": 97000,
            "commonsense_csqa": 9700,
        }

    def get_mixing_strategy(self, strategy="temperature", **kwargs):
        sizes = np.array(list(self.tasks.values()), dtype=float)
        names = list(self.tasks.keys())

        if strategy == "proportional":
            probs = sizes / sizes.sum()
        elif strategy == "temperature":
            probs = temperature_sampling(sizes, kwargs.get("temperature", 2.0))
        elif strategy == "capped":
            probs = examples_proportional_mixing(
                sizes, kwargs.get("max_examples", 50000)
            )
        else:
            probs = np.ones(len(sizes)) / len(sizes)

        return list(zip(names, probs))

    def sample_batch(self, batch_size, strategy="temperature"):
        mix = self.get_mixing_strategy(strategy)
        names, probs = zip(*mix)
        probs = np.array(probs)
        sampled = np.random.choice(len(names), size=batch_size, p=probs)
        return [names[i] for i in sampled]


def evaluate_generalization(n_train_tasks_list):
    """Simulate how held-out task performance scales with training tasks."""
    base_accuracy = 0.30
    results = []
    for n in n_train_tasks_list:
        acc = base_accuracy + 0.12 * np.log(n + 1)
        acc = min(acc, 0.90)
        results.append((n, acc))
    return results


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Instruction formatting
    print("--- Instruction Templates (same task, different formats) ---")
    text = "This restaurant has the best pasta I've ever eaten!"
    for i in range(3):
        prompt = format_instruction("sentiment", text, template_idx=i)
        print(f"\nTemplate {i+1}:")
        print(prompt)

    # CoT example
    print("\n--- Chain-of-Thought Instruction ---")
    cot = format_cot_instruction(
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        ["Distance = speed × time", "Distance = 60 × 2.5", "Distance = 150"],
        "150 miles"
    )
    print(cot)

    # Task mixing strategies
    print("\n--- Task Mixing Strategies ---")
    dataset = InstructionTuningDataset()

    for strategy in ["proportional", "temperature", "capped", "uniform"]:
        mix = dataset.get_mixing_strategy(strategy, temperature=2.0, max_examples=50000)
        print(f"\n{strategy.upper()}:")
        for name, prob in sorted(mix, key=lambda x: -x[1])[:5]:
            print(f"  {name:<30} {prob:.3%}")

    # Generalization scaling
    print("\n--- Task Count → Held-out Accuracy ---")
    results = evaluate_generalization([1, 5, 10, 50, 100, 500, 1000, 1800])
    for n, acc in results:
        bar = "█" * int(acc * 40)
        print(f"  {n:>5} tasks: {acc:.1%} {bar}")
```

---

## Interview Importance

FLAN is important for understanding the **SFT stage** of the alignment pipeline and how **data engineering** drives model capabilities.

### Difficulty Level: ⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: How does instruction tuning differ from pre-training in data and objective?

**Answer:** The objective is the same (next-token prediction), but the data distribution is fundamentally different:
- **Pre-training:** Raw web text, books, code — the model learns language patterns
- **Instruction tuning:** Curated instruction-response pairs from 1,800+ tasks — the model learns to parse instructions and produce appropriate responses

The shift in data distribution teaches the model a new "behavior": instead of completing arbitrary text, it learns to follow instructions.

### Q2: Why include CoT demonstrations in FLAN variants?

**Answer:** Including chain-of-thought examples teaches the model **when and how to reason step-by-step**:
1. The model learns that some tasks benefit from showing intermediate reasoning
2. It learns the **format** of step-by-step reasoning (numbered steps, "therefore")
3. On held-out reasoning tasks, the model is more likely to generate useful reasoning traces
4. Without CoT in the training mix, instruction-tuned models may default to giving direct answers even when reasoning would help

### Q3: What limits zero-shot transfer to truly novel tasks?

**Answer:**
1. **Task structure:** If the novel task requires fundamentally different reasoning (e.g., spatial reasoning when training only had text tasks), transfer fails
2. **Instruction ambiguity:** Novel tasks may require instructions the model can't parse
3. **Domain knowledge:** Tasks requiring specific knowledge not in pre-training data
4. **Output format:** If the expected output format (e.g., structured JSON, specific notation) wasn't in the training mix
5. **Distribution gap:** The further the novel task is from any training task, the worse transfer gets

### Q4: How does FLAN relate to RLHF and chat models?

**Answer:** FLAN is effectively **Stage 1 (SFT)** of the modern alignment pipeline:
1. **FLAN/SFT:** Teaches the model to follow instructions using demonstrations
2. **RLHF (Stage 2+3):** Further refines the model using human preferences

Chat models like ChatGPT start with FLAN-style instruction tuning, then add RLHF for quality refinement. FLAN showed that SFT alone gets you surprisingly far — a well-instruction-tuned model is already useful without RLHF.

---

## Connections to Other Papers

- **T5** → FLAN uses T5's text-to-text framework with task prefixes
- **InstructGPT** → FLAN is the SFT component; InstructGPT adds RLHF on top
- **Chain-of-Thought** → CoT data included in FLAN training mix
- **GPT-3** → FLAN improves on GPT-3's zero-shot ability through instruction tuning
- **PaLM** → FLAN-PaLM applies instruction tuning to PaLM

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Fine-tune on 1,800+ tasks phrased as instructions |
| Key insight | Task diversity → general instruction-following ability |
| Loss function | Same as LM loss, but on instruction-response pairs |
| CoT integration | Include step-by-step reasoning in training data |
| Scaling axes | Tasks × model size × CoT data |
| Legacy | Stage 1 of the chat model pipeline (SFT) |
