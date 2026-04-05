# T5: Encoder-Decoder Architecture

## Why This Matters for LLMs

Modern large language models are often discussed as a single stack: a decoder-only Transformer that predicts the next token. T5 (*Text-to-Text Transfer Transformer*, Raffel et al., 2019) is historically important because it demonstrated a different contract with the same underlying building blocks: **every supervised NLP task can be expressed as taking a string and producing another string**. That unification matters for interviews because it connects pre-training objectives, multi-task learning, instruction tuning, and production system design. When you explain how Flan-T5 or instruction-tuned seq2seq models behave, you are often explaining variations of the same text-to-text interface that T5 popularized at scale.

The encoder-decoder split is not a small detail. An encoder with bidirectional self-attention can represent the full input in one forward pass, while a decoder with causal self-attention and cross-attention to the encoder can generate outputs that depend on the entire source. That pattern aligns naturally with translation, summarization, data-to-text, and many structured prediction problems where the model must **condition** on long, fixed context before emitting a shorter or differently formatted target. Decoder-only models can approximate this with long prompts and careful formatting, but the architectural separation encodes an inductive bias that still shows up in benchmarks and in how teams shard work across encoder and decoder stacks in research prototypes.

Finally, T5 is one of the most cited empirical studies of Transformer design choices. The paper’s ablations give you interview-ready anchors: span corruption versus token masking, mean span length, corruption rate, encoder-decoder versus decoder-only at matched compute, and the interaction between model size and dataset size. Combined with later work such as Flan-T5 (instruction tuning on many tasks), you can tell a coherent story from pre-training through supervised adaptation, which is exactly what senior ML and systems interviewers often want when they ask how you would train and deploy a general-purpose language system.

---

## Text-to-Text Framing with Concrete Examples

T5 casts each task as **input text** maps to **target text**. The task is signaled by a natural-language prefix (sometimes called a **task prefix** or **control code**) prepended to the input.

**Example A — Machine translation**

- **Input:** `translate English to French: The cat`
- **Target:** `Le chat`

The prefix `translate English to French:` tells the model which mapping to apply. The model is trained to emit the French string as the decoded sequence.

**Example B — Abstractive summarization**

- **Input:** `summarize: On Tuesday regional officials opened cooling centers across the metro area as a heat advisory remained in effect through Thursday evening. Residents were urged to check on neighbors and to limit outdoor activity during peak afternoon hours.`
- **Target:** `Officials opened cooling centers during a multi-day heat advisory and urged residents to limit outdoor activity.`

Here the source is long-form news-like text after the `summarize:` cue. The target is a shorter paraphrase. The same tokenizer, vocabulary, and decoding procedure are used as for translation: there is no separate classification head.

**Example C — Sentiment as generation**

- **Input:** `sentiment: The movie was great`
- **Target:** `positive`

Binary or multi-class sentiment becomes string prediction. At training time you provide the label string; at inference time you decode a short label string (often with constrained decoding or a small allowed set in production systems).

These three examples share a common interface: **concatenate a human-readable task description and the instance, then train the model to generate the answer text token by token**.

---

## Mathematical View: Sequence-to-Sequence Objective

Let the tokenizer map input and output strings to token sequences. Let \(\mathbf{x} = (x_1, \ldots, x_{T_x})\) denote source tokens (including the task prefix) and \(\mathbf{y} = (y_1, \ldots, y_{T_y})\) denote target tokens. T5 is trained to minimize the conditional negative log-likelihood:

\[
\mathcal{L}_{\text{seq2seq}} = - \sum_{j=1}^{T_y} \log p_\theta\bigl(y_j \mid y_{<j}, \mathbf{x}\bigr)
\]

!!! math-intuition "In Plain English"
    The formula says: look at every position \(j\) in the target sequence. The model should assign high probability to the true next token \(y_j\) given all earlier target tokens \(y_{<j}\) and the full source sequence \(\mathbf{x}\). Summing \(-\log p\) over positions gives a standard cross-entropy loss for autoregressive decoding. The encoder supplies a representation of \(\mathbf{x}\); the decoder is trained to be a conditional language model over \(\mathbf{y}\).

The encoder-decoder Transformer parameterizes \(p_\theta(y_j \mid y_{<j}, \mathbf{x})\) using:

- Encoder layers with **bidirectional** self-attention over \(\mathbf{x}\).
- Decoder layers with **causal** self-attention over \(y_{<j}\) and **cross-attention** into encoder hidden states.

!!! example "Worked Example"
    Suppose a toy vocabulary maps tokens to integers and the target is the two-token sequence \((y_1, y_2)\). At position \(j=1\), the decoder predicts a distribution over the vocabulary for \(y_1\). If the correct \(y_1\) is token ID 42 and the model assigns probability \(0.6\), the contribution to the loss is \(-\log(0.6) \approx 0.51\). At \(j=2\), the decoder conditions on the embedding of \(y_1\) and predicts \(y_2\). If the correct token has probability \(0.25\), the contribution is \(-\log(0.25) \approx 1.39\). The total loss for this toy example is about \(0.51 + 1.39 = 1.90\) nats (or nats divided by \(\ln 2\) if you report bits). Real models use tens of thousands of tokens and millions of training examples; the mechanics are the same: sum cross-entropy across target positions.

---

## Architecture: Encoder, Decoder, and Cross-Attention

**Data flow (conceptual)**

```
Source tokens → Encoder (bidirectional self-attention, depth N)
                    ↓
              Encoder memory (per-position vectors)
                    ↓ cross-attention (decoder queries, encoder keys/values)
Target tokens → Decoder (causal self-attention, depth N) → logits → softmax over vocabulary
```

The decoder cannot attend to future target positions because of the causal mask. It can attend to all encoder positions because the source is fully observed when modeling each \(y_j\).

**Cross-attention (decoder queries, encoder keys and values)**

Let \(H_{\text{enc}} \in \mathbb{R}^{T_x \times d}\) be encoder hidden states and let \(h_{t}^{\text{dec}} \in \mathbb{R}^{d}\) be a decoder position hidden state before the cross-attention sublayer. Learned projections produce query \(\mathbf{q}_t = W_Q h_{t}^{\text{dec}}\), keys \(K = W_K H_{\text{enc}}\), and values \(V = W_V H_{\text{enc}}\). One standard scoring form is:

\[
\mathbf{a}_t = \text{softmax}\left(\frac{\mathbf{q}_t K^\top}{\sqrt{d_k}}\right), \quad \mathbf{o}_t = \mathbf{a}_t V
\]

!!! math-intuition "In Plain English"
    The decoder position \(t\) asks a **question** vector \(\mathbf{q}_t\). Every encoder position supplies a **key** describing what it contains. Large dot products mean "this encoder position is relevant for generating the next target token." Softmax turns scores into weights that sum to one. The output \(\mathbf{o}_t\) is a weighted blend of encoder **value** vectors: a soft pointer into the source representation.

**Relative position bias**

T5 does not use sinusoidal absolute position embeddings in the same way as the original Vaswani et al. formulation. Instead, it often relies on **learned relative position biases** (bucketed by distance) inside attention. The exact parameterization is an implementation detail, but the interview point is: **positions enter as relative pairwise relationships**, which can help length generalization compared to a purely absolute scheme in some regimes.

??? deep-dive "Deep Dive: Why Cross-Attention Reuses Encoder Memory Efficiently"
    In translation, the source sentence is fixed while the target grows one token at a time. The encoder runs once per source string and produces a tensor of shape roughly \((\text{batch}, T_x, d_{\text{model}})\). At each decoder step, cross-attention reads from that fixed tensor. In serving systems, this enables caching **encoder activations** across decoder steps so you do not recompute the encoder for each new target token. The decoder still updates its own KV cache for causal self-attention, but the encoder side is shared and stable. This is why encoder-decoder stacks can be attractive for **fixed-input, variable-output** workloads when batching and caching are aligned with the workload pattern.

---

## Span Corruption Pre-Training Walkthrough

T5’s unsupervised objective is **span corruption**. Unlike BERT’s token-level mask, T5 deletes contiguous spans and replaces them with **unique sentinel tokens**, then trains the model to emit the missing spans in the target.

**Step 1 — Start with a natural sentence**

`The cat sat on the mat and watched the birds fly.`

**Step 2 — Sample spans to corrupt**

Suppose the corruption procedure selects two spans:

- Span 1: `cat sat`
- Span 2: `mat and watched`

**Step 3 — Replace spans with sentinels**

Each replaced span gets a distinct sentinel such as `<extra_id_0>` and `<extra_id_1>` (exact strings depend on the vocabulary configuration). The corrupted **input** might become:

`The <extra_id_0> on the <extra_id_1> the birds fly.`

**Step 4 — Form the target sequence**

The **target** lists the sentinels and the recovered tokens in order:

`<extra_id_0> cat sat <extra_id_1> mat and watched`

The model learns to copy sentinel order and fill span contents. Intuitively, the decoder’s job is **denoising** with a compressed input: it must infer what was removed.

!!! math-intuition "In Plain English"
    Span corruption defines a distribution over noisy inputs \(\tilde{\mathbf{x}}\) given clean text, and trains the network to maximize \(\log p_\theta(\mathbf{r} \mid \tilde{\mathbf{x}})\) where \(\mathbf{r}\) is the sequence of removed pieces in a canonical order. Compared to reconstructing every token, the target sequence is shorter when spans are long, which can reduce the number of decoder steps during pre-training for a fixed amount of masking noise.

!!! example "Worked Example"
    **Original token sequence (toy tokenization):**  
    `[The, cat, sat, on, the, mat, and, watched, the, birds, fly, .]`  

    **Mask spans (lengths 2 and 3 for illustration):**  
    - Span A: `[cat, sat]`  
    - Span B: `[mat, and, watched]`  

    **Corrupted source string (conceptual):**  
    `[The, <extra_id_0>, on, the, <extra_id_1>, the, birds, fly, .]`  

    **Target string for the decoder:**  
    `[<extra_id_0>, cat, sat, <extra_id_1>, mat, and, watched]`  

    Count decoder steps: the target has 7 tokens versus 12 tokens for a full copy of the original sentence. For the same amount of masking, span targets are often shorter than per-token reconstruction, which is one reason T5 reports efficiency gains over some MLM setups.

---

## Ablation Findings from the T5 Paper

The T5 authors ran large-scale comparisons. The table below summarizes **directional** findings you can cite in interviews. Exact leaderboard numbers evolve with newer baselines, but the **relative** conclusions remain pedagogically useful.

| Design choice | What was tested | Practical takeaway |
| --- | --- | --- |
| **Architecture** | Encoder-decoder versus decoder-only versus encoder-only | Encoder-decoder often strongest on diverse supervised tasks when compute is matched using their protocol; decoder-only remains competitive and simpler at scale. |
| **Objective** | Span corruption versus language modeling versus BERT-style MLM | Span corruption performed best among the objectives they tried for their text-to-text setup. |
| **Corruption rate** | Fraction of tokens removed | Around 15% of tokens corrupted matched BERT’s rate in spirit and worked well in their sweep. |
| **Mean span length** | Distribution of masked span lengths | Mean span length around 3 was a sweet spot in their experiments. |
| **Data versus size** | Scaling laws with data and parameters | Both matter; increasing pre-training data remained valuable even when increasing model size. |
| **Multi-task fine-tuning** | Joint fine-tuning on many tasks | Helpful but sensitive to mixing rates and task sampling; naive mixing can hurt some tasks. |

??? deep-dive "Deep Dive: How to Interpret 'Matched Compute' in Ablations"
    When papers compare architectures at "matched FLOPs," they typically fix a training budget and adjust width, depth, or batching so that total compute is similar. Interviewers may probe whether encoder-decoder models pay an **implementation tax**: extra parameters for two stacks, more complex serving paths, and additional engineering for cross-attention caching. The fair answer is: **benchmarks measure idealized training objectives**, while product teams weigh simplicity, tooling, and in-context learning. That is why production LLMs are often decoder-only even when encoder-decoder might win on narrow academic metrics.

---

## Encoder-Decoder versus Decoder-Only: Trade-Offs

| Dimension | Encoder-decoder (T5-style) | Decoder-only (GPT-style) |
| --- | --- | --- |
| **Input representation** | Bidirectional over the full prompt | Causal; left context only inside the model unless you recompute or use tricks |
| **Best native fit** | Translation, summarization, structured transformation | Open-ended chat, long-form continuation, tool-augmented loops |
| **Parameter efficiency (conceptual)** | Two stacks can spend parameters on separate roles | Single stack reuses one block design everywhere |
| **Inference cost** | Encoder pass plus incremental decoding; cross-attention per step | Single forward path per token with KV cache |
| **Ecosystem** | Seq2seq tooling in older NLP; still used in research | Dominant stack for frontier LLM training codebases |

**When encoder-decoder is often better**

- The input is long and **static** while the output is short: summarization, parsing to string, many extraction-to-string tasks.
- You want a **clean separation** between "understand the page" and "produce the answer."

**When decoder-only is often preferred**

- You want **one model class** for pre-training, chat, and tool use.
- You rely on **in-context learning** and long prompts rather than supervised fine-tuning for every skill.

!!! math-intuition "In Plain English"
    Inference cost is often summarized as "encoder once plus decoder for each new token." If \(T_x\) is long and \(T_y\) is short, that can be favorable. If both are long and you interleave user and assistant turns in one causal stream, decoder-only stacks map naturally to the conversational pattern.

---

## Flan-T5 and Instruction Tuning

**Flan-T5** (Chung et al., 2022) applies **instruction fine-tuning** to T5 checkpoints: the model sees many tasks phrased as instructions paired with target outputs. The high-level objective remains supervised next-token prediction, but the task distribution becomes:

- Diverse NLP tasks (question answering, reasoning-style prompts, translation, and more).
- Multiple prompt templates per task to reduce overfitting to a single wording.

For interviews, emphasize the pattern: **start from a text-to-text model, then broaden behavior with large-scale multi-task instruction data**, analogous in spirit to instruction tuning on decoder-only models, even though the architecture differs.

??? deep-dive "Deep Dive: What Flan-T5 Does Not Solve"
    Instruction tuning improves generalization to held-out task phrasings, but it does not remove factual errors, safety issues, or evaluation gaming. Teams still need retrieval, monitoring, policy layers, and measurement. Mentioning Flan-T5 as "instruction-tuned T5" is correct; claiming it makes the model truthful or aligned by itself is not.

---

## Full Code: Inference, Fine-Tuning, and Span Corruption Demo

The script below uses Hugging Face `transformers`. It runs **multi-task inference**, a **tiny fine-tuning loop** for demonstration only, and a **printed span corruption** example. Install dependencies in your environment before running: `pip install torch transformers`.

```python
"""
T5 text-to-text: multi-task inference, illustrative fine-tuning, span corruption demo.

Dependencies: torch, transformers
This file is educational: fine-tuning uses a toy dataset and a few epochs only.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import torch
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5Tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def t5_multitask_inference(model_name: str) -> None:
    """
    Run several tasks in one model: translation, summarization, sentiment-style labeling.

    The model predicts target text conditioned on the prefixed input string.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    task_inputs: List[str] = [
        "translate English to French: The cat",
        "summarize: State authorities deployed repair crews along the coastal highway "
        "after overnight flooding closed two lanes. Officials asked travelers to delay "
        "nonessential trips until water levels recede.",
        "sentiment: The movie was great",
    ]

    print("=== Multi-task inference ===")
    for text in task_inputs:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                enc.input_ids,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"INPUT:  {text}")
        print(f"OUTPUT: {decoded}")
        print("-" * 72)


def t5_toy_finetune(model_name: str, epochs: int = 5, lr: float = 3e-4) -> None:
    """
    Illustrative fine-tune on six paired examples. Real training uses datasets and schedules.

    The task prefix here is 'classify:' to force a consistent format at inference time.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.train()

    train_pairs: List[Tuple[str, str]] = [
        ("classify: I love this product!", "positive"),
        ("classify: Terrible experience, never again.", "negative"),
        ("classify: Absolutely wonderful service.", "positive"),
        ("classify: Worst purchase I ever made.", "negative"),
        ("classify: Highly recommended!", "positive"),
        ("classify: Complete waste of money.", "negative"),
    ]

    optimizer = AdamW(model.parameters(), lr=lr)

    print("=== Toy fine-tuning (demo only) ===")
    for epoch in range(epochs):
        total_loss = 0.0
        for src, tgt in train_pairs:
            inputs = tokenizer(src, return_tensors="pt", truncation=True)
            labels = tokenizer(tgt, return_tensors="pt", truncation=True).input_ids
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
        avg = total_loss / max(len(train_pairs), 1)
        print(f"epoch={epoch + 1}  avg_loss={avg:.4f}")

    model.eval()
    tests = [
        "classify: This is the best thing ever!",
        "classify: I hate waiting in long lines.",
    ]
    print("=== After fine-tuning: quick checks ===")
    for src in tests:
        inputs = tokenizer(src, return_tensors="pt", truncation=True)
        with torch.no_grad():
            gen = model.generate(inputs.input_ids, max_new_tokens=8, num_beams=2)
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        print(f"{src}  ->  {out}")


def span_corruption_toy_demo() -> None:
    """
    Print a hand-constructed span corruption triple: original, corrupted source, target.
    This mirrors T5 pre-training structure without running a full training loop.
    """
    original = "The cat sat on the mat and watched the birds fly."
    corrupted = "The <extra_id_0> on the <extra_id_1> the birds fly."
    target = "<extra_id_0> cat sat <extra_id_1> mat and watched"
    print("=== Span corruption (hand-built illustration) ===")
    print(f"original:  {original}")
    print(f"corrupted: {corrupted}")
    print(f"target:    {target}")
    print(
        "The decoder learns to emit sentinel-aligned spans. "
        "Unique sentinels disambiguate multiple corrupted regions."
    )


def main() -> None:
    set_seed(0)
    model_name = "t5-small"
    t5_multitask_inference(model_name)
    t5_toy_finetune(model_name, epochs=5)
    span_corruption_toy_demo()


if __name__ == "__main__":
    main()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Explain T5’s text-to-text framing and give three tasks expressed as input strings and output strings.
    2. How does span corruption differ from BERT-style masked language modeling at the objective level?
    3. What is the role of sentinel tokens in span corruption, and why use unique sentinels per span?
    4. Compare encoder self-attention, decoder self-attention, and cross-attention in one sentence each.
    5. Why might encoder-decoder models be favorable for summarization versus a decoder-only model with a long prompt?
    6. What did T5’s ablations suggest about corruption rate and mean span length?
    7. Describe Flan-T5 and how it relates to instruction tuning in modern LLMs.
    8. How does relative position bias differ conceptually from sinusoidal absolute positions?
    9. What are deployment considerations for caching encoder representations in a seq2seq service?
    10. When would you still choose a decoder-only stack for a product despite T5-style advantages on paper?

!!! interview "Follow-up Probes"
    - "Walk me through the cross-attention tensors: what are queries, keys, and values tied to?"
    - "What happens to compute if the source sequence doubles in length?"
    - "How would you batch heterogeneous tasks in fine-tuning without catastrophic interference?"
    - "Why might span corruption reduce decoder steps compared to reconstructing every token?"
    - "How does beam search interact with task prefixes in text-to-text models?"

!!! key-phrases "Key Phrases to Use in Interviews"
    - "Unified text-to-text interface with task prefixes"
    - "Bidirectional encoder memory with causal decoder and cross-attention"
    - "Span corruption with sentinel tokens to mark deleted spans"
    - "Ablations on objective, span length, and architecture at matched compute"
    - "Flan-T5: large-scale instruction fine-tuning on diverse tasks"

---

## References

- Raffel et al. (2019), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*
- Xue et al. (2020), *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*
- Chung et al. (2022), *Scaling Instruction-Finetuned Language Models* (Flan-T5)
- Tay et al. (2022), *UL2: Unifying Language Learning Paradigms*
