# 2.6 — T5: Encoder–Decoder Transformers

## Intuition

T5 (*Text-to-Text Transfer Transformer*, Raffel et al., 2019) unifies every NLP task into a single format: **text in → text out**. Sentiment analysis becomes "sentiment: I love it" → "positive". Translation becomes "translate English to French: hello" → "bonjour". This framing lets one model and one training procedure handle classification, generation, QA, and summarization — with no task-specific architecture changes.

T5 also serves as the most thorough **ablation study** ever published on Transformer design choices. If someone asks "which Transformer configuration works best?", the T5 paper has the data.

---

## Core concepts

### Architecture: full encoder–decoder

T5 uses the **original Transformer** layout:

```
Source text → Encoder (bidirectional self-attention × N)
                ↓ cross-attention
Target text → Decoder (causal self-attention + cross-attention × N) → output tokens
```

- **Encoder** — bidirectional self-attention (like BERT). Reads the full input.
- **Decoder** — causal self-attention (like GPT) + cross-attention to encoder outputs. Generates output left-to-right.
- **Relative position bias** — T5 uses a **learned relative position bias** (bucketed distances) instead of sinusoidal or absolute embeddings.

### Text-to-text framing

Every task is cast as sequence-to-sequence with a **task prefix**:

| Task | Input | Target |
| --- | --- | --- |
| Translation | `translate English to French: The house is beautiful.` | `La maison est belle.` |
| Sentiment | `sst2 sentence: This movie was great` | `positive` |
| Summarization | `summarize: <article text>` | `<summary text>` |
| QA | `question: What color is the sky? context: The sky is blue.` | `blue` |
| Grammar | `cola sentence: The cat sat on the the mat.` | `unacceptable` |

This eliminates task-specific heads — the model always outputs text, decoded with the same vocabulary and beam search.

### Pre-training: span corruption

T5 uses a variant of MLM called **span corruption**: randomly select spans of tokens, replace each with a sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...), and train the decoder to produce the missing spans:

**Input:** `The <extra_id_0> sat on the <extra_id_1>.`

**Target:** `<extra_id_0> cat <extra_id_1> mat`

This is more efficient than BERT's token-level MLM because the decoder generates fewer tokens.

### T5 ablation findings

The T5 paper systematically tested design choices. Key results:

| Design choice | Winner |
| --- | --- |
| Architecture | Encoder–decoder slightly beats decoder-only for the same FLOP budget |
| Pre-training objective | Span corruption > token-level MLM > language modeling |
| Corruption rate | 15% of tokens (matching BERT) |
| Span length | Mean span length 3 is optimal |
| Model size vs. data | Both matter; more data is undervalued |
| Multi-task vs. single-task fine-tuning | Multi-task helps but requires tuning mixing rates |

### T5 family evolution

| Model | Year | Params | Key change |
| --- | --- | --- | --- |
| T5 | 2019 | 60M–11B | Original encoder–decoder |
| mT5 | 2020 | 300M–13B | Multilingual (101 languages) |
| Flan-T5 | 2022 | 80M–11B | Instruction-tuned on 1,800+ tasks |
| UL2 | 2022 | 20B | Mixture-of-Denoisers (multiple objectives) |

### Encoder–decoder vs. decoder-only: when each wins

| Scenario | Better choice | Why |
| --- | --- | --- |
| Conditional generation (translation, summary) | Encoder–decoder | Encoder reads full source; decoder generates conditioned on it |
| Open-ended generation / chat | Decoder-only | Simpler, scales better with in-context learning |
| Classification / regression | Either (encoder-only is cheapest) | Encoder-only avoids decoding overhead |
| Same FLOP budget, diverse tasks | Encoder–decoder | T5 paper showed slight advantage |

In practice, decoder-only dominates at extreme scale (GPT-4, LLaMA, Mistral) because:

- Simpler to train and serve (one stack, no cross-attention KV cache).
- In-context learning reduces the need for specialized architectures.
- Industry training infrastructure is optimized for causal LMs.

---

## Code — T5 fine-tuning and inference with HuggingFace

```python
"""
T5: text-to-text fine-tuning and generation.
Demonstrates the unified input/output framing and span corruption.
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


def t5_inference_demo() -> None:
    """Use a pre-trained T5 for translation and summarization."""
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.eval()

    tasks = [
        "translate English to French: The house is beautiful.",
        "translate English to German: Good morning, how are you?",
        "summarize: State authorities dispatched emergency crews Tuesday "
        "to survey damage from prior-day storms that killed at least seven people.",
    ]

    for task_input in tasks:
        input_ids = tokenizer(task_input, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=64, num_beams=4)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input:  {task_input}")
        print(f"Output: {result}\n")


def t5_finetune_demo() -> None:
    """Fine-tune T5-small on a tiny sentiment dataset."""
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    train_data = [
        ("classify: I love this product!", "positive"),
        ("classify: Terrible experience, never again.", "negative"),
        ("classify: Absolutely wonderful service.", "positive"),
        ("classify: Worst purchase I ever made.", "negative"),
        ("classify: Highly recommended!", "positive"),
        ("classify: Complete waste of money.", "negative"),
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()

    for epoch in range(5):
        total_loss = 0.0
        for input_text, target_text in train_data:
            input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
            labels = tokenizer(target_text, return_tensors="pt", truncation=True).input_ids

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}  loss={total_loss / len(train_data):.4f}")

    # Test
    model.eval()
    test_inputs = [
        "classify: This is the best thing ever!",
        "classify: I hate waiting in long lines.",
    ]
    for text in test_inputs:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=5)
        print(f"  {text} → {tokenizer.decode(output_ids[0], skip_special_tokens=True)}")


def span_corruption_demo() -> None:
    """Demonstrate T5's span corruption pre-training objective."""
    print("\n=== Span Corruption (Pre-training Objective) ===")
    original = "The cat sat on the mat and watched the birds fly."
    corrupted = "The <extra_id_0> on the <extra_id_1> the birds fly."
    target = "<extra_id_0> cat sat <extra_id_1> mat and watched"
    print(f"Original:  {original}")
    print(f"Corrupted: {corrupted}")
    print(f"Target:    {target}")
    print("The model learns to reconstruct masked spans.")


if __name__ == "__main__":
    print("=== T5 Inference Demo ===\n")
    t5_inference_demo()
    print("=== T5 Fine-tuning Demo ===")
    t5_finetune_demo()
    span_corruption_demo()
```

---

## Interview takeaways

1. **Text-to-text** — the key idea. Every task becomes seq2seq. The task prefix acts as an instruction. This was a precursor to instruction tuning (FLAN, Part 3.4).
2. **Span corruption vs. MLM** — span corruption is more efficient because the decoder generates only the missing spans, not the entire sequence. Know that mean span length = 3 and corruption rate = 15%.
3. **Encoder–decoder vs. decoder-only** — at equal compute, encoder–decoder has a slight edge on conditional generation tasks (T5 paper evidence). But decoder-only wins at scale because of simpler infrastructure and strong in-context learning.
4. **Relative position bias** — T5 uses learned, bucketed relative positions (different from RoPE or sinusoidal). This allows some length generalization.
5. **Flan-T5** — instruction-tuned T5 on 1,800+ tasks. Often asked about as an example of how instruction tuning improves zero-shot performance. Know it's the bridge between T5 and ChatGPT-era instruction following.
6. **Cross-attention KV cache** — in encoder–decoder models, the encoder outputs are computed once and reused for every decoder step. This is efficient for fixed-input tasks (translation) but adds complexity vs. decoder-only serving.

---

## References

- Raffel et al. (2019), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*
- Xue et al. (2020), *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*
- Chung et al. (2022), *Scaling Instruction-Finetuned Language Models* (Flan-T5)
- Tay et al. (2022), *UL2: Unifying Language Learning Paradigms*
