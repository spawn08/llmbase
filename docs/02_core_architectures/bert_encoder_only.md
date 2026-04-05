# 2.5 — BERT: Encoder-Only Transformers

## Intuition

While GPT predicts the next token left-to-right, BERT (*Bidirectional Encoder Representations from Transformers*, Devlin et al., 2018) reads the **entire sentence at once** and learns to fill in blanks. This bidirectional context produces richer representations for classification, entity recognition, and question answering — tasks where you have the full input before making a decision.

BERT dominated NLP benchmarks from 2018–2022 and remains the architecture of choice for **embedding models**, **rerankers**, and **classification** in production systems.

---

## Core concepts

### Architecture

BERT is an **encoder-only** Transformer — no causal mask, no decoder:

```
[CLS] tokens [SEP]  →  Embedding (token + segment + position)  →  Encoder × N  →  hidden states
```

- **No masking** — every token attends to every other token (bidirectional).
- **[CLS] token** — a special token prepended to every input. Its final hidden state serves as the sequence representation for classification.
- **[SEP] token** — separates sentence pairs (for tasks like NLI, QA).
- **Segment embeddings** — distinguish sentence A from sentence B.

### Pre-training objectives

**1. Masked Language Model (MLM)**

Randomly mask 15% of tokens; predict the originals:

\[
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i \mid w_{\setminus i})
\]

Of the 15% selected, 80% are replaced with `[MASK]`, 10% with a random token, 10% left unchanged. This prevents the model from learning that `[MASK]` = "something to predict."

**2. Next Sentence Prediction (NSP)** — *historical*

Given sentences A and B, predict whether B follows A. Later work (RoBERTa) showed **NSP hurts performance** and should be dropped.

### Fine-tuning

BERT's key insight: pre-train once on massive data, then **fine-tune** on small labeled datasets:

| Task | Input | Output layer |
| --- | --- | --- |
| Classification (e.g., sentiment) | Single sentence | Linear on `[CLS]` |
| NLI / paraphrase | Sentence pair with `[SEP]` | Linear on `[CLS]` |
| Token classification (NER) | Single sentence | Linear on each token |
| Question answering | Question `[SEP]` passage | Start/end span logits on each token |

### BERT variants

| Model | Year | Key change | Params |
| --- | --- | --- | --- |
| BERT-base | 2018 | Original | 110M |
| BERT-large | 2018 | Wider + deeper | 340M |
| RoBERTa | 2019 | Drop NSP, more data, dynamic masking | 355M |
| ALBERT | 2019 | Factorized embeddings, cross-layer sharing | 12M–235M |
| DistilBERT | 2019 | Knowledge distillation, 60% size, 97% perf | 66M |
| DeBERTa | 2020 | Disentangled attention (content + position) | 134M–1.5B |

### Why BERT can't generate text

BERT sees all tokens simultaneously — it doesn't have a left-to-right ordering. Predicting masked tokens is a **fill-in-the-blank** task, not sequential generation. For text generation, you need GPT's causal mask (Part 2.4).

### BERT as an embedding model

BERT's hidden states (especially from the last few layers) produce high-quality **sentence embeddings**. Models like **Sentence-BERT (SBERT)** fine-tune BERT with contrastive objectives to produce embeddings where cosine similarity reflects semantic similarity. These are used in:

- Semantic search and retrieval (RAG pipelines)
- Clustering and deduplication
- Reranking search results

---

## Code — BERT MLM and fine-tuning with HuggingFace

```python
"""
BERT: Masked Language Model inference and text classification fine-tuning.
Uses HuggingFace Transformers for practical, production-relevant examples.
"""
import torch
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    BertForSequenceClassification,
    AdamW,
)


# ── 1. Masked Language Model — Fill in the blank ───────────────────────
def mlm_demo() -> None:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    text = "The capital of France is [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    mask_idx = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        logits = model(**inputs).logits

    top_tokens = logits[0, mask_idx].topk(5, dim=-1)
    print("MLM: 'The capital of France is [MASK].'")
    for score, idx in zip(top_tokens.values[0], top_tokens.indices[0]):
        print(f"  {tokenizer.decode(idx):12s} (score: {score:.2f})")


# ── 2. Fine-tuning for classification ─────────────────────────────────
def classification_demo() -> None:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    # Toy labeled data
    texts = [
        ("I love this movie, it was fantastic!", 1),
        ("Terrible film, waste of time.", 0),
        ("Best performance I have ever seen.", 1),
        ("Boring and predictable plot.", 0),
    ]

    optimizer = AdamW(model.parameters(), lr=2e-5)
    model.train()

    for epoch in range(3):
        total_loss = 0.0
        for text, label in texts:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs["labels"] = torch.tensor([label])
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}  loss={total_loss / len(texts):.4f}")

    # Inference
    model.eval()
    test_text = "This is an amazing product!"
    inputs = tokenizer(test_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1).item()
    print(f"\nPrediction for '{test_text}': {'positive' if pred == 1 else 'negative'}")


# ── 3. Extract embeddings ─────────────────────────────────────────────
def embedding_demo() -> None:
    from transformers import BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    sentences = ["The cat sat on the mat.", "A kitten rested on the rug."]
    embeddings = []

    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] embedding
        embeddings.append(cls_emb)

    cos_sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1])
    print(f"\nCosine similarity between:")
    print(f"  '{sentences[0]}'")
    print(f"  '{sentences[1]}'")
    print(f"  = {cos_sim.item():.4f}")


if __name__ == "__main__":
    print("=== MLM Demo ===")
    mlm_demo()
    print("\n=== Classification Fine-tuning ===")
    classification_demo()
    print("\n=== Embedding Demo ===")
    embedding_demo()
```

---

## Interview takeaways

1. **Bidirectional vs. autoregressive** — BERT sees all tokens simultaneously (encoder). GPT sees only past tokens (decoder). This is the most fundamental architectural distinction in NLP. Be clear about which tasks each excels at.
2. **MLM objective** — mask 15%, predict the original. Know the 80/10/10 split (mask/random/keep) and *why* — to avoid a distribution mismatch between pre-training (has `[MASK]`) and fine-tuning (no `[MASK]`).
3. **NSP is harmful** — RoBERTa showed that removing NSP and training longer with more data improves quality. Know that RoBERTa also uses dynamic masking (different mask per epoch) vs. BERT's static masking.
4. **[CLS] token** — its final hidden state is used as the sequence-level representation. In contrastive embedding models (SBERT), this becomes the sentence embedding.
5. **Fine-tuning is cheap** — BERT pre-training takes weeks on TPUs; fine-tuning takes minutes on a single GPU with a few thousand labeled examples. This asymmetry is the economic insight of the BERT era.
6. **BERT in production today** — still heavily used for classification, NER, reranking, and embedding. Decoder-only models dominate generation, but encoders dominate understanding/retrieval.

---

## References

- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Liu et al. (2019), *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- He et al. (2020), *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
