# BERT: Pre-training of Deep Bidirectional Transformers

**Authors:** Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova  
**Year:** 2018 &nbsp;|&nbsp; **Venue:** NAACL  
**Link:** [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

---

## TL;DR

BERT pre-trains a **bidirectional** Transformer encoder using **Masked Language Modeling (MLM)**: randomly mask 15% of input tokens and predict them using both left and right context. A second objective, **Next Sentence Prediction (NSP)**, teaches the model to understand inter-sentence relationships. Fine-tuning replaces only a thin task-specific head on top, yielding strong results on classification, NER, QA, and more.

---

## Why This Paper Matters

BERT demonstrated that **pre-training + fine-tuning** is wildly effective. Before BERT, NLP models were typically trained from scratch per task. BERT changed the paradigm: pre-train once on a large corpus, fine-tune cheaply on any downstream task. BERT-style encoders still power **retrieval models**, **rerankers**, **classification**, and **NER** in production systems today. The MLM objective is fundamentally different from GPT's causal LM, and interviewers love contrasting the two.

---

## Key Concepts Explained Simply

### Masked Language Modeling (MLM)

Think of it as a **fill-in-the-blank** exercise. Given "The cat [MASK] on the mat," the model must predict "sat" using context from **both sides**. This is what makes BERT **bidirectional** — unlike GPT, which can only look left.

The masking strategy has three cases (for the 15% of tokens selected):
- **80%** of the time: Replace with `[MASK]`
- **10%** of the time: Replace with a random token
- **10%** of the time: Keep the original token

This prevents the model from learning a shortcut where it only pays attention to `[MASK]` tokens.

### Next Sentence Prediction (NSP)

Given two sentences A and B, predict whether B actually follows A in the corpus (IsNext) or is randomly sampled (NotNext). This was designed to help tasks like question answering where understanding sentence relationships matters.

**Important:** NSP was later shown to be **unnecessary** or even harmful. RoBERTa dropped it entirely and got better results.

### Fine-Tuning

For downstream tasks, you add a simple output layer:
- **Classification:** Take the `[CLS]` token representation → linear layer → softmax
- **Token-level tasks (NER):** Take each token's representation → linear layer per token
- **QA:** Predict start and end positions of the answer span

---

## The Math — Explained Step by Step

### MLM Loss

\[
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{\mathbf{x}} \sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid \tilde{\mathbf{x}})
\]

**Breaking it down:**

1. \(\tilde{\mathbf{x}}\): The corrupted input (with masks applied)
2. \(\mathcal{M}\): The set of masked positions (about 15% of tokens)
3. \(P_\theta(x_i \mid \tilde{\mathbf{x}})\): The model's predicted probability for the original token at position \(i\), given the corrupted input
4. The loss only computes over masked positions — unmasked tokens don't contribute gradients to the MLM head

The key insight: the model sees the **entire** corrupted sequence (left and right context) when predicting each masked token. This is fundamentally different from GPT's left-to-right factorization.

### NSP Loss

\[
\mathcal{L}_{\text{NSP}} = -\left[y \log P_\theta(\text{IsNext}) + (1-y) \log P_\theta(\text{NotNext})\right]
\]

Standard binary cross-entropy. \(y=1\) when B follows A, \(y=0\) when B is random.

### Total Pre-training Loss

\[
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
\]

---

## Python Implementation

```python
import numpy as np


def stable_softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def mlm_mask_tokens(token_ids, vocab_size, mask_token_id,
                    mask_prob=0.15, seed=None):
    """
    Apply BERT-style masking: 80% [MASK], 10% random, 10% keep.
    Returns masked input and positions that were masked.
    """
    if seed is not None:
        np.random.seed(seed)

    masked = token_ids.copy()
    mask_positions = []

    for i in range(len(token_ids)):
        if np.random.random() < mask_prob:
            mask_positions.append(i)
            r = np.random.random()
            if r < 0.8:
                masked[i] = mask_token_id
            elif r < 0.9:
                masked[i] = np.random.randint(0, vocab_size)
            # else: keep original (10%)

    return masked, mask_positions


def mlm_loss(logits_at_masked, true_labels):
    """
    Cross-entropy loss at masked positions only.
    logits_at_masked: [num_masked, vocab_size]
    true_labels: [num_masked] — true token IDs
    """
    probs = stable_softmax(logits_at_masked)
    loss = 0.0
    for i, label in enumerate(true_labels):
        loss -= np.log(probs[i, label] + 1e-12)
    return loss / len(true_labels)


def nsp_loss(cls_logits, is_next):
    """
    Binary cross-entropy for Next Sentence Prediction.
    cls_logits: [2] — logits for [IsNext, NotNext]
    is_next: 0 or 1
    """
    probs = stable_softmax(cls_logits)
    return -np.log(probs[is_next] + 1e-12)


def bert_embedding(token_ids, segment_ids, position_ids,
                   token_emb, segment_emb, position_emb):
    """
    BERT input = token embedding + segment embedding + position embedding.
    """
    return token_emb[token_ids] + segment_emb[segment_ids] + position_emb[position_ids]


class SimpleBERTDemo:
    """Minimal BERT forward pass for understanding the architecture."""

    def __init__(self, vocab_size=1000, d_model=64, n_heads=4, max_len=128):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads

        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.segment_emb = np.random.randn(2, d_model) * 0.02
        self.position_emb = np.random.randn(max_len, d_model) * 0.02

        self.mlm_head = np.random.randn(d_model, vocab_size) * 0.02
        self.nsp_head = np.random.randn(d_model, 2) * 0.02

    def forward(self, token_ids, segment_ids, mask_positions):
        seq_len = len(token_ids)
        position_ids = np.arange(seq_len)

        x = bert_embedding(
            token_ids, segment_ids, position_ids,
            self.token_emb, self.segment_emb, self.position_emb
        )

        # MLM logits at masked positions
        mlm_logits = x[mask_positions] @ self.mlm_head

        # NSP logits from [CLS] (position 0)
        nsp_logits = x[0] @ self.nsp_head

        return mlm_logits, nsp_logits


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    vocab_size = 1000
    mask_token_id = 999

    tokens = np.array([101, 45, 200, 67, 88, 102, 33, 55, 102])
    segments = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])

    masked_tokens, mask_pos = mlm_mask_tokens(
        tokens, vocab_size, mask_token_id, seed=42
    )
    print(f"Original:  {tokens}")
    print(f"Masked:    {masked_tokens}")
    print(f"Mask positions: {mask_pos}")

    model = SimpleBERTDemo(vocab_size=vocab_size)
    mlm_logits, nsp_logits = model.forward(masked_tokens, segments, mask_pos)

    true_labels = tokens[mask_pos]
    loss_mlm = mlm_loss(mlm_logits, true_labels)
    loss_nsp = nsp_loss(nsp_logits, is_next=1)

    print(f"\nMLM loss: {loss_mlm:.4f}")
    print(f"NSP loss: {loss_nsp:.4f}")
    print(f"Total loss: {loss_mlm + loss_nsp:.4f}")
```

---

## Interview Importance

BERT is a **top-5 most-asked** paper. It's the canonical encoder-only model and the contrast point against GPT-style decoders.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Explain MLM vs. causal language modeling. What are the trade-offs?

**Answer:** MLM masks random tokens and predicts them using **bidirectional** context — the model sees both left and right. Causal LM (GPT) predicts each token using **only left** context. 

**Trade-offs:**
- MLM produces better **representations** for understanding tasks (classification, retrieval) because every position has full context
- Causal LM is naturally suited for **generation** — you can sample token-by-token
- MLM can't easily generate text because masked positions are independent of each other (the model assumes masked tokens are conditionally independent given the rest)

### Q2: Why was NSP criticized, and what did RoBERTa change?

**Answer:** NSP was found to be too easy — the model could often distinguish real vs. random sentence pairs just from topic mismatch rather than learning real discourse structure. RoBERTa showed that removing NSP and training with **full-document** sequences improved performance. The key changes in RoBERTa: (1) drop NSP, (2) dynamic masking (re-sample masks each epoch), (3) larger batches, (4) more data, (5) longer training.

### Q3: How would you use a BERT-style model in a RAG pipeline?

**Answer:** Two main roles:
- **Retriever:** Use BERT as a bi-encoder to embed both queries and documents into the same vector space. Find relevant documents via nearest-neighbor search (e.g., FAISS).
- **Reranker:** Use BERT as a cross-encoder — concatenate query + document as `[CLS] query [SEP] document [SEP]` and predict a relevance score. Cross-encoders are more accurate than bi-encoders but can't be pre-computed.

### Q4: Why does BERT use three masking strategies (80/10/10) instead of always using [MASK]?

**Answer:** If BERT always replaced selected tokens with `[MASK]`, there would be a **train-test mismatch**: during fine-tuning, the model never sees `[MASK]` tokens, but during pre-training it learned to attend heavily to them. The 10% random replacement forces the model to maintain good representations for all tokens (not just masked ones). The 10% keep-original ensures the model doesn't learn "if a token is not [MASK], it must be correct."

### Q5: Can BERT generate text? Why or why not?

**Answer:** BERT is not designed for generation. MLM assumes masked positions are **conditionally independent** — it predicts each mask separately. For generation, you need the probability of a full sequence, which requires either left-to-right factorization (GPT) or iterative refinement. Some work (e.g., mask-predict for machine translation) uses BERT-like models for generation by iteratively masking and predicting, but this is inefficient compared to autoregressive models.

---

## Connections to Other Papers

- **Transformer** → BERT uses the encoder stack
- **GPT-2** → Contrasting approach: decoder-only, causal LM
- **RoBERTa** → "BERT done right" — better training recipe
- **ELECTRA** → Replaced token detection instead of MLM (more efficient)
- **XLNet** → Permutation LM to get bidirectional context without masking

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Architecture | Transformer encoder (bidirectional attention) |
| Pre-training | MLM (15% mask → predict) + NSP (dropped by RoBERTa) |
| Masking strategy | 80% [MASK], 10% random, 10% keep |
| Fine-tuning | Add thin task head on top of pre-trained encoder |
| Best for | Classification, NER, retrieval, reranking |
| Not for | Text generation (use GPT for that) |
