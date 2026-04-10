# BERT: Encoder-Only Transformers

## Why This Matters for LLMs

BERT (*Bidirectional Encoder Representations from Transformers*) established that **masked language modeling** on large text could pre-train deep bidirectional representations that transfer to classification, span prediction, and sentence-pair tasks with light fine-tuning. Even though generative decoder-only models now dominate open-ended chat, **encoder-only** and **encoder-style** models remain the backbone of **search relevance**, **reranking**, **named entity recognition**, **semantic retrieval**, and **embedding APIs** in production. Interviewers still use BERT as the canonical contrast to GPT because the **attention mask** and **training objective** differ in ways that change what the model can do at inference.

A second reason to master BERT is **representation learning**: BERT’s `[CLS]` vector and token-level hidden states are the historical starting point for **sentence embeddings** and **dense retrieval**. Methods such as Sentence-BERT explicitly fine-tune BERT with **siamese** and **contrastive** objectives so that cosine similarity aligns with semantic similarity. Modern embedding models evolved from this lineage. If you can explain **MLM**, **why bidirectional attention breaks autoregressive generation**, and **how fine-tuning adds a small task head**, you demonstrate literacy in both classical NLP transfer learning and today’s retrieval stacks.

Third, BERT is a clean laboratory for **pre-training distribution** questions: the `[MASK]` token appears during pre-training but rarely at fine-tuning time, so BERT uses a **mixed masking policy** (replace with mask, random token, or keep original) to reduce train–test skew. Similar concerns reappear in instruction tuning and tool-use formatting for large models. Understanding BERT’s objectives trains you to ask the right question: **does the model see the same kind of inputs at deployment that it saw during training?**

---

## Core Concepts

### Architecture

BERT is an **encoder-only** Transformer stack. Every layer uses **full self-attention**: each token may attend to **all** other tokens in the segment (within a maximum length). There is **no causal mask** that hides the future.

Typical single-segment input places a variable number of content tokens between `[CLS]` and `[SEP]`. For a sequence of length \(T\), you can read the layout as `[CLS]` followed by tokens \(w_1\) through \(w_T\) in order, then `[SEP]`. A minimal concrete example with three words is `[CLS] w_1 w_2 w_3 [SEP]`.

For sentence-pair tasks (natural language inference, paraphrase), the input looks like:

```
[CLS]  sentence_A_tokens  [SEP]  sentence_B_tokens  [SEP]
```

**Token embeddings**, **positional embeddings**, and **segment embeddings** (token-type IDs) are summed to form the input to the first encoder layer. The final hidden state of `[CLS]` is often used as a **sequence-level** vector for classification.

---

### Masked Language Model (MLM)

BERT randomly selects **15%** of input subword tokens. For each selected token, one of three actions is applied:

- **80%** of the time replace it with the special `[MASK]` token.
- **10%** of the time replace it with a **random** vocabulary token.
- **10%** of the time **keep** the original token unchanged.

The model is trained to predict the **original** token identity at masked positions using bidirectional context.

Let \(\mathcal{M}\) be the set of masked positions in a sequence. The MLM contribution to training is:

\[
\mathcal{L}_{\text{MLM}} = -\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \log P_\theta(w_i \mid \mathbf{x}_{\setminus \mathcal{M}}, \mathbf{x}_{\mathcal{M}\setminus\{i\}})
\]

In plain implementation terms, the model outputs a distribution over the vocabulary at each position, but you compute the loss only at positions in \(\mathcal{M}\).

!!! math-intuition "In Plain English"
    At each masked slot, BERT must guess the original word using **both left and right** context. The sum over positions in \(\mathcal{M}\) means you average (or sum) cross-entropy losses only where you placed a mask or a surrogate token. The random-token and keep-original tricks prevent the model from learning trivial correlations that only fire when the input contains the literal `[MASK]` string.

The probability uses a softmax over vocabulary logits \(\boldsymbol{\ell}_i \in \mathbb{R}^V\) at position \(i\):

\[
P_\theta(w_i = v \mid \cdot) = \frac{\exp(\ell_{i,v})}{\sum_{v'=1}^{V}\exp(\ell_{i,v'})}
\]

!!! math-intuition "In Plain English"
    This is the same softmax machinery as language modeling, but the **conditioning event** is different: the model may look at future words because there is no causal mask. The predicted distribution answers “which word belongs in this slot given the entire sentence?” rather than “what is the next word left-to-right?”

!!! example "Worked Example: Masking `cat` in a Sentence"
    Input sentence after tokenization (simplified word-level vocabulary for readability):  
    `The [MASK] sat on the mat`  
    Suppose the original token at the mask position is **`cat`**.
    
    **Model outputs** at the `[MASK]` position might produce logits (only four toy words shown):

    | token | logit |
    |-------|-------|
    | cat   | 3.0   |
    | dog   | 1.5   |
    | mat   | 0.2   |
    | the   | 0.1   |

    Softmax denominator:  
    \(\exp(3.0) + \exp(1.5) + \exp(0.2) + \exp(0.1) \approx 20.09 + 4.48 + 1.22 + 1.11 = 26.90\).

    \(P(\text{cat}) \approx 20.09 / 26.90 \approx 0.75\).

    Cross-entropy for the true label `cat`: \(-\log(0.75) \approx 0.29\) nats.

    **Why bidirectional context helps:** From the left, `The` raises mass on nouns and noun phrases; from the right, `sat` prefers animate subjects in simple training corpora. The model combines these cues. A unidirectional left-to-right model at the position of `The` would not yet see `sat` when predicting the token after `The`, so it would lack that future evidence unless you use a different objective.

    During training, this position might not always show `[MASK]`: sometimes the token is replaced by a random word (forcing the model to rely on context to detect the error) or left as `cat` (forcing the model to use contextualized hidden states rather than only copying the surface token).

---

### Next Sentence Prediction (NSP) — Historical Objective

Original BERT included a **binary classification** auxiliary task: given two segments A and B, predict whether B is the actual next sentence of A in the corpus.

\[
\mathcal{L}_{\text{NSP}} = -\big(y \log \sigma(z) + (1-y)\log(1-\sigma(z))\big)
\]

where \(z\) is a learned linear function of the `[CLS]` representation after the encoder, and \(y \in \{0,1\}\) indicates true continuation versus random pairing.

!!! math-intuition "In Plain English"
    This is standard binary cross-entropy. The model must produce a single score from the `[CLS]` vector that separates **genuine adjacent pairs** from **random pairs**. Later work, especially **RoBERTa**, found NSP **unnecessary or harmful** for many tasks when training was otherwise improved. You should know NSP for historical accuracy but also know that modern BERT-family training often **drops** it.

---

### Why BERT Cannot Generate Text Like GPT

Autoregressive generation requires a **factorization** of joint probability as a product of next-token conditionals:

\[
P(w_1,\ldots,w_T) = \prod_{t=1}^{T} P(w_t \mid w_1,\ldots,w_{t-1})
\]

!!! math-intuition "In Plain English"
    The product says the probability of the whole string factors into a chain of **next-step** choices. Step \(t\) only conditions on symbols that appear **before** \(t\). That matches how humans often model language left-to-right for generation: pick the first word, then the second given the first, and continue. Causal Transformers implement this conditioning with a **triangular mask** so the computation at position \(t\) never reads positions \(t+1\) and beyond.

A causal decoder enforces this by construction: the representation at position \(t\) does not depend on future tokens.

BERT with bidirectional attention **does not** define a unique ordering for generating an open-ended continuation. If you attempted to sample the next token left-to-right using BERT logits at each step, you would be **misusing** a model trained for **fill-in-the-blank** with global context. The training objective never taught BERT to produce a **sequential** conditional distribution consistent with causal generation. At a masked position, BERT’s predicted distribution assumes **all other tokens are fixed**; during generation those other tokens are not yet fixed at the right margin.

!!! math-intuition "In Plain English"
    Generative language modeling answers “what comes next given only the past?” BERT answers “what word belongs in this hole given **everything** around it?” Those are different questions. You can **impute** tokens with iterative masking strategies (like some cloze-style procedures), but that is not the same clean autoregressive story as GPT-style sampling.

---

### Fine-Tuning: `[CLS]` and the Classification Head

For **sequence-level classification** (sentiment, topic), BERT feeds the final hidden vector \(\mathbf{h}_{\texttt{[CLS]}} \in \mathbb{R}^{d_{\text{model}}}\) into a **dropout** layer (optional in inference) and a **linear** map to \(C\) classes:

\[
\boldsymbol{\ell} = \mathbf{W} \mathbf{h}_{\texttt{[CLS]}} + \mathbf{b},
\qquad
P(c) = \frac{\exp(\ell_c)}{\sum_{c'=1}^{C}\exp(\ell_{c'})}
\]

!!! math-intuition "In Plain English"
    `[CLS]` is a learned “aggregate slot.” During fine-tuning, gradients flow back through supervised loss into all layers so that the `[CLS]` vector becomes informative for your downstream labels. The linear head is tiny compared to the transformer body.

**Parameter count sketch for BERT-base:** \(d_{\text{model}} = 768\). A **two-class** head uses \(\mathbf{W} \in \mathbb{R}^{2 \times 768}\) and \(\mathbf{b} \in \mathbb{R}^{2}\), which is \(2 \times 768 + 2 = 1538\) parameters. The **body** of BERT-base is about **110 million** parameters, so the head is **far smaller than one tenth of one percent** of the total. Fine-tuning therefore adjusts primarily **shared representations** inside BERT, not a large external module.

For **token-level** tasks (NER), you place a linear head on **every token** hidden state independently (often with a CRF on top in classical pipelines, though plain per-token softmax remains a baseline).

---

### BERT as an Embedding Model

Given input tokens, BERT outputs a hidden vector for each subword token. Common pooling strategies to form a **single sentence embedding** include:

- **`[CLS]` pooling:** use \(\mathbf{h}_{\texttt{[CLS]}}\) from the last layer (or a fixed earlier layer in some pipelines).
- **Mean pooling:** average token hidden states over **non-padding** positions:
  \[
  \mathbf{s} = \frac{1}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \mathbf{h}_i
  \]
  where \(\mathcal{I}\) indexes real tokens.

!!! math-intuition "In Plain English"
    Mean pooling spreads evidence across all content words; `[CLS]` relies on a single learned slot that must compress the sequence. Mean pooling often works better **without** supervised fine-tuning for semantic textual similarity when the base model was not optimized for `[CLS]` alone. `[CLS]` can work well after task-specific fine-tuning that teaches aggregation behavior.

**When to prefer which:** If you have **labeled data** for a task closely related to your deployment distribution, fine-tune and use `[CLS]` if that matches training. If you need a **generic sentence vector** from off-the-shelf BERT without task labels, **mean pooling** over last-layer states is a strong baseline. Always apply **L2 normalization** before cosine similarity if that is your metric.

---

### Sentence-BERT and Contrastive Learning

**Sentence-BERT (SBERT)** fine-tunes Siamese BERT encoders so that **semantic similarity** correlates with **embedding distance**. A typical training pair \((a, b)\) with similarity label \(y\) might use a contrastive or triplet objective. A simple **classification-style** formulation for regression on similarity scores uses cosine similarity \(\cos(\mathbf{s}_a, \mathbf{s}_b)\) fed into a small classifier; **triplet loss** pushes positives closer than negatives:

\[
\mathcal{L}_{\text{triplet}} = \max\big(0,\; \|\mathbf{s}_a - \mathbf{s}_p\|_2^2 - \|\mathbf{s}_a - \mathbf{s}_n\|_2^2 + m\big)
\]

for anchor \(a\), positive \(p\), negative \(n\), and margin \(m\).

!!! math-intuition "In Plain English"
    Standard BERT with mean pooling does not directly optimize cosine space geometry for semantic similarity. SBERT **aligns** the encoder to a metric that retrieval systems need. Contrastive objectives are the conceptual bridge from BERT hidden states to modern **embedding APIs** and **RAG** retrievers.

---

### BERT Variants (Detailed)

| Model | Year | Key change | Parameters (representative) |
|-------|------|------------|-----------------------------|
| BERT-base | 2018 | Original Transformer encoder, MLM + NSP | 110M |
| BERT-large | 2018 | More layers and wider hidden size | 340M |
| RoBERTa | 2019 | Removed NSP, dynamic masking, more data, longer training | 125M / 355M |
| ALBERT | 2019 | Factorized embeddings, cross-layer parameter sharing | 12M–235M depending on config |
| DistilBERT | 2019 | Knowledge distillation from BERT to smaller student | 66M |
| DeBERTa | 2020 | Disentangled attention with separate content and relative position terms | 134M–1.5B |

**RoBERTa** trains longer with larger batches and does not rely on NSP. **Dynamic masking** recomputes which tokens are masked on each epoch, increasing effective diversity.

**ALBERT** reduces parameter count by sharing weights across layers and factorizing the embedding matrix, trading some capacity per layer for efficiency.

**DistilBERT** matches a fraction of BERT’s behavior using a distillation loss that encourages the student logits to mimic the teacher.

**DeBERTa** refines how attention scores combine token content and relative positions, improving robustness on several benchmarks.

---

## Deep Dive

??? deep-dive "Deep Dive: Train–Test Mismatch in MLM and What BERT Did About It"
    If BERT always replaced masked tokens with `[MASK]`, the model might overfit **surface patterns** that only occur around the literal mask token. Real fine-tuning data rarely contains `[MASK]` in the same way. By replacing **10%** of selected tokens with random words and leaving **10%** unchanged, BERT forces the model to use **contextualized hidden states** to detect wrong or surprising tokens and to predict tokens even when the input still shows the original word. This is an early instance of a broader lesson: **match deployment statistics** when possible, and when not possible, **augment training** so the model cannot rely on brittle cues.

??? deep-dive "Deep Dive: Encoder-Only Models in Modern Retrieval Pipelines"
    Contemporary systems often combine **generative** models with **retrieval**. Dense retrievers may still use BERT-derived architectures or their descendants (E5, BGE, sentence-transformers). The engineering pattern is: **encode queries and documents** into vectors, perform **approximate nearest neighbor** search, then pass top passages to a generator. Encoder-only thinking remains central to the **first stage** even when the second stage is decoder-only. Understanding pooling, normalization, and contrastive fine-tuning is how you connect classical BERT concepts to **RAG** latency budgets and recall–precision trade-offs.

---

## Code

The following script uses **Hugging Face Transformers** and **PyTorch**. It demonstrates **masked language modeling inference**, **short classification fine-tuning** on toy data, and **embedding extraction** with cosine similarity. Install dependencies with `pip install torch transformers` in your environment.

```python
"""
BERT with Hugging Face: MLM inference, classification fine-tuning, embeddings.
Requires: pip install torch transformers
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import (
    AdamW,
    BertForMaskedLM,
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
)


def mlm_demo() -> None:
    """Fill-in-the-blank using bert-base-uncased Masked LM."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    model.eval()

    text = "The capital of France is [MASK]."
    inputs = tokenizer(text, return_tensors="pt")
    mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(
        as_tuple=True
    )[1]

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits

    # Logits at the first [MASK] position: shape (vocab_size,)
    mask_logits = logits[0, mask_positions[0], :]
    probs = F.softmax(mask_logits, dim=-1)
    top_k = torch.topk(probs, k=8, dim=-1)

    print("MLM candidates for [MASK] in:", repr(text))
    for score, idx in zip(top_k.values.tolist(), top_k.indices.tolist()):
        tok = tokenizer.decode([idx])
        print(f"  {tok!r:20s}  P ≈ {score:.4f}")


def classification_finetune_demo() -> None:
    """Toy binary sentiment fine-tuning (demonstrates mechanics only)."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
    )
    model.train()

    texts = [
        ("I love this movie, it was fantastic!", 1),
        ("Terrible film, a complete waste of time.", 0),
        ("Brilliant acting and a moving story.", 1),
        ("Dull, predictable, and poorly edited.", 0),
    ]

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(5):
        total_loss = 0.0
        for sentence, label in texts:
            batch = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            batch["labels"] = torch.tensor([label], dtype=torch.long)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.item())
        print(f"epoch {epoch + 1}  avg_loss = {total_loss / len(texts):.4f}")

    model.eval()
    test_sentence = "This is an amazing product and I would buy it again."
    batch = tokenizer(
        test_sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        logits = model(**batch).logits
        pred = int(torch.argmax(logits, dim=-1).item())
    label_name = "positive" if pred == 1 else "negative"
    print(f"\nPrediction for {pred} ({label_name}) on: {test_sentence!r}")


def embedding_demo() -> None:
    """Last-layer [CLS] embeddings and cosine similarity between two sentences."""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    sentences = [
        "The cat sat on the mat.",
        "A kitten rested on the rug.",
    ]
    cls_embeddings: list[torch.Tensor] = []
    mean_embeddings: list[torch.Tensor] = []

    for sent in sentences:
        batch = tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model(**batch)
        last_hidden = out.last_hidden_state  # (1, seq, hidden)
        mask = batch["attention_mask"].unsqueeze(-1)  # (1, seq, 1)
        summed = (last_hidden * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1e-6)
        mean_pool = summed / lengths

        cls_embeddings.append(last_hidden[:, 0, :].squeeze(0))
        mean_embeddings.append(mean_pool.squeeze(0))

    def cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    print("Cosine similarity ([CLS] pooling):", cos(cls_embeddings[0], cls_embeddings[1]))
    print("Cosine similarity (mean pooling): ", cos(mean_embeddings[0], mean_embeddings[1]))


if __name__ == "__main__":
    print("=== Masked Language Modeling ===")
    mlm_demo()
    print("\n=== Classification Fine-Tuning (toy) ===")
    classification_finetune_demo()
    print("\n=== Embeddings ===")
    embedding_demo()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What is masked language modeling, and why does BERT use the 80/10/10 replacement strategy?**
    *Answer:* MLM trains the model to predict **original** tokens at masked positions using **bidirectional** context. **80%** `[MASK]`, **10%** random token, **10%** unchanged prevents the model from relying only on the literal `[MASK]` surface form (train–test mismatch at fine-tune time) and forces use of **contextual** representations.
    2. **Explain the difference between bidirectional encoder attention and causal decoder attention using one sentence each for what information a position may use.**
    *Answer:* **Encoder:** position \(i\) may attend to **all** tokens in the segment (past and future). **Decoder:** position \(i\) may attend only to tokens **at indices \(\le i\)** (past and self), never future.
    3. **Why is BERT unsuitable for standard autoregressive text generation the way GPT is trained?**
    *Answer:* BERT is trained for **joint** fill-in-the-blank with full context, not the product \(\prod_t P(w_t\mid w_{<t})\). It does not define a coherent **left-to-right** conditional for sampling the next token without misusing the training distribution (unless you use iterative masking heuristics, not standard GPT decoding).
    4. **What was Next Sentence Prediction, why was it used originally, and what did RoBERTa conclude?**
    *Answer:* NSP binary-predicted whether segment B **followed** segment A—intended to teach **inter-sentence** relations. **RoBERTa** found NSP often **unnecessary or harmful** when other training improvements (more data, dynamic masking, longer sequences) were applied; many modern recipes **drop** NSP.
    5. **How does fine-tuning add a classification head on top of `[CLS]`, and roughly how many parameters does that head contribute for BERT-base with two labels?**
    *Answer:* Final \(\mathbf{h}_{\texttt{[CLS]}}\in\mathbb{R}^{768}\) is passed through a linear layer to \(C\) logits (often with dropout). For **two** labels: \(W\in\mathbb{R}^{2\times768}\), \(b\in\mathbb{R}^2\) → **1,538** parameters—negligible vs ~110M in the encoder body.
    6. **Compare `[CLS]` pooling versus mean token pooling for sentence embeddings. When might each be preferred?**
    *Answer:* **`[CLS]`** is a single learned aggregate slot—strong when **fine-tuned** for classification with that objective. **Mean pooling** averages real token vectors—often better **off-the-shelf** similarity without task-specific `[CLS]` training. Always **L2-normalize** for cosine similarity.
    7. **What problem does Sentence-BERT solve that raw BERT mean pooling does not directly optimize?**
    *Answer:* Standard BERT + mean pool is trained for **MLM**, not **metric geometry**. SBERT uses **siamese/contrastive** objectives so cosine distance aligns with **semantic similarity**—critical for retrieval and clustering.
    8. **Name three BERT variants and one distinctive idea for each (RoBERTa, ALBERT, DistilBERT, DeBERTa).**
    *Answer:* **RoBERTa**: no NSP, dynamic masking, more/better training. **ALBERT**: factorized embeddings + **cross-layer weight sharing** for fewer params. **DistilBERT**: **distillation** from BERT teacher. **DeBERTa**: **disentangled** content vs relative-position in attention.
    9. **How would you use BERT hidden states in a retrieval pipeline, and what metric would you use between query and document vectors?**
    *Answer:* Encode query and documents to fixed vectors (e.g. mean-pool last layer, or dedicated retriever checkpoints), index with **ANN** (FAISS, ScaNN). **Cosine similarity** (or dot product on L2-normalized vectors) is standard for ranking.
    10. **If validation accuracy plateaus during fine-tuning while training loss keeps dropping, what steps would you take to diagnose overfitting or data issues?**
    *Answer:* Suspect **overfitting**: check gap train/val, early stopping, smaller LR, more data/augmentation, **class balance** on val. Verify **val leakage** or **label noise**; plot per-class metrics; try stronger **regularization** (dropout, weight decay).

!!! interview "Follow-up Probes"
    - **Why might max pooling over tokens be risky compared to mean pooling for variable-length sentences?**
    - **How does segment embedding interact with single-sentence versus pair inputs?**
    - **What happens to gradients for unmasked positions in standard MLM loss computation?**
    - **Why is cosine similarity more common than raw dot product for sentence vectors of different lengths?**
    - **How does distillation for DistilBERT differ from masked language modeling alone?**

!!! key-phrases "Key Phrases to Use in Interviews"
    - **“BERT is encoder-only with full self-attention: every token sees every other token in the segment.”**
    - **“MLM trains the model to predict masked tokens using bidirectional context.”**
    - **“The 80/10/10 masking policy reduces train–test mismatch around the `[MASK]` token.”**
    - **“Autoregressive generation requires a causal factorization; BERT’s objective is fill-in-the-blank with global context.”**
    - **“Fine-tuning attaches a small task head; most parameters live in the shared encoder.”**
    - **“Sentence-BERT aligns embedding space geometry with semantic similarity using siamese or contrastive training.”**
    - **“RoBERTa improved training recipe and often drops NSP; dynamic masking increases data diversity.”**

---

## References

- Devlin et al. (2018), *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*
- Liu et al. (2019), *RoBERTa: A Robustly Optimized BERT Pretraining Approach*
- Lan et al. (2019), *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*
- Sanh et al. (2019), *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*
- Reimers & Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*
- He et al. (2020), *DeBERTa: Decoding-enhanced BERT with Disentangled Attention*
