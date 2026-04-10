# Foundations of Language Modeling

The mathematical and conceptual bedrock that existed before the Transformer. Understanding these topics deeply is what separates candidates who can explain *why* things work from those who only know *that* they work.

!!! info "What Changed in This Section"
    Every page now includes **everyday analogies**, **"Think of it like..."** callouts, and **simplified introductions** before diving into the math. The mathematical content is unchanged — we've added layers of explanation on top so that anyone with high school math can follow along. If a formula looks intimidating, read the paragraph right above it first.

---

## Before You Start

### Prerequisites

This section builds on the **Deep Learning Fundamentals** (Part 0). You should be comfortable with:

- **Perceptrons, MLPs, and forward passes** — see [The Perceptron and Feedforward Networks](../00_prerequisites/01_perceptron_and_ffn.md)
- **Activation functions** (sigmoid, tanh, ReLU, softmax) — see [Activation Functions](../00_prerequisites/02_activation_functions.md)
- **Backpropagation and gradient descent** — see [Backpropagation and Gradient Descent](../00_prerequisites/03_backpropagation.md)
- **Cross-entropy loss** — see [Loss Functions and Regularization](../00_prerequisites/04_loss_and_regularization.md)
- **Sequence modeling and RNN basics** — see [Sequence Modeling and RNNs](../00_prerequisites/06_sequence_modeling_and_rnns.md)

**Mathematics used in this section:**
- Conditional probability and the chain rule — reviewed in [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#5-probability-basics)
- Vector notation \(\mathbf{v} \in \mathbb{R}^d\) — explained in [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#1-vectors-and-matrices)
- Logarithms and exponentials — reviewed in [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#4-exponentials-and-logarithms)
- Summation and product notation — explained in [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#3-summation-and-product-notation)

### Reading Strategy

This section progresses from **classical** language models (n-grams) → **distributional** semantics (embeddings) → **recurrent** networks → **attention**. If you're new to language modeling:

**First Pass (Build Intuition):**
- Focus on language_modeling_basics.md and word_embeddings.md — these introduce core concepts
- Read neural_language_models.md for LSTM/GRU gate intuition (skip spectral norm discussion)
- Read sequence_to_sequence.md for the encoder-decoder pattern and attention motivation
- Read information_theory.md for entropy, cross-entropy, and perplexity (skip forward/reverse KL deep dive)
- Read attention_math.md for scaled dot-product attention derivation and the 4×4 worked example

**Second Pass (Deepen Understanding):**
- Re-read with the "Deep Dive" sections included
- Study GloVe matrix factorization, PMI, and noise contrastive estimation
- Work through the full LSTM numerical trace
- Study KL divergence mode-covering vs mode-seeking behavior
- Analyze attention complexity (MAC counts) and multi-head projections

!!! tip "What to Skip on First Reading"
    - Deep dives on PMI, noise contrastive estimation, and GloVe factorization (word_embeddings.md)
    - Spectral norm and Jacobian analysis in vanishing gradients (neural_language_models.md)
    - Scheduled sampling deep dive (sequence_to_sequence.md)
    - Forward vs reverse KL divergence, DPO, and Bradley-Terry model (information_theory.md)
    - FlashAttention and low-rank kernel approximation (attention_math.md)

---

## Goals

After completing this section you will be able to:

- Derive the chain rule decomposition of language models and explain the Markov assumption
- Explain how Word2Vec, GloVe, and FastText learn vector representations of meaning
- Trace data through an LSTM cell gate by gate with actual numbers
- Describe the encoder-decoder framework and how attention solves the bottleneck problem
- Connect entropy, cross-entropy, and KL divergence to LLM training objectives
- Derive scaled dot-product attention from first principles and explain why we scale by \(\sqrt{d_k}\)

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Language Modeling](language_modeling_basics.md) | N-grams, chain rule, smoothing, perplexity |
| 2 | [Word Embeddings](word_embeddings.md) | Word2Vec, GloVe, FastText, embedding arithmetic |
| 3 | [Neural Language Models](neural_language_models.md) | RNNs, vanishing gradients, LSTM gates, GRU |
| 4 | [Sequence-to-Sequence](sequence_to_sequence.md) | Encoder-decoder, Bahdanau attention, teacher forcing |
| 5 | [Information Theory](information_theory.md) | Entropy, cross-entropy, KL divergence, perplexity |
| 6 | [Attention Mathematics](attention_math.md) | Scaled dot-product attention, multi-head attention, masking |

---

## Hands-On Notebooks

Practice with interactive Jupyter notebooks — each combines toy examples (build from scratch with NumPy/PyTorch) with real-world usage (HuggingFace transformers, gensim):

| Notebook | Covers Topics |
|----------|---------------|
| [Language Modeling & Embeddings](../notebooks/04_language_modeling_and_embeddings.ipynb) | Language Modeling, Word Embeddings |
| [Neural LM & Seq2Seq](../notebooks/05_neural_lm_and_seq2seq.ipynb) | Neural Language Models, Sequence-to-Sequence |
| [Information Theory & Attention](../notebooks/06_information_theory_and_attention.ipynb) | Information Theory, Attention Mathematics |

---

Every page includes:

1. **Simple analogies** — everyday comparisons to build intuition before any math
2. **"In Plain English"** — what each equation means in words
3. **Worked examples** — step-by-step calculations with real numbers
4. **Runnable Python code** — so you can verify the math yourself
5. **FAANG-level interview questions** — with expected answer depth

!!! tip "Reading Order for Beginners"
    If you're encountering these topics for the first time, we recommend reading in order (1 → 6). Each page builds on concepts from the previous one. The "Think of it like..." boxes at the start of each section give you the intuition before the math arrives.
