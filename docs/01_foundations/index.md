# Part 1 — Foundations

Before transformers and billion-parameter models, language modeling rests on probability, representations, and sequence models. This part builds the vocabulary and math you will reuse everywhere else.

## Goals

- Relate **classical and neural language models** (n-grams → LSTMs) to modern **token prediction**.
- Understand **embeddings** and why geometry matters for similarity and analogy.
- Derive **scaled dot-product attention** and softmax weights from first principles.

## Topics

| # | Topic | Key ideas |
| --- | --- | --- |
| 1.1 | [Language Modeling Basics](language_modeling_basics.md) | N-grams, chain rule, perplexity, smoothing |
| 1.2 | [Word Embeddings](word_embeddings.md) | Word2Vec, GloVe, FastText, t-SNE visualization |
| 1.3 | [Neural Language Models](neural_language_models.md) | FFNN LM, RNNs, LSTMs, vanishing gradients |
| 1.4 | [Sequence-to-Sequence](sequence_to_sequence.md) | Encoder–decoder, Bahdanau & Luong attention, teacher forcing |
| 1.5 | [Information Theory](information_theory.md) | Entropy, cross-entropy, KL divergence, perplexity link |
| 1.6 | [Mathematics of Attention](attention_math.md) | Scaled dot-product, softmax, causal masking, multi-head preview |

## How to read

Each page follows: **Intuition → math → code (with full imports) → interview takeaways → references.** Start at 1.1 and work forward — each topic builds on the previous one, culminating in the attention equation that powers Part 2.

## Status

All six foundation topics are complete with math, runnable code, and interview takeaways. Diagrams (draw.io SVGs) and Jupyter notebooks will be added in a future polish pass.
