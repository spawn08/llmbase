# Foundations of Language Modeling

The mathematical and conceptual bedrock that existed before the Transformer. Understanding these topics deeply is what separates candidates who can explain *why* things work from those who only know *that* they work.

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

Every page includes plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions with expected answer depth.
