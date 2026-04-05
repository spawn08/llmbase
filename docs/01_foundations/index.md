# Part 1 — Foundations

Before transformers and billion-parameter models, language modeling rests on probability, representations, and sequence models. This part builds the vocabulary and math you will reuse everywhere else.

## Goals

- Relate **classical and neural language models** (n-grams → LSTMs) to modern **token prediction**.
- Understand **embeddings** and why geometry matters for similarity and analogy.
- Derive **scaled dot-product attention** and softmax weights from first principles.

## Planned topics

| Topic | What you will get |
| --- | --- |
| Language modeling basics | N-grams, perplexity, probability trees |
| Word embeddings | Word2Vec, GloVe, FastText + visualization hooks |
| Neural language models | FFNN / RNN / LSTM language models |
| Sequence-to-sequence | Encoder–decoder and attention precursors |
| Information theory | Entropy, cross-entropy, KL divergence |
| Mathematics of attention | Dot-product, scaling, softmax, attention maps |

## Artifacts (per topic)

Each future page will follow: **Intuition → diagram (SVG) → math → code (full imports) → interview takeaways → references.**

## Status

Phase 1 defines this section shell; **Phase 2** adds full pages, `diagrams/*.drawio`, `docs/assets/diagrams/*.svg`, and notebooks under `notebooks/foundations/`.
