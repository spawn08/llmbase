# 1.1 — Language Modeling Basics

## Intuition

A **language model** assigns a probability to a sequence of words (or tokens). Given a partial sentence, it answers: *what comes next?* Every modern LLM — from GPT to Llama — is fundamentally doing this, just at enormous scale. Understanding the basics here is the foundation for everything that follows.

---

## Core concepts

### What is a language model?

A language model learns a probability distribution over sequences of tokens:

\[
P(w_1, w_2, \ldots, w_T)
\]

Using the **chain rule of probability**, this decomposes into:

\[
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})
\]

Every autoregressive LM (including GPT) models exactly this — it predicts the next token given all previous tokens.

### N-gram language models

An **n-gram model** makes a Markov assumption: the probability of a word depends only on the previous \(n - 1\) words.

**Unigram** (\(n = 1\)):

\[
P(w_t) = \frac{\text{count}(w_t)}{\sum_{w} \text{count}(w)}
\]

**Bigram** (\(n = 2\)):

\[
P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}
\]

**Trigram** (\(n = 3\)):

\[
P(w_t \mid w_{t-2}, w_{t-1}) = \frac{\text{count}(w_{t-2}, w_{t-1}, w_t)}{\text{count}(w_{t-2}, w_{t-1})}
\]

The trade-off is clear: higher \(n\) captures more context, but the number of possible n-grams explodes exponentially, leading to **data sparsity** — most n-grams are never observed in the training corpus.

### Smoothing

To handle unseen n-grams, we use **smoothing** techniques:

- **Add-k (Laplace):** Add a small constant \(k\) to every count.
- **Backoff:** Fall back to a lower-order n-gram when the higher-order count is zero.
- **Interpolation:** Combine probabilities from different orders: \(P_{\text{interp}}(w_t) = \lambda_1 P_{\text{tri}} + \lambda_2 P_{\text{bi}} + \lambda_3 P_{\text{uni}}\).

### Perplexity — measuring model quality

**Perplexity** (PPL) is the standard intrinsic metric for language models. It measures how "surprised" a model is by a test corpus:

\[
\text{PPL} = \exp\!\Bigl(-\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1})\Bigr)
\]

- **Lower is better** — a perfect model that always assigns probability 1 to the correct next word has \(\text{PPL} = 1\).
- Perplexity can be interpreted as the **weighted average branching factor**: a PPL of 50 means the model is "choosing" among ~50 equally likely words at each step.
- It is the exponential of the cross-entropy loss, connecting it directly to how models are trained.

---

## Code — N-gram language model from scratch

```python
"""
N-gram Language Model with Laplace smoothing and perplexity evaluation.
"""
import math
from collections import Counter, defaultdict
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Lowercase split tokenizer with sentence boundary tokens."""
    tokens = text.lower().split()
    return ["<s>"] + tokens + ["</s>"]


def build_ngram_counts(
    corpus: List[List[str]], n: int
) -> Tuple[Counter, Counter]:
    """Return (ngram_counts, context_counts) for order n."""
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    for tokens in corpus:
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            context = ngram[:-1]
            ngram_counts[ngram] += 1
            context_counts[context] += 1
    return ngram_counts, context_counts


def ngram_probability(
    word: str,
    context: Tuple[str, ...],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
    k: float = 1.0,
) -> float:
    """P(word | context) with add-k smoothing."""
    ngram = context + (word,)
    numerator = ngram_counts[ngram] + k
    denominator = context_counts[context] + k * vocab_size
    return numerator / denominator


def perplexity(
    test_tokens: List[str],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
    n: int,
    k: float = 1.0,
) -> float:
    """Compute perplexity of a token sequence under the n-gram model."""
    log_prob_sum = 0.0
    count = 0
    for i in range(n - 1, len(test_tokens)):
        context = tuple(test_tokens[i - n + 1 : i])
        word = test_tokens[i]
        prob = ngram_probability(
            word, context, ngram_counts, context_counts, vocab_size, k
        )
        log_prob_sum += math.log(prob)
        count += 1
    return math.exp(-log_prob_sum / count) if count > 0 else float("inf")


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus_text = [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the log",
        "the dog ate the bone",
    ]
    N = 3  # trigram
    corpus = [tokenize(s) for s in corpus_text]
    vocab = set(tok for sent in corpus for tok in sent)
    V = len(vocab)

    ngram_c, context_c = build_ngram_counts(corpus, N)

    test = tokenize("the cat sat on the bone")
    ppl = perplexity(test, ngram_c, context_c, V, N)
    print(f"Trigram model vocabulary size: {V}")
    print(f"Test sentence: 'the cat sat on the bone'")
    print(f"Perplexity: {ppl:.2f}")

    # Show bigram probabilities for comparison
    bi_c, bi_ctx = build_ngram_counts(corpus, 2)
    for ctx_word, next_word in [("the", "cat"), ("the", "dog"), ("cat", "sat")]:
        p = ngram_probability(next_word, (ctx_word,), bi_c, bi_ctx, V)
        print(f"  P({next_word} | {ctx_word}) = {p:.4f}")
```

---

## Interview takeaways

1. **Chain rule decomposition** — every autoregressive LM, including GPT-4, models \(P(w_t \mid w_{<t})\). The difference is *how* that conditional is parameterized (count tables vs. neural networks).
2. **N-gram sparsity** — explain why higher-order n-grams require smoothing and why this motivated neural approaches that generalize through continuous representations.
3. **Perplexity** — know the formula, know it is \(\exp(\text{cross-entropy})\), and know that lower is better. Be ready to say: "GPT-2 achieved ~30 PPL on WikiText-103 vs. ~60 for the best n-gram models."
4. **Vocabulary and tokenization** — interviewers may ask why modern LLMs use subword tokenizers (BPE) instead of word-level vocabularies. The answer is data sparsity — same root problem as n-grams.

---

## References

- Jurafsky & Martin, *Speech and Language Processing*, Ch. 3 — N-gram Language Models
- Shannon (1951), "Prediction and Entropy of Printed English"
- [Stanford CS 224N — Language Modeling lecture](https://web.stanford.edu/class/cs224n/)
