# XLNet: Generalized Autoregressive Pretraining

**Authors:** Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le  
**Year:** 2019 &nbsp;|&nbsp; **Venue:** NeurIPS  
**Link:** [arXiv:1906.08237](https://arxiv.org/abs/1906.08237)

---

## TL;DR

XLNet combines the best of **autoregressive** modeling (GPT) and **bidirectional** context (BERT) using **permutation language modeling**: tokens are predicted in a **random order**, so each position can attend to any subset of other positions according to the permutation. A **two-stream attention** mechanism separates content and query representations to prevent information leakage.

---

## Why This Paper Matters

XLNet clarifies the fundamental trade-off between causal and bidirectional factorizations. While it didn't become the default pre-training recipe (RoBERTa + scale won engineering mindshare), it introduced concepts that resurface in diffusion-based sequence models and non-autoregressive generation. It's a great paper for understanding **why** factorization order matters.

---

## Key Concepts Explained Simply

### The Problem with BERT's MLM

BERT masks tokens and predicts them **independently**. If you mask both "New" and "York" in "I visited [MASK] [MASK] last summer," BERT predicts "New" and "York" separately — it doesn't model the dependence between them. This is called the **independence assumption**, and it weakens the pre-training signal.

### Permutation Language Modeling

Instead of always predicting left-to-right, XLNet considers **all possible orderings** of tokens. For a 4-token sequence, there are 4! = 24 permutations. In permutation \(\pi = (3, 1, 4, 2)\), token 3 is predicted first (no context), then token 1 (seeing token 3), then token 4 (seeing tokens 3, 1), etc.

This way, **every token eventually gets to condition on every other token**, but through a proper autoregressive factorization — no independence assumption.

### Two-Stream Attention

A subtle problem: if token at position 3 is predicting itself, it shouldn't see its own content (that would be cheating). But it should know it's at position 3 (to use positional information). XLNet solves this with two representations:

- **Content stream (h):** Standard hidden state — knows what token is at this position
- **Query stream (g):** Knows the position and context from other tokens, but **not** the token at this position

---

## The Math — Explained Step by Step

### Permutation Language Modeling Objective

\[
\mathcal{L}_{\text{PLM}} = \mathbb{E}_{\pi \sim \mathcal{S}_n}\left[\sum_{t=1}^{n} \log P_\theta\big(x_{\pi(t)} \mid \mathbf{x}_{\pi(1:t-1)}\big)\right]
\]

**Breaking it down:**

1. \(\pi\): A random permutation of positions \(\{1, 2, \ldots, n\}\)
2. \(\pi(t)\): The position predicted at step \(t\) in this permutation
3. \(\mathbf{x}_{\pi(1:t-1)}\): All tokens at positions that come **before** position \(\pi(t)\) in this permutation
4. The expectation is over **random permutations** — each training step samples a different order

**Key insight:** Different permutations expose different factorizations of the same joint distribution. Across many permutations, every token sees every possible context combination.

### Two-Stream Equations

**Content stream** (standard self-attention, used for tokens already "revealed"):

\[
h_{\pi(t)}^{(l)} = \text{Attention}\big(Q = h_{\pi(t)}^{(l-1)}, \; KV = h_{\pi(1:t)}^{(l-1)}\big)
\]

**Query stream** (for the token being predicted — no self-content):

\[
g_{\pi(t)}^{(l)} = \text{Attention}\big(Q = g_{\pi(t)}^{(l-1)}, \; KV = h_{\pi(1:t-1)}^{(l-1)}\big)
\]

The query stream uses the position embedding but not the token embedding at position \(\pi(t)\).

---

## Python Implementation

```python
import numpy as np
from itertools import permutations


def stable_softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def sample_permutation(n):
    """Sample a random permutation of [0, 1, ..., n-1]."""
    perm = list(range(n))
    np.random.shuffle(perm)
    return perm


def permutation_attention_mask(perm):
    """
    Build attention mask for a given permutation.
    Position perm[t] can attend to positions perm[0], ..., perm[t-1].
    """
    n = len(perm)
    mask = np.zeros((n, n))
    for t, pos in enumerate(perm):
        for prev in perm[:t]:
            mask[pos][prev] = 1.0
        # Content stream: can also see itself
        mask[pos][pos] = 1.0
    return mask


def query_stream_mask(perm):
    """
    Query stream mask: same as content stream but WITHOUT self-attention.
    Position perm[t] can attend to perm[0], ..., perm[t-1] but NOT itself.
    """
    n = len(perm)
    mask = np.zeros((n, n))
    for t, pos in enumerate(perm):
        for prev in perm[:t]:
            mask[pos][prev] = 1.0
    return mask


def two_stream_attention(h, g, W_q, W_k, W_v, content_mask, query_mask):
    """
    h: content stream [n, d]
    g: query stream [n, d]
    Returns updated h and g.
    """
    d_k = h.shape[-1]

    # Content stream self-attention
    Q_h = h @ W_q
    K_h = h @ W_k
    V_h = h @ W_v
    scores_h = (Q_h @ K_h.T) / np.sqrt(d_k)
    scores_h = np.where(content_mask == 0, -1e9, scores_h)
    h_new = stable_softmax(scores_h) @ V_h

    # Query stream attention (uses content stream K, V from visible positions)
    Q_g = g @ W_q
    scores_g = (Q_g @ K_h.T) / np.sqrt(d_k)
    scores_g = np.where(query_mask == 0, -1e9, scores_g)
    g_new = stable_softmax(scores_g) @ V_h

    return h_new, g_new


def plm_log_prob(tokens, log_prob_fn, perm):
    """
    Compute total log probability under a permutation ordering.
    log_prob_fn(token, context_tokens) -> log probability
    """
    total = 0.0
    for t, idx in enumerate(perm):
        context = [tokens[perm[j]] for j in range(t)]
        total += log_prob_fn(tokens[idx], context)
    return total


def compare_factorizations(tokens, log_prob_fn, n_perms=10):
    """Show how different permutations give different factorizations."""
    results = []
    for _ in range(n_perms):
        perm = sample_permutation(len(tokens))
        lp = plm_log_prob(tokens, log_prob_fn, perm)
        results.append((perm, lp))
    return results


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model = 5, 16

    perm = sample_permutation(seq_len)
    print(f"Permutation: {perm}")
    print(f"Prediction order: {[f'pos {p}' for p in perm]}")

    c_mask = permutation_attention_mask(perm)
    q_mask = query_stream_mask(perm)

    print("\nContent mask (rows=query, cols=key):")
    print(c_mask.astype(int))
    print("\nQuery mask (no self-attention):")
    print(q_mask.astype(int))

    # Two-stream attention demo
    h = np.random.randn(seq_len, d_model)
    g = np.random.randn(seq_len, d_model)
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1

    h_new, g_new = two_stream_attention(h, g, W_q, W_k, W_v, c_mask, q_mask)
    print(f"\nContent stream output shape: {h_new.shape}")
    print(f"Query stream output shape: {g_new.shape}")
```

---

## Interview Importance

XLNet is less frequently asked directly but understanding it shows depth of knowledge about pre-training objectives and their trade-offs.

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: Explain permutation LM vs. standard left-to-right training.

**Answer:** Standard autoregressive LM always predicts left-to-right: \(P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdots\). Permutation LM samples random orderings, so \(x_3\) might be predicted first (no context), then \(x_1\) (given \(x_3\)), etc. Both are valid factorizations of the same joint distribution. The benefit: across many permutations, every token gets to condition on every possible subset of other tokens — achieving bidirectional context within an autoregressive framework.

### Q2: What problem does two-stream attention solve?

**Answer:** In standard attention, when predicting token at position \(t\), the model's query contains the token embedding itself — it would "see the answer." Two-stream attention separates:
- **Content stream:** Contains the actual token embedding (used when providing context to other positions)
- **Query stream:** Contains only the position embedding (used when this position is being predicted)

This prevents the model from trivially copying its own token while still allowing it to use positional information.

### Q3: Why did RoBERTa + scale often win over XLNet in practice?

**Answer:** Several practical factors:
1. XLNet is **more complex** to implement (two-stream attention, permutation sampling)
2. Training is **slower** per step due to the additional stream
3. RoBERTa showed that simply **fixing BERT's training recipe** (more data, dynamic masking, longer training, no NSP) achieved most of the gains with less complexity
4. The independence assumption in MLM, which XLNet targets, matters less with large models and data
5. Engineering teams prefer simpler architectures that are easier to debug and optimize

---

## Connections to Other Papers

- **BERT** → XLNet addresses BERT's masked token independence assumption
- **GPT-2** → XLNet extends GPT-style autoregressive training with permutations
- **RoBERTa** → Simpler alternative that achieved competitive results
- **Transformer-XL** → XLNet builds on its segment-level recurrence for long contexts

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Permutation language modeling — random factorization orders |
| Problem solved | BERT's independence assumption for masked tokens |
| Key mechanism | Two-stream attention (content vs. query) |
| Practical outcome | Competitive but complex; RoBERTa won mindshare |
| Interview value | Shows deep understanding of pre-training objectives |
