# 1.2 — Word Embeddings

## Intuition

Words are discrete symbols — a vocabulary of 50,000 tokens gives no notion of similarity between "king" and "queen." **Word embeddings** map each token to a dense, low-dimensional vector where geometric proximity reflects semantic relatedness. This single idea — *meaning lives in continuous space* — unlocked nearly every advance from RNNs to Transformers.

---

## Core concepts

### One-hot encoding (the baseline)

A vocabulary of size \(V\) represents each word as a \(V\)-dimensional binary vector with a single 1. Problems:

- Vectors are **orthogonal** — \(\text{cos}(\mathbf{v}_{\text{cat}}, \mathbf{v}_{\text{dog}}) = 0\).
- Dimension grows with vocabulary.
- No generalization: seeing "cat sat" teaches nothing about "dog sat."

### Word2Vec

Mikolov et al. (2013) proposed two shallow architectures to learn dense embeddings from co-occurrence:

**Skip-gram:** Predict context words from the center word.

\[
\max \sum_{t=1}^{T} \sum_{-c \le j \le c, \, j \ne 0} \log P(w_{t+j} \mid w_t)
\]

where the conditional is a softmax over the vocabulary:

\[
P(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}
\]

**CBOW (Continuous Bag of Words):** Predict the center word from the sum of context word vectors. Faster to train, slightly lower quality on rare words.

**Training tricks** that matter at scale:

- **Negative sampling** — instead of computing the full softmax, sample \(k\) negative examples and use a binary logistic objective.
- **Subsampling** of frequent words — down-weight "the", "of", etc.
- **Hierarchical softmax** — organize vocabulary in a binary tree for \(O(\log V)\) output.

### GloVe (Global Vectors)

Pennington et al. (2014) showed that the ratio of co-occurrence probabilities encodes meaning:

\[
\frac{P(\text{ice} \mid \text{solid})}{P(\text{ice} \mid \text{gas})} \gg 1
\]

GloVe factorizes the log co-occurrence matrix:

\[
\mathbf{w}_i^T \mathbf{\tilde{w}}_j + b_i + \tilde{b}_j = \log X_{ij}
\]

where \(X_{ij}\) is the count of word \(j\) appearing in the context of word \(i\). It combines the global statistics of matrix factorization with the efficiency of local context windows.

### FastText

Bojanowski et al. (2017) extended Word2Vec by representing each word as a **bag of character n-grams**. The embedding for "where" includes vectors for `<wh`, `whe`, `her`, `ere`, `re>`. Benefits:

- Handles **out-of-vocabulary (OOV)** words by composing subword vectors.
- Captures **morphology** — "unhappy" shares n-grams with "happy."
- Better for morphologically rich languages.

### Embedding arithmetic

Famous analogy result:

\[
\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}
\]

This works because the embedding space learns relational offsets. In practice, accuracy on analogy tasks varies (60–80%), but the principle that **directions encode relationships** is important for understanding why Transformer representations work.

---

## Code — Training Word2Vec and visualizing embeddings

```python
"""
Train Word2Vec with gensim, inspect analogies, and visualize with t-SNE.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# gensim's Word2Vec handles data loading, training, and analogy tests
from gensim.models import Word2Vec

# ── Toy corpus ──────────────────────────────────────────────────────────
sentences = [
    ["the", "king", "ruled", "the", "kingdom"],
    ["the", "queen", "ruled", "the", "kingdom"],
    ["the", "prince", "will", "be", "king"],
    ["the", "princess", "will", "be", "queen"],
    ["man", "and", "woman", "are", "humans"],
    ["boy", "and", "girl", "are", "children"],
    ["king", "is", "a", "man", "who", "rules"],
    ["queen", "is", "a", "woman", "who", "rules"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "cat", "sat", "on", "the", "mat"],
]

# ── Train Skip-gram model ──────────────────────────────────────────────
model = Word2Vec(
    sentences,
    vector_size=50,   # embedding dimension
    window=3,         # context window size
    min_count=1,      # include all words (toy corpus)
    sg=1,             # 1 = skip-gram, 0 = CBOW
    epochs=200,       # more epochs for tiny data
    seed=42,
)

# ── Inspect similarities ───────────────────────────────────────────────
print("Most similar to 'king':")
for word, score in model.wv.most_similar("king", topn=5):
    print(f"  {word:12s} {score:.4f}")

# ── Analogy: king - man + woman ≈ queen ────────────────────────────────
try:
    result = model.wv.most_similar(
        positive=["king", "woman"], negative=["man"], topn=3
    )
    print("\nking - man + woman →")
    for word, score in result:
        print(f"  {word:12s} {score:.4f}")
except KeyError as e:
    print(f"Analogy skipped (word not in vocab): {e}")

# ── t-SNE 2D visualization ─────────────────────────────────────────────
words = list(model.wv.key_to_index.keys())
vectors = np.array([model.wv[w] for w in words])

tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(words) - 1))
coords = tsne.fit_transform(vectors)

plt.figure(figsize=(10, 7))
plt.scatter(coords[:, 0], coords[:, 1], s=40, alpha=0.7)
for i, word in enumerate(words):
    plt.annotate(
        word,
        (coords[i, 0], coords[i, 1]),
        fontsize=9,
        ha="center",
        va="bottom",
    )
plt.title("Word2Vec Embeddings — t-SNE Projection")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150)
plt.show()
print("Saved: word2vec_tsne.png")
```

---

## Interview takeaways

1. **Why embeddings matter** — they convert the discrete sparsity problem into a continuous optimization problem. This is the bridge from n-grams (Part 1.1) to neural LMs (Part 1.3).
2. **Skip-gram vs. CBOW** — skip-gram works better for rare words; CBOW is faster. Know which objective maximizes what.
3. **Negative sampling** — be ready to explain why computing the full softmax is expensive (\(O(V)\) per word) and how negative sampling reduces it to \(O(k)\).
4. **GloVe vs. Word2Vec** — GloVe is count-based (global matrix); Word2Vec is prediction-based (local window). In practice, quality is comparable; the design philosophy differs.
5. **FastText for OOV** — subword n-grams let the model handle unseen words, which is the same insight behind BPE tokenization used in all modern LLMs.
6. **Contextual vs. static** — Word2Vec/GloVe give one vector per word type. BERT/GPT give one vector per word *token in context*. Know the distinction and why it matters (polysemy: "bank" of a river vs. "bank" for money).

---

## References

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*
- Pennington et al. (2014), *GloVe: Global Vectors for Word Representation*
- Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information*
- [Stanford CS 224N — Word Vectors lecture](https://web.stanford.edu/class/cs224n/)
