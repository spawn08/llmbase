# Word Embeddings

## Why This Matters for LLMs

Modern LLMs do not read text as a lookup table of unrelated symbols. They map every token into a **dense vector** so that computation can blend meaning, syntax, and shallow world knowledge inside linear algebra. Static word embeddings (Word2Vec, GloVe, FastText) were the first widely successful demonstration that **geometry can track semantics**: directions between vectors mirror analogies and co-occurrence statistics. That idea survived every architecture change: Transformer layers still operate on vectors in \(\mathbb{R}^d\), and analysis of hidden states still uses cosine similarity and linear probes.

Interviewers care about embeddings because they separate “I used BERT” from “I know why discrete NLP fails.” One-hot vectors have no notion of distance. Two different words are always orthogonal. Embeddings convert the **curse of dimensionality** in sparse counting into a **smooth optimization problem** where similar words share evidence. Every paper on instruction tuning, retrieval-augmented generation, or contrastive pretraining assumes you are comfortable with vector spaces and why cosine similarity appears everywhere.

Finally, embeddings are the historical bridge from **count-based n-grams** to **neural language models**. Word2Vec shows how prediction objectives on local windows induce global structure. GloVe shows how global co-occurrence matrices factor into the same space. FastText shows how **subwords** reduce out-of-vocabulary failures. Those three threads reappear inside byte-pair encoding tokenizers and sentence-piece models that feed billion-parameter models today.

---

## Core Concepts

### From One-Hot Vectors to Dense Embeddings

#### The Idea

A vocabulary of size \(V\) can represent each word type as a vector \(\mathbf{e}_w \in \mathbb{R}^V\) with a single \(1\) in position \(w\) and zeros elsewhere. That is the **one-hot** encoding. It is honest about discreteness but useless for similarity: every pair of distinct words has dot product zero, so **no gradient** pushes “cat” toward “dog” when the model sees them in interchangeable syntactic slots. A **word embedding** replaces the sparse vector with \(\mathbf{v}_w \in \mathbb{R}^d\) where \(d \ll V\). Similar words end up near each other because the training objective forces the model to share statistical evidence across neighbors in meaning or distribution.

#### The Math (One-Hot Orthogonality)

Let \(\mathbf{u}_a\) and \(\mathbf{u}_b\) be one-hot vectors for distinct words \(a\) and \(b\). Then:

\[
\mathbf{u}_a^\top \mathbf{u}_b = 0,
\qquad
\|\mathbf{u}_a\|_2 = \|\mathbf{u}_b\|_2 = 1.
\]

The **cosine similarity** \(\cos(\mathbf{u}_a, \mathbf{u}_b) = \frac{\mathbf{u}_a^\top \mathbf{u}_b}{\|\mathbf{u}_a\|\|\mathbf{u}_b\|}\) is zero for any distinct pair.

!!! math-intuition "In Plain English"
    The dot product counts coordinate-wise overlap. Two different one-hot vectors never share a nonzero coordinate, so the dot product is zero. Geometry cannot reflect “cat and dog are both nouns” because every word lives on a different corner of a simplex, equidistant from all unrelated corners.

### Why Geometry Encodes Meaning

Co-occurrence statistics say: words that appear in similar contexts behave like synonyms, antonyms in contrastive frames, or related entities. A model forced to predict missing words from neighbors must place words that share contexts near one another in vector space so that the same weight matrices can service many tokens. **Linear structure** emerges because many linguistic relations look like offsets: gender, tense, and capital-country pairs often appear as nearly parallel vector differences. That is not mysticism; it is a consequence of bilinear objectives and shared hidden layers compressing PMI-like statistics into low rank.

!!! example "Worked Example: Cosine Similarity in Two Dimensions (Illustrative)"
    Let \(\mathbf{v}_{\text{cat}} = (1.0, 0.2)\) and \(\mathbf{v}_{\text{dog}} = (0.9, 0.3)\).  
    Dot product: \(1.0 \cdot 0.9 + 0.2 \cdot 0.3 = 0.96\).  
    Norms: \(\|\mathbf{v}_{\text{cat}}\| = \sqrt{1 + 0.04} \approx 1.02\), \(\|\mathbf{v}_{\text{dog}}\| = \sqrt{0.81 + 0.09} \approx 0.95\).  
    Cosine: \(0.96 / (1.02 \cdot 0.95) \approx 0.99\).  
    Compare \(\mathbf{v}_{\text{rocket}} = (0.1, 1.0)\): dot with \(\mathbf{v}_{\text{cat}}\) is \(1.0 \cdot 0.1 + 0.2 \cdot 1.0 = 0.3\), norms product \(\approx 1.02 \cdot 1.005\), cosine \(\approx 0.29\).  
    The small toy vectors separate **domestic animals** from **vehicles** by angle, which is how evaluators summarize embedding quality at scale.

### Word2Vec Skip-Gram

#### Objective

**Skip-gram** maximizes the sum of log probabilities of **context** words given a **center** word. For corpus length \(T\), window radius \(c\):

\[
\mathcal{L} = \sum_{t=1}^{T} \sum_{\substack{-c \le j \le c \\ j \ne 0}} \log P(w_{t+j} \mid w_t).
\]

The conditional is a softmax over the vocabulary using an **input** vector \(\mathbf{v}_{w}\) for \(w_t\) and an **output** vector \(\mathbf{v}'_{w}\) for each possible output word:

\[
P(w_O \mid w_I) = \frac{\exp({\mathbf{v}'_{w_O}}^\top \mathbf{v}_{w_I})}{\sum_{w \in \mathcal{V}} \exp({\mathbf{v}'_{w}}^\top \mathbf{v}_{w_I})}.
\]

!!! math-intuition "In Plain English"
    For each position \(t\), the center word \(w_t\) is the conditioning event. Words within \(\pm c\) positions are targets. The model must raise the score \({\mathbf{v}'_{w_{t+j}}}^\top \mathbf{v}_{w_t}\) for true context words relative to every other word in the vocabulary. The softmax turns scores into a probability vector that sums to \(1\).

!!! example "Worked Example: Skip-Gram Step with Concrete Numbers"
    **Vocabulary size \(V = 3\)** with words `king`, `queen`, `child`.  
    **Center word:** `king` with input embedding \(\mathbf{v}_{\text{king}} = (1.0, 0.0)\).  
    **Output embeddings** as rows of a \(3 \times 2\) matrix:
    \(\mathbf{v}'_{\text{king}} = (0.5, 0.1)\), \(\mathbf{v}'_{\text{queen}} = (0.4, 0.2)\), \(\mathbf{v}'_{\text{child}} = (0.1, 0.0)\).
    **Scores** \(s_i = {\mathbf{v}'_{i}}^\top \mathbf{v}_{\text{king}}\):  
    \(s_{\text{king}} = 0.5\), \(s_{\text{queen}} = 0.4\), \(s_{\text{child}} = 0.1\).
    **Softmax denominators:** \(\exp(0.5) + \exp(0.4) + \exp(0.1) \approx 1.649 + 1.492 + 1.105 = 4.246\).  
    \(P(\text{queen} \mid \text{king}) \approx 1.492 / 4.246 \approx 0.35\).  
    If the true context word is `queen`, the contribution to \(\mathcal{L}\) is \(\log 0.35 \approx -1.05\).  
    **Gradient intuition (conceptual):** optimization increases \(s_{\text{queen}}\) by nudging \(\mathbf{v}_{\text{king}}\) to align better with \(\mathbf{v}'_{\text{queen}}\) and by nudging all \(\mathbf{v}'_{w}\) for incorrect words downward relative to the numerator. You do not need the full Jacobian on paper; you need the story: **pull true pair together, push negatives apart under the softmax pressure**.

### Negative Sampling

Full softmax costs \(O(V)\) per training step. **Negative sampling** replaces the full denominator with a small set of **negative draws** from a noise distribution \(P_n(w)\), typically proportional to unigram frequency raised to the \(\frac{3}{4}\) power. For a true pair \((w_I, w_O)\) and \(k\) negative words \(w_{i}^{\text{neg}}\), the binary logistic objective resembles:

\[
\log \sigma({\mathbf{v}'_{w_O}}^\top \mathbf{v}_{w_I})
+ \sum_{i=1}^{k} \mathbb{E}_{w \sim P_n}\bigl[\log \sigma(-{\mathbf{v}'_{w}}^\top \mathbf{v}_{w_I})\bigr].
\]

!!! math-intuition "In Plain English"
    Instead of comparing the true context word against all \(V\) words simultaneously, the model learns by **contrasting** the true word against a handful of impostors sampled from noise. Sigmoid maps the score to \((0,1)\) so the objective looks like logistic regression: predict “is this the true context word?” The sum over negatives says: low scores are good for junk words.

!!! example "Worked Example: Negative Sampling Probability Arithmetic"
    Suppose \(k = 2\) and the sampled negatives are `child` and `table` for center `king`, with true context `queen`. Write scores \(s = {\mathbf{v}'_{w}}^\top \mathbf{v}_{\text{king}}\). Use \(\sigma(z) = 1 / (1 + \exp(-z))\).
    **Positive score** \(s_{\text{queen}} = 2.0\): \(\log \sigma(2.0) \approx \log(0.881) \approx -0.127\).
    **Negative `child`** with \(s_{\text{child}} = -1.0\): the training term is \(\log \sigma(-s_{\text{child}}) = \log \sigma(1.0) \approx \log(0.731) \approx -0.314\).
    **Negative `table`** with \(s_{\text{table}} = 0.5\): \(\log \sigma(-0.5) \approx \log(0.378) \approx -0.975\).
    **Sum of the three log terms:** \(-0.127 + (-0.314) + (-0.975) = -1.416\). Stochastic gradient steps **increase** this sum by raising \(s_{\text{queen}}\) while pushing \(s_{\text{child}}\) and \(s_{\text{table}}\) downward so that impostor scores stay below zero after the negative sign inside the sigmoid.

### GloVe and Co-occurrence Structure

GloVe constructs a global word–word co-occurrence matrix \(X\) where \(X_{ij}\) counts how often word \(j\) appears in the context window of word \(i\). The model learns vectors \(\mathbf{w}_i\) and \(\tilde{\mathbf{w}}_j\) plus biases \(b_i\) and \(\tilde{b}_j\) to satisfy:

\[
\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j \approx \log X_{ij}.
\]

Ratios like \(P(j \mid i) / P(k \mid i)\) encode meaning because they cancel generic frequency effects.

!!! math-intuition "In Plain English"
    The log co-occurrence \(\log X_{ij}\) grows when words co-occur often. The bilinear term \(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j\) tries to match that log count in a low-dimensional inner product space. Biases absorb words that are frequent everywhere.

!!! example "Worked Example: Five-Word Vocabulary Co-occurrence Counts"
    **Vocabulary in order:** `river`, `bank`, `money`, `water`, `loan`.  
    **Window size 1** (immediate neighbors only) on the synthetic corpus:
    - `river bank water`
    - `bank money loan`
    - `river water bank`
    **Symmetric count** \(X_{ij}\): count \(i\) as focus and \(j\) as context, then add the symmetric focus-context swap if you use a symmetric definition. Here we accumulate ordered pairs within each sentence twice if both directions matter; for simplicity, count **unordered co-occurrence** within the window once per ordered pair \((i,j)\) where \(i\) is left of \(j\) or right of \(i\) within distance 1.

    Pairs:
    - Sentence 1: `(river,bank)`, `(bank,water)`
    - Sentence 2: `(bank,money)`, `(money,loan)`
    - Sentence 3: `(river,water)`, `(water,bank)`

    Aggregate unique unordered pairs with counts:
    `(river,bank):1`, `(bank,water):2` (from sentences 1 and 3), `(bank,money):1`, `(money,loan):1`, `(river,water):2`.

    Row `bank` touches `river`, `water`, `money` with total co-occurrence mass \(1 + 2 + 1 = 4\) across distinct partners when you sum \(X_{\text{bank},j}\) for \(j \ne \text{bank}\) in this toy (self-loops omitted). GloVe would regress \(\log X_{ij}\) for each nonzero cell while weighting rare pairs differently in the full algorithm.

### FastText and Subwords

FastText represents word \(w\) as the sum of embeddings of its character \(n\)-grams. For word `where` with boundary symbols `<` and `>`, 3-grams include `<wh`, `whe`, `her`, `ere`, `re>`.

!!! math-intuition "In Plain English"
    Rare words share character sequences with frequent words, so gradients update subword vectors even when the full word is uncommon. At test time, an unknown word is composed from its \(n\)-grams, giving a non-random vector instead of a generic UNK bucket.

!!! example "Worked Example: Decomposing `running`"
    Use \(n = 3\) with `<` and `>` boundaries: `<ru`, `run`, `unn`, `nni`, `ing`, `ng>`.  
    Suppose each 3-gram has a 2-D vector; FastText sets
    \(\mathbf{v}_{\text{running}} = \sum_{\text{g} \in \text{grams}} \mathbf{z}_g\).
    If `run` and `running` share `run`, `unn`, their sum overlaps partially, pulling their full-word vectors closer than unrelated strings like `rocket`.

### Contextual Versus Static Embeddings

**Static** embeddings assign one vector per word type. **Contextual** models (ELMo, BERT, GPT) produce a vector that depends on the full sentence: the representation of `bank` differs in “river bank” versus “investment bank.”

| Property | Static (Word2Vec, GloVe, FastText) | Contextual (Transformer hidden states) |
|----------|-------------------------------------|----------------------------------------|
| Polysemy | Collapsed into one vector | Disambiguated per position |
| Precompute | Single matrix lookup | Forward pass required |
| Downstream use | Feature input | Fine-tune or prompt |

!!! math-intuition "In Plain English"
    Static embeddings are a **dictionary**. Contextual embeddings are a **function** \(f(\text{sentence}, \text{position})\). Modern LLMs are almost entirely contextual; static methods remain relevant for data efficiency, retrieval indexes, and understanding history.

---

## Deep Dive: PMI, Noise Contrastive Estimation, and Modern Retrieval

??? deep-dive "Deep Dive: From Skip-Gram to Contrastive Learning"
    Skip-gram with negative sampling is related to **noise contrastive estimation**: discriminate true context draws from a known noise distribution. Pointwise mutual information \(\text{PMI}(w,c) = \log \frac{P(w,c)}{P(w)P(c)}\) connects co-occurrence ratios to log-odds. Modern sentence embeddings and RAG systems reuse the same contrastive shape: pull aligned pairs, push mismatched pairs, avoid full-softmax denominators at scale.

---

## Code

The script below trains skip-gram Word2Vec with `gensim`, inspects neighbors and analogies, and projects vectors with t-SNE. Inline comments connect parameters to the math above.

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
    vector_size=50,   # embedding dimension d in R^d
    window=3,         # context radius c in the skip-gram sum
    min_count=1,      # include all words (toy corpus only)
    sg=1,             # 1 = skip-gram (predict context from center)
    epochs=200,       # multiple passes because the corpus is tiny
    seed=42,
)

# ── Inspect similarities ───────────────────────────────────────────────
print("Most similar to 'king':")
for word, score in model.wv.most_similar("king", topn=5):
    # cosine similarity between learned vectors
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

## Interview Guide

### What Interviewers Actually Ask

!!! interview "FAANG-Level Questions"
    1. **Why are one-hot vectors inadequate for neural NLP?**  
       *Depth:* Orthogonality, no similarity, parameter explosion if used as inputs to large layers without compression.
    2. **Explain skip-gram versus CBOW in one minute.**  
       *Depth:* Skip-gram predicts context from center (better for rare words); CBOW predicts center from averaged context (faster, slightly worse on rares).
    3. **What problem does negative sampling solve?**  
       *Depth:* Full softmax is \(O(V)\) per update; negatives reduce to \(O(k)\) logistic contrasts.
    4. **How does GloVe differ philosophically from Word2Vec?**  
       *Depth:* Global matrix factorization of log co-occurrences versus local window prediction; both learn geometric structure.
    5. **What is the out-of-vocabulary problem and how does FastText mitigate it?**  
       *Depth:* Unknown words have no row in a word-level table; character \(n\)-grams compose a vector from subword units.
    6. **Define cosine similarity and say when you prefer it over Euclidean distance.**  
       *Depth:* Normalize by magnitude; useful when length correlates with frequency rather than semantics.
    7. **How does polysemy challenge static embeddings?**  
       *Depth:* Single vector averages all senses; contextual models disambiguate per token position.
    8. **Describe embedding arithmetic limitations.**  
       *Depth:* Analogies fail for symmetric or multi-relation words; small corpora give noisy directions.

!!! interview "Follow-up Probes"
    1. Why might you raise unigram frequencies to the \(\frac{3}{4}\) power when sampling negatives?
    2. What does hierarchical softmax achieve in complexity terms?
    3. When would you still use static embeddings in a Transformer pipeline?
    4. How does tokenization interact with FastText’s original word-boundary \(n\)-grams?
    5. Why can t-SNE distances mislead qualitative interpretation?

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Skip-gram maximizes \(\sum \log P(\text{context} \mid \text{center})\)”
    - “Noise contrastive estimation with sampled negatives”
    - “Global log co-occurrence matrix factorization (GloVe)”
    - “Subword composition for OOV robustness”
    - “Cosine similarity on \(\ell_2\)-normalized vectors”

---

## References

- Mikolov et al. (2013), *Efficient Estimation of Word Representations in Vector Space*
- Pennington et al. (2014), *GloVe: Global Vectors for Word Representation*
- Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information*
- [Stanford CS224N — Word Vectors lecture](https://web.stanford.edu/class/cs224n/)
