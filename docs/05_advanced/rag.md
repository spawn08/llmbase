# Retrieval-Augmented Generation

## Why This Matters for LLMs

Parametric language models store knowledge in weights and can **confidently hallucinate** when facts are rare, stale, or outside the training distribution. Retrieval-Augmented Generation (RAG) grounds generation in **external evidence**: documents, APIs, or structured stores retrieved at query time. For engineers, RAG is the default pattern for enterprise assistants, support bots, and research copilots where **verifiability** and **freshness** matter as much as fluency.

Second, RAG decomposes the problem into **retrieval quality** and **generation quality**. You can iterate on chunking strategies, embedding models, hybrid search, and re-ranking without retraining a 70B-parameter backbone. Interviews often probe whether you understand failure modes: irrelevant chunks, duplicate context, context-length limits, and “faithful but wrong” answers when the model ignores retrieved text.

Third, RAG sits at the intersection of **NLP**, **information retrieval**, and **systems engineering**. Vector databases, batch indexing pipelines, latency budgets, and evaluation harnesses (faithfulness, citation accuracy) are first-class concerns. A strong answer ties dense retrieval (embeddings) to classical IR (BM25), explains bi-encoder scoring, and names production trade-offs between approximate nearest neighbors and exact search.

---

## Core Concepts

### The RAG Loop: Retrieve → Augment → Generate

A minimal RAG pipeline has three stages:

1. **Retrieve**: Given a query \(q\), fetch a set of candidate passages \(\mathcal{D}(q) = \{d_1,\ldots,d_k\}\) from a corpus or index.
2. **Augment**: Pack \(\mathcal{D}(q)\) into a **prompt** (often with instructions, citations, and ordering heuristics) subject to a context budget \(L_{\max}\) tokens.
3. **Generate**: Sample an answer \(a\) from the LM conditioned on the augmented prompt.

Formally, if \(P_\theta\) is the LM and \(C\) is the constructed context string,

\[
a \sim P_\theta(\cdot \mid [C; q])
\]

where \([C; q]\) denotes token concatenation with delimiters. The retrieval module is often **non-differentiable** in production (API calls, BM25), though training with contrastive objectives can align retrievers and readers.

!!! math-intuition "In Plain English"
    Think of RAG as **open-book exam** mode: the model reads a few allowed pages before answering. If the right page is missing, the model may still sound authoritative—so **retrieval recall** is often the bottleneck, not decoding perplexity.

### Indexing Pipeline: Chunking

Documents are split into **chunks** \(c_i\) to match embedding model limits and improve retrieval granularity. Common strategies:

- **Fixed token windows** with overlap (e.g., 512 tokens, 64-token stride) to avoid cutting entities across boundaries.
- **Semantic chunking** (sentence/paragraph splits) with a maximum size cap.
- **Structure-aware** splits for Markdown, HTML, or PDF sections.

Let chunk \(c\) have token length \(|c|\). The indexer maps each chunk to a record \((\text{id}, \text{text}, \text{metadata})\). Metadata (source URL, timestamp, ACL) enables filtering **before** vector search.

### Embedding and the Vector Store

A **bi-encoder** maps queries and documents to vectors in \(\mathbb{R}^d\):

\[
\mathbf{e}_q = f_\phi(q),\quad \mathbf{e}_c = g_\phi(c)
\]

In symmetric setups \(f_\phi = g_\phi\). Retrieval scores dense candidates by **cosine similarity** or inner product:

\[
s_{\text{dense}}(q, c) = \frac{\langle \mathbf{e}_q, \mathbf{e}_c \rangle}{\|\mathbf{e}_q\|\|\mathbf{e}_c\|}
\]

The vector store persists \(\{\mathbf{e}_{c_i}\}\) with identifiers pointing back to raw text. Approximate nearest neighbor (ANN) libraries (FAISS, HNSW graphs in hosted DBs) trade exact ranking for latency at billion-scale corpora.

!!! math-intuition "In Plain English"
    Dense retrieval is **semantic matching**: “CEO of OpenAI” can match a passage that never uses that exact phrase. It can also retrieve **plausible but wrong** neighbors if embeddings confuse related entities—hybrid search mitigates this.

### Sparse Retrieval: BM25

**BM25** ranks lexical overlap between query \(q\) and document \(d\). A standard form for term \(t\):

\[
\text{BM25}(d,q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\cdot (k_1+1)}{f(t,d) + k_1\left(1 - b + b\frac{|d|}{\text{avgdl}}\right)}
\]

where \(f(t,d)\) is term frequency in \(d\), \(|d|\) is document length, \(\text{avgdl}\) is average length, and \(k_1,b\) tune saturation and length normalization.

!!! math-intuition "In Plain English"
    BM25 is **keyword glue**: it loves rare discriminative tokens (IDs, SKUs, legal citations) that embeddings sometimes smear. Hybrid pipelines combine BM25 recall with dense semantic recall.

### Hybrid Retrieval

A common fusion is **linear combination** or **Reciprocal Rank Fusion (RRF)**. For ranks \(r_{\text{dense}}, r_{\text{sparse}}\), a simple RRF score for document \(d\) is:

\[
\text{RRF}(d) = \sum_{i \in \{\text{dense},\text{sparse}\}} \frac{1}{k + r_i(d)}
\]

with small constant \(k\) (e.g., 60). Hybrid search reduces single-method blind spots.

### Augmentation: Context Packing and Citations

Given retrieved chunks, the augmenter must:

- **Deduplicate** near-duplicate passages.
- **Order** by score or reverse-chronological for news.
- **Truncate** to fit \(L_{\max}\), often reserving budget for system instructions and user query.
- **Cite**: attach `[n]` markers mapping spans to source IDs for user-facing verification.

Instruction-following prompts often demand: “Answer using only the context; if unknown, say you do not know.” That reduces—but does not eliminate—**unfaithful** generations.

### Vector Databases and Libraries

| System | Notes |
|--------|-------|
| **FAISS** | Facebook AI Similarity Search: GPU/CPU ANN, IVF, PQ, HNSW; common in research and self-hosted stacks |
| **Pinecone** | Managed vectors, metadata filters, production SLAs |
| **Weaviate** | GraphQL API, hybrid search, multi-tenancy patterns |
| **Chroma** | Lightweight embedded/local-first developer experience |

Engineering choices include **freshness** (reindex cadence), **sharding**, **embedding version pinning**, and **ACL-aware** retrieval (tenant isolation).

### Evaluation: Faithfulness, Relevance, Answer Quality

- **Retrieval**: nDCG@k, MRR@k, recall@k on labeled (q, relevant doc) sets.
- **Generation**: **Faithfulness** (supported by context), **relevance** to the question, **correctness** vs gold answers.
- **RAGAS-style** automated metrics (learned judge models) are popular but should be spot-checked with human eval.

Human rubrics often score 1–5 on: **groundedness**, **completeness**, **citation accuracy**, **harmlessness**.

### Dense Passage Retrieval Training (DPR-Style)

In supervised open-domain QA, we are given \((q, p^+, p^-_1,\ldots,p^-_m)\) where \(p^+\) is a gold passage and negatives are mined from the corpus. A **InfoNCE** / softmax contrastive loss over in-batch negatives trains encoders \(E_Q, E_P\):

\[
\mathcal{L} = - \log \frac{\exp\bigl(\langle E_Q(q), E_P(p^+) \rangle / \tau\bigr)}{\sum_{j} \exp\bigl(\langle E_Q(q), E_P(p_j) \rangle / \tau\bigr)}
\]

Temperature \(\tau\) scales logits; in-batch negatives reuse other \(p_j\) in the mini-batch for efficiency. Hard negatives (BM25 top documents without the answer) sharpen decision boundaries.

!!! math-intuition "In Plain English"
    DPR **pulls** the query vector toward the true passage and **pushes** it away from distractors. At inference, you never enumerate all passages with a cross-attention transformer—only a dot product in embedding space—so training must align that geometry with user questions.

### Re-Ranking: Cross-Encoder vs Bi-Encoder

A **cross-encoder** concatenates \([q; p]\) and runs a Transformer to emit a relevance score. It is more accurate but \(O(|q|+|p|)\) per pair—use on **top 50–200** BM25/dense candidates. Bi-encoders embed \(q\) and \(p\) independently (**linear in corpus size** after indexing). Production stacks: **retrieve wide** (high recall), **re-rank narrow** (high precision).

### Latency and Freshness

End-to-end latency is roughly:

\[
T_{\text{total}} \approx T_{\text{embed}(q)} + T_{\text{ANN}} + T_{\text{rerank}} + T_{\text{LLM}}
\]

Streaming answers while retrieving (speculative) complicates tracing; most systems **block** on retrieval for auditability. Freshness requires **incremental indexing** or **change-data-capture** from source systems.

### ANN: IVF, PQ, and HNSW (Conceptual)

**Product quantization (PQ)** compresses vectors into subcodes to reduce memory and distance computation at the cost of approximation. **Inverted file (IVF)** clusters vectors into coarse cells; at query time only a subset of cells is probed. **HNSW** builds a multi-layer navigable small-world graph for greedy search. Complexity is often sublinear in \(N\) for fixed recall targets, but **recall–latency** curves must be measured on your corpus.

!!! math-intuition "In Plain English"
    ANN is **deliberately sloppy** nearest neighbors: you accept a 1–2% miss rate on the true top-1 to cut latency 10×. For RAG, tune recall@k on **held-out** QA pairs from your domain—not on random nearest-neighbor benchmarks.

### Query Reformulation and HyDE

**Hypothetical Document Embeddings (HyDE)** ask the LM to invent a **fake answer passage**, embed it, and retrieve with that vector—sometimes improving recall when short queries omit jargon. Risks include hallucinated details steering retrieval; use **guardrails** and **score thresholds**.

### Multi-Vector and ColBERT-Style Retrieval

**ColBERT** late-interaction models store multiple token vectors per document and score with MaxSim operations—more expressive than one vector per document, heavier storage. Trade-offs matter at billion-token scale: **single-vector** bi-encoders dominate latency-sensitive stacks; **late interaction** wins on hard semantic matching benchmarks.

### End-to-End RAG: Reference Python

The following script sketches chunking, synthetic “embedding” via hash features for illustration, cosine top-k, and prompt construction. Swap `fake_embed` for `sentence_transformers` or an API in production.

```python
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|\S", text.lower())


def chunk_text(text: str, max_tokens: int = 64, stride: int = 48) -> List[str]:
    tokens = text.split()
    if not tokens:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        piece = tokens[start : start + max_tokens]
        chunks.append(" ".join(piece))
        if start + max_tokens >= len(tokens):
            break
        start += stride
    return chunks


def stable_hash_vector(text: str, dim: int = 128) -> List[float]:
    """Deterministic bag-of-hashes embedding for demo only — use real encoders in prod."""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0 if (h >> 8) % 2 == 0 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def dense_top_k(
    query: str, corpus_chunks: Sequence[str], k: int = 3
) -> List[Tuple[int, float]]:
    q = stable_hash_vector(query)
    scores: List[Tuple[int, float]] = []
    for i, chunk in enumerate(corpus_chunks):
        c = stable_hash_vector(chunk)
        scores.append((i, cosine_sim(q, c)))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores[:k]


def bm25_scores(query: str, corpus: Sequence[str], k1: float = 1.5, b: float = 0.75):
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    doc_lens = [len(toks) for toks in tokenized_corpus]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
    df = {}
    for toks in tokenized_corpus:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    N = len(corpus)
    q_terms = tokenize(query)
    scores = []
    for doc_idx, toks in enumerate(tokenized_corpus):
        tf = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = doc_lens[doc_idx]
        s = 0.0
        for t in q_terms:
            if t not in tf:
                continue
            idf = math.log(1 + (N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            num = tf[t] * (k1 + 1)
            den = tf[t] + k1 * (1 - b + b * (dl / avgdl))
            s += idf * num / den
        scores.append((doc_idx, s))
    scores.sort(key=lambda t: t[1], reverse=True)
    return scores


def build_rag_prompt(query: str, contexts: Sequence[str]) -> str:
    blocks = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        "You are a careful assistant. Use ONLY the numbered passages.\n"
        f"Passages:\n{blocks}\n\nQuestion: {query}\n"
        "Answer with citations like [1] where applicable."
    )


@dataclass
class RAGDemo:
    chunks: List[str]

    @classmethod
    def from_documents(cls, docs: Iterable[str]) -> "RAGDemo":
        chunks: List[str] = []
        for d in docs:
            chunks.extend(chunk_text(d))
        return cls(chunks=chunks)

    def answer_plan(self, query: str, k: int = 2) -> Tuple[List[str], str]:
        dense_hits = dense_top_k(query, self.chunks, k=k)
        bm25_hits = bm25_scores(query, self.chunks)
        fused = {i for i, _ in dense_hits} | {i for i, _ in bm25_hits[:k]}
        ordered = [self.chunks[i] for i in sorted(fused)]
        prompt = build_rag_prompt(query, ordered[:k])
        return ordered[:k], prompt


if __name__ == "__main__":
    docs = [
        "Paris is the capital of France. The Eiffel Tower is in Paris.",
        "Berlin is the capital of Germany. Brandenburg Gate is a landmark.",
        "France uses the euro. Germany uses the euro.",
    ]
    rag = RAGDemo.from_documents(docs)
    q = "What is the capital of Germany?"
    contexts, prompt = rag.answer_plan(q)
    print("Contexts:\n", contexts)
    print("\nPrompt tail:\n", prompt[-400:])
```

### Metadata Filters and ACL-Aware Retrieval

Enterprise corpora carry **tenant IDs**, **classification labels**, and **expiry dates**. Prefer **structured pre-filtering** (SQL/JSON predicates) **before** ANN search when possible—smaller candidate sets improve precision and **latency**. Pure vector stores often **over-fetch** \(k' \gg k\) then **post-filter** on metadata, accepting wasted ANN work as a trade-off for simpler pipelines.

### Query Expansion and Multi-Query Retrieval

**Query expansion** adds synonyms or LLM-generated paraphrases; **multi-query** retrieval embeds several variants and **fuses** results (union + RRF). Both improve **recall** when user phrasing diverges from document phrasing, at the cost of **latency** and **noise** (more irrelevant passages).

### Production Checklist (RAG)

- **Version** embedding models and chunking code with each index build.
- **Monitor** empty retrieval, duplicate chunks, and **stale** sources.
- **Cap** retrieved tokens; **truncate** or **summarize** when the LM budget is tight.
- **Log** retrieval scores and chunk IDs to debug “ignored context” incidents.

### Negative Mining for Dense Retrievers

Hard **negatives** are **high-scoring** wrong passages (often BM25 neighbors without answers). Training with hard negatives sharpens the **embedding** geometry:

\[
\mathcal{L}_{\text{hard}} = - \log \frac{e^{s(q,p^+)}}{e^{s(q,p^+)} + \sum_{p^- \in \mathcal{N}_{\text{hard}}} e^{s(q,p^-)}}
\]

### Chunking and Entity Boundaries

**Named entities** split across chunks degrade **retrieval**—prefer **sentence** boundaries or **document structure** (headings) when chunking **technical** manuals.

### User-Visible Failure Modes

- **Empty retrieval** → model **hallucinates** without evidence—surface “no sources found.”
- **Conflicting** chunks → instruct the model to **compare** sources or **defer**.

### Normalization and Tokenization Alignment

**Retrieval** and **generation** should share **compatible** **normalization** (Unicode, **casing**) where possible—**mismatches** hurt **exact** **match** **recall**.

### Cold-Start Corpus

New **products** lack **click** **logs** for **relevance**—use **weak** **supervision** (BM25 **positives**) until **human** labels arrive.

---

## Interview Takeaways

- RAG splits **memory** (corpus index) from **reasoning** (LM); failures often trace to **bad retrieval**, not bad decoding.
- **Bi-encoders** enable fast ANN search; **cross-encoders** re-rank top-k with attention between query and passage but cost more.
- **Hybrid** (dense + BM25 + fusion) is standard for keyword-heavy domains (legal, medical codes, internal IDs).
- **Context packing** and **citation prompts** reduce hallucination but require evaluation for **faithfulness**.
- **Vector DB** choice hinges on scale, filtering, SLA, and whether you need **exact** k-NN for compliance audits.
- Offline metrics (recall@k) must align with online **task success** (resolved tickets, correct SQL).

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020): [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020): [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Johnson et al., *Billion-scale similarity search with GPUs* (FAISS): [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023): [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
