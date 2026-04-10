# Retrieval-Augmented Generation (RAG)

## Why This Matters for LLMs

Retrieval-Augmented Generation is the dominant production pattern for grounding large language models in **external knowledge** that can change daily—policies, tickets, documentation, and proprietary data stores. Instead of baking facts into billions of weights, RAG **retrieves** evidence at query time and **conditions** generation on that evidence, which reduces confident hallucination on rare or time-sensitive facts and lets you update the knowledge base without retraining the backbone model. For LLM system design interviews, RAG is the default follow-up to “how do you reduce hallucinations?” because it cleanly separates **retrieval quality** from **generation quality** and exposes measurable knobs: chunking, embedding models, hybrid search, re-ranking, and faithfulness evaluation.

Second, RAG sits at the intersection of **information retrieval**, **representation learning**, and **systems engineering**. Dense bi-encoders enable approximate nearest-neighbor search at scale; classical sparse methods like BM25 remain indispensable for SKU-like tokens and exact strings; fusion strategies (linear blending, reciprocal rank fusion) combine complementary recall. Vector databases, batch indexing pipelines, embedding version pinning, and access-control-aware filtering are first-class design concerns. Interviewers probe whether you understand failure modes—irrelevant chunks, duplicate context, conflicting sources, empty retrieval, and “faithful but wrong” answers when the model ignores retrieved text—rather than treating the LM as an oracle.

Third, RAG is the bridge between **research papers** and **shipping**. Contrastive training for dense passage retrieval (DPR-style), hard negative mining, cross-encoder re-ranking, and automated RAG evaluation harnesses (e.g., faithfulness scores) give you vocabulary to discuss trade-offs: latency budgets \(T_{\text{embed}} + T_{\text{ANN}} + T_{\text{rerank}} + T_{\text{LLM}}\), freshness via incremental indexing, and offline metrics (recall@\(k\), nDCG) that must correlate with user-visible task success. A strong answer names bi-encoder vs cross-encoder costs, hybrid retrieval, and citation-aware prompting.

---

## Core Concepts

### The RAG Pipeline

A production RAG stack has three stages:

1. **Indexing**: ingest documents, **chunk** them, **embed** chunks with a bi-encoder, and store vectors plus metadata in a **vector index** (often with ANN).
2. **Retrieval**: embed the **query**, run **dense** and/or **sparse** search, optionally **re-rank** a shortlist with a cross-encoder, and pack the top-\(k\) chunks into a context budget.
3. **Augmented generation**: prepend retrieved passages to the user question under strict instructions (“answer only from context; cite `[n]`; say you do not know if missing”).

Let \(q\) be the query, \(\{c_1,\ldots,c_N\}\) the corpus chunks, and \(P_\theta\) the language model. If \(C\) is the string formed from selected chunks, a common sampling path is:

\[
a \sim P_\theta(\cdot \mid [C; q])
\]

where \([C; q]\) denotes templated concatenation with delimiters.

!!! math-intuition "In Plain English"
    This equation says the model **never sees the whole internet**—only the **retrieved slice** \(C\) you stuffed into the prompt. If \(C\) is empty or wrong, the distribution \(P_\theta(\cdot \mid [C;q])\) can still produce fluent nonsense; **retrieval recall** is often the real bottleneck.

### Document Chunking

Chunking trades **granularity** against **context**. Too small: entities split across chunks, poor semantic coherence. Too large: embeddings average away detail, one chunk dominates the budget.

**Fixed-size windows** use max length \(L\) and stride \(s < L\) (overlap \(L-s\)). If a document has token count \(T\), a simple count of windows is:

\[
N_{\text{chunks}} = \left\lceil \frac{T - L}{L - s} \right\rceil + 1 \quad (\text{for } T > L)
\]

!!! math-intuition "In Plain English"
    Overlap \(L-s\) lets the same sentence appear in two adjacent chunks so **boundaries** do not always cut through the only place an answer lives—at the cost of **storage** and **redundant** retrieval hits.

**Recursive** splitting walks a hierarchy (paragraphs → sentences → tokens). **Semantic** chunking clusters sentences by embedding similarity. **Parent–child** indexes small chunks for retrieval but injects a **parent span** (section) into the prompt for broader context.

!!! example "Worked Example: Chunking a 2000-Token Document"
    Suppose you use **512 tokens per chunk** and **50-token overlap**, so stride \(s = 462\). Approximate token positions (conceptually): chunk 1 covers tokens \(1\)–\(512\), chunk 2 covers \(463\)–\(974\) (because \(512-50=462\), next start is \(463\)), chunk 3 covers \(925\)–\(1436\), chunk 4 covers \(1387\)–\(1898\), and a short tail may merge or stand alone. **Boundary selection**: prefer splitting after `\n\n` or sentence ends inside the window so you do not break `"Invoice #"` / `"SKU-"` pairs—recursive splitting implements that policy before counting tokens.

### Embedding Models and Similarity

A **bi-encoder** maps query and chunk to vectors in \(\mathbb{R}^d\):

\[
\mathbf{e}_q = f_\phi(q),\quad \mathbf{e}_c = g_\phi(c)
\]

!!! math-intuition "In Plain English"
    The bi-encoder **compresses** query and passage into **one vector each** so you can precompute **all** passage vectors offline. At query time you only embed \(q\) once and compare to stored vectors—**sublinear** in stored chunks for ANN, unlike cross-attention over every pair.

For **cosine similarity** after L2 normalization \(\hat{\mathbf{e}} = \mathbf{e}/\|\mathbf{e}\|_2\):

\[
s_{\text{cos}}(q,c) = \hat{\mathbf{e}}_q^\top \hat{\mathbf{e}}_c
\]

!!! math-intuition "In Plain English"
    Cosine similarity is **direction matching**. After L2 normalization, dot product **equals** cosine: long passages and short queries become comparable because magnitude is stripped—**semantic** “aboutness,” not raw token counts.

**Matryoshka** embeddings are trained so that **prefix dimensions** of the vector remain meaningful; you can truncate to \(d' < d\) for storage without retraining, trading accuracy for memory.

### Vector Databases

| Database | Type | ANN / index | Cloud / self-hosted | Typical use |
|----------|------|-------------|---------------------|-------------|
| Pinecone | Managed | Proprietary | Cloud | SaaS production |
| Weaviate | Open-source | HNSW + filters | Both | Hybrid + metadata |
| Qdrant | Open-source | HNSW | Both | Low-latency OSS |
| pgvector | Postgres ext. | IVFFlat / HNSW | Self | SQL-centric teams |
| Chroma | Embedded | HNSW / others | Self | Prototyping, local |

### Sparse Retrieval: BM25

BM25 scores **lexical** overlap. For query terms \(t \in q\), document \(d\), collection statistics:

\[
\text{BM25}(d,q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)\,(k_1+1)}{f(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}
\]

Here \(f(t,d)\) is term frequency, \(|d|\) document length, \(\text{avgdl}\) average length, \(k_1\) and \(b\) are tuning constants.

!!! math-intuition "In Plain English"
    Rare terms (high IDF) with strong presence in \(d\) push the score up; length normalization stops long boilerplate docs from winning just because they repeat common words.

### Hybrid Retrieval and RRF

**Reciprocal Rank Fusion** merges ranked lists. If \(r_i(d)\) is the rank of document \(d\) in list \(i\) (1-based), a standard form is:

\[
\text{RRF}(d) = \sum_i \frac{1}{k + r_i(d)}
\]

with constant \(k\) (often 60).

!!! math-intuition "In Plain English"
    RRF rewards documents that appear **near the top** in **any** strong list—dense **or** sparse—without hand-tuning score scales between incompatible retrievers.

### HyDE and Multi-Query

**HyDE** generates a **hypothetical document** from the query, embeds it, and retrieves against that vector—useful when user queries are telegraphic. **Multi-query** expands the query via paraphrases and unions results (often with RRF). Both increase **recall** at the cost of **latency** and **noise**.

### Re-Ranking

**Cross-encoders** score pairs \((q, c)\) with joint attention; they are more accurate than bi-encoders but **cannot** precompute corpus-side vectors for every pair. Typical pipeline: bi-encoder or hybrid **top-100** → cross-encoder **top-5** → LLM.

### DPR-Style Contrastive Loss

Given query \(q\), positive passage \(p^+\), and negatives \(\{p^-_j\}\), a softmax contrastive loss (InfoNCE-style) is:

\[
\mathcal{L} = - \log \frac{\exp\bigl(s(q,p^+)/\tau\bigr)}{\exp\bigl(s(q,p^+)/\tau\bigr) + \sum_j \exp\bigl(s(q,p^-_j)/\tau\bigr)}
\]

with temperature \(\tau\).

!!! math-intuition "In Plain English"
    Pull the query embedding toward the **true** passage and push it away from **negatives**. In-batch negatives reuse other positives’ passages as cheap negatives—large batches matter.

### Retrieval Evaluation: nDCG

For ranked results and graded relevance \(rel_i\) at position \(i\), DCG at rank \(K\) is:

\[
\text{DCG}_K = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}
\]

nDCG normalizes by the ideal DCG: \(\text{nDCG}_K = \text{DCG}_K / \text{IDCG}_K\).

!!! math-intuition "In Plain English"
    nDCG rewards putting **highly relevant** chunks **early**. It captures ranking quality beyond binary hit/miss at \(k\).

### Prompting and Faithfulness

A common template:

> Answer **only** from the numbered passages. If the answer is not contained in the passages, say **“I don’t know.”** Cite passages like `[1]`.  
> **Passages:** …  
> **Question:** …

Faithfulness metrics check whether claims in \(a\) are **entailed** by \(C\) (NLI models, LLM judges—always spot-check).

### Mean Reciprocal Rank (MRR)

For a single query, if the first relevant chunk appears at rank \(r \in \{1,\ldots,K\}\):

\[
\text{RR} = \frac{1}{r},\quad \text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{r_q}
\]

If no relevant item appears in the list, \(\text{RR} = 0\).

!!! math-intuition "In Plain English"
    MRR punishes **late** first hits hard: rank 1 scores 1.0, rank 4 scores 0.25. It is sensitive to **ordering** of the first correct document—good when users read top results sequentially.

!!! example "Worked Example: Cosine Similarity on Toy Vectors"
    Let \(\mathbf{e}_q = (1,0,0)\), \(\mathbf{e}_{c_1} = (1,0,0)\), \(\mathbf{e}_{c_2} = (0,1,0)\) (already unit-norm). Then \(s_{\text{cos}}(q,c_1)=1\) and \(s_{\text{cos}}(q,c_2)=0\). If \(\mathbf{e}_{c_3} = \frac{1}{\sqrt{2}}(1,1,0)\), then \(s_{\text{cos}}(q,c_3)=\frac{1}{\sqrt{2}} \approx 0.707\). **Ranking**: \(c_1 \succ c_3 \succ c_2\). In a noisy corpus, **thresholds** on \(s_{\text{cos}}\) prevent low-score garbage from entering the prompt.

### Precision and Recall at \(k\)

With binary relevance labels:

\[
\text{Precision@}k = \frac{\text{\# relevant in top }k}{k},\quad
\text{Recall@}k = \frac{\text{\# relevant in top }k}{\text{\# relevant in corpus}}
\]

!!! math-intuition "In Plain English"
    Precision@\(k\) asks: **of what we showed**, how much was right? Recall@\(k\) asks: **of everything that should have been found**, how much did we surface in the top \(k\)? RAG systems often tune \(k\) from precision–recall curves on labeled QA pairs.

??? deep-dive "Deep Dive: HNSW and ANN recall"
    **HNSW** (Hierarchical Navigable Small World) builds a multi-layer graph: search starts at the top layer for long jumps, then refines downward. **ANN** trades **exact** nearest neighbors for **speed**—tune **efConstruction**, **M**, and **efSearch** for your recall@\(k\) vs latency curve. Always measure on **your** embedding model and corpus; public benchmarks are not transferable without re-tuning.

??? deep-dive "Deep Dive: ColBERT late interaction"
    **ColBERT** keeps multiple token vectors per document and scores with **MaxSim** between query and document tokens—richer than one vector per doc, **heavier** storage. Use when dense single-vector retrieval plateaus on your task.

## Code

The script below implements **chunking**, **SentenceTransformer** embeddings, **Chroma** storage, **top-k retrieval**, and **prompt construction**. The **generation** step is a stub you can swap for your chat API—no API keys are embedded.

```python
"""
Minimal RAG pipeline: chunk -> embed -> Chroma -> retrieve -> build prompt.
Requires: pip install sentence-transformers chromadb numpy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# --- Optional: real embeddings + vector store --------------------------------
try:
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Install dependencies: pip install sentence-transformers chromadb numpy\n" f"({e})"
    ) from e


def chunk_fixed_tokens(
    text: str, max_tokens: int = 512, overlap_tokens: int = 50
) -> List[str]:
    """Word-based chunking with overlap (pedagogical stand-in for tokenizer)."""
    words = text.split()
    if not words:
        return []
    stride = max(1, max_tokens - overlap_tokens)
    chunks: List[str] = []
    start = 0
    while start < len(words):
        piece = words[start : start + max_tokens]
        chunks.append(" ".join(piece))
        if start + max_tokens >= len(words):
            break
        start += stride
    return chunks


@dataclass
class RAGPipeline:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "kb"

    def __post_init__(self) -> None:
        self.encoder = SentenceTransformer(self.model_name)
        self.client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(list(texts), normalize_embeddings=True))

    def index_documents(self, docs: Sequence[str], ids: Sequence[str] | None = None) -> None:
        chunks: List[str] = []
        chunk_ids: List[str] = []
        for di, doc in enumerate(docs):
            parts = chunk_fixed_tokens(doc, max_tokens=64, overlap_tokens=8)
            for ci, part in enumerate(parts):
                chunks.append(part)
                base = ids[di] if ids else f"doc{di}"
                chunk_ids.append(f"{base}_chunk{ci}")
        if not chunks:
            return
        embs = self.embed(chunks)
        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embs.tolist(),
        )

    def retrieve(self, query: str, k: int = 4) -> Tuple[List[str], List[float]]:
        q = self.embed([query])[0].tolist()
        res = self.collection.query(
            query_embeddings=[q], n_results=min(k, max(1, self.collection.count()))
        )
        docs = res["documents"][0]
        dists = res["distances"][0] if res["distances"] else [0.0] * len(docs)
        # Chroma returns distance; convert cosine distance -> similarity-style display
        scores = [1.0 - float(d) for d in dists]
        return docs, scores

    @staticmethod
    def build_prompt(query: str, contexts: Sequence[str]) -> str:
        blocks = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        return (
            "You are a careful assistant. Answer using ONLY the numbered passages. "
            "If the answer is not in the passages, say you do not know.\n\n"
            f"Passages:\n{blocks}\n\nQuestion: {query}\n\nAnswer:"
        )


def demo_answer_stub(prompt: str) -> str:
    """Replace with OpenAI/Anthropic/local HF generate — no keys here."""
    return (
        "[Generation stub] Would call LM here. Prompt length (chars): "
        f"{len(prompt)} — plug in your chat completion."
    )


if __name__ == "__main__":
    corpus = [
        "Paris is the capital of France. The Eiffel Tower is a landmark in Paris.",
        "Berlin is the capital of Germany. The Brandenburg Gate is in Berlin.",
        "The euro is used in both France and Germany as of the common currency area.",
    ]
    rag = RAGPipeline()
    # Clear prior runs in ephemeral client: recreate collection
    try:
        rag.client.delete_collection(rag.collection_name)
    except Exception:
        pass
    rag.collection = rag.client.get_or_create_collection(
        name=rag.collection_name, metadata={"hnsw:space": "cosine"}
    )
    rag.index_documents(corpus, ids=["france_doc", "germany_doc", "euro_doc"])
    q = "What is the capital of Germany?"
    ctxs, scores = rag.retrieve(q, k=2)
    prompt = rag.build_prompt(q, ctxs)
    print("Retrieved:", list(zip(ctxs, scores)))
    print("\n--- Prompt (tail) ---\n", prompt[-600:])
    print("\n", demo_answer_stub(prompt))
```

!!! interview "FAANG-Level Questions"
    1. Walk through how you would design a RAG stack for a 10M-page internal wiki with ACLs per team.
    *Answer:* Partition the corpus by team or document ACL, store chunk metadata (doc id, version, clearance) beside vectors, and **filter** at query time so embeddings never leak across tenants—often with separate collections per isolation boundary or metadata predicates on every ANN query. Ingest with idempotent jobs (hash-stable chunk ids), incremental re-embed on change, and pin embedding model + chunking versions in the index manifest. For 10M pages, use batch embedding, sharded indices, hybrid retrieval with re-ranking on a shortlist, and observability on recall@k and empty-filter rates per team.
    2. When would you choose hybrid dense+BM25 over dense-only retrieval?
    *Answer:* Use hybrid when queries mix **lexical** signals (SKUs, error codes, legal citations, rare proper nouns) with **semantic** paraphrase—dense models can miss exact tokens while BM25 nails string overlap. Dense-only is simpler when vocabulary is fuzzy and exact strings rarely matter, but production wikis usually benefit from BM25 + dense fused with RRF or learned weights. The trade-off is pipeline complexity, tuning cost, and slightly higher latency versus better recall on “needle” queries.
    3. How does a bi-encoder differ from a cross-encoder in cost and accuracy?
    *Answer:* A bi-encoder embeds query and passage **independently**, so you precompute all passage vectors and score with dot products—cheap at corpus scale but shallower interaction. A cross-encoder joint-attends over \((q, c)\) pairs, which is far more accurate for relevance but **linear in corpus size** if used naïvely, so it is reserved for re-ranking tens—not millions—of candidates. Typical pattern: bi-encoder (or hybrid) → top-100 → cross-encoder → top-5 → LLM.
    4. What metrics would you use to evaluate retrieval vs generation quality separately?
    *Answer:* **Retrieval:** recall@k, MRR, nDCG on labeled chunk relevance, plus latency and empty-hit rate; optionally chunk-level precision@k when noise is costly. **Generation:** faithfulness/entailment of claims against \(C\), citation correctness, abstention rate when \(C\) is insufficient, and task success (e.g., user resolution)—not just BLEU. Separating the two avoids “good answer, bad retrieval” confusion and tells you whether to fix the index or the prompt/model.
    5. How do you handle conflicting information across retrieved chunks?
    *Answer:* Surface conflicts in the prompt (“sources disagree”) or ask the model to list claims per source with citations, and prefer **recency** or **authority** metadata (policy version, owner team) encoded in chunk metadata. For high-stakes domains, resolve conflicts with deterministic rules (latest timestamp wins) before generation, or escalate to the user. Blindly merging chunks can produce hedging or arbitrary picks—explicit conflict handling beats hoping the LM reconciles silently.
    6. Explain reciprocal rank fusion and why score scales differ between retrievers.
    *Answer:* RRF merges ranked lists by summing \(\sum_i 1/(k + r_i(d))\) so documents that rank highly in **any** list rise in the fused list without normalizing raw scores. Dense cosine, BM25, and learned scores live on **incompatible scales** and calibration—adding them directly is fragile; RRF only needs ranks. Trade-off: you lose fine-grained score magnitude (confidence) but gain robust fusion with minimal tuning.
    7. What breaks when you change the embedding model version without reindexing?
    *Answer:* Query vectors and indexed vectors then live in **different representation spaces**, so similarities become meaningless—recall collapses or ranks become arbitrary even if “distance” looks numeric. You need a full **re-embed + reindex** (often dual-write during migration), or a calibrated mapping layer that is rarely available off the shelf. Version skew also breaks A/B tests and any learned re-ranker trained on the old geometry.
    8. How does HyDE improve recall, and what are its failure modes?
    *Answer:* HyDE generates a **hypothetical document** that answers the query, embeds that text, and retrieves against it—helpful when user queries are short or keyword-poor versus document phrasing. Failure modes: the hypothetical doc invents facts (pollutes the query representation), adds topic drift, and costs an extra LLM call—so it can **hurt** precision or latency. It is a recall booster, not a substitute for good chunking or hybrid search.
    9. Describe parent–child chunking and when it helps.
    *Answer:* Index **small child** chunks for precise retrieval but inject the **parent** span (section or page) into the prompt so the model sees surrounding context and headings. It helps when answers need local precision (embedding match) plus broader discourse (parent narrative). Cost: more storage, trickier deduplication, and careful parent selection so the prompt is not dominated by huge parents.
    10. How would you detect and reduce hallucinations when context is present but ignored?
    *Answer:* Detect with **NLI/claim extraction** against retrieved passages, citation-required templates, and “answer must quote span” constraints; log cases where supported claims lack citation. Reduce by **forcing** stepwise attribution, lowering temperature on extractive settings, re-ranking to front-load the best evidence, and rejecting when confidence or entailment scores fall below threshold—treating “ignored context” as a first-class failure mode in eval dashboards.

!!! interview "Follow-up Probes"
    - “What latency budget do you assign to ANN vs cross-encoder vs LLM TTFT?”
    - “How do you shard and refresh indexes for streaming updates?”
    - “What’s your strategy for PDFs with tables and figures?”
    - “How do you log retrieval for debugging without storing PII?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Bi-encoder for recall at scale, cross-encoder for precision on the shortlist.”
    - “Hybrid retrieval: dense semantic plus BM25 for lexical exactness.”
    - “Reciprocal rank fusion merges ranked lists without score calibration.”
    - “Faithfulness is not accuracy: we measure entailment against retrieved passages.”
    - “Embedding version and chunking code must be pinned with every index build.”

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020): [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain Question Answering* (EMNLP 2020): [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- Gao et al., *Precise Zero-Shot Dense Retrieval without Relevance Labels* (HyDE, 2022): [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (2018)
- Khattab & Zaharia, *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction* (2020): [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)
- Es et al., *RAGAS: Automated Evaluation of Retrieval Augmented Generation* (2023): [arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
