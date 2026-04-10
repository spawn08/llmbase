# Pre-training at Scale

## Why This Matters for LLMs

Pre-training is the phase where a foundation model acquires broad linguistic competence, factual associations, reasoning patterns, and coding ability from raw text. The quality of that phase is bounded not only by architecture and hyperparameters but by **what data enters the process**, how it is **tokenized**, which **objective** the optimizer minimizes, and how much **compute** you can afford. When interviewers at large labs ask about “Chinchilla scaling,” “BPE merges,” or “tokens per parameter,” they are probing whether you can reason about the **full production stack** that turns internet-scale text into a useful prior, not just the forward pass of a transformer block.

Decisions made before the first training step—crawl selection, deduplication strategy, mixing ratios for code versus web versus books, tokenizer vocabulary size—often have **larger long-run effects** than small learning-rate tweaks. A tokenizer that explodes sequence length for your domain wastes FLOPs on padding and distorts the effective batch size. A data pipeline that fails to remove near-duplicates can inflate metrics while teaching brittle memorization. Understanding these trade-offs is what separates someone who can **run** a training job from someone who can **design** one that survives scaling and audit.

Finally, pre-training economics dominate the bill for frontier models: **data processing**, **storage**, **GPU-hours**, and **reproducibility** all hinge on estimates like training FLOPs and the compute-optimal token budget. If you cannot connect parameters \(N\), tokens \(D\), and total cost \(C\) in closed form, you cannot negotiate cluster time, choose between more data or more width, or explain to leadership why a 7B model trained on 200B tokens may be **under-trained** relative to Chinchilla-style budgets. This page ties those threads together with math, numbers, and code you can execute locally.

---

## Core Concepts

### Data Pipelines

Modern LLM corpora are assembled from **web crawls**, **curated books**, **code repositories**, **academic papers**, and sometimes **synthetic or conversational** data. Representative public sources include **Common Crawl** snapshots (raw web HTML), **C4** (filtered Common Crawl for English), **The Pile** (heterogeneous mixture including books and code), **RedPajama** (open reproduction-oriented mixes), and **FineWeb** (large-scale filtered English web). Teams rarely use a single dump: they **blend** sources so the model sees natural language, structured documents, and program text in proportions aligned with downstream product goals.

**Cleaning** typically combines:

- **Deduplication:** exact hashing (document-level hashes) and approximate matching (**MinHash** and **LSH** families) to drop near-duplicate pages that would dominate gradients.
- **Quality filtering:** heuristics (length, symbol ratios), **perplexity filters** using a small reference language model to discard outliers, and sometimes **classifier-based** scoring trained on “good” versus “bad” examples.
- **PII and safety scrubbing:** regex and NER pipelines to reduce emails, phone numbers, and other sensitive strings where policy requires it.

**Data mixing** assigns target fractions to each shard (for example, 70% web, 15% books, 10% code, 5% dialogue). The optimal mix is not universal: coding-heavy assistants need more code; chatty assistants may overweight dialogue after the base model exists.

For **near-duplicate** detection, **MinHash** estimates **Jaccard similarity** of \(k\)-shingle sets \(A\) and \(B\):

\[
J(A,B) = \frac{|A \cap B|}{|A \cup B|}.
\]

!!! math-intuition "In Plain English"
    Shingles are overlapping chunks of characters or words. **Jaccard** measures overlap of two bags of shingles as a number between 0 and 1. **MinHash** lets you estimate that overlap **without** comparing full sets pairwise across trillions of pairs—critical for web-scale deduplication budgets.

#### Chinchilla scaling (tokens per parameter)

Hoffmann and coauthors showed that for a fixed FLOPs budget, many models were **under-trained**: they had too many parameters relative to the number of tokens seen. A practical summary is the **20 tokens per parameter** rule of thumb for compute-optimal training in the regime they studied:

\[
D_{\text{chin}} \approx 20 \times N
\]

where \(D_{\text{chin}}\) is total training tokens and \(N\) is non-embedding parameter count.

!!! math-intuition "In Plain English"
    Think of **parameters** as capacity and **tokens** as experience. If capacity is huge but experience is short, the model memorizes shallow patterns. If experience is huge but capacity is tiny, the model underfits. The Chinchilla result says: **for a given compute budget**, there is a balanced pair \((N, D)\) that minimizes loss, and that pair implies **many more tokens** than early GPT-style recipes used. The factor 20 is not a law of physics; it is an empirical anchor from one scaling study. Use it as a **first-pass budget check**, then validate on your data and architecture.

!!! example "Worked Example: Token Budget vs Model Size"
    Suppose you plan a **7 billion** parameter dense model. Chinchilla-style budgeting suggests on the order of

    \[
    D_{\text{chin}} \approx 20 \times (7 \times 10^9) = 1.4 \times 10^{11}\ \text{tokens} = 140\ \text{billion tokens}.
    \]

    If your pipeline only yields **50 billion** tokens, you are **data-limited** relative to that reference: either train longer once more data exists, reduce \(N\), or accept that you are operating away from the compute-optimal frontier. If you instead train **2 trillion** tokens on the same 7B model, you may be **past diminishing returns** for that width unless your objective rewards additional epochs for specialization.

### Tokenization and BPE

#### The Idea

Tokenization maps raw UTF-8 text into an **integer sequence** drawn from a finite vocabulary \(\mathcal{V}\). **Byte-Pair Encoding (BPE)** starts from a base alphabet (often bytes or characters) and **iteratively merges** the most frequent adjacent symbols until the vocabulary reaches a chosen size \(|\mathcal{V}|\). The result is a **subword** vocabulary that avoids huge word-level lexicons while keeping frequent words as single tokens.

#### The Math

Let \(c\) denote the corpus as a sequence of base symbols. At each iteration, BPE counts adjacent pairs \((a,b)\) and selects

\[
(a^\*, b^\*) = \arg\max_{(a,b)} \text{count}(a,b).
\]

It introduces a new merged symbol by concatenating \(a^\*\) and \(b^\*\), replaces occurrences according to the implementation’s tie-breaking rules, and repeats until the target number of merges is reached.

\[
\text{BPE: iteratively merge most frequent byte-pair until target vocabulary size is reached.}
\]

!!! math-intuition "In Plain English"
    You are building a **compression dictionary** for your training set. The merge rule says: “Which two symbols appear side-by-side most often?” Those two get a **shortcut symbol** next. Repeating this captures frequent words and morphemes as **single tokens**, while rare words are represented as **several** subword pieces. That balances **shorter sequences** (fewer tokens per sentence) against a **bounded vocabulary** (manageable embedding tables and softmax layers).

!!! example "Worked Example: BPE on `lowest`"
    **Start** with character tokens for the word `lowest`:

    `['l', 'o', 'w', 'e', 's', 't']`

    Suppose a **tiny auxiliary corpus** makes the following **pair frequencies** inside this word the highest at each step (real BPE uses the **whole** corpus; this isolates three iterations):

    - **Iteration 1:** Pair \((l,o)\) is most frequent globally. **Merge** \(l+o \to lo\). The word becomes symbols `lo`, `w`, `e`, `s`, `t` (five symbols).
    - **Iteration 2:** Among pairs inside the word, \((w,e)\) wins next. **Merge** \(w+e \to we\). Sequence: `lo`, `we`, `s`, `t` (four symbols).
    - **Iteration 3:** Pair \((s,t)\) wins. **Merge** \(s+t \to st\). Sequence: `lo`, `we`, `st` (three symbols).

    After three merges, `lowest` is encoded as **three** subword tokens instead of **six** characters, illustrating how BPE **shortens** frequent patterns.

**Ecosystem notes:** **SentencePiece** trains subword models with explicit normalization and can encode raw sentences to IDs without pre-tokenization. **tiktoken** is OpenAI’s fast runtime encoder for GPT models. **Hugging Face tokenizers** provides Rust-backed implementations for many architectures.

**Byte-level BPE** (GPT-2 family onward) starts from **256 byte types** so there is no unknown-token hole for Unicode. **Vocabulary size trade-offs:** 32K yields smaller embedding tables but longer token sequences; 50K–64K is a common compromise; 100K+ shortens sequences but grows softmax and embedding costs.

### Training Objective

#### Next-token prediction (autoregressive)

The standard objective is **next-token prediction**. For a sequence \(w_1,\dots,w_T\), the model defines conditional probabilities \(P_\theta(w_t \mid w_{<t})\). Training minimizes the **negative log-likelihood**:

\[
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(w_t \mid w_{<t}).
\]

!!! math-intuition "In Plain English"
    At each position \(t\), the model sees **only past tokens** and must assign **high probability to the true next token**. Summing \(-\log P\) over the sequence measures **total surprise** under the model. Lower loss means the model is **less surprised** by real text. This is **maximum likelihood estimation** for causal modeling.

#### Masked language modeling (BERT-style)

BERT masks a random subset \(\mathcal{M}\) and predicts masked tokens using bidirectional context:

\[
\mathcal{L}_{\text{MLM}} = -\sum_{t \in \mathcal{M}} \log P_\theta(w_t \mid w_{\setminus \mathcal{M}}).
\]

!!! math-intuition "In Plain English"
    Positions attend **left and right** to infer a masked token. This builds **rich token representations** for classification and span tasks but does not directly give a **left-to-right** generative story without additional machinery.

#### Span corruption (T5-style)

T5 deletes contiguous spans and trains the encoder-decoder to **emit** the removed text in the decoder. The loss remains **token-level cross-entropy** on decoder outputs.

!!! math-intuition "In Plain English"
    You are training **denoising** with an explicit **text-to-text** interface: input is corrupted, output is the reconstruction. This differs from standard left-to-right perplexity minimization on a single stream.

#### Fill-in-the-middle (code models)

Code models often permute chunks into **prefix, middle, suffix** and train the model to predict the **middle** given outer context. The loss is still cross-entropy, but the **data layout** encourages **infilling** used in IDEs.

### Compute Budgets and Scaling Laws

For transformer decoder training, a widely used **FLOPs** approximation is:

\[
C \approx 6 N D
\]

where \(N\) is the **parameter count** (conventions vary on embeddings), \(D\) is **total tokens processed**, and the factor **6** accounts for forward and backward passes through matrix multiplies in a rough average sense for dense models.

!!! math-intuition "In Plain English"
    Each token participates in forward and backward passes through parameter tensors; the factor 6 is a **rule-of-thumb constant** from counting multiply-adds in linear layers. Multiply **bigger models** and **more tokens** and cost grows **linearly in each** in this first-order estimate. Embedding lookups and attention have different constants, but \(6ND\) is the industry shorthand for **order-of-magnitude** cluster planning.

!!! example "Worked Example: Compute Budget"
    Take \(N = 7 \times 10^9\) and \(D = 2 \times 10^{12}\) tokens:

    \[
    C \approx 6 \times (7 \times 10^9) \times (2 \times 10^{12})
      = 84 \times 10^{21}
      = 8.4 \times 10^{22}\ \text{FLOPs}.
    \]

    Suppose a **single A100** GPU sustains **effective** \(3.12 \times 10^{14}\) FLOP/s for this workload (a rounded illustrative throughput; real jobs vary with kernel mix, precision, and parallelism). Then **one GPU** would need

    \[
    \frac{8.4 \times 10^{22}}{3.12 \times 10^{14}} \approx 2.69 \times 10^{8}\ \text{seconds} \approx 8.5\ \text{years}.
    \]

    With **8192** GPUs assuming **perfect linear speedup** (optimistic),

    \[
    \frac{8.5\ \text{years}}{8192} \approx 0.00104\ \text{years} \approx 9.1\ \text{hours}.
    \]

    Real runs include **communication overhead**, **pipeline bubbles**, **checkpointing**, and **straggler effects**, so wall-clock is **longer** than this lower bound. The calculation still shows how \(C\) translates into **human-scale time** once you pick a throughput and GPU count.

### Learning Rate Schedules

A common schedule uses **linear warmup** for \(T_{\text{warm}}\) steps, then **cosine decay** toward a floor \(\eta_{\min}\):

\[
\eta_t =
\begin{cases}
\eta_{\max} \cdot \dfrac{t}{T_{\text{warm}}}, & t < T_{\text{warm}} \\[6pt]
\eta_{\min} + \dfrac{1}{2}\bigl(\eta_{\max} - \eta_{\min}\bigr)\bigl(1 + \cos(\pi \cdot \dfrac{t - T_{\text{warm}}}{T_{\text{cos}}})\bigr), & t \ge T_{\text{warm}}
\end{cases}
\]

where \(T_{\text{cos}}\) spans the remaining training steps after warmup.

!!! math-intuition "In Plain English"
    **Warmup** slowly raises the learning rate so early batches with noisy gradients do not **destroy** weights. **Cosine decay** gently reduces the rate, giving **fine late-training moves** without a hard cliff. **Linear decay** is simpler but can **cut too aggressively** if mis-tuned. **WSD (Warmup-Stable-Decay)** holds a **plateau** for a long time, then decays—useful when you want **most tokens** at full learning rate before a short anneal.

---

## Deep Dive

??? deep-dive "Deep Dive: Data Quality vs Quantity: What Matters More?"
    **Quantity** buys coverage: rare facts, stylistic diversity, and robustness to phrasing only appear if the tail of the distribution is sampled enough times. **Quality** buys gradient signal per token: a page of spam or boilerplate consumes the same sequence positions as a textbook page but teaches less usable structure. Strong pipelines therefore **remove** toxic or repetitive junk, **deduplicate** aggressively, and **upweight** sources with lasting value (well-edited prose, vetted technical writing).

    Empirical studies repeatedly show that **two epochs of mediocre web** often lose to **one epoch of cleaner, deduped web** at the same token budget. That does not mean “small data wins”: it means **marginal tokens must earn their place**. Teams sometimes train **quality classifiers** on trusted versus suspect documents, or filter by **perplexity under a small LM** to drop non-linguistic sludge.

    The interview-ready takeaway: articulate **both** a **scaling law** story (more tokens helps if compute allows) and a **data-centric** story (bad tokens are **negative progress** at finite budgets). Mention **instrumentation**: log **loss by shard**, track **contamination** of evaluation sets, and version **filters** like code releases.

??? deep-dive "Deep Dive: The Controversy Around Training Data and Copyright"
    Large corpora include **copyrighted text** scraped from the web, **licensed books**, **user-generated content** with unclear license chains, and **code** under a spectrum of open-source terms. Legal interpretations differ by jurisdiction; this page does not offer legal advice. The engineering reality is that teams must align **data acquisition** with **corporate risk tolerance**, sometimes **excluding** categories of sites, **respecting** robots exclusion standards where applicable, and **documenting** provenance for enterprise customers.

    **Opt-out** requests and **publisher negotiations** are becoming common. **Open-weight** releases sometimes ship with **datasheets** describing mixtures at a high level without full URLs. For researchers, the actionable point is: **reproducibility** now includes **policy** and **consent** layers, not only hashes and hyperparameters.

    In interviews, show you understand **trade secrecy** versus **open science**, the role of **synthetic data** as a partial substitute, and why **evaluation contamination** from benchmark leakage is an ethical and scientific issue distinct from copyright itself.

---

## Code

```python
"""
Minimal byte-pair encoding trainer and encoder from scratch (educational).
Save as a script and run: python bpe_from_scratch.py
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


def get_pair_counts(tokens: List[str]) -> Counter:
    pairs: Counter = Counter()
    for i in range(len(tokens) - 1):
        pairs[(tokens[i], tokens[i + 1])] += 1
    return pairs


def merge_pair(tokens: List[str], pair: Tuple[str, str], new_symbol: str) -> List[str]:
    merged: List[str] = []
    i = 0
    a, b = pair
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
            merged.append(new_symbol)
            i += 2
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def train_bpe(corpus: List[str], num_merges: int) -> Tuple[Dict[int, Tuple[str, str]], List[str]]:
    """
    corpus: list of strings; characters are initial tokens.
    Returns merge rules and vocabulary list.
    """
    tokens: List[str] = list("".join(corpus))
    merges: Dict[int, Tuple[str, str]] = {}
    vocab = sorted(set(tokens))
    next_id = 0

    for _step in range(num_merges):
        pair_counts = get_pair_counts(tokens)
        if not pair_counts:
            break
        best_pair, _freq = pair_counts.most_common(1)[0]
        new_symbol = "".join(best_pair)
        merges[next_id] = best_pair
        next_id += 1
        tokens = merge_pair(tokens, best_pair, new_symbol)
        vocab = sorted(set(tokens + [new_symbol]))
    return merges, vocab


def encode_string(s: str, merges: Dict[int, Tuple[str, str]]) -> List[str]:
    """Apply merges in training order to a new string."""
    tokens = list(s)
    for _idx, pair in sorted(merges.items(), key=lambda kv: kv[0]):
        merged_sym = "".join(pair)
        tokens = merge_pair(tokens, pair, merged_sym)
    return tokens


if __name__ == "__main__":
    tiny_corpus = ["lowest", "lower", "lowland"]
    merges, vocab = train_bpe(tiny_corpus, num_merges=6)
    print("Vocabulary size:", len(vocab))
    print("First merges:", [merges[i] for i in range(min(3, len(merges)))])
    print("encode('lowest'):", encode_string("lowest", merges))
```

```python
"""
Hugging Face Datasets pipeline sketch: local rows, tokenize, map to IDs.
Requires: pip install datasets transformers
"""
from __future__ import annotations

from datasets import Dataset
from transformers import AutoTokenizer


def build_tiny_local_dataset() -> Dataset:
    rows = {
        "text": [
            "Pre-training consumes massive compute budgets.",
            "Tokenization defines the bridge between bytes and logits.",
        ]
    }
    return Dataset.from_dict(rows)


def main() -> None:
    ds = build_tiny_local_dataset()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    print(tokenized[0])


if __name__ == "__main__":
    main()
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Chinchilla trade-off:** Explain why compute-optimal training favors more tokens per parameter than older recipes, and how you would estimate a token budget for a new 13B run.
        *Answer:* Older recipes often fixed model size and trained on too few tokens relative to capacity, so the model memorized shallow patterns instead of using parameters efficiently; Chinchilla-style studies show that for a fixed FLOPs budget, loss is lower when you pair smaller \(N\) with proportionally more tokens than early GPT-era mixes implied. A practical first-pass budget is \(D \approx 20 \times N\) non-embedding parameters (e.g., \(\sim\)260B tokens for 13B params), then you sanity-check against data availability, desired specialization, and measured validation curves—if data is scarce you shrink \(N\) or accept operating off the compute-optimal frontier.
    2. **Deduplication:** Compare exact hashing versus MinHash for near-duplicate removal at trillion-token scale.
        *Answer:* Exact hashing (e.g., SHA-256 of normalized document text) cheaply removes byte-identical duplicates but misses paraphrases and template-heavy near-copies that still dominate gradients. MinHash with LSH estimates Jaccard similarity of \(k\)-shingle sets so you can drop pairs above a similarity threshold without all-pairs comparison—critical at web scale where near-dupes are far more numerous than exact dupes. Production pipelines often combine both: exact dedupe first, then MinHash/LSH for fuzzy dedupe within budget.
    3. **Tokenizer domain shift:** Your base tokenizer was trained on English web text; you fine-tune on Japanese customer support. What breaks, and how do you detect it?
        *Answer:* A tokenizer tuned to English merges will split Japanese into many more byte-pair pieces per character, inflating sequence length, attention cost, and effective batch size while starving the model of coherent morpheme-level units. You detect this via per-language token length statistics, held-out perplexity or downstream accuracy by locale, and qualitative inspection of segmentations; mitigation is often retraining or adapting the tokenizer on in-domain text, or switching to a multilingual tokenizer whose merge stats match your deployment languages.
    4. **Objective choice:** When would you choose MLM, span corruption, or autoregressive LM as a pretraining objective?
        *Answer:* Autoregressive next-token prediction is the default for general-purpose **generative** LMs and directly matches left-to-right deployment. MLM (BERT-style) builds rich bidirectional token representations for classification and span tasks but does not yield a natural causal generator without extra machinery. Span corruption / encoder–decoder (T5-style) is attractive when you want a denoising, text-to-text interface or heavy conditional generation from structured inputs; choose it when the product is explicitly encoder–decoder or heavily infilling-oriented rather than open-ended chat completion.
    5. **Data mixing:** How would you allocate code versus web if the product is primarily a coding assistant?
        *Answer:* You overweight **code** (and often related technical docs) relative to a general web mix so the model sees enough syntax, APIs, and idioms for IDE-style completion and explanation—many public recipes use double-digit percentages of code for coding assistants versus single digits for general models. You still retain some natural-language web or dialogue so the assistant remains fluent in instructions, refusals, and explanations; the exact blend is validated with coding benchmarks (HumanEval-style), long-context repo tasks, and regression checks on general chat quality.
    6. **Scaling laws:** Derive an order-of-magnitude training FLOPs estimate from \(N\) and \(D\). Where does the factor 6 come from at a high level?
        *Answer:* A standard dense-transformer shorthand is \(C \approx 6 N D\) total training FLOPs, where \(N\) is parameter count and \(D\) is total tokens seen. The **6** arises from counting multiply–adds in forward and backward passes through the dominant matrix multiplies (roughly two for the forward pass and two more for activations and weights in the backward pass per layer, aggregated into a single rule-of-thumb constant). It is not exact—attention and embeddings change the constant—but \(6ND\) is the industry order-of-magnitude for cluster planning and Chinchilla-style comparisons.
    7. **Learning-rate schedule:** Why use warmup with cosine decay instead of a constant learning rate throughout?
        *Answer:* Early training steps have noisy gradients from random initialization and data order; **warmup** slowly increases the learning rate so updates do not destabilize weights in one shot. **Cosine decay** (or long stable plateaus with a short anneal) reduces the step size in late training so the optimizer can settle into flatter regions without the abrupt cutoff of a constant-then-drop schedule. A fixed LR throughout either stays too large (late-training oscillation) or too small (slow early progress); warmup+cosine is a practical compromise validated across large LM runs.
    8. **Evaluation contamination:** How do you prevent benchmark examples from leaking into pretraining corpora?
        *Answer:* Maintain **blocklists** of benchmark strings, n-gram hashes, or dataset URLs and scan crawls before mixing; use **minhash/containment** tests against known eval sets and **canary** strings inserted in eval data that should never appear in training logs. Version deduplication pipelines and document which eval suites were excluded when, because accidental inclusion inflates benchmark scores without improving true capability—reproducibility and honest reporting require treating eval leakage as a data bug.
    9. **PII handling:** What pipeline stages reduce exposure of emails and phone numbers without destroying utility?
        *Answer:* Typical stages include regex and pattern matchers for emails/phones, **NER** or specialized classifiers for names and addresses, and deterministic **redaction** or replacement with placeholders where policy requires it. Aggressive scrubbing can damage rare formats (URLs, code with `@` symbols), so teams often tune thresholds, log scrub rates, and sample audits rather than blindly deleting every substring—balancing privacy compliance with usable text for modeling.
    10. **Reproducibility:** Which artifacts must be versioned besides model weights to reproduce a pretraining run?
        *Answer:* You need **exact data lineage**: crawl snapshots, filter versions, dedupe parameters, and mixing ratios; **tokenizer** vocabulary, merge rules, and normalization; **code** commit hash for the training stack; **hyperparameters** (LR schedule, batch, precision, seeds); and **hardware/framework** notes (CUDA, PyTorch, DeepSpeed/FSDP versions). Without these, two teams can “train the same architecture” and get different models—weights alone are insufficient for scientific or compliance-grade reproducibility.

!!! interview "Follow-up Probes"
    - “Walk through how a single MinHash signature is computed for a document shingle set.”
    - “If loss improves but downstream QA does not, name three dataset or tokenizer hypotheses you would test.”
    - “How does byte-level BPE handle Unicode emoji compared to word-level tokenization?”
    - “What happens to softmax cost as vocabulary increases, and how do architectures mitigate it?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Compute-optimal frontier** between parameters and tokens.”
    - “**Quality-filtered web** versus **raw Common Crawl**.”
    - “**Next-token cross-entropy** as maximum likelihood.”
    - “**Subword tokenization** balances **OOV handling** and **sequence length**.”
    - “**Warmup stabilizes early optimization**; **cosine decay** anneals step sizes.”

---

## References

1. Hoffmann, J., Borgeaud, S., Mensch, A., et al. **Training Compute-Optimal Large Language Models.** arXiv:2203.15556 (2022).
2. Sennrich, R., Haddow, B., Birch, A. **Neural Machine Translation of Rare Words with Subword Units.** ACL (2016).
3. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I. **Language Models are Unsupervised Multitask Learners.** OpenAI technical report (2019).
4. Raffel, C., Shazeer, N., Roberts, A., et al. **Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.** JMLR (2020).
5. Kudo, T., Richardson, J. **SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing.** EMNLP (2018).
6. Gao, L., Tow, J., Abbasi, B., et al. **The Pile: An 800GB Dataset of Diverse Text for Language Modeling.** arXiv:2101.00027 (2020).
7. Penedo, G., Kydlicek, L., allal, L.B., et al. **The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale.** arXiv:2406.00131 (2024).
8. Brown, T.B., Mann, B., Ryder, N., et al. **Language Models are Few-Shot Learners.** NeurIPS (2020).
