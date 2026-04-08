# Sequence-to-Sequence Models

## Why This Matters for LLMs

Modern large language models did not appear from a vacuum: **sequence-to-sequence (seq2seq)** was the dominant paradigm for translation, summarization, and conditional generation before the Transformer. Seq2seq introduced the **encoder–decoder split** that still appears in T5, BART, and encoder–decoder multimodal models. Crucially, seq2seq's main weakness—a **single-vector bottleneck**—motivated **Bahdanau attention**, which generalized into **scaled dot-product self-attention** in Transformers. Interviewers often probe seq2seq not for nostalgia, but to test whether you understand **where attention came from**, how **teacher forcing** creates **exposure bias**, and why **additive vs. multiplicative** attention foreshadows today's \(QK^\top V\) machinery.

Sequence-to-sequence (seq2seq) is the idea behind machine translation: read an English sentence, compress it into a 'meaning vector', then generate the French translation word by word. It's like a person who reads a sentence in one language, understands the meaning, and then expresses that meaning in another language.

!!! tip "Notation Help"
    - Superscripts \(\text{enc}\) and \(\text{dec}\) distinguish **encoder** from **decoder** states
    - Subscript \(t\) is the **time step** (position in the sequence)
    - \(T_{\text{src}}\) means "length of the source sequence"

---

## Core Concepts

### The Encoder–Decoder Framework

#### The Idea (plain English)

Think of one person **reading** a book in English and taking **notes** (compressed meaning), and a second person **writing** the book in French using only those notes. The encoder reads the source sequence once and produces internal states; the decoder writes the target **one token at a time**, conditioning on what it has already written and (classically) on a **summary** of the source. Variable-length input and output are both handled because the decoder’s loop can stop when it emits an end token.

#### The Math

At time \(t\), the encoder consumes source token \(\mathbf{x}_t\) (often an embedding) and updates a recurrent hidden state:

\[
\mathbf{h}_t^{\text{enc}} = f_{\text{enc}}(\mathbf{x}_t, \mathbf{h}_{t-1}^{\text{enc}})
\]

The simplest seq2seq models take the **last** encoder state as the sole summary (**context vector**):

\[
\mathbf{c} = \mathbf{h}_{T_{\text{src}}}^{\text{enc}}
\]

The decoder is another recurrent network. At decoder step \(t\) it ingests the previous target token \(\mathbf{y}_{t-1}\) (embedding) and its own hidden state:

\[
\mathbf{h}_t^{\text{dec}} = f_{\text{dec}}(\mathbf{y}_{t-1}, \mathbf{h}_{t-1}^{\text{dec}})
\]

Token probabilities are produced via a linear layer and softmax:

\[
P(y_t \mid y_{<t}, \mathbf{c}) = \text{softmax}\bigl(W \mathbf{h}_t^{\text{dec}} + \mathbf{b}\bigr)
\]

(Early models also **initialize** \(\mathbf{h}_0^{\text{dec}}\) from \(\mathbf{c}\) or **concatenate** \(\mathbf{c}\) at every step; implementations vary.)

!!! math-intuition "In Plain English"
    - \(\mathbf{h}_t^{\text{enc}}\): “What I believe about the source **so far** after reading tokens \(1..t\).” It is a running summary; LSTMs/GRUs reduce vanishing-gradient issues but do not remove the **capacity** limit of a fixed state size.
    - \(\mathbf{c}\): A **single bag of numbers** meant to capture *everything* about the source sentence. In the basic model, it is literally one vector (e.g., 256 or 1024 dimensions)—that is the **bottleneck**.
    - \(\mathbf{h}_t^{\text{dec}}\): “What I believe about the partial translation **so far** while generating token \(t\).” It drives the next-word distribution.
    - The softmax turns a vector of real scores into **probabilities** over the target vocabulary: higher logit → higher probability for that word piece.

!!! example "Worked Example: Translating 'I am a student'"
    **Setup (toy sizes).** Suppose embeddings are tiny (4-D) and we read the English words as tokens `I`, `am`, `a`, `student`. After each encoder step we get \(\mathbf{h}_1^{\text{enc}}, \ldots, \mathbf{h}_4^{\text{enc}}\). In the **vanilla** model, \(\mathbf{c} = \mathbf{h}_4^{\text{enc}}\): the encoder’s state after seeing `student` is the **only** source information passed to the decoder.

    **Encoder pass (conceptual).**
    - After `I`: \(\mathbf{h}_1^{\text{enc}}\) might emphasize subject = first person.
    - After `am`: \(\mathbf{h}_2^{\text{enc}}\) adds copula / present tense cues.
    - After `a`: \(\mathbf{h}_3^{\text{enc}}\) adds indefinite article semantics.
    - After `student`: \(\mathbf{h}_4^{\text{enc}}\) adds noun semantics; **this** is \(\mathbf{c}\).

    **Decoder pass (greedy, one step at a time).**
    - Start token `<sos>` initializes \(\mathbf{h}_0^{\text{dec}}\) (often from \(\mathbf{c}\)).
    - Step 1: softmax over French vocabulary peaks at `je` → \(\hat{y}_1 = \texttt{je}\).
    - Step 2: feed `je` in; state updates; peak at `suis` → \(\hat{y}_2 = \texttt{suis}\).
    - Step 3: `un`; Step 4: `étudiant`; Step 5: perhaps `</s>`.

    **What the math enforced:** At each \(t\), \(P(y_t \mid y_{<t}, \mathbf{c})\) only sees **one** \(\mathbf{c}\) summarizing the whole English sentence. If \(\mathbf{c}\) “forgets” early words, the decoder cannot recover them—hence **attention** later.

### The Bottleneck Problem

Concrete failure mode: encode a **50-word** paragraph into \(\mathbf{c} \in \mathbb{R}^{256}\). Fifty words × thousands of possible meanings must be compressed into 256 numbers. Long-range dependencies (e.g., **gender agreement** between subject on word 2 and verb on word 40) are easily lost. This is not only vanishing gradients—it is **representation limits**: the final \(\mathbf{h}_T^{\text{enc}}\) cannot be a lossless code of the input. **Attention** fixes this by letting the decoder **read from every encoder position** at every step instead of one \(\mathbf{c}\).

### Bahdanau Attention

#### The Idea

Instead of one static summary \(\mathbf{c}\), at **each** decoder step \(t\) compute a **new** context \(\mathbf{c}_t\) as a **weighted sum** of **all** encoder hidden states \(\mathbf{h}_j^{\text{enc}}\). The weights \(\alpha_{t,j}\) say: “for producing word \(t\) of the translation, how much should I look at source word \(j\)?” Intuitively, when generating a verb, the model may attend strongly to the source **verb**; when generating an adjective, to the **noun**.

Attention was invented because of a simple problem: how do you squeeze a 50-word sentence into a single vector? You can't, without losing information. Attention says: 'Instead of one summary vector, let the decoder LOOK BACK at every word in the source sentence and decide which ones are relevant right now.' It's like a translator who glances back at the original text while writing each word of the translation.

#### The Math

**Alignment scores** (one scalar per encoder position \(j\)):

\[
e_{t,j} = \mathbf{v}^\top \tanh\bigl(W_1 \mathbf{h}_j^{\text{enc}} + W_2 \mathbf{h}_{t-1}^{\text{dec}}\bigr)
\]

**Attention weights** (normalize across \(j\)):

\[
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_{\text{src}}} \exp(e_{t,k})}
\]

**Context vector** (weighted blend of encoder states):

\[
\mathbf{c}_t = \sum_{j=1}^{T_{\text{src}}} \alpha_{t,j} \, \mathbf{h}_j^{\text{enc}}
\]

The decoder then uses \(\mathbf{c}_t\) (e.g., concat with \(\mathbf{h}_t^{\text{dec}}\)) to predict \(y_t\). Often the recurrence uses \(\mathbf{h}_{t-1}^{\text{dec}}\) in the score—**previous** decoder state—so alignment moves as translation unfolds.

!!! math-intuition "In Plain English"
    - \(e_{t,j}\): **Alignment score**—“How relevant is encoder time step \(j\) **right now**, when I am about to emit target token \(t\)?” Larger \(e_{t,j}\) means “pay more attention to source position \(j\).”
    - \(\tanh(W_1 \mathbf{h}_j^{\text{enc}} + W_2 \mathbf{h}_{t-1}^{\text{dec}})\): A small **feed-forward** mixes “what the encoder said at \(j\)” with “where the decoder is in its output.” Nonlinearity allows rich interactions; \(\mathbf{v}^\top\) projects that to one number \(e_{t,j}\).
    - \(\alpha_{t,j}\): **Soft attention weights**—positive, sum to 1 over \(j\). They behave like a probability distribution over source positions.
    - \(\mathbf{c}_t\): A **context-specific** summary: different at each \(t\), unlike fixed \(\mathbf{c}\). If the model aligns to the correct source words, \(\mathbf{c}_t\) carries **local** source meaning for the current output.

!!! example "Worked Example: Attention at Decoder Step 3"
    **Toy setting:** \(T_{\text{src}} = 3\) encoder steps, hidden size 3. Encoder states (row vectors for readability):

    \(\mathbf{h}_1^{\text{enc}} = [1.0,\, 0.0,\, 0.0]\), \(\mathbf{h}_2^{\text{enc}} = [0.0,\, 1.0,\, 0.0]\), \(\mathbf{h}_3^{\text{enc}} = [0.2,\, 0.2,\, 0.6]\).

    **Pedagogical shortcut:** Suppose the network has produced **alignment logits** (before softmax) that happen to equal dot products \(\tilde{e}_{3,j} = \mathbf{h}_{t-1}^{\text{dec}} \cdot \mathbf{h}_j^{\text{enc}}\) with \(\mathbf{h}_{t-1}^{\text{dec}} = [0.5,\, 0.8,\, 0.3]\) (full Bahdanau replaces this dot with \(\mathbf{v}^\top\tanh(\ldots)\), but the **softmax pipeline** is identical).

    **Scores:**
    - \(\tilde{e}_{3,1} = 0.5\cdot1 + 0.8\cdot0 + 0.3\cdot0 = 0.5\)
    - \(\tilde{e}_{3,2} = 0.5\cdot0 + 0.8\cdot1 + 0.3\cdot0 = 0.8\)
    - \(\tilde{e}_{3,3} = 0.5\cdot0.2 + 0.8\cdot0.2 + 0.3\cdot0.6 = 0.1 + 0.16 + 0.18 = 0.44\)

    **Softmax** (numerical, unnormalized exponentials): \(\exp(0.5)\approx 1.649\), \(\exp(0.8)\approx 2.226\), \(\exp(0.44)\approx 1.553\). Sum \(\approx 5.428\).

    \(\alpha_{3,1} \approx 1.649/5.428 \approx 0.304\), \(\alpha_{3,2} \approx 2.226/5.428 \approx 0.410\), \(\alpha_{3,3} \approx 1.553/5.428 \approx 0.286\).

    **Context** \(\mathbf{c}_3 = \sum_j \alpha_{3,j} \mathbf{h}_j^{\text{enc}}\):

    - Dim 1: \(0.304\cdot1 + 0.410\cdot0 + 0.286\cdot0.2 \approx 0.304 + 0.057 = 0.361\)
    - Dim 2: \(0 + 0.410\cdot1 + 0.286\cdot0.2 \approx 0.410 + 0.057 = 0.467\)
    - Dim 3: \(0 + 0 + 0.286\cdot0.6 \approx 0.172\)

    **Interpretation:** **Source position 2** got the **largest** weight (\(\approx 0.41\))—the decoder’s current state “matches” \(\mathbf{h}_2^{\text{enc}}\) most. In real NMT, position 2 might be the main verb or head noun the third target word depends on.

### Luong Attention (Multiplicative)

Luong et al. simplify scoring. **General** form:

\[
e_{t,j} = (\mathbf{h}_t^{\text{dec}})^\top W_a \, \mathbf{h}_j^{\text{enc}}
\]

**Dot** form (when dimensions match and \(W_a = I\)):

\[
e_{t,j} = (\mathbf{h}_t^{\text{dec}})^\top \mathbf{h}_j^{\text{enc}}
\]

**Comparison:** Bahdanau is **additive** (feed-forward in the score); Luong is **multiplicative** (bilinear or dot). Dot-product scores are **cheaper** and parallelize cleanly—the same structural idea as Transformer attention, which uses scaled dot products between **learned** projections \(Q\) and \(K\).

!!! math-intuition "In Plain English"
    - \(e_{t,j} = \mathbf{h}_t^\top \mathbf{h}_j\): **Similarity** measure—if two vectors point the same direction, the score is large. No separate \(\tanh\) MLP: “match decoder state to encoder state by cosine-like dot product.”
    - \(W_a\): Introduces **learned mixing** of dimensions before the dot product—like a **soft alignment** in a specific subspace.

### Teacher Forcing

Teacher forcing is a training trick: instead of feeding the model its own (possibly wrong) predictions as input, feed it the CORRECT previous word. It's like learning to cook from a recipe — you follow the correct steps, not your own mistakes. The downside: at test time, there's no recipe to follow, and errors can snowball.

**Plain English:** During training, the decoder receives **ground-truth** \(y_{t-1}\) as input, not the model’s own prediction \(\hat{y}_{t-1}\). So the model always sees a **correct** prefix and learns one-step-ahead prediction in a **stable** regime.

**Exposure bias:** At test time, errors compound—the model feeds **its own** (possibly wrong) outputs. Training never exposed it to its mistakes, so it may **drift** off distribution. Mitigations include **scheduled sampling** (sometimes feed \(\hat{y}\)), **self-critical training**, and **beam search** at inference.

---

## Deep Dive

??? deep-dive "Scheduled Sampling and Exposure Bias"
    **Scheduled sampling** (Bengio et al., 2015): with probability \(p_t\) increasing over training, replace ground-truth \(y_{t-1}\) by \(\hat{y}_{t-1}\). Early training uses mostly truth (easy); later mixes in model samples (harder, closer to inference). **Why it helps:** the model experiences **its** prefix distribution, reducing train/test mismatch.

    **Curriculum nuance:** Too much noise too early hurts learning; schedules matter. Modern Transformers still rely heavily on teacher forcing; **RL fine-tuning** (RLHF) and **sequence-level losses** partly address mismatch for dialogue and reasoning tasks.

??? deep-dive "Bidirectional Encoders"
    A **unidirectional** encoder RNN only sees **left context** at each \(j\). **Bidirectional** encoders run a forward and backward RNN and **concatenate** \(\mathbf{h}_j = [\overrightarrow{\mathbf{h}}_j ; \overleftarrow{\mathbf{h}}_j]\). Each position then encodes **both** sides—closer to BERT-style context, though RNNs are shallow compared to stacked Transformers. Seq2seq + bidirectional encoder was common in NMT before Transformers; **decoders** stay unidirectional when generating autoregressively.

---

## Code (keep existing, add comments)

```python
"""
Minimal Seq2Seq with Bahdanau (additive) attention.
Demonstrates: encoder, attention layer, decoder, and greedy decoding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        # Map token ids to dense vectors for the RNN
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Bidirectional GRU: each position gets left+right context in enc_out
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Project the two final directions into one init vector for the decoder
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(src)                          # (B, S, E)
        enc_out, hidden = self.rnn(embedded)                # enc_out: (B, S, 2H) per-position; hidden: (D*, B, H)
        # Merge bidirectional final states: last layer forward + last layer backward
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # (B, 2H)
        # Single nonlinearity + unsqueeze to GRU rank (1, B, H) for decoder
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)   # (1, B, H)
        return enc_out, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, enc_dim: int):
        super().__init__()
        # W1 h_j^enc — per-position encoder transform
        self.W1 = nn.Linear(enc_dim, hidden_dim, bias=False)
        # W2 h_{t-1}^dec — decoder state transform (broadcast over time in forward)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # v^T tanh(...) — score to scalar per position
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, dec_hidden: torch.Tensor, enc_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dec_hidden: (B, H)  enc_out: (B, S, 2H)
        # Additive attention: tanh(W1 enc + W2 dec) then v -> (B, S, 1)
        score = self.v(
            torch.tanh(self.W1(enc_out) + self.W2(dec_hidden).unsqueeze(1))
        )                                                    # (B, S, 1)
        weights = F.softmax(score, dim=1)                    # softmax over source positions
        # Context = convex combo of encoder outputs
        context = (weights * enc_out).sum(dim=1)             # (B, 2H)
        return context, weights.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, enc_dim)
        # Input is concat(embedding, context) — attend-and-merge before RNN step
        self.rnn = nn.GRU(embed_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embed(tgt_token)                     # (B, 1, E)
        context, attn_w = self.attention(hidden[-1], enc_out)# (B, 2H) + alphas
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)       # (B, 1, H)
        logits = self.fc(output.squeeze(1))                  # (B, V)
        return logits, hidden, attn_w


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        embed_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        enc_dim = hidden_dim * 2  # bidirectional encoder channels
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim, enc_dim)
        self.tgt_vocab = tgt_vocab

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        enc_out, hidden = self.encoder(src)
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.tgt_vocab)

        dec_input = tgt[:, 0:1]  # Usually <sos>; teacher forcing uses gold prefix
        for t in range(1, T):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out)
            outputs[:, t] = logits
            dec_input = tgt[:, t : t + 1]  # Teacher forcing: always gold previous token
        return outputs


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SRC_V, TGT_V = 20, 20
    B, SRC_LEN, TGT_LEN = 4, 8, 6

    model = Seq2Seq(SRC_V, TGT_V)
    src = torch.randint(0, SRC_V, (B, SRC_LEN))
    tgt = torch.randint(0, TGT_V, (B, TGT_LEN))

    out = model(src, tgt)
    print(f"Source shape : {src.shape}")
    print(f"Target shape : {tgt.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")
```

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What problem did seq2seq solve that plain RNN language models did not?**  
       Variable-length **input** and **output** with different lengths; conditional generation from a source sequence (MT, summarization).

    2. **Why is a single context vector a bottleneck, and how does attention fix it?**  
       Fixed \(\mathbf{c}\) has limited capacity and loses long-range detail; attention builds **per-step** \(\mathbf{c}_t\) over **all** encoder states.

    3. **Write the Bahdanau attention equations.**  
       Score \(e_{t,j} = \mathbf{v}^\top\tanh(W_1 \mathbf{h}_j^{\text{enc}} + W_2 \mathbf{h}_{t-1}^{\text{dec}})\), softmax for \(\alpha_{t,j}\), \(\mathbf{c}_t = \sum_j \alpha_{t,j}\mathbf{h}_j^{\text{enc}}\).

    4. **Additive vs. multiplicative attention—tradeoffs?**  
       Additive is more flexible per pair; multiplicative (dot) is faster and parallel-friendly—ancestor to Transformer attention.

    5. **What is teacher forcing? What is exposure bias?**  
       Training uses gold \(y_{t-1}\); test uses \(\hat{y}_{t-1}\)—mismatch can cause **error accumulation**.

    6. **How does Luong dot attention relate to Transformer attention?**  
       Both score by compatibility between **query-like** and **key-like** vectors; Transformers add scaling, multi-head, and residual stacks.

    7. **Why bidirectional encoder but unidirectional decoder?**  
       Encoder sees full source; decoder must not peek at future **target** tokens when generating left-to-right.

    8. **Interpretability:** what do attention weights \(\alpha_{t,j}\) show?  
       Soft alignment between output step \(t\) and source position \(j\)—useful but **not** guaranteed to be “true” alignment.

    9. **How does seq2seq connect to T5/BART?**  
       Same **encoder–decoder** pattern; RNNs replaced by **Transformer** blocks; attention generalized to **self-** and **cross-attention**.

    10. **Scheduled sampling—when would you use it?**  
        When exposure bias measurably hurts (some seq tasks); language modeling at scale often relies on **teacher forcing + later RL/DPO** instead.

!!! interview "Follow-up Probes"
    - “What happens to gradients if softmax attention saturates?” → vanishing flow through weights; compare to **scaled** dot-product in Transformers.
    - “Could you use last-step encoder state **and** attention?” → yes; hybrid init is implementation-dependent.
    - “Is attention **causal** in the encoder?” → encoder is typically **full** self-attention or bidirectional RNN; **decoder** causality matters for autoregression.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Encoder–decoder** maps variable-length input to variable-length output.”
    - “**Bottleneck**: one \(\mathbf{c}\) cannot preserve all source information.”
    - “**Bahdanau attention** is **content-based addressing** over encoder timesteps.”
    - “**Teacher forcing** speeds training but causes **train/test mismatch**.”
    - “**Dot-product attention** scaled by \(\sqrt{d_k}\) is the **modern** stable variant.”

### Failure Modes You Can Name in Interviews

- **Long sentences:** Early content is washed out before \(\mathbf{h}_T^{\text{enc}}\); attention mitigates by **direct paths** to each \(\mathbf{h}_j^{\text{enc}}\).
- **Rare words:** If embeddings are poor, all later steps suffer; subword units (BPE, SentencePiece) became standard partly because of this.
- **Word-order mismatch** (e.g., English SVO vs. Japanese SOV): A single \(\mathbf{c}\) must encode reordering; attention learns **soft reordering** via \(\alpha_{t,j}\).

---

## References

- Sutskever et al. (2014), *Sequence to Sequence Learning with Neural Networks*
- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al. (2015), *Effective Approaches to Attention-Based Neural Machine Translation*
- Bengio et al. (2015), *Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*
