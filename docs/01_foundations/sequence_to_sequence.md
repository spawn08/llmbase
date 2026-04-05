# 1.4 — Sequence-to-Sequence Models

## Intuition

Translation, summarization, and question-answering all share a pattern: **read a variable-length input → produce a variable-length output**. The *Seq2Seq* architecture (Sutskever et al., 2014) solved this by chaining an **encoder** that compresses the source into a fixed vector with a **decoder** that generates the target one token at a time. Its critical weakness — squeezing an entire sentence into a single vector — led directly to the **attention mechanism** (Bahdanau et al., 2015), the precursor to the Transformer.

---

## Core concepts

### Encoder–decoder framework

```
Source: "the cat sat"  →  Encoder  →  context vector c  →  Decoder  →  "le chat assis"
```

**Encoder** — an RNN (or LSTM) reads source tokens left-to-right. The final hidden state \(\mathbf{h}_T^{\text{enc}}\) is the **context vector** \(\mathbf{c}\):

\[
\mathbf{h}_t^{\text{enc}} = f_{\text{enc}}(\mathbf{x}_t, \mathbf{h}_{t-1}^{\text{enc}})
\]

\[
\mathbf{c} = \mathbf{h}_T^{\text{enc}}
\]

**Decoder** — another RNN generates target tokens autoregressively, initialized with \(\mathbf{c}\):

\[
\mathbf{h}_t^{\text{dec}} = f_{\text{dec}}(\mathbf{y}_{t-1}, \mathbf{h}_{t-1}^{\text{dec}})
\]

\[
P(y_t \mid y_{<t}, \mathbf{c}) = \text{softmax}(W \mathbf{h}_t^{\text{dec}})
\]

### The bottleneck problem

All source information must fit in one fixed-size vector \(\mathbf{c}\). For long sequences, the encoder "forgets" early tokens — the same vanishing-gradient problem from Part 1.3, now hitting the representation capacity.

### Bahdanau attention (additive attention)

Instead of a single context vector, let the decoder **look back** at every encoder hidden state at each generation step.

For decoder step \(t\):

1. Compute an **alignment score** between the current decoder state and each encoder state:

\[
e_{t,j} = \mathbf{v}^T \tanh(W_1 \mathbf{h}_j^{\text{enc}} + W_2 \mathbf{h}_{t-1}^{\text{dec}})
\]

2. Normalize into **attention weights** via softmax:

\[
\alpha_{t,j} = \frac{\exp(e_{t,j})}{\sum_{k=1}^{T_{\text{src}}} \exp(e_{t,k})}
\]

3. Compute a **context vector** as a weighted sum of encoder states:

\[
\mathbf{c}_t = \sum_{j=1}^{T_{\text{src}}} \alpha_{t,j} \, \mathbf{h}_j^{\text{enc}}
\]

4. Concatenate \(\mathbf{c}_t\) with \(\mathbf{h}_t^{\text{dec}}\) to predict the next token.

This is called **additive** (or **concat**) attention because the score function uses an additive layer with `tanh`.

### Luong attention (multiplicative attention)

Luong et al. (2015) simplified the score to a dot product (or a bilinear form):

\[
e_{t,j} = (\mathbf{h}_t^{\text{dec}})^T W_a \, \mathbf{h}_j^{\text{enc}}
\quad \text{(general)}
\]

\[
e_{t,j} = (\mathbf{h}_t^{\text{dec}})^T \mathbf{h}_j^{\text{enc}}
\quad \text{(dot)}
\]

The **dot-product** variant is the direct ancestor of the attention in "Attention Is All You Need" (Part 2.1).

### Teacher forcing

During training, the decoder receives the **ground-truth** previous token \(y_{t-1}\) instead of its own prediction. This makes training faster and more stable, but creates a train/test mismatch called **exposure bias** — the decoder never learns to recover from its own mistakes.

---

## Code — Seq2Seq with Bahdanau attention in PyTorch

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
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.embed(src)                          # (B, S, E)
        enc_out, hidden = self.rnn(embedded)                # enc_out: (B, S, 2H)
        # Merge bidirectional final states for decoder init
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1) # (B, 2H)
        hidden = torch.tanh(self.fc(hidden)).unsqueeze(0)   # (1, B, H)
        return enc_out, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.W1 = nn.Linear(enc_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self, dec_hidden: torch.Tensor, enc_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # dec_hidden: (B, H)  enc_out: (B, S, 2H)
        score = self.v(
            torch.tanh(self.W1(enc_out) + self.W2(dec_hidden).unsqueeze(1))
        )                                                    # (B, S, 1)
        weights = F.softmax(score, dim=1)                    # (B, S, 1)
        context = (weights * enc_out).sum(dim=1)             # (B, 2H)
        return context, weights.squeeze(-1)


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, enc_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim, enc_dim)
        self.rnn = nn.GRU(embed_dim + enc_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tgt_token: torch.Tensor,
        hidden: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.embed(tgt_token)                     # (B, 1, E)
        context, attn_w = self.attention(hidden[-1], enc_out)# (B, 2H)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)         # (B, 1, H)
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
        enc_dim = hidden_dim * 2  # bidirectional
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim, enc_dim)
        self.tgt_vocab = tgt_vocab

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> torch.Tensor:
        enc_out, hidden = self.encoder(src)
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.tgt_vocab)

        dec_input = tgt[:, 0:1]  # <sos> token
        for t in range(1, T):
            logits, hidden, _ = self.decoder(dec_input, hidden, enc_out)
            outputs[:, t] = logits
            dec_input = tgt[:, t : t + 1]  # teacher forcing
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

## Interview takeaways

1. **Encoder = read, Decoder = write** — the encoder produces representations; the decoder consumes them autoregressively. This division persists in T5 and encoder–decoder Transformers.
2. **Fixed-vector bottleneck** — the single context vector cannot scale to long documents. Attention solved this by making context a *dynamic* weighted sum of all encoder states.
3. **Additive vs. multiplicative attention** — Bahdanau uses a learned layer; Luong uses a dot product. Dot-product attention is cheaper and is what Transformers use (after scaling by \(\sqrt{d_k}\)).
4. **Attention weights are interpretable** — they show which source tokens the decoder "looks at" for each output token. This interpretability carried into Transformer attention heads.
5. **Teacher forcing vs. scheduled sampling** — teacher forcing is fast but causes exposure bias. Some interviewers ask about **scheduled sampling** (gradually replacing ground truth with model predictions during training).
6. **Bidirectional encoder** — using a bidirectional RNN lets the encoder capture both left and right context, a pattern that BERT generalizes with masked self-attention.

---

## References

- Sutskever et al. (2014), *Sequence to Sequence Learning with Neural Networks*
- Bahdanau et al. (2015), *Neural Machine Translation by Jointly Learning to Align and Translate*
- Luong et al. (2015), *Effective Approaches to Attention-Based Neural Machine Translation*
