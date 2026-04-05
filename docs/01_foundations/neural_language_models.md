# 1.3 — Neural Language Models

## Intuition

N-gram models hit a wall: exponential sparsity as context grows. **Neural language models** replace count tables with learned functions that map a *continuous* representation of context to a distribution over the next word. This section traces the progression from feedforward nets through RNNs to LSTMs — the architectures that dominated NLP from 2003 to 2017, and whose limitations motivated the Transformer.

---

## Core concepts

### Feedforward Neural Language Model (Bengio, 2003)

The first neural LM concatenates embeddings of the previous \(n-1\) words and passes them through a hidden layer:

\[
\mathbf{h} = \tanh\bigl(W \cdot [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}] + \mathbf{b}\bigr)
\]

\[
P(w_t \mid w_{t-n+1:t-1}) = \text{softmax}(U \mathbf{h} + \mathbf{d})
\]

**Advantages over n-grams:**

- Embedding sharing — similar words get similar predictions automatically.
- The hidden layer learns non-linear combinations of context words.

**Limitations:**

- Fixed window size — context is still bounded by \(n\).
- No parameter sharing across positions — the model sees position 1 differently from position 3.

### Recurrent Neural Networks (RNNs)

An RNN processes tokens **one at a time**, maintaining a hidden state \(\mathbf{h}_t\) that theoretically summarizes all past tokens:

\[
\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b})
\]

\[
P(w_{t+1} \mid w_{\le t}) = \text{softmax}(W_{hy} \mathbf{h}_t)
\]

**Why RNNs were a breakthrough:**

- **Variable-length context** — no fixed window.
- **Parameter sharing** across time steps — the same weights handle position 1 and position 100.

**Why they struggle:**

- **Vanishing gradients** — during backpropagation through time (BPTT), gradients are multiplied by \(W_{hh}\) at each step. When \(\|W_{hh}\| < 1\), gradients shrink exponentially; the model cannot learn long-range dependencies.
- **Exploding gradients** — when \(\|W_{hh}\| > 1\), gradients blow up. Mitigated by **gradient clipping** but doesn't solve the learning problem.
- **Sequential computation** — each step depends on the previous one, so RNNs cannot be parallelized across the sequence.

### LSTM (Long Short-Term Memory)

Hochreiter & Schmidhuber (1997) introduced **gating mechanisms** to control information flow. An LSTM cell has three gates and a cell state \(\mathbf{c}_t\):

**Forget gate** — what to discard from the cell state:

\[
\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)
\]

**Input gate** — what new information to store:

\[
\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)
\]

\[
\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)
\]

**Cell state update:**

\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t
\]

**Output gate** — what to expose as the hidden state:

\[
\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)
\]

\[
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)
\]

The cell state \(\mathbf{c}_t\) acts as a **highway**: the forget gate can set \(\mathbf{f}_t \approx 1\), allowing gradients to flow unchanged across many time steps. This directly addresses the vanishing gradient problem.

### GRU (Gated Recurrent Unit)

Cho et al. (2014) simplified LSTM into two gates (reset and update), merging the cell and hidden states. GRUs train faster with comparable quality for many tasks. The key simplification:

\[
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
\]

where \(\mathbf{z}_t\) is the update gate — an interpolation between "keep old state" and "take new candidate."

---

## Code — LSTM language model in PyTorch

```python
"""
Character-level LSTM Language Model in PyTorch.
Trains on a small corpus and generates text autoregressively.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple


class CharLSTMLM(nn.Module):
    """Character-level language model using a single LSTM layer."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embed(x)                      # (B, T, E)
        out, hidden = self.lstm(emb, hidden)      # (B, T, H)
        logits = self.fc(out)                     # (B, T, V)
        return logits, hidden


def build_vocab(text: str) -> Tuple[dict, dict]:
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


def make_sequences(
    text: str, char2idx: dict, seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [char2idx[c] for c in text]
    inputs, targets = [], []
    for i in range(0, len(encoded) - seq_len):
        inputs.append(encoded[i : i + seq_len])
        targets.append(encoded[i + 1 : i + seq_len + 1])
    return torch.tensor(inputs), torch.tensor(targets)


def generate(
    model: CharLSTMLM,
    seed: str,
    char2idx: dict,
    idx2char: dict,
    length: int = 100,
    temperature: float = 0.8,
) -> str:
    model.eval()
    hidden = None
    input_ids = torch.tensor([[char2idx[c] for c in seed]])
    result = list(seed)

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_ids, hidden)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            result.append(idx2char[next_id.item()])
            input_ids = next_id
    return "".join(result)


# ── Training loop ──────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus = (
        "the cat sat on the mat. the dog sat on the log. "
        "the cat ate the fish. the dog ate the bone. "
        "a king ruled the kingdom. a queen ruled the kingdom. "
    )
    SEQ_LEN = 20
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    EPOCHS = 300
    LR = 0.003

    char2idx, idx2char = build_vocab(corpus)
    V = len(char2idx)
    X, Y = make_sequences(corpus, char2idx, SEQ_LEN)

    model = CharLSTMLM(V, EMBED_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, V), Y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}  loss={loss.item():.4f}")

    print("\n── Generated text ──")
    print(generate(model, "the ", char2idx, idx2char, length=80))
```

---

## Interview takeaways

1. **FFNN → RNN → LSTM progression** — explain *why* each step was needed: fixed window → variable context → gradient flow through gates.
2. **Vanishing gradient** — the core reason RNNs fail on long sequences. Relate it to repeated multiplication of \(W_{hh}\). Know that LSTM's additive cell state update is the fix.
3. **Gates as soft switches** — sigmoid outputs in \([0, 1]\) act as differentiable binary gates. The forget gate is the most important: setting it to 1 lets information persist indefinitely.
4. **Sequential bottleneck** — RNN/LSTM must process tokens one-by-one, so training cannot be parallelized across the time axis. This is the key limitation that Transformers solve (Part 2.1).
5. **Gradient clipping** — a practical trick for exploding gradients. Know it clips the gradient *norm*, not individual values.
6. **Bidirectional RNNs** — sometimes asked: run two RNNs (forward + backward) and concatenate hidden states. Used in ELMo and early BERT-era feature extractors.

---

## References

- Bengio et al. (2003), *A Neural Probabilistic Language Model*
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder*
- Pascanu et al. (2013), *On the Difficulty of Training Recurrent Neural Networks* (vanishing/exploding gradients)
