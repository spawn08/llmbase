# Neural Language Models

## Why This Matters for LLMs

Neural language models replaced count tables with **differentiable functions** that map a representation of the past to a distribution over the next token. Every decoder-only Transformer in production is a descendant of that idea: parameters are tuned by gradient descent on cross-entropy, not by storing raw n-gram frequencies. Interviewers probe this lineage because **gradient flow** and **context representation** separate memorized diagrams from real understanding.

The progression **feedforward window → RNN → LSTM** is the historical answer to two questions: how far can context reach, and how can error signals survive many time steps? Vanishing gradients in plain RNNs are not a footnote; they are the concrete reason gating was invented. When you explain why Transformers replaced recurrence for large-scale training, you are partly explaining **parallelism** and **path length**, but you should still articulate why LSTM gates helped the previous generation and what failure mode they addressed.

Finally, bidirectional recurrence and stacked layers (ELMo) sit between static embeddings and full self-attention. Knowing what an LSTM cell computes line by line lets you read older papers and ablation studies without confusion, and it gives you vocabulary for discussing **gradient highways**, **forget gates**, and why depth plus recurrence was hard to train before modern optimizers and initialization.

---

## Core Concepts

### Feedforward Neural Language Model (Bengio, 2003)

The model concatenates embeddings of the previous \(n-1\) words and feeds them through a tanh hidden layer:

\[
\mathbf{h} = \tanh\bigl(W [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}] + \mathbf{b}\bigr),
\qquad
P(w_t \mid w_{t-n+1:t-1}) = \text{softmax}(U \mathbf{h} + \mathbf{d}).
\]

!!! math-intuition "In Plain English"
    The bracket \([\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}]\) stacks embedding vectors into one long column. Matrix \(W\) mixes them nonlinearly into \(\mathbf{h}\). Matrix \(U\) turns \(\mathbf{h}\) into logits over the vocabulary; softmax makes them probabilities that sum to \(1\).

!!! example "Worked Example: Two-Word Context Window"
    Suppose \(n = 3\) so two context embeddings \(\mathbf{e}_1, \mathbf{e}_2 \in \mathbb{R}^2\) feed the network. Let \(\mathbf{e}_1 = (1, 0)\), \(\mathbf{e}_2 = (0, 1)\), and pretend \(W\) is a \(2 \times 4\) map so \(\mathbf{h} \in \mathbb{R}^2\) after tanh. If \(U \mathbf{h} = (0.2, 1.5, 0.1)\) for a three-word vocabulary, softmax yields \(P \approx (0.21, 0.68, 0.11)\). The model assigns the highest mass to vocabulary index \(2\). Changing embeddings changes \(\mathbf{h}\) and therefore the entire next-token distribution **without** touching a sparse count table.

### Recurrent Neural Networks

An RNN updates a hidden state \(\mathbf{h}_t\) using the previous hidden state and the current input embedding \(\mathbf{x}_t\):

\[
\mathbf{h}_t = \tanh(W_{hh} \mathbf{h}_{t-1} + W_{xh} \mathbf{x}_t + \mathbf{b}),
\qquad
P(w_{t+1} \mid w_{\le t}) = \text{softmax}(W_{hy} \mathbf{h}_t).
\]

!!! math-intuition "In Plain English"
    The same weight matrices \(W_{hh}\) and \(W_{xh}\) apply at every time step: **parameter sharing** across positions. \(\mathbf{h}_t\) is a summary of the past produced recursively. The softmax reads the current hidden state and emits next-token probabilities.

### Why RNNs Failed: Vanishing Gradients (Concrete Numbers)

Backpropagation through time multiplies Jacobian factors across steps. When the recurrent Jacobian has spectral norm below \(1\) on average, gradients shrink **exponentially** with sequence length.

!!! example "Worked Example: Repeated Multiplication by 0.9"
    If each step scales the gradient by a factor \(0.9\), after \(50\) steps the cumulative factor is \(0.9^{50}\). Compute \(\log(0.9^{50}) = 50 \log 0.9 \approx 50 \cdot (-0.10536) = -5.268\), so \(0.9^{50} \approx \exp(-5.268) \approx 0.00515\).  
    A gradient component that started at magnitude \(1\) ends near **0.005** after \(50\) steps: updates to early inputs nearly vanish. Learning long-range dependencies becomes impossible without architecture changes or gates.

!!! math-intuition "In Plain English"
    Deep networks already multiply Jacobians layer by layer; RNNs multiply them **across time**. Values below one compound into near-zero signals. Values above one **explode** unless clipped. The tanh saturation region makes matters worse by shrinking derivatives.

### LSTM: Equations Gate by Gate

An LSTM maintains a **cell state** \(\mathbf{c}_t\) and **hidden state** \(\mathbf{h}_t\). Let \([\mathbf{h}_{t-1}; \mathbf{x}_t]\) denote concatenation. All gates use sigmoid \(\sigma\) or tanh as written.

**Forget gate:**

\[
\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f).
\]

!!! math-intuition "In Plain English"
    Each coordinate of \(\mathbf{f}_t\) lies in \((0,1)\). It scales what to **keep** from the old cell state \(\mathbf{c}_{t-1}\). Near \(1\) means “retain that dimension”; near \(0\) means “erase.”

**Input gate and candidate:**

\[
\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i),
\qquad
\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c).
\]

!!! math-intuition "In Plain English"
    \(\mathbf{i}_t\) decides **how much** of the new candidate \(\tilde{\mathbf{c}}_t\) to write into the cell. \(\tilde{\mathbf{c}}_t\) is a proposed update in \([-1,1]\) per coordinate after tanh.

**Cell update:**

\[
\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t.
\]

!!! math-intuition "In Plain English"
    The symbol \(\odot\) is elementwise multiplication. Old memory \(\mathbf{c}_{t-1}\) is filtered by \(\mathbf{f}_t\), then new evidence is added scaled by \(\mathbf{i}_t\). This **additive** path lets gradients flow across time when \(\mathbf{f}_t\) stays near \(1\).

**Output gate and hidden state:**

\[
\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o),
\qquad
\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
\]

!!! math-intuition "In Plain English"
    \(\mathbf{o}_t\) picks which parts of \(\mathbf{c}_t\) (after tanh squashing) become visible as \(\mathbf{h}_t\), the vector passed to the next softmax or layer.

### Worked Example: Token `cat` Through One LSTM Step

Use **scalar** hidden and cell sizes \(d = 1\) for transparency. Suppose before processing `cat` at time \(t\):

- \(\mathbf{h}_{t-1} = 0.5\) (previous summary)
- Input embedding for `cat`: \(x_t = 1.2\)
- Previous cell: \(c_{t-1} = 0.8\)

Assume learned weights produce these **pre-activation** values before sigmoid or tanh:

- Forget pre: \(z_f = 0.0\) so \(f_t = \sigma(0) = 0.5\)
- Input pre: \(z_i = 1.0\) so \(i_t = \sigma(1) \approx 0.731\)
- Candidate pre: \(z_c = 0.5\) so \(\tilde{c}_t = \tanh(0.5) \approx 0.462\)
- Output pre: \(z_o = 0.5\) so \(o_t = \sigma(0.5) \approx 0.622\)

**Cell update:**

\[
c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde{c}_t
= 0.5 \cdot 0.8 + 0.731 \cdot 0.462
\approx 0.4 + 0.338 = 0.738.
\]

**Hidden output:**

\[
h_t = o_t \cdot \tanh(c_t) \approx 0.622 \cdot \tanh(0.738) \approx 0.622 \cdot 0.586 \approx 0.365.
\]

The softmax layer would map \(h_t\) (after a projection in real models) to vocabulary logits. The story: gates **combined** old memory and new input into \(c_t\), then **filtered** \(c_t\) into \(h_t\) for downstream use.

!!! math-intuition "In Plain English"
    This numeric trace shows \(\mathbf{h}_{t-1}\) and \(x_t\) jointly influenced every gate through the shared concatenated vector. The cell \(c_t\) carries a smoothed memory; \(h_t\) is the public interface after the output gate.

### GRU (Gated Recurrent Unit)

GRU merges cell and hidden into one state \(\mathbf{h}_t\) with **reset** \( \mathbf{r}_t \) and **update** \( \mathbf{z}_t \) gates:

\[
\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}; \mathbf{x}_t]),
\qquad
\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}; \mathbf{x}_t]),
\]

\[
\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}; \mathbf{x}_t]),
\qquad
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\]

!!! math-intuition "In Plain English"
    \(\mathbf{z}_t\) interpolates between “keep old \(\mathbf{h}_{t-1}\)” and “take new candidate \(\tilde{\mathbf{h}}_t\).” \(\mathbf{r}_t\) decides how much past \(\mathbf{h}_{t-1}\) enters the candidate computation. Fewer parameters than LSTM; often similar accuracy on many NLP tasks when depth and width match.

!!! example "Worked Example: Scalar GRU Step"
    Let \(h_{t-1} = 0.4\), \(x_t = 0.2\). Suppose \(z_t = 0.3\), \(r_t = 0.8\), and \(\tilde{h}_t = 0.9\) after the tanh. Then \(h_t = (1 - 0.3) \cdot 0.4 + 0.3 \cdot 0.9 = 0.28 + 0.27 = 0.55\). The gate blended old and new states without a separate cell vector.

### Bidirectional RNNs and ELMo

A **bidirectional** RNN runs one RNN left-to-right and another right-to-left. Token \(t\) receives \(\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]\) (concatenation or sum, depending on design).

**ELMo** (Embeddings from Language Models) stacks bidirectional LSTMs (actually two **independent** directions without cross-attention between them in the original formulation for language modeling constraints) and combines layer outputs into contextual embeddings. Modern reading: ELMo showed that **deep recurrent stacks** produce token representations far richer than Word2Vec.

!!! math-intuition "In Plain English"
    Forward \(\overrightarrow{\mathbf{h}}_t\) sees left context; backward \(\overleftarrow{\mathbf{h}}_t\) sees right context. Concatenation gives each position a summary of the **whole sentence** locally available to shallow layers. This is not the same as Transformer self-attention, but it fixes the “only left context” limitation of GPT-style models for **encoding** tasks.

---

## Deep Dive: Gradient Highways and What Replaced LSTMs in Training at Scale

??? deep-dive "Deep Dive: Why LSTMs Lost to Transformers for Large Pretraining"
    **Parallelism:** RNN steps are sequential on the time axis; GPUs idle when you cannot batch long sequences efficiently. Transformers compute attention in parallel across positions (with positional structure added explicitly).
    **Path length:** Attention can relate any two positions in \(O(1)\) layers of depth (within a layer), while an RNN needs \(O(T)\) steps to bridge \(T\) tokens.
    **Optimization at scale:** Residual connections, layer norms, and attention stability recipes scaled more predictably than very deep recurrent stacks. LSTMs remain in production for tiny footprints and streaming audio, but decoder-only Transformers dominate text.

---

## Code

Character-level LSTM language model in PyTorch: training loop, gradient clipping, and generation. Comments map tensors to the equations above.

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
        # Maps token ids to continuous vectors x_t (input embeddings)
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # One LSTM: computes h_t, c_t per the gate equations internally
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # Projects hidden state h_t to vocabulary logits before softmax
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embed(x)                      # (batch, time, embed_dim) = x_t pathway
        out, hidden = self.lstm(emb, hidden)   # out contains h_t sequence; hidden is (h_T, c_T)
        logits = self.fc(out)                     # (batch, time, vocab_size) unnormalized log-probs
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
            logits = logits[:, -1, :] / temperature  # flatten distribution: high T = more random
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
    criterion = nn.CrossEntropyLoss()  # equals negative log-softmax on true next char
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        logits, _ = model(X)
        loss = criterion(logits.view(-1, V), Y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        # Mitigate exploding gradients from recurrent depth in time
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d}  loss={loss.item():.4f}")

    print("\n── Generated text ──")
    print(generate(model, "the ", char2idx, idx2char, length=80))
```

---

## Interview Guide

### What Interviewers Actually Ask

!!! interview "FAANG-Level Questions"
    1. **Write the RNN hidden-state update and say what \(W_{hh}\) does across time.**  
       *Depth:* Same weights at every step; summarizes history recurrently.
    2. **Why do vanilla RNNs struggle with long sequences?**  
       *Depth:* Vanishing and exploding gradients; repeated Jacobian products; saturated nonlinearities.
    3. **What problem does the LSTM forget gate solve?**  
       *Depth:* Selective memory retention; enables additive cell-state updates.
    4. **Walk through LSTM cell update in plain language.**  
       *Depth:* Forget old cell, add gated new candidate, output-filtered hidden.
    5. **How does GRU differ from LSTM structurally?**  
       *Depth:* Two gates versus three; fused hidden state; fewer parameters.
    6. **Why did Transformers replace LSTMs for large-scale LM training?**  
       *Depth:* Parallelism across sequence length; shorter path between distant tokens; stable scaling with residuals and layer norms.
    7. **What is gradient clipping and what failure mode does it target?**  
       *Depth:* Caps global norm of gradients; mitigates explosion, not vanishing.
    8. **What is a bidirectional RNN and where is it inappropriate?**  
       *Depth:* Uses future and past context; invalid for strict autoregressive generation at training time for left-to-right LMs without independence assumptions.
    9. **How does ELMo differ from Word2Vec?**  
       *Depth:* Deep bidirectional contextualization versus static type vectors.
    10. **Why does the feedforward neural LM still use a fixed window?**  
       *Depth:* Context length bounded by \(n-1\); no recurrence to summarize arbitrary history.

!!! interview "Follow-up Probes"
    1. What initialization schemes reduce vanishing gradients in RNNs?
    2. How does truncated BPTT trade bias for memory?
    3. Why might you still use an LSTM on-device for streaming low-latency inference?
    4. What does `batch_first=True` change in PyTorch LSTM calls?
    5. How would you detect exploding gradients in training logs?

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Backpropagation through time and repeated Jacobian products”
    - “Additive cell-state update as a gradient highway”
    - “Gated nonlinearities as differentiable memory controllers”
    - “Sequential recurrence prevents time-parallel training”
    - “Bidirectional encoders for classification; unidirectional for autoregressive generation”

---

## References

- Bengio et al. (2003), *A Neural Probabilistic Language Model*
- Hochreiter & Schmidhuber (1997), *Long Short-Term Memory*
- Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder*
- Peters et al. (2018), *Deep Contextualized Word Representations (ELMo)*
- Pascanu et al. (2013), *On the Difficulty of Training Recurrent Neural Networks*
