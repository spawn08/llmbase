# Sequence Modeling and RNNs

## Why This Matters for LLMs

Large language models are **sequence models**: they consume ordered tokens and produce distributions or continuations where **position and order** matter. Before the Transformer era, recurrent networks were the default tool for learning these temporal dependencies end-to-end. Even if you never ship an LSTM in production, interviewers expect you to connect three ideas: **variable-length structure**, **shared parameters across positions**, and **why gradients behave badly across long horizons**—because those same themes reappear when you discuss attention depth, residual paths, and long-context training.

Understanding the sequence modeling problem also clarifies **why autoregressive generation** (predict one token at a time, condition on the past) became the dominant pretraining paradigm. RNNs implemented that recursion explicitly through a hidden state; Transformers implement it with causal masking and parallel attention over the prefix. When you explain “what problem self-attention solved,” part of the honest answer is **scalable training and direct paths between positions**, not the mere existence of long-range dependencies.

This page is a **primer**: it frames the problem, defines the vanilla RNN update, and sketches vanishing gradients and gating at an intuitive level. For **full LSTM/GRU gate equations**, ELMo-style bidirectionality, and the detailed vanishing-gradient analysis, see [Neural Language Models](../01_foundations/neural_language_models.md) (Part 1).

!!! tip "Notation Help"
    The equation \(\prod_{t=1}^{T}\) means "multiply terms from \(t=1\) to \(t=T\)." See [Math Prerequisites](00_math_prerequisites.md#3-summation-and-product-notation) for notation basics. The \(\tanh\) function is an activation explained in [Activation Functions](02_activation_functions.md).

---

## Core Concepts

### The Sequence Modeling Problem

Many real inputs are **ordered sequences**: words in a sentence, samples in a time series, bases along DNA, frames in video. The modeling goal is usually one or more of:

- produce a **label** or **vector** for the whole sequence (classification, retrieval);
- produce an **output sequence** aligned to the input or to a target length (translation, summarization);
- assign a **probability** to the next element given all previous elements (**language modeling**).

Unlike fixed-size tabular features, sequences can have **variable length**, and permuting elements generally **changes the meaning**. A feedforward network that flattens the sequence into one vector must either use a fixed window (losing long context) or pad to a maximum length (wasting capacity and distorting short inputs).

**Why plain MLPs struggle.** Treating position \(i\) as a separate weight column breaks **weight sharing**: the model cannot generalize the notion of “local pattern” across positions. **Convolutional** models respect locality and sharing along a grid, but classic 1-D convolutions with small kernels build receptive fields gradually; they do not, by themselves, maintain a compact **summary state** of unbounded history the way recurrence does—though dilated convolutions and later architectures narrowed the gap.

---

### Autoregressive vs Sequence-to-Sequence

**Autoregressive modeling** factorizes a joint over tokens using the chain rule:

\[
P(x_1,\ldots,x_T) = \prod_{t=1}^{T} P(x_t \mid x_1,\ldots,x_{t-1}).
\]

At each step the model outputs a distribution over the next token given the **prefix**. Decoder-only Transformers (GPT-style) are trained in this regime: causal attention ensures each position only sees the past.

**Sequence-to-sequence** maps an input sequence to an output sequence, often with different lengths—e.g. machine translation. Encoder–decoder RNNs (and later Transformers) read the source into a representation, then generate the target **step by step**, sometimes with attention between encoder and decoder. Text-to-text models (T5-style) cast many tasks into “string in, string out,” but the high-level split remains: **next-token prediction along one stream** versus **conditional generation of another stream**.

---

### The Vanilla RNN

A recurrent layer maintains a **hidden state** \(\mathbf{h}_t\) that summarizes information from time \(1\) through \(t\). With input embedding \(\mathbf{x}_t\), the standard update is:

\[
\mathbf{h}_t = \tanh\bigl(W_{hh}\,\mathbf{h}_{t-1} + W_{xh}\,\mathbf{x}_t + \mathbf{b}\bigr).
\]

The **same** matrices \(W_{hh}\), \(W_{xh}\), and bias \(\mathbf{b}\) are reused at every time step: **parameter sharing across time**. A readout layer (for example softmax on \(W_{hy}\mathbf{h}_t\)) maps the state to logits for the next token or task label.

!!! math-intuition "In Plain English"
    Think of \(\mathbf{h}_{t-1}\) as a compressed “memory” of what you have seen so far. Each new observation \(\mathbf{x}_t\) is blended with that memory through the affine map, then squashed by \(\tanh\) into a bounded vector for the next step. One set of weights implements **every** time step—analogous to applying the same convolution kernel at every spatial location.

!!! example "Worked Example: Three Steps of a Scalar RNN"
    Let \(h_t, x_t \in \mathbb{R}\) (one-dimensional state and input). Take \(W_{hh} = 0.5\), \(W_{xh} = 1.0\), \(b = 0\), and initial state \(h_0 = 0\). Inputs \(x_1 = 0.8\), \(x_2 = -0.3\), \(x_3 = 0.5\).

    - **Step \(t=1\):** \(z_1 = 0.5 \cdot 0 + 1.0 \cdot 0.8 = 0.8\), so \(h_1 = \tanh(0.8) \approx 0.664\).
    - **Step \(t=2\):** \(z_2 = 0.5 \cdot 0.664 + 1.0 \cdot (-0.3) \approx 0.032\), so \(h_2 = \tanh(0.032) \approx 0.032\).
    - **Step \(t=3\):** \(z_3 = 0.5 \cdot 0.032 + 1.0 \cdot 0.5 \approx 0.516\), so \(h_3 = \tanh(0.516) \approx 0.474\).

    The state carries forward a nonlinear mix of all prior inputs; early values influence later \(h_t\) through repeated application of the same recurrence.

---

### Unrolling in Time

Training uses **backpropagation through time (BPTT)**: the recurrence is **unrolled** into a feedforward graph with one layer per time step. Conceptually:

```text
x_1 → [RNN] → h_1 → [RNN] → h_2 → … → h_T
       ↑         ↑         ↑
    shared W   shared W   shared W
```

Each box applies the **same** \(W_{hh}\), \(W_{xh}\), \(\mathbf{b}\). The unrolled network is deep in **time**; gradients flow backward along edges that repeat the Jacobian of the transition. That depth-in-time is exactly what makes long-range credit assignment hard for vanilla tanh RNNs.

---

### The Vanishing Gradient Problem (Intuition)

During BPTT, the derivative of a loss at time \(T\) with respect to an early state or input involves a **product** of Jacobian factors across intermediate steps. If typical magnitudes are **smaller than one**, the product **shrinks exponentially** with sequence length—early inputs receive negligible updates. Saturating \(\tanh\) regions worsen this by driving derivatives toward zero.

!!! example "Why 0.9^50 Is Tiny"
    If each time step scales a gradient component by a factor around \(0.9\) in magnitude, after \(50\) steps the cumulative factor is \(0.9^{50}\). Since \(\log(0.9^{50}) = 50 \log 0.9 \approx -5.27\), we get \(0.9^{50} \approx e^{-5.27} \approx 0.005\). A signal that started at unit scale is already **below half a percent** after \(50\) multiplications.

!!! math-intuition "In Plain English"
    Vanishing gradients mean **short memory in optimization**: the model cannot easily learn to blame an error at \(t=100\) on what happened at \(t=5\). The full spectral-norm and long-term dependency analysis lives in [Neural Language Models](../01_foundations/neural_language_models.md).

---

### Why Gating Was Invented

**LSTM** and **GRU** introduce **gates** (sigmoid multipliers) and often an **additive** path for memory updates. Instead of only multiplicative composition through \(\tanh\), additive updates to a cell state allow error signals to flow more directly across many steps—a **gradient highway** when gates stay open. Intuitively: gates learn **what to forget, what to write, and what to expose**.

!!! math-intuition "In Plain English"
    Picture a conveyor belt (cell state) you can edit at each station. Gates decide how much old material stays, how much new evidence arrives, and what the next hidden output shows. **Part 1** walks through LSTM and GRU equations line by line—here you only need the motivation: **control flow** and **additive paths** mitigate vanishing gradients in depth-through-time.

---

### Bidirectional Processing

A **bidirectional RNN** runs one recurrence left-to-right and another right-to-left, then **concatenates** or combines hidden states. For **encoding** tasks—tagging, classification of a whole sentence—seeing both contexts is often helpful.

For **autoregressive generation**, you must **not** condition on future tokens you have not generated yet. Decoder-only LLMs use **causal** attention for exactly this reason. Bidirectional encoders (BERT-style) are powerful for understanding, but they are not a drop-in for left-to-right generation without architectural changes.

---

## Deep Dive

??? deep-dive "Deep Dive: Teacher forcing, truncated BPTT, and when RNNs still win"
    **Teacher forcing.** In seq2seq training, the decoder is often fed the **ground-truth** previous token as input while learning to predict the next token. At inference, the model feeds **its own** predictions—so training and test distributions can **diverge** (exposure bias). Mitigations include scheduled sampling and careful inference strategies; Transformers in translation faced the same tension before better decoding and large-scale data.

    **Truncated BPTT.** Full BPTT over very long sequences is expensive and can be unstable. **Truncated** BPTT limits the backward pass to a **window** of recent time steps (or periodically resets state). You get cheaper gradients that still capture medium-range dependencies, at the cost of cutting off gradients to very early tokens within a chunk.

    **Where RNNs still beat Transformers (sometimes).** For **tiny** models and **tight** compute or memory budgets, small GRUs/LSTMs can be competitive. **Streaming** audio or sensor data favors \(O(1)\) state updates per step versus growing KV caches. **On-device** or **low-latency** pipelines sometimes prefer a compact recurrent core over a full attention stack—though efficient attention, state-space models, and hybrids blur the line.

---

## Code

Below is a **minimal** character-level language model: embedding lookup, **hand-written** vanilla RNN transition (no `torch.nn.RNN`), softmax logits, one step of gradient descent on a toy corpus. It is meant to mirror the equations, not to reach low perplexity.

```python
"""
Vanilla RNN: next-character prediction on a tiny corpus (NumPy only).
Implements h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h) with one-hot inputs.
"""

from __future__ import annotations

import numpy as np


def one_hot(idx: int, vocab_size: int) -> np.ndarray:
    v = np.zeros((vocab_size, 1), dtype=np.float64)
    v[idx, 0] = 1.0
    return v


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def tanh_grad(t: np.ndarray) -> np.ndarray:
    return 1.0 - t**2


def rnn_forward(
    indices: list[int],
    emb: np.ndarray,
    w_xh: np.ndarray,
    w_hh: np.ndarray,
    b_h: np.ndarray,
    w_hy: np.ndarray,
    b_y: np.ndarray,
    h0: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Returns per-step hidden states (after tanh), pre-activations, and logits."""
    vocab, d_in = emb.shape
    d_h = w_hh.shape[0]
    h = h0.copy()
    hs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []

    for t_idx in indices:
        x = emb[t_idx : t_idx + 1].T  # (d_in, 1)
        z = w_xh @ x + w_hh @ h + b_h
        h = np.tanh(z)
        logit = w_hy @ h + b_y
        hs.append(h.copy())
        zs.append(z.copy())
        logits_list.append(logit.copy())
    return hs, zs, logits_list


def main() -> None:
    rng = np.random.default_rng(0)
    text = "hello world " * 3  # tiny repeating corpus
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    v = len(chars)

    d_in, d_h, d_out = 8, 12, v
    emb = 0.1 * rng.standard_normal((v, d_in))
    w_xh = 0.1 * rng.standard_normal((d_h, d_in))
    w_hh = 0.1 * rng.standard_normal((d_h, d_h))
    b_h = np.zeros((d_h, 1))
    w_hy = 0.1 * rng.standard_normal((d_out, d_h))
    b_y = np.zeros((d_out, 1))
    h0 = np.zeros((d_h, 1))

    idxs = [stoi[c] for c in text]
    # Predict next char: inputs are idxs[:-1], targets are idxs[1:]
    ins, targets = idxs[:-1], idxs[1:]

    lr = 0.5
    for step in range(200):
        # --- forward ---
        h = h0.copy()
        loss = 0.0
        dh_next = np.zeros_like(h)
        # storage for BPTT (manual backprop through time)
        cache: list[tuple] = []

        for t in range(len(ins)):
            x = emb[ins[t] : ins[t] + 1].T
            z_pre = w_xh @ x + w_hh @ h + b_h
            h_new = np.tanh(z_pre)
            logits = w_hy @ h_new + b_y
            probs = softmax(logits)
            y = one_hot(targets[t], v)
            loss += float(-(y * np.log(probs + 1e-12)).sum())

            cache.append((x, h, z_pre, h_new, probs, y))
            h = h_new

        loss /= len(ins)

        # --- backward (truncated: full sequence here since tiny) ---
        d_emb = np.zeros_like(emb)
        dw_xh = np.zeros_like(w_xh)
        dw_hh = np.zeros_like(w_hh)
        db_h = np.zeros_like(b_h)
        dw_hy = np.zeros_like(w_hy)
        db_y = np.zeros_like(b_y)

        for t in reversed(range(len(ins))):
            x, h_prev, z_pre, h_new, probs, y = cache[t]
            dlogit = (probs - y) / len(ins)
            dw_hy += dlogit @ h_new.T
            db_y += dlogit
            dh = w_hy.T @ dlogit + dh_next
            dz = tanh_grad(h_new) * dh
            dw_xh += dz @ x.T
            dw_hh += dz @ h_prev.T
            db_h += dz
            d_emb[ins[t] : ins[t] + 1] += (dz.T @ w_xh).ravel()
            dh_next = w_hh.T @ dz

        # --- SGD update ---
        emb -= lr * d_emb
        w_xh -= lr * dw_xh
        w_hh -= lr * dw_hh
        b_h -= lr * db_h
        w_hy -= lr * dw_hy
        b_y -= lr * db_y

        if step % 50 == 0:
            print(f"step {step:3d}  avg_nll ≈ {loss:.4f}")

    # Demo: forward pass matching the math for first 3 steps
    demo_idx = ins[:3]
    hs, zs, logits = rnn_forward(demo_idx, emb, w_xh, w_hh, b_h, w_hy, b_y, h0)
    print("\nFirst 3 steps — hidden state norms ||h_t||:")
    for i, h in enumerate(hs, start=1):
        print(f"  t={i}: ||h||_2 ≈ {np.linalg.norm(h):.4f}")


if __name__ == "__main__":
    main()
```

Running the script prints decreasing average negative log-likelihood and shows that the forward pass implements the same recurrence used in the derivation above.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What is the sequence modeling problem, and why do fixed-window MLPs fail to generalize across positions?**  
       *Depth:* Variable length, order matters, and weight sharing across time/position is needed; flat MLPs on fixed concatenations do not share structure across offsets.

    2. **Write the vanilla RNN hidden update and explain parameter sharing across time.**  
       *Depth:* \(\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b})\); same \(W_{hh}, W_{xh}\) at every step—unrolled depth shares weights.

    3. **Contrast autoregressive language modeling with encoder–decoder sequence-to-sequence training.**  
       *Depth:* Chain-rule factorization for \(P(x_{1:T})\) vs conditional generation of an output sequence; different masking and loss alignment.

    4. **Why do gradients vanish in deep unrolled RNNs, and what does \(0.9^{50}\) illustrate?**  
       *Depth:* Products of Jacobian factors across steps shrink if typical multipliers are \(<1\); \(0.9^{50} \approx 0.005\) shows exponential decay over horizon.

    5. **What problem did LSTM/GRU gating address, without deriving every gate?**  
       *Depth:* Long-range credit assignment; additive cell paths and gates allow controlled memory and better gradient flow than plain tanh recurrence.

    6. **When is bidirectional recurrence appropriate, and why not for GPT-style decoding?**  
       *Depth:* Encoding with full context vs causal generation—future tokens are unavailable at inference.

    7. **What is teacher forcing, and what mismatch does it introduce?**  
       *Depth:* Training with gold previous tokens vs sampling from the model at inference; exposure bias.

    8. **Name two settings where compact RNNs sometimes remain attractive versus Transformers.**  
       *Depth:* Streaming \(O(1)\) state updates, tiny on-device models, extreme memory constraints—accepting lower ceiling on parallelization.

!!! interview "Follow-up Probes"
    1. **How does unrolling relate RNN depth to sequence length?**  
       *Depth:* Each step is a layer in the computational graph; BPTT backprops through that temporal depth.

    2. **Why truncated BPTT?**  
       *Depth:* Cost and stability; limits how far gradients reach within one backward pass.

    3. **What does tanh saturation do to gradients?**  
       *Depth:* Derivatives \(\approx 0\) in saturation regions, worsening vanishing products in BPTT.

    4. **How does weight sharing in CNNs differ from RNNs?**  
       *Depth:* CNNs share across spatial offsets; RNNs share across time—both encode translation structure, different domains.

    5. **Where do you read the full LSTM equations in LLMBase?**  
       *Depth:* [Neural Language Models](../01_foundations/neural_language_models.md)—gate-by-gate math and ELMo context.

!!! key-phrases "Key Phrases to Use in Interviews"
    1. “Recurrence shares parameters across time; the unrolled graph is deep in the time dimension.”
    2. “Autoregressive factorization: each token is predicted from the prefix.”
    3. “Vanishing gradients come from products of Jacobians across many steps.”
    4. “Gating and additive cell updates create highways for gradient flow.”
    5. “Bidirectional encoders see future context; causal decoders cannot.”
    6. “Teacher forcing trains with ground-truth inputs; inference uses model samples.”

---

## References

1. Elman, J. L. (1990). *Finding structure in time.* Cognitive Science. (Foundational RNN motivation.)
2. Bengio, Y., Simard, P., & Frasconi, P. (1994). *Learning long-term dependencies with gradient descent is difficult.* IEEE Transactions on Neural Networks. (Vanishing gradients in recurrence.)
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory.* Neural Computation. (LSTM—details in Part 1.)
4. Cho, K., et al. (2014). *Learning phrase representations using RNN encoder–decoder for statistical machine translation.* EMNLP. (GRU variant widely used.)
5. Williams, R. J., & Zipser, D. (1989). *A learning algorithm for continually running fully recurrent neural networks.* Neural Computation. (BPTT.)
6. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to sequence learning with neural networks.* NeurIPS. (Seq2seq framing.)

**LLMBase (Part 1):** [Neural Language Models](../01_foundations/neural_language_models.md) — LSTM/GRU equations, ELMo, and vanishing-gradient analysis in full detail.
