# ELECTRA: Pre-training Text Encoders as Discriminators

**Authors:** Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning  
**Year:** 2020 &nbsp;|&nbsp; **Venue:** ICLR  
**Link:** [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)

---

## TL;DR

ELECTRA replaces BERT's Masked Language Modeling with **replaced token detection**: a small generator proposes replacement tokens, and a discriminator predicts **which tokens were replaced** — a binary classification per position. This is far more **sample-efficient** than predicting a full vocabulary softmax at every masked position, because every token in the sequence provides a training signal (not just the 15% that are masked).

---

## Why This Paper Matters

ELECTRA showed that **you can pre-train strong encoders with much less compute** by changing the objective from generation to discrimination:

1. **Sample efficiency:** Every token contributes to the loss (not just 15% masked ones)
2. **Computational efficiency:** Binary classification is cheaper than full-vocabulary softmax
3. **Strong small models:** ELECTRA-Small matches GPT-large with 1/10th the compute
4. Connects to **GAN-like** training and **contrastive** learning ideas

---

## Key Concepts Explained Simply

### The Problem with MLM

In BERT, only **15% of tokens** provide a training signal (the masked ones). The other 85% are just context — the model gets no direct gradient from them. This is wasteful: you process the entire sequence but learn from only a fraction.

### ELECTRA's Solution: Replaced Token Detection

Two models work together:

1. **Generator (small):** A small masked language model that proposes replacement tokens for masked positions. Think of it as a "corruptor."

2. **Discriminator (main model):** Looks at the corrupted sequence and predicts, for **every** token, whether it is "original" or "replaced." This is binary classification.

The discriminator is the model you keep. It's trained on **100% of positions** — every token either IS or IS NOT replaced, and the discriminator must figure out which.

### Why It's More Efficient

| Aspect | BERT (MLM) | ELECTRA |
|---|---|---|
| Training signal | 15% of tokens | 100% of tokens |
| Output layer | Full vocabulary softmax | Binary sigmoid |
| Compute per position | \(O(\text{vocab\_size})\) | \(O(1)\) |
| Effective training | ~6.7× less per token | Full utilization |

---

## The Math — Explained Step by Step

### Generator

The generator is a standard MLM model. For masked positions, it samples replacements:

\[
\hat{x}_i \sim P_G(x_i \mid \tilde{\mathbf{x}}) \quad \text{for } i \in \mathcal{M}
\]

The generator is trained with MLM loss. It's intentionally **small** (e.g., 1/4 the size of the discriminator) — it just needs to produce plausible replacements, not be a great language model.

### Discriminator

The discriminator sees the generator's output (with replacements at masked positions) and classifies each position:

\[
\mathcal{L}_D = -\sum_{i=1}^{n} \left[y_i \log \sigma(s_i) + (1 - y_i) \log(1 - \sigma(s_i))\right]
\]

where:
- \(s_i\): The discriminator's logit for position \(i\)
- \(\sigma\): Sigmoid function
- \(y_i = 1\) if token \(i\) is **original** (not replaced), \(y_i = 0\) if replaced
- The sum is over **all** positions — not just masked ones

### Combined Loss

\[
\mathcal{L} = \mathcal{L}_{\text{MLM}}^{(G)} + \lambda \cdot \mathcal{L}_D
\]

The generator is trained with standard MLM loss. The discriminator loss is weighted by \(\lambda\) (typically 50). Note: gradients do **not** flow from the discriminator back through the generator (unlike GANs) — the generator is trained independently with MLM.

### Why Not a GAN?

It looks like a GAN (generator + discriminator), but there are key differences:
- The generator is trained with **maximum likelihood** (MLM), not adversarial loss
- No gradient flows from discriminator to generator
- The generator operates on **discrete** tokens (sampling breaks differentiability)
- This makes training much more stable than GANs

---

## Python Implementation

```python
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def stable_softmax(logits):
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


def generator_sample(token_ids, mask_positions, generator_logits, vocab_size):
    """
    Small generator produces replacement tokens at masked positions.
    generator_logits: [num_masked, vocab_size]
    """
    replaced = token_ids.copy()
    for idx, pos in enumerate(mask_positions):
        probs = stable_softmax(generator_logits[idx])
        sampled = np.random.choice(vocab_size, p=probs)
        replaced[pos] = sampled
    return replaced


def discriminator_labels(original_ids, replaced_ids):
    """
    y_i = 1 if original (not replaced), 0 if replaced.
    Note: even if generator samples the correct token, it's still
    considered "replaced" in the original paper's implementation.
    """
    return (original_ids == replaced_ids).astype(float)


def discriminator_loss(scores, labels):
    """
    Binary cross-entropy over ALL positions.
    scores: [seq_len] — discriminator logits
    labels: [seq_len] — 1=original, 0=replaced
    """
    s = sigmoid(scores)
    loss = -(labels * np.log(s + 1e-12) + (1 - labels) * np.log(1 - s + 1e-12))
    return np.mean(loss)


def mlm_generator_loss(logits, true_labels):
    """
    Standard MLM loss for the generator (at masked positions only).
    logits: [num_masked, vocab_size]
    true_labels: [num_masked]
    """
    probs = stable_softmax(logits)
    loss = 0.0
    for i, label in enumerate(true_labels):
        loss -= np.log(probs[i, label] + 1e-12)
    return loss / len(true_labels)


class ELECTRATrainer:
    """Simplified ELECTRA training loop."""

    def __init__(self, vocab_size=1000, d_model=64, mask_prob=0.15, lam=50.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_prob = mask_prob
        self.lam = lam

        # Tiny generator (1/4 the size conceptually)
        self.gen_proj = np.random.randn(d_model // 4, vocab_size) * 0.02
        # Discriminator head
        self.disc_proj = np.random.randn(d_model, 1) * 0.02

    def train_step(self, token_ids):
        seq_len = len(token_ids)

        # Step 1: Select mask positions
        mask_positions = [i for i in range(seq_len)
                         if np.random.random() < self.mask_prob]
        if not mask_positions:
            mask_positions = [0]

        # Step 2: Generator produces replacements
        gen_logits = np.random.randn(len(mask_positions), self.vocab_size)
        replaced_ids = generator_sample(
            token_ids, mask_positions, gen_logits, self.vocab_size
        )

        # Step 3: Generator MLM loss
        true_at_masked = token_ids[mask_positions]
        gen_loss = mlm_generator_loss(gen_logits, true_at_masked)

        # Step 4: Discriminator classifies every position
        labels = discriminator_labels(token_ids, replaced_ids)
        disc_scores = np.random.randn(seq_len)
        disc_loss = discriminator_loss(disc_scores, labels)

        # Step 5: Combined loss
        total_loss = gen_loss + self.lam * disc_loss

        return {
            "gen_loss": gen_loss,
            "disc_loss": disc_loss,
            "total_loss": total_loss,
            "n_replaced": int(sum(labels == 0)),
            "n_original": int(sum(labels == 1)),
        }


def compare_training_signal():
    """Compare MLM vs ELECTRA training signal per sequence."""
    seq_len = 100
    mask_rate = 0.15

    mlm_signal_positions = int(seq_len * mask_rate)
    electra_signal_positions = seq_len

    print(f"Sequence length: {seq_len}")
    print(f"MLM training signal: {mlm_signal_positions} positions ({mask_rate:.0%})")
    print(f"ELECTRA training signal: {electra_signal_positions} positions (100%)")
    print(f"ELECTRA is {electra_signal_positions/mlm_signal_positions:.1f}x more efficient")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    compare_training_signal()

    trainer = ELECTRATrainer(vocab_size=100)
    tokens = np.random.randint(0, 100, 20)

    print("\nTraining step results:")
    result = trainer.train_step(tokens)
    for k, v in result.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
```

---

## Interview Importance

ELECTRA demonstrates an important principle: **efficiency innovations in the training objective can matter as much as scaling**. Good for showing you understand pre-training beyond just "make the model bigger."

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Why is replaced-token detection more compute-efficient than MLM?

**Answer:** Two reasons:
1. **100% token utilization:** In MLM, only 15% of positions contribute to the loss. In ELECTRA, every position is classified as original or replaced. This means 6.7× more training signal per sequence.
2. **Binary sigmoid vs. full softmax:** MLM computes a softmax over the entire vocabulary (30K-100K tokens) at each masked position. ELECTRA computes a binary sigmoid — vastly cheaper per position.

The net effect: ELECTRA-Small (14M params) matches BERT-Base (110M params) on GLUE benchmarks.

### Q2: What role does the small generator play — could it be shared with the discriminator?

**Answer:** The generator provides **plausible** replacements that force the discriminator to learn fine-grained language understanding. If replacements were random, the task would be too easy (random tokens are obviously wrong).

**Weight sharing** was tested: sharing all weights hurts because the generator needs to be weak enough to produce detectable fakes. If it's too good, the discriminator's task becomes impossible. The paper found using a generator 1/4 to 1/2 the discriminator's size works best.

### Q3: Compare ELECTRA to contrastive learning (e.g., SimCSE).

**Answer:**
- **ELECTRA:** Binary discrimination at token level — "is this token original or replaced?" Uses a generator to create hard negatives.
- **SimCSE/Contrastive:** Operates at sequence level — pull together similar sentences, push apart different ones. Negatives are other sequences in the batch.
- **Shared principle:** Both learn by distinguishing positive from negative examples rather than generating full outputs
- **Key difference:** ELECTRA's negatives are generated by a model (harder, more informative) vs. random in-batch negatives in contrastive learning

---

## Connections to Other Papers

- **BERT** → ELECTRA replaces MLM with a more efficient objective
- **RoBERTa** → Both show training innovations > architecture innovations
- **GPT-2** → Contrasting generative vs. discriminative pre-training
- **CLIP** → Another use of contrastive/discriminative objectives

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Replace MLM with replaced-token detection |
| Key advantage | 100% of tokens contribute to loss (vs. 15% in MLM) |
| Architecture | Small generator + large discriminator |
| Not a GAN | Generator uses MLE, no adversarial gradient |
| Efficiency | ELECTRA-Small ≈ GPT-Large at 1/10th compute |
| Discriminator output | Binary sigmoid per position (not full softmax) |
