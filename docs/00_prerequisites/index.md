# Deep Learning Fundamentals

The building blocks that every neural network — including every Transformer and every LLM — is made of. This section exists so you never have to wave your hands when someone asks "but how does a neural network actually learn?"

---

## Goals

After completing Part 0 you will be able to:

- Explain how a single neuron computes a weighted sum, applies a nonlinearity, and produces output
- Trace forward and backward passes through a multi-layer network with actual numbers
- Compare activation functions (sigmoid, tanh, ReLU, GELU) and explain when each is preferred
- Derive the chain rule on a computational graph and connect it to gradient descent
- Explain cross-entropy loss and why it is the standard objective for classification and language modeling
- Describe how convolutions extract spatial features and why CNNs matter for vision encoders in multimodal LLMs
- Articulate the sequence modeling problem and why vanilla feedforward networks cannot solve it
- Draw the abstract encoder-decoder paradigm that underpins seq2seq, Transformers, and modern LLM architectures

---

## Before You Start

### Prerequisites

If any of these look unfamiliar, **start with the [Math Prerequisites](00_math_prerequisites.md)** page. It explains every symbol, notation, and mathematical concept used throughout this documentation, with worked numerical examples.

**You should be comfortable with:**
- High school algebra (solving equations, working with variables)
- Basic coordinate geometry (points, lines, slopes)
- What a function is (input → output mapping)

**We'll teach you:**
- Vector and matrix operations (with step-by-step examples)
- Derivatives and the chain rule (intuition-first, not proof-heavy)
- Probability basics (what distributions, expectations, and variance mean)

### Reading Strategy: Two Passes

This documentation is designed to be read **in two passes**:

**First Pass (Build Intuition):**
- Read the "Why This Matters" and "Core Concepts" sections
- Focus on the "In Plain English" callout boxes
- Work through the numerical examples
- Skip the "Deep Dive" sections on first reading
- Goal: Understand *what* each concept does and *why* it matters

**Second Pass (Deepen Understanding):**
- Re-read with the "Deep Dive" sections included
- Study the code implementations
- Attempt the interview questions
- Goal: Understand *how* to implement and *when* to apply each concept

!!! tip "What to Skip on First Reading"
    - "Deep Dive" collapsible sections (marked with ??? deep-dive)
    - Detailed optimizer derivations (Adam bias correction proofs)
    - Advanced regularization theory (KL divergence decompositions)
    - Come back to these after you've built intuition from the core concepts.

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 0 | [Math Prerequisites](00_math_prerequisites.md) | **Start here if math notation is unfamiliar** — vectors, matrices, derivatives, summation, log/exp, probability basics |
| 1 | [The Perceptron and Feedforward Networks](01_perceptron_and_ffn.md) | Single neuron, MLP, universal approximation, forward pass |
| 2 | [Activation Functions](02_activation_functions.md) | Sigmoid, tanh, ReLU, GELU, softmax — saturation, dying neurons, when to use each |
| 3 | [Backpropagation and Gradient Descent](03_backpropagation.md) | Chain rule, computational graphs, SGD, momentum, Adam, learning rate schedules |
| 4 | [Loss Functions and Regularization](04_loss_and_regularization.md) | MSE, cross-entropy, L1/L2 penalties, dropout, batch normalization |
| 5 | [Convolutional Neural Networks](05_convolutional_neural_networks.md) | Convolution operation, pooling, feature hierarchies, LeNet to ResNet, vision encoders |
| 6 | [Sequence Modeling and RNNs](06_sequence_modeling_and_rnns.md) | Why order matters, vanilla RNN intuition, limitations that motivate LSTM and attention |
| 7 | [The Encoder-Decoder Paradigm](07_encoder_decoder_paradigm.md) | Compress-then-generate pattern, information bottleneck, bridge to Transformers |

---

## How this connects to the rest of LLMBase

This section gives you the vocabulary and intuition that the Foundations section assumes. Once you are comfortable with backprop, activations, and the idea that a network maps inputs to outputs through differentiable layers, the Foundations section will build on that with language-specific concepts: n-grams, word embeddings, LSTM gate equations, and attention.

Every page follows the same structure as the rest of LLMBase: plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions with expected answer depth.
