# Part 0 — Deep Learning Fundamentals

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

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [The Perceptron and Feedforward Networks](01_perceptron_and_ffn.md) | Single neuron, MLP, universal approximation, forward pass |
| 2 | [Activation Functions](02_activation_functions.md) | Sigmoid, tanh, ReLU, GELU, softmax — saturation, dying neurons, when to use each |
| 3 | [Backpropagation and Gradient Descent](03_backpropagation.md) | Chain rule, computational graphs, SGD, momentum, Adam, learning rate schedules |
| 4 | [Loss Functions and Regularization](04_loss_and_regularization.md) | MSE, cross-entropy, L1/L2 penalties, dropout, batch normalization |
| 5 | [Convolutional Neural Networks](05_convolutional_neural_networks.md) | Convolution operation, pooling, feature hierarchies, LeNet to ResNet, vision encoders |
| 6 | [Sequence Modeling and RNNs](06_sequence_modeling_and_rnns.md) | Why order matters, vanilla RNN intuition, limitations that motivate LSTM and attention |
| 7 | [The Encoder-Decoder Paradigm](07_encoder_decoder_paradigm.md) | Compress-then-generate pattern, information bottleneck, bridge to Transformers |

---

## How this connects to the rest of LLMBase

Part 0 gives you the vocabulary and intuition that Part 1 (Foundations) assumes. Once you are comfortable with backprop, activations, and the idea that a network maps inputs to outputs through differentiable layers, Part 1 will build on that with language-specific concepts: n-grams, word embeddings, LSTM gate equations, and attention.

Every page follows the same structure as the rest of LLMBase: plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions with expected answer depth.
