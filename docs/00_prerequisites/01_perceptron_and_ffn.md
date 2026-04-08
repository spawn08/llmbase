# The Perceptron and Feedforward Networks

## Why This Matters for LLMs

Modern LLMs are built from **Transformers**, and every Transformer block is essentially **attention plus feedforward computation**. The self-attention sublayer mixes information across positions, but the **feedforward network (FFN)** sublayer is what applies a large, position-wise nonlinear transformation after mixing. If you cannot read a matrix multiply, a bias, and a composition of layers, you cannot read a Transformer block on paper—and you will struggle to reason about capacity, scaling laws, and why depth and width trade off the way they do.

Feedforward networks are also the universal template for how neural networks **approximate functions**: a stack of linear maps and nonlinearities. That template is the same whether you are classifying XOR with a tiny MLP or running a 70B-parameter model: the objects get bigger, but the local operations are the same. Solid intuition here transfers directly to interview questions about initialization, vanishing gradients, and why depth buys **compositional** representations.

Finally, interview loops at top tech companies routinely probe whether you can **trace a forward pass** with concrete numbers and connect equations to code. The perceptron and MLP are the cleanest place to practice that skill before attention and softmax add moving parts.

!!! tip "Notation Help"
    If symbols like \(\mathbf{x} \in \mathbb{R}^d\) or \(W \in \mathbb{R}^{m \times n}\) look unfamiliar, check the [Math Prerequisites](00_math_prerequisites.md) page first. It explains every notation with worked numerical examples.

## Core Concepts

### The Single Neuron (Perceptron)

A single neuron computes a **weighted sum** of its inputs, adds a **bias**, and passes the result through an **activation function** \(\sigma\). For input vector \(\mathbf{x} \in \mathbb{R}^d\), weights \(\mathbf{w} \in \mathbb{R}^d\), and bias \(b \in \mathbb{R}\):

\[
y = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \sigma\left(\sum_{i=1}^{d} w_i x_i + b\right).
\]

In words: each feature \(x_i\) is scaled by \(w_i\), the terms are summed, the bias shifts the sum, and \(\sigma\) introduces nonlinearity so the model is not limited to linear decision boundaries when \(\sigma\) is nonlinear (or when the neuron sits inside a deeper network). When two activations combine **elementwise** (as in gating), that product is written \(\mathbf{u} \odot \mathbf{v}\), distinct from the matrix product \(UV\).

!!! math-intuition "In Plain English"
    Think of the neuron as a **vote**: each input gets a signed weight (how much it pushes “yes” vs “no”), the bias is a default tilt, and the activation decides how strongly the neuron “fires.” Without \(\sigma\) (or without stacking layers), a single neuron only implements a linear separator in feature space.

!!! example "Worked Example: three inputs"
    Take \(d = 3\) with \(\mathbf{x} = [1,\, 2,\, 3]^\top\), \(\mathbf{w} = [0.5,\, -1,\, 0.25]^\top\), and \(b = 1\).

    **Step 1 — weighted sum plus bias**

    \[
    \mathbf{w} \cdot \mathbf{x} + b = (0.5)(1) + (-1)(2) + (0.25)(3) + 1 = 0.5 - 2 + 0.75 + 1 = 0.25.
    \]

    **Step 2 — activation (ReLU: \(\sigma(z)=\max(0,z)\))**

    \[
    y = \max(0,\, 0.25) = 0.25.
    \]

    If instead \(\sigma\) were the logistic sigmoid \(\sigma(z)=1/(1+e^{-z})\), then \(y \approx 0.562\) because the sigmoid maps the pre-activation \(0.25\) into \((0,1)\).

### From One Neuron to a Layer

To produce **multiple outputs at once**, stack many neurons in parallel. Each output dimension has its own weight row. For batch input \(\mathbf{x} \in \mathbb{R}^d\) (a single column vector) or more generally a batch matrix, a **fully connected layer** is:

\[
\mathbf{z} = W\mathbf{x} + \mathbf{b},
\]

where \(W \in \mathbb{R}^{m \times d}\), \(\mathbf{b} \in \mathbb{R}^m\), and \(\mathbf{z} \in \mathbb{R}^m\). Row \(j\) of \(W\) is the weight vector of the \(j\)-th neuron; the matrix-vector product computes all \(m\) weighted sums simultaneously.

In words: **one matrix multiply** replaces a loop of dot products, and the bias vector adds a per-neuron offset before any activation is applied elementwise.

!!! math-intuition "In Plain English"
    A layer is a **bundle of neurons** acting on the same input \(\mathbf{x}\). Each row of \(W\) defines one neuron’s weights; multiplying \(W\mathbf{x}\) computes all those dot products in parallel. This is the workhorse operation behind both MLP blocks and the linear projections inside attention and FFNs.

!!! example "Worked Example: two neurons, two inputs"
    Let \(\mathbf{x} = [2,\, -1]^\top\),

    \[
    W = \begin{bmatrix} 1 & 3 \\ -2 & 0.5 \end{bmatrix}, \quad \mathbf{b} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}.
    \]

    Then

    \[
    W\mathbf{x} + \mathbf{b}
    = \begin{bmatrix} 1\cdot 2 + 3\cdot (-1) \\ -2\cdot 2 + 0.5\cdot (-1) \end{bmatrix}
    + \begin{bmatrix} 0 \\ 1 \end{bmatrix}
    = \begin{bmatrix} -1 \\ -4.5 \end{bmatrix}
    + \begin{bmatrix} 0 \\ 1 \end{bmatrix}
    = \begin{bmatrix} -1 \\ -3.5 \end{bmatrix}.
    \]

    If the layer uses ReLU, the post-activation is \(\max(0, z)\) per coordinate: \([0,\, 0]\).

### Multi-Layer Perceptron (MLP)

An **MLP** chains layers: linear map, activation, linear map, activation, and so on. A two-hidden-layer style composition (using \(f_1, f_2\) for activation bundles) can be written as:

\[
\mathbf{h} = f_2\!\left(W_2\, f_1(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2\right).
\]

Here \(f_1\) and \(f_2\) are applied **elementwise** (or include a final softmax for classification). Depth matters because each layer can transform the representation of \(\mathbf{x}\) before the next linear map sees it.

In words: the network is a **nested sequence** of affine maps and nonlinearities; the nonlinearities let the model bend what would otherwise be a single large linear map.

!!! math-intuition "In Plain English"
    An MLP is “linear → nonlinearity → linear → nonlinearity …” stacked. Each block can warp the input space so the next linear layer operates on **features** that are easier to separate or regress. This is the same structural idea as a Transformer FFN, just with different dimensions and often GELU instead of ReLU.

!!! example "Worked Example: 2 inputs, 2 hidden units, 1 output"
    Use ReLU for \(f_1\) and **identity** for the output (a scalar linear readout) so we can read the number cleanly.

    **Inputs**

    \[
    \mathbf{x} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}.
    \]

    **First layer**

    \[
    W_1 = \begin{bmatrix} 2 & -1 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0.5 \\ -0.5 \end{bmatrix}.
    \]

    Pre-activation:

    \[
    W_1 \mathbf{x} + \mathbf{b}_1
    = \begin{bmatrix} 2\cdot 1 + (-1)\cdot 0 + 0.5 \\ 1\cdot 1 + 3\cdot 0 - 0.5 \end{bmatrix}
    = \begin{bmatrix} 2.5 \\ 0.5 \end{bmatrix}.
    \]

    Hidden activations:

    \[
    \mathbf{h} = f_1(W_1 \mathbf{x} + \mathbf{b}_1) = \mathrm{ReLU}\!\left(\begin{bmatrix} 2.5 \\ 0.5 \end{bmatrix}\right) = \begin{bmatrix} 2.5 \\ 0.5 \end{bmatrix}.
    \]

    **Second layer (1 output)**

    \[
    W_2 = \begin{bmatrix} 1 & 2 \end{bmatrix}, \quad b_2 = -1.
    \]

    Scalar logit:

    \[
    z = W_2 \mathbf{h} + b_2 = 1\cdot 2.5 + 2\cdot 0.5 - 1 = 2.5 + 1 - 1 = 2.5.
    \]

    With identity output, \(y = 2.5\). If this were binary classification, you might apply sigmoid to \(z\) or use cross-entropy on logits.

### Universal Approximation Theorem

Informally: under mild conditions, a **single hidden layer** feedforward network with enough neurons can approximate **any continuous function** on a compact domain, provided you use a reasonable nonlinearity (for example sigmoid or ReLU-style activations in practical settings).

In words: shallow networks can be **universally expressive** in a function-space sense, which is part of why neural networks are a credible modeling class at all.

Why it matters: it gives a **theoretical reason** that neural nets are not obviously doomed—nonlinear feature learning plus width can represent rich maps.

Limitation: the theorem is **existence**, not a recipe. It does not tell you how many neurons you need for a given accuracy, nor how to **optimize** weights from data, nor how depth changes sample efficiency or optimization dynamics. Modern practice often prefers **deeper** models for representation and optimization reasons even when width alone is theoretically sufficient in the limit.

!!! math-intuition "In Plain English"
    Imagine you can dial the number of hidden units. Universal approximation says that, if you dial it high enough, you can match a target continuous function arbitrarily well on a bounded region. It does **not** say training will find those weights easily, or that a one-hidden-layer net is the best way to learn it in practice.

!!! example "Worked Example: what the theorem does *not* specify"
    Suppose you want to approximate \(f(x)=x^2\) on \([-1,1]\). The theorem says there exists some one-hidden-layer network \(g(x)\) with enough hidden units such that \(|g(x)-x^2|\) is tiny for all \(x\) in that interval. It does **not** tell you whether 10 or 10,000 units are enough for \(\varepsilon = 0.01\), and it does not construct the weights—you only know they exist in principle.

### The Feedforward Network Inside Every Transformer

Transformer blocks typically apply attention, then an **FFN** at each position. A common form (up to layout conventions) is a two-layer MLP with a middle nonlinearity:

\[
\mathrm{FFN}(\mathbf{x}) = \big(\sigma(\mathbf{x} W_1 + \mathbf{b}_1)\big) W_2 + \mathbf{b}_2,
\]

where \(\sigma(z)=\max(0,z)\) applied **elementwise** is the ReLU case, matching \(\max(0, \cdot)\) on the pre-activations. In many LLMs the nonlinearity is **GELU** rather than ReLU, but structurally it is still “expand dimension → nonlinearity → project back.”

In words: after attention has mixed tokens, the FFN applies the **same MLP pattern** independently at each position, processing each token’s vector with two linear transforms and a nonlinearity. That is why feedforward intuition transfers directly: the “LLM block” still contains plain affine maps and activations.

!!! math-intuition "In Plain English"
    Attention routes information between positions; the FFN **processes** each position’s vector locally. So every Transformer layer is “communication (attention) + computation (FFN),” repeated. If you understand \(W\mathbf{x}+\mathbf{b}\) and elementwise \(\sigma\), you already understand the computational core of the FFN—only the shapes and activation choices change.

!!! example "Worked Example: match tensors to the equation (tiny dimensions)"
    Use row-vector layout \(\mathbf{x} \in \mathbb{R}^{1 \times d}\) multiplying \(W_1 \in \mathbb{R}^{d \times d_{\mathrm{ff}}}\), matching common frameworks. Take \(d = 2\), \(d_{\mathrm{ff}} = 2\):

    \[
    \mathbf{x} = \begin{bmatrix} 1 & -1 \end{bmatrix}, \quad
    W_1 = \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}, \quad \mathbf{b}_1 = \begin{bmatrix} 0 & 0 \end{bmatrix}.
    \]

    Inner pre-activation (before ReLU):

    \[
    \mathbf{x} W_1 + \mathbf{b}_1
    = \begin{bmatrix} 1 & -1 \end{bmatrix}
    \begin{bmatrix} 1 & 2 \\ 0 & -1 \end{bmatrix}
    = \begin{bmatrix} 1 & 3 \end{bmatrix}.
    \]

    Apply ReLU to each coordinate: \(\mathbf{h}_i = \max(0, z_i)\).

    \[
    \mathbf{h} = \begin{bmatrix} 1 & 3 \end{bmatrix}.
    \]

    Let \(W_2 \in \mathbb{R}^{2 \times 1}\) and \(b_2 \in \mathbb{R}\):

    \[
    W_2 = \begin{bmatrix} 0.5 \\ 1 \end{bmatrix}, \quad b_2 = 0.
    \]

    Output scalar at this position:

    \[
    \mathrm{FFN}(\mathbf{x}) = \mathbf{h} W_2 + b_2 = 1\cdot 0.5 + 3\cdot 1 + 0 = 3.5.
    \]

    In full-sized models, \(d_{\mathrm{ff}} \gg d\) (expand–contract), and the activation may be GELU instead of ReLU, but the affine–nonlinear–affine pattern is unchanged.

## Deep Dive

??? deep-dive "Deep Dive: width vs depth, compositional features, and initialization"
    **Width vs depth.** Width (more units per layer) increases the capacity of a single layer’s transformations; depth (more layers) increases the ability to build **hierarchical** features where later layers operate on increasingly abstract representations. Very wide shallow nets can be expressive, but deep nets often generalize better on complex data because structure can be **factorized** across layers.

    **Why depth helps (compositional learning).** Deep stacks encourage **composition**: early layers can detect simple patterns, later layers combine them into parts, objects, or linguistic constructs. This is not guaranteed by architecture alone—data and training matter—but it explains why adding layers is often more effective than simply widening one layer when the target function has compositional structure.

    **Initialization: Xavier/Glorot and He/Kaiming.** Random initialization sets the scale of activations and gradients at the start of training. **Xavier/Glorot** initialization is designed to keep variance roughly stable across layers for **tanh/sigmoid**-like activations in many settings. **He/Kaiming** initialization is tailored for **ReLU** families, accounting for how ReLU zeros half the activations on average. Poor initialization can make activations explode or vanish, making optimization slow or unstable—especially in deep networks.

    **Why this matters for LLMs.** Large models are extremely deep and wide; stable signal propagation at initialization is part of why training is tractable at all. Modern Transformer training stacks careful initialization, normalization (often LayerNorm), and residual connections—each interacts with the raw affine maps \(W\mathbf{x}+\mathbf{b}\) you already understand.

## Code

The XOR problem is not linearly separable: a single neuron cannot solve it, but a small MLP with a hidden layer can. The tensors below map directly to \(\mathbf{z}_1 = W_1\mathbf{x}+\mathbf{b}_1\), \(\mathbf{h}=\mathrm{ReLU}(\mathbf{z}_1)\), and \(\mathbf{z}_2 = W_2\mathbf{h}+\mathbf{b}_2\).

```python
import torch
import torch.nn as nn


class XORMLP(nn.Module):
    def __init__(self, hidden: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


def main() -> None:
    torch.manual_seed(0)
    xs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    ys = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    model = XORMLP(hidden=8)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for step in range(2000):
        opt.zero_grad()
        pred = model(xs)
        loss = loss_fn(pred, ys)
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"step {step:4d}  loss={loss.item():.6f}")

    with torch.no_grad():
        out = model(xs)
    print("predictions:", out.squeeze().tolist())
    print("targets:    ", ys.squeeze().tolist())


if __name__ == "__main__":
    main()
```

Running the script should drive loss down and produce predictions close to \(0\) and \(1\) on the XOR pattern.

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What does a single neuron compute, and why is the bias term needed?**  
       *Depth:* Weighted sum \(\mathbf{w}\cdot\mathbf{x}\), plus bias \(b\), then activation; bias shifts the decision boundary so it need not pass through the origin.

    2. **Why can’t a single-layer perceptron solve XOR?**  
       *Depth:* XOR is not linearly separable in the original 2D input space; a single neuron defines a half-plane separator.

    3. **Write the matrix form of a fully connected layer and state tensor shapes for batch size \(B\).**  
       *Depth:* \(\mathbf{Z} = XW^\top + \mathbf{b}\) in common PyTorch layout with \(X\in\mathbb{R}^{B\times d}\); each row is one example.

    4. **What changes from one neuron to a layer?**  
       *Depth:* A layer stacks many neurons; weights become rows of \(W\), biases become a vector, activations apply elementwise.

    5. **State the universal approximation theorem in one sentence and name one limitation.**  
       *Depth:* A sufficiently wide one-hidden-layer network can approximate continuous functions on compact sets; limitation is non-constructive and silent on optimization and sample complexity.

    6. **What is the role of the FFN inside a Transformer block?**  
       *Depth:* Position-wise MLP after attention; processes each token vector with two linear maps and a nonlinearity (often GELU in LLMs).

    7. **Explain width vs depth tradeoffs at a high level.**  
       *Depth:* Width increases per-layer expressivity; depth enables hierarchical composition; very deep models often train better on structured data than one enormous hidden layer.

    8. **Why are Xavier and He initialization different?**  
       *Depth:* They target different activation statistics; He accounts for ReLU sparsity; Xavier suits bounded activations like tanh/sigmoid in classical analysis.

    9. **What problem does elementwise nonlinearity solve that stacked linear layers cannot?**  
       *Depth:* Composition of linear maps is linear; nonlinearities enable curved decision boundaries and universal approximation.

    10. **How does ReLU act elementwise on a hidden vector?**  
        *Depth:* \(\mathrm{ReLU}(\mathbf{z})\) applies \(\max(0,z_i)\) per coordinate; not the same as a dot product or matrix multiply.

!!! interview "Follow-up Probes"
    1. **What happens if you remove all nonlinearities from an MLP?**  
       *Depth:* The entire network collapses to one linear map (of reduced rank), no matter how many “layers” you stack.

    2. **Why is XOR a standard sanity check for MLP code?**  
       *Depth:* It requires hidden layers; if training fails trivially, shapes, loss, or optimization are likely wrong.

    3. **How does LayerNorm interact with the FFN in Transformers?**  
       *Depth:* Normalization stabilizes activations/residual streams; FFN still applies \(W\) and nonlinearity on normalized vectors within the block’s pre/post-norm design.

    4. **Does universal approximation imply trainability?**  
       *Depth:* No—existence of weights does not mean SGD will find them or that finite data suffices.

    5. **Why might GELU replace ReLU in some LLM FFNs?**  
       *Depth:* Smoother, probabilistic intuition; empirical gains in some regimes; still a scalar nonlinearity applied elementwise.

!!! key-phrases "Key Phrases to Use in Interviews"
    1. “Affine map then elementwise nonlinearity—repeat.”
    2. “Attention mixes positions; the FFN is position-wise computation.”
    3. “Stacking linear layers without nonlinearities collapses to a single linear map.”
    4. “Universal approximation is about existence of weights, not efficient learnability.”
    5. “Initialization controls activation and gradient scale at the start of training.”
    6. “Depth enables compositional features; width scales instantaneous basis capacity.”
    7. “The Transformer FFN is structurally a two-layer MLP applied per token.”

## References

1. Cybenko, G. (1989). *Approximation by superpositions of a sigmoidal function.* Mathematics of Control, Signals and Systems.
2. Hornik, K., Stinchcombe, M., & White, H. (1989). *Multilayer feedforward networks are universal approximators.* Neural Networks.
3. Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks.* AISTATS.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.* ICCV.
5. Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS. (FFN block in Transformer)
