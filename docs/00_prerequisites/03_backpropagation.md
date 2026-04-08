# Backpropagation and Gradient Descent

## Why This Matters for LLMs

**Backpropagation** is the algorithm that actually *fits* every large language model. Whether you are pre-training GPT-4, LLaMA, or a domain-specific model, the training loop is fundamentally: **forward pass → loss (usually cross-entropy over next-token predictions) → backward pass → optimizer step**, repeated over enormous corpora. At the scale of trillions of tokens, nothing mystical replaces this pattern—you are still differentiating a scalar loss with respect to millions or billions of parameters and using those gradients to update weights.

Understanding **computational graphs** and **gradient flow** is not optional for serious ML systems work. When loss spikes, gradients explode, or throughput collapses, interview-level debugging assumes you can reason about *which tensor depends on which*, where activations are stored, and how errors propagate backward through attention, layer normalization, and residual connections. The same mental model underpins **mixed-precision training** (which tensors are safe in `float16` / `bfloat16`), **gradient accumulation** (when to scale the loss and when to step the optimizer), and **checkpointing** (what to recompute vs store).

Finally, FAANG-style interviews often probe whether you can **derive or trace** a backward pass for a small network, explain **SGD vs momentum vs Adam** at the update-rule level, and connect **learning-rate schedules** (warmup, cosine decay) to training stability—the topics on this page.

## Core Concepts

### The Chain Rule (Single Variable)

If \(y = g(x)\) and \(z = f(y)\), then \(z\) depends on \(x\) through the composition \(f(g(x))\). The chain rule states:

\[
\frac{dz}{dx} = \frac{dz}{dy}\,\frac{dy}{dx}.
\]

Equivalently, with Leibniz notation: multiply derivatives along the path from \(z\) down to \(x\).

!!! math-intuition "In Plain English"
    When a small change \(\Delta x\) nudges \(y\), which then nudges \(z\), the **total** sensitivity of \(z\) to \(x\) is “how sensitive \(z\) is to \(y\)” times “how sensitive \(y\) is to \(x\).” Backpropagation is this idea, systematically applied on every edge of the graph.

!!! example "Worked Example: chain rule with numbers"
    Let \(g(x) = x^2 + 1\) and \(f(y) = 3y\), so \(z = f(g(x)) = 3(x^2 + 1)\).

    At \(x = 2\):

    - \(y = g(2) = 4 + 1 = 5\).
    - \(\dfrac{dy}{dx} = 2x = 4\) at \(x = 2\).
    - \(\dfrac{dz}{dy} = 3\).
    - \(\dfrac{dz}{dx} = \dfrac{dz}{dy}\,\dfrac{dy}{dx} = 3 \times 4 = 12\).

    Sanity check: \(z = 3x^2 + 3\), so \(\dfrac{dz}{dx} = 6x = 12\) at \(x = 2\), matching the chain rule.

### Computational Graphs

A **computational graph** represents a function as **nodes** (operations: add, matmul, ReLU, softmax, etc.) and **edges** (tensors flowing forward). The **forward pass** evaluates each node in topological order, materializing intermediate activations. The **backward pass** applies the chain rule in reverse, propagating \(\partial L/\partial (\cdot)\) from the loss \(L\) back to parameters.

\[
\text{Forward: } \mathbf{x} \rightarrow \cdots \rightarrow L; \qquad
\text{Backward: } \frac{\partial L}{\partial L}=1 \rightarrow \cdots \rightarrow \frac{\partial L}{\partial \mathbf{W}}.
\]

!!! math-intuition "In Plain English"
    Think of the graph as a pipeline: data flows **downstream** to compute \(L\), and gradient signals flow **upstream** from \(L\) to every parameter that influenced \(L\). If an edge does not exist on the forward path, that parameter gets **zero** gradient.

!!! example "Worked Example: tiny graph"
    Suppose \(L = (a + b)^2\) with \(a = 2\), \(b = 3\).

    - Forward: \(s = a + b = 5\), \(L = s^2 = 25\).
    - Backward from \(L\): \(\dfrac{\partial L}{\partial s} = 2s = 10\).
    - \(\dfrac{\partial s}{\partial a} = 1\), \(\dfrac{\partial s}{\partial b} = 1\), so \(\dfrac{\partial L}{\partial a} = \dfrac{\partial L}{\partial s}\dfrac{\partial s}{\partial a} = 10\), and likewise \(\dfrac{\partial L}{\partial b} = 10\).

### Backpropagation Algorithm

**Backpropagation** is automatic differentiation in reverse mode: for a scalar loss \(L\), compute \(\partial L/\partial w\) for every parameter \(w\) in one backward sweep.

**Procedure (conceptual):**

1. **Forward pass:** Evaluate the network, storing tensors needed for derivatives (pre-activations, activations).
2. **Compute loss** \(L\) (e.g., squared error for regression, cross-entropy for classification / LM).
3. **Backward pass:** Initialize \(\partial L/\partial L = 1\). Visit nodes in reverse topological order; for each operation, apply local Jacobian–vector products to propagate gradients to inputs and accumulate parameter gradients.

For a weight \(W\) in a linear layer \(\mathbf{z} = W\mathbf{x} + \mathbf{b}\), you eventually obtain \(\partial L/\partial W\) as an outer product of upstream gradient with the forward input (for appropriate shapes).

!!! math-intuition "In Plain English"
    Backprop is “push the blame backward”: the loss tells each parameter how much it contributed to the error, in the sense of a first-order (linear) approximation.

!!! example "Worked Example: 2-layer network with ReLU and squared loss"
    **Architecture.** Input \(\mathbf{x} \in \mathbb{R}^2\), one hidden layer with ReLU, scalar linear output. Parameters:

\[
W_1 = \begin{bmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \end{bmatrix}, \quad
\mathbf{b}_1 = \begin{bmatrix} 0.01 \\ 0.02 \end{bmatrix}, \quad
W_2 = \begin{bmatrix} 0.5 & 0.6 \end{bmatrix}, \quad
b_2 = 0.03.
\]

    Let \(\mathbf{x} = [1,\,2]^\top\), target \(y = 1\).

    **Forward**

\[
\mathbf{z}_1 = W_1 \mathbf{x} + \mathbf{b}_1
= \begin{bmatrix} 0.51 \\ 1.12 \end{bmatrix}, \quad
\mathbf{h} = \mathrm{ReLU}(\mathbf{z}_1) = \begin{bmatrix} 0.51 \\ 1.12 \end{bmatrix}.
\]

\[
z_2 = W_2 \mathbf{h} + b_2 = 0.5(0.51) + 0.6(1.12) + 0.03 = 0.957.
\]

    Squared loss:

\[
L = (z_2 - y)^2 = (0.957 - 1)^2 = 0.001849.
\]

    **Backward** (scalar output; \(\mathrm{ReLU}'(z) = \mathbb{1}_{z>0}\))

\[
\frac{\partial L}{\partial z_2} = 2(z_2 - y) = -0.086.
\]

\[
\frac{\partial L}{\partial W_2} = \frac{\partial L}{\partial z_2}\,\mathbf{h}^\top
= \begin{bmatrix} -0.04386 & -0.09632 \end{bmatrix}, \quad
\frac{\partial L}{\partial b_2} = -0.086.
\]

\[
\frac{\partial L}{\partial \mathbf{h}} = W_2^\top \frac{\partial L}{\partial z_2}
= \begin{bmatrix} -0.043 \\ -0.0516 \end{bmatrix}.
\]

    Since both entries of \(\mathbf{z}_1\) are positive, \(\partial L/\partial \mathbf{z}_1 = \partial L/\partial \mathbf{h}\).

\[
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \mathbf{z}_1}\,\mathbf{x}^\top
= \begin{bmatrix} -0.043 & -0.086 \\ -0.0516 & -0.1032 \end{bmatrix}, \quad
\frac{\partial L}{\partial \mathbf{b}_1} = \begin{bmatrix} -0.043 \\ -0.0516 \end{bmatrix}.
\]

    These are the values you should recover from a manual implementation or from PyTorch autograd for the same tensors.

### Stochastic Gradient Descent (SGD)

Given a loss \(L(w)\), **gradient descent** uses:

\[
w \leftarrow w - \eta\,\frac{\partial L}{\partial w},
\]

where \(\eta > 0\) is the **learning rate**.

!!! math-intuition "In Plain English"
    Each step nudges weights in the direction that **reduces** \(L\) fastest (steepest descent in the local linear model). The learning rate \(\eta\) sets how large that nudge is.

**Stochastic** GD uses a **mini-batch** of examples to estimate \(\partial L/\partial w\), trading lower per-step cost for noisier updates—often helpful for generalization and essential when the full dataset does not fit in one pass.

\[
\nabla_w L \approx \frac{1}{B}\sum_{i \in \mathrm{batch}} \nabla_w \ell_i(w).
\]

!!! math-intuition "In Plain English"
    A mini-batch gradient is an **average** of per-example gradients: cheaper than the full dataset each step, noisy but often a cheap, noisy compass that points roughly toward lower average loss.

!!! example "Worked Example: one SGD update"
    Scalar weight \(w = 2.0\), learning rate \(\eta = 0.1\), batch gives \(\dfrac{\partial L}{\partial w} = 0.4\).

\[
w_{\mathrm{new}} = 2.0 - 0.1 \times 0.4 = 2.0 - 0.04 = 1.96.
\]

    *In plain terms:* one SGD step subtracts \(\eta\) times the observed gradient from the current weight—here a positive gradient **decreases** \(w\) because we step opposite to \(\nabla L\).

    If the gradient were negative, the update would **increase** \(w\) (move opposite to the direction of steepest ascent).

### Momentum

**Momentum** maintains a velocity vector \(\mathbf{v}\) that accumulates past gradients, smoothing updates:

\[
\mathbf{v} \leftarrow \beta \mathbf{v} + \frac{\partial L}{\partial \mathbf{w}}, \qquad
\mathbf{w} \leftarrow \mathbf{w} - \eta\,\mathbf{v}.
\]

Here \(\beta \in [0,1)\) (e.g., \(0.9\)) controls how long past gradients influence the step. In valleys where the loss surface oscillates across a narrow gorge, momentum **dampens** left–right wobble; along a consistent downhill direction, \(\mathbf{v}\) **builds** and steps accelerate.

!!! math-intuition "In Plain English"
    Momentum is “inertia”: the optimizer remembers recent gradient directions instead of reacting jerkily to every mini-batch.

!!! example "Worked Example: two momentum steps"
    Scalar \(w = 1.0\), \(\eta = 0.1\), \(\beta = 0.9\), initial velocity \(v = 0\).

    - Step 1: gradient \(g_1 = 0.5\). Then \(v_1 = 0.9\cdot 0 + 0.5 = 0.5\), and \(w_1 = 1.0 - 0.1\cdot 0.5 = 0.95\).
    - Step 2: gradient \(g_2 = -0.3\). Then \(v_2 = 0.9\cdot 0.5 + (-0.3) = 0.15\), and \(w_2 = 0.95 - 0.1\cdot 0.15 = 0.935\).

    The second gradient points the other way, but velocity still carries memory of the first step.

### Adam Optimizer

**Adam** (Adaptive Moment Estimation) is the most popular optimizer for training Transformers. It combines two ideas:

1. **Momentum** (remembers past gradient directions)
2. **Adaptive learning rates** (each parameter gets its own step size based on recent gradient magnitudes)

**The core update (intuition first):**

\[
\mathbf{w}_t = \mathbf{w}_{t-1} - \eta\,\frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon}
\]

where:
- \(\mathbf{m}_t\) = smoothed average of recent gradients (like momentum)
- \(\mathbf{v}_t\) = smoothed average of recent **squared** gradients (tracks gradient magnitude per parameter)
- \(\epsilon\) = tiny constant (e.g., \(10^{-8}\)) to avoid division by zero

!!! math-intuition "In Plain English"
    Think of Adam as giving each parameter its own "smart learning rate":
    
    - If a parameter's gradients have been **large recently**, \(\mathbf{v}_t\) is large, so the learning rate gets **scaled down** (take smaller steps)
    - If a parameter's gradients have been **small recently**, \(\mathbf{v}_t\) is small, so the learning rate gets **scaled up** (take bigger steps)
    - The \(\mathbf{m}_t\) term ensures we keep moving in a consistent direction, smoothing out noisy mini-batch gradients

    This makes Adam robust across noisy, high-dimensional Transformer losses. That is why it (or close variants like AdamW with decoupled weight decay) is a **default** in many LLM recipes.

??? deep-dive "Deep Dive: Full Adam Update with Bias Correction"
    The complete Adam algorithm includes **bias correction** to handle the fact that \(\mathbf{m}\) and \(\mathbf{v}\) start at zero.
    
    **Step 1: Update moments**
    

\[
\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t, \qquad
\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t \odot \mathbf{g}_t.
\]

    
    Typical values: \(\beta_1 = 0.9\) (gradient memory), \(\beta_2 = 0.999\) (magnitude tracking).
    
    **Step 2: Bias correction**
    
    Because \(\mathbf{m}_0=\mathbf{0}\) and \(\mathbf{v}_0=\mathbf{0}\), the early estimates are biased toward zero. We correct this:
    

\[
\widehat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \qquad
\widehat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}.
\]

    
    !!! math-intuition "Why bias correction?"
        Early in training, EMAs are **biased toward zero** because they start at zero; dividing by \(1-\beta^t\) inflates them so the first steps are not artificially timid. At \(t=1\) with \(\beta_1=0.9\), we divide by \(1-0.9=0.1\), multiplying by 10!
    
    **Step 3: Parameter update**
    

\[
\mathbf{w}_t = \mathbf{w}_{t-1} - \eta\,\frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t} + \epsilon}
\quad \text{(elementwise)}.
\]

    
    !!! example "Worked Example: Adam for two steps (scalar)"
        Use \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\epsilon = 10^{-8}\), \(\eta = 0.01\), starting \(w = 1.0\), \(\mathbf{m}_0=\mathbf{v}_0=0\).
        
        **Step \(t=1\), gradient \(g_1 = 0.4\):**
        
        \[
        m_1 = 0.9\cdot 0 + 0.1\cdot 0.4 = 0.04, \quad v_1 = 0.999\cdot 0 + 0.001\cdot 0.16 = 0.00016.
        \]

        
        \[
        \widehat{m}_1 = \frac{0.04}{1-0.9} = 0.4, \quad
        \widehat{v}_1 = \frac{0.00016}{0.001} = 0.16.
        \]

        
        \[
        w \leftarrow 1.0 - 0.01\cdot\frac{0.4}{\sqrt{0.16}+\epsilon}
        = 1.0 - 0.01\cdot\frac{0.4}{0.4} = 0.99.
        \]

        
        **Step \(t=2\), gradient \(g_2 = -0.2\):**
        
        \[
        m_2 = 0.9\cdot 0.04 + 0.1\cdot(-0.2) = 0.016, \quad
        v_2 = 0.999\cdot 0.00016 + 0.001\cdot 0.04 = 0.00019984.
        \]

        
        \[
        \widehat{m}_2 = \frac{0.016}{1-0.81} \approx 0.08421, \quad
        \widehat{v}_2 = \frac{0.00019984}{1-0.998001} \approx 0.09997.
        \]

        
        \[
        w \leftarrow 0.99 - 0.01\cdot\frac{0.08421}{\sqrt{0.09997}+\epsilon}
        \approx 0.99 - 0.00266 \approx 0.9873.
        \]

        
        (Numerical values rounded for display; implementations keep full precision.)

### Learning Rate Schedules

LLM **pre-training** often combines **linear warmup** with **cosine decay**. Warmup gradually raises \(\eta\) from a small value to a peak; cosine annealing decays \(\eta\) smoothly toward a floor.

**Linear warmup** for steps \(t \in [0, T_{\mathrm{warm}})\):

\[
\eta(t) = \frac{t}{T_{\mathrm{warm}}}\,\eta_{\max}.
\]

!!! math-intuition "In Plain English"
    Linear warmup **ramps** the learning rate from zero (or a small floor) to \(\eta_{\max}\) over \(T_{\mathrm{warm}}\) steps so early updates stay small while the network and optimizer statistics stabilize.

**Cosine decay** from peak \(\eta_{\max}\) to \(\eta_{\min}\) over steps \(t \in [T_{\mathrm{warm}}, T]\):

\[
\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})
\left(1 + \cos\left(\pi\,\frac{t - T_{\mathrm{warm}}}{T - T_{\mathrm{warm}}}\right)\right).
\]

!!! math-intuition "In Plain English"
    The cosine term starts near \(1\) (so \(\eta \approx \eta_{\max}\) right after warmup) and ends near \(-1\) (so \(\eta \approx \eta_{\min}\)), giving a **smooth** decay without sudden jumps—common in published LLM schedules.

!!! example "Worked Example: warmup then cosine"
    Let \(T_{\mathrm{warm}} = 100\), \(T = 1000\), \(\eta_{\max} = 3\times 10^{-4}\), \(\eta_{\min} = 3\times 10^{-5}\).

    - At \(t = 50\): \(\eta(50) = \dfrac{50}{100}\eta_{\max} = 1.5\times 10^{-4}\) (half of peak).
    - At \(t = 100\) (end of warmup): \(\eta = \eta_{\max}\).
    - At \(t = 550\) (midpoint of cosine phase): \(\dfrac{t - T_{\mathrm{warm}}}{T - T_{\mathrm{warm}}} = \dfrac{450}{900} = 0.5\), \(\cos(\pi/2)=0\), so \(\eta = \eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min}) = \dfrac{\eta_{\max}+\eta_{\min}}{2}\).

## Deep Dive

??? deep-dive "Scaling tricks used in LLM training"
    **Gradient accumulation**  
    When a full batch does not fit in GPU memory, split a batch into **micro-batches**: run forward/backward on each micro-batch, **accumulate** \(\partial L/\partial w\) (and typically **average** or scale loss by \(1/\text{grad accumulation steps}\)), then apply **one** optimizer step. This **simulates** a larger batch size for gradient estimation without holding all activations at once.

    **Gradient checkpointing**  
    Forward pass normally stores activations for backprop. **Checkpointing** stores only a subset of layers’ activations; during backward, missing values are **recomputed** by a partial forward from the last checkpoint. You trade **extra compute** for **lower peak memory**—critical for long sequences and large models.

    **Mixed precision and loss scaling**  
    Training in `float16` / `bfloat16` speeds matmuls and reduces bandwidth; dynamic range is limited. **Loss scaling** multiplies the loss by a large factor before backward so small gradients are representable; gradients are **unscaled** before the optimizer step. **Automatic** loss scaling adjusts the factor when overflows/underflows are detected.

    **Why warmup for LLMs**  
    Early in training, gradients and activations can be **large relative to random initialization**. A small \(\eta\) during warmup prevents optimizer states (especially adaptive second moments) and weights from being thrown into a bad region before the loss landscape has “settled.” Warmup is standard in Transformer pre-training for **stability**, not just folklore.

## Code

The script below implements **manual backpropagation** for the same two-layer ReLU network as in the worked example, then repeats the computation with **PyTorch autograd** and prints gradients side by side. Run with `python 03_backpropagation_demo.py` (or paste into a single file).

```python
"""Manual backprop vs PyTorch autograd for a 2-layer ReLU MLP (scalar MSE loss)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def manual_two_layer_grads() -> dict[str, np.ndarray]:
    x = np.array([[1.0], [2.0]])  # (2, 1)
    y = 1.0
    w1 = np.array([[0.1, 0.2], [0.3, 0.4]])  # (2, 2)
    b1 = np.array([[0.01], [0.02]])  # (2, 1)
    w2 = np.array([[0.5, 0.6]])  # (1, 2)
    b2 = np.array([[0.03]])  # (1, 1)

    # Forward
    z1 = w1 @ x + b1
    h = np.maximum(0.0, z1)
    z2 = (w2 @ h + b2).item()
    loss = (z2 - y) ** 2

    # Backward — MSE L = (z2 - y)^2 => dL/dz2 = 2(z2 - y)
    dz2 = 2.0 * (z2 - y)
    dw2 = dz2 * h.T  # (1, 2)
    db2 = np.array([[dz2]])
    dh = w2.T * dz2  # (2, 1)
    dz1 = dh * (z1 > 0).astype(np.float64)
    dw1 = dz1 @ x.T
    db1 = dz1

    return {
        "loss": np.array(loss),
        "d_w1": dw1,
        "d_b1": db1,
        "d_w2": dw2,
        "d_b2": db2,
    }


def torch_two_layer_grads() -> dict[str, torch.Tensor]:
    x = torch.tensor([[1.0], [2.0]], dtype=torch.double, requires_grad=False)
    y = torch.tensor(1.0, dtype=torch.double)
    w1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.double, requires_grad=True)
    b1 = torch.tensor([[0.01], [0.02]], dtype=torch.double, requires_grad=True)
    w2 = torch.tensor([[0.5, 0.6]], dtype=torch.double, requires_grad=True)
    b2 = torch.tensor([[0.03]], dtype=torch.double, requires_grad=True)

    z1 = w1 @ x + b1
    h = torch.relu(z1)
    z2 = (w2 @ h + b2).squeeze()
    loss = (z2 - y) ** 2
    loss.backward()

    return {
        "loss": loss.detach(),
        "d_w1": w1.grad,
        "d_b1": b1.grad,
        "d_w2": w2.grad,
        "d_b2": b2.grad,
    }


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)
    torch.set_printoptions(precision=6, sci_mode=False)

    m = manual_two_layer_grads()
    t = torch_two_layer_grads()

    print("Loss (manual):", float(m["loss"]))
    print("Loss (torch): ", float(t["loss"].item()))
    print()

    for name in ["d_w1", "d_b1", "d_w2", "d_b2"]:
        mn = m[name]
        tt = t[name].detach().numpy()
        diff = np.max(np.abs(mn - tt))
        print(f"{name} manual:\n{mn}")
        print(f"{name} torch:\n{tt}")
        print(f"max |manual - torch|: {diff:.2e}\n")


if __name__ == "__main__":
    main()
```

Expected output: **max absolute differences on the order of \(10^{-16}\)** (floating-point noise), confirming that the manual Jacobian chain matches autograd.

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **State the chain rule for composed scalar functions and explain how backprop applies it.**  
       *Depth:* \(\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}\); reverse-mode AD multiplies Jacobian-vector products along edges from \(L\) to each parameter.

    2. **What is stored during the forward pass for a backward pass through a ReLU layer?**  
       *Depth:* Typically the pre-activation (or a boolean mask) to know where ReLU was active; zero input to ReLU gets zero gradient.

    3. **Write the SGD update and contrast full-batch vs mini-batch vs stochastic (one example).**  
       *Depth:* \(w \leftarrow w - \eta \nabla L\); mini-batch estimates the gradient with variance–compute tradeoffs.

    4. **Why does momentum help in ill-conditioned loss landscapes?**  
       *Depth:* Smooths oscillations across steep curved valleys; accelerates along consistent gradient directions.

    5. **Write Adam’s moment updates and explain bias correction.**  
       *Depth:* EMAs of \(g\) and \(g^2\); early steps are biased toward zero, so divide by \(1-\beta^t\).

    6. **Why is Adam often preferred over plain SGD for Transformers out of the box?**  
       *Depth:* Per-parameter adaptive scales + momentum-like behavior; still needs LR schedule and often AdamW-style decoupled weight decay.

    7. **What is gradient accumulation, and how must loss scaling interact with it?**  
       *Depth:* Sum/average gradients over micro-batches; loss is usually divided by accumulation steps so the effective gradient matches a larger batch.

    8. **Explain gradient checkpointing: what do you trade, and when is it worth it?**  
       *Depth:* Recompute activations vs store them; worth it when memory bounds sequence length or model width.

    9. **Why do LLM training recipes use learning-rate warmup?**  
       *Depth:* Large initial updates can destabilize training; small \(\eta\) early lets optimizer statistics and activations settle.

    10. **How does mixed-precision training interact with gradient magnitudes?**  
        *Depth:* FP16 has limited dynamic range; loss scaling preserves small gradient signals; GradScaler adjusts automatically in frameworks.

!!! interview "Follow-up Probes"
    1. **What happens to backprop if you detach a tensor from the graph in PyTorch?**  
       *Depth:* Gradients do not flow through detached tensors; useful for stopping credit assignment or freezing parts of the model.

    2. **Why might cosine decay be preferred over step decay for large pre-training runs?**  
       *Depth:* Smooth schedule; fewer discrete hyperparameter cliffs; common in published LM recipes.

    3. **Does Adam eliminate the need for learning-rate tuning?**  
       *Depth:* No—global \(\eta\), schedule, and weight decay still matter; Adam adapts per-parameter scales, not all optimization difficulty.

    4. **What is the difference between `float16` and `bfloat16` for training stability?**  
       *Depth:* `bfloat16` has wider exponent range, often more forgiving; `float16` may need more careful loss scaling.

    5. **How does batch normalization change the backward pass conceptually (even if LayerNorm is more common in LLMs)?**  
       *Depth:* Extra dependencies on batch statistics; gradients flow through mean/variance terms unless stopped; LayerNorm uses per-token statistics instead.

!!! key-phrases "Key Phrases to Use in Interviews"
    1. “Reverse-mode automatic differentiation: one forward, one backward, all parameter gradients.”
    2. “Mini-batch SGD: unbiased gradient estimate with controlled variance–compute tradeoff.”
    3. “Momentum smooths the path; Adam adapts step sizes using first and second moments.”
    4. “Warmup stabilizes early training; cosine decay anneals the learning rate smoothly.”
    5. “Gradient accumulation simulates large batches; checkpointing trades compute for memory.”
    6. “Loss scaling preserves small gradients in low-precision matmuls.”
    7. “The computational graph is the dependency structure for credit assignment.”

## References

1. Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). *Learning representations by back-propagating errors.* Nature.
2. Robbins, H., & Monro, S. (1951). *A stochastic approximation method.* The Annals of Mathematical Statistics.
3. Polyak, B. T. (1964). *Some methods of speeding up the convergence of iteration methods.* USSR Computational Mathematics and Mathematical Physics.
4. Kingma, D. P., & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR.
5. Loshchilov, I., & H. Hutter, F. (2017). *SGDR: Stochastic gradient descent with warm restarts.* ICLR. (cosine-style schedules)
6. Micikevicius, P., et al. (2018). *Mixed precision training.* arXiv:1710.03740.
7. Chen, T., et al. (2016). *Training deep nets with sublinear memory cost.* arXiv:1604.06174. (checkpointing)
