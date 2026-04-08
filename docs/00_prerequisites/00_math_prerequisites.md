# Mathematics Prerequisites

Before diving into the deep learning fundamentals, let's build a shared vocabulary of the mathematical notation and concepts used throughout this documentation. **You don't need to be a math expert**—this section explains every symbol and equation you'll encounter, with worked examples using actual numbers.

---

## How to Read This Documentation

### Notation Conventions

We use consistent notation throughout. Here's what each symbol means:

| Symbol | Meaning | Example |
|--------|---------|---------|
| \(x\), \(y\), \(z\) | Scalars (single numbers) | \(x = 3.14\), \(y = -2\) |
| \(\mathbf{x}\), \(\mathbf{y}\) | **Vectors** (lists of numbers) | \(\mathbf{x} = [1, 2, 3]\) |
| \(W\), \(X\), \(Y\) | **Matrices** (2D grids of numbers) | \(W\) is a weight matrix |
| \(\mathbb{R}\) | The set of **real numbers** (all decimal numbers) | \(x \in \mathbb{R}\) means "\(x\) is a real number" |
| \(\mathbb{R}^d\) | A **\(d\)-dimensional vector** | \(\mathbf{x} \in \mathbb{R}^3\) means \(\mathbf{x}\) has 3 numbers |
| \(\mathbb{R}^{m \times n}\) | An **\(m \times n\) matrix** | \(W \in \mathbb{R}^{2 \times 3}\) has 2 rows, 3 columns |

!!! example "Understanding \(\mathbf{x} \in \mathbb{R}^d\)"
    Let's break down \(\mathbf{x} \in \mathbb{R}^d\) step by step:
    
    - \(\mathbb{R}\) = "real numbers" (like 1, -3.14, 0.001, \(\pi\), etc.)
    - \(\mathbb{R}^d\) = "a list of \(d\) real numbers"
    - \(\mathbf{x} \in \mathbb{R}^d\) = "\(\mathbf{x}\) is a list of \(d\) real numbers"
    
    **Concrete example**: If \(d = 3\), then \(\mathbf{x} \in \mathbb{R}^3\) could be:
    

\[
\mathbf{x} = \begin{bmatrix} 1.5 \\ -2.0 \\ 3.14 \end{bmatrix}
\]

    
    This is a **column vector** with 3 numbers stacked vertically. In code, this is like a list or array: `[1.5, -2.0, 3.14]`.

---

## 1. Vectors and Matrices

### What is a Vector?

A **vector** is just an ordered list of numbers. Think of it as a point in space, or a set of features describing something.

\[
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ \vdots \\ x_d \end{bmatrix} \in \mathbb{R}^d
\]

- \(x_1\) is the **first element** (position 1)
- \(x_2\) is the **second element** (position 2)
- \(x_d\) is the **last element** (position \(d\))

!!! example "Real-world example: describing a house"
    You could represent a house as a vector in \(\mathbb{R}^4\):
    

\[
\text{house} = \begin{bmatrix} 3 \\ 2.5 \\ 1800 \\ 2015 \end{bmatrix}
\]

    
    - \(x_1 = 3\) bedrooms
    - \(x_2 = 2.5\) bathrooms
    - \(x_3 = 1800\) square feet
    - \(x_4 = 2015\) year built
    
    Each number is a **feature**. The vector bundles them together.

### What is a Matrix?

A **matrix** is a 2D grid of numbers, like a spreadsheet:

\[
W = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \in \mathbb{R}^{2 \times 3}
\]

- This matrix has **2 rows** and **3 columns**
- We write \(W \in \mathbb{R}^{2 \times 3}\) to say "W is a 2-by-3 matrix"
- Element \(W_{i,j}\) means "row \(i\), column \(j\)" (e.g., \(W_{1,2} = 2\))

### Vector Operations

#### Addition

Add vectors **element by element** (they must have the same size):

\[
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} + \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1+4 \\ 2+5 \\ 3+6 \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \\ 9 \end{bmatrix}
\]

#### Scalar Multiplication

Multiply every element by a single number (a **scalar**):

\[
3 \times \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} = \begin{bmatrix} 3 \\ 6 \\ 9 \end{bmatrix}
\]

#### Dot Product (Inner Product)

The **dot product** multiplies matching elements and sums them up:

\[
\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^{d} w_i x_i = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d
\]

!!! example "Dot product with numbers"
    Let \(\mathbf{w} = [0.5, -1, 0.25]\) and \(\mathbf{x} = [1, 2, 3]\):
    

\[
\mathbf{w} \cdot \mathbf{x} = (0.5)(1) + (-1)(2) + (0.25)(3) = 0.5 - 2 + 0.75 = -0.75
\]

    
    **What this means**: The dot product measures how "aligned" two vectors are. If they point in similar directions, the result is large and positive. If they point in opposite directions, it's negative. If they're perpendicular, it's zero.

### Matrix-Vector Multiplication

This is the **core operation** in neural networks. When a matrix \(W\) multiplies a vector \(\mathbf{x}\):

\[
\mathbf{z} = W\mathbf{x}
\]

Each **row** of \(W\) does a dot product with \(\mathbf{x}\):

!!! example "Matrix-vector multiplication step by step"
    Let \(W = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}\) and \(\mathbf{x} = \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix}\):
    

\[
W\mathbf{x} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \\ -1 \end{bmatrix} = \begin{bmatrix} (1)(1) + (2)(0) + (3)(-1) \\ (4)(1) + (5)(0) + (6)(-1) \end{bmatrix} = \begin{bmatrix} -2 \\ -2 \end{bmatrix}
\]

    
    - **Row 1** \(\cdot\) \(\mathbf{x}\) = \(1 + 0 - 3 = -2\) → first output
    - **Row 2** \(\cdot\) \(\mathbf{x}\) = \(4 + 0 - 6 = -2\) → second output
    
    **Think of it this way**: The matrix \(W\) transforms the input vector \(\mathbf{x}\) into a new vector. Each row of \(W\) is a different "lens" or "filter" looking at the input.

---

## 2. Functions and Derivatives

### What is a Function?

A **function** \(f\) takes an input \(x\) and produces an output \(y\):

\[
y = f(x)
\]

!!! example "Common functions"
    - \(f(x) = x^2\): squares the input (\(f(3) = 9\))
    - \(f(x) = 2x + 1\): doubles and adds 1 (\(f(3) = 7\))
    - \(\sigma(x) = \frac{1}{1 + e^{-x}}\) (sigmoid): squishes any number to \((0, 1)\)

### What is a Derivative?

The **derivative** \(\frac{df}{dx}\) measures **how sensitive** the output is to small changes in the input:

\[
\frac{df}{dx} = \text{rate of change of } f \text{ with respect to } x
\]

!!! math-intuition "In Plain English"
    The derivative answers: **"If I nudge \(x\) a tiny bit, how much does \(f(x)\) change?"**
    
    - If \(\frac{df}{dx} = 3\): a small increase in \(x\) causes **3x bigger** increase in \(f(x)\)
    - If \(\frac{df}{dx} = -2\): a small increase in \(x\) causes **2x bigger decrease** in \(f(x)\)
    - If \(\frac{df}{dx} = 0\): changing \(x\) barely affects \(f(x)\) at all

!!! example "Derivatives with numbers"
    Let \(f(x) = x^2\). The derivative is \(\frac{df}{dx} = 2x\).
    
    - At \(x = 3\): \(\frac{df}{dx} = 2(3) = 6\)
      - Nudge \(x\) from 3 to 3.01 (change of 0.01)
      - \(f(x)\) changes from 9 to ≈ 9.06 (change of 0.06, which is \(6 \times 0.01\))
    
    - At \(x = -1\): \(\frac{df}{dx} = 2(-1) = -2\)
      - Nudge \(x\) from -1 to -0.99 (change of +0.01)
      - \(f(x)\) changes from 1 to ≈ 0.98 (change of -0.02, which is \(-2 \times 0.01\))

### Common Derivative Rules

| Function \(f(x)\) | Derivative \(\frac{df}{dx}\) |
|-------------------|------------------------------|
| \(c\) (constant) | \(0\) |
| \(cx\) | \(c\) |
| \(x^n\) | \(nx^{n-1}\) |
| \(e^x\) | \(e^x\) |
| \(\ln(x)\) | \(\frac{1}{x}\) |

### The Chain Rule

The **chain rule** is the most important derivative rule for neural networks. If you have **nested functions**:

\[
z = f(y) \quad \text{and} \quad y = g(x) \quad \Rightarrow \quad z = f(g(x))
\]

Then:

\[
\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}
\]

!!! example "Chain rule with numbers"
    Let \(y = x^2 + 1\) and \(z = 3y\). What is \(\frac{dz}{dx}\) at \(x = 2\)?
    
    **Step 1:** Find intermediate values
    - At \(x = 2\): \(y = 2^2 + 1 = 5\)
    - \(\frac{dy}{dx} = 2x = 4\) at \(x = 2\)
    - \(\frac{dz}{dy} = 3\) (constant)
    
    **Step 2:** Apply chain rule
    - \(\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} = 3 \times 4 = 12\)
    
    **Check:** \(z = 3(x^2 + 1) = 3x^2 + 3\), so \(\frac{dz}{dx} = 6x = 12\) at \(x = 2\). ✓

---

## 3. Summation and Product Notation

### Summation (\(\Sigma\))

The capital sigma \(\Sigma\) means "add up a sequence of terms":

\[
\sum_{i=1}^{n} a_i = a_1 + a_2 + a_3 + \cdots + a_n
\]

!!! example "Summation with numbers"

\[
\sum_{i=1}^{4} i^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
\]

### Product (\(\Pi\))

The capital pi \(\Pi\) means "multiply a sequence of terms":

\[
\prod_{i=1}^{n} a_i = a_1 \times a_2 \times a_3 \times \cdots \times a_n
\]

!!! example "Product with numbers"

\[
\prod_{i=1}^{4} i = 1 \times 2 \times 3 \times 4 = 24
\]

---

## 4. Exponentials and Logarithms

### The Natural Exponential \(e^x\)

The number \(e \approx 2.71828\) is a mathematical constant. The function \(e^x\) (or \(\exp(x)\)) grows rapidly:

- \(e^0 = 1\)
- \(e^1 \approx 2.718\)
- \(e^2 \approx 7.389\)
- \(e^{-1} \approx 0.368\) (negative exponent = reciprocal)

### The Natural Logarithm \(\ln(x)\) or \(\log(x)\)

The **natural log** is the inverse of \(e^x\):

\[
\ln(e^x) = x \quad \text{and} \quad e^{\ln(x)} = x
\]

Key properties:

| Property | Formula | Example |
|----------|---------|---------|
| Log of product | \(\ln(ab) = \ln(a) + \ln(b)\) | \(\ln(6) = \ln(2) + \ln(3)\) |
| Log of quotient | \(\ln(a/b) = \ln(a) - \ln(b)\) | \(\ln(2) = \ln(6) - \ln(3)\) |
| Log of power | \(\ln(a^b) = b \ln(a)\) | \(\ln(8) = 3 \ln(2)\) |
| Log of 1 | \(\ln(1) = 0\) | Always |

!!! math-intuition "Why logs matter in ML"
    Logarithms convert **multiplication into addition**, which is easier to compute and more numerically stable. In machine learning, we often work with **products of probabilities** (which are tiny numbers), so we take logs to convert them into **sums of log-probabilities**.

---

## 5. Probability Basics

### Random Variables and Distributions

A **random variable** \(X\) can take different values with different probabilities. A **probability distribution** tells us how likely each value is.

!!! example "Fair die roll"
    Let \(X\) be the outcome of rolling a fair 6-sided die:
    

\[
P(X = 1) = P(X = 2) = \cdots = P(X = 6) = \frac{1}{6}
\]

    
    Each outcome has probability \(\frac{1}{6}\).

### Probability Mass Function (Discrete)

For discrete outcomes, \(P(X = x)\) is the probability that \(X\) equals \(x\). Two rules:

1. **Non-negative**: \(P(X = x) \geq 0\) for all \(x\)
2. **Sums to 1**: \(\sum_x P(X = x) = 1\)

### Expected Value

The **expected value** \(\mathbb{E}[X]\) is the "average" value you'd expect over many trials:

\[
\mathbb{E}[X] = \sum_x x \cdot P(X = x)
\]

!!! example "Expected value of a die roll"

\[
\mathbb{E}[X] = 1\cdot\frac{1}{6} + 2\cdot\frac{1}{6} + 3\cdot\frac{1}{6} + 4\cdot\frac{1}{6} + 5\cdot\frac{1}{6} + 6\cdot\frac{1}{6} = 3.5
\]

### Variance

The **variance** \(\mathrm{Var}(X)\) measures how "spread out" values are around the mean:

\[
\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
\]

- **Low variance**: values cluster tightly around the mean
- **High variance**: values are spread out

---

## 6. Elementwise Operations

When we apply a function **elementwise** to a vector, we apply it to each element independently:

\[
\mathrm{ReLU}\!\left(\begin{bmatrix} -1 \\ 2 \\ -3 \end{bmatrix}\right) = \begin{bmatrix} \max(0, -1) \\ \max(0, 2) \\ \max(0, -3) \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \\ 0 \end{bmatrix}
\]

In neural network notation, when you see \(\sigma(\mathbf{z})\) where \(\mathbf{z}\) is a vector, it means \(\sigma\) is applied to **each element** of \(\mathbf{z}\).

---

## 7. The Hadamard (Elementwise) Product

The symbol \(\odot\) denotes **elementwise multiplication** of two vectors of the same size:

\[
\mathbf{u} \odot \mathbf{v} = \begin{bmatrix} u_1 \\ u_2 \\ u_3 \end{bmatrix} \odot \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} = \begin{bmatrix} u_1 v_1 \\ u_2 v_2 \\ u_3 v_3 \end{bmatrix}
\]

!!! example "Hadamard product"

\[
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \odot \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 4 \\ 10 \\ 18 \end{bmatrix}
\]

    
    Contrast with the **dot product** \(\mathbf{u} \cdot \mathbf{v}\), which produces a **scalar** (single number), while \(\mathbf{u} \odot \mathbf{v}\) produces a **vector**.

---

## 8. Gradients and Partial Derivatives

When a function depends on **multiple variables**, we can ask how sensitive it is to each one individually.

### Partial Derivative

The **partial derivative** \(\frac{\partial f}{\partial x}\) measures how \(f\) changes when we nudge \(x\), **holding all other variables constant**.

!!! example "Partial derivative"
    Let \(f(x, y) = x^2 + 3y\).
    
    - \(\frac{\partial f}{\partial x} = 2x\) (treat \(y\) as constant)
    - \(\frac{\partial f}{\partial y} = 3\) (treat \(x\) as constant)

### Gradient

The **gradient** \(\nabla f\) is a vector containing **all partial derivatives**:

\[
\nabla_{\mathbf{w}} L = \begin{bmatrix} \frac{\partial L}{\partial w_1} \\ \frac{\partial L}{\partial w_2} \\ \vdots \\ \frac{\partial L}{\partial w_d} \end{bmatrix}
\]

!!! math-intuition "In Plain English"
    The gradient points in the direction of **steepest increase** of the function. In machine learning, we go in the **opposite direction** (gradient descent) to **minimize** the loss:
    

\[
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \nabla_{\mathbf{w}} L
\]

    
    where \(\eta\) is the learning rate (step size).

---

## Quick Reference Card

Print this out or bookmark it for easy reference while reading:

| Concept | Notation | Meaning |
|---------|----------|---------|
| Scalar | \(x\), \(y\) | Single number |
| Vector | \(\mathbf{x}\), \(\mathbf{y}\) | List of numbers (1D array) |
| Matrix | \(W\), \(X\) | 2D grid of numbers |
| "belongs to" | \(\in\) | "is an element of" |
| Real numbers | \(\mathbb{R}\) | All decimal numbers |
| Dimension | \(\mathbb{R}^d\) | Vector with \(d\) elements |
| Matrix size | \(\mathbb{R}^{m \times n}\) | Matrix with \(m\) rows, \(n\) columns |
| Dot product | \(\mathbf{w} \cdot \mathbf{x}\) | Sum of elementwise products |
| Matrix multiply | \(W\mathbf{x}\) | Each row dot-producted with \(\mathbf{x}\) |
| Derivative | \(\frac{df}{dx}\) | Sensitivity of output to input |
| Partial derivative | \(\frac{\partial f}{\partial x}\) | Derivative w.r.t. one variable |
| Gradient | \(\nabla f\) | Vector of all partial derivatives |
| Summation | \(\sum_{i=1}^n\) | Add up terms from 1 to \(n\) |
| Exponential | \(e^x\), \(\exp(x)\) | Rapid growth function |
| Logarithm | \(\ln(x)\), \(\log(x)\) | Inverse of \(e^x\) |
| Expected value | \(\mathbb{E}[X]\) | Average value over many trials |
| Elementwise product | \(\mathbf{u} \odot \mathbf{v}\) | Multiply matching elements |

---

## What's Next?

Now that you understand the notation, you're ready for the deep learning fundamentals. Each topic will:

1. **Start with intuition** — plain English explanation
2. **Show the math** — equations with worked numerical examples
3. **Connect to code** — PyTorch implementations
4. **Test understanding** — interview questions

**Tip**: If you encounter an equation you don't understand, come back to this page and look up the symbols. Every equation in this documentation follows the notation explained here.
