# Convolutional Neural Networks

## Why This Matters for LLMs

Convolutional neural networks (CNNs) are the **historical backbone of computer vision**—and vision is how multimodal large language models **see**. Systems like GPT-4V, Gemini, and open recipes such as LLaVA pair a **vision encoder** (often CNN- or ViT-based) with a language model so images become tokens in a shared latent space. Even when the encoder is a **Vision Transformer (ViT)** rather than a classic CNN stack, the **ideas** behind convolutions—**local receptive fields**, **hierarchical features**, and **translation-equivariant** processing—remain the mental model for what “good” spatial representations look like.

Vision Transformers have displaced pure CNNs for many large-scale image tasks, but **1D convolutions** still appear in **efficient sequence models** (depthwise separable variants, local mixing layers, and hybrid designs). Understanding convolutions therefore completes the **“big three”** of classical neural architectures alongside **feedforward networks (FFNs)** and **recurrent networks (RNNs)**: the vocabulary interviewers expect when you discuss **inductive bias**, **parameter sharing**, and **multimodal fusion**.

Finally, **residual connections**—popularized in depth by **ResNet**—are the same structural idea as the **residual stream** in Transformers: a clean path for gradients and a stable place to add sub-layer outputs. Tracing CNNs through **ResNet → ViT → CLIP** is one of the shortest routes from “2010s vision” to “2020s multimodal LLMs.”

---

## Core Concepts

### The Convolution Operation

A **2D discrete convolution** (in the **deep-learning sense**, frameworks typically implement **cross-correlation** without flipping the kernel) produces one output value by placing a **kernel** \(\mathbf{W} \in \mathbb{R}^{K \times K}\) over a \(K \times K\) patch of the input and taking the **sum of elementwise products**. For a single-channel input \(X\) and output \(Y\):

\[
Y[i,j] = \sum_{u=0}^{K-1}\sum_{v=0}^{K-1} X[i+u,\, j+v]\, W[u,v] + b.
\]

!!! math-intuition "In Plain English"
    Each output location \((i,j)\) answers: “How much does this **local pattern** \(\mathbf{W}\) match the patch of \(X\) sitting under it?” The kernel **slides** across the input; **parameter sharing** means the **same** weights \(W[u,v]\) are reused at every position—unlike a fully connected layer that would learn separate weights per pixel.

Mathematical **convolution** flips the kernel; **cross-correlation** does not. In interviews, say what your framework does (PyTorch `Conv2d` uses cross-correlation) and that the distinction is mostly **notational** when weights are learned.

**Sliding a \(3 \times 3\) kernel over a \(5 \times 5\) input (valid, no padding):** the kernel visits every position where it fits fully inside the input. With stride \(1\), the output grid is \(3 \times 3\) (see the output-size formula below). Conceptually:

```text
5×5 input                    3×3 kernel (reused at each position)
┌─────────────────┐          ┌─────────┐
│ ■ ■ ■ · ·       │   →      │ w w w │
│ ■ ■ ■ · ·       │          │ w w w │
│ ■ ■ ■ · ·       │          │ w w w │
│ · · · · ·       │          └─────────┘
│ · · · · ·       │          dot with each 3×3 patch → one scalar per position
└─────────────────┘
```

!!! example "Worked Example: one output pixel with real numbers"
    Take the **top-left** \(3 \times 3\) patch of \(X\) and a kernel \(\mathbf{W}\):

    \[
    X[0{:}3,\,0{:}3] = \begin{bmatrix}
    1 & 1 & 1\\
    2 & 2 & 2\\
    3 & 3 & 3
    \end{bmatrix},
    \quad
    \mathbf{W} = \begin{bmatrix}
    1 & 0 & 0\\
    0 & 1 & 0\\
    0 & 0 & 1
    \end{bmatrix}.
    \]

    The output at \(Y[0,0]\) is the **sum of elementwise products**:

    \[
    Y[0,0] = 1\cdot 1 + 1\cdot 0 + 1\cdot 0 + 2\cdot 0 + 2\cdot 1 + 2\cdot 0 + 3\cdot 0 + 3\cdot 0 + 3\cdot 1 = 1 + 2 + 3 = 6.
    \]

    Sliding the kernel to the next position (e.g. one step right) repeats the operation on a **new** patch, producing \(Y[0,1]\), and so on.

### Stride, Padding, and Output Size

For a square input of side \(I\), square kernel of side \(K\), padding \(P\) (zeros added symmetrically), and stride \(S\), the **spatial output size** (per dimension) is:

\[
O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1.
\]

!!! math-intuition "In Plain English"
    **Padding** grows the effective canvas so borders contribute like interior pixels. **Stride** subsamples the output grid: larger stride \(\Rightarrow\) fewer positions \(\Rightarrow\) smaller feature maps (and cheaper downstream computation). The fraction must land on a valid grid; the floor captures integer downsampling.

!!! example "Worked Example: stride and padding"
    - **\(I=5,\, K=3,\, P=0,\, S=1\):** \(O = \lfloor(5-3+0)/1\rfloor + 1 = 3\).
    - **\(I=5,\, K=3,\, P=1,\, S=1\):** \(O = \lfloor(5-3+2)/1\rfloor + 1 = 5\) (same spatial size as input—typical for “same” convolutions when \(K\) is odd and \(P=(K-1)/2\)).
    - **\(I=7,\, K=3,\, P=0,\, S=2\):** \(O = \lfloor(7-3)/2\rfloor + 1 = 3\) (downsampling without pooling).

### Multiple Channels and Filters

Real inputs have **multiple channels** \(C_{\mathrm{in}}\) (e.g. RGB \(C_{\mathrm{in}}=3\); intermediate layers often have hundreds). A **single filter** produces one output channel by summing over **all input channels**:

\[
Y[c_{\mathrm{out}}, i, j] = \sum_{c_{\mathrm{in}}=0}^{C_{\mathrm{in}}-1}\sum_{u=0}^{K-1}\sum_{v=0}^{K-1}
X[c_{\mathrm{in}}, i+u, j+v]\, W[c_{\mathrm{out}}, c_{\mathrm{in}}, u, v] + b[c_{\mathrm{out}}].
\]

The full weight tensor has shape **\((C_{\mathrm{out}},\, C_{\mathrm{in}},\, K,\, K)\)**—\(C_{\mathrm{out}}\) independent filters, each of depth \(C_{\mathrm{in}}\).

!!! math-intuition "In Plain English"
    Each output channel is a **different learned detector** applied everywhere on the map. **Depth** in the tensor is “what mix of input channels defines this pattern”; **spatial** \(K\times K\) is “what local arrangement.”

**Parameter count** (including one bias per output channel):

\[
\text{\#params} = C_{\mathrm{out}}\big(C_{\mathrm{in}} K^2 + 1\big).
\]

!!! math-intuition "In Plain English"
    You pay \(C_{\mathrm{in}} K^2\) weights **per** output channel (full connectivity across input channels inside the kernel), plus one bias per channel.

!!! example "Worked Example: parameter count"
    \(C_{\mathrm{in}}=64,\, C_{\mathrm{out}}=128,\, K=3\):

    \[
    128\,(64\cdot 9 + 1) = 128 \times 577 = 73{,}856 \text{ parameters for that layer.}
    \]

### Pooling Layers

**Pooling** reduces spatial resolution using a local rule over non-overlapping or overlapping windows. The most common are:

- **Max pooling:** \(Y[i,j] = \max_{(u,v)\in \text{window}} X\) in that window.
- **Average pooling:** \(Y[i,j] = \text{mean}\) over the window.

!!! math-intuition "In Plain English"
    Pooling builds **small translation tolerance** (a feature can shift slightly within the window and still activate) and **reduces dimension** so deeper layers see broader context without exploding compute. Max pooling emphasizes **whether a feature fired**; average pooling smooths.

!!! example "Worked Example: \(2\times 2\) max pool, stride \(2\) on a \(4\times 4\) map"
    Input (one channel):

    \[
    \begin{bmatrix}
    1 & 3 & 2 & 4\\
    5 & 6 & 1 & 0\\
    2 & 2 & 7 & 8\\
    0 & 1 & 3 & 2
    \end{bmatrix}.
    \]

    Partition into four \(2\times 2\) blocks. Max in each block:

    \[
    \begin{bmatrix}
    \max(1,3,5,6) & \max(2,4,1,0)\\
    \max(2,2,0,1) & \max(7,8,3,2)
    \end{bmatrix}
    =
    \begin{bmatrix}
    6 & 4\\
    2 & 8
    \end{bmatrix}.
    \]

    Spatial size \(4\to 2\); each pooled value summarizes a **region**.

### Feature Hierarchies

CNNs typically learn a **compositional hierarchy**: **early** layers respond to **edges, colors, and simple textures**; **middle** layers combine these into **parts and patterns**; **deep** layers encode **object-level** or **scene-level** abstractions (especially with global pooling or fully connected heads).

!!! math-intuition "In Plain English"
    Small kernels see only local context; **stacking** layers enlarges the **effective receptive field** so deeper units integrate information from larger image regions. **Compositionality**—simple features feeding into complex ones—is why convnets match the structure of many natural images.

This structure is **not guaranteed** by the architecture alone (data and training matter), but it is the standard intuition for **what depth buys you** in vision—and it parallels how later multimodal models expect **patch tokens** to carry increasingly global meaning after Transformer layers.

### Classic Architectures in Brief

| Model | Year | Signature idea |
|-------|------|----------------|
| **LeNet-5** | 1998 | Convolution + subsampling for digit recognition; an early blueprint for **conv → pool → conv → FC**. |
| **AlexNet** | 2012 | Deeper CNN on ImageNet; **ReLU**, **GPU** training at scale, **dropout**—kicked off the deep learning boom in vision. |
| **VGGNet** | 2014 | Very **small \(3\times 3\)** filters stacked deeply; **uniform**, simple blocks; showed depth + small kernels works. |
| **ResNet** | 2015 | **Residual (skip) connections**: layers learn **residuals** \(\mathcal{F}(\mathbf{x})\) added to identity \(\mathbf{x}\), enabling **very deep** trainable networks. |

**Residual block (concept):**

\[
\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x}).
\]

!!! math-intuition "In Plain English"
    Instead of forcing a stack of layers to approximate the **whole** desired mapping, each block only needs to learn a **correction** on top of what is already in \(\mathbf{x}\). That **eases optimization** and keeps a **gradient highway** through the skip path—ideas that transfer directly to **Transformer blocks**.

### From CNN to Vision Transformer

**Vision Transformer (ViT)** cuts an image into fixed-size **patches** (e.g. \(16\times 16\)), **linearly embeds** each patch to a vector, adds **positional encoding**, and runs **standard Transformer encoder** blocks. Patches are **tokens**—the same abstraction as words in NLP.

!!! math-intuition "In Plain English"
    ViT **removes** hand-crafted spatial convolutions in favor of **global self-attention** between patch tokens. The trade-off is data and compute: ViTs often need **large-scale pretraining** to match strong CNNs on smaller data regimes. Hybrid models use **convolutional stems** (a few conv layers before patches) to bake in local priors.

**CLIP** (Contrastive Language–Image Pre-training) trains **image and text encoders** so matched pairs align in a **shared embedding space**—the bridge from **vision backbones** (CNN or ViT) to **language** used in many multimodal LLM recipes. Knowing CNN/ViT basics is how you parse those system diagrams in interviews.

---

## Deep Dive

??? deep-dive "Residual connections, depthwise separable convolutions, and CNNs inside multimodal LLMs"
    **Residual connections: why they work.** Very deep stacks make it hard for each layer to learn **small perturbations** to an already-useful representation. Residual learning reframes the target as \(\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})\). The skip path carries \(\mathbf{x}\) forward unchanged, so gradients can **flow** along it even if \(\mathcal{F}\) temporarily has small gradients—often described as a **gradient highway**. In Transformers, **pre-norm** or **post-norm** residual blocks wrap attention and MLP sublayers the same way: the **residual stream** is the backbone; sublayers **add** processed information.

    **Depthwise separable convolutions.** A standard conv mixes **spatial** and **channel** mixing in one tensor. **Depthwise separable** conv splits this into: (1) **depthwise** conv—one \(K\times K\) filter **per input channel** (no cross-channel mixing); then (2) **pointwise** \(1\times 1\) conv to mix channels. Fewer parameters and FLOPs for similar expressivity in many mobile/efficient settings (**MobileNet**, **EfficientNet**). Interview angle: **factorization** of operations as a systems motif (also seen in low-rank adapters and grouped convolutions).

    **CNN as vision encoder in multimodal LLMs.** Pipelines like **CLIP** and **SigLIP** use strong image towers—**ResNet** variants or **ViT**—to produce image embeddings aligned with text. Some multimodal LLMs use **convolutional stems** before ViT blocks to stabilize local structure. Even when the top stack is Transformer-only, **convolutional priors** still show up in production encoders and efficiency discussions.

---

## Code

A compact **PyTorch** CNN for **MNIST** ( \(1\times 28\times 28\) grayscale digits, 10 classes). Two convolutional blocks, max pooling, then fully connected layers. Run on CPU or GPU if available.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    """Two conv layers + pool + two FC layers — enough to reach high MNIST accuracy."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 28 -> 14 -> 7 spatially; 64 channels
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(model)
    print(f"Device: {device}")

    for epoch in range(1, 4):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg = total_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = 100.0 * correct / total
        print(f"epoch {epoch}  train_loss={avg:.4f}  test_acc={acc:.2f}%")

    print("Done.")


if __name__ == "__main__":
    main()
```

**What to expect:** after a few epochs on MNIST, test accuracy is typically **well above 98%** with this architecture and default hyperparameters—enough to demonstrate that **conv → pool → conv → pool → FC** works end-to-end.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **Define 2D convolution (or cross-correlation) for CNNs and explain parameter sharing.**  
       *Depth:* Sliding kernel; same weights at every spatial location; fewer parameters than fully connecting to local patches.

    2. **Derive or recall the output-size formula \(O = \lfloor(I - K + 2P)/S\rfloor + 1\) and explain each symbol.**  
       *Depth:* Padding expands effective input; stride downsamples; floor accounts for discrete grid.

    3. **Why does stacking conv layers increase the receptive field?**  
       *Depth:* Each layer integrates neighboring responses; effective field grows with depth (exact formula depends on kernel, stride, dilation).

    4. **What is the parameter count for `Conv2d(C_in, C_out, K)` with bias?**  
       *Depth:* \(C_{\mathrm{out}}(C_{\mathrm{in}} K^2 + 1)\); relate to grouping and depthwise separable variants.

    5. **Compare max pooling and average pooling—when might each be preferred?**  
       *Depth:* Max emphasizes presence/sparsity; average smooths; both reduce spatial size and add local translation robustness.

    6. **Describe the feature hierarchy typically learned by CNNs on natural images.**  
       *Depth:* Edges/textures → parts → objects; compositionality; connect to inductive bias for spatial locality.

    7. **What problem do residual connections solve, and how does \(\mathbf{y}=\mathbf{x}+\mathcal{F}(\mathbf{x})\) help optimization?**  
       *Depth:* Residual learning; gradient flow; direct parallel to Transformer residual streams.

    8. **Explain depthwise separable convolution and why MobileNet-style models use it.**  
       *Depth:* Split spatial per-channel mixing from cheap \(1\times 1\) channel mixing; efficiency on edge devices.

    9. **How does a Vision Transformer turn an image into tokens, and what role did CNNs play historically?**  
       *Depth:* Patchify + linear embed + position encoding; CNNs as prior art and still common in stems/hybrid encoders.

    10. **What does CLIP optimize at a high level, and why does that matter for multimodal LLMs?**  
        *Depth:* Contrastive alignment of image/text embeddings; shared space enables zero-shot transfer and downstream prompting.

!!! interview "Follow-up Probes"
    1. **Does PyTorch `Conv2d` implement convolution or cross-correlation? Does it matter after training?**  
       *Depth:* Cross-correlation; flipping is absorbed into learned weights.

    2. **What is dilation (atrous convolution), and how does it affect receptive field?**  
       *Depth:* Spaces out kernel taps; expands field without same resolution cost as stacking stride-1 convs.

    3. **Why might a ViT need more data than a ResNet for similar accuracy?**  
       *Depth:* Less built-in locality bias; global attention from scratch; data/regularization/scale dependent.

    4. **How does \(1\times 1\) convolution change channel dimension?**  
       *Depth:* Per-pixel linear mix across channels—used for bottleneck expansions and in separable convolutions.

    5. **Where do CNN ideas show up outside 2D images?**  
       *Depth:* 1D conv for sequences; local mixing; speech; time series; some efficient Transformer variants.

!!! key-phrases "Key Phrases to Use in Interviews"
    1. “Translation equivariance (or approximate invariance after pooling) from shared weights and local connectivity.”
    2. “Hierarchical features: local edges compose into parts and objects.”
    3. “Parameter sharing slashes weights versus fully connecting to local receptive fields.”
    4. “Residual connections give a gradient highway—same motif as the Transformer residual stream.”
    5. “Depthwise separable conv factorizes spatial filtering and channel mixing for efficiency.”
    6. “ViT patchifies images into tokens; CLIP aligns vision and language in one embedding space.”
    7. “Effective receptive field grows with depth—stacking convolutions sees larger context.”

## References

1. LeCun, Y., et al. (1998). *Gradient-based learning applied to document recognition.* Proceedings of the IEEE (**LeNet**).
2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks.* NeurIPS (**AlexNet**).
3. Simonyan, K., & Zisserman, A. (2015). *Very deep convolutional networks for large-scale image recognition.* ICLR (**VGG**).
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR (**ResNet**).
5. Howard, A. G., et al. (2017). *MobileNets: Efficient convolutional neural networks for mobile vision applications.* arXiv (**depthwise separable**).
6. Dosovitskiy, A., et al. (2021). *An image is worth 16x16 words: Transformers for image recognition at scale.* ICLR (**ViT**).
7. Radford, A., et al. (2021). *Learning transferable visual models from natural language supervision.* ICML (**CLIP**).
