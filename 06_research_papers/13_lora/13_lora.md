# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen  
**Year:** 2021 &nbsp;|&nbsp; **Venue:** ICLR  
**Link:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

---

## TL;DR

LoRA freezes the pretrained weights \(W_0\) and injects trainable **low-rank** update matrices \(\Delta W = BA\) into linear layers, where \(B \in \mathbb{R}^{d \times r}\) and \(A \in \mathbb{R}^{r \times k}\) with \(r \ll \min(d, k)\). During inference, the update can be **merged** into the original weight matrix (\(W = W_0 + BA\)) with **zero additional latency**. This enables cheap task-specific or domain-specific adaptation without full fine-tuning.

---

## Why This Paper Matters

LoRA is the **default PEFT (Parameter-Efficient Fine-Tuning)** method in practice:

1. **Multi-tenant serving:** One base model, many LoRA adapters (per customer, per language, per task)
2. **Memory efficiency:** Train only \(r(d+k)\) parameters instead of \(d \times k\)
3. **No inference overhead:** Merge adapters into base weights
4. **QLoRA extension:** Combined with quantization, fine-tune 65B models on a single GPU
5. Used in nearly every fine-tuning workflow today (Hugging Face PEFT, LitGPT, etc.)

---

## Key Concepts Explained Simply

### The Intuition: Why Low-Rank?

When you fine-tune a large model on a specific task, the weight changes (\(\Delta W\)) tend to be **low-rank** — they don't need the full capacity of the original weight matrix. Think of it this way:

- Pre-training learns **general** language understanding across millions of weight dimensions
- Fine-tuning for a specific task (e.g., "classify sentiment") only needs to adjust a **small subspace** of those dimensions
- A low-rank matrix \(BA\) with rank \(r = 4\) or \(r = 16\) captures this subspace efficiently

### How LoRA Works

For any linear layer \(h = W_0 x\):

1. **Freeze** \(W_0\) (no gradients)
2. **Add** a parallel path: \(h = W_0 x + BAx\)
3. **Initialize** \(A\) with small random values, \(B\) with zeros (so \(\Delta W = 0\) at the start)
4. **Train** only \(A\) and \(B\) (much fewer parameters)
5. **Merge** for inference: \(W = W_0 + BA\) (no runtime cost)

### Where to Apply LoRA

The paper finds that applying LoRA to **query and value projection matrices** (\(W_Q, W_V\)) in attention layers gives the best results. You can also apply it to key projections (\(W_K\)), output projections (\(W_O\)), and FFN layers.

### Parameter Savings

| | Full Fine-Tuning | LoRA (r=8) |
|---|---|---|
| GPT-3 (175B) | 175B parameters | 4.7M parameters |
| Fraction | 100% | **0.003%** |
| GPU memory | Multiple A100s | Single A100 |

---

## The Math — Explained Step by Step

### Forward Pass with LoRA

\[
h = W_0 x + \Delta W x = W_0 x + BAx
\]

**Parameter counts:**
- \(W_0\): \(d \times k\) (frozen)
- \(B\): \(d \times r\) (trainable)
- \(A\): \(r \times k\) (trainable)
- Total trainable: \(r(d + k)\)

For \(d = k = 4096\) and \(r = 8\):
- Full: \(4096 \times 4096 = 16.8M\) parameters
- LoRA: \(8 \times (4096 + 4096) = 65.5K\) parameters — **256× reduction**

### Initialization

- \(A \sim \mathcal{N}(0, \sigma^2)\): Random Gaussian initialization
- \(B = 0\): Zero initialization

This ensures \(\Delta W = BA = 0\) at the start of training — the model begins identical to the pre-trained model.

### Scaling Factor

In practice, LoRA uses a scaling factor \(\alpha / r\):

\[
h = W_0 x + \frac{\alpha}{r} BAx
\]

where \(\alpha\) is a hyperparameter (often set to \(r\) or \(2r\)). This keeps the magnitude of the update stable when changing rank.

### Merging for Inference

\[
W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA
\]

Once merged, the forward pass is identical to a standard linear layer — **no additional computation or memory** at inference time. To switch tasks, swap the merged weights.

---

## Python Implementation

```python
import numpy as np


class LoRALinear:
    """Linear layer with LoRA adaptation."""

    def __init__(self, d_in, d_out, rank=4, alpha=None):
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank

        # Pre-trained weight (frozen)
        self.W0 = np.random.randn(d_out, d_in) * 0.02

        # LoRA matrices (trainable)
        self.A = np.random.randn(rank, d_in) * 0.01  # Gaussian init
        self.B = np.zeros((d_out, rank))  # Zero init → ΔW starts at 0

        self._merged = False

    def forward(self, x):
        """Forward pass: W0*x + scaling * B*A*x."""
        base = x @ self.W0.T

        if self._merged:
            return base

        lora = x @ self.A.T @ self.B.T * self.scaling
        return base + lora

    def merge(self):
        """Merge LoRA into base weights for inference (no overhead)."""
        self.W0 = self.W0 + self.scaling * self.B @ self.A
        self._merged = True

    def unmerge(self):
        """Remove LoRA from base weights (for switching adapters)."""
        self.W0 = self.W0 - self.scaling * self.B @ self.A
        self._merged = False

    @property
    def trainable_params(self):
        return self.rank * (self.d_in + self.d_out)

    @property
    def total_params(self):
        return self.d_in * self.d_out

    @property
    def compression_ratio(self):
        return self.total_params / self.trainable_params


class MultiAdapterServing:
    """Demonstrates serving multiple LoRA adapters on one base model."""

    def __init__(self, d_in, d_out, rank=4):
        self.base_weight = np.random.randn(d_out, d_in) * 0.02
        self.adapters = {}
        self.rank = rank
        self.d_in = d_in
        self.d_out = d_out

    def add_adapter(self, name):
        """Add a new task-specific adapter."""
        self.adapters[name] = {
            'A': np.random.randn(self.rank, self.d_in) * 0.01,
            'B': np.random.randn(self.d_out, self.rank) * 0.01,
        }

    def forward(self, x, adapter_name=None):
        """Forward with optional adapter."""
        out = x @ self.base_weight.T
        if adapter_name and adapter_name in self.adapters:
            A = self.adapters[adapter_name]['A']
            B = self.adapters[adapter_name]['B']
            out += x @ A.T @ B.T
        return out


def compare_ranks(d_in=4096, d_out=4096):
    """Show parameter savings for different LoRA ranks."""
    full_params = d_in * d_out
    print(f"Full fine-tuning: {full_params:,} parameters\n")
    print(f"{'Rank':>6} {'LoRA Params':>12} {'Fraction':>10} {'Compression':>12}")
    print("-" * 44)
    for r in [1, 2, 4, 8, 16, 32, 64, 128]:
        lora_params = r * (d_in + d_out)
        fraction = lora_params / full_params
        compression = full_params / lora_params
        print(f"{r:>6} {lora_params:>12,} {fraction:>10.4%} {compression:>12.0f}×")


def simulate_training(layer, x, target, lr=0.01, steps=100):
    """Simple gradient descent on LoRA parameters."""
    losses = []
    for step in range(steps):
        out = layer.forward(x)
        error = out - target
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Gradient for B and A (simplified)
        d_out = error  # [batch, d_out]
        lora_hidden = x @ layer.A.T  # [batch, rank]

        grad_B = d_out.T @ lora_hidden / len(x) * layer.scaling
        grad_A = (d_out @ layer.B).T @ x / len(x) * layer.scaling

        layer.B -= lr * grad_B
        layer.A -= lr * grad_A

    return losses


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Parameter comparison
    compare_ranks()

    # LoRA forward pass
    print("\n--- LoRA Forward Pass ---")
    layer = LoRALinear(d_in=256, d_out=256, rank=8, alpha=16)
    x = np.random.randn(4, 256)

    out_before = layer.forward(x)
    print(f"Output before merge: shape={out_before.shape}")

    layer.merge()
    out_after = layer.forward(x)
    print(f"Output after merge:  shape={out_after.shape}")
    print(f"Max difference (should be ~0): {np.max(np.abs(out_before - out_after)):.2e}")

    # Multi-adapter serving
    print("\n--- Multi-Adapter Serving ---")
    server = MultiAdapterServing(d_in=128, d_out=128, rank=4)
    server.add_adapter("sentiment")
    server.add_adapter("translation")
    server.add_adapter("coding")

    x = np.random.randn(2, 128)
    for name in [None, "sentiment", "translation", "coding"]:
        out = server.forward(x, name)
        label = name or "base"
        print(f"  {label}: output norm = {np.linalg.norm(out):.4f}")

    print(f"\n  Memory: 1 base model + {len(server.adapters)} adapters "
          f"({server.rank * 128 * 2 * len(server.adapters):,} adapter params total)")

    # Training simulation
    print("\n--- Training Simulation ---")
    layer = LoRALinear(d_in=64, d_out=64, rank=4)
    x = np.random.randn(32, 64)
    target = np.random.randn(32, 64) * 0.1
    losses = simulate_training(layer, x, target, lr=0.001, steps=50)
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} ({(1-losses[-1]/losses[0])*100:.1f}% reduction)")
```

---

## Interview Importance

LoRA is a **top-5 most practical** topic in LLM interviews. It comes up in system design ("how do you serve many customers with different fine-tunes?") and ML engineering ("how do you adapt a model cheaply?").

### Difficulty Level: ⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Why does low-rank structure make sense for task-specific updates?

**Answer:** Pre-training creates a high-dimensional weight space optimized for general language understanding. When fine-tuning for a specific task, the model only needs to adjust its behavior along a **small number of directions** — the task-relevant subspace. Research shows that the weight updates during fine-tuning have low intrinsic dimensionality. A rank-4 or rank-8 matrix can capture these directions while ignoring the vast majority of dimensions that should remain unchanged.

### Q2: Where would you NOT use LoRA?

**Answer:**
1. **Continued pre-training:** When adapting to a very different domain (e.g., training a code model on legal text), the required weight changes may be high-rank and LoRA's capacity is insufficient
2. **Training from scratch:** LoRA assumes a good base model exists
3. **When full fine-tuning is affordable:** If you have the compute and just need one model, full fine-tuning gives better results
4. **Very different tasks:** If the target task is fundamentally different from pre-training (e.g., vision tasks on a language model), low-rank updates may not suffice

### Q3: How does merging adapters affect serving?

**Answer:** After training, \(\Delta W = BA\) can be added to \(W_0\) directly:
- **Merged model:** Same inference speed and memory as the original model — zero overhead
- **Multi-tenant serving:** Keep base weights in memory, apply per-request adapter on the fly (small matrix multiply per layer)
- **Adapter switching:** Can merge/unmerge adapters to switch between tasks
- **Stacking:** Multiple LoRA adapters can be combined (with limitations — the sum of low-rank matrices is still low-rank up to the sum of ranks)

### Q4: What is QLoRA and why does it matter?

**Answer:** QLoRA (Quantized LoRA) combines LoRA with 4-bit quantization of the base model:
1. Base weights are stored in **4-bit NormalFloat** (NF4) format
2. LoRA adapters are trained in full precision (float16/bfloat16)
3. During forward pass, base weights are dequantized on-the-fly
4. Result: Fine-tune a **65B model on a single 48GB GPU** (vs. 8× A100s for full fine-tuning)
5. Quality is surprisingly close to full fine-tuning and standard LoRA

### Q5: How do you choose the rank \(r\)?

**Answer:**
- **r = 4-8:** Good default for most fine-tuning tasks. Sufficient for classification, QA, and standard NLP tasks
- **r = 16-32:** For more complex adaptations or when you have enough data to support more parameters
- **r = 64+:** Approaching full fine-tuning territory; diminishing returns vs. more parameters
- **Rule of thumb:** Start with r=8, evaluate, increase if underfitting the task
- **Which layers:** Applying to Q, V projections is standard; adding K, O, and FFN layers can help if rank is very low

---

## Connections to Other Papers

- **LLaMA** → Most commonly LoRA-fine-tuned model family
- **InstructGPT** → LoRA enables cheap SFT on instruction data
- **Chinchilla** → LoRA makes it practical to specialize optimally-trained models
- **FLAN** → LoRA fine-tuning on instruction data as alternative to full FLAN tuning

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Freeze W₀, add trainable low-rank ΔW = BA |
| Savings | ~0.01% of parameters (e.g., 4.7M for GPT-3) |
| Initialization | A ~ Gaussian, B = 0 (start identical to pre-trained) |
| Merging | W = W₀ + αBA/r → zero inference overhead |
| Best targets | Query and value projection matrices |
| Scaling | α/r factor keeps updates stable across ranks |
| Multi-tenant | One base model + many small adapters |
| Extension | QLoRA = 4-bit base + full-precision LoRA |
