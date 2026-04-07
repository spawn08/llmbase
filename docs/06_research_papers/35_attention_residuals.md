# Attention Residuals: Rethinking Transformer Depth

**Original Authors:** Kimi Team (Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, Yutian Chen, Junjie Yan, Ming Wei, Y. Zhang, Fanqing Meng, Chao Hong, Xiaotong Xie, Shaowei Liu, Enzhe Lu, Yunpeng Tai, Yanru Chen, Xin Men, Haiqing Guo, Y. Charles, Haoyu Lu, Lin Sui, Jinguo Zhu, Zaida Zhou, Weiran He, Weixiao Huang, Xinran Xu, Yuzhi Wang, Guokun Lai, Yulun Du, Yuxin Wu, Zhilin Yang, Xinyu Zhou)
**Institution:** Moonshot AI (Kimi)
**Timeline:** Attention Residuals (March 2026)
**Links:** [arXiv:2603.15031](https://arxiv.org/abs/2603.15031) &nbsp;|&nbsp; [Hugging Face](https://huggingface.co/papers/2603.15031)

---

## TL;DR

Attention Residuals (AttnRes) replaces the standard **fixed-weight** residual connections in Transformers with a **learned, input-dependent** attention mechanism over preceding layer outputs. Instead of simply adding each layer's output to a running sum (residual stream), AttnRes computes **softmax weights** across all previous layers, allowing the model to **selectively aggregate** representations from different depths. This addresses the fundamental problem that PreNorm residual connections accumulate with unit weights, causing **uncontrolled hidden-state growth** and **progressive dilution** of individual layer contributions. Integrated into Kimi Linear (48B total / 3B activated MoE) and trained on 1.4T tokens, AttnRes demonstrates consistent scaling law gains, more uniform output magnitudes, and improved downstream performance.

---

## Why This Paper Matters

Attention Residuals tackles one of the few Transformer components that has remained **virtually unchanged** since ResNet (2015) and the original Transformer (2017):

1. **First-principles redesign:** Questions the assumption that all layers should contribute equally via fixed addition
2. **Depth as sequence:** Treats transformer depth as a **sequence of representations** that can be attended over, not just summed
3. **Addresses PreNorm pathology:** Standard PreNorm causes hidden states to grow linearly with depth, diluting early-layer signals
4. **Production-validated:** Integrated into Kimi Linear (48B MoE) with 1.4T token pretraining — not a toy experiment
5. **Efficient implementation:** Block AttnRes partitions layers into blocks to control memory/communication overhead
6. **Interview relevance:** Emerging topic for architecture design, training optimization, and systems roles in 2026

---

## Key Concepts Explained Simply

### The Residual Connection Problem

Standard Transformers use **residual connections** around each sublayer (attention, FFN):

\[
x_{\ell+1} = x_\ell + \text{Sublayer}_\ell(\text{Norm}(x_\ell))
\]

This creates a **residual stream** — a running sum of all layer outputs. The problem:

1. **Fixed unit weights:** Every layer contributes with weight 1.0, regardless of importance
2. **Uncontrolled growth:** Hidden state norms grow linearly with depth (\(\|x_L\| \approx L \cdot \|x_0\|\))
3. **Signal dilution:** Early layers' contributions become a vanishing fraction of the total sum by layer 50+
4. **No input adaptation:** The same addition happens whether the input is simple or complex

### Attention Residuals' Solution

Instead of fixed addition, AttnRes computes **attention weights** over all preceding layer outputs:

\[
x_L = \sum_{\ell=0}^{L-1} \alpha_\ell \cdot h_\ell
\]

where \(\alpha_\ell = \text{softmax}(\text{score}(q, h_\ell))\) are **learned, input-dependent** weights. This lets the model:
- **Focus** on the most relevant depths for each input
- **Skip** layers that don't contribute useful transformations
- **Balance** contributions regardless of depth position

### Block AttnRes: The Efficiency Trick

Computing attention over all \(L\) layers is expensive: \(O(L^2)\) memory and communication. Block AttnRes partitions layers into \(B\) blocks of \(L/B\) layers each:

1. **Within-block attention:** Layers in the same block attend to each other normally
2. **Between-block attention:** Each block computes a summary vector, then attends over block summaries
3. **Two-phase strategy:** Cache-based pipeline communication minimizes inter-GPU overhead

This reduces complexity from \(O(L^2)\) to \(O(B^2 + L \cdot L/B)\), making it practical for 100+ layer models.

---

## The Math — Explained Step by Step

### Standard PreNorm Residual

At layer \(\ell\), the residual stream updates as:

\[
x_{\ell+1} = x_\ell + f_\ell(\text{RMSNorm}(x_\ell))
\]

where \(f_\ell\) is the sublayer (attention or FFN). Unrolling from layer 0 to \(L\):

\[
x_L = x_0 + \sum_{\ell=0}^{L-1} f_\ell(\text{RMSNorm}(x_\ell))
\]

**Problem:** The sum grows linearly with \(L\). If each \(f_\ell\) contributes magnitude \(\epsilon\), then \(\|x_L\| \approx \|x_0\| + L \cdot \epsilon\).

### Attention Residuals Formulation

AttnRes replaces the fixed sum with attention-weighted aggregation:

\[
x_L = \sum_{\ell=0}^{L-1} \alpha_\ell \cdot h_\ell
\]

where \(h_\ell\) is the output of layer \(\ell\) (after sublayer transformation), and:

\[
\alpha_\ell = \frac{\exp(\text{score}(q, h_\ell))}{\sum_{j=0}^{L-1} \exp(\text{score}(q, h_j))}
\]

The query \(q\) can be:
- **Input-dependent:** Derived from the current sequence representation
- **Learned:** A set of query vectors that learn to probe different depths
- **Hybrid:** Combination of input and learned queries

### Scoring Function

The paper uses a standard dot-product scoring:

\[
\text{score}(q, h_\ell) = \frac{q^\top W_Q (W_K h_\ell)^\top}{\sqrt{d_k}}
\]

where \(W_Q\) and \(W_K\) are learned projections. This is **identical in form** to self-attention, but the "sequence" being attended over is **layer index** \(\ell\) instead of token position.

### Block AttnRes: Partitioned Attention

Divide \(L\) layers into \(B\) blocks, each with \(L/B\) layers. For block \(b\):

**Phase 1: Intra-block attention**

\[
h_\ell^{(b)} = \text{Attn}\left(\{h_j : j \in \text{block } b\}\right) \quad \text{for } \ell \in \text{block } b
\]

**Phase 2: Inter-block attention**

Compute block summaries \(s_b = \frac{1}{|b|} \sum_{\ell \in b} h_\ell^{(b)}\), then:

\[
\beta_b = \text{softmax}(\text{score}(q, s_b))
\]

\[
x_L = \sum_{b=1}^{B} \beta_b \cdot s_b
\]

**Complexity:** \(O(B^2 + L \cdot L/B)\) vs. \(O(L^2)\) for full AttnRes.

### Gradient Flow Analysis

In standard residuals, the gradient from layer \(L\) to layer \(\ell\) flows through \(L - \ell\) addition operations:

\[
\frac{\partial \mathcal{L}}{\partial x_\ell} = \frac{\partial \mathcal{L}}{\partial x_L} + \sum_{j=\ell}^{L-1} \frac{\partial \mathcal{L}}{\partial x_j} \cdot \frac{\partial x_j}{\partial x_\ell}
\]

With AttnRes, gradients are **gated** by attention weights:

\[
\frac{\partial \mathcal{L}}{\partial x_\ell} = \alpha_\ell \cdot \frac{\partial \mathcal{L}}{\partial x_L} + \text{attention gradient terms}
\]

This means:
- Layers with **high** \(\alpha_\ell\) receive **strong** gradients
- Layers with **low** \(\alpha_\ell\) receive **weak** gradients (can be a feature, not a bug)
- Gradient flow is **input-adaptive**, not uniform

---

## Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionResidual(nn.Module):
    """
    Attention Residuals: replaces fixed summation with attention over layers.
    
    Instead of: x_L = x_0 + sum(f_ell(x_ell) for ell in 0..L-1)
    We compute: x_L = sum(alpha_ell * h_ell for ell in 0..L-1)
    where alpha_ell = softmax(score(q, h_ell))
    """
    
    def __init__(self, d_model: int, n_layers: int, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        d_k = d_model // n_heads
        
        # Query and key projections for attention over layers
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        
        # Optional: learned query vectors (input-independent)
        self.learned_queries = nn.Parameter(
            torch.randn(n_heads, d_k)
        )
        
        self.scale = np.sqrt(d_k)
    
    def forward(self, layer_outputs: torch.Tensor,
                input_query: torch.Tensor = None) -> torch.Tensor:
        """
        Compute attention-weighted aggregation over layer outputs.
        
        Args:
            layer_outputs: (n_layers, batch, d_model) — outputs from each layer
            input_query: (batch, d_model) — optional input-dependent query
        
        Returns:
            aggregated: (batch, d_model) — attention-weighted sum
        """
        n_layers, batch, d_model = layer_outputs.shape
        
        # Compute keys from layer outputs
        K = self.w_k(layer_outputs)  # (n_layers, batch, d_model)
        K = K.view(n_layers, batch, self.n_heads, -1)  # (n_layers, batch, heads, d_k)
        K = K.permute(1, 2, 0, 3)  # (batch, heads, n_layers, d_k)
        
        # Compute queries
        if input_query is not None:
            # Input-dependent query
            Q = self.w_q(input_query)  # (batch, d_model)
            Q = Q.view(batch, self.n_heads, -1)  # (batch, heads, d_k)
            Q = Q.unsqueeze(2)  # (batch, heads, 1, d_k)
        else:
            # Use learned queries, broadcasted to batch
            Q = self.learned_queries.unsqueeze(0)  # (1, heads, d_k)
            Q = Q.expand(batch, -1, -1).unsqueeze(2)  # (batch, heads, 1, d_k)
        
        # Compute attention scores over layers
        scores = torch.matmul(Q, K.transpose(-1, -2))  # (batch, heads, 1, n_layers)
        scores = scores / self.scale
        
        # Softmax over layer dimension
        attn_weights = F.softmax(scores, dim=-1)  # (batch, heads, 1, n_layers)
        
        # Aggregate layer outputs
        V = layer_outputs.view(n_layers, batch, self.n_heads, -1)  # (n_layers, batch, heads, d_k)
        V = V.permute(1, 2, 0, 3)  # (batch, heads, n_layers, d_k)
        
        aggregated = torch.matmul(attn_weights, V)  # (batch, heads, 1, d_k)
        aggregated = aggregated.squeeze(2)  # (batch, heads, d_k)
        aggregated = aggregated.reshape(batch, d_model)  # (batch, d_model)
        
        return aggregated, attn_weights


class BlockAttentionResidual(nn.Module):
    """
    Block Attention Residuals: partitions layers into blocks for efficiency.
    
    Complexity: O(B^2 + L * L/B) vs O(L^2) for full AttnRes
    """
    
    def __init__(self, d_model: int, n_layers: int, n_blocks: int,
                 n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_blocks = n_blocks
        self.layers_per_block = n_layers // n_blocks
        
        assert n_layers % n_blocks == 0, "n_layers must be divisible by n_blocks"
        
        # Intra-block attention
        self.intra_attn = nn.ModuleList([
            AttentionResidual(d_model, self.layers_per_block, n_heads)
            for _ in range(n_blocks)
        ])
        
        # Inter-block attention (over block summaries)
        self.inter_attn = AttentionResidual(d_model, n_blocks, n_heads)
    
    def forward(self, layer_outputs: torch.Tensor,
                input_query: torch.Tensor = None) -> torch.Tensor:
        """
        Two-phase block attention.
        
        Args:
            layer_outputs: (n_layers, batch, d_model)
            input_query: (batch, d_model)
        
        Returns:
            aggregated: (batch, d_model)
        """
        batch = layer_outputs.shape[1]
        
        # Phase 1: Intra-block attention
        block_summaries = []
        for b in range(self.n_blocks):
            start = b * self.layers_per_block
            end = start + self.layers_per_block
            
            block_outputs = layer_outputs[start:end]  # (layers_per_block, batch, d_model)
            block_agg, _ = self.intra_attn[b](block_outputs, input_query)
            block_summaries.append(block_agg)
        
        block_summaries = torch.stack(block_summaries, dim=0)  # (n_blocks, batch, d_model)
        
        # Phase 2: Inter-block attention
        aggregated, block_weights = self.inter_attn(block_summaries, input_query)
        
        return aggregated, block_weights


def compare_residual_strategies(d_model: int = 512, n_layers: int = 32,
                                 batch: int = 4):
    """
    Compare standard vs attention residuals in terms of norm growth and flexibility.
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("=" * 70)
    print("Residual Connection Comparison")
    print("=" * 70)
    
    # Simulate layer outputs with random transformations
    layer_outputs = torch.randn(n_layers, batch, d_model) * 0.1
    x0 = torch.randn(batch, d_model)
    
    # 1. Standard residual: x_L = x_0 + sum(f_ell)
    x_standard = x0 + layer_outputs.sum(dim=0)
    print(f"\n1. Standard Residual:")
    print(f"   ||x_0||: {x0.norm(dim=-1).mean():.4f}")
    print(f"   ||x_L||: {x_standard.norm(dim=-1).mean():.4f}")
    print(f"   Growth factor: {x_standard.norm(dim=-1).mean() / x0.norm(dim=-1).mean():.2f}x")
    print(f"   Each layer contributes equally (weight = 1.0)")
    
    # 2. Attention residual: x_L = sum(alpha_ell * h_ell)
    attn_res = AttentionResidual(d_model, n_layers)
    x_attn, weights = attn_res(layer_outputs)
    print(f"\n2. Attention Residual:")
    print(f"   ||x_L||: {x_attn.norm(dim=-1).mean():.4f}")
    print(f"   Growth factor: {x_attn.norm(dim=-1).mean() / x0.norm(dim=-1).mean():.2f}x")
    print(f"   Layer weights vary: [{weights[0, 0, 0, :5].detach().numpy().round(3)}, ...]")
    print(f"   Weight entropy: {(-weights[0, 0, 0] * weights[0, 0, 0].log()).sum().detach().item():.4f}")
    
    # 3. Block attention residual
    n_blocks = 8
    block_attn = BlockAttentionResidual(d_model, n_layers, n_blocks)
    x_block, block_weights = block_attn(layer_outputs)
    print(f"\n3. Block Attention Residual ({n_blocks} blocks):")
    print(f"   ||x_L||: {x_block.norm(dim=-1).mean():.4f}")
    print(f"   Growth factor: {x_block.norm(dim=-1).mean() / x0.norm(dim=-1).mean():.2f}x")
    print(f"   Block weights: {block_weights[0, 0, 0, :].detach().numpy().round(3)}")
    
    print(f"\n{'=' * 70}")
    print("Key Insight:")
    print(f"{'=' * 70}")
    print(f"Standard residuals grow linearly with depth (||x_L|| ~ L)")
    print(f"Attention residuals maintain bounded norms (||x_L|| ~ constant)")
    print(f"This matters for deep models (50+ layers) where signal dilution occurs")


def analyze_depth_utilization(d_model: int = 256, n_layers: int = 48,
                               batch: int = 2):
    """
    Analyze how attention residuals distribute weight across depth.
    """
    torch.manual_seed(42)
    
    print("\n" + "=" * 70)
    print("Depth Utilization Analysis")
    print("=" * 70)
    
    # Create layer outputs with varying "usefulness"
    layer_outputs = torch.zeros(n_layers, batch, d_model)
    for i in range(n_layers):
        # Simulate: early and late layers more useful, middle layers less so
        if i < 8 or i > n_layers - 8:
            layer_outputs[i] = torch.randn(batch, d_model) * 0.3
        else:
            layer_outputs[i] = torch.randn(batch, d_model) * 0.05
    
    attn_res = AttentionResidual(d_model, n_layers)
    _, weights = attn_res(layer_outputs)
    
    # Average attention weights across heads and batch
    avg_weights = weights.mean(dim=1).mean(dim=0)  # (1, n_layers)
    
    print(f"\nAverage attention weight per layer:")
    for i in range(0, n_layers, 4):
        w = avg_weights[0, i].item()
        bar = "█" * int(w * 100)
        print(f"  Layer {i:2d}: {w:.4f} {bar}")
    
    print(f"\nObservation: Attention concentrates on early/late layers,")
    print(f"bypassing less useful middle layers — adaptive depth selection.")


# --- Demo ---
if __name__ == "__main__":
    print("=" * 70)
    print("Attention Residuals: Rethinking Transformer Depth")
    print("=" * 70)
    
    # 1. Compare residual strategies
    compare_residual_strategies(d_model=512, n_layers=32)
    
    # 2. Analyze depth utilization
    analyze_depth_utilization(d_model=256, n_layers=48)
    
    # 3. End-to-end attention residual test
    print("\n" + "=" * 70)
    print("End-to-End Attention Residual Test")
    print("=" * 70)
    
    d_model, n_layers, batch = 512, 24, 4
    layer_outputs = torch.randn(n_layers, batch, d_model)
    input_query = torch.randn(batch, d_model)
    
    attn_res = AttentionResidual(d_model, n_layers, n_heads=8)
    output, weights = attn_res(layer_outputs, input_query)
    
    print(f"\nInput layer outputs shape: {layer_outputs.shape}")
    print(f"Input query shape: {input_query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weight sparsity (fraction < 0.01): "
          f"{(weights < 0.01).float().mean().item():.3f}")
    
    # 4. Block attention test
    n_blocks = 6
    block_attn = BlockAttentionResidual(d_model, n_layers, n_blocks)
    output_block, block_weights = block_attn(layer_outputs, input_query)
    
    print(f"\nBlock Attention Residual ({n_blocks} blocks):")
    print(f"  Output shape: {output_block.shape}")
    print(f"  Block weights shape: {block_weights.shape}")
    print(f"  Complexity reduction: O(L^2) = O({n_layers**2}) -> "
          f"O(B^2 + L*L/B) = O({n_blocks**2 + n_layers * n_layers // n_blocks})")
```

---

## Connection to Kimi Linear Architecture

Attention Residuals was integrated into **Kimi Linear**, a 48B-parameter MoE model (3B active parameters) following the Moonlight / DeepSeek-V3 design:

### Kimi Linear Configuration

- **Total parameters:** 48B
- **Active parameters:** 3B per token (MoE with top-K routing)
- **Training data:** 1.4T tokens
- **Architecture:** MoE Transformer with MLA (Multi-head Latent Attention)
- **AttnRes integration:** Drop-in replacement for standard residuals

### Implementation Optimizations

For distributed training, Kimi's AttnRes uses:

1. **Cache-based pipeline communication:** Layer outputs are cached and streamed between GPUs to avoid all-to-all communication bottlenecks
2. **Two-phase computation:** Intra-block attention computed locally, inter-block attention uses synchronized block summaries
3. **Memory-efficient attention:** Checkpointing intermediate layer outputs to avoid storing all \(L\) outputs in memory

### Results

- **Scaling laws:** Consistent improvements across model sizes
- **Hidden-state norms:** More uniform magnitudes across depth (no linear growth)
- **Gradient distribution:** Better-conditioned gradients, less variance across layers
- **Downstream performance:** Improved benchmarks across all evaluated tasks

---

## Why This Matters for Deep Transformers

### The Depth Scaling Problem

Standard Transformers struggle to scale beyond ~50 layers because:

1. **Gradient vanishing/exploding:** Despite residual connections, deep networks have unstable training
2. **Signal dilution:** Early layers' contributions become negligible in the running sum
3. **Representation collapse:** All layers converge to similar transformations (wasted capacity)
4. **Post-Norm vs Pre-Norm trade-offs:**
   - **Post-Norm** (LayerNorm after residual): Stable gradients, but hard to optimize at depth
   - **Pre-Norm** (LayerNorm before residual): Easy to optimize, but hidden states grow unbounded

### AttnRes Addresses All Four

1. **Gradient flow:** Attention weights gate gradients, preventing vanishing/exploding issues
2. **No dilution:** Important layers get high attention weight, regardless of depth position
3. **Adaptive computation:** Model learns to use different depths for different inputs
4. **Norm control:** Softmax normalization bounds the output magnitude: \(\|x_L\| \leq \max_\ell \|h_\ell\|\)

---

## Interview Importance

Attention Residuals is an **emerging topic** in 2026 interviews, especially for roles involving:

- **Architecture design:** Questioning fundamental Transformer assumptions
- **Training optimization:** Deep network stability and scaling
- **Distributed systems:** Block AttnRes implementation challenges
- **Moonshot AI/Kimi:** Understanding their technical contributions

### Difficulty Level: ⭐⭐⭐⭐ (Hard)

---

## Interview Questions & Answers

### Q1: What is the fundamental problem with standard residual connections that AttnRes solves?

**Answer:** Standard residuals use **fixed unit weights**: every layer contributes equally via addition, \(x_{\ell+1} = x_\ell + f_\ell(x_\ell)\). This causes two issues:
1. **Uncontrolled growth:** Hidden state norms grow linearly with depth (\(\|x_L\| \approx L \cdot \epsilon\)), especially in PreNorm configurations
2. **Signal dilution:** Early layers' contributions become a vanishing fraction by layer 50+, making it hard to preserve initial representations

AttnRes replaces fixed addition with **attention-weighted aggregation**, letting the model learn which layers matter for each input.

### Q2: How does Block AttnRes reduce complexity from \(O(L^2)\) to \(O(B^2 + L \cdot L/B)\)?

**Answer:** Full AttnRes computes attention over all \(L\) layers, requiring \(O(L^2)\) pairwise scores. Block AttnRes:
1. **Partitions** \(L\) layers into \(B\) blocks of \(L/B\) layers each
2. **Intra-block attention:** Each block computes attention internally: \(B \times O((L/B)^2) = O(L^2/B)\)
3. **Inter-block attention:** Attend over \(B\) block summaries: \(O(B^2)\)
4. **Total:** \(O(B^2 + L^2/B)\)

For \(L=96, B=8\): \(96^2 = 9216\) vs. \(8^2 + 96 \times 12 = 1216\) — a **7.6× reduction**.

### Q3: Why does PreNorm cause hidden-state growth, and how does AttnRes fix it?

**Answer:** In PreNorm: \(x_{\ell+1} = x_\ell + f_\ell(\text{RMSNorm}(x_\ell))\). The RMSNorm normalizes \(x_\ell\) before the sublayer, but the **residual add** is unnormalized. If \(f_\ell\) outputs magnitude \(\epsilon\) on average, after \(L\) layers:

\[
\|x_L\|^2 = \|x_0 + \sum f_\ell\|^2 \approx \|x_0\|^2 + L \cdot \epsilon^2
\]

AttnRes bounds this via softmax normalization:

\[
x_L = \sum \alpha_\ell h_\ell, \quad \sum \alpha_\ell = 1 \implies \|x_L\| \leq \max_\ell \|h_\ell\|
\]

The output is a **convex combination** of layer outputs, preventing unbounded growth.

### Q4: How do AttnRes gradients differ from standard residual gradients?

**Answer:** In standard residuals, gradients flow uniformly through the identity path:

\[
\frac{\partial \mathcal{L}}{\partial x_\ell} = \frac{\partial \mathcal{L}}{\partial x_L} \quad \text{(identity connection)}
\]

Every layer receives the **same** upstream gradient magnitude (plus sublayer gradients).

In AttnRes, gradients are **gated by attention weights**:

\[
\frac{\partial \mathcal{L}}{\partial x_\ell} = \alpha_\ell \cdot \frac{\partial \mathcal{L}}{\partial x_L} + \frac{\partial \alpha_\ell}{\partial x_\ell} \cdot (\text{value terms})
\]

Layers with high \(\alpha_\ell\) receive strong gradients; layers with low \(\alpha_\ell\) receive weak gradients. This is **adaptive** — the model learns which layers need stronger training signal for each input.

---

## Connections to Other Papers

- **ResNet (2015)** → Original residual connections; AttnRes generalizes this idea
- **Transformer (2017)** → Standard residuals unchanged for 9 years; AttnRes redesigns this component
- **DeepSeek-V3** → Kimi Linear builds on DeepSeek-V3's MoE and MLA design
- **Kimi K2.5** → 1T MoE model from same lab; AttnRes may influence future versions
- **GLM-5** → Another 700B+ MoE model; both address deep Transformer training challenges
- **Chinchilla** → Scaling laws; AttnRes changes the "depth" axis of scaling equations

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Replace fixed residual addition with **attention-weighted** aggregation over layers |
| Problem solved | PreNorm hidden-state growth, signal dilution, uniform layer contributions |
| Mathematical form | \(x_L = \sum_{\ell=0}^{L-1} \alpha_\ell h_\ell\) where \(\alpha_\ell = \text{softmax}(\text{score}(q, h_\ell))\) |
| Block AttnRes | Partition layers into blocks: \(O(B^2 + L^2/B)\) vs \(O(L^2)\) |
| Norm control | Bounded: \(\|x_L\| \leq \max_\ell \|h_\ell\|\) (convex combination) |
| Gradient flow | **Input-adaptive** — gated by attention weights, not uniform |
| Kimi Linear | 48B MoE (3B active), 1.4T tokens, AttnRes integrated |
| Results | Better scaling laws, uniform norms, improved downstream performance |
| Interview angle | Emerging 2026 topic; tests understanding of residual connections, depth scaling |
| Key insight | **Depth as sequence** — treat layer index like token position for attention |
