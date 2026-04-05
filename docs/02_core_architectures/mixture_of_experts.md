# 2.7 — Mixture of Experts (MoE)

## Intuition

Standard Transformers apply every parameter to every token. **Mixture of Experts** (MoE) breaks the FFN into multiple **expert networks** and routes each token to only a subset (typically 1–2) of them. This lets you scale parameters massively — 8× the FFN capacity — while keeping per-token compute nearly constant.

Mixtral 8×7B has 46.7B total parameters but activates only ~12.9B per token, matching the quality of a 70B dense model at a fraction of the inference cost.

---

## Core concepts

### Architecture

An MoE layer replaces the dense FFN in a Transformer block:

```
Input x → Router (linear + softmax) → select top-k experts → weighted sum of outputs
```

Formally, with \(E\) experts and a gating function \(G\):

\[
G(\mathbf{x}) = \text{softmax}(W_g \mathbf{x}) \in \mathbb{R}^{E}
\]

\[
\text{MoE}(\mathbf{x}) = \sum_{i \in \text{TopK}} G(\mathbf{x})_i \cdot \text{Expert}_i(\mathbf{x})
\]

Each expert is a standard FFN (e.g., SwiGLU). Only the top-\(k\) experts (usually \(k = 1\) or \(k = 2\)) are computed; the rest are skipped entirely.

### Router design

The **router** (or **gate**) is a small linear layer that decides which experts handle each token. Key challenges:

**Load balancing** — without constraints, the router tends to send most tokens to a few "popular" experts, wasting the others. Solutions:

- **Auxiliary load-balancing loss** — penalize uneven expert usage:

\[
\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i
\]

where \(f_i\) = fraction of tokens routed to expert \(i\), \(P_i\) = average router probability for expert \(i\).

- **Expert capacity factor** — cap how many tokens each expert can process per batch. Overflow tokens are dropped or sent to a shared "default" expert.

- **Random noise** — add Gaussian noise to router logits during training to encourage exploration.

**Expert choice routing** (Zhou et al., 2022) — flip the selection: instead of tokens choosing experts, experts choose their top-\(k\) tokens. This guarantees perfect load balance.

### MoE scaling properties

| Property | Dense | MoE |
| --- | --- | --- |
| Total params | \(P\) | \(E \times P_{\text{expert}} + P_{\text{shared}}\) (much larger) |
| Active params per token | \(P\) | \(\approx k \times P_{\text{expert}} + P_{\text{shared}}\) |
| Training FLOP | Proportional to \(P\) | Proportional to active params (cheaper) |
| Memory | \(P\) | All \(E\) experts must fit in memory |

The memory requirement is the main drawback: you need to load all expert weights even though only \(k\) are used per token. This makes MoE models memory-bound, not compute-bound.

### Notable MoE models

| Model | Year | Experts | Top-k | Total params | Active params |
| --- | --- | --- | --- | --- | --- |
| Switch Transformer | 2021 | 128–2048 | 1 | Up to 1.6T | Dense-equivalent |
| GLaM | 2022 | 64 | 2 | 1.2T | ~97B |
| Mixtral 8×7B | 2023 | 8 | 2 | 46.7B | 12.9B |
| Mixtral 8×22B | 2024 | 8 | 2 | 141B | 39B |
| DeepSeek-V2 | 2024 | 160 | 6 | 236B | 21B |
| Grok-1 | 2024 | 8 | 2 | 314B | ~79B |

### MoE training challenges

1. **Instability** — MoE models are harder to train stably. Router collapse (all tokens go to one expert) can happen early in training.
2. **Expert specialization** — ideally each expert specializes in different domains/skills. In practice, specialization is weak and hard to interpret.
3. **Communication overhead** — in distributed training, tokens must be dispatched to the GPU hosting the correct expert (all-to-all communication).
4. **Fine-tuning** — MoE models are harder to fine-tune efficiently because only a subset of experts see each example.

---

## Code — Simple MoE layer in PyTorch

```python
"""
Mixture of Experts (MoE) layer in PyTorch.
Implements: top-k routing, load balancing loss, and expert capacity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A single expert — standard SwiGLU FFN."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """
    Top-k Mixture of Experts with load-balancing loss.

    Args:
        d_model: hidden dimension
        d_ff: expert FFN inner dimension
        n_experts: total number of experts
        top_k: number of experts per token (1 or 2)
        balance_coeff: weight for load-balancing auxiliary loss
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        balance_coeff: float = 0.01,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff

        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D)
            aux_loss: scalar load-balancing loss
        """
        B, T, D = x.shape
        flat_x = x.view(-1, D)  # (B*T, D)
        N = flat_x.size(0)

        # Router
        router_logits = self.gate(flat_x)         # (N, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-k selection
        topk_probs, topk_indices = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)   # renormalize

        # Compute expert outputs (simple loop — production uses scatter/gather)
        output = torch.zeros_like(flat_x)
        for k_idx in range(self.top_k):
            expert_idx = topk_indices[:, k_idx]  # (N,)
            weight = topk_probs[:, k_idx]        # (N,)
            for e in range(self.n_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        # Load-balancing loss
        # f_i = fraction of tokens routed to expert i
        # P_i = average router probability for expert i
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        for k_idx in range(self.top_k):
            for e in range(self.n_experts):
                tokens_per_expert[e] += (topk_indices[:, k_idx] == e).float().sum()
        f = tokens_per_expert / (N * self.top_k)
        P = router_probs.mean(dim=0)
        aux_loss = self.balance_coeff * self.n_experts * (f * P).sum()

        return output.view(B, T, D), aux_loss


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(42)
    D, D_FF, E, K = 128, 256, 8, 2
    B, T = 2, 16

    moe = MoELayer(D, D_FF, n_experts=E, top_k=K)
    x = torch.randn(B, T, D)

    output, aux_loss = moe(x)
    print(f"Input:       {x.shape}")
    print(f"Output:      {output.shape}")
    print(f"Aux loss:    {aux_loss.item():.6f}")

    # Parameter count comparison
    dense_ffn = Expert(D, D_FF)
    dense_params = sum(p.numel() for p in dense_ffn.parameters())
    moe_params = sum(p.numel() for p in moe.parameters())
    print(f"\nDense FFN params:  {dense_params:,}")
    print(f"MoE layer params:  {moe_params:,} ({E} experts)")
    print(f"Ratio:             {moe_params / dense_params:.1f}× total, "
          f"~{K}/{E} = {K / E:.0%} active per token")

    # Check expert utilization
    with torch.no_grad():
        router_logits = moe.gate(x.view(-1, D))
        selections = router_logits.argmax(dim=-1)
        for e in range(E):
            count = (selections == e).sum().item()
            print(f"  Expert {e}: {count} tokens ({count / (B * T):.0%})")
```

---

## Interview takeaways

1. **MoE = conditional computation** — not every parameter is used for every token. This decouples parameter count from compute cost. Be ready to explain the economic argument.
2. **Top-k routing** — each token is routed to the top \(k\) experts (usually 2). The outputs are weighted by the router's softmax probabilities.
3. **Load balancing** — without it, the router collapses. Know the auxiliary loss formula and that it's added to the main training loss.
4. **Mixtral numbers** — 8 experts, top-2, 46.7B total, 12.9B active. Matches Llama 2 70B quality at ~3× lower inference cost.
5. **Memory vs. compute** — MoE saves compute but not memory. All experts must be loaded. This makes MoE models harder to serve on consumer hardware.
6. **Expert specialization** — a common interview question. In practice, experts don't cleanly specialize by domain/topic. Specialization is more at the token/feature level.
7. **All-to-all communication** — in distributed training/inference, tokens may need to be sent to different GPUs hosting different experts. This is a systems bottleneck.

---

## References

- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*
- Fedus et al. (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*
- Jiang et al. (2024), *Mixtral of Experts*
- Zhou et al. (2022), *Mixture-of-Experts with Expert Choice Routing*
