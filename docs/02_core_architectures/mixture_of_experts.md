# Mixture of Experts (MoE)

## Why This Matters for LLMs

Dense Transformers spend the same compute on every token: each layer’s feed-forward network (FFN) activates all weights for every position in the sequence. **Mixture of Experts** replaces that dense FFN with a bank of parallel **expert** networks and a lightweight **router** that sends each token to only a small subset of experts (often top-1 or top-2). Total parameter count can grow into the tens or hundreds of billions while **per-token active parameters** stay closer to a mid-sized dense model. That split is central to how teams reason about frontier systems: you separate **memory footprint and total capacity** from **realized FLOPs per token**.

MoE is a recurring topic in **system design** interviews because routing creates new failure modes: load imbalance across GPUs, all-to-all communication in distributed training, expert collapse during optimization, and tricky fine-tuning dynamics when only a fraction of parameters receives gradients for a given batch. Saying "MoE scales parameters" without mentioning routing stability and serving costs is incomplete. A strong answer connects the gating softmax, top-\(k\) selection, auxiliary losses, and the operational reality that **all experts must typically reside in device memory** even when inactive for a particular token.

Finally, MoE sits at the intersection of model architecture and **hardware**. Parallel expert shards map to multi-GPU layouts; expert choice routing inverts the assignment problem to improve balance; products like open-weight Mixtral models made MoE tangible for developers who experiment locally. Understanding MoE is therefore not only a reading-comprehension exercise about papers: it is preparation for discussing how a 47B-parameter checkpoint might still behave like a 13B active model at inference time, and why that matters for latency budgets and batching strategies.

---

## Router and Gate: From Logits to a Weighted Mixture

Let \(\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}\) be a token hidden vector entering an MoE layer. A learned linear map produces **logits** over \(E\) experts:

\[
\mathbf{z} = W_g \mathbf{x} + \mathbf{b}_g, \quad \mathbf{z} \in \mathbb{R}^E
\]

!!! math-intuition "In Plain English"
    Logits are **raw scores** assigned to each expert before normalization. A higher \(z_i\) means the linear router favors expert \(i\) for this hidden vector \(\mathbf{x}\), but logits are not probabilities until passed through softmax.

\[
\mathbf{g} = \text{softmax}(\mathbf{z}), \quad g_i = \frac{\exp(z_i)}{\sum_{j=1}^{E} \exp(z_j)}
\]

!!! math-intuition "In Plain English"
    Softmax converts logits into a vector \(\mathbf{g}\) whose entries are positive and sum to one. You can read \(g_i\) as the router’s **intent** to use expert \(i\) before sparsification. If one logit is much larger than the others, \(\mathbf{g}\) becomes sharply peaked.

For **top-\(k\)** routing with small \(k\) (often 1 or 2), define the set \(\mathcal{S}(\mathbf{x})\) as the indices of the \(k\) largest entries of \(\mathbf{g}\). The MoE output is a weighted sum of expert functions \(f_i\):

\[
\text{MoE}(\mathbf{x}) = \sum_{i \in \mathcal{S}(\mathbf{x})} \tilde{g}_i \, f_i(\mathbf{x})
\]

!!! math-intuition "In Plain English"
    This sum is a **sparse mixture**: only experts whose indices lie in \(\mathcal{S}(\mathbf{x})\) are evaluated. Each selected expert maps \(\mathbf{x}\) through its own FFN \(f_i\). The mixture combines those vector outputs using nonnegative weights \(\tilde{g}_i\).

where \(\tilde{g}_i\) are the selected gate values renormalized to sum to one over the chosen experts:

\[
\tilde{g}_i = \frac{g_i}{\sum_{j \in \mathcal{S}(\mathbf{x})} g_j}
\]

!!! math-intuition "In Plain English"
    Top-\(k\) masking removes most experts, so the surviving raw weights \(g_i\) may no longer sum to one. Renormalization divides by their sum so \(\{\tilde{g}_i\}_{i \in \mathcal{S}}\) forms a proper convex combination of the expert outputs.

!!! example "Worked Example: Eight Experts, Top-Two Routing for One Token"
    Let \(E = 8\). Suppose the router logits for a specific token are:

    \[
    \mathbf{z} = [2.0,\ 0.5,\ -1.0,\ 3.5,\ 0.0,\ -0.5,\ 1.0,\ -2.0]
    \]

    Compute softmax probabilities (using natural exponentials):

    - \(\exp(3.5) \approx 33.12\), \(\exp(2.0) \approx 7.39\), \(\exp(1.0) \approx 2.72\), \(\exp(0.5) \approx 1.65\), \(\exp(0.0) = 1.00\), \(\exp(-0.5) \approx 0.61\), \(\exp(-1.0) \approx 0.37\), \(\exp(-2.0) \approx 0.14\)

    Sum \(\approx 33.12 + 7.39 + 2.72 + 1.65 + 1.00 + 0.61 + 0.37 + 0.14 \approx 47.00\) (rounded).

    Normalized weights (approximate):

    - \(g_4 \approx 33.12 / 47.00 \approx 0.705\)  
    - \(g_1 \approx 7.39 / 47.00 \approx 0.157\)  
    - \(g_8 \approx 2.72 / 47.00 \approx 0.058\)  
    - remaining indices carry the rest.

    Top-\(k\) with \(k = 2\) selects experts 4 and 1 (indices 4 and 1 if one-indexed naming is used carefully; here 0-indexed they are index 3 and index 0). Suppose the two largest are indices **3** and **0** in zero-based numbering (values 3.5 and 2.0). Renormalize:

    \[
    \tilde{g}_3 = \frac{g_3}{g_3 + g_0}, \quad \tilde{g}_0 = \frac{g_0}{g_3 + g_0}
    \]

    Using approximate \(g_3 \approx 0.705\) and \(g_0 \approx 0.157\):

    \[
    \tilde{g}_3 \approx \frac{0.705}{0.705 + 0.157} \approx 0.818, \quad \tilde{g}_0 \approx \frac{0.157}{0.862} \approx 0.182
    \]

    If expert outputs are vectors \(f_0(\mathbf{x}) = \mathbf{o}_0\) and \(f_3(\mathbf{x}) = \mathbf{o}_3\), the MoE output is \(\mathbf{y} \approx 0.818\, \mathbf{o}_3 + 0.182\, \mathbf{o}_0\). Only those two experts execute a full FFN for this token.

---

## Load Balancing and the Auxiliary Loss

A practical MoE must avoid **router collapse**, where a few experts receive almost all tokens. A common training-time fix adds an auxiliary loss that encourages **both** high routing probability mass and **actual token counts** to spread across experts.

Let \(f_i\) be the fraction of tokens in a batch whose routing includes expert \(i\) (counting each token’s top-\(k\) slots), normalized so \(\sum_i f_i = k\) when each token selects \(k\) experts and each selection is counted. Let \(P_i\) be the mean of \(g_i\) across tokens in the batch. A Switch-style balance term often resembles:

\[
\mathcal{L}_{\text{balance}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i
\]

!!! math-intuition "In Plain English"
    Think of \(f_i\) as **how often expert \(i\) actually fires** and \(P_i\) as **how strongly the router wanted expert \(i\) on average**. If the router assigns high probability to an expert but that expert rarely runs, or the opposite, the product \(f_i P_i\) can drift away from a uniform ideal. Multiplying by \(E\) scales the penalty relative to the number of experts. The coefficient \(\alpha\) trades off routing regularization against the main language modeling loss.

!!! example "Worked Example: Load Balancing Loss"
    **Setup:** \(E = 8\) experts, batch of **16 tokens**, top-\(k = 2\) so each token contributes two expert selections, 32 expert-slots total.

    **Step 1 — Count routed slots per expert (toy but concrete).**  
    Suppose after routing, expert slot counts are:

    | Expert index | Routed slots (out of 32) |
    | --- | --- |
    | 0 | 10 |
    | 1 | 2 |
    | 2 | 2 |
    | 3 | 2 |
    | 4 | 2 |
    | 5 | 2 |
    | 6 | 6 |
    | 7 | 6 |

    Check sum: \(10 + 2 + 2 + 2 + 2 + 2 + 2 + 6 + 6 = 32\).

    Convert to \(f_i\) as **fraction of expert-slots**:

    \[
    f_0 = 10/32 = 0.3125,\ f_1 = 2/32 = 0.0625,\ f_2 = 0.0625,\ f_3 = 0.0625,
    \]
    \[
    f_4 = 0.0625,\ f_5 = 0.0625,\ f_6 = 6/32 = 0.1875,\ f_7 = 0.1875
    \]

    **Step 2 — Router probability mass per expert (batch mean of \(g_i\)).**  
    Suppose the average softmax probabilities are:

    | Expert | \(P_i\) |
    | --- | --- |
    | 0 | 0.30 |
    | 1 | 0.08 |
    | 2 | 0.08 |
    | 3 | 0.08 |
    | 4 | 0.08 |
    | 5 | 0.08 |
    | 6 | 0.15 |
    | 7 | 0.15 |

    Sum \(P_i = 1.00\).

    **Step 3 — Compute \(\sum_i f_i P_i\):**

    - \(f_0 P_0 = 0.3125 \times 0.30 = 0.09375\)  
    - \(f_1 P_1 = 0.0625 \times 0.08 = 0.005\)  
    - Experts 2–5 each: \(0.0625 \times 0.08 = 0.005\) (four experts → \(0.02\))  
    - \(f_6 P_6 = 0.1875 \times 0.15 = 0.028125\)  
    - \(f_7 P_7 = 0.1875 \times 0.15 = 0.028125\)

    Sum:

    \[
    0.09375 + 0.02 + 0.028125 + 0.028125 = 0.17
    \]

    (since \(0.09375 + 0.02 = 0.11375\), plus \(0.05625 = 0.17\)).

    **Step 4 — Scale by \(E\) and \(\alpha\).**  
    With \(\alpha = 0.01\):

    \[
    \mathcal{L}_{\text{balance}} \approx 0.01 \times 8 \times 0.17 = 0.0136
    \]

    **Interpretation:** Expert 0 is both frequently chosen and strongly preferred by the router, driving a larger product term. Training nudges the router and representations so that **traffic** and **intent** align more evenly, reducing idle experts.

---

## Expert Choice Routing (Tokens versus Experts)

Classic routing: **tokens pick experts** via softmax and top-\(k\). **Expert choice** routing flips the assignment: each expert selects its top tokens from a pool. This can enforce **perfect balance** because each expert takes a fixed quota.

!!! math-intuition "In Plain English"
    Imagine a job fair. Token-choice routing lets every candidate apply to their favorite employers, which can produce long lines at popular booths. Expert-choice routing lets each booth interview a fixed number of applicants chosen from the crowd. The second scheme caps wait imbalance by construction, which helps hardware utilization at the cost of a different scheduling algorithm and sometimes different gradient statistics.

??? deep-dive "Deep Dive: Why Expert Choice Changes Implementation"
    Token-choice routing maps naturally to "for each token, gather expert outputs." Expert-choice routing requires sorting or grouping tokens per expert after an initial scoring pass. Distributed systems must implement **buffering** and **capacity** carefully to avoid dropping tokens. Interview answers should mention that load balance improves but **communication patterns** and **kernel fusion** requirements change.

---

## Scaling Properties: Mixtral and the Active Parameter Budget

Public Mixtral-style models illustrate the **total versus active** split. A commonly cited profile for **Mixtral 8×7B** is on the order of **47B total parameters** with roughly **13B active parameters per token** when two experts are selected from eight (the precise headline numbers vary slightly by counting conventions and shared tensors).

!!! math-intuition "In Plain English"
    Total parameters count every expert’s weights sitting on disk or in GPU memory. Active parameters count the FFN weights actually multiplied during the forward pass for a token, plus shared components like attention projections. The ratio explains why MoE can match a much larger dense model’s quality without multiplying inference FLOPs by the full depth of all experts.

---

## Notable Models (Orientation Table)

| Model family | Experts (order of magnitude) | Top-\(k\) (typical) | Notes |
| --- | --- | --- | --- |
| Switch Transformer | Hundreds to low thousands | 1 | Early large-scale MoE training recipes |
| GLaM | Dozens | 2 | Large sparse mixture with dense counterparts in studies |
| Mixtral 8×7B | 8 | 2 | Widely deployed open-weights MoE LLM |
| Mixtral 8×22B | 8 | 2 | Larger experts, higher total parameter count |
| DeepSeek-V2 | Large (hundreds) | >2 | Strong open model line with MoE FFN blocks |
| Grok-1 | 8 (reported in many summaries) | 2 | Closed weights; cited in public benchmarking discussions |

**GPT-4** is widely speculated to use MoE-style sparsity, but public sources do not provide authoritative architecture tables. In interviews, phrase it as **industry rumor** unless you have private knowledge.

??? deep-dive "Deep Dive: Memory Versus Compute in Serving"
    MoE reduces compute per token relative to a dense model with the same expert width, but the **checkpoint size** still includes all experts. Serving stacks therefore worry about **VRAM**, **expert offloading**, **CPU RAM fallback**, and **batching** to saturate GPUs. A complete system answer mentions **all-to-all** dispatch when experts shard across devices.

---

## Training Challenges: Collapse, Instability, Communication

1. **Expert collapse** — the router sends most tokens to one expert, wasting capacity.
2. **Router instability** — sharp softmax distributions early in training can freeze exploration; remedies include noise on logits, temperature, auxiliary losses, and curriculum.
3. **Communication overhead** — distributed training must move activations to the devices that own the selected experts.
4. **Fine-tuning brittleness** — small task-specific datasets may overfit a subset of experts unless learning rates, regularization, and data mixing are tuned.

!!! interview "FAANG-Level Questions"
    1. Define MoE at the layer level and contrast dense FFN with sparse expert FFN.
    *Answer:* MoE replaces one dense FFN with **\(E\)** expert FFNs and a **router**; each token activates only **top-\(k\)** experts (often 1–2). **Dense** runs all FFN weights every time; **sparse** runs a **subset**, so **active FLOPs** stay small while **total** capacity (parameters on disk) is large.
    2. Write the softmax gate equation and explain top-\(k\) sparsification.
    *Answer:* \(g_i=\exp(z_i)/\sum_j\exp(z_j)\) over expert logits \(z=W_g x\). **Top-\(k\)** keeps only the \(k\) largest \(g_i\), **zeroing** the rest, then **renormalizes** the surviving weights—sparse mixture before expert FFN evaluation.
    3. Why does load balancing matter, and what does an auxiliary loss penalize?
    *Answer:* If routing collapses to a few experts, **others idle**—wasted capacity and **poor GPU utilization**. Auxiliary losses (e.g. Switch-style \( \propto \sum_i f_i P_i\)) penalize mismatch between **actual token counts** \(f_i\) and **average router mass** \(P_i\), encouraging **uniform** load.
    4. Compare token-choice and expert-choice routing at a systems level.
    *Answer:* **Token-choice:** each token picks experts—popular experts get **hotspots** and imbalance. **Expert-choice:** each expert picks a **quota** of tokens—**balance by construction**, but scheduling/sorting and gradient patterns differ; implementation is more like **assignment** than per-token gather.
    5. Given total parameters and active parameters, how do you reason about inference cost?
    *Answer:* **Latency/FLOPs** track **active** experts per token (e.g. 2 of 8) plus shared attention—not the sum of all experts. **Throughput** is bounded by **executing** only those FFNs; total params inflate **memory** for storing all experts, not per-token multiply count.
    6. What causes all-to-all communication in distributed MoE training?
    *Answer:* Experts are **sharded across devices**; after routing, each token’s activations must be **sent to the devices owning the selected experts** and results **gathered** back—**all-to-all** (or grouped) communication across the expert mesh.
    7. Why might MoE fine-tuning behave differently from dense fine-tuning?
    *Answer:* Only **routed** experts receive strong gradients for a given batch—**sparse updates** can overfit subsets, cause **router instability**, or need lower LR / auxiliary losses. Small domain datasets may **activate** a narrow expert subset unless regularized.
    8. How would you detect expert collapse in telemetry (histograms of expert indices)?
    *Answer:* Track **histograms** of `argmax` or top-\(k\) expert IDs per step: collapse shows **few bars** dominating mass. Alert when entropy of the routing distribution drops or min/max load ratio exceeds thresholds across experts.
    9. Why is VRAM pressure often worse than FLOPs for MoE deployment?
    *Answer:* **All** expert weights usually **reside in device memory** (or slow offload) even if inactive per token—checkpoint size scales with \(E\). FLOPs are low per token, but **fitting** dozens of expert shards in VRAM and **bandwidth** for routing dominate serving.
    10. Explain why Mixtral can approach larger dense quality with smaller active compute.
    *Answer:* **Total** parameters (~47B in 8×7B style) increase **capacity** and specialization; **top-2** routing activates only ~**13B**-scale FFN work per token. Quality tracks **total knowledge** while inference FLOPs track **active** experts—similar to a large ensemble with sparse execution.

!!! interview "Follow-up Probes"
    - "What happens if one expert is overloaded beyond its capacity cap?"
    - "How does mixed-precision training interact with router numerics?"
    - "Would you shard experts by layer or across layers on the same device?"
    - "How do you evaluate whether experts specialize meaningfully?"

!!! key-phrases "Key Phrases to Use in Interviews"
    - "Conditional computation: sparse activation of FFN experts"
    - "Gating softmax plus top-\(k\) selection with renormalized weights"
    - "Auxiliary load-balancing loss on \(f_i\) and \(P_i\)"
    - "Total parameters versus active parameters per token"
    - "Distributed all-to-all dispatch for expert shards"

---

## Full Code: MoE Layer with Top-\(k\) Routing and Auxiliary Loss

The module below is **self-contained**, uses explicit loops for clarity (production stacks use fused kernels), and reports the auxiliary loss using the same structural terms as the explanation above.

```python
"""
Mixture of Experts (MoE) layer: top-k routing, renormalized weights, load-balancing loss.

Requires: torch >= 2.0 recommended (works with 1.x for basic tensors).

Educational implementation: loops per expert for transparency; production code uses grouping.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """One expert: SwiGLU-style feed-forward block."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(w1(x)) * w3(x) then w2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoELayer(nn.Module):
    """
    Top-k MoE with Switch-style balance loss:
      aux = balance_coeff * E * sum_i f_i * P_i
    where f_i is the fraction of routed slots to expert i,
    P_i is the mean softmax probability for expert i across tokens.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 8,
        top_k: int = 2,
        balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k
        self.balance_coeff = balance_coeff
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUExpert(d_model, d_ff) for _ in range(n_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, time, d_model)
        returns: output with same shape, scalar aux_loss
        """
        b, t, d = x.shape
        flat = x.reshape(b * t, d)
        n_tokens = flat.size(0)

        logits = self.gate(flat)
        probs = F.softmax(logits, dim=-1)

        topk_weight, topk_idx = probs.topk(self.top_k, dim=-1)
        # Renormalize selected weights so each token's mixture sums to 1.
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        out = torch.zeros_like(flat)

        # Route: for each k slot, accumulate weighted expert outputs.
        for slot in range(self.top_k):
            sel = topk_idx[:, slot]
            w = topk_weight[:, slot]
            for expert_id in range(self.n_experts):
                mask = sel == expert_id
                if not mask.any():
                    continue
                sub_in = flat[mask]
                sub_out = self.experts[expert_id](sub_in)
                out[mask] = out[mask] + w[mask].unsqueeze(-1) * sub_out

        # Load-balancing: f_i uses routed slots (k per token).
        counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        for slot in range(self.top_k):
            counts.scatter_add_(0, topk_idx[:, slot], torch.ones_like(topk_idx[:, slot], dtype=torch.float32))
        total_slots = float(n_tokens * self.top_k)
        f = counts / max(total_slots, 1.0)
        p_mean = probs.mean(dim=0)
        aux_loss = self.balance_coeff * float(self.n_experts) * torch.sum(f * p_mean)

        return out.view(b, t, d), aux_loss


def demo_parameter_ratio(d_model: int, d_ff: int, n_experts: int, top_k: int) -> None:
    """Print parameter counts: gate plus all experts versus active expert FFNs per token."""
    moe = MoELayer(d_model, d_ff, n_experts=n_experts, top_k=top_k)
    gate_params = moe.gate.weight.numel()
    one_expert_params = sum(p.numel() for p in moe.experts[0].parameters())
    all_expert_params = sum(p.numel() for p in moe.experts.parameters())
    print(f"Gate parameters: {gate_params:,}")
    print(f"One expert (SwiGLU) parameters: {one_expert_params:,}")
    print(f"All {n_experts} experts total: {all_expert_params:,}")
    print(f"MoE layer total (gate + experts): {gate_params + all_expert_params:,}")
    print(f"Per token, top-{top_k} uses about {top_k} expert forward passes out of {n_experts}.")


if __name__ == "__main__":
    torch.manual_seed(0)
    d_model, d_ff = 128, 256
    batch, time, experts, k = 2, 16, 8, 2
    layer = MoELayer(d_model, d_ff, n_experts=experts, top_k=k, balance_coeff=0.01)
    x = torch.randn(batch, time, d_model)
    y, aux = layer(x)
    print("output:", tuple(y.shape))
    print("aux loss:", float(aux))
    demo_parameter_ratio(d_model, d_ff, experts, k)
```

---

## References

- Shazeer et al. (2017), *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*
- Fedus et al. (2021), *Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*
- Jiang et al. (2024), *Mixtral of Experts*
- Zhou et al. (2022), *Mixture-of-Experts with Expert Choice Routing*
