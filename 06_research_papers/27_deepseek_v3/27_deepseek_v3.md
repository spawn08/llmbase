# DeepSeek-V3 Technical Report

**Authors:** DeepSeek-AI **Year:** 2024 &nbsp;|&nbsp; **Venue:** arXiv **Link:** [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

---

## TL;DR

DeepSeek-V3 is a **671B** mixture-of-experts (MoE) language model with **37B active parameters per token**, trained on **14.8T tokens** in about **2.788M H800 GPU-hours**. Three headline innovations: **(1)** auxiliary-loss-free load balancing via per-expert bias terms in the router softmax, **(2)** **multi-token prediction (MTP)** to provide denser training signal with auxiliary heads, and **(3)** **FP8** mixed-precision training for roughly **2×** throughput. The model reuses **multi-head latent attention (MLA)** from DeepSeek-V2. Empirically it **matches frontier closed-source models** on many benchmarks.

If you remember only three symbols for the whiteboard: **\(g_{i,t}=\mathrm{Softmax}(e_{i,t}+b_i)\)** for routing, **\(\mathcal{L}_{\mathrm{MTP}}\)** for extra-token supervision, and **\(\hat{B}=\mathrm{round}(B/s_B)\)** for FP8 blocks—then you can reconstruct the rest of the story.

---

## Why This Paper Matters

MoE training historically trades off **routing quality** against **load balance**: auxiliary balancing losses keep experts from collapsing, but they **perturb the main LM objective**. DeepSeek-V3 shows you can **balance experts without that auxiliary term**, using **bias-controlled softmax routing** updated from utilization statistics. That matters for anyone building large MoE systems at scale.

The paper also packages **MTP** as a practical training trick—extra heads predict future tokens for a **stronger gradient signal**, then disappear at inference—and documents **FP8** recipes and **DualPipe** pipeline parallelism that make **671B-scale** training tractable. For interviews, it is a compact reference for **routing math**, **speculative decoding hooks**, and **low-precision training** narratives.

Benchmark framing in the report emphasizes **closed-model parity** on reasoning, coding, and knowledge tasks—useful when you need a **single citation** that ties together **MoE routing**, **training efficiency**, and **end-task quality** rather than only architecture diagrams.

---

## Key Concepts Explained Simply

### 1. Auxiliary-loss-free load balancing

Classic MoE adds a **balancing loss** so no expert is starved or overloaded. That loss is **not** the same as next-token prediction, so it can **interfere** with what you actually want to optimize.

DeepSeek-V3 keeps a **per-expert bias** \(b_i\) in the router. The gate for expert \(i\) at time \(t\) is:

\[
g_{i,t} = \frac{\exp(e_{i,t} + b_i)}{\sum_j \exp(e_{j,t} + b_j)}.
\]

**Intuition:** If expert \(i\) is **over-used**, **lower** \(b_i\) so its softmax mass drops; if **under-used**, **raise** \(b_i\). The primary loss stays clean; balance is enforced **outside** the main cross-entropy.

### 2. MTP (Multi-Token Prediction)

Besides the main head that predicts token \(t{+}1\), **auxiliary heads** predict \(t{+}2, t{+}3, \ldots, t{+}k\) given the **causal** representations up to \(t\). Training gets **more supervision per forward**; at **inference** those heads are **discarded**. Reported gains are on the order of **~2%** average on benchmarks; MTP can also pair with **speculative decoding** with high acceptance (**85–90%** in their discussion).

### 3. FP8 training

Activations and weights are represented in **FP8** with **block-wise** (or vector) **scale factors**. Accumulation can be **FP8 inside** a block and **FP32 across** blocks for stability. Net effect: about **2×** FLOPs throughput and **half** the memory bandwidth of BF16 for the quantized tensors—when the stack and kernels support it.

### 4. DualPipe

**Pipeline parallelism** is arranged so **forward/backward** stages overlap with **communication**, using **duplicated parameter shards** so compute and transfer **hide** each other—aiming for **near-perfect** overlap and higher hardware utilization.

### 5. Architecture snapshot

- **MLA** (from V2) for attention efficiency.
- **DeepSeekMoE**: **256** routed experts **+ 1** shared expert, **top-8** routing per token.

The **shared expert** is always applied (alongside the selected routed experts), giving a **stable** representation pathway and separating **always-on** computation from **sparse** expert choice—similar in spirit to **shared + specialized** splits in other MoE designs, but tuned to DeepSeek’s depth and width.

### Practical pitfalls (interview color)

- **Routing jitter:** If \(\gamma\) is too large, \(b\) oscillates and **hurts convergence**; production systems use **EMA** on utilization, **smaller steps**, and **bounds** on \(b_i\).
- **Top‑\(K\) vs softmax mass:** Training often uses **noisy top‑\(K\)** or **straight-through** estimators; know whether your story is **discrete routing** or **soft** gates.
- **FP8 outliers:** A few **large-magnitude** activations can dominate a block scale—**clipping**, **gradient norms**, and **layer norm placement** interact with FP8 more than BF16.
- **MTP depth \(K\):** Larger \(K\) adds **compute** in training; teams tune \(\alpha_k\) so MTP helps **representations** without **dominating** the main LM loss.

---

## The Math — Explained Step by Step

### Expert routing with biases

Let \(e_{i,t}\) be the **router logit** for expert \(i\) at position \(t\). With biases \(b_i\):

\[
g_{i,t} = \mathrm{Softmax}_i\bigl(e_{\cdot,t} + b\bigr)_i = \frac{\exp(e_{i,t} + b_i)}{\sum_{j=1}^{N} \exp(e_{j,t} + b_j)}.
\]

A simple **utilization-driven** update (conceptually): let \(\bar{u}_i\) be a **moving average** of how often expert \(i\) is selected (or the load proxy used in the paper). Compare to a **target** utilization \(u^\*\). One schematic rule:

\[
b_i \leftarrow b_i - \gamma \cdot \mathrm{sign}(\bar{u}_i - u^\*),
\]

with small \(\gamma > 0\): **over-utilized** experts (\(\bar{u}_i > u^\*\)) get **smaller** \(b_i\); **under-utilized** experts get **larger** \(b_i\). The paper’s implementation uses **statistics over batches** and **clipping** for stability; the key idea is **decoupling balance from the CE loss**.

### MTP auxiliary loss

Let \(h_t\) be the hidden state after processing tokens up through \(t\) (causal). The main LM loss predicts \(x_{t+1}\). For \(k \geq 2\), an auxiliary head \(f_k\) predicts \(x_{t+k}\):

\[
\mathcal{L}_{\mathrm{MTP}} = \sum_{k=2}^{K} \alpha_k \, \mathbb{E}_t\bigl[-\log p_\theta(x_{t+k} \mid h_t)\bigr],
\]

with \(\alpha_k\) weights; total training loss is \(\mathcal{L} = \mathcal{L}_{\mathrm{LM}} + \mathcal{L}_{\mathrm{MTP}}\) (plus any other terms **except** a large MoE balancing loss). At inference, only the standard next-token head is used.

**Causal chain:** each MTP head conditions on the **same** \(h_t\) (information available at \(t\)), not on future hidden states—so the **graph stays causal** and matches autoregressive training. The extra heads only **reuse** \(h_t\) to peek **forward along the label sequence**, similar in flavor to **multi-step** auxiliary losses in older sequence models but implemented for **large Transformers**.

### FP8 block-wise quantization

Split a matrix into blocks of size \(N_c \times N_c\). For each block \(B\), choose a **scale** \(s_B\) (often derived from max absolute value in the block). **Quantize** entries to FP8:

\[
\hat{B} = \mathrm{round}\bigl(B / s_B\bigr),
\]

and **dequantize** \(\tilde{B} = s_B \cdot \hat{B}\) for higher-precision accumulation where needed. **Matmul** can use **FP8 tensor cores** on \( \hat{A}, \hat{B} \) with scales, while **partial sums** across blocks accumulate in **FP32** to reduce drift.

### Block-scaled GEMM (sketch)

For blocks \(p,q\) in an output tile, a schematic product is:

\[
C_{pq} \approx \sum_r (s^{(A)}_{pr} s^{(B)}_{rq}) \cdot (\hat{A}_{pr} \hat{B}_{rq}),
\]

with **FP32** accumulation of the sum. Hardware typically fuses **scale multiply** with **dot** into one kernel; the **point** for interviews is: **scales are per-block** (or per-vector), not one global scalar—this captures **dynamic range** without giving up FP8 speed.

---

## Python Implementation

Below is a **minimal, didactic** simulation of **auxiliary-loss-free** routing: softmax logits plus **per-expert biases** updated from **empirical selection rates** vs a **target** share. This is **not** the full production kernel; it shows **how \(b_i\) moves** over training steps.

```python
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def route_one_step(
    logits: np.ndarray,
    bias: np.ndarray,
    top_k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return gate weights g_i and binary top-k mask for one token."""
    g = softmax(logits + bias)
    top_idx = np.argpartition(-g, top_k)[:top_k]
    mask = np.zeros_like(g)
    mask[top_idx] = 1.0
    return g, mask


def update_bias_no_aux_loss(
    bias: np.ndarray,
    usage_count: np.ndarray,
    batch_tokens: int,
    target_share: float,
    gamma: float,
    b_min: float,
    b_max: float,
) -> np.ndarray:
    """
    usage_count: how many times each expert was selected (e.g., in top-k) in the window.
    Move bias down for over-used experts, up for under-used ones.
    """
    share = usage_count / max(batch_tokens, 1)
    delta = target_share - share  # positive => under-used => increase bias
    bias = bias + gamma * delta
    return np.clip(bias, b_min, b_max)


def demo_training_loop(
    num_experts: int = 256,
    steps: int = 200,
    tokens_per_step: int = 1024,
    top_k: int = 8,
    gamma: float = 0.02,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    bias = np.zeros(num_experts, dtype=np.float64)

    for step in range(steps):
        # Fake router logits: some experts "naturally" preferred — biases should correct load
        base = rng.normal(0.0, 1.0, size=(tokens_per_step, num_experts))
        trend = np.linspace(-1.0, 1.0, num_experts)
        logits_batch = base + trend  # skewed preferences

        usage = np.zeros(num_experts, dtype=np.float64)
        for t in range(tokens_per_step):
            _, mask = route_one_step(logits_batch[t], bias, top_k, rng)
            usage += mask

        # Each of (tokens * top_k) slot assignments; uniform load => fraction 1/N per expert
        target = 1.0 / num_experts
        bias = update_bias_no_aux_loss(
            bias,
            usage,
            tokens_per_step * top_k,
            target_share=target,
            gamma=gamma,
            b_min=-2.0,
            b_max=2.0,
        )

        if step % 40 == 0 or step == steps - 1:
            share = usage / usage.sum()
            print(f"step={step:4d}  H(share)={-np.sum(share * np.log(share + 1e-12)):.3f}  max_share={share.max():.4f}")

    print("Final bias stats: mean=%.4f std=%.4f" % (bias.mean(), bias.std()))


if __name__ == "__main__":
    demo_training_loop()
```

**EMA variant (closer to production spirit):** maintain \(\bar{u}_i \leftarrow \beta \bar{u}_i + (1-\beta) \hat{u}_i\) with batch utilization \(\hat{u}_i\), then update \(b_i\) from \(\bar{u}_i - u^\*\). This **smooths** noise when **batch size** or **sequence length** varies.

**How to read the run:** early steps often concentrate mass on a subset of experts; as **bias updates** fire, **selection shares** spread toward the **target** \( \approx 1/N \) of total **top‑\(K\)** slot assignments—**without** adding a balancing term to the cross-entropy.

---

## Interview Importance

- **MoE routing** is a standard senior/Staff topic: you should explain **top‑k gating**, **load imbalance**, and **why auxiliary losses hurt**.
- **Training at FP8** signals you know **numerics**, **scaling**, and **hardware** (bandwidth vs compute).
- **MTP** connects to **better training signals** and **speculative decoding**—common follow-up for inference optimization.
- **DualPipe / pipeline efficiency** separates “model math” from **systems** questions on **bubble reduction** and **all-to-all** expert communication.
- Be ready to write **\(g_{i,t}\)** on a whiteboard and explain **why \(b_i\) is not the same as** a learned logit **shift** tied to CE—it's **load-driven**.

---

## Interview Questions & Answers (6 Q&As with ### Q1: format, **Answer:** format)

### Q1: What is expert collapse in MoE, and how does DeepSeek-V3 mitigate it without a balancing loss?

**Answer:** **Expert collapse** means a few experts receive almost all tokens while others are idle—wasting capacity and destabilizing training. Classical MoE often adds an **auxiliary balancing loss** to equalize load, but that **adds gradient interference**. DeepSeek-V3 instead uses **per-expert biases** \(b_i\) inside \(\mathrm{Softmax}(e + b)\) and **updates** \(b_i\) from **utilization** so over-used experts are **down-weighted** and under-used experts **up-weighted**, keeping the **primary LM loss unmodified**.

### Q2: Why might an auxiliary balancing loss “pollute” the main objective?

**Answer:** The LM is trained to minimize **next-token cross-entropy** (and related terms). A balancing loss pushes **routing** toward uniform usage; its gradients can **conflict** with logits that would best **minimize CE**. You get a **multi-objective tradeoff** where \(\lambda\) on the auxiliary term is hard to tune. **Bias-based balancing** moves that concern **outside** the main CE by adjusting **router parameters** (here, \(b\)) on **statistics**, not by mixing another loss into the token prediction objective at the same weight level.

### Q3: What does MTP buy you in training, and why are auxiliary heads dropped at inference?

**Answer:** MTP adds **auxiliary predictions** of \(x_{t+2}, \ldots, x_{t+K}\) from \(h_t\), increasing **supervision density** per sequence position. That tends to improve **representation quality** and gave ~**2%** average benchmark gains in their report. At inference, the model only needs **standard autoregressive** next-token sampling; the extra heads are **training-only** and can be **removed** to save memory and latency.

### Q4: How does FP8 training improve efficiency, and what keeps it stable?

**Answer:** FP8 uses **narrower** formats so **tensor cores** achieve higher **throughput** and **memory bandwidth** drops vs BF16—often ~**2×** FLOPs for supported ops. Stability comes from **mixed precision**: **block-wise scales**, **FP8 accumulation inside blocks**, and **FP32 accumulation across blocks** limits error growth; careful **loss scaling** and **clipping** (as in large-model practice) remain important.

### Q5: What is DualPipe in one sentence, and why does overlap matter?

**Answer:** **DualPipe** is a **pipeline-parallel** schedule with **duplicated shards** so **forward/backward compute** overlaps **inter-device communication**, approaching **full overlap** and higher **MFU**. Overlap matters because at this scale **network** time otherwise **bubbles** the pipeline and wastes **H800 hours**.

### Q6: How does DeepSeek-V3 compare conceptually to Mixtral-style MoE?

**Answer:** **Mixtral** popularized **top‑k experts per layer** with a clear **open recipe**; DeepSeek-V3 is **much larger** (671B total, 37B active), uses **DeepSeekMoE** (**256** routed + **1** shared, **top‑8**), **MLA**, **MTP**, **FP8**, and **bias-only load control** instead of a heavy **auxiliary balance loss**. Both are MoE, but V3 emphasizes **training-system** innovations (precision, pipeline, routing) at **frontier scale**.

---

## Connections to Other Papers

| Paper / family | Connection |
|----------------|------------|
| **DeepSeek-V2** | **MLA** and MoE lineage; V3 extends training and routing efficiency. |
| **Mixtral** | Reference MoE **open** baseline; contrast routing, scale, and training tricks. |
| **LLaMA** | Dense **decoder-only** LLM line; useful contrast to **sparse** activation. |
| **Chinchilla** | **Compute-optimal scaling** context for **14.8T tokens** and model size tradeoffs. |
| **DeepSeek-R1** | Later reasoning-focused work; V3 is the **base model / stack** story. |
| **Switch / GShard** | Earlier MoE at scale; compare **routing**, **capacity**, and **balancing** mechanisms. |

**Study order:** read **V2 (MLA)** first for attention compression, then **V3** for **routing + training stack**, then **R1** if the interview focuses on **reasoning**—V3 is the **pre-reasoning** capability base in that narrative.

---

## Key Takeaways for Quick Review (table)

| Topic | One-liner |
|-------|-----------|
| **Scale** | 671B MoE, **37B active**/token, **14.8T** tokens, **~2.788M H800** GPU-hours. |
| **Routing** | \(g_{i,t}=\mathrm{Softmax}(e_{i,t}+b_i)\); **bias updates** from **utilization**, **no** classic balance loss. |
| **MTP** | Extra heads predict **t+2…t+k** in training; **~2%** avg gains; **removed** at inference; helps **speculative** paths. |
| **FP8** | **Block-scaled** FP8, **FP32** across blocks; ~**2×** throughput vs BF16 in their setting. |
| **DualPipe** | **Pipeline** parallelism with **duplication** for **compute/comm overlap**. |
| **Architecture** | **MLA** + **256** routed + **1** shared expert, **top-8** routing. |
| **Result** | **Matches** strong **closed-source** baselines on reported benchmarks. |
| **Bias update** | Track **utilization** \(\bar{u}_i\) vs \(u^\*\); adjust \(b_i\) so **over-used** experts lose mass—**no** auxiliary balance term in CE. |
| **Whiteboard check** | Write \(\mathcal{L}_{\mathrm{MTP}}\), \(\hat{B}= \mathrm{round}(B/s_B)\), and **top‑\(K\)** on \(g_{i,t}\). |

---

### Revision checklist (60 seconds)

1. State **671B / 37B active** and **14.8T tokens**.
2. Contrast **bias routing** vs **auxiliary balance loss** in one sentence.
3. Name **MTP** training-only heads and **~2%** uplift + **speculative** link.
4. Recite **FP8**: block scales, **FP32** across blocks, **~2×** throughput vs BF16.
5. Mention **DualPipe** as **overlap** for pipeline parallelism.
6. Close with **MLA + 256 + 1 shared, top‑8** and **closed-model** parity.
