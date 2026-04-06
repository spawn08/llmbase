# Kimi K2.5: Trillion-Parameter MoE with Native Vision and Agent Swarm

**Authors:** Moonshot AI (Kimi Team) &nbsp;|&nbsp; **Year:** 2026 &nbsp;|&nbsp; **Venue:** arXiv &nbsp;|&nbsp; **Link:** [moonshot.ai/kimi-k2.5](https://moonshot.ai/kimi-k2.5)

---

## TL;DR

Kimi K2.5 is a **1-trillion-parameter MoE** model (32B active, 384 experts with 8 activated per token) released open-source in January 2026. Three innovations define this release: (1) **native multimodal** capabilities through continual pretraining on ~15T mixed visual-text tokens with a **MoonViT 400M** vision encoder, (2) **dual operating modes** — a single checkpoint switches between step-by-step thinking and fast direct responses, and (3) **Agent Swarm** — a research-preview system that orchestrates up to **100 parallel sub-agents** making up to **1,500 tool calls** in a single workflow, trained with **Parallel-Agent Reinforcement Learning (PARL)**.

**Headline results:** 4.5× speedup on embarrassingly parallel tasks via Agent Swarm; 256K context at 32B active parameters; visual-to-code generation from UI screenshots and Figma exports.

**One-sentence pitch:** K2.5 pushes the frontier from **single-agent reasoning** to **multi-agent coordination**, trained end-to-end with RL to decompose and parallelize complex tasks.

---

## Why This Paper Matters

- **Multi-agent RL:** PARL is the first published approach to training a **coordinator** agent to decompose tasks for **parallel execution** by sub-agents, using RL with task-completion rewards. This moves beyond prompt-engineering multi-agent systems into **learned orchestration**.
- **Native vs bolt-on multimodality:** Continual pretraining on interleaved image-text data with MoonViT produces fundamentally different vision-language representations than plugging a frozen vision encoder post-hoc — relevant for interview questions about multimodal architectures.
- **Dual thinking modes:** A single checkpoint serving both fast and reasoning workloads simplifies deployment — one model, one serving stack, two latency profiles.
- **Massive MoE at 1T scale:** 384 experts with top-8 routing at 1T total parameters pushes MoE engineering to its limits in load balancing, memory management, and expert parallelism.
- **Visual-to-code pipeline:** End-to-end training for converting UI designs (screenshots, Figma) into frontend code eliminates separate OCR + code-generation stages.
- **Interview relevance:** Combines MoE scaling, multimodal pretraining, RL for multi-agent systems, and practical serving — a rich intersection of ML and systems topics.

---

## Key Concepts Explained Simply

### 1. Agent Swarm and PARL (Parallel-Agent Reinforcement Learning)

Most LLM agent systems execute tasks **sequentially** — the model thinks, acts, observes, repeats. Agent Swarm introduces **parallel execution**: a **coordinator** agent decomposes a request into independent sub-tasks, dispatches them to specialized **sub-agents** (each with tool access), collects results, and synthesizes a final output.

!!! math-intuition "In Plain English"
    Think of **MapReduce for LLM tasks**: the coordinator is the **mapper** (decomposes work), sub-agents are **workers** (execute in parallel), and the coordinator's synthesis step is the **reducer** (aggregates results). PARL trains the coordinator to make good decomposition decisions via RL — the reward is task completion quality, so the coordinator learns which tasks can be parallelized and which must be sequential.

**Training PARL:**

1. Coordinator proposes a task decomposition \(D = \{t_1, t_2, \ldots, t_n\}\) with dependency graph.
2. Sub-agents execute independent tasks in parallel; dependent tasks wait.
3. Coordinator synthesizes results.
4. **Reward:** task-completion quality + latency bonus for parallelism.
5. **RL update:** coordinator policy is updated to improve decomposition quality.

### 2. Native multimodality with MoonViT

**Bolt-on approach** (LLaVA-style): pretrain a language model, freeze a vision encoder (CLIP), train a projection layer to map visual features into the language model's embedding space. The language model never sees visual tokens during pretraining.

**Native approach** (K2.5): perform **continual pretraining** on interleaved image-text data. MoonViT 400M processes images at **variable resolution** (preserving native aspect ratios), and visual tokens are interleaved with text tokens in the pretraining data. The language model learns joint representations from the start.

| Approach | Pros | Cons |
|----------|------|------|
| **Bolt-on** | Simpler, preserves text-only quality | Vision features don't influence language representations; limited grounding |
| **Native** | Deep vision-language integration; better on spatial/visual reasoning | Requires retraining; potential regression on text-only tasks without careful mixing |

### 3. Dual thinking modes

A single K2.5 checkpoint supports:

- **Thinking mode:** generates internal chain-of-thought reasoning tokens before the final answer. Activated via system prompt or API parameter.
- **Non-thinking mode:** skips reasoning overhead for straightforward queries.

Unlike earlier approaches that deployed separate "fast" and "reasoning" model variants, K2.5 learns both behaviors during RL training — the model itself selects the appropriate depth when in automatic mode.

### 4. MoE at 1T scale (384 experts, top-8)

With 384 experts and 8 active per token:

- **Expert selection diversity:** each token chooses from \(\binom{384}{8} \approx 1.2 \times 10^{14}\) possible expert combinations — enormous representational diversity.
- **Load balancing challenge:** with 384 experts, ensuring roughly uniform utilization requires careful routing bias adjustment or auxiliary loss.
- **Memory footprint:** all 384 expert FFN blocks must be accessible, even though only 8 are used per token — requiring distributed expert placement across GPUs.

### 5. Visual-to-code pipeline

K2.5 converts UI designs (screenshots, Figma exports, video walkthroughs) directly into frontend code. This is trained **end-to-end** — the model processes visual tokens from the UI image and generates code tokens, without intermediate OCR or layout-detection stages. The variable-resolution MoonViT encoder is critical here, as UI screenshots vary widely in aspect ratio and detail level.

---

## The Math — Explained Step by Step

### 1. PARL objective for coordinator training

Let the coordinator policy \(\pi_C\) produce a decomposition \(D\) given a task description \(q\):

\[
D \sim \pi_C(\cdot \mid q), \quad D = \{(t_i, \text{deps}_i)\}_{i=1}^n
\]

Sub-agents execute tasks: \(r_i = \text{SubAgent}(t_i)\). The coordinator synthesizes: \(y = \text{Synthesize}(q, \{r_i\})\).

The PARL reward combines **quality** and **efficiency**:

\[
R(D, y) = R_{\text{quality}}(y, q) + \beta \cdot \frac{T_{\text{sequential}}}{T_{\text{parallel}}(D)}
\]

where \(T_{\text{sequential}}\) is the baseline sequential execution time and \(T_{\text{parallel}}(D)\) is the actual wall-clock time with parallelism. The ratio rewards decompositions that achieve genuine speedup.

The coordinator is updated via policy gradient:

\[
\nabla_\theta J = \mathbb{E}_{D \sim \pi_C}\left[R(D, y) \cdot \nabla_\theta \log \pi_C(D \mid q)\right]
\]

### 2. MoonViT variable-resolution encoding

Standard vision transformers resize all images to a fixed resolution (e.g., 224×224), losing detail in high-resolution images and wasting compute on low-resolution ones. MoonViT processes images at their **native resolution** by:

1. Dividing the image into patches of fixed size \(p \times p\) (e.g., 14×14 pixels).
2. The number of patches varies: \(N_{\text{patches}} = \lceil H/p \rceil \times \lceil W/p \rceil\).
3. 2D positional encodings preserve spatial layout regardless of image dimensions.

For an image of height \(H\) and width \(W\):

\[
\text{Visual tokens} = \text{MoonViT}\left(\text{patches}(I, p)\right) \in \mathbb{R}^{N_{\text{patches}} \times d}
\]

These visual tokens are then interleaved with text tokens in the transformer's input sequence.

### 3. MoE routing at 384-expert scale

For input \(h\), router scores for all 384 experts:

\[
s_i = W_r^{(i)} \cdot h, \quad i = 1, \ldots, 384
\]

Top-8 selection:

\[
\text{TopK}(s, 8) = \{i_1, \ldots, i_8\} \quad \text{where } s_{i_1} \geq \cdots \geq s_{i_8}
\]

Gating weights (normalized over selected experts):

\[
g_{i_k} = \frac{\exp(s_{i_k})}{\sum_{j=1}^{8} \exp(s_{i_j})}
\]

Output: \(\text{MoE}(h) = \sum_{k=1}^{8} g_{i_k} \cdot E_{i_k}(h)\)

**Load balancing** at this scale uses dynamic bias adjustment (per DeepSeek-V3): maintain a running utilization count per expert; if expert \(i\) is over-utilized, decrease \(b_i\); if under-utilized, increase \(b_i\). Modified scores: \(s_i' = s_i + b_i\).

### 4. Memory footprint analysis

With 384 experts, each containing FFN parameters:

\[
\text{Memory}_{\text{experts}} = 384 \times 2 \times d_{\text{model}} \times d_{\text{ffn}} \times \text{bytes\_per\_param}
\]

For 1T total params in BF16 (~2 bytes/param): \(\approx 2\text{TB}\) just for weights. KV cache for 256K context at 32B active params adds significant memory. Practical serving requires **expert parallelism** across 25+ GPUs (80GB each), with careful placement to minimize inter-GPU communication for the 8 active experts per token.

### 5. Dual-mode training objective

The training combines standard language modeling with mode-conditioned behavior:

\[
\mathcal{L} = \mathcal{L}_{\text{LM}} + \alpha \cdot \mathcal{L}_{\text{thinking}} + \gamma \cdot \mathcal{L}_{\text{PARL}}
\]

where \(\mathcal{L}_{\text{thinking}}\) trains the model to produce (or skip) chain-of-thought based on a mode token, and \(\mathcal{L}_{\text{PARL}}\) trains the coordinator decomposition behavior. Mode selection during RL training teaches the model to **automatically** choose thinking depth based on query complexity.

---

## Python Implementation

The following demonstrates **task decomposition and parallel execution** — the core Agent Swarm pattern, with a simplified PARL-style reward computation.

```python
"""
Simplified Agent Swarm: coordinator decomposes tasks, sub-agents execute
in parallel, coordinator synthesizes. Educational implementation.
"""
from __future__ import annotations

import time
import concurrent.futures
from dataclasses import dataclass


@dataclass
class SubTask:
    id: int
    description: str
    dependencies: list[int]


@dataclass
class SubTaskResult:
    task_id: int
    output: str
    execution_time: float


def execute_subtask(task: SubTask) -> SubTaskResult:
    """Simulate a sub-agent executing a task."""
    start = time.monotonic()
    time.sleep(0.1)  # simulate work
    output = f"Result for: {task.description}"
    elapsed = time.monotonic() - start
    return SubTaskResult(task_id=task.id, output=output, execution_time=elapsed)


def topological_layers(tasks: list[SubTask]) -> list[list[SubTask]]:
    """Group tasks into layers for parallel execution respecting dependencies."""
    completed: set[int] = set()
    remaining = list(tasks)
    layers: list[list[SubTask]] = []

    while remaining:
        layer = [t for t in remaining if all(d in completed for d in t.dependencies)]
        if not layer:
            raise ValueError("Circular dependency detected")
        layers.append(layer)
        completed.update(t.id for t in layer)
        remaining = [t for t in remaining if t.id not in completed]

    return layers


def agent_swarm_execute(tasks: list[SubTask]) -> tuple[list[SubTaskResult], float, float]:
    """
    Execute tasks with dependency-aware parallelism.
    Returns (results, parallel_time, sequential_time_estimate).
    """
    layers = topological_layers(tasks)
    results: list[SubTaskResult] = []
    parallel_start = time.monotonic()

    for layer in layers:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(layer)) as pool:
            futures = {pool.submit(execute_subtask, t): t for t in layer}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

    parallel_time = time.monotonic() - parallel_start
    sequential_time = sum(r.execution_time for r in results)
    return results, parallel_time, sequential_time


def parl_reward(
    quality_score: float,
    sequential_time: float,
    parallel_time: float,
    beta: float = 0.3,
) -> float:
    """
    PARL-style reward: quality + parallelism bonus.
    Higher speedup ratio = higher reward.
    """
    speedup = sequential_time / max(parallel_time, 1e-6)
    return quality_score + beta * speedup


def demo() -> None:
    tasks = [
        SubTask(0, "Parse requirements", []),
        SubTask(1, "Design database schema", [0]),
        SubTask(2, "Design API endpoints", [0]),
        SubTask(3, "Write backend code", [1, 2]),
        SubTask(4, "Write frontend code", [2]),
        SubTask(5, "Write tests", [3, 4]),
    ]

    layers = topological_layers(tasks)
    print("Execution layers (parallel within each layer):")
    for i, layer in enumerate(layers):
        print(f"  Layer {i}: {[t.description for t in layer]}")

    results, par_t, seq_t = agent_swarm_execute(tasks)
    reward = parl_reward(quality_score=0.85, sequential_time=seq_t, parallel_time=par_t)

    print(f"\nSequential time: {seq_t:.3f}s")
    print(f"Parallel time:   {par_t:.3f}s")
    print(f"Speedup:         {seq_t/par_t:.2f}x")
    print(f"PARL reward:     {reward:.3f}")


if __name__ == "__main__":
    demo()
```

---

## Interview Importance

K2.5 is a **four-topic interview model**: (1) MoE at extreme scale (384 experts), (2) native multimodal pretraining, (3) multi-agent RL, and (4) dual inference modes. Expect questions comparing Agent Swarm to prompt-based multi-agent frameworks (AutoGen, CrewAI) — the key difference is **learned decomposition** via PARL vs **hand-designed** orchestration.

**Drill themes:** (1) Native vs bolt-on vision — when does each win? (2) 384-expert load balancing — what happens when experts collapse? (3) PARL — how do you assign credit to the coordinator vs sub-agents? (4) 256K context at 32B active — memory vs compute bottleneck.

---

## Interview Questions & Answers (6 Q&As)

**Q1: What are the trade-offs between native multimodal pretraining (K2.5) vs bolt-on vision encoders (LLaVA)?**
**A:** **Native** pretraining interleaves image-text data during continual training, so the language model's representations are influenced by visual inputs from early in training — better for **spatially grounded** reasoning (UI layouts, diagrams, charts). **Bolt-on** freezes a vision encoder and trains only a projection layer, preserving text-only quality but limiting how deeply visual features integrate into language representations. Native is better for tasks requiring **joint reasoning** (visual-to-code, diagram understanding); bolt-on is simpler and preserves text benchmarks.

**Q2: How does PARL train the coordinator to decompose tasks for parallel execution?**
**A:** The coordinator proposes a decomposition \(D = \{(t_i, \text{deps}_i)\}\) via autoregressive generation. Sub-agents execute, and the result quality plus a **speedup bonus** (sequential time / parallel time) forms the reward. Policy gradient updates the coordinator to improve both **task decomposition quality** (correct sub-tasks, right dependencies) and **parallelism** (more independent sub-tasks = higher speedup bonus). Credit assignment is challenging — the coordinator gets credit for decomposition, not sub-agent execution quality.

**Q3: With 384 experts and 8 active per token, what is the memory footprint for serving K2.5?**
**A:** All 384 expert FFN blocks must be loaded (~2TB for 1T params in BF16), even though only 8 are active per token. Compute cost per token matches a ~32B dense model, but memory must hold the full 1T. This requires at least **25 × 80GB GPUs** for weights alone, plus KV cache for 256K context. **Expert parallelism** distributes experts across GPUs; the router determines which 8 experts activate, and tokens must be communicated to the GPUs hosting those experts — **all-to-all communication** is the bottleneck.

**Q4: Explain the practical challenges of running 100 parallel sub-agents with 1,500 tool calls.**
**A:** (1) **Rate limiting:** external tools (APIs, databases) may throttle concurrent requests. (2) **Error propagation:** one sub-agent failure can cascade — need retry logic, fallback strategies, and partial-result aggregation. (3) **Result consistency:** parallel sub-agents may produce conflicting outputs that the coordinator must reconcile. (4) **Resource contention:** 100 concurrent model inferences compete for GPU memory and compute. (5) **Latency tail:** the overall latency is bounded by the slowest sub-agent. Production systems need timeouts and graceful degradation.

**Q5: How does the dual thinking mode work in a single checkpoint?**
**A:** During RL training, the model learns to produce chain-of-thought tokens when a "thinking" mode token is present and to skip them otherwise. Both behaviors are trained in the same RL loop — thinking mode gets harder problems where reasoning helps, non-thinking mode gets simpler queries. The key insight is **mode-conditional behavior** stored in the same weights, selected by a control token rather than separate model architectures. Automatic mode selection adds a routing decision trained to predict query complexity.

**Q6: Compare K2.5's Agent Swarm to prompt-based multi-agent frameworks (AutoGen, CrewAI).**
**A:** **Prompt-based** frameworks define agent roles and communication protocols through prompts and code — the decomposition logic is **hand-designed** by engineers. **Agent Swarm** uses PARL to **learn** decomposition from task-completion rewards — the coordinator discovers what can be parallelized through trial and error. Prompt-based is more **interpretable** and **controllable**; PARL is more **adaptive** but harder to debug. In practice, production systems likely combine both — PARL for decomposition, hand-designed guardrails for safety.

---

## Connections to Other Papers

| Paper / Line | Connection to Kimi K2.5 |
|--------------|------------------------|
| **Kimi k1.5** | Direct predecessor — K2.5 inherits the long-context RL philosophy and extends it with multimodality and multi-agent coordination. |
| **DeepSeek-R1** | Both use RL for reasoning; R1 focuses on **single-agent** GRPO, K2.5 extends to **multi-agent** PARL coordination. |
| **GLM-5** | Both target agentic capabilities; GLM-5 via single-agent tool mastery with async RL (Slime), K2.5 via parallel multi-agent orchestration. |
| **CLIP** | Foundational vision-language contrastive learning; K2.5's MoonViT is a descendant approach with variable resolution for LLM integration. |
| **Gemini** | Both natively multimodal; Gemini uses early fusion at pretraining, K2.5 uses continual pretraining with MoonViT — different integration strategies. |
| **Llama 4** | Both open-weight and natively multimodal MoE; Scout's 10M context vs K2.5's 256K represent different scaling priorities (context vs agents). |
| **ReAct** | Single-agent thought-action-observation loop; Agent Swarm generalizes to N parallel ReAct loops coordinated by a learned decomposer. |
| **Qwen 3** | Shares the dual thinking mode design; both train a single checkpoint for fast and reasoning modes via RL. |

---

## Key Takeaways for Quick Review (table)

| Topic | One-liner |
|-------|-----------|
| **Scale** | **1T** total, **32B** active, **384** experts (top-8); 256K context; open-source. |
| **Agent Swarm** | Up to **100 parallel sub-agents**, **1,500 tool calls** per workflow; trained with **PARL**. |
| **PARL** | RL trains a **coordinator** to decompose tasks; reward = quality + speedup bonus. |
| **Native vision** | **MoonViT 400M** with variable resolution; continual pretraining on 15T mixed tokens. |
| **Dual modes** | Single checkpoint: **thinking** (CoT) and **non-thinking** (fast); auto-selectable. |
| **Visual-to-code** | End-to-end UI screenshot/Figma → frontend code pipeline, no separate OCR stage. |
| **Memory** | ~2TB for weights; 25+ GPUs at 80GB each; expert parallelism + all-to-all routing. |
| **Speedup** | **4.5×** on embarrassingly parallel tasks; gains diminish with sequential dependencies. |
| **vs prompt agents** | PARL **learns** decomposition; AutoGen/CrewAI **hand-design** it — different trade-offs. |
| **Interview frame** | MoE scale, native multimodal, multi-agent RL, serving costs — four themes in one model. |
