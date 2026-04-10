# Qwen 3 — April 2025

## What Changed

**Alibaba's Qwen 3** is a family of eight open-weight models (Apache 2.0) released in April 2025, spanning dense architectures (0.6B to 32B) and MoE architectures (30B-A3B and the flagship **235B-A22B**). Qwen 3 introduced **hybrid thinking modes** — a single checkpoint that switches between step-by-step reasoning and fast direct responses — and expanded language support from 29 to **119 languages**. A follow-up, **Qwen 3.5**, was released in February 2026 with further benchmark improvements.

## Key Technical Details

**Model lineup:**

| Model | Architecture | Total Params | Active Params | Context |
|-------|-------------|-------------|--------------|---------|
| Qwen3-0.6B to 32B | Dense | 0.6B–32B | All | 32K–128K |
| Qwen3-30B-A3B | MoE (128E / 8A) | 30B | 3B | 128K |
| Qwen3-235B-A22B | MoE (128E / 8A) | 235B | 22B | 128K |

All models use: **Grouped Query Attention (GQA)**, **SwiGLU** activation, **Rotary Positional Embeddings (RoPE)**, extendable to 1M tokens via **YaRN**.

**Hybrid thinking modes:** Unlike separate "thinking" and "non-thinking" model variants, Qwen 3 unifies both in a single model:

- **Thinking mode:** Generates internal chain-of-thought reasoning tokens before the final answer. Activated via system prompt or API parameter.
- **Non-thinking mode:** Skips reasoning overhead for straightforward queries.
- **Thinking budget:** Users can cap the number of reasoning tokens, creating a latency-quality trade-off slider.

!!! math-intuition "In Plain English"
    Previous generations required deploying **two models** (e.g., a fast model and a reasoning model) and routing between them. Qwen 3 merges both into one checkpoint — the model learned when and how deeply to reason during RL training. The "thinking budget" is conceptually similar to giving the model a compute allowance: more budget = deeper reasoning = higher accuracy on hard problems, but higher latency.

**Training pipeline:**

1. **Pre-training** on a large multilingual corpus
2. **Long-context extension** via YaRN (progressive scaling of RoPE base frequency)
3. **Thinking-mode training:** a two-stage RL process:
    - Stage 1: RL with only thinking mode enabled (learns to reason)
    - Stage 2: RL with both modes, teaching the model to select the appropriate mode

**Qwen3-Next (September 2025):** A hybrid architecture variant combining:

- **GatedDeltaNet** (linear attention) for efficient long-range processing
- **GatedAttention** (standard softmax attention) for precise local reasoning
- **MoE routing** with 512 experts

At sequences longer than 32K tokens, Qwen3-Next delivers **10× the throughput** of Qwen3-32B by using linear attention for most of the sequence and reserving full attention for critical positions.

## Benchmark Performance

Qwen3-235B-A22B is competitive with frontier models:

| Benchmark Category | Competitive With |
|-------------------|-----------------|
| Coding | DeepSeek-R1, o1 |
| Mathematics | o3-mini, Grok-3 |
| General knowledge | Gemini 2.5 Pro |
| Multilingual | SotA across 119 languages |

The small Qwen3-4B reportedly rivals Qwen2.5-72B-Instruct, demonstrating significant efficiency improvements in the training recipe.

## Practical Implications

**Unified thinking model** simplifies deployment — one model serves both fast and reasoning workloads, reducing infrastructure complexity. The thinking budget provides a practical latency-cost control that aligns well with tiered API pricing.

**119 languages** makes Qwen 3 the broadest multilingual open model, relevant for applications in underserved language markets.

**Qwen3-Next's hybrid architecture** points toward a future where linear attention handles the bulk of long sequences efficiently, while full attention is reserved for positions that need it — a potential path to truly sub-quadratic LLMs.

!!! interview "Interview Questions"
    1. How does Qwen 3's **thinking budget** mechanism work? How is it different from simply truncating the chain-of-thought output?
    2. Compare the **two-stage RL** training for hybrid thinking modes with DeepSeek-R1's approach. What does each stage optimize for?
    3. Explain how **YaRN** extends context from 128K to 1M tokens. What happens to positional encoding frequencies, and why does this preserve quality?
    4. What is the advantage of Qwen3-Next's **hybrid linear + softmax attention** over pure linear attention (like Mamba) or pure softmax attention? What are the trade-offs?
    5. With 128 experts and 8 active, how does the **expert routing** in Qwen3-235B-A22B compare to Mixtral's 8-expert/2-active design? What are the implications for load balancing?
