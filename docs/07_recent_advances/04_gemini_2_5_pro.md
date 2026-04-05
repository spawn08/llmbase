# Gemini 2.5 Pro — March–June 2025

## What Changed

**Google DeepMind's Gemini 2.5 Pro** became the top-ranked model on LMArena (Elo 1470) and WebDevArena (1443) after its full release in June 2025. Its defining features: **1 million token context**, **Deep Think mode** (configurable reasoning budget up to 32K thinking tokens), and **grounded search** natively integrated.

Gemini 2.5 Pro moved the context scaling frontier from 128K to 1M tokens while maintaining competitive quality on multi-hop reasoning and coding tasks.

## Key Technical Details

**Configurable thinking budget**: unlike o1/o3 which have opaque reasoning, Gemini 2.5 Pro exposes `thinking_budget` as a parameter controlling how many tokens the model may spend on internal reasoning before answering. This makes cost/quality trade-offs explicit and debuggable.

\[
\text{Total cost} = \text{prefill tokens} + \text{thinking tokens} + \text{output tokens}
\]

**Thought summaries** expose a compressed version of the reasoning trace — valuable for enterprise auditability without exposing full internal chains.

??? info "Technical Details"
    - **1,048,576 input tokens**: enables loading entire codebases, legal corpora, or book-length documents in a single prompt.
    - **Deep Think**: routes complex requests to a reasoning mode analogous to o1 — math, code, planning tasks. Light requests bypass it.
    - **Multimodal**: text, code, images, audio, video in the same context.
    - **Knowledge cutoff**: January 2025 (at GA).

```json
{
  "model": "gemini-2.5-pro",
  "thinking_config": {
    "thinking_budget": 8000,
    "include_thoughts": false
  },
  "contents": [...]
}
```

## Practical Implications

Use **`thinking_budget`** when you want explicit control over internal reasoning depth versus models with opaque reasoning. Bill and forecast cost using **prefill + thinking + output** tokens. Use **thought summaries** for enterprise auditability without exposing full internal chains. The **1,048,576**-token context window enables whole codebases, legal corpora, or book-length documents in a single prompt.

!!! interview "Interview Questions"
    1. How does exposing `thinking_budget` as an API parameter change the cost/quality trade-off compared to a model with a fixed reasoning depth?
    2. Why does extending context from 128K to 1M tokens create new **serving** challenges even if the model quality holds — what tensor shapes and memory layouts are affected?
    3. What is the "lost-in-the-middle" problem, and how does it interact with 1M-token contexts?

## Code Example

```json
{
  "model": "gemini-2.5-pro",
  "thinking_config": {
    "thinking_budget": 8000,
    "include_thoughts": false
  },
  "contents": [...]
}
```
