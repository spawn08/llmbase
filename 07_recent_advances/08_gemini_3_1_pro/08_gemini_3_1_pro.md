# Gemini 3.1 Pro — February 2026

## What Changed

**Gemini 3.1 Pro** scored 77.1% on ARC-AGI-2 and 94.3% on GPQA Diamond — tying GPT-5.4 Pro on the Intelligence Index at roughly one-third the cost per token. It cemented Google's position at the frontier and demonstrated that **compute efficiency** had become as important a competitive axis as raw benchmark performance.

## Key Technical Details

ARC-AGI-2 tests novel reasoning patterns not seen in training — it is specifically designed to resist memorization. High performance requires genuine compositional generalization:

\[
P(\text{correct} \mid \text{novel pattern}) \neq P(\text{correct} \mid \text{seen pattern})
\]

A model that memorizes training distributions fails ARC-AGI-2 even at very large scale.

!!! math-intuition "In Plain English"
    **Compositional generalization** means solving new puzzles built from familiar primitives in unfamiliar combinations. If the model only pattern-matches training examples, ARC-AGI-2 punishes that — it rewards **rule induction** and **transfer**, not recall.

??? info "Technical Details"
    - **Cost efficiency**: ~3x cheaper per token than GPT-5.4 Pro at comparable quality — a meaningful production trade-off.
    - **Native multimodal**: text, images, audio, video; continuation of the Gemini native-multimodal approach.
    - **Grounded search**: integrated Google Search grounding produces citations and factual grounding in outputs.

## Practical Implications

When **cost per token** drops at fixed quality, re-evaluate batch jobs (summarization, classification) that were previously uneconomical at frontier quality. For **grounded search**, design prompts that require citations and validate URLs in post-processing. For **ARC-style** reasoning, prefer evaluations that mix novel rule systems over static knowledge benchmarks alone.

!!! interview "Interview Questions"
    1. Why is **ARC-AGI-2** considered a harder benchmark than MMLU or GPQA? What reasoning failure mode does it specifically target?
    2. How does **cost per token** become a competitive axis at the frontier, and what architectural choices drive inference efficiency at scale?

## Code Example

Illustrative **grounded generation** request (parameters and names vary by API version):

```json
{
  "model": "gemini-3.1-pro",
  "contents": [{ "role": "user", "parts": [{ "text": "Summarize today's news on EU AI regulation with citations." }] }],
  "tools": [{ "google_search": {} }]
}
```
