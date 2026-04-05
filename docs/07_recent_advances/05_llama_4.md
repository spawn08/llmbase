# Llama 4 — April 2025

## What Changed

**Meta's Llama 4 family** (Scout, Maverick, Behemoth) introduced the first open-weight **natively multimodal MoE** models — trained on up to 40 trillion tokens across 200 languages with early-fusion multimodality (images and text in the same token sequence, not a bolt-on vision encoder).

**Scout** (17B active / 16 experts, ~109B total) achieved a **10 million token** context window and fits on a single H100 GPU. **Maverick** (17B active / 128 experts, ~400B total) matched GPT-4o and DeepSeek-V3 on reasoning and coding benchmarks. **Behemoth** (288B active / 16 experts) is in training as a distillation teacher.

## Key Technical Details

**Early-fusion multimodality** trains from the beginning with interleaved image patches and text tokens in a single sequence. The model does not need a separate vision encoder — images are tokenized into patch embeddings in the same space as text from pretraining:

\[
\mathbf{h} = \mathrm{Transformer}\bigl([\mathbf{e}^{\text{img}}_1,\ldots,\mathbf{e}^{\text{img}}_{N_v},\; \mathbf{e}^{\text{txt}}_1,\ldots,\mathbf{e}^{\text{txt}}_{N_t}]\bigr)
\]

**MoE architecture with iRoPE** (interleaved RoPE): attention layers alternate between standard RoPE and NoPE (no positional encoding) to support extreme context lengths without positional aliasing.

!!! math-intuition "In Plain English"
    Scout's 10M token context works in part because some attention layers have no positional encoding — they rely on content-based attention rather than position-based patterns. This avoids the frequency collapse that limits standard RoPE at extreme lengths.

??? info "Technical Details"
    - **Scout efficiency**: 17B active parameters; total ~109B experts. Fits on one H100 — comparable latency to a dense 7B model with more capacity.
    - **Maverick routing**: 128 experts with top-K routing; 17B active. Beats GPT-4o on several benchmarks.
    - **Behemoth**: Teacher model for distillation into Scout/Maverick. Not yet publicly released.
    - **License**: Llama 4 Community License — commercial use permitted with usage caps; review before production.
    - **Training data**: 40T tokens, 200 languages, image+text interleaved from day 1.

## Practical Implications

**Scout** fits on a single H100 GPU with a 10M-token context window; **Maverick** matches GPT-4o and DeepSeek-V3 on reasoning and coding. Review the **Llama 4 Community License** before production—commercial use is permitted with usage caps. **Behemoth** remains a distillation teacher, not a generally deployed endpoint.

!!! interview "Interview Questions"
    1. What is **early-fusion multimodality** and how does it differ from a frozen vision encoder + projection approach? What are the training cost trade-offs?
    2. Scout uses **iRoPE** with alternating RoPE/NoPE layers to achieve 10M context — why does removing positional encoding from some layers help at extreme context lengths?
    3. How does Maverick's 128-expert MoE at 17B active parameters achieve comparable quality to much larger dense models?
    4. Why is Behemoth used as a distillation teacher rather than deployed directly?

## Code Example

```python
import transformers
pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device_map="auto",
    torch_dtype="auto",
)
out = pipeline([{"role": "user", "content": "Explain MoE routing in two sentences."}])
print(out[0]["generated_text"][-1]["content"])
```
