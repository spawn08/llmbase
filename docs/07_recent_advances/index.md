# Recent Advances in LLM Research

## How to Use This Section

A chronological log of frontier LLM developments from January 2025 onward. Each entry explains **what changed technically**, **why it matters**, and **what interviewers probe**. Use the timeline to orient yourself, then drill into individual entries for depth.

## Last Updated

April 2026

---

## Timeline at a Glance

| Date | Event | Significance |
|------|-------|-------------|
| Jan 2025 | DeepSeek-R1 & R1-Zero released | RL-only reasoning without SFT; GRPO goes mainstream |
| Jan 2025 | Kimi k1.5 (Moonshot AI) | Long-context RL scaling; matches o1 without MCTS |
| Feb 2025 | GPT-4.5 (OpenAI) | Last major GPT-4 generation; unsupervised learning scaling |
| Mar 2025 | Gemini 2.5 Pro preview (Google) | 1M context, Deep Think mode, configurable reasoning budget |
| Apr 2025 | Llama 4 Scout & Maverick (Meta) | First open-weight natively multimodal MoE; 10M ctx Scout |
| May 2025 | Gemini 2.5 Pro GA + I/O updates | Top LMArena ranking, grounded search |
| Jun 2025 | Gemini 2.5 Flash GA | 25% faster, 85% cheaper than Gemini 1.5 Pro |
| Aug 2025 | GPT-5 (OpenAI) | Unified fast+reasoning router; unified model replaces GPT-4o/o1 split |
| Sep 2025 | GLM-4.6 (Zhipu AI) | 357B MoE, 200K context, open-weight under MIT |
| Nov 2025 | Claude Opus 4.5 (Anthropic) | SotA on software engineering benchmarks |
| Feb 2026 | Gemini 3.1 Pro (Google) | 77.1% on ARC-AGI-2; cost-competitive with proprietary frontier |
| Mar 2026 | GPT-5.4 family (OpenAI) | 1M+ ctx, native computer use, mini/nano variants |
| Apr 2026 | Llama 4 Behemoth (Meta, in training) | 288B active / 16-expert MoE; distillation teacher |

---

## 1. DeepSeek-R1 and R1-Zero — January 2025

### What Changed

**DeepSeek-R1-Zero** became the first model trained via large-scale RL **without any supervised fine-tuning** as a prerequisite. Starting from DeepSeek-V3 base weights, RL with verifiable rewards taught the model to self-verify, reflect, and produce chain-of-thought traces—entirely from scratch. The model displayed an emergent **"Aha moment"** during training: a sudden qualitative shift in reasoning depth as RL scaling continued.

**DeepSeek-R1** then added a small cold-start SFT phase followed by the same RL recipe, producing stronger and more stable outputs. Both models matched OpenAI o1-1217 on AIME and MATH-500 benchmarks. Distilled student models (**R1-Distill-Qwen**, **R1-Distill-Llama**) brought reasoning-grade performance to 7B–70B open checkpoints.

### Key Technical Details

**Group Relative Policy Optimization (GRPO)** replaces the critic/value-network of PPO with group-averaged advantages. For a prompt \(q\), sample \(G\) completions \(\{o_1, \ldots, o_G\}\), score each with reward \(r_i\), and compute the advantage for completion \(i\) relative to the group mean:

\[
\hat{A}_i = \frac{r_i - \text{mean}(r_1,\ldots,r_G)}{\text{std}(r_1,\ldots,r_G)}
\]

The policy gradient objective clips the probability ratio as in PPO:

\[
\mathcal{J}_{\text{GRPO}} = \mathbb{E}_q \frac{1}{G}\sum_{i=1}^{G} \min\!\left(\rho_i \hat{A}_i,\ \mathrm{clip}(\rho_i, 1\!-\!\epsilon, 1\!+\!\epsilon)\hat{A}_i\right) - \beta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{ref}})
\]

where \(\rho_i = \pi_\theta(o_i \mid q) / \pi_{\mathrm{old}}(o_i \mid q)\).

!!! math-intuition "In Plain English"
    Instead of training a separate value network (expensive, unstable), GRPO uses the **other completions in the same batch** as the baseline. A response that scores better than its peers gets upweighted; one that scores worse gets downweighted. No critic, no bootstrapping.

**Verifiable reward signals** are central: math problems have ground-truth answers; code has unit tests. These hard signals are far more stable than human preference labels for multi-step reasoning.

??? info "Technical Details"
    - **R1-Zero only**: RL applied directly to the base model with no SFT warm-up — proved that reasoning capability is latent in well-pretrained models and can be elicited purely through RL.
    - **Cold-start data**: R1 adds ~thousands of long-CoT examples as SFT to stabilize initial RL training and prevent degenerate outputs (e.g., language mixing, repetition).
    - **Length penalty**: Without a penalty, models learn to pad traces. R1 adds a soft length reward to encourage concise but complete reasoning.
    - **Distillation**: R1-Distill models are trained on R1-generated traces via SFT — no RL required for the student. This transfers reasoning style at a fraction of the compute.
    - **GRPO bias**: Subsequent research (Dr. GRPO) found GRPO artificially inflates sequence length, particularly for incorrect responses, due to a normalization artifact — leading to corrected variants.

### Practical Implications

Parse `<think>...</think>` blocks before showing users final answers. Moderate the reasoning trace for PII and IP — the trace contains more information than the final reply. For production: budget reasoning tokens via max-length caps, not just final token caps.

!!! interview "Interview Questions"
    1. Why does GRPO not need a critic/value network, and what does it use instead? How does this differ from PPO?
    2. What is the "Aha moment" in R1-Zero training, and why is it significant for understanding emergent capabilities?
    3. Why are verifiable rewards (unit tests, symbolic solvers) more stable for RL training than human preference labels?
    4. How does distillation from R1 differ from training a student with GRPO directly?
    5. What failure mode does the KL penalty in GRPO prevent, and how does the \(\beta\) coefficient control the trade-off?

---

## 2. Kimi k1.5 — January 2025

### What Changed

**Moonshot AI's Kimi k1.5** demonstrated that long-context scaling — extending the policy's context window to 128K tokens — is itself a key axis for improving reasoning quality. Without MCTS, value functions, or process reward models, Kimi k1.5 matched OpenAI o1 on AIME (77.5) and MATH 500 (96.2) using a simpler RL framework.

The insight: a long context window lets the model **plan, reflect, and self-correct** within a single forward trajectory, because earlier reasoning steps remain in-context and can be referenced and revised.

### Key Technical Details

**Partial rollout reuse** avoids regenerating entire trajectories from scratch. When exploring long reasoning chains, the system reuses large chunks of already-computed trajectories and only generates novel suffixes. This dramatically reduces the per-training-step compute cost for 128K-token sequences.

**Online mirror descent** replaces standard gradient descent for policy optimization in this long-context regime:

\[
\pi_{t+1} = \arg\min_\pi \left[ -\mathbb{E}_\pi[R] + \frac{1}{\eta} D_{\mathrm{KL}}(\pi \| \pi_t) \right]
\]

!!! math-intuition "In Plain English"
    Mirror descent keeps each policy update close to the previous policy (via a KL ball) while maximizing expected reward. This is more stable than unconstrained gradient steps when the reward landscape is non-smooth over long sequences.

**Length penalty**: a soft penalty discourages unnecessarily long chains while preserving chains that need depth for correctness.

??? info "Technical Details"
    - **No MCTS**: MCTS requires many forward passes and a value function. Kimi k1.5 shows that a plain RL loop with long context achieves comparable benchmark results — a simpler, more scalable alternative.
    - **Short-CoT regime**: Kimi k1.5 also achieves strong results in the short-CoT regime (no explicit reasoning steps in the prompt), outperforming GPT-4o and Claude Sonnet 3.5 by up to +550% on some tasks.
    - **Multimodal**: trained jointly on text and vision data, enabling reasoning over image inputs.

### Practical Implications

Long-context RL is now a viable alternative to search-augmented inference. When deploying reasoning models, measure cost at the reasoning-token level, not just final output tokens — the 128K traces are the expensive part.

!!! interview "Interview Questions"
    1. Why does extending a model's context window improve its reasoning ability in RL training?
    2. How does online mirror descent stabilize policy updates compared to vanilla gradient ascent on reward?
    3. Kimi k1.5 achieves o1-level results without MCTS — what does this tell us about the relationship between search and in-context reasoning?

---

## 3. GPT-4.5 — February 2025

### What Changed

**GPT-4.5** was the last major release in the GPT-4 generation, focused on scaling **unsupervised learning** further — improving pattern recognition, creative reasoning, and emotional intelligence without primarily adding more instruction tuning. It reduced hallucinations and showed improved calibration (better uncertainty acknowledgment). It was the first model explicitly positioned around "knowing what it doesn't know."

GPT-4.5 was succeeded by GPT-5 in August 2025.

### Key Technical Details

The key investment was in pre-training compute and data quality rather than post-training RL. The model showed that **unsupervised scaling** (richer world models, better latent representations) independently improves alignment behaviors like honesty and calibration, without requiring more RLHF data.

!!! math-intuition "In Plain English"
    A better base model is easier to align. If the model has richer internal representations of uncertainty, post-training can refine rather than fight against the base distribution.

??? info "Technical Details"
    - **Pricing**: $75/M input tokens, $150/M output tokens at launch — significantly above GPT-4o, reflecting the compute investment.
    - **Emotional intelligence**: Notably improved on tasks requiring social reasoning, empathy modeling, and long-form coherent narrative.
    - **Deprecation**: Folded into "Legacy Models" after GPT-5 launch, still accessible for Pro subscribers.

!!! interview "Interview Questions"
    1. What is the distinction between scaling unsupervised pre-training vs. scaling post-training alignment — and how might each affect calibration differently?
    2. Why might a better base model be "easier to align," and what does this imply for the compute allocation between pre- and post-training?

---

## 4. Gemini 2.5 Pro — March–June 2025

### What Changed

**Google DeepMind's Gemini 2.5 Pro** became the top-ranked model on LMArena (Elo 1470) and WebDevArena (1443) after its full release in June 2025. Its defining features: **1 million token context**, **Deep Think mode** (configurable reasoning budget up to 32K thinking tokens), and **grounded search** natively integrated.

Gemini 2.5 Pro moved the context scaling frontier from 128K to 1M tokens while maintaining competitive quality on multi-hop reasoning and coding tasks.

### Key Technical Details

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

!!! interview "Interview Questions"
    1. How does exposing `thinking_budget` as an API parameter change the cost/quality trade-off compared to a model with a fixed reasoning depth?
    2. Why does extending context from 128K to 1M tokens create new **serving** challenges even if the model quality holds — what tensor shapes and memory layouts are affected?
    3. What is the "lost-in-the-middle" problem, and how does it interact with 1M-token contexts?

---

## 5. Llama 4 — April 2025

### What Changed

**Meta's Llama 4 family** (Scout, Maverick, Behemoth) introduced the first open-weight **natively multimodal MoE** models — trained on up to 40 trillion tokens across 200 languages with early-fusion multimodality (images and text in the same token sequence, not a bolt-on vision encoder).

**Scout** (17B active / 16 experts, ~109B total) achieved a **10 million token** context window and fits on a single H100 GPU. **Maverick** (17B active / 128 experts, ~400B total) matched GPT-4o and DeepSeek-V3 on reasoning and coding benchmarks. **Behemoth** (288B active / 16 experts) is in training as a distillation teacher.

### Key Technical Details

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

!!! interview "Interview Questions"
    1. What is **early-fusion multimodality** and how does it differ from a frozen vision encoder + projection approach? What are the training cost trade-offs?
    2. Scout uses **iRoPE** with alternating RoPE/NoPE layers to achieve 10M context — why does removing positional encoding from some layers help at extreme context lengths?
    3. How does Maverick's 128-expert MoE at 17B active parameters achieve comparable quality to much larger dense models?
    4. Why is Behemoth used as a distillation teacher rather than deployed directly?

---

## 6. GPT-5 — August 2025

### What Changed

**OpenAI's GPT-5** unified the previously split product line (GPT-4o for speed, o1/o3 for reasoning) into a single system with an **intelligent router** that dynamically selects between a fast main model and a deeper reasoning model based on query complexity.

GPT-5 represented a qualitative jump in: coding (complex front-end generation, debugging), health-domain questions, writing structure, and hallucination reduction. Available to all users; Pro users get access to `gpt-5-thinking-pro` for extended reasoning.

### Key Technical Details

**Adaptive routing** is the architectural novelty: a lightweight classifier routes each query to either `gpt-5-main` (fast, efficient) or `gpt-5-thinking` (deeper, slower). The router is trained on signals including user preferences, measured correctness, conversation type, and tool requirements.

\[
\text{route}(q) = \arg\max_{m \in \{\text{main, thinking}\}} P(\text{correct} \mid m, q) \cdot w_m
\]

where \(w_m\) incorporates latency and cost penalties for the heavy model.

!!! math-intuition "In Plain English"
    The router is itself a learned policy: given the query, decide whether it is worth paying the latency and cost of the reasoning model. Over time, the router improves from feedback — user thumbs-down on fast answers that were wrong teaches it to route more carefully.

**API surface**: introduces a `reasoning` field with `effort` level (`minimal`, `low`, `medium`, `high`) and a `verbosity` parameter, making reasoning budget explicit.

??? info "Technical Details"
    - **Reduced sycophancy**: GPT-5 was specifically trained to minimize agreement-seeking behavior — a known failure mode of RLHF-trained models.
    - **Improved calibration**: Better uncertainty expression; more frequent "I don't know" or "I'm not sure" when the model genuinely lacks knowledge.
    - **Computer use**: not a primary feature at GPT-5 launch but available in GPT-5.4.
    - **Pricing**: lower than GPT-4.5 at launch, reflecting inference efficiency improvements.

!!! interview "Interview Questions"
    1. What are the engineering challenges of a **routing architecture** (fast main + deep reasoning model)? How do you ensure the router decision is itself low-latency?
    2. Why is **sycophancy** a specific failure mode of RLHF training — what in the training signal causes it, and how would you measure and mitigate it?
    3. How does the unified router model change the **user experience** compared to explicitly choosing between GPT-4o and o3?

---

## 7. Claude Opus 4.5 — November 2025

### What Changed

**Anthropic's Claude Opus 4.5** achieved state-of-the-art on software engineering benchmarks (SWE-bench Verified), with particular strength in multi-step agentic coding tasks. Designed for **computer use** — controlling browsers and GUIs autonomously — and long-running agents that must maintain coherent plans across many tool calls.

### Key Technical Details

Constitutional AI at scale: Opus 4.5 continues Anthropic's RLAIF approach where model-generated critiques guided by written principles provide training signal without large-scale human labeling. The key advance: **critique models** specialized per domain (code review, safety, factual grounding) rather than a single general critic.

\[
\pi_{\theta}^{*} = \arg\max_\theta \mathbb{E}_{x \sim \pi_\theta}\bigl[r_\phi^{\text{domain}}(x) - \beta \log\frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}\bigr]
\]

!!! math-intuition "In Plain English"
    Each domain critic (code, factual, safety) provides its own reward signal. The policy maximizes a blend of domain-specific rewards while staying close to the reference model.

??? info "Technical Details"
    - **SWE-bench Verified**: Anthropic claims SotA; measures a model's ability to resolve GitHub issues end-to-end in a sandboxed environment.
    - **Computer use API**: standardized interface for GUI actions (click, type, screenshot) — enables "agent as employee" workflows.
    - **Pricing**: $5/$25 per million input/output tokens.
    - **Context**: 200K token context window.

!!! interview "Interview Questions"
    1. What is **RLAIF** (RL from AI Feedback) and how does it differ from RLHF? What are the scalability advantages and the risks?
    2. What does SWE-bench Verified measure that MMLU or HumanEval does not? Why is it a better proxy for real software engineering ability?
    3. What safety challenges are unique to **computer use** agents that are not present in chat-only deployments?

---

## 8. Gemini 3.1 Pro — February 2026

### What Changed

**Gemini 3.1 Pro** scored 77.1% on ARC-AGI-2 and 94.3% on GPQA Diamond — tying GPT-5.4 Pro on the Intelligence Index at roughly one-third the cost per token. It cemented Google's position at the frontier and demonstrated that **compute efficiency** had become as important a competitive axis as raw benchmark performance.

### Key Technical Details

ARC-AGI-2 tests novel reasoning patterns not seen in training — it is specifically designed to resist memorization. High performance requires genuine compositional generalization:

\[
P(\text{correct} \mid \text{novel pattern}) \neq P(\text{correct} \mid \text{seen pattern})
\]

A model that memorizes training distributions fails ARC-AGI-2 even at very large scale.

??? info "Technical Details"
    - **Cost efficiency**: ~3x cheaper per token than GPT-5.4 Pro at comparable quality — a meaningful production trade-off.
    - **Native multimodal**: text, images, audio, video; continuation of the Gemini native-multimodal approach.
    - **Grounded search**: integrated Google Search grounding produces citations and factual grounding in outputs.

!!! interview "Interview Questions"
    1. Why is **ARC-AGI-2** considered a harder benchmark than MMLU or GPQA? What reasoning failure mode does it specifically target?
    2. How does **cost per token** become a competitive axis at the frontier, and what architectural choices drive inference efficiency at scale?

---

## 9. GPT-5.4 Family — March 2026

### What Changed

The **GPT-5.4 family** (Thinking, Pro, mini, nano) brought **native computer use**, up to **1,050,000 token context**, and a tiered pricing structure down to \$0.20/M tokens for nano. This established the pattern of model families: a flagship for quality, a mini for cost, and a nano for edge/volume use cases — all from the same training run via distillation.

### Key Technical Details

**Native computer use**: the model generates structured action sequences (click, type, scroll, screenshot-then-reason) in a loop — not just text. The action space is formalized:

\[
a_t = \pi_\theta(s_t), \quad s_{t+1} = \text{env}(s_t, a_t)
\]

where \(s_t\) is the current screenshot/DOM state and \(a_t\) is a structured action.

!!! math-intuition "In Plain English"
    The model is now an agent operating in a GUI environment. Each step: observe state (screenshot), reason, emit action (click, type), observe new state. This is a direct MDP/RL framing at inference time.

??? info "Technical Details"
    - **GPT-5.4 standard**: $2.50/$15 per million tokens.
    - **GPT-5.4 Pro**: $30/$180 per million tokens — extended reasoning.
    - **GPT-5.4 mini**: $0.75/$4.50 per million tokens.
    - **GPT-5.4 nano**: $0.20/$1.25 per million tokens — optimized for high-volume, latency-sensitive calls.
    - **1M token context**: full document, codebase, or conversation history in a single prompt.

!!! interview "Interview Questions"
    1. How do you distill a frontier model into mini/nano variants without losing critical capabilities? What training signals are preserved vs. compressed away?
    2. What new **safety challenges** arise from native computer use that don't exist for text-only models?

---

## 10. GLM-4.6 — September 2025

### What Changed

**Zhipu AI's GLM-4.6** is a 357B-parameter MoE model (32B active per forward pass) released open-weight under MIT license in September 2025. It extends context from 128K (GLM-4.5) to **200K tokens** with **128K-token output** support — enabling long-form generation at a scale most models cannot match. The GLM family has deep roots in **GLM (General Language Model) pretraining**, which uses autoregressive blank infilling rather than causal LM or masked LM as the pretraining objective.

### Key Technical Details

**GLM pretraining objective**: the model predicts shuffled spans in a document using bidirectional attention for the context and causal attention for each span:

\[
\mathcal{L}_{\text{GLM}} = -\mathbb{E}\left[\sum_{s \in S} \log P_\theta(s \mid x_{\text{corrupt}}, s_{<i})\right]
\]

where \(S\) is the set of masked spans and \(x_{\text{corrupt}}\) is the text with those spans replaced by mask tokens.

!!! math-intuition "In Plain English"
    GLM unifies the MLM objective (bidirectional context, like BERT) with the autoregressive objective (causal generation, like GPT) in a single pretraining task. The model sees full context when encoding, but generates each masked span left-to-right. This trains both understanding and generation simultaneously.

**Agentic capabilities** in GLM-4.6 are enhanced via RL-trained tool use: the model autonomously plans, invokes tools, and coordinates across tool calls without explicit orchestration code.

??? info "Technical Details"
    - **Architecture**: MoE Transformer, 357B total / ~32B active parameters.
    - **Context**: 200K input, 128K output — one of the highest output-token limits available.
    - **License**: MIT/Apache 2.0 — commercially permissive.
    - **Languages**: strong Chinese and English, 24 additional languages.
    - **GLM-Z1**: companion reasoning model with "deep thinking" mode.
    - **GLM-4-32B**: a dense 32B variant trained on 15T tokens, competitive with much larger models.

!!! interview "Interview Questions"
    1. How does the **GLM pretraining objective** differ from BERT's MLM and GPT's causal LM? What tasks does each objective best prepare the model for?
    2. Why is a **200K output token** limit practically significant, and what types of tasks become possible that were not with a 4K output limit?
    3. How does MoE sparsity (32B active out of 357B total) affect **per-token inference cost** vs. a dense 32B model?

---

## Cross-Cutting Themes (2025–2026)

**Reasoning as a first-class axis**: every frontier lab now has a reasoning model (o3, R1, Gemini Deep Think, GLM-Z1, Claude extended thinking). RL with verifiable rewards is the standard recipe.

**Open-weight MoE at scale**: Llama 4, GLM-4.6, Qwen2.5-Plus — open MoE models are now competitive with closed frontiers. Self-hosting is viable for many production use cases.

**Context window beyond 128K**: 1M (Maverick, Gemini 2.5), 10M (Scout), 200K (GLM-4.6). Cost-per-token matters as much as quality at these lengths.

**Natively multimodal pretraining**: early fusion (Llama 4) is gaining over bolt-on vision encoders. Models learn joint text-image representations from day one.

**Model families via distillation**: GPT-5.4 (Pro/mini/nano), R1 (Distill variants), Scout (distilled from Behemoth). One training run, multiple deployment tiers.

**Cost as a frontier**: Gemini 3.1 at 1/3 the cost of GPT-5.4 Pro at comparable quality — efficiency is now a primary competitive dimension, not just a secondary consideration.

## Further Reading

- *Core Architectures* (MoE, MLA, iRoPE) — technical details on the attention and expert mechanisms underlying these models
- *Training and Alignment* (RLHF, DPO, GRPO) — the RL recipes powering reasoning models
- *Inference and Serving* (KV cache, speculative decoding, quantization) — how these large models run in production
- *Top 25 Papers* — the research lineage: DeepSeekMath → R1, Chinchilla → compute-optimal training, InstructGPT → RLHF

Verify **vendor docs** and **licenses** before production use — this page reflects the state as of April 2026.
