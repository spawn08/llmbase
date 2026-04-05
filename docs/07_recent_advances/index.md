# Recent Advances in LLM Research

## How to Use This Section

Frontier research as of April 2026: each entry ties a development to **technical** detail and **practice**. Skim **What Changed**, open **Technical Details** for depth, use **Interview Questions** for trade-off drills—not paper titles.

## Last Updated

April 2026

---

## 1. GPT-4o and Multimodal Reasoning

### What Changed

**GPT-4o** (and “omni”-style APIs) unified text, image, and often **audio** in one model: fewer OCR→caption→LLM pipelines, more **joint** reasoning (charts, UI screenshots, object counting). **Interleaved** tokens—patches + text in one sequence—share transformer depth; tool calls and JSON outputs align with **grounded** actions. By 2026 multimodal is the default surface; text-only is a **subset**.

### Key Technical Details

Multimodal LLMs typically concatenate **visual tokens** \(v_1,\ldots,v_{N_v}\) with **text tokens** \(t_1,\ldots,t_{N_t}\) into one sequence. Self-attention then mixes across modalities:

\[
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V
\]

where \(Q,K,V\) are built from **both** image and text positions—so “reasoning” can be **cross-modal** without an explicit retrieval step.

!!! math-intuition "In Plain English"
    The model does not “see then read” as two black boxes—it **attends** from a word to a patch and back in the same layer stack. That is why tasks like “what does the red arrow point to?” can work in one shot if the alignment is learned well.

**Vision tokenization** often uses a ViT-style encoder or a single large CNN/conv stem that maps an image to a grid of tokens; **projection** layers align visual dimensionality to the transformer width \(d_{\text{model}}\).

\[
h^{\text{vis}}_i = W_{\text{proj}} \cdot \mathrm{Enc}(I)_i + b
\]

??? info "Technical Details"
    - **Interleaved sequences**: Chat with multiple images is just longer \(T\); positional encodings must extend to **image slots** (often 2D-aware or treated as extra “time” steps).
    - **Resolution trade-offs**: More patches \(\Rightarrow\) more tokens \(\Rightarrow\) quadratic attention cost in full attention; **tile** or **crop** strategies matter for PDFs and slides.
    - **Audio** (when present): waveforms or mel features are quantized or embedded into the same space; streaming adds **causal** masks for real-time use.
    - **Alignment objectives**: Pretraining may combine **captioning**, **contrastive** losses, and **instruction** fine-tuning so the model follows **natural-language** constraints on pixels.

### Practical Implications

Prefer one multimodal endpoint when layout matters; track **tokens/image** and **resize/tile** for cost. **Safety** shifts to media abuse; **eval** on vision+text tasks, not text-only leaderboards.

!!! interview "Interview Questions"
    1. Why does adding image tokens increase **prefill** cost roughly quadratically with full self-attention, and what mitigations exist at serving time?
    2. How would you design an eval for a **UI assistant** that must ground clicks in screenshots—what failure modes are unique vs text-only agents?
    3. Explain **interleaved** vs **encoder-fusion** multimodal stacks; what breaks if the vision encoder is frozen and only a small projector trains?

### Code Example — OpenAI-Compatible Multimodal Request

Many providers expose **chat completions** with a `content` array mixing `text` and `image_url`:

```python
import os, requests
payload = {
    "model": "gpt-4o",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is unusual about this chart?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/metrics.png", "detail": "high"}},
        ],
    }],
    "max_tokens": 512,
    "temperature": 0.2,
}
r = requests.post(
    "https://api.openai.com/v1/chat/completions",
    json=payload,
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    timeout=120,
)
print(r.json()["choices"][0]["message"]["content"])
```

Tune **`detail`** (or equivalent) to balance **token count** vs **readability** of small text in images.

---

## 2. Claude 3/4 Architecture Insights

### What Changed

**Claude 3/4**-era models stress **long-context** stability, **steerable** system prompts, **constitutional** / critique-style alignment, and **production-grade** tools. Public layer diagrams are scarce; **post-training** (RLAIF, preferences) and **KV/context** engineering explain the behavioral signature—strong coding, careful refusals, long documents. Enterprise positioning favors **auditability** and **less sycophancy** (“helpful without overclaiming”).

### Key Technical Details

From first principles, a **decoder-only** stack with **RoPE** or similar positional encoding defines next-token distribution:

\[
P(x_t \mid x_{<t}) = \mathrm{softmax}(W_o \, h_t)
\]

**Alignment** layers adjust the policy \(\pi_\theta\) toward human/AI preferences using reward models or implicit preferences—conceptually:

\[
\max_\theta \; \mathbb{E}\bigl[r_\phi(x,y) - \beta \log \pi_\theta(y \mid x) / \pi_{\text{ref}}(y \mid x)\bigr]
\]

!!! math-intuition "In Plain English"
    After base training gives you a fluent LM, **RLHF/RLAIF** nudges outputs toward **higher reward** answers while a **KL penalty** stops the model from collapsing into weird hacks that fool the reward model.

??? info "Technical Details"
    - **Mixture-of-Experts** (in larger variants): sparsely activated FFN layers keep **per-token compute** lower than dense width suggests; routing is learned and can be **load-balanced** with auxiliary losses.
    - **Long context**: Attention **kernels** (FlashAttention-class), **KV cache** layout, and possibly **hybrid** attention (local + global) are standard; exact public layer maps are proprietary.
    - **Constitutional / critique-based** training: models compare responses to **principles**, not only scalar rewards—improving **oversight** scalability.
    - **Artifacts & tools**: Post-training emphasizes **JSON**, **function calling**, and **multi-step** workflows with **error recovery**.

### Practical Implications

Strong **system** prompts (tone, uncertainty, citations); pair tools with **clear** success tests; **chunk** long logs even when context fits. Plan **refusal** UX; abstract vendor-only features (**artifacts**, memory) behind your own interfaces.

!!! interview "Interview Questions"
    1. How does **KL-regularized** RLHF differ from pure reward maximization, and what failure mode does it mitigate?
    2. Why might **MoE** improve **throughput** per dollar even if peak quality per parameter differs from dense models?
    3. What is **sycophancy** in RLHF-trained models, and what training or eval mitigations would you propose?

### Configuration Example — System Prompt Skeleton

```yaml
# claude_agent_config.yaml — illustrative pattern for tool-using agents
system_prompt: |
  You are a senior backend engineer. Rules:
  - Never invent library APIs; if unsure, say so and propose how to verify.
  - Prefer small, testable edits; show diffs as unified patches.
  - Use tools: run_terminal_cmd only after restating the exact command.
tools:
  - name: read_file
    description: Read a UTF-8 text file from the workspace.
  - name: run_terminal_cmd
    description: Run a whitelisted command; no network by default.
model:
  name: claude-sonnet-4-20250514
  temperature: 0.2
  max_tokens: 8192
```

---

## 3. DeepSeek R1 and Reasoning Models

### What Changed

**DeepSeek-R1** showed **CoT-style** reasoning can be **trained in** (not only prompted), with strong **math/code** and inspectable traces—blurring model vs **test-time search**. **Open** weights and **R1-Distill** students brought step-by-step reasoning **on-prem**; by 2026 traces are normal in STEM, coding copilots, and **verifier**-scored pipelines. Teams increasingly **parse** and **store** traces for analytics, not only final answers—raising **PII** and **IP** handling requirements in logs.

### Key Technical Details

**GRPO** (Group Relative Policy Optimization) compares **multiple** sampled completions per prompt and applies **clipped** policy gradients using **group-relative** advantages—often **without** a full **critic** network. Symbolically, think PPO-style \(\min(r_t \hat A_t, \mathrm{clip}(r_t)\hat A_t)\) averaged over tokens and samples, with \(r_{i,t}=\pi_\theta/\pi_{\text{old}}\).

!!! math-intuition "In Plain English"
    Draw **several** answers, score them, **upweight** trajectories that beat their peers—PPO-like, but **baselines** come from the **group** instead of a value net.

**Cold-start** CoT data plus **RL** on **verifiable** rewards (math, unit tests) anchors training in **objective** signal.

??? info "Technical Details"
    - **Reward design**: Mix **rule-based** (unit tests, symbolic checks) with **model-based** grading where ground truth is fuzzy.
    - **Overthinking**: Long chains cost **latency** and can **hallucinate** confidently—**length penalties** and **stop** criteria matter.
    - **Distillation**: Train smaller students on **teacher traces**; watch for **distribution shift** when the teacher uses tools the student lacks.

### Practical Implications

Show traces when **audit** matters; hide for **latency**. **Eval** process quality, not only final answers. **Safety**: moderate **chains**, not only final text.

!!! interview "Interview Questions"
    1. Why are **verifiable rewards** (tests, symbolic solvers) more stable than general **human preference** for RL on reasoning models?
    2. Compare **GRPO**-style group baselines to a learned **critic** in PPO—trade-offs in stability and compute?
    3. How does **distillation from a reasoning teacher** differ from standard SFT on raw answers?

### Code Example — vLLM Serving a Reasoning Checkpoint

```bash
# Serve an open reasoning-style model with vLLM (names illustrative)
export MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --port 8000
```

Client-side, **parse** `<think>...</think>` or provider-specific **reasoning** blocks before showing user-facing text.

---

## 4. Phi-3 / Phi-4 — Small Models, Large Knowledge

### What Changed

**Phi-3/Phi-4** show **data quality** + **curriculum** can make **sub-10B** models **frontier-adjacent** on reasoning/code/chat—**synthetic** and **filtered** pretraining plus strong **SFT/alignment** compress capability into phones, laptops, and **cheap** GPUs. The **scaling-law** story shifts toward **better tokens** when parameter budgets are fixed. Product teams use Phi-class models as **default** baselines before jumping to 70B+ APIs, especially when **latency** and **cost** caps dominate.

### Key Technical Details

**Loss** on a token-level CE objective remains standard:

\[
\mathcal{L} = -\sum_{t} \log P_\theta(x_t \mid x_{<t})
\]

**Data filtering** maximizes **information per token**—reducing redundancy and toxic/low-utility text.

!!! math-intuition "In Plain English"
    Small models hit **capacity limits** fast if you feed them **sloppy** web dumps; **curating** “textbook-like” data teaches **patterns** with fewer parameters.

**Scaling laws** relate loss to parameters \(N\), data \(D\), and compute \(C\):

\[
L \approx A N^{-\alpha} + B D^{-\beta} + \text{irreducible}
\]

!!! math-intuition "In Plain English"
    For a fixed **tiny** \(N\), you still gain from **better** \(D\) and training—until you hit the **floor** where the model simply cannot memorize or generalize further.

??? info "Technical Details"
    - **Architecture**: Often **dense** decoder-only transformers; **GQA** for KV cache efficiency.
    - **Tokenizer**: Efficient tokenization reduces **effective** sequence length for the same text.
    - **Alignment**: Heavy **SFT** + preference tuning; **safety** layers tuned for consumer deployment.

### Practical Implications

Great for **quantized** edge; pair with **strong RAG**/prompting. Batch small jobs locally to save **API** cost.

!!! interview "Interview Questions"
    1. Why does **data quality** disproportionately help **small** models vs large ones?
    2. How would you decide between **Phi-class** local inference vs **frontier** API for a coding assistant?
    3. What **eval** would you run to detect **overfitting** to synthetic textbook style?

### Code Example — Transformers + 4-bit Loading (Illustrative)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "microsoft/Phi-3-mini-4k-instruct"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto",
)
prompt = "<|user|>\nExplain KV cache in two sentences.\n<|assistant|>\n"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 5. Llama 3.1 / 3.2 and Open-Weight Scaling

### What Changed

**Llama 3.1** pushed **open** weights toward frontier quality (**405B** tier, **128K** context, multilingual). **Llama 3.2** adds **efficient** and **VLM** variants for **edge**. Ecosystem default: **LoRA**, tools, **GGUF**/**vLLM**/**MLX** target Llama checkpoints first.

### Key Technical Details

**Grouped-Query Attention (GQA)** shares \(K,V\) heads across query groups to shrink KV cache:

\[
\text{KV pairs per layer: } O\left(\frac{H}{G} \cdot T \cdot d_h\right)
\]

for \(G\) query groups vs \(H\) Q heads.

!!! math-intuition "In Plain English"
    You **duplicate queries** for expressiveness but **reuse** keys/values—cutting memory bandwidth and cache size with modest quality impact when designed well.

**RoPE** scaling (e.g., **YaRN**) adjusts frequencies for **long** evaluation:

\[
f'_i = f_i \cdot s_i
\]

where scaling \(s_i\) may vary by band to preserve **local** vs **global** structure.

??? info "Technical Details"
    - **405B**: Requires **multi-GPU** tensor/pipeline parallelism for interactive serving; **8-bit** or **FP8** weights common.
    - **Tool calling**: Chat templates (`special tokens`) standardize **assistant/user/tool** turns—critical for **agent** frameworks.
    - **License**: **Community** license with usage rules—**compliance** review before production.

### Practical Implications

Weights **port** across stacks; **LoRA** most domains; **full FT** only with eval gates. **No recall** for leaks—**app-layer** safety required.

!!! interview "Interview Questions"
    1. Quantify how **GQA** changes **KV cache** size vs **MHA** at fixed \(H\) and \(T\).
    2. What breaks if you **double** context length without **positional** scaling—where do you see failures first?
    3. Compare **open-weight** risk management to **API-only** models for regulated industries.

### Code Example — llama.cpp Server (GGUF)

```bash
# Build llama.cpp with CUDA/Metal as appropriate, then:
./llama-server \
  -m ./models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
  -c 131072 \
  --parallel 4 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 8080
```

Tune **`-c`** (context) to the **minimum** your app needs to preserve **speed** and **RAM**.

---

## 6. Mixture-of-Depths

### What Changed

**MoD** routes **per-token depth**: a light **router** skips or runs full layers—**dynamic FLOPs** in the forward pass, complementary to **MoE** (which sparsifies **width**). Goal: **full depth** only where hard; 2025–2026 stacks explore MoD for **serving** efficiency and tail latency control. Unlike **early exit** at the **final** layer only, MoD can allocate depth **throughout** the stack—closer to **conditional computation** in classical nets. Training must avoid **router collapse** (always skip or always run), usually via **auxiliary** losses.

### Key Technical Details

Let \(x_t^{(\ell)}\) be the hidden state at layer \(\ell\). A **binary** or **soft** gate \(g_t^{(\ell)} \in [0,1]\) mixes **transformed** and **identity/skip** paths:

\[
x_t^{(\ell+1)} = g_t^{(\ell)} \cdot F^{(\ell)}(x_t^{(\ell)}) + (1 - g_t^{(\ell)}) \cdot x_t^{(\ell)}
\]

!!! math-intuition "In Plain English"
    If the model is already **sure** what comes next, **skip** the heavy block; if not, **run** the full layer. Think **conditional** depth per token.

**Load balancing** losses may encourage **non-degenerate** routing (avoid always-skip or always-run collapse).

??? info "Technical Details"
    - **Implementation**: Skipping must interact cleanly with **KV cache** for autoregressive decoding—**cache** alignment for skipped layers is a systems detail vendors handle carefully.
    - **Training**: Auxiliary losses stabilize **router**; **straight-through** or **Gumbel-Softmax** tricks appear in some formulations.
    - **MoE + MoD**: In principle **composable**—**depth** and **expert** sparsity target different axes.

### Practical Implications

Watch **p99** under **uneven** routing; **dynamic** graphs may stress kernels. Report **quality per FLOP**, not only wall-clock.

!!! interview "Interview Questions"
    1. Contrast **MoD** with **early exiting**: where does each apply in the stack?
    2. How could **routing collapse** harm training, and what loss terms mitigate it?
    3. Why is MoD **trickier** for **decode** than **prefill** in transformer serving?

### Pseudocode — Gated Residual (Conceptual)

```python
def mod_block(x, layer_fn, router_fn):
    # x: [batch, seq, dim]
    g = torch.sigmoid(router_fn(x))  # [batch, seq, 1]
    y = layer_fn(x)
    return g * y + (1.0 - g) * x
```

Production code must fuse with **attention** + **KV** update rules; treat this as **illustrative** math, not a drop-in kernel.

---

## 7. Test-Time Compute Scaling (o1 / o3 Paradigm)

### What Changed

**o1/o3**-style APIs sell **test-time compute**: extra **thinking** tokens, **tool** loops, **sampling**, **verifiers**—quality from **inference** spend, not only bigger pretrain. UX: hidden/summarized chain + final answer with **budgets**. **Agentic** code execution is the default for **max reliability** when latency allows.

### Key Technical Details

A simple **best-of-N** estimator samples \(N\) answers and selects via verifier score \(V\):

\[
\hat{y} = \arg\max_{y \in \{y_1,\ldots,y_N\}} V(x,y)
\]

!!! math-intuition "In Plain English"
    If **any** sample crosses the correctness bar, you win more often—at linear **cost** in \(N\) without retraining.

**Process supervision** assigns credit to **steps**:

\[
R = \sum_{k} r_k \quad \text{for steps } k
\]

!!! math-intuition "In Plain English"
    Grade the **route**, not only the **destination**—critical when final answers are **small** but reasoning is **long**.

??? info "Technical Details"
    - **Compute allocators** decide when to **stop thinking**—MDP / bandit views apply.
    - **Consistency checks**: **Self-consistency** (majority vote over chains) trades cost for robustness.
    - **Tool-augmented** search: **MCTS**-like exploration appears in research systems; product stacks combine **retrieval** + **Python** sandboxes.

### Practical Implications

Expose **effort**/budget knobs; log **steps** (PII-safe). **Moderate** intermediate states—long chains can surface harm before the final reply.

!!! interview "Interview Questions"
    1. Map **test-time scaling** to a **compute-optimal** frontier—when does doubling **N** stop helping?
    2. How do you **evaluate** hidden chains without leaking **test set** answers into prompts?
    3. Compare **verifier-based** selection to **single-sample** long-CoT at equal latency budget.

### Configuration — “Reasoning Effort” API Pattern

```json
{
  "model": "o3-mini",
  "reasoning": { "effort": "high", "max_reasoning_tokens": 32000, "summary": "auto" },
  "tools": [{ "type": "function", "function": { "name": "python_exec" } }]
}
```

Fields vary by provider—wrap in your **SDK**.

---

## 8. Long-Context Retrieval Without RAG

### What Changed

**128K–1M+** contexts let teams **stuff** whole repos/PDFs into **one** prompt—less **classical** chunk+RAG glue for some tasks. You still **pay** **prefill** and **I/O**; the shift is **vector retrieval → load-into-prompt**. **Lost-in-the-middle** and **distraction** remain without careful **curation**.

### Key Technical Details

Full attention **cost** scales as \(O(T^2)\) per layer; **FlashAttention** reduces **memory** traffic but not **asymptotic** FLOPs:

\[
\text{Attention FLOPs} \propto L \cdot H \cdot T^2 \cdot d_h
\]

!!! math-intuition "In Plain English"
    **Long** prompts are **compute-heavy** even if they fit in RAM—**prefill** dominates before token-one is emitted.

**Information-theoretic** view: stuffing \(T\) tokens does not guarantee **attention** to all positions—**entropy** of the attention distribution matters.

??? info "Technical Details"
    - **YaRN / NTK** scaling extends RoPE-trained models beyond train length.
    - **Hybrid** models: **local** windows + **global** tokens for **register** effects.
    - **Distraction**: Irrelevant tokens **compete** for attention mass—**curation** still helps.

### Practical Implications

**Stuffing** wins for single corpora and tight **privacy** (no index). **RAG** still wins at **TB** scale, **freshness**, and **stable citations**. Use **prompt caching** for static prefixes to cut **prefill** cost.

!!! interview "Interview Questions"
    1. Compare **paged attention** benefits for **long prefill** vs **decode**—what tensor shapes dominate each phase?
    2. Describe an experiment to measure **lost-in-the-middle** for **your** documents—what controls would you use?
    3. Why might **RAG + long context** compose better than either alone for **enterprise** search?

### Code Example — Prompt Caching Hint (OpenAI-Style)

```python
# Pseudocode: stable prefix hashed & billed at cache rate where supported
messages = [
    {"role": "system", "content": HUGE_POLICY_MANUAL},
    {"role": "user", "content": "Summarize section 7.3 only."},
]
# Client SDK may expose cache_control on the static system block
```

Measure **cached vs uncached** token pricing for static prefixes.

---

## 9. Continuous Learning and Online Fine-Tuning

### What Changed

**Continual** / **online** FT adapts to **drift** without full pretrain—**LoRA** rotations, streaming **DPO**, **canary** evals—with defenses against **forgetting** and **poisoning**. MLOps is **data lineage** + **GPU** ops. **Regulatory** pressure (retention limits, **right to erasure**) collides with “**always learning**”—you need **retention policies** and sometimes **adapter rollback**. **Shadow** deployments catch regressions before full traffic moves to a new adapter.

### Key Technical Details

**Elastic Weight Consolidation (EWC)**-style regularization penalizes movement in **important** directions for past tasks:

\[
\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{new}} + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_i^\star)^2
\]

where \(F_i\) approximates **Fisher** information for parameter \(i\).

!!! math-intuition "In Plain English"
    **Protect** weights that mattered for old tasks while learning new ones—**trade** plasticity vs stability.

**Experience replay** mixes **old** samples into **new** batches to mitigate forgetting:

\[
\mathcal{L} = \mathbb{E}_{(x,y)\sim \mathcal{D}_{\text{new}} \cup \mathcal{D}_{\text{replay}}}[\ell(f_\theta(x),y)]
\]

??? info "Technical Details"
    - **LoRA**: Low-rank \(\Delta W = BA\) keeps **base** frozen; swapping adapters enables **tenant** isolation.
    - **RLHF drift**: Online preference updates can **overoptimize** quirks—**KL** to reference remains important.
    - **Security**: **Data poisoning** in user logs → **backdoors**; **human** review for high-risk corpora.

### Practical Implications

Version **datasets** and **adapters**; **canary** before promote. **GDPR**/forget conflicts need **unlearn** or **excludes**. Monitor **cohorts**—regressions often hit **minority** styles first.

!!! interview "Interview Questions"
    1. What is **catastrophic forgetting**, and how do **replay** and **EWC** differ in mechanism?
    2. How would you detect **preference hacking** in an online **DPO** loop?
    3. Why might **full fine-tuning** on streams be riskier than **LoRA** for production LLMs?

### Code Example — PEFT LoRA Training Snippet

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
model = get_peft_model(base, lora)  # train adapters only
```

---

## 10. LLM Inference on Edge Devices

### What Changed

**NPUs** and **SoCs** run **3B–8B** models **locally** (**GGUF**, **ONNX**, **Core ML**, **ExecuTorch**) with **quant** + **KV** reuse. **Privacy**/**cost** vs **quality**, **context**, and **thermal** limits. **Hybrid** apps run **small** models on-device for **PII-sensitive** steps and **escalate** only anonymized subtasks to the cloud. Battery-aware UX **throttles** max tokens/sec when thermals spike.

### Key Technical Details

**Quantized** matmuls approximate weights \(W\) with \(\hat{W}\):

\[
y = \hat{W} x,\quad \hat{W} = \mathrm{quantize}(W)
\]

**INT4** / **INT8** schemes pair with **per-channel** scales \(s\) and zero-points \(z\):

\[
\hat{W}_{ij} = s_j (q_{ij} - z_j)
\]

!!! math-intuition "In Plain English"
    **Fewer bits** \(\Rightarrow\) smaller memory and higher **ops/s**—but **error** accumulates across layers; **calibration** on representative activations reduces outliers.

**Memory bandwidth** often bounds **decode**:

\[
\text{Tokens/sec} \approx \frac{\text{BW}}{\text{bytes moved per token}}
\]

??? info "Technical Details"
    - **KV cache quantization**: 8-bit KV reduces **memory** for long sessions.
    - **Speculative decoding**: Small **draft** model proposes tokens; large model **verifies**—harder on edge but emerging.
    - **Power**: **Sustained** vs **burst** clocks—**thermal** throttling changes **SLA**.

### Practical Implications

Rough guide: **~3B** phones, **7–8B** desktop **UM**; stream tokens, **fallback** to cloud if uncertain. **Eval** downstream tasks under **quant**, not PPL alone.

!!! interview "Interview Questions"
    1. Why is **decode** often **memory-bandwidth** bound while **prefill** is **compute** bound?
    2. Compare **GGUF** inference vs **ONNX** + vendor EP for a mobile deployment.
    3. How does **INT4** quantization interact with **outlier** dimensions in activations?

### Code Example — llama.cpp CLI on Apple Silicon

```bash
# Metal backend example — paths and model vary
./llama-cli \
  -m ./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  -p "List three bullet points about edge LLM trade-offs." \
  -n 256 \
  -ngl 99 \
  --ctx-size 8192 \
  --threads 8 \
  --temp 0.7
```

**`-ngl`** offloads **layers** to GPU/NPU backends where supported—profile **tokens/sec** vs **power** draw for **your** device class.

---

## Cross-Cutting Themes (April 2026)

**Multimodal** defaults; **test-time** spend buys reliability; **open weights** imply **your** safety/compliance; **MoD/GQA/quant/small-data** stack; **long context vs RAG** is **cost+eval**, not religion.

## Further Reading

*Inference and Serving* (KV, batching, quant); *Training and Alignment* (RLHF/DPO); *Advanced* (long context, RAG). Verify **vendor** docs and **licenses**—this page is a **snapshot**.
