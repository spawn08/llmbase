# DeepSeek-R1 and R1-Zero — January 2025

## What Changed

**DeepSeek-R1-Zero** became the first model trained via large-scale RL **without any supervised fine-tuning** as a prerequisite. Starting from DeepSeek-V3 base weights, RL with verifiable rewards taught the model to self-verify, reflect, and produce chain-of-thought traces—entirely from scratch. The model displayed an emergent **"Aha moment"** during training: a sudden qualitative shift in reasoning depth as RL scaling continued.

**DeepSeek-R1** then added a small cold-start SFT phase followed by the same RL recipe, producing stronger and more stable outputs. Both models matched OpenAI o1-1217 on AIME and MATH-500 benchmarks. Distilled student models (**R1-Distill-Qwen**, **R1-Distill-Llama**) brought reasoning-grade performance to 7B–70B open checkpoints.

## Key Technical Details

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

## Practical Implications

Parse `<redacted_thinking>...</redacted_thinking>` blocks before showing users final answers. Moderate the reasoning trace for PII and IP — the trace contains more information than the final reply. For production: budget reasoning tokens via max-length caps, not just final token caps.

!!! interview "Interview Questions"
    1. Why does GRPO not need a critic/value network, and what does it use instead? How does this differ from PPO?
    2. What is the "Aha moment" in R1-Zero training, and why is it significant for understanding emergent capabilities?
    3. Why are verifiable rewards (unit tests, symbolic solvers) more stable for RL training than human preference labels?
    4. How does distillation from R1 differ from training a student with GRPO directly?
    5. What failure mode does the KL penalty in GRPO prevent, and how does the \(\beta\) coefficient control the trade-off?

## Code Example — vLLM Serving a Reasoning Checkpoint

```bash
export MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --dtype auto \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --enable-chunked-prefill \
  --port 8000
```

Client-side, **parse** `<redacted_thinking>...</redacted_thinking>` or provider-specific **reasoning** blocks before showing user-facing text.
