# Kimi k1.5 — January 2025

## What Changed

**Moonshot AI's Kimi k1.5** demonstrated that long-context scaling — extending the policy's context window to 128K tokens — is itself a key axis for improving reasoning quality. Without MCTS, value functions, or process reward models, Kimi k1.5 matched OpenAI o1 on AIME (77.5) and MATH 500 (96.2) using a simpler RL framework.

The insight: a long context window lets the model **plan, reflect, and self-correct** within a single forward trajectory, because earlier reasoning steps remain in-context and can be referenced and revised.

## Key Technical Details

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

## Practical Implications

Long-context RL is now a viable alternative to search-augmented inference. When deploying reasoning models, measure cost at the reasoning-token level, not just final output tokens — the 128K traces are the expensive part.

!!! interview "Interview Questions"
    1. Why does extending a model's context window improve its reasoning ability in RL training?
    2. How does online mirror descent stabilize policy updates compared to vanilla gradient ascent on reward?
    3. Kimi k1.5 achieves o1-level results without MCTS — what does this tell us about the relationship between search and in-context reasoning?

## Code Example

*(No standalone code sample in the source entry; partial rollout reuse and online mirror descent are primarily training-stack concerns.)*
