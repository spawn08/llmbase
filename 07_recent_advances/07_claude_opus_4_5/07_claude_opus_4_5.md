# Claude Opus 4.5 — November 2025

## What Changed

**Anthropic's Claude Opus 4.5** achieved state-of-the-art on software engineering benchmarks (SWE-bench Verified), with particular strength in multi-step agentic coding tasks. Designed for **computer use** — controlling browsers and GUIs autonomously — and long-running agents that must maintain coherent plans across many tool calls.

## Key Technical Details

Constitutional AI at scale: Opus 4.5 continues Anthropic's RLAIF approach where model-generated critiques guided by written principles provide training signal without large-scale human labeling. The key advance: **critique models** specialized per domain (code review, safety, factual grounding) rather than a single general critic.

\[
\pi_{\theta}^{*} = \arg\max_\theta \mathbb{E}_{x \sim \pi_\theta}\bigl[r_\phi^{\text{domain}}(x) - \beta \log\frac{\pi_\theta(x)}{\pi_{\text{ref}}(x)}\bigr]
\]

!!! math-intuition "In Plain English"
    Each domain critic (code, factual, safety) provides its own reward signal. The policy maximizes a blend of domain-specific rewards while staying close to the reference model — so improvements are steered by **AI-generated feedback** aligned to principles, not only by human preference labels.

??? info "Technical Details"
    - **SWE-bench Verified**: Anthropic claims SotA; measures a model's ability to resolve GitHub issues end-to-end in a sandboxed environment.
    - **Computer use API**: standardized interface for GUI actions (click, type, screenshot) — enables "agent as employee" workflows.
    - **Pricing**: $5/$25 per million input/output tokens.
    - **Context**: 200K token context window.

## Practical Implications

For **agentic coding**, evaluate on task-level benchmarks (issue → patch → tests), not just HumanEval-style single-function completion. For **computer use**, run actions inside **sandboxed VMs** with network and filesystem policies; log every action for audit. For **RLAIF**, monitor reward hacking (critics agreeing with each other without grounding) and refresh principle sets as product risks evolve.

!!! interview "Interview Questions"
    1. What is **RLAIF** (RL from AI Feedback) and how does it differ from RLHF? What are the scalability advantages and the risks?
    2. What does SWE-bench Verified measure that MMLU or HumanEval does not? Why is it a better proxy for real software engineering ability?
    3. What safety challenges are unique to **computer use** agents that are not present in chat-only deployments?

## Code Example

Conceptual **computer use** loop (API shapes vary; consult Anthropic docs for exact schemas):

```python
# Pseudocode: observe → act → observe until task done or step limit
state = env.reset()  # e.g. initial screenshot + accessibility tree
for step in range(max_steps):
    action = client.computer_use(model="claude-opus-4-5", state=state, goal=user_goal)
    state = env.step(action)  # click, type, scroll, wait, etc.
    if state.task_complete:
        break
```
