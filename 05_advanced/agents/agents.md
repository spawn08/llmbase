# Agents and Tool Use

## Why This Matters for LLMs

A standalone language model is a **next-token predictor**. **Agents** wrap that predictor in a **control loop** that can **query tools, call APIs, execute code, and mutate state** in external systems. This is the architectural shift behind coding assistants, enterprise copilots with CRM connectors, and “GPT with plugins.” Interviews consistently test whether you can separate **policy** (what to do next) from **tools** (how the environment responds) and whether you understand **failure modes**: wrong tool selection, malformed JSON arguments, infinite loops, and **prompt injection** via untrusted tool outputs.

Second, tool use **relocates knowledge** from model weights into **ground-truth channels**: flight status from an airline API, account balance from a ledger, file contents from disk. The failure signature changes from “fluent hallucination” to **structured errors**—missing parameters, 404s, permission denied—that you must surface to users and log for observability. Reasoning traces (**ReAct**: thought → action → observation) give debugging hooks that pure chain-of-thought lacks when there is no external feedback.

Third, **multi-agent** and **hierarchical** patterns (router, planner–worker, critic–actor) decompose complex tasks but add **coordination cost** and **safety surface area**—agents granting permissions to one another, duplicate tool calls, or contradictory observations. Memory tiers (context window vs vector store vs structured DB), **max-step** cutoffs, and **human-in-the-loop** approvals for irreversible actions are production requirements, not optional polish.

---

## Core Concepts

### What Is an LLM Agent?

An agent augments an LM with:

1. A **tool registry** (name, description, JSON Schema parameters).
2. A **runtime** that validates calls, executes tools, and appends **observations** to the transcript.
3. A **termination rule** (explicit “finish” action, answer delimiter, or step budget).

Abstractly, at step \(t\) the policy produces an action \(a_t\) given history \(h_t\); the environment returns \(o_t\):

\[
a_t \sim \pi_\theta(a_t \mid h_t),\quad o_t = \mathrm{Env}(a_t)
\]

and \(h_{t+1} = h_t \oplus (a_t, o_t)\).

!!! math-intuition "In Plain English"
    The LM is only choosing **discrete or JSON-structured** actions from an **allowlist** you exposed. “Environment” can be HTTP, SQL, Python, or a mock—what matters is **grounding**: the next thought is conditioned on **real** \(o_t\), not imagined facts.

### ReAct: Reasoning + Acting

**ReAct** interleaves natural-language **thought**, **action**, and **observation**. A trace might look like: *Thought: I need the population.* → *Action: search("France population 2024")* → *Observation: 68.4 million (estimate).* → *Thought: I can answer.*

Compared to chain-of-thought **without** tools, ReAct reduces **ungrounded** reasoning when tools are faithful.

!!! example "Worked Example: ReAct Trace (Abbreviated)"
    **User:** “What is the capital of Germany?”  
    **Thought:** “I should verify with the geography tool rather than guessing.”  
    **Action:** `lookup_capital(country="Germany")`  
    **Observation:** `{"capital": "Berlin", "source": "static_kv"}`  
    **Thought:** “Observation is authoritative; respond.”  
    **Action:** `finish("Berlin")`  
    **System:** Return **Berlin** to the user.  
    If the observation had been an error JSON, the next thought should **repair** arguments or try a fallback tool—not hallucinate **Munich**.

### Tool Calling and JSON Schema

Tools are declared with **JSON Schema**-compatible metadata. The model emits a structured call:

\[
\text{call} = (\text{name}, \text{args}),\quad \text{args} \in \mathbb{R}^{d_{\text{json}}}
\]

Validation uses the schema **before** execution: required fields, types, enums.

!!! math-intuition "In Plain English"
    Structured args are **not** free-form prose—they must parse with `json.loads` and pass **pydantic** (or equivalent) checks. This is your **firewall** against creative but unusable “tool names.”

### Softmax over Tool Choices (Conceptual)

Some implementations score each tool \(t \in \{1,\ldots,T\}\) with logits \(z_t\) and sample or take argmax:

\[
P(\text{tool}=t) = \frac{\exp(z_t / \tau)}{\sum_{j=1}^{T} \exp(z_j / \tau)}
\]

Temperature \(\tau\) controls exploration in **policy** heads or distilled routers.

!!! math-intuition "In Plain English"
    Lower \(\tau\): almost **deterministic** tool choice—good for **compliance** workflows. Higher \(\tau\): more randomness—risky for production unless you want exploratory agents.

### Planning: Chain-of-Thought vs Tree-of-Thought

**Chain-of-thought** expands a **single** reasoning trace. **Tree-of-thought** maintains **multiple** partial plans \(\mathcal{B}\) and selects:

\[
b^\star = \arg\max_{b \in \mathcal{B}} V(h \parallel b)
\]

where \(V\) is a value estimate (learned model, heuristic, or LLM self-critique).

!!! math-intuition "In Plain English"
    CoT is one draft; ToT is **try several drafts**, score them, commit. Cost scales with **branching factor**—use when puzzles justify it, not for every API call.

### Memory Types

| Memory | Mechanism | Role |
|--------|-----------|------|
| Short-term | Conversation in context | Coreference, recent tool I/O |
| Long-term | Vector DB / KV / SQL | User prefs, org facts across sessions |
| Episodic | Summaries of past runs | “Last time you asked for CSV export” |
| Working | Scratchpad / variables | Intermediate numbers in code agents |

Long-term retrieval often follows **embed query → top-k → merge** into the system message.

### Multi-Agent Patterns

- **Orchestrator**: delegates subtasks to specialists.
- **Debate**: agents propose conflicting plans; aggregator picks or synthesizes.
- **Critic–actor**: critic blocks unsafe tool args.

Message passing may be **shared transcript** (synchronous) or **event bus** (async).

### Bellman Optimality (Pointer)

For MDPs with reward \(r_t\), the **value** obeys:

\[
V^\pi(s) = \mathbb{E}_\pi \Bigl[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \Bigr]
\]

RL fine-tuning of tool policies can shape intermediate rewards (correct tool) and terminal rewards (task success).

!!! math-intuition "In Plain English"
    Most shipped agents use **supervised** tool traces or **rejection sampling** before full RL—but the Bellman view explains **why** dense rewards on tool correctness help credit assignment.

### Contrast with RAG

| Pattern | Strength | Weakness |
|---------|----------|----------|
| RAG | Grounded facts from corpus | Limited **actions** |
| Agent | Side effects & multi-step | Loops, bad args, injection |
| Hybrid | Retrieve then act | Orchestration complexity |

??? deep-dive "Deep Dive: Partially Observable MDPs"
    Tool outputs are **observations**, not full world state—the LM’s belief is **implicit** in the truncated transcript. **Summarization** compresses history but can drop **constraints**; for long tasks, persist **structured state** (JSON) outside the model.

??? deep-dive "Deep Dive: Prompt Injection via Tools"
    A malicious webpage fetched by `browse()` can contain “ignore previous instructions.” Mitigations: **tool output sandboxing**, **privilege separation**, **static system prompt priority**, and **downstream content filters** on combined text.

## Code

### ReAct-style loop with tool registry (no network required)

```python
"""
Self-contained ReAct-style agent: regex-parse actions, deterministic mock LLM.
Extend call_llm with your provider; keep tool execution server-side validated.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


ToolFn = Callable[..., str]


@dataclass
class ToolSpec:
    name: str
    description: str
    json_schema: Dict[str, Any]
    fn: ToolFn


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON object parse from model output."""
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class ReactAgent:
    tools: Dict[str, ToolSpec]
    max_steps: int = 8
    history: List[str] = field(default_factory=list)

    def run(self, user_goal: str, call_llm: Callable[[str], str]) -> str:
        self.history = [f"User: {user_goal}"]
        for _ in range(self.max_steps):
            prompt = self._format_prompt()
            reply = call_llm(prompt)
            self.history.append(f"Assistant raw:\n{reply}")

            jcall = _extract_json_object(reply)
            if jcall and "tool" in jcall:
                name = str(jcall["tool"])
                args = jcall.get("arguments", {})
                if name == "finish":
                    return str(args.get("answer", ""))
                if name not in self.tools:
                    obs = json.dumps({"error": f"unknown tool {name}"})
                else:
                    spec = self.tools[name]
                    try:
                        out = spec.fn(**args)
                        obs = out if isinstance(out, str) else json.dumps(out)
                    except TypeError as e:
                        obs = json.dumps({"error": f"bad args: {e}"})
                self.history.append(f"Observation: {obs}")
                continue

            # Fallback: plain text answer if model stops calling tools
            if "FINAL:" in reply:
                return reply.split("FINAL:", 1)[1].strip()
            return reply.strip()
        return "Stopped: max steps reached."

    def _format_prompt(self) -> str:
        tools_desc = "\n".join(
            f"- {t.name}: {t.description} schema={json.dumps(t.json_schema)}"
            for t in self.tools.values()
        )
        header = (
            "You are a tool-using assistant. Reply with a JSON object ONLY:\n"
            '{"tool": "<name>", "arguments": {...}} or {"tool":"finish", "arguments":{"answer":"..."}}\n'
            f"Tools:\n{tools_desc}\n"
        )
        return header + "\n".join(self.history)


def tool_get_capital(country: str) -> str:
    data = {"france": "Paris", "germany": "Berlin", "japan": "Tokyo"}
    return json.dumps({"capital": data.get(country.lower().strip(), "unknown")})


TOOLS: Dict[str, ToolSpec] = {
    "get_capital": ToolSpec(
        name="get_capital",
        description="Return capital city for a country name.",
        json_schema={
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
        },
        fn=tool_get_capital,
    ),
}


def scripted_llm(prompt: str) -> str:
    """Deterministic mock: first turn calls tool; second turn finishes."""
    if "Observation:" not in prompt:
        return '{"tool": "get_capital", "arguments": {"country": "Germany"}}'
    if '"capital": "Berlin"' in prompt:
        return '{"tool": "finish", "arguments": {"answer": "Berlin"}}'
    return '{"tool": "finish", "arguments": {"answer": "Could not resolve."}}'


if __name__ == "__main__":
    agent = ReactAgent(tools=TOOLS)
    print(agent.run("What is the capital of Germany?", scripted_llm))
```

### OpenAI-compatible tool loop (requires `OPENAI_API_KEY`)

```python
"""Optional: real API tool loop. pip install openai"""
import json
import os
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[misc, assignment]


def run_openai_tool_example() -> None:
    if OpenAI is None:
        print("pip install openai to run run_openai_tool_example()")
        return
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run live example.")
        return
    client = OpenAI(api_key=api_key)
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "What is 19 + 23? Use the tool."},
    ]

    def add(a: float, b: float) -> str:
        return json.dumps({"result": a + b})

    for _ in range(4):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append(msg)
            for call in msg.tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments)
                if name == "add":
                    out = add(args["a"], args["b"])
                else:
                    out = json.dumps({"error": "unknown"})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": out,
                    }
                )
        else:
            print(msg.content)
            return
    print("max turns")


if __name__ == "__main__" and os.environ.get("RUN_OPENAI_EXAMPLE"):
    run_openai_tool_example()
```

!!! interview "FAANG-Level Questions"
    1. Describe the ReAct loop and how observations change the next policy input.
    *Answer:* ReAct alternates **thought → action → observation**: the policy conditions on the full transcript so far, picks a tool call, the runtime executes it and appends **observation** \(o_t\) (API result, error, file contents). The next forward pass sees \(h_{t+1} = h_t \oplus (a_t, o_t)\), so the model’s input distribution shifts from **speculation** to **grounded** facts—unlike CoT alone, where the next step only sees prior text. Stops when a finish action fires or budgets exhaust.
    2. How do you prevent tool argument injection and unsafe tool outputs from reaching the user?
    *Answer:* Treat tool outputs as **untrusted**: validate arguments with JSON Schema server-side, sandbox or allowlist URLs, strip instruction-like patterns, and run output through policy filters before merging into user-visible text. Separate **system** instructions from retrieved/tool content structurally (e.g., XML delimiters) and never execute embedded “new instructions.” For sensitive tools, require human approval or secondary authorization—not just model intent.
    3. Compare chain-of-thought-only vs tool-augmented reasoning for factual queries.
    *Answer:* CoT improves **internal** scratch work but still samples from parametric knowledge—fluent wrong answers remain possible for time-sensitive or niche facts. Tool use routes factual claims through **authoritative channels** (APIs, DBs), shifting errors to parseable failures you can retry or surface. Trade-off: higher latency, integration burden, and injection risk—but for stock prices, tickets, or account data, tools dominate CoT alone.
    4. What is the difference between JSON Schema validation and prompt instructions for tools?
    *Answer:* **Schema validation** is enforced by code: types, required fields, enums—deterministic and auditable. Prompt instructions only **bias** the model; they do not stop malformed calls or malicious payloads. Production systems validate **after** parsing and **before** execution; prompts describe semantics (“when to call”) while schema defines **what is legal**.
    5. How would you implement max-steps, deduplication, and circuit breakers in an agent service?
    *Answer:* Cap iterations with `max_steps` and wall-clock timeouts; **deduplicate** identical tool+args within a trace (hash key) to stop thrash loops; **circuit-break** upstream dependencies after error rates or latency SLO breaches and return degraded responses. Combine with idempotent tool design so retries do not double-charge. Log step counts and breaker trips as first-class metrics for tuning budgets.
    6. Explain short-term vs long-term memory and consistency trade-offs.
    *Answer:* **Short-term** is the live context window—fast, strongly consistent with the session, but bounded. **Long-term** (vector DB, SQL) persists across sessions but can be **stale**, permission-inconsistent, or retrieval-noisy. Trade-offs: syncing profile facts vs privacy; summarization compresses history but drops constraints—often persist structured state outside the model and inject summaries selectively.
    7. How does tree-of-thought scale in cost, and when would you avoid it?
    *Answer:* ToT branches multiple partial plans and evaluates each—cost scales roughly with **branching × depth × forward passes** per candidate. Avoid it for low-latency API agents or simple lookups where a single CoT or ReAct trace suffices. Use when search-heavy puzzles or coding tasks justify exploration and you have budget for scoring/value models.
    8. Describe a multi-agent routing architecture for customer support.
    *Answer:* A **router** (classifier or small LM) maps intent to specialized workers: billing, technical, account—each with its own tools and prompts. A **supervisor** can summarize, enforce policy, and merge outputs; async handoffs use a shared ticket state store. Key design: explicit escalation rules, consistent user-facing persona, and deduplicated tool access so subagents do not fight over writes.
    9. How do you log and trace tool calls for compliance without logging secrets?
    *Answer:* Log **tool name**, latency, status codes, correlation ids, and **redacted** arguments (mask tokens, PANs, emails); never log full OAuth tokens or passwords. Use structured spans (OpenTelemetry) linking user/session id to tool spans for audit. Store sensitive payloads encrypted with tight retention—or omit bodies entirely and log hashes for replay debugging.
    10. What is idempotency and why does it matter for retried tool calls?
    *Answer:* An **idempotent** operation yields the same effect when applied once or multiple times (e.g., “set status to refunded” with a unique idempotency key). Agents retry on timeouts and race conditions—without idempotency keys, payments ship twice or tickets duplicate. Expose idempotency keys in your tool API and dedupe server-side.

!!! interview "Follow-up Probes"
    - “Show how you’d unit-test a tool router without calling live APIs.”
    - “What happens to your MDP view when context window truncates?”
    - “How do you prioritize latency vs parallel subagent exploration?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Separate policy from tools: schema validation on the server, not trust-the-model.”
    - “ReAct grounds reasoning in observations; CoT alone has no environmental feedback.”
    - “Cross-encoder precision doesn’t apply here—agents are sequential decision problems with tool latency.”
    - “Treat tool outputs as untrusted input—prompt-injection is a first-class threat.”

## References

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023): [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023): [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (2023): [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- OpenAI Function Calling: [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023): [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* (2023): [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
