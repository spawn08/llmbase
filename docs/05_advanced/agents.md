# Agents & Tool Use

## Why This Matters for LLMs

A standalone language model maps tokens to tokens. Real assistants must **act**: call APIs, query databases, execute code, and chain steps toward a goal. **Agents** wrap the LM in a control loop that interleaves reasoning traces with **tool invocations**. This is the architecture behind OpenAI plugins, function-calling APIs, coding agents, and enterprise orchestration layers. Interviewers expect you to separate **policy** (what to do next) from **tools** (how the world responds).

Second, tool use mitigates **parametric knowledge limits**: instead of memorizing flight schedules, the model emits structured calls to a flight API. The failure modes shift toward **wrong tool choice**, **bad arguments**, and **infinite loops**—topics that pure perplexity metrics barely capture.

Third, **multi-agent** systems decompose roles (planner, critic, executor) and enable parallel exploration, but they introduce coordination overhead and new safety surfaces (agents granting each other permissions). Understanding ReAct-style loops, memory scopes, and orchestration patterns is table stakes for senior LLM systems design.

---

## Core Concepts

### ReAct: Reasoning + Acting

The **ReAct** pattern interleaves **thought**, **action**, and **observation** traces. At step \(t\), the model may output natural language reasoning, then select an action \(a_t\) from a discrete tool set; the environment returns observation \(o_t\); the history \(h_{t+1} = h_t \oplus (a_t, o_t)\) conditions the next step.

\[
\pi_\theta(a_t \mid h_t),\quad o_t = \text{Env}(a_t)
\]

Compared with **chain-of-thought** alone, ReAct grounds decisions in **external feedback**, reducing ungrounded speculation when tools exist.

!!! math-intuition "In Plain English"
    ReAct is **stop, look, go**: think a little, poke the world, read the result, repeat. The LM is not required to carry all facts in weights—only to **choose** tools and **interpret** observations.

### Tool Calling: Schemas and APIs

Tools are declared with **JSON Schema**-like signatures: name, description, parameter types, and required fields. The model emits a structured call:

```json
{"name": "weather.get", "arguments": {"city": "Berlin", "units": "metric"}}
```

The runtime validates arguments, executes, and appends the tool output to the chat. **Function calling** APIs (OpenAI, Anthropic, Gemini) fine-tune models to emit such blocks reliably.

### Planning: Chain-of-Thought and Tree-of-Thought

- **Chain-of-thought (CoT)** expands a single reasoning trace before an answer.
- **Tree-of-thought (ToT)** maintains **multiple partial plans**, scores them (by a value model or self-critique), and prunes—useful for puzzles and combinatorial tasks.

A simplified scoring step might choose branch \(b^\star\) by:

\[
b^\star = \arg\max_{b \in \mathcal{B}} V_\psi(h \parallel b)
\]

where \(V_\psi\) is a learned or prompted value head.

!!! math-intuition "In Plain English"
    CoT is one draft outline; ToT is **try a few outlines**, peek ahead, and commit to the most promising. Compute cost grows with branching factor—use sparingly in production unless tasks justify it.

### Memory: Short-Term and Long-Term

| Scope | Mechanism | Role |
|-------|-----------|------|
| **Short-term** | Conversation context window | Immediate coreference, last tool outputs |
| **Long-term** | Vector store + summarization | User prefs, prior sessions, org knowledge |
| **Working set** | Scratchpad variables in code agents | Intermediate computations |

Long-term memory often uses **retrieve-then-merge**: on each turn, embed the query, fetch top-k memories, prepend summaries to the system prompt.

### Multi-Agent Orchestration

Patterns include:

- **Router**: classify intent → dispatch to specialist agents.
- **Critic–actor**: generator proposes; critic flags risks; loop.
- **Hierarchical planner**: high-level planner expands subtasks delegated to workers.

Message-passing can be synchronous (shared transcript) or asynchronous (event bus). Conflicts arise when agents **contradict** tool results—consensus protocols or a **single source of truth** layer help.

### OpenAI-Style Function Calling (Illustrative)

The following uses the public API shape conceptually—always consult current vendor docs for exact fields.

```python
import json
from typing import Any, Callable, Dict, List


TOOLS: List[Dict[str, Any]] = [
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


def add(a: float, b: float) -> float:
    return a + b


TOOL_DISPATCH: Dict[str, Callable[..., Any]] = {
    "add": add,
}


def handle_tool_call(name: str, arguments_json: str) -> str:
    fn = TOOL_DISPATCH[name]
    args = json.loads(arguments_json)
    result = fn(**args)
    return json.dumps({"result": result})


# Pseudocode for a single assistant turn:
def fake_assistant_turn(user_text: str) -> None:
    # In production: messages = chat.completions.create(..., tools=TOOLS)
    # Model returns assistant_message with tool_calls[...]
    simulated_call = {
        "name": "add",
        "arguments": json.dumps({"a": 2, "b": 3}),
    }
    obs = handle_tool_call(simulated_call["name"], simulated_call["arguments"])
    print("Tool observation:", obs)


if __name__ == "__main__":
    fake_assistant_turn("What is 2+3?")
```

### ReAct Agent Loop (Minimal Python)

This loop does not call a remote LM; it shows how **history**, **tool routing**, and **termination** fit together. Plug `call_llm` into your provider.

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class ReactAgent:
    tools: Dict[str, Callable[..., str]]
    max_steps: int = 6
    history: List[str] = field(default_factory=list)

    def act(self, user_goal: str, call_llm: Callable[[str], str]) -> str:
        self.history.append(f"User: {user_goal}")
        for _ in range(self.max_steps):
            prompt = "\n".join(self.history) + "\nAssistant:"
            text = call_llm(prompt)
            self.history.append(f"Assistant: {text}")
            action = self._parse_action(text)
            if action is None:
                return text
            name, argstr = action
            if name.lower() in {"answer", "finish"}:
                return argstr
            if name not in self.tools:
                obs = f"ERROR: unknown tool {name}"
            else:
                try:
                    obs = self.tools[name](argstr)
                except Exception as exc:  # noqa: BLE001
                    obs = f"ERROR: {exc}"
            self.history.append(f"Observation: {obs}")
        return "Stopped: max steps reached."

    @staticmethod
    def _parse_action(text: str) -> Tuple[str, str] | None:
        m = re.search(r"Action:\s*(\w+)\s*\(([^)]*)\)", text, re.IGNORECASE)
        if not m:
            return None
        return m.group(1), m.group(2).strip().strip("'\"")


def tool_search_stub(query: str) -> str:
    knowledge = {"capital of france": "Paris", "capital of germany": "Berlin"}
    return knowledge.get(query.lower(), "No hit.")


if __name__ == "__main__":
    agent = ReactAgent(tools={"search": tool_search_stub})

    def dummy_llm(prompt: str) -> str:
        if "capital of Germany" in prompt and "Observation" not in prompt:
            return 'Thought: I should look it up.\nAction: search("capital of germany")'
        if "Observation:" in prompt:
            return "Thought: I can answer now.\nAction: answer(Berlin)"
        return "Thought: unclear."

    print(agent.act("What is the capital of Germany?", dummy_llm))
```

### Failure Modes and Guardrails

- **Tool hallucination**: model invents nonexistent functions—mitigate with strict schema validation and **allowlists**.
- **Argument errors**: types and ranges—use pydantic validation server-side.
- **Loops**: same action repeated—detect with deduplication hashes and **step caps**.
- **Prompt injection** via tool outputs—sanitize and **privilege** system instructions.

### Agents as Sequential Decision Processes

Abstractly, a tool-using agent is a **partially observed Markov decision process** (POMDP): states are hidden (full world), observations are tool outputs, actions are tool calls or final answers. Policies \(\pi_\theta(a \mid o_{\le t})\) implemented by LMs are **history-dependent**; the context window truncates history, so **summarization** becomes part of state compression.

The **Bellman** structure motivates **reward shaping** when you fine-tune with RL: intermediate rewards for correct tool use, terminal reward for task success. In practice many systems rely on **behavior cloning** from human traces or **rejection sampling** rather than full RL.

### Delegation and Permissions

Production agents need **capability-based** access: tools declare OAuth scopes, rate limits, and **human-in-the-loop** gates for irreversible actions (payments, deletions). Log **tool transcripts** for audit but **redact** secrets—never log raw API keys from model-emitted JSON.

### Contrast with RAG-Only Systems

| Pattern | Strength | Weakness |
|---------|----------|----------|
| **RAG** | Grounded factual lookup | Limited “actions” beyond retrieval |
| **Tool agent** | Writes, transacts, multi-step | Higher failure surface, loop risk |
| **Hybrid** | Retrieve then act | Orchestration complexity |

### OpenAI Chat Completions with Tools (Pattern)

Below is a **minimal** illustration of the **messages** + **tool_calls** loop using the public Python SDK shape. Replace `client` creation with your API key from environment variables—**never** hardcode secrets.

```python
import json
import os
from typing import Any, Dict, List

from openai import OpenAI


def run_tool_loop(
    user_message: str,
    model: str = "gpt-4o-mini",
) -> str:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    tools: List[Dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "get_capital",
                "description": "Return capital city for a country.",
                "parameters": {
                    "type": "object",
                    "properties": {"country": {"type": "string"}},
                    "required": ["country"],
                },
            },
        }
    ]

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "Use tools when factual lookup is needed."},
        {"role": "user", "content": user_message},
    ]

    def get_capital(country: str) -> str:
        return {"france": "Paris", "germany": "Berlin"}.get(country.lower(), "unknown")

    for _ in range(4):
        resp = client.chat.completions.create(
            model=model,
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
                if name == "get_capital":
                    out = get_capital(args["country"])
                else:
                    out = "unsupported tool"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps({"result": out}),
                    }
                )
        else:
            return msg.content or ""
    return "max turns exceeded"
```

### Planning as Search in Latent Space

Tree-of-Thought can be viewed as **beam search** over partial reasoning strings with a **pruning** heuristic. If each node expands \(b\) children and depth is \(d\), worst-case evaluations grow as \(O(b^d)\)—**branching limits** and **value model** pruning are mandatory online.

### Observability for Agents

Structured logs should capture: **tool name**, **latency**, **argument hash** (not raw PII), **outcome status**, and **retry count**. **Traces** (OpenTelemetry) correlate user sessions with backend tool failures.

### Cost Controls for Agent Loops

Each **tool** call may hit **paid** APIs—enforce **per-session** budgets, **exponential backoff** on errors, and **circuit breakers** when downstream **latency** spikes.

### Determinism vs Exploration

**Temperature** \(>0\) increases **exploration** in **ReAct** traces—useful for **brainstorming**, harmful for **auditable** workflows. For **compliance**, prefer **low** temperature and **structured** logs.

### Human-in-the-Loop Approvals

Irreversible actions (payments, **deletes**) should require **explicit** human **approval** steps—not **model** discretion alone.

### Testing Agents

**Unit-test** tool **routers** with **mock** environments; **integration-test** with **staging** APIs. **Record/replay** traces for **regression** when **prompts** change.

---

## Interview Takeaways

- **ReAct** interleaves reasoning, actions, and observations; it reduces ungrounded chains when tools are faithful.
- **Bi-encoder retrieval** scales; **ReAct** is sequential—profile latency when combining both.
- **Function calling** requires **schema discipline** and **server-side validation**, not blind `eval`.
- **Memory** tiers (context vs vector vs structured DB) map to different consistency and privacy requirements.
- **Multi-agent** gains come from specialization; costs include orchestration complexity and failure propagation.
- Always specify **termination conditions**, **max steps**, and **fallback** behaviors for production agents.

## References

- Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models* (ICLR 2023): [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- Schick et al., *Toolformer: Language Models Can Teach Themselves to Use Tools* (2023): [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Yao et al., *Tree of Thoughts: Deliberate Problem Solving with Large Language Models* (2023): [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- OpenAI Function Calling documentation: [https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling)
- Park et al., *Generative Agents: Interactive Simulacra of Human Behavior* (2023): [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)
