# Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors:** Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom  
**Year:** 2023 &nbsp;|&nbsp; **Venue:** arXiv  
**Link:** [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

---

## TL;DR

Toolformer teaches a language model to **call APIs** (calculator, search, translation, calendar, QA) by **self-supervision**: the model proposes API calls, the calls are executed, and only calls that **reduce the model's prediction loss** on subsequent tokens are kept as training data. The result is **selective** tool use — the model learns **when** calling a tool helps and **when** to rely on its own knowledge.

---

## Why This Paper Matters

Toolformer bridges the gap between prompted tool use (ReAct) and natively learned tool use:

1. **Self-supervised:** No human-labeled tool-use data needed
2. **Selective:** The model learns when tools help vs. when they don't
3. **Precursor to function calling:** Modern API calling (OpenAI function calls) follows this pattern
4. **No RL needed:** Unlike some approaches, tool use is learned through standard supervised training

---

## Key Concepts Explained Simply

### The Core Idea

1. Start with a language model and a set of tool APIs
2. For each training example, have the model **propose** positions where an API call might help
3. **Execute** each proposed call and get the result
4. **Compare** the model's loss on future tokens **with** vs. **without** the tool result
5. **Keep** only the API calls that reduced loss — the tool genuinely helped
6. **Fine-tune** the model on this filtered data

### Why Self-Supervision Works

The key insight: if inserting a calculator result like `[Calculator(3.7 * 8.2) → 30.34]` into the text makes the model **better at predicting the next word**, then the calculator call was useful. If the model already knew the answer, the tool call adds no value and is filtered out.

### API Call Format

Tool calls are embedded as special tokens in the text:

```
The population of Paris is [QA("population of Paris") → 2.16 million] approximately 2 million.
```

The model learns to generate the `[API_NAME(input)` tokens, and the ` → result]` is provided by executing the API.

---

## The Math — Explained Step by Step

### Counterfactual Filtering

For a candidate API call at position \(i\) with result \(o\):

\[
\Delta \mathcal{L} = \mathcal{L}_{\text{LM}}(x_{i+1:} \mid x_{:i}) - \mathcal{L}_{\text{LM}}(x_{i+1:} \mid x_{:i}, o)
\]

**Breaking it down:**

1. \(\mathcal{L}_{\text{LM}}(x_{i+1:} \mid x_{:i})\): Loss on future tokens **without** the tool result
2. \(\mathcal{L}_{\text{LM}}(x_{i+1:} \mid x_{:i}, o)\): Loss on future tokens **with** the tool result inserted
3. **Keep the call if** \(\Delta \mathcal{L} > \tau\) (threshold, e.g., 0) — the tool helped predict future text

This is a **counterfactual test**: would the model have been better off if it had called this tool?

### When to Insert API Calls

The model is prompted with examples showing API calls. It generates candidates by sampling positions where it would insert `[API_NAME(`:

\[
P(\text{API call at position } i) = P_\theta(\text{[API\_NAME(} \mid x_{:i})
\]

Positions with high probability of generating the API token are candidate insertion points.

### Training Loss

After filtering, the model is fine-tuned on the augmented data:

\[
\mathcal{L} = -\sum_{t} \log P_\theta(x_t \mid x_{<t})
\]

where the training text now includes the API call tokens `[API(input) → result]` at selected positions.

---

## Python Implementation

```python
import numpy as np


class Tool:
    """Base class for a tool/API."""

    def __init__(self, name):
        self.name = name

    def execute(self, query):
        raise NotImplementedError


class Calculator(Tool):
    def __init__(self):
        super().__init__("Calculator")

    def execute(self, expression):
        try:
            result = eval(expression, {"__builtins__": {}})
            return str(result)
        except Exception:
            return "Error"


class QATool(Tool):
    def __init__(self, knowledge):
        super().__init__("QA")
        self.knowledge = knowledge

    def execute(self, question):
        question_lower = question.lower()
        for key, value in self.knowledge.items():
            if key in question_lower:
                return value
        return "Unknown"


class Calendar(Tool):
    def __init__(self):
        super().__init__("Calendar")

    def execute(self, query):
        return "2026-04-05"  # Simplified


def compute_loss_with_tool(text_tokens, tool_result_tokens, position,
                           base_logprobs, augmented_logprobs):
    """
    Compare loss with and without tool result insertion.

    base_logprobs: log probs for tokens AFTER position without tool
    augmented_logprobs: log probs for tokens AFTER position with tool result
    """
    loss_without = -np.sum(base_logprobs)
    loss_with = -np.sum(augmented_logprobs)
    return loss_without - loss_with  # Positive = tool helped


def filter_api_calls(candidates, threshold=0.0):
    """
    Keep only API calls where the loss reduction exceeds threshold.
    candidates: list of (position, api_call, delta_loss)
    """
    accepted = []
    rejected = []
    for pos, call, delta in candidates:
        if delta > threshold:
            accepted.append((pos, call, delta))
        else:
            rejected.append((pos, call, delta))
    return accepted, rejected


def augment_text(text, api_calls):
    """Insert accepted API calls into the text."""
    # Sort by position (reverse to maintain indices)
    sorted_calls = sorted(api_calls, key=lambda x: x[0], reverse=True)

    words = text.split()
    for pos, call_str, _ in sorted_calls:
        if pos < len(words):
            words.insert(pos, call_str)
    return " ".join(words)


class ToolformerTrainer:
    """Simulated Toolformer training pipeline."""

    def __init__(self, tools, threshold=0.0):
        self.tools = {t.name: t for t in tools}
        self.threshold = threshold

    def propose_calls(self, text, n_candidates=3):
        """
        Simulate model proposing API calls.
        In practice, the model generates these via sampling.
        """
        words = text.split()
        candidates = []

        for i in range(len(words)):
            for tool_name, tool in self.tools.items():
                if np.random.random() < 0.1:  # Simulate proposal probability
                    # Generate a plausible input for this tool
                    context = " ".join(words[max(0, i-3):i+1])
                    candidates.append({
                        "position": i,
                        "tool": tool_name,
                        "input": context,
                    })

        return candidates[:n_candidates]

    def evaluate_candidates(self, text, candidates):
        """Evaluate each candidate API call using counterfactual loss."""
        results = []
        for cand in candidates:
            tool = self.tools[cand["tool"]]
            result = tool.execute(cand["input"])

            # Simulate loss comparison (in practice, run the model twice)
            delta_loss = np.random.uniform(-0.5, 1.5)
            call_str = f'[{cand["tool"]}("{cand["input"]}") → {result}]'

            results.append({
                "position": cand["position"],
                "call_str": call_str,
                "delta_loss": delta_loss,
                "accepted": delta_loss > self.threshold,
            })

        return results

    def process_example(self, text):
        """Full pipeline for one training example."""
        candidates = self.propose_calls(text)
        evaluated = self.evaluate_candidates(text, candidates)

        accepted = [e for e in evaluated if e["accepted"]]
        rejected = [e for e in evaluated if not e["accepted"]]

        return {
            "original": text,
            "candidates": len(candidates),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "details": evaluated,
        }


def demonstrate_selective_tool_use():
    """Show when tools help vs. when they don't."""
    examples = [
        {
            "text": "The capital of France is Paris.",
            "tool": "QA",
            "query": "capital of France",
            "result": "Paris",
            "helps": False,
            "reason": "Model already knows this — tool adds no information"
        },
        {
            "text": "The square root of 1764 is 42.",
            "tool": "Calculator",
            "query": "1764 ** 0.5",
            "result": "42.0",
            "helps": True,
            "reason": "Arithmetic is unreliable — calculator reduces error"
        },
        {
            "text": "Today's date is needed for the report.",
            "tool": "Calendar",
            "query": "today",
            "result": "2026-04-05",
            "helps": True,
            "reason": "Model's training data doesn't contain today's date"
        },
    ]

    print("--- When Tools Help vs. Don't ---")
    for ex in examples:
        status = "✓ KEEP" if ex["helps"] else "✗ FILTER OUT"
        print(f"\n  Text: \"{ex['text']}\"")
        print(f"  Tool: {ex['tool']}(\"{ex['query']}\") → {ex['result']}")
        print(f"  {status}: {ex['reason']}")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)

    # Selective tool use
    demonstrate_selective_tool_use()

    # Toolformer pipeline
    print("\n--- Toolformer Training Pipeline ---")
    tools = [
        Calculator(),
        QATool({"france": "Paris", "python": "1991", "transformer": "2017"}),
        Calendar(),
    ]
    trainer = ToolformerTrainer(tools, threshold=0.0)

    texts = [
        "The distance is calculated as 45.7 times 12.3 kilometers",
        "Python was created by Guido van Rossum many years ago",
        "Today we will discuss the meeting schedule",
    ]

    for text in texts:
        result = trainer.process_example(text)
        print(f"\n  Text: \"{result['original'][:60]}...\"")
        print(f"  Candidates: {result['candidates']}, "
              f"Accepted: {result['accepted']}, "
              f"Rejected: {result['rejected']}")
        for d in result['details']:
            status = "✓" if d['accepted'] else "✗"
            print(f"    {status} ΔL={d['delta_loss']:+.3f}: {d['call_str'][:60]}")

    # Counterfactual comparison
    print("\n--- Counterfactual Loss Test ---")
    base_logprobs = np.array([-2.5, -3.1, -1.8, -2.0])
    augmented_logprobs = np.array([-1.2, -1.5, -1.0, -1.3])
    delta = compute_loss_with_tool(
        None, None, 0, base_logprobs, augmented_logprobs
    )
    print(f"  Loss without tool: {-np.sum(base_logprobs):.2f}")
    print(f"  Loss with tool: {-np.sum(augmented_logprobs):.2f}")
    print(f"  ΔL = {delta:.2f} {'(KEEP — tool helps)' if delta > 0 else '(FILTER — tool hurts)'}")
```

---

## Interview Importance

Toolformer is valuable for understanding **learned** vs. **prompted** tool use, and the self-supervised approach to teaching models when to use tools.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: How does Toolformer discover useful calls without hand-labeled tool traces?

**Answer:** Toolformer uses **self-supervision** through counterfactual loss comparison:
1. The model proposes candidate API calls at various positions in the text
2. Each proposed call is executed to get the real result
3. The model's loss on subsequent tokens is compared **with** vs. **without** the tool result
4. Only calls that **reduce** the prediction loss are kept as training data
5. The model is then fine-tuned on this filtered, augmented data

No human needs to label "this is where a tool should be called" — the model discovers it by measuring whether the tool actually helps predict future text.

### Q2: What risks arise from letting models invoke external APIs?

**Answer:**
1. **Security:** Malicious prompt injection could trick the model into calling dangerous APIs
2. **Cost:** Unconstrained API calls can be expensive (e.g., paid search APIs)
3. **Privacy:** The model might send sensitive user data to external services
4. **Rate limiting:** Excessive calls can overwhelm external services
5. **Correctness:** API results may be wrong, and the model might trust them blindly
6. **Latency:** Tool calls add latency; cascading failures in tools can hang the system
7. **Authorization:** The model acts with the user's permissions, which may be too broad

### Q3: Compare Toolformer to ReAct-style prompt engineering with a frozen model.

**Answer:**
| Aspect | Toolformer | ReAct |
|---|---|---|
| Training | Fine-tuned to call tools | Frozen model with prompts |
| Tool decision | **Learned** — model knows when to call | **Prompted** — depends on prompt quality |
| Selectivity | High — filtered by loss reduction | Variable — depends on the LLM's reasoning |
| New tools | Requires re-training | Just update the prompt |
| Flexibility | Fixed tool set from training | Any tool describable in a prompt |
| Efficiency | Direct generation of API tokens | Verbose Thought/Action/Observation format |

---

## Connections to Other Papers

- **ReAct** → Prompted tool use vs. Toolformer's learned tool use
- **Chain-of-Thought** → CoT reasons internally; Toolformer uses external computation
- **GPT-3** → Base model for Toolformer experiments
- **InstructGPT** → Function calling in ChatGPT extends these ideas to production

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Core idea | Self-supervised tool-use learning via loss comparison |
| Filtering rule | Keep API calls that reduce prediction loss (ΔL > 0) |
| No RL needed | Standard supervised fine-tuning on filtered data |
| Selectivity | Model learns when tools help vs. when to use own knowledge |
| API format | `[ToolName(input) → result]` embedded in text |
| vs. ReAct | Learned tool use (training) vs. prompted tool use (inference) |
