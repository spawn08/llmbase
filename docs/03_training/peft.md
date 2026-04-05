# Parameter-Efficient Fine-Tuning (PEFT)

## Why This Matters for LLMs

Full fine-tuning updates **every weight** in a multi-billion-parameter model. That demands enormous GPU memory for optimizer states (often multiple copies per parameter in mixed-precision training), long training times, and frequent operational pain: checkpoint sizes that are hard to move, merge, and serve. **Parameter-efficient fine-tuning (PEFT)** freezes most of the pretrained weights and injects small trainable modules (adapters, low-rank matrices, soft prompts) so practitioners can specialize a model on a domain or task with **orders of magnitude** fewer trainable parameters. For applied ML and LLM platform roles, this is among the most practical skills: it is how teams ship custom assistants on single-GPU workstations or modest cloud instances.

Second, PEFT changes **what you are allowed to do in production**. A merged LoRA can be folded into base weights for **zero inference overhead**, while adapter-based methods may require extra forward passes or module routing. Quantized training with **QLoRA** (4-bit base weights + LoRA in higher precision) makes 7B–70B-class adaptation feasible on consumer or single professional GPUs without giving up much quality when configured carefully. Interviewers often ask you to **compute parameter counts**, compare memory footprints, and explain **when** to merge adapters versus keep them separate for multi-tenant serving.

Third, understanding PEFT is inseparable from understanding **rank**, **where** to attach adapters (attention vs. MLP), and **optimization stability**. Low-rank adaptation assumes that task-specific updates to large linear layers lie near a **low-dimensional subspace**—a useful inductive bias that works remarkably well for language tasks. You should be able to sketch \(W' = W + BA\), explain scaling rules (e.g., \(\alpha/r\)), and connect PEFT to broader ideas like prompt tuning and prefix tuning as points on a **capacity vs. compute** spectrum.

---

## Core Concepts

### Why Not Full Fine-Tune?

- **Memory**: Adam states multiply footprint; sharded optimizers help but do not erase cost.
- **Storage**: One full fine-tune per customer is often impractical; PEFT checkpoints are small.
- **Catastrophic interference**: Smaller trainable sets can reduce destructive overwrite of general capabilities (not guaranteed, but observed in practice).

??? deep-dive "Deep Dive: Multi-tenant LoRA serving"
    Keep one base model in GPU memory and swap small LoRA weights per tenant, or batch requests that share adapters—latency and memory planning depend on your inference stack (vLLM, TensorRT-LLM, custom).

---

## LoRA (Low-Rank Adaptation)

### The Idea

For a frozen weight matrix \(W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}\), LoRA adds a trainable low-rank product:

\[
W' = W + \Delta W, \quad \Delta W = B A
\]

with \(B \in \mathbb{R}^{d_{\mathrm{out}} \times r}\), \(A \in \mathbb{R}^{r \times d_{\mathrm{in}}}\), and rank \(r \ll \min(d_{\mathrm{out}}, d_{\mathrm{in}})\).

!!! math-intuition "In Plain English"
    Instead of touching millions or billions of entries in \(W\), you learn two **thin** matrices whose product **approximates** the update you would have learned with full fine-tuning. Most entries of \(W\) stay fixed; a compact “delta” path captures task-specific adjustments.

### Forward Pass

Given input \(x \in \mathbb{R}^{d_{\mathrm{in}}}\) (row-vector convention as common in code; transpose as needed):

\[
h = x W'^\top = x W^\top + x A^\top B^\top
\]

!!! math-intuition "In Plain English"
    Activations flow through the original layer **plus** a side branch: project \(x\) through \(A^\top\), then \(B^\top\), and add the result to the usual output. At inference, this can be **fused** or **merged** into a single matrix for speed.

### Scaling

LoRA implementations often include a scale \(\alpha\) (sometimes combined as \(\alpha/r\)):

\[
h = x W^\top + \frac{\alpha}{r}\, x A^\top B^\top
\]

!!! math-intuition "In Plain English"
    The scale controls how aggressively the low-rank pathway contributes **relative** to frozen weights when you change rank \(r\). Increasing \(r\) increases capacity; \(\alpha/r\) keeps updates comparable across different ranks during experimentation.

### Parameter Savings (Counting)

Trainable parameters for one LoRA pair:

\[
\#\mathrm{params} = d_{\mathrm{out}} r + r d_{\mathrm{in}} = r (d_{\mathrm{out}} + d_{\mathrm{in}})
\]

!!! math-intuition "In Plain English"
    You pay linearly in \(r\) and in layer width—**much** cheaper than full \(d_{\mathrm{out}} \times d_{\mathrm{in}}\) when \(r\) is small.

!!! example "Worked Example: LoRA Parameter Savings"
    Let \(d_{\mathrm{out}} = d_{\mathrm{in}} = 4096\). Full \(W\) has \(4096^2 = 16{,}777{,}216\) parameters.
    
    With \(r=16\):
    
    \[
    \#\mathrm{LoRA} = 16 \cdot (4096 + 4096) = 131{,}072
    \]
    
    !!! math-intuition "In Plain English"
        This is **arithmetic**: multiply rank by the sum of input and output dimensions because \(B\) has \(d_{\mathrm{out}}\times r\) entries and \(A\) has \(r\times d_{\mathrm{in}}\)—add them for total LoRA parameters on that layer.
    
    Ratio \(16{,}777{,}216 / 131{,}072 \approx 128\times\) fewer trainable parameters **for that matrix**. A transformer repeats this pattern across \(q,v,k,o\) projections per layer—total trainable counts depend on **which modules** you adapt.

??? deep-dive "Deep Dive: Which modules to train?"
    Common defaults adapt **attention projections** and sometimes **MLP** layers. Training only \(q\) and \(v\) may suffice for some tasks; adding MLP adapters increases capacity and trainable parameter count.

---

## QLoRA

QLoRA keeps the **base model in low-bit precision** (e.g., 4-bit NF4) and computes LoRA updates in **float16/bfloat16**.

### Quantized Linear (Conceptual)

A quantized weight uses a mapping from discrete codes to reals; the forward pass conceptually is:

\[
y = \mathrm{dequant}(W_{\mathrm{q}})\, x
\]

!!! math-intuition "In Plain English"
    Weights are stored compactly; at compute time they are reconstructed (fully or partially) to multiply activations—implementation details vary by kernel.

### Memory Sketch

If full-precision \(W\) needs \(M\) bytes, 4-bit weights need roughly \(M/8\) bytes **for parameters** (plus small overhead), dramatically shrinking **GPU RAM** for the frozen backbone.

??? deep-dive "Deep Dive: Double quantization and NF4"
    QLoRA uses NormalFloat4 (NF4) quantization with a block structure; “double quantization” quantizes the quantization constants to save extra bits. Read the QLoRA paper for exact storage formulas.

---

## Adapters (Houlsby / Pfeiffer Variants)

Classic adapters insert a bottleneck MLP after attention or MLP sublayers:

\[
h_{\mathrm{adapt}} = h + U\,\sigma(D\,h)
\]

with \(D \in \mathbb{R}^{r \times d}\), \(U \in \mathbb{R}^{d \times r}\), \(r \ll d\).

!!! math-intuition "In Plain English"
    Down-project activations to a small width \(r\), apply a nonlinearity, up-project back—**residual** style so the base representation dominates unless adapters learn otherwise.

Trainable fraction is higher than LoRA per layer if you stack many adapters, but still far below full fine-tuning.

---

## Prefix Tuning and Prompt Tuning

**Prefix tuning** learns continuous vectors prepended to keys/values in attention:

\[
K' = [P_K; K], \quad V' = [P_V; V]
\]

!!! math-intuition "In Plain English"
    You pretend there are extra “virtual tokens” at the start of keys and values whose embeddings are **learned**—they steer attention without changing word embeddings for real tokens.

**Prompt tuning** (Lester *et al.*) prepends soft prompt embeddings to inputs only at the embedding layer; simpler, fewer parameters, often competitive at large model sizes.

---

## Comparison Table (Rule of Thumb)

| Method | Trainable % | Memory (typical) | Merge at inference | Notes |
|--------|-------------|------------------|--------------------|-------|
| Full FT | 100% | Very high | n/a | Best capacity, highest cost |
| LoRA | 0.01–1% | Moderate | Yes | Very common default |
| QLoRA | Similar trainable | Low (4-bit base) | Yes | Great for single GPU |
| Adapters | ~1–4% | Moderate | Optional | Module insertion |
| Prefix / Prompt | <0.1% | Low | Prompt-only | Scales with model size |

---

## Worked Examples

!!! example "Worked Example: Rank vs. Alpha Ablation (Conceptual)"
    Suppose \(r=8\) underfits on a hard coding task, raising validation error. Increasing to \(r=32\) multiplies LoRA parameters roughly by \(32/8=4\) for those layers. If loss improves marginally but overfitting appears, you might reduce epochs or add dropout on adapters—PEFT is not immune to overfitting.

!!! example "Worked Example: Merging LoRA into Base Weights"
    For a linear with \(h = x(W^\top + \frac{\alpha}{r} A^\top B^\top)\), merged weight (for inference) can be formed as \(W_{\mathrm{merged}}^\top = W^\top + \frac{\alpha}{r} A^\top B^\top\). After merging, you can discard \(A,B\) and run a standard single-matrix multiply.

---

## Self-Contained Runnable Python

Below: **LoRA** via Hugging Face `peft` + **optional BitsAndBytes 4-bit** path (commented) for QLoRA-style setup. Uses a small public model.

```python
"""
LoRA fine-tuning skeleton with PEFT + Transformers.
CPU-capable with tiny model; GPU recommended for real workloads.

pip install torch transformers peft datasets accelerate
# Optional QLoRA: pip install bitsandbytes
"""
from dataclasses import dataclass

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def build_tiny_dataset(tok: AutoTokenizer) -> Dataset:
    texts = [
        "Instruction: Say hello.\nResponse: Hello!",
        "Instruction: Capital of France?\nResponse: Paris.",
    ]

    def _tok(batch):
        return tok(batch["text"], truncation=True, max_length=128)

    ds = Dataset.from_dict({"text": texts})
    return ds.map(_tok, batched=True, remove_columns=["text"])


@dataclass
class TrainCfg:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


def main() -> None:
    cfg = TrainCfg()
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    # Uncomment for QLoRA-style loading (requires GPU + bitsandbytes):
    # from transformers import BitsAndBytesConfig
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # model = AutoModelForCausalLM.from_pretrained(cfg.model_name, quantization_config=bnb_config)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust per architecture
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = build_tiny_dataset(tok)

    def _labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    tokenized = tokenized.map(_labels, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    args = TrainingArguments(
        output_dir="./peft_demo",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=200,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )
    trainer.train()

    # Merge LoRA into base weights for inference-only deployment (optional)
    merged = model.merge_and_unload()  # PeftModel method when merge supported
    print("Merged model type:", type(merged))


if __name__ == "__main__":
    main()
```

**Notes:**

- `target_modules` must match **actual layer names** (e.g., `q_proj` vs `query`); inspect `model.named_modules()`.
- For SFT with instruction masking, prefer **TRL `SFTTrainer`** with PEFT—this script demonstrates **PEFT wiring** only.

??? deep-dive "Deep Dive: Gradient checkpointing + LoRA"
    Activations can dominate memory during long sequences; gradient checkpointing trades compute for memory—often necessary alongside QLoRA for large contexts.

### Frobenius Norm Intuition for \(\Delta W = BA\)

A standard rank inequality gives:

\[
\|\Delta W\|_F \le \|B\|_F \|A\|_F
\]

!!! math-intuition "In Plain English"
    The **size** of the weight update (measured by Frobenius norm) is controlled by the norms of the two small factors—large norms can mean aggressive adaptation; regularization or dropout on LoRA layers reins in runaway updates.

### Effective Learning Rate and Optimizer Choice

Many practitioners use **higher** learning rates for LoRA than for full fine-tuning because fewer parameters update and the frozen backbone stabilizes features. There is no universal recipe—**always** validate on a dev set.

!!! example "Worked Example: Layer-Wise Trainable Count (Single Transformer Block)"
    Suppose one self-attention block has four projections \(W_q, W_k, W_v, W_o \in \mathbb{R}^{4096 \times 4096}\). Adapting all four with \(r=16\) yields per-matrix LoRA parameters \(16 \cdot (4096+4096) = 131{,}072\), times four \(\approx 524{,}288\) trainable parameters **per layer** from attention alone. Multiply by layer count \(L\) to estimate total LoRA size—still tiny versus full \(4096^2\) per matrix times many layers.

### When PEFT Is a Poor Fit

- You need **full-model** capability shifts (new languages, massive domain retraining).
- Data is abundant and **underfitting** is obvious even at high rank.
- Regulatory settings requiring **full-weight** provenance and auditing without adapter indirection.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. Write the LoRA decomposition and explain why rank \(r\) controls capacity.
    2. How do you compute trainable parameter counts for LoRA on a given layer?
    3. What is QLoRA, and why does 4-bit quantization matter for memory?
    4. Compare LoRA vs. adapter layers vs. prompt tuning—when would you pick each?
    5. How does \(\alpha/r\) scaling affect training dynamics?
    6. What are the steps to merge LoRA into base weights for serving?
    7. Which modules would you target first for instruction tuning, and why?
    8. What failure modes remain with PEFT (overfitting, underfitting, catastrophic forgetting)?
    9. How does multi-tenant serving change if each tenant has a different LoRA?
    10. Explain NF4 at a high level and one reason it works with LoRA fine-tuning.

!!! interview "Follow-up Probes"
    - “Show how you’d verify `target_modules` for a new architecture.”
    - “What changes if you use rank 4 vs 64 on the same task?”
    - “How would you detect that LoRA is underfitting?”
    - “What’s your process for choosing learning rate for LoRA vs full FT?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “Low-rank adaptation: \(W' = W + BA\), train \(A,B\) only.”
    - “QLoRA: 4-bit backbone, FP16/BF16 LoRA grads—big VRAM savings.”
    - “Merge adapters for zero inference overhead single-matrix matmuls.”
    - “PEFT enables per-customer checkpoints without full-weight copies.”

---

## References

1. Hu *et al.*, “LoRA: Low-Rank Adaptation of Large Language Models,” 2021. `https://arxiv.org/abs/2106.09685`
2. Dettmers *et al.*, “QLoRA: Efficient Finetuning of Quantized LLMs,” 2023. `https://arxiv.org/abs/2305.14314`
3. Houlsby *et al.*, “Parameter-Efficient Transfer Learning for NLP,” 2019 (adapters). `https://arxiv.org/abs/1902.00751`
4. Li & Liang, “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” 2021. `https://arxiv.org/abs/2101.00190`
5. Lester *et al.*, “The Power of Scale for Parameter-Efficient Prompt Tuning,” 2021. `https://arxiv.org/abs/2104.08691`
6. Hugging Face PEFT documentation. `https://huggingface.co/docs/peft`

---

*Last updated: 2026 — API names aligned with Hugging Face `peft` and `transformers` conventions.*
