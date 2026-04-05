# Part 3 — Training and Alignment

How models go from random weights to following instructions. Pre-training data pipelines, distributed training at scale, quantization, supervised fine-tuning, reinforcement learning from human feedback, and parameter-efficient adaptation.

---

## Goals

After completing Part 3 you will be able to:

- Design a data pipeline for pre-training including deduplication and quality filtering
- Calculate memory budgets for distributed training and choose the right parallelism strategy
- Explain the three-stage RLHF pipeline and contrast it with DPO
- Implement LoRA and QLoRA fine-tuning and explain why low-rank adaptation works
- Describe Constitutional AI and how it scales alignment with AI feedback

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Pre-training at Scale](pretraining.md) | Data pipelines, BPE tokenization, compute budgets, scaling laws |
| 2 | [Distributed Training](distributed_training.md) | Data/model/pipeline parallelism, FSDP, DeepSpeed ZeRO |
| 3 | [Mixed Precision and Quantization](mixed_precision.md) | FP16, BF16, INT8, INT4, GPTQ, AWQ, quantization-aware training |
| 4 | [Instruction Tuning and SFT](instruction_tuning.md) | FLAN, InstructGPT, chat templates, SFT data formats |
| 5 | [RLHF and DPO](rlhf.md) | Reward models, PPO, DPO, KL penalty, preference optimization |
| 6 | [Constitutional AI](constitutional_ai.md) | RLAIF, critique-revision, self-alignment at scale |
| 7 | [Parameter-Efficient Fine-Tuning](peft.md) | LoRA, QLoRA, adapters, prefix tuning, prompt tuning |

---

Every page includes plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions.
