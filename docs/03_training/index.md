# Part 3 — Training and Alignment

Pre-training at scale, distributed training, mixed precision and quantization, instruction tuning, preference optimization (RLHF/DPO), Constitutional AI, and parameter-efficient fine-tuning. This part bridges **foundation models** (what you train) with **aligned assistants** (what you ship).

---

## Goals

After completing Part 3 you will be able to:

- Describe end-to-end data and tokenization pipelines for large-scale LM pre-training
- Compare data/model parallelism, ZeRO stages, and communication primitives used in distributed training
- Explain FP16/BF16/FP8 training, loss scaling, and affine quantization (INT8/INT4) with trade-offs between PTQ and QAT
- Walk through supervised fine-tuning with masked loss on completions and mitigation strategies for catastrophic forgetting
- Derive policy-gradient and DPO objectives and contrast RLHF with direct preference optimization
- Summarize Constitutional AI / RLAIF and how they relate to human-feedback RLHF
- Compute parameter savings for LoRA/QLoRA and place adapters in a modern PEFT workflow

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [Pre-training at Scale](pretraining.md) | Data pipelines, BPE, CLM/MLM, scaling laws, LR schedules |
| 2 | [Distributed Training](distributed_training.md) | DDP, FSDP, tensor/pipeline parallel, ZeRO, AllReduce |
| 3 | [Mixed Precision & Quantization](mixed_precision.md) | FP formats, loss scaling, GPTQ/AWQ, affine quantization |
| 4 | [Instruction Tuning (SFT)](instruction_tuning.md) | FLAN/InstructGPT, masking, forgetting |
| 5 | [RLHF & DPO](rlhf.md) | Reward models, PPO, DPO, comparisons |
| 6 | [Constitutional AI](constitutional_ai.md) | RLAIF, critique-revision, red-teaming vs RLHF |
| 7 | [Parameter-Efficient Fine-Tuning](peft.md) | LoRA, QLoRA, adapters, prefix/prompt tuning |

---

Every page includes plain-English math intuition, runnable Python where it clarifies algorithms, and interview-oriented takeaways tied to production LLM training and alignment.
