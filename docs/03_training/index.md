# Part 3 — Training & alignment

Training is more than “call `Trainer`”: data quality, scale, parallelism, precision, and **alignment** (SFT, preference optimization, constitutional / RLAIF-style methods) determine what the model actually does in production.

## Goals

- Outline a **pre-training pipeline** (tokenization, data mix, compute).
- Name **parallelism** strategies and what they optimize (memory vs communication).
- Contrast **RLHF**, **DPO**, and **PEFT** (LoRA / QLoRA) at a design level.

## Planned topics

| Topic | What you will get |
| --- | --- |
| Pre-training at scale | BPE, data pipeline, scaling context |
| Distributed training | Data / tensor / pipeline / model parallel overview |
| Mixed precision & quantization | FP16/BF16/INT8/GPTQ — training vs inference |
| Instruction tuning | SFT, instruction datasets, chat templates |
| RLHF | Reward model + policy optimization (high level) |
| Constitutional AI | RLAIF / critique loops (conceptual) |
| Parameter-efficient fine-tuning | LoRA, adapters, prompts |

## Status

Shell for **Phase 1**; detailed notebooks and TRL/HF examples land in **Phase 4**.
