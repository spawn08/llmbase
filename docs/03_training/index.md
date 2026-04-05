# Part 3 — Training and Alignment

Pre-training pipelines, distributed training, mixed precision and quantization, instruction tuning (SFT), RLHF, DPO, and parameter-efficient fine-tuning (LoRA, QLoRA).

## Pages in this section

- **[Pre-training at Scale](pretraining.md)** — Data pipelines, BPE tokenization, training objectives, Chinchilla-style budgets, compute FLOPs, learning-rate schedules, runnable BPE and Hugging Face dataset code.
- **[Distributed Training](distributed_training.md)** — Data and model parallelism, FSDP, DeepSpeed ZeRO, collectives, mixed precision, memory sketches, PyTorch FSDP and DeepSpeed JSON examples.
- **[Mixed Precision and Quantization](quantization.md)** — FP16, BF16, INT8, INT4, PTQ methods (GPTQ, AWQ, GGUF, SmoothQuant), QAT, bitsandbytes loading, manual INT8 quantize-dequantize in NumPy.

Additional topics (SFT, RLHF, DPO, LoRA) will be added in follow-on pages.
