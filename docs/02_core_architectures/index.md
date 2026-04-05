# Part 2 — Core architectures

This part is the backbone of LLM interviews: how the **Transformer** works, how **decoder-only**, **encoder-only**, and **encoder–decoder** families differ, and what **MoE** and **state-space** layers add at scale.

## Goals

- Read a **Transformer block diagram** and map it to tensors and code.
- Compare **GPT-style**, **BERT-style**, and **T5-style** objectives and masks.
- Recognize **positional encoding** choices (sinusoidal, learned, RoPE, ALiBi) and when they appear in open models.

## Planned topics

| Topic | What you will get |
| --- | --- |
| The Transformer | End-to-end stack, residual stream, FFN |
| Self-attention & MHA | QKV, heads, masking patterns |
| Positional encoding | Sinusoidal, learned, RoPE, ALiBi |
| GPT (decoder-only) | Causal LM, generation loop |
| BERT (encoder-only) | MLM, NSP (historical), fine-tuning |
| T5 (encoder–decoder) | Text-to-text framing |
| Mixture of Experts | Routing, load balancing, sparsity |
| State space models | S4 / Mamba intuition + minimal blocks |

## Artifacts

Diagrams live as **draw.io** sources in `diagrams/` and **SVG** under `docs/assets/diagrams/`. Code will prefer small, readable PyTorch modules with explicit imports.

## Status

Section shell for **Phase 1**; deep dives scheduled in **Phase 3** of the project roadmap.
