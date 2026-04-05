# Part 2 — Core Architectures

This part is the backbone of LLM interviews: how the **Transformer** works, how **decoder-only**, **encoder-only**, and **encoder–decoder** families differ, and what **MoE** and **state-space** layers add at scale.

## Goals

- Read a **Transformer block diagram** and map it to tensors and code.
- Compare **GPT-style**, **BERT-style**, and **T5-style** objectives and masks.
- Recognize **positional encoding** choices (sinusoidal, learned, RoPE, ALiBi) and when they appear in open models.
- Understand **MoE** routing and **SSM** state dynamics as alternatives to dense attention.

## Topics

| # | Topic | Key ideas |
| --- | --- | --- |
| 2.1 | [The Transformer](transformer.md) | Full block walkthrough, Pre-Norm, RMSNorm, SwiGLU, residual stream |
| 2.2 | [Self-Attention & MHA](self_attention_mha.md) | QKV, multi-head, GQA/MQA, masking patterns, cross-attention |
| 2.3 | [Positional Encoding](positional_encoding.md) | Sinusoidal, learned, RoPE, ALiBi — all four implemented |
| 2.4 | [GPT (Decoder-Only)](gpt_decoder_only.md) | Causal LM, weight tying, in-context learning, nanoGPT |
| 2.5 | [BERT (Encoder-Only)](bert_encoder_only.md) | MLM, fine-tuning, embeddings, BERT variants |
| 2.6 | [T5 (Encoder-Decoder)](t5_encoder_decoder.md) | Text-to-text framing, span corruption, ablation findings |
| 2.7 | [Mixture of Experts](mixture_of_experts.md) | Top-k routing, load balancing, Mixtral, scaling properties |
| 2.8 | [State Space Models](state_space_models.md) | S4, Mamba selective scan, dual modes, hybrid architectures |

## How to read

Start at 2.1 (the Transformer block) — it's the foundation for all variants. Then 2.2–2.3 go deeper into attention and positional encoding. Pages 2.4–2.6 cover the three major architecture families. Pages 2.7–2.8 cover modern extensions beyond dense Transformers.

## Status

All eight topics are complete with math, runnable code (full imports), and interview takeaways. Diagrams (draw.io SVGs) will be added in a future polish pass.
