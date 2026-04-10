# Core Architectures

How modern sequence models work at the tensor level — from the original Transformer through GPT, BERT, T5, Mixture of Experts, and state-space models. This is the most interview-critical section of LLMBase.

---

## Goals

After completing this section you will be able to:

- Trace a token through every layer of a Transformer block and state the dimension at each step
- Explain Multi-Head, Grouped-Query, and Multi-Query Attention with memory cost trade-offs
- Compare sinusoidal, learned, RoPE, and ALiBi positional encodings and know when each is preferred
- Implement a minimal GPT from scratch and explain the KV cache
- Contrast BERT's masked language modeling with GPT's autoregressive objective
- Describe T5's text-to-text framing and span corruption pre-training
- Explain MoE routing, load balancing, and why sparse models scale better
- Describe how Mamba's selective state spaces achieve linear-time sequence modeling

---

## Topics

| # | Topic | What You Will Learn |
|---|-------|---------------------|
| 1 | [The Transformer](transformer.md) | Full architecture walkthrough, residual stream, Pre-Norm |
| 2 | [Self-Attention and MHA](self_attention_mha.md) | Self vs cross attention, GQA, MQA, masking patterns |
| 3 | [Positional Encoding](positional_encoding.md) | Sinusoidal, Learned, RoPE, ALiBi, extrapolation |
| 4 | [GPT (Decoder-Only)](gpt_decoder_only.md) | Causal attention, next-token prediction, KV cache, scaling |
| 5 | [BERT (Encoder-Only)](bert_encoder_only.md) | MLM, fine-tuning, embeddings, BERT variants |
| 6 | [T5 (Encoder-Decoder)](t5_encoder_decoder.md) | Text-to-text, span corruption, Flan-T5 |
| 7 | [Mixture of Experts](mixture_of_experts.md) | Router design, load balancing, Mixtral, DeepSeek |
| 8 | [State Space Models](state_space_models.md) | S4, Mamba, selective gating, hybrid architectures |

---

Every page includes plain-English math walkthroughs, worked numerical examples, runnable Python code, and FAANG-level interview questions with expected answer depth.
