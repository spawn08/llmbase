# Interview Question Bank

A comprehensive collection of 85+ questions organized by topic and difficulty. These are the questions asked at Google, Meta, OpenAI, Anthropic, Amazon, Microsoft, Apple, and top AI startups for roles involving LLMs.

Each question includes the **expected answer depth**, **common follow-ups**, and **red-flag answers** that suggest shallow understanding.

---

## How to use this page

- Questions marked **(Core)** test fundamental understanding — you must nail these.
- Questions marked **(Advanced)** test depth — strong answers here set you apart.
- Questions marked **(System Design)** test practical engineering judgment.
- Questions marked **(Research)** test awareness of the frontier.

---

## Foundations (30 Questions)

### Language Modeling

!!! interview "1. (Core) Explain the chain rule decomposition for language models."
    **Expected depth:** Write the joint probability \(P(w_1, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1})\). Explain that every autoregressive LM — from bigrams to GPT-4 — is an instantiation of this factorization. The only difference is how they parameterize the conditional.

    **Follow-up:** "What assumption does an n-gram model make about this factorization?"

    **Red flag:** Cannot write the formula or confuses it with Bayes' rule.

!!! interview "2. (Core) What is perplexity and why do we use it?"
    **Expected depth:** Perplexity = \(\exp(H)\) where \(H\) is the average cross-entropy per token. Intuitively, it measures the effective vocabulary size the model is choosing from at each step. Lower is better. A perplexity of 1 means perfect prediction. Mention that perplexity is only comparable between models using the same tokenizer on the same test set.

    **Follow-up:** "If model A has perplexity 15 and model B has perplexity 30 on the same test set, what does that mean concretely?"

    **Red flag:** Says "it measures how confused the model is" without explaining the exponential relationship to cross-entropy.

!!! interview "3. (Core) Why do n-gram models fail at scale?"
    **Expected depth:** The number of possible n-grams grows as \(V^n\). For a vocabulary of 50,000 and \(n=5\), that is \(50000^5 \approx 3 \times 10^{23}\) possible 5-grams — most never appear in training data. This is the curse of dimensionality. Smoothing helps but cannot generalize to unseen contexts. Neural models solve this by mapping words to continuous vectors where similar words share parameters.

    **Follow-up:** "How does smoothing address the zero-probability problem?"

    **Red flag:** Only mentions "sparsity" without quantifying the combinatorial explosion.

### Word Embeddings

!!! interview "4. (Core) Explain how Word2Vec learns word representations."
    **Expected depth:** Describe both Skip-gram (predict context from center word) and CBOW (predict center word from context). The key insight is that the hidden-to-output weight matrix learns vectors where words appearing in similar contexts get similar vectors. Mention the training objective — maximize \(\log P(\text{context} \mid \text{center})\) — and why negative sampling is needed (full softmax over 50K+ words is prohibitive). Negative sampling turns the problem into binary classification: distinguish real context words from random noise words.

    **Follow-up:** "What is the complexity difference between full softmax and negative sampling?"

    **Red flag:** Cannot explain the training objective or confuses Skip-gram and CBOW.

!!! interview "5. (Core) Why does cosine similarity work for word embeddings?"
    **Expected depth:** Cosine similarity measures the angle between vectors regardless of magnitude. Embeddings trained with co-occurrence objectives push words into regions of the vector space where direction encodes semantic similarity. Cosine handles the fact that some words have larger magnitude (higher frequency) without that affecting similarity judgments. Contrast with Euclidean distance which is magnitude-sensitive.

    **Follow-up:** "When would Euclidean distance be more appropriate?"

    **Red flag:** Cannot explain why direction matters more than magnitude.

!!! interview "6. (Advanced) Explain the king - man + woman = queen analogy and when it fails."
    **Expected depth:** The embedding space captures relational structure as vector offsets. The gender direction (man → woman) is roughly parallel to (king → queen). This works because Word2Vec implicitly factorizes a PMI matrix. It fails for: infrequent words (poor embedding quality), multi-sense words (one vector per word), abstract relationships, and relationships not well-represented in the training corpus.

    **Follow-up:** "How do contextual embeddings handle polysemy?"

    **Red flag:** Treats it as magic without explaining why linear structure exists.

!!! interview "7. (Advanced) Compare static embeddings (Word2Vec) with contextual embeddings (BERT)."
    **Expected depth:** Static embeddings assign one fixed vector per word regardless of context — "bank" in "river bank" and "bank account" gets the same vector. Contextual embeddings produce different vectors for the same token depending on surrounding words. BERT's embeddings come from running the full Transformer and reading the final hidden state for each token position. ELMo was the first contextual model (bidirectional LSTM). Trade-offs: static embeddings are fast and small (lookup table), contextual embeddings require a forward pass but handle polysemy and syntax.

    **Follow-up:** "Can you get contextual embeddings from GPT? How do they differ from BERT's?"

    **Red flag:** Cannot articulate the polysemy problem that contextual embeddings solve.

### Neural Language Models

!!! interview "8. (Core) Explain the vanishing gradient problem in RNNs."
    **Expected depth:** During backpropagation through time, gradients are multiplied by the recurrent weight matrix at each step. If the largest singular value of this matrix is less than 1, gradients shrink exponentially: \(0.9^{50} \approx 0.005\). This means the network cannot learn dependencies beyond roughly 10-20 steps. The exploding gradient problem (singular value > 1) is solved by gradient clipping, but vanishing gradients require architectural solutions like LSTM/GRU.

    **Follow-up:** "How exactly does LSTM's constant error carousel solve this?"

    **Red flag:** Cannot give the mathematical intuition (repeated multiplication by values < 1).

!!! interview "9. (Core) Walk through the LSTM gate equations and explain what each gate does."
    **Expected depth:** Four components: forget gate \(f_t = \sigma(W_f[h_{t-1}, x_t])\) decides what to discard from cell state; input gate \(i_t = \sigma(W_i[h_{t-1}, x_t])\) decides what new information to store; candidate \(\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t])\) creates candidate new values; cell update \(c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t\) applies forget and add; output gate \(o_t = \sigma(W_o[h_{t-1}, x_t])\) decides what to output; hidden state \(h_t = o_t \odot \tanh(c_t)\). The cell state acts as a conveyor belt — gradients can flow through multiplication by the forget gate without repeated matrix multiplication.

    **Follow-up:** "Why does GRU merge the forget and input gates? What are the trade-offs?"

    **Red flag:** Cannot name all four gates or explain the cell state's role in gradient flow.

!!! interview "10. (Advanced) Why were LSTMs replaced by Transformers?"
    **Expected depth:** LSTMs process sequences sequentially — token \(t\) requires the output of token \(t-1\). This prevents parallelism during training. Transformers compute attention over all positions simultaneously, enabling GPU parallelism and much faster training. Transformers also handle long-range dependencies better because every position directly attends to every other position (O(1) path length vs O(T) for RNNs). The cost is O(T^2) memory for attention, which SSMs now address.

    **Follow-up:** "Are there any tasks where LSTMs still outperform Transformers?"

    **Red flag:** Only says "Transformers are better" without explaining the parallelism advantage.

### Sequence-to-Sequence and Attention

!!! interview "11. (Core) Explain the bottleneck problem in encoder-decoder models and how attention solves it."
    **Expected depth:** In vanilla seq2seq, the entire input sequence is compressed into a single fixed-size vector (the encoder's final hidden state). For long sequences, this vector cannot represent all the information — it is a lossy compression. Attention solves this by letting the decoder look at all encoder hidden states at every decoding step. It computes a weighted sum of encoder states where the weights (attention scores) indicate which source positions are relevant for the current decoding step.

    **Follow-up:** "What is the difference between Bahdanau and Luong attention?"

    **Red flag:** Cannot explain why a fixed-size vector is a bottleneck.

!!! interview "12. (Core) Explain scaled dot-product attention step by step."
    **Expected depth:** Given queries Q, keys K, values V: (1) compute scores \(S = QK^T\), (2) scale by \(\sqrt{d_k}\) to prevent softmax saturation, (3) optionally apply mask, (4) apply softmax row-wise to get weights, (5) multiply weights by V. The scaling is necessary because for large \(d_k\), dot products have variance proportional to \(d_k\), pushing softmax into regions with near-zero gradients.

    **Follow-up:** "What happens if you remove the scaling factor?"

    **Red flag:** Cannot explain why we divide by \(\sqrt{d_k}\).

!!! interview "13. (Advanced) What is teacher forcing and what is its downside?"
    **Expected depth:** During training, the decoder receives the ground-truth previous token as input instead of its own prediction. This stabilizes training and speeds convergence. The downside is exposure bias: at inference time, the model feeds its own (potentially wrong) predictions, a distribution it never saw during training. Errors compound. Scheduled sampling mitigates this by gradually mixing in model predictions during training.

    **Follow-up:** "How does beam search interact with exposure bias?"

    **Red flag:** Cannot name exposure bias.

### Information Theory

!!! interview "14. (Core) Why is cross-entropy the standard training loss for language models?"
    **Expected depth:** The training objective is to minimize the KL divergence between the true data distribution and the model's distribution. Since \(D_{KL}(p \| q) = H(p, q) - H(p)\) and the entropy of the data \(H(p)\) is constant, minimizing KL divergence is equivalent to minimizing cross-entropy \(H(p, q) = -\sum p(x) \log q(x)\). For language models with one-hot targets, this simplifies to \(-\log q(w_{\text{true}})\) — the negative log probability of the correct token. Maximum likelihood estimation and cross-entropy minimization are mathematically identical.

    **Follow-up:** "What is the relationship between cross-entropy and perplexity?"

    **Red flag:** Cannot connect cross-entropy to KL divergence or maximum likelihood.

!!! interview "15. (Core) Explain KL divergence. Is it symmetric?"
    **Expected depth:** \(D_{KL}(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)}\). It measures how much information is lost when using \(q\) to approximate \(p\). It is NOT symmetric — \(D_{KL}(p \| q) \neq D_{KL}(q \| p)\). Forward KL (\(D_{KL}(p \| q)\)) is mode-covering: it forces \(q\) to cover all modes of \(p\), potentially spreading probability mass. Reverse KL (\(D_{KL}(q \| p)\)) is mode-seeking: it allows \(q\) to collapse onto a single mode of \(p\). This matters for RLHF where the KL penalty uses \(D_{KL}(\pi \| \pi_{\text{ref}})\) to keep the policy close to the reference model.

    **Follow-up:** "In RLHF, why do we use KL divergence as a penalty?"

    **Red flag:** Says KL divergence is a "distance metric" (it is not — it violates symmetry and triangle inequality).

!!! interview "16. (Advanced) Explain forward vs reverse KL divergence with a concrete example."
    **Expected depth:** Suppose the true distribution \(p\) is bimodal (two peaks) and we approximate it with a unimodal \(q\). Forward KL forces \(q\) to spread across both modes (because wherever \(p > 0\), \(q\) must also be > 0 to avoid infinite divergence). Reverse KL allows \(q\) to lock onto one mode (because wherever \(q > 0\), \(p\) must also be > 0 — but \(q\) can be zero where \(p\) is positive). This is why forward KL is used for variational autoencoders (mode-covering) and reverse KL appears in policy optimization (mode-seeking behavior).

    **Follow-up:** "Which direction of KL is used in DPO vs PPO?"

    **Red flag:** Cannot articulate mode-covering vs mode-seeking.

---

## Architecture (30 Questions)

### The Transformer

!!! interview "17. (Core) Walk through a Transformer block step by step."
    **Expected depth:** Input \(x\) (shape \(B \times T \times d\)) → LayerNorm (Pre-Norm) → Multi-Head Self-Attention → Add residual (\(x + \text{MHA}(\text{LN}(x))\)) → LayerNorm → Feed-Forward Network (two linear layers with activation, typically SwiGLU in modern models) → Add residual. The output has the same shape as the input. In the Pre-Norm variant (used by GPT-2+), normalization happens before each sublayer. In Post-Norm (original Transformer), normalization happens after.

    **Follow-up:** "Why did modern LLMs switch from Post-Norm to Pre-Norm?"

    **Red flag:** Cannot describe the residual connections or confuses Pre/Post-Norm.

!!! interview "18. (Core) Explain the Feed-Forward Network in a Transformer and SwiGLU."
    **Expected depth:** The FFN applies two linear transformations with a nonlinearity: \(\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2\). It operates independently on each position. The inner dimension is typically 4x the model dimension. SwiGLU replaces ReLU with a gated variant: \(\text{SwiGLU}(x) = (\text{Swish}(W_1 x) \odot (W_3 x)) W_2\), adding a third weight matrix. Despite more parameters per layer, SwiGLU yields better performance per FLOP. The FFN is often interpreted as a key-value memory where \(W_1\) rows are keys and \(W_2\) columns are values.

    **Follow-up:** "How does the FFN's parameter count compare to attention's? Which dominates in large models?"

    **Red flag:** Cannot state the dimensions or explain why there is an inner expansion.

!!! interview "19. (Advanced) Explain the residual stream interpretation of Transformers."
    **Expected depth:** Instead of viewing a Transformer as a sequential pipeline, think of a shared residual stream that every layer reads from and writes to. Each attention head and FFN adds a small vector to the stream. The final output is the sum of the original input embedding plus all the contributions from every head and FFN across all layers. This view explains superposition (multiple features encoded in the same dimensions), skip connections' importance, and why shallow layers' representations persist to the output.

    **Follow-up:** "What is superposition and why does it matter for interpretability?"

    **Red flag:** Only mentions "skip connections prevent vanishing gradients" without the stream interpretation.

!!! interview "20. (Core) What is RMSNorm and why is it preferred over LayerNorm?"
    **Expected depth:** LayerNorm normalizes by subtracting the mean and dividing by the standard deviation, then applies learned scale and shift parameters. RMSNorm skips the mean centering — it only divides by the root mean square: \(\text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2}\). This is faster (no mean computation) and empirically works just as well. LLaMA, PaLM, and most modern LLMs use RMSNorm.

    **Follow-up:** "Where does normalization happen in the block — before or after the sublayer?"

    **Red flag:** Cannot explain the mathematical difference between LayerNorm and RMSNorm.

### Attention Variants

!!! interview "21. (Core) Why do we use multiple attention heads instead of one big head?"
    **Expected depth:** Multiple heads allow the model to attend to different aspects of the input simultaneously. Head 1 might learn positional relationships, head 2 might learn syntactic dependencies, head 3 might learn semantic similarity. Each head operates on a \(d_k = d_{\text{model}} / h\)-dimensional subspace. The outputs are concatenated and projected. Having \(h\) heads with dimension \(d_k\) costs the same as one head with dimension \(d_{\text{model}}\), but the factored structure acts as an inductive bias for learning diverse attention patterns.

    **Follow-up:** "What happens if you prune some attention heads? How much can you prune?"

    **Red flag:** Says "more heads = more parameters" (false — parameter count is the same).

!!! interview "22. (Core) Explain Grouped-Query Attention and why it matters."
    **Expected depth:** Standard MHA has \(h\) query, key, and value heads. GQA groups query heads so multiple query heads share the same K and V heads. For example, 32 query heads with 8 KV heads (GQA-4): each KV head serves 4 query heads. This reduces the KV cache by 4x during inference (critical for serving). Multi-Query Attention (MQA) is the extreme case with 1 KV head. GQA was introduced by LLaMA-2 and strikes a balance between MQA's memory savings and MHA's quality.

    **Follow-up:** "Compute the KV cache size for a 70B parameter model with GQA-8 serving a 4096-token context."

    **Red flag:** Cannot explain the memory savings or confuses GQA with MQA.

!!! interview "23. (Advanced) Explain causal masking and why it is necessary for autoregressive models."
    **Expected depth:** Causal masking sets the upper triangle of the attention score matrix to \(-\infty\) before softmax, so each position can only attend to itself and earlier positions. Without this, the model could see future tokens during training, making it trivial to predict the next word (just copy the answer). This enforces the autoregressive property: \(P(w_t \mid w_{1:t-1})\). During inference, causal masking is implicit because we only have tokens up to position \(t\).

    **Follow-up:** "What is prefix masking and when is it used?"

    **Red flag:** Cannot explain why masking is needed for training (not just inference).

### Positional Encoding

!!! interview "24. (Core) Compare RoPE and ALiBi for positional encoding."
    **Expected depth:** RoPE rotates query and key vectors by an angle proportional to their position. Relative position information emerges from the dot product of rotated vectors: \(q_m^T k_n\) depends on \(m - n\). ALiBi adds a linear bias to attention scores: \(-m|i - j|\) where \(m\) is head-specific and \(|i-j|\) is the distance. RoPE modifies the vectors; ALiBi modifies the scores. ALiBi extrapolates better to longer contexts out of the box. RoPE needs scaling tricks (NTK-aware, YaRN) for extrapolation but retains more positional expressiveness. Most open-source LLMs (LLaMA, Mistral) use RoPE.

    **Follow-up:** "What is NTK-aware RoPE scaling and why does it help with long contexts?"

    **Red flag:** Cannot explain the difference between modifying vectors vs modifying scores.

!!! interview "25. (Advanced) Why is attention permutation-equivariant without positional encoding?"
    **Expected depth:** Attention computes \(\text{softmax}(QK^T / \sqrt{d_k})V\). If you permute the input sequence, Q, K, V are permuted in the same way, and the output is permuted identically. The function does not inherently know which token is at which position. This means "dog bites man" and "man bites dog" would produce the same attention pattern (up to permutation). Positional encoding breaks this symmetry by injecting position information.

    **Follow-up:** "How would you test whether a model has learned positional information?"

    **Red flag:** Does not understand what permutation equivariance means.

### GPT and Decoder-Only Models

!!! interview "26. (Core) Explain in-context learning. Why is it surprising?"
    **Expected depth:** In-context learning is the ability to learn new tasks from examples provided in the prompt — without any gradient updates. You give GPT a few input-output pairs and it generalizes to new inputs. This is surprising because the model weights are frozen; learning happens purely through the forward pass. One theory is that Transformer layers implement implicit gradient descent — the attention pattern at deeper layers mimics what fine-tuning would do. The quality improves with model scale (a hallmark of emergent abilities).

    **Follow-up:** "What is the relationship between in-context learning and meta-learning?"

    **Red flag:** Thinks in-context learning involves updating the model weights.

!!! interview "27. (Core) Explain the KV cache. How does it speed up autoregressive generation?"
    **Expected depth:** During autoregressive generation, at step \(t\) we only need to compute Q, K, V for the new token. The K and V vectors for all previous tokens are unchanged (because of causal masking — they don't attend to the new token). So we cache them and only compute the new K, V row. This reduces the per-step computation from \(O(T \cdot d)\) to \(O(d)\) for the K, V projection. The cache grows linearly with sequence length and is the dominant memory cost during inference.

    **Follow-up:** "How does GQA reduce KV cache size? Compute the size for a specific model configuration."

    **Red flag:** Cannot explain why K and V (but not Q) can be cached.

!!! interview "28. (Advanced) Explain weight tying in GPT models."
    **Expected depth:** Weight tying means the input token embedding matrix and the output projection matrix (which maps hidden states to vocabulary logits) share the same weights. If the embedding matrix is \(E \in \mathbb{R}^{V \times d}\), the output logits are \(h \cdot E^T\). This reduces parameter count by \(V \times d\) (significant for large vocabularies), acts as a regularizer, and makes intuitive sense: a token's output representation should be close to its input embedding in the same space.

    **Follow-up:** "Does LLaMA-2 use weight tying? Why or why not?"

    **Red flag:** Cannot explain why sharing these specific matrices makes sense.

### BERT and Encoder Models

!!! interview "29. (Core) Why can't BERT generate text like GPT?"
    **Expected depth:** BERT uses bidirectional attention — every token can attend to every other token including future ones. During pre-training with MLM, BERT sees the entire sequence (with some tokens masked) and predicts the masked tokens. It was never trained to predict the *next* token given only *previous* tokens. Autoregressive generation requires causal masking, which BERT does not have. You could technically do iterative masked prediction, but BERT wasn't trained for this and it violates the independence assumptions of MLM.

    **Follow-up:** "Could you modify BERT to do generation? What would you lose?"

    **Red flag:** Says "BERT is too small to generate" or does not mention bidirectional attention.

!!! interview "30. (Core) Compare BERT fine-tuning with GPT prompting for a classification task."
    **Expected depth:** BERT fine-tuning: add a classification head on top of the [CLS] token, train on labeled data with supervised cross-entropy, update all parameters. Typically needs 1K-10K labeled examples. The model learns task-specific representations. GPT prompting: craft a text prompt with task description and few examples, generate a classification label as text. No gradient updates. Works with very few examples (zero/few-shot) but less reliable. Fine-tuning gives higher accuracy; prompting gives more flexibility and no training cost. Modern approach: use GPT for prototyping and few-shot, switch to fine-tuned smaller models for production.

    **Follow-up:** "When would you choose a fine-tuned BERT over prompting GPT-4?"

    **Red flag:** Does not mention the [CLS] token or the parameter update difference.

!!! interview "31. (Advanced) Explain the key differences between RoBERTa, ALBERT, and DeBERTa."
    **Expected depth:** RoBERTa: removes NSP, uses dynamic masking, trains longer on more data — shows BERT was undertrained. ALBERT: factorizes the embedding matrix (\(V \times d \to V \times e + e \times d\), \(e \ll d\)), shares parameters across layers — dramatic parameter reduction. DeBERTa: disentangles content and position embeddings, uses relative position encoding, applies absolute position only at the decoding layer — consistently outperforms BERT-base/large. Know that DeBERTa's disentangled attention computes separate content-to-content, content-to-position, and position-to-content attention.

    **Follow-up:** "Why is DeBERTa's disentangled attention better than adding position to the input?"

    **Red flag:** Cannot name at least one specific change each model makes.

### T5 and Encoder-Decoder

!!! interview "32. (Core) What is text-to-text framing and why does T5 use it?"
    **Expected depth:** Every NLP task is cast as mapping an input text string to an output text string. Classification: "sentiment: I love this → positive". Translation: "translate English to German: Hello → Hallo". Summarization: "summarize: <long text> → <summary>". This unifies all tasks under a single seq2seq architecture and training objective (teacher-forced cross-entropy). The benefit is simplicity: one model, one objective, one architecture, applied to everything. Task prefixes act as implicit task embeddings.

    **Follow-up:** "What are the downsides of text-to-text for classification tasks?"

    **Red flag:** Cannot give a concrete example of how a classification task becomes text-to-text.

!!! interview "33. (Advanced) Explain span corruption pre-training and how it differs from MLM."
    **Expected depth:** BERT's MLM masks individual tokens (15% of tokens, with 80/10/10 mask/random/keep split). T5's span corruption masks contiguous spans of tokens and replaces each span with a single sentinel token (\<extra_id_0\>, \<extra_id_1\>, ...). The target is the concatenation of sentinels + the original tokens in the masked spans. This is more efficient: the target sequence is shorter (only the masked spans, not the full sequence), and the model learns to predict spans of varying length, which is closer to generation.

    **Follow-up:** "What span length did the T5 paper find optimal?"

    **Red flag:** Confuses span corruption with MLM or cannot explain the sentinel tokens.

### Mixture of Experts

!!! interview "34. (Core) Explain how the router in a Mixture of Experts layer works."
    **Expected depth:** Each token's hidden state \(x\) is passed through a linear layer (the router/gate) to produce logits over \(N\) experts: \(g = W_g x\). Top-k experts are selected (typically k=2). The token is processed by each selected expert, and the outputs are combined using the softmax-normalized gate values as weights: \(\text{output} = \sum_{i \in \text{top-k}} \text{softmax}(g)_i \cdot E_i(x)\). Most of the model's parameters sit in the experts, but only k/N fraction activates per token.

    **Follow-up:** "What is expert collapse and how do you prevent it?"

    **Red flag:** Cannot explain the top-k selection or how outputs are weighted.

!!! interview "35. (Core) Explain load balancing in MoE models."
    **Expected depth:** Without intervention, the router often sends most tokens to a few popular experts (the rich-get-richer problem). Load balancing uses an auxiliary loss that penalizes uneven distributions: \(\mathcal{L}_{\text{aux}} = N \sum_{i=1}^{N} f_i \cdot P_i\) where \(f_i\) is the fraction of tokens routed to expert \(i\) and \(P_i\) is the mean router probability for expert \(i\). This loss is minimized when all experts receive equal traffic. The coefficient \(\alpha\) (typically 0.01) controls the trade-off between routing quality and balance.

    **Follow-up:** "Mixtral uses 8 experts with top-2. How many parameters activate per token compared to a dense model of the same size?"

    **Red flag:** Cannot explain what happens without load balancing (expert collapse).

!!! interview "36. (Advanced) Compare expert-choice routing with token-choice routing."
    **Expected depth:** In standard token-choice routing, each token picks its top-k experts. This can lead to some experts being overloaded and others idle. In expert-choice routing, each expert independently selects its top-k tokens from the batch. This guarantees perfect load balance (each expert processes exactly the same number of tokens) and removes the need for auxiliary loss, capacity factors, and dropped tokens. The trade-off: variable compute per token (some tokens may be processed by zero experts, some by many).

    **Follow-up:** "What happens to tokens that no expert selects?"

    **Red flag:** Cannot articulate the inversion of selection direction.

### State Space Models

!!! interview "37. (Core) What problem do SSMs solve that Transformers have?"
    **Expected depth:** Self-attention has \(O(T^2)\) time and memory complexity in sequence length \(T\). For a 100K-token context, that is \(10^{10}\) operations per layer — prohibitive. SSMs process sequences in \(O(T)\) by maintaining a fixed-size hidden state that gets updated at each step (like an RNN, but designed to be parallelizable during training via convolution). Mamba showed that SSMs with input-dependent state transitions can match Transformer quality while being much faster for long sequences.

    **Follow-up:** "If SSMs are O(T), why don't all models use them?"

    **Red flag:** Cannot quantify the attention complexity bottleneck.

!!! interview "38. (Advanced) Explain Mamba's selective state space mechanism."
    **Expected depth:** Classic SSMs (like S4) have fixed state transition matrices A, B, C — the same dynamics apply to every token regardless of content. Mamba makes B, C, and the discretization step Δ input-dependent: they are computed from the current token's hidden state via linear projections. This lets the model decide, for each token, how much to update the state (large Δ = big update, small Δ = ignore). This selectivity is critical for language tasks where some tokens carry information and others don't.

    **Follow-up:** "How does Mamba achieve parallelism during training despite the recurrent structure?"

    **Red flag:** Cannot explain what "selective" means or why fixed SSMs are limited.

!!! interview "39. (Advanced) What are hybrid architectures like Jamba?"
    **Expected depth:** Jamba interleaves Transformer attention layers with Mamba SSM layers. The idea is that attention excels at precise, content-based retrieval (looking up specific tokens), while SSMs excel at compressing sequential patterns efficiently. By alternating them, you get Transformer-quality reasoning with SSM-efficiency for long contexts. Jamba-1.5 also incorporates MoE. The ratio of attention to SSM layers is a key design choice — more attention gives better quality, more SSM gives better throughput.

    **Follow-up:** "Where in the model (early vs late layers) should attention be placed vs SSMs?"

    **Red flag:** Cannot explain the complementary strengths of attention and SSMs.

---

## System Design (15 Questions)

!!! interview "40. (System Design) Design an LLM serving system for a product with 10,000 concurrent users."
    **Expected depth:** Key components: (1) Model hosting — GPU cluster with enough VRAM for model weights + KV cache. (2) Batching — continuous batching (not static) to maximize GPU utilization. (3) Load balancing — distribute requests across replicas. (4) KV cache management — PagedAttention (vLLM) to avoid memory fragmentation. (5) Quantization — INT8 or INT4 to fit larger models on fewer GPUs. (6) Streaming — server-sent events for token-by-token delivery. (7) Monitoring — TTFT (time to first token), TPS (tokens per second), p99 latency. Mention specific tools: vLLM, TGI, or TensorRT-LLM.

    **Follow-up:** "How would you handle a traffic spike that doubles concurrent users?"

    **Red flag:** Only mentions "put the model on a GPU" without discussing batching, caching, or quantization.

!!! interview "41. (System Design) Design a RAG pipeline for an enterprise knowledge base."
    **Expected depth:** (1) Ingestion: chunk documents (512-1024 tokens), generate embeddings with a bi-encoder (e.g., BGE, E5). (2) Vector store: Pinecone, Weaviate, or pgvector. (3) Retrieval: embed the user query, ANN search for top-k chunks, optionally re-rank with a cross-encoder. (4) Augmentation: inject retrieved chunks into the LLM prompt with clear formatting. (5) Generation: LLM generates answer grounded in retrieved context. (6) Evaluation: faithfulness (does the answer match the sources?), relevance (are the right chunks retrieved?), answer correctness. Address: chunk overlap, metadata filtering, hybrid search (BM25 + dense), citation extraction.

    **Follow-up:** "How do you handle documents that update frequently?"

    **Red flag:** Cannot describe the retrieval step or confuses RAG with fine-tuning.

!!! interview "42. (System Design) How would you reduce the inference cost of a 70B parameter model by 4x?"
    **Expected depth:** Multiple strategies, ordered by ease: (1) INT4 quantization (GPTQ, AWQ) — 4x memory reduction, ~5% quality loss. (2) GQA — if the model doesn't have it, distill into a GQA variant. (3) Speculative decoding — use a small draft model to propose tokens, large model verifies in parallel. (4) Continuous batching — amortize the cost of attention across many concurrent requests. (5) Prompt caching — for repeated system prompts. (6) Model distillation — train a 7B model to mimic the 70B. Mention that strategies stack: INT4 + speculative decoding + continuous batching can achieve >4x reduction.

    **Follow-up:** "What quality benchmarks would you use to verify the quantized model is acceptable?"

    **Red flag:** Only mentions quantization without discussing batching, speculative decoding, or distillation.

!!! interview "43. (System Design) Design a system to detect and prevent LLM hallucinations in a customer-facing product."
    **Expected depth:** (1) Output grounding: RAG to provide source material, instruct the model to cite sources. (2) Self-consistency: generate N answers, check for agreement. (3) Fact-checking pipeline: extract claims from the output, verify against a knowledge base or search engine. (4) Confidence estimation: token-level log probabilities — low-confidence spans are flagged. (5) Guardrails: Regex/LLM-based checks for factual consistency. (6) Human-in-the-loop: route low-confidence answers to human reviewers. (7) Fine-tuning: train the model to say "I don't know" when appropriate (calibration). Mention that no single technique eliminates hallucination entirely — defense in depth is required.

    **Follow-up:** "How do you measure hallucination rate in production?"

    **Red flag:** Only says "use RAG" without addressing verification or confidence estimation.

!!! interview "44. (System Design) Design the training infrastructure for a 7B parameter model from scratch."
    **Expected depth:** (1) Hardware: minimum 8-16 A100 80GB GPUs (or H100s). (2) Data: 1-2T tokens of cleaned, deduplicated web text + high-quality curated sources. (3) Distributed training: FSDP or DeepSpeed ZeRO-3 for sharding model/optimizer/gradient states across GPUs. (4) Training pipeline: tokenization → data loading with shuffling → forward/backward pass → gradient accumulation → optimizer step. (5) Hyperparameters: learning rate warmup + cosine decay, batch size ramp, BF16 mixed precision. (6) Checkpointing: save every N steps to handle preemption. (7) Monitoring: loss curves, gradient norms, learning rate schedule. (8) Estimated cost: ~$100K-500K depending on hardware and training length.

    **Follow-up:** "How do you decide between FSDP and DeepSpeed?"

    **Red flag:** Cannot estimate the number of GPUs needed or mention a sharding strategy.

!!! interview "45. (System Design) Design an evaluation pipeline for an LLM-based product."
    **Expected depth:** (1) Automated benchmarks: standard tasks (MMLU, HumanEval, GSM8K) for regression detection. (2) Task-specific metrics: BLEU/ROUGE for summarization, pass@k for code generation, F1 for extraction. (3) LLM-as-judge: use a stronger model to rate outputs on dimensions like helpfulness, accuracy, safety. (4) Human evaluation: Elo-style pairwise comparison (like Chatbot Arena). (5) Safety testing: red-teaming prompts, toxicity classifiers, refusal detection. (6) A/B testing: deploy to a subset of users, measure task completion rate. (7) Regression testing: maintain a golden dataset of prompt-response pairs.

    **Follow-up:** "How do you handle evaluation when there is no single correct answer?"

    **Red flag:** Only mentions perplexity as an evaluation metric.

!!! interview "46. (System Design) How would you implement a multi-turn conversation system with memory?"
    **Expected depth:** (1) Short-term memory: include recent conversation turns in the prompt (sliding window). (2) Context management: when the window fills up, summarize earlier turns using the LLM itself. (3) Long-term memory: store key facts in a database (user preferences, prior decisions), retrieve via RAG at each turn. (4) Session management: each conversation has a unique ID, turns are stored persistently. (5) System prompt: persistent instructions and user profile injected at the start. (6) Token budget: allocate tokens between system prompt, retrieved context, conversation history, and generation. Mention attention over long histories is expensive — summarization is essential.

    **Follow-up:** "How do you handle contradictions between early conversation context and recent messages?"

    **Red flag:** Only suggests "put the full conversation in the prompt" without addressing token limits.

!!! interview "47. (System Design) Design a code generation system using LLMs."
    **Expected depth:** (1) Model selection: code-specific model (CodeLlama, StarCoder, DeepSeek-Coder) or general model (GPT-4, Claude). (2) Context: provide relevant files, function signatures, docstrings, and test cases in the prompt. (3) Retrieval: use code search to find relevant snippets from the codebase (embedding-based or keyword). (4) Generation: generate code with explicit instruction about language, style, and constraints. (5) Validation: run syntax check, type check, unit tests automatically. (6) Iterative refinement: if tests fail, feed error messages back to the model for correction. (7) Security: sandbox execution, scan for vulnerabilities, never execute untrusted code outside sandbox.

    **Follow-up:** "How would you handle a monorepo with 10M lines of code?"

    **Red flag:** Does not mention execution-based validation (just returns generated code as-is).

!!! interview "48. (System Design) Design a system for fine-tuning LLMs on private enterprise data."
    **Expected depth:** (1) Data pipeline: clean, deduplicate, format into instruction-response pairs. (2) Privacy: data never leaves the VPC, use differential privacy if needed, audit logs. (3) Method selection: full fine-tuning (expensive, best quality), LoRA/QLoRA (parameter-efficient, 10x cheaper), or prompt tuning (cheapest). (4) Infrastructure: managed service (Azure OpenAI, Vertex AI) or self-hosted (Axolotl, TRL). (5) Evaluation: hold-out test set, domain-specific metrics, A/B test against base model. (6) Versioning: track model versions, dataset versions, hyperparameters (MLflow, W&B). (7) Deployment: swap models with zero downtime using blue-green deployment.

    **Follow-up:** "How do you prevent catastrophic forgetting when fine-tuning on domain-specific data?"

    **Red flag:** Cannot distinguish between full fine-tuning, LoRA, and prompt tuning.

!!! interview "49. (System Design) Design a real-time content moderation system using LLMs."
    **Expected depth:** (1) Fast classifier: small model (DistilBERT) or ML classifier as first filter — reject obvious violations in <50ms. (2) LLM review: ambiguous content goes to an LLM for nuanced judgment (with policy guidelines in the prompt). (3) Categories: hate speech, violence, self-harm, sexual content, misinformation, PII exposure. (4) Latency: the fast path (classifier) handles 95% of traffic; the LLM path handles edge cases asynchronously. (5) Feedback loop: human reviewers correct errors, retraining the classifier periodically. (6) Multi-language: embeddings-based approach for language-agnostic moderation. (7) Metrics: precision (don't over-flag), recall (don't miss harmful content), latency p99.

    **Follow-up:** "How do you handle context-dependent content (sarcasm, medical discussions)?"

    **Red flag:** Proposes only regex or keyword matching as the moderation approach.

!!! interview "50. (System Design) Design an LLM-based agent that can use tools."
    **Expected depth:** (1) Architecture: ReAct loop — the LLM generates a thought, selects a tool (from a defined set), provides arguments, receives the tool output, then continues reasoning. (2) Tool definitions: each tool has a name, description, parameter schema (JSON schema). (3) Tool selection: the LLM outputs a structured function call; parse and validate against the schema. (4) Execution: sandbox tool execution, timeout, retry logic. (5) Memory: maintain a scratchpad of observations and intermediate results. (6) Safety: whitelist allowed tools, validate arguments, rate-limit API calls. (7) Evaluation: task completion rate, tool selection accuracy, number of tool calls (efficiency).

    **Follow-up:** "How do you handle tool calls that return errors?"

    **Red flag:** Cannot explain the observe-think-act loop.

!!! interview "51. (System Design) How would you implement streaming inference for an LLM API?"
    **Expected depth:** (1) Server-Sent Events (SSE) or WebSocket connection. (2) The model generates tokens one at a time; each token is immediately sent to the client. (3) Buffering: optionally buffer until a complete word or sentence for better UX. (4) Backpressure: if the client is slow, buffer server-side with a limit, then drop or pause. (5) Cancellation: client can close the connection, server detects and stops generation (saves compute). (6) Load balancing: sticky sessions or request routing to ensure the same GPU handles the full generation. (7) Metrics: time to first token (TTFT), inter-token latency (ITL), total generation time.

    **Follow-up:** "What are the networking implications of thousands of concurrent SSE connections?"

    **Red flag:** Describes a simple request-response API without understanding why streaming exists.

!!! interview "52. (System Design) Design a prompt management and versioning system."
    **Expected depth:** (1) Prompt registry: central store of all prompts with versioning (like a model registry). (2) Templating: prompts use variables that are filled at runtime (Jinja-style). (3) A/B testing: route traffic between prompt versions, measure output quality metrics. (4) Evaluation: each prompt version is tested against a golden dataset before production rollout. (5) Rollback: one-click revert to a previous prompt version. (6) Audit trail: who changed what, when, and why. (7) Environment promotion: dev → staging → production with gates. Tools: LangSmith, Promptflow, or custom.

    **Follow-up:** "How do you version prompts that include few-shot examples?"

    **Red flag:** Treats prompts as hardcoded strings without versioning or testing.

!!! interview "53. (System Design) Design a multi-model routing system that selects the best LLM for each query."
    **Expected depth:** (1) Router: a small classifier that categorizes the query (simple factual, complex reasoning, code, creative writing). (2) Model pool: GPT-4 for complex reasoning, Claude for long context, Mixtral for cost-sensitive queries, CodeLlama for code. (3) Routing logic: confidence-based — if the small model is confident, use it; otherwise escalate. (4) Cost optimization: route 80% of queries to cheap models, 20% to expensive ones. (5) Fallback chain: if the primary model fails (timeout, error), route to a backup. (6) Quality monitoring: track output quality per route, adjust routing rules dynamically. (7) Latency SLA: route to the fastest model that meets quality requirements.

    **Follow-up:** "How do you train the router without labeled data?"

    **Red flag:** Proposes using only one model or cannot articulate cost-quality trade-offs.

!!! interview "54. (System Design) How would you build a document Q&A system that handles 10,000 PDFs?"
    **Expected depth:** (1) Ingestion: extract text from PDFs (PyMuPDF, Unstructured), handle tables/images with OCR or multimodal models. (2) Chunking: recursive character splitting at 512-token chunks with 50-token overlap, preserve section boundaries. (3) Embeddings: batch-embed all chunks with a bi-encoder (BGE, E5), store in a vector database with metadata (filename, page number, section). (4) Retrieval: embed query, ANN top-20, re-rank with cross-encoder to top-5. (5) Generation: inject top-5 chunks with citations into LLM prompt. (6) Scaling: partition vector index by document collection, use approximate search (HNSW). (7) Updates: incremental indexing for new documents, deletion propagation.

    **Follow-up:** "How do you handle queries that span information across multiple documents?"

    **Red flag:** Suggests putting all PDFs in the context window.

---

## Research Awareness (10 Questions)

!!! interview "55. (Research) Explain the Chinchilla scaling laws and their impact."
    **Expected depth:** Chinchilla (Hoffmann et al., 2022) found that most LLMs were undertrained — for a given compute budget, optimal performance comes from scaling data and model size roughly equally. The compute-optimal ratio is approximately 20 tokens per parameter. GPT-3 (175B params, 300B tokens) was over-parameterized; the compute-optimal model at the same budget would be ~70B params trained on 1.4T tokens. This finding shifted the field toward training smaller models on more data (LLaMA-1 trained 7B on 1T tokens).

    **Follow-up:** "How do the Chinchilla laws change when you account for inference cost?"

    **Red flag:** Cannot state the approximate tokens-per-parameter ratio.

!!! interview "56. (Research) Explain RLHF and how it aligns LLMs with human preferences."
    **Expected depth:** Three stages: (1) Supervised fine-tuning (SFT): train on human-written examples of good behavior. (2) Reward model training: collect pairwise preferences (human ranks two model outputs), train a reward model to predict which output a human prefers. (3) RL optimization: use PPO to update the LM policy to maximize the reward model's score, with a KL penalty to prevent the policy from diverging too far from the SFT model. The KL penalty is critical — without it, the model exploits reward model weaknesses (reward hacking).

    **Follow-up:** "What is DPO and why is it simpler than RLHF?"

    **Red flag:** Cannot name the three stages or explain the reward model.

!!! interview "57. (Research) What is DPO and how does it differ from PPO-based RLHF?"
    **Expected depth:** Direct Preference Optimization (Rafailov et al., 2023) eliminates the reward model entirely. It reparameterizes the RLHF objective to derive a closed-form loss that directly optimizes the policy on preference data: the preferred response should have higher log-probability (relative to a reference policy) than the dispreferred response. DPO is simpler (no RL loop, no reward model, no sampling during training), more stable, and computationally cheaper. The trade-off: DPO is less flexible than PPO for complex reward signals and can be sensitive to the reference policy.

    **Follow-up:** "When would you choose PPO over DPO?"

    **Red flag:** Cannot explain how DPO avoids the reward model.

!!! interview "58. (Research) Explain LoRA and why it works."
    **Expected depth:** Low-Rank Adaptation freezes the pre-trained weights and injects trainable low-rank decomposition matrices into each layer: instead of updating \(W\), learn \(\Delta W = BA\) where \(B \in \mathbb{R}^{d \times r}\) and \(A \in \mathbb{R}^{r \times d}\) with rank \(r \ll d\). The intuition is that weight updates during fine-tuning are low-rank — most of the model's knowledge is preserved, and task-specific adaptation can be captured by a small subspace. Benefits: 10-100x fewer trainable parameters, multiple LoRA adapters can be swapped at inference time, merged back into the base weights with zero inference overhead.

    **Follow-up:** "What rank do you typically use? How do you choose?"

    **Red flag:** Cannot explain the low-rank decomposition or why it works.

!!! interview "59. (Research) What are reasoning models (o1, DeepSeek-R1) and how do they work?"
    **Expected depth:** Reasoning models use chain-of-thought (CoT) prompting or training to decompose complex problems into intermediate steps. OpenAI's o1 uses test-time compute scaling — it generates multiple reasoning chains and selects the best one (or uses verification). DeepSeek-R1 trains on process-level reward models that evaluate each reasoning step, not just the final answer. The key insight is that more compute at inference time (longer thinking, more attempts) can substitute for more compute at training time. These models excel at math, coding, and multi-step logical reasoning.

    **Follow-up:** "What is the trade-off between training-time and test-time compute?"

    **Red flag:** Thinks chain-of-thought is just a prompt trick, not a fundamental capability enhancement.

!!! interview "60. (Research) Explain speculative decoding."
    **Expected depth:** Speculative decoding uses a small, fast draft model to propose \(k\) tokens, then the large target model verifies all \(k\) tokens in a single forward pass (which is almost as fast as verifying 1 token due to parallelism). If the draft model's tokens match what the target model would have generated (using rejection sampling), they are accepted. If they diverge, the target model's token is used and the draft restarts. The speedup comes from the draft model running much faster (7B draft for a 70B target). Acceptance rate depends on how well the draft model matches the target.

    **Follow-up:** "What is the expected speedup for a draft model with 70% acceptance rate?"

    **Red flag:** Cannot explain the verify-then-accept mechanism.

!!! interview "61. (Research) What is Mixture of Experts and how does Mixtral use it?"
    **Expected depth:** Mixtral 8x7B has 8 expert FFN modules per layer with a router that selects the top-2 for each token. Total parameters: ~47B. Active parameters per token: ~13B. This gives near-70B quality at 13B inference cost. The router is a learned linear layer. Load balancing uses an auxiliary loss. The architecture is otherwise a standard Transformer with GQA. Training challenges include expert collapse and communication overhead for distributed training across GPUs.

    **Follow-up:** "How do you shard MoE models across GPUs?"

    **Red flag:** Cannot explain the total vs active parameter distinction.

!!! interview "62. (Research) Explain the attention mechanism in the original 'Attention Is All You Need' paper."
    **Expected depth:** The paper introduced scaled dot-product attention with multi-head structure. Key innovations: (1) Self-attention replaces recurrence entirely. (2) Multi-head attention runs \(h\) parallel attention functions on different \(d_k\)-dimensional projections. (3) Scaling by \(\sqrt{d_k}\) prevents softmax saturation. (4) Positional encoding (sinusoidal) provides sequence order. (5) The encoder-decoder architecture with cross-attention. (6) The training used label smoothing, learning rate warmup, and dropout. Know the original hyperparameters: \(d_{\text{model}}=512\), \(h=8\), \(d_k=64\), \(d_{ff}=2048\), 6 layers.

    **Follow-up:** "What were the specific training tricks that made this work?"

    **Red flag:** Cannot describe multi-head attention or the scaling factor.

!!! interview "63. (Research) What are the key ideas in the LLaMA family of models?"
    **Expected depth:** LLaMA-1 (Meta, 2023): showed that smaller models trained on more data (Chinchilla-optimal) outperform larger undertrained models. Used RoPE, RMSNorm, SwiGLU, Pre-Norm. LLaMA-2: added GQA for efficient inference, extended context to 4096, and released chat models fine-tuned with RLHF. LLaMA-3: scaled to 405B parameters, 128K context, 15T training tokens, multilingual. Key architectural choices: no bias terms, RMSNorm instead of LayerNorm, SwiGLU instead of ReLU, RoPE instead of learned PE. The LLaMA family established the open-weight foundation model standard.

    **Follow-up:** "How does LLaMA's tokenizer differ from GPT's?"

    **Red flag:** Cannot name at least 3 specific architectural choices.

!!! interview "64. (Research) What is test-time compute scaling?"
    **Expected depth:** The idea that model performance can be improved by spending more compute at inference time rather than training time. Methods: (1) Best-of-N sampling — generate N answers, select the best using a verifier or reward model. (2) Chain-of-thought — let the model reason step by step, using more tokens for harder problems. (3) Tree search — explore multiple reasoning paths (like Monte Carlo tree search). (4) Self-consistency — generate diverse reasoning chains and majority-vote. OpenAI's o1 and o3 demonstrated dramatic improvements from test-time compute, especially on math and coding benchmarks. The trade-off: latency and cost increase linearly with compute spent.

    **Follow-up:** "When is it more efficient to scale test-time compute vs training compute?"

    **Red flag:** Only mentions "generate longer outputs" without the verification/selection component.

---

## Quick Reference: Concepts to Know Cold

| Concept | One-Sentence Summary |
|---------|---------------------|
| Chain rule decomposition | Joint probability factors into conditional probabilities: \(P(w_1 \ldots w_T) = \prod P(w_t \mid w_{<t})\) |
| Perplexity | Exponential of average cross-entropy — the effective vocabulary size the model chooses from |
| Attention | Weighted sum of values where weights come from query-key similarity |
| Causal masking | Upper-triangle mask preventing tokens from attending to future positions |
| KV cache | Reuse of previously computed key-value pairs during autoregressive generation |
| RoPE | Rotary positional encoding — encodes position by rotating Q and K vectors |
| GQA | Grouped-query attention — multiple query heads share K/V heads to reduce cache size |
| SwiGLU | Gated activation function in FFN — better than ReLU per FLOP |
| RMSNorm | Normalization by root mean square — faster than LayerNorm, no mean subtraction |
| MoE | Mixture of Experts — sparse activation, only k of N experts process each token |
| LoRA | Low-rank adaptation — freeze weights, train small rank-r matrices |
| RLHF | Reinforcement learning from human feedback — SFT → reward model → PPO |
| DPO | Direct preference optimization — eliminates reward model, directly optimizes on preferences |
| RAG | Retrieval-augmented generation — retrieve relevant documents and inject into prompt |
| Speculative decoding | Draft model proposes tokens, target model verifies in parallel |

---

---

## Cross-Paper Interview Scenarios

For advanced interview preparation, see the [Cross-Paper Interview Scenarios](06_research_papers/index.md#cross-paper-interview-scenarios) section in the Research Papers guide. These scenarios test your ability to connect ideas across multiple papers — a hallmark of strong candidates.

Topics covered:

1. **MoE Serving Under Load** — memory vs. compute trade-offs at scale ([Mixtral](06_research_papers/16_mixtral.md), [DeepSeek-V3](06_research_papers/27_deepseek_v3.md), [GLM-5](06_research_papers/32_glm5.md))
2. **Async RL for Tool-Heavy Tasks** — keeping GPUs utilized during slow rollouts ([DeepSeek-R1](06_research_papers/28_deepseek_r1.md), [GLM-5](06_research_papers/32_glm5.md))
3. **KV Cache Efficiency Stack** — composing MLA + FlashAttention + continuous batching ([DeepSeek-V2](06_research_papers/26_deepseek_v2.md), [FlashAttention](06_research_papers/14_flash_attention.md))
4. **Single Agent vs. Multi-Agent** — when to use a swarm vs. a single powerful agent ([GLM-5](06_research_papers/32_glm5.md), [Kimi K2.5](06_research_papers/33_kimi_k2_5.md), [ReAct](06_research_papers/19_react.md))
5. **Distillation vs. On-Student RL** — when to copy teacher traces vs. train from scratch ([DeepSeek-R1](06_research_papers/28_deepseek_r1.md), [Chinchilla](06_research_papers/11_chinchilla.md))
6. **Data Quality as a Scaling Multiplier** — why 72B can match 405B ([Qwen2.5](06_research_papers/31_qwen2_5.md), [Chinchilla](06_research_papers/11_chinchilla.md))
7. **Evaluation Hygiene** — ensuring trustworthy benchmark scores ([Qwen2.5](06_research_papers/31_qwen2_5.md), [GLM-5](06_research_papers/32_glm5.md))

---

*This question bank is updated as new topics are covered in LLMBase. See individual topic pages for in-depth explanations and worked examples.*
