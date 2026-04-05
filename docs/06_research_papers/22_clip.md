# CLIP: Learning Transferable Visual Models From Natural Language Supervision

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, and 9 more  
**Year:** 2021 &nbsp;|&nbsp; **Venue:** ICML  
**Link:** [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

---

## TL;DR

CLIP trains **dual encoders** — one for images, one for text — on **400M (image, text) pairs** from the web using **contrastive learning**. Matching pairs are pulled together in embedding space while non-matching pairs are pushed apart. The resulting image encoder can classify images **zero-shot** using natural language descriptions ("a photo of a cat") without any task-specific training data.

---

## Why This Paper Matters

CLIP is the foundational multimodal model:

1. **Zero-shot image classification:** No labeled dataset needed — just describe categories in text
2. **Multimodal embeddings:** Power image retrieval, image-text search, and VLMs
3. **Diffusion model backbone:** CLIP text encoder is used in Stable Diffusion and DALL-E
4. **Image RAG:** CLIP embeddings enable searching images with text queries
5. **Robustness:** Much more robust to distribution shift than supervised ImageNet models

---

## Key Concepts Explained Simply

### Contrastive Learning

Given a batch of \(N\) (image, text) pairs:
- **Positive pairs:** The image and its matching text → pull embeddings close together
- **Negative pairs:** The image with non-matching text (all other texts in the batch) → push embeddings apart

With batch size 32,768, each image has 1 positive and 32,767 negatives.

### Dual Encoder Architecture

Two separate encoders that map to the **same** embedding space:
- **Image encoder:** ViT (Vision Transformer) or ResNet → image embedding
- **Text encoder:** Transformer → text embedding

Both produce vectors of the same dimension (e.g., 512). Similarity is measured by **cosine similarity** or dot product.

### Zero-Shot Classification

To classify an image:
1. Create text prompts for each class: "a photo of a dog", "a photo of a cat", "a photo of a car"
2. Encode the image and all text prompts
3. Compute cosine similarity between the image embedding and each text embedding
4. The class with highest similarity wins

No training on labeled examples needed.

---

## The Math — Explained Step by Step

### InfoNCE Loss

For a batch of \(N\) image-text pairs with normalized embeddings \(g_i\) (image) and \(t_j\) (text):

\[
\mathcal{L}_i^{\text{image}} = -\log \frac{\exp(g_i^\top t_i / \tau)}{\sum_{j=1}^{N} \exp(g_i^\top t_j / \tau)}
\]

\[
\mathcal{L}_i^{\text{text}} = -\log \frac{\exp(t_i^\top g_i / \tau)}{\sum_{j=1}^{N} \exp(t_i^\top g_j / \tau)}
\]

**Breaking it down:**

1. **Numerator:** Similarity between the matching pair \((g_i, t_i)\) — should be high
2. **Denominator:** Sum over all pairs in the batch (including positives) — normalizes into a probability
3. **\(\tau\):** Temperature parameter (learned, ~0.07) — controls how sharp the distribution is
4. **Symmetric:** Loss is computed both ways (image→text and text→image)
5. **Total loss:** \(\mathcal{L} = \frac{1}{2N}\sum_i (\mathcal{L}_i^{\text{image}} + \mathcal{L}_i^{\text{text}})\)

### Why Large Batch Sizes Matter

With batch size \(N\):
- Each sample has \(N-1\) negatives
- Larger batches → harder negatives → better representations
- CLIP uses batch size 32,768 — each sample competes with 32,767 negatives
- Small batch sizes lead to easy negatives and weaker representations

### Temperature

\(\tau\) controls discrimination sharpness:
- **Small \(\tau\) (0.01):** Very peaked — model must be very confident about the correct match
- **Large \(\tau\) (1.0):** Flatter — less discriminating
- CLIP learns \(\tau\) during training, typically converging to ~0.07

---

## Python Implementation

```python
import numpy as np


def l2_normalize(x):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-8)


def clip_loss(image_embeddings, text_embeddings, temperature=0.07):
    """
    Symmetric InfoNCE contrastive loss.
    image_embeddings: [batch_size, embed_dim]
    text_embeddings: [batch_size, embed_dim]
    """
    # Normalize
    image_norm = l2_normalize(image_embeddings)
    text_norm = l2_normalize(text_embeddings)

    # Cosine similarity matrix [batch, batch]
    logits = (image_norm @ text_norm.T) / temperature
    batch_size = len(image_embeddings)
    labels = np.arange(batch_size)

    # Image-to-text loss
    log_probs_i2t = logits - np.max(logits, axis=1, keepdims=True)
    log_probs_i2t = log_probs_i2t - np.log(
        np.sum(np.exp(log_probs_i2t), axis=1, keepdims=True)
    )
    loss_i2t = -np.mean(log_probs_i2t[np.arange(batch_size), labels])

    # Text-to-image loss
    log_probs_t2i = logits.T - np.max(logits.T, axis=1, keepdims=True)
    log_probs_t2i = log_probs_t2i - np.log(
        np.sum(np.exp(log_probs_t2i), axis=1, keepdims=True)
    )
    loss_t2i = -np.mean(log_probs_t2i[np.arange(batch_size), labels])

    return (loss_i2t + loss_t2i) / 2


def zero_shot_classify(image_embedding, class_text_embeddings, class_names,
                       temperature=0.07):
    """
    Zero-shot classification using CLIP.
    image_embedding: [embed_dim]
    class_text_embeddings: [n_classes, embed_dim]
    """
    image_norm = l2_normalize(image_embedding.reshape(1, -1))
    text_norm = l2_normalize(class_text_embeddings)

    similarities = (image_norm @ text_norm.T).flatten() / temperature

    # Softmax for probabilities
    exp_sim = np.exp(similarities - np.max(similarities))
    probs = exp_sim / np.sum(exp_sim)

    top_idx = np.argmax(probs)
    return class_names[top_idx], probs


def prompt_engineering(class_names, templates=None):
    """
    Generate text prompts for each class using templates.
    Ensembling multiple templates improves accuracy.
    """
    if templates is None:
        templates = [
            "a photo of a {}.",
            "a photo of the {}.",
            "an image of a {}.",
            "a picture of a {}.",
            "a photo of a {}, a type of object.",
        ]

    prompts = {}
    for cls in class_names:
        prompts[cls] = [t.format(cls) for t in templates]
    return prompts


def image_text_retrieval(image_embeddings, text_embeddings,
                         query_type="image", query_idx=0, top_k=5):
    """
    Retrieve top-k matches across modalities.
    """
    image_norm = l2_normalize(image_embeddings)
    text_norm = l2_normalize(text_embeddings)

    if query_type == "image":
        query = image_norm[query_idx:query_idx+1]
        similarities = (query @ text_norm.T).flatten()
    else:
        query = text_norm[query_idx:query_idx+1]
        similarities = (query @ image_norm.T).flatten()

    top_indices = np.argsort(-similarities)[:top_k]
    top_scores = similarities[top_indices]
    return list(zip(top_indices, top_scores))


def batch_size_analysis():
    """Show how batch size affects contrastive learning quality."""
    print("--- Batch Size Effect on Contrastive Learning ---")
    print(f"{'Batch Size':>12} {'Negatives':>12} {'GPU Memory':>12}")
    print("-" * 40)
    for bs in [32, 128, 512, 2048, 8192, 32768]:
        negatives = bs - 1
        # Rough memory estimate for similarity matrix (float32)
        mem_mb = bs * bs * 4 / 1e6
        print(f"{bs:>12,} {negatives:>12,} {mem_mb:>10.1f}MB")


# --- Demo ---
if __name__ == "__main__":
    np.random.seed(42)
    embed_dim = 64
    batch_size = 8

    # Contrastive loss
    image_emb = np.random.randn(batch_size, embed_dim)
    text_emb = np.random.randn(batch_size, embed_dim)

    loss = clip_loss(image_emb, text_emb)
    print(f"CLIP loss (random embeddings): {loss:.4f}")
    print(f"Expected for random (log(N)): {np.log(batch_size):.4f}")

    # Zero-shot classification
    print("\n--- Zero-Shot Classification ---")
    class_names = ["cat", "dog", "car", "airplane", "bird"]
    n_classes = len(class_names)

    image_emb_single = np.random.randn(embed_dim)
    class_embs = np.random.randn(n_classes, embed_dim)
    # Make "cat" embedding similar to the image
    class_embs[0] = image_emb_single + np.random.randn(embed_dim) * 0.3

    predicted, probs = zero_shot_classify(
        image_emb_single, class_embs, class_names
    )
    print(f"Predicted: {predicted}")
    for name, prob in zip(class_names, probs):
        bar = "█" * int(prob * 40)
        print(f"  {name:>10}: {prob:.1%} {bar}")

    # Prompt engineering
    print("\n--- Prompt Engineering ---")
    prompts = prompt_engineering(["cat", "dog"])
    for cls, templates in prompts.items():
        print(f"  {cls}:")
        for t in templates:
            print(f"    - {t}")

    # Batch size analysis
    print()
    batch_size_analysis()

    # Retrieval demo
    print("\n--- Image-Text Retrieval ---")
    n_images, n_texts = 10, 10
    image_embs = np.random.randn(n_images, embed_dim)
    text_embs = np.random.randn(n_texts, embed_dim)
    results = image_text_retrieval(image_embs, text_embs, "image", 0, top_k=3)
    print(f"Top-3 texts for image 0: {results}")
```

---

## Interview Importance

CLIP is **essential** for multimodal AI roles and increasingly important for general LLM positions as models become multimodal.

### Difficulty Level: ⭐⭐⭐ (Medium)

---

## Interview Questions & Answers

### Q1: Why do in-batch negatives work, and what breaks at small batch size?

**Answer:** In-batch negatives are "free" — you get \(N-1\) negatives without extra computation. At large batch sizes, the batch likely contains hard negatives (similar but non-matching pairs), which forces the model to learn fine-grained distinctions.

At small batch sizes (e.g., 32): negatives are mostly easy (a "dog" image vs. "quantum physics" text), so the model doesn't learn to distinguish similar concepts. The loss becomes trivially low without learning useful representations.

### Q2: How does CLIP enable zero-shot classification via prompts?

**Answer:** CLIP maps images and text into a shared embedding space where matching pairs are close. To classify:
1. Create text descriptions for each class: "a photo of a [class]"
2. Encode each description with the text encoder
3. Encode the image with the image encoder
4. Compare via cosine similarity — the most similar text = predicted class

This works because CLIP learned from 400M image-text pairs that images of dogs are similar to text about dogs, etc.

### Q3: Name failure modes of CLIP.

**Answer:**
1. **Texture bias:** CLIP sometimes relies on texture over shape (inherited from training data)
2. **OCR gaps:** Struggles with text in images despite seeing web data with text
3. **Fine-grained categories:** Distinguishing dog breeds or bird species is hard without specialized training
4. **Abstract concepts:** Difficulty with abstract or subjective descriptions
5. **Distribution shift:** While more robust than supervised models, still degrades on out-of-distribution data
6. **Counting:** Poor at "a photo of three cats" vs "a photo of two cats"
7. **Spatial relationships:** Poor at "cat on top of dog" vs "dog on top of cat"

### Q4: How is CLIP used in diffusion models like Stable Diffusion?

**Answer:** CLIP's text encoder converts text prompts into embeddings that **condition** the diffusion model's denoising process. The image encoder is used for image-to-image similarity and CLIP-guided generation. Specifically, the CLIP text embedding is injected via cross-attention into the U-Net at each denoising step, steering the generated image to match the text description.

---

## Connections to Other Papers

- **Transformer** → Both CLIP encoders are Transformers (ViT + text Transformer)
- **GPT-2** → CLIP's text encoder follows GPT-2 architecture
- **Gemini** → Native multimodal models extend CLIP's vision-language alignment
- **Codex** → CLIP for images; Codex for code — both transfer via pre-training
- **InstructGPT** → Multimodal RLHF builds on CLIP embeddings

---

## Key Takeaways for Quick Review

| Concept | Remember |
|---|---|
| Architecture | Dual encoder (image ViT + text Transformer) |
| Training | Contrastive learning on 400M image-text pairs |
| Loss | Symmetric InfoNCE with learned temperature |
| Key ability | Zero-shot image classification via text prompts |
| Batch size | 32,768 — large batches critical for quality |
| Temperature | ~0.07 (learned) |
| Used in | Stable Diffusion, image retrieval, VLMs |
| Limitation | Texture bias, poor counting/spatial reasoning |
