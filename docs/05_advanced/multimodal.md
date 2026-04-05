# Multimodal LLMs

## Why This Matters for LLMs

Language is not the only modality in production: assistants must read **screenshots**, **charts**, **PDFs**, and **camera frames**. **Multimodal LLMs** connect **vision encoders** (or audio encoders) to **decoder-only LMs** via projection layers, cross-attention, or interleaved tokenization. Understanding CLIP-style alignment, ViT backbones, and LLaVA-class architectures is essential for interview discussions about GPT-4V-class systems and open models like LLaVA, Qwen-VL, and Gemini.

Second, **cross-modal alignment** is a representation-learning problem: image patches and text tokens must live in a **shared space** where cosine similarity reflects semantic correspondence. Contrastive objectives (CLIP) and generative objectives (captioning) produce different inductive biases—engineers need vocabulary for **zero-shot transfer**, **instruction tuning**, and **hallucinated objects** in vision answers.

Third, **throughput and memory** multiply: vision transformers produce hundreds of patch tokens per image; feeding them into a 70B LM dominates latency. **Resolution trade-offs**, **adaptive cropping**, and **connector** design (MLP projector vs Q-Former) are systems questions as much as modeling questions.

---

## Core Concepts

### CLIP: Contrastive Language–Image Pre-training

**CLIP** trains dual encoders \(f_I\) (image) and \(f_T\) (text) on **image–text pairs** \((x_i, t_i)\) from the web. For a batch of \(N\) pairs, similarity is measured with normalized embeddings:

\[
\hat{f}_I = \frac{f_I(x)}{\|f_I(x)\|},\quad \hat{f}_T = \frac{f_T(t)}{\|f_T(t)\|}
\]

The **symmetric InfoNCE** loss for image-to-text and text-to-image uses temperature \(\tau\):

\[
\mathcal{L} = - \frac{1}{2N} \sum_{i=1}^{N} \left[
\log \frac{e^{\langle \hat{f}_I(x_i), \hat{f}_T(t_i)\rangle / \tau}}{\sum_{j=1}^{N} e^{\langle \hat{f}_I(x_i), \hat{f}_T(t_j)\rangle / \tau}}
+
\log \frac{e^{\langle \hat{f}_T(t_i), \hat{f}_I(x_i)\rangle / \tau}}{\sum_{j=1}^{N} e^{\langle \hat{f}_T(t_i), \hat{f}_I(x_j)\rangle / \tau}}
\right]
\]

Negatives are **in-batch**; large batch sizes improve contrastive learning but demand multi-GPU coordination.

!!! math-intuition "In Plain English"
    CLIP **does not** generate captions at pre-training—it only learns that matching image/text pairs sit nearby in embedding space. Generation comes later when you attach a decoder or prompt a LM with visual features.

### Vision Encoders: ViT

**Vision Transformer (ViT)** splits an image into fixed-size patches, linearly embeds each patch, adds position embeddings, and runs Transformer blocks. For \(H \times W\) pixels and patch size \(P\), there are \(N = HW/P^2\) patch tokens—often 196–576 for standard inputs.

Let patch embeddings be \(z_0 \in \mathbb{R}^{N \times d}\). After \(L\) blocks, pooled or token features feed downstream heads.

### SigLIP and Improved Contrastive Training

**SigLIP** (sigmoid loss for language–image pre-training) replaces softmax with **pairwise sigmoid** losses, improving stability and data efficiency in some regimes—useful when batch sizes cannot scale to CLIP levels.

### BLIP-2 and Q-Former Bottlenecks

**BLIP-2** freezes pretrained **image encoder** and **LLM**, and learns a lightweight **Q-Former** (Transformer with learnable query tokens) that **extracts** a fixed number of visual tokens via cross-attention to image features. The output is linearly projected into the LM embedding space. This **bottleneck** reduces visual tokens from hundreds of patches to 32–256 **latent queries**, cutting LM compute.

\[
Z = \text{QFormer}(Q, F_{\text{img}}),\quad H_{\text{in}} = W Z + b
\]

where \(Q\) are learnable queries and \(F_{\text{img}}\) are frozen image features.

!!! math-intuition "In Plain English"
    Q-Former is **interview summarization for pixels**: a small network picks the few visual facts the LM should see, instead of dumping every patch token into context.

### LLaVA Architecture: Vision Encoder + Projector + LLM

**LLaVA** freezes a pretrained **ViT** (often CLIP ViT-L/14), maps patch tokens through a **projection MLP** into the **word embedding space** of a frozen or LoRA-tuned LLM, and trains on **instruction-following** multimodal dialog data. The model sees a sequence:

\[
[\text{system}],\, [\text{USER}: \text{image tokens} + \text{text}],\, [\text{ASSISTANT}: \text{answer}]
\]

**Image tokens** are long—compression via **perceiver resampler** or **Q-Former** (BLIP-2) reduces tokens before the LM.

!!! math-intuition "In Plain English"
    The projector is a **Rosetta stone**: it turns CNN/ViT patch vectors into fake “words” the LM already knows how to reason about. The LM supplies **reasoning**; the encoder supplies **pixels**.

### Flamingo: Gated Cross-Attention and Perceiver Resamplers

**Flamingo** interleaves **gated cross-attention** layers in the LM stack so visual tokens attend into hidden states at selected layers. **Perceiver resampler** compresses variable-size image/video feature maps to a fixed token count. The design supports **few-shot** in-context examples with images by concatenating multimodal demonstrations.

### GPT-4V / Gemini-Style Systems

Closed multimodal systems typically combine:

- A **high-capacity vision encoder** (sometimes with **dynamic resolution** tiling).
- A **connector** (perceiver, MLP, or cross-attention bottleneck).
- A **large decoder-only LM** with interleaved tool use and safety filters.

**Interleaved** image/text sequences support multiple images and follow-up questions; **system prompts** constrain harmful vision outputs.

### Cross-Modal Alignment

Alignment objectives include:

- **Contrastive** (CLIP): global image–text matching.
- **Masked modeling** (some Flamingo-style setups): predict masked patches or tokens.
- **Captioning**: maximize \(P(\text{caption} \mid \text{image})\) with autoregressive decoding.

Evaluation uses **VQA** accuracy, **text-image retrieval** (Recall@k), **human preference** on helpfulness, and **red-team** probes for OCR misuse.

### Code: CLIP-Style Similarity (Synthetic)

The following uses **random** encoders for shape demonstration only—replace with `open_clip` or `transformers` in real work.

```python
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyCLIP(nn.Module):
    def __init__(self, image_dim: int = 512, text_dim: int = 512, embed_dim: int = 256):
        super().__init__()
        self.img_enc = nn.Sequential(
            nn.Linear(image_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.txt_enc = nn.Sequential(
            nn.Linear(text_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        z = self.img_enc(x)
        return F.normalize(z, dim=-1)

    def encode_text(self, t: torch.Tensor) -> torch.Tensor:
        z = self.txt_enc(t)
        return F.normalize(z, dim=-1)

    def clip_loss(self, image_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        logits = image_emb @ text_emb.transpose(0, 1) * self.logit_scale.exp()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.transpose(0, 1), labels)
        return (loss_i + loss_t) / 2


def similarity_rank(
    model: ToyCLIP, image: torch.Tensor, texts: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        ie = model.encode_image(image)
        te = model.encode_text(texts)
        scores = (ie @ te.transpose(0, 1)).squeeze(0)
        return scores, scores.argsort(descending=True)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ToyCLIP()
    img = torch.randn(1, 512)
    captions = torch.randn(4, 512)  # fake text features
    loss = model.clip_loss(model.encode_image(img.expand(4, 512)), model.encode_text(captions))
    print("symmetric clip loss:", float(loss))
    scores, order = similarity_rank(model, img, captions)
    print("scores:", scores.tolist())
    print("best caption index:", int(order[0]))
```

### Modalities Beyond Vision

**Audio** (Whisper-style encoders), **video** (frame sampling + temporal pooling), and **structured sensors** follow the same template: encode → align → fuse in LM context. **Token budget** management across modalities is a recurring design constraint.

### Failure Modes

- **Object hallucination**: LM invents objects not present—mitigate with **grounding** boxes or segmentation-assisted tools.
- **OCR errors**: small text misread—higher resolution or **cropped zoom** tools.
- **Prompt injection via images**: adversarial pixels—**policy layers** and **tool sandboxing**.

### Vision-Language Tasks (Evaluation Map)

| Task | Measures | Caveats |
|------|------------|---------|
| **Image–text retrieval** | Recall@k on Flickr30K/MSCOCO | Domain bias toward web imagery |
| **VQA** | Accuracy on open-ended or multiple-choice | **Shortcut** answers from language priors |
| **Text-rich VQA** (DocVQA) | Reading comprehension on documents | OCR errors propagate |
| **Chart QA** | Reasoning over plots | Requires **visual** + **numeric** reasoning |

Report **fine-grained** metrics (per category) rather than a single **accuracy** headline.

### Token Budgeting for Images

If \(N\) patch tokens are projected into the LM, each image consumes roughly \(N \cdot L_{\text{forward}}\) FLOPs in early layers—**reducing** \(N\) via pooling, Q-Former, or **dynamic resolution** (tile + select) is standard for low-latency chat.

### Contrastive vs Generative Pre-training

**CLIP** (contrastive) excels at **retrieval** and zero-shot classification; **generative** image-to-text (captioning) optimizes \(P(\text{text} \mid \text{image})\) directly. Multimodal LMs often **compose** both: contrastive encoder for alignment, autoregressive LM for **reasoning** and dialog.

### Licensing and Data Provenance

Web-scale image–text pairs carry **copyright** and **PII** risks. Production pipelines apply **filters**, **deduplication**, and **consent** policies—aligning with **Responsible AI** commitments.

### Patchification and Positional Structure

ViT treats an \(H \times W\) image as a sequence of \(P \times P\) patches. The number of tokens is:

\[
N_{\text{patches}} = \frac{H}{P} \cdot \frac{W}{P}
\]

**CLS** token or **global average pooling** yields an image embedding for CLIP-style losses. **2D** positional encodings (absolute or **relative**) help the encoder preserve **spatial** locality.

### Image Resolution vs Latency

Doubling **resolution** roughly **quadruples** patch count (for fixed \(P\)), increasing **Transformer** cost **superlinearly** in patches for standard attention across patches—**hierarchical** encoders and **adaptive** cropping mitigate this.

### Adapter Tuning and LoRA on Multimodal Stacks

Common **PEFT** patterns:

- **LoRA** on **LLM** layers only, projector **full** fine-tune (small).
- **Freeze** vision encoder initially; unfreeze **later** stages when data is ample.

This reduces **catastrophic forgetting** of **language** abilities while adapting **vision** grounding.

### GPT-4V-Style Tool Use

Multimodal assistants often chain **vision** with **tools** (code execution, web browsing). **Safety** policies must consider **screenshots** containing **secrets**—**redact** before logging and **warn** users.

### Interview Pitfall: “CLIP = VLM”

**CLIP** alone does **not** perform **free-form** multimodal chat—it produces **embeddings**. **VLMs** add **generative** decoding with an **autoregressive** LM head or **fusion** layers.

### SigLIP Sigmoid Loss (Sketch)

Instead of a **softmax** over the batch for each image, **SigLIP** uses **sigmoid** losses on **pairs** \((i,j)\), improving stability when **batch** sizes are **small**—relevant for **edge** devices and **fine-tuning** runs.

### OCR-Heavy Pipelines

For **screenshots** and **PDFs**, add a dedicated **text detection** stage (e.g., **detector** + **recognizer**) before **VLM** reasoning—**end-to-end** pixel-only models may **miss** small fonts.

### Video as Many Images

A simple **video** baseline **samples** frames at **fps** \(f\) for **duration** \(D\): **frame count** \(\approx f \cdot D\). **Long** videos exceed **context** budgets—use **event detection**, **keyframes**, or **learned** frame selectors.

### Evaluation Pitfalls: Shortcut Learning

VLMs sometimes answer **VQA** using **language priors** alone (“What color are bananas?”). **Balanced** datasets and **adversarial** image edits reduce **shortcut** exploitation.

### Future-Proofing Connectors

When upgrading **vision** backbones, **retrain** **projectors**; **frozen** LLMs with **new** encoders may need **alignment** stages to restore **quality**.

### CLIP Zero-Shot Classifier

Given **class** names \(c_k\) embedded as text \(t_k\) and image \(x\), CLIP predicts:

\[
k^\star = \arg\max_k \langle f_I(x), f_T(t_k)\rangle
\]

**Prompt engineering** (“a photo of a {label}”) materially affects **zero-shot** accuracy.

### Normalization and Temperature

CLIP uses **L2** normalization so **cosine** similarity equals **dot** product. **Temperature** \(\tau\) in contrastive loss sharpens or softens the **softmax**—analogous to **logit scaling** at inference for **calibration**.

### Data Mixing for Multimodal Fine-Tunes

**Instruction** data often mixes **pure text** and **image-text** samples—ratios affect **forgetting** of **language** skills. Track **per-modality** perplexity during training.

### Accessibility and Alt Text

**Screen reader** users rely on **alt text**—multimodal systems should not **replace** accessibility pipelines; **generate** descriptions only when **appropriate** and **safe**.

### Gemini / GPT-4V: System-Level Observations

Closed multimodal APIs often combine **vision** with **tool use** and **safety** classifiers in **one** stack—**latency** budgets include **multiple** model calls. **Engineering** discussions should separate **model** capability from **system** orchestration.

### Open-Weight VLMs

Open releases (LLaVA, Qwen-VL, InternVL) differ in **connector** design, **resolution**, and **instruction** data—**compare** on **your** domain with **private** evals, not only **public** leaderboards.

### Multimodal Prompting Patterns

- **Single image + question**: baseline **chat** template.
- **Multi-image** reasoning: watch **token** budget—**compress** with **per-image** summaries first.
- **Charts**: ask for **extracted** numbers before **interpretation** to reduce **hallucination**.

### Risk: Visual Misinformation

Images can **fabricate** evidence (fake screenshots). **Policies** should discourage **high-stakes** decisions from **unverified** visuals—**metadata** and **provenance** APIs help.

### Spatial Reasoning Limits

VLMs can **struggle** with **precise** spatial relations (“left of”, “smaller than”)—**explicit** measurement tools or **structured** outputs help.

### Audio and Speech (Pointer)

**Speech** models often use **Conformer**/**Whisper** encoders feeding **text** LMs—modalities compose similarly: **encode** → **project** → **decode**.

### Benchmarks: MMMU, MathVista

**Multimodal** reasoning benchmarks combine **images** with **knowledge** and **math**—better proxies for **assistant** quality than ImageNet-style metrics.

---

## Interview Takeaways

- **CLIP** trains **dual encoders** with symmetric contrastive loss; it enables zero-shot retrieval and seeds multimodal LMs.
- **ViT** turns images into **token sequences**; **patch count** drives LM context cost.
- **LLaVA-class** models **project** visual tokens into LM embedding space and fine-tune with **multimodal instruction** data.
- Closed systems emphasize **interleaved** multimodal chat, **safety**, and **dynamic resolution**—open models vary in connector design.
- **Cross-modal alignment** quality limits downstream reasoning—bad embeddings cannot be fixed by prompting alone.
- Production stacks must budget **latency** for vision encoding + **long visual token** sequences.

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP): [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT): [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Liu et al., *Visual Instruction Tuning* (LLaVA): [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning* (2022): [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP): [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
