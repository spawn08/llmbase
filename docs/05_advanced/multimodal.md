# Multimodal LLMs

## Why This Matters for LLMs

**Multimodal LLMs** process **images, audio, video, and structured signals** alongside text—frontier assistants are judged on **screenshots, charts, UI mockups, and camera input**, not prose alone. Architecturally, this means **vision encoders** (ViT, CLIP, SigLIP, DINOv2), **audio encoders** (Whisper-style), and **connector layers** that map continuous inputs into the **token embedding space** of a decoder-only LM. Interviews increasingly ask how **CLIP-style alignment** differs from **generative** VLMs, how **visual token budgets** affect latency, and how **hallucinated objects** differ from text-only hallucinations.

Second, **cross-modal alignment** is a **representation-learning** problem: image patches and text spans must occupy a **shared geometry** where cosine similarity reflects **semantic** correspondence. **Contrastive** training (CLIP) optimizes **global** image–text matching; **generative** training optimizes \(P(\text{text} \mid \text{image})\) directly. Engineers need vocabulary for **zero-shot transfer**, **instruction tuning** on multimodal chat data, and **evaluation** pitfalls (language shortcuts in VQA).

Third, **systems constraints** bite harder: a single image may yield **hundreds of patch tokens**; projecting them into a 70B LM dominates **prefill FLOPs** and **context** usage. **Resolution**, **dynamic cropping**, **Q-Former bottlenecks**, and **interleaved** image–text sequences are as important as backbone parameter counts.

---

## Core Concepts

### Architecture Patterns

1. **Early fusion**: interleave **image tokens** and **text tokens** in one sequence (LLaVA-class with projector).
2. **Late fusion**: separate encoders; combine at **pooled** representations (some retrieval stacks).
3. **Cross-attention fusion**: **Flamingo**-style gated cross-attention layers inject vision into LM hidden states at multiple depths.

### CLIP: Contrastive Language–Image Pre-training

Dual encoders \(f_I\) (image) and \(f_T\) (text) produce embeddings. With L2 normalization \(\hat{f} = f/\|f\|_2\), **cosine similarity** equals dot product:

\[
s_{ij} = \langle \hat{f}_I(x_i), \hat{f}_T(t_j) \rangle
\]

!!! math-intuition "In Plain English"
    After L2 normalization, each \(s_{ij}\) is a **cosine** in \([-1,1]\): **high** when image \(i\) and text \(j\) **align** in the joint space. The **diagonal** \(s_{ii}\) is what training pushes up versus all off-diagonal \(j\) in the same row.

For batch size \(N\), the **symmetric** contrastive loss uses temperature \(\tau\):

\[
\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^{N} \left[
\log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ij}/\tau}}
+
\log \frac{e^{s_{ii}/\tau}}{\sum_{j=1}^{N} e^{s_{ji}/\tau}}
\right]
\]

!!! math-intuition "In Plain English"
    Each image \(i\) should be **closest** to its own caption among the \(N\) captions in the batch (softmax row), and each caption symmetrically **matches** its image column. **In-batch negatives** mean **large batches** matter.

!!! example "Worked Example: 4×4 Similarity and Row Softmax"
    Suppose \(N=4\), \(\tau=1\), and **unnormalized** logits (cosines after L2 norm) are:

    \[
    S = \begin{bmatrix}
    0.9 & 0.1 & 0.2 & 0.0 \\
    0.0 & 0.85 & 0.3 & 0.1 \\
    0.1 & 0.0 & 0.95 & 0.05 \\
    0.2 & 0.1 & 0.0 & 0.88
    \end{bmatrix}
    \]

    **Diagonal** entries are highest in each row—good. For **row 1** (image 1), softmax weights are proportional to \(\exp(0.9), \exp(0.1), \exp(0.2), \exp(0.0)\). Numerically: \(\exp(0.9)\approx 2.46\), \(\exp(0.1)\approx 1.11\), \(\exp(0.2)\approx 1.22\), \(\exp(0)=1\). Sum \(\approx 5.79\). **Probability** for the correct caption \(j=1\) is \(\approx 2.46/5.79 \approx 0.42\). **Cross-entropy** \(-\log 0.42 \approx 0.87\) for that row—training **pushes** the diagonal logits **up** and off-diagonals **down**. **Symmetric** loss repeats for **columns** (text-to-image).

### Vision Transformer (ViT)

Split an \(H \times W\) image into \(P \times P\) patches. The number of patch tokens is:

\[
N_{\text{patches}} = \frac{H}{P} \cdot \frac{W}{P}
\]

Each patch is linearly embedded; add position embeddings; run Transformer blocks.

!!! math-intuition "In Plain English"
    **ViT** is “BERT for pixels”: a **sequence** model on patches—**quadratic** patch self-attention cost drives **high-resolution** bottlenecks.

### SigLIP (Sigmoid Loss)

**SigLIP** uses **pairwise sigmoid** losses instead of softmax over the batch dimension—often more stable when **batch size** is small:

\[
\mathcal{L}_{\text{sig}} \approx - \sum_{i,j} \Bigl[ y_{ij} \log \sigma(s_{ij}/\tau) + (1-y_{ij})\log(1-\sigma(s_{ij}/\tau)) \Bigr]
\]

with \(y_{ij}=1\) iff \(i=j\) for matched pairs (formulations vary).

!!! math-intuition "In Plain English"
    Softmax **competes** every image against every caption in one normalization; sigmoid treats **pairs** more independently—better when you **cannot** afford giant batches.

### LLaVA: Projector + LLM

Let \(F_{\text{img}} \in \mathbb{R}^{N \times d_v}\) be vision tokens from a frozen ViT. A **projection** \(W \in \mathbb{R}^{d_v \times d_{\text{LLM}}}\) maps into LM embedding space:

\[
H_{\text{vis}} = F_{\text{img}} W
\]

The LM sees an interleaved sequence: **system / user** text + **visual tokens** + **assistant** tokens.

!!! math-intuition "In Plain English"
    The projector is a **dictionary** from **patch semantics** to **fake word embeddings** the LM already understands—**language** reasoning stays in the **decoder**; **vision** is encoded **upstream**.

### Flamingo: Cross-Attention and Perceiver

**Flamingo** inserts **gated cross-attention** layers that let LM hidden states attend to **visual** features. **Perceiver resampler** compresses variable-size feature maps to a **fixed** token count.

### Gemini / Native Multimodal (Conceptual)

**Natively multimodal** models tokenize **audio, image, video** in a **unified** vocabulary from early training—not only a **vision tower bolted** onto a text-only checkpoint—enabling **interleaved** multimodal pretraining at scale (public details vary by vendor).

### Video and Audio (Sketch)

**Video**: sample frames at rate \(f_s\), encode each frame → **temporal pooling** or **transformer** over frames. **Audio**: log-mel spectrogram → ConvNet/Transformer → projector to LM space.

\[
T_{\text{video tokens}} \approx N_{\text{frame}} \cdot N_{\text{patches per frame}}
\]

!!! math-intuition "In Plain English"
    Long videos **explode** token counts—**frame selection** and **compression** (perceiver, pooling) are **mandatory** for real-time assistants.

### CLIP Zero-Shot Classification

Given class names \(\{c_k\}_{k=1}^{K}\), embed prompts \(t_k = f_T(\text{“a photo of a } c_k \text{”})\) and image \(x\):

\[
k^\star = \arg\max_k \langle \hat{f}_I(x), \hat{t}_k \rangle
\]

!!! math-intuition "In Plain English"
    You never train a **classifier head**—**cosine** against **text** prototypes is the **decision rule**. Prompt wording (“photo”, “sketch”) shifts accuracy—**prompt engineering** is real.

!!! example "Worked Example: Patch Count vs Resolution"
    ViT-B/16 on \(224 \times 224\) uses \(P=16\): \(N = (224/16)^2 = 14 \times 14 = 196\) patches. At \(448 \times 448\) with the same \(P\): \((448/16)^2 = 28^2 = 784\) patches—**4×** patches for **2×** resolution per side—**quadratic** in resolution for fixed patch size.

### Evaluation Map (Practical)

| Benchmark family | What it measures | Caveat |
|------------------|------------------|--------|
| Image–text retrieval (MSCOCO/Flickr) | Recall@\(k\) | Domain bias to web photos |
| VQA v2 / GQA | Visual question answering | **Shortcuts** from language priors |
| DocVQA / Chart QA | Reading + reasoning | OCR errors propagate |
| MMMU / MathVista | Multimodal knowledge + math | Harder proxy for assistants |

### Failure Modes (Multimodal)

- **Object hallucination**: model names objects not present—mitigate with **grounding** boxes or tool-assisted detection.
- **OCR errors**: small text—**increase** resolution or **crop** regions.
- **Prompt injection via pixels**: adversarial images—**policy** layers + **no** execution of instructions in images.

### Licensing and Data

Web-scale image–text pairs carry **copyright** and **consent** issues—production systems apply **filtering**, **deduplication**, and **safety** classifiers.

### Adapter Tuning (PEFT)

Common pattern: **LoRA** on LM layers + **full** or **LoRA** fine-tune on **projector**; freeze vision encoder early, **unfreeze** later layers when data is ample.

### Contrastive vs Generative Objectives

| Objective | Optimizes | Strength |
|------------|-----------|----------|
| Contrastive (CLIP) | Match image–text pairs in embedding space | Retrieval, zero-shot |
| Generative captioning | \(P(\text{caption}\mid\text{image})\) | Fluent descriptions |

**VLMs** for chat often **compose** both: contrastive **encoder** + **autoregressive** LM.

### Visual Misinformation

Screenshots can **fabricate** evidence—**policy** should constrain **high-stakes** decisions from unverified visuals; **metadata** and **provenance** APIs help.

### Spatial Reasoning Limits

Precise relations (“slightly left of center”, “smaller than”) can be **hard**—explicit **measurement** tools or **structured** outputs (bounding boxes) reduce error.

### Data Mixing During Instruction Tuning

Let \(\alpha\) be the fraction of **multimodal** instruction pairs vs **text-only** SFT:

\[
\mathcal{L} = \alpha \mathcal{L}_{\text{mm}} + (1-\alpha) \mathcal{L}_{\text{text}}
\]

!!! math-intuition "In Plain English"
    Too much multimodal data can **erode** pure language skills—track **per-modality** metrics during training.

### Temperature and Logit Scale at Inference

CLIP training uses **learned logit scale** \(s\) (or fixed \(\tau^{-1}\)):

\[
\text{logits}_{ij} = s \cdot \langle \hat{f}_I(x_i), \hat{f}_T(t_j) \rangle
\]

!!! math-intuition "In Plain English"
    Larger \(s\) **sharpens** the softmax—more **confident** similarities; calibration for downstream **zero-shot** depends on this scale—**OpenCLIP** and **HF** checkpoints bake in different scales.

### Accessibility Note

**Alt text** and **screen readers** remain essential—multimodal generation must **not** replace accessible pipelines; generated captions can be **wrong** or **unsafe**.

??? deep-dive "Deep Dive: Q-Former (BLIP-2)"
    **Q-Former** learns **query tokens** that **cross-attend** to frozen image features and output a **fixed** small set of visual tokens—**bottleneck** that cuts LM FLOPs vs feeding every patch.

??? deep-dive "Deep Dive: OCR and Small Text"
    VLMs can **misread** tiny fonts; **pipeline** systems often add **detection + OCR** before VLM reasoning for **documents** and **screenshots** with dense text.

## Code

### CLIP similarity with `transformers` (downloads weights on first run)

```python
"""Compute CLIP image-text similarity — requires: pip install torch transformers pillow requests"""
from __future__ import annotations

import torch
from PIL import Image

def main() -> None:
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    # Tiny inline image: 64x64 red square (no external file needed)
    img = Image.new("RGB", (64, 64), color=(200, 40, 40))
    texts = [
        "a red square",
        "a blue ocean",
        "a photo of a dog",
    ]
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = model(**inputs)
        logits_per_image = out.logits_per_image  # shape (1, num_texts)
        probs = logits_per_image.softmax(dim=1)
    print("softmax over texts:", probs.tolist())
    best = int(probs.argmax(dim=1).item())
    print("best caption index:", best, "->", texts[best])


if __name__ == "__main__":
    main()
```

### Toy CLIP loss (no external deps beyond torch)

```python
"""Symmetric CLIP loss on random features — pedagogical."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyCLIP(nn.Module):
    def __init__(self, dim_in: int = 128, dim: int = 64):
        super().__init__()
        self.img_enc = nn.Linear(dim_in, dim)
        self.txt_enc = nn.Linear(dim_in, dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode(self, x: torch.Tensor, enc: nn.Linear) -> torch.Tensor:
        return F.normalize(enc(x), dim=-1)

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        ie = self.encode(image, self.img_enc)
        te = self.encode(text, self.txt_enc)
        logits = ie @ te.transpose(0, 1) * self.logit_scale.exp()
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.transpose(0, 1), targets)
        return (loss_i + loss_t) / 2


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d_in = 8, 128
    model = ToyCLIP(d_in=d_in)
    img = torch.randn(n, d_in)
    txt = torch.randn(n, d_in)
    loss = model(img, txt)
    print("symmetric clip loss:", float(loss))
```

### LLaVA-style inference (optional, large weights)

```python
"""
Optional LLaVA inference — downloads multi-GB weights if run.
pip install torch transformers accelerate protobuf
"""
from __future__ import annotations


def llava_demo() -> None:
    try:
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration
    except ImportError:
        print("Install transformers + torch to run llava_demo.")
        return
    model_id = "llava-hf/llava-1.5-7b-hf"
    print("Loading (large) model:", model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    from PIL import Image

    image = Image.new("RGB", (336, 336), color=(30, 120, 80))
    prompt = "USER: <image>\nWhat color dominates this image?\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=32)
    text = processor.decode(out[0], skip_special_tokens=True)
    print(text)


if __name__ == "__main__" and __import__("os").environ.get("RUN_LAVA"):
    llava_demo()
```

!!! interview "FAANG-Level Questions"
    1. How does CLIP training differ from image captioning (generative) training?
    *Answer:* **CLIP** is **contrastive**: push matched image–text pairs together and negatives apart in embedding space—great for retrieval and zero-shot, no autoregressive image modeling. **Captioning** optimizes \(P(\text{text}\mid\text{image})\) with a decoder—fluent descriptions and dense supervision, but not inherently aligned for symmetric retrieval. VLMs often combine pieces: contrastive encoders plus LM fine-tuning for chat.
    2. Why are L2-normalized embeddings paired with cosine similarity in CLIP?
    *Answer:* L2 normalization maps vectors to the **unit sphere**, so dot product **equals** cosine similarity and removes magnitude effects from patch brightness or caption length. Training with a **temperature** or learned scale then sharpens the softmax distribution. Without normalization, norm drift can dominate similarity and destabilize contrastive learning.
    3. Describe LLaVA-style fusion: what is frozen, what is trained?
    *Answer:* Typical LLaVA recipes **freeze** a pretrained **ViT** (or partially unfreeze later), **train** a lightweight **projector** (MLP) mapping vision tokens into the LLM embedding space, and **LoRA/SFT** the LLM on instruction data. Vision provides features; language reasoning stays in the decoder—data-efficient but projector quality gates multimodal fidelity.
    4. How does ViT patch count scale when you double image resolution?
    *Answer:* With fixed patch size \(P\), patches per side scale with \(H/P\) and \(W/P\), so **doubling** each spatial dimension **quadruples** patch count (\(2\times\) per axis \(\Rightarrow 4\times\) tokens). Prefill FLOPs and memory grow accordingly—why dynamic resolution, pooling, or Q-Former bottlenecks matter for high-res inputs.
    5. What is the difference between early fusion and cross-attention fusion (Flamingo)?
    *Answer:* **Early fusion** (LLaVA-style) concatenates **projected image tokens** with text tokens in one decoder sequence—simple, but vision tokens consume context budget upfront. **Flamingo cross-attention** injects visual features via **gated cross-attention layers** at multiple depths—richer interaction and variable visual compression (e.g., perceiver), often heavier to train/serve but flexible fusion.
    6. Why might SigLIP help when batch size is small?
    *Answer:* Softmax CLIP **competes** each image against all captions in-batch—small batches mean **fewer negatives**, noisier gradients, and unstable training. **SigLIP** uses **sigmoid** losses over pairs, behaving more like independent binary decisions—often more stable and data-efficient when you cannot afford huge batches. Trade-off: different calibration and engineering compared to classic CLIP.
    7. How do multimodal LLMs hallucinate objects that are not in the image?
    *Answer:* The LM prior is **strong**: it can complete plausible scenes (e.g., “stop sign”) from weak evidence or language cues alone—**visual grounding** is imperfect. Low resolution, occlusion, or ambiguous patches let the decoder **confabulate** details to satisfy the instruction. Mitigations: grounding boxes, detection tools, “point to evidence,” and refusal when attention/segmentation is uncertain.
    8. What is the Q-Former bottleneck solving in BLIP-2?
    *Answer:* Feeding **all** ViT patches into a large LLM explodes **tokens and FLOPs**. The **Q-Former** learns a small set of **query tokens** that cross-attend to image features, producing a **fixed-length** visual summary for the LM—compressing vision to a budget without hand-tuned pooling. Quality depends on query count and training stage alignment.
    9. How would you budget latency for vision encoder + LLM prefill?
    *Answer:* Profile end-to-end: **vision forward** (often large at high resolution), **projector**, then **LLM prefill** on image+text tokens—vision can dominate short text. Reduce image tokens (cropping, lower res, pooling), batch vision across requests, use faster encoders or INT8, and cap max image side. Product SLA drives whether you **async** thumbnail-first responses or stream partial UI.
    10. What are shortcut behaviors in VQA benchmarks?
    *Answer:* Models exploit **language priors** and dataset biases—answering “yes/no” from question wording or frequent object co-occurrences **without** using pixels. This inflates VQA accuracy while failing on counterfactual or adversarial images. Mitigations: balanced splits, **compositionality** tests, visual perturbations, and **open-ended** evaluation with human or tool verification.

!!! interview "Follow-up Probes"
    - “How do you evaluate chart understanding separately from OCR?”
    - “When would you add a dedicated OCR stage instead of end-to-end pixels?”
    - “How does interleaved image-text pretraining differ from vision-tower-only fine-tune?”

!!! key-phrases "Key Phrases to Use in Interviews"
    - “CLIP aligns image and text in a shared embedding space with contrastive InfoNCE.”
    - “LLaVA projects ViT tokens into the LM embedding space with a trained projector.”
    - “Patch-count drives prefill cost—compress with Q-Former or pooling before the LLM.”
    - “VQA shortcuts: models may answer from language priors without looking at pixels.”

## References

- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision* (CLIP): [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT): [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- Liu et al., *Visual Instruction Tuning* (LLaVA): [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- Alayrac et al., *Flamingo: a Visual Language Model for Few-Shot Learning* (2022): [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- Zhai et al., *Sigmoid Loss for Language Image Pre-Training* (SigLIP): [arXiv:2303.15343](https://arxiv.org/abs/2303.15343)
- Li et al., *BLIP-2: Bootstrapping Language-Image Pre-training* (2023): [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision* (2023): [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
