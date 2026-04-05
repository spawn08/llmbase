# The Encoder-Decoder Paradigm

## Why This Matters for LLMs

The encoder–decoder pattern is one of the most important **design templates** in deep learning: split computation into (1) a module that **compresses** or **summarizes** inputs into an internal representation and (2) a module that **expands** that representation into an output. The same abstract idea appears in **autoencoders** (reconstruction), **sequence-to-sequence** translation (source sentence \(\rightarrow\) target sentence), **Transformer encoder–decoder** models such as T5 and BART, **variational autoencoders** (probabilistic latents used in generative modeling and as building blocks for diffusion), and **vision–language** systems where an image encoder feeds a text decoder. Interviewers reward candidates who can name this pattern and map concrete architectures to it—because it shows you see the **common thread** beneath different papers and API names.

For LLMs specifically, “encoder–decoder vs decoder-only vs encoder-only” is not a trivia quiz: it reflects **what supervision signal you have**, **whether generation must be conditional on a separate input**, and **how you pay for cross-attention and bidirectional context**. Models like GPT are decoder-only; BERT is encoder-only; T5/BART are full encoder–decoder stacks. Multimodal assistants often reuse a **frozen image encoder** (ViT, CLIP) plus a **causal text decoder**—again, encoder feeding decoder across modalities. Understanding the paradigm helps you explain **why** attention splits into **self-attention** and **cross-attention**, and why a single-vector summary was historically a **bottleneck** (the RNN seq2seq story is developed with equations in [Sequence-to-Sequence Models](../01_foundations/sequence_to_sequence.md)—this page stays at the pattern level).

Once you internalize the abstraction, many interview questions become variations on one theme: *What is the bottleneck? What information is preserved? What is the training objective?* That framing travels well beyond translation—into summarization, VQA, captioning, and tool-using agents where “observation” is encoded and “action or answer” is decoded.

## Core Concepts

### The Abstract Pattern

An **encoder** \(f_{\mathrm{enc}}\) maps an input \(\mathbf{x}\) to an intermediate representation \(\mathbf{z}\) (sometimes called **context**, **code**, or **latent**):

\[
\mathbf{z} = f_{\mathrm{enc}}(\mathbf{x})
\]

A **decoder** \(f_{\mathrm{dec}}\) maps \(\mathbf{z}\) to an output \(\hat{\mathbf{y}}\):

\[
\hat{\mathbf{y}} = f_{\mathrm{dec}}(\mathbf{z})
\]

The representation \(\mathbf{z}\) is the **information bottleneck**: its dimensionality and structure determine what can be carried forward. In the simplest designs, \(\mathbf{z}\) is a **fixed-size vector**; in others it is a **set of tokens** (e.g., encoder hidden states attended by a decoder).

!!! math-intuition "In Plain English"
    - \(\mathbf{z}\) is the “summary variable”: everything the decoder is allowed to know about \(\mathbf{x}\) must pass through it (unless you add skip connections, attention, or side channels).
    - **Bottleneck** = inductive bias: force the model to learn **salient** factors because it cannot copy the input verbatim.

### Autoencoders

The **autoencoder** is the cleanest non-sequential instance: encode, then decode to **reconstruct** the input. Training minimizes reconstruction error, e.g. squared loss for real-valued vectors:

\[
\mathcal{L}_{\mathrm{AE}} = \lVert \mathbf{x} - f_{\mathrm{dec}}(f_{\mathrm{enc}}(\mathbf{x})) \rVert_2^2
\]

An **undercomplete** autoencoder uses \(\dim(\mathbf{z}) \ll \dim(\mathbf{x})\), so the network cannot memorize trivially; it must learn **compressed features** that still allow reconstruction. That compression is the same **bottleneck intuition** as in seq2seq—only the target is \(\mathbf{x}\) itself rather than another sequence.

!!! example "Worked Example: MNIST flattened to a 32-D bottleneck"
    Flatten a \(28\times 28\) grayscale image to \(\mathbf{x} \in \mathbb{R}^{784}\). Choose \(\mathbf{z} \in \mathbb{R}^{32}\) with an MLP encoder \(784 \rightarrow 128 \rightarrow 32\) and decoder \(32 \rightarrow 128 \rightarrow 784\). **Rate:** \(784\) input pixels \(\rightarrow\) \(32\) scalars (**\(\approx 24\times\)** compression if we count raw numbers naïvely). **Distortion:** whatever cannot be represented in \(32\) dims shows up as blur or smearing after reconstruction. Training pushes the model to allocate those \(32\) dimensions to **digit identity**, stroke structure, and other recurring factors—because those minimize average error across the dataset.

### Sequence-to-Sequence (Preview)

In **sequence-to-sequence** models, the encoder reads the **source** sequence (e.g. English tokens) and produces a summary; the decoder generates the **target** sequence (e.g. French) **one token at a time**, conditioned on that summary. A classic RNN formulation compresses the entire source into a **single context vector** \(\mathbf{c}\) (often the last encoder hidden state)—so the full source must “fit” in one vector. That **bottleneck** is exactly why **attention** became essential: it gives the decoder **selective access** to all encoder positions instead of only \(\mathbf{c}\).

!!! note "Where to go for the full RNN story"
    The concrete recurrence, **Bahdanau attention**, and **teacher forcing** are covered in depth in [Sequence-to-Sequence Models](../01_foundations/sequence_to_sequence.md). Here we only need the **pattern**: encode source \(\rightarrow\) represent \(\rightarrow\) decode target.

### The Information Bottleneck

Compression is both **strength** and **weakness**. By forcing \(\mathbf{z}\) to be smaller (or lower-rate) than \(\mathbf{x}\), you encourage **abstraction**—shared structure across examples. But any strictly lossy bottleneck **discards detail** that might matter for a particular downstream task. **Rate–distortion theory** (intuitively) formalizes the trade-off: lower **rate** (fewer bits or smaller \(\mathbf{z}\)) generally implies higher **distortion** (worse fidelity) unless the data truly lie on a low-dimensional manifold. Deep networks learn **nonlinear** compressors, so they can achieve better distortion at a given rate than linear PCA—but the trade-off does not disappear.

### Encoder–Decoder in Transformers

**T5** and **BART** instantiate the paradigm with **Transformer blocks**: the encoder stack is **bidirectional** self-attention over the source; the decoder stack is **causal** self-attention over the target **plus** **cross-attention** into encoder outputs. **Cross-attention** is the modern bridge that generalizes “read from encoder memory at each decode step.”

By contrast:

| Family | Encoder | Decoder | Typical use |
|--------|---------|---------|-------------|
| **Encoder–decoder** (e.g. T5, BART) | Full self-attn on input | Causal self-attn + cross-attn | Conditional generation (MT, summarization) |
| **Decoder-only** (e.g. GPT) | — | Causal self-attn on concatenated prompt+output | General LM; “encoder” role folded into prefix tokens |
| **Encoder-only** (e.g. BERT) | Full self-attn on input | — | Representation, classification, MLM |

**GPT** is not “missing half a model” in a careless sense: the **prompt** is represented inside the same causal stack, so conditioning is **in-context** rather than via a separate encoder. **BERT** never models \(P(\text{output}\mid\text{input})\) as an autoregressive factorization for generation—it excels at **fill-in** and **encoding** for downstream heads.

### Vision–Language Models

In multimodal setups, an **image encoder** (ViT, CLIP visual tower, etc.) maps pixels to a sequence of **visual tokens** or a pooled embedding; a **text decoder** (often a causal Transformer LM) generates captions, answers, or actions **conditioned** on those tokens. Architecturally, this is still **encoder \(\rightarrow\) bottleneck representation \(\rightarrow\) decoder**, with **cross-attention** (or early/late fusion variants) wiring modalities together. The bottleneck may be a **small set of learned queries** (e.g. Q-Former style interfaces) rather than one vector—same pattern, richer representation.

### Variational Autoencoders (VAE)

A **VAE** replaces a deterministic \(\mathbf{z}\) with a **distribution**: the encoder outputs parameters \((\boldsymbol{\mu}, \boldsymbol{\sigma})\) (often diagonal Gaussian), a latent sample is drawn as

\[
\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
\]

(the **reparameterization trick**, with \(\odot\) element-wise), and the decoder maps \(\mathbf{z}\) to \(\hat{\mathbf{x}}\). Training adds a **KL term** toward a prior—encouraging a smooth latent space usable for generation. VAE ideas appear in **some text models** and as conceptual kin to **diffusion** (encode toward noise, decode toward data), but LLM interviews more often probe **Transformers** and **seq2seq** than full VAE derivations.

!!! math-intuition "In Plain English"
    - \(\boldsymbol{\mu}\): “best guess” latent location for this input.
    - \(\boldsymbol{\sigma}\): uncertainty / spread; sampling injects diversity.
    - Reparameterization moves randomness to \(\boldsymbol{\epsilon}\) so gradients can flow into \(\boldsymbol{\mu}\) and \(\boldsymbol{\sigma}\).

## Deep Dive

??? deep-dive "Cross-attention, prefix LMs, and encoder–decoder trade-offs"
    **Cross-attention as the encoder–decoder bridge.** In Transformers, decoder layer \(l\) forms queries \(\mathbf{Q}\) from decoder states; keys and values \(\mathbf{K},\mathbf{V}\) come from **encoder** outputs. One schematic step is \(\mathrm{Attention}(\mathbf{Q}_{\mathrm{dec}}, \mathbf{K}_{\mathrm{enc}}, \mathbf{V}_{\mathrm{enc}})\). That lets every target position **retrieve** source information dynamically—addressing the classical single-vector bottleneck without hand-designed alignments (contrast with the RNN story in Part 1).

    **Prefix LMs vs encoder–decoder vs decoder-only.** A **prefix LM** (or decoder with **prefix masking**) processes a non-causal “prefix” (e.g. prompt or source) with bidirectional or partially masked attention, then generates causally—blurring the line between “encoder” and “decoder” while keeping one stack. A **pure decoder-only** model uses **left-to-right** attention on the **concatenation** of conditioning and output; implementation is uniform, but long bidirectional conditioning is **not** native unless you add special attention masks or external memory. **Encoder–decoder** cleanly separates **bidirectional encoding** of the source from **causal decoding**, which can help **conditional** tasks at the cost of more parameters and training complexity.

    **When encoder–decoder vs decoder-only.** Encoder–decoder shines when there is a **clear input–output asymmetry** (summarization, translation, denoising reconstruction as in BART) and you want **cross-attention** into a full encoded source. Decoder-only dominates **general-purpose autoregressive LMs** and chat because a single stack scales well and **prompting** handles many tasks without architectural specialization—though task quality on strict **input-to-output** mappings can still favor encoder–decoder designs in some benchmarks.

## Code

The following **PyTorch** script defines a minimal **fully connected autoencoder** on synthetic “fake MNIST” tensors (shape \(784\)), trains it briefly with MSE reconstruction loss, and prints the **bottleneck** activations for one batch.

```python
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """784 -> 128 -> 32 -> 128 -> 784 MLP autoencoder."""

    def __init__(self, input_dim: int = 784, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def synthetic_mnist_like(batch: int, dim: int = 784, seed: int = 0) -> torch.Tensor:
    """Random non-negative 'images' in [0,1] — stand-in for flattened pixels."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand((batch, dim), generator=g)


def main() -> None:
    torch.manual_seed(0)
    input_dim = 784
    batch_size = 64
    steps = 200
    lr = 1e-2

    model = AutoEncoder(input_dim=input_dim, latent_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for step in range(steps):
        x = synthetic_mnist_like(batch_size, input_dim, seed=step)
        opt.zero_grad()
        x_hat, z = model(x)
        loss = loss_fn(x_hat, x)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  recon_loss={loss.item():.6f}  z.shape={tuple(z.shape)}")

    model.eval()
    with torch.no_grad():
        demo = synthetic_mnist_like(4, input_dim, seed=123)
        recon, bottleneck = model(demo)
    print("bottleneck (first row, first 8 dims):", bottleneck[0, :8].tolist())
    print("recon error (last step):", loss_fn(recon, demo).item())


if __name__ == "__main__":
    main()
```

The tensor `bottleneck` is \(\mathbf{z} \in \mathbb{R}^{32}\): the **compressed code** the decoder must use to approximate \(\mathbf{x}\). On real MNIST, you would swap `synthetic_mnist_like` for DataLoader batches of flattened images normalized to \([0,1]\).

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What is the encoder–decoder pattern at a high level?**  
       *Depth:* Encoder maps input to an internal representation; decoder maps that representation to output; the representation is the **bottleneck** that forces abstraction.

    2. **How does an autoencoder relate to seq2seq?**  
       *Depth:* Same split (encode \(\rightarrow\) decode); AE reconstructs \(\mathbf{x}\), seq2seq predicts a **different** sequence \(\mathbf{y}\). Both wrestle with **limited \(\mathbf{z}\)**.

    3. **Why was a single RNN context vector problematic?**  
       *Depth:* Entire source compressed into one \(\mathbf{c}\)—limited capacity; **attention** exposes all encoder timesteps (see Part 1 for Bahdanau details).

    4. **What role does cross-attention play in T5/BART?**  
       *Depth:* Decoder queries **encoder memory** each step—dynamic access vs. one static summary vector.

    5. **Contrast GPT, BERT, and encoder–decoder Transformers.**  
       *Depth:* Decoder-only (causal LM), encoder-only (bidirectional MLM/encoding), encoder–decoder (conditional generation with cross-attn).

    6. **Why might vision–language models still be “encoder–decoder”?**  
       *Depth:* Image tower encodes pixels to tokens/embeddings; text decoder generates language **conditioned** on them via cross-attention or fusion.

    7. **What is the rate–distortion trade-off (informally)?**  
       *Depth:* Smaller/fewer latent bits \(\Rightarrow\) less detail retained unless the data are highly structured.

    8. **How does a VAE differ from a plain autoencoder?**  
       *Depth:* Latent **distribution** + KL regularization; **reparameterization** for backprop through sampling.

    9. **When would you prefer encoder–decoder over decoder-only?**  
       *Depth:* Strong input–output asymmetry, need **full bidirectional** encoding of source, classic conditional NLP tasks—though decoder-only can still be competitive with prompting/IT.

    10. **What is a prefix LM, conceptually?**  
        *Depth:* One stack with a **prefix** region and a **causal** generation region—intermediate between strict encoder–decoder and vanilla GPT-style prompting.

!!! interview "Follow-up Probes"
    - “Is the bottleneck always a vector?” → No—can be **encoder token sequence** attended by decoder.
    - “Does cross-attention imply alignment?” → **Soft** alignment weights, similar interpretability caveats as seq2seq attention.
    - “Can decoder-only models do translation?” → Yes via **prompting**, but inductive bias differs from explicit encoder–decoder.
    - “What does undercomplete mean?” → \(\dim(\mathbf{z}) < \dim(\mathbf{x})\), forcing compression.
    - “VAE vs AE for anomaly detection?” → VAE can flag **low likelihood** under \(p(\mathbf{x})\); AE uses reconstruction error—both need careful calibration.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Encoder–decoder** separates **encoding** (summarize input) from **decoding** (produce output).”
    - “The **bottleneck** \(\mathbf{z}\) is the **information channel**—capacity limits what can be preserved.”
    - “**Cross-attention** lets the decoder **query** encoder representations each step.”
    - “**Decoder-only** models fold conditioning into the **prompt** inside one causal stack.”
    - “**Rate–distortion**: smaller latents \(\Rightarrow\) more abstraction, less fine detail.”
    - “**VAE**: encoder outputs **\(\boldsymbol{\mu}, \boldsymbol{\sigma}\)**; **reparameterization** enables learning.”
    - “Multimodal systems often pair an **image encoder** with a **text decoder**—same abstract pattern.”

## References

- Kingma & Welling (2014), *Auto-Encoding Variational Bayes*
- Rumerhart, Hinton, Williams (1986) — classic backprop / representation learning context for autoencoders
- Vaswani et al. (2017), *Attention Is All You Need* (encoder–decoder Transformer, cross-attention)
- Raffel et al. (2020), *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5)
- Lewis et al. (2020), *BART: Denoising Sequence-to-Sequence Pre-training*
- Dosovitskiy et al. (2021), *An Image is Worth 16x16 Words* (ViT—encoder for images)
- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* (CLIP—dual encoders; often paired with decoders in later systems)
