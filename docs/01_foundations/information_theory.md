# Information Theory for LLMs

## Why This Matters for LLMs

Information theory is the **shared vocabulary** for uncertainty, coding length, and distance between distributions. When you read “**minimize cross-entropy**,” “**perplexity 12**,” or “**KL penalty to the reference model**” in an RLHF paper, those are not slogans—they are precise statements about **entropy**, **expected code length**, and **distributional mismatch**. Modern LLM **pretraining** is overwhelmingly **maximum likelihood** = **cross-entropy minimization**. **Evaluation** uses perplexity (exp of average cross-entropy). **Alignment** often adds **reverse KL**-style penalties so the policy does not drift from a trusted base model. Interviewers expect you to connect **Shannon entropy → cross-entropy → KL → training loss** in one coherent chain.

---

## Core Concepts

### Entropy — Expected Surprise

For discrete \(X\) with PMF \(p(x)\):

\[
H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x)
\]

Convention: \(0 \log 0 = 0\). Log base 2 gives **bits**; natural log gives **nats**.

!!! math-intuition "In Plain English"
    - \(p(x)\): how often symbol \(x\) occurs.
    - \(-\log p(x)\): **surprise** (rare events have large surprise).
    - The sum \(\sum_x p(x)(-\log p(x))\): **average surprise** if you draw from \(p\)—that is **entropy**.
    - **High entropy:** hard to predict (spread mass). **Low entropy:** nearly deterministic.

!!! example "Worked Example: Fair vs. Biased Coin (every step of \(-p\log p\))"
    **Fair coin:** \(P(H)=0.5\), \(P(T)=0.5\). Use \(\log_2\).

    - Term for \(H\): \(-0.5 \log_2(0.5) = -0.5 \cdot (-1) = 0.5\) bits.
    - Term for \(T\): same, \(0.5\) bits.
    - **Total:** \(H = 0.5 + 0.5 = 1\) bit. Intuition: one fair binary question resolves the outcome.

    **Biased coin:** \(P(H)=0.9\), \(P(T)=0.1\).

    - \(H\) term: \(-0.9 \log_2(0.9)\). Now \(\log_2(0.9) \approx -0.152\), so \(-0.9 \cdot (-0.152) \approx 0.137\) bits.
    - \(T\) term: \(-0.1 \log_2(0.1)\). \(\log_2(0.1) \approx -3.322\), so \(-0.1 \cdot (-3.322) \approx 0.332\) bits.
    - **Total:** \(H \approx 0.137 + 0.332 = 0.469\) bits **< 1**. The outcome is more predictable—entropy dropped.

    **Deterministic:** \(P(H)=1\). Only term is \(-1 \cdot \log_2(1) = 0\). **Zero entropy**—no surprise.

### Cross-Entropy — Coding Under the Wrong Distribution

True distribution \(p\); model \(q\) (your approximate probabilities):

\[
H(p, q) = -\sum_x p(x) \log q(x)
\]

!!! math-intuition "In Plain English"
    You **believe** \(q\); nature follows \(p\). Cross-entropy is the **average number of nats/bits** you spend encoding outcomes **if** your code is optimal for \(q\) but events come from \(p\). It is always \(\ge H(p)\) with equality iff \(q = p\) (Gibbs’ inequality).

!!! example "Worked Example: Token Prediction with Three Words"
    Vocabulary order: **[cat, dog, fish]**. Model probabilities \(q = [0.7,\, 0.2,\, 0.1]\). True next token: **cat** (one-hot \(p = [1,0,0]\)).

    \[
    H(p, q) = -\sum_i p_i \log q_i = -\log q(\texttt{cat}) = -\log(0.7)
    \]

    Natural log: \(-\ln(0.7) \approx 0.357\) nats. If training averages this over tokens, **lower is better**—higher \(q\) on the true token reduces loss.

    If the true token had been **fish** instead:

    \[
    H(p, q) = -\log(0.1) \approx 2.302\ \text{nats}
    \]

    **Penalty explodes** when the model assigns near-zero mass to the truth—this is why **label smoothing** and **numerical floors** matter in practice.

### Why Cross-Entropy **IS** the LLM Training Loss (step-by-step)

1. **Data distribution:** a corpus induces a **true** conditional distribution over next tokens \(p(w_t \mid w_{<t})\) (unknown, but we sample from it).
2. **Model:** \(q_\theta(w_t \mid w_{<t})\) from softmax logits.
3. **Objective:** **maximum likelihood**—maximize \(\mathbb{E}_{w \sim \text{data}}[\log q_\theta(w_t \mid w_{<t})]\).
4. **Negate:** minimize \(-\mathbb{E}[\log q_\theta] = \mathbb{E}[-\log q_\theta]\).
5. For a **single** correct token \(w^*\) (empirical one-hot), that expectation is **one term** \(-\log q_\theta(w^*)\)—the **cross-entropy** between the empirical one-hot and the model distribution.

So **“train with cross-entropy”** and **“do MLE on next-token prediction”** are the same sentence in different notation.

!!! math-intuition "In Plain English"
    Each training token says: “**raise** probability on **this** symbol.” Summing \(-\log q_\theta(w^*)\) punishes **confident wrongness** exponentially more than mild mistakes—gradient magnitude \(\propto 1/q\) for softmax targets.

### KL Divergence — Extra Cost of Using \(q\) Instead of \(p\)

\[
D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
\]

!!! math-intuition "In Plain English"
    - \(H(p)\): unavoidable average surprise if you use the **true** code.
    - \(H(p, q)\): average surprise using a code tuned to **\(q\)** while nature draws from **\(p\)**.
    - Their difference is the **overhead** from misspecification—**KL**—always \(\ge 0\), zero iff \(p=q\) a.e. on the support of \(p\).

!!! example "Worked Example: KL with Two Distributions, Term by Term"
    **True** \(p\): \([0.5,\, 0.5]\) over \(\{a,b\}\). **Model** \(q\): \([0.6,\, 0.4]\).

    \[
    D_{\text{KL}}(p \| q) = 0.5 \log\frac{0.5}{0.6} + 0.5 \log\frac{0.5}{0.4}
    \]

    Natural logs:

    - First term: \(0.5 \cdot (\ln 0.5 - \ln 0.6) = 0.5 \cdot (-0.6931 + 0.5108) \approx 0.5 \cdot (-0.1823) \approx -0.0912\)
    - Second: \(0.5 \cdot (\ln 0.5 - \ln 0.4) = 0.5 \cdot (-0.6931 + 0.9163) \approx 0.5 \cdot (0.2231) \approx 0.1116\)

    Sum \(\approx 0.0204\) nats **\(\ge 0\)**. Small mismatch → small KL.

    **Sanity check:** \(H(p) = -\ln 0.5 = 0.693\) nats. \(H(p,q) = -0.5\ln 0.6 - 0.5\ln 0.4 \approx 0.5(0.511) + 0.5(0.916) \approx 0.714\) nats. Difference \(0.714 - 0.693 \approx 0.021\) ✓.

### Forward vs. Reverse KL (Mode-Covering vs. Mode-Seeking)

- **Forward KL** \(D_{\text{KL}}(p \| q)\): **expectation under \(p\)**. Wherever \(p\) has mass, \(q\) must place mass or pay \(\log(p/q)\) **huge** penalty—**mode-covering** behavior. **MLE / cross-entropy** aligns with this direction when \(p\) is data and \(q\) is the model.

- **Reverse KL** \(D_{\text{KL}}(q \| p)\): **expectation under \(q\)**. Penalty emphasizes regions **\(q\) thinks likely**. If \(q\) is narrow, it can **ignore** a secondary mode of \(p\) — **mode-seeking**. Used in some **variational** objectives and in **KL-to-reference** penalties where \(p\) is a **frozen** reference LM and \(q\) is the policy.

!!! example "Bimodal Target in Text (no plot needed)"
    Suppose **true** \(p\) places 0.5 mass on “**The answer is 42.**” and 0.5 on “**The answer is π.**”

    - **Forward-KL-optimal** \(q\) tends to cover **both** modes (mass on both sentences), because missing either mode incurs infinite log penalty where \(p>0\) but \(q \approx 0\).
    - **Reverse-KL-optimal** \(q\) may put almost all mass on **one** mode (unimodal \(q\))—cheaper to be confidently wrong about the **other** mode if \(q\) rarely visits it.

    **RLHF intuition:** a **KL penalty** \(\beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})\) discourages the tuned policy \(\pi\) from straying from \(\pi_{\text{ref}}\)—often implemented with a **reverse-KL-like** form on sequences or approximations (see Schulman’s notes / PPO-KL variants).

### Perplexity = exp(cross-entropy)

Average **per-token** cross-entropy (nats): \(\hat{H} = -\frac{1}{T}\sum_{t=1}^T \log q(w_t \mid w_{<t})\).

\[
\text{PPL} = \exp(\hat{H})
\]

!!! math-intuition "In Plain English"
    If PPL = 30, the model’s predictive distribution is, on average, as “flat” as if you were **uniform over ~30 choices**—a **calibrated** intuition for comparing LMs **on the same vocabulary and test set**.

!!! example "Worked Example: Real Numbers"
    Suppose for a short sentence the average **negative log-likelihood** is **3.0 nats** per token. Then \(\text{PPL} = e^{3} \approx 20.09\).

    If you improve to **2.5 nats**, \(\text{PPL} = e^{2.5} \approx 12.18\)—**perplexity drops multiplicatively** with small CE improvements.

### Connection to RLHF’s KL Penalty

**Policy** \(\pi_\theta\) (tuned model) vs. **reference** \(\pi_{\text{ref}}\) (SFT or base LM). A typical surrogate adds:

\[
-\beta \, \mathbb{E}\bigl[D_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x))\bigr]
\]

!!! math-intuition "In Plain English"
    - Keeps \(\pi_\theta\) **close** to something trusted—reduces **reward hacking** and **incoherent** text.
    - \(\beta\) trades **helpfulness** vs. **staying on-distribution**.
    - Implementations approximate KL with **closed-form** expressions for Gaussians in some diffusion work; for discrete tokens, **Monte Carlo** or **analytic** softmax-KL pieces appear depending on algorithm (PPO, DPO, etc.).

### Minimizing Cross-Entropy ≡ Minimizing Forward KL to Data

\[
\arg\min_\theta H(p, q_\theta) = \arg\min_\theta D_{\text{KL}}(p \| q_\theta)
\]

because \(H(p)\) does not depend on \(\theta\).

!!! math-intuition "In Plain English"
    **MLE** pushes \(q_\theta\) toward **covering** the data distribution—**forward KL** intuition. This is why **hallucination** is not solved by CE alone: **missing** modes in data are not heavily penalized if the model stays **sharp** elsewhere (plus many other factors).

### Bits vs. Nats (Interview Hygiene)

\[
H_{\text{bits}} = -\sum_x p(x) \log_2 p(x), \qquad
H_{\text{nats}} = -\sum_x p(x) \ln p(x), \quad H_{\text{nats}} = H_{\text{bits}} \cdot \ln 2
\]

!!! example "Worked Example: Fair Coin in Both Units"
    Fair coin: \(H = 1\) bit \(= \ln 2 \approx 0.693\) nats. When comparing papers, **check the log base** before claiming one model has “lower entropy” than another.

### Mutual Information (Bridge to Advanced Topics)

\[
I(X; Y) = D_{\text{KL}}(p(x,y) \| p(x)p(y)) = H(X) - H(X \mid Y)
\]

!!! math-intuition "In Plain English"
    **How much** \(Y\) tells you about \(X\). Used in **information bottleneck** (“compress \(X\) into \(Z\) while keeping \(I(Z;Y)\) high”), **representation** probing, and some **interpretability** analyses of layers.

### DPO and Log-Ratio Views (High-Level)

**Direct Preference Optimization** rewrites Bradley–Terry-style preferences using **log-probability differences** between policy and reference—**implicitly** controlling deviation from \(\pi_{\text{ref}}\). You rarely need the full loss on a whiteboard, but you should say: **preference learning** still compares **policy vs. reference** in log-space; **KL** is the **units** of “how far we drifted” from a trusted LM.

### Full-Sentence Perplexity Chain (Worked)

Given token NLLs \( \ell_t = -\log q(w_t \mid w_{<t})\) in nats, **mean** \(\bar{\ell} = \frac{1}{T}\sum_t \ell_t\), then \(\text{PPL} = \exp(\bar{\ell})\).

!!! example "Worked Example: Three Tokens"
    Suppose \(\ell_1 = 0.5\), \(\ell_2 = 1.0\), \(\ell_3 = 1.5\) nats. Mean \(\bar{\ell} = 3/3 = 1.0\) nats/token. \(\text{PPL} = e^1 \approx 2.72\) — very small because these numbers were toy-low; real LMs on WikiText sit at **much** higher \(\bar{\ell}\).

---

## Code (with inline comments)

```python
"""
Entropy, cross-entropy, and KL divergence — computed from scratch and
verified against PyTorch / scipy implementations.
"""
import numpy as np
import torch
import torch.nn.functional as F


def entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in nats."""
    p = p[p > 0]  # 0 log 0 = 0: drop zero masses
    return -np.sum(p * np.log(p))


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy H(p, q) in nats. q must be > 0 wherever p > 0."""
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats."""
    return cross_entropy(p, q) - entropy(p)


# ── True distribution vs. two models ──────────────────────────────────
p = np.array([0.4, 0.3, 0.2, 0.1])       # true next-token distribution
q_good = np.array([0.35, 0.30, 0.20, 0.15])  # close model
q_bad = np.array([0.1, 0.1, 0.1, 0.7])       # poor model

print("=== NumPy from-scratch ===")
print(f"H(p)          = {entropy(p):.4f} nats")
print(f"H(p, q_good)  = {cross_entropy(p, q_good):.4f} nats")
print(f"H(p, q_bad)   = {cross_entropy(p, q_bad):.4f} nats")
print(f"KL(p || q_good) = {kl_divergence(p, q_good):.4f} nats")
print(f"KL(p || q_bad)  = {kl_divergence(p, q_bad):.4f} nats")

# ── Verify with PyTorch ────────────────────────────────────────────────
p_t = torch.tensor(p, dtype=torch.float32)
q_good_t = torch.tensor(q_good, dtype=torch.float32)
q_bad_t = torch.tensor(q_bad, dtype=torch.float32)

# PyTorch kl_div: input is log_q, target is p — matches KL(p || q) with sum reduction
kl_good_pt = F.kl_div(q_good_t.log(), p_t, reduction="sum").item()
kl_bad_pt = F.kl_div(q_bad_t.log(), p_t, reduction="sum").item()

print("\n=== PyTorch verification ===")
print(f"KL(p || q_good) = {kl_good_pt:.4f}")
print(f"KL(p || q_bad)  = {kl_bad_pt:.4f}")

# ── Cross-entropy as LLM loss ─────────────────────────────────────────
logits = torch.tensor([[2.0, 1.5, 0.8, 0.2]])  # raw model outputs (1,4)
target = torch.tensor([0])                       # correct token index (class 0)

ce_loss = F.cross_entropy(logits, target)  # softmax + NLL of true class
ppl = torch.exp(ce_loss)
print(f"\nCross-entropy loss for token 0: {ce_loss.item():.4f}")
print(f"Per-token perplexity: {ppl.item():.2f}")
```

---

## Deep Dive

??? deep-dive "Jensen–Shannon and Symmetric 'Distances'"
    **KL is not symmetric.** Jensen–Shannon divergence \(JSD(p,q) = \tfrac{1}{2}D_{\text{KL}}(p\|m)+\tfrac{1}{2}D_{\text{KL}}(q\|m)\) with \(m=\tfrac{p+q}{2}\) **is** symmetric and bounded. Mention when interviewers ask for a **metric-like** alternative—used in some GAN training (though modern LMs rarely cite JSD in loss).

??? deep-dive "Label Smoothing vs. True Entropy"
    **Label smoothing** replaces one-hot \(p\) with \((1-\epsilon)\) on true class and \(\epsilon/(K-1)\) elsewhere. This **raises** cross-entropy vs. hard targets but **lowers** overconfidence—acts as regularization. Connect to **calibration** and **generalization**.

??? deep-dive "Why KL \(\ge 0\) (Jensen Sketch)"
    Let \(f(t) = -\log t\), **convex** on \(t>0\). For any \(q\) with full support where needed, Jensen’s inequality on \(\sum_x p(x) f(q(x)/p(x))\) yields **non-negativity** of KL. Interviewers rarely want the full proof—**“convexity of \(-\log\)”** is the **buzzphrase**.

### Smoothing, Zero Masses, and Numerical Safety

If \(q(x)=0\) but \(p(x)>0\), cross-entropy has an **infinite** term—**why** training uses **label smoothing**, **temperature** bounds, and **epsilon** in some Keras/PyTorch recipes. In **evaluation**, perplexity on **held-out** text avoids **exact** zeros by construction (tokens appeared in vocab), but **subword** OOV handling still matters for fair comparison.

!!! example "Worked Example: Tiny Smoothing Effect"
    True class probability under label smoothing with \(K=4\), \(\epsilon=0.1\): target vector becomes \([0.925,\, 0.025,\, 0.025,\, 0.025]\) instead of \([1,0,0,0]\). Cross-entropy **no longer** blows up if the model puts \(0\) on a non-target class—gradients stay **finite**.

---

## Interview Guide

!!! interview "FAANG-Level Questions"
    1. **What is cross-entropy minimizing in LM training?**  
       Negative log-likelihood of observed tokens—equivalent to **MLE**.

    2. **Prove or argue that \(D_{\text{KL}}(p\|q) \ge 0\).**  
       Gibbs’ inequality / Jensen on \(\log\).

    3. **Why is minimizing cross-entropy the same as minimizing forward KL to data?**  
       \(H(p)\) constant w.r.t. \(\theta\); only \(H(p,q_\theta)\) changes.

    4. **Forward vs. reverse KL—give a bimodal example.**  
       Forward covers modes; reverse may collapse to one.

    5. **Perplexity 50 on a 50k vocab—how to interpret?**  
       **Not** “uniform over 50k words”—compare **relative** PPL across models on **same** corpus/tokenizer.

    6. **Why does RLHF add a KL penalty?**  
       Keep policy near reference; reduce incoherent exploitation of reward model.

    7. **Numerical issue:** what if \(q(w^*)=0\) for a true token?  
       Infinite CE—**why** masking, label smoothing, and **logit clipping** matter.

    8. **Bits per character vs. perplexity?**  
       Related reporting units; know **which log base** your metric uses.

    9. **Is KL a metric?**  
       No—not symmetric, triangle inequality fails; JSD is closer to a metric.

    10. **Connection between entropy rate and compression?**  
        Shannon source coding; expected length \(\ge H(p)\) per symbol (asymptotically).

!!! interview "Follow-up Probes"
    - “**Why** is reverse KL mode-seeking?” — mass where \(q\) focuses dominates expectation.
    - “**Difference** between **uncertainty** (entropy of Bayes posterior) and **epistemic** uncertainty?” — advanced; mention ensembles / latent variables.
    - “**How** does **temperature** affect implied perplexity?” — softmax sharpening spreads or concentrates mass.

!!! key-phrases "Key Phrases to Use in Interviews"
    - “**Cross-entropy** is the **expected** \(-\log q\) under the **data** distribution.”
    - “**MLE** on tokens **is** **cross-entropy** with **one-hot** targets.”
    - “**KL** measures **extra bits** from using \(q\) instead of \(p\).”
    - “**Forward KL** is **inclusive**; **reverse KL** can **miss modes**.”
    - “**Perplexity** is **exp(average cross-entropy)**—geometric mean **branching factor** intuition.”

### Quick Reference Card (Loss ↔ Information)

| Object | Formula (discrete) | Role in LLMs |
| --- | --- | --- |
| Entropy | \(-\sum p \log p\) | Irreducible uncertainty of **true** data |
| Cross-entropy | \(-\sum p \log q\) | **Training loss** (next-token) |
| KL | \(\sum p \log(p/q)\) | Gap between **target** and **model**; **RLHF** penalty |
| Perplexity | \(\exp(\text{avg NLL})\) | **Scalar** model comparison on fixed test |

**Sanity check:** On a **finite** alphabet, **entropy** is maximized by the **uniform** distribution—so if someone says “maximize entropy of outputs,” they may mean **encourage diversity** (different from CE training, which is **data-driven**).

**Compare across models only** when the **tokenizer**, **test set**, and **context length** match—otherwise perplexity is **not** comparable apples-to-apples.

For **byte-level** or **BPE** models, always note whether reported PPL is **per token** or **per byte** (BPC).

---

## References

- Shannon (1948), *A Mathematical Theory of Communication*
- Cover & Thomas, *Elements of Information Theory*
- Jurafsky & Martin, SLP — Language Modeling and Entropy
- Radford et al. (2019), *Language Models are Unsupervised Multitask Learners* (perplexity reporting)
- Schulman et al. — RLHF / PPO with KL notes (implementation details vary by repo)
