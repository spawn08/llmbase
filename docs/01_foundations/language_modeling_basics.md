# Language Modeling Basics

## Why This Matters for LLMs

At its core, every chatbot, translation system, and autocomplete feature is doing ONE thing: predicting the next word. Given "I love eating ice ___", what comes next? "cream" has high probability, "truck" has low probability. A language model is just a fancy probability calculator for "what word comes next?"

Every large language model you read about in papers or deploy in production is, at its core, a **language model**: a system that assigns probabilities to sequences of tokens and, in the autoregressive setting, predicts the next token given everything that came before. When you prompt GPT-4 or Llama 3, the model is not "thinking" in the human sense; it is evaluating conditional distributions \\(P(w_t \\mid w_1, \\ldots, w_{t-1})\\) (or their tokenized equivalents) billions of times, sampling or greedily picking the next piece of text. Interviewers at top technology companies return to this fact constantly because it separates candidates who memorized architecture diagrams from candidates who understand **what is being optimized** and **what the objective means**.

The **chain rule** is the single equation that ties classical n-gram models, recurrent neural language models, and modern Transformer decoders into one family. Whether the conditional is stored in a sparse count table, produced by an LSTM hidden state, or computed by stacked self-attention layers, the factorization of the joint probability of a sentence is the same. If you can explain that decomposition clearly, you can explain why training uses next-token cross-entropy, why perplexity is reported, and why "the model learned syntax" is shorthand for "the model assigns high probability to syntactically typical continuations."

Finally, language modeling fundamentals are how interviewers probe **generalization and data sparsity**. N-gram models make the limitations obvious: unseen contexts get zero probability without smoothing. Neural models hide the issue behind dense parameters, but the underlying tension remains: the space of possible histories grows faster than any corpus. Scaling laws and longer contexts are engineering responses to the same quantity the chain rule isolated: the conditional next-token distribution must be accurate at every position, which ties **training** (average cross-entropy) directly to **evaluation** (perplexity on held-out text). Demonstrating that you understand the Markov assumption, smoothing, and perplexity shows you know why subword tokenizers and massive pretraining budgets exist, not only what they are called.

!!! tip "Notation Help"
    - \\(P(w_t \\mid w_1, \\ldots, w_{t-1})\\) means "the probability of token \\(w_t\\) **given** all previous tokens" — see [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#5-probability-basics) for probability notation
    - The vertical bar \\(\\mid\\) means "given" or "conditioned on"
    - \\(\\mathbf{v} \\in \\mathbb{R}^d\\) means "vector v with d dimensions" — see [Math Prerequisites](../00_prerequisites/00_math_prerequisites.md#1-vectors-and-matrices)

---

## Core Concepts

### What is a Language Model?

#### The Idea

A **language model** is a probability machine. You feed it a prefix of text (full words, subwords, or characters depending on tokenization). It responds with a distribution over what token is allowed to come next. Good models assign high probability to fluent, factual, and contextually appropriate continuations and low probability to gibberish or contradictions. The same machinery supports autocomplete, machine translation decoders, speech recognition rescoring, and chat interfaces: anywhere you need a score or a ranked list of next units, a language model is the usual workhorse.

#### The Math

??? tip "🎯 Think of it like… Reading a Story Word by Word"
    Imagine you're reading a novel one word at a time, covering everything after the current word with a piece of paper. At each step, you try to **guess** the next word based on everything you've read so far.

    - After reading "The", you might guess "cat", "dog", "man"…
    - After reading "The cat", you narrow it down: "sat", "ran", "slept"…
    - After reading "The cat sat on the", your best guess might be "mat", "floor", "chair"…

    The **chain rule** just says: **the probability of the WHOLE sentence = the product of each of these individual guesses.** That's literally it. It's a formalization of the reading-word-by-word process you've been doing since childhood.

The chain rule in language modeling is just common sense written as math. When you read a sentence word by word, each new word's meaning depends on ALL the words before it. The probability of the whole sentence is just: probability of word 1 × probability of word 2 given word 1 × probability of word 3 given words 1–2 × ... and so on.

Let \\(w_1, w_2, \\ldots, w_T\\) denote a sequence of \\(T\\) tokens. A language model specifies the **joint probability**:

\\[
P(w_1, w_2, \\ldots, w_T).
\\]

By the **chain rule** of probability, that joint decomposes into a product of **conditional** probabilities:

\\[
P(w_1, w_2, \\ldots, w_T) = \\prod_{t=1}^{T} P(w_t \\mid w_1, \\ldots, w_{t-1}).
\\]

<div class="equation-breakdown" markdown>
#### 📝 Symbol-by-Symbol Breakdown

| Symbol | What it means | Analogy |
|--------|--------------|---------|
| \\(P(w_1, w_2, \\ldots, w_T)\\) | Probability of the **entire sentence** appearing exactly as written | "How likely is this specific sentence?" |
| \\(\\prod_{t=1}^{T}\\) | Multiply together \\(T\\) terms (one per word position) | "Take each word's contribution and multiply them all" |
| \\(P(w_t \\mid w_1, \\ldots, w_{t-1})\\) | Probability of word \\(t\\) given **all** previous words | "Given what I've read so far, how likely is this next word?" |
| \\(w_{t-1}\\) | The word at position \\(t-1\\) (one before the current) | "The previous word" |
| \\(T\\) | Total number of words in the sentence | Length of the sequence  |

**How to read the equation aloud:** "The probability of seeing words \\(w_1\\) through \\(w_T\\) together equals the product, for each position \\(t\\) from \\(1\\) to \\(T\\), of the probability of word \\(w_t\\) given all earlier words."
</div>

The first factor is \\(P(w_1)\\) (sometimes conditioned on a start symbol). Each later factor asks: given the entire past, how likely is the token at position \\(t\\)?

!!! math-intuition "In Plain English"
    The symbol \\(P(w_1, \\ldots, w_T)\\) is the probability that the whole string appears exactly as written: the joint probability of the entire sentence.
    The product \\(\\prod_{t=1}^{T} P(w_t \\mid w_1, \\ldots, w_{t-1})\\) means: multiply together the probability of the first token, then the probability of the second token given the first, then the third given the first two, and continue until the end.
    The expression \\(P(w_t \\mid w_1, \\ldots, w_{t-1})\\) reads: **given everything that appeared in positions \\(1\\) through \\(t-1\\), how likely is token \\(w_t\\) as the next symbol?** Autoregressive large language models are trained to estimate exactly these conditionals.

!!! example "Worked Example: Predicting the Next Word"
    **Prefix:** "The cat sat on the"  
    **Candidate next words:** `mat`, `floor`, `dog`, `chair`  
    The model assigns a **conditional probability** to each candidate using the full prefix as context: \\(P(\\text{mat} \\mid \\text{The}, \\text{cat}, \\text{sat}, \\text{on}, \\text{the})\\), \\(P(\\text{floor} \\mid \\text{the same context})\\), \\(P(\\text{dog} \\mid \\text{the same context})\\), \\(P(\\text{chair} \\mid \\text{the same context})\\).
    Suppose the model outputs the following nonnegative values that sum to \\(1\\):
    \\(P(\\text{mat}) = 0.45\\), \\(P(\\text{floor}) = 0.25\\), \\(P(\\text{dog}) = 0.12\\), \\(P(\\text{chair}) = 0.18\\).
    The **support** of the distribution is these four outcomes for this illustration. For a full vocabulary of size \\(V\\), the model defines a vector of length \\(V\\) where every entry is nonnegative and the entries sum to \\(1\\). That is the **probability simplex** constraint: one step of sampling or greedy decoding picks a single next token from this normalized distribution.

### N-gram Language Models

??? tip "🎯 Think of it like… Counting Past Experience"
    Imagine you're an autocomplete system, but instead of a neural network, you have a **notebook** where you wrote down every pair of consecutive words you've ever seen:

    - "ice cream" — seen 80 times
    - "ice water" — seen 15 times
    - "ice truck" — seen 2 times
    - "ice dragon" — seen 0 times

    When someone types "ice ___", you look up your notebook and say: "cream" is 80/(80+15+2) ≈ 82% likely! That's a **bigram** model — it only looks at the **one** previous word. A **trigram** looks at the **two** previous words. And so on.

    The catch? If you've NEVER seen "ice dragon" in your notebook, you say it's **impossible** — probability zero. That's clearly wrong! That's why we need **smoothing** (coming next).

N-grams are the "counting" approach to language modeling. Instead of a neural network, you just count how often word combinations appear in your training text. A bigram model says: "After the word ice, cream appeared 80 times and truck appeared 2 times, so cream is 40x more likely." Simple, but it breaks when you encounter a word combination you've never seen before.

#### Plain English First

An **n-gram model** refuses to look at the entire past. It assumes a **Markov property**: only the last \\(n-1\\) tokens matter when predicting the next one. That turns an enormous history into a short string of context, so counts become tractable. The price is realism: true English depends on subject–verb agreement and coreference that can span dozens of words, which an \\(n\\)-gram model with small \\(n\\) cannot see.

#### The Math

**Unigram** (\\(n = 1\\)): ignore all context.

\\[
P(w_t) = \\frac{\\text{count}(w_t)}{\\sum_{w \\in \\mathcal{V}} \\text{count}(w)}.
\\]

<div class="equation-breakdown" markdown>
#### 📝 What This Says

"The probability of word \\(w_t\\) = how many times \\(w_t\\) appeared ÷ total number of words." No context used at all — just raw word frequency. The word "the" might be 7% of all words, so \\(P(\\text{the}) \\approx 0.07\\).
</div>

**Bigram** (\\(n = 2\\)): the next word depends only on the immediately previous word.

\\[
P(w_t \\mid w_{t-1}) = \\frac{\\text{count}(w_{t-1}, w_t)}{\\text{count}(w_{t-1})}.
\\]

<div class="equation-breakdown" markdown>
#### 📝 What This Says

**Numerator:** How many times did \\(w_{t-1}\\) and \\(w_t\\) appear **next to each other** in training?  
**Denominator:** How many times did \\(w_{t-1}\\) appear **at all** (as the first word of any pair)?  
**Result:** The fraction of times \\(w_{t-1}\\) was followed by \\(w_t\\).

**Concrete example:** If "the" appeared 1000 times and "the cat" appeared 50 times, then \\(P(\\text{cat} \\mid \\text{the}) = 50/1000 = 0.05\\), or 5%.
</div>

**Trigram** (\\(n = 3\\)):

\\[
P(w_t \\mid w_{t-2}, w_{t-1}) = \\frac{\\text{count}(w_{t-2}, w_{t-1}, w_t)}{\\text{count}(w_{t-2}, w_{t-1})}.
\\]

Here \\(\\text{count}(w_{t-1})\\) is the number of times token \\(w_{t-1}\\) appears in the training corpus as the **first** element of a bigram (or more simply, the unigram count of \\(w_{t-1}\\), depending on convention; for standard bigram LM, the denominator is the number of times the context word occurred).

!!! math-intuition "In Plain English"
    \\(\\text{count}(w_{t-1}, w_t)\\) counts how often the pair (previous word, next word) appears in training. Dividing by \\(\\text{count}(w_{t-1})\\) turns that into a fraction: among all times you saw \\(w_{t-1}\\), what fraction of the time was \\(w_t\\) the follower?
    The Markov assumption of **order \\(n-1\\)** means the model pretends history longer than \\(n-1\\) words is irrelevant. That is why increasing \\(n\\) improves expressiveness but blows up the number of possible contexts.

!!! example "Worked Example: Bigram Counts from Four Sentences"
    **Training sentences (lowercased, word tokenization):**
    - Sentence A: `the cat sat on the mat`
    - Sentence B: `the cat ate the fish`
    - Sentence C: `the dog sat on the log`
    - Sentence D: `the dog ate the bone`
    **Step 1 — List every bigram inside each sentence (adjacent pairs):**
    - From A: `(the,cat)`, `(cat,sat)`, `(sat,on)`, `(on,the)`, `(the,mat)`
    - From B: `(the,cat)`, `(cat,ate)`, `(ate,the)`, `(the,fish)`
    - From C: `(the,dog)`, `(dog,sat)`, `(sat,on)`, `(on,the)`, `(the,log)`
    - From D: `(the,dog)`, `(dog,ate)`, `(ate,the)`, `(the,bone)`
    **Step 2 — Aggregate counts for each ordered pair:**
    `(the,cat)=2`, `(cat,sat)=1`, `(sat,on)=2`, `(on,the)=2`, `(the,mat)=1`, `(cat,ate)=1`, `(ate,the)=2`, `(the,fish)=1`, `(the,dog)=2`, `(dog,sat)=1`, `(dog,ate)=1`, `(the,log)=1`, `(the,bone)=1`
    **Step 3 — Unigram counts for words that appear as the first element of a bigram** (same as word counts in this corpus):
    `the` appears \\(8\\) times, `cat` appears \\(2\\) times, `sat` appears \\(2\\) times, `on` appears \\(2\\) times, `ate` appears \\(2\\) times, `dog` appears \\(2\\) times. Words that only appear as second elements in the listed bigrams still have unigram counts: `mat`,`fish`,`log`,`bone` each appear once.
    **Step 4 — Compute example conditional probabilities:**
    \\(P(\\text{cat} \\mid \\text{the}) = \\frac{\\text{count}(\\text{the},\\text{cat})}{\\text{count}(\\text{the})} = \\frac{2}{8} = 0.25\\).
    \\(P(\\text{dog} \\mid \\text{the}) = \\frac{2}{8} = 0.25\\).
    \\(P(\\text{mat} \\mid \\text{the}) = \\frac{1}{8} = 0.125\\).
    \\(P(\\text{sat} \\mid \\text{cat}) = \\frac{\\text{count}(\\text{cat},\\text{sat})}{\\text{count}(\\text{cat})} = \\frac{1}{2} = 0.5\\).
    \\(P(\\text{ate} \\mid \\text{cat}) = \\frac{1}{2} = 0.5\\).
    These numbers are **maximum-likelihood** estimates from the tiny corpus. A real system would use smoothing (next section) because any bigram absent from the table would get probability zero under raw counts.

#### 🔬 Try It Yourself: Interactive N-gram Calculator

<div class="viz-container" id="ngram-calc" markdown>
*Loading interactive N-gram calculator…*
</div>

### Smoothing

??? tip "🎯 Think of it like… Sharing Candy"
    Imagine you have 100 candies to distribute among students based on test scores. The top student scored 40/100 and gets 40 candies. But what about the student who was absent and scored 0? With pure "maximum likelihood" distribution, they get **nothing** — zero candies.

    **Smoothing** says: "Before distributing, I'll add 1 bonus candy to every student's score." Now the absent student gets 1 candy (not zero), the top student gets 41 (slightly less dominant), and nobody is left empty-handed.

    In language modeling, the "candies" are probability mass, the "students" are word pairs, and the "absent student" is an unseen n-gram that should NOT get zero probability.

Smoothing solves the "never seen before" problem. Without smoothing, if the model has never seen "ice dragon" in its training data, it assigns probability ZERO — which is too harsh. Smoothing says: "even if I've never seen this combination, it's probably not literally impossible."

#### Plain English

**Smoothing** reallocates probability mass so that unseen n-grams do not force the entire sentence probability to zero. Raw counts lie: absence in a finite corpus does not mean impossible in the language. **Add-k (Laplace) smoothing** adds a small positive constant \\(k\\) to every count before normalization, pulling probability slightly away from frequent events toward rare or unseen ones.

#### The Math

For bigrams with vocabulary size \\(V\\), **add-k** smoothing gives:

\\[
P(w_t \\mid w_{t-1}) = \\frac{\\text{count}(w_{t-1}, w_t) + k}{\\text{count}(w_{t-1}) + k \\cdot V}.
\\]

<div class="equation-breakdown" markdown>
#### 📝 Symbol-by-Symbol Breakdown

| Part | What it is | Why it's there |
|------|-----------|----------------|
| \\(\\text{count}(w_{t-1}, w_t)\\) | Times we saw this specific pair | The real evidence from data |
| \\(+ k\\) | A small bonus added to the numerator | Ensures **every** pair gets at least a tiny probability |
| \\(\\text{count}(w_{t-1})\\) | Total times we saw the context word | Our base for computing fractions |
| \\(+ k \\cdot V\\) | Correction to the denominator | We added \\(k\\) to \\(V\\) different pairs, so we must add \\(k \\times V\\) to keep everything summing to 1 |

**When \\(k = 0\\):** No smoothing — raw counts. Unseen pairs get probability 0.  
**When \\(k = 1\\):** Classic Laplace smoothing — pretend you saw every pair once extra.  
**When \\(k \\to \\infty\\):** Everything approaches \\(1/V\\) — uniform distribution, ignoring all data.
</div>

The denominator ensures probabilities over all \\(w_t\\) still sum to \\(1\\) for each fixed context \\(w_{t-1}\\).

!!! math-intuition "In Plain English"
    The numerator \\(\\text{count}(w_{t-1}, w_t) + k\\) says: pretend we saw each bigram \\(k\\) extra times. The denominator adds \\(k \\cdot V\\) pseudo-counts so that the total mass for context \\(w_{t-1}\\) matches the adjusted numerator sums. Larger \\(k\\) pushes the distribution closer to uniform.

!!! example "Worked Example: \\(k = 0\\) versus \\(k = 1\\)"
    Suppose vocabulary \\(\\mathcal{V} = \\{\\text{cat}, \\text{dog}, \\text{sat}\\}\\), so \\(V = 3\\). Context word is `cat`. Observed bigrams from `cat`: only `(cat, sat)` appears, once. So \\(\\text{count}(\\text{cat}) = 1\\) as context for the next word in this fragment.
    **Case \\(k = 0\\) (no smoothing):**  
    \\(P(\\text{sat} \\mid \\text{cat}) = \\frac{1}{1} = 1\\).  
    \\(P(\\text{dog} \\mid \\text{cat}) = \\frac{0}{1} = 0\\).  
    Any test sentence that needs `dog` after `cat` gets **zero** joint probability.
    **Case \\(k = 1\\) (Laplace, \\(V = 3\\)):**  
    \\(P(\\text{sat} \\mid \\text{cat}) = \\frac{1 + 1}{1 + 1 \\cdot 3} = \\frac{2}{4} = 0.5\\).  
    \\(P(\\text{dog} \\mid \\text{cat}) = \\frac{0 + 1}{4} = 0.25\\).  
    \\(P(\\text{cat} \\mid \\text{cat}) = \\frac{1}{4} = 0.25\\).  
    Unseen bigrams are no longer impossible; they share the smoothed mass.

#### Backoff and Interpolation (Advanced Counting Strategies)

**Backoff** means: if the trigram count is zero, use the bigram estimate; if that is zero, use the unigram. **Interpolation** means: always mix several orders with weights that sum to one:

\\[
P_{\\text{interp}}(w_t \\mid w_{t-2}, w_{t-1}) =
\\lambda_3 \\, P_{\\text{tri}}(w_t \\mid w_{t-2}, w_{t-1})
+ \\lambda_2 \\, P_{\\text{bi}}(w_t \\mid w_{t-1})
+ \\lambda_1 \\, P_{\\text{uni}}(w_t),
\\quad \\lambda_1 + \\lambda_2 + \\lambda_3 = 1.
\\]

<div class="equation-breakdown" markdown>
#### 📝 What This Says

Instead of relying on a single n-gram order, **interpolation blends** all orders together:

- \\(\\lambda_3 \\cdot P_{\\text{tri}}\\): Give weight \\(\\lambda_3\\) (e.g., 60%) to the trigram estimate (most specific)
- \\(\\lambda_2 \\cdot P_{\\text{bi}}\\): Give weight \\(\\lambda_2\\) (e.g., 30%) to the bigram estimate (backup)
- \\(\\lambda_1 \\cdot P_{\\text{uni}}\\): Give weight \\(\\lambda_1\\) (e.g., 10%) to the unigram estimate (last resort)

The lambdas (\\(\\lambda\\)) must sum to 1 because the result must be a valid probability. Even if the trigram count is zero, the bigram and unigram pieces keep the final probability positive.
</div>

!!! math-intuition "In Plain English"
    Interpolation keeps every probability contribution alive: even if the trigram table has a hole, the bigram and unigram pieces still supply positive mass. The lambdas are hyperparameters or values fit on held-out data. Backoff is sparser in spirit: use the highest-order estimate only when the count exists, otherwise **fall back** stepwise.

!!! example "Worked Example: Numeric Interpolation (Toy Weights)"
    Fix \\(\\lambda_3 = 0.6\\), \\(\\lambda_2 = 0.3\\), \\(\\lambda_1 = 0.1\\). Suppose for some context you obtain \\(P_{\\text{tri}} = 0\\), \\(P_{\\text{bi}} = 0.2\\), \\(P_{\\text{uni}} = 0.01\\). Then
    \\(P_{\\text{interp}} = 0.6 \\cdot 0 + 0.3 \\cdot 0.2 + 0.1 \\cdot 0.01 = 0 + 0.06 + 0.001 = 0.061\\).
    The trigram term vanishes, but the sentence still receives nonzero probability because lower-order pieces participate.

### Perplexity

??? tip "🎯 Think of it like… A Multiple-Choice Quiz"
    Imagine taking a quiz where every question is: "What's the next word?"

    - If the quiz has **2 options** per question and you always guess right, the test is easy → **perplexity ≈ 2**
    - If the quiz has **50 options** per question and you're equally unsure about all of them → **perplexity ≈ 50**
    - If the quiz has **50,000 options** per question (the full vocabulary!) and you have no clue → **perplexity ≈ 50,000**

    Modern LLMs like GPT-4 achieve perplexity around **3–8** on typical text. That means the model narrows things down to just a handful of likely candidates at each position — it's really good at this quiz!

    **Lower perplexity = smarter model.** A perplexity of 1 would mean perfect prediction (impossible in practice because language has genuine ambiguity).

Perplexity answers: "On average, how confused is the model about what comes next?" A perplexity of 10 means the model is as confused as if it were randomly choosing between 10 equally likely options. Lower perplexity = less confused = better model.

#### Plain English First

**Perplexity** answers: if the model were calibrated like a fair die at each step, how many sides would that die have on average? Lower perplexity means the model is **less surprised** by the correct next token: it spreads probability mass over fewer plausible alternatives. A perplexity of \\(50\\) is often described informally as the model behaving as if it were choosing among roughly \\(50\\) equiprobable words at each step, though the actual distribution is usually far from uniform.

#### The Math

For a sequence of length \\(T\\) (in tokens), with model predictions \\(\\hat{P}(w_t \\mid w_1, \\ldots, w_{t-1})\\):

\\[
\\text{PPL} = \\exp\\!\\left( -\\frac{1}{T} \\sum_{t=1}^{T} \\log \\hat{P}(w_t \\mid w_1, \\ldots, w_{t-1}) \\right).
\\]

<div class="equation-breakdown" markdown>
#### 📝 Symbol-by-Symbol Breakdown

Let's unpack this from the inside out:

| Part | What it computes | Intuition |
|------|-----------------|-----------|
| \\(\\hat{P}(w_t \\mid \\ldots)\\) | Model's predicted probability for the correct next word | "How confident was the model?" |
| \\(\\log \\hat{P}(\\ldots)\\) | Log of that probability (always negative or zero) | Converts to "surprise" scale |
| \\(-\\log \\hat{P}(\\ldots)\\) | **Surprise** — higher when the model was less confident | Surprise for a single word |
| \\(\\frac{1}{T} \\sum_{t=1}^{T}\\) | Average surprise across all \\(T\\) words | Average surprise per word |
| \\(\\exp(\\ldots)\\) | Convert back from log-scale to a number | Turn average surprise into "effective choices" |

**The inner sum** \\(-\\frac{1}{T}\\sum \\log P\\) is the **cross-entropy** (in nats if using \\(\\ln\\), in bits if using \\(\\log_2\\)).  
**The outer exp** converts cross-entropy back to an interpretable "branching factor."
</div>

The inner average is the **cross-entropy** (in nats if \\(\\log\\) is natural log). Perplexity is \\(\\exp\\) of that.

!!! math-intuition "In Plain English"
    The term \\(-\\log \\hat{P}(w_t \\mid \\cdots)\\) is **surprise** for step \\(t\\): higher probability means lower surprise. Averaging over \\(T\\) steps gives typical surprise per token. Exponentiating turns log-loss back into a "branching factor" scale that people find easier to interpret than raw nats or bits.

!!! example "Worked Example: Five-Token Sentence"
    **Sentence:** `the cat sat on mat`  
    Assume the model assigns the following **conditional** probabilities for the correct next token at each step (given the true prefix in training evaluation):
    - After start: \\(P(\\text{the}) = 0.20\\)
    - After `the`: \\(P(\\text{cat} \\mid \\text{the}) = 0.15\\)
    - After `the cat`: \\(P(\\text{sat} \\mid \\text{the}, \\text{cat}) = 0.10\\)
    - After `the cat sat`: \\(P(\\text{on} \\mid \\text{the}, \\text{cat}, \\text{sat}) = 0.30\\)
    - After `the cat sat on`: \\(P(\\text{mat} \\mid \\text{the}, \\text{cat}, \\text{sat}, \\text{on}) = 0.05\\)
    **Step 1 — Sum of log probabilities:**  
    \\(\\log 0.20 + \\log 0.15 + \\log 0.10 + \\log 0.30 + \\log 0.05 \\approx -1.609 + (-1.897) + (-2.303) + (-1.204) + (-2.996) = -10.009\\) (natural log).
    **Step 2 — Average negative log likelihood:**  
    \\(-\\frac{1}{5} \\sum \\log p_i \\approx \\frac{10.009}{5} = 2.002\\) nats per token.
    **Step 3 — Perplexity:**  
    \\(\\text{PPL} = \\exp(2.002) \\approx 7.4\\).  
    The model is about as uncertain as a uniform choice among roughly seven options at each step **on this illustrative path**.

---

## Deep Dive: From N-grams to Neural Language Models

??? deep-dive "Deep Dive: Why N-grams Hit a Wall"
    **Curse of dimensionality:** For a vocabulary of size \\(V\\), there are \\(V^{n}\\) possible \\(n\\)-grams. Most never appear in training, so maximum-likelihood estimates are zero for almost all contexts at moderate \\(n\\).
    **Sparsity:** Smoothing helps locally but does not create meaningful similarity between distinct contexts. `cat sat` and `dog sat` are unrelated rows in a count table even though they share structure.
    **Continuous representations:** Neural language models map contexts and words into dense vectors. Similar contexts receive similar hidden states, so the model **generalizes** beyond exact n-gram matches. That is the bridge from this chapter to word embeddings and recurrent networks.

---

## Code

The following script builds n-gram counts, applies add-k smoothing, and evaluates **perplexity** on a test token sequence. Read the inline comments for the mapping between formulas and code.

```python
"""
N-gram Language Model with Laplace smoothing and perplexity evaluation.
"""
import math
from collections import Counter, defaultdict
from typing import List, Tuple


def tokenize(text: str) -> List[str]:
    """Lowercase split tokenizer with sentence boundary tokens."""
    tokens = text.lower().split()
    # Boundary markers let the model learn start/end behavior like a real LM
    return ["<s>"] + tokens + ["</s>"]


def build_ngram_counts(
    corpus: List[List[str]], n: int
) -> Tuple[Counter, Counter]:
    """Return (ngram_counts, context_counts) for order n."""
    ngram_counts: Counter = Counter()
    context_counts: Counter = Counter()
    for tokens in corpus:
        # Slide a window of length n across each sentence
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])  # Full n-gram tuple for P(w | context)
            context = ngram[:-1]  # First (n-1) tokens are the conditioning context
            ngram_counts[ngram] += 1
            context_counts[context] += 1  # Denominator for P(w | context)
    return ngram_counts, context_counts


def ngram_probability(
    word: str,
    context: Tuple[str, ...],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
    k: float = 1.0,
) -> float:
    """P(word | context) with add-k smoothing."""
    ngram = context + (word,)
    # Add-k in numerator and k * V in denominator matches the bigram formula generalized
    numerator = ngram_counts[ngram] + k
    denominator = context_counts[context] + k * vocab_size
    return numerator / denominator


def perplexity(
    test_tokens: List[str],
    ngram_counts: Counter,
    context_counts: Counter,
    vocab_size: int,
    n: int,
    k: float = 1.0,
) -> float:
    """Compute perplexity of a token sequence under the n-gram model."""
    log_prob_sum = 0.0
    count = 0
    # Each position predicts test_tokens[i] from the previous (n-1) tokens
    for i in range(n - 1, len(test_tokens)):
        context = tuple(test_tokens[i - n + 1 : i])
        word = test_tokens[i]
        prob = ngram_probability(
            word, context, ngram_counts, context_counts, vocab_size, k
        )
        log_prob_sum += math.log(prob)
        count += 1
    # exp(- average log prob) == exp(cross-entropy in nats)
    return math.exp(-log_prob_sum / count) if count > 0 else float("inf")


# ── Demo ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    corpus_text = [
        "the cat sat on the mat",
        "the cat ate the fish",
        "the dog sat on the log",
        "the dog ate the bone",
    ]
    N = 3  # trigram: context length n-1 = 2
    corpus = [tokenize(s) for s in corpus_text]
    vocab = set(tok for sent in corpus for tok in sent)
    V = len(vocab)

    ngram_c, context_c = build_ngram_counts(corpus, N)

    test = tokenize("the cat sat on the bone")
    ppl = perplexity(test, ngram_c, context_c, V, N)
    print(f"Trigram model vocabulary size: {V}")
    print(f"Test sentence: 'the cat sat on the bone'")
    print(f"Perplexity: {ppl:.2f}")

    # Bigram (n=2) probabilities illustrate P(w_t | w_{t-1}) on the same corpus
    bi_c, bi_ctx = build_ngram_counts(corpus, 2)
    for ctx_word, next_word in [("the", "cat"), ("the", "dog"), ("cat", "sat")]:
        p = ngram_probability(next_word, (ctx_word,), bi_c, bi_ctx, V)
        print(f"  P({next_word} | {ctx_word}) = {p:.4f}")
```

---

## Interview Guide

### What Interviewers Actually Ask

!!! interview "FAANG-Level Questions"
    **1. State the chain rule for language modeling and name one training objective that implements it.**

    **Full Answer:** The chain rule decomposes the joint probability of a token sequence into a product of conditional probabilities:

    \\(P(w_1,\\ldots,w_T) = \\prod_{t=1}^{T} P(w_t \\mid w_1,\\ldots,w_{t-1})\\)

    This says: "the probability of a full sentence equals the product of each word's probability given everything before it." Every autoregressive language model — from bigrams to GPT-4 — is an instantiation of this factorization. The only difference is **how they parameterize the conditional**: n-grams use count tables, RNNs use recurrent hidden states, and Transformers use stacked self-attention.

    The standard training objective that implements this is **next-token cross-entropy**: minimize \\(-\\frac{1}{T}\\sum_t \\log P_\\theta(w_t \\mid w_{<t})\\). This is mathematically identical to maximum likelihood estimation (MLE) on the training corpus and also identical to minimizing the KL divergence between the data distribution and the model distribution.

    **What the interviewer is testing:** Can you connect the abstract factorization to the concrete loss function? The red flag is if you can only recite the formula but can't explain why cross-entropy training IS this factorization in action.

    ---

    **2. What is the Markov assumption in an n-gram model?**

    **Full Answer:** The Markov assumption states that the probability of the next token depends only on the **last \\(n-1\\) tokens**, not the entire history:

    \\(P(w_t \\mid w_1,\\ldots,w_{t-1}) \\approx P(w_t \\mid w_{t-n+1},\\ldots,w_{t-1})\\)

    For a bigram (\\(n=2\\)), only the immediately preceding word matters. For a trigram (\\(n=3\\)), only the last two words matter. This is a deliberate simplification that makes counting feasible — you can tabulate all possible \\(n\\)-grams — but it sacrifices the ability to model long-range dependencies. English sentences like "The **cat** that I saw at the park yesterday **was** orange" require the verb to agree with a subject 8 words back, which a trigram model cannot capture.

    **Why it matters for modern LLMs:** Transformers don't make the Markov assumption — they attend to the full context window (e.g., 128K tokens). But they still face a practical version of the problem: the context window is finite, and the quadratic cost of attention constrains how far back the model can "see." Understanding the Markov assumption helps you articulate why context length improvements (RoPE scaling, efficient attention) are research priorities.

    ---

    **3. Why does perplexity decrease when the model improves?**

    **Full Answer:** Perplexity = \\(\\exp(\\text{average cross-entropy})\\). When the model improves, it assigns **higher probabilities** to the correct next tokens. Higher probability → lower \\(-\\log P\\) (less surprise) → lower average cross-entropy → lower perplexity after exponentiation.

    Concretely: if a model improves its average per-token probability from 0.05 to 0.10, the surprise drops from \\(-\\ln(0.05) = 3.0\\) to \\(-\\ln(0.10) = 2.3\\) nats, and perplexity drops from \\(e^3 ≈ 20\\) to \\(e^{2.3} ≈ 10\\). The model went from "choosing among ~20 options" to "choosing among ~10 options" — more focused, less confused.

    **Caveat to mention:** Perplexity is only comparable between models using the **same tokenizer on the same test set**. A BPE tokenizer with 50K vocabulary and a character-level tokenizer will produce very different perplexity numbers even for the same model quality, because the "decisions per token" differ.

    ---

    **4. Why must you smooth n-gram counts?**

    **Full Answer:** Because finite training data yields **zero counts** for unseen n-grams, and a single zero in the chain rule product zeros out the **entire sentence probability**. If the test sentence contains "cat flies" and the training data never had that bigram, \\(P(\\text{flies} \\mid \\text{cat}) = 0/\\text{count}(\\text{cat}) = 0\\), making \\(P(\\text{entire sentence}) = 0\\) and perplexity = infinity.

    This isn't just a theoretical problem — it's extremely common. For vocabulary \\(V = 50,000\\), there are \\(V^5 = 3.125 \\times 10^{23}\\) possible 5-grams. Even with billions of training tokens, the vast majority of 5-grams will never appear. Smoothing (add-k, Kneser-Ney, interpolation) redistributes small amounts of probability mass from seen n-grams to unseen ones, ensuring zero probabilities never occur.

    **Neural angle:** Neural language models "smooth implicitly" through continuous word embeddings — similar words share parameters and gradient updates. This is a key advantage over n-grams: the model can generalize from "cat sat" to "dog sat" because `cat` and `dog` have similar embedding vectors, whereas in an n-gram table these are completely unrelated entries.

    ---

    **5. Compare unigram, bigram, and trigram models in one sentence each.**

    **Full Answer:**
    - **Unigram** (\\(n=1\\)): Predicts the next word based on overall word frequency alone, ignoring all context — essentially a bag-of-words frequency table. Very high bias, zero variance from context.
    - **Bigram** (\\(n=2\\)): Uses one word of context to predict the next word — captures simple local patterns ("New York", "ice cream") but misses anything beyond adjacent pairs.
    - **Trigram** (\\(n=3\\)): Uses two words of context — captures short phrases and common patterns ("New York City", "on the other") but requires \\(V^3\\) possible entries and most remain unseen.

    The trade-off is **expressiveness vs. data sparsity**: higher \\(n\\) models can capture richer patterns but need exponentially more data to estimate reliably. This is exactly why neural models won — they break the exponential dependence on \\(n\\) by using continuous representations.

    ---

    **6. How does perplexity relate to cross-entropy?**

    **Full Answer:** Perplexity is the **exponential of cross-entropy**: \\(\\text{PPL} = \\exp(H)\\) where \\(H\\) is the average per-token cross-entropy. If using \\(\\log_2\\), then \\(\\text{PPL} = 2^H\\); if using \\(\\ln\\), then \\(\\text{PPL} = e^H\\). The key insight is that cross-entropy measures average surprise in nats (or bits), while perplexity converts that into an interpretable "effective branching factor."

    This means a **1-nat decrease** in cross-entropy translates to a **factor-of-\\(e \\approx 2.72\\)** reduction in perplexity. Small cross-entropy improvements compound into significant perplexity drops, which is why researchers track cross-entropy loss curves obsessively during pretraining.

    ---

    **7. What is autoregressive factorization?**

    **Full Answer:** Autoregressive factorization is exactly the chain rule decomposition: generate tokens left-to-right, where each token is conditioned only on previously generated (or observed) tokens. The term "autoregressive" emphasizes that the model's own outputs feed back as inputs for subsequent predictions. GPT-style models are autoregressive by design (causal masking enforces this during training). BERT is NOT autoregressive — it uses masked language modeling where predictions are conditioned on both left and right context. T5's decoder is autoregressive even though its encoder sees bidirectional context.

    ---

    **8. Why are subword tokenizers related to n-gram sparsity?**

    **Full Answer:** Word-level tokenization creates massive vocabularies (100K+ word types in English) where most entries are rare. Bigram tables over 100K words have \\(10^{10}\\) possible entries — hopelessly sparse. Subword tokenizers (BPE, SentencePiece, WordPiece) break rare words into frequent character chunks: "unhappiness" → "un", "happi", "ness". This shrinks the vocabulary to 30K–50K tokens while ensuring every token occurs frequently enough for reliable probability estimation. The same subword units appear across many words, sharing statistical strength — "un" helps predict "undo", "unwind", "unlikely" collectively rather than treating each as an isolated event. This is essentially **built-in smoothing** at the tokenization level.

!!! interview "Follow-up Probes"
    1. If a bigram never appeared in training, what happens to ML estimates without smoothing?
    2. Why is perplexity not comparable across different tokenizers or vocabularies?
    3. How does interpolation combine unigram, bigram, and trigram scores?
    4. What is backoff versus interpolation in intuitive terms?
    5. Why is joint probability of a long sentence usually represented in log space in code?

!!! key-phrases "Key Phrases to Use in Interviews"
    - "Autoregressive factorization via the chain rule"
    - "Markov assumption of order \\(n-1\\)"
    - "Maximum-likelihood n-gram estimates"
    - "Add-k (Laplace) smoothing"
    - "Perplexity as exponential cross-entropy"
    - "Data sparsity in high-order n-grams"

---

## References

- Jurafsky & Martin, *Speech and Language Processing*, Chapter on N-gram Language Models
- Shannon (1951), "Prediction and Entropy of Printed English"
- [Stanford CS224N — Language Modeling lecture](https://web.stanford.edu/class/cs224n/)
