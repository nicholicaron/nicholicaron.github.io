---
layout: post
title: "Take Heed of the Hydra"
date: 2023-06-18
tags: [Transformers, AI, Math]
cover_image: /assets/images/multihead_attn/Hydra_ss-3463311282.jpg
---

Implementing Multi-Head Attention and Optimized Matrix Multiplication via Strassen's Algorithm. This post walks through the transformer's core mechanism at the matrix level — not as a black box, but as a series of linear algebra operations you can trace by hand.

Before we get started I have to apologize: firstly for the pun (there are only so many Multi-Head Attention puns available) and secondly for getting nerd sniped once again — I'll finish up the compiler series soon enough, I promise.

## What This Post Covers

- **Scaled Dot-Product Attention** — The QKV formulation, why we scale by $\sqrt{d_k}$, and the softmax bottleneck
- **Multi-Head Attention** — Parallel attention heads, concatenation, and the projection back to model dimension
- **Strassen's Algorithm** — Asymptotically faster matrix multiplication and its implications for attention computation
- **Implementation** — Building multi-head attention from scratch in PyTorch, with a focus on computational efficiency

---

## Why the Hydra?

The mythological Hydra — a creature with many heads, each operating independently but serving a single body — is a fitting metaphor for multi-head attention. Each attention head learns to focus on different aspects of the input: syntactic relationships, semantic similarity, positional patterns. The power comes from the diversity of perspectives, unified through concatenation and projection.

We may be standing at the single greatest lever point in human history. As cliche as it sounds I fundamentally believe that the repercussions of AI on society will rival the internet before it's all said and done. If computers are bicycles for the mind, Deep Learning methods are performance enhancing drugs — the average person now becomes Lance Armstrong. The Machine Learning industry has had a seasonal past. Public sentiment has oscillated between "the machines will kill us all" and "AI is a pipe dream". We're currently in the heat of an AI summer, thanks to consumer-facing products such as ChatGPT, and the robust developer ecosystem that has developed around Natural Language Processing, Image Classification and Generation, etc. I believe that we're at an inflection point in the pace of development, accelerating towards the singularity.

> "Midas was wrong, everything we touch turns to silicon"
> — George Hotz

But enough of the melodrama — let's get down to the nitty gritty. For the sake of precision, let's get some terminology out of the way. I've been using AI, Machine Learning, and Deep Learning interchangeably so far. There is a distinction between them though:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 520 320" width="520" height="320" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'JetBrains Mono', monospace; max-width: 100%;">
  <!-- Outermost: AI -->
  <ellipse cx="260" cy="165" rx="248" ry="148" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.4"/>
  <text x="480" y="55" text-anchor="end" fill="currentColor" font-size="13" font-weight="600">AI</text>
  <text x="480" y="72" text-anchor="end" fill="currentColor" font-size="9.5" opacity="0.55">e.g. Knowledge Bases</text>
  <!-- ML -->
  <ellipse cx="230" cy="175" rx="185" ry="115" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.55"/>
  <text x="395" y="120" text-anchor="end" fill="currentColor" font-size="13" font-weight="600">Machine Learning</text>
  <text x="395" y="137" text-anchor="end" fill="currentColor" font-size="9.5" opacity="0.55">e.g. Logistic Regression</text>
  <!-- Representation Learning -->
  <ellipse cx="200" cy="185" rx="130" ry="85" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.7"/>
  <text x="265" y="265" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">Representation</text>
  <text x="265" y="280" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">Learning</text>
  <text x="265" y="296" text-anchor="middle" fill="currentColor" font-size="9.5" opacity="0.55">e.g. Autoencoders</text>
  <!-- Innermost: Deep Learning -->
  <ellipse cx="165" cy="180" rx="75" ry="55" fill="none" stroke="currentColor" stroke-width="2" opacity="0.9"/>
  <text x="165" y="175" text-anchor="middle" fill="currentColor" font-size="12" font-weight="700">Deep</text>
  <text x="165" y="192" text-anchor="middle" fill="currentColor" font-size="12" font-weight="700">Learning</text>
  <text x="165" y="208" text-anchor="middle" fill="currentColor" font-size="9.5" opacity="0.55">e.g. MLPs</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The nested relationship between AI, Machine Learning, Representation Learning, and Deep Learning.</p>

From here on out I'll focus on Deep Learning as that is where most of the recent progress has been made. It's important to emphasize that Deep Learning is not a silver bullet — depending on the context, Machine Learning methods may be much more efficient or preferable. For example, despite all the exotic new Deep Learning architectures, banks still use Decision Forests and the like to determine whether to loan an individual money. Due to regulations, they have to be able to explain *why* an individual was turned down. Neural Networks (Deep Learning) are often thought of as "black boxes" and lack the level of interpretability necessary.

The seed that sparked the recent hype around the space was the advent of the Transformer architecture as described in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Transformers first saw rapid and extensive adoption in the Natural Language Processing space, secondly in Computer Vision, and have shot out from there. While we won't dive into the full transformer architecture — you'd be better off just reading the paper for that — we will give a thorough treatment to the attention mechanism and try to implement something similar to how it looks behind the scenes.

> The Transformer is a magnificent neural network architecture because it is a general-purpose differentiable computer. It is simultaneously: (1) expressive in the forward pass, (2) optimizable via backpropagation and gradient descent, (3) efficient as a high parallelism compute graph.
> — Andrej Karpathy

---

## Attention: The Secret Sauce

As previously noted by Phil Karlton, one of the hardest problems in Computer Science is naming things. The Attention mechanism is aptly named. Attention in the context of Deep Learning closely mirrors our everyday meaning of the word. It allows a system to keep both previous input and previous output in context when generating new output. Furthermore, through training a Transformer model, the model learns how to rank the components of a state's context in terms of relevance when predicting the next word or token.

Take, for example, a sentence like *"My dog hardly ate today, I hope he isn't sick."* If we couldn't confidently reference the past context, we'd be stumped. Who is "he"? What makes you think "he" might be sick? Long-range dependencies are key to understanding the meaning and structure of language and then acting on that understanding (e.g. through response or translation) in a human-like fashion.

Machine Learning allows us to learn from and make actionable conclusions based on real-world data. Just as Physics describes formulas that represent reality and can be used to predict future states of a given physical system, Machine Learning practitioners believe that many real-world systems subscribe to similar mappings from input to output. Often these functions are too complex to be determined analytically, but as determined by the **Universal Approximation Theorem**, Neural Networks are capable of approximating any function to arbitrary precision.

Linear Algebra plays a fundamental role in Machine Learning systems. The goal of ML is to approximate and exploit the intrinsic low-rank feature spaces of a given data set. We gather high-dimensional, noisy data, then try to determine and represent the defining pattern — the signal in the noise. Having a good intuition for vector spaces, matrices as linear transformations, and change of basis is invaluable for understanding these systems. Each ML algorithm has a different way of finding optimal embeddings for the data that are both informative and interpretable. In Deep Learning, Neural Networks are composed of several layers of Neurons. Each layer is represented by a matrix whose elements — called **weights** — are iteratively learned by our model.

### Why Not RNNs?

**Recurrent Neural Networks** were the previous state-of-the-art approach in NLP. RNNs used sequential methods to gather context: an input sequence was processed incrementally according to the order of words. Hidden states are computed recursively — each hidden state $h_t$ is generated as a function of the previous hidden state and the current input:

$$h_t = f(h_{t-1}, x_t)$$

While recursion often provides elegant solutions, this creates a fundamental bottleneck: **you can't compute $h_t$ until you've computed $h_{t-1}$**. This sequential dependency chain means that processing a sequence of length $n$ requires $n$ sequential steps, regardless of how many GPUs you throw at it. It also puts a hard cap on sequence length, since the hidden state has to compress the entire history into a fixed-size vector — information from early tokens gets diluted as the sequence grows.

The beauty of GPUs is that they are massively parallel machines optimized to perform matrix multiplication. A crude simplification of Deep Learning is that it is a way of brute-forcing matrix multiplications to solve an optimization problem. The parallelizability of the Attention architecture is precisely what brought Transformers to the forefront. Instead of processing tokens one at a time, attention computes **all pairwise relationships simultaneously**. Given an input sequence of $n$ tokens, the attention matrix $QK^T$ is an $n \times n$ matrix where every entry is computed independently — a single batched matrix multiplication that GPUs devour.

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."
> — Richard Sutton

---

## Scaled Dot-Product Attention

Let's formalize the attention mechanism. An attention function maps a query vector $Q$ and a set of key-value pair vectors $(K, V)$ to an output vector. There are multiple variants, but we'll focus on **scaled dot-product attention** since the dot product is fast and space-efficient:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 220 380" width="220" height="380" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'JetBrains Mono', monospace; max-width: 100%;">
  <defs>
    <marker id="arrow-attn" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="currentColor"/>
    </marker>
  </defs>
  <!-- Output -->
  <text x="110" y="20" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">Output</text>
  <line x1="110" y1="28" x2="110" y2="45" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 110, 36)"/>
  <!-- MatMul (top) -->
  <rect x="55" y="45" width="110" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="110" y="65" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">MatMul</text>
  <line x1="110" y1="75" x2="110" y2="100" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 110, 87)"/>
  <!-- V input to top MatMul -->
  <line x1="185" y1="350" x2="185" y2="60" stroke="currentColor" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.5"/>
  <line x1="185" y1="60" x2="165" y2="60" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)"/>
  <!-- SoftMax -->
  <rect x="55" y="100" width="110" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
  <text x="110" y="120" text-anchor="middle" fill="currentColor" font-size="11">SoftMax</text>
  <line x1="110" y1="130" x2="110" y2="155" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 110, 142)"/>
  <!-- Mask (opt.) -->
  <rect x="55" y="155" width="110" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.6"/>
  <text x="110" y="175" text-anchor="middle" fill="currentColor" font-size="11" opacity="0.7">Mask (opt.)</text>
  <line x1="110" y1="185" x2="110" y2="210" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 110, 197)"/>
  <!-- Scale -->
  <rect x="55" y="210" width="110" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
  <text x="110" y="230" text-anchor="middle" fill="currentColor" font-size="11">Scale (1/√d<tspan baseline-shift="sub" font-size="8">k</tspan>)</text>
  <line x1="110" y1="240" x2="110" y2="265" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 110, 252)"/>
  <!-- MatMul (bottom) -->
  <rect x="55" y="265" width="110" height="30" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="110" y="285" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">MatMul</text>
  <!-- Q, K, V inputs -->
  <line x1="80" y1="295" x2="80" y2="340" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 80, 317)"/>
  <line x1="140" y1="295" x2="140" y2="340" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)" transform="rotate(180, 140, 317)"/>
  <text x="50" y="358" text-anchor="middle" fill="currentColor" font-size="13" font-weight="700">Q</text>
  <text x="110" y="358" text-anchor="middle" fill="currentColor" font-size="13" font-weight="700">K</text>
  <text x="185" y="358" text-anchor="middle" fill="currentColor" font-size="13" font-weight="700">V</text>
  <!-- K connects to bottom matmul -->
  <line x1="110" y1="340" x2="140" y2="295" stroke="currentColor" stroke-width="1.2" opacity="0.5"/>
  <!-- Q connects to bottom matmul -->
  <line x1="50" y1="340" x2="50" y2="280" stroke="currentColor" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.5"/>
  <line x1="50" y1="280" x2="55" y2="280" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-attn)"/>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Scaled Dot-Product Attention: Q and K are multiplied, scaled, optionally masked, passed through softmax, then multiplied by V.</p>

Let's break this down step by step:

**Step 1: Compute similarity scores.** Multiply $Q$ by $K^T$. The superscript $T$ denotes the **transpose** — rows become columns, so a row vector $(1 \times n)$ becomes a column vector $(n \times 1)$. We transpose because matrix multiplication has stricter rules than scalar multiplication: the inner dimensions must match. The result $QK^T$ is an $n \times n$ matrix where each entry $(i, j)$ measures how much query $i$ should attend to key $j$.

**Step 2: Scale.** Divide element-wise by $\sqrt{d_k}$, where $d_k$ is the dimensionality of the key vectors. Why? Without scaling, when $d_k$ is large, the dot products grow large in magnitude, pushing the softmax into regions where it has extremely small gradients. The scaling keeps the variance of the dot products at approximately 1, keeping gradients healthy.

**Step 3: Mask (optional).** In the decoder, we need to prevent positions from attending to subsequent positions — the model shouldn't peek at the future when predicting the next token. We set these positions to $-\infty$ before softmax, which drives them to zero.

**Step 4: Softmax.** Apply softmax row-wise to convert raw scores into a probability distribution. Each row now sums to 1, giving us attention weights.

**Step 5: Weighted sum.** Multiply the attention weights by $V$. Each output position is now a weighted combination of all value vectors, where the weights reflect relevance.

---

## Matrices as Weighted Aggregation

Here's the deep insight that makes attention click: **matrix multiplication is weighted aggregation**. Let's build this intuition from scratch.

How would you contextualize some ordered set of numbers — say $2, 4, 6, 8, 10$ — to guess what comes next? One idea is to use a series of averages as a hint. Split the set into subsets to get several examples:

| Input | Next Element |
|-------|-------------|
| $[2]$ | $4$ |
| $[2, 4]$ | $6$ |
| $[2, 4, 6]$ | $8$ |
| $[2, 4, 6, 8]$ | $10$ |

If we didn't have the human intuition to recognize that we're dealing with the even integers, we could notice that the pattern seems to be: calculate the average value, then multiply it by two. That's a lossy compression of the underlying distribution — the average only conveys so much information — but it's a compression nonetheless.

The key to self-attention is the extension of this idea. Take a lower triangular matrix filled with ones and multiply it by a data matrix:

$$\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \end{bmatrix} \times \begin{bmatrix} 2 & 8 & 14 \\ 4 & 10 & 16 \\ 6 & 12 & 18 \end{bmatrix} = \begin{bmatrix} 2 & 8 & 14 \\ 6 & 18 & 30 \\ 12 & 30 & 48 \end{bmatrix}$$

This operation sums the elements of each column from top to bottom — it's a **cumulative sum**. Now, if we normalize each row of $A$ so that it sums to 1:

$$\begin{bmatrix} 1 & 0 & 0 \\ 1/2 & 1/2 & 0 \\ 1/3 & 1/3 & 1/3 \end{bmatrix} \times \begin{bmatrix} 2 & 8 & 14 \\ 4 & 10 & 16 \\ 6 & 12 & 18 \end{bmatrix} = \begin{bmatrix} 2 & 8 & 14 \\ 3 & 9 & 15 \\ 4 & 10 & 16 \end{bmatrix}$$

We get an **incremental moving average** as we go down the columns. Notice how this provides a decent representation for predicting the next element in column 1 (where the pattern is "even numbers in ascending order") — the average tracks the trend. But it's a one-size-fits-all compression. Different columns might have different patterns that require different weightings.

This is exactly what attention learns: **data-dependent weights**. Instead of fixed uniform averages, the model learns to assign weights based on the actual content of each position — some tokens are more relevant than others for predicting the next one.

### Masking and Softmax

In Transformer networks, we use the weight matrix as our set of learned attention weights. To avoid letting the model cheat by looking at future tokens during training, we initialize the weights carefully:

```python
tril = torch.tril(torch.ones(N, N))  # Lower triangular matrix of ones
weights = torch.zeros((N, N))         # Initialize to zeros
weights = weights.masked_fill(tril == 0, float('-inf'))  # Future positions → -∞
weights = F.softmax(weights, dim=-1)  # Softmax across rows
```

This seems like an overly complex way to initialize weights, but there's good reason. By setting future positions to $-\infty$ before softmax (rather than just using zeros), we ensure that those positions have **exactly zero** attention weight after softmax — they contribute nothing to the output. If we just used zeros, those positions would get small but nonzero attention weights ($\text{softmax}(0) > 0$), and worse, gradient descent could nudge them during training, allowing the model to "leak" information from the future.

Through **autodifferentiation** (see my [previous post on AD](/2023/06/15/autodifferentiation.html)), we iteratively tune these weights in search of an optimal compression of our input that produces a good probability distribution for prediction.

---

## Matrix Multiplication: The Computational Bottleneck

Let's take a step back and think about the computational cost of all these matrix multiplications. For two $n \times n$ matrices, the **naive algorithm** computes each entry of the product as a dot product of a row and a column:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

For two $2 \times 2$ matrices with elements:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \times \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix}$$

That's 8 scalar multiplications and 4 additions. In general: $n^3$ multiplications and $n^3 - n^2$ additions. Since multiplications are more expensive, we say this is $O(n^3)$. The triple-nested loop makes this immediately apparent:

```python
def matmul(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]
```

Most examples we can wrap our heads around are trivial in size to a computer. But imagine realistic matrices: the average monitor resolution of $1920 \times 1080$ has just over 2 million pixels — for a *grayscale* image. RGB triples that to 6 million. Now imagine training a computer vision model on a modern dataset like JFT-300M (that the original Vision Transformer was trained on) and multiply by 300 million. We're looking at $\sim 1.8 \times 10^{15}$ elements, with an untold number of matrix multiplications on each.

We have to find ways to improve tractability. We can compress the data, choose a parallelizable architecture like the Transformer, and we can also look for more efficient matrix multiplication algorithms.

---

## Strassen's Algorithm

The first major advancement in efficient matrix multiplication came in 1969 from Volker Strassen. The key idea is **divide and conquer**: recursively partition the factor matrices into sub-blocks and combine them cleverly.

For two $2 \times 2$ matrices, partition each into four scalars (or, for larger matrices, into equally-sized sub-matrices). Then define seven intermediate products:

$$\begin{aligned}
M_1 &= (A_{11} + A_{22})(B_{11} + B_{22}) \\
M_2 &= (A_{21} + A_{22}) \cdot B_{11} \\
M_3 &= A_{11} \cdot (B_{12} - B_{22}) \\
M_4 &= A_{22} \cdot (B_{21} - B_{11}) \\
M_5 &= (A_{11} + A_{12}) \cdot B_{22} \\
M_6 &= (A_{21} - A_{11})(B_{11} + B_{12}) \\
M_7 &= (A_{12} - A_{22})(B_{21} + B_{22})
\end{aligned}$$

Then assemble the result:

$$\begin{aligned}
C_{11} &= M_1 + M_4 - M_5 + M_7 \\
C_{12} &= M_3 + M_5 \\
C_{21} &= M_2 + M_4 \\
C_{22} &= M_1 - M_2 + M_3 + M_6
\end{aligned}$$

The reader can verify that this produces the same result as the naive algorithm. The critical insight: **7 multiplications instead of 8**. We introduced more additions, but since multiplications dominate the cost, this is a net win. The complexity drops from $O(n^3)$ to $O(n^{\log_2 7}) \approx O(n^{2.807})$.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 560 260" width="560" height="260" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'JetBrains Mono', monospace; max-width: 100%;">
  <!-- A matrix -->
  <rect x="10" y="30" width="80" height="80" fill="none" stroke="currentColor" stroke-width="1.5" rx="3"/>
  <line x1="50" y1="30" x2="50" y2="110" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <line x1="10" y1="70" x2="90" y2="70" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <text x="30" y="57" text-anchor="middle" fill="currentColor" font-size="11">A₁₁</text>
  <text x="70" y="57" text-anchor="middle" fill="currentColor" font-size="11">A₁₂</text>
  <text x="30" y="97" text-anchor="middle" fill="currentColor" font-size="11">A₂₁</text>
  <text x="70" y="97" text-anchor="middle" fill="currentColor" font-size="11">A₂₂</text>
  <text x="50" y="135" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">A</text>
  <!-- × -->
  <text x="110" y="75" text-anchor="middle" fill="currentColor" font-size="18" opacity="0.6">×</text>
  <!-- B matrix -->
  <rect x="130" y="30" width="80" height="80" fill="none" stroke="currentColor" stroke-width="1.5" rx="3"/>
  <line x1="170" y1="30" x2="170" y2="110" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <line x1="130" y1="70" x2="210" y2="70" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <text x="150" y="57" text-anchor="middle" fill="currentColor" font-size="11">B₁₁</text>
  <text x="190" y="57" text-anchor="middle" fill="currentColor" font-size="11">B₁₂</text>
  <text x="150" y="97" text-anchor="middle" fill="currentColor" font-size="11">B₂₁</text>
  <text x="190" y="97" text-anchor="middle" fill="currentColor" font-size="11">B₂₂</text>
  <text x="170" y="135" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">B</text>
  <!-- Arrow -->
  <text x="240" y="75" text-anchor="middle" fill="currentColor" font-size="16" opacity="0.6">→</text>
  <!-- M products -->
  <text x="290" y="25" text-anchor="start" fill="currentColor" font-size="12" font-weight="700">7 Products:</text>
  <text x="290" y="48" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₁ = (A₁₁+A₂₂)(B₁₁+B₂₂)</text>
  <text x="290" y="66" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₂ = (A₂₁+A₂₂)·B₁₁</text>
  <text x="290" y="84" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₃ = A₁₁·(B₁₂−B₂₂)</text>
  <text x="290" y="102" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₄ = A₂₂·(B₂₁−B₁₁)</text>
  <text x="290" y="120" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₅ = (A₁₁+A₁₂)·B₂₂</text>
  <text x="290" y="138" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₆ = (A₂₁−A₁₁)(B₁₁+B₁₂)</text>
  <text x="290" y="156" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">M₇ = (A₁₂−A₂₂)(B₂₁+B₂₂)</text>
  <!-- Arrow -->
  <text x="240" y="210" text-anchor="middle" fill="currentColor" font-size="16" opacity="0.6">→</text>
  <!-- Result -->
  <text x="290" y="190" text-anchor="start" fill="currentColor" font-size="12" font-weight="700">Assembly:</text>
  <text x="290" y="213" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">C₁₁ = M₁ + M₄ − M₅ + M₇</text>
  <text x="290" y="231" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">C₁₂ = M₃ + M₅</text>
  <text x="290" y="249" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8">C₂₁ = M₂ + M₄</text>
  <text x="290" y="267" text-anchor="start" fill="currentColor" font-size="10" opacity="0.8" font-weight="400">C₂₂ = M₁ − M₂ + M₃ + M₆</text>
  <!-- C matrix -->
  <rect x="70" y="175" width="80" height="80" fill="none" stroke="currentColor" stroke-width="1.5" rx="3"/>
  <line x1="110" y1="175" x2="110" y2="255" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <line x1="70" y1="215" x2="150" y2="215" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <text x="90" y="202" text-anchor="middle" fill="currentColor" font-size="11">C₁₁</text>
  <text x="130" y="202" text-anchor="middle" fill="currentColor" font-size="11">C₁₂</text>
  <text x="90" y="242" text-anchor="middle" fill="currentColor" font-size="11">C₂₁</text>
  <text x="130" y="242" text-anchor="middle" fill="currentColor" font-size="11">C₂₂</text>
  <text x="110" y="270" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">C = A × B</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Strassen's algorithm: partition A and B into sub-blocks, compute 7 intermediate products, assemble the 4 quadrants of C.</p>

For the $2 \times 2$ case, we dealt with $1 \times 1$ single-element sub-matrices. What about an $8 \times 8$? We'd have sub-matrices of size $4 \times 4$. For each of those, we break them down further into $2 \times 2$ sub-matrices, and so on. Isn't that beautiful? There's something elegant and pleasing about recursive algorithms. I think it's the same self-referential nature that makes fractals and inductive proofs so nice.

In practice, Strassen's algorithm has a crossover point — for small matrices, the overhead of the extra additions and recursive calls outweighs the savings from fewer multiplications. Most implementations switch to the naive algorithm below a threshold (typically around $n = 64$). There are also numerical stability concerns since the additions and subtractions can accumulate floating-point errors.

### A Brief Aside on Language Performance

Before getting into the implementation, let's think about the language we're using. Python is great for developer productivity, and as Tony Hoare said: "premature optimization is the root of all evil in programming." Once we have something that works, *then* we optimize. But Python struggles with raw performance.

Enter **Mojo** — a language that combines the usability of Python with the performance of C, by Chris Lattner (creator of Swift, LLVM, etc.). Mojo is a superset of Python with features like progressive types, zero-cost abstractions, and an ownership/borrow checker (makes me smile as a part-time Rustacean). Switching to Mojo for matrix multiplications can yield a speedup of 17.5x out of the box, and up to 14,050x if you optimize aggressively. Worth keeping an eye on.

But since the reader is more likely to be familiar with Python, we'll focus on improving the algorithm itself.

---

## Multi-Head Attention

Now we return to the Hydra. A single attention head computes one set of attention weights — one "perspective" on the input. But language has many simultaneous structures: syntactic dependencies, semantic similarity, coreference patterns, positional relationships. A single attention function can't capture all of these effectively.

**Multi-head attention** runs $h$ attention heads in parallel, each with its own learned projections, then concatenates and projects the results:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \, W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(Q W_i^Q, \, K W_i^K, \, V W_i^V)$$

The projection matrices $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ are all learned parameters. The key design choice is setting $d_k = d_v = d_{\text{model}} / h$, so that the total computational cost is similar to single-head attention with full dimensionality.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 440 360" width="440" height="360" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'JetBrains Mono', monospace; max-width: 100%;">
  <defs>
    <marker id="arrow-mha" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="currentColor"/>
    </marker>
  </defs>
  <!-- Output -->
  <text x="220" y="20" text-anchor="middle" fill="currentColor" font-size="12" font-weight="600">Output</text>
  <line x1="220" y1="28" x2="220" y2="42" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 220, 35)"/>
  <!-- Linear (final) -->
  <rect x="155" y="42" width="130" height="28" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="220" y="61" text-anchor="middle" fill="currentColor" font-size="11" font-weight="600">Linear (W^O)</text>
  <line x1="220" y1="70" x2="220" y2="88" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 220, 79)"/>
  <!-- Concat -->
  <rect x="135" y="88" width="170" height="28" rx="4" fill="none" stroke="currentColor" stroke-width="1.5" opacity="0.8"/>
  <text x="220" y="107" text-anchor="middle" fill="currentColor" font-size="11">Concat</text>
  <!-- Lines from concat to heads -->
  <line x1="160" y1="116" x2="90" y2="145" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 125, 130)"/>
  <line x1="220" y1="116" x2="220" y2="145" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 220, 130)"/>
  <line x1="280" y1="116" x2="350" y2="145" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 315, 130)"/>
  <!-- Head 1 -->
  <rect x="40" y="145" width="100" height="45" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="90" y="164" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Scaled Dot-</text>
  <text x="90" y="178" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Product Attn</text>
  <text x="90" y="205" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.6">head₁</text>
  <!-- Head 2 -->
  <rect x="170" y="145" width="100" height="45" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="220" y="164" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Scaled Dot-</text>
  <text x="220" y="178" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Product Attn</text>
  <text x="220" y="205" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.6">head₂</text>
  <!-- Head h -->
  <rect x="300" y="145" width="100" height="45" rx="4" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="350" y="164" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Scaled Dot-</text>
  <text x="350" y="178" text-anchor="middle" fill="currentColor" font-size="10" font-weight="600">Product Attn</text>
  <text x="350" y="205" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.6">head_h</text>
  <!-- Dots between head 2 and head h -->
  <text x="280" y="170" text-anchor="middle" fill="currentColor" font-size="14" opacity="0.4">···</text>
  <!-- Lines from heads down to linear projections -->
  <line x1="90" y1="190" x2="90" y2="218" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 90, 204)"/>
  <line x1="220" y1="190" x2="220" y2="218" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 220, 204)"/>
  <line x1="350" y1="190" x2="350" y2="218" stroke="currentColor" stroke-width="1.2" marker-end="url(#arrow-mha)" transform="rotate(180, 350, 204)"/>
  <!-- Linear layers -->
  <rect x="45" y="218" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1" opacity="0.7"/>
  <text x="90" y="233" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.7">Linear × 3</text>
  <rect x="175" y="218" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1" opacity="0.7"/>
  <text x="220" y="233" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.7">Linear × 3</text>
  <rect x="305" y="218" width="90" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="1" opacity="0.7"/>
  <text x="350" y="233" text-anchor="middle" fill="currentColor" font-size="9" opacity="0.7">Linear × 3</text>
  <!-- Lines from linear to V, K, Q -->
  <line x1="70" y1="240" x2="70" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="90" y1="240" x2="90" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="110" y1="240" x2="110" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="200" y1="240" x2="200" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="220" y1="240" x2="220" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="240" y1="240" x2="240" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="330" y1="240" x2="330" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="350" y1="240" x2="350" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <line x1="370" y1="240" x2="370" y2="270" stroke="currentColor" stroke-width="1" opacity="0.5"/>
  <!-- Converge all to V, K, Q -->
  <line x1="70" y1="270" x2="120" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="200" y1="270" x2="200" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="330" y1="270" x2="290" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="90" y1="270" x2="170" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="220" y1="270" x2="220" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="350" y1="270" x2="310" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="110" y1="270" x2="150" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="240" y1="270" x2="240" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <line x1="370" y1="270" x2="340" y2="300" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <!-- V, K, Q labels -->
  <text x="120" y="330" text-anchor="middle" fill="currentColor" font-size="14" font-weight="700">V</text>
  <text x="220" y="330" text-anchor="middle" fill="currentColor" font-size="14" font-weight="700">K</text>
  <text x="320" y="330" text-anchor="middle" fill="currentColor" font-size="14" font-weight="700">Q</text>
  <!-- h label -->
  <text x="420" y="170" text-anchor="start" fill="currentColor" font-size="11" opacity="0.5">×h</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Multi-Head Attention: V, K, Q are each linearly projected h times, attention is applied in parallel, and the results are concatenated and projected.</p>

Each head sees the input through different learned projections — different "lenses." One head might learn to attend to syntactic structure (subject-verb agreement), another to semantic similarity (synonyms and related concepts), another to positional patterns (nearby tokens). The concatenation and final linear projection fuse these perspectives into a single rich representation.

---

## Implementation

Let's build multi-head attention from scratch in PyTorch, then implement Strassen's algorithm.

### Multi-Head Attention in PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Learned projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Project and reshape: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention in parallel across all heads
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads: (batch, n_heads, seq_len, d_k) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final linear projection
        return self.W_o(attn_output)
```

The `view` and `transpose` operations are doing the heavy lifting. Rather than creating $h$ separate attention modules (wasteful), we project to the full $d_{\text{model}}$ dimension, then *reshape* to split the last dimension into $h$ heads of size $d_k$. The attention computation then broadcasts across the head dimension — all heads run in a single batched matrix multiplication. After attention, we reverse the reshaping and apply the final projection $W^O$.

### Strassen's Algorithm in Python

```python
import numpy as np

def strassen(A, B):
    """Multiply two square matrices using Strassen's algorithm."""
    n = A.shape[0]

    # Base case: fall back to naive multiplication for small matrices
    if n <= 64:
        return A @ B

    # Pad to even dimension if necessary
    if n % 2 != 0:
        A = np.pad(A, ((0, 1), (0, 1)))
        B = np.pad(B, ((0, 1), (0, 1)))
        C = strassen(A, B)
        return C[:n, :n]

    mid = n // 2

    # Partition into quadrants
    A11, A12 = A[:mid, :mid], A[:mid, mid:]
    A21, A22 = A[mid:, :mid], A[mid:, mid:]
    B11, B12 = B[:mid, :mid], B[:mid, mid:]
    B21, B22 = B[mid:, :mid], B[mid:, mid:]

    # 7 intermediate products (recursive!)
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    # Assemble result
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Combine quadrants
    C = np.zeros((n, n))
    C[:mid, :mid] = C11
    C[:mid, mid:] = C12
    C[mid:, :mid] = C21
    C[mid:, mid:] = C22

    return C
```

Note the base case threshold at $n = 64$ — below this, we use NumPy's built-in (BLAS-optimized) matrix multiplication, which is faster for small matrices despite its $O(n^3)$ complexity. The recursive structure mirrors the mathematical formulation exactly: partition, compute 7 products, assemble. Each recursive call halves the matrix dimension, and we make 7 such calls, giving us the $O(n^{\log_2 7})$ complexity.

---

## Cutting Off the Hydra's Heads

We've traced a path from the high-level intuition — attention as data-dependent weighted aggregation — down through the linear algebra, the computational complexity, and into working code. The Hydra metaphor holds: each attention head independently learns its own view of the data, and the concatenation fuses these into something more powerful than any single head could achieve.

The transformer's real innovation wasn't just attention itself — it was making attention **the only mechanism**, removing the sequential bottleneck of RNNs and letting GPUs do what they do best: massively parallel matrix multiplication. Strassen's algorithm (and its more exotic descendants like the Coppersmith-Winograd family) chip away at the cost of each multiplication. Together, they make it tractable to train models on the scale we see today.

If you want to understand what happens *after* attention computes its outputs — how the loss propagates backward through all these matrix operations to update the weights — check out my post on [Autodifferentiation](/2023/06/15/autodifferentiation.html). The chain rule through a multi-head attention layer is a beautiful exercise in the machinery we built there.
