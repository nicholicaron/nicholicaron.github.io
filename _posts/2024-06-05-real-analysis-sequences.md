---
layout: post
title: "Real Analysis I: Sequences and Convergence"
date: 2024-06-05
tags: [Math, Analysis, Real Analysis]
---

This is Part 3 of a four-part series on Real Analysis I. In [Part 1](/2024/05/01/real-analysis-real-numbers.html), we built the real number system and the Completeness Axiom. In [Part 2](/2024/05/08/real-analysis-topology.html), we developed the topology of the real line — neighborhoods, open and closed sets, accumulation points. Now we put these tools to work.

Sequences are the beating heart of analysis. Nearly every concept in this course — limits, continuity, compactness — can be expressed in terms of sequences. A sequence is the simplest kind of infinite process, and understanding when and why infinite processes converge is the central problem of real analysis.

## What This Post Covers

- **Sequences and Convergence** — The $\epsilon$-$N$ definition and the uniqueness of limits
- **Limit Theorems** — Algebraic operations on limits, the squeeze theorem
- **Monotone Sequences** — The Monotone Convergence Theorem and its consequences
- **Cauchy Sequences** — A criterion for convergence that doesn't require knowing the limit
- **Subsequences and Bolzano-Weierstrass** — Every bounded sequence has a convergent subsequence
- **Lim Sup and Lim Inf** — The eventual bounds of a sequence

---

## Sequences and Convergence

**Definition.** A **sequence** of real numbers is a function $f: \mathbb{N} \to \mathbb{R}$. We write $x_n = f(n)$ and denote the sequence by $(x_n)$ or $(x_1, x_2, x_3, \ldots)$.

**Examples.**
- $x_n = 1/n$ gives the sequence $(1, 1/2, 1/3, 1/4, \ldots)$.
- $x_n = 1 + (-1)^n$ gives the sequence $(0, 2, 0, 2, \ldots)$.

The fundamental question is: does a sequence "settle down" to a single value?

**Definition.** A sequence $(x_n)$ is said to **converge** to a real number $x$ if

$$
\forall\, \epsilon > 0, \quad \exists\, N \in \mathbb{N} \text{ such that } \forall\, n \in \mathbb{N}, \quad n \geq N \implies |x_n - x| < \epsilon.
$$

We write $x_n \to x$, or $\lim_{n \to \infty} x_n = x$, and call $x$ the **limit** of the sequence.

Think of this as a game: an adversary picks any tolerance $\epsilon > 0$, no matter how small. You must respond with a starting index $N$ such that *every* term of the sequence from position $N$ onward lies within $\epsilon$ of the limit $x$. If you can always win, the sequence converges.

<svg viewBox="0 0 480 180" style="max-width:480px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax4 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.2; }
    .lbl4 { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .lbl4s { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--text-secondary, #555); }
    .band { fill: rgba(148, 69, 43, 0.08); stroke: var(--primary, #94452b); stroke-width: 1; stroke-dasharray: 5,4; }
    .dot-in { fill: var(--primary, #94452b); }
    .dot-out { fill: var(--text-secondary, #555); }
    .limit-line { stroke: var(--primary, #94452b); stroke-width: 1.5; }
  </style>
  <defs>
    <marker id="arr4" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <path d="M0,0 L6,2.5 L0,5" fill="none" stroke="var(--text-primary, #1a1a1a)" stroke-width="1" />
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="50" y1="150" x2="460" y2="150" class="ax4" marker-end="url(#arr4)" />
  <line x1="50" y1="150" x2="50" y2="15" class="ax4" marker-end="url(#arr4)" />
  <text x="465" y="155" class="lbl4s">n</text>
  <!-- Limit line L -->
  <line x1="50" y1="70" x2="450" y2="70" class="limit-line" stroke-dasharray="8,4" />
  <text x="38" y="73" text-anchor="end" class="lbl4" fill="var(--primary, #94452b)" font-style="italic">L</text>
  <!-- Epsilon band -->
  <rect x="50" y="48" width="400" height="44" class="band" />
  <text x="38" y="52" text-anchor="end" class="lbl4s">L+ε</text>
  <text x="38" y="96" text-anchor="end" class="lbl4s">L−ε</text>
  <!-- N marker -->
  <line x1="210" y1="145" x2="210" y2="155" class="ax4" />
  <text x="210" y="165" text-anchor="middle" class="lbl4" font-style="italic">N</text>
  <!-- Dots before N (some outside band) -->
  <circle cx="80" cy="120" r="3" class="dot-out" />
  <circle cx="110" cy="40" r="3" class="dot-out" />
  <circle cx="140" cy="95" r="3" class="dot-out" />
  <circle cx="170" cy="55" r="3" class="dot-out" />
  <circle cx="200" cy="85" r="3" class="dot-out" />
  <!-- Dots after N (all inside band) -->
  <circle cx="230" cy="65" r="3" class="dot-in" />
  <circle cx="260" cy="74" r="3" class="dot-in" />
  <circle cx="290" cy="67" r="3" class="dot-in" />
  <circle cx="320" cy="72" r="3" class="dot-in" />
  <circle cx="350" cy="69" r="3" class="dot-in" />
  <circle cx="380" cy="71" r="3" class="dot-in" />
  <circle cx="410" cy="70" r="3" class="dot-in" />
  <circle cx="440" cy="70" r="3" class="dot-in" />
</svg>

*After index $N$, every term of the sequence lies within the $\epsilon$-band around $L$. The terms before $N$ may wander freely.*

A sequence that does not converge is said to **diverge**.

### Uniqueness of Limits

**Theorem.** The limit of a convergent sequence is unique.

*Proof.* Suppose $\ell_1$ and $\ell_2$ are both limits of $(x_n)$. Let $\epsilon > 0$ be given. Since $x_n \to \ell_1$, there exists $N_1 \in \mathbb{N}$ such that $n \geq N_1 \implies \lvert x_n - \ell_1 \rvert < \epsilon/2$. Since $x_n \to \ell_2$, there exists $N_2 \in \mathbb{N}$ such that $n \geq N_2 \implies \lvert x_n - \ell_2 \rvert < \epsilon/2$.

Define $N = \max\\{N_1, N_2\\}$. Then for this $N$:

$$
|\ell_1 - \ell_2| = |(\ell_1 - x_N) + (x_N - \ell_2)| \leq |x_N - \ell_1| + |x_N - \ell_2| < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon.
$$

Since $\epsilon$ was arbitrary and $0 \leq \lvert \ell_1 - \ell_2 \rvert < \epsilon$ for all $\epsilon > 0$, we conclude $\ell_1 = \ell_2$. $\square$

### A First Example from the Definition

**Example.** Prove that $\lim_{n \to \infty} \frac{1}{n} = 0$.

*Proof.* Let $\epsilon > 0$ be given. By the Archimedean Property, there exists $N \in \mathbb{N}$ with $\frac{1}{N} < \epsilon$. Then for all $n \geq N$:

$$
\left|\frac{1}{n} - 0\right| = \frac{1}{n} \leq \frac{1}{N} < \epsilon. \quad \square
$$

**Remark.** $a_n \to a$ if and only if $\lvert a_n - a \rvert \to 0$. Also, if $a_n \to a$, then $\lvert a_n \rvert \to \lvert a \rvert$ (by the reverse triangle inequality). The converse is false: $a_n = (-1)^n$ has $\lvert a_n \rvert \to 1$, but $(a_n)$ diverges.

---

## Limit Theorems

Computing limits directly from the $\epsilon$-$N$ definition every time would be tedious. The algebraic limit theorems let us build complex limits from simple ones.

### Convergent Sequences are Bounded

**Definition.** A sequence $(a_n)$ is **bounded** if there exists $M > 0$ such that $\lvert a_n \rvert \leq M$ for all $n \in \mathbb{N}$.

**Theorem.** Every convergent sequence is bounded.

*Proof.* Let $a_n \to a$. For $\epsilon = 1$, there exists $N \in \mathbb{N}$ such that $n \geq N \implies \lvert a_n - a \rvert < 1$. For $n \geq N$, we have $\lvert a_n \rvert = \lvert a_n - a + a \rvert \leq \lvert a_n - a \rvert + \lvert a \rvert < 1 + \lvert a \rvert$.

Define $M = \max\\{1 + \lvert a \rvert, \lvert a_1 \rvert, \lvert a_2 \rvert, \ldots, \lvert a_{N-1} \rvert\\}$. Then $\lvert a_n \rvert \leq M$ for all $n \in \mathbb{N}$. $\square$

The converse is false: $((-1)^n)$ is bounded but divergent.

### The Algebraic Limit Theorem

**Theorem.** Suppose $s_n \to s$, $t_n \to t$, and $k \in \mathbb{R}$ is fixed. Then:
1. $s_n + t_n \to s + t$
2. $k + s_n \to k + s$
3. $s_n \cdot t_n \to s \cdot t$
4. $s_n / t_n \to s / t$, provided $t \neq 0$ and $t_n \neq 0$ for all $n$
5. $k \cdot s_n \to k \cdot s$

The product rule is the most instructive to prove because it showcases a key analysis technique: adding and subtracting a strategic term to split an error into manageable pieces.

*Proof of (c).* Let $\epsilon > 0$ be given. Since $t_n \to t$, the sequence $(t_n)$ is bounded: there exists $M > 0$ with $\lvert t_n \rvert \leq M$ for all $n$.

Since $s_n \to s$, there exists $N_1$ with $n \geq N_1 \implies \lvert s_n - s \rvert < \frac{\epsilon}{2M}$.

Since $t_n \to t$, there exists $N_2$ with $n \geq N_2 \implies \lvert t_n - t \rvert < \frac{\epsilon}{2\lvert s \rvert + 1}$.

Set $N = \max\\{N_1, N_2\\}$. For $n \geq N$:

$$
\begin{aligned}
|s_n t_n - st| &= |s_n t_n - s_n t + s_n t - st| \\
&= |s_n(t_n - t) + t(s_n - s)| \\
&\leq |s_n||t_n - t| + |t||s_n - s| \\
&< M \cdot \frac{\epsilon}{2M} + |s| \cdot \frac{\epsilon}{2|s| + 1} \\
&< \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon. \quad \square
\end{aligned}
$$

### The Squeeze Theorem and Order Limit Theorem

**Theorem (Squeeze Theorem).** If $a_n \to L$, $b_n \to L$, and $a_n \leq x_n \leq b_n$ for all $n$, then $x_n \to L$.

**Theorem (Order Limit Theorem).** If $s_n \to s$, $t_n \to t$, and $s_n \leq t_n$ for all $n$, then $s \leq t$.

**Corollary.** If $t_n \to t$ and $t_n \geq 0$ for all $n$, then $t \geq 0$.

A useful application: if $t_n \to t$ and $t_n \geq 0$, then $\sqrt{t_n} \to \sqrt{t}$.

---

## Monotone Sequences

Some sequences don't oscillate — they march steadily in one direction. These are particularly well-behaved.

**Definition.** A sequence $(a_n)$ is **increasing** if $a_n \leq a_{n+1}$ for all $n$, and **decreasing** if $a_n \geq a_{n+1}$ for all $n$. A sequence is **monotone** if it is either increasing or decreasing.

**Example.** $a_n = 1/n$ is decreasing. $a_n = 1 - 1/n$ is increasing. $a_n = (-1)^n/n$ is neither.

### The Monotone Convergence Theorem

**Theorem (MCT).** A monotone sequence converges if and only if it is bounded.

*Proof.* $(\Leftarrow)$ Suppose $(a_n)$ is increasing and bounded. The set $S = \\{a_n : n \in \mathbb{N}\\}$ is nonempty and bounded above. By the Completeness Axiom, $a = \sup S$ exists in $\mathbb{R}$.

We claim $a_n \to a$. Let $\epsilon > 0$. Since $a = \sup S$, the value $a - \epsilon$ is not an upper bound for $S$, so there exists $N \in \mathbb{N}$ with $a_N > a - \epsilon$.

Since $(a_n)$ is increasing and $a = \sup S$:

$$
n \geq N \implies a - \epsilon < a_N \leq a_n \leq a < a + \epsilon.
$$

That is, $\lvert a_n - a \rvert < \epsilon$ for all $n \geq N$. So $a_n \to a = \sup S$.

If $(a_n)$ is decreasing and bounded, then $b_n = -a_n$ is increasing and bounded. By the above, $(b_n)$ converges, so $(a_n) = (-b_n)$ converges.

$(\Rightarrow)$ Every convergent sequence is bounded (proved earlier). $\square$

> The MCT is where the Completeness Axiom pays its first major dividend. In the rationals, this theorem *fails*: the sequence of decimal approximations $1, 1.4, 1.41, 1.414, \ldots$ to $\sqrt{2}$ is increasing, bounded above by $2$, and has no limit in $\mathbb{Q}$.

### Application: Nested Radicals and the Golden Ratio

**Example.** Let $s_1 = 1$, and $s_{n+1} = \sqrt{1 + s_n}$ for $n \geq 1$. The sequence is $(1, \sqrt{2}, \sqrt{1 + \sqrt{2}}, \ldots)$. We show it converges and find its limit.

**Step 1: $(s_n)$ is increasing.** Base case: $s_1 = 1 \leq \sqrt{2} = s_2$. Inductive step: if $s_k \leq s_{k+1}$, then $1 + s_k \leq 1 + s_{k+1}$, so $s_{k+1} = \sqrt{1 + s_k} \leq \sqrt{1 + s_{k+1}} = s_{k+2}$.

**Step 2: $(s_n)$ is bounded above by 2.** Base case: $s_1 = 1 < 2$. Inductive step: if $s_k \leq 2$, then $s_{k+1} = \sqrt{1 + s_k} \leq \sqrt{1 + 2} = \sqrt{3} \leq 2$.

By the MCT, $s = \lim s_n$ exists. Since $s_{n+1} = \sqrt{1 + s_n}$, taking limits of both sides:

$$
s = \sqrt{1 + s}, \quad s^2 = 1 + s, \quad s^2 - s - 1 = 0, \quad s = \frac{1 + \sqrt{5}}{2}.
$$

Since $s_n \geq 1$ for all $n$, we take the positive root: the limit is the **golden ratio** $\phi = \frac{1 + \sqrt{5}}{2}$.

### Unbounded Monotone Sequences

**Theorem.** If $(s_n)$ is increasing and unbounded (above), then $s_n \to +\infty$. If $(s_n)$ is decreasing and unbounded (below), then $s_n \to -\infty$.

A sequence **diverges to $+\infty$** if for every $M \in \mathbb{R}$, there exists $N \in \mathbb{N}$ such that $n \geq N \implies a_n > M$. Divergence to $-\infty$ is defined analogously.

---

## Cauchy Sequences

Here is a philosophical problem: the definition of convergence requires us to *know the limit* $x$ in advance. But what if we suspect a sequence converges but don't know to what? The Cauchy criterion solves this.

**Definition.** A sequence $(a_n)$ is **Cauchy** if

$$
\forall\, \epsilon > 0, \quad \exists\, N \in \mathbb{N} \text{ such that } \forall\, m, n \in \mathbb{N}, \quad m, n \geq N \implies |a_n - a_m| < \epsilon.
$$

A Cauchy sequence is one where the terms eventually cluster together — the distances between terms shrink to zero, regardless of which two terms you pick (past the index $N$). The key insight: this definition makes no reference to a limit.

**Theorem.** Every convergent sequence is Cauchy.

*Proof.* Let $(a_n)$ converge to $a$. Let $\epsilon > 0$. There exists $N$ with $n \geq N \implies \lvert a_n - a \rvert < \epsilon/2$. For $m, n \geq N$:

$$
|a_n - a_m| = |(a_n - a) - (a_m - a)| \leq |a_n - a| + |a_m - a| < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon. \quad \square
$$

**Theorem.** Every Cauchy sequence is bounded.

*Proof.* Take $\epsilon = 1$ in the definition to find $N$ with $\lvert a_n - a_N \rvert < 1$ for all $n \geq N$. Then $\lvert a_n \rvert < 1 + \lvert a_N \rvert$ for $n \geq N$. Set $M = \max\\{1 + \lvert a_N \rvert, \lvert a_1 \rvert, \ldots, \lvert a_{N-1} \rvert\\}$. $\square$

### The Cauchy Criterion

**Theorem.** A sequence of real numbers converges if and only if it is Cauchy.

*Proof.* $(\Rightarrow)$ Proved above.

$(\Leftarrow)$ Suppose $(a_n)$ is Cauchy. Since Cauchy sequences are bounded, $S = \\{a_n : n \in \mathbb{N}\\}$ is bounded.

**Case 1: $S$ is finite.** Let $\epsilon$ be the smallest distance between any two distinct points of $S$. Since $(a_n)$ is Cauchy, there exists $N$ with $\lvert a_n - a_m \rvert < \epsilon$ for all $m, n \geq N$. Since the minimum distance between distinct elements of $S$ is $\epsilon$, we must have $a_n = a_N$ for all $n \geq N$. So $(a_n)$ converges to $a_N$.

**Case 2: $S$ is infinite.** $S$ is bounded and infinite. By the Bolzano-Weierstrass theorem (below), $S$ has an accumulation point $a \in S'$. We claim $a_n \to a$.

Given $\epsilon > 0$: since $(a_n)$ is Cauchy, there exists $N$ with $\lvert a_n - a_m \rvert < \epsilon/2$ for all $m, n \geq N$. Since $a \in S'$, $N^*(a; \epsilon/2) \cap S$ is infinite, so we can choose $a_m$ with $m \geq N$ and $\lvert a_m - a \rvert < \epsilon/2$. For $n \geq N$:

$$
|a_n - a| \leq |a_n - a_m| + |a_m - a| < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon. \quad \square
$$

> The Cauchy criterion is remarkable: it lets you determine that a sequence converges *without knowing what it converges to*. This is essential in practice — many naturally arising sequences are easily shown to be Cauchy, but their limits are transcendental numbers that resist closed-form description.

---

## Subsequences and Bolzano-Weierstrass

**Definition.** Let $(a_n)_{n=1}^{\infty}$ be a sequence and let $(n_k)_{k=1}^{\infty}$ be a strictly increasing sequence of natural numbers (i.e., $n_k < n_{k+1}$ for all $k$). The sequence $(a_{n_k})_{k=1}^{\infty}$ is called a **subsequence** of $(a_n)$.

A subsequence picks out infinitely many terms of the original sequence, preserving their order but possibly skipping some.

**Example.** Let $a_n = 1/n$. Taking $n_k = 2k$ gives the subsequence $a_{n_k} = 1/(2k)$. Taking $n_k = 2^k$ gives $a_{n_k} = 1/2^k$.

**Lemma.** If $(n_k)$ is a strictly increasing sequence of natural numbers, then $n_k \geq k$ for all $k \in \mathbb{N}$.

*Proof.* By induction. Base case: $n_1 \geq 1$. If $n_k \geq k$, then $n_{k+1} > n_k \geq k$, so $n_{k+1} \geq k + 1$. $\square$

**Theorem.** If $x_n \to x$, then every subsequence of $(x_n)$ also converges to $x$.

This has a powerful contrapositive: if two subsequences converge to *different* limits, the original sequence diverges. For instance, $((-1)^n)$ diverges because the subsequence $a_{2k} = 1 \to 1$ and $a_{2k-1} = -1 \to -1$.

### The Bolzano-Weierstrass Theorem

**Theorem (Bolzano-Weierstrass).** Every bounded sequence of real numbers has a convergent subsequence.

*Proof.* Let $(a_n)$ be bounded, say $\lvert a_n \rvert \leq M$ for all $n$. Let $A = \\{a_n : n \in \mathbb{N}\\}$.

**Case 1:** If some value $x = a_n$ appears for infinitely many values of $n$, we can extract a constant subsequence converging to $x$.

**Case 2:** Otherwise, $A$ is infinite. Since $A$ is bounded and infinite, by the Bolzano-Weierstrass theorem for sets (which follows from the completeness axiom), $A$ has an accumulation point $y$. Then for each $k \in \mathbb{N}$, the neighborhood $N^*(y; 1/k)$ contains a point of $A$, say $a_{n_k}$, with $\lvert a_{n_k} - y \rvert < 1/k$.

We build the subsequence inductively: choose $n_1$ with $\lvert a_{n_1} - y \rvert < 1$. Having chosen $n_1 < n_2 < \cdots < n_k$, since $N^*(y; 1/(k+1)) \cap A$ is infinite, we can choose $n_{k+1} > n_k$ with $\lvert a_{n_{k+1}} - y \rvert < 1/(k+1)$.

Then $\lvert a_{n_k} - y \rvert < 1/k \to 0$, so $a_{n_k} \to y$. $\square$

> Bolzano-Weierstrass is one of the deepest consequences of completeness. It says that in $\mathbb{R}$, boundedness alone forces some kind of convergent behavior — you cannot have infinitely many points packed into a bounded region without them clustering somewhere. This fails in $\mathbb{Q}$.

### Applications

**Theorem.** A sequence converges if and only if every subsequence converges (to the same limit).

**Theorem.** If $(a_n)$ is an unbounded sequence, then there exists a subsequence $(a_{n_k})$ with $a_{n_k} \to +\infty$ or $a_{n_k} \to -\infty$.

---

## Lim Sup and Lim Inf

Not every bounded sequence converges, but every bounded sequence has a "best" upper and lower limit.

**Definition.** Let $(a_n)$ be a bounded sequence. The set of **subsequential limits** is

$$
S = \{a \in \mathbb{R} : \text{there exists a subsequence } (a_{n_k}) \text{ with } a_{n_k} \to a\}.
$$

By Bolzano-Weierstrass, $S \neq \emptyset$. Since $(a_n)$ is bounded, so is every subsequence, so $S$ is bounded. By the Completeness Axiom, $\sup S$ and $\inf S$ exist.

- The **limit superior** (or upper limit) is $\limsup_{n \to \infty} a_n = \sup S$.
- The **limit inferior** (or lower limit) is $\liminf_{n \to \infty} a_n = \inf S$.

**Examples.**

1. $a_n = (-1)^n$: $S = \\{-1, 1\\}$, so $\limsup a_n = 1$ and $\liminf a_n = -1$.
2. $b_n = 1/n$: $S = \\{0\\}$, so $\limsup b_n = \liminf b_n = 0$.
3. $t_n = \sin(n)$: $\limsup t_n = 1$ and $\liminf t_n = -1$.

**Key facts:**
- $\liminf a_n \leq \limsup a_n$ always.
- $a_n \to a$ if and only if $\limsup a_n = \liminf a_n = a$.
- If $\liminf a_n < \limsup a_n$, the sequence **oscillates**.

**Theorem.** Let $(s_n)$ be a bounded sequence and $s = \limsup s_n$. Then:
1. For every $\epsilon > 0$, there exists $N$ such that $s_n < s + \epsilon$ for all $n \geq N$ (all but finitely many terms are below $s + \epsilon$).
2. For every $\epsilon > 0$ and every $k$, there exists $n_k > k$ with $s_{n_k} > s - \epsilon$ (infinitely many terms are above $s - \epsilon$).

Moreover, $s$ is the unique real number satisfying both (i) and (ii).

---

## Looking Ahead

We have built a complete theory of sequential convergence: the $\epsilon$-$N$ definition, algebraic limit theorems for computation, the Monotone Convergence Theorem and Cauchy criterion as convergence tests, and Bolzano-Weierstrass as the deep structural result.

Sequences capture limits of *numbers*. But analysis ultimately cares about *functions* — and functions have their own, richer notion of limits. In [Part 4: Continuity and Compactness](/2024/06/12/real-analysis-continuity.html), we define limits of functions, develop the theory of continuous functions, and arrive at the crown jewels of the course: the Extreme Value Theorem and the Intermediate Value Theorem.
