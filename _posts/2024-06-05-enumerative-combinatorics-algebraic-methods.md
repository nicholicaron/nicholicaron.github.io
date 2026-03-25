---
layout: post
title: "Enumerative Combinatorics: Algebraic Proof Methods"
date: 2024-06-05
tags: [Math, Combinatorics, Enumerative Combinatorics]
---

These are my compiled notes from Enumerative Combinatorics, reorganized into a narrative that builds the theory from its core techniques. Enumerative combinatorics is the art of counting — not just "how many?" but "how many, exactly, and why?" The subject begins with simple principles (if you can break a set into non-overlapping pieces, count each piece) but quickly demands algebraic machinery to handle overlapping cases, infinite families, and recursive structures.

This is Part 1 of a three-part series. Here we develop two foundational proof techniques: the Principle of Inclusion-Exclusion, which handles overlapping cases that the basic sum principle cannot, and mathematical induction, which lets us prove statements about all natural numbers from a single domino-like argument.

## What This Post Covers

- **The Principle of Inclusion-Exclusion** — From two overlapping sets to the general formula for $m$ sets
- **PIE in Action** — Counting constrained integer solutions and coprime integers
- **Mathematical Induction** — The standard principle and its strong variant
- **Induction in Practice** — The Hockey Stick Identity and bounding the Fibonacci sequence

---

## The Principle of Inclusion-Exclusion

### From Partitions to Overlaps

Recall that if a set $A$ is **partitioned** into non-overlapping blocks $A_1, A_2, \ldots, A_k$, then by the sum principle, $\lvert A \rvert = \lvert A_1 \rvert + \lvert A_2 \rvert + \cdots + \lvert A_k \rvert$. But what if our cases overlap? Naively adding the sizes overcounts elements that belong to multiple blocks. The Principle of Inclusion-Exclusion (PIE) corrects for this.

### Two Sets

Suppose we want $\lvert A \cup B \rvert$. Adding $\lvert A \rvert + \lvert B \rvert$ counts every element of $A \cap B$ twice, so we subtract once:

$$\lvert A \cup B \rvert = \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert.$$

<svg viewBox="0 0 340 180" xmlns="http://www.w3.org/2000/svg" style="max-width:340px; margin: 1.5rem auto; display:block;">
  <style>
    .venn-label { font: 14px 'Inter', sans-serif; fill: var(--text-primary, #1a1a1a); }
    .venn-circle { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
    .venn-region { font: 12px 'JetBrains Mono', monospace; fill: var(--text-secondary, #666); }
  </style>
  <circle cx="130" cy="90" r="70" class="venn-circle" opacity="0.8"/>
  <circle cx="210" cy="90" r="70" class="venn-circle" opacity="0.8"/>
  <text x="95" y="95" class="venn-label" text-anchor="middle">A</text>
  <text x="245" y="95" class="venn-label" text-anchor="middle">B</text>
  <text x="170" y="95" class="venn-region" text-anchor="middle">A∩B</text>
</svg>

### Three Sets

With three sets the pattern extends. Adding all three overcounts pairwise overlaps, but subtracting all pairwise intersections removes the triple intersection one time too many:

$$\lvert A \cup B \cup C \rvert = \lvert A \rvert + \lvert B \rvert + \lvert C \rvert - \lvert A \cap B \rvert - \lvert A \cap C \rvert - \lvert B \cap C \rvert + \lvert A \cap B \cap C \rvert.$$

<svg viewBox="0 0 360 220" xmlns="http://www.w3.org/2000/svg" style="max-width:360px; margin: 1.5rem auto; display:block;">
  <style>
    .v3-label { font: 14px 'Inter', sans-serif; fill: var(--text-primary, #1a1a1a); }
    .v3-circle { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
  </style>
  <circle cx="145" cy="90" r="70" class="v3-circle" opacity="0.8"/>
  <circle cx="215" cy="90" r="70" class="v3-circle" opacity="0.8"/>
  <circle cx="180" cy="150" r="70" class="v3-circle" opacity="0.8"/>
  <text x="110" y="70" class="v3-label" text-anchor="middle">A</text>
  <text x="250" y="70" class="v3-label" text-anchor="middle">B</text>
  <text x="180" y="200" class="v3-label" text-anchor="middle">C</text>
</svg>

### The General Theorem

The pattern of alternating signs generalizes to any number of sets.

**Theorem (Principle of Inclusion-Exclusion).** Let $A_1, A_2, \ldots, A_m$ be subsets of a universal set $U$. Then

$$\left\lvert U \setminus (A_1 \cup A_2 \cup \cdots \cup A_m)\right\rvert = \sum_{I \subseteq [m]} (-1)^{\lvert I \rvert} \left\lvert\bigcap_{i \in I} A_i\right\rvert$$

where the sum ranges over all subsets $I$ of $[m] = \{1, 2, \ldots, m\}$, and by convention $\bigcap_{i \in \emptyset} A_i = U$.

Equivalently, expanding the right side:

$$\left\lvert U \setminus \bigcup_{i=1}^m A_i\right\rvert = \lvert U \rvert - \sum_{i} \lvert A_i \rvert + \sum_{i < j} \lvert A_i \cap A_j \rvert - \sum_{i < j < k} \lvert A_i \cap A_j \cap A_k \rvert + \cdots + (-1)^m \lvert A_1 \cap A_2 \cap \cdots \cap A_m \rvert.$$

> The key insight: each term in PIE corrects an overcounting (or undercounting) introduced by the previous term. The alternating signs ensure that every element of $U$ is counted exactly once.

---

### Example: Constrained Integer Solutions

**Problem.** How many non-negative integer solutions to $x_1 + x_2 + x_3 + x_4 = 10$ are there such that $\max\{x_1, x_2, x_3\} \geq 3$?

Without any upper-bound constraints, the number of non-negative integer solutions to $x_1 + x_2 + x_3 + x_4 = n$ is $\binom{n+3}{3}$ (a standard stars-and-bars count). We define the "bad" sets:

$$A = \{x_1 \geq 3\}, \quad B = \{x_2 \geq 3\}, \quad C = \{x_3 \geq 3\}.$$

We want $\lvert A \cup B \cup C \rvert$.

**Computing $\lvert A \rvert$:** If $x_1 \geq 3$, let $x_1' = x_1 - 3 \geq 0$. Then $x_1' + x_2 + x_3 + x_4 = 7$, giving $\binom{10}{3}$ solutions. By symmetry, $\lvert A \rvert = \lvert B \rvert = \lvert C \rvert = \binom{10}{3}$.

**Computing $\lvert A \cap B \rvert$:** If $x_1 \geq 3$ and $x_2 \geq 3$, let $x_1' = x_1 - 3$ and $x_2' = x_2 - 3$. Then $x_1' + x_2' + x_3 + x_4 = 4$, giving $\binom{7}{3}$ solutions. By symmetry, $\lvert A \cap B \rvert = \lvert A \cap C \rvert = \lvert B \cap C \rvert = \binom{7}{3}$.

**Computing $\lvert A \cap B \cap C \rvert$:** All three at least 3 gives $x_1' + x_2' + x_3' + x_4 = 1$, so $\binom{4}{3}$ solutions.

By PIE:

$$\lvert A \cup B \cup C \rvert = 3\binom{10}{3} - 3\binom{7}{3} + \binom{4}{3}.$$

---

### Example: Counting Coprimality

**Problem.** How many positive integers up to 100 are **not** divisible by 2, 3, or 5?

Let $U = [100]$ and define:

$$A = \{n \in U : 2 \mid n\}, \quad B = \{n \in U : 3 \mid n\}, \quad C = \{n \in U : 5 \mid n\}.$$

The sizes are:

| Set | Size | Reason |
|---|---|---|
| $A$ | $\lfloor 100/2 \rfloor = 50$ | Multiples of 2 |
| $B$ | $\lfloor 100/3 \rfloor = 33$ | Multiples of 3 |
| $C$ | $\lfloor 100/5 \rfloor = 20$ | Multiples of 5 |
| $A \cap B$ | $\lfloor 100/6 \rfloor = 16$ | Multiples of 6 |
| $A \cap C$ | $\lfloor 100/10 \rfloor = 10$ | Multiples of 10 |
| $B \cap C$ | $\lfloor 100/15 \rfloor = 6$ | Multiples of 15 |
| $A \cap B \cap C$ | $\lfloor 100/30 \rfloor = 3$ | Multiples of 30 |

By PIE:

$$\lvert A \cup B \cup C \rvert = 50 + 33 + 20 - 16 - 10 - 6 + 3 = 74.$$

Hence the number of integers up to 100 divisible by none of 2, 3, or 5 is:

$$\lvert U \setminus (A \cup B \cup C) \rvert = 100 - 74 = 26.$$

---

## Mathematical Induction

Induction is the tool for proving statements of the form "for all $n \in \mathbb{N}$, $P(n)$ is true." It comes in two flavors.

### The Principle of Mathematical Induction

**Axiom.** Let $P(n)$ be a statement for each $n \in \mathbb{N}$. Then $P(n)$ is true for all $n$ if and only if:

1. **(Base case)** $P(1)$ is true, and
2. **(Inductive step)** For all $k \in \mathbb{N}$, $P(k) \Rightarrow P(k+1)$.

### Strong Induction

In **strong induction**, the inductive step assumes $P(j)$ is true for *all* $j \leq k$, not just $P(k)$. This is useful when the recurrence at step $k+1$ depends on multiple earlier values (as in Fibonacci-type arguments).

> Every induction proof must contain four clearly identifiable components: (1) a base case, (2) an induction hypothesis ("let $k \in \mathbb{N}$ and assume $P(k)$"), (3) a proof that $P(k+1)$ follows, and (4) a concluding sentence invoking the principle ("by induction, we are done").

---

### Example: The Hockey Stick Identity

**Claim.** For all $n, k \in \mathbb{N}$ with $k \leq n$,

$$\binom{n+1}{k+1} = \binom{n}{k} + \binom{n-1}{k} + \binom{n-2}{k} + \cdots + \binom{k}{k}.$$

*Proof.* Fix $k \in \mathbb{N}$. We induct on $n$. Let $P(n)$ be the statement above.

**Base case:** $P(k)$ reads $\binom{k+1}{k+1} = \binom{k}{k}$, which is $1 = 1$. True.

**Inductive step:** Let $m \in \mathbb{N}$ and assume $P(m)$ holds:

$$\binom{m+1}{k+1} = \binom{m}{k} + \binom{m-1}{k} + \cdots + \binom{k}{k}.$$

We want to prove $P(m+1)$:

$$\binom{m+2}{k+1} = \binom{m+1}{k} + \binom{m}{k} + \cdots + \binom{k}{k}.$$

By Pascal's Identity, $\binom{m+2}{k+1} = \binom{m+1}{k+1} + \binom{m+1}{k}$. Substituting the induction hypothesis for $\binom{m+1}{k+1}$:

$$\binom{m+2}{k+1} = \left[\binom{m}{k} + \binom{m-1}{k} + \cdots + \binom{k}{k}\right] + \binom{m+1}{k} = \binom{m+1}{k} + \binom{m}{k} + \cdots + \binom{k}{k}.$$

This is exactly $P(m+1)$. By induction, we are done. $\square$

---

### Example: Bounding the Fibonacci Sequence

The Fibonacci sequence is defined by $F_0 = 1$, $F_1 = 1$, and $F_n = F_{n-1} + F_{n-2}$ for $n \geq 2$.

**Claim.** $F_n \leq (1.7)^n$ for all $n \geq 0$.

*Proof.* We use **strong induction**.

**Base cases:** $F_0 = 1 \leq (1.7)^0 = 1$, and $F_1 = 1 \leq (1.7)^1 = 1.7$. Both hold.

**Inductive step:** Let $k \geq 2$ and assume $F_j \leq (1.7)^j$ for all $j < k$ (strong induction hypothesis). In particular, $F_{k-1} \leq (1.7)^{k-1}$ and $F_{k-2} \leq (1.7)^{k-2}$. Then:

$$F_k = F_{k-1} + F_{k-2} \leq (1.7)^{k-1} + (1.7)^{k-2} = (1.7)^{k-2}(1.7 + 1) = 2.7 \cdot (1.7)^{k-2}.$$

The key inequality is whether $2.7 \cdot (1.7)^{k-2} \leq (1.7)^k$, which simplifies to $2.7 \leq (1.7)^2 = 2.89$. Since this holds, we conclude:

$$F_k \leq 2.7 \cdot (1.7)^{k-2} \leq (1.7)^2 \cdot (1.7)^{k-2} = (1.7)^k.$$

By (strong) induction, $F_n \leq (1.7)^n$ for all $n \geq 0$. $\square$

> The choice of 1.7 is not arbitrary — any base $b$ satisfying $b^2 \geq b + 1$ works (the golden ratio $\varphi = \frac{1+\sqrt{5}}{2} \approx 1.618$ is the tightest such bound). The argument reveals that Fibonacci numbers grow exponentially, a fact we will make precise in [Part 2](/2024/06/12/enumerative-combinatorics-generating-functions.html) using generating functions.

---

## Looking Ahead

PIE and induction are powerful, but they require cleverness for each new problem. In [Part 2: Generating Functions](/2024/06/12/enumerative-combinatorics-generating-functions.html), we develop a systematic algebraic framework — ordinary and exponential generating functions — that transforms counting problems into algebra. We will derive closed-form formulas for sequences defined by recurrences, prove combinatorial identities by comparing coefficients, and see why the Fibonacci sequence grows like powers of the golden ratio.
