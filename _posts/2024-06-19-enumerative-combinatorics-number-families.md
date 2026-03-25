---
layout: post
title: "Enumerative Combinatorics: Famous Number Families"
date: 2024-06-19
tags: [Math, Combinatorics, Enumerative Combinatorics]
---

In [Part 1](/2024/06/05/enumerative-combinatorics-algebraic-methods.html) we built the algebraic proof methods — PIE and induction. In [Part 2](/2024/06/12/enumerative-combinatorics-generating-functions.html) we developed the generating function machinery — OGFs, EGFs, the Binomial Theorem, and the four-step recurrence method. Now we put it all together to study the great recurring characters of combinatorics: number families that appear in problem after problem, each counted by a sequence with deep algebraic and combinatorial structure.

A **number family** is a sequence of numbers depending on one or more parameters. Fibonacci numbers depend on one index; binomial coefficients on two. What makes a number family "famous" is that it arises as the answer to many seemingly unrelated counting problems. The same number that counts bracketings of parentheses also counts non-crossing partitions, lattice paths, and binary trees — and the algebraic reason is always the same generating function.

## What This Post Covers

- **Multinomial Coefficients** — Generalizing the Binomial Theorem to multiple variables, with combinatorial interpretations
- **Fibonacci and Lucas Numbers** — Algebraic closed forms, combinatorial interpretations via $\{1,2\}$-lists and matchings
- **Catalan Numbers** — A nonlinear recurrence, a quadratic generating function, and the art of proper bracketing
- **Beyond the Course** — A glimpse at posets, Dilworth's Theorem, and spectral graph theory

---

## Multinomial Coefficients

### From Binomial to Multinomial

Recall the Binomial Theorem: for all $n \in \mathbb{N}$,

$$(x + y)^n = \sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k}, \qquad \text{where } \binom{n}{k} = \frac{n!}{k!(n-k)!}.$$

The **Trinomial Theorem** extends this to three variables. Expanding $(x + y + z)^3$ directly:

$$\begin{aligned}
(x+y+z)^3 &= x^3 + y^3 + z^3 + 3x^2y + 3xy^2 + 3x^2z \\
&\quad + 3xz^2 + 3y^2z + 3yz^2 + 6xyz.
\end{aligned}$$

The coefficient of $x^a y^b z^c$ (with $a+b+c = 3$) is $\frac{3!}{a!\, b!\, c!}$. This motivates the general definition.

### The Multinomial Theorem

**Definition.** The **multinomial coefficient** is

$$\binom{n}{a_1, a_2, \ldots, a_r} = \frac{n!}{a_1!\, a_2!\, \cdots\, a_r!}$$

where $a_1 + a_2 + \cdots + a_r = n$ and each $a_i \geq 0$.

**Theorem (Multinomial Theorem).** For all $n \in \mathbb{N}$,

$$(x_1 + x_2 + \cdots + x_r)^n = \sum_{\substack{(a_1, \ldots, a_r) \\ a_1 + \cdots + a_r = n}} \binom{n}{a_1, a_2, \ldots, a_r}\, x_1^{a_1}\, x_2^{a_2}\, \cdots\, x_r^{a_r}.$$

*Proof.* By induction on $r$. Fix $n$ and assume the result holds for $r = m$ variables.

**Base case ($r = 2$):** This is the ordinary Binomial Theorem.

**Inductive step:** Let $y = x_1 + x_2 + \cdots + x_m$. Then:

$$\begin{aligned}
(x_1 + \cdots + x_m + x_{m+1})^n &= (y + x_{m+1})^n = \sum_{k=0}^{n} \binom{n}{k}\, y^k\, x_{m+1}^{n-k} \\
&= \sum_{k=0}^{n} \binom{n}{k} \left[\sum_{\substack{(a_1,\ldots,a_m) \\ a_1+\cdots+a_m=k}} \binom{k}{a_1,\ldots,a_m}\, x_1^{a_1}\cdots x_m^{a_m}\right] x_{m+1}^{n-k}.
\end{aligned}$$

Since $\binom{n}{k} \cdot \binom{k}{a_1,\ldots,a_m} = \frac{n!}{k!(n-k)!} \cdot \frac{k!}{a_1!\cdots a_m!} = \frac{n!}{a_1!\cdots a_m!(n-k)!} = \binom{n}{a_1,\ldots,a_m,n-k}$, this equals the $(m+1)$-variable multinomial expansion. By induction, we are done. $\square$

### Combinatorial Interpretations

The multinomial coefficient $\binom{n}{a_1, a_2, \ldots, a_r}$ counts:

1. **Distributions** of $n$ distinct objects to $r$ distinct recipients, where recipient $i$ gets exactly $a_i$ objects.
2. **Rearrangements** of an $n$-element list with $a_1$ copies of symbol 1, $a_2$ copies of symbol 2, ..., $a_r$ copies of symbol $r$.

**Example.** How many rearrangements of the letters in MISSISSIPPI?

We have 1 M, 4 I's, 4 S's, and 2 P's (total 11 letters). The answer is:

$$\binom{11}{1, 4, 4, 2} = \frac{11!}{1!\, 4!\, 4!\, 2!} = \frac{39916800}{1 \cdot 24 \cdot 24 \cdot 2} = 34650.$$

**Example.** How many partitions of $[10]$ into 1 block of size 4 and 2 blocks of size 3?

First choose which elements go into each block: $\binom{10}{4, 3, 3}$. But the two blocks of size 3 are *indistinguishable* — swapping them gives the same partition. By the equivalence principle (recall: if an equivalence relation has classes of equal size $c$, then the number of classes is the total count divided by $c$), we divide by $2! = 2$:

$$\frac{1}{2}\binom{10}{4, 3, 3} = \frac{10!}{4!\, 3!\, 3!\, \cdot\, 2} = 2100.$$

### A Counting Identity

Setting $x_1 = x_2 = \cdots = x_r = 1$ in the Multinomial Theorem:

$$r^n = \sum_{\substack{a_1+\cdots+a_r=n}} \binom{n}{a_1, \ldots, a_r}.$$

The left side counts $r$-ary lists of length $n$. The right side partitions these lists by the frequency vector $(a_1, \ldots, a_r)$. Two ways of counting the same thing.

---

## Fibonacci and Lucas Numbers

### Fibonacci Numbers: The Algebraic View

The Fibonacci sequence is defined by $f_0 = 1$, $f_1 = 1$, and $f_n = f_{n-1} + f_{n-2}$ for $n \geq 2$, giving $(f_n) = (1, 1, 2, 3, 5, 8, 13, \ldots)$.

In [Part 2](/2024/06/12/enumerative-combinatorics-generating-functions.html) we derived the OGF $F(x) = \frac{1}{1-x-x^2}$ and Binet's closed form. We also showed via the geometric series approach that:

$$f_m = \sum_{k=0}^{m} \binom{m-k}{k},$$

obtained by writing $F(x) = \frac{1}{1-(x+x^2)} = \sum (x+x^2)^n$ and applying the Binomial Theorem to each term.

### The Combinatorial Interpretation

**Definition.** Let $P_n$ be the set of lists with entries in $\{1, 2\}$ whose entries sum to $n$.

For example, $P_3 = \{(1,1,1),\, (1,2),\, (2,1)\}$, so $\lvert P_3 \rvert = 3 = f_3$.

**Claim.** $\lvert P_n \rvert = f_n$ for all $n \geq 0$.

*Proof.* By strong induction.

**Base cases:** $P_0 = \{\emptyset\}$ (the empty list), so $\lvert P_0 \rvert = 1 = f_0$. And $P_1 = \{(1)\}$, so $\lvert P_1 \rvert = 1 = f_1$.

**Inductive step:** Let $k \geq 0$ and assume $\lvert P_j \rvert = f_j$ for all $j \leq k+1$. We show $\lvert P_{k+2} \rvert = f_{k+2}$.

Define a bijection $g: P_{k+2} \to P_{k+1} \cup P_k$ by removing the last entry of the list:

$$g((x_1, \ldots, x_\ell)) = (x_1, \ldots, x_{\ell-1}).$$

If $(x_1, \ldots, x_\ell) \in P_{k+2}$, then $x_1 + \cdots + x_\ell = k+2$. If $x_\ell = 1$, the remaining entries sum to $k+1$, so $g$ maps into $P_{k+1}$. If $x_\ell = 2$, the remaining entries sum to $k$, so $g$ maps into $P_k$. Moreover, $P_{k+1}$ and $P_k$ are disjoint (lists summing to different values), so $g$ maps $P_{k+2}$ bijectively onto $P_{k+1} \cup P_k$.

The inverse $g^{-1}: P_{k+1} \cup P_k \to P_{k+2}$ appends 1 (if the input is from $P_{k+1}$) or 2 (if from $P_k$).

Since $g$ is a bijection, $\lvert P_{k+2} \rvert = \lvert P_{k+1} \rvert + \lvert P_k \rvert = f_{k+1} + f_k = f_{k+2}$. By strong induction, $\lvert P_n \rvert = f_n$ for all $n$. $\square$

### An Alternative Proof (Without Induction)

Partition $P_n = A_0 \cup A_1 \cup \cdots \cup A_n$, where $A_j$ consists of lists in $\{1, 2\}$ of length $n-j$ with exactly $j$ entries equal to 2 (and thus $n - 2j$ entries equal to 1, requiring $n - j$ total entries summing to $n$).

The number of such lists is $\lvert A_j \rvert = \binom{n-j}{j}$ (choosing which $j$ positions among $n-j$ are 2's). Hence:

$$\lvert P_n \rvert = \sum_{j=0}^{\lfloor n/2 \rfloor} \binom{n-j}{j} = f_n.$$

### Fibonacci as Matchings

A **matching** of a graph is a set of edges with no shared vertices. Let $M_n$ be the set of matchings of the $n$-vertex **path graph** $1 - 2 - 3 - \cdots - n$.

<svg viewBox="0 0 520 130" xmlns="http://www.w3.org/2000/svg" style="max-width:520px; margin: 1.5rem auto; display:block;">
  <style>
    .match-node { fill: var(--surface-container-low, #f5f5f5); stroke: var(--primary, #94452b); stroke-width: 2; }
    .match-edge { stroke: var(--text-secondary, #666); stroke-width: 2; }
    .match-edge-sel { stroke: var(--primary, #94452b); stroke-width: 3; }
    .match-label { font: 13px 'JetBrains Mono', monospace; fill: var(--text-primary, #1a1a1a); text-anchor: middle; }
    .match-caption { font: italic 12px 'Inter', sans-serif; fill: var(--text-secondary, #666); text-anchor: middle; }
  </style>
  <!-- Path graph: 1-2-3-4-5-6-7 -->
  <g transform="translate(20, 30)">
    <text x="240" y="-10" class="match-caption">Path graph P₇ with matching {1-2, 3-4, 6-7} shown</text>
    <line x1="0" y1="20" x2="70" y2="20" class="match-edge-sel"/>
    <line x1="70" y1="20" x2="140" y2="20" class="match-edge"/>
    <line x1="140" y1="20" x2="210" y2="20" class="match-edge-sel"/>
    <line x1="210" y1="20" x2="280" y2="20" class="match-edge"/>
    <line x1="280" y1="20" x2="350" y2="20" class="match-edge"/>
    <line x1="350" y1="20" x2="420" y2="20" class="match-edge-sel"/>
    <circle cx="0" cy="20" r="14" class="match-node"/><text x="0" y="25" class="match-label">1</text>
    <circle cx="70" cy="20" r="14" class="match-node"/><text x="70" y="25" class="match-label">2</text>
    <circle cx="140" cy="20" r="14" class="match-node"/><text x="140" y="25" class="match-label">3</text>
    <circle cx="210" cy="20" r="14" class="match-node"/><text x="210" y="25" class="match-label">4</text>
    <circle cx="280" cy="20" r="14" class="match-node"/><text x="280" y="25" class="match-label">5</text>
    <circle cx="350" cy="20" r="14" class="match-node"/><text x="350" y="25" class="match-label">6</text>
    <circle cx="420" cy="20" r="14" class="match-node"/><text x="420" y="25" class="match-label">7</text>
  </g>
  <!-- Correspondence -->
  <g transform="translate(20, 90)">
    <text x="240" y="10" class="match-caption">↕  corresponds to list (2, 2, 1, 2) in P₇</text>
  </g>
</svg>

The bijection $g: P_n \to M_n$ works as follows: read the list $(x_1, x_2, \ldots, x_\ell) \in P_n$ from left to right. If $x_i = 1$, leave vertex $i$ isolated (unmatched). If $x_i = 2$, match the current vertex with the next one, then skip ahead.

Since $\lvert P_n \rvert = f_n$, the bijection gives $\lvert M_n \rvert = f_n$: the number of matchings of the path graph on $n$ vertices is the $n$-th Fibonacci number.

### Lucas Numbers

**Definition.** The **Lucas numbers** are defined by $L_0 = 2$, $L_1 = 1$, and $L_n = L_{n-1} + L_{n-2}$ for $n \geq 2$, giving $(L_n) = (2, 1, 3, 4, 7, 11, 18, \ldots)$.

**Algebraic approach.** The OGF is derived the same way as Fibonacci:

$$L(x) = \frac{2-x}{1-x-x^2}.$$

Since the denominator factors as $(1-ax)(1-bx)$ with $a = \frac{1+\sqrt{5}}{2}$ and $b = \frac{1-\sqrt{5}}{2}$ (same as Fibonacci), partial fractions give the closed form:

$$L_n = a^n + b^n = \left(\frac{1+\sqrt{5}}{2}\right)^n + \left(\frac{1-\sqrt{5}}{2}\right)^n.$$

Compare with Binet's formula for Fibonacci: the Fibonacci formula has $a^{n+1}$ and divides by $\sqrt{5}$, while the Lucas formula has $a^n$ and adds rather than subtracts.

**Combinatorial interpretation.** Lucas numbers count matchings of the **$n$-cycle graph** $C_n$ (vertices $1, 2, \ldots, n$ arranged in a circle, with edges connecting consecutive vertices and $n$ back to $1$).

<svg viewBox="0 0 440 180" xmlns="http://www.w3.org/2000/svg" style="max-width:440px; margin: 1.5rem auto; display:block;">
  <style>
    .cy-node { fill: var(--surface-container-low, #f5f5f5); stroke: var(--primary, #94452b); stroke-width: 2; }
    .cy-edge { stroke: var(--text-secondary, #666); stroke-width: 2; }
    .cy-label { font: 13px 'JetBrains Mono', monospace; fill: var(--text-primary, #1a1a1a); text-anchor: middle; }
    .cy-caption { font: italic 12px 'Inter', sans-serif; fill: var(--text-secondary, #666); text-anchor: middle; }
  </style>
  <!-- C3 triangle -->
  <g transform="translate(90, 90)">
    <text x="0" y="-75" class="cy-caption">C₃ (L₃ = 4 matchings)</text>
    <line x1="0" y1="-55" x2="-48" y2="28" class="cy-edge"/>
    <line x1="0" y1="-55" x2="48" y2="28" class="cy-edge"/>
    <line x1="-48" y1="28" x2="48" y2="28" class="cy-edge"/>
    <circle cx="0" cy="-55" r="12" class="cy-node"/><text x="0" y="-51" class="cy-label">1</text>
    <circle cx="-48" cy="28" r="12" class="cy-node"/><text x="-48" y="32" class="cy-label">2</text>
    <circle cx="48" cy="28" r="12" class="cy-node"/><text x="48" y="32" class="cy-label">3</text>
  </g>
  <!-- C4 square -->
  <g transform="translate(330, 90)">
    <text x="0" y="-75" class="cy-caption">C₄ (L₄ = 7 matchings)</text>
    <line x1="-42" y1="-42" x2="42" y2="-42" class="cy-edge"/>
    <line x1="42" y1="-42" x2="42" y2="42" class="cy-edge"/>
    <line x1="42" y1="42" x2="-42" y2="42" class="cy-edge"/>
    <line x1="-42" y1="42" x2="-42" y2="-42" class="cy-edge"/>
    <circle cx="-42" cy="-42" r="12" class="cy-node"/><text x="-42" y="-38" class="cy-label">1</text>
    <circle cx="42" cy="-42" r="12" class="cy-node"/><text x="42" y="-38" class="cy-label">2</text>
    <circle cx="42" cy="42" r="12" class="cy-node"/><text x="42" y="46" class="cy-label">3</text>
    <circle cx="-42" cy="42" r="12" class="cy-node"/><text x="-42" y="46" class="cy-label">4</text>
  </g>
</svg>

**Claim.** $\lvert C_n \rvert = L_n$ for all $n \geq 3$.

The proof follows the same pattern as the Fibonacci matching result: find a bijection $g: C_n \to C_{n-1} \cup C_{n-2}$ by examining whether a fixed vertex (say vertex 1) is matched or isolated, then apply strong induction.

---

## Catalan Numbers

The Catalan numbers are among the most ubiquitous sequences in all of combinatorics. They satisfy a **nonlinear** recurrence — unlike Fibonacci, the Catalan recurrence involves a sum of products — and their generating function satisfies a quadratic rather than a linear equation.

### Definition and First Values

**Definition.** The **Catalan numbers** are defined by $C_0 = 1$ and

$$C_n = \sum_{k=0}^{n-1} C_k\, C_{n-k-1} \quad \text{for } n \geq 1.$$

Computing the first few values:

| $n$ | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $C_n$ | 1 | 1 | 2 | 5 | 14 | 42 |

For instance, $C_4 = C_0 C_3 + C_1 C_2 + C_2 C_1 + C_3 C_0 = 1 \cdot 5 + 1 \cdot 2 + 2 \cdot 1 + 5 \cdot 1 = 14$.

### Algebraic Derivation

Let $C(x) = \sum_{n=0}^{\infty} C_n x^n$. What is $x \cdot C(x)^2$?

By the convolution formula:

$$C(x)^2 = \sum_{n=0}^{\infty}\left(\sum_{k=0}^{n} C_k\, C_{n-k}\right) x^n,$$

so $x\, C(x)^2 = \sum_{n=0}^{\infty}\left(\sum_{k=0}^{n} C_k\, C_{n-k}\right) x^{n+1} = \sum_{n=1}^{\infty} C_n\, x^n = C(x) - 1$.

This gives the functional equation:

$$x\, C(x)^2 - C(x) + 1 = 0.$$

By the quadratic formula:

$$C(x) = \frac{1 \pm \sqrt{1-4x}}{2x}.$$

### Resolving the Sign

The $\pm$ presents a choice. Let us investigate both branches.

**The $+$ branch:** $C(x) = \frac{1 + \sqrt{1-4x}}{2x}$. Expanding $\sqrt{1-4x}$ by the Binomial Theorem:

$$\sqrt{1-4x} = \sum_{k=0}^{\infty} \binom{1/2}{k}(-4x)^k = 1 + (-4)\tfrac{1}{2}x + \cdots = 1 - 2x - \cdots$$

So $\frac{1 + \sqrt{1-4x}}{2x} = \frac{2 - 2x - \cdots}{2x} = \frac{1}{x} - 1 - \cdots$ This has a $\frac{1}{x}$ term — not a formal power series. Rejected.

**The $-$ branch:** $C(x) = \frac{1 - \sqrt{1-4x}}{2x}$. Now:

$$\frac{1 - (1 - 2x - \cdots)}{2x} = \frac{2x + \cdots}{2x} = 1 + \cdots$$

This is a proper formal power series starting with $C_0 = 1$. Accepted.

Working through the full expansion (and using our earlier result that $\binom{-1/2}{n}(-4)^n = \binom{2n}{n}$ from [Part 2](/2024/06/12/enumerative-combinatorics-generating-functions.html)):

$$C(x) = \sum_{n=0}^{\infty} \frac{1}{n+1}\binom{2n}{n} x^n.$$

**Closed form:**

$$\boxed{C_n = \frac{1}{n+1}\binom{2n}{n}.}$$

### Combinatorial Interpretation: Proper Bracketings

**Definition.** A **proper bracketing** of $n$ pairs of parentheses is a string of $n$ open and $n$ close parentheses where every prefix has at least as many opens as closes.

Let $A_n$ be the set of proper bracketings with $n$ pairs:

- $A_0 = \{\epsilon\}$ (the empty string), so $\lvert A_0 \rvert = 1$.
- $A_1 = \{()\}$, so $\lvert A_1 \rvert = 1$.
- $A_2 = \{()()\,,\, (())\}$, so $\lvert A_2 \rvert = 2$.
- $A_3 = \{()()()\,,\, (())()\,,\, ()(())\,,\, ((()))\,,\, (()())\}$, so $\lvert A_3 \rvert = 5$.

**Proposition.** $\lvert A_n \rvert = C_n$ for all $n \geq 0$.

*Proof.* By strong induction.

**Base cases:** $\lvert A_0 \rvert = 1 = C_0$, $\lvert A_1 \rvert = 1 = C_1$, $\lvert A_2 \rvert = 2 = C_2$, $\lvert A_3 \rvert = 5 = C_3$. All match.

**Inductive step:** Let $k \geq 0$ and assume $\lvert A_j \rvert = C_j$ for all $j < k$. We show $\lvert A_k \rvert = C_k$.

Take any proper bracketing $B \in A_k$. The first character is an open parenthesis "(". It has a matching close ")". Write:

$$B = (\underbrace{U}_{\text{inner}})\ \underbrace{V}_{\text{tail}}$$

where $U$ is the proper bracketing *inside* the first matched pair, and $V$ is the proper bracketing *after* it.

<svg viewBox="0 0 400 80" xmlns="http://www.w3.org/2000/svg" style="max-width:400px; margin: 1.5rem auto; display:block;">
  <style>
    .br-text { font: 18px 'JetBrains Mono', monospace; fill: var(--text-primary, #1a1a1a); }
    .br-accent { font: 18px 'JetBrains Mono', monospace; fill: var(--primary, #94452b); font-weight: bold; }
    .br-label { font: italic 13px 'Inter', sans-serif; fill: var(--text-secondary, #666); text-anchor: middle; }
    .br-brace { stroke: var(--primary, #94452b); stroke-width: 1.5; fill: none; }
  </style>
  <text x="30" y="35" class="br-accent">(</text>
  <text x="80" y="35" class="br-text">U</text>
  <text x="130" y="35" class="br-accent">)</text>
  <text x="210" y="35" class="br-text">V</text>
  <!-- Braces -->
  <path d="M 50,45 Q 80,60 110,45" class="br-brace"/>
  <text x="80" y="72" class="br-label">∈ Aⱼ</text>
  <path d="M 160,45 Q 210,60 260,45" class="br-brace"/>
  <text x="210" y="72" class="br-label">∈ A_{k-j-1}</text>
</svg>

If $U \in A_j$, then $V \in A_{k-j-1}$ (the remaining $k - j - 1$ pairs). The map $B \mapsto (U, V)$ is a **bijection** between $A_k$ and $\bigcup_{j=0}^{k-1} A_j \times A_{k-j-1}$.

Therefore:

$$\lvert A_k \rvert = \left\lvert\bigcup_{j=0}^{k-1} A_j \times A_{k-j-1}\right\rvert = \sum_{j=0}^{k-1} \lvert A_j \rvert \cdot \lvert A_{k-j-1} \rvert = \sum_{j=0}^{k-1} C_j\, C_{k-j-1} = C_k.$$

By strong induction, $\lvert A_n \rvert = C_n$ for all $n \geq 0$. $\square$

> Catalan numbers count an extraordinary variety of structures: proper bracketings, binary trees, non-crossing partitions, Dyck paths, triangulations of polygons, and many more. In each case, the same recursive decomposition — splitting a structure into a "first piece" and a "remaining piece" — yields the same recurrence $C_n = \sum C_j C_{n-j-1}$.

---

## Beyond the Course: A Glimpse Ahead

The tools we have built — PIE, induction, generating functions, and number families — form the foundation of enumerative combinatorics. But the subject extends much further. Here are a few directions the final lectures pointed toward.

### Partially Ordered Sets

A **poset** $(P, \preceq)$ is a set $P$ equipped with a partial order: a reflexive, antisymmetric, transitive relation. For example, binary strings of length 3 ordered coordinatewise form a poset (the Boolean lattice $B_3$).

A **chain** is a totally ordered subset; an **antichain** is a totally unordered subset (no two elements are comparable).

**Theorem (Dilworth).** In any finite poset, the size of the largest antichain equals the minimum number of chains needed to cover the entire poset.

This theorem connects the "width" of a poset (largest antichain) to its "decomposition complexity" (chain covers) — a beautiful min-max duality.

### Graph Theory and Spectral Methods

Combinatorics meets linear algebra through the **adjacency matrix** $A = [a_{ij}]$ of a graph, where $a_{ij} = 1$ if vertices $i$ and $j$ are connected.

A remarkable theorem: the number of **cyclic walks** of length $2\ell$ on the cube graph $Q_2$ is $\frac{1}{4}(3^{2\ell} + 3)$. The proof uses the eigenvalues of the adjacency matrix — the trace of $A^{2\ell}$ equals both $\sum_i (A^{2\ell})_{ii}$ (counting closed walks) and $\sum_i \lambda_i^{2\ell}$ (summing eigenvalue powers).

A **spanning tree** of a connected graph is a minimal subset of edges that keeps all vertices connected (equivalently, a maximal acyclic subgraph). **Kirchhoff's Matrix Tree Theorem** counts spanning trees as a determinant of a modified Laplacian matrix — another instance where linear algebra provides exact combinatorial answers.

These connections to order theory, linear algebra, and algebraic combinatorics show that enumerative techniques are just the beginning of a much larger landscape.

---

## Series Conclusion

Across these three posts, we have built a toolkit for exact counting:

- [**Part 1**](/2024/06/05/enumerative-combinatorics-algebraic-methods.html): PIE handles overlapping cases; induction proves universal statements.
- [**Part 2**](/2024/06/12/enumerative-combinatorics-generating-functions.html): Generating functions transform counting into algebra; the Binomial Theorem and partial fractions extract closed forms.
- **Part 3** (this post): Number families — multinomials, Fibonacci, Lucas, Catalan — illustrate how a single algebraic identity can count dozens of apparently different structures.

The central lesson of enumerative combinatorics is that counting problems, which seem to require case-by-case cleverness, often yield to systematic algebraic methods. Encode your sequence as a generating function, manipulate it with formal power series algebra, and read off the answer.
