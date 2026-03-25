---
layout: post
title: "Real Analysis I: Topology of the Real Line"
date: 2024-05-08
tags: [Math, Analysis, Real Analysis]
---

This is Part 2 of a four-part series on Real Analysis I. In [Part 1](/2024/05/01/real-analysis-real-numbers.html), we built the real numbers from their axioms and discovered the Completeness Axiom — the property that fills the gaps in $\mathbb{Q}$. Now we need a language for *nearness*. To talk about limits, continuity, and convergence precisely, we need to formalize what it means for points to be "close to" each other and "close to" a set. That language is the topology of the real line.

## What This Post Covers

- **Intervals and Neighborhoods** — The building blocks of "nearness"
- **Interior, Exterior, and Boundary Points** — Three ways a point can relate to a set
- **Open and Closed Sets** — The two fundamental types, and why they are not opposites
- **Unions and Intersections** — How open and closed sets behave under set operations
- **Accumulation Points and Closure** — The limiting behavior of sets

---

## Intervals and Neighborhoods

The basic building blocks of topology on $\mathbb{R}$ are intervals. Given $a, b \in \mathbb{R}$ with $a < b$:

| Notation | Definition | Type |
|---|---|---|
| $(a, b)$ | $\\{x \in \mathbb{R} : a < x < b\\}$ | Open interval |
| $[a, b]$ | $\\{x \in \mathbb{R} : a \leq x \leq b\\}$ | Closed interval |
| $(a, b]$ | $\\{x \in \mathbb{R} : a < x \leq b\\}$ | Half-open |
| $[a, b)$ | $\\{x \in \mathbb{R} : a \leq x < b\\}$ | Half-open |

We also have rays: $(a, \infty) = \\{x : x > a\\}$, $(-\infty, a] = \\{x : x \leq a\\}$, and $(-\infty, \infty) = \mathbb{R}$.

The distance between two points $x, y \in \mathbb{R}$ is $\lvert x - y \rvert$. This lets us define the most important concept in point-set topology.

**Definition.** Given $x \in \mathbb{R}$ and $\epsilon > 0$, the **$\epsilon$-neighborhood** of $x$ is

$$
N(x;\, \epsilon) = \{y \in \mathbb{R} : |x - y| < \epsilon\} = (x - \epsilon,\, x + \epsilon).
$$

The **deleted neighborhood** (or **punctured neighborhood**) of $x$ with radius $\epsilon$ is

$$
N^*(x;\, \epsilon) = N(x;\, \epsilon) \setminus \{x\} = \{y \in \mathbb{R} : 0 < |x - y| < \epsilon\}.
$$

<svg viewBox="0 0 440 90" style="max-width:440px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax2 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; }
    .lbl2 { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); }
    .lbl2s { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-secondary, #555); }
    .nbhd { fill: rgba(148, 69, 43, 0.1); }
    .ep-line { stroke: var(--primary, #94452b); stroke-width: 1.2; }
  </style>
  <!-- Axis -->
  <line x1="30" y1="50" x2="410" y2="50" class="ax2" />
  <!-- Neighborhood region -->
  <rect x="120" y="40" width="200" height="20" rx="10" class="nbhd" />
  <!-- Open endpoints -->
  <circle cx="120" cy="50" r="4" fill="var(--background, #fff)" stroke="var(--primary, #94452b)" stroke-width="2" />
  <circle cx="320" cy="50" r="4" fill="var(--background, #fff)" stroke="var(--primary, #94452b)" stroke-width="2" />
  <!-- Center point -->
  <circle cx="220" cy="50" r="4" fill="var(--primary, #94452b)" />
  <!-- Labels -->
  <text x="220" y="75" text-anchor="middle" class="lbl2" font-style="italic">x</text>
  <text x="120" y="75" text-anchor="middle" class="lbl2s">x - ε</text>
  <text x="320" y="75" text-anchor="middle" class="lbl2s">x + ε</text>
  <!-- Epsilon arrows -->
  <line x1="220" y1="30" x2="120" y2="30" class="ep-line" marker-end="url(#arr2)" />
  <line x1="220" y1="30" x2="320" y2="30" class="ep-line" marker-end="url(#arr2)" />
  <text x="170" y="25" text-anchor="middle" class="lbl2s" fill="var(--primary, #94452b)">ε</text>
  <text x="270" y="25" text-anchor="middle" class="lbl2s" fill="var(--primary, #94452b)">ε</text>
  <!-- Caption -->
  <text x="220" y="88" text-anchor="middle" class="lbl2s" font-style="italic">N(x; ε)</text>
  <defs>
    <marker id="arr2" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="none" stroke="var(--primary, #94452b)" stroke-width="1.2" />
    </marker>
  </defs>
</svg>

*An $\epsilon$-neighborhood is a symmetric open interval centered at $x$. It captures all points within distance $\epsilon$ of $x$.*

Neighborhoods formalize the idea of "closeness": to say $y$ is close to $x$ means $y \in N(x; \epsilon)$ for some small $\epsilon$.

---

## Interior, Exterior, and Boundary Points

Given a set $S \subseteq \mathbb{R}$, every point $x \in \mathbb{R}$ falls into exactly one of three categories.

**Definition.** Let $S \subseteq \mathbb{R}$ and $x \in \mathbb{R}$.

- $x$ is an **interior point** of $S$ if there exists $\epsilon > 0$ such that $N(x; \epsilon) \subseteq S$. The set of all interior points is $\operatorname{int} S$.
- $x$ is an **exterior point** of $S$ if there exists $\epsilon > 0$ such that $N(x; \epsilon) \subseteq \mathbb{R} \setminus S$. The set of all exterior points is $\operatorname{ext} S$.
- $x$ is a **boundary point** of $S$ if for every $\epsilon > 0$, $N(x; \epsilon) \cap S \neq \emptyset$ and $N(x; \epsilon) \cap (\mathbb{R} \setminus S) \neq \emptyset$. The set of all boundary points is $\operatorname{bd} S$ (or $\partial S$).

In words: interior points have a whole neighborhood inside $S$, exterior points have a whole neighborhood outside $S$, and boundary points have *every* neighborhood straddling the edge.

<svg viewBox="0 0 460 110" style="max-width:460px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax3 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; }
    .lbl3 { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); }
    .lbl3s { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--text-secondary, #555); }
    .int-region { fill: rgba(148, 69, 43, 0.12); }
    .ext-region { fill: rgba(42, 122, 226, 0.06); }
  </style>
  <!-- Axis -->
  <line x1="30" y1="55" x2="430" y2="55" class="ax3" />
  <!-- Set S region [a, b] -->
  <rect x="130" y="42" width="200" height="26" rx="3" class="int-region" />
  <!-- Exterior shading -->
  <rect x="30" y="42" width="100" height="26" class="ext-region" />
  <rect x="330" y="42" width="100" height="26" class="ext-region" />
  <!-- Endpoint markers -->
  <circle cx="130" cy="55" r="5" fill="var(--primary, #94452b)" />
  <circle cx="330" cy="55" r="5" fill="var(--primary, #94452b)" />
  <!-- Labels -->
  <text x="130" y="82" text-anchor="middle" class="lbl3" font-style="italic">a</text>
  <text x="330" y="82" text-anchor="middle" class="lbl3" font-style="italic">b</text>
  <text x="230" y="35" text-anchor="middle" class="lbl3s" fill="var(--primary, #94452b)">interior points</text>
  <text x="75" y="35" text-anchor="middle" class="lbl3s" fill="#2a7ae2">exterior</text>
  <text x="385" y="35" text-anchor="middle" class="lbl3s" fill="#2a7ae2">exterior</text>
  <text x="130" y="100" text-anchor="middle" class="lbl3s">boundary</text>
  <text x="330" y="100" text-anchor="middle" class="lbl3s">boundary</text>
  <!-- S label -->
  <text x="230" y="20" text-anchor="middle" class="lbl3">S = [a, b]</text>
</svg>

**Examples.**

1. $S = (0, 1)$: $\operatorname{int} S = (0, 1)$, $\operatorname{bd} S = \\{0, 1\\}$, $\operatorname{ext} S = (-\infty, 0) \cup (1, \infty)$.
2. $S = [0, 1]$: $\operatorname{int} S = (0, 1)$, $\operatorname{bd} S = \\{0, 1\\}$, $\operatorname{ext} S = (-\infty, 0) \cup (1, \infty)$.
3. $S = \mathbb{Q}$: $\operatorname{int} S = \emptyset$, $\operatorname{bd} S = \mathbb{R}$, $\operatorname{ext} S = \emptyset$. Every neighborhood of every real number contains both rationals and irrationals (by density).

**Remark.** For any $S \subseteq \mathbb{R}$ with $S \neq \emptyset$:
- $S \subseteq \operatorname{int} S \cup \operatorname{bd} S$, and $\operatorname{int} S \cap \operatorname{bd} S = \emptyset$.
- $\mathbb{R} = \operatorname{int} S \cup \operatorname{bd} S \cup \operatorname{ext} S$ (disjoint union).

---

## Open and Closed Sets

The two most important classes of sets in topology are defined by their relationship to the boundary.

**Definition.** A set $S \subseteq \mathbb{R}$ is **open** if $S \subseteq \operatorname{int} S$ (equivalently, $S = \operatorname{int} S$). A set $S \subseteq \mathbb{R}$ is **closed** if $\operatorname{bd} S \subseteq S$.

An open set contains none of its boundary; a closed set contains all of it. Equivalently:
- $S$ is open if and only if every point of $S$ is an interior point.
- $S$ is closed if and only if $\mathbb{R} \setminus S$ is open.

> Open and closed are *not* opposites. A set can be both open and closed (like $\mathbb{R}$ and $\emptyset$), or neither (like $[0, 1)$, which contains the boundary point $0$ but not $1$). Most sets are neither.

**Examples.**

- $(a, b)$ is open. $[a, b]$ is closed.
- $\mathbb{R}$ is both open and closed. $\emptyset$ is both open and closed.
- $[0, 1)$ is neither open nor closed.
- $\mathbb{Q}$ is neither open nor closed: $\operatorname{int} \mathbb{Q} = \emptyset \neq \mathbb{Q}$, and $\operatorname{bd} \mathbb{Q} = \mathbb{R} \not\subseteq \mathbb{Q}$.

**Theorem.** $S$ is open if and only if $S \cap \partial S = \emptyset$.

*Proof.* If $S$ is open, then $S = \operatorname{int} S$. Since $\operatorname{int} S \cap \operatorname{bd} S = \emptyset$, we get $S \cap \partial S = \emptyset$. Conversely, if $S \cap \partial S = \emptyset$, then every $x \in S$ is not a boundary point. Since $x \in S$, it cannot be an exterior point either. So $x \in \operatorname{int} S$. Thus $S \subseteq \operatorname{int} S$, and $S$ is open. $\square$

### How Open and Closed Sets Combine

**Theorem.** Let $S \subseteq \mathbb{R}$.
1. $S$ is open if and only if $S = \operatorname{int} S$.
2. $S$ is open if and only if $\mathbb{R} \setminus S$ is closed.

The second statement gives us a duality: every theorem about open sets translates into a theorem about closed sets via complementation. This is powered by **De Morgan's Laws**.

---

## Unions and Intersections

The following theorem governs how open and closed sets behave under set operations.

**Theorem.**
1. The union of an arbitrary collection of open sets is open.
2. The intersection of a *finite* collection of open sets is open.

*Proof of (a).* Let $\Lambda$ be a nonempty index set and let $\\{A_\alpha\\}_{\alpha \in \Lambda}$ be a collection of open sets. Let $x \in \bigcup_{\alpha \in \Lambda} A_\alpha$. Then $x \in A_\alpha$ for some $\alpha \in \Lambda$. Since $A_\alpha$ is open, there exists $\epsilon > 0$ such that $N(x; \epsilon) \subseteq A_\alpha$. Since $A_\alpha \subseteq \bigcup_{\alpha \in \Lambda} A_\alpha$, we have $N(x; \epsilon) \subseteq \bigcup_{\alpha \in \Lambda} A_\alpha$. So $x \in \operatorname{int}\left(\bigcup_{\alpha \in \Lambda} A_\alpha\right)$. $\square$

*Proof of (b).* Let $A_1, \ldots, A_n$ be open, and let $x \in \bigcap_{i=1}^n A_i$. Then $x \in A_i$ for each $i = 1, \ldots, n$. Since each $A_i$ is open, there exist $\epsilon_1, \ldots, \epsilon_n > 0$ with $N(x; \epsilon_i) \subseteq A_i$. Define $\epsilon = \min\\{\epsilon_1, \ldots, \epsilon_n\\}$. Since $n$ is finite, $\epsilon > 0$. Then $N(x; \epsilon) \subseteq N(x; \epsilon_i) \subseteq A_i$ for all $i$, so $N(x; \epsilon) \subseteq \bigcap_{i=1}^n A_i$. $\square$

The word "finite" in (b) is essential. Here is why:

**Counterexample.** Let $A_n = (-1/n,\, 1/n)$ for $n \in \mathbb{N}$. Each $A_n$ is open. But

$$
\bigcap_{n=1}^{\infty} A_n = \{0\},
$$

which is *not* open (no neighborhood of $0$ fits inside $\\{0\\}$). The trick $\epsilon = \min\\{\epsilon_1, \ldots, \epsilon_n\\}$ breaks down for infinitely many sets because the infimum of positive numbers can be $0$.

By De Morgan's Laws, the dual statements hold for closed sets:

**Corollary.**
1. The intersection of an arbitrary collection of closed sets is closed.
2. The union of a *finite* collection of closed sets is closed.

---

## Accumulation Points and Closure

The topological concepts so far have been about *sets*. Now we ask about individual points and their relationship to the "limit behavior" of a set.

**Definition.** Let $S \subseteq \mathbb{R}$. A point $x \in \mathbb{R}$ is called an **accumulation point** (or **limit point** or **cluster point**) of $S$ if

$$
\text{for every } \epsilon > 0, \quad N^*(x; \epsilon) \cap S \neq \emptyset.
$$

That is, every deleted neighborhood of $x$ contains a point of $S$. The set of all accumulation points of $S$ is denoted $S'$.

**Definition.** A point $x \in \mathbb{R}$ is an **isolated point** of $S \subseteq \mathbb{R}$ if there exists $\epsilon > 0$ such that $N(x; \epsilon) \cap S = \\{x\\}$.

An isolated point belongs to $S$ but has a neighborhood containing no *other* points of $S$. An accumulation point may or may not belong to $S$, but every neighborhood contains infinitely many points of $S$ (you can prove this).

**Examples.**

1. $S = (0, 1) \cup (1, 5)$: The point $1$ is an accumulation point (every neighborhood hits $(0,1)$ or $(1,5)$), even though $1 \notin S$.
2. $A = (1, 7] \cup \\{10\\}$: $\operatorname{int} A = (1, 7)$. The point $10$ is an isolated point. $A' = [1, 7]$.
3. $S = \\{1/n : n \in \mathbb{N}\\}$: The only accumulation point is $0$, and $0 \notin S$. Every point $1/n \in S$ is isolated.

For the last example: to see that $0 \in S'$, let $\epsilon > 0$. By the Archimedean Property, there exists $n \in \mathbb{N}$ with $1/n < \epsilon$. Then $1/n \in N^*(0; \epsilon) \cap S$. On the other hand, no point $1/n$ is an accumulation point: the neighborhood $N^*(1/n;\, 1/n - 1/(n+1))$ contains no points of $S$.

### Closure

**Definition.** Let $S \subseteq \mathbb{R}$. The **closure** of $S$, denoted $\operatorname{cl} S$ or $\overline{S}$, is

$$
\operatorname{cl} S = S \cup S'.
$$

The closure adds to $S$ all the points that $S$ "approaches." Alternatively, $\operatorname{cl} S$ is the intersection of all closed sets containing $S$ — the smallest closed set that contains $S$.

**Examples.**
- $\operatorname{cl}(0, 1) = [0, 1]$
- $\operatorname{cl} \mathbb{Q} = \mathbb{R}$ (every real number is a limit of rationals)
- $\operatorname{cl}\\{1/n : n \in \mathbb{N}\\} = \\{1/n : n \in \mathbb{N}\\} \cup \\{0\\} = \\{0, 1, 1/2, 1/3, \ldots\\}$

We also have the decomposition $\operatorname{cl} S = \operatorname{int} S \cup \operatorname{bd} S$.

### Closed Sets and Accumulation Points

The following theorem gives the most useful characterization of closed sets.

**Theorem.** Let $S \subseteq \mathbb{R}$. Then $S$ is closed if and only if $S' \subseteq S$.

*Proof.* $(\Rightarrow)$ Suppose $S$ is closed and $x \in S'$. We show $x \in S$. If $x \notin S$, then $x \in \mathbb{R} \setminus S$. Since $S$ is closed, $\mathbb{R} \setminus S$ is open. So there exists $\epsilon > 0$ with $N(x; \epsilon) \subseteq \mathbb{R} \setminus S$. But then $N^*(x; \epsilon) \cap S = \emptyset$, contradicting $x \in S'$. So $x \in S$.

$(\Leftarrow)$ Suppose $S' \subseteq S$. We show $\mathbb{R} \setminus S$ is open. Let $x \in \mathbb{R} \setminus S$. Since $x \notin S$ and $x \notin S'$ (because $S' \subseteq S$ and $x \notin S$), there exists $\epsilon > 0$ with $N^*(x; \epsilon) \cap S = \emptyset$. Since $x \notin S$, we also have $x \notin S$, so $N(x; \epsilon) \cap S = \emptyset$. Thus $N(x; \epsilon) \subseteq \mathbb{R} \setminus S$, and $\mathbb{R} \setminus S$ is open. $\square$

> A closed set is one that "holds onto" all its limit points. If a sequence of points in $S$ converges to some limit, that limit must be in $S$. This is the topological essence of closedness, and it foreshadows the sequence characterization we will develop in Part 3.

**Theorem.** For any $S \subseteq \mathbb{R}$:
1. $\operatorname{cl} S$ is closed.
2. $S$ is closed if and only if $S = \operatorname{cl} S$.

---

## Looking Ahead

We now have the topological vocabulary to describe how sets and points relate on the real line. Interior points, boundary points, open sets, closed sets, and accumulation points give us a precise language for "nearness."

With this language in place, we can finally ask the central question of analysis: what does it mean for a sequence to *converge*? In [Part 3: Sequences and Convergence](/2024/06/05/real-analysis-sequences.html), we define convergence, prove the Monotone Convergence Theorem, develop the Cauchy criterion, and arrive at the Bolzano-Weierstrass theorem — the cornerstone results that make analysis work.
