---
layout: post
title: "Real Analysis I: Building the Real Numbers"
date: 2024-05-01
tags: [Math, Analysis, Real Analysis]
---

These are my compiled notes from Real Analysis I, reorganized into a narrative that builds the theory from first principles. Real analysis is the subject that puts calculus on solid ground. You used limits, derivatives, and integrals throughout calculus — but what *is* a limit, really? Why can you take the square root of 2? Why does a continuous function on a closed interval achieve its maximum? These questions seem trivial until you try to answer them precisely, and the answers turn out to be deep.

This is Part 1 of a four-part series. Here we build the real number system from its axioms, discover the Completeness Axiom that separates $\mathbb{R}$ from $\mathbb{Q}$, and explore the consequences that make analysis possible.

## What This Post Covers

- **Natural Numbers and Induction** — The principle of mathematical induction and its equivalence to well-ordering
- **Ordered Fields** — The axioms that govern arithmetic and order on $\mathbb{R}$
- **The Gaps in the Rationals** — Why $\sqrt{2}$ doesn't exist in $\mathbb{Q}$, and what that means
- **Bounded Sets and Completeness** — The single axiom that makes the reals special
- **The Archimedean Property** — No infinitely large or infinitely small real numbers
- **Density of the Rationals** — Between any two reals, a rational (and an irrational) lurks

---

## Natural Numbers and Induction

We begin where counting begins: with $\mathbb{N} = \\{1, 2, 3, \ldots\\}$.

The natural numbers carry the basic arithmetic operations — addition ($m + n$), subtraction ($\ell - n = m$ when defined), multiplication (repeated addition), and a natural ordering ($n < \ell$ if $\exists\, m \in \mathbb{N}$ such that $n + m = \ell$). But the most important structural property of $\mathbb{N}$ is one you might not expect: *every nonempty subset has a smallest element*.

**Axiom (Well-Ordering).** If $\emptyset \neq S \subseteq \mathbb{N}$, then $S$ has a least element: there exists $m \in S$ such that $m \leq k$ for all $k \in S$.

This axiom powers one of the most useful proof techniques in mathematics.

**Theorem (Principle of Mathematical Induction).** Let $P(n)$ be a statement that is either true or false for each $n \in \mathbb{N}$. Then $P(n)$ is true for all $n \in \mathbb{N}$, provided that:

1. **(Base case)** $P(1)$ is true, and
2. **(Inductive step)** For each $k \in \mathbb{N}$, if $P(k)$ is true, then $P(k+1)$ is true.

*Proof.* Suppose (a) and (b) hold, but $P(n)$ is false for some $n \in \mathbb{N}$. Let $S = \\{n \in \mathbb{N} : P(n) \text{ is false}\\}$. Then $\emptyset \neq S \subseteq \mathbb{N}$. By the Well-Ordering Axiom, $S$ has a least element $m$, so $m \leq k$ for all $k \in S$. By (a), $1 \notin S$, so $m > 1$, which means $m - 1 \in \mathbb{N}$. Since $m$ is the least element of $S$, we have $m - 1 \notin S$, so $P(m-1)$ is true. By (b) with $k = m - 1$, we get $P(m)$ is true. But then $m \notin S$, contradicting our choice of $m$. $\square$

> The Principle of Mathematical Induction and the Well-Ordering Axiom are logically equivalent — each can be derived from the other. They are two faces of the same deep property of $\mathbb{N}$.

### Induction in Practice

**Example.** Prove that $1^2 + 2^2 + \cdots + n^2 = \dfrac{n(n+1)(2n+1)}{6}$ for all $n \in \mathbb{N}$.

*Proof.* Let $P(n)$ denote the statement $1^2 + 2^2 + \cdots + n^2 = \frac{n(n+1)(2n+1)}{6}$.

**Base case:** $P(1)$ reads $1^2 = \frac{1 \cdot 2 \cdot 3}{6} = 1$. True.

**Inductive step:** Assume $P(k)$ is true. We show $P(k+1)$ holds:

$$
\begin{aligned}
1^2 + 2^2 + \cdots + k^2 + (k+1)^2 &= \frac{k(k+1)(2k+1)}{6} + (k+1)^2 \\
&= (k+1)\left[\frac{k(2k+1)}{6} + (k+1)\right] \\
&= (k+1) \cdot \frac{k(2k+1) + 6(k+1)}{6} \\
&= (k+1) \cdot \frac{2k^2 + 7k + 6}{6} \\
&= \frac{(k+1)(k+2)(2k+3)}{6}.
\end{aligned}
$$

This is exactly $P(k+1)$. By the Principle of Mathematical Induction, $P(n)$ is true for all $n \in \mathbb{N}$. $\square$

There is also a **Generalized PMI**: if the base case starts at $m$ instead of $1$, the conclusion holds for all $n \geq m$.

---

## Ordered Fields

We have been using real numbers our entire mathematical lives. But what *are* they, axiomatically? Real analysis begins by assuming the existence of a set $\mathbb{R}$ equipped with two operations — addition ($+$) and multiplication ($\cdot$) — satisfying a precise list of axioms.

### The Field Axioms

The following properties hold for all $x, y, z \in \mathbb{R}$:

**Addition axioms:**

| | Axiom | Name |
|---|---|---|
| A1 | $x + y \in \mathbb{R}$, and if $x = w$, $y = z$, then $x + y = w + z$ | Closure |
| A2 | $x + y = y + x$ | Commutativity |
| A3 | $x + (y + z) = (x + y) + z$ | Associativity |
| A4 | $\exists\, 0 \in \mathbb{R}$ such that $0 + x = x$ for all $x$ | Existence of zero |
| A5 | For each $x$, $\exists\, (-x) \in \mathbb{R}$ such that $x + (-x) = 0$ | Existence of negatives |

**Multiplication axioms:**

| | Axiom | Name |
|---|---|---|
| M1 | $x \cdot y \in \mathbb{R}$, and if $x = w$, $y = z$, then $x \cdot y = w \cdot z$ | Closure |
| M2 | $x \cdot y = y \cdot x$ | Commutativity |
| M3 | $x \cdot (y \cdot z) = (x \cdot y) \cdot z$ | Associativity |
| M4 | $\exists\, 1 \in \mathbb{R}$ such that $1 \cdot x = x$ for all $x$ | Existence of unity |
| M5 | For each $x \neq 0$, $\exists\, x^{-1} \in \mathbb{R}$ such that $x \cdot x^{-1} = 1$ | Existence of inverses |

**Distributive law:**

| | Axiom | |
|---|---|---|
| D1 | $x \cdot (y + z) = x \cdot y + x \cdot z$ | Distributivity |

Any set satisfying A1–A5, M1–M5, and D1 is called a **field**. Both $\mathbb{Q}$ and $\mathbb{R}$ are fields. The set of irrational numbers $\mathbb{R} \setminus \mathbb{Q}$ is *not* a field (it is not closed under addition: $\sqrt{2} + (-\sqrt{2}) = 0 \notin \mathbb{R} \setminus \mathbb{Q}$).

### Consequences of the Field Axioms

From these eleven axioms, we can derive all the familiar algebraic rules. Here is one that reveals the power of the axiomatic approach.

**Theorem.** For any $x \in \mathbb{R}$, $x \cdot 0 = 0$.

*Proof.* We compute:

$$
\begin{aligned}
x \cdot 0 &= x \cdot (0 + 0) & \text{(by A4)} \\
&= x \cdot 0 + x \cdot 0 & \text{(by D1)}.
\end{aligned}
$$

Adding $-(x \cdot 0)$ to both sides (which exists by A5):

$$
0 = x \cdot 0 + x \cdot 0 + (-(x \cdot 0)) = x \cdot 0 + 0 = x \cdot 0. \quad \square
$$

Other familiar facts that follow from the axioms include:
- **Cancellation:** If $x + z = y + z$, then $x = y$
- $(-1) \cdot x = -x$
- $(-x)(-y) = xy$
- $xy = 0$ if and only if $x = 0$ or $y = 0$

### The Order Axioms

To do analysis, we need more than algebra — we need the ability to compare elements. We assume there is a relation "$<$" on $\mathbb{R}$ satisfying:

| | Axiom | Name |
|---|---|---|
| O1 | For any $x, y \in \mathbb{R}$, exactly one holds: $x < y$, $y < x$, or $x = y$ | Trichotomy |
| O2 | If $x < y$ and $y < z$, then $x < z$ | Transitivity |
| O3 | If $x < y$, then $x + z < y + z$ for all $z$ | Addition preserves order |
| O4 | If $x < y$ and $z > 0$, then $xz < yz$ | Positive multiplication preserves order |

A field with these four properties is called an **ordered field**. Both $\mathbb{Q}$ and $\mathbb{R}$ are ordered fields. But they are *not* the same — and the difference matters enormously.

### Absolute Value

Given the order axioms, we define the **absolute value**:

$$
|x| = \begin{cases} x & \text{if } x \geq 0 \\ -x & \text{if } x < 0 \end{cases}
$$

The absolute value measures *distance* — $\lvert x - y \rvert$ is the distance between $x$ and $y$ on the number line. Its most important property is the **triangle inequality**:

$$
|a + b| \leq |a| + |b|
$$

and its useful cousin, the **reverse triangle inequality**:

$$
\bigl||a| - |b|\bigr| \leq |a - b|.
$$

These inequalities are the workhorses of analysis. Nearly every $\epsilon$-proof you will see uses the triangle inequality at its core.

---

## The Gaps in the Rationals

The rationals $\mathbb{Q}$ form an ordered field — they satisfy all the axioms above. So why do we need $\mathbb{R}$?

A real number $r$ is **rational** if there exist $p, q \in \mathbb{Z}$ with $q \neq 0$ such that $r = p/q$. A real number that is not rational is **irrational**. The rationals are closed under addition and multiplication (the sum and product of two rationals are rational), but they have *holes*.

**Theorem.** Let $p$ be a prime number. Then $\sqrt{p}$ is irrational.

*Proof.* Suppose $\sqrt{p}$ is rational. Then there exist $m, n \in \mathbb{Z}$ with $n \neq 0$ such that $\sqrt{p} = m/n$. We may assume $m$ and $n$ share no common prime factor.

Squaring both sides gives $p = m^2/n^2$, so $pn^2 = m^2$. This means $p$ divides $m^2$. Since $p$ is prime, $p$ must divide $m$ — write $m = kp$ for some integer $k$.

Then $pn^2 = (kp)^2 = k^2 p^2$, so $n^2 = k^2 p$. This means $p$ divides $n^2$, and since $p$ is prime, $p$ divides $n$.

But now $p$ divides both $m$ and $n$, contradicting our assumption that they share no common prime factor. $\square$

This is a problem: the equation $x^2 = 2$ has no solution in $\mathbb{Q}$. The number line, as the rationals see it, has gaps — places where a number *should* be but isn't. The real numbers exist precisely to fill those gaps.

---

## Bounded Sets and the Completeness Axiom

To state the axiom that fills the gaps, we need the language of bounds.

**Definition.** Let $S \subseteq \mathbb{R}$.

- $S$ is **bounded above** if there exists $M \in \mathbb{R}$ such that $s \leq M$ for all $s \in S$. Such an $M$ is an **upper bound** for $S$.
- $S$ is **bounded below** if there exists $m \in \mathbb{R}$ such that $m \leq s$ for all $s \in S$. Such an $m$ is a **lower bound** for $S$.
- $S$ is **bounded** if it is both bounded above and bounded below.

A set can have many upper bounds. The interesting one is the *smallest*.

**Definition.** Let $S \subseteq \mathbb{R}$ with $S \neq \emptyset$. A real number $M$ is called the **least upper bound** (or **supremum**) of $S$, written $M = \sup S$, if:

1. $s \leq M$ for all $s \in S$ (i.e., $M$ is an upper bound), and
2. For all $M' < M$, there exists $s' \in S$ such that $s' > M'$ (i.e., no smaller number is an upper bound).

The **greatest lower bound** (or **infimum**), written $\inf S$, is defined analogously.

<svg viewBox="0 0 500 120" style="max-width:500px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; }
    .label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); }
    .label-sm { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-secondary, #555); }
    .set-region { fill: rgba(148, 69, 43, 0.12); stroke: var(--primary, #94452b); stroke-width: 2; }
    .marker { fill: var(--primary, #94452b); }
    .marker-open { fill: var(--background, #fff); stroke: var(--primary, #94452b); stroke-width: 2; }
    .brace { stroke: var(--text-secondary, #555); stroke-width: 1.2; fill: none; }
  </style>
  <!-- Axis -->
  <line x1="30" y1="60" x2="470" y2="60" class="axis" />
  <line x1="30" y1="55" x2="30" y2="65" class="axis" />
  <line x1="470" y1="55" x2="470" y2="65" class="axis" />
  <!-- Set region -->
  <rect x="120" y="50" width="200" height="20" rx="3" class="set-region" />
  <!-- Open endpoint at left (inf S might not be in S) -->
  <circle cx="120" cy="60" r="5" class="marker-open" />
  <!-- Closed endpoint at right -->
  <circle cx="320" cy="60" r="5" class="marker" />
  <!-- sup S marker -->
  <line x1="320" y1="35" x2="320" y2="55" stroke="var(--primary, #94452b)" stroke-width="1.5" stroke-dasharray="4,3" />
  <text x="320" y="28" text-anchor="middle" class="label" fill="var(--primary, #94452b)">sup S</text>
  <!-- inf S marker -->
  <line x1="120" y1="35" x2="120" y2="55" stroke="var(--primary, #94452b)" stroke-width="1.5" stroke-dasharray="4,3" />
  <text x="120" y="28" text-anchor="middle" class="label" fill="var(--primary, #94452b)">inf S</text>
  <!-- S label -->
  <text x="220" y="95" text-anchor="middle" class="label">S</text>
  <!-- Upper bounds region -->
  <line x1="320" y1="100" x2="460" y2="100" stroke="var(--text-secondary, #555)" stroke-width="1.5" />
  <text x="390" y="115" text-anchor="middle" class="label-sm">upper bounds</text>
  <!-- Arrow on upper bounds -->
  <polygon points="460,96 460,104 470,100" fill="var(--text-secondary, #555)" />
</svg>

*The supremum is the "tightest" upper bound — any smaller value fails to bound $S$ from above. The infimum is the tightest lower bound. Note that $\sup S$ may or may not belong to $S$ itself.*

**Example.** Let $S = \\{1/n : n \in \mathbb{N}\\} = \\{1, 1/2, 1/3, 1/4, \ldots\\}$. Then:

- $\sup S = \max S = 1$ (achieved at $n = 1$)
- $\inf S = 0$ (but $0 \notin S$, so there is no minimum)

Now comes the axiom that separates $\mathbb{R}$ from $\mathbb{Q}$.

**The Completeness Axiom.** Every nonempty subset of $\mathbb{R}$ that is bounded above has its supremum in $\mathbb{R}$.

> This single axiom is what separates $\mathbb{R}$ from $\mathbb{Q}$. Everything else in real analysis flows from it. The rationals satisfy all the field and order axioms, but they fail completeness: the set $\\{r \in \mathbb{Q} : r^2 < 2\\}$ is bounded above in $\mathbb{Q}$ but has no supremum in $\mathbb{Q}$.

The completeness axiom also gives us infima for free: if $S$ is nonempty and bounded below, consider $T = \\{-s : s \in S\\}$. Then $T$ is bounded above, so $\sup T$ exists, and $\inf S = -\sup T$.

One useful consequence is the behavior of sums of sets.

**Theorem.** Suppose $A, B \subseteq \mathbb{R}$, $A \neq \emptyset$, $B \neq \emptyset$. Define $C = \\{a + b : a \in A,\, b \in B\\}$. If $\sup A$ and $\sup B$ exist, then $\sup C = \sup A + \sup B$.

---

## The Archimedean Property

The Completeness Axiom has an immediate and powerful consequence: the natural numbers are not bounded above.

**Theorem (Archimedean Property).** $\mathbb{N}$ is not bounded above in $\mathbb{R}$.

*Proof.* Suppose $\mathbb{N}$ is bounded above. Since $\mathbb{N} \neq \emptyset$ and $\mathbb{N} \subseteq \mathbb{R}$, the Completeness Axiom gives us $m = \sup \mathbb{N} \in \mathbb{R}$. Since $m$ is the least upper bound, $m - 1$ is not an upper bound, so there exists $n \in \mathbb{N}$ with $n > m - 1$. But then $n + 1 > m$ and $n + 1 \in \mathbb{N}$, contradicting $m = \sup \mathbb{N}$. $\square$

This innocent-looking result has three equivalent formulations that appear constantly in analysis:

**Theorem.** The following are equivalent:
1. **(AP)** $\mathbb{N}$ is not bounded above.
2. For every $z \in \mathbb{R}$, there exists $n \in \mathbb{N}$ such that $z < n$.
3. For every $x > 0$ and $y \in \mathbb{R}$, there exists $n \in \mathbb{N}$ such that $y < nx$.
4. For every $x > 0$, there exists $n \in \mathbb{N}$ such that $0 < 1/n < x$.

Form (4) is especially important: it says there are no "infinitely small" positive real numbers. No matter how small a positive number you name, some $1/n$ is smaller.

### The Existence of Square Roots

Now we reach the payoff. The completeness axiom doesn't just assert that $\mathbb{R}$ has no gaps — it lets us *construct* the numbers that fill those gaps.

**Theorem.** Let $p$ be a prime. Then there exists a unique $x > 0$ in $\mathbb{R}$ such that $x^2 = p$.

*Proof.* Define $S = \\{r \in \mathbb{R} : 0 < r \text{ and } r^2 < p\\}$. Since $p > 1$, we have $1 \in S$ (because $1^2 = 1 < p$), so $S \neq \emptyset$. Moreover, if $r \in S$ then $r^2 < p < p^2$, so $r < p$. Thus $S$ is bounded above by $p$.

By the Completeness Axiom, $x = \sup S$ exists in $\mathbb{R}$. We claim $x^2 = p$.

**Case 1: $x^2 < p$.** Since $\frac{p - x^2}{2x + 1} > 0$, by the Archimedean Property there exists $n \in \mathbb{N}$ with $\frac{1}{n} < \frac{p - x^2}{2x + 1}$. Then:

$$
\left(x + \frac{1}{n}\right)^2 = x^2 + \frac{2x}{n} + \frac{1}{n^2} \leq x^2 + \frac{2x + 1}{n} < x^2 + (p - x^2) = p.
$$

So $x + 1/n \in S$, contradicting $x = \sup S$.

**Case 2: $x^2 > p$.** Since $\frac{x^2 - p}{2x} > 0$, there exists $m \in \mathbb{N}$ with $\frac{1}{m} < \frac{x^2 - p}{2x}$ and $\frac{1}{m} < x$. Then:

$$
\left(x - \frac{1}{m}\right)^2 = x^2 - \frac{2x}{m} + \frac{1}{m^2} > x^2 - \frac{2x}{m} > x^2 - (x^2 - p) = p.
$$

So $x - 1/m$ is an upper bound for $S$ (since every $r \in S$ satisfies $r^2 < p < (x - 1/m)^2$, hence $r < x - 1/m$). But $x - 1/m < x = \sup S$, a contradiction.

Since both $x^2 < p$ and $x^2 > p$ lead to contradictions, we conclude $x^2 = p$ by trichotomy.

**Uniqueness:** If $y > 0$ satisfies $y^2 = p$ as well, then $x > y$ would give $x^2 > y^2$, i.e., $p > p$ — a contradiction. Similarly $x < y$ fails. So $x = y$. $\square$

> This proof is a template you will see again and again in analysis: define a set, take its supremum, then show the supremum *must* be the object you want by ruling out the alternatives. The Completeness Axiom guarantees the supremum exists; the Archimedean Property provides the "wiggle room" to derive contradictions.

---

## Density of the Rationals

Here is a remarkable fact: despite the holes in $\mathbb{Q}$, the rationals are *everywhere*. Between any two real numbers, no matter how close, there is always a rational number.

We first need a lemma.

**Theorem.** Let $x > 0$. Then there exists a unique integer $n \in \mathbb{Z}$ such that $n - 1 \leq x < n$.

**Theorem (Density of $\mathbb{Q}$).** Let $x, y \in \mathbb{R}$ with $x < y$. Then there exists $r \in \mathbb{Q}$ such that $x < r < y$.

*Proof.* First assume $0 < x < y$. Since $y - x > 0$, by the Archimedean Property there exists $n \in \mathbb{N}$ with $\frac{1}{n} < y - x$, i.e., $1 < n(y - x)$, i.e., $nx + 1 < ny$.

By the previous theorem, there exists $m \in \mathbb{N}$ with $m - 1 \leq nx < m$, so $m \leq nx + 1 < ny$. This gives $nx < m < ny$, hence $x < m/n < y$. Set $r = m/n \in \mathbb{Q}$.

For $x \leq 0$: choose $k \in \mathbb{N}$ with $k > \lvert x \rvert$. Apply the above to $x + k > 0$ and $y + k$ to find $q \in \mathbb{Q}$ with $x + k < q < y + k$. Then $r = q - k$ is rational and $x < r < y$. $\square$

<svg viewBox="0 0 500 100" style="max-width:500px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; }
    .lbl { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); }
    .lbl-s { font-family: 'Inter', sans-serif; font-size: 10px; }
    .dot-r { fill: var(--primary, #94452b); }
    .dot-i { fill: none; stroke: #2a7ae2; stroke-width: 1.5; }
    .region { fill: rgba(148, 69, 43, 0.06); stroke: var(--primary, #94452b); stroke-width: 1; stroke-dasharray: 4,3; }
  </style>
  <!-- Axis -->
  <line x1="30" y1="50" x2="470" y2="50" class="ax" />
  <!-- x and y markers -->
  <line x1="140" y1="44" x2="140" y2="56" class="ax" />
  <line x1="360" y1="44" x2="360" y2="56" class="ax" />
  <text x="140" y="72" text-anchor="middle" class="lbl" font-style="italic">x</text>
  <text x="360" y="72" text-anchor="middle" class="lbl" font-style="italic">y</text>
  <!-- Highlighted region between x and y -->
  <rect x="140" y="40" width="220" height="20" rx="2" class="region" />
  <!-- Rational dots -->
  <circle cx="175" cy="50" r="3" class="dot-r" />
  <circle cx="210" cy="50" r="3" class="dot-r" />
  <circle cx="235" cy="50" r="3" class="dot-r" />
  <circle cx="270" cy="50" r="3" class="dot-r" />
  <circle cx="305" cy="50" r="3" class="dot-r" />
  <circle cx="330" cy="50" r="3" class="dot-r" />
  <!-- Irrational marks (x shapes) -->
  <g stroke="#2a7ae2" stroke-width="1.5">
    <line x1="190" y1="47" x2="196" y2="53" /><line x1="196" y1="47" x2="190" y2="53" />
    <line x1="222" y1="47" x2="228" y2="53" /><line x1="228" y1="47" x2="222" y2="53" />
    <line x1="250" y1="47" x2="256" y2="53" /><line x1="256" y1="47" x2="250" y2="53" />
    <line x1="285" y1="47" x2="291" y2="53" /><line x1="291" y1="47" x2="285" y2="53" />
    <line x1="315" y1="47" x2="321" y2="53" /><line x1="321" y1="47" x2="315" y2="53" />
    <line x1="345" y1="47" x2="351" y2="53" /><line x1="351" y1="47" x2="345" y2="53" />
  </g>
  <!-- Legend -->
  <circle cx="160" cy="90" r="3" class="dot-r" />
  <text x="168" y="93" class="lbl-s" fill="var(--primary, #94452b)">rationals</text>
  <g stroke="#2a7ae2" stroke-width="1.5">
    <line x1="237" y1="87" x2="243" y2="93" /><line x1="243" y1="87" x2="237" y2="93" />
  </g>
  <text x="250" y="93" class="lbl-s" fill="#2a7ae2">irrationals</text>
</svg>

*Between any two real numbers $x < y$, both rationals and irrationals are packed infinitely densely.*

The irrationals are dense too:

**Corollary.** Let $x, y \in \mathbb{R}$ with $x < y$. Then there exists an irrational number $c$ with $x < c < y$.

*Proof.* By the density theorem, there exists $r \in \mathbb{Q}$ with $\frac{x}{\sqrt{2}} < r < \frac{y}{\sqrt{2}}$. Then $c = r\sqrt{2}$ satisfies $x < c < y$. If $c$ were rational, then $\sqrt{2} = c/r$ would be rational — contradiction. So $c$ is irrational. $\square$

> The rationals and irrationals are interleaved in an impossibly fine weave: between any two rationals lies an irrational, and between any two irrationals lies a rational. Yet the irrationals vastly outnumber the rationals — in a measure-theoretic sense, the rationals take up "zero space" on the number line.

---

## Looking Ahead

We have built the real numbers from their axioms and discovered the property that makes them complete. The Completeness Axiom, a single sentence asserting the existence of suprema, has already yielded the Archimedean Property, the existence of square roots, and the density of the rationals.

But analysis needs more than algebra and order — it needs a notion of *nearness*. What does it mean for a point to be "close to" a set? When is a set "open" or "closed"? In [Part 2: Topology of the Real Line](/2024/05/08/real-analysis-topology.html), we develop the topological language that makes limits and continuity precise.
