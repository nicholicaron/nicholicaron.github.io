---
layout: post
title: "Modern Algebra I: Permutations and Structure"
date: 2024-05-22
tags: [Math, Algebra, Group Theory]
---

This is Part 2 of my Modern Algebra I notes. In [Part 1](/2024/05/15/modern-algebra-groups.html) we built groups from scratch — operations, axioms, Cayley tables, dihedral symmetries, and subgroups. Now we develop the central machinery of the course: permutation groups give us a concrete way to *compute* inside any group, and isomorphisms give us a precise way to say when two groups are "the same."

## What This Post Covers

- **Functions Review** — Injectivity, surjectivity, bijectivity, and composition
- **Permutation Groups** — The symmetric group $S_n$, arrow diagrams, two-line notation
- **Cycle Notation** — Disjoint cycles, transpositions, and computing with cycles
- **Parity of Permutations** — Even and odd permutations, and why parity is well-defined
- **Isomorphisms** — When two groups are structurally identical, and how to prove it
- **Order of Elements** — How element orders reveal group structure
- **Cyclic Groups** — Classification and the direct product criterion

---

## Functions: A Quick Review

Before diving into permutations, we need precise language for functions.

**Definition.** A **function** $f: A \to B$ assigns exactly one output $f(a) \in B$ to every input $a \in A$.

**Definition.** A function $f: A \to B$ is:
- **Injective** (one-to-one) if each $b \in B$ is mapped onto by *at most* one $a \in A$. That is, $f(a_1) = f(a_2) \Rightarrow a_1 = a_2$.
- **Surjective** (onto) if every $b \in B$ is equal to $f(a)$ for some $a \in A$. The whole range is "used."
- **Bijective** if it is both injective and surjective — every element of $B$ is paired with exactly one element of $A$.

Bijective functions are exactly the functions that have **inverses**: if $f: A \to B$ is bijective, there exists $f^{-1}: B \to A$ such that $f^{-1}(f(a)) = a$ and $f(f^{-1}(b)) = b$.

Given $f: A \to B$ and $g: B \to C$, the **composite function** $g \circ f: A \to C$ is defined by $(g \circ f)(a) = g(f(a))$. Note the order: $g \circ f$ means "do $f$ first, then $g$." Composition of bijections is bijective.

---

## Permutation Groups

**Definition.** A **permutation** of a set $A$ is a bijective function $f: A \to A$.

For $A = \lbrace 1, 2, 3\rbrace$, there are $3! = 6$ permutations. We can visualize them as arrow diagrams:

<svg viewBox="0 0 680 150" xmlns="http://www.w3.org/2000/svg" style="max-width:680px; margin: 1.5rem auto; display:block;">
  <style>
    .perm-num { font: 15px 'JetBrains Mono', monospace; fill: var(--text-primary, #1a1a1a); }
    .perm-arrow { stroke: var(--primary, #94452b); stroke-width: 1.8; fill: none; marker-end: url(#parrow); }
    .perm-label { font: italic 13px 'Inter', sans-serif; fill: var(--text-secondary, #666); }
  </style>
  <defs>
    <marker id="parrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="var(--primary, #94452b)"/>
    </marker>
  </defs>

  <!-- Identity -->
  <text x="20" y="20" class="perm-num">1</text><text x="60" y="20" class="perm-num">1</text>
  <text x="20" y="50" class="perm-num">2</text><text x="60" y="50" class="perm-num">2</text>
  <text x="20" y="80" class="perm-num">3</text><text x="60" y="80" class="perm-num">3</text>
  <line x1="32" y1="16" x2="55" y2="16" class="perm-arrow"/>
  <line x1="32" y1="46" x2="55" y2="46" class="perm-arrow"/>
  <line x1="32" y1="76" x2="55" y2="76" class="perm-arrow"/>
  <text x="40" y="105" class="perm-label">e</text>

  <!-- (1 2) -->
  <text x="120" y="20" class="perm-num">1</text><text x="160" y="20" class="perm-num">1</text>
  <text x="120" y="50" class="perm-num">2</text><text x="160" y="50" class="perm-num">2</text>
  <text x="120" y="80" class="perm-num">3</text><text x="160" y="80" class="perm-num">3</text>
  <line x1="132" y1="16" x2="155" y2="46" class="perm-arrow"/>
  <line x1="132" y1="46" x2="155" y2="16" class="perm-arrow"/>
  <line x1="132" y1="76" x2="155" y2="76" class="perm-arrow"/>
  <text x="135" y="105" class="perm-label">(1,2)</text>

  <!-- (1 3) -->
  <text x="230" y="20" class="perm-num">1</text><text x="270" y="20" class="perm-num">1</text>
  <text x="230" y="50" class="perm-num">2</text><text x="270" y="50" class="perm-num">2</text>
  <text x="230" y="80" class="perm-num">3</text><text x="270" y="80" class="perm-num">3</text>
  <line x1="242" y1="16" x2="265" y2="76" class="perm-arrow"/>
  <line x1="242" y1="46" x2="265" y2="46" class="perm-arrow"/>
  <line x1="242" y1="76" x2="265" y2="16" class="perm-arrow"/>
  <text x="245" y="105" class="perm-label">(1,3)</text>

  <!-- (2 3) -->
  <text x="340" y="20" class="perm-num">1</text><text x="380" y="20" class="perm-num">1</text>
  <text x="340" y="50" class="perm-num">2</text><text x="380" y="50" class="perm-num">2</text>
  <text x="340" y="80" class="perm-num">3</text><text x="380" y="80" class="perm-num">3</text>
  <line x1="352" y1="16" x2="375" y2="16" class="perm-arrow"/>
  <line x1="352" y1="46" x2="375" y2="76" class="perm-arrow"/>
  <line x1="352" y1="76" x2="375" y2="46" class="perm-arrow"/>
  <text x="355" y="105" class="perm-label">(2,3)</text>

  <!-- (1 2 3) -->
  <text x="450" y="20" class="perm-num">1</text><text x="490" y="20" class="perm-num">1</text>
  <text x="450" y="50" class="perm-num">2</text><text x="490" y="50" class="perm-num">2</text>
  <text x="450" y="80" class="perm-num">3</text><text x="490" y="80" class="perm-num">3</text>
  <line x1="462" y1="16" x2="485" y2="46" class="perm-arrow"/>
  <line x1="462" y1="46" x2="485" y2="76" class="perm-arrow"/>
  <line x1="462" y1="76" x2="485" y2="16" class="perm-arrow"/>
  <text x="458" y="105" class="perm-label">(1,2,3)</text>

  <!-- (1 3 2) -->
  <text x="560" y="20" class="perm-num">1</text><text x="600" y="20" class="perm-num">1</text>
  <text x="560" y="50" class="perm-num">2</text><text x="600" y="50" class="perm-num">2</text>
  <text x="560" y="80" class="perm-num">3</text><text x="600" y="80" class="perm-num">3</text>
  <line x1="572" y1="16" x2="595" y2="76" class="perm-arrow"/>
  <line x1="572" y1="46" x2="595" y2="16" class="perm-arrow"/>
  <line x1="572" y1="76" x2="595" y2="46" class="perm-arrow"/>
  <text x="568" y="105" class="perm-label">(1,3,2)</text>
</svg>

**Definition.** The **symmetric group** of a set $A$, written $S_A$, is the set of all permutations of $A$ with function composition as the operation. When $A = \lbrace 1, 2, \ldots, n\rbrace$, we write $S_n$.

Let's verify $S_A$ is actually a group:
- **Associativity**: Function composition is always associative — $(f \circ g) \circ h = f \circ (g \circ h)$.
- **Identity**: The identity function $e(x) = x$.
- **Inverses**: Every bijection has an inverse bijection, computed by "reversing the arrows."

**Fact.** $\lvert S_n \rvert = n!$ — the set $\lbrace1, 2, \ldots, n\rbrace$ has exactly $n!$ permutations.

### Notation

**Two-line notation** writes the inputs on top and outputs on the bottom:

$$\sigma = \begin{pmatrix} 1 & 2 & 3 & 4 & 5 \\ 4 & 1 & 3 & 2 & 5 \end{pmatrix}$$

This means $\sigma(1) = 4$, $\sigma(2) = 1$, $\sigma(3) = 3$, $\sigma(4) = 2$, $\sigma(5) = 5$.

**One-line notation** omits the top row (since it's always $1, 2, \ldots, n$ in order): just write $41325$.

To **compose** in two-line notation, treat columns as dominoes. To find $\sigma \circ \tau$, follow the outputs of $\tau$ through $\sigma$. To **invert**, simply swap the rows and re-sort by the top.

---

## Cycle Notation

Two-line notation is explicit but bulky. **Cycle notation** is far more compact and reveals the structure of a permutation.

Consider $\alpha = 32761458 \in S_8$ (one-line notation). Track what happens when we repeatedly apply $\alpha$:

- $1 \to 3 \to 7 \to 5 \to 1$ — a cycle of length 4
- $2 \to 2$ — a fixed point
- $4 \to 6 \to 4$ — a cycle of length 2
- $8 \to 8$ — a fixed point

In cycle notation: $\alpha = (1, 3, 7, 5)(4, 6)$. Fixed points (length-1 cycles) are omitted. The cycle $(1, 3, 7, 5)$ means $1 \mapsto 3 \mapsto 7 \mapsto 5 \mapsto 1$.

**Key facts about cycle notation:**
- Every permutation decomposes uniquely (up to ordering of cycles and rotation within cycles) into **disjoint cycles**.
- Disjoint cycles **commute**: $(1, 3)(4, 6) = (4, 6)(1, 3)$.
- A cycle $(a_1, a_2, \ldots, a_k)$ has order $k$: applying it $k$ times returns to the identity.

### Computing with Cycles

To **multiply** (compose) cycles, trace elements through right to left.

**Example.** Compute $(1, 4, 2)(3, 5) \cdot (1, 3)(2, 4)(5)$ in $S_5$.

Read right to left. Where does 1 go? The right permutation sends $1 \to 3$, then the left sends $3 \to 5$. So $1 \to 5$. Continue for each element:

- $1 \to 3 \to 5$
- $5 \to 5 \to 1$ (wait — right sends $5 \to 5$, left sends $5 \to 3$... let me be more careful)

Actually, let's be precise. Right permutation: $(1,3)(2,4)$. Left permutation: $(1,4,2)(3,5)$.

- $1$: right gives $1 \to 3$, left gives $3 \to 5$. So $1 \to 5$.
- $5$: right gives $5 \to 5$, left gives $5 \to 3$. So $5 \to 3$.
- $3$: right gives $3 \to 1$, left gives $1 \to 4$. So $3 \to 4$.
- $4$: right gives $4 \to 2$, left gives $2 \to 1$. So $4 \to 1$.
- $2$: right gives $2 \to 4$, left gives $4 \to 2$. So $2 \to 2$.

Result: $(1, 5, 3, 4)$. Element 2 is fixed.

To **invert** a cycle, reverse it: $(1, 3, 7, 5)^{-1} = (5, 7, 3, 1) = (1, 5, 7, 3)$. For a product of disjoint cycles, reverse each cycle individually (order doesn't matter since they commute).

### Computing Large Powers

Cycle notation makes computing large powers trivial. Suppose $\beta = (2, 4, 5, 1, 3) \in S_8$. What is $\beta^{100}$?

In cycle notation, $\beta = (1, 2, 4)(3, 5)$. The cycle $(1, 2, 4)$ has order 3 and $(3, 5)$ has order 2. After 100 applications:
- $(1, 2, 4)^{100} = (1, 2, 4)^{33 \cdot 3 + 1} = (1, 2, 4)$ — since $100 \bmod 3 = 1$
- $(3, 5)^{100} = (3, 5)^{50 \cdot 2} = e$ — since $100 \bmod 2 = 0$

So $\beta^{100} = (1, 2, 4)$.

---

## Transpositions and Parity

**Definition.** A **transposition** is a cycle of length 2 — it swaps two elements and fixes everything else.

**Fact.** Every permutation can be written as a composition of transpositions.

**Example.** The permutation $23154 \in S_5$ can be decomposed: starting from $12345$, we need three swaps to reach $41235$... but the decomposition is **not unique**. We might need a different number of transpositions depending on how we do it. However, something remarkable is true:

**Theorem.** If a permutation $\pi = t_m \cdots t_1$ is written as a product of $m$ transpositions, and also as $\pi = u_p \cdots u_1$ with $p$ transpositions, then $m$ and $p$ are either **both even or both odd**.

The proof proceeds by examining $e = \pi\pi^{-1} = t_m \cdots t_1 \cdot u_1^{-1} \cdots u_p^{-1}$ (each transposition is its own inverse). This is a product of $m + p$ transpositions equaling the identity, and one shows (through a careful case analysis on how transpositions interact) that the identity can only be written as a product of an even number of transpositions. Thus $m + p$ is even, so $m$ and $p$ have the same parity.

**Definition.** A permutation $\pi \in S_n$ is:
- **Even** if $\pi$ is a product of an even number of transpositions
- **Odd** if $\pi$ is a product of an odd number of transpositions

> Parity under composition behaves just like parity of integers under addition: even $\circ$ even = even, odd $\circ$ odd = even, even $\circ$ odd = odd.

This means there's a "homomorphism" from $S_n$ to $\mathbb{Z}_2$ — a structure-preserving map that we'll formalize in Part 3.

**Quick test:** A $k$-cycle is even if $k$ is odd, and odd if $k$ is even (since a $k$-cycle decomposes into $k - 1$ transpositions). For a product of disjoint cycles, just count the total parity.

**Remark.** If $\pi \in S_n$, then $\pi^{2024}$ is always even (an even power of anything is even). And $\pi^{2025}$ has the same parity as $\pi$ itself.

---

## Isomorphisms

This is one of the most important ideas in all of algebra: when are two groups "really the same"?

Consider these two groups of size 2:

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px;">$G_1$</caption>
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">0</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td></tr>
</table>

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px;">$G_2$</caption>
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th><th style="border: 1px solid #999; padding: 6px 12px;">-1</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">-1</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">-1</td><td style="border: 1px solid #999; padding: 6px 12px;">-1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
</table>

These are the same group up to relabeling: $0 \leftrightarrow 1$ and $1 \leftrightarrow -1$.

**Definition.** A (group) **isomorphism** is a bijective function $f: G_1 \to G_2$ such that for all $a, b \in G_1$:

$$f(a *_1 b) = f(a) *_2 f(b)$$

If such an isomorphism exists, we say $G_1$ and $G_2$ are **isomorphic** and write $G_1 \cong G_2$.

**Example.** $(\mathbb{R}, +) \cong (\mathbb{R}^{pos}, \cdot)$ via the isomorphism $f(x) = 2^x$ (with inverse $f^{-1}(y) = \log_2(y)$). Check: $f(a + b) = 2^{a+b} = 2^a \cdot 2^b = f(a) \cdot f(b)$.

### Proving Groups Are Isomorphic

To show $G_1 \cong G_2$: **construct** an explicit bijection $f: G_1 \to G_2$ and verify $f(ab) = f(a)f(b)$.

### Proving Groups Are NOT Isomorphic

Any isomorphism must preserve every structural property. So $G_1 \cong G_2$ implies:
- $\lvert G_1 \rvert = \lvert G_2 \rvert$
- Same commutativity (both abelian or both non-abelian)
- Same cyclicity
- Same number of subgroups
- Same number of elements of each order

To show $G_1 \not\cong G_2$, find any single property they don't share.

**Example.** $\mathbb{Z}_4 \not\cong \mathbb{Z}_2 \times \mathbb{Z}_2$: the first is cyclic, the second is not.

**Example.** $\mathbb{Z}_6 \not\cong S_3$: the first is abelian, the second is not.

> Isomorphism is an equivalence relation: $G \cong G$ (reflexive), $G_1 \cong G_2 \Rightarrow G_2 \cong G_1$ (symmetric), and $G_1 \cong G_2 \cong G_3 \Rightarrow G_1 \cong G_3$ (transitive). In some sense, group theory is the study of these equivalence classes.

### Cayley's Theorem

**Theorem (Cayley).** Every group $G$ is isomorphic to a subgroup of the symmetric group $S_G$.

The isomorphism is given by $f: G \to S_G$ where $f(g) = \pi_g$, the permutation $\pi_g(x) = g * x$.

**Example.** For $G = \mathbb{Z}_3$: $0 \mapsto \begin{pmatrix}0&1&2\\0&1&2\end{pmatrix}$, $1 \mapsto \begin{pmatrix}0&1&2\\1&2&0\end{pmatrix}$, $2 \mapsto \begin{pmatrix}0&1&2\\2&0&1\end{pmatrix}$.

This is a deep result: it says that no matter how abstract a group may seem, it can always be realized as a concrete collection of symmetries (permutations). Permutation groups are "universal."

---

## Order of Group Elements

**Definition.** The **order** of an element $a \in G$, written $\text{ord}(a)$ or $\lvert a \rvert$, is the smallest positive integer $m$ such that $a^m = e$. If no such $m$ exists, $\text{ord}(a) = \infty$.

Equivalently, $\text{ord}(a) = \lvert\langle a \rangle\rvert$ — the order of the element equals the size of the cyclic subgroup it generates.

**Examples in** $(\mathbb{Z}, +)$: $\text{ord}(0) = 1$, $\text{ord}(1) = \infty$, $\text{ord}(n) = \infty$ for all $n \neq 0$.

**Orders in** $\mathbb{Z}_n$: In $\mathbb{Z}_6$, the orders are $\text{ord}(0) = 1$, $\text{ord}(1) = 6$, $\text{ord}(2) = 3$, $\text{ord}(3) = 2$, $\text{ord}(4) = 3$, $\text{ord}(5) = 6$.

**Key facts:**
- $\text{ord}(a) = 1 \iff a = e$
- If $G$ is finite, then $\text{ord}(a)$ divides $\lvert G \rvert$ for all $a \in G$ (this will follow from Lagrange's theorem in Part 3)
- If $f: G_1 \to G_2$ is an isomorphism, then $\text{ord}(f(a)) = \text{ord}(a)$ for all $a \in G_1$

That last fact gives us a powerful tool for proving groups are not isomorphic.

**Example.** $\mathbb{Z}_9 \not\cong \mathbb{Z}_3 \times \mathbb{Z}_3$. In $\mathbb{Z}_9$, element 1 has order 9. In $\mathbb{Z}_3 \times \mathbb{Z}_3$, the maximum order is 3 (every non-identity element has order 3). Since the sets of element orders differ, the groups are not isomorphic.

**Example.** $\mathbb{R}^{\ast} \not\cong \mathbb{C}^{\ast}$ (under multiplication). In $\mathbb{R}^{\ast}$, only $\pm 1$ have finite order (orders 1 and 2). But in $\mathbb{C}^{\ast}$, $i$ has order 4. Since $\mathbb{R}^{\ast}$ has no element of order 4, the groups are not isomorphic.

### The Division Theorem for Orders

**Theorem.** Suppose $\text{ord}(a) = n < \infty$. Then $a^s = a^t \iff s \equiv t \pmod{n}$.

In particular, $a^k = e \iff n \mid k$.

**Example.** If $\text{ord}(a) = 6$, is $a^{10} = a^{20}$? We check: $10 \bmod 6 = 4$ and $20 \bmod 6 = 2$. Since $4 \neq 2$, no.

---

## Cyclic Groups

Recall from Part 1 that a group is **cyclic** if it's generated by a single element: $G = \langle a \rangle$.

### Classification

**Theorem.** Every cyclic group is isomorphic to either $\mathbb{Z}$ (if infinite) or $\mathbb{Z}_n$ (if finite of order $n$).

In particular, any two cyclic groups of the same order are isomorphic. This is a clean, complete classification.

### Exponent Laws

In any group, using multiplicative notation with $a^n = a \cdot a \cdots a$ ($n$ times):

$$a^m \cdot a^n = a^{m+n}, \quad (a^m)^n = a^{mn}, \quad a^{-n} = (a^n)^{-1}$$

> **Caution.** There is no notion of $n$th roots or logarithms in a general group. The expression $a^{1/2}$ is meaningless unless the group has additional structure.

### When Is a Direct Product Cyclic?

This is an elegant result that connects number theory to group theory:

**Theorem.**

$$\mathbb{Z}_m \times \mathbb{Z}_n \cong \mathbb{Z}_{mn} \quad \text{if and only if} \quad \gcd(m, n) = 1$$

Equivalently, $\mathbb{Z}_m \times \mathbb{Z}_n$ is cyclic iff $\text{lcm}(m, n) = mn$.

**Example.** $\mathbb{Z}_2 \times \mathbb{Z}_3 \cong \mathbb{Z}_6$ since $\gcd(2, 3) = 1$. The isomorphism sends $x \mapsto (x \bmod 2, x \bmod 3)$.

**Example.** $\mathbb{Z}_2 \times \mathbb{Z}_2 \not\cong \mathbb{Z}_4$ since $\gcd(2, 2) = 2 \neq 1$.

**Proof idea.** The isomorphism is $f(x) = (x \bmod m, \; x \bmod n)$. When $\gcd(m,n) = 1$, it's injective: if $f(a) = f(b)$, then both $m$ and $n$ divide $a - b$, so $mn$ divides $a - b$ by coprimality. Since both groups have the same finite size, injectivity implies bijectivity.

This generalizes: a direct product of cyclic groups is cyclic iff their orders are pairwise coprime. Looking further ahead, every finitely-generated abelian group decomposes as a product of cyclic groups — but that's a story for a second course.

---

## Looking Ahead

We've now built a rich toolkit: permutation groups give us concrete objects to compute with, cycle notation gives us an efficient language, and isomorphisms tell us when two groups are really the same structure wearing different labels.

In [Part 3](/2024/05/29/modern-algebra-quotients.html), we'll use **cosets** to partition groups into equal-sized pieces (leading to Lagrange's theorem), develop **homomorphisms** as structure-preserving maps that don't need to be bijective, build **quotient groups** by "dividing out" a normal subgroup, and culminate with the **Fundamental Homomorphism Theorem** — the crown jewel connecting kernels, images, and quotients into a single elegant picture.
