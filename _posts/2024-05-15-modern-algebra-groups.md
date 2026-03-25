---
layout: post
title: "Modern Algebra I: Groups from the Ground Up"
date: 2024-05-15
tags: [Math, Algebra, Group Theory]
---

These are my compiled notes from Modern Algebra I, reorganized into a narrative that builds the theory of groups from first principles. Abstract algebra strips away the specifics of numbers, matrices, and functions to study the *structure* of mathematical operations themselves. The payoff is enormous: a single theorem about groups can simultaneously tell you something about clock arithmetic, symmetries of molecules, Rubik's cube solutions, and error-correcting codes.

This is Part 1 of a three-part series. Here we build the foundations — from operations on sets to the definition of a group, through the first major examples and subgroup theory.

## What This Post Covers

- **Operations on Sets** — What it means to combine elements, and how to organize combinations in a table
- **Properties of Operations** — Commutativity, identity elements, and inverses
- **Associativity and the Definition of a Group** — The three axioms that make a group
- **Elementary Group Properties** — Uniqueness theorems, cancellation, and the "Sudoku property"
- **Symmetry Groups** — The dihedral group of the triangle, where algebra meets geometry
- **Subgroups** — Direct products, the subgroup criterion, cyclic subgroups, and generators

---

## Operations on Sets

The starting point of abstract algebra is almost absurdly simple. Forget about addition, multiplication, and all the specific operations you know. Instead, ask: what does it *mean* to combine two things?

**Definition.** An **operation** on a set $A$ is a rule that assigns exactly one element of $A$ to every ordered pair of elements of $A$. Equivalently, it is a function $*: A \times A \to A$.

We write $a * b$ (or $a + b$, $a \cdot b$, etc.) instead of $\ast(a, b)$ because it's more natural. The key word is *closed* — the output must land back in $A$. Addition is an operation on $\mathbb{Z}$, but division is not an operation on $\mathbb{Z}$ (since $1 \div 2 \notin \mathbb{Z}$).

Some familiar examples:

- $+$ on $\mathbb{Z}$, $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$
- $\div$ on $\mathbb{Q} - \lbrace 0\rbrace$, $\mathbb{R} - \lbrace 0\rbrace$, $\mathbb{C} - \lbrace 0\rbrace$
- $\cdot$ on $n \times n$ matrices

### How Many Operations Exist?

Here's a fun counting question: how many operations are there on the set $A = \lbrace 0, 1\rbrace$?

An operation on $A$ must assign an output to every ordered pair. The possible inputs are $(0,0)$, $(0,1)$, $(1,0)$, and $(1,1)$ — four pairs. For each pair, we can choose either $0$ or $1$ as the output. By the product principle, there are $2^4 = 16$ possible operations.

We can organize any operation on a finite set into a **Cayley table** (also called an operation table). Here are two of the sixteen operations on $\lbrace0, 1\rbrace$:

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">0</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
</table>

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">0</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td></tr>
</table>

The left table is ordinary multiplication. The right is addition modulo 2. To read these: the entry in row $a$, column $b$ gives $a * b$.

In general, an $n$-element set has $n^{n^2}$ possible operations. This grows absurdly fast — a 3-element set has $3^9 = 19{,}683$ operations. Most of these are unstructured noise. The goal of algebra is to focus on the "nice" ones.

---

## Properties of Operations

What makes an operation "nice"? Three properties stand out.

### Commutativity

**Definition.** An operation $*$ on a set $A$ is **commutative** if $a * b = b * a$ for all $a, b \in A$.

Addition on $\mathbb{Z}$ is commutative. Matrix multiplication is not — in general, $AB \neq BA$. There's a neat visual test: an operation is commutative if and only if its Cayley table is symmetric across the main diagonal. If $G$ is the matrix representing the Cayley table, the operation is commutative iff $G^T = G$.

### Identity Elements

**Definition.** An operation $*$ on $A$ has an **identity element** if there is some $e \in A$ such that $a * e = e * a = a$ for every $a \in A$.

The identity for addition is $0$. The identity for multiplication is $1$. For matrix multiplication, it's the identity matrix $I$. A critical point: the identity must be a *fixed constant* — it cannot depend on which element $a$ you're combining it with.

> **Non-example.** Exponentiation on $\lbrace0, 1\rbrace$: we'd need $x^e = x$ for all $x$, which gives $e = 1$. But we also need $e^x = x$ for all $x$, which gives $1^0 = 0$ — false. So exponentiation has no identity.

In a Cayley table, the identity element is easy to spot: its row and column reproduce the header row and column exactly.

<table style="margin: 1rem auto; border-collapse: collapse; text-align: center;">
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">a</th><th style="border: 1px solid #999; padding: 6px 12px;">b</th><th style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">c</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">a</td><td style="border: 1px solid #999; padding: 6px 12px;">b</td><td style="border: 1px solid #999; padding: 6px 12px;">b</td><td style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">a</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">b</td><td style="border: 1px solid #999; padding: 6px 12px;">a</td><td style="border: 1px solid #999; padding: 6px 12px;">c</td><td style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">b</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold; background: rgba(100,200,100,0.15);">c</td><td style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">a</td><td style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">b</td><td style="border: 1px solid #999; padding: 6px 12px; background: rgba(100,200,100,0.15);">c</td></tr>
</table>

Here $c$ is the identity — the highlighted row and column just reproduce the labels.

### Inverses

**Definition.** If an operation $*$ on a set $A$ has an identity element $e$, then an element $a \in A$ has an **inverse** $a^{-1} \in A$ if $a * a^{-1} = e = a^{-1} * a$.

The inverse of $3$ under addition is $-3$. The inverse of $3$ under multiplication is $\frac{1}{3}$ (but only if we're in $\mathbb{Q}$ or $\mathbb{R}$, not $\mathbb{Z}$). Zero has no multiplicative inverse. You cannot have inverses without first having an identity element.

Finding inverses in a Cayley table amounts to looking for copies of $e$: to find $a^{-1}$, scan row $a$ for $e$, and the column header gives you $a^{-1}$.

An operation can be commutative and/or have an identity and/or have inverses for every element — these properties are independent.

---

## Associativity and the Birth of Groups

### Associativity

**Definition.** An operation $*$ on a set $A$ is **associative** if $(a * b) * c = a * (b * c)$ for all $a, b, c \in A$.

Addition, multiplication, and matrix multiplication are all associative. Subtraction is not: $8 - (4 - 2) = 6$, but $(8 - 4) - 2 = 2$. Unlike commutativity and identity elements, there is no simple visual test for associativity from a Cayley table — you just have to check.

**Example.** Consider the operation $a * b = -a - b + 1$ on $\mathbb{Z}$.

- *Commutative?* Yes: $-a - b + 1 = -b - a + 1$.
- *Identity?* We need $a * e = a$, i.e., $-a - e + 1 = a$, giving $e = 1 - 2a$. But $e$ depends on $a$ — so **no identity exists**.
- *Associative?* $(a * b) * c = (-a - b + 1) * c = -(-a-b+1) - c + 1 = a + b - c$. Meanwhile, $a * (b * c) = a * (-b - c + 1) = -a - (-b-c+1) + 1 = -a + b + c$. These aren't equal, so **not associative**.

### The Definition of a Group

Now we can state the central definition of the course.

**Definition.** A **group** $(G, *)$ is a set $G$ together with an operation $*$ such that:

1. $*$ is **associative**: $(a * b) * c = a * (b * c)$ for all $a, b, c \in G$
2. $*$ has an **identity element** $e \in G$
3. Every element of $G$ has an **inverse** under $*$

That's it. Three axioms. Everything in group theory flows from these.

> A group is not necessarily commutative. Commutative groups are called **abelian groups** (after Niels Henrik Abel).

### First Examples

- $(\mathbb{Z}, +)$, $(\mathbb{Q}, +)$, $(\mathbb{R}, +)$ are abelian groups.
- $(\mathbb{Q} - \lbrace 0\rbrace, \cdot)$, $(\mathbb{R} - \lbrace 0\rbrace, \cdot)$ are abelian groups.
- $(\mathbb{R}^{pos}, \cdot)$ and $([0, \infty), \cdot)$ are abelian groups... wait, is $[0, \infty)$ a group under multiplication? No — $0$ has no multiplicative inverse!
- **Modular addition** gives us finite groups: $\mathbb{Z}_n = \lbrace 0, 1, 2, \ldots, n-1\rbrace$ with addition modulo $n$.

Here are the Cayley tables for $\mathbb{Z}_2$ and $\mathbb{Z}_3$:

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px;">$\mathbb{Z}_2$</caption>
<tr><th style="border: 1px solid #999; padding: 6px 12px;">+</th><th style="border: 1px solid #999; padding: 6px 12px;">0</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td></tr>
</table>

<table style="display:inline-table; margin: 1rem 2rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px;">$\mathbb{Z}_3$</caption>
<tr><th style="border: 1px solid #999; padding: 6px 12px;">+</th><th style="border: 1px solid #999; padding: 6px 12px;">0</th><th style="border: 1px solid #999; padding: 6px 12px;">1</th><th style="border: 1px solid #999; padding: 6px 12px;">2</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">2</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td><td style="border: 1px solid #999; padding: 6px 12px;">2</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">2</td><td style="border: 1px solid #999; padding: 6px 12px;">2</td><td style="border: 1px solid #999; padding: 6px 12px;">0</td><td style="border: 1px solid #999; padding: 6px 12px;">1</td></tr>
</table>

$\mathbb{Z}_n$ is an abelian group for any $n$, and it's our first example of a **cyclic group** — every element can be obtained by repeatedly adding $1$.

The set of invertible $n \times n$ real matrices under multiplication is a **non-commutative** group. This is a big deal: it means group theory captures structures where order matters.

---

## Elementary Properties of Groups

What can we deduce from just the three axioms? Quite a lot.

### Uniqueness of the Identity

**Proposition.** Every group has exactly one identity element.

*Proof.* Suppose $e_1$ and $e_2$ are both identity elements in $(G, *)$. Then:

$$e_1 = e_1 * e_2 = e_2$$

The first equality holds because $e_2$ is an identity, and the second holds because $e_1$ is an identity. So $e_1 = e_2$. $\square$

### Uniqueness of Inverses

**Proposition.** For any element $a$ in a group $(G, *)$:

1. $a^{-1}$ is unique (no element has two different inverses)
2. $(a^{-1})^{-1} = a$
3. $(a * b)^{-1} = b^{-1} * a^{-1}$

> **Watch the order in (3)!** The inverse of a product reverses the order. This is the same reason you take off your shoes before your socks. Only when the group is abelian do we get $(a * b)^{-1} = a^{-1} * b^{-1}$.

More generally, $(a_1 * a_2 * \cdots * a_n)^{-1} = a_n^{-1} * \cdots * a_2^{-1} * a_1^{-1}$.

### Cancellation Laws

**Proposition.** In any group $(G, *)$ with elements $a, b, c$:

1. If $a * b = a * c$, then $b = c$ (left cancellation)
2. If $b * a = c * a$, then $b = c$ (right cancellation)

*Proof of (1).* Multiply both sides on the left by $a^{-1}$:

$$a^{-1} * (a * b) = a^{-1} * (a * c)$$

$$(a^{-1} * a) * b = (a^{-1} * a) * c \quad \text{(associativity)}$$

$$e * b = e * c \quad \text{(inverses)}$$

$$b = c \quad \text{(identity)} \quad \square$$

> **WARNING.** In a non-abelian group, $a * b = c * a$ does **not** imply $b = c$. Similarly, $b * a = a * c$ does **not** imply $b = c$. You can only cancel from the *same side*.

### The "Sudoku Property"

The cancellation laws have a beautiful consequence: in the Cayley table of any group, every element appears exactly once in every row and exactly once in every column. This is sometimes called the **Sudoku property**.

Why? If some element $d$ appeared twice in row $a$ — say in columns $b$ and $c$ with $b \neq c$ — then $a * b = d = a * c$, which by left cancellation gives $b = c$, a contradiction.

This property is surprisingly powerful. Suppose you know $(\lbrace e, a\rbrace, *)$ is a group. The table must start with:

<table style="margin: 1rem auto; border-collapse: collapse; text-align: center;">
<tr><th style="border: 1px solid #999; padding: 6px 12px;">*</th><th style="border: 1px solid #999; padding: 6px 12px;">e</th><th style="border: 1px solid #999; padding: 6px 12px;">a</th></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">e</td><td style="border: 1px solid #999; padding: 6px 12px;">e</td><td style="border: 1px solid #999; padding: 6px 12px;">a</td></tr>
<tr><td style="border: 1px solid #999; padding: 6px 12px; font-weight:bold;">a</td><td style="border: 1px solid #999; padding: 6px 12px;">a</td><td style="border: 1px solid #999; padding: 6px 12px;">e</td></tr>
</table>

The identity row and column are forced, and then the Sudoku property forces $a * a = e$. So up to renaming elements, there is **exactly one group of size 2**.

Similarly, there is exactly one group of size 3, but there are **two** groups of size 4 (up to isomorphism — a concept we'll develop in Part 2).

From here on, we adopt **multiplicative notation**: we write $ab$ instead of $a * b$, and $a^n$ for the product of $n$ copies of $a$.

---

## Symmetry: The Dihedral Group

Groups become vivid when we connect them to geometry. Consider an equilateral triangle with vertices labeled $A$, $B$, $C$. What are all the ways we can pick up this triangle and set it back down so that it occupies the same space?

<svg viewBox="0 0 700 320" xmlns="http://www.w3.org/2000/svg" style="max-width:700px; margin: 1.5rem auto; display:block;">
  <style>
    .tri-label { font: bold 16px 'Inter', sans-serif; fill: var(--text-primary, #1a1a1a); }
    .sym-label { font: 14px 'Inter', sans-serif; fill: var(--text-secondary, #555); }
    .sym-desc { font: italic 12px 'Inter', sans-serif; fill: var(--text-secondary, #777); }
    .tri-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 2; fill: none; }
    .tri-fill { fill: rgba(148, 69, 43, 0.08); stroke: var(--text-primary, #1a1a1a); stroke-width: 2; }
    .arrow-line { stroke: var(--primary, #94452b); stroke-width: 1.5; fill: none; marker-end: url(#arrowhead); }
    .dash-line { stroke: var(--text-secondary, #999); stroke-width: 1; stroke-dasharray: 5,4; }
  </style>
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="var(--primary, #94452b)"/>
    </marker>
  </defs>

  <!-- Center triangle (reference) -->
  <polygon points="350,30 280,150 420,150" class="tri-fill"/>
  <text x="350" y="22" text-anchor="middle" class="tri-label">A</text>
  <text x="270" y="168" text-anchor="middle" class="tri-label">B</text>
  <text x="430" y="168" text-anchor="middle" class="tri-label">C</text>

  <!-- e: identity (top-left) -->
  <polygon points="80,40 45,100 115,100" class="tri-edge"/>
  <text x="80" y="33" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="37" y="115" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="123" y="115" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="80" y="128" text-anchor="middle" class="sym-label" font-weight="bold">e</text>
  <text x="80" y="143" text-anchor="middle" class="sym-desc">identity</text>

  <!-- r: 120° CW (top-right) -->
  <polygon points="620,40 585,100 655,100" class="tri-edge"/>
  <text x="620" y="33" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="577" y="115" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="663" y="115" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="620" y="128" text-anchor="middle" class="sym-label" font-weight="bold">r</text>
  <text x="620" y="143" text-anchor="middle" class="sym-desc">120° CW</text>

  <!-- s: 240° CW (mid-right) -->
  <polygon points="620,180 585,240 655,240" class="tri-edge"/>
  <text x="620" y="173" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="577" y="255" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="663" y="255" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="620" y="268" text-anchor="middle" class="sym-label" font-weight="bold">s</text>
  <text x="620" y="283" text-anchor="middle" class="sym-desc">240° CW</text>

  <!-- a: reflect across A (bottom-left) -->
  <polygon points="80,190 45,250 115,250" class="tri-edge"/>
  <text x="80" y="183" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="37" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="123" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="80" y="278" text-anchor="middle" class="sym-label" font-weight="bold">a</text>
  <text x="80" y="293" text-anchor="middle" class="sym-desc">reflect thru A</text>

  <!-- b: reflect across B (bottom-center) -->
  <polygon points="280,190 245,250 315,250" class="tri-edge"/>
  <text x="280" y="183" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="237" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="323" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="280" y="278" text-anchor="middle" class="sym-label" font-weight="bold">b</text>
  <text x="280" y="293" text-anchor="middle" class="sym-desc">reflect thru B</text>

  <!-- c: reflect across C (bottom-right) -->
  <polygon points="420,190 385,250 455,250" class="tri-edge"/>
  <text x="420" y="183" text-anchor="middle" class="tri-label" style="font-size:13px;">B</text>
  <text x="377" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">A</text>
  <text x="463" y="265" text-anchor="middle" class="tri-label" style="font-size:13px;">C</text>
  <text x="420" y="278" text-anchor="middle" class="sym-label" font-weight="bold">c</text>
  <text x="420" y="293" text-anchor="middle" class="sym-desc">reflect thru C</text>

  <!-- Arrows from center -->
  <line x1="310" y1="60" x2="140" y2="60" class="arrow-line"/>
  <line x1="390" y1="60" x2="560" y2="60" class="arrow-line"/>
  <line x1="600" y1="160" x2="610" y2="175" class="arrow-line"/>
  <line x1="310" y1="140" x2="130" y2="200" class="arrow-line"/>
  <line x1="340" y1="155" x2="300" y2="190" class="arrow-line"/>
  <line x1="395" y1="155" x2="415" y2="185" class="arrow-line"/>
</svg>

There are exactly six symmetries:

- **$e$**: do nothing (identity)
- **$r$**: rotate 120° clockwise
- **$s$**: rotate 240° clockwise (equivalently, 120° counter-clockwise)
- **$a$**: reflect across the line through vertex $A$
- **$b$**: reflect across the line through vertex $B$
- **$c$**: reflect across the line through vertex $C$

The set is $D_3 = \lbrace e, r, s, a, b, c\rbrace$ and the operation is **composition**, applied right to left. To compute $r * a$ ("do $a$ first, then $r$"), we track where each vertex goes.

This group is called the **dihedral group** $D_3$. Its full Cayley table is:

<table style="margin: 1rem auto; border-collapse: collapse; text-align: center;">
<tr><th style="border: 1px solid #999; padding: 5px 10px;">*</th><th style="border: 1px solid #999; padding: 5px 10px;">e</th><th style="border: 1px solid #999; padding: 5px 10px;">r</th><th style="border: 1px solid #999; padding: 5px 10px;">s</th><th style="border: 1px solid #999; padding: 5px 10px;">a</th><th style="border: 1px solid #999; padding: 5px 10px;">b</th><th style="border: 1px solid #999; padding: 5px 10px;">c</th></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td></tr>
<tr><td style="border: 1px solid #999; padding: 5px 10px; font-weight:bold;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">c</td><td style="border: 1px solid #999; padding: 5px 10px;">a</td><td style="border: 1px solid #999; padding: 5px 10px;">b</td><td style="border: 1px solid #999; padding: 5px 10px;">r</td><td style="border: 1px solid #999; padding: 5px 10px;">s</td><td style="border: 1px solid #999; padding: 5px 10px;">e</td></tr>
</table>

Notice that this table is **not** symmetric across the diagonal — $D_3$ is non-abelian. For instance, $r * a = c$ but $a * r = b$. The rotations commute with each other, and each reflection is its own inverse, but mixing rotations and reflections does not commute.

The dihedral group $D_n$ generalizes to the symmetries of a regular $n$-gon. $D_4$ (the square) has 8 elements, $D_5$ (the pentagon) has 10, and in general $\lvert D_n \rvert = 2n$.

---

## Subgroups

### Direct Products

**Definition.** Given groups $(G_1, *_1)$ and $(G_2, *_2)$, the **direct product** $G_1 \times G_2$ consists of all ordered pairs $\lbrace(a_1, a_2) : a_1 \in G_1, a_2 \in G_2\rbrace$ with componentwise operation:

$$(a_1, a_2) * (b_1, b_2) = (a_1 *_1 b_1, \; a_2 *_2 b_2)$$

For example, $\mathbb{Z}_2 \times \mathbb{Z}_2$ has elements $\lbrace(0,0), (0,1), (1,0), (1,1)\rbrace$ with modular addition in each coordinate. Its Cayley table:

<table style="margin: 1rem auto; border-collapse: collapse; text-align: center; font-size: 0.9rem;">
<tr><th style="border: 1px solid #999; padding: 4px 8px;">+</th><th style="border: 1px solid #999; padding: 4px 8px;">(0,0)</th><th style="border: 1px solid #999; padding: 4px 8px;">(0,1)</th><th style="border: 1px solid #999; padding: 4px 8px;">(1,0)</th><th style="border: 1px solid #999; padding: 4px 8px;">(1,1)</th></tr>
<tr><td style="border: 1px solid #999; padding: 4px 8px; font-weight:bold;">(0,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,1)</td></tr>
<tr><td style="border: 1px solid #999; padding: 4px 8px; font-weight:bold;">(0,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,0)</td></tr>
<tr><td style="border: 1px solid #999; padding: 4px 8px; font-weight:bold;">(1,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,1)</td></tr>
<tr><td style="border: 1px solid #999; padding: 4px 8px; font-weight:bold;">(1,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(1,0)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,1)</td><td style="border: 1px solid #999; padding: 4px 8px;">(0,0)</td></tr>
</table>

### The Subgroup Criterion

**Definition.** If $(G, *)$ is a group and $H \subseteq G$, then $H$ is a **subgroup** of $G$ if $(H, *)$ is itself a group.

For example, $(\mathbb{Z}, +)$ is a subgroup of $(\mathbb{Q}, +)$, which is a subgroup of $(\mathbb{R}, +)$.

You don't need to re-verify all three axioms from scratch. Since the operation is inherited from $G$, associativity is automatic. You only need to check:

1. **Closed under $*$**: for all $a, b \in H$, $a * b \in H$
2. **Closed under inverses**: for all $a \in H$, $a^{-1} \in H$

(These two conditions together guarantee the identity is in $H$ — can you see why?)

**Example.** What are the subgroups of $\mathbb{Z}_4 = \lbrace 0, 1, 2, 3\rbrace$?

- $\lbrace0\rbrace$ — the trivial subgroup (always exists)
- $\lbrace0, 2\rbrace$ — closed under addition mod 4
- $\lbrace0, 1, 2, 3\rbrace$ — the whole group (always a subgroup of itself)

Note that any divisor $d$ of $n$ yields a subgroup $\lbrace0, d, 2d, \ldots\rbrace$ of $\mathbb{Z}_n$ of size $\frac{n}{d}$. So the number of subgroups of $\mathbb{Z}_n$ equals the number of divisors of $n$.

### Cyclic Subgroups and Generators

**Definition.** Given an element $a$ in a group $(G, *)$, the **subgroup generated by $a$** is:

$$\langle a \rangle = \{e, a, a^2, a^3, \ldots, a^{-1}, a^{-2}, \ldots\}$$

This is always a subgroup (the smallest one containing $a$).

**Definition.** A group $G$ is **cyclic** if there exists some element $a$ such that $\langle a \rangle = G$ — that is, every element of $G$ can be obtained by repeatedly applying $a$ or $a^{-1}$.

**Examples:**
- $(\mathbb{Z}, +) = \langle 1 \rangle = \langle -1 \rangle$, so $\mathbb{Z}$ is cyclic.
- $\mathbb{Z}_n = \langle 1 \rangle$, so every $\mathbb{Z}_n$ is cyclic.
- $\mathbb{Z}_2 \times \mathbb{Z}_2$ is **not** cyclic — no single element generates the whole group. Each non-identity element generates a subgroup of size 2: $\langle(0,1)\rangle = \lbrace(0,0), (0,1)\rbrace$, $\langle(1,0)\rangle = \lbrace(0,0), (1,0)\rbrace$, and $\langle(1,1)\rangle = \lbrace(0,0), (1,1)\rbrace$.

> Groups like $\mathbb{Z}_m \times \mathbb{Z}_n$ may or may not be cyclic, depending on $m$ and $n$. We'll see the precise criterion in Part 2.

### Generators and Relations

We can describe a group by specifying a set of **generators** and **relations** between them. For instance:

- $(\mathbb{Z}, +)$ can be presented as $\langle 1 \rangle$ — one generator, no relations (other than the group axioms).
- $\mathbb{Z}_n$ can be presented as $\langle 1 \mid n \cdot 1 = 0 \rangle$ (meaning $n$ copies of the generator sum to the identity).
- The dihedral group $D_3$ can be presented as $\langle r, s \mid rrr, ss, rsrs \rangle$ — two generators (a rotation and a reflection) with relations encoding that $r^3 = e$, $s^2 = e$, and $rsrs = e$.

The **free group** $\langle A \rangle$ on an alphabet $A$ consists of all "words" in the letters of $A$ (and their inverses), with concatenation as the operation and cancellation ($aa^{-1} = e$) as the only relation. Adding more relations restricts which words are considered equal, carving out a quotient of the free group — a concept we'll formalize in Part 3.

### Intersections of Subgroups

**Proposition.** If $H$ and $K$ are both subgroups of $(G, *)$, then $H \cap K$ is also a subgroup of $(G, *)$.

*Proof.* We need $H \cap K$ to be closed under $*$ and inverses. If $a, b \in H \cap K$, then $a * b \in H$ (because $H$ is closed) and $a * b \in K$ (because $K$ is closed), so $a * b \in H \cap K$. Similarly for inverses. $\square$

> **Caution.** Unions of subgroups are generally *not* subgroups. For instance, in $(\mathbb{Z}, +)$, the subgroups $2\mathbb{Z}$ and $3\mathbb{Z}$ have union containing both 2 and 3, but $2 + 3 = 5 \notin 2\mathbb{Z} \cup 3\mathbb{Z}$.

---

## Looking Ahead

We've built the foundations: operations, the group axioms, basic properties, symmetry groups, and subgroups. In [Part 2](/2024/05/22/modern-algebra-permutations.html), we'll develop the machinery of permutation groups and cycle notation, then use isomorphisms to understand when two groups are "really the same" despite looking different. The punchline — Cayley's theorem — will show that *every* group is secretly a permutation group.
