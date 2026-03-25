---
layout: post
title: "Modern Algebra II: Rings, Ideals, and Quotient Rings"
date: 2024-06-05
tags: [Math, Algebra, Ring Theory]
---

In my Modern Algebra I notes ([Part 1](/2024/05/15/modern-algebra-groups.html), [Part 2](/2024/05/22/modern-algebra-permutations.html), [Part 3](/2024/05/29/modern-algebra-quotients.html)), we built the theory of groups from first principles — a single operation on a set, governed by three axioms. We found subgroups, defined normal subgroups, constructed quotient groups, and tied everything together with the Fundamental Homomorphism Theorem.

Now we add a second operation.

A **ring** is a set equipped with *two* operations — addition and multiplication — that interact through distributivity. This seemingly modest upgrade unlocks an enormous amount of new structure: zero divisors, units, ideals, quotient rings, and the distinction between domains and fields. The beautiful thing is that the entire journey mirrors the one we already took through group theory. The parallels are not accidental — they reflect deep structural patterns that persist across all of algebra.

Historically, algebra has been about solving equations. Abel and Galois proved that there is no consistent formula for the roots of a general quintic polynomial — a result that ultimately depends on the structure of rings and fields. Ring theory gives us the language to make this precise.

## What This Post Covers

- **The Definition of a Ring** — Two operations, one set, and the axioms that bind them
- **Properties of Rings** — Why $a \cdot 0 = 0$ is a theorem, not an axiom
- **Ring Vocabulary** — Unital, commutative, units, zero divisors, and cancellation
- **Integral Domains and Fields** — The hierarchy of "well-behaved" rings
- **Subrings** — Finding rings inside rings
- **Ideals** — The ring-theoretic analog of normal subgroups
- **An Exotic Ring** — A worked example proving a strange pair of operations forms a ring
- **Ring Homomorphisms** — Structure-preserving maps between rings
- **Quotient Rings** — Building new rings by collapsing an ideal
- **The Fundamental Homomorphism Theorem** — Connecting kernels and quotients, again
- **Maximal and Prime Ideals** — When quotients are fields or integral domains

---

## The Definition of a Ring

Recall that a **group** is a set $G$ with a single operation $*$ satisfying associativity, identity, and inverses. A ring keeps all of that structure for addition, then layers on a second operation — multiplication — with its own rules.

**Definition.** A **ring** $(R, +, \cdot)$ is a set $R$ with two operations $+$ and $\cdot$ such that:

1. $(R, +)$ is an **abelian group** (associative, commutative, has identity $0$, has inverses $-a$)
2. $\cdot$ is **associative**: for all $x, y, z \in R$, $(x \cdot y) \cdot z = x \cdot (y \cdot z)$
3. $\cdot$ **distributes** over $+$ from both sides: for all $x, y, z \in R$,

$$x \cdot (y + z) = x \cdot y + x \cdot z \quad \text{and} \quad (x + y) \cdot z = x \cdot z + y \cdot z$$

Condition (1) gives us a lot for free: an additive identity $0$, additive inverses $-a$ for every element, and commutativity of addition. But notice what is **not** required of multiplication: it need not be commutative, it need not have an identity element, and elements need not have multiplicative inverses.

> A ring is an abelian group (under $+$) with a compatible multiplication bolted on. The additive structure is always well-behaved; all the interesting asymmetry lives in the multiplication.

The structural journey we're about to take parallels our group theory journey almost step-for-step:

<svg viewBox="0 0 640 230" xmlns="http://www.w3.org/2000/svg" style="max-width:640px; margin: 1.5rem auto; display:block;">
  <defs>
    <marker id="ring-arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-auto">
      <path d="M 0 0 L 10 3.5 L 0 7 z" fill="var(--primary, #94452b)"/>
    </marker>
  </defs>
  <!-- Group Theory Column -->
  <text x="150" y="28" text-anchor="middle" font-family="'Newsreader', serif" font-size="18" font-weight="bold" fill="var(--text-primary)">Group Theory</text>
  <rect x="50" y="42" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="150" y="63" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Group $(G, *)$</text>
  <rect x="50" y="84" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="150" y="105" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Subgroup</text>
  <rect x="50" y="126" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="150" y="147" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Normal Subgroup</text>
  <rect x="50" y="168" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="150" y="189" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Quotient Group / FHT</text>
  <!-- Ring Theory Column -->
  <text x="490" y="28" text-anchor="middle" font-family="'Newsreader', serif" font-size="18" font-weight="bold" fill="var(--text-primary)">Ring Theory</text>
  <rect x="390" y="42" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="490" y="63" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Ring $(R, +, \cdot)$</text>
  <rect x="390" y="84" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="490" y="105" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Subring</text>
  <rect x="390" y="126" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="490" y="147" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Ideal</text>
  <rect x="390" y="168" width="200" height="32" rx="6" fill="none" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="490" y="189" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-primary)">Quotient Ring / FHT</text>
  <!-- Arrows -->
  <line x1="255" y1="58" x2="385" y2="58" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#ring-arrow)"/>
  <line x1="255" y1="100" x2="385" y2="100" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#ring-arrow)"/>
  <line x1="255" y1="142" x2="385" y2="142" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#ring-arrow)"/>
  <line x1="255" y1="184" x2="385" y2="184" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#ring-arrow)"/>
  <!-- Vertical label -->
  <text x="320" y="222" text-anchor="middle" font-family="'Inter', sans-serif" font-size="12" fill="var(--text-secondary)" font-style="italic">same structural template, richer objects</text>
</svg>

### First Examples

The most familiar rings come from number systems:

- $(\mathbb{Z}, +, \cdot)$, $(\mathbb{Q}, +, \cdot)$, $(\mathbb{R}, +, \cdot)$ — the integers, rationals, and reals under ordinary addition and multiplication.
- $(\mathbb{Z}_n, + \bmod n, \cdot \bmod n)$ — modular arithmetic is a ring.
- **Polynomials** with coefficients in any ring form a ring under polynomial addition and multiplication.
- **Square matrices** $M_n(\mathbb{R})$ — the $n \times n$ matrices over $\mathbb{R}$ form a ring. This is our first example where multiplication is **not commutative**.
- $(2\mathbb{Z}, +, \cdot)$ — the even integers form a ring with **no multiplicative identity**. There is no even integer $e$ such that $e \cdot a = a$ for all even $a$.

### What Is Not a Ring

The natural numbers $(\lbrace 0, 1, 2, \ldots \rbrace, +, \cdot)$ do **not** form a ring because most elements lack additive inverses. (This structure is called a **semi-ring**.) More generally, the most common way for a structure to fail to be a ring is by not being closed under addition or multiplication, or by lacking additive inverses.

> **Warning.** In a ring, $a \cdot b = 0$ does **not** imply $a = 0$ or $b = 0$. For example, in $\mathbb{Z}_6$, we have $2 \cdot 3 = 0$ even though neither $2$ nor $3$ is zero. This is a fundamental departure from the arithmetic of $\mathbb{Z}$, $\mathbb{Q}$, and $\mathbb{R}$.

---

## Properties of Rings

Despite the minimal axioms, some surprisingly useful properties hold in **every** ring.

**Theorem.** Let $a$ and $b$ be elements of a ring $(R, +, \cdot)$. Then:

1. $a \cdot 0 = 0 \cdot a = 0$
2. $a \cdot (-b) = -(a \cdot b) = (-a) \cdot b$
3. $(-a) \cdot (-b) = a \cdot b$

*Proof of (1).* We have $a \cdot 0 = a \cdot (0 + 0) = a \cdot 0 + a \cdot 0$ by distributivity. Adding $-(a \cdot 0)$ to both sides gives $0 = a \cdot 0$. A similar argument shows $0 \cdot a = 0$. $\square$

*Proof of (2).* By distributivity, $a \cdot (-b) + a \cdot b = a \cdot (-b + b) = a \cdot 0 = 0$ by part (1). So $a \cdot (-b) = -(a \cdot b)$. A similar argument shows $(-a) \cdot b = -(a \cdot b)$. $\square$

*Proof of (3).* Applying part (2) twice: $(-a) \cdot (-b) = -(a \cdot (-b)) = -(-(a \cdot b)) = a \cdot b$. $\square$

> Property (1) is subtle: $0$ here is the *additive* identity, and the proof uses distributivity to bootstrap from additive structure. The fact that $a \cdot 0 = 0$ is a theorem, not an axiom — it follows from the interplay between the two operations.

---

## Ring Vocabulary

Be careful with assuming familiar algebraic properties in an arbitrary ring. Several things you might take for granted can fail.

### Unital and Commutative Rings

**Definition.** A ring is **unital** (or a "ring with unity") if it has a multiplicative identity, usually written $1$. A ring is **non-unital** if no such element exists.

**Definition.** A ring is **commutative** if $a \cdot b = b \cdot a$ for all $a, b \in R$. Otherwise it is **noncommutative**.

Most rings we encounter are unital and commutative, but important exceptions exist. The ring $(2\mathbb{Z}, +, \cdot)$ is commutative but non-unital. The ring of $n \times n$ matrices is unital (with identity matrix $I$) but noncommutative.

### Units

Not every element of a ring has a multiplicative inverse — in fact, $0$ never does.

**Definition.** An element $a$ in a unital ring is a **unit** (or is **invertible**) if there exists $b \in R$ such that $a \cdot b = b \cdot a = 1$.

The set of all units of $R$ forms a group under multiplication, often denoted $R^\times$ or $U(R)$.

### Zero Divisors

**Definition.** Nonzero elements $a, b$ in a ring are **zero divisors** if $a \cdot b = 0$.

Zero divisors break cancellation. In $\mathbb{Z}$, if $a \cdot b = a \cdot c$ and $a \neq 0$, you can cancel $a$ to conclude $b = c$. In a ring with zero divisors, you cannot.

**Theorem (Cancellation).** In a ring $R$, the following are equivalent:

$$\big(a \cdot b = a \cdot c \Rightarrow b = c\big) \;\text{and}\; \big(b \cdot a = c \cdot a \Rightarrow b = c\big) \quad \Longleftrightarrow \quad R \text{ has no zero divisors}$$

### Examples: $\mathbb{Z}_6$, $\mathbb{Z}_7$, $\mathbb{Z}_8$

<table style="margin: 1.5rem auto; border-collapse: collapse; text-align: center;">
<tr>
<th style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Ring</th>
<th style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Unital?</th>
<th style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Commutative?</th>
<th style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Units</th>
<th style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Zero Divisors</th>
</tr>
<tr>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px; font-weight:bold;">$\mathbb{Z}_6$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">$\lbrace 1, 5 \rbrace$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">$\lbrace 2, 3, 4 \rbrace$</td>
</tr>
<tr>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px; font-weight:bold;">$\mathbb{Z}_7$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">$\lbrace 1,2,3,4,5,6 \rbrace$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">None</td>
</tr>
<tr>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px; font-weight:bold;">$\mathbb{Z}_8$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">Yes</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">$\lbrace 1,3,5,7 \rbrace$</td>
<td style="border: 1px solid var(--text-secondary); padding: 6px 16px;">$\lbrace 2, 4, 6 \rbrace$</td>
</tr>
</table>

> The pattern: in $\mathbb{Z}_n$, the units are exactly the elements coprime to $n$, and the zero divisors are the nonzero elements sharing a common factor with $n$. When $n$ is prime, every nonzero element is a unit and there are no zero divisors at all.

---

## Integral Domains and Fields

At the top of the ring hierarchy sit the structures where multiplication is best behaved.

**Definition.** An **integral domain** is a unital commutative ring with no zero divisors.

The integers $\mathbb{Z}$ are the prototypical integral domain — hence the name. In an integral domain, cancellation works exactly as you'd expect: $a \cdot b = a \cdot c$ with $a \neq 0$ implies $b = c$.

**Definition.** A **field** is an integral domain in which every nonzero element is a unit (has a multiplicative inverse).

Fields are where algebra is most comfortable — you can add, subtract, multiply, and divide (by nonzero elements) freely.

**Examples of fields:** $\mathbb{Q}$, $\mathbb{R}$, $\mathbb{C}$, and $\mathbb{Z}_p$ for any prime $p$.

**Integral domains that are not fields:** $\mathbb{Z}$ is the classic example — it has no zero divisors, but $2$ has no multiplicative inverse in $\mathbb{Z}$.

The hierarchy looks like this:

<svg viewBox="0 0 560 170" xmlns="http://www.w3.org/2000/svg" style="max-width:560px; margin: 1.5rem auto; display:block;">
  <defs>
    <marker id="hier-arrow" viewBox="0 0 10 7" refX="10" refY="3.5" markerWidth="8" markerHeight="6" orient="auto-start-auto">
      <path d="M 0 0 L 10 3.5 L 0 7 z" fill="var(--primary, #94452b)"/>
    </marker>
  </defs>
  <!-- Boxes -->
  <rect x="10" y="60" width="100" height="50" rx="8" fill="rgba(148,69,43,0.06)" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="60" y="82" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Fields</text>
  <text x="60" y="100" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" fill="var(--text-secondary)">$\mathbb{Q}, \mathbb{R}, \mathbb{Z}_p$</text>
  <rect x="145" y="60" width="120" height="50" rx="8" fill="rgba(148,69,43,0.06)" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="205" y="78" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Integral</text>
  <text x="205" y="93" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Domains</text>
  <text x="205" y="106" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" fill="var(--text-secondary)">$\mathbb{Z}$</text>
  <rect x="300" y="60" width="120" height="50" rx="8" fill="rgba(148,69,43,0.06)" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="360" y="78" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Commutative</text>
  <text x="360" y="93" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Rings</text>
  <text x="360" y="106" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" fill="var(--text-secondary)">$\mathbb{Z}_6$</text>
  <rect x="455" y="60" width="95" height="50" rx="8" fill="rgba(148,69,43,0.06)" stroke="var(--text-secondary)" stroke-width="1.5"/>
  <text x="502" y="82" text-anchor="middle" font-family="'Inter', sans-serif" font-size="13" font-weight="bold" fill="var(--text-primary)">Rings</text>
  <text x="502" y="100" text-anchor="middle" font-family="'Inter', sans-serif" font-size="11" fill="var(--text-secondary)">$M_n(\mathbb{R})$</text>
  <!-- Subset arrows -->
  <line x1="113" y1="85" x2="141" y2="85" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#hier-arrow)"/>
  <line x1="268" y1="85" x2="296" y2="85" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#hier-arrow)"/>
  <line x1="423" y1="85" x2="451" y2="85" stroke="var(--primary, #94452b)" stroke-width="2" marker-end="url(#hier-arrow)"/>
  <!-- Superset labels -->
  <text x="127" y="50" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-secondary)">$\subset$</text>
  <text x="282" y="50" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-secondary)">$\subset$</text>
  <text x="437" y="50" text-anchor="middle" font-family="'Inter', sans-serif" font-size="14" fill="var(--text-secondary)">$\subset$</text>
  <!-- Bottom label -->
  <text x="280" y="145" text-anchor="middle" font-family="'Inter', sans-serif" font-size="12" fill="var(--text-secondary)" font-style="italic">each category adds more structure to multiplication</text>
</svg>

> Every field is an integral domain, and every finite integral domain is a field. The proof of the latter uses the pigeonhole principle: in a finite ring with no zero divisors, the map $x \mapsto a \cdot x$ is injective for $a \neq 0$, hence surjective, so $1$ is in the image.

---

## Subrings

Just as we found groups inside groups, we can find rings inside rings.

**Definition.** A subset $S \subseteq R$ is a **subring** of $(R, +, \cdot)$ if $(S, +, \cdot)$ is itself a ring.

**Subring Test.** A nonempty subset $S \subseteq R$ is a subring if and only if $S$ is closed under subtraction and multiplication:

- $a, b \in S \Rightarrow a - b \in S$
- $a, b \in S \Rightarrow a \cdot b \in S$

Closure under subtraction guarantees $(S, +)$ is a subgroup (we proved this in group theory), and closure under multiplication means $\cdot$ is an operation on $S$. Associativity and distributivity are inherited from $R$.

**Examples.**

- $\mathbb{Z} \subseteq \mathbb{Q} \subseteq \mathbb{R}$ — each is a subring of the next.
- $\lbrace 0, 2, 4 \rbrace$ is a subring of $\mathbb{Z}_6$ (using $+$ and $\cdot$ mod 6).
- $\mathbb{Z}_6$ is **not** a subring of $\mathbb{Z}$ — different operations.

**Example (Dyadic rationals).** The set $S = \lbrace \frac{a}{2^n} : a \in \mathbb{Z}, \; n \geq 0 \rbrace$ is a subring of $\mathbb{Q}$.

*Proof.* Closed under subtraction: $\frac{a}{2^m} - \frac{b}{2^n} = \frac{a \cdot 2^n - b \cdot 2^m}{2^{m+n}} \in S$. Closed under multiplication: $\frac{a}{2^m} \cdot \frac{b}{2^n} = \frac{ab}{2^{m+n}} \in S$. $\square$

Note that $\frac{10}{3} \notin S$ — not every rational is dyadic.

---

## Ideals

This is the central new concept of ring theory — and the one that makes quotient rings possible.

### Definition and the Ideal Test

In group theory, a **normal subgroup** is a subgroup that is "closed under conjugation." The ring-theoretic analog is an **ideal**: a subring that is "closed under multiplication by arbitrary ring elements."

**Definition.** A subring $I$ of a ring $R$ is a **(two-sided) ideal** if for every $r \in R$ and every $s \in I$:

$$r \cdot s \in I \quad \text{and} \quad s \cdot r \in I$$

This is the **absorption property** — the ideal "absorbs" multiplication by any element of the ring, not just elements of the ideal itself.

**Ideal Test.** A subset $I \subseteq R$ is an ideal if and only if:

1. $a, b \in I \Rightarrow a - b \in I$ (closed under subtraction)
2. $r \in R, \; s \in I \Rightarrow r \cdot s \in I$ and $s \cdot r \in I$ (absorption)

> An ideal is to a ring what a normal subgroup is to a group. Just as normal subgroups are exactly the kernels of group homomorphisms, ideals are exactly the kernels of ring homomorphisms. This parallel is not a coincidence — it's the same structural phenomenon appearing in a richer setting.

**Fact.** Every ideal is a subring, but not every subring is an ideal.

### Examples and Non-examples

**Example.** $\mathbb{Z}$ is a subring of $\mathbb{Q}$. Is it an ideal? **No.** Take $r = \frac{1}{2} \in \mathbb{Q}$ and $s = 3 \in \mathbb{Z}$. Then $r \cdot s = \frac{3}{2} \notin \mathbb{Z}$. The absorption property fails.

**Example.** $\mathbb{Q}$ is a subring of $\mathbb{Q}$. Is it an ideal? **No** — wait, $\mathbb{Q}$ is the whole ring, so it's trivially an ideal. But is it a *proper* ideal? No, because $1 \in \mathbb{Q}$, so for any $r \in \mathbb{Q}$, $r \cdot 1 = r \in \mathbb{Q}$ — and this means $I = \mathbb{Q}$, so the only ideals of $\mathbb{Q}$ are $\lbrace 0 \rbrace$ and $\mathbb{Q}$ itself.

**Example.** $2\mathbb{Z}$ is an ideal of $\mathbb{Z}$. Check: the even integers are closed under subtraction (even $-$ even $=$ even), and an integer times an even number is even. More generally:

**Fact.** For any integer $n$, the set $n\mathbb{Z} = \lbrace \ldots, -2n, -n, 0, n, 2n, \ldots \rbrace$ is an ideal of $\mathbb{Z}$.

**Example.** $\lbrace 0, 2, 4 \rbrace$ is a subring of $\mathbb{Z}_6$. Is it an ideal? **Yes** — you can verify that multiplying any element of $\mathbb{Z}_6$ by $0$, $2$, or $4$ stays in $\lbrace 0, 2, 4 \rbrace$.

### Properties of Ideals

**Proposition.** Suppose $I$ and $J$ are ideals of $R$. Then:

1. $I \cap J$ is an ideal of $R$.
2. $I + J = \lbrace i + j : i \in I, \; j \in J \rbrace$ is an ideal of $R$.

**Warning.** The union $I \cup J$ is **not** necessarily an ideal. For example, $4\mathbb{Z} \cup 6\mathbb{Z}$ contains $4$ and $6$, but $4 + 6 = 10 \notin 4\mathbb{Z} \cup 6\mathbb{Z}$ (since $10$ is neither a multiple of $4$ nor $6$), so it's not even a subring.

**Example.** $4\mathbb{Z} \cap 6\mathbb{Z} = 12\mathbb{Z}$ (multiples of both $4$ and $6$ are multiples of $12$). And $4\mathbb{Z} + 6\mathbb{Z} = \lbrace 4a + 6b : a, b \in \mathbb{Z} \rbrace = 2\mathbb{Z}$ (since $\gcd(4, 6) = 2$).

### Principal Ideals and PIDs

**Definition.** If an ideal $I$ of a commutative ring $R$ can be written as $I = \lbrace r \cdot a : r \in R \rbrace$ for some fixed $a \in R$, we say $I$ is a **principal ideal** and write $I = \langle a \rangle$.

**Fact.** In $\mathbb{Z}$, every ideal is principal: every ideal looks like $n\mathbb{Z} = \langle n \rangle$ for some $n$.

**Definition.** A **principal ideal domain (PID)** is an integral domain in which every ideal is principal.

$\mathbb{Z}$ is a PID. $\mathbb{Z}[x]$ (polynomials in $x$ with integer coefficients) is **not** a PID — the ideal $\langle 2, x \rangle = \lbrace 2f(x) + x \cdot g(x) : f, g \in \mathbb{Z}[x] \rbrace$ is not principal.

---

## An Exotic Ring

To see that ring structure depends entirely on the operations and not on the underlying set, consider this unusual construction.

**Claim.** $(\mathbb{Z}, \oplus, \odot)$ is a ring, where:

$$a \oplus b = a + b - 1 \qquad \text{and} \qquad a \odot b = a \cdot b - a - b + 2$$

*Proof.* We verify the ring axioms one by one.

**$\oplus$ is associative:**

$$(a \oplus b) \oplus c = (a + b - 1) + c - 1 = a + b + c - 2$$

$$a \oplus (b \oplus c) = a + (b + c - 1) - 1 = a + b + c - 2 \quad \checkmark$$

**$\oplus$ is commutative:**

$$a \oplus b = a + b - 1 = b + a - 1 = b \oplus a \quad \checkmark$$

**$\oplus$ has an identity:** Solve $a \oplus e = a$, i.e., $a + e - 1 = a$, giving $e = 1$.

**$\oplus$ has inverses:** For each $a$, solve $a \oplus x = 1$, i.e., $a + x - 1 = 1$, giving $x = 2 - a$. Check: $a \oplus (2 - a) = a + (2 - a) - 1 = 1 = e$. $\checkmark$

So $(\mathbb{Z}, \oplus)$ is an abelian group with identity $1$ and inverses $a^{-1} = 2 - a$.

**$\odot$ distributes over $\oplus$:** We need $a \odot (b \oplus c) = (a \odot b) \oplus (a \odot c)$.

Left side:

$$a \odot (b \oplus c) = a \odot (b + c - 1) = a(b + c - 1) - a - (b + c - 1) + 2 = ab + ac - 2a - b - c + 3$$

Right side:

$$(a \odot b) \oplus (a \odot c) = (ab - a - b + 2) + (ac - a - c + 2) - 1 = ab + ac - 2a - b - c + 3 \quad \checkmark$$

Since $\odot$ is commutative (which can be verified directly), left distributivity follows from right distributivity.

**$\odot$ is associative:** We compute $(a \odot b) \odot c$ and $a \odot (b \odot c)$ and verify both equal $abc - ab - ac - bc + a + b + c$ (straightforward but tedious calculation).

Therefore $(\mathbb{Z}, \oplus, \odot)$ is a ring. $\square$

> This example illustrates that ring structure depends on the operations, not on the underlying set. The "same" set $\mathbb{Z}$ can carry completely different ring structures — with different identities, different inverses, and different multiplication tables.

---

## Ring Homomorphisms

**Definition.** A **ring homomorphism** is a function $f: R_1 \to R_2$ between rings $(R_1, +_1, \cdot_1)$ and $(R_2, +_2, \cdot_2)$ such that for all $a, b \in R_1$:

$$f(a +_1 b) = f(a) +_2 f(b) \qquad \text{and} \qquad f(a \cdot_1 b) = f(a) \cdot_2 f(b)$$

A **ring isomorphism** is a bijective ring homomorphism. If $f$ is also bijective, we say $R_1 \cong R_2$.

**Definition.** The **kernel** of a ring homomorphism $f: R_1 \to R_2$ is:

$$\ker(f) = \lbrace a \in R_1 : f(a) = 0_{R_2} \rbrace$$

**Fact.** The kernel of any ring homomorphism is an ideal of the domain. (This is the ring-theoretic version of "the kernel of a group homomorphism is a normal subgroup.")

**Example.** Define $f: \mathbb{Z} \to \mathbb{Z}_2$ by $f(\text{even}) = 0$ and $f(\text{odd}) = 1$. This preserves addition (even $+$ even $=$ even maps to $0 + 0 = 0$, etc.) and multiplication (odd $\times$ odd $=$ odd maps to $1 \cdot 1 = 1$, etc.). The kernel is $\ker(f) = 2\mathbb{Z}$, which is indeed an ideal of $\mathbb{Z}$.

---

## Quotient Rings

Just as we built quotient groups by "collapsing" a normal subgroup, we build quotient rings by collapsing an ideal.

### Cosets of an Ideal

Given an ideal $I$ of a ring $R$ and an element $a \in R$, the **coset** of $a$ with respect to $I$ is:

$$I + a = \lbrace i + a : i \in I \rbrace$$

Since $(R, +)$ is an abelian group and $I$ is a subgroup (in fact, a normal subgroup — every subgroup of an abelian group is normal), the cosets of $I$ partition $R$ into disjoint pieces.

**Definition.** The **quotient ring** $R / I$ is the set of all cosets $\lbrace I + a : a \in R \rbrace$ with operations:

$$(I + a) + (I + b) = I + (a + b) \qquad \text{and} \qquad (I + a) \cdot (I + b) = I + (a \cdot b)$$

Addition is well-defined by the group theory we already proved. The key question is: **is multiplication well-defined?** We need to show that the result doesn't depend on which coset representatives we choose.

*Proof that multiplication is well-defined.* Suppose $I + a = I + a'$ and $I + b = I + b'$. Then $a' = a + i_1$ and $b' = b + i_2$ for some $i_1, i_2 \in I$. We compute:

$$a' \cdot b' = (a + i_1)(b + i_2) = ab + a \cdot i_2 + i_1 \cdot b + i_1 \cdot i_2$$

The last three terms all belong to $I$ by the absorption property (this is exactly where we use the fact that $I$ is an ideal, not just a subring). So $a'b' - ab \in I$, which means $I + a'b' = I + ab$. $\square$

> This is why we need ideals rather than just subrings for quotient constructions. Subrings are closed under multiplication *by their own elements*, but ideals are closed under multiplication *by any ring element*. Without the absorption property, the quotient multiplication would not be well-defined.

### Example: $\mathbb{Z}/4\mathbb{Z}$

The ideal $4\mathbb{Z} = \lbrace \ldots, -8, -4, 0, 4, 8, \ldots \rbrace$ partitions $\mathbb{Z}$ into four cosets:

- $4\mathbb{Z} + 0 = \lbrace \ldots, -8, -4, 0, 4, 8, \ldots \rbrace$
- $4\mathbb{Z} + 1 = \lbrace \ldots, -7, -3, 1, 5, 9, \ldots \rbrace$
- $4\mathbb{Z} + 2 = \lbrace \ldots, -6, -2, 2, 6, 10, \ldots \rbrace$
- $4\mathbb{Z} + 3 = \lbrace \ldots, -5, -1, 3, 7, 11, \ldots \rbrace$

Writing $\bar{0}, \bar{1}, \bar{2}, \bar{3}$ for these cosets, the addition and multiplication tables are:

<table style="display:inline-table; margin: 1rem 1.5rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px; font-size: 0.9rem;">Addition in $\mathbb{Z}/4\mathbb{Z}$</caption>
<tr><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$+$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</th></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td></tr>
</table>

<table style="display:inline-table; margin: 1rem 1.5rem; border-collapse: collapse; text-align: center;">
<caption style="font-weight:bold; margin-bottom:4px; font-size: 0.9rem;">Multiplication in $\mathbb{Z}/4\mathbb{Z}$</caption>
<tr><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\cdot$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</th><th style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</th></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td></tr>
<tr><td style="border: 1px solid var(--text-secondary); padding: 6px 12px; font-weight:bold;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{0}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{3}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{2}$</td><td style="border: 1px solid var(--text-secondary); padding: 6px 12px;">$\bar{1}$</td></tr>
</table>

This quotient ring $\mathbb{Z}/4\mathbb{Z}$ is isomorphic to $\mathbb{Z}_4$ — the familiar integers mod 4, with both addition and multiplication mod 4. Remember: column $\cdot$ row order (these rings happen to be commutative, but the convention matters for non-commutative rings).

### A Non-example

What happens if we try to quotient by a subring that isn't an ideal? Consider $\mathbb{Z}$ as a subring of $\mathbb{Q}$. Since $\mathbb{Z}$ is **not** an ideal of $\mathbb{Q}$ (as we showed earlier), we cannot form a quotient ring $\mathbb{Q}/\mathbb{Z}$. Attempting to do so, multiplication would not be well-defined: the cosets $(\mathbb{Z} + \frac{1}{2})$ and $(\mathbb{Z} + \frac{1}{3})$ would give $(\mathbb{Z} + \frac{1}{2}) \cdot (\mathbb{Z} + \frac{1}{3}) = \mathbb{Z} + \frac{1}{6}$, but choosing different representatives yields different results.

---

## The Fundamental Homomorphism Theorem for Rings

**Theorem (FHT for Rings).** If $f: R \to S$ is a surjective ring homomorphism with kernel $K = \ker(f)$, then:

$$R / K \cong S$$

More precisely, the map $\varphi: R/K \to S$ defined by $\varphi(K + a) = f(a)$ is a well-defined ring isomorphism.

*Proof sketch.* Almost everything carries over directly from the group-theoretic FHT applied to the abelian group $(R, +)$ and the normal subgroup $K$. That gives us a well-defined group isomorphism $\varphi$ with $\varphi(K + a) = f(a)$. We only need to check that $\varphi$ also preserves multiplication:

$$\varphi\big((K + a) \cdot (K + b)\big) = \varphi(K + a \cdot b) = f(a \cdot b) = f(a) \cdot f(b) = \varphi(K + a) \cdot \varphi(K + b)$$

Thus $\varphi$ respects multiplication. $\square$

> The FHT for rings is almost a freebie once you have the FHT for groups. The additive structure does all the heavy lifting; you just check that multiplication comes along for the ride.

**Corollary.** The kernel of any ring homomorphism is always an ideal.

---

## Maximal and Prime Ideals

Given an ideal $I$ of a ring $R$, what can we say about the quotient $R/I$? Two special types of ideals determine whether the quotient has particularly nice properties.

**Definition.** An ideal $I$ of a commutative ring $R$ is **maximal** if the only ideals containing $I$ are $I$ itself and $R$. In other words, there is no ideal "between" $I$ and $R$.

**Definition.** An ideal $I$ of a commutative ring $R$ is **prime** if whenever $a \cdot b \in I$, either $a \in I$ or $b \in I$.

**Theorem.** Let $I$ be an ideal of a commutative unital ring $R$. Then:

- $R/I$ is a **field** $\iff$ $I$ is a **maximal** ideal.
- $R/I$ is an **integral domain** $\iff$ $I$ is a **prime** ideal.

Since every field is an integral domain, every maximal ideal is prime — but the converse is not true in general.

### Example: $I = 4\mathbb{Z}$

Consider $I = 4\mathbb{Z}$ as an ideal of $\mathbb{Z}$. Then $\mathbb{Z}/4\mathbb{Z} \cong \mathbb{Z}_4$.

- Is $\mathbb{Z}/4\mathbb{Z}$ a field? **No** — $\bar{2}$ has no multiplicative inverse (check the multiplication table above). So $4\mathbb{Z}$ is **not maximal**. Indeed, $4\mathbb{Z} \subsetneq 2\mathbb{Z} \subsetneq \mathbb{Z}$, so there is an ideal strictly between $4\mathbb{Z}$ and $\mathbb{Z}$.

- Is $\mathbb{Z}/4\mathbb{Z}$ an integral domain? **No** — we have $\bar{2} \cdot \bar{2} = \bar{0}$, so $\bar{2}$ is a zero divisor. So $4\mathbb{Z}$ is **not prime**. Indeed, $2 \cdot 2 = 4 \in 4\mathbb{Z}$ but $2 \notin 4\mathbb{Z}$.

**Contrast.** The ideal $3\mathbb{Z}$ **is** both maximal and prime. The quotient $\mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}_3$ is a field (every nonzero element has an inverse: $\bar{1}^{-1} = \bar{1}$, $\bar{2}^{-1} = \bar{2}$).

> The general pattern in $\mathbb{Z}$: the ideal $n\mathbb{Z}$ is maximal (and prime) if and only if $n$ is prime. This is because $\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n$, which is a field precisely when $n$ is prime.

---

## The Big Picture

Looking back, the arc of this post has followed the same structural template as our group theory journey:

**Groups** asked: What is a set with one well-behaved operation? We found subgroups, discovered that normal subgroups are the right notion for building quotients, and the Fundamental Homomorphism Theorem tied kernels to quotients.

**Rings** asked: What happens when we add a second operation? We found subrings, discovered that ideals (not just subrings) are needed for well-defined quotients, and the FHT for rings carries over almost verbatim — the additive group does the structural work, and we just verify multiplication is compatible.

The new phenomena — zero divisors, units, the spectrum of behavior from general rings to integral domains to fields — all arise from the richer structure that a second operation provides. And the classification results for ideals (maximal $\Leftrightarrow$ field quotient, prime $\Leftrightarrow$ domain quotient) give us powerful tools for understanding ring structure through quotients.

Where does the story go from here? Fields — the rings where every nonzero element is invertible — lead to **field extensions** and **Galois theory**, the machinery that finally explains why there is no formula for the roots of a general quintic polynomial. The question that motivated the birth of abstract algebra comes full circle: the impossibility of the quintic is not a failure of cleverness, but a consequence of the structure of certain field extensions, revealed through the symmetry groups that act on their roots.

The tools we've built — groups, rings, ideals, quotients, homomorphisms — are the language in which that story is told.
