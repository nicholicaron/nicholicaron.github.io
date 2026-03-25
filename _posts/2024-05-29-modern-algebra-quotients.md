---
layout: post
title: "Modern Algebra I: Quotients and Actions"
date: 2024-05-29
tags: [Math, Algebra, Group Theory]
---

This is Part 3 of my Modern Algebra I notes, covering the deepest material in the course. In [Part 1](/2024/05/15/modern-algebra-groups.html) we built groups from axioms, and in [Part 2](/2024/05/22/modern-algebra-permutations.html) we developed permutations and isomorphisms. Now we reach the structural heart of group theory: cosets and Lagrange's theorem reveal hidden arithmetic constraints on subgroups, homomorphisms generalize isomorphisms to maps that can "compress" structure, and quotient groups let us build new groups by collapsing old ones. The Fundamental Homomorphism Theorem ties it all together.

## What This Post Covers

- **Partitions and Equivalence Relations** — The set theory underlying cosets
- **Cosets and Lagrange's Theorem** — Why subgroup sizes divide group sizes
- **Homomorphisms** — Structure-preserving maps and their kernels
- **Normal Subgroups** — The subgroups that play nicely with quotients
- **Quotient Groups** — Building new groups by "dividing out" a subgroup
- **The Fundamental Homomorphism Theorem** — The bridge between kernels and quotients
- **Groups Acting on Sets** — Orbits, stabilizers, and Burnside's counting theorem

---

## Partitions and Equivalence Relations

Before cosets, we need one piece of set theory.

**Definition.** A **partition** of a set $S$ is a collection of disjoint, non-empty subsets whose union is all of $S$.

**Definition.** An **equivalence relation** $\sim$ on $S$ is a relation that is:
- **Reflexive**: $a \sim a$
- **Symmetric**: $a \sim b \Rightarrow b \sim a$
- **Transitive**: $a \sim b$ and $b \sim c \Rightarrow a \sim c$

The **equivalence class** of $a$ is $[a] = \lbrace x \in S : x \sim a\rbrace$.

**Theorem.** Every partition of $S$ defines a unique equivalence relation (elements are related iff they're in the same piece), and vice versa.

This might seem abstract, but it's exactly what cosets do: they partition a group into equal-sized pieces.

---

## Cosets and Lagrange's Theorem

**Definition.** Given a group $(G, *)$, a subgroup $H$ of $G$, and an element $a \in G$:
- The **right coset** of $H$ with respect to $a$ is $Ha = \lbrace h * a : h \in H\rbrace$
- The **left coset** of $H$ with respect to $a$ is $aH = \lbrace a * h : h \in H\rbrace$

> Left and right cosets are the same when the group is abelian. In non-abelian groups, they can differ — and this difference turns out to be critical.

**Example.** In $(\mathbb{Z}, +)$ with subgroup $3\mathbb{Z} = \lbrace\ldots, -6, -3, 0, 3, 6, \ldots\rbrace$:
- $3\mathbb{Z} + 0 = \lbrace\ldots, -6, -3, 0, 3, 6, \ldots\rbrace$
- $3\mathbb{Z} + 1 = \lbrace\ldots, -5, -2, 1, 4, 7, \ldots\rbrace$
- $3\mathbb{Z} + 2 = \lbrace\ldots, -4, -1, 2, 5, 8, \ldots\rbrace$

These three cosets partition $\mathbb{Z}$ into three pieces of "equal size." This is modular arithmetic in disguise — the cosets are exactly the residue classes mod 3.

### Key Properties

- $H = He$ is always a coset of $H$. It's the only coset that is a subgroup.
- $Ha$ is the (unique) coset containing the element $a$.
- For a fixed subgroup $H$, the cosets of $H$ **partition** $G$ into parts of equal size (there are bijections between any two cosets).

**Definition.** The **index** of $H$ in $G$, written $(G : H)$, is the number of distinct cosets of $H$ in $G$:

$$(G : H) = \frac{\lvert G \rvert}{\lvert H \rvert}$$

### Lagrange's Theorem

**Theorem (Lagrange).** If $G$ is a finite group and $H$ is a subgroup of $G$, then $\lvert H \rvert$ divides $\lvert G \rvert$.

*Proof.* The cosets of $H$ partition $G$ into $(G : H)$ pieces, each of size $\lvert H \rvert$. So $\lvert G \rvert = (G : H) \cdot \lvert H \rvert$. $\square$

This simple-looking theorem has powerful consequences:

**Corollary.** If $G$ is a finite group and $a \in G$, then $\text{ord}(a)$ divides $\lvert G \rvert$.

*Proof.* $\text{ord}(a) = \lvert\langle a \rangle\rvert$, and $\langle a \rangle$ is a subgroup of $G$. $\square$

**Corollary.** If $G$ is a finite group of prime order $p$, then $G$ is cyclic (and therefore isomorphic to $\mathbb{Z}_p$).

*Proof.* The only divisors of $p$ are 1 and $p$. Pick any $a \neq e$; then $\lvert\langle a \rangle\rvert$ divides $p$ and is greater than 1, so $\lvert\langle a \rangle\rvert = p = \lvert G \rvert$. $\square$

---

## Homomorphisms

Isomorphisms are bijective structure-preserving maps. What if we drop the bijectivity requirement?

**Definition.** A **homomorphism** from $(G_1, *_1)$ to $(G_2, *_2)$ is a function $f: G_1 \to G_2$ such that:

$$f(x *_1 y) = f(x) *_2 f(y) \quad \text{for all } x, y \in G_1$$

An isomorphism is a bijective homomorphism — but homomorphisms in general can "lose information."

**Key properties** of a homomorphism $f: G \to H$:
- $f(e_G) = e_H$ (identities map to identities)
- $f(a^{-1}) = f(a)^{-1}$ (inverses map to inverses)

> **Important.** Unlike isomorphisms, a homomorphism can map a non-abelian group to an abelian one, a large group to a small one, or a non-cyclic group to a cyclic one. Homomorphisms can compress, but never "create" structure.

### The Kernel

**Definition.** The **kernel** of a homomorphism $f: G \to H$ is:

$$\ker(f) = \{g \in G : f(g) = e_H\}$$

The kernel measures how much information $f$ loses. If $\ker(f) = \lbrace e_G\rbrace$, then $f$ is injective (no information lost). If $\ker(f) = G$, then $f$ maps everything to the identity (all information lost).

**Fact.** The kernel of a homomorphism is always a subgroup of the domain. The image of a homomorphism is always a subgroup of the codomain.

---

## Normal Subgroups

Not every subgroup can serve as a kernel. The ones that can are special.

**Definition.** Two elements $a, b \in G$ are **conjugate** if there exists $g \in G$ such that $gag^{-1} = b$.

**Definition.** A subgroup $H$ of $G$ is **normal** (written $H \trianglelefteq G$) if $gHg^{-1} = H$ for all $g \in G$. Equivalently, $H$ is normal iff $gH = Hg$ for all $g \in G$ — the left and right cosets coincide.

> In abelian groups, every subgroup is normal (since $gH = Hg$ is automatic when everything commutes). Normal subgroups become interesting precisely in non-abelian groups.

**Key fact.** $H$ is normal iff $H$ is "closed under conjugation" — whenever $h \in H$ and $g \in G$, we have $ghg^{-1} \in H$.

**Fact.** The kernel of any homomorphism is a normal subgroup.

**Fact.** A normal subgroup is a subgroup which is "closed under conjugates."

---

## Quotient Groups

Here's the payoff of normality. Given a normal subgroup $N \trianglelefteq G$, we can build a brand new group whose elements are the *cosets* of $N$.

**Definition.** The **quotient group** $G/N$ consists of the set of cosets $\lbrace gN : g \in G\rbrace$ with the operation:

$$(gN)(hN) = (gh)N$$

This operation is well-defined (independent of which coset representatives $g, h$ we choose) **if and only if** $N$ is normal.

**Example.** $\mathbb{Z}/3\mathbb{Z}$ has three elements: $\lbrace 0 + 3\mathbb{Z}, \; 1 + 3\mathbb{Z}, \; 2 + 3\mathbb{Z}\rbrace$. The operation is addition of cosets, and $\mathbb{Z}/3\mathbb{Z} \cong \mathbb{Z}_3$.

> Think of a quotient group as "collapsing" all elements of $N$ to a single identity, and grouping every other element by which coset it belongs to. The quotient $G/N$ is the group of "equivalence classes" under the relation $a \sim b \iff ab^{-1} \in N$.

---

## The Fundamental Homomorphism Theorem

This is the crown jewel of the course — it connects everything.

**Theorem (Fundamental Homomorphism Theorem).** If $f: G \to H$ is a surjective homomorphism with kernel $K$, then:

$$G/K \cong H$$

More precisely, the map $\phi: G/K \to H$ defined by $\phi(Kx) = f(x)$ is a well-defined isomorphism.

In other words: the quotient of $G$ by the kernel of a surjective homomorphism perfectly "divides out" the redundancy, yielding a group isomorphic to the image.

**Example.** Consider $f: \mathbb{Z} \to \mathbb{Z}_n$ defined by $f(x) = x \bmod n$. This is a surjective homomorphism with $\ker(f) = n\mathbb{Z}$. The theorem gives us $\mathbb{Z}/n\mathbb{Z} \cong \mathbb{Z}_n$ — which is exactly how we defined $\mathbb{Z}_n$ in the first place.

**Practical use:** If $f: G \to H$ is a homomorphism with kernel $K$:

- $f(a) = f(b) \iff Ka = Kb$ (elements have the same image iff they're in the same coset of the kernel)
- The image of $f$ is a subgroup of $H$
- If $f$ is surjective, then $G/K$ has the same structure as $H$

> If you can find a surjective homomorphism $f: G \to H$ with kernel $K$, then $G/K$ has the same structure as $H$. This gives a powerful indirect way to identify quotient groups.

---

## Groups Acting on Sets

We close with one of the most versatile ideas in algebra: using groups to describe symmetry in a general setting.

**Definition.** A **group action** of $G$ on a set $X$ is a function $G \times X \to X$, written $(g, x) \mapsto g \cdot x$, such that:

1. $e \cdot x = x$ for all $x \in X$ (the identity acts trivially)
2. $(g_1 g_2) \cdot x = g_1 \cdot (g_2 \cdot x)$ for all $g_1, g_2 \in G$ and $x \in X$ (the action is compatible with the group operation)

**Definition.** Given a group action of $G$ on $X$:
- The **orbit** of $x$ is $\text{Orb}(x) = \lbrace g \cdot x : g \in G\rbrace$ — all the places $x$ can be moved by the group.
- The **stabilizer** of $x$ is $G_x = \lbrace g \in G : g \cdot x = x\rbrace$ — the elements that fix $x$.
- The **fixed point set** of $g$ is $X_g = \lbrace x \in X : g \cdot x = x\rbrace$ — the points fixed by $g$.

The orbits partition $X$ (elements are in the same orbit iff one can be transformed into the other). The stabilizer $G_x$ is always a subgroup of $G$.

### Burnside's Counting Theorem

When we want to count objects "up to symmetry" — distinct necklaces, distinct colorings, distinct molecular configurations — Burnside's theorem is the tool.

**Theorem (Burnside).** The number of distinct orbits under a group action of $G$ on $X$ is:

$$\text{number of orbits} = \frac{1}{\lvert G \rvert} \sum_{g \in G} \lvert X_g \rvert$$

In words: the number of "truly different" objects equals the average number of fixed points across all group elements.

This theorem connects group actions to combinatorics in a way that makes otherwise intractable counting problems elegant. It appears throughout chemistry (counting molecular isomers), combinatorics (counting distinct colorings), and physics (classifying particle states).

---

## The Big Picture

Looking back across all three parts, the arc of Modern Algebra I follows a clear trajectory:

**Part 1** asked: *What is a group?* We built the definition from operations and axioms, saw it come alive in symmetry groups, and learned to find substructure through subgroups.

**Part 2** asked: *How do groups relate?* Permutation groups gave us a universal representation (Cayley's theorem), cycle notation gave us efficient computation, and isomorphisms told us precisely when two groups are structurally identical.

**Part 3** asked: *How can we build new groups from old ones?* Cosets and Lagrange's theorem revealed hidden divisibility constraints, homomorphisms gave us structure-preserving maps that can compress, and quotient groups let us "divide" a group by a normal subgroup. The Fundamental Homomorphism Theorem unified these ideas: the image of a homomorphism is always isomorphic to the quotient by its kernel.

The beauty of abstract algebra is that these ideas — developed for sets with a single operation — echo through all of mathematics. Rings add a second operation, fields add multiplicative inverses, modules generalize vector spaces, and category theory abstracts the morphisms themselves. But the core intuition is always the same: study structure, not specific objects, and you'll see patterns that were invisible before.

**Next up:** In [Modern Algebra II: Rings, Ideals, and Quotient Rings](/2024/06/05/modern-algebra-rings.html), we add a second operation and watch the entire journey — definitions, substructures, quotients, and the FHT — play out again in a richer setting.
