---
layout: post
title: "Graph Theory II: Degree Sequences and Trees"
date: 2024-07-10
tags: [Math, Graph Theory, Combinatorics]
---

In [Part 1](/2024/07/03/graph-theory-foundations.html), we learned to speak the language of graphs: vertices, edges, walks, components, degrees. Now we push deeper. If someone hands you a list of numbers — say $4, 3, 2, 2, 1$ — can you tell whether there is a graph with those vertex degrees? And what are the simplest possible connected graphs — the ones with no redundancy at all?

This is Part 2 of a four-part series. Here we develop the theory of degree sequences, study graph isomorphism, and explore trees: the minimally connected graphs that serve as the backbone of all connected structures.

## What This Post Covers

- **Degree Sequences** — Which lists of numbers can actually be realized as degrees of a graph?
- **The Havel-Hakimi Algorithm** — A constructive test for graphic sequences
- **Graph Isomorphism** — When two graphs are "the same" despite different vertex names
- **Trees** — Minimally connected, maximally acyclic, and equivalent characterizations
- **Properties of Trees** — Leaf structure, edge counts, induction on trees
- **Spanning Trees** — BFS trees, DFS trees, and finding the backbone of a connected graph
- **Counting Trees** — Prufer codes and Cayley's formula $n^{n-2}$

---

## Degree Sequences

The **degree sequence** of a graph $G$ is the list of all vertex degrees, sorted in descending order. For example, a graph with degrees $1, 4, 4, 2, 1, 0, 2, 2$ has degree sequence $4, 4, 2, 2, 2, 1, 1, 0$.

A natural question: given a sequence of numbers, can you tell if it is the degree sequence of some graph? Such a sequence is called **graphic**. We already know two necessary conditions:

1. Every term must be between 0 and $n - 1$ (where $n$ is the length of the sequence).
2. The sum of all terms must be even (by the Handshake Lemma).

But these tests are not sufficient. The sequence $4, 3, 2, 1, 0$ passes both tests but is *not* graphic — a vertex of degree 4 in a 5-vertex graph must be adjacent to every other vertex, but a vertex of degree 0 is adjacent to no one.

### Regular Graphs

A **regular graph** is one where every vertex has the same degree. An **$r$-regular graph** has every vertex of degree $r$. The degree sequence is just $r, r, \ldots, r$.

**Theorem.** An $r$-regular graph on $n$ vertices exists whenever $0 \leq r \leq n - 1$ and at least one of $r$ or $n$ is even.

The proof constructs these graphs explicitly using **Harary graphs**: arrange $n$ vertices around a circle, and connect vertices that are within distance $r/2$ steps of each other. When $r$ is odd and $n$ is even, add a "diameter matching" connecting opposite vertices. This elegant construction covers all valid cases.

---

## The Havel-Hakimi Algorithm

For general sequences, we need a more powerful test. The Havel-Hakimi algorithm provides both a test and a construction.

**The idea.** If a sequence $d_1 \geq d_2 \geq \cdots \geq d_n$ is graphic, then there should be some graph $G$ realizing it where the vertex of highest degree $d_1$ is adjacent to the vertices of the next-highest degrees $d_2, d_3, \ldots, d_{d_1+1}$. If we delete that vertex, the remaining graph has degree sequence:

$$
d_2 - 1,\; d_3 - 1,\; \ldots,\; d_{d_1+1} - 1,\; d_{d_1+2},\; \ldots,\; d_n.
$$

Sort this and repeat. If we eventually reach all zeros, the original sequence was graphic and we can build the graph by reversing the steps. If we get a negative number or an impossibility, it was not graphic.

### A Worked Example

Let's test $3, 3, 2, 2, 2$:

**Step 1.** Delete the first 3, subtract 1 from the next three terms: $2, 1, 1, 2$. Sort: $2, 2, 1, 1$.

**Step 2.** Delete the first 2, subtract 1 from the next two terms: $1, 0, 1$. Sort: $1, 1, 0$.

**Step 3.** Delete the first 1, subtract 1 from the next term: $0, 0$. This is graphic (two isolated vertices).

Now reverse: add back each vertex with edges to the appropriate neighbors. The result is a graph with degree sequence $3, 3, 2, 2, 2$.

### The Havel-Hakimi Theorem

The "guess" that the highest-degree vertex is adjacent to the next-highest-degree vertices is justified by the following theorem:

**Theorem (Havel-Hakimi).** Let $G$ be a graph with vertices $v_1, v_2, \ldots, v_n$ such that $\deg(v_1) \geq \deg(v_2) \geq \cdots \geq \deg(v_n)$, and let $d = \deg(v_1)$. Then there is a graph $H$ with the same vertices and the same degree sequence in which $v_1$ is adjacent to $v_2, v_3, \ldots, v_{d+1}$.

The proof uses **edge swaps**: replace edges $v_1 x$ and $v_i y$ (where $v_i$ has higher degree than $x$) with edges $v_1 v_i$ and $xy$. This doesn't change any vertex degrees but moves $v_1$'s neighbors toward the high-degree vertices. By the extremal principle, the graph that maximizes the number of high-degree neighbors of $v_1$ must already have $v_1$ adjacent to the top-$d$ vertices.

Edge swaps also prove a more general result: any two graphs with the same degree sequence can be transformed into each other by a sequence of edge swaps. This is useful for randomly sampling graphs with a given degree sequence.

---

## Graph Isomorphism

Two graphs $G$ and $H$ are **isomorphic** if there is a bijection $f: V(G) \to V(H)$ that preserves adjacency: $vw$ is an edge of $G$ if and only if $f(v)f(w)$ is an edge of $H$. Such a function $f$ is a **graph isomorphism**. Think of it as a "relabeling" of vertices.

Isomorphic graphs share all structural properties: same number of vertices and edges, same degree sequence, same diameter, same number of components, same bipartiteness. These shared properties are called **invariants**.

To prove two graphs are *not* isomorphic, find an invariant that differs. To prove they *are* isomorphic, exhibit the bijection.

<svg viewBox="0 0 400 160" style="max-width:420px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.8; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
    .gt-map { stroke: var(--primary, #94452b); stroke-width: 1; stroke-dasharray: 4,3; opacity: 0.6; }
    .gt-maplabel { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--primary, #94452b); text-anchor: middle; }
  </style>
  <!-- Graph G (left) -->
  <line x1="30" y1="30" x2="130" y2="30" class="gt-edge"/>
  <line x1="130" y1="30" x2="130" y2="130" class="gt-edge"/>
  <line x1="130" y1="130" x2="30" y2="130" class="gt-edge"/>
  <line x1="30" y1="130" x2="30" y2="30" class="gt-edge"/>
  <line x1="30" y1="30" x2="130" y2="130" class="gt-edge"/>
  <circle cx="30" cy="30" r="13" class="gt-vertex"/>
  <circle cx="130" cy="30" r="13" class="gt-vertex"/>
  <circle cx="130" cy="130" r="13" class="gt-vertex"/>
  <circle cx="30" cy="130" r="13" class="gt-vertex"/>
  <text x="30" y="30" class="gt-label">1</text>
  <text x="130" y="30" class="gt-label">2</text>
  <text x="130" y="130" class="gt-label">3</text>
  <text x="30" y="130" class="gt-label">4</text>
  <!-- Graph H (right) -->
  <line x1="290" y1="20" x2="360" y2="80" class="gt-edge"/>
  <line x1="360" y1="80" x2="290" y2="140" class="gt-edge"/>
  <line x1="290" y1="140" x2="250" y2="80" class="gt-edge"/>
  <line x1="250" y1="80" x2="290" y2="20" class="gt-edge"/>
  <line x1="290" y1="20" x2="290" y2="140" class="gt-edge"/>
  <circle cx="290" cy="20" r="13" class="gt-vertex"/>
  <circle cx="360" cy="80" r="13" class="gt-vertex"/>
  <circle cx="290" cy="140" r="13" class="gt-vertex"/>
  <circle cx="250" cy="80" r="13" class="gt-vertex"/>
  <text x="290" y="20" class="gt-label">a</text>
  <text x="360" y="80" class="gt-label">b</text>
  <text x="290" y="140" class="gt-label">c</text>
  <text x="250" y="80" class="gt-label">d</text>
  <!-- Mapping arrows -->
  <text x="190" y="75" class="gt-maplabel">1&#x2192;a, 2&#x2192;b</text>
  <text x="190" y="92" class="gt-maplabel">3&#x2192;c, 4&#x2192;d</text>
</svg>

*Two isomorphic graphs drawn differently. The mapping $1 \mapsto a, 2 \mapsto b, 3 \mapsto c, 4 \mapsto d$ preserves all edges.*

An **automorphism** is an isomorphism from a graph to itself — a symmetry. Automorphisms let us reduce case analysis in proofs. For example, the cycle graph $C_n$ has automorphisms that can send any edge to any other edge, so when proving something about "an edge of $C_n$," we can assume without loss of generality that it is the edge $v_1 v_n$.

### Self-Complementary Graphs

A graph $G$ is **self-complementary** if $G$ is isomorphic to its complement $\overline{G}$. The path graph $P_4$ is self-complementary, as is the cycle $C_5$. A self-complementary graph on $n$ vertices has exactly $\frac{n(n-1)}{4}$ edges, which forces $n \equiv 0$ or $1 \pmod{4}$. It turns out self-complementary graphs exist for all such $n$.

---

## Trees: Minimally Connected Graphs

A **tree** is a connected graph that is *minimally* connected: removing any edge disconnects it. Equivalently, every edge of a tree is a **bridge**.

<svg viewBox="0 0 380 170" style="max-width:380px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 2; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-leaf { fill: rgba(148, 69, 43, 0.2); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- Edges -->
  <line x1="190" y1="30" x2="110" y2="80" class="gt-edge"/>
  <line x1="190" y1="30" x2="270" y2="80" class="gt-edge"/>
  <line x1="110" y1="80" x2="50" y2="140" class="gt-edge"/>
  <line x1="110" y1="80" x2="150" y2="140" class="gt-edge"/>
  <line x1="270" y1="80" x2="230" y2="140" class="gt-edge"/>
  <line x1="270" y1="80" x2="330" y2="140" class="gt-edge"/>
  <!-- Internal vertices -->
  <circle cx="190" cy="30" r="13" class="gt-vertex"/>
  <circle cx="110" cy="80" r="13" class="gt-vertex"/>
  <circle cx="270" cy="80" r="13" class="gt-vertex"/>
  <!-- Leaves (highlighted) -->
  <circle cx="50" cy="140" r="13" class="gt-leaf"/>
  <circle cx="150" cy="140" r="13" class="gt-leaf"/>
  <circle cx="230" cy="140" r="13" class="gt-leaf"/>
  <circle cx="330" cy="140" r="13" class="gt-leaf"/>
  <!-- Labels -->
  <text x="190" y="30" class="gt-label">r</text>
  <text x="110" y="80" class="gt-label">a</text>
  <text x="270" y="80" class="gt-label">b</text>
  <text x="50" y="140" class="gt-label">c</text>
  <text x="150" y="140" class="gt-label">d</text>
  <text x="230" y="140" class="gt-label">e</text>
  <text x="330" y="140" class="gt-label">f</text>
</svg>

*A tree on 7 vertices. Leaves (degree 1) are shaded. Removing any edge disconnects the tree.*

### Equivalent Characterizations

**Theorem (Bridge Criterion).** In a connected graph $G$, an edge $vw$ is a bridge if and only if $vw$ does not lie on any cycle.

*Proof.* If $vw$ is on a cycle, then any walk using $vw$ can "go the long way around" the cycle instead, so $G - vw$ is still connected. Conversely, if $vw$ is not a bridge, then $G - vw$ is connected, so there is a $v$–$w$ path in $G - vw$, which together with edge $vw$ forms a cycle. $\square$

This immediately gives us multiple equivalent definitions of a tree:

**Proposition.** The following are equivalent for a graph $T$:

1. $T$ is a tree (connected, and every edge is a bridge).
2. $T$ is connected and acyclic (has no cycles).
3. $T$ is acyclic, but adding any edge would create a cycle (maximally acyclic).
4. Between any two vertices of $T$, there is exactly one path.

> Trees are *simultaneously* the most connected acyclic graphs and the least connected connected graphs. They sit at the boundary between two worlds.

---

## Properties of Trees

### Edge Count

**Theorem.** Every tree on $n$ vertices has exactly $n - 1$ edges.

*Proof sketch.* A graph with $n$ vertices and $m$ edges has at least $n - m$ connected components (proved by induction on $m$: each edge merges at most two components). A tree is connected (one component), so $n - m \leq 1$, giving $m \geq n - 1$. But a connected graph with $n - 1$ edges must be minimally connected — deleting any edge drops the count below $n - 1$ and disconnects it. So trees have exactly $n - 1$ edges. $\square$

More generally, a **forest** (an acyclic graph, not necessarily connected) with $k$ components has exactly $n - k$ edges. Each component is a tree, and the edge counts add up.

### Leaves

A vertex of degree 1 in a graph is a **leaf**. Trees always have them:

**Theorem.** Any tree with $n \geq 2$ vertices has at least 2 leaves.

*Proof.* Trees have no cycles, so $\delta(T) \leq 1$ (otherwise every vertex has degree $\geq 2$, which forces a cycle). Since trees are connected, there are no isolated vertices (degree 0) when $n \geq 2$. So at least one leaf exists.

Suppose there is only one leaf. Then one vertex has degree 1 and the other $n - 1$ have degree $\geq 2$. The degree sum would be at least $2(n - 1) + 1 = 2n - 1$. But the degree sum equals $2(n - 1) = 2n - 2$. Contradiction. $\square$

### Induction on Trees

Leaves give us a powerful induction template. If $T$ is a tree and $v$ is a leaf, then $T - v$ is a tree on $n - 1$ vertices ($n - 2$ edges, still connected, still acyclic). So we can prove things about trees by induction: delete a leaf, apply the hypothesis, then add the leaf back.

**Theorem.** All trees are bipartite.

*Proof.* By induction on $n$. A single vertex is trivially bipartite. For $n \geq 2$, let $v$ be a leaf of $T$ with neighbor $w$. By induction, $T - v$ has a bipartition $(A, B)$. If $w \in A$, put $v$ in $B$; if $w \in B$, put $v$ in $A$. The only new edge $vw$ crosses the bipartition. $\square$

---

## Spanning Trees

**Theorem.** A graph $G$ is connected if and only if it has a spanning tree.

*Proof.* If $G$ has a spanning tree $T$, then any two vertices have a path between them in $T$ (hence in $G$). Conversely, if $G$ is connected, repeatedly delete non-bridge edges. The process terminates at a spanning tree. $\square$

This theorem is practically important: spanning trees are the "most efficient" way to keep a graph connected.

### BFS and DFS Trees

A **breadth-first search (BFS) tree** from vertex $v$ is built by exploring all vertices at distance 1, then distance 2, and so on. Whenever a new vertex $x$ is discovered as a neighbor of an already-visited vertex $y$, include edge $xy$ in the tree. The BFS tree naturally computes distances from $v$.

A **depth-first search (DFS) tree** from vertex $v$ is built by walking as far as possible without revisiting vertices, then backtracking. This produces long paths in the tree and is useful for finding cycles and bridges.

### Minimum-Cost Spanning Trees

When edges have costs, a **minimum-cost spanning tree** minimizes total edge cost while keeping the graph connected. A simple greedy algorithm works: process edges from most to least expensive, deleting any edge that is not a bridge. The remaining edges form the minimum-cost spanning tree.

---

## Counting Trees: Prufer Codes and Cayley's Formula

How many trees can be built on a fixed set of $n$ labeled vertices? The answer is given by one of the most elegant results in combinatorics.

### Prufer Codes

A **Prufer code** is a bijection between labeled trees on $\{v_1, v_2, \ldots, v_n\}$ and sequences of length $n - 2$ from the alphabet $\{1, 2, \ldots, n\}$.

**Encoding:** Given a tree $T$, repeatedly find the leaf $v_i$ with smallest index $i$, record the index of its unique neighbor, then delete $v_i$. After $n - 2$ deletions, two vertices remain.

**Example.** Consider a tree on $\{v_1, \ldots, v_7\}$ with edges $v_1 v_4, v_3 v_4, v_4 v_6, v_5 v_2, v_6 v_2, v_2 v_7$:

- Smallest leaf is $v_1$, neighbor is $v_4$. Record 4. Delete $v_1$.
- Smallest leaf is $v_3$, neighbor is $v_4$. Record 4. Delete $v_3$.
- Smallest leaf is $v_4$, neighbor is $v_6$. Record 6. Delete $v_4$.
- Smallest leaf is $v_5$, neighbor is $v_2$. Record 2. Delete $v_5$.
- Smallest leaf is $v_6$, neighbor is $v_2$. Record 2. Delete $v_6$.

Prufer code: $(4, 4, 6, 2, 2)$. The remaining vertices $v_2, v_7$ form the last edge.

**Decoding.** The code can be reversed: the number of times vertex $v_i$ appears in the code equals $\deg(v_i) - 1$. Vertices that never appear in the code are leaves. This is enough information to reconstruct the tree.

The key property is that every sequence in $\{1, 2, \ldots, n\}^{n-2}$ decodes to a valid tree, and the encoding and decoding are inverses. This establishes a bijection.

**Theorem (Cayley's Formula).** The number of labeled trees on $n$ vertices is $n^{n-2}$.

*Proof.* There are $n^{n-2}$ possible Prufer codes (sequences of length $n - 2$ from an $n$-element alphabet). Since the Prufer code gives a bijection between labeled trees and such sequences, there are exactly $n^{n-2}$ labeled trees. $\square$

For small values: $n = 1$ gives $1^{-1} = 1$ tree, $n = 2$ gives $2^0 = 1$ tree, $n = 3$ gives $3^1 = 3$ trees, $n = 4$ gives $4^2 = 16$ trees. These match the exhaustive counts.

> Cayley's formula is remarkable not just for the clean answer, but for the method. The Prufer code transforms a geometric/combinatorial object (a tree) into a purely algebraic one (a sequence), making counting trivial.

---

Trees are the skeleton of graph theory — they underlie connected graphs as spanning trees, provide induction frameworks, and have a beautiful counting theory. But the real power of graph theory emerges when we ask optimization questions. In [Part 3](/2024/07/17/graph-theory-matchings.html), we study the matching problem: given a bipartite graph, how many edges can we select so that no vertex is used twice?
