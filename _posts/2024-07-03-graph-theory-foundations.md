---
layout: post
title: "Graph Theory I: Foundations and First Explorations"
date: 2024-07-03
tags: [Math, Graph Theory, Combinatorics]
---

These are my compiled notes from Graph Theory (Math 3322), reorganized into a narrative that builds the subject from its most basic question: what happens when you connect dots with lines? Graph theory is the study of networks in their purest form — stripped of geometry, physics, and everything except the bare relationship of "these two things are linked." It shows up everywhere: social networks, circuit boards, road maps, scheduling problems, even puzzles.

This is Part 1 of a four-part series. Here we define graphs, learn to navigate them, meet the most important named examples, and discover the first structural theorems that constrain what graphs can look like.

## What This Post Covers

- **What Is a Graph?** — Vertices, edges, and the formal definition
- **Walks, Paths, and Distances** — How to move through a graph, and why the shortest route never revisits a vertex
- **Connected Components** — When a graph breaks into pieces
- **The Graph Zoo** — Path graphs, cycles, complete graphs, bipartite graphs, hypercubes, and the Petersen graph
- **Bipartite Graphs** — The elegant characterization: no odd cycles
- **Vertex Degree and the Handshake Lemma** — Counting edges by counting endpoints

---

## What Is a Graph?

Suppose you want to take a road trip through the 48 contiguous US states without visiting any state twice. To plan this, you don't need a full geographic map — you just need to know which states share a border. You could encode this as follows:

1. Make a list of all states: AL, AZ, AR, ..., WY.
2. Make a list of all pairs of states with a road between them, like $\{\text{AL}, \text{FL}\}$ or $\{\text{MO}, \text{TN}\}$.

This is exactly what a graph is.

**Definition.** A **graph** $G$ is a pair $(V, E)$ where:

- $V$ is a set of objects called **vertices** (these can be anything),
- $E$ is a set of **edges**; each edge is a pair $\{v, w\}$ of vertices $v, w \in V$, and tells us that $v$ and $w$ are **adjacent**.

We often write the edge between $v$ and $w$ as $vw$ instead of $\{v, w\}$. This is unordered: $vw$ and $wv$ are the same edge.

It is often convenient to represent a graph by a diagram where the edges are drawn as lines connecting vertices. The way the diagram is drawn — the positions, the lengths of lines, whether edges cross — is irrelevant. Only the connections matter.

<svg viewBox="0 0 340 180" style="max-width:340px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 2; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- Edges -->
  <line x1="50" y1="40" x2="150" y2="40" class="gt-edge"/>
  <line x1="150" y1="40" x2="250" y2="40" class="gt-edge"/>
  <line x1="50" y1="40" x2="100" y2="140" class="gt-edge"/>
  <line x1="150" y1="40" x2="100" y2="140" class="gt-edge"/>
  <line x1="150" y1="40" x2="200" y2="140" class="gt-edge"/>
  <line x1="250" y1="40" x2="200" y2="140" class="gt-edge"/>
  <line x1="100" y1="140" x2="200" y2="140" class="gt-edge"/>
  <line x1="250" y1="40" x2="300" y2="140" class="gt-edge"/>
  <!-- Vertices -->
  <circle cx="50" cy="40" r="14" class="gt-vertex"/>
  <circle cx="150" cy="40" r="14" class="gt-vertex"/>
  <circle cx="250" cy="40" r="14" class="gt-vertex"/>
  <circle cx="100" cy="140" r="14" class="gt-vertex"/>
  <circle cx="200" cy="140" r="14" class="gt-vertex"/>
  <circle cx="300" cy="140" r="14" class="gt-vertex"/>
  <!-- Labels -->
  <text x="50" y="40" class="gt-label">a</text>
  <text x="150" y="40" class="gt-label">b</text>
  <text x="250" y="40" class="gt-label">c</text>
  <text x="100" y="140" class="gt-label">d</text>
  <text x="200" y="140" class="gt-label">e</text>
  <text x="300" y="140" class="gt-label">f</text>
</svg>

*A graph on six vertices. For example, $a$ and $b$ are adjacent (there is an edge $ab$), but $a$ and $e$ are not.*

Graphs model an astonishing variety of problems. The Towers of Hanoi puzzle becomes a graph where each vertex is a puzzle state and each edge connects states that are one move apart. A circuit board becomes a graph where components are vertices and wires are edges. Even tiling puzzles can be modeled as graphs: place a vertex for each way to position a tile, and put an edge between incompatible placements. Then a valid tiling corresponds to an **independent set** — a set of vertices with no edges between them.

---

## Walks, Paths, and Distances

Once we have a graph, the most natural thing to do is move around in it. This leads to one of the most fundamental concepts in graph theory.

**Definition.** A **$v$–$w$ walk** in a graph $G$ is a sequence of vertices $v_0, v_1, v_2, \ldots, v_\ell$ where $v_0 = v$, $v_\ell = w$, and for each $i = 0, 1, \ldots, \ell - 1$, the pair $v_i v_{i+1}$ is an edge. The number $\ell$ is the **length** of the walk.

A **$v$–$w$ path** is a walk in which no vertices are repeated.

For example, in the graph above, the sequence $(a, b, e, d)$ is an $a$–$d$ path of length 3. But $(a, b, e, d, b)$ is an $a$–$b$ walk of length 4 that is *not* a path, because $b$ appears twice.

Walks can be arbitrarily long and redundant — you can wander back and forth forever. Paths, on the other hand, are efficient. The key theorem connecting the two is proved using a technique called the **extremal principle**: pick the best object of a certain type, then argue it must have the property you want.

**Theorem.** If there is a $v$–$w$ walk in $G$, then there is also a $v$–$w$ path. Moreover, the shortest $v$–$w$ walk is always a path.

*Proof.* Let $(v_0, v_1, \ldots, v_\ell)$ with $v_0 = v$ and $v_\ell = w$ be the shortest $v$–$w$ walk. We claim it is a path.

Suppose not: some vertex repeats, say $v_i = v_j$ with $i < j$. Then we could "shortcut" the walk by removing the loop:

$$
(v_0, v_1, \ldots, v_i, v_{j+1}, \ldots, v_\ell).
$$

This is still a valid walk (consecutive vertices are still adjacent, since the critical pair $v_i$ and $v_{j+1}$ are adjacent because $v_i = v_j$ and $v_j v_{j+1}$ was an edge in the original walk). But its length is $\ell - (j - i) < \ell$, contradicting our choice of the *shortest* walk. So no vertex repeats, and the walk is a path. $\square$

> This theorem lets us always prove the easier thing (a walk exists) and assume the stronger thing (a path exists). It is a recurring pattern in graph theory: prove existence loosely, then tighten for free.

**Definition.** The **distance** $d(v, w)$ between vertices $v$ and $w$ is the length of the shortest $v$–$w$ path. If no such path exists (because $v$ and $w$ are in different pieces of the graph), we write $d(v, w) = \infty$. The **diameter** of a graph is the largest distance between any two of its vertices.

---

## Connected Components

The three cups puzzle illustrates an important phenomenon. You have three cups in a row, and in one move you flip two consecutive cups. Starting with all cups up (UUU), can you reach all cups down (DDD)?

If we draw the graph of all $8$ states with edges between states one move apart, it splits into two disconnected pieces — and UUU and DDD land in different pieces. No sequence of moves can bridge the gap.

**Definition.** We say $v \sim w$ if there is a $v$–$w$ walk in $G$. This relation is an equivalence relation on $V(G)$ — it is reflexive (the zero-length walk $(v)$), symmetric (reverse a walk), and transitive (concatenate two walks).

The equivalence classes of $\sim$ are the **connected components** of $G$. A graph with exactly one connected component is called **connected**.

**Theorem.** The vertices of any graph can be partitioned into connected components.

This is powerful because it means we can often solve problems for each component separately. If we are looking for an independent set, or a circuit board layout, or a tiling, we can handle each component independently and combine the results.

### Subgraphs

A **subgraph** $H$ of $G$ is a graph whose vertices $V(H) \subseteq V(G)$ and edges $E(H) \subseteq E(G)$. If $S \subseteq V(G)$ and $H$ has all edges of $G$ between vertices in $S$, then $H$ is the **induced subgraph** $G[S]$. A **spanning subgraph** includes all vertices of $G$ but possibly not all edges.

---

## The Graph Zoo

Graph theory has a collection of named graphs that appear everywhere. Think of this as a bestiary — characters you will encounter again and again.

### Path and Cycle Graphs

The **path graph** $P_n$ has vertices $v_1, v_2, \ldots, v_n$ and edges $v_i v_{i+1}$ for $i = 1, \ldots, n-1$. Its diameter is $n - 1$ (the endpoints are as far apart as possible).

The **cycle graph** $C_n$ (for $n \geq 3$) adds one more edge to $P_n$: the edge $v_1 v_n$, closing the path into a loop. Every vertex has degree 2.

### Complete Graphs

The **complete graph** $K_n$ has $n$ vertices and *every* possible edge $v_i v_j$ with $i \neq j$. The number of edges is $\binom{n}{2} = \frac{n(n-1)}{2}$. Every vertex has degree $n - 1$.

### Complete Bipartite Graphs

The **complete bipartite graph** $K_{m,n}$ has vertex set split into parts $A = \{v_1, \ldots, v_m\}$ and $B = \{w_1, \ldots, w_n\}$, with every possible edge between $A$ and $B$ (but no edges within $A$ or within $B$). It has $mn$ edges.

### The Hypercube

The **hypercube graph** $Q_d$ has vertices $\{0, 1\}^d$ — all binary strings of length $d$ — with an edge between strings that differ in exactly one coordinate. The cube graph $Q_3$ looks like a cube when drawn in 3D. The hypercube $Q_d$ has $2^d$ vertices and $d \cdot 2^{d-1}$ edges. Its diameter is $d$ (opposite corners like $000\ldots0$ and $111\ldots1$ differ in every coordinate).

<svg viewBox="0 0 300 260" style="max-width:300px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.8; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'JetBrains Mono', monospace; font-size: 11px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- Back face edges -->
  <line x1="100" y1="40" x2="200" y2="40" class="gt-edge" stroke-dasharray="6,3" opacity="0.5"/>
  <line x1="100" y1="40" x2="100" y2="140" class="gt-edge" stroke-dasharray="6,3" opacity="0.5"/>
  <line x1="200" y1="40" x2="200" y2="140" class="gt-edge" stroke-dasharray="6,3" opacity="0.5"/>
  <line x1="100" y1="140" x2="200" y2="140" class="gt-edge" stroke-dasharray="6,3" opacity="0.5"/>
  <!-- Connecting edges (front to back) -->
  <line x1="60" y1="80" x2="100" y2="40" class="gt-edge"/>
  <line x1="240" y1="80" x2="200" y2="40" class="gt-edge"/>
  <line x1="60" y1="180" x2="100" y2="140" class="gt-edge"/>
  <line x1="240" y1="180" x2="200" y2="140" class="gt-edge"/>
  <!-- Front face edges -->
  <line x1="60" y1="80" x2="240" y2="80" class="gt-edge"/>
  <line x1="60" y1="80" x2="60" y2="180" class="gt-edge"/>
  <line x1="240" y1="80" x2="240" y2="180" class="gt-edge"/>
  <line x1="60" y1="180" x2="240" y2="180" class="gt-edge"/>
  <!-- Back vertices -->
  <circle cx="100" cy="40" r="13" class="gt-vertex"/>
  <circle cx="200" cy="40" r="13" class="gt-vertex"/>
  <circle cx="100" cy="140" r="13" class="gt-vertex"/>
  <circle cx="200" cy="140" r="13" class="gt-vertex"/>
  <!-- Front vertices -->
  <circle cx="60" cy="80" r="13" class="gt-vertex"/>
  <circle cx="240" cy="80" r="13" class="gt-vertex"/>
  <circle cx="60" cy="180" r="13" class="gt-vertex"/>
  <circle cx="240" cy="180" r="13" class="gt-vertex"/>
  <!-- Labels -->
  <text x="100" y="40" class="gt-label">101</text>
  <text x="200" y="40" class="gt-label">111</text>
  <text x="100" y="140" class="gt-label">100</text>
  <text x="200" y="140" class="gt-label">110</text>
  <text x="60" y="80" class="gt-label">001</text>
  <text x="240" y="80" class="gt-label">011</text>
  <text x="60" y="180" class="gt-label">000</text>
  <text x="240" y="180" class="gt-label">010</text>
  <!-- Caption -->
  <text x="150" y="220" style="font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-secondary, #666); text-anchor: middle; font-style: italic;">The hypercube Q&#x2083; with binary vertex labels</text>
</svg>

The hypercube is always bipartite: vertices with an even number of 1's go on one side, vertices with an odd number on the other. Any edge changes exactly one coordinate, flipping the parity.

### The Petersen Graph

The **Petersen graph** is the most famous small graph in all of graph theory. Its vertices are the $\binom{5}{2} = 10$ two-element subsets of $\{1, 2, 3, 4, 5\}$, and two vertices are adjacent if their subsets are disjoint: $\{a, b\}$ is adjacent to $\{c, d\}$ whenever $\{a, b\} \cap \{c, d\} = \emptyset$. Every vertex has degree 3, and the graph has 15 edges.

<svg viewBox="0 0 300 300" style="max-width:280px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.8; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 9px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- Outer pentagon edges -->
  <line x1="150" y1="20" x2="274" y2="110" class="gt-edge"/>
  <line x1="274" y1="110" x2="227" y2="258" class="gt-edge"/>
  <line x1="227" y1="258" x2="73" y2="258" class="gt-edge"/>
  <line x1="73" y1="258" x2="26" y2="110" class="gt-edge"/>
  <line x1="26" y1="110" x2="150" y2="20" class="gt-edge"/>
  <!-- Inner pentagram edges (star shape) -->
  <line x1="150" y1="100" x2="198" y2="225" class="gt-edge"/>
  <line x1="198" y1="225" x2="80" y2="152" class="gt-edge"/>
  <line x1="80" y1="152" x2="220" y2="152" class="gt-edge"/>
  <line x1="220" y1="152" x2="102" y2="225" class="gt-edge"/>
  <line x1="102" y1="225" x2="150" y2="100" class="gt-edge"/>
  <!-- Spokes (outer to inner) -->
  <line x1="150" y1="20" x2="150" y2="100" class="gt-edge"/>
  <line x1="274" y1="110" x2="220" y2="152" class="gt-edge"/>
  <line x1="227" y1="258" x2="198" y2="225" class="gt-edge"/>
  <line x1="73" y1="258" x2="102" y2="225" class="gt-edge"/>
  <line x1="26" y1="110" x2="80" y2="152" class="gt-edge"/>
  <!-- Outer vertices -->
  <circle cx="150" cy="20" r="14" class="gt-vertex"/>
  <circle cx="274" cy="110" r="14" class="gt-vertex"/>
  <circle cx="227" cy="258" r="14" class="gt-vertex"/>
  <circle cx="73" cy="258" r="14" class="gt-vertex"/>
  <circle cx="26" cy="110" r="14" class="gt-vertex"/>
  <!-- Inner vertices -->
  <circle cx="150" cy="100" r="14" class="gt-vertex"/>
  <circle cx="220" cy="152" r="14" class="gt-vertex"/>
  <circle cx="198" cy="225" r="14" class="gt-vertex"/>
  <circle cx="102" cy="225" r="14" class="gt-vertex"/>
  <circle cx="80" cy="152" r="14" class="gt-vertex"/>
  <!-- Labels: outer -->
  <text x="150" y="20" class="gt-label">{1,2}</text>
  <text x="274" y="110" class="gt-label">{2,3}</text>
  <text x="227" y="258" class="gt-label">{3,4}</text>
  <text x="73" y="258" class="gt-label">{4,5}</text>
  <text x="26" y="110" class="gt-label">{5,1}</text>
  <!-- Labels: inner -->
  <text x="150" y="100" class="gt-label">{3,4}</text>
  <text x="220" y="152" class="gt-label">{4,5}</text>
  <text x="198" y="225" class="gt-label">{5,1}</text>
  <text x="102" y="225" class="gt-label">{1,2}</text>
  <text x="80" y="152" class="gt-label">{2,3}</text>
</svg>

*The Petersen graph. Each vertex is a 2-element subset of $\{1,2,3,4,5\}$; edges join disjoint subsets.*

The Petersen graph is a counterexample to many naive conjectures and will reappear throughout the series.

### Operations on Graphs

The **complement** $\overline{G}$ of a graph $G$ has the same vertices, but exactly the edges that $G$ does *not* have. For any two vertices $u$ and $v$, exactly one of $uv \in E(G)$ or $uv \in E(\overline{G})$ holds.

The **union** $G \cup H$ of two graphs combines their vertices and edges. When $G$ and $H$ share the same vertex set, the union overlays their edges. When they have no vertices in common, it places them side by side as disconnected components.

---

## Bipartite Graphs

A **bipartite graph** is a graph $G$ whose vertex set can be split into two parts $A$ and $B$ such that every edge has one endpoint in $A$ and one in $B$. No edge runs within $A$ or within $B$.

Sometimes bipartiteness is built into the problem — like matching instructors to courses, where edges only go between people and classes. But sometimes a graph turns out to be bipartite for deeper structural reasons. How can we recognize this?

### An Algorithm

Pick any vertex $v$ and put it on side $A$. Put all its neighbors on side $B$. Put all their unvisited neighbors on side $A$. Continue alternating until every reachable vertex has a side. If we ever try to assign a vertex to one side but find it already assigned to the other, the graph is not bipartite.

This algorithm assigns $w$ to side $A$ when $d(v, w)$ is even and to side $B$ when $d(v, w)$ is odd.

<svg viewBox="0 0 400 200" style="max-width:400px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.8; }
    .gt-vA { fill: rgba(148, 69, 43, 0.15); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-vB { fill: rgba(70, 130, 180, 0.15); stroke: steelblue; stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
    .gt-side { font-family: 'Inter', sans-serif; font-size: 14px; font-weight: 600; text-anchor: middle; }
  </style>
  <!-- Side labels -->
  <text x="200" y="22" class="gt-side" fill="var(--primary, #94452b)">Side A (even distance)</text>
  <text x="200" y="192" class="gt-side" fill="steelblue">Side B (odd distance)</text>
  <!-- Edges -->
  <line x1="60" y1="50" x2="100" y2="160" class="gt-edge"/>
  <line x1="60" y1="50" x2="200" y2="160" class="gt-edge"/>
  <line x1="160" y1="50" x2="100" y2="160" class="gt-edge"/>
  <line x1="160" y1="50" x2="200" y2="160" class="gt-edge"/>
  <line x1="160" y1="50" x2="300" y2="160" class="gt-edge"/>
  <line x1="260" y1="50" x2="200" y2="160" class="gt-edge"/>
  <line x1="260" y1="50" x2="300" y2="160" class="gt-edge"/>
  <line x1="350" y1="50" x2="300" y2="160" class="gt-edge"/>
  <!-- Side A vertices (top) -->
  <circle cx="60" cy="50" r="14" class="gt-vA"/>
  <circle cx="160" cy="50" r="14" class="gt-vA"/>
  <circle cx="260" cy="50" r="14" class="gt-vA"/>
  <circle cx="350" cy="50" r="14" class="gt-vA"/>
  <!-- Side B vertices (bottom) -->
  <circle cx="100" cy="160" r="14" class="gt-vB"/>
  <circle cx="200" cy="160" r="14" class="gt-vB"/>
  <circle cx="300" cy="160" r="14" class="gt-vB"/>
  <!-- Labels -->
  <text x="60" y="50" class="gt-label">a</text>
  <text x="160" y="50" class="gt-label">b</text>
  <text x="260" y="50" class="gt-label">c</text>
  <text x="350" y="50" class="gt-label">d</text>
  <text x="100" y="160" class="gt-label">e</text>
  <text x="200" y="160" class="gt-label">f</text>
  <text x="300" y="160" class="gt-label">g</text>
</svg>

*A bipartite graph with its two sides colored. Every edge crosses between the sides.*

### The Characterization Theorem

The following theorem gives three equivalent ways to think about bipartiteness:

**Theorem (Bipartite Characterization).** The following are equivalent for a graph $G$:

1. $G$ is bipartite.
2. $G$ has no closed walks of odd length.
3. $G$ has no cycles of odd length.

*Proof.* We prove $1 \Rightarrow 2 \Rightarrow 3 \Rightarrow 1$ (showing that if any one fails, the previous one does too).

**$(1 \Rightarrow 2)$:** Suppose $G$ has a bipartition $(A, B)$ and a closed walk $(v_0, v_1, \ldots, v_{2k+1})$ with $v_0 = v_{2k+1}$. Without loss of generality, $v_0 \in A$. Then $v_1 \in B$ (since $v_0 v_1$ is an edge), $v_2 \in A$, and so on: $v_i \in A$ when $i$ is even, $v_i \in B$ when $i$ is odd. Therefore $v_{2k+1} \in B$. But $v_{2k+1} = v_0 \in A$ — contradiction.

**$(2 \Rightarrow 3)$:** Every cycle is a closed walk, so this direction is immediate.

**$(3 \Rightarrow 1)$:** Suppose $G$ has no odd cycles. We show $G$ is bipartite by running the BFS-coloring algorithm. If the algorithm fails, some edge $xy$ has both endpoints on the same side, meaning $d(v, x)$ and $d(v, y)$ have the same parity. Consider the closed walk: follow a shortest path from $v$ to $x$, take edge $xy$, then reverse a shortest path from $v$ to $y$. This has odd total length $d(v, x) + 1 + d(v, y)$.

Now use the extremal principle: take the *shortest* closed walk of odd length. Its length is at least 3 (since a vertex is never adjacent to itself). If it is not already a cycle, then some vertex repeats, splitting it into two shorter closed walks whose lengths sum to the original odd length. One of them must also have odd length — contradicting minimality. So the shortest odd closed walk is an odd cycle, contradicting our assumption. $\square$

> The power of this theorem lies in what it gives us in each direction. Assuming bipartiteness? Use the bipartition. Assuming *not* bipartite? Use the odd cycle. Proving not bipartite? Construct an odd closed walk (easier than finding a cycle).

---

## Vertex Degree and the Handshake Lemma

**Definition.** The **degree** of a vertex $v$, written $\deg(v)$, is the number of edges incident on $v$. A vertex of degree 0 is **isolated**; a vertex of degree 1 is a **leaf**. The **maximum degree** is $\Delta(G)$ and the **minimum degree** is $\delta(G)$.

The most fundamental relationship between vertex degrees and edge counts is:

**Theorem (Handshake Lemma).** In any graph $G$,

$$
\sum_{v \in V(G)} \deg(v) = 2|E(G)|.
$$

*Proof.* By induction on the number of edges $m$. When $m = 0$, all degrees are 0 and both sides equal 0. For the inductive step, let $G$ have $m \geq 1$ edges, and let $xy$ be any edge. In $G - xy$, the degrees of $x$ and $y$ each decrease by 1; all other degrees are unchanged. By the inductive hypothesis, the degree sum in $G - xy$ is $2(m - 1)$. So the degree sum in $G$ is $2(m - 1) + 2 = 2m$. $\square$

The name comes from a party analogy: if everyone at a party counts how many handshakes they participated in, the total count is exactly twice the number of handshakes (because each handshake is counted by two people).

**Corollary.** Every graph has an even number of odd-degree vertices.

*Proof.* The degree sum is even (it equals $2|E|$). The even-degree vertices contribute an even amount. So the odd-degree vertices must also contribute an even amount, which requires an even number of them. $\square$

### Minimum Degree and Cycles

**Theorem.** If $\delta(G) \geq 2$ (every vertex has degree at least 2), then $G$ contains a cycle.

*Proof.* Let $(v_1, v_2, \ldots, v_k)$ be a longest path in $G$. Since $\deg(v_1) \geq 2$, vertex $v_1$ has at least two neighbors. Can $v_1$ have a neighbor $w$ outside the path? No — then $(w, v_1, v_2, \ldots, v_k)$ would be a longer path. So all of $v_1$'s neighbors lie among $v_2, v_3, \ldots, v_k$.

One neighbor is $v_2$ (the next vertex on the path). There must be another, say $v_i$ with $i > 2$. Then $(v_1, v_2, \ldots, v_i, v_1)$ is a cycle. $\square$

### Average Degree and Subgraphs

Even graphs with many edges can have minimum degree 0 (just add an isolated vertex to $K_{99}$). But high *average* degree still forces interesting structure:

**Theorem.** If $G$ has average degree at least $d$, then $G$ contains a subgraph $H$ with $\delta(H) > \frac{d}{2}$.

The proof repeatedly deletes vertices of low degree. Each deletion preserves or raises the average degree. Eventually, all remaining vertices have degree above $\frac{d}{2}$.

**Corollary.** If $G$ has $n$ vertices and at least $n$ edges, then $G$ contains a cycle.

*Proof.* Average degree is at least 2. By the theorem above, $G$ has a subgraph with minimum degree at least 2. By the previous theorem, that subgraph contains a cycle — which is also a cycle in $G$. $\square$

---

We now have a vocabulary for graphs — vertices, edges, walks, paths, components, degrees — and a toolkit of first results. We know how to recognize bipartite graphs, count edges from degrees, and force cycles from minimum degree conditions. In [Part 2](/2024/07/10/graph-theory-degree-sequences-and-trees.html), we ask deeper structural questions: what numerical signatures capture a graph's shape, and what are the simplest possible connected graphs?
