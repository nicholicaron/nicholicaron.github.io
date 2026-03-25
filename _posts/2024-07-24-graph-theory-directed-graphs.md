---
layout: post
title: "Graph Theory IV: Directed Graphs and Eulerian Tours"
date: 2024-07-24
tags: [Math, Graph Theory, Combinatorics]
---

In 1736, Leonhard Euler asked a deceptively simple question about the city of Königsberg: can you walk through the city, crossing each of its seven bridges exactly once, and return to where you started? His negative answer — and the reasoning behind it — launched graph theory as a mathematical discipline. To fully appreciate Euler's theorem, we need to generalize our notion of graphs to allow multiple edges and directed edges. Along the way, we will discover a striking contrast: Eulerian tours (traversing every *edge* once) have a clean characterization, while Hamiltonian cycles (visiting every *vertex* once) are computationally intractable.

This is Part 4 of a four-part series. In [Part 1](/2024/07/03/graph-theory-foundations.html) we built foundations, in [Part 2](/2024/07/10/graph-theory-degree-sequences-and-trees.html) we studied trees, and in [Part 3](/2024/07/17/graph-theory-matchings.html) we developed matching theory. Here we extend the framework to multigraphs and digraphs, prove Euler's theorem, and explore tournaments.

## What This Post Covers

- **Multigraphs** — Loops, parallel edges, and why the Königsberg bridges demand them
- **Directed Graphs** — Arcs, indegree, outdegree, and directed walks
- **Eulerian Tours** — Euler's theorem: connected and all even degrees
- **Cycle Decompositions** — Every even-degree graph splits into edge-disjoint cycles
- **Hamiltonian Cycles** — The hard twin of Eulerian tours
- **Tournaments** — Directed complete graphs, score sequences, and Hamiltonian paths

---

## Multigraphs

Until now, our graphs have been *simple*: no vertex is adjacent to itself, and no two vertices are connected by more than one edge. But Euler's original problem requires a more flexible model.

The city of Königsberg (now Kaliningrad) had four landmasses connected by seven bridges. If we model each landmass as a vertex, we need *multiple edges* between some pairs of vertices to represent the multiple bridges.

<svg viewBox="0 0 360 220" style="max-width:360px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 2.5; fill: none; }
    .gt-vertex { fill: rgba(148, 69, 43, 0.15); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 14px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; font-weight: 600; }
    .gt-bridgelabel { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--text-secondary, #666); text-anchor: middle; }
  </style>
  <!-- Edges: A-B (two bridges) -->
  <path d="M 80,60 Q 130,30 180,60" class="gt-edge"/>
  <path d="M 80,60 Q 130,90 180,60" class="gt-edge"/>
  <!-- Edges: A-C (two bridges) -->
  <path d="M 80,60 Q 50,110 80,160" class="gt-edge"/>
  <path d="M 80,60 Q 110,110 80,160" class="gt-edge"/>
  <!-- Edge: A-D (one bridge) -->
  <line x1="80" y1="60" x2="280" y2="110" class="gt-edge"/>
  <!-- Edge: B-D (one bridge) -->
  <line x1="180" y1="60" x2="280" y2="110" class="gt-edge"/>
  <!-- Edge: C-D (one bridge) -->
  <line x1="80" y1="160" x2="280" y2="110" class="gt-edge"/>
  <!-- Vertices -->
  <circle cx="80" cy="60" r="20" class="gt-vertex"/>
  <circle cx="180" cy="60" r="20" class="gt-vertex"/>
  <circle cx="80" cy="160" r="20" class="gt-vertex"/>
  <circle cx="280" cy="110" r="20" class="gt-vertex"/>
  <!-- Labels -->
  <text x="80" y="60" class="gt-label">A</text>
  <text x="180" y="60" class="gt-label">B</text>
  <text x="80" y="160" class="gt-label">C</text>
  <text x="280" y="110" class="gt-label">D</text>
  <!-- Degree annotations -->
  <text x="180" y="200" class="gt-bridgelabel">deg(A)=5, deg(B)=3, deg(C)=3, deg(D)=3</text>
  <text x="180" y="215" class="gt-bridgelabel" style="font-style: italic;">All degrees odd — no Eulerian tour exists</text>
</svg>

*The Königsberg bridge problem as a multigraph. Four landmasses (vertices), seven bridges (edges). Every vertex has odd degree.*

A **multigraph** allows:
- **Loops**: edges that join a vertex to itself.
- **Parallel edges**: two or more edges with the same endpoints.

Most of what we have learned still applies. The Handshake Lemma still holds (a loop at $v$ contributes 2 to $\deg(v)$). Trees are always simple. Matchings apply unchanged (bipartite multigraphs can have parallel edges but not loops).

---

## Directed Graphs

A **directed graph** (or **digraph**) $D$ has a vertex set $V(D)$ and a set of **arcs** (directed edges) $E(D)$, where each arc is an ordered pair $(v, w)$ going *from* $v$ *to* $w$. Arcs are drawn as arrows.

In a digraph, each vertex $v$ has two degree measures:
- **Indegree** $\deg^-(v)$: the number of arcs pointing *into* $v$.
- **Outdegree** $\deg^+(v)$: the number of arcs pointing *out of* $v$.

**Directed Handshake Lemma.** In any digraph $D$:

$$
\sum_{v \in V(D)} \deg^-(v) = \sum_{v \in V(D)} \deg^+(v) = |E(D)|.
$$

Each arc $(v, w)$ contributes 1 to $\deg^+(v)$ and 1 to $\deg^-(w)$.

A **walk** in a digraph must follow the arrows: $(v_0, v_1, \ldots, v_\ell)$ where each $(v_i, v_{i+1})$ is an arc. Distance becomes asymmetric: $d(u, v) \neq d(v, u)$ in general.

A digraph is **strongly connected** if for every pair of vertices $u, v$, there is a directed $u$–$v$ walk *and* a directed $v$–$u$ walk. It is **weakly connected** if the underlying undirected graph is connected.

---

## Eulerian Tours

An **Eulerian tour** in a (multi)graph $G$ is a closed walk that uses every edge of $G$ exactly once. A graph is **Eulerian** if it has an Eulerian tour.

### The Necessary Condition

**Lemma.** If a multigraph $G$ has an Eulerian tour, then every vertex of $G$ has even degree.

*Proof.* Every time the tour enters a vertex $v$, it must also leave $v$, using two edges incident to $v$. After the tour uses all edges, $v$ must have been entered and exited the same number of times, say $k$. This accounts for $2k$ edges at $v$, so $\deg(v) = 2k$ is even. $\square$

This is why the Königsberg bridge problem has no solution: all four vertices have odd degree.

### Cycle Decompositions

To prove the converse, we need a stepping stone: graphs with all even degrees can be decomposed into cycles.

**Lemma.** A multigraph $G$ has a **cycle decomposition** (a partition of its edges into edge-disjoint cycles) if and only if every vertex has even degree.

*Proof.* **Only if:** Each cycle contributes 2 to the degree of every vertex it passes through, so degrees are even.

**If:** By strong induction on the number of edges. If $G$ has no edges, the empty set is a cycle decomposition. Otherwise, pick a component with at least one edge. Every vertex in this component has even degree $\geq 2$, so minimum degree is $\geq 2$, which guarantees a cycle $C$.

In $G - C$ (delete the edges of $C$, keep vertices), all degrees are still even: the degree of each vertex on $C$ decreased by exactly 2. Since $G - C$ has fewer edges, it has a cycle decomposition by the inductive hypothesis. Adding $C$ gives a cycle decomposition of $G$. $\square$

### The Main Theorem

**Theorem (Euler).** A connected multigraph $G$ is Eulerian if and only if every vertex has even degree.

*Proof.* We proved necessity above. For sufficiency, assume $G$ is connected and every vertex has even degree. By the cycle decomposition lemma, $G$ has a decomposition into cycles $C_1, C_2, \ldots, C_k$.

We build the Eulerian tour by **gluing cycles together**. Start with a "partial tour" $PT = C_1$. Then repeatedly:

1. Find a cycle $C_i$ in the decomposition that shares a vertex $v$ with $PT$.
2. **Splice** $C_i$ into $PT$ at $v$: when $PT$ visits $v$, detour through all of $C_i$ before continuing.

Such a cycle must always exist (otherwise $PT$ and the remaining cycles would form disconnected parts of $G$, contradicting connectivity). After splicing all cycles, $PT$ traverses every edge exactly once. $\square$

<svg viewBox="0 0 400 130" style="max-width:400px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-c1 { stroke: var(--primary, #94452b); stroke-width: 2.5; fill: none; }
    .gt-c2 { stroke: steelblue; stroke-width: 2.5; fill: none; }
    .gt-c3 { stroke: #2d8a4e; stroke-width: 2.5; fill: none; }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--text-primary, #1a1a1a); stroke-width: 2; }
    .gt-shared { fill: rgba(148, 69, 43, 0.25); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- Cycle 1 (red/primary) -->
  <path d="M 60,30 L 140,30 L 140,100 L 60,100 Z" class="gt-c1"/>
  <!-- Cycle 2 (blue) -->
  <path d="M 140,30 L 250,30 L 250,100 L 140,100 Z" class="gt-c2"/>
  <!-- Cycle 3 (green) -->
  <path d="M 250,30 L 340,30 L 340,100 L 250,100 Z" class="gt-c3"/>
  <!-- Vertices -->
  <circle cx="60" cy="30" r="8" class="gt-vertex"/>
  <circle cx="60" cy="100" r="8" class="gt-vertex"/>
  <circle cx="140" cy="30" r="8" class="gt-shared"/>
  <circle cx="140" cy="100" r="8" class="gt-shared"/>
  <circle cx="250" cy="30" r="8" class="gt-shared"/>
  <circle cx="250" cy="100" r="8" class="gt-shared"/>
  <circle cx="340" cy="30" r="8" class="gt-vertex"/>
  <circle cx="340" cy="100" r="8" class="gt-vertex"/>
  <!-- Legend -->
  <line x1="50" y1="122" x2="70" y2="122" class="gt-c1"/>
  <text x="90" y="124" class="gt-label">C₁</text>
  <line x1="120" y1="122" x2="140" y2="122" class="gt-c2"/>
  <text x="160" y="124" class="gt-label">C₂</text>
  <line x1="190" y1="122" x2="210" y2="122" class="gt-c3"/>
  <text x="230" y="124" class="gt-label">C₃</text>
  <text x="320" y="124" class="gt-label" style="font-style: italic;">splice at shared vertices</text>
</svg>

*Three cycles sharing vertices (highlighted). The Eulerian tour is built by splicing: traverse $C_1$, detour into $C_2$ at a shared vertex, detour into $C_3$, then finish.*

### Eulerian Paths

What if we only want to traverse every edge once but don't need to return to the start? An **Eulerian path** (a non-closed walk using every edge once) exists if and only if the graph is connected and has exactly 0 or 2 vertices of odd degree. With 0 odd-degree vertices, we get a closed tour. With 2, the path must start and end at the two odd-degree vertices.

### Directed Eulerian Tours

The theorem generalizes naturally to directed graphs. A **directed Eulerian tour** in a digraph $D$ is a closed directed walk that traverses every arc exactly once.

**Theorem (Directed Euler).** A digraph $D$ has a directed Eulerian tour if and only if $D$ is (weakly) connected and $\deg^+(v) = \deg^-(v)$ for every vertex $v$.

The condition $\deg^+(v) = \deg^-(v)$ is the directed analogue of "all degrees are even" — every time the tour enters a vertex, it must leave along a different arc.

---

## Hamiltonian Cycles

An **Eulerian tour** visits every *edge* once. A **Hamiltonian cycle** visits every *vertex* once (and returns to the start). Despite the similar-sounding definitions, the computational landscape is completely different.

For Eulerian tours, we have a complete characterization (connected + all even degrees) and an efficient algorithm. For Hamiltonian cycles, no simple necessary-and-sufficient condition is known, and the decision problem is NP-complete — one of the hardest problems in computer science.

We do have *sufficient* conditions. The most classical is:

**Theorem (Dirac, 1952).** If $G$ has $n \geq 3$ vertices and $\delta(G) \geq \frac{n}{2}$, then $G$ has a Hamiltonian cycle.

The intuition: if every vertex has at least $n/2$ neighbors, the graph is "dense enough" that you can always extend a partial path into a full cycle. But the proof is more involved than Euler's theorem, and the condition is far from necessary — many sparse graphs are Hamiltonian.

> The contrast between Eulerian and Hamiltonian problems is one of the great lessons of combinatorics. Small changes in a problem statement — edges vs. vertices — can shift the difficulty from polynomial time to (presumably) exponential.

---

## Tournaments

A **tournament** is a special kind of directed graph: for every pair of vertices $v$ and $w$, exactly one of the arcs $(v, w)$ or $(w, v)$ exists. Think of it as a round-robin competition where every pair of players meets exactly once, and one of them wins.

<svg viewBox="0 0 300 260" style="max-width:280px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="gt-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 Z" fill="var(--text-primary, #1a1a1a)"/>
    </marker>
  </defs>
  <style>
    .gt-arc { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.8; fill: none; marker-end: url(#gt-arrow); }
    .gt-vertex { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 14px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; font-weight: 600; }
    .gt-score { font-family: 'JetBrains Mono', monospace; font-size: 10px; fill: var(--text-secondary, #666); text-anchor: middle; }
  </style>
  <!-- Arcs (carefully positioned to not overlap vertices) -->
  <!-- 1→2 -->
  <line x1="160" y1="37" x2="246" y2="96" class="gt-arc"/>
  <!-- 1→3 -->
  <line x1="143" y1="45" x2="90" y2="98" class="gt-arc"/>
  <!-- 2→5 -->
  <line x1="258" y1="122" x2="200" y2="210" class="gt-arc"/>
  <!-- 3→2 -->
  <line x1="82" y1="108" x2="245" y2="108" class="gt-arc"/>
  <!-- 4→1 -->
  <line x1="82" y1="200" x2="138" y2="44" class="gt-arc"/>
  <!-- 4→3 -->
  <line x1="68" y1="205" x2="68" y2="130" class="gt-arc"/>
  <!-- 5→3 -->
  <line x1="180" y1="218" x2="80" y2="125" class="gt-arc"/>
  <!-- 5→4 -->
  <line x1="185" y1="225" x2="85" y2="218" class="gt-arc"/>
  <!-- 1→5 -->
  <line x1="155" y1="42" x2="192" y2="206" class="gt-arc"/>
  <!-- 2→4 -->
  <line x1="250" y1="120" x2="86" y2="208" class="gt-arc"/>
  <!-- Vertices (drawn on top) -->
  <circle cx="150" cy="30" r="16" class="gt-vertex"/>
  <circle cx="260" cy="110" r="16" class="gt-vertex"/>
  <circle cx="68" cy="110" r="16" class="gt-vertex"/>
  <circle cx="68" cy="218" r="16" class="gt-vertex"/>
  <circle cx="195" cy="218" r="16" class="gt-vertex"/>
  <!-- Labels -->
  <text x="150" y="30" class="gt-label">1</text>
  <text x="260" y="110" class="gt-label">2</text>
  <text x="68" y="110" class="gt-label">3</text>
  <text x="68" y="218" class="gt-label">4</text>
  <text x="195" y="218" class="gt-label">5</text>
  <!-- Scores -->
  <text x="150" y="250" class="gt-score">Scores: 1→3, 2→2, 3→1, 4→2, 5→2</text>
</svg>

*A tournament on 5 vertices. Each pair has exactly one arc. Vertex 1 beats three others (outdegree 3).*

### Transitive Tournaments

An **acyclic** tournament is called **transitive**: if $u$ beats $v$ and $v$ beats $w$, then $u$ beats $w$. Transitive tournaments have a clear "ranking" — a topological ordering where all arcs go from earlier to later vertices.

**Theorem.** Up to isomorphism, there is only one acyclic tournament on $n$ vertices: the transitive tournament.

*Proof.* An acyclic digraph has a topological ordering: vertices $v_1, v_2, \ldots, v_n$ such that all arcs go from $v_i$ to $v_j$ with $i < j$. In a tournament, *every* pair has an arc, so all $\binom{n}{2}$ arcs of the form $(v_i, v_j)$ with $i < j$ must be present. This determines the tournament uniquely. $\square$

### Score Sequences

In a tournament, the **score** of a vertex is its outdegree — the number of opponents it beats. The **score sequence** is the list of all scores in ascending order $p_1 \leq p_2 \leq \cdots \leq p_n$.

The sum of all scores is $\binom{n}{2}$ (each arc contributes 1 to exactly one score). The score sequence of the transitive tournament is $0, 1, 2, \ldots, n-1$.

Not every sequence with the right sum is a valid score sequence. The necessary and sufficient conditions are given by **Landau's theorem**: an ascending sequence $p_1 \leq \cdots \leq p_n$ is a score sequence if and only if

$$
p_1 + p_2 + \cdots + p_k \geq \binom{k}{2} \quad \text{for } k = 1, \ldots, n-1,
$$

with equality when $k = n$.

### Hamiltonian Paths in Tournaments

Hamiltonian cycles are hard to find in general graphs. But in tournaments, even the seemingly weaker structure of a *Hamiltonian path* is guaranteed:

**Theorem.** Every tournament has a Hamiltonian path.

*Proof.* Let $v_1, v_2, \ldots, v_n$ be an ordering of the vertices that maximizes the number of **forward arcs** $(v_i, v_j)$ with $i < j$.

We claim that for every $i$, the arc between $v_i$ and $v_{i+1}$ is forward: $(v_i, v_{i+1})$. If instead we had the backward arc $(v_{i+1}, v_i)$, we could swap $v_i$ and $v_{i+1}$ in the ordering. The arc $(v_{i+1}, v_i)$ was backward and becomes forward; no other arc changes direction. This increases the number of forward arcs, contradicting maximality.

Therefore $(v_1, v_2, \ldots, v_n)$ is a Hamiltonian path: consecutive vertices are connected by forward arcs. $\square$

> Compare this to the Hamiltonian cycle problem in undirected graphs: merely *deciding* whether a Hamiltonian path exists is NP-complete in general, but in tournaments, one always exists, and the proof is constructive!

---

## Coda

We have traveled from the most basic definition — dots connected by lines — through the rich structure of trees, the duality of matchings and covers, and the elegance of Euler's theorem. Along the way, a few themes emerged:

**Characterization theorems** are the crown jewels: bipartite iff no odd cycle, tree iff connected and acyclic, Eulerian iff connected and all even degrees. These are the results that turn fuzzy intuition into precise, checkable conditions.

**Easy versus hard.** The gap between Eulerian tours and Hamiltonian cycles — between checking edges and checking vertices — reminds us that similar-sounding problems can have wildly different computational complexities.

**Duality.** König's theorem reveals that matchings (sets of independent edges) and vertex covers (sets that touch every edge) are two sides of the same coin in bipartite graphs. This min-max duality pattern recurs throughout combinatorial optimization.

Graph theory extends far beyond what we have covered here. **Planar graphs** and Euler's formula for faces, edges, and vertices. **Graph coloring** and the Four Color Theorem. **Ramsey theory** and the inevitability of structure in large graphs. **Spectral methods** that analyze graphs through the eigenvalues of their adjacency matrices. Each of these topics deserves its own series — and together, they make graph theory one of the richest and most applicable branches of discrete mathematics.
