---
layout: post
title: "Graph Theory III: Matchings and Covers"
date: 2024-07-17
tags: [Math, Graph Theory, Combinatorics]
---

Imagine you are scheduling courses for a math department. You have a list of professors and a list of courses, and each professor has preferences about what they are willing to teach. Can you assign every professor exactly one course and every course exactly one professor? This is the **bipartite matching problem**, and it turns out to have a beautiful theory with clean necessary and sufficient conditions.

This is Part 3 of a four-part series. In [Part 1](/2024/07/03/graph-theory-foundations.html) we built the foundations of graph theory, and in [Part 2](/2024/07/10/graph-theory-degree-sequences-and-trees.html) we studied degree sequences and trees. Here we develop the theory of matchings in bipartite graphs, culminating in two landmark theorems: König's theorem and Hall's marriage theorem.

## What This Post Covers

- **The Matching Problem** — Matchings, maximum vs. maximal, and vertex covers
- **Augmenting Paths** — The key technique for improving matchings
- **König's Theorem** — Maximum matching equals minimum vertex cover in bipartite graphs
- **Hall's Marriage Theorem** — When a perfect matching exists
- **Perfect Matchings in Regular Bipartite Graphs** — Regularity guarantees a perfect matching

---

## The Matching Problem

A **matching** $M$ in a graph $G$ is a set of edges such that no two edges in $M$ share an endpoint. Equivalently, it is a subgraph where every vertex has degree at most 1.

A vertex is **covered** by $M$ if it is an endpoint of some edge in $M$, and **uncovered** otherwise.

<svg viewBox="0 0 380 180" style="max-width:400px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; opacity: 0.3; }
    .gt-matched { stroke: var(--primary, #94452b); stroke-width: 3; opacity: 1; }
    .gt-covered { fill: rgba(148, 69, 43, 0.2); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-uncovered { fill: var(--surface, #fcf9f4); stroke: var(--text-primary, #1a1a1a); stroke-width: 2; opacity: 0.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
    .gt-side { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-secondary, #666); text-anchor: middle; font-style: italic; }
  </style>
  <!-- Non-matching edges (faded) -->
  <line x1="50" y1="40" x2="100" y2="140" class="gt-edge"/>
  <line x1="50" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="100" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="280" y2="140" class="gt-edge"/>
  <line x1="230" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="230" y1="40" x2="280" y2="140" class="gt-edge"/>
  <line x1="320" y1="40" x2="280" y2="140" class="gt-edge"/>
  <!-- Matching edges (bold) -->
  <line x1="50" y1="40" x2="100" y2="140" class="gt-matched"/>
  <line x1="140" y1="40" x2="190" y2="140" class="gt-matched"/>
  <line x1="230" y1="40" x2="280" y2="140" class="gt-matched"/>
  <!-- Covered vertices -->
  <circle cx="50" cy="40" r="14" class="gt-covered"/>
  <circle cx="140" cy="40" r="14" class="gt-covered"/>
  <circle cx="230" cy="40" r="14" class="gt-covered"/>
  <circle cx="100" cy="140" r="14" class="gt-covered"/>
  <circle cx="190" cy="140" r="14" class="gt-covered"/>
  <circle cx="280" cy="140" r="14" class="gt-covered"/>
  <!-- Uncovered vertex -->
  <circle cx="320" cy="40" r="14" class="gt-uncovered"/>
  <!-- Labels -->
  <text x="50" y="40" class="gt-label">a₁</text>
  <text x="140" y="40" class="gt-label">a₂</text>
  <text x="230" y="40" class="gt-label">a₃</text>
  <text x="320" y="40" class="gt-label">a₄</text>
  <text x="100" y="140" class="gt-label">b₁</text>
  <text x="190" y="140" class="gt-label">b₂</text>
  <text x="280" y="140" class="gt-label">b₃</text>
  <text x="190" y="168" class="gt-side">A matching of size 3. Vertex a₄ is uncovered.</text>
</svg>

There are two natural optimization questions:

1. **Is there a perfect matching?** — one that covers every vertex.
2. **What is the maximum matching size?** — the largest $|M|$ over all matchings $M$.

### Maximum vs. Maximal

A matching is **maximum** if no larger matching exists. It is **maximal** if no edge can be added to it without violating the matching condition. Every maximum matching is maximal, but the converse fails: a maximal matching might be far from maximum.

This distinction matters because greedy algorithms naturally find maximal matchings, but we need maximum ones. The gap between the two is what makes the matching problem interesting.

### Vertex Covers

A **vertex cover** of $G$ is a set $U \subseteq V(G)$ such that every edge of $G$ has at least one endpoint in $U$. We write:

- $\alpha'(G)$ for the size of a maximum matching,
- $\beta(G)$ for the size of a minimum vertex cover.

**Claim.** For any matching $M$ and vertex cover $U$ in the same graph, $|M| \leq |U|$.

*Proof.* Every edge in $M$ must have at least one endpoint in $U$ (since $U$ covers all edges). No two edges of $M$ share an endpoint, so no vertex of $U$ can "account for" more than one edge of $M$. Therefore $|M| \leq |U|$. $\square$

This gives us $\alpha'(G) \leq \beta(G)$ for all graphs $G$. The remarkable fact about bipartite graphs is that equality holds.

---

## Augmenting Paths

The key to improving a suboptimal matching is the notion of an **augmenting path**.

**Definition.** Given a matching $M$ in a graph $G$, an **$M$-augmenting path** is a path $P$ in $G$ such that:

- $P$ begins and ends at vertices not covered by $M$,
- The edges of $P$ alternate between "not in $M$" and "in $M$."

An augmenting path has odd length $2k + 1$, with $k + 1$ unmatched edges and $k$ matched edges. If we **flip** the path — remove the matched edges from $M$ and add the unmatched ones — we get a new matching $M \triangle P$ with one more edge than $M$.

<svg viewBox="0 0 580 120" style="max-width:580px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-unmatched { stroke: var(--text-primary, #1a1a1a); stroke-width: 2; }
    .gt-matchededge { stroke: var(--primary, #94452b); stroke-width: 3; }
    .gt-covered { fill: rgba(148, 69, 43, 0.2); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-uncovered { fill: var(--surface, #fcf9f4); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; }
  </style>
  <!-- Before: augmenting path -->
  <text x="145" y="15" class="gt-label" font-weight="600">Before (augmenting path)</text>
  <line x1="20" y1="50" x2="70" y2="80" class="gt-unmatched"/>
  <line x1="70" y1="80" x2="120" y2="50" class="gt-matchededge"/>
  <line x1="120" y1="50" x2="170" y2="80" class="gt-unmatched"/>
  <line x1="170" y1="80" x2="220" y2="50" class="gt-matchededge"/>
  <line x1="220" y1="50" x2="270" y2="80" class="gt-unmatched"/>
  <circle cx="20" cy="50" r="8" class="gt-uncovered"/>
  <circle cx="70" cy="80" r="8" class="gt-covered"/>
  <circle cx="120" cy="50" r="8" class="gt-covered"/>
  <circle cx="170" cy="80" r="8" class="gt-covered"/>
  <circle cx="220" cy="50" r="8" class="gt-covered"/>
  <circle cx="270" cy="80" r="8" class="gt-uncovered"/>
  <!-- After: flipped -->
  <text x="435" y="15" class="gt-label" font-weight="600">After (flip)</text>
  <line x1="310" y1="50" x2="360" y2="80" class="gt-matchededge"/>
  <line x1="360" y1="80" x2="410" y2="50" class="gt-unmatched"/>
  <line x1="410" y1="50" x2="460" y2="80" class="gt-matchededge"/>
  <line x1="460" y1="80" x2="510" y2="50" class="gt-unmatched"/>
  <line x1="510" y1="50" x2="560" y2="80" class="gt-matchededge"/>
  <circle cx="310" cy="50" r="8" class="gt-covered"/>
  <circle cx="360" cy="80" r="8" class="gt-covered"/>
  <circle cx="410" cy="50" r="8" class="gt-covered"/>
  <circle cx="460" cy="80" r="8" class="gt-covered"/>
  <circle cx="510" cy="50" r="8" class="gt-covered"/>
  <circle cx="560" cy="80" r="8" class="gt-covered"/>
  <!-- Edge labels -->
  <text x="145" y="108" class="gt-label" style="font-size:10px;">3 unmatched, 2 matched</text>
  <text x="435" y="108" class="gt-label" style="font-size:10px;">3 matched, 2 unmatched</text>
</svg>

*An augmenting path with 5 edges: 3 unmatched (thin) and 2 matched (bold). After flipping, the matching gains one edge.*

The **symmetric difference** $M \triangle P$ swaps matched and unmatched edges along the path. The crucial observation is that this always produces a valid matching: the endpoints of the path gain coverage, interior vertices simply swap which edge covers them, and vertices outside the path are unaffected.

**Theorem.** If $M$ is a matching in any graph $G$, then either $M$ is a maximum matching, or $G$ has an $M$-augmenting path.

*Proof.* Suppose $M$ is not maximum: there exists a larger matching $N$. Consider the symmetric difference $M \triangle N$ as a subgraph. Every vertex has degree at most 2 in this subgraph (at most one edge from $M$, one from $N$). So the components are paths and even cycles. Since $|N| > |M|$, some component must have more $N$-edges than $M$-edges — and that component is an $M$-augmenting path. $\square$

---

## König's Theorem

**Theorem (König).** For any bipartite graph $G$, $\alpha'(G) = \beta(G)$.

In words: the maximum matching size equals the minimum vertex cover size. We already know $\alpha'(G) \leq \beta(G)$ for all graphs. The content of König's theorem is the reverse inequality, but only for bipartite graphs. (It fails for general graphs: $C_5$ has $\alpha'(C_5) = 2$ but $\beta(C_5) = 3$.)

*Proof.* Let $G$ have bipartition $(A, B)$ and let $M$ be any matching. We describe an algorithm that either finds an augmenting path (improving $M$) or constructs a vertex cover $U$ with $|U| = |M|$.

Partition $A$ into $A_0$ (uncovered by $M$) and $A_1$ (covered). Similarly, $B = B_0 \cup B_1$.

Starting from every vertex in $A_0$, explore the graph by alternating: follow unmatched edges from $A$ to $B$, then matched edges from $B$ back to $A$. Let $Z$ be the set of all vertices reachable from $A_0$ by such alternating walks.

**Case 1:** Some vertex in $B_0$ is in $Z$. Then the alternating walk from $A_0$ to this uncovered vertex in $B$ is an $M$-augmenting path. Replace $M$ by $M \triangle P$ and repeat.

**Case 2:** No vertex in $B_0$ is in $Z$. Define the vertex cover:

$$
U = (A_1 \setminus Z) \cup (B_1 \cap Z).
$$

We check that $U$ is a vertex cover: every edge of $G$ has at least one endpoint in $U$. Consider any edge $ab$ with $a \in A$, $b \in B$:

- If $a \in A_0$, then $a \in Z$, so $b \in Z$ (reachable via the unmatched edge $ab$). Since $b \in B_0 \cap Z$ would give us Case 1, we must have $b \in B_1 \cap Z \subseteq U$.
- If $a \in A_1 \setminus Z$, then $a \in U$.
- If $a \in A_1 \cap Z$, then $a$ was reached via a matched edge from some $b' \in B_1 \cap Z$. If $ab$ is unmatched, then $b$ is also reachable (through $a$), so $b \in Z$. If $b \in B_0$, that's Case 1. So $b \in B_1 \cap Z \subseteq U$. If $ab$ is matched, then $b = b' \in B_1 \cap Z \subseteq U$.

Next, $|U| = |M|$: each vertex in $A_1 \setminus Z$ is matched to some vertex in $B_1 \setminus Z$, and each vertex in $B_1 \cap Z$ is matched to some vertex in $A_1 \cap Z$. These are disjoint parts of $M$, and they account for all edges in $M$.

When the algorithm terminates (no more augmenting paths), we have a matching $M$ and a vertex cover $U$ with $|M| = |U|$. Since $\alpha'(G) \leq \beta(G) \leq |U| = |M| \leq \alpha'(G)$, equality holds everywhere. $\square$

<svg viewBox="0 0 380 180" style="max-width:400px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-edge { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; opacity: 0.3; }
    .gt-matched { stroke: var(--primary, #94452b); stroke-width: 3; opacity: 1; }
    .gt-incover { fill: rgba(148, 69, 43, 0.35); stroke: var(--primary, #94452b); stroke-width: 2.5; }
    .gt-notcover { fill: var(--surface, #fcf9f4); stroke: var(--text-primary, #1a1a1a); stroke-width: 1.5; }
    .gt-label { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); text-anchor: middle; dominant-baseline: central; }
  </style>
  <!-- All edges (faded) -->
  <line x1="50" y1="40" x2="80" y2="140" class="gt-edge"/>
  <line x1="50" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="80" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="140" y1="40" x2="300" y2="140" class="gt-edge"/>
  <line x1="230" y1="40" x2="190" y2="140" class="gt-edge"/>
  <line x1="230" y1="40" x2="300" y2="140" class="gt-edge"/>
  <line x1="320" y1="40" x2="300" y2="140" class="gt-edge"/>
  <!-- Matching -->
  <line x1="50" y1="40" x2="80" y2="140" class="gt-matched"/>
  <line x1="140" y1="40" x2="190" y2="140" class="gt-matched"/>
  <line x1="230" y1="40" x2="300" y2="140" class="gt-matched"/>
  <!-- Vertices: in cover (shaded) vs not -->
  <circle cx="50" cy="40" r="14" class="gt-notcover"/>
  <circle cx="140" cy="40" r="14" class="gt-incover"/>
  <circle cx="230" cy="40" r="14" class="gt-notcover"/>
  <circle cx="320" cy="40" r="14" class="gt-notcover"/>
  <circle cx="80" cy="140" r="14" class="gt-incover"/>
  <circle cx="190" cy="140" r="14" class="gt-notcover"/>
  <circle cx="300" cy="140" r="14" class="gt-incover"/>
  <!-- Labels -->
  <text x="50" y="40" class="gt-label">a₁</text>
  <text x="140" y="40" class="gt-label">a₂</text>
  <text x="230" y="40" class="gt-label">a₃</text>
  <text x="320" y="40" class="gt-label">a₄</text>
  <text x="80" y="140" class="gt-label">b₁</text>
  <text x="190" y="140" class="gt-label">b₂</text>
  <text x="300" y="140" class="gt-label">b₃</text>
  <text x="190" y="170" style="font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-secondary, #666); text-anchor: middle; font-style: italic;">König's theorem: matching size 3 = cover size 3</text>
</svg>

*A maximum matching (bold edges) and a minimum vertex cover (shaded vertices) of equal size. Every edge touches at least one shaded vertex.*

---

## Hall's Marriage Theorem

König's theorem tells us *how large* a maximum matching is. Hall's theorem tells us *when* a matching covers an entire side of a bipartite graph.

**Theorem (Hall).** A bipartite graph $G$ with bipartition $(A, B)$ has a matching that covers all of $A$ if and only if **Hall's condition** holds:

$$
\text{For all } S \subseteq A, \quad |N(S)| \geq |S|,
$$

where $N(S)$ is the set of all vertices in $B$ adjacent to at least one vertex in $S$.

Hall's condition says: there is no "bottleneck." No group of vertices on side $A$ shares too few neighbors on side $B$. If $|A| = |B|$, this becomes the condition for a **perfect matching**.

*Proof.* **Necessity.** If a matching covers all of $A$, then each vertex $u \in S$ is matched to a distinct vertex $v \in N(S)$. So $|N(S)| \geq |S|$.

**Sufficiency.** Suppose Hall's condition holds but no matching covers all of $A$. By König's theorem, there is a vertex cover $U$ with $|U| = \alpha'(G) < |A|$.

Let $S = A \setminus U$ — the vertices in $A$ not in the cover. Since $|U| < |A|$, we have $S \neq \emptyset$. For any $v \in S$ adjacent to $w$, the edge $vw$ must be covered by $U$; since $v \notin U$, we need $w \in U$. So $N(S) \subseteq U \cap B$.

Now $U$ contains at least $|A| - |S|$ vertices from $A$ (those not in $S$) and all of $N(S)$ from $B$. So:

$$
|U| \geq (|A| - |S|) + |N(S)| \geq (|A| - |S|) + |S| = |A|.
$$

But $|U| < |A|$ — contradiction. So a matching covering all of $A$ must exist. $\square$

> The marriage interpretation: if every group of $k$ people collectively know at least $k$ potential partners, then everyone can be paired up. Hall's theorem makes this precise.

---

## Perfect Matchings in Regular Bipartite Graphs

Hall's theorem has a beautiful corollary for regular graphs:

**Theorem.** If $G$ is an $r$-regular bipartite graph (every vertex has degree $r \geq 1$), then $G$ has a perfect matching.

*Proof.* First, $|A| = |B|$: the total degree from $A$ is $r|A|$ and from $B$ is $r|B|$, and they must be equal (both count the number of edges).

To apply Hall's theorem, take any $S \subseteq A$. There are $r|S|$ edges with one endpoint in $S$. Each vertex in $N(S)$ has degree $r$, so it accounts for at most $r$ of these edges. Therefore:

$$
|N(S)| \geq \frac{r|S|}{r} = |S|.
$$

Hall's condition is satisfied, so a matching covering all of $A$ exists. Since $|A| = |B|$, this is a perfect matching. $\square$

> This theorem generalizes to $(r, s)$-biregular graphs (degree $r$ on one side, $s$ on the other). By double counting, $r|A| = s|B|$, so $|A| \neq |B|$ when $r \neq s$. But there is always a matching covering the smaller side.

---

We have seen that bipartite matching theory is remarkably clean: the duality between matchings and vertex covers (König), the neighborhood condition for complete matchings (Hall), and the automatic perfection of regular bipartite graphs. In [Part 4](/2024/07/24/graph-theory-directed-graphs.html), we generalize our notion of graphs to allow directed edges and multiple edges, and discover Euler's elegant characterization of graphs where every edge can be traversed exactly once.
