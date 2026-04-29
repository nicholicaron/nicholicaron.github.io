---
layout: post
title: "Branch-and-Bound from First Principles"
date: 2026-04-28
tags: [Optimization, Combinatorics, AI, MILP]
cover_image: /assets/images/chop/peutinger-table-map-1619.jpg
---

In the late 1950s, two RAND researchers solved the largest Traveling Salesman Problem ever attempted: 49 cities, optimally. The number of possible tours through 49 cities is around $10^{60}$ — more than the atoms in your body — so they did not check them all. They invented a way to check almost none of them, and prove they had the answer anyway. The technique was *branch-and-bound*, and seventy years later it is still how every commercial Mixed-Integer Linear Program solver — Gurobi, CPLEX, FICO Xpress, the multi-billion-dollar industry that powers airline scheduling, logistics planning, and supply chain optimization — finds the optimal solution.

This post is the foundation for [a research diary I wrote about CHOP]({{ site.url }}/2026/04/28/chop-a-research-diary-in-learning-to-branch.html), an attempt to use deep reinforcement learning to make branch-and-bound smarter. To enjoy that post, you need to know what branch-and-bound is and why a smarter version of it is even worth chasing. So we'll get there together.

## What This Post Covers

- **Why optimization is the secret backbone of modern industry** — and why the math behind it is surprisingly accessible
- **Linear programs vs integer programs** — why one is easy and the other is the rest of the iceberg
- **The branch-and-bound algorithm** — step-by-step on a tiny example with diagrams
- **Heuristics: the human decisions inside the algorithm** — node selection, variable selection, and why they matter
- **When the strongest classical heuristic isn't actually best** — the perverse instances where folklore fails

This post assumes nothing. If you've never seen a linear program, you'll leave knowing what one is. If you have, skim the first few sections.

---

## The Quiet Empire of Optimization

Most of the moments your day works without friction were arranged by an optimization solver. Your morning flight got a gate, a crew, and a tail number from one solver. The package you ordered yesterday took a route through five sortation hubs picked by another. The wind farm contract your utility signed last quarter exists because a third decided which turbines to build, in what order, and where to draw the lines for transmission.

These problems all have the same shape: *make a discrete sequence of decisions to maximize value or minimize cost, subject to constraints*. They're called **combinatorial optimization** problems because the answer is some combination of choices. They are also, for reasons that will become clear, *brutally hard*. The decision variables are usually binary — assign this driver to that route, or don't; build this turbine, or don't — and the number of possible combinations explodes faster than physics will let you check them.

When the constraints and objective are linear, the problem is called a **Mixed-Integer Linear Program** (MILP). MILPs are the workhorse formulation: expressive enough to capture most of industrial planning, restrictive enough that we have algorithms that actually solve them. *Branch-and-bound* is that algorithm.

To appreciate it, we need to start one rung lower.

---

## Linear Programs: The Easy Case

A **linear program** (LP) asks for the largest value of a linear objective $\mathbf{c}^\top \mathbf{x}$ subject to linear constraints $\mathbf{A}\mathbf{x} \le \mathbf{b}$ with continuous variables $\mathbf{x} \ge \mathbf{0}$. Geometrically, the constraints carve out a convex region in $\mathbb{R}^n$ — think of it as a many-faced gem — and the objective points in some direction. The optimum sits at one of the gem's corners.

Here's a two-variable LP: maximize $3x + 4y$ subject to $x + 2y \le 10$, $3x + y \le 15$, with $x, y \ge 0$:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 460 360" width="460" height="360" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <pattern id="lp-fill" patternUnits="userSpaceOnUse" width="6" height="6">
      <path d="M 0 0 L 6 6" stroke="var(--primary, #94452b)" stroke-width="0.4" opacity="0.4"/>
    </pattern>
    <marker id="lp-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="currentColor"/>
    </marker>
  </defs>

  <!-- Axes -->
  <line x1="60" y1="320" x2="420" y2="320" stroke="currentColor" stroke-width="1.2" marker-end="url(#lp-arrow)"/>
  <line x1="60" y1="320" x2="60" y2="40" stroke="currentColor" stroke-width="1.2" marker-end="url(#lp-arrow)"/>

  <!-- Tick marks (every 1 unit, scale = 30 px) -->
  <g stroke="currentColor" stroke-width="0.6" opacity="0.4">
    <line x1="60" y1="320" x2="60" y2="324"/>
    <line x1="90" y1="320" x2="90" y2="324"/>
    <line x1="120" y1="320" x2="120" y2="324"/>
    <line x1="150" y1="320" x2="150" y2="324"/>
    <line x1="180" y1="320" x2="180" y2="324"/>
    <line x1="210" y1="320" x2="210" y2="324"/>
    <line x1="240" y1="320" x2="240" y2="324"/>
    <line x1="270" y1="320" x2="270" y2="324"/>
    <line x1="60" y1="290" x2="56" y2="290"/>
    <line x1="60" y1="260" x2="56" y2="260"/>
    <line x1="60" y1="230" x2="56" y2="230"/>
    <line x1="60" y1="200" x2="56" y2="200"/>
    <line x1="60" y1="170" x2="56" y2="170"/>
    <line x1="60" y1="140" x2="56" y2="140"/>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.7">
    <text x="60" y="338" text-anchor="middle">0</text>
    <text x="120" y="338" text-anchor="middle">2</text>
    <text x="180" y="338" text-anchor="middle">4</text>
    <text x="240" y="338" text-anchor="middle">6</text>
    <text x="48" y="293" text-anchor="end">2</text>
    <text x="48" y="233" text-anchor="end">4</text>
    <text x="48" y="173" text-anchor="end">6</text>
  </g>
  <text x="430" y="324" font-size="13" fill="currentColor" font-style="italic">x</text>
  <text x="60" y="32" font-size="13" fill="currentColor" font-style="italic" text-anchor="middle">y</text>

  <!-- Feasible region polygon: (0,0) -> (5,0) -> (4,3) -> (0,5) -->
  <!-- Pixel coords: (60,320), (210,320), (180,230), (60,170) -->
  <polygon points="60,320 210,320 180,230 60,170" fill="url(#lp-fill)" stroke="var(--primary, #94452b)" stroke-width="1.5"/>

  <!-- Constraint lines (extended) -->
  <!-- 3x + y = 15 -> y = 15 - 3x. Endpoints: (0,15) -> pixel (60, -130 clamped); (5,0) -> (210, 320). Visible portion. -->
  <line x1="60" y1="-130" x2="210" y2="320" stroke="var(--primary-dim, #a35338)" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.6"/>
  <text x="225" y="245" font-size="10" fill="var(--primary-dim, #a35338)" font-style="italic">3x + y ≤ 15</text>

  <!-- x + 2y = 10 -> y = 5 - x/2. Endpoints: (0,5) -> (60, 170); (10,0) -> (360, 320) -->
  <line x1="60" y1="170" x2="360" y2="320" stroke="var(--primary-dim, #a35338)" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.6"/>
  <text x="290" y="290" font-size="10" fill="var(--primary-dim, #a35338)" font-style="italic">x + 2y ≤ 10</text>

  <!-- Optimum at (4,3): pixel (180,230) -->
  <circle cx="180" cy="230" r="6" fill="var(--primary, #94452b)" stroke="white" stroke-width="2"/>
  <text x="195" y="218" font-size="11" fill="var(--primary, #94452b)" font-weight="600">optimum (4, 3)</text>
  <text x="195" y="232" font-size="10" fill="currentColor" opacity="0.7">value = 24</text>

  <!-- Objective gradient arrow at corner -->
  <line x1="60" y1="290" x2="120" y2="218" stroke="currentColor" stroke-width="1.5" opacity="0.6" marker-end="url(#lp-arrow)"/>
  <text x="125" y="215" font-size="10" fill="currentColor" opacity="0.7" font-style="italic">∇ = (3, 4)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">A linear program with two constraints. The feasible region is the hatched quadrilateral; the optimum sits at the corner where both constraints bind.</p>

The **simplex algorithm**, invented in 1947, walks from corner to corner, always moving in the objective's direction, and proves it has reached the optimum when no neighboring corner is better. In practice it solves problems with millions of variables in seconds. LPs, in short, are *solved* — the polynomial-time interior-point methods that came later only made an already-fast algorithm faster.

The LP optimum to our example is exactly $(x, y) = (4, 3)$, with objective value $24$. Notice that both numbers happened to be integers. That is luck.

---

## Integer Programs: The Hard Case

Now add one constraint: $x$ and $y$ must be **integers**. The feasible region is no longer a continuous polygon — it's the *lattice points inside* the polygon, the dots:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 460 360" width="460" height="360" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <pattern id="ip-fill" patternUnits="userSpaceOnUse" width="6" height="6">
      <path d="M 0 0 L 6 6" stroke="var(--primary, #94452b)" stroke-width="0.3" opacity="0.2"/>
    </pattern>
    <marker id="ip-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="currentColor"/>
    </marker>
  </defs>

  <line x1="60" y1="320" x2="420" y2="320" stroke="currentColor" stroke-width="1.2" marker-end="url(#ip-arrow)"/>
  <line x1="60" y1="320" x2="60" y2="40" stroke="currentColor" stroke-width="1.2" marker-end="url(#ip-arrow)"/>
  <g font-size="10" fill="currentColor" opacity="0.7">
    <text x="60" y="338" text-anchor="middle">0</text>
    <text x="120" y="338" text-anchor="middle">2</text>
    <text x="180" y="338" text-anchor="middle">4</text>
    <text x="240" y="338" text-anchor="middle">6</text>
    <text x="48" y="293" text-anchor="end">2</text>
    <text x="48" y="233" text-anchor="end">4</text>
    <text x="48" y="173" text-anchor="end">6</text>
  </g>
  <text x="430" y="324" font-size="13" fill="currentColor" font-style="italic">x</text>
  <text x="60" y="32" font-size="13" fill="currentColor" font-style="italic" text-anchor="middle">y</text>

  <polygon points="60,320 210,320 180,230 60,170" fill="url(#ip-fill)" stroke="var(--primary, #94452b)" stroke-width="0.8" opacity="0.6"/>

  <!-- Integer points inside feasible region. Region: x>=0, y>=0, x+2y<=10, 3x+y<=15.
       Pixel (px,py) = (60 + 30*x, 320 - 30*y) -->
  <g fill="currentColor" opacity="0.4">
    <circle cx="60" cy="320" r="3"/>
    <circle cx="90" cy="320" r="3"/>
    <circle cx="120" cy="320" r="3"/>
    <circle cx="150" cy="320" r="3"/>
    <circle cx="180" cy="320" r="3"/>
    <circle cx="60" cy="290" r="3"/>
    <circle cx="90" cy="290" r="3"/>
    <circle cx="120" cy="290" r="3"/>
    <circle cx="150" cy="290" r="3"/>
    <circle cx="180" cy="290" r="3"/>
    <circle cx="60" cy="260" r="3"/>
    <circle cx="90" cy="260" r="3"/>
    <circle cx="120" cy="260" r="3"/>
    <circle cx="150" cy="260" r="3"/>
    <circle cx="60" cy="230" r="3"/>
    <circle cx="90" cy="230" r="3"/>
    <circle cx="120" cy="230" r="3"/>
    <circle cx="60" cy="200" r="3"/>
    <circle cx="90" cy="200" r="3"/>
    <circle cx="60" cy="170" r="3"/>
  </g>

  <!-- LP optimum (still continuous) -->
  <circle cx="180" cy="230" r="5" fill="none" stroke="var(--primary-dim, #a35338)" stroke-width="1.5" stroke-dasharray="2,2"/>
  <text x="195" y="232" font-size="9" fill="var(--primary-dim, #a35338)" font-style="italic">LP opt (4,3)</text>

  <!-- Integer optimum at (4,3): conveniently the same here -->
  <circle cx="180" cy="230" r="6" fill="var(--primary, #94452b)" stroke="white" stroke-width="2"/>
  <text x="195" y="218" font-size="11" fill="var(--primary, #94452b)" font-weight="600">integer opt</text>

  <!-- Annotation -->
  <text x="280" y="80" font-size="11" fill="currentColor" opacity="0.8" font-style="italic">For this LP, integer</text>
  <text x="280" y="95" font-size="11" fill="currentColor" opacity="0.8" font-style="italic">optimum and LP optimum</text>
  <text x="280" y="110" font-size="11" fill="currentColor" opacity="0.8" font-style="italic">happen to coincide.</text>
  <text x="280" y="135" font-size="11" fill="currentColor" opacity="0.8" font-style="italic">In general they don't.</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The same problem with the added constraint that x, y must be integers. The feasible region collapses to a finite set of lattice points.</p>

This particular instance is friendly: the LP optimum landed on a lattice point, so the integer optimum is the same. In general, the LP optimum is some real-valued $(x^\ast, y^\ast)$ with fractional coordinates, and the *true* integer optimum is some lattice point nearby — but possibly *far* from $(x^\ast, y^\ast)$, with a substantially worse objective value. The gap between the LP optimum and the integer optimum is the **integrality gap**, and bridging it is what makes integer programming hard.

How hard? **NP-hard**. There is no known polynomial-time algorithm for general MILP, and the consensus assumption (P ≠ NP) is that there isn't one. The number of integer points inside the feasible region grows combinatorially with the number of variables: a problem with $n$ binary variables has up to $2^n$ candidate solutions. At $n=50$ you've already exceeded the lifetime of the universe at one nanosecond per check.

So we don't check them all. We do something cleverer.

---

## Branch-and-Bound: Prove You Don't Have to Look

Branch-and-bound has two ideas, both from 1960 (the *Land and Doig* paper), and both deceptively simple.

**Idea 1 — Branching: split a hard problem into two easier ones.** Take the LP relaxation. If a variable $x_i$ comes back fractional — say $x_i = 2.7$ in the LP solution — then in any integer solution it must satisfy *either* $x_i \le 2$ *or* $x_i \ge 3$. Both can't be true; one must be. So the original problem decomposes into two sub-problems with the extra constraint added on each side. Recurse.

**Idea 2 — Bounding: prove a whole subtree is hopeless.** When you solve the LP relaxation of a sub-problem, you get an *upper bound* on what the integer solution in that subtree can possibly achieve. (LP is a relaxation of MILP, so its optimum is at least as good as the integer optimum.) Meanwhile, you have an *incumbent* — the best integer-feasible solution you've found so far. If the LP upper bound of a sub-problem is $\le$ the incumbent value, no integer solution in that subtree can beat the incumbent. **Prune the entire subtree without exploring it.**

Together: build a search tree, expand it, prune ruthlessly. You only explore the subtrees that *could* contain a better solution than what you already have.

Here is the algorithm running on a tiny problem — maximize $5x_1 + 4x_2$ subject to $x_1 + x_2 \le 5$, $10x_1 + 6x_2 \le 45$, with $x_1, x_2 \in \{0, 1, 2, ...\}$:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 700 500" width="700" height="500" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <marker id="bb-arrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="currentColor"/>
    </marker>
  </defs>

  <!-- Root node -->
  <g>
    <rect x="280" y="20" width="140" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="var(--primary, #94452b)" stroke-width="2"/>
    <text x="350" y="40" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">Root</text>
    <text x="350" y="55" text-anchor="middle" font-size="10" fill="currentColor">LP: x = (3.75, 1.25)</text>
    <text x="350" y="70" text-anchor="middle" font-size="10" font-weight="600" fill="var(--primary, #94452b)">obj = 23.75</text>
  </g>

  <!-- Edges from root -->
  <line x1="320" y1="80" x2="180" y2="130" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <line x1="380" y1="80" x2="520" y2="130" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <text x="220" y="105" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₁ ≤ 3</text>
  <text x="455" y="105" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₁ ≥ 4</text>

  <!-- Node A: x1 <= 3 -->
  <g>
    <rect x="100" y="135" width="160" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
    <text x="180" y="155" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">A: x₁ ≤ 3</text>
    <text x="180" y="170" text-anchor="middle" font-size="10" fill="currentColor">LP: x = (3, 2)</text>
    <text x="180" y="185" text-anchor="middle" font-size="10" font-weight="600" fill="#0a8a3f">obj = 23 ✓ integer</text>
  </g>

  <!-- Node B: x1 >= 4 -->
  <g>
    <rect x="440" y="135" width="160" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
    <text x="520" y="155" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">B: x₁ ≥ 4</text>
    <text x="520" y="170" text-anchor="middle" font-size="10" fill="currentColor">LP: x = (4, 0.83)</text>
    <text x="520" y="185" text-anchor="middle" font-size="10" font-weight="600" fill="var(--primary, #94452b)">obj = 23.33</text>
  </g>

  <!-- Edges from B (B is fractional, branch on x2) -->
  <line x1="490" y1="195" x2="380" y2="245" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <line x1="550" y1="195" x2="640" y2="245" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <text x="395" y="220" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₂ ≤ 0</text>
  <text x="595" y="220" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₂ ≥ 1</text>

  <!-- Node B-floor: x1 >= 4, x2 <= 0 -->
  <g>
    <rect x="280" y="250" width="200" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
    <text x="380" y="270" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">B₁: x₁ ≥ 4, x₂ = 0</text>
    <text x="380" y="285" text-anchor="middle" font-size="10" fill="currentColor">LP: x = (4.5, 0)</text>
    <text x="380" y="300" text-anchor="middle" font-size="10" font-weight="600" fill="var(--primary, #94452b)">obj = 22.5</text>
  </g>

  <!-- Node B-ceil: x1 >= 4, x2 >= 1 -- INFEASIBLE -->
  <g>
    <rect x="540" y="250" width="150" height="60" rx="8" fill="var(--surface-container-low, #f7f3ee)" stroke="currentColor" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.7"/>
    <text x="615" y="270" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor" opacity="0.8">B₂: x₁ ≥ 4, x₂ ≥ 1</text>
    <text x="615" y="288" text-anchor="middle" font-size="10" fill="#a64542">infeasible</text>
    <text x="615" y="302" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">10·4 + 6·1 = 46 &gt; 45</text>
  </g>

  <!-- B1 children: x1 = 4 vs x1 >= 5 -->
  <line x1="350" y1="310" x2="280" y2="360" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <line x1="410" y1="310" x2="480" y2="360" stroke="currentColor" stroke-width="1.2" marker-end="url(#bb-arrow)"/>
  <text x="290" y="335" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₁ = 4</text>
  <text x="455" y="335" font-size="10" fill="currentColor" font-style="italic" opacity="0.75">x₁ ≥ 5</text>

  <!-- Final integer-feasible candidate at x = (4, 0) -->
  <g>
    <rect x="180" y="365" width="200" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
    <text x="280" y="385" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">B₁₁: x = (4, 0)</text>
    <text x="280" y="400" text-anchor="middle" font-size="10" fill="currentColor">obj = 20</text>
    <text x="280" y="415" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">≤ incumbent (23) → prune</text>
  </g>

  <!-- B12: x1 = 5, x2 = 0 -->
  <g>
    <rect x="400" y="365" width="200" height="60" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
    <text x="500" y="385" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">B₁₂: x = (4.5, 0) → (5,0)</text>
    <text x="500" y="400" text-anchor="middle" font-size="10" fill="currentColor">obj = 25 — better!</text>
    <text x="500" y="415" text-anchor="middle" font-size="10" fill="#0a8a3f">but check feasibility...</text>
  </g>

  <!-- Hint text near root -->
  <text x="50" y="40" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">Step 1:</text>
  <text x="50" y="55" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">solve LP at root.</text>
  <text x="50" y="70" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">Variables fractional →</text>
  <text x="50" y="85" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">branch on x₁.</text>

  <text x="50" y="465" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">Each box is a sub-problem with extra constraints.</text>
  <text x="50" y="480" font-size="10" fill="currentColor" opacity="0.65" font-style="italic">Dashed = infeasible. The optimum is the best integer-feasible leaf.</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">A small branch-and-bound tree. Each node is a sub-problem; we solve its LP relaxation to get a bound; if it's fractional, we branch.</p>

A few things to notice:

1. **The LP relaxation is doing the heavy lifting.** Every sub-problem just adds bound constraints to the parent's LP. Solving each LP is fast; solving the *integer* version of even one of them would defeat the purpose.
2. **The bound prunes aggressively.** Node B₁₁ found an integer solution with value 20, but the incumbent was already 23, so we discard it without exploring further. The incumbent acts as a moving floor.
3. **We branch on a *fractional* variable.** If the LP solution is already integer, we're done — no branching needed. Some problem classes are lucky like that. Most aren't.
4. **The tree shape depends on choices we made.** We branched on $x_1$ first, but we could have branched on $x_2$. We expanded the right subtree before the left. Different choices, different trees.

That last point is the entire reason this post exists. Branch-and-bound's *correctness* doesn't depend on those choices — any choice will find the optimum eventually. But its *speed* depends entirely on them. A great solver explores few nodes; a bad one explores exponentially many. The choices are heuristics, and the heuristics are the soul of the algorithm.

---

## The Two Decisions That Define a Solver

Inside the main loop of every modern MILP solver, two decisions are made over and over:

**Variable selection (branching):** *which fractional variable do we branch on?* If the LP returned $(x_1, x_2, x_3) = (1.5, 0.5, 4.2)$, all three are candidates. Branching on $x_1$ gives one tree shape; branching on $x_3$ gives a different one. The "correct" choice would require knowing the future — which split will close the gap fastest? — and we don't have it. Heuristics try to estimate.

**Node selection:** *of all the open sub-problems, which one do we expand next?* When you've branched a few levels deep, your "to-do list" of unexplored sub-problems can have hundreds of entries. You have to pick one. The "correct" choice would, again, require seeing the future. Heuristics estimate.

CHOP focuses on the second decision. It's the cleaner one to study — the action space is just "pick one of the open nodes", and the per-step compute is small enough to do interesting RL on a laptop.

### The classical menu of node-selection heuristics

Four standard rules cover most of what classical solvers actually do:

- **Best-bound.** Always expand the open node with the highest LP value. The intuition: that node is the most "promising" — it has the most room to improve over the incumbent. Best-bound is the gold standard in MILP folklore. If you only know one heuristic, know this one.
- **Depth-first.** Always expand the deepest open node. The intuition: dive to a leaf, find an integer solution, raise the incumbent, then start pruning. Memory-efficient (the open list stays shallow), and good when finding *any* integer solution is the bottleneck.
- **Breadth-first.** Always expand the shallowest open node. Explore level by level. Rarely the right answer in MILP, but a useful baseline.
- **Random.** Pick uniformly at random from the open list. The control group.

If you ran these four on the airline-scheduling MILPs that real solvers face, best-bound would win most of the time. The asymptotic theory backs this up: in the limit of perfect bound functions, best-bound minimizes the number of nodes explored. It's the right thing to do *given the bound*.

But "given the bound" is doing a lot of work in that sentence.

---

## When Best-Bound Isn't Best

The Set Cover problem asks: given a universe of elements and a collection of sets that each cover some elements at some cost, pick a minimum-cost collection of sets that together cover the whole universe. It's the canonical NP-hard problem behind crew scheduling, sensor placement, and a hundred other applications.

Set Cover's LP relaxation is famously *fractional*. If you solve the LP without the integer constraint, you typically get a solution where many variables sit around $0.5$ — half this set, half that one. The LP value ends up close to the integer optimum, but the LP *solution* is far from an integer point. Worse, the LP value barely changes between adjacent search nodes — branching on one variable nudges the LP value by a tiny amount, then the next node looks almost identical.

Best-bound, in this regime, dives into deep fractional subtrees. Every step looks promising by the LP value, but progress toward an actual integer solution is slow. Random and breadth-first, on the other hand, get bounced around the tree and stumble onto integer-feasible leaves earlier. They raise the incumbent sooner, and the incumbent's job is to prune the rest.

Empirically, on `SetCover(50 elements × 80 sets, density 0.10)` with my in-house simplex backend:

| Heuristic     | Mean nodes ± std | vs best_bound |
|---------------|-------------------|---------------|
| best_bound    | 19.1 ± 16.1       | 1.00x         |
| depth_first   | 10.9 ± 10.5       | 1.75x better  |
| breadth_first | 11.7 ± 9.8        | 1.63x better  |
| random        | 13.2 ± 9.8        | 1.45x better  |

Best-bound — the textbook recommendation — explores roughly *twice* as many nodes as any of the alternatives on this distribution. It's not a quirk of one instance; the result is stable over 40 random instances drawn from the same distribution.

Two things to take away:

1. **Heuristic quality is *problem-distribution-dependent*.** No single rule wins everywhere. The shape of the LP relaxation, the structure of constraints, the typical integrality gap — these vary by problem class, and the right heuristic varies with them.

2. **There is room above the classical baseline.** On Set Cover specifically, *anything* would be better than best-bound. That's the regime where a learned heuristic — one that adapts to the problem distribution — could plausibly help.

---

## Where We Go Next

Picking heuristics by hand is the way it has always been done. Practitioners stare at instances, eyeball patterns, write code. The result is the impressive-but-finicky hybrid branching rule inside SCIP, the proprietary recipes inside Gurobi. Each rule is a person's intuition crystallized into a function.

But intuition has a ceiling. And modern industrial MILP problems don't come from nowhere — they come from *distributions*. An airline solves a similar scheduling problem every Monday. A logistics company solves the same vehicle routing structure every morning. The instances aren't arbitrary; they're samples from a stationary process that the operator has been running for years.

That's exactly the regime where machine learning should win. *Given a distribution of similar instances, can we learn a heuristic that's better than the hand-crafted one for that specific distribution?* The 2019 NeurIPS paper from Maxime Gasse and collaborators showed convincingly that the answer is "yes, by a clear margin" for variable selection (branching) on Set Cover, Combinatorial Auctions, Capacitated Facility Location, and Maximum Independent Set. They imitated the strong-branching expert with a graph neural network and beat the SCIP defaults on every problem class they tested.

CHOP picks up the question for *node selection* and tries to answer it on a laptop, with reinforcement learning instead of imitation, using whatever combination of architectures we can stand up in a few iterations.

That story — from "it doesn't even import" to seven trained architectures, three trainers, and a ~2x improvement over best-bound — is [Part 2]({{ site.url }}/2026/04/28/chop-a-research-diary-in-learning-to-branch.html).

---

## A Note on Honesty

Optimization research, and combinatorial optimization in particular, is a field where it is easy to overclaim. Numbers move with random seeds. Variance can hide trends. Benchmarks are gameable. Throughout the next post I try to report results honestly — including the results that *don't* support the headline. If you pick up only one thing from this series, let it be that any claim of "X% better than baseline" should be read with a question mark until you've seen the seed-by-seed table. The Bipartite-GCN-with-attention architecture I'll show you in Part 2 is the headline champion *on one seed*. On a different seed it underperforms the simpler bare-encoder version. We'll get to that.

For now: branch-and-bound is the algorithm. Heuristics are the soul. Best-bound is the textbook. And on Set Cover, the textbook is wrong.
