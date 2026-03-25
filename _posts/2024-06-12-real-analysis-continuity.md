---
layout: post
title: "Real Analysis I: Continuity and Compactness"
date: 2024-06-12
tags: [Math, Analysis, Real Analysis]
---

This is Part 4 — the finale — of a four-part series on Real Analysis I. In [Part 1](/2024/05/01/real-analysis-real-numbers.html), we built the reals and the Completeness Axiom. In [Part 2](/2024/05/08/real-analysis-topology.html), we developed the topology of the real line. In [Part 3](/2024/06/05/real-analysis-sequences.html), we mastered sequences and convergence. Now we bring everything together.

This post is about the theorems that make calculus work. The Extreme Value Theorem says a continuous function on a closed interval achieves its maximum. The Intermediate Value Theorem says a continuous function that changes sign must cross zero. You used these facts freely in calculus — now we prove them, and in doing so, we will see every major result from the previous three parts play its role.

## What This Post Covers

- **Limits of Functions** — The $\epsilon$-$\delta$ definition and the sequential characterization
- **Continuous Functions** — The definition, equivalent formulations, and the Dirichlet function
- **Properties of Continuous Functions** — Algebraic operations and the open set characterization
- **Compact Sets and Heine-Borel** — Compactness as the bridge between topology and analysis
- **The Extreme Value Theorem** — Continuous functions on compact sets achieve their bounds
- **The Intermediate Value Theorem** — Continuous functions preserve connectedness

---

## Limits of Functions

The sequence limit asks: what happens to $a_n$ as $n$ grows? The function limit asks: what happens to $f(x)$ as $x$ approaches a point $c$?

**Definition.** Let $\emptyset \neq D \subseteq \mathbb{R}$, $c \in D'$ (an accumulation point of $D$), and $f: D \to \mathbb{R}$. A real number $L$ is the **limit of $f$ at $x = c$** if

$$
\forall\, \epsilon > 0, \quad \exists\, \delta > 0 \text{ such that } \forall\, x \in D, \quad 0 < |x - c| < \delta \implies |f(x) - L| < \epsilon.
$$

We write $\lim_{x \to c} f(x) = L$ or $f(x) \to L$ as $x \to c$.

The key difference from sequences: in a sequence, the index $n$ approaches infinity from one direction. For functions, $x$ can approach $c$ from *any* direction — left, right, or oscillating. The condition $0 < \lvert x - c \rvert$ ensures we never evaluate $f$ at $c$ itself; the limit is about the *behavior near* $c$, not the value at $c$.

<svg viewBox="0 0 400 260" style="max-width:380px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax5 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.2; }
    .lbl5 { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .lbl5s { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--text-secondary, #555); }
    .box-h { stroke: var(--primary, #94452b); stroke-width: 1; stroke-dasharray: 5,4; fill: none; }
    .box-v { stroke: #2a7ae2; stroke-width: 1; stroke-dasharray: 5,4; fill: none; }
    .curve { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
    .region-x { fill: rgba(42, 122, 226, 0.06); }
    .region-y { fill: rgba(148, 69, 43, 0.06); }
  </style>
  <defs>
    <marker id="arr5" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
      <path d="M0,0 L6,2.5 L0,5" fill="none" stroke="var(--text-primary, #1a1a1a)" stroke-width="1" />
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="50" y1="220" x2="370" y2="220" class="ax5" marker-end="url(#arr5)" />
  <line x1="50" y1="220" x2="50" y2="20" class="ax5" marker-end="url(#arr5)" />
  <!-- Delta band on x-axis -->
  <rect x="160" y="210" width="80" height="20" class="region-x" />
  <line x1="160" y1="210" x2="160" y2="230" class="ax5" stroke-dasharray="3,3" />
  <line x1="240" y1="210" x2="240" y2="230" class="ax5" stroke-dasharray="3,3" />
  <!-- Epsilon band on y-axis -->
  <rect x="40" y="85" width="310" height="50" class="region-y" />
  <line x1="40" y1="85" x2="350" y2="85" stroke="var(--primary, #94452b)" stroke-width="0.8" stroke-dasharray="5,4" />
  <line x1="40" y1="135" x2="350" y2="135" stroke="var(--primary, #94452b)" stroke-width="0.8" stroke-dasharray="5,4" />
  <!-- Delta lines projected up -->
  <line x1="160" y1="220" x2="160" y2="85" stroke="#2a7ae2" stroke-width="0.8" stroke-dasharray="5,4" />
  <line x1="240" y1="220" x2="240" y2="85" stroke="#2a7ae2" stroke-width="0.8" stroke-dasharray="5,4" />
  <!-- c and L markers -->
  <line x1="200" y1="215" x2="200" y2="225" class="ax5" />
  <text x="200" y="242" text-anchor="middle" class="lbl5" font-style="italic">c</text>
  <text x="160" y="242" text-anchor="middle" class="lbl5s">c−δ</text>
  <text x="240" y="242" text-anchor="middle" class="lbl5s">c+δ</text>
  <text x="38" y="113" text-anchor="end" class="lbl5" font-style="italic" fill="var(--primary, #94452b)">L</text>
  <text x="38" y="89" text-anchor="end" class="lbl5s">L+ε</text>
  <text x="38" y="139" text-anchor="end" class="lbl5s">L−ε</text>
  <!-- L line -->
  <line x1="45" y1="110" x2="350" y2="110" stroke="var(--primary, #94452b)" stroke-width="1" stroke-dasharray="8,4" />
  <!-- Curve -->
  <path d="M80,180 Q140,140 180,118 Q200,108 220,102 Q260,88 320,60" class="curve" />
</svg>

*The $\epsilon$-$\delta$ definition: if $x$ lands in the $\delta$-neighborhood of $c$ (blue band), then $f(x)$ lands in the $\epsilon$-neighborhood of $L$ (red band). The function "maps nearness to nearness."*

**Theorem.** Limits of functions are unique (same proof technique as for sequences).

### The Sequential Characterization

The sequential characterization of limits is one of the most useful tools in analysis: it lets us transfer the entire machinery of sequences to functions.

**Theorem.** Let $D \subseteq \mathbb{R}$, $c \in D'$, and $f: D \to \mathbb{R}$. Suppose $L \in \mathbb{R}$. Then $\lim_{x \to c} f(x) = L$ if and only if for every sequence $(x_n)$ in $D \setminus \\{c\\}$ with $x_n \to c$, we have $f(x_n) \to L$.

The forward direction is a direct $\epsilon$-$\delta$ argument. The contrapositive of the reverse direction is especially powerful:

**Corollary (TFAE).** The following are equivalent:
1. $f$ does not have a limit at $c$.
2. There exists a sequence $(x_n)$ in $D \setminus \\{c\\}$ with $x_n \to c$ but $(f(x_n))$ does not converge in $\mathbb{R}$.

**Example.** $\lim_{x \to 0} \sin(1/x)$ does not exist. Take $x_n = \frac{2}{\pi(4n+1)}$, so $f(x_n) = \sin(\frac{\pi}{2}(4n+1)) = 1$. Take $y_n = \frac{1}{n\pi}$, so $f(y_n) = \sin(n\pi) = 0$. Two subsequences give different limits.

### Worked Examples

**Example.** Let $f: \mathbb{R} \to \mathbb{R}$ with $f(x) = x^2 + 4x + 9$. Prove $\lim_{x \to 3} f(x) = 30$.

*Proof.* Let $\epsilon > 0$. We need $\lvert x^2 + 4x + 9 - 30 \rvert = \lvert x^2 + 4x - 21 \rvert = \lvert x - 3 \rvert\lvert x + 7 \rvert < \epsilon$. Choose $\delta = \min\\{1/2,\, \epsilon/16\\}$. If $0 < \lvert x - 3 \rvert < \delta$, then $\lvert x \rvert < 3 + 1/2$ so $\lvert x + 7 \rvert \leq \lvert x \rvert + 7 < 10.5 < 16$. Hence $\lvert f(x) - 30 \rvert < 16 \cdot (\epsilon/16) = \epsilon$. $\square$

### One-Sided Limits

**Definition.** $f$ has the **left-hand limit** $L$ at $c$ if for every $\epsilon > 0$, there exists $\delta > 0$ such that $c - \delta < x < c \implies \lvert f(x) - L \rvert < \epsilon$. Similarly for the **right-hand limit**.

**Theorem.** If $f$ has both the left-hand and right-hand limits at $c$, and they are equal to $L$, then $\lim_{x \to c} f(x)$ exists and equals $L$.

---

## Continuous Functions

Limits of functions describe behavior *near* a point. Continuity is the special case where the limit *agrees* with the function value.

**Definition.** Let $D \subseteq \mathbb{R}$ and $c \in D$. We say $f: D \to \mathbb{R}$ is **continuous at $c$** if

$$
\forall\, \epsilon > 0, \quad \exists\, \delta > 0 \text{ such that } \forall\, x \in D, \quad |x - c| < \delta \implies |f(x) - f(c)| < \epsilon.
$$

If $c \in D'$, this is equivalent to $\lim_{x \to c} f(x) = f(c)$.

**Remark.** Every function is automatically continuous at each isolated point of its domain, since for an isolated point $c$, there exists $\delta > 0$ with $N(c; \delta) \cap D = \\{c\\}$, so the condition $\lvert f(x) - f(c) \rvert < \epsilon$ is vacuously satisfied.

### Equivalent Formulations

**Theorem.** Let $f: D \to \mathbb{R}$ and $c \in D$. The following are equivalent:
1. $f$ is continuous at $c$.
2. For every sequence $(x_n)$ in $D$ with $x_n \to c$, we have $f(x_n) \to f(c)$.
3. For every neighborhood $N(f(c); \epsilon)$, there exists a neighborhood $N(c; \delta)$ such that $f(N(c; \delta) \cap D) \subseteq N(f(c); \epsilon)$.
4. $\lim_{x \to c} f(x)$ exists and equals $f(c)$ (when $c \in D'$).

### The Dirichlet Function: Continuous Nowhere

The **Dirichlet function** is defined by:

$$
f(x) = \begin{cases} 1 & \text{if } x \in \mathbb{Q} \\ 0 & \text{if } x \notin \mathbb{Q} \end{cases}
$$

**Theorem.** The Dirichlet function is discontinuous at every $c \in \mathbb{R}$.

*Proof.* Let $c \in \mathbb{R}$.

**Case 1: $c \in \mathbb{Q}$.** Then $f(c) = 1$. By the density of the irrationals, there exists a sequence of irrationals $(y_n)$ with $y_n \to c$. Then $f(y_n) = 0$ for all $n$, so $f(y_n) \to 0 \neq 1 = f(c)$. By the sequential characterization, $f$ is not continuous at $c$.

**Case 2: $c \notin \mathbb{Q}$.** Then $f(c) = 0$. By the density of the rationals, there exists a sequence of rationals $(r_n)$ with $r_n \to c$. Then $f(r_n) = 1$ for all $n$, so $f(r_n) \to 1 \neq 0 = f(c)$. So $f$ is not continuous at $c$. $\square$

> The Dirichlet function is the ultimate stress test for our definition of continuity. It shows that continuity is truly a *local* property that depends on the fine structure of the domain near each point.

In contrast, $g(x) = x \sin(1/x)$ for $x \neq 0$, $g(0) = 0$ is continuous at $0$ (since $\lvert g(x) \rvert \leq \lvert x \rvert \to 0$), even though it oscillates infinitely near the origin. Continuity does not require "niceness" — only that the oscillations are controlled.

### Discontinuity

**Theorem.** $f: D \to \mathbb{R}$ is discontinuous at $c \in D$ if and only if there exists a sequence $(x_n)$ in $D$ with $x_n \to c$ but $f(x_n) \not\to f(c)$.

**Example.** $f(x) = 1/x$ on $D = (-\infty, 0) \cup (0, \infty)$. This function is continuous on $D$. The function is *not* defined at $x = 0$, so the question of continuity there does not arise. But $f$ is discontinuous "at $x = 0$" in the sense that $\lim_{x \to 0} f(x)$ does not exist — the left- and right-hand limits are $-\infty$ and $+\infty$.

---

## Properties of Continuous Functions

Continuous functions are well-behaved under the standard algebraic operations.

**Theorem.** If $f, g: D \to \mathbb{R}$ are continuous at $c \in D$, then:
1. $f + g$, $f - g$, and $f \cdot g$ are continuous at $c$.
2. $f/g$ is continuous at $c$, provided $g(c) \neq 0$.

**Theorem.** If $f: D \to \mathbb{R}$ is continuous at $c$ and $g: E \to \mathbb{R}$ is continuous at $f(c)$, with $f(D) \subseteq E$, then $g \circ f$ is continuous at $c$.

*Proof.* Let $W$ be a neighborhood of $g(f(c))$. Since $g$ is continuous at $f(c)$, there exists a neighborhood $V$ of $f(c)$ with $g(V \cap E) \subseteq W$. Since $f$ is continuous at $c$ and $V$ is a neighborhood of $f(c)$, there exists a neighborhood $U$ of $c$ with $f(U \cap D) \subseteq V$. Since $f(D) \subseteq E$, we have $f(U \cap D) \subseteq V \cap E$, so $g(f(U \cap D)) \subseteq g(V \cap E) \subseteq W$. $\square$

### The Open Set Characterization

There is a beautiful topological characterization of continuity that connects everything we've built.

**Theorem.** $f: D \to \mathbb{R}$ is continuous on $D$ if and only if for every open set $G \subseteq \mathbb{R}$, there exists an open set $H \subseteq \mathbb{R}$ such that $H \cap D = f^{-1}(G)$.

This is the "grown-up" definition of continuity used in topology: *continuous functions are those that pull open sets back to open sets*.

---

## Compact Sets and the Heine-Borel Theorem

The theorems of calculus — the Extreme Value Theorem and the Intermediate Value Theorem — don't hold for arbitrary domains. They need the domain to be *compact*. This concept ties together all of our topology.

**Definition.** A set $C \subseteq \mathbb{R}$ is **compact** if for every collection of open sets $\\{U_\alpha\\}_{\alpha \in \Lambda}$ satisfying $C \subseteq \bigcup_{\alpha \in \Lambda} U_\alpha$ (an **open cover** of $C$), there is a finite subcollection $U_{\alpha_1}, \ldots, U_{\alpha_k}$ such that $C \subseteq U_{\alpha_1} \cup \cdots \cup U_{\alpha_k}$ (a **finite subcover**).

Compactness says: you can't "cover" $C$ with open sets in a way that genuinely requires infinitely many of them. Some finite selection always suffices.

**Examples.**
- Any finite subset of $\mathbb{R}$ is compact.
- $[a, b]$ is compact (this is the content of the Heine-Borel theorem below).
- $(a, b)$, $(a, b]$, and $[a, b)$ are *not* compact.
- $\\{1/n : n \in \mathbb{N}\\}$ is not compact (the cover $\\{N(1/n;\, 1/n - 1/(n+1))\\}$ has no finite subcover), but $\\{0\\} \cup \\{1/n : n \in \mathbb{N}\\}$ is compact (closed and bounded).

**Non-example.** $\\{1/n : n \in \mathbb{N}\\}$ is bounded but not closed ($0$ is an accumulation point not in the set), hence not compact. The open cover $U_n = N(1/n;\, \frac{1}{2}(\frac{1}{n} - \frac{1}{n+1}))$ gives each $1/n$ its own tiny interval, and the $U_n$'s are pairwise disjoint — so no finite subcollection can cover all the points.

The following landmark theorem gives a clean characterization of compact subsets of $\mathbb{R}$.

**Theorem (Heine-Borel).** A set $C \subseteq \mathbb{R}$ is compact if and only if $C$ is closed and bounded.

The proof is involved (the hard direction uses a bisection/nested intervals argument), but the result is clean and practical: in $\mathbb{R}$, "compact" is just another word for "closed and bounded."

---

## The Extreme Value Theorem

Now the payoff begins. We first show that continuous functions preserve compactness.

**Theorem.** Let $D \subseteq \mathbb{R}$ be compact and $f: D \to \mathbb{R}$ be continuous. Then $f(D)$ is compact.

*Proof.* Let $\\{G_\alpha\\}_{\alpha \in \Lambda}$ be an open cover of $f(D)$. Then $D \subseteq f^{-1}\left(\bigcup_{\alpha \in \Lambda} G_\alpha\right) = \bigcup_{\alpha \in \Lambda} f^{-1}(G_\alpha)$. Since $f$ is continuous and each $G_\alpha$ is open, each $f^{-1}(G_\alpha)$ is open relative to $D$: there exists an open set $H_\alpha$ with $f^{-1}(G_\alpha) = H_\alpha \cap D$. Then $D \subseteq \bigcup_{\alpha \in \Lambda} H_\alpha$.

Since $D$ is compact, there exist $\alpha_1, \ldots, \alpha_k \in \Lambda$ with $D \subseteq H_{\alpha_1} \cup \cdots \cup H_{\alpha_k}$. Then:

$$
f(D) \subseteq f(H_{\alpha_1} \cap D) \cup \cdots \cup f(H_{\alpha_k} \cap D) = f(f^{-1}(G_{\alpha_1})) \cup \cdots \cup f(f^{-1}(G_{\alpha_k})) \subseteq G_{\alpha_1} \cup \cdots \cup G_{\alpha_k}.
$$

So $\\{G_{\alpha_1}, \ldots, G_{\alpha_k}\\}$ is a finite subcover of $f(D)$. $\square$

**Corollary (Extreme Value Theorem).** Let $D \subseteq \mathbb{R}$ be compact and $f: D \to \mathbb{R}$ be continuous. Then $f$ attains its absolute maximum and absolute minimum on $D$. That is, there exist $x_1, x_2 \in D$ with $f(x_1) \leq f(x) \leq f(x_2)$ for all $x \in D$.

*Proof.* By the previous theorem, $f(D)$ is compact, hence closed and bounded (Heine-Borel). Since $f(D)$ is bounded, $M = \sup f(D)$ and $m = \inf f(D)$ exist. Since $f(D)$ is closed, $M \in \operatorname{bd}(f(D)) \cup \operatorname{int}(f(D)) \subseteq f(D)$ (closed sets contain their boundary). So $M, m \in f(D)$, which means $M = f(x_2)$ and $m = f(x_1)$ for some $x_1, x_2 \in D$. $\square$

> This is *why* we close the interval in calculus when looking for extrema. The function $f(x) = x$ on the open interval $(0, 1)$ is continuous and bounded, but it never achieves its supremum $1$ or infimum $0$. Compactness is not optional.

---

## The Intermediate Value Theorem

The final theorem of the course ties completeness back to the intuitive notion that continuous functions "draw unbroken curves."

**Theorem (Bolzano's Intermediate Value Theorem).** Let $f: [a, b] \to \mathbb{R}$ be continuous. If $f(a) < 0 < f(b)$ (or $f(a) > 0 > f(b)$), then there exists $c \in (a, b)$ such that $f(c) = 0$.

*Proof.* Assume $f(a) < 0 < f(b)$. Define

$$
S = \{x \in [a, b] : f(x) \leq 0\}.
$$

Since $f(a) < 0$, we have $a \in S$, so $S \neq \emptyset$. Since $S \subseteq [a, b]$, $S$ is bounded above by $b$. By the Completeness Axiom, $c = \sup S$ exists, and $a \leq c \leq b$.

We claim $f(c) = 0$.

**$f(c) < 0$ leads to a contradiction.** Clearly $c \neq b$ (since $f(b) > 0$). By continuity of $f$ at $c$, there exists $\delta > 0$ such that $f(x) < 0$ for all $x \in (c - \delta, c + \delta) \cap [a, b]$. Pick any $q \in (c, c + \delta) \cap [a, b]$. Then $f(q) < 0$, so $q \in S$. But $q > c = \sup S$, a contradiction.

**$f(c) > 0$ leads to a contradiction.** Then $c \neq a$. By continuity, there exists $\delta > 0$ such that $f(x) > 0$ for all $x \in (c - \delta, c + \delta) \cap [a, b]$. This means no point of $(c - \delta, c]$ belongs to $S$ (since $f > 0$ there). So $c - \delta$ is an upper bound for $S$. But $c - \delta < c = \sup S$, contradicting the leastness of $c$.

Since both alternatives are impossible, $f(c) = 0$ by trichotomy. $\square$

> The IVT says continuous functions on closed intervals "don't jump over values." It seems obvious — draw any curve from below the $x$-axis to above it, and it must cross. But this intuition relies on completeness. In $\mathbb{Q}$, the function $f(x) = x^2 - 2$ satisfies $f(1) < 0$ and $f(2) > 0$, but there is no $c \in \mathbb{Q}$ with $f(c) = 0$. The IVT fails without completeness.

The general form handles any intermediate value, not just zero:

**Corollary.** If $f: [a, b] \to \mathbb{R}$ is continuous and $v$ is between $f(a)$ and $f(b)$ (i.e., $f(a) < v < f(b)$ or $f(b) < v < f(a)$), then there exists $c \in (a, b)$ with $f(c) = v$.

*Proof.* Apply the theorem to $g(x) = f(x) - v$. $\square$

**Application.** Every positive real number has an $n$th root. For any $a > 0$ and $n \in \mathbb{N}$, the function $f(x) = x^n$ is continuous, $f(0) = 0 < a$, and $f(M) > a$ for $M$ large enough. By the IVT, there exists $c \in (0, M)$ with $c^n = a$.

---

## Conclusion

We started this series with the axioms for the real numbers — a set equipped with addition, multiplication, order, and one additional property: *completeness*. That single axiom, asserting the existence of suprema for bounded sets, has been the engine behind every major result:

- The **Archimedean Property** — no infinitely large or small reals (Part 1)
- The **existence of square roots** — filling the gaps in $\mathbb{Q}$ (Part 1)
- The **density of $\mathbb{Q}$** — rationals and irrationals interleaved everywhere (Part 1)
- The **Monotone Convergence Theorem** — bounded monotone sequences converge (Part 3)
- The **Cauchy Criterion** — convergence without knowing the limit (Part 3)
- The **Bolzano-Weierstrass Theorem** — bounded sequences have convergent subsequences (Part 3)
- The **Extreme Value Theorem** — continuous functions on compact sets achieve their bounds (Part 4)
- The **Intermediate Value Theorem** — continuous functions preserve connectedness (Part 4)

Real analysis is, at its heart, the study of what completeness makes possible. The rationals have algebra and order but are full of holes. The reals, by filling those holes, enable the entire edifice of limits, continuity, differentiation, and integration.

The journey continues in Real Analysis II with differentiation, the Riemann integral, and sequences and series of functions — but the foundation is now complete.
