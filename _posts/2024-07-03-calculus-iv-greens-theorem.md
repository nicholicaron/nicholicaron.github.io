---
layout: post
title: "Calculus IV: Conservative Fields and Green's Theorem"
date: 2024-07-03
tags: [Math, Calculus, Multivariable Calculus]
---

In [Part 2](/2024/06/26/calculus-iv-curves-line-integrals.html), we defined two types of line integrals: the work integral $\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r}$ and the flux integral $\int_C \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}s$. We noticed that for certain "nice" vector fields, the work integral depends only on the endpoints — not on the path. This post is about understanding *why*, and about the crown jewel of two-dimensional vector calculus: **Green's theorem**, which connects line integrals to double integrals.

The thread running through everything is the idea that local properties of a vector field (how much it spins or expands at a point) determine global properties (the total circulation or flux around a curve). This is the first instance of a pattern that culminates in the general Stokes' theorem.

This is Part 3 of a four-part series on Calculus IV.

## What This Post Covers

- **Path Independence** — When does the value of a line integral depend only on endpoints?
- **The Fundamental Theorem of Line Integrals** — The multivariable analogue of $\int_a^b f'(x)\,\mathrm{d}x = f(b) - f(a)$
- **Finding Potential Functions** — How to recover $f$ from $\nabla f$
- **The Component Test** — A quick check for whether a vector field could be conservative
- **The Tricky $\mathrm{d}\theta$ Form** — A cautionary tale about domains and topology
- **Curl and Divergence** — Local measures of rotation and expansion
- **Green's Theorem** — The bridge between line integrals and double integrals
- **Applications** — Computing area via line integrals, the shoelace formula, and regions with holes

---

## Path Independence and Conservative Fields

We observed in Part 2 that the vector field $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}$ gave the same work integral $\frac{3}{2}$ along two very different paths from $(0,0,0)$ to $(1,1,1)$. This is called **path independence**: the integral $\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r}$ depends only on where $C$ starts and ends, not on the route it takes.

A vector field with this property is called **conservative**. The name comes from physics: for a conservative force, the total energy (kinetic + potential) is conserved.

Path independence has an immediate consequence for closed curves: if $C$ is any closed loop (starting and ending at the same point), then

$$
\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = 0
$$

for a conservative field. This is because we can think of a closed loop as a path from $A$ to $B$ followed by a path from $B$ back to $A$; the second path gives the negative of the first.

### The fundamental theorem of line integrals

Why was $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}$ path-independent? Because it is the gradient of $f(x,y,z) = \frac{1}{2}(x^2 + y^2 + z^2)$:

$$
\nabla f = \frac{\partial f}{\partial x}\,\mathbf{i} + \frac{\partial f}{\partial y}\,\mathbf{j} + \frac{\partial f}{\partial z}\,\mathbf{k} = x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k} = \mathbf{F}.
$$

This is the key to the entire theory.

**Theorem (Fundamental Theorem of Line Integrals).** *Let $f\colon \mathbb{R}^n \to \mathbb{R}$ be differentiable with continuous gradient $\mathbf{F} = \nabla f$. Let $C$ be a smooth curve from point $A$ to point $B$. Then*

$$
\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = f(B) - f(A).
$$

*Proof.* Using the chain rule, $\frac{\mathrm{d}}{\mathrm{d}t}f(\mathbf{r}(t)) = \nabla f(\mathbf{r}(t)) \cdot \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}$. So

$$
\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \int_a^b \nabla f(\mathbf{r}(t)) \cdot \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\,\mathrm{d}t = \int_a^b \frac{\mathrm{d}}{\mathrm{d}t}f(\mathbf{r}(t))\,\mathrm{d}t = f(\mathbf{r}(b)) - f(\mathbf{r}(a)) = f(B) - f(A). \quad \square
$$

The converse is also true: every conservative field is the gradient of some scalar function $f$, called a **potential function**.

> In the language of differential forms, this theorem says $\int_C \mathrm{d}f = f(B) - f(A)$ — the exact analogue of the one-variable fundamental theorem of calculus.

---

## Finding Potential Functions

Given a vector field $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j}$ that we believe to be conservative, how do we find the potential function $f$ with $\nabla f = \mathbf{F}$?

The idea is like finding an antiderivative, but with multiple variables. The $+C$ from single-variable calculus becomes much richer: a "constant" with respect to $x$ can be any function of $y$.

### Example in $\mathbb{R}^2$

Let $\mathbf{F} = 2xy^3\,\mathbf{i} + 3(x^2 - 1)y^2\,\mathbf{j}$. We want $f$ with $\frac{\partial f}{\partial x} = 2xy^3$ and $\frac{\partial f}{\partial y} = 3(x^2 - 1)y^2$.

**Step 1.** Integrate $\frac{\partial f}{\partial x} = 2xy^3$ with respect to $x$:

$$
f(x,y) = x^2 y^3 + g(y)
$$

where $g(y)$ is an unknown function of $y$ (the "constant" of integration).

**Step 2.** Differentiate our expression with respect to $y$ and set it equal to $N$:

$$
\frac{\partial f}{\partial y} = 3x^2 y^2 + g'(y) = 3(x^2 - 1)y^2 \implies g'(y) = -3y^2.
$$

The crucial check: $g'(y)$ depends only on $y$, not on $x$. If it still contained $x$, then no potential function would exist.

**Step 3.** Integrate: $g(y) = -y^3 + C$. The potential function is

$$
f(x,y) = x^2 y^3 - y^3 + C.
$$

### Example in $\mathbb{R}^3$

For $\mathbf{F} = (2xy + 3z^2)\,\mathbf{i} + (x^2 + 4z^2)\,\mathbf{j} + (6xz + 8yz)\,\mathbf{k}$:

1. Integrate $\frac{\partial f}{\partial x} = 2xy + 3z^2$ to get $f = x^2 y + 3xz^2 + g(y,z)$.
2. Differentiate with respect to $y$: $x^2 + \frac{\partial g}{\partial y} = x^2 + 4z^2$, so $\frac{\partial g}{\partial y} = 4z^2$, giving $g = 4yz^2 + h(z)$.
3. Differentiate with respect to $z$: $6xz + 8yz + h'(z) = 6xz + 8yz$, so $h'(z) = 0$.

Therefore $f(x,y,z) = x^2 y + 3xz^2 + 4yz^2 + C$.

---

## The Component Test

Before going through the work of finding a potential function, it is useful to have a quick test for whether one could exist at all.

**Theorem (Component Test in $\mathbb{R}^2$).** *If $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j}$ is a gradient field, then $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$.*

This follows immediately from Clairaut's theorem: if $f$ exists with $\frac{\partial f}{\partial x} = M$ and $\frac{\partial f}{\partial y} = N$, then $\frac{\partial M}{\partial y} = \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial N}{\partial x}$.

**In $\mathbb{R}^3$**, the test becomes three conditions:

$$
\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}, \qquad \frac{\partial M}{\partial z} = \frac{\partial P}{\partial x}, \qquad \frac{\partial N}{\partial z} = \frac{\partial P}{\partial y}.
$$

If any one of these fails, $\mathbf{F}$ is definitely not conservative. If all three hold, then $\mathbf{F}$ is *usually* conservative — but there is a subtle catch.

---

## The Tricky $\mathrm{d}\theta$ Form

Consider the vector field

$$
\mathbf{F} = \frac{-y}{x^2 + y^2}\,\mathbf{i} + \frac{x}{x^2 + y^2}\,\mathbf{j}.
$$

This is the gradient of $\theta(x,y) = \arctan(y/x)$ — the polar angle. And indeed, $\mathbf{F}$ passes the component test: $\frac{\partial M}{\partial y} = \frac{\partial N}{\partial x}$ everywhere that $\mathbf{F}$ is defined. So is it conservative?

The answer is: **it depends on the domain**.

The problem is that $\theta$ is not a continuous function on all of $\mathbb{R}^2 \setminus \{(0,0)\}$. As you walk counterclockwise around the origin, $\theta$ increases from $0$ toward $2\pi$, then abruptly jumps back to $0$. No matter where you place the "cut" (the ray where $\theta$ is discontinuous), the function $\theta$ cannot be made continuous on any domain that encircles the origin.

Computing directly on the unit circle $\mathbf{r}(t) = (\cos t, \sin t)$:

$$
\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \int_0^{2\pi} 1\,\mathrm{d}t = 2\pi \neq 0.
$$

A conservative field cannot have a nonzero integral around a closed loop. So $\mathbf{F}$ is *not* conservative on $\mathbb{R}^2 \setminus \{(0,0)\}$.

However, on any **simply connected** domain that avoids the origin (like a half-plane), $\mathbf{F}$ *is* conservative: the component test works, and a potential function (some branch of $\theta$) exists.

> **Theorem.** If $D$ is an open, simply connected domain (no holes), and $\mathbf{F}$ is defined on all of $D$ with continuous partial derivatives and passes the component test, then $\mathbf{F}$ is conservative on $D$.

The lesson: the component test is necessary but not quite sufficient. Topology matters — specifically, whether the domain has "holes" that a closed curve could wrap around.

---

## Curl and Divergence

We now develop two local measurements of a vector field's behavior. Both are motivated by thinking about what a tiny closed curve "sees."

### Curl (circulation density)

Imagine tossing a small chip of wood into a flowing stream. Two things happen: it floats downstream, and it *spins*. The spinning is caused by different parts of the chip experiencing different velocities.

The **curl** of $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j}$ at a point $(x,y)$ quantifies this spinning tendency:

$$
\operatorname{curl}\mathbf{F} = \frac{\partial N}{\partial x} - \frac{\partial M}{\partial y}.
$$

This formula emerges naturally from computing the circulation $\oint_{\square} \mathbf{F} \cdot \mathrm{d}\mathbf{r}$ around a tiny square of side $2h$ centered at $(x,y)$, then dividing by the area $4h^2$:

$$
\lim_{h \to 0} \frac{\displaystyle\oint_{\square} \mathbf{F} \cdot \mathrm{d}\mathbf{r}}{4h^2} = \frac{\partial N}{\partial x} - \frac{\partial M}{\partial y}.
$$

Notice: $\operatorname{curl}\mathbf{F} = 0$ is exactly the component test. Conservative fields have zero curl everywhere — they produce no local spinning.

### Divergence (flux density)

Similarly, the **divergence** measures the tendency of a vector field to *expand* at a point — like an air current flowing outward from a high-pressure region.

$$
\operatorname{div}\mathbf{F} = \frac{\partial M}{\partial x} + \frac{\partial N}{\partial y}.
$$

This comes from computing the outward flux across the boundary of a tiny square and dividing by the area. A positive divergence means the field is "creating" flow at that point (a source); a negative divergence means it is absorbing flow (a sink).

<svg viewBox="0 0 440 160" style="max-width:440px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .cd-lb { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .cd-ar { stroke: var(--primary, #94452b); stroke-width: 1.2; marker-end: url(#cda); }
  </style>
  <defs>
    <marker id="cda" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4" fill="var(--primary, #94452b)" stroke="none"/></marker>
  </defs>
  <!-- Left: curl (spinning) -->
  <circle cx="100" cy="80" r="30" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" fill="none" stroke-dasharray="3,2"/>
  <line x1="60" y1="65" x2="75" y2="65" class="cd-ar"/>
  <line x1="60" y1="95" x2="45" y2="95" class="cd-ar"/>
  <line x1="100" y1="50" x2="115" y2="48" class="cd-ar"/>
  <line x1="100" y1="110" x2="85" y2="112" class="cd-ar"/>
  <line x1="130" y1="65" x2="135" y2="55" class="cd-ar"/>
  <line x1="130" y1="95" x2="125" y2="105" class="cd-ar"/>
  <!-- spinning arrow -->
  <path d="M 108,60 A 15,15 0 1,1 93,62" stroke="var(--text-primary, #1a1a1a)" stroke-width="1" fill="none" marker-end="url(#cda)"/>
  <text x="50" y="145" class="cd-lb">curl ≠ 0: field induces rotation</text>
  <!-- Right: divergence (expanding) -->
  <circle cx="340" cy="80" r="30" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" fill="none" stroke-dasharray="3,2"/>
  <line x1="340" y1="80" x2="370" y2="65" class="cd-ar"/>
  <line x1="340" y1="80" x2="370" y2="95" class="cd-ar"/>
  <line x1="340" y1="80" x2="310" y2="65" class="cd-ar"/>
  <line x1="340" y1="80" x2="310" y2="95" class="cd-ar"/>
  <line x1="340" y1="80" x2="340" y2="52" class="cd-ar"/>
  <line x1="340" y1="80" x2="340" y2="108" class="cd-ar"/>
  <line x1="340" y1="80" x2="365" y2="80" class="cd-ar"/>
  <line x1="340" y1="80" x2="315" y2="80" class="cd-ar"/>
  <text x="280" y="145" class="cd-lb">div > 0: field expands (source)</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Curl measures local rotation; divergence measures local expansion.</p>

### Connection to differential forms

Both curl and divergence arise from the **exterior derivative** of a 1-form $\phi = M\,\mathrm{d}x + N\,\mathrm{d}y$:

$$
\mathrm{d}\phi = \left(\frac{\partial N}{\partial x} - \frac{\partial M}{\partial y}\right)\mathrm{d}x \wedge \mathrm{d}y.
$$

The coefficient is exactly the curl! And the divergence version comes from the flux form of the same integral. This unification is a hint that differential forms are the "right" language for all of this.

---

## Green's Theorem

We have all the pieces. Green's theorem is the bridge that connects line integrals (around a boundary) to double integrals (over a region).

### Green's theorem for circulation

**Theorem (Green's Theorem).** *Let $R$ be a bounded, simply connected region in $\mathbb{R}^2$ with boundary $C$ oriented counterclockwise. Let $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j}$ be a vector field with continuous partial derivatives on an open set containing $R$. Then:*

$$
\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \iint_R \operatorname{curl}\mathbf{F}\,\mathrm{d}A = \iint_R \left(\frac{\partial N}{\partial x} - \frac{\partial M}{\partial y}\right)\mathrm{d}x\,\mathrm{d}y.
$$

*In words: the circulation of $\mathbf{F}$ around $C$ equals the integral of the circulation density over $R$.*

### The proof idea

The proof is a beautiful argument about cancellation.

<svg viewBox="0 0 500 170" style="max-width:500px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .gt-lb { font-family: 'Inter', sans-serif; font-size: 10px; fill: var(--text-primary, #1a1a1a); }
    .gt-cell { stroke: var(--primary, #94452b); stroke-width: 0.8; fill: none; }
    .gt-bd { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
    .gt-ar { stroke: var(--primary, #94452b); stroke-width: 0.8; }
  </style>
  <!-- Left: region subdivided -->
  <rect x="20" y="20" width="130" height="130" fill="none" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" stroke-dasharray="2,2"/>
  <!-- grid lines -->
  <line x1="20" y1="46" x2="150" y2="46" class="gt-cell"/>
  <line x1="20" y1="72" x2="150" y2="72" class="gt-cell"/>
  <line x1="20" y1="98" x2="150" y2="98" class="gt-cell"/>
  <line x1="20" y1="124" x2="150" y2="124" class="gt-cell"/>
  <line x1="46" y1="20" x2="46" y2="150" class="gt-cell"/>
  <line x1="72" y1="20" x2="72" y2="150" class="gt-cell"/>
  <line x1="98" y1="20" x2="98" y2="150" class="gt-cell"/>
  <line x1="124" y1="20" x2="124" y2="150" class="gt-cell"/>
  <!-- blob boundary -->
  <path d="M 55,30 Q 120,18 140,50 Q 155,90 135,130 Q 100,155 60,140 Q 25,120 30,80 Q 28,50 55,30 Z" class="gt-bd"/>
  <text x="45" y="165" class="gt-lb">Subdivide into cells</text>
  <!-- Arrow -->
  <text x="170" y="85" style="font-size:18px; fill: var(--text-primary, #1a1a1a);">→</text>
  <!-- Middle: circulation on each cell -->
  <rect x="200" y="20" width="130" height="130" fill="none" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" stroke-dasharray="2,2"/>
  <line x1="200" y1="46" x2="330" y2="46" class="gt-cell"/>
  <line x1="200" y1="72" x2="330" y2="72" class="gt-cell"/>
  <line x1="200" y1="98" x2="330" y2="98" class="gt-cell"/>
  <line x1="200" y1="124" x2="330" y2="124" class="gt-cell"/>
  <line x1="226" y1="20" x2="226" y2="150" class="gt-cell"/>
  <line x1="252" y1="20" x2="252" y2="150" class="gt-cell"/>
  <line x1="278" y1="20" x2="278" y2="150" class="gt-cell"/>
  <line x1="304" y1="20" x2="304" y2="150" class="gt-cell"/>
  <!-- tiny circulation arrows in a couple cells -->
  <path d="M 262,58 A 6,6 0 1,1 262,56" stroke="var(--primary, #94452b)" stroke-width="0.6" fill="none"/>
  <path d="M 288,84 A 6,6 0 1,1 288,82" stroke="var(--primary, #94452b)" stroke-width="0.6" fill="none"/>
  <path d="M 236,84 A 6,6 0 1,1 236,82" stroke="var(--primary, #94452b)" stroke-width="0.6" fill="none"/>
  <path d="M 262,110 A 6,6 0 1,1 262,108" stroke="var(--primary, #94452b)" stroke-width="0.6" fill="none"/>
  <text x="210" y="165" class="gt-lb">Circulate around each cell</text>
  <!-- Arrow -->
  <text x="345" y="85" style="font-size:18px; fill: var(--text-primary, #1a1a1a);">→</text>
  <!-- Right: only boundary remains -->
  <path d="M 425,30 Q 490,18 510,50 Q 525,90 505,130 Q 470,155 430,140 Q 395,120 400,80 Q 398,50 425,30 Z" stroke="var(--primary, #94452b)" stroke-width="2" fill="none"/>
  <!-- direction arrows on boundary -->
  <text x="390" y="165" class="gt-lb">Interior cancels → boundary</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The proof of Green's theorem: subdivide the region, circulate around each cell, and interior boundaries cancel.</p>

1. Subdivide $R$ into many tiny rectangular cells.
2. The double integral $\iint_R \operatorname{curl}\mathbf{F}\,\mathrm{d}A$ is approximated by summing $\operatorname{curl}\mathbf{F}(x_i, y_i) \cdot \Delta A$ over all cells.
3. By the definition of curl, each term is approximately the circulation $\oint_{\square_i} \mathbf{F} \cdot \mathrm{d}\mathbf{r}$ around the $i$-th cell.
4. When we add up these tiny circulations, each *interior* boundary segment appears twice — once for each adjacent cell — in opposite directions. They cancel.
5. The only boundary segments that don't cancel are those on the *outer boundary* $C$.
6. Therefore: $\iint_R \operatorname{curl}\mathbf{F}\,\mathrm{d}A \approx \sum_i \oint_{\square_i} \mathbf{F} \cdot \mathrm{d}\mathbf{r} \approx \oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r}$.

### Green's theorem for flux

There is a companion version for flux:

$$
\oint_C \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}s = \iint_R \operatorname{div}\mathbf{F}\,\mathrm{d}A = \iint_R \left(\frac{\partial M}{\partial x} + \frac{\partial N}{\partial y}\right)\mathrm{d}x\,\mathrm{d}y.
$$

The proof is essentially identical, but with outward flux across each cell boundary instead of circulation around it.

### The differential forms perspective

From the viewpoint of differential forms, both versions are the same theorem. If $\phi = M\,\mathrm{d}x + N\,\mathrm{d}y$ is a 1-form, then Green's theorem says

$$
\int_C \phi = \iint_R \mathrm{d}\phi
$$

where $\mathrm{d}\phi = \left(\frac{\partial N}{\partial x} - \frac{\partial M}{\partial y}\right)\mathrm{d}x \wedge \mathrm{d}y$ is the exterior derivative. The circulation version uses the vector line integral form; the flux version uses the flux integral form. Both reduce to integrating the exterior derivative over the region.

### Example: circulation via Green's theorem

Let $\mathbf{F} = x\,\mathbf{i} + xy\,\mathbf{j}$ and let $C$ be the boundary of the region $\{(x,y) : -2 \leq x \leq 2,\ -2 \leq y \leq 2 - x^2\}$, oriented counterclockwise.

Instead of parameterizing $C$ (which has a straight-line piece and a parabolic piece), we use Green's theorem:

$$
\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \iint_R \left(\frac{\partial(xy)}{\partial x} - \frac{\partial(x)}{\partial y}\right)\mathrm{d}A = \iint_R y\,\mathrm{d}A.
$$

Setting up the integral:

$$
\int_{-2}^{2}\int_{-2}^{2-x^2} y\,\mathrm{d}y\,\mathrm{d}x = \int_{-2}^{2}\frac{(2-x^2)^2 - 4}{2}\,\mathrm{d}x = \int_{-2}^{2}\left(\frac{x^4}{2} - 2x^2\right)\mathrm{d}x.
$$

Since $\frac{x^4}{2}$ and $-2x^2$ are both even functions:

$$
= 2\int_0^2 \left(\frac{x^4}{2} - 2x^2\right)\mathrm{d}x = 2\left(\frac{x^5}{10} - \frac{2x^3}{3}\right)\Bigg|_0^2 = 2\left(\frac{32}{10} - \frac{16}{3}\right) = -\frac{64}{15}.
$$

---

## Applications of Green's Theorem

### Area by line integral

Green's theorem can run "backwards": instead of simplifying a line integral into a double integral, we can compute a double integral via a line integral. The most common application is **area**.

The area of a region $R$ is $\iint_R 1\,\mathrm{d}A$. We need a vector field $\mathbf{F}$ with $\operatorname{curl}\mathbf{F} = 1$. A convenient choice is $\mathbf{F} = -\frac{1}{2}y\,\mathbf{i} + \frac{1}{2}x\,\mathbf{j}$, which gives:

$$
\text{Area}(R) = \frac{1}{2}\oint_C x\,\mathrm{d}y - y\,\mathrm{d}x.
$$

**The area of an ellipse.** For the ellipse $\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1$ with parameterization $\mathbf{r}(t) = (a\cos t, b\sin t)$:

$$
\text{Area} = \frac{1}{2}\int_0^{2\pi} \left(a\cos t \cdot b\cos t - b\sin t \cdot (-a\sin t)\right)\mathrm{d}t = \frac{1}{2}\int_0^{2\pi} ab\,\mathrm{d}t = \pi ab.
$$

### The shoelace formula

For a polygon with vertices $(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k)$ listed counterclockwise, applying the same area formula to each line segment and simplifying gives:

$$
\text{Area} = \frac{x_1 y_2 - x_2 y_1 + x_2 y_3 - x_3 y_2 + \cdots + x_k y_1 - x_1 y_k}{2}.
$$

This is the **shoelace formula** — named for the criss-cross pattern of the terms.

### Regions with holes

Green's theorem, as stated, requires $R$ to be simply connected. But we can extend it to regions with holes by carefully orienting the boundaries.

If $R$ is the region between an outer boundary $C_2$ and an inner boundary $C_1$ (like an annulus), then the "boundary" of $R$ consists of $C_2$ oriented counterclockwise and $C_1$ oriented *clockwise*. Green's theorem then gives:

$$
\iint_R \operatorname{curl}\mathbf{F}\,\mathrm{d}A = \oint_{C_2} \mathbf{F} \cdot \mathrm{d}\mathbf{r} - \oint_{C_1} \mathbf{F} \cdot \mathrm{d}\mathbf{r}
$$

where both $C_1$ and $C_2$ are traversed counterclockwise (the minus sign on $C_1$ accounts for the clockwise orientation of the inner boundary).

This extension is essential for handling vector fields with singularities. For the gravity field $\mathbf{F} = -\frac{x\,\mathbf{i} + y\,\mathbf{j}}{(x^2+y^2)^{3/2}}$, which has $\operatorname{curl}\mathbf{F} = 0$ everywhere except the origin, we can show that $\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = 0$ for any closed curve not passing through the origin — either by finding a potential function, or by deforming the curve to a circle where the integral can be computed directly.

---

## Looking Ahead

Green's theorem is the first of several powerful results that relate a boundary integral to an interior integral. In [Part 4](/2024/07/10/calculus-iv-surfaces.html), we move from curves to surfaces: we learn to parameterize surfaces in $\mathbb{R}^3$, compute surface area via cross products, and set the stage for the surface integrals and Stokes' theorem that complete the story.
