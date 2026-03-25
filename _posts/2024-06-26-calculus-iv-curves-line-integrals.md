---
layout: post
title: "Calculus IV: Curves and Line Integrals"
date: 2024-06-26
tags: [Math, Calculus, Multivariable Calculus]
---

In [Part 1](/2024/06/19/calculus-iv-coordinate-systems.html), we developed coordinate systems and the Jacobian for integrating over *regions*. But many natural questions in physics and geometry involve integrating along a *path*: how much work does a force do on a moving object? What is the mass of a wire with varying density? How much does a fluid flow across a boundary?

These are all **line integrals** — a somewhat misleading name, since the "line" is usually a curve, not a straight line. This post covers the two main flavors: scalar line integrals (which accumulate a function's value along a curve) and vector line integrals (which accumulate how much a vector field follows or crosses a curve).

This is Part 2 of a four-part series on Calculus IV.

## What This Post Covers

- **Curves and Parameterizations** — What a curve is, how to parameterize it, and why orientation matters
- **Scalar Line Integrals** — Integrating a function along a curve, the arc length special case, and the mass of a wire
- **Vector Fields** — Definitions, examples, the gradient field, and the concept of work
- **Vector Line Integrals** — The work integral $\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r}$, its formal derivation, and the differential form notation
- **Flux Integrals** — Measuring how much a vector field crosses a curve, and the normal vector construction

---

## Curves and Parameterizations

A **curve** in $\mathbb{R}^n$ is informally a one-dimensional set — it has length but no thickness. Think of a wire, a path through space, or the trajectory of a particle. A curve can be open (with a start and end) or **closed** (ending where it began).

To do calculus on a curve, we need a **parameterization**: a function $\mathbf{r}\colon [a,b] \to \mathbb{R}^n$ whose image traces out the curve $C$. For example, the standard parameterization of the unit circle is

$$
\mathbf{r}(t) = (\cos t, \sin t), \qquad t \in [0, 2\pi].
$$

There are two useful ways to think about this:
- **The particle picture:** $t$ represents time, and $\mathbf{r}(t)$ is the position of a particle at time $t$. As $t$ increases from $a$ to $b$, the particle traces out the curve.
- **The coordinate picture:** $t$ is like a GPS coordinate for creatures living on the curve. The function $\mathbf{r}$ converts this 1D coordinate into a position in $\mathbb{R}^n$.

The same curve can have many parameterizations — sometimes very different ones! A curve also carries an **orientation**: the direction of travel. Choosing a parameterization $\mathbf{r}(t)$ implicitly chooses an orientation (the direction of increasing $t$).

### Building parameterizations

**From a function graph.** If a curve is described by $y = f(x)$ for $a \leq x \leq b$, we can simply set $\mathbf{r}(t) = (t, f(t))$ for $t \in [a,b]$.

**From coordinate systems.** Alternate coordinates can suggest parameterizations. For a spiral winding around the cylinder $x^2 + y^2 = 1$, climbing from $(1,0,0)$ to $(1,0,4)$ in one full turn, we use cylindrical coordinates: $r = 1$, $\theta = t$, $z = \frac{2t}{\pi}$. Converting to rectangular:

$$
\mathbf{r}(t) = \left(\cos t, \sin t, \frac{2t}{\pi}\right), \qquad t \in [0, 2\pi].
$$

**Line segments.** The segment from point $\mathbf{a}$ to point $\mathbf{b}$ has a clean parameterization as a weighted average:

$$
\mathbf{r}(t) = (1-t)\,\mathbf{a} + t\,\mathbf{b}, \qquad t \in [0,1].
$$

At $t = 0$ we are at $\mathbf{a}$; at $t = 1$ we are at $\mathbf{b}$; at $t = 1/2$ we are at the midpoint.

**Piecewise curves.** When a curve has corners or changes behavior, we can break it into pieces and parameterize each one separately. For example, the boundary of the unit square (counterclockwise from the origin) splits into four line segments, each parameterized on $[0,1]$.

**Transformations.** Parameterizations behave nicely under geometric transformations. To get a circle of radius 2 centered at $(1, 3)$, start with $\mathbf{r}(t) = (\cos t, \sin t)$ and apply the transformation: $\mathbf{r}(t) = (2\cos t + 1, 2\sin t + 3)$.

---

## Scalar Line Integrals

### The wall visualization

We start with the scalar line integral in $\mathbb{R}^2$, which has a nice geometric picture. Given a function $f\colon \mathbb{R}^2 \to \mathbb{R}$ and a curve $C$ in the $xy$-plane, imagine plotting the surface $z = f(x,y)$ in 3D. Above the curve $C$, the surface traces out a 3D curve $C'$. Imagine a wall built between $C$ and $C'$ — like a curved fence whose height at each point $(x,y)$ is $f(x,y)$.

The scalar line integral $\int_C f(x,y)\,\mathrm{d}s$ measures the **area of this wall** (at least when $f$ is positive).

As a special case, when $f(x,y) = 1$, the wall has constant height 1, so its area equals the **arc length** of $C$. This is why $\int_C 1\,\mathrm{d}s$ is called an **arc length integral**.

### Formal definition

Let $C$ be parameterized by $\mathbf{r}(t)$ for $t \in [a,b]$. Divide $[a,b]$ into tiny intervals $[t_i, t_{i+1}]$ of width $\Delta t$. For each interval, approximate the wall's area by a tiny rectangle:

- **Height:** $f(\mathbf{r}(t_i))$
- **Width:** $\|\mathbf{r}(t_{i+1}) - \mathbf{r}(t_i)\| \approx \left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\| \Delta t$

Summing these rectangles and taking the limit gives the definition:

$$
\boxed{\int_C f(x,y)\,\mathrm{d}s := \int_{t=a}^{b} f(\mathbf{r}(t)) \left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\| \mathrm{d}t.}
$$

The factor $\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\|$ converts a change in the parameter $t$ into a change in arc length along the curve. This is what makes the integral independent of parameterization: if you parameterize the curve twice as fast, $\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\|$ doubles, but $\Delta t$ halves, and the product stays the same.

> A parameterization where $\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\| = 1$ everywhere is called a **natural parameterization** or **unit-speed parameterization**. In this case, $t$ directly measures arc length, and the formula simplifies to $\int_a^b f(\mathbf{r}(t))\,\mathrm{d}t$.

### Example: integrating $x^2$ over the unit circle

Using $\mathbf{r}(t) = (\cos t, \sin t)$, we have $\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = (-\sin t, \cos t)$, so $\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\| = 1$ (a natural parameterization!). With $f(x,y) = x^2$:

$$
\int_C x^2\,\mathrm{d}s = \int_0^{2\pi} \cos^2 t \cdot 1\,\mathrm{d}t = \int_0^{2\pi}\frac{1+\cos 2t}{2}\,\mathrm{d}t = \pi.
$$

### Scalar line integrals in 3D and the space dust analogy

The definition in $\mathbb{R}^3$ is identical: $\int_C f(x,y,z)\,\mathrm{d}s = \int_a^b f(\mathbf{r}(t))\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\|\mathrm{d}t$. We lose the wall visualization (since $w = f(x,y,z)$ would require four dimensions to plot), but we gain a nice analogy: imagine a spaceship traveling through a region of space dust. At each point, $f(x,y,z)$ measures the dust density. The scalar line integral is the total amount of dust the spaceship collects along its path.

> An important subtlety: the scalar line integral does *not* depend on how fast the spaceship travels. A spaceship that lingers at a dusty point does not collect more dust, because it has already collected the dust when it first passed through. (Think of the dust as a finite resource that gets scooped up.)

### Mass of a wire

A physical application: if a thin wire has the shape of a curve $C$ and its **mass per unit length** at a point is $\delta$, then the total mass is

$$
\text{Mass} = \int_C \delta\,\mathrm{d}s.
$$

---

## Vector Fields

A **vector field** in $\mathbb{R}^2$ is a function $\mathbf{F}\colon \mathbb{R}^2 \to \mathbb{R}^2$ that assigns a vector to every point. In $\mathbb{R}^3$, it is a function $\mathbf{F}\colon \mathbb{R}^3 \to \mathbb{R}^3$. We write

$$
\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j} + P\,\mathbf{k}
$$

where $M$, $N$, $P$ are scalar functions. The standard visualization is a plot of arrows: at each sample point, draw an arrow in the direction and (proportional to the) magnitude of $\mathbf{F}$.

<svg viewBox="0 0 300 300" style="max-width:300px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .vf-ax { stroke: var(--text-primary, #1a1a1a); stroke-width: 1; }
    .vf-lb { font-family: 'Newsreader', serif; font-size: 13px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .vf-arrow { stroke: var(--primary, #94452b); stroke-width: 1.2; fill: none; marker-end: url(#vfa); }
  </style>
  <defs>
    <marker id="vfa" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto">
      <path d="M0,0 L5,2 L0,4" fill="var(--primary, #94452b)" stroke="none"/>
    </marker>
  </defs>
  <!-- axes -->
  <line x1="10" y1="150" x2="290" y2="150" class="vf-ax"/>
  <line x1="150" y1="290" x2="150" y2="10" class="vf-ax"/>
  <text x="280" y="145" class="vf-lb">x</text>
  <text x="155" y="20" class="vf-lb">y</text>
  <!-- F = x i + y j arrows radiating outward -->
  <!-- Row y=2 (top) -->
  <line x1="60" y1="50" x2="52" y2="42" class="vf-arrow"/>
  <line x1="110" y1="50" x2="106" y2="42" class="vf-arrow"/>
  <line x1="150" y1="50" x2="150" y2="40" class="vf-arrow"/>
  <line x1="190" y1="50" x2="194" y2="42" class="vf-arrow"/>
  <line x1="240" y1="50" x2="248" y2="42" class="vf-arrow"/>
  <!-- Row y=1 -->
  <line x1="60" y1="100" x2="54" y2="95" class="vf-arrow"/>
  <line x1="110" y1="100" x2="107" y2="95" class="vf-arrow"/>
  <line x1="150" y1="100" x2="150" y2="94" class="vf-arrow"/>
  <line x1="190" y1="100" x2="193" y2="95" class="vf-arrow"/>
  <line x1="240" y1="100" x2="246" y2="95" class="vf-arrow"/>
  <!-- Row y=0 (middle) -->
  <line x1="60" y1="150" x2="52" y2="150" class="vf-arrow"/>
  <line x1="110" y1="150" x2="105" y2="150" class="vf-arrow"/>
  <line x1="190" y1="150" x2="195" y2="150" class="vf-arrow"/>
  <line x1="240" y1="150" x2="248" y2="150" class="vf-arrow"/>
  <!-- Row y=-1 -->
  <line x1="60" y1="200" x2="54" y2="205" class="vf-arrow"/>
  <line x1="110" y1="200" x2="107" y2="205" class="vf-arrow"/>
  <line x1="150" y1="200" x2="150" y2="206" class="vf-arrow"/>
  <line x1="190" y1="200" x2="193" y2="205" class="vf-arrow"/>
  <line x1="240" y1="200" x2="246" y2="205" class="vf-arrow"/>
  <!-- Row y=-2 (bottom) -->
  <line x1="60" y1="250" x2="52" y2="258" class="vf-arrow"/>
  <line x1="110" y1="250" x2="106" y2="258" class="vf-arrow"/>
  <line x1="150" y1="250" x2="150" y2="260" class="vf-arrow"/>
  <line x1="190" y1="250" x2="194" y2="258" class="vf-arrow"/>
  <line x1="240" y1="250" x2="248" y2="258" class="vf-arrow"/>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The vector field $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j}$: every arrow points away from the origin, with magnitude proportional to distance.</p>

Common physical examples:

- **Velocity fields:** $\mathbf{F}(x,y)$ gives the velocity of a fluid (air, water) at the point $(x,y)$. Wind maps are vector fields.
- **Force fields:** $\mathbf{F}(x,y,z)$ gives the force (gravity, electromagnetism) acting on an object at $(x,y,z)$.

### The gradient field

Given a scalar function $f\colon \mathbb{R}^3 \to \mathbb{R}$, its **gradient** is the vector field

$$
\nabla f = \frac{\partial f}{\partial x}\,\mathbf{i} + \frac{\partial f}{\partial y}\,\mathbf{j} + \frac{\partial f}{\partial z}\,\mathbf{k}.
$$

The gradient points in the direction where $f$ increases the fastest, and its magnitude is the rate of increase in that direction. If $f(x,y)$ represents elevation on a topographic map, then $\nabla f$ points uphill, and $-\nabla f$ is the direction water flows.

The vector field $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j}$ is the gradient of $f(x,y) = \frac{1}{2}(x^2 + y^2)$. Not every vector field is a gradient field — determining which ones are is a deep question we tackle in [Part 3](/2024/07/03/calculus-iv-greens-theorem.html).

---

## Vector Line Integrals and Work

### The physics motivation

Imagine a force field $\mathbf{F}$ pushing on an object as it moves along a curve $C$. The **work** done by $\mathbf{F}$ is a measure of how much the force helped (or hindered) the motion.

For a constant force $\mathbf{F}$ and a straight-line displacement $\mathbf{s}$, the work is $\mathbf{F} \cdot \mathbf{s}$:
- If the object moves parallel to $\mathbf{F}$, work is $\|\mathbf{F}\|\,\|\mathbf{s}\|$ (the force helps fully).
- If the object moves perpendicular to $\mathbf{F}$, work is $0$ (the force neither helps nor hinders).
- If the object moves opposite to $\mathbf{F}$, work is negative (the force resists the motion).

For a varying force along a curve, we divide $C$ into tiny segments. Over each tiny segment $[\mathbf{r}(t_i), \mathbf{r}(t_{i+1})]$, the force is approximately $\mathbf{F}(\mathbf{r}(t_i))$ and the displacement is approximately $\mathbf{r}(t_{i+1}) - \mathbf{r}(t_i) \approx \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\,\Delta t$. The work over this segment is approximately

$$
\mathbf{F}(\mathbf{r}(t_i)) \cdot \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}(t_i)\,\Delta t.
$$

Summing and taking the limit:

$$
\boxed{\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} := \int_{t=a}^{b} \mathbf{F}(\mathbf{r}(t)) \cdot \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\,\mathrm{d}t.}
$$

This is the **vector line integral** (or **work integral**). Unlike the scalar line integral, it depends on the *orientation* of the curve: reversing direction flips the sign.

Writing $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j} + P\,\mathbf{k}$ and $\mathbf{r}(t) = (x(t), y(t), z(t))$, the integrand expands to

$$
\mathbf{F} \cdot \frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = M\frac{\mathrm{d}x}{\mathrm{d}t} + N\frac{\mathrm{d}y}{\mathrm{d}t} + P\frac{\mathrm{d}z}{\mathrm{d}t}.
$$

This motivates the shorthand $\int_C M\,\mathrm{d}x + N\,\mathrm{d}y + P\,\mathrm{d}z$, where $\mathrm{d}x$ stands for $\frac{\mathrm{d}x}{\mathrm{d}t}\,\mathrm{d}t$, and so on. The expression $M\,\mathrm{d}x + N\,\mathrm{d}y + P\,\mathrm{d}z$ is called a **differential form** (specifically, a 1-form).

### Example: work done by a radial force

Let $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}$ (a force pushing outward from the origin) and let a particle move along $\mathbf{r}(t) = (t, t^2, t^3)$ from $(0,0,0)$ to $(1,1,1)$.

We compute $\mathbf{F}(\mathbf{r}(t)) = t\,\mathbf{i} + t^2\,\mathbf{j} + t^3\,\mathbf{k}$ and $\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = \mathbf{i} + 2t\,\mathbf{j} + 3t^2\,\mathbf{k}$. Their dot product is

$$
t \cdot 1 + t^2 \cdot 2t + t^3 \cdot 3t^2 = t + 2t^3 + 3t^5.
$$

Integrating:

$$
\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \int_0^1 (t + 2t^3 + 3t^5)\,\mathrm{d}t = \frac{1}{2} + \frac{1}{2} + \frac{1}{2} = \frac{3}{2}.
$$

An interesting fact: if we take the *straight-line* path $\mathbf{r}(t) = (t, t, t)$ from $(0,0,0)$ to $(1,1,1)$ instead, we get $\int_0^1 3t\,\mathrm{d}t = \frac{3}{2}$ — the same answer. This is not a coincidence! The force $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j} + z\,\mathbf{k}$ is a **conservative** force (it is the gradient of $f = \frac{1}{2}(x^2 + y^2 + z^2)$), and for conservative forces, the work depends only on the endpoints. We explore this deeply in [Part 3](/2024/07/03/calculus-iv-greens-theorem.html).

### Scalar vs. vector line integrals

The two types of line integrals are closely related. If $\mathbf{T}(t) = \frac{\mathrm{d}\mathbf{r}/\mathrm{d}t}{\|\mathrm{d}\mathbf{r}/\mathrm{d}t\|}$ is the unit tangent vector to $C$, then

$$
\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \int_C \mathbf{F} \cdot \mathbf{T}\,\mathrm{d}s.
$$

The vector line integral is a scalar line integral of the function $f = \mathbf{F} \cdot \mathbf{T}$ — the component of $\mathbf{F}$ along the curve.

---

## Flux and the Flux Integral

The vector line integral measures how much $\mathbf{F}$ *follows* a curve. There is a companion integral that measures how much $\mathbf{F}$ *crosses* a curve.

### The idea of flux

**Flux** measures the amount of a vector field that passes through a boundary. Imagine stretching a net in a body of water: the flux of the current across the net is the total rate at which water passes through.

In $\mathbb{R}^2$, a boundary is a curve $C$. The flux of $\mathbf{F}$ across $C$ is signed: crossing $C$ from left to right (relative to the direction of travel along $C$) counts as positive flux.

<svg viewBox="0 0 460 160" style="max-width:460px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .fl-ax { stroke: var(--text-primary, #1a1a1a); stroke-width: 0.8; }
    .fl-lb { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .fl-curve { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
    .fl-arrow { stroke: var(--primary, #94452b); stroke-width: 1; marker-end: url(#fla); }
    .fl-vec { stroke: #4a7ab5; stroke-width: 1.2; marker-end: url(#flb); }
  </style>
  <defs>
    <marker id="fla" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4" fill="var(--primary, #94452b)" stroke="none"/></marker>
    <marker id="flb" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4" fill="#4a7ab5" stroke="none"/></marker>
  </defs>
  <!-- Left: positive flux -->
  <path d="M 30,130 Q 80,20 130,80 Q 160,120 200,50" class="fl-curve"/>
  <line x1="80" y1="40" x2="110" y2="25" class="fl-vec"/>
  <line x1="120" y1="75" x2="150" y2="55" class="fl-vec"/>
  <line x1="155" y1="105" x2="185" y2="85" class="fl-vec"/>
  <text x="60" y="155" class="fl-lb">Positive flux (left → right)</text>
  <!-- Right: negative flux -->
  <path d="M 260,130 Q 310,20 360,80 Q 390,120 430,50" class="fl-curve"/>
  <line x1="340" y1="55" x2="310" y2="40" class="fl-vec"/>
  <line x1="350" y1="75" x2="320" y2="55" class="fl-vec"/>
  <line x1="380" y1="105" x2="350" y2="85" class="fl-vec"/>
  <text x="290" y="155" class="fl-lb">Negative flux (right → left)</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Positive flux: the field crosses from left to right. Negative flux: right to left.</p>

For a closed curve bounding a region $R$, the standard convention is to orient $C$ counterclockwise, so that "positive" flux means *outward* — the field is leaving the region.

### Defining the flux integral

The vector line integral used the dot product $\mathbf{F} \cdot \mathbf{T}$ (the tangential component of $\mathbf{F}$). The flux integral uses $\mathbf{F} \cdot \mathbf{n}$, the **normal** component.

Given the unit tangent vector $\mathbf{T}$, we obtain the unit normal $\mathbf{n}$ by rotating $\mathbf{T}$ clockwise by $90°$. If $\mathbf{T} = a\,\mathbf{i} + b\,\mathbf{j}$, then $\mathbf{n} = b\,\mathbf{i} - a\,\mathbf{j}$. We can write this rotation as the cross product $\mathbf{T} \times \mathbf{k}$.

Since $\mathbf{n}$ is a scaled version of $\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} \times \mathbf{k}$, the flux integral works out to:

$$
\int_C \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}s = \int_{t=a}^{b} \mathbf{F}(\mathbf{r}(t)) \cdot \left(\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} \times \mathbf{k}\right)\,\mathrm{d}t.
$$

Expanding with $\mathbf{F} = M\,\mathbf{i} + N\,\mathbf{j}$ and $\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} = \frac{\mathrm{d}x}{\mathrm{d}t}\,\mathbf{i} + \frac{\mathrm{d}y}{\mathrm{d}t}\,\mathbf{j}$:

$$
\boxed{\int_C \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}s = \int_C M\,\mathrm{d}y - N\,\mathrm{d}x.}
$$

### Example: flux of a constant field across the unit circle

Let $\mathbf{F} = 2\,\mathbf{i} + 3\,\mathbf{j}$ and let $C$ be the counterclockwise unit circle. Parameterize $C$ by $\mathbf{r}(t) = (\cos t, \sin t)$, so $\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t} \times \mathbf{k} = \cos t\,\mathbf{i} + \sin t\,\mathbf{j}$ (pointing radially outward — which makes sense for outward flux). Then:

$$
\int_0^{2\pi} (2\cos t + 3\sin t)\,\mathrm{d}t = (2\sin t - 3\cos t)\Big|_0^{2\pi} = 0.
$$

The outward flux is zero. Intuitively, a constant field sends as much fluid *into* the circle on one side as it sends *out* on the other.

### Example: flux of an expanding field across a triangle

Now let $\mathbf{F} = x\,\mathbf{i} + y\,\mathbf{j}$ (the expanding field from our earlier example), and let $C$ be the counterclockwise boundary of the triangle with vertices $(1,-1)$, $(0,1)$, $(-1,-1)$.

Computing the flux integral piece by piece using $\int_C M\,\mathrm{d}y - N\,\mathrm{d}x = \int_C x\,\mathrm{d}y - y\,\mathrm{d}x$:

- **Side from $(1,-1)$ to $(0,1)$:** Parameterize by $\mathbf{r}(t) = (1-t, 2t-1)$ for $t \in [0,1]$. We get $\int_0^1 ((1-t)(2) - (2t-1)(-1))\,\mathrm{d}t = \int_0^1 1\,\mathrm{d}t = 1$.
- **Side from $(0,1)$ to $(-1,-1)$:** By symmetry with the first side, this also gives $1$.
- **Base from $(-1,-1)$ to $(1,-1)$:** Here $\mathbf{n} = -\mathbf{j}$ and $\mathbf{F} \cdot \mathbf{n} = -y = 1$ everywhere on this side, with arc length $2$. So the flux is $2$.

Total outward flux: $1 + 1 + 2 = 4$.

---

## Looking Ahead

We have built two powerful types of line integrals: the vector line integral $\int_C \mathbf{F} \cdot \mathrm{d}\mathbf{r}$ measuring how much a field *follows* a curve, and the flux integral $\int_C \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}s$ measuring how much a field *crosses* a curve.

In [Part 3](/2024/07/03/calculus-iv-greens-theorem.html), we discover the remarkable fact that conservative vector fields make these integrals path-independent, and Green's theorem connects line integrals around a closed curve to double integrals over the enclosed region — linking everything back to the integration techniques from Part 1.
