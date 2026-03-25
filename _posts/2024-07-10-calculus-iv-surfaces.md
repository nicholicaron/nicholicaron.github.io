---
layout: post
title: "Calculus IV: Surfaces and Surface Area"
date: 2024-07-10
tags: [Math, Calculus, Multivariable Calculus]
---

In the first three parts of this series, we integrated over regions ([Part 1](/2024/06/19/calculus-iv-coordinate-systems.html)), over curves ([Part 2](/2024/06/26/calculus-iv-curves-line-integrals.html)), and connected the two via Green's theorem ([Part 3](/2024/07/03/calculus-iv-greens-theorem.html)). Now we take the next step: integrating over **surfaces** in $\mathbb{R}^3$.

A surface is to three dimensions what a curve is to two dimensions — a boundary that separates regions of space. Just as we parameterized curves to set up line integrals, we need to parameterize surfaces to set up surface integrals. And just as the arc length element $\mathrm{d}s$ involved the derivative of the parameterization, the surface area element will involve the **cross product** of partial derivatives.

This post develops the machinery of surface parameterizations and surface area. It sets the stage for surface integrals and Stokes' theorem — the full generalization of everything we have built.

This is Part 4 of a four-part series on Calculus IV.

## What This Post Covers

- **Parameterizing Surfaces** — The rubber-sheet and GPS intuitions, rectangular vs. wacky domains
- **Building Parameterizations** — From function graphs, coordinate systems, and transformations
- **The Möbius Strip** — A non-orientable surface and its elegant parameterization
- **Cross Products** — A review of the key tool for surface area
- **Surface Area** — Deriving the integral formula and computing examples
- **Surface Area of a Sphere** — A slick Jacobian trick that avoids a painful direct computation

---

## Parameterizing Surfaces

When we parameterized curves, we used functions $\mathbf{r}\colon [a,b] \to \mathbb{R}^3$ — one parameter $t$ tracing out a one-dimensional path. For surfaces, we need **two parameters**. A parameterization of a surface $S$ is a function

$$
\mathbf{r}\colon D \to \mathbb{R}^3, \qquad \mathbf{r}(u,v) = (x(u,v),\ y(u,v),\ z(u,v))
$$

where $D \subseteq \mathbb{R}^2$ is the domain, and the image $S = \{\mathbf{r}(u,v) : (u,v) \in D\}$ is the surface.

### Three ways to think about it

**The rubber sheet.** Imagine $D$ as a flat rectangular sheet of rubber. The parameterization $\mathbf{r}$ tells you how to bend and stretch that sheet to fit the shape of the surface. A flat piece of rubber becomes a hemisphere, a paraboloid, or a Möbius strip.

**The GPS coordinates.** Imagine tiny creatures living on the surface. They describe their location using two numbers $(u, v)$ — like latitude and longitude on Earth. The parameterization $\mathbf{r}$ is the function that converts these local coordinates into actual positions in 3D space.

**The family of curves.** Fix one parameter and vary the other. For each constant $u_0$, the function $v \mapsto \mathbf{r}(u_0, v)$ is a curve on the surface. As $u_0$ changes, the curve sweeps out the surface. (And similarly for fixed $v$.) This is exactly how we think of surface parameterizations built from cylindrical or spherical coordinates.

### Rectangular vs. wacky domains

We prefer the domain $D$ to be a **rectangle** $[a,b] \times [c,d]$, because this makes setting up double integrals straightforward. A parameterization with a rectangular domain is called a **rectangular-domain parameterization**.

When $D$ is some other shape (a disk, a triangle, etc.), we call it a **wacky-domain parameterization**. These work fine in theory, but they make integration harder and describing boundaries trickier. Always check whether a clever choice of coordinates can make the domain rectangular.

### Injectivity

A good parameterization should be *mostly* injective: different points $(u,v)$ should map to different points on the surface. Failures on the boundary of $D$ are acceptable (just as $\mathbf{r}(0) = \mathbf{r}(2\pi)$ was fine for curves). But failures in the interior mean the surface is self-intersecting, which causes formulas to break.

---

## Building Parameterizations

### From function graphs

The simplest case: a surface given by $z = f(x,y)$ over a region in the $xy$-plane. Just set $x = u$, $y = v$:

$$
\mathbf{r}(u,v) = (u, v, f(u,v)).
$$

For example, the paraboloid $z = x^2 + y^2$ over the disk $x^2 + y^2 \leq 3$ can be parameterized as $\mathbf{r}(u,v) = (u, v, u^2 + v^2)$ — but this is a wacky-domain parameterization (the domain is a disk). Better: switch to cylindrical coordinates. Setting $r = u$, $\theta = v$:

$$
\mathbf{r}(u,v) = (u\cos v, u\sin v, u^2), \qquad (u,v) \in [0, \sqrt{3}] \times [0, 2\pi].
$$

Now the domain is a rectangle.

### From coordinate systems

Whenever a surface has a nice description in cylindrical or spherical coordinates, we can use the coordinate conversion as a parameterization.

**The hemisphere** $x^2 + y^2 + z^2 = \rho^2$, $z \geq 0$ is described in spherical coordinates by $\rho = \text{const}$, $0 \leq \phi \leq \pi/2$, $0 \leq \theta \leq 2\pi$. Using $\phi$ and $\theta$ as parameters:

$$
\mathbf{r}(\phi, \theta) = (\rho\sin\phi\cos\theta,\ \rho\sin\phi\sin\theta,\ \rho\cos\phi), \qquad (\phi, \theta) \in [0, \pi/2] \times [0, 2\pi].
$$

> Remember: the parameterization must always output rectangular coordinates $(x, y, z)$. Cylindrical and spherical coordinates are used as a *source of inspiration* for choosing $u$ and $v$, not as the output format.

### Transformations

Just as with curves, we can modify parameterizations with geometric transformations:

- **Translation:** add constants to shift the surface.
- **Scaling:** multiply components by constants to stretch or compress.
- **Rotation:** apply a rotation matrix to the output.

For example, starting with a paraboloid $\mathbf{r}(u,v) = (u\cos v, u\sin v, u^2)$, we can rotate it $45°$ around the $x$-axis by applying the rotation matrix $\begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos 45° & -\sin 45° \\ 0 & \sin 45° & \cos 45° \end{bmatrix}$ to each point.

---

## The Möbius Strip

One of the most fascinating surfaces in mathematics is the **Möbius strip**: take a long rectangular strip of paper, give it a half-twist, and tape the ends together. The result is a surface with only *one side* — an ant crawling along the strip will traverse both "sides" before returning to its starting point.

We can build the parameterization step by step.

**Step 1: A non-twisted strip.** Start with a cylinder of radius 5. A strip of paper wrapped around it (without twisting) is parameterized by:

$$
\mathbf{r}(u,v) = (5\cos u, 5\sin u, v), \qquad (u,v) \in [0, 2\pi] \times [-1, 1].
$$

Here $u$ is the angle around the cylinder, and $v$ runs from the bottom of the strip ($v = -1$) to the top ($v = 1$).

**Step 2: Add the twist.** Think of the strip as sweeping out a line segment $\mathbf{r}_u(v)$ for each angle $u$. Currently each segment is vertical (parallel to the $z$-axis). To create a Möbius strip, we need this segment to *rotate* as it goes around the cylinder.

For a Möbius strip, the segment makes a half-twist (180°) as $u$ goes from $0$ to $2\pi$. So the rotation angle should be $\alpha = u/2$. In the $rz$-half-plane (cylindrical coordinates), the segment at angle $u$ connects $(r,z) = (5 - v\sin(u/2),\ v\cos(u/2))$.

Converting to rectangular coordinates:

$$
\boxed{\mathbf{r}(u,v) = \left(\left(5 - v\sin\frac{u}{2}\right)\cos u,\ \left(5 - v\sin\frac{u}{2}\right)\sin u,\ v\cos\frac{u}{2}\right)}
$$

for $(u,v) \in [0, 2\pi] \times [-1, 1]$.

The Möbius strip is a **non-orientable surface** — you cannot consistently define "inside" and "outside" (or, equivalently, a continuously varying normal vector). This has profound consequences: Stokes' theorem, which requires a compatible orientation between a surface and its boundary, cannot be applied to the Möbius strip. Non-orientability is not a bug — it is a genuine topological property.

---

## Cross Products

Before we can compute surface area, we need one key tool from linear algebra: the **cross product**.

Given vectors $\mathbf{a} = a_1\,\mathbf{i} + a_2\,\mathbf{j} + a_3\,\mathbf{k}$ and $\mathbf{b} = b_1\,\mathbf{i} + b_2\,\mathbf{j} + b_3\,\mathbf{k}$, their cross product is:

$$
\mathbf{a} \times \mathbf{b} = \det \begin{bmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{bmatrix} = (a_2 b_3 - a_3 b_2)\,\mathbf{i} + (a_3 b_1 - a_1 b_3)\,\mathbf{j} + (a_1 b_2 - a_2 b_1)\,\mathbf{k}.
$$

Two essential properties:

1. **$\mathbf{a} \times \mathbf{b}$ is perpendicular to both $\mathbf{a}$ and $\mathbf{b}$.** The direction is determined by the right-hand rule.
2. **$\|\mathbf{a} \times \mathbf{b}\|$ is the area of the parallelogram** with sides $\mathbf{a}$ and $\mathbf{b}$.

The second property is the key to surface area.

> The cross product is **anti-commutative**: $\mathbf{b} \times \mathbf{a} = -(\mathbf{a} \times \mathbf{b})$. Swapping the order flips the direction (and changes which side of the surface is "up").

---

## Surface Area

### Deriving the formula

The idea parallels the derivation of line integrals. We want to measure the area of a surface $S$ parameterized by $\mathbf{r}(u,v)$ over a rectangular domain $[a,b] \times [c,d]$.

<svg viewBox="0 0 480 180" style="max-width:480px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .sa-lb { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .sa-lbi { font-family: 'Newsreader', serif; font-size: 12px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .sa-cell { stroke: var(--text-primary, #1a1a1a); stroke-width: 0.5; fill: none; }
    .sa-hl { fill: var(--primary, #94452b); opacity: 0.2; stroke: var(--primary, #94452b); stroke-width: 1.5; }
    .sa-vec { stroke: var(--primary, #94452b); stroke-width: 1.5; marker-end: url(#saa); }
  </style>
  <defs>
    <marker id="saa" markerWidth="5" markerHeight="4" refX="5" refY="2" orient="auto"><path d="M0,0 L5,2 L0,4" fill="var(--primary, #94452b)" stroke="none"/></marker>
  </defs>
  <!-- Left: uv domain with grid -->
  <rect x="20" y="20" width="140" height="140" class="sa-cell"/>
  <line x1="55" y1="20" x2="55" y2="160" class="sa-cell"/>
  <line x1="90" y1="20" x2="90" y2="160" class="sa-cell"/>
  <line x1="125" y1="20" x2="125" y2="160" class="sa-cell"/>
  <line x1="20" y1="55" x2="160" y2="55" class="sa-cell"/>
  <line x1="20" y1="90" x2="160" y2="90" class="sa-cell"/>
  <line x1="20" y1="125" x2="160" y2="125" class="sa-cell"/>
  <!-- highlight one cell -->
  <rect x="90" y="90" width="35" height="35" class="sa-hl"/>
  <text x="60" y="175" class="sa-lb">uv-domain</text>
  <text x="22" y="16" class="sa-lbi">v</text>
  <text x="162" y="165" class="sa-lbi">u</text>
  <!-- Arrow -->
  <text x="180" y="95" style="font-size:16px; fill: var(--text-primary, #1a1a1a);">r →</text>
  <!-- Right: curved surface with parallelogram -->
  <path d="M 260,140 Q 300,150 360,120 Q 420,80 440,30" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.8" fill="none"/>
  <path d="M 250,100 Q 300,105 360,85 Q 420,55 450,20" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.8" fill="none"/>
  <path d="M 280,30 Q 290,70 295,100 Q 300,130 310,155" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" fill="none" stroke-dasharray="3,2"/>
  <path d="M 340,20 Q 350,55 355,80 Q 360,105 370,140" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.5" fill="none" stroke-dasharray="3,2"/>
  <!-- parallelogram approximation -->
  <polygon points="330,95 365,80 375,100 340,115" class="sa-hl"/>
  <!-- tangent vectors -->
  <line x1="330" y1="95" x2="365" y2="80" class="sa-vec"/>
  <line x1="330" y1="95" x2="340" y2="115" class="sa-vec"/>
  <text x="360" y="74" class="sa-lbi" font-size="10">∂r/∂u</text>
  <text x="342" y="127" class="sa-lbi" font-size="10">∂r/∂v</text>
  <text x="320" y="175" class="sa-lb">surface S</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Each small rectangle in the $uv$-domain maps to an approximate parallelogram on the surface, with sides $\frac{\partial\mathbf{r}}{\partial u}\,\Delta u$ and $\frac{\partial\mathbf{r}}{\partial v}\,\Delta v$.</p>

Divide the domain into tiny cells of size $\Delta u \times \Delta v$. Each cell maps to a tiny patch of the surface. For small $\Delta u$ and $\Delta v$, this patch is approximately a **parallelogram** with sides:

$$
\mathbf{a} = \frac{\partial\mathbf{r}}{\partial u}\,\Delta u, \qquad \mathbf{b} = \frac{\partial\mathbf{r}}{\partial v}\,\Delta v.
$$

The area of a parallelogram with sides $\mathbf{a}$ and $\mathbf{b}$ is $\|\mathbf{a} \times \mathbf{b}\|$. Factoring out $\Delta u\,\Delta v$ (using linearity of the cross product):

$$
\text{Area of patch} \approx \left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\| \Delta u\,\Delta v.
$$

Summing over all patches and taking the limit:

$$
\boxed{\text{Surface area} = \int_a^b \int_c^d \left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\| \mathrm{d}v\,\mathrm{d}u.}
$$

The quantity $\left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\|$ plays the same role for surfaces that $\left\|\frac{\mathrm{d}\mathbf{r}}{\mathrm{d}t}\right\|$ plays for curves: it converts from parameter-space area to surface area.

### Example: area of a saddle surface

Consider $z = xy$ over $[-1,1] \times [-1,1]$, parameterized by $\mathbf{r}(u,v) = (u, v, uv)$. We compute:

$$
\frac{\partial\mathbf{r}}{\partial u} = \mathbf{i} + v\,\mathbf{k}, \qquad \frac{\partial\mathbf{r}}{\partial v} = \mathbf{j} + u\,\mathbf{k}.
$$

Their cross product:

$$
(\mathbf{i} + v\,\mathbf{k}) \times (\mathbf{j} + u\,\mathbf{k}) = \det\begin{bmatrix}\mathbf{i} & \mathbf{j} & \mathbf{k} \\ 1 & 0 & v \\ 0 & 1 & u\end{bmatrix} = -v\,\mathbf{i} - u\,\mathbf{j} + \mathbf{k}.
$$

So $\left\|\frac{\partial\mathbf{r}}{\partial u} \times \frac{\partial\mathbf{r}}{\partial v}\right\| = \sqrt{v^2 + u^2 + 1} = \sqrt{1 + u^2 + v^2}$, and the surface area is

$$
\int_{-1}^{1}\int_{-1}^{1} \sqrt{1 + u^2 + v^2}\,\mathrm{d}v\,\mathrm{d}u.
$$

This integral, while not elementary to evaluate in closed form, can be computed numerically or with the help of a computer algebra system.

---

## Surface Area of a Sphere

Let's find the surface area of a sphere of radius $\rho$ centered at the origin. The spherical-coordinate parameterization is

$$
\mathbf{r}(\phi, \theta) = (\rho\sin\phi\cos\theta,\ \rho\sin\phi\sin\theta,\ \rho\cos\phi), \qquad (\phi, \theta) \in [0, \pi] \times [0, 2\pi].
$$

### The direct computation

Computing the cross product $\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta}$ by brute force:

$$
\frac{\partial\mathbf{r}}{\partial\phi} = (\rho\cos\phi\cos\theta,\ \rho\cos\phi\sin\theta,\ -\rho\sin\phi),
$$

$$
\frac{\partial\mathbf{r}}{\partial\theta} = (-\rho\sin\phi\sin\theta,\ \rho\sin\phi\cos\theta,\ 0).
$$

Their cross product simplifies (after considerable algebra) to

$$
\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta} = \rho^2\cos\theta\sin^2\phi\,\mathbf{i} + \rho^2\sin\theta\sin^2\phi\,\mathbf{j} + \rho^2\sin\phi\cos\phi\,\mathbf{k}.
$$

Taking the norm involves simplifications like $\cos^2\theta\sin^4\phi + \sin^2\theta\sin^4\phi = \sin^4\phi$, and ultimately:

$$
\left\|\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta}\right\| = \rho^2\sin\phi.
$$

### The elegant Jacobian trick

There is a much slicker way to arrive at the same answer. Recall that the spherical coordinate substitution has Jacobian determinant $\frac{\partial(x,y,z)}{\partial(\rho,\phi,\theta)} = \rho^2\sin\phi$.

Now, $\frac{\partial\mathbf{r}}{\partial\rho}$ (the derivative with respect to the radial coordinate) points radially outward. It is perpendicular to both $\frac{\partial\mathbf{r}}{\partial\phi}$ and $\frac{\partial\mathbf{r}}{\partial\theta}$ (which are tangent to the sphere), and is therefore *parallel* to the cross product $\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta}$.

Since each component of $\mathbf{r}$ is linear in $\rho$, we have $\left\|\frac{\partial\mathbf{r}}{\partial\rho}\right\| = 1$. The Jacobian determinant is the dot product of $\frac{\partial\mathbf{r}}{\partial\rho}$ with $\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta}$, and since these vectors are parallel:

$$
\left\|\frac{\partial\mathbf{r}}{\partial\phi} \times \frac{\partial\mathbf{r}}{\partial\theta}\right\| = \left|\frac{\partial(x,y,z)}{\partial(\rho,\phi,\theta)}\right| = \rho^2\sin\phi.
$$

This is the same surface area element as the volume element, without the $\mathrm{d}\rho$.

### Computing the area

$$
\text{Surface area} = \int_0^{2\pi}\int_0^{\pi} \rho^2\sin\phi\,\mathrm{d}\phi\,\mathrm{d}\theta.
$$

The inner integral is $\int_0^{\pi}\sin\phi\,\mathrm{d}\phi = [-\cos\phi]_0^{\pi} = 1 - (-1) = 2$. Multiplying by $\rho^2$ and then by $2\pi$:

$$
\text{Surface area} = 4\pi\rho^2.
$$

The familiar formula, derived from first principles.

---

## Looking Back

Over these four posts, we have traveled from coordinate systems and the Jacobian, through curves and line integrals, to Green's theorem and surface parameterizations. The thread connecting everything is the interplay between *boundaries* and *interiors*:

- The fundamental theorem of line integrals: $\int_C \nabla f \cdot \mathrm{d}\mathbf{r} = f(B) - f(A)$ — a 1D boundary (two points) determines a 1D integral (along a curve).
- Green's theorem: $\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \iint_R \operatorname{curl}\mathbf{F}\,\mathrm{d}A$ — a 1D boundary (a curve) determines a 2D integral (over a region).

The next step in this pattern would be the **divergence theorem** ($\oiint_S \mathbf{F} \cdot \mathbf{n}\,\mathrm{d}S = \iiint_V \operatorname{div}\mathbf{F}\,\mathrm{d}V$) and **Stokes' theorem** in 3D ($\oint_C \mathbf{F} \cdot \mathrm{d}\mathbf{r} = \iint_S \operatorname{curl}\mathbf{F} \cdot \mathrm{d}\mathbf{S}$) — where a 2D boundary (a surface) determines a 3D integral (over a solid). All of these are instances of the **generalized Stokes' theorem**: $\int_{\partial \Omega} \omega = \int_{\Omega} \mathrm{d}\omega$.

The differential forms, wedge products, and orientation machinery we developed along the way are exactly the language needed to state this theorem cleanly. From the Jacobian to Green's theorem to surface area, every topic has been a step toward this unifying principle.
