---
layout: post
title: "Calculus IV: Coordinate Systems and the Jacobian"
date: 2024-06-19
tags: [Math, Calculus, Multivariable Calculus]
---

These are my compiled notes from Calculus IV, reorganized into a narrative that builds the theory from its natural starting point. Calculus IV is, at its heart, about one goal: generalizing the fundamental theorem of calculus to higher dimensions. The ultimate destination is Stokes' theorem and its cousins, which relate integrals over a region to integrals over its boundary. But the journey starts here, with a seemingly unrelated question: *what coordinate system should we use?*

This is Part 1 of a four-part series. Here we develop the coordinate systems and substitution techniques that make multivariable integration tractable, culminating in the Jacobian determinant and a first taste of differential forms.

## What This Post Covers

- **Cylindrical Coordinates** — Extending polar coordinates to 3D, and the $rz$-half-plane trick for visualizing shapes
- **Spherical Coordinates** — A second generalization of polar coordinates, with the Earth-as-sphere intuition
- **Integration in Alternative Coordinates** — Why $\mathrm{d}x\,\mathrm{d}y$ becomes $r\,\mathrm{d}r\,\mathrm{d}\theta$, and the volume elements for cylindrical and spherical systems
- **Centroids and Centers of Mass** — The averaging interpretation and worked examples
- **The Jacobian Determinant** — The general theory of multivariable substitution, in 2D and 3D
- **Wedge Products and Oriented Integrals** — A preview of the differential forms machinery that will power Green's theorem

---

## Cylindrical Coordinates

You have likely seen polar coordinates before: a point in $\mathbb{R}^2$ is represented as a pair $(r, \theta)$, where $r$ is the distance from the origin and $\theta$ is the angle from the positive $x$-axis. Cylindrical coordinates are the simplest way to extend this idea to three dimensions.

A point $P$ in $\mathbb{R}^3$ gets the triple $(r, \theta, z)$. The $z$-coordinate means exactly what it usually does: the height above the $xy$-plane. The pair $(r, \theta)$ is the polar representation not of $P$ itself, but of $P$'s *projection* $Q$ onto the $xy$-plane — the "shadow" of $P$ directly below (or above) it.

<svg viewBox="0 0 340 260" style="max-width:340px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .axis { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.2; }
    .label { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); }
    .label-i { font-family: 'Newsreader', serif; font-size: 14px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .dim { stroke: var(--primary, #94452b); stroke-width: 1.5; stroke-dasharray: 5,4; }
    .dim-solid { stroke: var(--primary, #94452b); stroke-width: 1.5; }
    .point { fill: var(--primary, #94452b); }
    .angle-arc { stroke: var(--primary, #94452b); stroke-width: 1.2; fill: none; }
  </style>
  <!-- axes -->
  <line x1="50" y1="220" x2="10" y2="250" class="axis" marker-end="url(#ah1)"/>
  <line x1="50" y1="220" x2="200" y2="220" class="axis" marker-end="url(#ah1)"/>
  <line x1="50" y1="220" x2="50" y2="30" class="axis" marker-end="url(#ah1)"/>
  <defs><marker id="ah1" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="var(--text-primary, #1a1a1a)" stroke="none"/></marker></defs>
  <text x="5" y="245" class="label-i">x</text>
  <text x="200" y="215" class="label-i">y</text>
  <text x="42" y="28" class="label-i">z</text>
  <!-- point Q on xy-plane -->
  <line x1="50" y1="220" x2="150" y2="195" class="dim-solid"/>
  <circle cx="150" cy="195" r="3" class="point" opacity="0.5"/>
  <text x="155" y="210" class="label-i">Q</text>
  <!-- dashed line up to P -->
  <line x1="150" y1="195" x2="150" y2="85" class="dim"/>
  <!-- point P -->
  <circle cx="150" cy="85" r="4" class="point"/>
  <text x="158" y="82" class="label-i">P</text>
  <!-- z label -->
  <text x="155" y="145" class="label-i">z</text>
  <!-- r label -->
  <text x="88" y="215" class="label-i">r</text>
  <!-- theta arc -->
  <path d="M 70,220 Q 75,215 80,213" class="angle-arc"/>
  <text x="78" y="232" class="label-i" font-size="12">θ</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The cylindrical coordinates $(r, \theta, z)$ of a point $P$ and its projection $Q$ onto the $xy$-plane.</p>

The conversion formulas are the same as for polar coordinates, with $z$ tagging along:

$$
x = r\cos\theta, \qquad y = r\sin\theta, \qquad z = z.
$$

### Visualizing shapes: the $rz$-half-plane trick

A powerful technique for understanding 3D shapes in cylindrical coordinates is to *forget about $\theta$* at first and work only with $r$ and $z$. Since $r$ is nonnegative, the pair $(r, z)$ lives in a half-plane. You draw your shape there in 2D, then mentally rotate it around the $z$-axis as $\theta$ sweeps from $0$ to $2\pi$.

For example, the equation $z = r^2$ is a simple parabola in the $rz$-half-plane. Rotating it around the $z$-axis produces a **paraboloid** — the familiar bowl shape.

<svg viewBox="0 0 480 180" style="max-width:480px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax2 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.2; }
    .lb2 { font-family: 'Inter', sans-serif; font-size: 12px; fill: var(--text-primary, #1a1a1a); }
    .lb2i { font-family: 'Newsreader', serif; font-size: 13px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .curve2 { stroke: var(--primary, #94452b); stroke-width: 2; fill: none; }
    .arrow2 { fill: var(--text-primary, #1a1a1a); font-size: 22px; }
  </style>
  <!-- Left: rz-half-plane -->
  <line x1="30" y1="160" x2="180" y2="160" class="ax2" marker-end="url(#ah2)"/>
  <line x1="30" y1="160" x2="30" y2="15" class="ax2" marker-end="url(#ah2)"/>
  <defs><marker id="ah2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="var(--text-primary, #1a1a1a)" stroke="none"/></marker></defs>
  <text x="178" y="155" class="lb2i">r</text>
  <text x="22" y="18" class="lb2i">z</text>
  <!-- parabola z = r^2, scaled: r from 0 to ~1.2, z from 0 to ~1.4 -->
  <path d="M 30,160 Q 70,158 90,120 Q 110,65 140,20" class="curve2"/>
  <text x="145" y="30" class="lb2i" font-size="11">z = r²</text>
  <text x="60" y="175" class="lb2" font-size="11">rz-half-plane</text>
  <!-- Arrow -->
  <text x="200" y="95" class="arrow2">→</text>
  <text x="195" y="115" class="lb2" font-size="10">rotate</text>
  <!-- Right: paraboloid sketch -->
  <ellipse cx="360" cy="155" rx="80" ry="18" stroke="var(--primary, #94452b)" stroke-width="1.5" fill="none" stroke-dasharray="4,3"/>
  <path d="M 280,155 Q 300,130 320,80 Q 340,30 360,18 Q 380,30 400,80 Q 420,130 440,155" stroke="var(--primary, #94452b)" stroke-width="2" fill="none"/>
  <line x1="360" y1="170" x2="360" y2="5" class="ax2" marker-end="url(#ah2)"/>
  <text x="352" y="8" class="lb2i">z</text>
  <text x="310" y="175" class="lb2" font-size="11">paraboloid z = r²</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The $rz$-half-plane trick: draw the 2D cross-section, then rotate around the $z$-axis.</p>

Similarly, the region $0 \leq z \leq 1 - r$ is a triangle in the $rz$-half-plane (bounded by $z \geq 0$, $r \geq 0$, and $z \leq 1 - r$). Rotating it produces a solid cone — oriented like a party hat with the tip pointing up, though squatter than the typical traffic cone.

Not all shapes decompose neatly this way. When the bounds on $\theta$ are tangled with $r$ and $z$ — such as a rectangular prism — the conversion to cylindrical coordinates can be painful. But when a shape has **rotational symmetry** around the $z$-axis, cylindrical coordinates are exactly the right tool.

---

## Spherical Coordinates

Cylindrical coordinates kept the old polar variables and added a new rectangular dimension $z$. Spherical coordinates take a different approach: they keep the *philosophy* of polar coordinates — specifying a point by a distance and a direction — and extend it fully to 3D.

A point $P$ in $\mathbb{R}^3$ is represented by $(\rho, \theta, \phi)$:

- $\rho$ is the distance from the origin to $P$ (always $\rho \geq 0$),
- $\theta$ is the same angle as in cylindrical coordinates (the angle in the $xy$-plane, $0 \leq \theta \leq 2\pi$),
- $\phi$ is the angle between the positive $z$-axis and the line from the origin to $P$ (where $0 \leq \phi \leq \pi$).

<svg viewBox="0 0 340 270" style="max-width:340px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax3 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1.2; }
    .lb3 { font-family: 'Inter', sans-serif; font-size: 13px; fill: var(--text-primary, #1a1a1a); }
    .lb3i { font-family: 'Newsreader', serif; font-size: 14px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .rho-line { stroke: var(--primary, #94452b); stroke-width: 1.8; }
    .dim3 { stroke: var(--primary, #94452b); stroke-width: 1.2; stroke-dasharray: 5,4; }
    .pt3 { fill: var(--primary, #94452b); }
    .arc3 { stroke: var(--primary, #94452b); stroke-width: 1.2; fill: none; }
  </style>
  <defs><marker id="ah3" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><path d="M0,0 L7,2.5 L0,5" fill="var(--text-primary, #1a1a1a)" stroke="none"/></marker></defs>
  <!-- axes -->
  <line x1="80" y1="210" x2="30" y2="248" class="ax3" marker-end="url(#ah3)"/>
  <line x1="80" y1="210" x2="250" y2="210" class="ax3" marker-end="url(#ah3)"/>
  <line x1="80" y1="210" x2="80" y2="25" class="ax3" marker-end="url(#ah3)"/>
  <text x="22" y="250" class="lb3i">x</text>
  <text x="252" y="207" class="lb3i">y</text>
  <text x="72" y="23" class="lb3i">z</text>
  <!-- origin label -->
  <text x="62" y="222" class="lb3i" font-size="12">O</text>
  <!-- point P -->
  <line x1="80" y1="210" x2="200" y2="70" class="rho-line"/>
  <circle cx="200" cy="70" r="4" class="pt3"/>
  <text x="208" y="67" class="lb3i">P</text>
  <!-- rho label -->
  <text x="128" y="125" class="lb3i">ρ</text>
  <!-- dashed projection to Q -->
  <line x1="200" y1="70" x2="200" y2="183" class="dim3"/>
  <line x1="80" y1="210" x2="200" y2="183" class="dim3"/>
  <text x="205" y="200" class="lb3i">Q</text>
  <!-- z height -->
  <text x="207" y="135" class="lb3i" font-size="12">z</text>
  <!-- r label -->
  <text x="135" y="205" class="lb3i" font-size="12">r</text>
  <!-- phi arc (from z-axis toward rho line) -->
  <path d="M 80,170 Q 90,155 105,145" class="arc3"/>
  <text x="90" y="155" class="lb3i" font-size="12">ϕ</text>
  <!-- theta arc -->
  <path d="M 100,210 Q 105,205 110,203" class="arc3"/>
  <text x="108" y="222" class="lb3i" font-size="12">θ</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The spherical coordinates $(\rho, \theta, \phi)$ of a point $P$.</p>

A useful way to build intuition is to imagine the sphere $\rho = 1$ as the Earth:

- Holding $\theta$ constant and varying $\phi$ traces a half-circle from the North Pole ($\phi = 0$) to the South Pole ($\phi = \pi$). These are lines of **longitude**.
- Holding $\phi$ constant and varying $\theta$ traces a circle parallel to the equator. These are lines of **latitude**.

> Be careful: the mathematical convention differs from the geographical one. In geography, latitude is measured from the equator ($\pm 90°$); in mathematics, $\phi$ is measured from the North Pole ($0$ to $\pi$). So $\phi = 0$ is the North Pole, $\phi = \pi/2$ is the equator, and $\phi = \pi$ is the South Pole.

### Conversion formulas

Looking at the triangle $\triangle OPQ$, we can derive the relationship between spherical and rectangular coordinates. The hypotenuse has length $\rho$, the vertical leg is $z$, and the horizontal leg is the cylindrical radius $r$. Since $\phi$ is the angle from the $z$-axis:

$$
z = \rho\cos\phi, \qquad r = \rho\sin\phi.
$$

Combining with the polar-to-rectangular formulas $x = r\cos\theta$ and $y = r\sin\theta$:

$$
\boxed{x = \rho\sin\phi\cos\theta, \qquad y = \rho\sin\phi\sin\theta, \qquad z = \rho\cos\phi.}
$$

These are worth memorizing. A quick sanity check: when $\phi = 0$, we should be on the positive $z$-axis, and indeed $(x,y,z) = (0, 0, \rho)$. When $\phi = \pi/2$ and $\theta = 0$, we get $(x,y,z) = (\rho, 0, 0)$ — the positive $x$-axis.

---

## Integration in Alternative Coordinates

When we integrate over a region in polar coordinates, we cannot simply replace $\mathrm{d}x\,\mathrm{d}y$ by $\mathrm{d}r\,\mathrm{d}\theta$. We must use $r\,\mathrm{d}r\,\mathrm{d}\theta$. Why the extra factor of $r$?

### The area element in polar coordinates

The differential $\mathrm{d}x\,\mathrm{d}y$ stands in for $\Delta x \cdot \Delta y$ in a Riemann sum: the area of a tiny rectangle. In polar coordinates, a tiny "cell" corresponds to a change of $\Delta\theta$ in angle and $\Delta r$ in radius. This cell is not a rectangle — it is a thin sliver, roughly a rectangle with:

- **Width** (radial direction): $\Delta r$
- **Height** (angular direction): $r \Delta\theta$ — the arc length of a $\frac{\Delta\theta}{2\pi}$ fraction of a circle with perimeter $2\pi r$

<svg viewBox="0 0 260 200" style="max-width:260px; margin: 1.5rem auto; display:block;" xmlns="http://www.w3.org/2000/svg">
  <style>
    .ax4 { stroke: var(--text-primary, #1a1a1a); stroke-width: 1; }
    .lb4 { font-family: 'Inter', sans-serif; font-size: 11px; fill: var(--text-primary, #1a1a1a); }
    .lb4i { font-family: 'Newsreader', serif; font-size: 12px; font-style: italic; fill: var(--text-primary, #1a1a1a); }
    .cell { fill: var(--primary, #94452b); opacity: 0.2; stroke: var(--primary, #94452b); stroke-width: 1.5; }
    .arc4 { stroke: var(--text-primary, #1a1a1a); stroke-width: 0.8; fill: none; stroke-dasharray: 3,2; }
  </style>
  <defs><marker id="ah4" markerWidth="6" markerHeight="4" refX="6" refY="2" orient="auto"><path d="M0,0 L6,2 L0,4" fill="var(--text-primary, #1a1a1a)" stroke="none"/></marker></defs>
  <!-- axes -->
  <line x1="30" y1="170" x2="240" y2="170" class="ax4" marker-end="url(#ah4)"/>
  <line x1="30" y1="170" x2="30" y2="15" class="ax4" marker-end="url(#ah4)"/>
  <text x="237" y="165" class="lb4i">x</text>
  <text x="22" y="18" class="lb4i">y</text>
  <!-- radial lines -->
  <line x1="30" y1="170" x2="220" y2="70" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.6" stroke-dasharray="3,2"/>
  <line x1="30" y1="170" x2="200" y2="50" stroke="var(--text-primary, #1a1a1a)" stroke-width="0.6" stroke-dasharray="3,2"/>
  <!-- arcs -->
  <path d="M 155,108 A 120,120 0 0,0 140,96" class="arc4"/>
  <path d="M 185,125 A 150,150 0 0,0 167,110" class="arc4"/>
  <!-- shaded cell -->
  <path d="M 155,108 L 185,125 A 150,150 0 0,1 167,110 L 140,96 A 120,120 0 0,0 155,108 Z" class="cell"/>
  <!-- labels -->
  <text x="110" y="80" class="lb4" font-size="10">Δθ</text>
  <text x="188" y="140" class="lb4" font-size="10">Δr</text>
  <text x="170" y="96" class="lb4i" font-size="10">r Δθ</text>
</svg>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">A tiny area element in polar coordinates has area approximately $r\,\Delta r\,\Delta\theta$.</p>

So the area of this sliver is approximately $\Delta r \cdot r\,\Delta\theta = r\,\Delta r\,\Delta\theta$. In the limit, this becomes the differential area element $r\,\mathrm{d}r\,\mathrm{d}\theta$.

### Volume elements

The same reasoning extends to three dimensions:

**Cylindrical coordinates:** A tiny volume element is a sliver of area $r\,\Delta r\,\Delta\theta$ with a height $\Delta z$. So:

$$
\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z = r\,\mathrm{d}z\,\mathrm{d}r\,\mathrm{d}\theta.
$$

**Spherical coordinates:** The derivation is a bit more involved. We chop space into nested shells of thickness $\Delta\rho$, then divide each shell's surface by curves of constant $\theta$ and constant $\phi$. A single cell on a shell of radius $\rho$ has:

- Height: $\rho\,\Delta\phi$ (the arc length from the pole-to-pole measurement)
- Width: $\rho\sin\phi\,\Delta\theta$ (the circumference of the latitude circle scales with $r = \rho\sin\phi$)
- Depth: $\Delta\rho$

Multiplying these gives the volume element:

$$
\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z = \rho^2 \sin\phi\,\mathrm{d}\rho\,\mathrm{d}\phi\,\mathrm{d}\theta.
$$

> As a quick check, integrate $\rho^2 \sin\phi\,\mathrm{d}\rho\,\mathrm{d}\phi\,\mathrm{d}\theta$ over a full sphere of radius $R$ (that is, $0 \leq \rho \leq R$, $0 \leq \phi \leq \pi$, $0 \leq \theta \leq 2\pi$). You should get the familiar $\frac{4}{3}\pi R^3$.

---

## Centroids and Centers of Mass

The **centroid** of a solid region $R$ is its geometric center — the point $(\overline{x}, \overline{y}, \overline{z})$ where each coordinate is the *average* over the region. For instance:

$$
\overline{z} = \frac{\iiint_R z\,\mathrm{d}V}{\iiint_R \mathrm{d}V}.
$$

This is exactly the continuous version of averaging: the numerator sums up all the $z$-values, weighted by volume, and the denominator is the total volume.

### Example: centroid of a paraboloid

Consider the solid $R$ bounded by $0 \leq \theta \leq 2\pi$ and $r^2 \leq z \leq 1$ in cylindrical coordinates — a filled-in version of the paraboloid $z = r^2$, capped at height $z = 1$. Where is its centroid?

By symmetry, $R$ is rotationally symmetric in $x$ and $y$, so $\overline{x} = \overline{y} = 0$. We only need $\overline{z}$.

**The denominator** (volume):

$$
\iiint_R \mathrm{d}V = \int_0^{2\pi}\int_0^{1}\int_{r^2}^{1} r\,\mathrm{d}z\,\mathrm{d}r\,\mathrm{d}\theta = 2\pi \int_0^{1} r(1 - r^2)\,\mathrm{d}r = 2\pi \int_0^{1}(r - r^3)\,\mathrm{d}r = 2\pi\left(\frac{1}{2} - \frac{1}{4}\right) = \frac{\pi}{2}.
$$

**The numerator** ($\iiint z\,\mathrm{d}V$): the inner integral gives $\int_{r^2}^{1} rz\,\mathrm{d}z = \frac{r}{2}(1 - r^4)$. Then:

$$
\int_0^1 \frac{r - r^5}{2}\,\mathrm{d}r = \frac{1}{2}\left(\frac{1}{2} - \frac{1}{6}\right) = \frac{1}{6}, \qquad \text{so } \iiint_R z\,\mathrm{d}V = 2\pi \cdot \frac{1}{6} = \frac{\pi}{3}.
$$

Therefore $\overline{z} = \frac{\pi/3}{\pi/2} = \frac{2}{3}$. The centroid is at $(0, 0, \tfrac{2}{3})$ — two-thirds of the way from the vertex to the flat cap. This makes sense: the paraboloid is "top-heavy," with more volume concentrated near $z = 1$.

### Centers of mass

If the object has non-uniform density $\delta(x,y,z)$, the centroid becomes the **center of mass**. We simply weight everything by $\delta$:

$$
\overline{x} = \frac{\iiint_R x\,\delta\,\mathrm{d}V}{\iiint_R \delta\,\mathrm{d}V}, \qquad \overline{y} = \frac{\iiint_R y\,\delta\,\mathrm{d}V}{\iiint_R \delta\,\mathrm{d}V}, \qquad \overline{z} = \frac{\iiint_R z\,\delta\,\mathrm{d}V}{\iiint_R \delta\,\mathrm{d}V}.
$$

---

## The Jacobian Determinant

So far, the substitutions $r\,\mathrm{d}r\,\mathrm{d}\theta$ and $\rho^2\sin\phi\,\mathrm{d}\rho\,\mathrm{d}\phi\,\mathrm{d}\theta$ have been derived by geometric reasoning — picturing tiny slivers and cells. But there is a powerful general theory that handles *any* change of variables, not just the standard coordinate systems.

### A motivating example

Suppose we want to find the area of the region $x^2 - 2xy + 5y^2 \leq 1$. Setting this up in rectangular coordinates is horrific. But the expression can be rewritten as $(x - y)^2 + (2y)^2$, so if we set $u = x - y$ and $v = 2y$, the region becomes $u^2 + v^2 \leq 1$ — a unit disk! We know its area in the $uv$-plane is just $\pi$.

But the area of the original ellipse is not $\pi$. The substitution distorts areas. By how much?

The key insight is that a linear transformation $\mathbf{f}\colon \mathbb{R}^2 \to \mathbb{R}^2$ scales all areas by $|\det(\mathbf{f})|$. In our case, the inverse transformation from $(u,v)$ back to $(x,y)$ is $x = u + \frac{1}{2}v$, $y = \frac{1}{2}v$, represented by the matrix

$$
\begin{bmatrix} 1 & 1/2 \\ 0 & 1/2 \end{bmatrix}, \qquad \text{determinant} = 1 \cdot \frac{1}{2} - \frac{1}{2} \cdot 0 = \frac{1}{2}.
$$

So the transformation scales areas by a factor of $\frac{1}{2}$, and the ellipse has area $\frac{\pi}{2}$.

### The general theory

For a nonlinear substitution $(x, y) = \mathbf{f}(u, v)$, the local scaling factor varies from point to point. The best linear approximation at each point is given by the matrix of partial derivatives, and its determinant is the **Jacobian determinant**:

$$
\frac{\partial(x,y)}{\partial(u,v)} = \det \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} \\[4pt] \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} \end{bmatrix}.
$$

The substitution rule for double integrals becomes:

$$
\iint_R g(x,y)\,\mathrm{d}x\,\mathrm{d}y = \iint_S g(\mathbf{f}(u,v)) \left|\frac{\partial(x,y)}{\partial(u,v)}\right| \mathrm{d}u\,\mathrm{d}v.
$$

### The 3D Jacobian

The same idea extends to three variables. For a substitution $(x,y,z) = \mathbf{f}(u,v,w)$:

$$
\frac{\partial(x,y,z)}{\partial(u,v,w)} = \det \begin{bmatrix} \frac{\partial x}{\partial u} & \frac{\partial x}{\partial v} & \frac{\partial x}{\partial w} \\[4pt] \frac{\partial y}{\partial u} & \frac{\partial y}{\partial v} & \frac{\partial y}{\partial w} \\[4pt] \frac{\partial z}{\partial u} & \frac{\partial z}{\partial v} & \frac{\partial z}{\partial w} \end{bmatrix}.
$$

### Re-deriving the standard volume elements

With the Jacobian machinery, we can rederive our coordinate substitution rules from first principles.

**Cylindrical coordinates** ($x = r\cos\theta$, $y = r\sin\theta$, $z = z$):

$$
\frac{\partial(x,y,z)}{\partial(r,\theta,z)} = \det \begin{bmatrix} \cos\theta & -r\sin\theta & 0 \\ \sin\theta & r\cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} = r\cos^2\theta + r\sin^2\theta = r.
$$

This confirms $\mathrm{d}x\,\mathrm{d}y\,\mathrm{d}z = |r|\,\mathrm{d}r\,\mathrm{d}\theta\,\mathrm{d}z = r\,\mathrm{d}r\,\mathrm{d}\theta\,\mathrm{d}z$ (since $r \geq 0$).

**Spherical coordinates** ($x = \rho\sin\phi\cos\theta$, $y = \rho\sin\phi\sin\theta$, $z = \rho\cos\phi$): the computation is longer, but produces $-\rho^2\sin\phi$. Taking the absolute value gives $\rho^2\sin\phi$, confirming the volume element.

### Example: a clean substitution in 3D

Consider the integral

$$
\int_{-1}^{1}\int_{4-x}^{5-x}\int_{2y-1}^{2y+1}(x^2 + xy)\,\mathrm{d}z\,\mathrm{d}y\,\mathrm{d}x.
$$

The bounds suggest the substitution $u = x$, $v = x + y$, $w = z - 2y$, which gives $4 \leq v \leq 5$ and $-1 \leq w \leq 1$. Solving back: $x = u$, $y = v - u$, $z = w + 2(v-u)$. The Jacobian is

$$
\frac{\partial(x,y,z)}{\partial(u,v,w)} = \det \begin{bmatrix} 1 & 0 & 0 \\ -1 & 1 & 0 \\ -2 & 2 & 1 \end{bmatrix} = 1
$$

(the determinant of a lower-triangular matrix is the product of the diagonal entries). So the substitution preserves volumes!

Moreover, the integrand becomes $x^2 + xy = u^2 + u(v-u) = uv$. Since $uv$ is an odd function of $u$ and the integration region $[-1,1] \times [4,5] \times [-1,1]$ is symmetric in $u$, the integral evaluates to $0$ without any computation.

---

## Oriented Integrals and Wedge Products

The Jacobian determinant requires an absolute value because areas are always positive. But there is something interesting hiding behind that absolute value.

If the determinant of a linear transformation is *negative*, the transformation is **orientation-reversing** — it turns shapes into their mirror images. A substitution is **orientation-preserving** if the positive direction of increasing $v$ is a counterclockwise rotation from the positive direction of increasing $u$ (when drawn in the $xy$-plane), and orientation-reversing otherwise.

If we drop the absolute value and use signed areas, we get **oriented integrals**. These use slightly different notation: instead of $\mathrm{d}x\,\mathrm{d}y$, we write $\mathrm{d}x \wedge \mathrm{d}y$. The $\wedge$ (pronounced "wedge") signals that orientation matters.

### The wedge product

The wedge product provides an alternative way to compute the Jacobian. Here are its rules:

1. Replace $\mathrm{d}x$ by $\frac{\partial x}{\partial u}\,\mathrm{d}u + \frac{\partial x}{\partial v}\,\mathrm{d}v$ (the multivariate chain rule), and similarly for $\mathrm{d}y$.
2. Distribute $\wedge$ over addition.
3. Apply two simplification rules:
   - $\mathrm{d}v \wedge \mathrm{d}u = -(\mathrm{d}u \wedge \mathrm{d}v)$ — swapping order flips the sign (opposite orientations)
   - $\mathrm{d}u \wedge \mathrm{d}u = 0$ — a "coordinate system" with the same variable repeated measures no area

For example, with our ellipse substitution ($x = u + \frac{1}{2}v$, $y = \frac{1}{2}v$):

$$
\mathrm{d}x \wedge \mathrm{d}y = \left(\mathrm{d}u + \tfrac{1}{2}\,\mathrm{d}v\right) \wedge \left(\tfrac{1}{2}\,\mathrm{d}v\right) = \tfrac{1}{2}\,\mathrm{d}u \wedge \mathrm{d}v + \tfrac{1}{4}\,\mathrm{d}v \wedge \mathrm{d}v = \tfrac{1}{2}\,\mathrm{d}u \wedge \mathrm{d}v.
$$

This gives the same answer as the determinant, with the correct sign. The wedge product will turn out to be much more than a computational convenience: it is the language of **differential forms**, which will unify all the major theorems of multivariable calculus. We will return to it when we study Green's theorem in [Part 3](/2024/07/03/calculus-iv-greens-theorem.html).

---

## Looking Ahead

We now have the tools to set up integrals in whatever coordinate system makes a problem tractable, and the Jacobian determinant to handle the bookkeeping. In [Part 2](/2024/06/26/calculus-iv-curves-line-integrals.html), we shift from integrating over *regions* to integrating over *curves* — entering the world of line integrals, vector fields, and the concept of work from physics.
