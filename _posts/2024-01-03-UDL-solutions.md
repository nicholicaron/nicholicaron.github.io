---
layout: post
title: "Understanding Deep Learning Solutions"
date: 2024-01-03
tags: [Deep Learning, Math, AI]
cover_image: /assets/images/UDL/UDL-Solutions-01.jpg
---

My attempt at an unofficial, comprehensive solution set for the book *Understanding Deep Learning* by Simon J.D. Prince.

The book builds deep learning from first principles — linear regression, shallow networks, loss functions, optimization, and backpropagation — with exercises that demand you work through the mathematics by hand. That friction is the point. These solutions cover chapters 2 through 7, and represent my notes as I worked through them with pen, paper, and occasionally a healthy dose of skepticism toward my own algebra.

This is a living document. Some solutions are polished; others carry honest annotations where something still feels unresolved. Problems marked with an asterisk (\*) are from the challenge exercises in the book.

## Contents

- [Chapter 2: Linear Regression](#chapter-2-linear-regression)
- [Chapter 3: Shallow Neural Networks](#chapter-3-shallow-neural-networks)
- [Chapter 4: Deep Networks](#chapter-4-deep-networks)
- [Chapter 5: Loss Functions](#chapter-5-loss-functions)
- [Chapter 6: Training & Optimization](#chapter-6-training--optimization)
- [Chapter 7: Backpropagation](#chapter-7-backpropagation)

**Reference:** [Understanding Deep Learning](https://udlbook.github.io/udlbook/) by Simon J.D. Prince

---

## Chapter 2: Linear Regression

The book begins with the simplest supervised learning model: fit a line to data. The loss function measures how far off our predictions are, and we minimize it by computing gradients and either solving analytically or descending iteratively.

### Problem 2.1

Compute the partial derivatives of the least squares loss $L[\phi]$ with respect to the parameters $\phi_0$ and $\phi_1$.

The loss function is:

$$\hat{\phi} = \operatorname*{argmin}_{\phi} \left[ \sum_{i=1}^{I} (\phi_0 + \phi_1 x_i - y_i)^2 \right]$$

Expanding the squared term:

$$(\phi_0 + \phi_1 x_i - y_i)^2 = \phi_0^2 + 2\phi_0\phi_1 x_i - 2\phi_0 y_i + \phi_1^2 x_i^2 - 2\phi_1 x_i y_i + y_i^2$$

Taking derivatives:

$$\boxed{\frac{\partial L}{\partial \phi_0} = \sum_i 2(\phi_0 + \phi_1 x_i - y_i) = \sum_i \left[2\phi_0 + 2\phi_1 x_i\right]}$$

$$\boxed{\frac{\partial L}{\partial \phi_1} = \sum_i 2(\phi_0 + \phi_1 x_i - y_i) \cdot x_i = \sum_i \left[2\phi_0 x_i + 2\phi_1 x_i^2\right]}$$

### Problem 2.2

Find the minimum of the loss in closed form by setting the derivatives from Problem 2.1 to zero and solving for $\phi_0$ and $\phi_1$.

Setting $\frac{\partial L}{\partial \phi_0} = 0$:

$$\begin{aligned}
2\phi_0 + \phi_1 x_i - 2y_i &= 0 \\
2\phi_0 - 2\phi_0 x_i &= 2\phi_1 x_i^2 - 2x_i y_i - \phi_1 x_i + 2y_i \\
\phi_0(2 - 2x_i) &= 2\phi_1 x_i^2 - 2x_i y_i - \phi_1 x_i + 2y_i \\
\phi_0 &= \frac{2\phi_1 x_i^2 - 2x_i y_i - \phi_1 x_i + 2y_i}{2 - 2x_i}
\end{aligned}$$

Setting $\frac{\partial L}{\partial \phi_1} = 0$:

$$\begin{aligned}
\phi_1 x_i - 2\phi_1 x_i^2 &= -2\phi_0 + 2y_i + 2\phi_0 x_i - 2x_i y_i \\
\phi_1(x_i - 2x_i^2) &= 2\phi_0 x_i + 2y_i - 2\phi_0 - 2x_i y_i \\
\phi_1 &= \frac{2\phi_0 x_i + 2y_i - 2\phi_0 - 2x_i y_i}{x_i - 2x_i^2}
\end{aligned}$$

This works for linear regression but not for more complex models — which is why we use iterative methods like gradient descent (figure 2.4 in the book).

### Problem 2.3\*

Reformulate linear regression as a generative model where $x = g[y, \phi] = \phi_0 + \phi_1 y$.

The new loss function swaps the roles of $x$ and $y$:

$$\hat{\phi} = \operatorname*{argmin}_{\phi} \left[ \sum_{i=1}^{I} (\phi_0 + \phi_1 y_i - x_i)^2 \right]$$

The inverse function for inference is:

$$y = \frac{x - \phi_0}{\phi_1} = g^{-1}[x, \phi]$$

This model will generally *not* produce the same predictions as the discriminative version for a given training set $\{x_i, y_i\}$, since they minimize different objectives — one minimizes vertical residuals, the other horizontal residuals. You can verify this by fitting a line to three data points using both methods and comparing the results.

---

## Chapter 3: Shallow Neural Networks

This chapter builds the core intuition for what a single hidden layer can represent. A shallow network with ReLU activations creates a piecewise linear function — and the exercises systematically explore how the number of pieces, their slopes, and their boundaries depend on the parameters.

### Problem 3.1

If the activation function were linear ($a[z] = \psi_0 + \psi_1 z$) or simply the identity ($a[z] = z$), what kind of mapping would the network create?

With a linear activation, the network output is:

$$y = f[x, \phi] = \phi_0 + \phi_1(\psi_0 + \psi_1(\theta_{10} + \theta_{11}x)) + \phi_2(\psi_0 + \psi_1(\theta_{20} + \theta_{21}x)) + \phi_3(\psi_0 + \psi_1(\theta_{30} + \theta_{31}x))$$

Expanding fully:

$$y = \phi_0 + \phi_1\psi_0 + \phi_1\psi_1\theta_{10} + \phi_1\psi_1\theta_{11}x + \phi_2\psi_0 + \phi_2\psi_1\theta_{20} + \phi_2\psi_1\theta_{21}x + \phi_3\psi_0 + \phi_3\psi_1\theta_{30} + \phi_3\psi_1\theta_{31}x$$

This is still a linear mapping — just a constant plus a term proportional to $x$. No matter how many parameters we have, a linear activation collapses the network into a simple linear function.

With the identity activation ($a[z] = z$), the same collapse happens:

$$y = \phi_0 + \phi_1(\theta_{10} + \theta_{11}x) + \phi_2(\theta_{20} + \theta_{21}x) + \phi_3(\theta_{30} + \theta_{31}x)$$

Still linear. The nonlinearity of the activation function is what gives neural networks their expressive power.

### Problem 3.2

For the four linear regions in figure 3.3j, identify which hidden units are active (passing the input through) and which are inactive (clipping to zero).

Examining the piecewise linear output $y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3$, where each $h_d = a[\theta_{d0} + \theta_{d1}x]$:

<div style="text-align: center;">
<svg viewBox="0 0 500 220" width="500" height="220" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; max-width: 100%;">
  <!-- Axes -->
  <line x1="50" y1="180" x2="460" y2="180" stroke="currentColor" stroke-width="1.5"/>
  <line x1="50" y1="20" x2="50" y2="180" stroke="currentColor" stroke-width="1.5"/>
  <!-- Axis labels -->
  <text x="250" y="210" text-anchor="middle" fill="currentColor" font-size="13" font-family="Inter, sans-serif">Input, x</text>
  <text x="15" y="100" text-anchor="middle" fill="currentColor" font-size="13" font-family="Inter, sans-serif" transform="rotate(-90, 15, 100)">Output, y</text>
  <!-- Region boundaries (vertical dashed lines) -->
  <line x1="150" y1="25" x2="150" y2="180" stroke="currentColor" stroke-width="0.8" stroke-dasharray="5,4" opacity="0.5"/>
  <line x1="250" y1="25" x2="250" y2="180" stroke="currentColor" stroke-width="0.8" stroke-dasharray="5,4" opacity="0.5"/>
  <line x1="350" y1="25" x2="350" y2="180" stroke="currentColor" stroke-width="0.8" stroke-dasharray="5,4" opacity="0.5"/>
  <!-- Joint labels -->
  <text x="150" y="195" text-anchor="middle" fill="currentColor" font-size="11" font-family="Inter, sans-serif">j₁</text>
  <text x="250" y="195" text-anchor="middle" fill="currentColor" font-size="11" font-family="Inter, sans-serif">j₂</text>
  <text x="350" y="195" text-anchor="middle" fill="currentColor" font-size="11" font-family="Inter, sans-serif">j₃</text>
  <!-- Region labels -->
  <text x="100" y="40" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" font-style="italic">R₁</text>
  <text x="200" y="40" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" font-style="italic">R₂</text>
  <text x="300" y="40" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" font-style="italic">R₃</text>
  <text x="400" y="40" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" font-style="italic">R₄</text>
  <!-- Piecewise linear function -->
  <polyline points="50,140 150,160 250,60 350,100 460,80" fill="none" stroke="currentColor" stroke-width="2"/>
  <!-- Shaded region R₂ -->
  <rect x="150" y="25" width="100" height="155" fill="currentColor" opacity="0.06"/>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Four linear regions created by three hidden units. Each joint corresponds to where a hidden unit's pre-activation crosses zero.</p>

| Region | Active Hidden Units |
|--------|-------------------|
| $R_1$ | $h_3$ only |
| $R_2$ | $h_1$ and $h_3$ |
| $R_3$ | $h_1$, $h_2$, and $h_3$ |
| $R_4$ | $h_1$ and $h_2$ |

Each time we cross a joint $j_d$, the corresponding hidden unit $h_d$ transitions between active and inactive, changing the slope of the output.

### Problem 3.3\*

The "joints" in the piecewise linear function occur where each hidden unit's pre-activation crosses zero. Setting each $\theta_{d0} + \theta_{d1}x = 0$:

$$\begin{aligned}
j_1: \quad \theta_{10} + \theta_{11}x = 0 \quad &\Longrightarrow \quad x = -\theta_{10}/\theta_{11} \\
j_2: \quad \theta_{20} + \theta_{21}x = 0 \quad &\Longrightarrow \quad x = -\theta_{20}/\theta_{21} \\
j_3: \quad \theta_{30} + \theta_{31}x = 0 \quad &\Longrightarrow \quad x = -\theta_{30}/\theta_{31}
\end{aligned}$$

The slopes of the four linear regions are determined by which combination of $\phi_d \cdot \theta_{d1}$ terms are active.

### Problem 3.4

Redraw figure 3.3 with a modified third hidden unit (changed y-intercept and slope as in figure 3.14c), keeping all other parameters the same.

The key insight is that changing one hidden unit's parameters shifts its joint position and slope contribution, which ripples through all regions where that unit is active. The output remains a piecewise linear function with three joints, but the shape changes in regions $R_1$, $R_2$, and $R_3$ (everywhere $h_3$ contributes).

### Problem 3.5

Prove the non-negative homogeneity property of ReLU: for $\alpha \in \mathbb{R}^+$,

$$\text{ReLU}[\alpha \cdot z] = \alpha \cdot \text{ReLU}[z]$$

Starting from the definition:

$$\text{ReLU}(x) = \begin{cases} 0 & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases}$$

For $\alpha > 0$:

$$\alpha \cdot \text{ReLU}(x) = \alpha \cdot \begin{cases} 0 & \text{if } x < 0 \\ x & \text{if } x \geq 0 \end{cases} = \begin{cases} 0 & \text{if } x < 0 \\ \alpha x & \text{if } x \geq 0 \end{cases}$$

And since $\alpha > 0$ preserves the sign of the argument:

$$\text{ReLU}(\alpha x) = \begin{cases} 0 & \text{if } \alpha x < 0 \\ \alpha x & \text{if } \alpha x \geq 0 \end{cases} = \begin{cases} 0 & \text{if } x < 0 \\ \alpha x & \text{if } x \geq 0 \end{cases}$$

These are identical. $\square$

### Problem 3.6

Following from Problem 3.5, what happens to the shallow network when we multiply $\theta_{10}$ and $\theta_{11}$ by a positive constant $\alpha$ and divide $\phi_1$ by $\alpha$? What if $\alpha$ is negative?

If $\alpha$ is positive, nothing changes:

$$\frac{1}{\alpha} \text{ReLU}(\alpha x) = \frac{1}{\alpha} \cdot \alpha \cdot \text{ReLU}(x) = \text{ReLU}(x)$$

The non-negative homogeneity property means we can freely rescale the hidden unit's parameters as long as we compensate in the output weights.

If $\alpha$ is negative:

$$\frac{1}{\alpha} \text{ReLU}(\alpha x) = \frac{1}{\alpha} \begin{cases} 0 & \text{if } x > 0 \\ \alpha x & \text{if } x \leq 0 \end{cases} = \begin{cases} 0 & \text{if } x > 0 \\ x & \text{if } x \leq 0 \end{cases}$$

This flips which side of the origin is active — it's no longer the same function. The homogeneity property only holds for non-negative scalars.

### Problem 3.7

Does the least squares loss for fitting the shallow network model (equation 3.1) have a unique minimum?

We're looking for critical points where all partial derivatives are zero. Expanding $f[x, \phi]^2$ and taking derivatives with respect to every parameter:

$$L[\phi] = \sum_i (f[x_i, \phi] - y_i)^2$$

where $f[x, \phi] = \phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x] + \phi_2 a[\theta_{20} + \theta_{21}x] + \phi_3 a[\theta_{30} + \theta_{31}x]$.

Taking the partial derivatives with respect to the output weights and biases:

$$\frac{\partial L}{\partial \phi_0} = 2(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i)$$

$$\frac{\partial L}{\partial \phi_1} = 2(a[\theta_{10} + \theta_{11}x_i] + \phi_1)(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i)$$

Setting $\partial L/\partial \theta_{\cdot} = 0$ for the hidden unit parameters introduces piecewise conditions from the ReLU derivative. Critical points occur wherever all derivatives simultaneously equal zero. For $\partial/\partial \phi_0 \to 0$, $\phi_0 = 0$ for $\text{ReLU}(z) > 0$ cases.

> *Author's note: This one still feels off. The critical point analysis doesn't feel complete — the piecewise nature of ReLU makes this trickier than it appears, and I suspect the answer involves showing non-convexity rather than uniqueness. I'll revisit after more chapters.*

### Problem 3.8

Replace ReLU with three alternative activation functions and describe the family of functions each can represent with one input, three hidden units, and one output.

Using the same parameters $\phi = \{-0.23, -1.3, 1.3, 0.66, -0.2, 0.4, -0.9, 0.9, 1.1, -0.7\}$:

**(i) Heaviside step function:**

$$\text{heaviside}[z] = \begin{cases} 0 & z < 0 \\ 1 & z \geq 0 \end{cases}$$

Each hidden unit contributes a step, so the output is a piecewise constant function — a sum of weighted step functions. With three hidden units, we get up to four constant regions. The family is all piecewise constant functions with at most three jump discontinuities.

**(ii) Hyperbolic tangent:**

Each hidden unit contributes a smooth sigmoid-like curve. The output is a smooth function that can approximate any continuous function. The tanh activation creates smooth transitions rather than sharp joints, producing the family of all sums of three shifted, scaled sigmoid curves.

**(iii) Rectangular function:**

$$\text{rect}[z] = \begin{cases} 0 & z < 0 \\ 1 & 0 \leq z \leq 1 \\ 0 & z > 1 \end{cases}$$

Each hidden unit contributes a "bump" — active over a finite interval. With three hidden units, the output is a piecewise constant function with up to six jump points (each rect has two edges). The family is all sums of three weighted rectangular pulses.

### Problem 3.9\*

The third linear region in figure 3.3 has a slope that is the sum of the slopes of the first and fourth regions. This follows directly from Problem 3.2: in $R_3$, all three hidden units are active, so the slope is $\phi_1\theta_{11} + \phi_2\theta_{21} + \phi_3\theta_{31}$. In $R_1$ only $h_3$ is active (slope $\phi_3\theta_{31}$), and in $R_4$ only $h_1$ and $h_2$ are active (slope $\phi_1\theta_{11} + \phi_2\theta_{21}$). The sum of slopes in $R_1$ and $R_4$ equals the slope in $R_3$.

### Problem 3.10

A shallow network with one input, one output, and three hidden units typically creates four linear regions. Under what circumstances could it produce fewer?

If two or more of the linear functions representing the hidden units are linearly dependent, there would be fewer than four linear regions. Concretely, if two hidden units have joints at the same $x$ position (i.e., $-\theta_{d0}/\theta_{d1}$ is the same for two units), then two joints coincide and we get three regions instead of four. In the extreme case where all three joints coincide, we get just two regions.

### Problem 3.11\*

The model in figure 3.6 has: 1 bias ($\phi_0$), plus $2 \times 4 = 8$ weights connecting inputs to hidden units, plus $4$ biases in the hidden layer, plus $4 \times 1 = 4$ weights connecting hidden units to outputs, plus $2$ additional input-to-output weights. Total:

$$1(4) + 2(4) + 4 + 2 = 18 \text{ parameters}$$

### Problem 3.12

The model in figure 3.7 has: input dimension 2, hidden dimension 4, output dimension 1. With the standard fully connected architecture:

$$2(3) + 3(4) + 3 + 1 = 13 \text{ parameters}$$

### Problem 3.13

For the seven regions in figure 3.8j (a network with $D_i = 2$ inputs, $D = 3$ hidden units, $D_o = 1$ output), the activation pattern for each region describes which hidden units are active (passing the input) and which are clipped.

The two-dimensional input space is partitioned by three hyperplanes (lines in 2D), one per hidden unit. Each line is defined by $\theta_{d0} + \theta_{d1}x_1 + \theta_{d2}x_2 = 0$. On one side the unit is active; on the other it's clipped. The seven regions correspond to seven distinct combinations of active/inactive states across the three hidden units.

### Problem 3.14

Write the equations for the two-layer network in figure 3.11 (two inputs, three hidden units, two outputs):

$$y_1 = \psi_{11}\big[a[\theta_{10} + \theta_{11}x]\big] + \psi_{21}\big[a[\theta_{20} + \theta_{21}x]\big] + \psi_{31}\big[a[\theta_{30} + \theta_{31}x]\big]$$

$$y_2 = \psi_{12}\big[a[\theta_{10} + \theta_{11}x]\big] + \psi_{22}\big[a[\theta_{20} + \theta_{21}x]\big] + \psi_{32}\big[a[\theta_{30} + \theta_{31}x]\big]$$

Each output is a different linear combination of the same hidden unit activations.

### Problem 3.15\*

The maximum number of 3D linear regions created by the network in figure 3.11 is determined by three hidden units, which define three intersecting planes in 3D. Three planes in general position create $2^3 = 8$ linear regions.

### Problem 3.16

For a network with two inputs ($x_1, x_2$), four hidden units, and three outputs, the equations in compact form are:

$$h_d = a\bigg[\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di} x_i\bigg], \qquad y_j = \phi_{j0} + \sum_{d=1}^{D} \phi_{jd} \cdot a\bigg[\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di} x_i\bigg]$$

Writing out explicitly:

$$\begin{aligned}
y_1 &= \phi_{10} + \phi_{11}\,a[\theta_{10} + \theta_{11}x_1 + \theta_{12}x_2 + \theta_{13}x_3] + \phi_{12}\,a[\theta_{20} + \theta_{21}x_1 + \theta_{22}x_2 + \theta_{23}x_3] \\
&\quad + \phi_{13}\,a[\theta_{30} + \theta_{31}x_1 + \theta_{32}x_2 + \theta_{33}x_3]
\end{aligned}$$

And similarly for $y_2$ and $y_3$, each with their own output weights $\phi_{2d}$ and $\phi_{3d}$.

### Problem 3.17\*

For a general network with $D_i$ inputs, $D$ hidden units, and $D_o$ outputs, the total parameter count is:

$$\underbrace{(D_i + 1) \times D}_{\text{input-to-hidden weights + biases}} + \underbrace{(D + 1) \times D_o}_{\text{hidden-to-output weights + biases}}$$

The $+1$ terms account for the bias at each layer.

### Problem 3.18\*

The maximum number of regions created by a shallow network with $D_i$-dimensional input and $D$ hidden units is given by Zaslavsky's theorem (1975) — the maximum number of regions formed by partitioning $D_i$-dimensional space with $D$ hyperplanes:

$$N = \sum_{j=0}^{D_i} \binom{D}{j}$$

For $D_i = 2$, $D = 3$:

$$\sum_{j=0}^{2} \binom{3}{j} = \binom{3}{0} + \binom{3}{1} + \binom{3}{2} = 1 + 3 + 3 = 7$$

This matches figure 3.8. If we add two more hidden units ($D = 5$):

$$\sum_{j=0}^{2} \binom{5}{j} = \binom{5}{0} + \binom{5}{1} + \binom{5}{2} = 1 + 5 + 10 = 16$$

---

## Chapter 4: Deep Networks

Depth fundamentally changes what neural networks can represent. A deep network composes simple functions, and each additional layer can exponentially increase the number of linear regions — achieving with far fewer parameters what a shallow network would need a massive hidden layer to match.

### Problem 4.1\*

Composing the two networks in figure 4.8 produces an output $y'$ for $x \in [-1, 1]$ that exhibits rapid oscillations. The first network maps $x$ to a piecewise linear function, and feeding that output into the second network applies another set of piecewise linear transformations. The composition multiplies the number of linear pieces.

<div style="text-align: center;">
<svg viewBox="0 0 460 200" width="460" height="200" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; max-width: 100%;">
  <!-- Axes -->
  <line x1="50" y1="170" x2="420" y2="170" stroke="currentColor" stroke-width="1.2"/>
  <line x1="50" y1="20" x2="50" y2="170" stroke="currentColor" stroke-width="1.2"/>
  <!-- Axis labels -->
  <text x="235" y="195" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif">Input, x</text>
  <text x="15" y="95" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" transform="rotate(-90, 15, 95)">Output, y'</text>
  <!-- Tick marks -->
  <line x1="50" y1="170" x2="50" y2="175" stroke="currentColor" stroke-width="1"/>
  <text x="50" y="188" text-anchor="middle" fill="currentColor" font-size="10">-1.0</text>
  <line x1="235" y1="170" x2="235" y2="175" stroke="currentColor" stroke-width="1"/>
  <text x="235" y="188" text-anchor="middle" fill="currentColor" font-size="10">0.0</text>
  <line x1="420" y1="170" x2="420" y2="175" stroke="currentColor" stroke-width="1"/>
  <text x="420" y="188" text-anchor="middle" fill="currentColor" font-size="10">1.0</text>
  <!-- Zero line -->
  <line x1="50" y1="95" x2="420" y2="95" stroke="currentColor" stroke-width="0.5" stroke-dasharray="4,4" opacity="0.3"/>
  <!-- Composed function (rapid oscillation) -->
  <polyline points="50,95 80,45 110,145 140,45 170,145 200,45 230,145 260,45 290,145 320,45 350,145 380,45 420,95" fill="none" stroke="currentColor" stroke-width="2"/>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Composing two piecewise linear networks produces rapid oscillations — the number of linear pieces multiplies with each composition.</p>

### Problem 4.2

The four hyperparameters in figure 4.6 (a deep network with three hidden layers) are:

- $D_1$ — number of units in hidden layer 1
- $D_2$ — number of units in hidden layer 2
- $D_3$ — number of units in hidden layer 3
- Choice of activation function

### Problem 4.3

Using the non-negative homogeneity of ReLU (Problem 3.5), show that weight matrices can be rescaled arbitrarily as long as biases are adjusted and scale factors re-applied at the output:

$$\text{ReLU}\big[\boldsymbol{\beta}_1 + \lambda_1 \cdot \boldsymbol{\Omega}_1 \text{ReLU}[\boldsymbol{\beta}_0 + \lambda_0 \cdot \boldsymbol{\Omega}_0 \mathbf{x}]\big] = \lambda_0 \lambda_1 \cdot \text{ReLU}\bigg[\frac{1}{\lambda_0 \lambda_1}\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \text{ReLU}\bigg[\frac{1}{\lambda_0}\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}\bigg]\bigg]$$

The proof proceeds by pulling constants through ReLU using homogeneity:

$$\begin{aligned}
&\text{ReLU}\big[\boldsymbol{\beta}_1 + \lambda_1 \boldsymbol{\Omega}_1\,\text{ReLU}[\boldsymbol{\beta}_0 + \lambda_0 \boldsymbol{\Omega}_0 \mathbf{x}]\big] \\
&= \lambda_0 \lambda_1\,\text{ReLU}\bigg[\frac{1}{\lambda_0 \lambda_1}\boldsymbol{\beta}_1 + \frac{1}{\lambda_0}\boldsymbol{\Omega}_1\,\text{ReLU}[\boldsymbol{\beta}_0 + \lambda_0 \boldsymbol{\Omega}_0 \mathbf{x}]\bigg] \\
&= \lambda_0 \lambda_1\,\text{ReLU}\bigg[\frac{1}{\lambda_0 \lambda_1}\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\,\text{ReLU}\bigg[\frac{1}{\lambda_0}\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}\bigg]\bigg]
\end{aligned}$$

At each step, we multiply outside by $\lambda$ and divide inside by $\lambda$, using $\text{ReLU}[\alpha \cdot z] = \alpha \cdot \text{ReLU}[z]$ to pull the scale factor out.

### Problem 4.4

For a deep network with $D_i = 5$ inputs, $D_o = 4$ outputs, and three hidden layers of sizes $D_1 = 20$, $D_2 = 10$, $D_3 = 7$:

The layer-by-layer equations:

$$\begin{aligned}
\mathbf{h}_1 &= a[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}] \\
\mathbf{h}_2 &= a[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1 \mathbf{h}_1] \\
\mathbf{h}_3 &= a[\boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2 \mathbf{h}_2] \\
\mathbf{y} &= \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3 \mathbf{h}_3
\end{aligned}$$

Or as a single composed expression:

$$\mathbf{y} = \boldsymbol{\beta}_3 + \boldsymbol{\Omega}_3\,a\big[\boldsymbol{\beta}_2 + \boldsymbol{\Omega}_2\,a[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\,a[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0 \mathbf{x}]]\big]$$

Weight matrix and bias vector sizes:

| Parameter | Size | Bias | Size |
|-----------|------|------|------|
| $\boldsymbol{\Omega}_0$ | $20 \times 5$ | $\boldsymbol{\beta}_0$ | $\in \mathbb{R}^{20}$ |
| $\boldsymbol{\Omega}_1$ | $10 \times 20$ | $\boldsymbol{\beta}_1$ | $\in \mathbb{R}^{10}$ |
| $\boldsymbol{\Omega}_2$ | $7 \times 10$ | $\boldsymbol{\beta}_2$ | $\in \mathbb{R}^{7}$ |
| $\boldsymbol{\Omega}_3$ | $4 \times 7$ | $\boldsymbol{\beta}_3$ | $\in \mathbb{R}^{4}$ |

### Problem 4.5

A deep network with $D_i = 5$ inputs, $D_o = 1$ output, and $K = 20$ hidden layers of $D = 30$ units each has:

- **Width** = 30 (the number of hidden units per layer)
- **Depth** = 20 (the number of hidden layers)

### Problem 4.6

For a network with $D_i = 1$, $D_o = 1$, $K = 10$ layers, and $D = 10$ hidden units per layer:

Number of weights as a function of width $x$: $f(x) = 10x$, so $\frac{\partial f}{\partial x} = 10$.

Number of weights as a function of depth $y$: $f(y) = 10y$, so $\frac{\partial f}{\partial y} = 10$.

The number of weights grows at an equal rate of 10 whether we increase the depth or the width by one. However, increasing depth gives an exponential increase in representational capacity (number of linear regions), while increasing width gives only a linear increase.

### Problem 4.7

Choose parameters for the shallow network in equation 3.1 to define an identity function over a finite range $x \in [a, b]$.

The trick is to handle negative values with a piecewise construction:

$$y = \begin{cases} 0 + \frac{1}{3}a[0 + 1 \cdot x] + \frac{1}{3}a[0 + 1 \cdot x] + \frac{1}{3}a[0 + 1 \cdot x] & \text{if } x > 0 \\
0 - \frac{1}{3}a[0 - 1 \cdot x] - \frac{1}{3}a[0 - 1 \cdot x] - \frac{1}{3}a[0 - 1 \cdot x] & \text{if } x \leq 0
\end{cases}$$

For $x > 0$: $\phi = \{0,\; \frac{1}{3},\; \frac{1}{3},\; \frac{1}{3},\; 0, 1,\; 0, 1,\; 0, \frac{1}{3}\}$

For $x \leq 0$: $\phi = \{0,\; -\frac{1}{3},\; -\frac{1}{3},\; -\frac{1}{3},\; 0, -1,\; 0, -1,\; 0, -\frac{1}{3}\}$

### Problem 4.8\*

Figure 4.9 shows three hidden unit activations with slopes 1.0, 1.0, and -1.0, with joints at positions $1/6$, $2/6$, and $4/6$ respectively. We need $\phi_0, \phi_1, \phi_2, \phi_3$ so that $y = \phi_0 + \phi_1 h_1 + \phi_2 h_2 + \phi_3 h_3$ oscillates between 0 and 1 across four linear regions (positive, negative, positive, negative slopes).

Working region by region:
- **First region** ($0 < x < 1/6$): only $h_3$ is active. Slope = $\phi_3 \cdot (-1) = +1$, so $\phi_3 = -1$. Need $y = 0$ at start and $y = 1$ at end, giving $\phi_0 = 4, \phi_3 = -6$.
- **Second region** ($1/6 < x < 2/6$): $h_3$ and $h_1$ are active. Slope $= \phi_1 - \phi_3$, and we want slope $= -12$.
- **Third region** ($2/6 < x < 4/6$): $h_1$, $h_2$, $h_3$ all active. Slope $= \phi_1 + \phi_2 + \phi_3 \cdot (-1)$.

Composing this network with itself essentially folds the function, doubling the number of oscillations. For $K$ compositions: $4 \times K$ linear regions. The number of linear regions grows as $4^K$ when composing the network with itself $K$ times.

### Problem 4.9\*

Can we create a function with three linear regions that oscillates between 0 and 1 using a shallow network with two hidden units? **No** — two hidden units create at most three linear regions, but to oscillate you need the function to go up, down, and up again, which requires two "direction changes" and hence three joints (four regions minimum for a full oscillation).

However, it *is* possible to create five linear regions using a shallow network with four hidden units. In general, for $n \geq 3$ hidden units, we can create a function with $n + 1$ linear regions that oscillates.

### Problem 4.10

A deep network with a single input, single output, and $K$ hidden layers each containing $D$ hidden units has:

$$\text{Total parameters} = 3D + 1 + (K - 1)D(D + 1)$$

Breaking this down:
- $\boldsymbol{\beta}_0 \in \mathbb{R}^D$ contributes $D$
- $\boldsymbol{\Omega}_0 \in \mathbb{R}^{D \times 1}$ contributes $D$
- Each hidden-to-hidden layer: $\boldsymbol{\Omega}_k \in \mathbb{R}^{D \times D}$ contributes $D^2$, $\boldsymbol{\beta}_k \in \mathbb{R}^D$ contributes $D$, total $(K-1)(D^2 + D)$
- Final layer: $\boldsymbol{\Omega}_K \in \mathbb{R}^{1 \times D}$ contributes $D$, $\boldsymbol{\beta}_K \in \mathbb{R}^1$ contributes $1$

Total: $D + D + (K-1)D(D+1) + D + 1 = 3D + 1 + (K-1)D(D+1)$.

### Problem 4.11\*

Compare two networks mapping a scalar input to a scalar output:

**Network 1 (shallow):** $D = 95$ hidden units.

Using the formula from Problem 3.17: $(D_i + 1) \times D + (D + 1) \times D_o = 2 \times 95 + 96 \times 1 = 286$ parameters.

Maximum linear regions:

$$\sum_{j=0}^{1}\binom{95}{j} = 1 + 95 = 96 \text{ linear regions}$$

**Network 2 (deep):** $K = 10$ layers, $D = 5$ units each.

Using the formula from Problem 4.10: $3(5) + 1 + (10-1)(5)(6) = 16 + 270 = 286$ parameters.

Maximum linear regions: $324$ linear regions.

Both networks have the same parameter count (286), but the deep network produces far more linear regions (324 vs 96). Network 1 would likely run faster — a single wide matrix multiplication is more parallelizable than ten sequential narrow ones, even with the same total parameter count.

---

## Chapter 5: Loss Functions

Loss functions encode our assumptions about the data. The book develops a principled recipe: choose a probability distribution for the output, substitute the network's prediction for the distribution's parameters, then minimize the negative log-likelihood. Every standard loss function falls out of this framework.

### Problem 5.1

Show that the logistic sigmoid $\text{sig}[z] = \frac{1}{1 + \exp[-z]}$ maps $z = -\infty$ to 0, $z = 0$ to 0.5, and $z = \infty$ to 1.

$$\text{sig}(-\infty) = \frac{1}{1 + e^{\infty}} = \frac{1}{\infty} \to 0$$

$$\text{sig}(0) = \frac{1}{1 + e^{0}} = \frac{1}{2}$$

$$\text{sig}(\infty) = \frac{1}{1 + e^{-\infty}} = \frac{1}{1 + 0} = 1$$

### Problem 5.2

The binary cross-entropy loss for a single training pair $\{x, y\}$ where $y \in \{0, 1\}$ is:

$$L = -(1-y)\log\big[1 - \text{sig}[f[\mathbf{x}, \phi]]\big] - y\log\big[\text{sig}[f[\mathbf{x}, \phi]]\big]$$

When $y = 0$: $L = -\log[1 - \text{sig}[f[\mathbf{x}, \phi]]]$

When $y = 1$: $L = -\log[\text{sig}[f[\mathbf{x}, \phi]]]$

<div style="text-align: center;">
<svg viewBox="0 0 400 220" width="400" height="220" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; max-width: 100%;">
  <!-- Axes -->
  <line x1="50" y1="190" x2="370" y2="190" stroke="currentColor" stroke-width="1.2"/>
  <line x1="50" y1="20" x2="50" y2="190" stroke="currentColor" stroke-width="1.2"/>
  <!-- Labels -->
  <text x="210" y="215" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif">sig[f[x, φ]]</text>
  <text x="15" y="105" text-anchor="middle" fill="currentColor" font-size="12" font-family="Inter, sans-serif" transform="rotate(-90, 15, 105)">Loss, L</text>
  <!-- y=1 curve: -log(p), high loss at left, zero at right -->
  <path d="M 60,20 C 100,40 150,80 200,120 Q 250,155 360,185" fill="none" stroke="currentColor" stroke-width="2"/>
  <text x="90" y="45" fill="currentColor" font-size="11" font-family="Inter, sans-serif">y = 1</text>
  <!-- y=0 curve: -log(1-p), zero at left, high loss at right -->
  <path d="M 60,185 Q 170,155 220,120 C 270,80 320,40 360,20" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="6,3"/>
  <text x="310" y="45" fill="currentColor" font-size="11" font-family="Inter, sans-serif">y = 0</text>
  <!-- Tick marks -->
  <text x="50" y="205" text-anchor="middle" fill="currentColor" font-size="10">0</text>
  <text x="370" y="205" text-anchor="middle" fill="currentColor" font-size="10">1</text>
  <!-- Reference lines -->
  <text x="40" y="192" text-anchor="end" fill="currentColor" font-size="10">0</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Binary cross-entropy loss. When y=1 (solid), loss is high for predictions near 0 and zero for predictions near 1. When y=0 (dashed), the reverse.</p>

In both cases, the loss penalizes confident wrong predictions exponentially — a desirable property that makes gradient descent converge faster when the model is confidently wrong.

### Problem 5.3\*

Build a loss function for predicting wind direction $y$ (in radians) from barometric pressure $\mathbf{x}$ using the von Mises distribution.

Following the recipe:

**1. Choose distribution:** Von Mises with parameters $\mu$ (mean direction) and $\kappa$ (concentration):

$$Pr(y|\mu, \kappa) = \frac{\exp[\kappa \cos(y - \mu)]}{2\pi \cdot \text{Bessel}_0[\kappa]}$$

**2. Substitute network prediction:** Replace $\mu$ with $f[\mathbf{x}, \phi]$, treat $\kappa$ as constant:

$$Pr(y|f[\mathbf{x}, \phi], \kappa) = \exp[\kappa \cos(y - f[\mathbf{x}, \phi])]$$

(We drop the denominator since it doesn't depend on $\mu$ and won't affect the argmin.)

**3. Minimize negative log-likelihood:**

$$\begin{aligned}
\hat{\phi} &= \operatorname*{argmin}_{\phi}\bigg[-\sum_{i=1}^{I}\log Pr(y_i | f[\mathbf{x}_i, \phi])\bigg] \\
&= \operatorname*{argmin}_{\phi}\bigg[-\sum_{i=1}^{I} \kappa\cos[y_i - f[\mathbf{x}_i, \phi]]\bigg]
\end{aligned}$$

Since $\kappa$ is a positive constant that doesn't change the position of the minimum:

$$\boxed{L[\phi] = \cos[y_i - f[\mathbf{x}_i, \phi]]}$$

To perform inference, we compute $\hat{y} = f[\mathbf{x}, \hat{\phi}]$ — the predicted mean direction.

### Problem 5.4\*

Construct a loss function for a mixture of two Gaussians model with parameters $\boldsymbol{\theta} = \{\lambda, \mu_1, \sigma_1^2, \mu_2, \sigma_2^2\}$:

**1. Distribution:**

$$Pr(y|\boldsymbol{\theta}) = \frac{\lambda}{\sqrt{2\pi}\sigma_1}\exp\bigg[\frac{-(y-\mu_1)^2}{2\sigma_1^2}\bigg] + \frac{1-\lambda}{\sqrt{2\pi}\sigma_2}\exp\bigg[\frac{-(y-\mu_2)^2}{2\sigma_2^2}\bigg]$$

**2. Substitute predictions:** Since $\lambda \in [0,1]$, constrain it via sigmoid: $\lambda = \text{sig}[f_1[\mathbf{x}, \phi]]$. The means and variances come from separate network heads $f_2$ through $f_5$.

**3. Loss function:**

$$L[\phi] = -\sum_i \log\bigg[\frac{\text{sig}[f_1]}{\sqrt{2\pi}\,f_3^2}\exp\bigg[\frac{-(y_i - f_2)^2}{2f_3^2}\bigg] + \frac{1-\text{sig}[f_1]}{\sqrt{2\pi}\,f_5^2}\exp\bigg[\frac{-(y_i - f_4)^2}{2f_5^2}\bigg]\bigg]$$

Inference would be difficult since there is no closed-form expression for the mode of a mixture of Gaussians.

### Problem 5.5

Extend Problem 5.3 to predict wind direction using a mixture of two von Mises distributions:

$$Pr(y|\boldsymbol{\theta}) = \lambda \cdot \frac{\exp[\kappa_1 \cos(y - \theta_1)]}{2\pi\,\text{Bessel}_0[\kappa_1]} + (1-\lambda) \cdot \frac{\exp[\kappa_2 \cos(y - \theta_2)]}{2\pi\,\text{Bessel}_0[\kappa_2]}$$

Dropping the denominators (constant with respect to the means):

$$Pr(y|\boldsymbol{\theta}) \approx \lambda\,\exp[\kappa_1\cos(y - \theta_1)] + (1-\lambda)\,\exp[\kappa_2\cos(y - \theta_2)]$$

The network needs to predict **5 values**:
- $\lambda$ (mixing weight, constrained via sigmoid)
- $\kappa_1, \kappa_2$ (concentrations, i.e., inverse variances)
- $\theta_1, \theta_2$ (predicted directions)

### Problem 5.6

Design a loss function for predicting pedestrian counts $y \in \{0, 1, 2, \ldots\}$ using the Poisson distribution:

**1. Distribution:** $Pr(y = k) = \frac{\lambda^k e^{-\lambda}}{k!}$, parameter $\lambda > 0$ (the rate).

**2. Substitute:** Since $\lambda > 0$, use $\lambda = f[\mathbf{x}, \phi]^2$ (squaring ensures positivity).

$$Pr(y|f[\mathbf{x}, \phi]) = f[\mathbf{x}, \phi]^{2k} \cdot e^{-f[\mathbf{x}, \phi]^2}$$

(We drop the $k!$ denominator — it doesn't depend on $\phi$.)

**3. Negative log-likelihood:**

$$L[\phi] = -\sum_i \big[2k_i \log f[\mathbf{x}_i, \phi] - f[\mathbf{x}_i, \phi]^2\big] = \sum_i \big[f[\mathbf{x}_i, \phi]^2 - 2k_i\log f[\mathbf{x}_i, \phi]\big]$$

### Problem 5.7

For multivariate regression predicting $\mathbf{y} \in \mathbb{R}^{10}$ with independent normal distributions (each with mean $\mu_d$ predicted by the network and constant variance $\sigma^2$):

$$Pr(\mathbf{y}|f[\mathbf{x}, \phi]) = \prod_{d=1}^{10} \frac{1}{\sqrt{2\pi}\sigma} \exp\bigg[\frac{-(y_d - f_d[\mathbf{x}, \phi])^2}{2\sigma^2}\bigg]$$

Since $\sigma^2$ is constant, we can drop it and the normalizing constant:

$$Pr(\mathbf{y}|f[\mathbf{x}, \phi]) = \prod_{d=1}^{10} \exp\big[-(y_d - f_d[\mathbf{x}, \phi])^2\big]$$

Taking the negative log:

$$\hat{\phi} = \operatorname*{argmin}_{\phi} \sum_{d=1}^{10} (y_d - f_d[\mathbf{x}, \phi])^2$$

This is just the sum of squared errors — the familiar least squares loss. When variances are constant, maximum likelihood estimation reduces to ordinary least squares.

### Problem 5.8\*

Construct a loss for multivariate predictions $\mathbf{y}$ with independent normal distributions where both $\mu_d$ and $\sigma_d^2$ vary as functions of the input (a heteroscedastic model):

$$Pr(\mathbf{y}|\boldsymbol{\theta}) = \prod_d \frac{1}{\sqrt{2\pi}\,\sigma_d}\exp\bigg[\frac{-(y_d - \mu_d)^2}{2\sigma_d^2}\bigg]$$

Substituting network outputs $f_{1,d}[\mathbf{x}, \phi]$ for $\mu_d$ and $f_{2,d}[\mathbf{x}, \phi]$ for $\sigma_d$:

$$L[f[\mathbf{x}, \phi]] = -\log\prod_d Pr(y_d | f[\mathbf{x}, \phi])$$

$$= \sum_d \bigg[\log\big[\sqrt{2\pi}\,f_{1,d}[\mathbf{x}, \phi]\big] + \frac{(y_d - f_{2,d}[\mathbf{x}, \phi])^2}{2\,f_{1,d}[\mathbf{x}, \phi]^2}\bigg]$$

$$= \sum_d \log\big[\sqrt{2\pi}\,f_{1,d}[\mathbf{x}, \phi]^2\big] + \frac{(y_d - f_{2,d}[\mathbf{x}, \phi])^2}{2\,f_{1,d}[\mathbf{x}, \phi]^2}$$

Unlike the constant-variance case, we can't drop the log-variance term — it acts as a regularizer that prevents the network from making the variance arbitrarily large to reduce the squared error term.

### Problem 5.9\*

When predicting height (meters) and weight (kilos) from the same input, the units differ by roughly two orders of magnitude. A standard least squares loss will be dominated by weight prediction errors.

**Solution 1:** Rescale the outputs so both metrics have the same standard deviation, train the model, then rescale back during inference.

**Solution 2:** Learn separate variances for each output dimension (as in Problem 5.8). This lets the model handle the scaling automatically — the heteroscedastic loss naturally balances the contributions of each output dimension.

### Problem 5.10

Extend Problem 5.3 to predict both wind direction and speed. The likelihood combines a von Mises distribution for direction with an appropriate distribution for speed:

$$Pr(y|f[\mathbf{x}, \phi]) = \frac{\exp[f_1[\mathbf{x}, \phi] \cos(y - f_2[\mathbf{x}, \phi])]}{2\pi \cdot \text{Bessel}_0[f_1[\mathbf{x}, \phi]]}$$

The loss function:

$$L[f[\mathbf{x}, \phi]] = -f_1[\mathbf{x}, \phi]\cos[y - f_2[\mathbf{x}, \phi]] + \log\big[2\pi \cdot \text{Bessel}_0[f_1[\mathbf{x}, \phi]]\big]$$

Note that unlike Problem 5.3, we can no longer drop the Bessel function term because $f_1$ (the concentration parameter) is now predicted by the network rather than held constant.

---

## Chapter 6: Training & Optimization

All the theory from previous chapters is useless without a way to actually find good parameters. This chapter covers the mechanics of gradient descent, the conditions under which it behaves well (convexity), and the tricks that make it work in practice (stochastic batching, momentum).

### Problem 6.1

Show that the derivatives of the least squares loss $L[\phi] = \sum_i (\phi_0 + \phi_1 x_i - y_i)^2$ match equations 6.5 and 6.7 in the book.

By the chain rule:

$$\frac{\partial L[\phi]}{\partial \phi_0} = \sum_i 2(\phi_0 + \phi_1 x_i - y_i)$$

$$\frac{\partial L[\phi]}{\partial \phi_1} = \sum_i 2(\phi_0 + \phi_1 x_i - y_i) \cdot x_i$$

### Problem 6.2

The Hessian matrix of the linear regression loss function is:

$$\mathbf{H}[\phi] = \begin{bmatrix} \frac{\partial^2 L}{\partial \phi_0^2} & \frac{\partial^2 L}{\partial \phi_0 \partial \phi_1} \\ \frac{\partial^2 L}{\partial \phi_1 \partial \phi_0} & \frac{\partial^2 L}{\partial \phi_1^2} \end{bmatrix}$$

Computing each entry:

$$\frac{\partial^2 L}{\partial \phi_0^2} = 2, \quad \frac{\partial^2 L}{\partial \phi_0 \partial \phi_1} = 2x_i, \quad \frac{\partial^2 L}{\partial \phi_1^2} = 2x_i^2$$

$$\mathbf{H}[\phi] = \begin{bmatrix} 2 & 2x_i \\ 2x_i & 2x_i^2 \end{bmatrix}$$

To check convexity, we verify that both eigenvalues are positive by checking the trace and determinant:

- $\text{Tr}[\mathbf{H}] = 2(x_i^2 + 1) = 4x_i^2 > 0$ ✓
- $\det[\mathbf{H}] = 2(2x_i^2) - (2x_i)^2 = 4x_i^2 - 4x_i^2 = 0$

The determinant is zero, which means one eigenvalue is zero. The Hessian is positive *semi*-definite — the surface is convex but not strictly convex. It depends on $x_i$, meaning the curvature of the loss landscape changes with the data.

### Problem 6.3

Compute the derivatives of the least squares loss with respect to the Gabor model parameters $\phi_0$ and $\phi_1$:

$$f[x, \phi] = \sin[\phi_0 + 0.06 \cdot \phi_1 x] \cdot \exp\bigg(-\frac{(\phi_0 + 0.06 \cdot \phi_1 x)^2}{32.0}\bigg)$$

Using the product rule and chain rule:

$$\frac{\partial L}{\partial \phi_0} = \sum_i 2\bigg(\cos[\phi_0 + 0.06\phi_1 x_i] \exp\bigg(-\frac{(\phi_0 + 0.06\phi_1 x_i)^2}{32}\bigg) + \sin[\phi_0 + 0.06\phi_1 x_i] \cdot \bigg(-\frac{2(\phi_0 + 0.06\phi_1 x_i)}{32}\bigg)\exp\bigg(-\frac{(\phi_0 + 0.06\phi_1 x_i)^2}{32}\bigg)\bigg)$$

$$\frac{\partial L}{\partial \phi_1} = \sum_i 2\bigg(0.06x_i\cos[\phi_0 + 0.06\phi_1 x_i]\exp\bigg(-\frac{(\phi_0 + 0.06\phi_1 x_i)^2}{32}\bigg) + \sin[\phi_0 + 0.06\phi_1 x_i] \cdot \bigg(-\frac{2 \cdot 0.06 x_i(\phi_0 + 0.06\phi_1 x_i)}{32}\bigg)\exp\bigg(-\frac{(\phi_0 + 0.06\phi_1 x_i)^2}{32}\bigg)\bigg)$$

The Gabor model's loss landscape is highly non-convex, making it a good illustration of why gradient descent can get stuck in local minima.

### Problem 6.4\*

The logistic regression model for 1D binary classification:

$$Pr(y = 1|x) = \text{sig}[\phi_0 + \phi_1 x]$$

where $\text{sig}[z] = \frac{1}{1 + \exp[-z]}$.

The parameter $\phi_0$ controls the decision boundary position (where $Pr = 0.5$), and $\phi_1$ controls the steepness of the transition. A suitable loss function is the binary cross-entropy from Problem 5.2.

### Problem 6.5\*

Compute all ten partial derivatives of the least squares loss for the shallow network:

$$f[x, \phi] = \phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x] + \phi_2 a[\theta_{20} + \theta_{21}x] + \phi_3 a[\theta_{30} + \theta_{31}x]$$

The ReLU derivative introduces piecewise conditions. Noting that $\frac{\partial}{\partial z}\text{ReLU}[z] = \mathbb{1}[z > 0]$:

$$\frac{\partial L}{\partial \phi_0} = 2(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i)$$

$$\frac{\partial L}{\partial \phi_1} = 2(a[\theta_{10} + \theta_{11}x_i])(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i)$$

For the hidden unit parameters, the derivatives are piecewise:

$$\frac{\partial L}{\partial \theta_{10}} = \begin{cases} 2(\phi_1)(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i) & \text{if } \theta_{10} > -\theta_{11}x_i \\ 0 & \text{otherwise} \end{cases}$$

$$\frac{\partial L}{\partial \theta_{11}} = \begin{cases} 2(\phi_1 x_i)(\phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x_i] + \phi_2 a[\theta_{20} + \theta_{21}x_i] + \phi_3 a[\theta_{30} + \theta_{31}x_i] - y_i) & \text{if } \theta_{10} > -\theta_{11}x_i \\ 0 & \text{otherwise} \end{cases}$$

And similarly for $\theta_{20}, \theta_{21}, \theta_{30}, \theta_{31}$ — each gated by whether its respective hidden unit is active.

### Problem 6.6

Classify each critical point in the three 1D loss functions in figure 6.11:

| Point | Classification |
|-------|---------------|
| 1 | Local minimum |
| 2 | Global minimum |
| 3 | Local minimum |
| 4 | Neither (saddle/inflection) |
| 5 | Global minimum (ambiguous — could tie with 6) |
| 6 | Global minimum |
| 7 | Neither (saddle/inflection) |

A convex surface is one where no chord between two points on the surface dips below the function. Of the three loss functions, only (b) is convex.

### Problem 6.7\*

The gradient descent trajectory for path 1 in figure 6.5a oscillates because it follows the negative gradient, which points perpendicular to the contour lines. In a narrow valley, successive steps overshoot from side to side.

This happens because the next step $S_i$ is not orthogonal to the previous step $S_{i-1}$. Gradient descent walks along the curve by following the negative gradient, moving in the direction of maximum decrease until the loss stops decreasing. But for a narrow valley, the gradient is dominated by the steep walls rather than the gentle slope along the valley floor.

**Momentum** helps: by accumulating a running average of past gradients, the oscillations cancel out while the consistent downhill component reinforces, smoothing the trajectory along the valley.

### Problem 6.8\*

Can non-stochastic gradient descent with a fixed learning rate escape local minima?

**No.** At a local minimum, all gradients are zero. With zero gradients, there are no parameter updates — the algorithm is stuck. This is fundamentally different from stochastic gradient descent, where the random mini-batch sampling introduces noise that can push the parameters out of shallow local minima.

### Problem 6.9

With 1,000 iterations, a dataset of size 100, and a batch size of 20:

$$1000 \text{ iterations} \times 20 \text{ examples/iteration} = 20{,}000 \text{ examples processed}$$

$$\frac{20{,}000}{100} = \boxed{20 \text{ epochs}}$$

### Problem 6.10

Show that the momentum term $m_t$ (equation 6.11) is an infinite weighted sum of past gradients and derive the weights.

The momentum update rule is:

$$m_{t+1} \leftarrow \beta \cdot m_t + (1 - \beta)\sum_{i \in B_t}\frac{\partial \ell_i[\phi_t]}{\partial \phi}$$

Since $m_t$ is defined recursively, expanding it reveals that $m_t$ is an infinite weighted sum of all previous gradients, with exponentially decaying weights.

> *This solution is still in progress. Check back for updates.*

### Problem 6.11

If the model has one million parameters, the Hessian matrix has dimensions:

$$\mathbf{H} = n \times n = 1{,}000{,}000 \times 1{,}000{,}000$$

That's $10^{12}$ entries — one trillion floating-point numbers. This is why second-order optimization methods (which require the Hessian) are impractical for large neural networks, and we rely on first-order methods like SGD and Adam.

---

## Chapter 7: Backpropagation

Backpropagation is just the chain rule applied systematically. These problems strip away the abstraction and have you compute derivatives by hand through a small network — which is the best way to internalize what the algorithm actually does.

### Problem 7.1

For a two-layer network with two hidden units per layer:

$$y = \phi_0 + \phi_1 a\big[\psi_{01} + \psi_{11}a[\theta_{01} + \theta_{11}x] + \psi_{21}a[\theta_{02} + \theta_{12}x]\big] + \phi_2 a\big[\psi_{02} + \psi_{12}a[\theta_{01} + \theta_{11}x] + \psi_{22}a[\theta_{02} + \theta_{12}x]\big]$$

Computing all 13 derivatives directly (not using backpropagation):

$$\frac{\partial y}{\partial \phi_0} = 1, \qquad \frac{\partial y}{\partial \phi_1} = a[\psi_{01} + \psi_{11}a[\theta_{01} + \theta_{11}x] + \psi_{21}a[\theta_{02} + \theta_{12}x]]$$

$$\frac{\partial y}{\partial \phi_2} = a[\psi_{02} + \psi_{12}a[\theta_{01} + \theta_{11}x] + \psi_{22}a[\theta_{02} + \theta_{12}x]]$$

$$\frac{\partial y}{\partial \psi_{01}} = 1, \quad \frac{\partial y}{\partial \psi_{11}} = a[\theta_{01} + \theta_{11}x], \quad \frac{\partial y}{\partial \psi_{21}} = a[\theta_{02} + \theta_{12}x]$$

$$\frac{\partial y}{\partial \theta_{01}} = 1, \quad \frac{\partial y}{\partial \theta_{11}} = x$$

$$\frac{\partial y}{\partial \theta_{02}} = 1, \quad \frac{\partial y}{\partial \theta_{12}} = x$$

(Each derivative is gated by indicator functions $\mathbb{1}[\cdot > 0]$ from the ReLU — omitted here for clarity but present whenever a ReLU argument must be checked.)

### Problem 7.2

For the computation graph in equation 7.12, each of the five chains of derivatives terminates with:

$$\begin{aligned}
\frac{\partial h_3}{\partial f_2} &= \sin(f_2) \\
\frac{\partial f_5}{\partial h_2} &= \omega_2 \\
\frac{\partial h_2}{\partial f_1} &= \exp[f_1] \\
\frac{\partial f_3}{\partial h_1} &= \omega_1 \\
\frac{\partial h_1}{\partial f_0} &= \cos(f_0)
\end{aligned}$$

Each terminal derivative corresponds to the local gradient of a single node in the computation graph.

### Problem 7.3

The derivative $\frac{\partial \ell_i}{\partial f_0}$ in equation 7.19 is expressed as a chain:

$$\frac{\partial \ell_i}{\partial f_0} = \frac{\partial h_1}{\partial f_0} \cdot \left(\frac{\partial h_2}{\partial f_1} \cdot \frac{\partial h_3}{\partial f_2} \cdot \frac{\partial f_5}{\partial h_3} \cdot \frac{\partial \ell_i}{\partial f_5}\right)$$

The sizes of each term:

| Term | Type | Size |
|------|------|------|
| $\frac{\partial h_1}{\partial f_0}$ | Diagonal (activation derivative) | $D_1 \times D_1$ |
| $\frac{\partial f_1}{\partial h_1}$ | Weight matrix $\boldsymbol{\Omega}_1$ | $D_2 \times D_1$ |
| $\frac{\partial h_2}{\partial f_1}$ | Diagonal (activation derivative) | $D_2 \times D_2$ |
| $\frac{\partial f_2}{\partial h_2}$ | Weight matrix $\boldsymbol{\Omega}_2$ | $D_3 \times D_2$ |
| $\frac{\partial \ell_i}{\partial f_2}$ | Output gradient | $D_3 \times 1$ |

The diagonal matrices come from the element-wise activation derivative ($\text{diag}[\mathbb{1}[f_k > 0]]$ for ReLU), and the dense matrices are the weight matrices connecting layers. The product of these terms is exactly what backpropagation computes efficiently by reusing intermediate results rather than recomputing shared sub-expressions.

---

## Closing Notes

Working through these problems by hand was slow, occasionally frustrating, and exactly the right thing to do. There's a difference between reading that ReLU networks create piecewise linear functions and actually tracing the activation patterns through four regions. Between knowing that the loss landscape is non-convex and computing the Hessian to see the zero eigenvalue.

Some solutions here are clean. Others carry annotations where the algebra felt wrong or incomplete — those stay, because pretending otherwise would defeat the purpose.

More chapters may follow as I continue through the book. The full text is freely available at [udlbook.github.io/udlbook](https://udlbook.github.io/udlbook/).
