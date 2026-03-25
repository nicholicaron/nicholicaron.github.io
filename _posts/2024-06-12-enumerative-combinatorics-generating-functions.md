---
layout: post
title: "Enumerative Combinatorics: Generating Functions"
date: 2024-06-12
tags: [Math, Combinatorics, Enumerative Combinatorics]
---

In [Part 1](/2024/06/05/enumerative-combinatorics-algebraic-methods.html) we developed PIE and induction — tools that work problem-by-problem, each requiring its own insight. Now we introduce the single most powerful idea in enumerative combinatorics: **generating functions**. The idea is deceptively simple: encode a sequence $(a_0, a_1, a_2, \ldots)$ as the coefficients of a power series, then use algebra on the series to discover formulas for the sequence. What makes this so effective is that operations on power series — multiplication, differentiation, composition — correspond to natural combinatorial operations on the sequences they encode.

This is Part 2 of a three-part series. We build the theory of ordinary and exponential generating functions, prove the Generalized Binomial Theorem, and apply the machinery to solve recurrences — culminating in a closed-form formula for the Fibonacci numbers.

## What This Post Covers

- **Ordinary Generating Functions** — Encoding sequences as power series
- **Formal Power Series** — The algebraic rules for manipulating generating functions
- **The Generalized Binomial Theorem** — Expanding $(1+cx)^m$ for any $m \in \mathbb{C}$
- **Coefficient Extraction** — Reading off answers and discovering identities
- **Exponential Generating Functions** — When order matters
- **Solving Recurrences** — A systematic four-step method, applied to Fibonacci and factorial sequences

---

## Ordinary Generating Functions

**Definition.** Given a sequence $(a_n)_{n \geq 0} = (a_0, a_1, a_2, \ldots)$, its **ordinary generating function** (OGF) is the formal power series

$$A(x) = \sum_{n=0}^{\infty} a_n x^n = a_0 + a_1 x + a_2 x^2 + \cdots$$

We treat $A(x)$ as an algebraic object — a formal power series — rather than an analytic function. Questions of convergence do not arise; what matters is the sequence of coefficients.

### First Examples

**Example.** If $a_n = 1$ for all $n \geq 0$, then

$$A(x) = \sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots$$

This is the geometric series. We verify the closed form $A(x) = \frac{1}{1-x}$ by checking that $(1-x)(1 + x + x^2 + \cdots) = 1$: the $x$ terms telescope, leaving only the constant term.

> The identity $\displaystyle\sum_{n=0}^{\infty} x^n = \frac{1}{1-x}$ is the single most important generating function formula. Nearly every OGF computation builds on it.

**Example.** If $b_n = n + 1$ for all $n \geq 0$, then

$$B(x) = \sum_{n=0}^{\infty} (n+1) x^n = 1 + 2x + 3x^2 + 4x^3 + \cdots$$

Notice that $B(x) = \frac{d}{dx}\left[\sum_{n=0}^{\infty} x^{n+1}/(n+1) \cdot (n+1)\right]$. More directly, $B(x) = \frac{d}{dx}\!\left[\frac{1}{1-x}\right]$, since differentiating $\sum x^n$ term-by-term gives $\sum n x^{n-1} = \sum (n+1)x^n$. Hence:

$$B(x) = \frac{1}{(1-x)^2}.$$

**Example.** Let $C(x) = \frac{1}{(1-x)(1-x^2)}$. What are its coefficients $(c_n)$?

We factor: $C(x) = \frac{1}{1-x} \cdot \frac{1}{1-x^2} = (1 + x + x^2 + \cdots)(1 + x^2 + x^4 + \cdots)$.

To find $c_n$, we collect all ways to pick $x^a$ from the first factor and $x^{2b}$ from the second with $a + 2b = n$. There are $\lfloor n/2 \rfloor + 1$ such pairs, so $c_n = \lfloor n/2 \rfloor + 1$.

---

## Formal Power Series Algebra

We treat generating functions as **formal power series**: algebraic objects where the rules are defined coefficient-by-coefficient, with no concern for convergence.

Given $A(x) = \sum a_n x^n$ and $B(x) = \sum b_n x^n$:

**Addition.** $A(x) + B(x) = \sum (a_n + b_n) x^n$.

**Scalar multiplication.** $\gamma \cdot A(x) = \sum \gamma\, a_n\, x^n$.

**Multiplication (Convolution).** The product $A(x) \cdot B(x)$ has coefficients given by the **convolution**:

$$A(x) \cdot B(x) = \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} a_k\, b_{n-k}\right) x^n.$$

This is the algebraic heart of generating functions: multiplying two OGFs corresponds to convolving their coefficient sequences.

**Reciprocal.** $1/B(x)$ is a well-defined formal power series if and only if $b_0 \neq 0$.

**Composition.** $A(B(x))$ is well-defined if and only if $b_0 = 0$ (ensuring each coefficient of the composite involves only finitely many terms).

**Formal derivative.** $\frac{d}{dx}[A(x)] = \sum_{n=1}^{\infty} n\, a_n\, x^{n-1} = \sum_{m=0}^{\infty} (m+1)\, a_{m+1}\, x^m$.

**Formal integral.** $\int A(x)\, dx = \sum_{n=0}^{\infty} a_n \frac{x^{n+1}}{n+1} = \sum_{m=1}^{\infty} \frac{a_{m-1}}{m}\, x^m$.

---

## The Generalized Binomial Theorem

The classical Binomial Theorem says $(1+x)^n = \sum_{k=0}^{n} \binom{n}{k} x^k$ for $n \in \mathbb{N}$. The generalization extends this to *any* exponent $m \in \mathbb{C}$ — at the cost of an infinite series.

**Definition.** For any $m \in \mathbb{C}$ and $k \in \mathbb{N}_0$, the **generalized binomial coefficient** is

$$\binom{m}{k} = \frac{(m)_k}{k!} = \frac{m(m-1)(m-2)\cdots(m-k+1)}{k!}.$$

When $m$ is a non-negative integer and $k > m$, one of the factors in the numerator is zero, so $\binom{m}{k} = 0$ and the sum is finite. For other values of $m$, all terms are nonzero.

**Theorem (Generalized Binomial Theorem).** For all $c, m \in \mathbb{C}$,

$$(1 + cx)^m = \sum_{k=0}^{\infty} \binom{m}{k} c^k x^k.$$

### Negative Exponents

A particularly useful special case: when $-m$ is a positive integer,

$$\binom{-m}{k} = (-1)^k \binom{m+k-1}{k}.$$

*Proof.* We compute both sides. The left side is $\frac{(-m)(-m-1)\cdots(-m-k+1)}{k!}$. Factoring out $(-1)^k$ from the $k$ terms in the numerator gives $(-1)^k \frac{m(m+1)\cdots(m+k-1)}{k!}$. Rewriting the rising product as a falling factorial: $\frac{(m+k-1)(m+k-2)\cdots m}{k!} = \binom{m+k-1}{k}$. $\square$

**Corollary.** For $m$ a positive integer:

$$\frac{1}{(1-cx)^m} = \sum_{k=0}^{\infty} \binom{m+k-1}{k} c^k x^k.$$

This is the **multiset coefficient** formula — $\binom{m+k-1}{k}$ counts the number of multisets of size $k$ from $m$ types, or equivalently, the non-negative integer solutions to $x_1 + \cdots + x_m = k$.

> The Binomial Theorem for integers gives *finite* sums. The Generalized Binomial Theorem for arbitrary exponents gives *infinite* series that serve as generating functions. This is the bridge between algebra and analysis in combinatorics.

### Exercises in Closed Form

**(a)** If $a_n = \binom{n+3}{n}$, then $A(x) = \sum \binom{n+3}{n} x^n = \frac{1}{(1-x)^4}$ by the corollary with $m = 4$, $c = 1$.

**(b)** The sequence $\binom{n+3}{n} = 4, 10, 20, 35, \ldots$ confirms this: it matches the coefficients of $\frac{1}{(1-x)^4}$.

**(c)** For $a_n = $ number of non-negative integer solutions to $z_1 + z_2 + z_3 = n$ where $z_1$ is odd, $z_2 \in \{1, 5, 8\}$, and $z_3 \geq 0$: build the OGF factor by factor. The contribution from odd $z_1$ is $x + x^3 + x^5 + \cdots = \frac{x}{1-x^2}$; from $z_2 \in \{1,5,8\}$ is $x + x^5 + x^8$; from unconstrained $z_3$ is $\frac{1}{1-x}$. So:

$$A(x) = \frac{x}{1-x^2}\left(x + x^5 + x^8\right)\frac{1}{1-x}.$$

---

## Coefficient Extraction and Identities

**Definition.** For a formal power series $A(x) = \sum a_n x^n$, we write

$$[A(x)]_{x^n} = a_n,$$

read as "the coefficient of $x^n$ in $A(x)$."

### Extracting Coefficients

**Example.** Let $A(x) = \frac{3x^2}{1-2x}$. Find $[A(x)]_{x^5}$.

We rewrite: $A(x) = 3x^2 \cdot \frac{1}{1-2x} = 3x^2 \sum_{n=0}^{\infty} (2x)^n = 3x^2 \sum_{n=0}^{\infty} 2^n x^n = \sum_{n=0}^{\infty} 3 \cdot 2^n x^{n+2}$.

So $[A(x)]_{x^5} = 3 \cdot 2^3 = 24$.

**Example.** Let $B(x) = \frac{1}{1-3x+2x^2}$. Find $[B(x)]_{x^k}$.

**Option 1 (Factoring).** Note $1-3x+2x^2 = (1-2x)(1-x)$, so $B(x) = \frac{1}{1-2x} \cdot \frac{1}{1-x} = \left(\sum 2^n x^n\right)\left(\sum x^n\right)$. By the convolution formula:

$$[B(x)]_{x^k} = \sum_{j=0}^{k} 2^j \cdot 1 = \frac{2^{k+1}-1}{2-1} = 2^{k+1} - 1.$$

**Option 2 (Partial Fractions).** Decompose $\frac{1}{(1-2x)(1-x)} = \frac{r}{1-2x} + \frac{s}{1-x}$.

Clearing denominators: $1 = r(1-x) + s(1-2x)$. Setting $x = 1$ gives $s = -1$; setting $x = 1/2$ gives $r = 2$. So $B(x) = \frac{2}{1-2x} - \frac{1}{1-x}$, and:

$$[B(x)]_{x^k} = 2 \cdot 2^k - 1 = 2^{k+1} - 1.$$

### The Central Binomial Coefficients

**Claim.** If $C(x) = \frac{1}{\sqrt{1-4x}}$, then $[C(x)]_{x^k} = \binom{2k}{k}$.

*Proof.* Write $C(x) = (1-4x)^{-1/2}$. By the Generalized Binomial Theorem:

$$C(x) = \sum_{n=0}^{\infty} \binom{-1/2}{n}(-4x)^n = \sum_{n=0}^{\infty} \binom{-1/2}{n}(-4)^n x^n.$$

We compute $\binom{-1/2}{n}$:

$$\binom{-1/2}{n} = \frac{(-1/2)(-3/2)(-5/2)\cdots(-(2n-1)/2)}{n!} = \frac{(-1)^n}{2^n} \cdot \frac{1 \cdot 3 \cdot 5 \cdots (2n-1)}{n!}.$$

Multiplying numerator and denominator by $2 \cdot 4 \cdot 6 \cdots (2n) = 2^n \cdot n!$:

$$\binom{-1/2}{n} = \frac{(-1)^n}{2^n} \cdot \frac{(2n)!}{n! \cdot 2^n \cdot n!} = \frac{(-1)^n}{4^n}\binom{2n}{n}.$$

Therefore $\binom{-1/2}{n}(-4)^n = \frac{(-1)^n}{4^n}\binom{2n}{n} \cdot (-4)^n = \binom{2n}{n}$. $\square$

### Identities from Squaring

Since $C(x)^2 = \frac{1}{1-4x} = \sum 4^n x^n$, we can extract coefficients from both sides:

$$[C(x)^2]_{x^m} = \sum_{k=0}^{m}\binom{2k}{k}\binom{2m-2k}{m-k} = 4^m.$$

This is a non-trivial identity derived purely from generating function algebra!

### The Convolution Formula

**Convolution Formula.** If $A(x) = \sum a_n x^n$ and $B(x) = \sum b_n x^n$, then

$$[A(x) B(x)]_{x^n} = \sum_{k=0}^{n} a_k\, b_{n-k}.$$

**Example.** From $(1+x)^{12} = (1+x)^7 (1+x)^5$, comparing coefficients of $x^k$:

$$\binom{12}{k} = \sum_{i=0}^{k} \binom{7}{i}\binom{5}{k-i}.$$

This is the **Vandermonde identity**, proved here without any combinatorial argument — just by comparing coefficients of a product.

### The Hockey Stick via OGFs

From $\frac{1}{(1-x)^{n+1}} = \frac{1}{(1-x)^n} \cdot \frac{1}{1-x}$, extract the coefficient of $x^k$:

$$\text{LHS: } \binom{n+k}{k}, \qquad \text{RHS: } \sum_{i=0}^{k} \binom{n+i-1}{i} \cdot 1 = \sum_{i=0}^{k}\binom{n-1+i}{i}.$$

Rewriting with $n \to n+1$: $\binom{n+1}{k+1} = \sum_{i=0}^{k} \binom{n-i}{k-i}$, which is the Hockey Stick Identity from [Part 1](/2024/06/05/enumerative-combinatorics-algebraic-methods.html) — now proved with zero induction.

### Integer Partitions

Let $a_n = p(n)$, the number of **integer partitions** of $n$ (ways to write $n$ as a sum of positive integers, ignoring order).

**Claim.** The OGF for partitions is the infinite product:

$$A(x) = \prod_{k=1}^{\infty} \frac{1}{1-x^k} = \frac{1}{1-x} \cdot \frac{1}{1-x^2} \cdot \frac{1}{1-x^3} \cdots$$

The factor $\frac{1}{1-x^k} = 1 + x^k + x^{2k} + \cdots$ accounts for using 0, 1, 2, ... copies of the part $k$. When we multiply out, the coefficient of $x^n$ counts the number of ways to select non-negative integers $(i_1, i_2, i_3, \ldots)$ with $1 \cdot i_1 + 2 \cdot i_2 + 3 \cdot i_3 + \cdots = n$ — exactly a partition of $n$.

---

## Exponential Generating Functions

Ordinary generating functions encode sequences as $\sum a_n x^n$. But some sequences are better served by a different encoding.

**Definition.** The **exponential generating function** (EGF) of a sequence $(a_n)$ is

$$A(x) = \sum_{n=0}^{\infty} a_n \frac{x^n}{n!}.$$

We extract coefficients using the notation $\left[A(x)\right]_{x^n/n!} = a_n$.

### Comparing OGF and EGF

The same sequence can have very different OGF and EGF.

| Sequence $a_n$ | OGF $\sum a_n x^n$ | EGF $\sum a_n \frac{x^n}{n!}$ |
|---|---|---|
| $a_n = 1$ | $\frac{1}{1-x}$ | $e^x$ |
| $b_n = n$ | $\frac{x}{(1-x)^2}$ | $xe^x$ |
| $c_n = 3^n$ | $\frac{1}{1-3x}$ | $e^{3x}$ |
| $d_n = n!$ | does not converge | $\frac{1}{1-x}$ |

The last row is striking: $d_n = n!$ grows too fast for its OGF $\sum n!\, x^n$ to converge anywhere (radius of convergence zero). But the EGF $\sum n! \cdot \frac{x^n}{n!} = \sum x^n = \frac{1}{1-x}$ is perfectly well-behaved. This is the key motivation for EGFs: they tame fast-growing sequences.

### Coefficient Extraction for EGFs

**Example.** $\left[e^{-4x}\right]_{x^6/6!} = \left[\sum \frac{(-4x)^n}{n!}\right]_{x^6/6!} = (-4)^6 = 4096$.

**Example.** $\left[\frac{1}{1-x}\right]_{x^n/n!} = \left[\sum x^n\right]_{x^n/n!} = n!$, since $\sum x^n = \sum n! \cdot \frac{x^n}{n!}$.

### The EGF Convolution Formula

The product of two EGFs yields a **binomial convolution**, which arises naturally when counting labeled structures.

**Theorem.** If $A(x) = \sum a_n \frac{x^n}{n!}$ and $B(x) = \sum b_n \frac{x^n}{n!}$, then

$$A(x) \cdot B(x) = \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} \binom{n}{k} a_k\, b_{n-k}\right) \frac{x^n}{n!}.$$

*Proof.* Compute directly:

$$\begin{aligned}
A(x) \cdot B(x) &= \left(\sum_i \frac{a_i}{i!} x^i\right)\left(\sum_j \frac{b_j}{j!} x^j\right) \\
&= \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} \frac{a_k}{k!} \cdot \frac{b_{n-k}}{(n-k)!}\right) x^n \\
&= \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} \frac{n!}{k!(n-k)!} a_k\, b_{n-k}\right) \frac{x^n}{n!} \\
&= \sum_{n=0}^{\infty} \left(\sum_{k=0}^{n} \binom{n}{k} a_k\, b_{n-k}\right) \frac{x^n}{n!}. \quad\square
\end{aligned}$$

> The binomial coefficient $\binom{n}{k}$ in the EGF convolution arises because EGFs naturally handle *labeled* objects: the $\binom{n}{k}$ chooses which $k$ labels go to one structure and which $n-k$ go to the other.

---

## Solving Recurrences with Generating Functions

Generating functions provide a systematic four-step method for solving linear recurrences.

### The Method

Suppose $(a_n)$ satisfies initial conditions $a_0 = c_0, \ldots, a_{r-1} = c_{r-1}$ and a recurrence $a_n = f(a_{n-1}, \ldots, a_{n-r})$ for $n \geq r$.

**(I)** Multiply both sides of the recurrence by $x^n$ and sum over all valid $n$.

**(II)** Let $A(x) = \sum a_n x^n$ and derive a functional equation for $A(x)$.

**(III)** Solve the functional equation for $A(x)$.

**(IV)** Compute $a_n = [A(x)]_{x^n}$.

### Example: A Simple Recurrence

Let $(a_n)$ be defined by $a_0 = 2$ and $a_n = 3a_{n-1}$ for $n \geq 1$.

**(I)** Multiply by $x^n$ and sum: $\sum_{n=1}^{\infty} a_n x^n = \sum_{n=1}^{\infty} 3a_{n-1} x^n$.

**(II)** The LHS is $A(x) - a_0 = A(x) - 2$. The RHS is $3x \sum_{n=1}^{\infty} a_{n-1} x^{n-1} = 3x A(x)$.

So $A(x) - 2 = 3x A(x)$.

**(III)** Solving: $A(x)(1-3x) = 2$, hence $A(x) = \frac{2}{1-3x}$.

**(IV)** Extract: $a_n = [A(x)]_{x^n} = \left[\frac{2}{1-3x}\right]_{x^n} = 2 \cdot 3^n$.

### Example: The Fibonacci Numbers

The Fibonacci sequence $(f_n)$ is defined by $f_0 = 1$, $f_1 = 1$, $f_n = f_{n-1} + f_{n-2}$ for $n \geq 2$.

**(I)** Sum the recurrence: $\sum_{n=2}^{\infty} f_n x^n = \sum_{n=2}^{\infty} f_{n-1} x^n + \sum_{n=2}^{\infty} f_{n-2} x^n$.

**(II)** Let $F(x) = \sum_{n=0}^{\infty} f_n x^n$. Then:

- LHS: $F(x) - f_1 x - f_0 = F(x) - x - 1$.
- First term of RHS: $x \sum_{n=2}^{\infty} f_{n-1} x^{n-1} = x\sum_{m=1}^{\infty} f_m x^m = x(F(x) - 1)$.
- Second term: $x^2 \sum_{n=2}^{\infty} f_{n-2} x^{n-2} = x^2 F(x)$.

So: $F(x) - x - 1 = x(F(x) - 1) + x^2 F(x) = xF(x) - x + x^2 F(x)$.

Simplifying: $F(x) - xF(x) - x^2 F(x) = 1$, hence $F(x)(1 - x - x^2) = 1$.

**(III)** Solve:

$$F(x) = \frac{1}{1 - x - x^2}.$$

**(IV)** To extract $f_n$, we use **partial fractions**. Factor $1-x-x^2 = (1-ax)(1-bx)$ where $a, b$ are the roots of $t^2 - t - 1 = 0$ (after substituting $x = 1/t$):

$$a = \frac{1+\sqrt{5}}{2} = \varphi, \qquad b = \frac{1-\sqrt{5}}{2} = \hat{\varphi}.$$

Decompose:

$$\frac{1}{(1-ax)(1-bx)} = \frac{P}{1-ax} + \frac{Q}{1-bx}.$$

Clearing denominators and solving: $P = \frac{a}{a-b} = \frac{a}{\sqrt{5}}$, $Q = \frac{b}{b-a} = \frac{-b}{\sqrt{5}}$.

Extracting $[x^n]$:

$$f_n = P \cdot a^n + Q \cdot b^n = \frac{a^{n+1}}{\sqrt{5}} - \frac{b^{n+1}}{\sqrt{5}}.$$

This gives **Binet's formula**:

$$\boxed{f_n = \frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^{n+1} - \frac{1}{\sqrt{5}}\left(\frac{1-\sqrt{5}}{2}\right)^{n+1}.}$$

> It is remarkable that the Fibonacci numbers — which are always integers — are expressed as a difference of powers of irrational numbers. The second term $\lvert b \rvert^{n+1}/\sqrt{5}$ shrinks to zero exponentially, so $f_n$ is simply the nearest integer to $\varphi^{n+1}/\sqrt{5}$.

### Example: When OGF Fails — The Factorial Sequence

Let $(a_n)$ be defined by $a_0 = 1$ and $a_n = n \cdot a_{n-1}$ for $n \geq 1$. We expect $a_n = n!$.

**Attempting the OGF method:**

**(I-II)** Let $A(x) = \sum a_n x^n$. The recurrence gives $A(x) - 1 = x \frac{d}{dx}[x A(x)]$. This is a **differential equation** for $A(x)$, which is very hard to solve. Moreover, $A(x) = \sum n! \, x^n$ has radius of convergence zero — the OGF simply does not work well here.

**The EGF trick:** Replace $A(x) = \sum a_n x^n$ with the EGF $A(x) = \sum a_n \frac{x^n}{n!}$.

**(I)** The recurrence $a_n = n \cdot a_{n-1}$ becomes $\sum_{n=1}^{\infty} a_n \frac{x^n}{n!} = \sum_{n=1}^{\infty} n \cdot a_{n-1} \frac{x^n}{n!}$.

**(II)** LHS $= A(x) - 1$. RHS $= \sum_{n=1}^{\infty} a_{n-1} \frac{x^n}{(n-1)!} = x \sum_{n=1}^{\infty} a_{n-1} \frac{x^{n-1}}{(n-1)!} = x A(x)$.

So $A(x) - 1 = x A(x)$.

**(III)** Solving: $A(x)(1-x) = 1$, hence $A(x) = \frac{1}{1-x}$.

**(IV)** Extract: $a_n = \left[\frac{1}{1-x}\right]_{x^n/n!} = \left[\sum x^n\right]_{x^n/n!} = n!$, since $\sum x^n = \sum n! \cdot \frac{x^n}{n!}$.

> The moral: when a recurrence involves factors of $n$ (like $a_n = n \cdot a_{n-1}$), the OGF leads to differential equations, but the EGF absorbs the factorial and yields a simple algebraic equation. This is the key heuristic for choosing between OGF and EGF.

---

## Looking Ahead

We now have a complete algebraic toolkit: OGFs, EGFs, the Binomial Theorem, convolutions, and coefficient extraction. In [Part 3: Famous Number Families](/2024/06/19/enumerative-combinatorics-number-families.html), we apply these tools to study the great number families of combinatorics — multinomial coefficients, Fibonacci and Lucas numbers (now from a combinatorial perspective), and the Catalan numbers, whose generating function satisfies a quadratic equation.
