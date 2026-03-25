---
layout: post
title: "On the Decomposition of Square Matrices"
date: 2022-02-07
tags: [Math, Linear Algebra]
cover_image: /assets/images/matrix_decomp/honors_proj_ss-1.png
---

An extracurricular honors project I wrote at Georgia Highlands College under the supervision of Dr. Paul J. Kapitza. This paper surveys the major decomposition techniques for square matrices — symmetric eigenvalue decomposition, Jordan decomposition, LU decomposition, and the singular value decomposition — with a focus on building intuition for *when* and *why* each factorization is useful.

> "In the language of Computer Science, the expression of [a matrix] A as a product amounts to a pre-processing of the data in A, organizing that data into two or more parts whose structures are more useful in some way, perhaps more accessible for computation."
> — David C. Lay

**The full paper is available as a [PDF download](/assets/pdfs/GHC_Honors_Nicholi-2.pdf).**

---

## Background

I tend to take a top-down approach when learning new materials — looking at a subject from a macro-level to understand the context of its pieces before delving into the details. Professor Kapitza showed me that inverting that approach, spending more time on the fundamentals, can lead to substantial revelations. This paper was my attempt to do exactly that: start from the concrete mechanics of matrix factorizations and build upward to their applications.

Matrix decompositions are among the most powerful tools in applied mathematics. The core idea is simple: take a complex matrix and express it as a product of simpler matrices whose structure we understand better. Every time you train a neural network, compress an image, solve a differential equation, or run PCA on a dataset, you are relying on one of these factorizations.

An important takeaway from this survey is that these decompositions are *related by their form* with progressively demanding requirements and constructions.

---

## Chapter 1: Symmetric Eigenvalue Decomposition

### Eigenvalues and Eigenvectors

A matrix $A$ can be thought of as a function which takes a vector $\vec{x}$ as input and outputs a transformed vector $A\vec{x}$. Eigenvectors are the special vectors for which $A\vec{x}$ is parallel to $\vec{x}$:

$$A\vec{x} = \lambda\vec{x}$$

where the factor by which $\vec{x}$ is stretched (the constant $\lambda$) is the eigenvalue corresponding to that particular eigenvector.

**Determining Eigenvalues.** Eigenvalues of a matrix $A$ can be found by solving for $\lambda$ in the equation:

$$\det(A - \lambda I) = 0$$

The polynomial this reduces to is referred to as the characteristic polynomial of $A$. If there are any repeated eigenvalues, the numbers of times they repeat are referred to as their multiplicities.

**Determining Eigenvectors.** To find an eigenvector $\vec{e}$ associated with a particular eigenvalue $\lambda$, we must substitute $\lambda$ into the equation $\det(A - \lambda I) = 0$ and bring the resulting matrix to row reduced echelon form, solving for all $x_n$.

### Orthogonal Matrices

Orthogonal matrices consist of orthogonal (i.e. perpendicular) column vectors. The dot product of two orthogonal vectors is equal to zero. Conveniently, orthogonal matrices have the property

$$Q^TQ = I$$

When dealing with orthogonal *square* matrices, they have the additional property

$$Q^T = Q^{-1}$$

### Symmetric Matrices

A symmetric matrix is one for which $A = A^T$. If a symmetric matrix has some special properties, its eigenvalues and eigenvectors likely have special properties as well:

- If $A$ consists of real number entries, its eigenvalues are real and there exists a complete set of eigenvectors that are orthogonal and of unit length (orthonormal).

**Theorem 1.1 (Spectral Theorem).** *We can decompose any Hermitian (i.e. real-symmetric) matrix $A$ with the symmetric eigenvalue decomposition (SED):*

$$A = \sum_{i=1}^{n} \lambda_i p_i p_i^T = P\Lambda P^T = P\Lambda P^{-1},$$

$$\Lambda = \text{diag}(\lambda_1 \cdots \lambda_n)$$

*where $P$ is orthonormal.*

*The following corollaries are used in the examples:*

1. *For any real-symmetric matrix, there are exactly $n$ (not necessarily distinct) real eigenvalues.*
2. *The associated eigenvectors can be chosen to form an orthonormal basis.*
3. *A factorization of the form $P\Lambda P^T$ can be crafted given the previous information.*

### A Worked Example

Find the eigenvalues, eigenvectors, and the symmetric eigenvalue decomposition of the matrix

$$A = \begin{bmatrix} -2 & 6 \\ 6 & -2 \end{bmatrix}$$

- First, let's find $A$'s eigenvalues:

$$\det\left(\begin{bmatrix} -2 - \lambda & 6 \\ 6 & -2 - \lambda \end{bmatrix}\right) = 0$$

$$\begin{aligned}
\Rightarrow \lambda^2 + 4\lambda - 32 &= 0 \\
\Rightarrow (\lambda + 8)(\lambda - 4) &= 0
\end{aligned}$$

$$\lambda_1 = -8, \quad \lambda_2 = 4$$

- Next, we'll substitute our eigenvalues into $(A - \lambda I)$ to find their associated eigenvectors.

**Case** $\lambda_1 = -8$:

$$\begin{bmatrix} -2 & 6 \\ 6 & -2 \end{bmatrix} - \begin{bmatrix} -8 & 0 \\ 0 & -8 \end{bmatrix} = \begin{bmatrix} 6 & 6 \\ 6 & 6 \end{bmatrix}$$

To find the eigenvector $\vec{e_1}$ corresponding with this eigenvalue, we must bring this matrix to row reduced echelon form:

$$\begin{bmatrix} 6 & 6 \\ 6 & 6 \end{bmatrix} \Rightarrow \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$$

$$\begin{cases} x_1 = -x_2 \\ x_2 \text{ is free} \end{cases}$$

$$\Rightarrow \vec{x} = \begin{bmatrix} -x_2 \\ x_2 \end{bmatrix} = x_2 \begin{bmatrix} -1 \\ 1 \end{bmatrix}, \quad \vec{e_1} = \begin{bmatrix} -1 \\ 1 \end{bmatrix}$$

**Case** $\lambda_2 = 4$:

$$\begin{bmatrix} -2 & 6 \\ 6 & -2 \end{bmatrix} - \begin{bmatrix} 4 & 0 \\ 0 & 4 \end{bmatrix} = \begin{bmatrix} -6 & 6 \\ 6 & -6 \end{bmatrix}$$

$$\begin{bmatrix} -6 & 6 \\ 6 & -6 \end{bmatrix} \Rightarrow \begin{bmatrix} 1 & -1 \\ 0 & 0 \end{bmatrix}$$

$$\begin{cases} x_1 = x_2 \\ x_2 \text{ is free} \end{cases}$$

$$\Rightarrow \vec{x} = \begin{bmatrix} x_2 \\ x_2 \end{bmatrix} = x_2 \begin{bmatrix} 1 \\ 1 \end{bmatrix}, \quad \vec{e_2} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$$

- Therefore the eigenvectors of $A$ are:

$$\vec{e_1}, \vec{e_2} = \left\\{ \begin{bmatrix} -1 \\ 1 \end{bmatrix}, \begin{bmatrix} 1 \\ 1 \end{bmatrix} \right\\}$$

These vectors are already orthogonal, as $\vec{e_1} \cdot \vec{e_2} = 0$. Let's normalize them to form our orthonormal matrix $P$:

$$\vec{p_1} = \frac{\vec{e_1}}{||\vec{e_1}||} = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{-1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}$$

$$\vec{p_2} = \frac{\vec{e_2}}{||\vec{e_2}||} = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{bmatrix}$$

$$P = \begin{bmatrix} \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}$$

- Applying Theorem 1.1: $A = P\Lambda P^{-1}$
  - Because $P$ happens to be symmetric, $P = P^T$
  - Also, since $P$ is orthogonal and square, $P^T = P^{-1}$
  - Therefore $P = P^T = P^{-1}$

$$\begin{bmatrix} -2 & 6 \\ 6 & -2 \end{bmatrix} = \begin{bmatrix} \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} -8 & 0 \\ 0 & 4 \end{bmatrix} \begin{bmatrix} \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \quad \checkmark$$

---

## Chapter 2: Jordan Decomposition

### Background

**Positive Definiteness.** A symmetric, square matrix $A$ is positive definite if:

- $\vec{x}^T A \vec{x}$ is positive except when $\vec{x} = \vec{0}$.

For a symmetric matrix $A$:

1. All of $A$'s eigenvalues are greater than 0,
2. The determinant of $A$ is greater than 0,
3. All pivots of $A$ are greater than 0.

**Similarity.** Two matrices $A$ and $D$ are similar if

$$D = P^{-1}AP$$

for some matrix $P$. This allows us to group together matrices that perform similar transformations.

**The Factorization.** The Jordan Decomposition is the factorization of a square matrix $A$ into the product of three matrices:

$$A = PDP^{-1}$$

where:

1. $A$ and $D$ are similar matrices,
2. $D$ is a diagonal matrix,
3. $P^{-1}$ is the matrix inverse of $P$.

### Powers of a Matrix

Let's say we want to square a matrix $A$; supposing $A$ has a Jordan decomposition:

$$\begin{aligned}
A &= PDP^{-1} \\
A^2 &= PDP^{-1}PDP^{-1} \\
\Rightarrow A^2 &= PD^2P^{-1}
\end{aligned}$$

Clearly, this pattern can be extrapolated to higher powers, giving us the general form:

$$A^n = PD^nP^{-1}$$

Since $D$ is diagonal, this works out quite nicely:

$$\Rightarrow A^n = P \begin{bmatrix} d_1^n & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & d_m^n \end{bmatrix} P^{-1}$$

This makes for a great reduction in complexity. We can avoid $n$ matrix multiplications, which can be very expensive computations — namely $O(m^3)$ scalar multiplications each, using the classical algorithm. And instead only perform three matrix multiplications and raise $m$ scalar values to the $n^{th}$ power, which is relatively trivial.

### A Worked Example

> "The heart of mathematics consists of concrete examples and concrete problems."
> — P. R. Halmos

Given the matrices

$$A = \begin{bmatrix} 1 & -2 & 8 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix}, \quad P = \begin{bmatrix} 1 & -4 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}.$$

Confirm that $P$ diagonalizes $A$, then compute $A^{1000}$.

1. First, we have to find $P^{-1}$:

$$P^{-1} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & -1 & 4 \end{bmatrix}.$$

2. By rearranging the previous equation, the product of $P^{-1}AP$ should give us a diagonal matrix $D$:

$$D = P^{-1}AP$$

$$D = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & -1 & 4 \end{bmatrix} \begin{bmatrix} 1 & -2 & 8 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{bmatrix} \begin{bmatrix} 1 & -4 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}$$

$$\Rightarrow D = \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$P$ diagonalizes $A$. $\checkmark$

3. Next we have to solve for $A$:

$$A = PDP^{-1}$$

Since $A$ and $D$ are similar:

$$A^n = PD^nP^{-1}$$

Thus

$$A^{1000} = PD^{1000}P^{-1}$$

$$A^{1000} = P \begin{bmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 1 \end{bmatrix}^{1000} P^{-1}$$

$$\Rightarrow A^{1000} = P \begin{bmatrix} (-1)^{1000} & 0 & 0 \\ 0 & (-1)^{1000} & 0 \\ 0 & 0 & 1^{1000} \end{bmatrix} P^{-1}$$

$$\Rightarrow A^{1000} = PI_3P^{-1}$$

$$\Rightarrow A^{1000} = PP^{-1}$$

$$A^{1000} = I_3$$

where $I_3$ is the $3 \times 3$ identity matrix.

### The Fibonacci Sequence

The Fibonacci Sequence is an example of an additive recurrence with initial values of 0 and 1. The next digit of the Fibonacci Sequence is the sum of its two preceding values and can be represented by the form

$$F_{k+2} = F_{k+1} + F_k.$$

Although this is a second order scalar equation, it can be converted to a first order linear system by letting

$$\vec{u_k} = \begin{bmatrix} F_{k+1} \\ F_k \end{bmatrix}$$

1. $F_{k+2} = F_{k+1} + F_k$,
2. $F_{k+1} = F_1$.

This can be rewritten as the linear system

$$\vec{u_{k+1}} = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} \vec{u_k}.$$

Thus, our transformation matrix

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}.$$

The state of a difference equation can be represented by a vector $\vec{u_k}$ where the second entry is the preceding state, and the first entry is the current state. Difference equations can be represented by the product of some growth matrix $A$ and a seed vector $\vec{u_0}$:

$$\vec{u_0} = A^k u_0 = \sum_{n=1}^{n} c_n \lambda_n^k \vec{x_n}$$

**Finding a Formula for the $k^{th}$ Fibonacci Number:**

- First, we must calculate the eigenvalues of $A$:

$$\det\left(\begin{bmatrix} 1 - \lambda & 1 \\ 1 & -\lambda \end{bmatrix}\right) = 0$$

$$\Rightarrow \lambda^2 - \lambda - 1 = 0$$

$$\lambda = \frac{1 \pm \sqrt{5}}{2}.$$

- Next, let's find our eigenvectors:

$$(A - \lambda I)\vec{x} = \begin{bmatrix} 1 - \lambda & 1 \\ 1 & -\lambda \end{bmatrix} \vec{x} = \vec{0}$$

when $\vec{x} = \begin{bmatrix} \lambda \\ 1 \end{bmatrix}$:

$$\Rightarrow \vec{x_1} = \begin{bmatrix} \lambda_1 \\ 1 \end{bmatrix}, \quad \vec{x_2} = \begin{bmatrix} \lambda_2 \\ 1 \end{bmatrix}.$$

- We can use our initial conditions to solve for our constants:

$$\vec{u_0} = \begin{bmatrix} F_1 \\ F_0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = c_1 x_1 + c_2 x_2$$

$$\Rightarrow c_1 = -c_2 = \frac{1}{\sqrt{5}}.$$

- Finally, since $\begin{bmatrix} F_{k+1} \\ F_k \end{bmatrix} = c_1 \lambda_1 x_1 + c_2 \lambda_2 x_2$, we can plug in our now-known quantities to reveal a closed form expression:

$$F_k = \frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^k - \frac{1}{\sqrt{5}}\left(\frac{1-\sqrt{5}}{2}\right)^k.$$

The eigenvalues of a first order system can determine how the system performs as the inputs increase. Since $\lambda_1 = \frac{1+\sqrt{5}}{2}$ is the only eigenvalue with an absolute value greater than 1, the growth of the Fibonacci Sequence in the limit is determined by $\lambda_1$, and can thus be approximated by

$$F_k \approx \frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^k.$$

The growth of the sequence is dominated by $\lambda_1 = (1+\sqrt{5})/2$ — the golden ratio.

---

## Chapter 3: LU Decomposition

LU decomposition factors a square matrix $A$ into the product of a **L**ower triangular matrix and an **U**pper triangular matrix:

$$A = LU$$

- $L$ is a lower triangular matrix, i.e., all entries above the diagonal are equal to zero.
- $U$ is an upper triangular matrix, i.e., all entries below the diagonal are equal to zero.

### Finding L and U

1. To compute a LU decomposition, we perform Gaussian elimination on our original matrix $A$, until it is of upper triangular form (not necessarily reduced echelon form). This newfound upper triangular matrix will serve as $U$.
2. Along the way, we must keep track of which row operations we have used, as well as the order in which they were applied. The elementary matrices which represent the row operations will be multiplied successively on the left of $A$.
3. Lastly, we have to isolate $A$. To do so, we must multiply on the left of both sides by the inverse product of our elementary matrices to reveal our lower triangular matrix $L$.

### A 2x2 Example

> "Practice yourself, for heaven's sake, in little things; and thence proceed to greater"
> — Epictetus

Find an LU Decomposition of the previously established Fibonacci matrix:

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix}.$$

- To turn $A$ into an upper triangular matrix, we must subtract Row 1 from Row 2. Representing this row operation as an elementary matrix gives us

$$E_1 = \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix}.$$

- Multiplying our original matrix $A$ on the left by this elementary matrix is equivalent to performing the row operation, which gives us the equation

$$E_1 A = U$$

$$\begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}.$$

- All that's left is to multiply on the left of both sides by $E_1^{-1}$:

$$E_1^{-1} E_1 A = E_1^{-1} U$$

$$\begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ -1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}.$$

- This simplifies down to $I_2 A = E^{-1}U$. Since $E^{-1}$ is of lower triangular form,

$$I_2 A = LU$$

$$\Rightarrow A = LU$$

$$\Rightarrow \begin{bmatrix} 1 & 1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 0 & -1 \end{bmatrix}. \quad \checkmark$$

### A 3x3 Example with Variables

Find an LU Decomposition of the matrix $A$

$$A = \begin{bmatrix} 1 & 0 & 1 \\ a & a & a \\ b & b & a \end{bmatrix},$$

for which real numbers $a$ and $b$ does it exist?

1. Our primary objective is to bring $A$ to an upper triangular matrix by way of Gaussian elimination.

   a) First we'd like to clear the row 2, column 1 entry. To do so, we will subtract $a$ times row 1 from row 2, which is equivalent to multiplying $A$ on the left by the matrix $E_1$:

   $$E_1 = \begin{bmatrix} 1 & 0 & 0 \\ -a & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

   $$E_1 A = U^*$$

   $$\begin{bmatrix} 1 & 0 & 0 \\ -a & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ a & a & a \\ b & b & a \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & a & 0 \\ b & b & a \end{bmatrix}.$$

   b) Next, we want to eliminate the $U^*_{3,1}$ entry. To complete this, we will subtract $b$ times row 1 from row 3, or effectively multiply $E_1 A$ on the left by $E_2$:

   $$E_2 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -b & 0 & 1 \end{bmatrix}$$

   $$E_2 E_1 A = U^*$$

   $$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -b & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ -a & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ a & a & a \\ b & b & a \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & a & 0 \\ 0 & b & a - b \end{bmatrix}.$$

   c) Our last objective is to clear the $U^*_{3,2}$ entry. We will multiply on the left by the matrix $E_3$ which corresponds to a subtraction of $b/a$ times row 2 from row 3:

   $$E_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -b/a & 1 \end{bmatrix}$$

   $$E_3 E_2 E_1 A = U$$

   $$\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & -b/a & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ -b & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ -a & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ a & a & a \\ b & b & a \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & a & 0 \\ 0 & 0 & a - b \end{bmatrix}.$$

2. Subsequently, we have to isolate $A$ by multiplying both sides of the equation on the left by the inverse product of our elimination matrices:

$$(E_3 E_2 E_1)^{-1}(E_3 E_2 E_1)A = (E_3 E_2 E_1)^{-1}U$$

$$\Rightarrow I_3 A = E_1^{-1} E_2^{-1} E_3^{-1} U.$$

The product of inverse elimination matrices will give us our lower triangular matrix $L$:

$$L = E_1^{-1} E_2^{-1} E_3^{-1}$$

$$\Rightarrow L = \begin{bmatrix} 1 & 0 & 0 \\ a & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ b & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & b/a & 1 \end{bmatrix}$$

$$\Rightarrow L = \begin{bmatrix} 1 & 0 & 0 \\ a & 1 & 0 \\ b & b/a & 1 \end{bmatrix}.$$

Therefore we have that $A = LU$, where

$$\Rightarrow \begin{bmatrix} 1 & 0 & 1 \\ a & a & a \\ b & b & a \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ a & 1 & 0 \\ b & b/a & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & a & 0 \\ 0 & 0 & a - b \end{bmatrix}. \quad \checkmark$$

Conclusion: The matrix $A$ has an LU Decomposition for all real values of $a$ and $b$; given $a \neq 0$.

---

## Chapter 4: The Singular Value Decomposition

The SVD is widely regarded as the most valuable matrix factorization. Unlike eigendecomposition, **an SVD exists for any matrix** — square or rectangular. SVD is efficiently computed for large matrices, and can be used to approximate a matrix to arbitrary precision using the Eckart-Young Low Rank Approximation Theorem.

$$A = U\Sigma V^T$$

where:

- $U$ is an orthogonal matrix,
- $\Sigma$ is a diagonal matrix and consists of singular values, i.e. $\sigma_n$ (square roots of the eigenvalues of $A$), and
- $V$ is an orthogonal matrix.

If a matrix $A$ is symmetric positive definite — $A = A^T$, its eigenvectors are orthogonal and consist of positive, real values — we are allowed to use a special case of the SVD where $U = V = P$, allowing for the familiar form

$$A = P\Lambda P^T$$

In section 1.3, we established that, given a symmetric $n \times n$ matrix, we can form a set of $n$ eigenvectors. It is known that the matrix $A^TA$ is symmetric for a square matrix $A$ and shares many important properties with $A$. These properties allow an SVD of $A$ to be performed on the symmetric matrix $A^TA$, even if our original matrix $A$ is not symmetric.

### Calculation

The key part of finding the singular value decomposition is finding an orthonormal basis $\vec{v}$ for the row space of $A$ where

$$A \begin{bmatrix} v_1 & v_2 & \ldots & v_n \end{bmatrix} = \begin{bmatrix} \sigma_1 u_1 & \sigma_2 u_2 & \ldots & \sigma_n u_n \end{bmatrix}$$

where $\vec{u}$ is an orthonormal basis for the column space of $A$, and $\sigma_n$ are termed singular values. After incorporating the nullspaces, and making $\vec{v}$ and $\vec{u}$ to be orthonormal bases for the entire space $\mathbb{R}^n$ the equation becomes

$$AV = U\Sigma$$

Because $V$ is orthogonal, we can multiply both sides on the right by $V^{-1} = V^T$ to isolate $A$:

$$A = U\Sigma V^T$$

It wouldn't be wise to attempt to solve for $U$, $V$, and $\Sigma$ simultaneously, so instead we will multiply both sides on the left by their transpose $A^T = V\Sigma^T U^T$ where $\Sigma$ can be substituted for $\Sigma^T$ since $\Sigma$ is diagonal and $U^{-1}$ can be substituted for $U^T$ since $U$ is orthogonal, to get the result

$$A^TA = V\Sigma U^{-1} U\Sigma V^T$$

$$\Rightarrow A^TA = V\Sigma^2 V^T$$

$$A^TA = V \begin{bmatrix} \sigma_1^2 & 0 & 0 & 0 \\ 0 & \sigma_2^2 & 0 & 0 \\ 0 & 0 & \ddots & 0 \\ 0 & 0 & 0 & \sigma_n^2 \end{bmatrix} V^T$$

We can find $V$ by diagonalizing the symmetric positive definite matrix $A^TA$. The eigenvectors of $A^TA$ form the columns of $V$. Similarly, the eigenvalues of $A^TA$ form the diagonal elements of $\Sigma^2$ ($\sigma_i^2$), where $\sigma_i$ are the corresponding positive square roots of $\lambda_i$.

To find $U$, we can either repeat this process with $AA^T$, or use the now known matrices $A$, $V$, and $\Sigma$ to solve for $U$:

$$\begin{aligned}
AV &= U\Sigma \\
\Rightarrow AV\Sigma^{-1} &= U\Sigma\Sigma^{-1} \\
\Rightarrow AV\Sigma^{-1} &= U.
\end{aligned}$$

### A Worked Example

> "... a concrete life preserver thrown to students sinking in a sea of abstraction"
> — W. Gottschalk

Determine the singular value decomposition of the matrix

$$A = \begin{bmatrix} 5 & 5 \\ -1 & 7 \end{bmatrix}.$$

- First, we must find $A^TA$:

$$A^TA = \begin{bmatrix} 5 & -1 \\ 5 & 7 \end{bmatrix} \begin{bmatrix} 5 & 5 \\ -1 & 7 \end{bmatrix}$$

$$\Rightarrow A^TA = \begin{bmatrix} 26 & 18 \\ 18 & 74 \end{bmatrix}.$$

- Next, let's find the eigenvalues of $A^TA$:

$$\det(A^TA - \lambda I) = 0$$

$$\det\left(\begin{bmatrix} 26 - \lambda & 18 \\ 18 & 74 - \lambda \end{bmatrix}\right) = 0$$

$$\begin{aligned}
\Rightarrow (26 - \lambda)(74 - \lambda) - 18^2 &= 0 \\
\Rightarrow \lambda^2 - 100\lambda + 1600 &= 0 \\
\Rightarrow (\lambda - 20)(\lambda - 80) &= 0
\end{aligned}$$

$$\lambda_1 = 20, \quad \lambda_2 = 80.$$

- After finding our eigenvalues of $A^TA$, we must find their associated eigenvectors to form $V$.

**Case** $\lambda_1 = 20$:

$$A^TA - 20I = \begin{bmatrix} 6 & 18 \\ 18 & 54 \end{bmatrix}$$

$$v_1 = \begin{bmatrix} -3 \\ 1 \end{bmatrix}.$$

Time to normalize $v_1$:

$$v_1 = \begin{bmatrix} \frac{-3}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} \end{bmatrix}.$$

**Case** $\lambda_2 = 80$:

$$A^TA - 80I = \begin{bmatrix} -54 & 18 \\ 18 & -6 \end{bmatrix}$$

$$v_2 = \begin{bmatrix} 1 \\ 3 \end{bmatrix}.$$

Time to normalize $v_2$:

$$v_2 = \begin{bmatrix} \frac{1}{\sqrt{10}} \\ \frac{3}{\sqrt{10}} \end{bmatrix}.$$

Thus,

$$V = \begin{bmatrix} \frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \end{bmatrix}.$$

Since $V$ is symmetric, $V^T = V$:

$$V^T = \begin{bmatrix} \frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \end{bmatrix}.$$

- Now let's use the eigenvalues of $A^TA$ to form $\Sigma$:

$$\Sigma^2 = \begin{bmatrix} \sigma_1^2 & 0 \\ 0 & \sigma_2^2 \end{bmatrix} = \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} = \begin{bmatrix} 20 & 0 \\ 0 & 80 \end{bmatrix}$$

$$\Rightarrow \Sigma = \begin{bmatrix} \sqrt{20} & 0 \\ 0 & \sqrt{80} \end{bmatrix}$$

$$\Sigma = \begin{bmatrix} 2\sqrt{5} & 0 \\ 0 & 4\sqrt{5} \end{bmatrix}.$$

- Next we find $U$:

$$U = AV\Sigma^{-1}$$

$$U = \begin{bmatrix} 5 & 5 \\ -1 & 7 \end{bmatrix} \begin{bmatrix} \frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \end{bmatrix} \begin{bmatrix} \frac{1}{2\sqrt{5}} & 0 \\ 0 & \frac{1}{4\sqrt{5}} \end{bmatrix}$$

$$U = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix}.$$

- Finally, we can conclude our factorization of $A$:

$$A = U\Sigma V^T$$

$$\begin{bmatrix} 5 & 5 \\ -1 & 7 \end{bmatrix} = \begin{bmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{-1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \end{bmatrix} \begin{bmatrix} 2\sqrt{5} & 0 \\ 0 & 4\sqrt{5} \end{bmatrix} \begin{bmatrix} \frac{-3}{\sqrt{10}} & \frac{1}{\sqrt{10}} \\ \frac{1}{\sqrt{10}} & \frac{3}{\sqrt{10}} \end{bmatrix}. \quad \checkmark$$

### Applications

#### The Eckart-Young Theorem and Low-Rank Approximation

> "The signal is the truth. The noise is what distracts us from the truth."
> — Nate Silver

**Theorem 4.1 (Eckart-Young).** *If a matrix $B$ has rank $k$ then*

$$||A - A_k|| \leq ||A - B||.$$

This can be interpreted as saying that truncating $A$ to rank $k$ gives a better approximation to $A$ than any other matrix $B$, given $B$ is rank $k$ or lower.

**Consequences of the Eckart-Young Theorem.** The Eckart-Young theorem is an important extension of the singular value decomposition in which a matrix may be approximated to arbitrary precision by truncating our matrices $U$, $\Sigma$, and $V^T$ to a common rank $r$, of our choice. The truncated matrix $A$ is not only a good approximation, it is the best rank $r$ approximation to $A$ possible.

In order to accurately approximate our original matrix $A$, our matrices must be arranged in a particular fashion. The diagonal entries of $\Sigma$, $\sigma_i$ correspond to the column vectors $\vec{u_i}$, and the row vectors $\vec{v_i}^T$. The values of $\sigma_i$ indicate the relative importance of their corresponding $\vec{u_i}$ and $\vec{v_i}^T$ in communicating the signal of $A$. Therefore, the more important a set of column vectors and row vectors are, the greater their analogous $\sigma_i$ value will be. Once our matrices are in order, we can begin extracting signal from our original matrix:

1. $U$ will be truncated to only include its first $r$ columns,
2. $\Sigma$ will be truncated to include its first $r$ columns and $r$ rows, becoming an $r \times r$ matrix,
3. $V^T$ will be truncated to include its first $r$ rows;

$$A \approx \begin{bmatrix} | & & | \\ \vec{u_1} & \ldots & \vec{u_r} \\ | & & | \end{bmatrix} \begin{bmatrix} \sigma_1 & 0 & 0 \\ 0 & \ddots & 0 \\ 0 & 0 & \sigma_r \end{bmatrix} \begin{bmatrix} — & \vec{v_1} & — \\ & \vdots & \\ — & \vec{v_r} & — \end{bmatrix}^T.$$

#### Image Compression

Since images can be represented as matrices of pixel intensities, SVD provides a natural compression scheme. A 1920x1080 greyscale image requires over 2 million entries. By keeping only the top-$r$ singular values, we can dramatically reduce storage while preserving visual fidelity.

![Image compression via SVD](/assets/images/matrix_decomp/23.png){: .post-image }
*SVD-based image compression applied to the Georgia Highlands College campus, showing approximations at ranks 1, 5, 30, 65, 80, and 100.*

#### Collaborative Filtering

Netflix, Amazon, and other recommender systems use SVD-based techniques to extract patterns from user-item interaction matrices. By decomposing a sparse ratings matrix, collaborative filtering can predict what a user might enjoy based on the preferences of similar users.

#### Latent Semantic Indexing

LSI uses the truncated SVD to process natural language, enabling computers to see past *synonymy* (different words, same meaning) and *polysemy* (one word, multiple meanings) by analyzing word usage structure across documents.

---

## Why This Matters

These four decompositions form the computational backbone of modern scientific computing and machine learning:

- **LU** is the workhorse behind solving systems of linear equations
- **Eigendecomposition** powers PCA, spectral clustering, and dynamical systems analysis
- **Jordan form** gives us efficient matrix exponentiation for recurrences and differential equations
- **SVD** underlies everything from image compression to recommendation engines to NLP

Understanding them at the matrix level — not just as API calls — is what separates engineers who can debug numerical instability from those who can't.

---

*This paper was written as an extracurricular honors project at Georgia Highlands College. Thanks to Dr. Paul J. Kapitza for supervising the project and for demonstrating that spending time on fundamentals leads to substantial revelations.*

**[Download the full paper (PDF)](/assets/pdfs/GHC_Honors_Nicholi-2.pdf)**
