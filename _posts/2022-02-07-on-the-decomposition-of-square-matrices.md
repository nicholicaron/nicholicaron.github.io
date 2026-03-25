---
layout: post
title: "On the Decomposition of Square Matrices"
date: 2022-02-07
tags: [Math, Linear Algebra]
cover_image: /assets/images/matrix_decomp/honors_proj_ss-1.png
---

An extracurricular honors project I wrote at Georgia Highlands College under the supervision of Dr. Paul J. Kapitza. This paper surveys the major decomposition techniques for square matrices — LU, symmetric eigenvalue decomposition, Jordan decomposition, and the singular value decomposition — with a focus on building intuition for *when* and *why* each factorization is useful.

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

A matrix **A** can be thought of as a function which takes a vector **x** as input and outputs a transformed vector **Ax**. Eigenvectors are the special vectors for which **Ax** is parallel to **x**:

$$A\vec{x} = \lambda\vec{x}$$

where the stretching factor λ is the eigenvalue. Finding eigenvalues means solving det(**A** − λ**I**) = 0, which reduces to a characteristic polynomial.

### Orthogonal and Symmetric Matrices

Orthogonal matrices consist of perpendicular column vectors and have the elegant property **Q**<sup>T</sup>**Q** = **I**. For square orthogonal matrices, this means **Q**<sup>T</sup> = **Q**<sup>−1</sup> — the transpose *is* the inverse.

Symmetric matrices (**A** = **A**<sup>T</sup>) with real entries have a powerful guarantee: their eigenvalues are real, and there exists a complete set of orthonormal eigenvectors.

This leads to the **Spectral Theorem**: any real-symmetric matrix **A** can be decomposed as:

$$A = P\Lambda P^T = P\Lambda P^{-1}$$

where **P** is orthonormal and **Λ** is diagonal.

### A Worked Example

The paper walks through a complete decomposition of the matrix **A** = [[-2, 6], [6, -2]], finding eigenvalues λ₁ = −8 and λ₂ = 4, computing the corresponding eigenvectors, normalizing them to form the orthonormal matrix **P**, and assembling the full factorization.

![Eigenvalue decomposition worked example](/assets/images/matrix_decomp/7.png){: .post-image }
*Pages from the paper showing the symmetric eigenvalue decomposition worked example.*

---

## Chapter 2: Jordan Decomposition

The Jordan Decomposition factorizes a square matrix **A** into:

$$A = PDP^{-1}$$

where **D** is diagonal and **A** and **D** are *similar* matrices (they perform the same transformation, just expressed in different bases).

### Powers of a Matrix

This is where things get computationally beautiful. If **A** = **P** **D** **P**<sup>−1</sup>, then:

$$A^n = PD^nP^{-1}$$

Since **D** is diagonal, raising it to the nth power just means raising each diagonal entry to the nth power — a trivial operation. This avoids *n* full matrix multiplications (each O(m³) with the classical algorithm) and replaces them with three matrix multiplications and *m* scalar exponentiations.

The paper includes a worked example computing **A**<sup>1000</sup> for a 3×3 matrix, which elegantly reduces to the identity matrix.

### The Fibonacci Sequence

One of my favorite applications: the Fibonacci recurrence F<sub>k+2</sub> = F<sub>k+1</sub> + F<sub>k</sub> can be rewritten as a linear system using the transformation matrix **A** = [[1, 1], [1, 0]]. By diagonalizing this matrix, we arrive at a closed-form expression for the kth Fibonacci number:

$$F_k = \frac{1}{\sqrt{5}}\left(\frac{1+\sqrt{5}}{2}\right)^k - \frac{1}{\sqrt{5}}\left(\frac{1-\sqrt{5}}{2}\right)^k$$

The growth of the sequence is dominated by λ₁ = (1+√5)/2 — the golden ratio.

![Jordan decomposition and Fibonacci derivation](/assets/images/matrix_decomp/12.png){: .post-image }
*The Jordan decomposition applied to derive a closed-form Fibonacci formula.*

---

## Chapter 3: LU Decomposition

LU decomposition factors a square matrix **A** into the product of a **L**ower triangular matrix and an **U**pper triangular matrix:

$$A = LU$$

The procedure is systematic: perform Gaussian elimination on **A** to get **U**, track the elementary row operation matrices along the way, then multiply their inverses to recover **L**.

> "Practice yourself, for heaven's sake, in little things; and thence proceed to greater"
> — Epictetus

### The Fibonacci Matrix (Again)

The paper starts with a clean 2×2 example using the Fibonacci matrix **A** = [[1, 1], [1, 0]], which decomposes into **L** = [[1, 0], [1, 1]] and **U** = [[1, 1], [0, -1]].

### A 3×3 Example with Variables

The more involved example decomposes a parameterized 3×3 matrix, stepping through three elimination stages (E₁, E₂, E₃) and their inverses. The result reveals that the decomposition exists for all real values of the parameters — provided *a* ≠ 0 (we need non-zero pivots).

![LU decomposition 3x3 example](/assets/images/matrix_decomp/16.png){: .post-image }
*The 3×3 LU decomposition with variables, showing the full elimination sequence.*

---

## Chapter 4: The Singular Value Decomposition

The SVD is widely regarded as the most valuable matrix factorization. Unlike eigendecomposition, **an SVD exists for any matrix** — square or rectangular.

$$A = U\Sigma V^T$$

where **U** and **V** are orthogonal matrices and **Σ** is diagonal, containing the singular values (square roots of the eigenvalues of **A**<sup>T</sup>**A**).

The key insight is that we can find **V** by diagonalizing the symmetric positive definite matrix **A**<sup>T</sup>**A**, extract the singular values from its eigenvalues, and then solve for **U** using **U** = **A** **V** **Σ**<sup>−1</sup>.

The paper includes a complete 2×2 worked example for **A** = [[5, 5], [-1, 7]], computing **A**<sup>T</sup>**A**, its eigenvalues (20 and 80), the corresponding eigenvectors to form **V**, the singular values (2√5 and 4√5), and finally **U**.

### Applications

#### The Eckart-Young Theorem and Low-Rank Approximation

The Eckart-Young theorem states that truncating a matrix to rank *k* via SVD gives the *best possible* rank-k approximation. The singular values σᵢ indicate the relative importance of their corresponding column and row vectors in communicating the "signal" of the original matrix. By keeping only the largest singular values and discarding the rest, we separate signal from noise.

#### Image Compression

Since images can be represented as matrices of pixel intensities, SVD provides a natural compression scheme. A 1920×1080 greyscale image requires over 2 million entries. By keeping only the top-*r* singular values, we can dramatically reduce storage while preserving visual fidelity.

![Image compression via SVD](/assets/images/matrix_decomp/23.png){: .post-image }
*SVD-based image compression applied to the Georgia Highlands College campus.*

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
