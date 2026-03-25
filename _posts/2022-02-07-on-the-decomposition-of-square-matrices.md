---
layout: post
title: "On the Decomposition of Square Matrices"
date: 2022-02-07
tags: [Math, Linear Algebra]
cover_image: /assets/images/matrix_decomp/honors_proj_ss-1.png
---

An extracurricular paper I wrote for an introductory Linear Algebra course. Matrix decompositions are among the most powerful tools in applied mathematics — they reveal the hidden structure of linear transformations and make otherwise intractable computations feasible.

This paper surveys the major decomposition techniques for square matrices, with a focus on building intuition for when and why each decomposition is useful.

## Decompositions Covered

- **LU Decomposition** — Gaussian elimination as matrix factorization. The workhorse of numerical linear algebra for solving systems of equations.
- **QR Decomposition** — Orthogonal factorization via Gram-Schmidt, Householder reflections, or Givens rotations. Essential for least-squares problems and eigenvalue algorithms.
- **Eigendecomposition** — Diagonalization of matrices with linearly independent eigenvectors. The bridge between linear algebra and dynamical systems.
- **Singular Value Decomposition (SVD)** — The most general decomposition. Works for any matrix, reveals rank, range, and null space in a single factorization.
- **Cholesky Decomposition** — The efficient special case for symmetric positive definite matrices. Half the work of LU, with better numerical stability.

## Why Decompositions Matter

Every time you train a neural network, compress an image, solve a differential equation, or run PCA on a dataset, you're relying on matrix decompositions. They are the computational backbone of modern scientific computing.

**Reference:** The full paper is available as a [PDF download](/assets/pdfs/GHC_Honors_Nicholi-2.pdf).
