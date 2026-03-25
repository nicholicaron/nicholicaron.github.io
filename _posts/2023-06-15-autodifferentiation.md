---
layout: post
title: "Autodifferentiation"
date: 2023-06-15
tags: [AI, Math, Deep Learning]
cover_image: /assets/images/autodif/autodiff.jpg
---

We build autodifferentiation (AD) from the ground up. Having a deep understanding of AD is essential for anyone doing serious deep learning research — it's the mechanism that makes training neural networks possible, and understanding it transforms how you think about computation itself.

## What This Post Covers

- **Numerical vs. Symbolic vs. Automatic Differentiation** — Why AD is neither finite differences nor symbolic math, and why that distinction matters
- **Forward Mode AD** — Dual numbers, the chain rule applied left-to-right, and when forward mode is efficient
- **Reverse Mode AD (Backpropagation)** — The computational graph, adjoint variables, and why reverse mode dominates deep learning
- **The Tape** — How modern AD frameworks record operations for later differentiation
- **Implementation** — Building a minimal autodiff engine in Python that supports scalar and tensor operations

## Why Build It From Scratch?

Frameworks like PyTorch and JAX abstract away the differentiation machinery. This is powerful for productivity, but dangerous for understanding. When your gradients explode, vanish, or produce NaN — and they will — you need a mental model of what's actually happening inside `loss.backward()`. Building AD from scratch gives you that model.

*In progress — full implementation and derivations coming soon.*
