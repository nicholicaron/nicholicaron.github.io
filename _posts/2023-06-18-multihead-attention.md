---
layout: post
title: "Take Heed of the Hydra"
date: 2023-06-18
tags: [Transformers, AI, Math]
cover_image: /assets/images/multihead_attn/Hydra_ss-3463311282.jpg
---

Implementing Multi-Head Attention and Optimized Matrix Multiplication via Strassen's Algorithm. This post walks through the transformer's core mechanism at the matrix level — not as a black box, but as a series of linear algebra operations you can trace by hand.

## What This Post Covers

- **Scaled Dot-Product Attention** — The QKV formulation, why we scale by $$\sqrt{d_k}$$, and the softmax bottleneck
- **Multi-Head Attention** — Parallel attention heads, concatenation, and the projection back to model dimension
- **Strassen's Algorithm** — Asymptotically faster matrix multiplication and its implications for attention computation
- **Implementation** — Building multi-head attention from scratch in PyTorch, with a focus on computational efficiency

## Why the Hydra?

The mythological Hydra — a creature with many heads, each operating independently but serving a single body — is a fitting metaphor for multi-head attention. Each attention head learns to focus on different aspects of the input: syntactic relationships, semantic similarity, positional patterns. The power comes from the diversity of perspectives, unified through concatenation and projection.

*In progress — full derivations and implementation coming soon.*
