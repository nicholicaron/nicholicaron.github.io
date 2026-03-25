---
layout: post
title: "Convolutions of Grandeur"
date: 2023-06-17
tags: [CNNs, Computer Vision, AI, Math]
cover_image: /assets/images/cnns/IMG-5203-1.jpg
---

A dive into computer vision via Convolutional Neural Networks and how the Winograd Convolution is implemented. CNNs are one of the most elegant examples of inductive bias in deep learning — the assumption that spatial locality matters, baked directly into the architecture.

## What This Post Covers

- **Convolution as a Linear Operation** — The convolution theorem, Toeplitz matrices, and why convolution is really just structured matrix multiplication
- **CNN Architectures** — From LeNet to ResNet, how the field evolved its understanding of depth and skip connections
- **The Winograd Transform** — A lesser-known optimization that reduces the number of multiplications in convolution by trading them for additions, yielding significant speedups for small filter sizes
- **Implementation Details** — Memory layout (NCHW vs NHWC), im2col tricks, and why GPU implementations don't always match the textbook

## Why Winograd?

Standard convolution requires $$O(m \cdot k^2)$$ multiplications per output element. Winograd's minimal filtering algorithm reduces this to roughly $$O(m)$$ multiplications at the cost of more additions and a larger transformed tile. For 3x3 filters — the most common in modern architectures — this translates to real-world speedups of 2-4x.

*In progress — full derivations and benchmarks coming soon.*
