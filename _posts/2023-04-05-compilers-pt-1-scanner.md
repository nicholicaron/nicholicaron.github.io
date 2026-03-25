---
layout: post
title: "Compilers Pt. 1: Scanners"
date: 2023-04-05
tags: [Compilers, Computer Science]
cover_image: /assets/images/compilers/scanner/turing-machine.png
---

Part 1 of the [Compiler Design & Implementation](/2023/03/21/compiler-design-and-implementation.html) series. This post provides background on scanners, regular expressions, and finite automata — then implements a simple scanner for the C0 language in Rust.

## What This Post Covers

- **Lexical Analysis** — What the scanner does and why it's the first phase of compilation
- **Regular Expressions** — The formal language for specifying token patterns
- **Finite Automata** — NFAs, DFAs, and the subset construction algorithm
- **Thompson's Construction** — Converting regular expressions to NFAs systematically
- **DFA Minimization** — Hopcroft's algorithm for producing optimal recognizers
- **Implementation** — A hand-written scanner for C0 in Rust, handling identifiers, keywords, operators, literals, and whitespace

## Why Hand-Write a Scanner?

Tools like Flex and re2c can generate scanners automatically from regular expression specifications. But hand-writing a scanner teaches you what these tools do under the hood, gives you full control over error handling and performance, and — for simple languages — often produces cleaner, faster code than generated alternatives.

*In progress — full implementation coming soon.*
