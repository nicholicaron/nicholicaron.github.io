---
layout: post
title: "Compilers Pt. 2: Parsers"
date: 2023-05-30
tags: [Compilers, Computer Science]
cover_image: /assets/images/compilers/parser/rm_linguistics-102950941.jpg
---

Part 2 of the [Compiler Design & Implementation](/2023/03/21/compiler-design-and-implementation.html) series. Parsing is where the compiler determines the validity of an input program's syntax. Conceptually, the parser takes the flat stream of tokens produced by the scanner and gives it structure — a tree that captures the grammatical relationships between language constructs.

## What This Post Covers

- **Context-Free Grammars** — Productions, derivations, and the Chomsky hierarchy
- **Parse Trees vs. Abstract Syntax Trees** — What information we keep and what we discard
- **Top-Down Parsing** — Recursive descent, LL(1) grammars, and FIRST/FOLLOW sets
- **Bottom-Up Parsing** — Shift-reduce, LR parsing, and why most production compilers use LALR(1)
- **Error Recovery** — Panic mode, phrase-level recovery, and producing useful error messages
- **Implementation** — Building a recursive descent parser in Rust for the C0 language

## The Parser's Job

The scanner doesn't understand structure. It sees `if`, `(`, `x`, `>`, `0`, `)`, `{` as a flat sequence of tokens. The parser's job is to recognize that this sequence forms an if-statement, with a condition and a body — and to reject sequences that don't form valid programs.

*In progress — full implementation coming soon.*
