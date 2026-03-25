---
layout: post
title: "Compiler Design & Implementation"
date: 2023-03-21
tags: [Compilers, Computer Science]
cover_image: /assets/images/compilers/intro/SIGABA-patent-2.png
---

## 15-411 Compiler Design

**15-411 — Carnegie Mellon Course that I will be Auditing**

It's been a long time since my last post. In addition to continuing my Math undergrad, I've been working in IT -- first as a Helpdesk Technician, and more recently as a System Administrator. It has been hard to find the time for my personal projects. I would spend a spare hour here and there but I was missing out on the economies of scale that accompany deep, uninterrupted focus. Several of my peers' career goals were to become System Administrators, and I had reached that point without much effort. While I could have continued down the IT path and lived a very comfortable life, I felt that I was still settling. Two quotes really sparked something within me to take a step back and go all in on my goals:

> "Will inertia be your guide or will you follow your passions?"
> — Jeff Bezos

> "The important thing is this: to be able at any moment to sacrifice what we are for what we could become"
> — Charles Du Bos

I felt like I was leaving a lot on the table, and that an unchecked complacency had set in. So I quit my job as a System Administrator, and enrolled back into school full-time to finish out my math degree while also pursuing computer science and machine learning. This is not something I would recommend for anyone in a different situation than mine, but sometimes the most impressive things are done by those who were too naive to know they couldn't. I say all of this, because recently it has become far more feasible for me to pursue these kinds of projects.

So here we are: compilers. I've always been fascinated by the intersection of mathematics and computer science, and compilers sit right at the heart of that intersection. They are among the most complex and well-studied systems that us programmers use on a day-to-day basis. They are what allows us to abstract away the details of the hardware that runs the code we write -- reducing our mental load. This affords us great leverage, and provides automated double-checking of our code to help ensure correctness. By better understanding the tools of the trade, we become more efficient artisans.

> "For the programmer, compiler construction reveals the cost of the abstractions that programs use... A programmer cannot effectively tune an application's performance without knowing the costs of its individual parts."
> — Cooper & Torczon (Engineering a Compiler)

Additionally, with the end of Dennard Scaling and Moore's Law, we can no longer rely on regular improvements in clock speeds to improve our programs performance for us. The future of speedups will likely have to come from improvements in architecture and parallelism -- through improvements in optimization and code generation via specialized hardware/software co-design. Compilers are a critical piece to this puzzle.

## Why learn about compilers?

> "You are a mass of mass-produced, off-the-shelf components, and so is every piece of software that you use. Your life is lived on top of hardware, firmware, operating systems, network stacks, programming languages, runtime libraries, and applications designed and assembled by thousands of others. And that entire tall, teetering stack of abstractions, upon which your very consciousness rests, bottlenecks through compilers and the hardware they produce code for. You should know this. You should understand this."

> "Ultimately, because of control. Your entire life is entombed in an elaborate, entirely abstract labyrinth of machines that define so many aspects of every moment of your waking life. You're here to better understand the prison you find yourself within. You're here to sympathize with the machine."
> — Tyler Neely

The real reason to learn about compilers is so we can utilize the infamous Double Compile whenever we find ourselves in desperate times.

## Why use Rust?

Well, if the question is a general one, see this [previous article](/2022/02/08/why-rust.html). If you mean why use Rust for writing compilers, then see the introduction section to this online book. It is extremely important for the literal thing that creates your binaries that you run on your machine to, itself, be as robust as possible for obvious reasons. To quote David Wheeler's paper *Countering Trusting Trust through Diverse Double-Compiling*: "compilers can be subverted to insert malicious Trojan horses into critical software, including themselves."

If there's one piece of software that you want to be formally verified and memory safe, it's the compiler. On that note, if there's a second piece of software that you want to be formally verified and memory safe, it's the operating system. There's a good chance that Redox OS will be the first general-purpose OS written entirely in Rust.

## A Personal Aside

I know AI may be all the rage, but I tend to admire systems programmers — and more generally, people who choose the more difficult paths in life. Since the consequence of admiration is emulation, this, mixed with masochism and a proclivity for patterns, has led me to develop a penchant for tinkering with low-level systems.

Some people I look up to: John Carmack, Chris Lattner, Jim Keller, Jon Gjengset, Dan Luu, Brandon Falk, and George Hotz.

I also enjoy learning languages. In high school I took five Spanish classes, plus Arabic and American Sign Language. My dad's side of the family are Francophones from Quebec, so that's next on my list. Languages in general are all about expressing intent (semantics) via agreed-upon form (syntax) and abstractions. One of the primary differences between spoken languages and programming languages is ambiguity.

## What Compilers Make Good Use Of

Compilers sit at the intersection of nearly every subfield in CS. From *Engineering a Compiler*:

- **Greedy Algorithms** — Register allocation
- **Heuristic Search** — List scheduling
- **Graph Algorithms** — Dead-code elimination
- **Dynamic Programming** — Instruction selection
- **Automata Theory** — Parsing & scanning
- **Fixed-Point Algorithms** — Data-flow analysis

And they *apply* theory in practice in concrete ways:

| Theory Applied | Where in the Compiler |
|---|---|
| Formal language theory | Scanners & parsers |
| Lattice theory, number theory | Type checking & static analysis |
| Tree-pattern matching, dynamic programming | Code generators |

> "Compiler construction brings together ideas and techniques from across the breadth of computer science and applies them in a constrained setting to solve some truly hard problems."
> — Cooper & Torczon

## Fundamental Principles of Compilation

Two axioms govern all compiler design:

1. **The compiler must preserve the meaning of the input program.** Correctness lies at the heart of the social contract between you and your users as the author of a compiler.
2. **The compiler must discernibly improve the input program.**

> "Good compilers approximate the solutions to hard problems. They emphasize efficiency, in their own implementations and in the code they generate."

> "A successful compiler executes an unimaginable number of times... Thus, compiler writers must pay attention to compile time costs, such as the asymptotic complexity of algorithms and the space used by data structures."

And for JIT compilers, the stakes are even higher:

> "JIT construction may be the ultimate feat of compiler engineering. Because the system must recoup compile time through improved running time..."

## The Anatomy of a Compiler

The compilation process can be broken down into stages, each responsible for a specific transformation of the source code. At a high level, a compiler takes human-readable source code and transforms it into machine code that can be executed by a processor.

![Compiler structure](/assets/images/compilers/intro/compiler_structure-3.png){: .post-image }

### Front End: Source → IR

The front end encodes the intent (semantics) of the source program into an **Intermediate Representation** (IR) — some set of data structures to represent the code it processes. The **definitive IR** is the version of the program passed between independent phases (e.g., front end → back end).

- **Scanner** — Tokenizes the source code
- **Parser** — Builds a parse tree from the tokens
- **Elaborator** — Performs additional computation: type checking, constant folding, laying out storage, building the IR

### Optimizer (Optional): IR → IR

May make several passes over the IR, recursively integrating metadata and creating shortcuts from previous passes to optimize for some metric.

- **Data-flow analysis** — Typically solves a system of simultaneous set equations based on facts derived from the IR
- **Control-flow analysis** — Computes information like dominance relations
- **Dependence analysis** — Uses number-theoretic tests to reason about the relative independence of memory references

### Back End: IR → Target ISA

- **Instruction Selector** — Translates IR into the target machine's ISA (x86, ARM, RISC-V, JVM, etc.)
- **Register Allocator** — Assigns registers to variables
- **Instruction Scheduler** — Orders instructions to minimize pipeline stalls

## Thinking About Time

When building compilers, several temporal slots matter:

- **Design time** — Choosing algorithms and data structures
- **Build time** — Constructing the compiler itself
- **Compile time** — When the compiler runs on user code
- **Runtime** — When the compiled program executes

The key insight: behavior that occurs at one time is *planned* at another. A compiler writer must reason about which data structures exist at each phase and how decisions made at build time affect performance at compile time and runtime.

## Separate Compilation

Most real compilers invoke separately for each source file:

- Each source file compiles independently into an **object file**
- A **linker** combines the object files into a single executable

**Advantages:** Faster incremental compilation (only changed files recompile), easier code reuse (libraries compile once, used by many programs).

**Disadvantages:** More complex build systems (dependency tracking), more complex debugging (which object file has the error?).

## Written Assignments

#### Assignment 1: Backend
Code generation and register allocation.

#### Assignment 2: Frontend
Lexical analysis and parsing.

#### Assignment 3: Middle
Semantic analysis and intermediate representations.

#### Assignment 4: Semantics
Type systems and formal verification.

#### Continuing with this project — What's next?

Beyond the CMU course, we'll also explore using digital logic and FPGAs to create a RISC-V board to run our compiled binaries on! Let's also beef up our compiler with Cornell's CS 6120: Advanced Compilers: The Self-Guided Online Course.

## Primary References

- *Engineering a Compiler* — Cooper & Torczon
- *Crafting Interpreters* — Robert Nystrom
- *Computer Organization and Design RISC-V Edition*
- *Hacker's Delight*
