---
layout: post
title: "Multiply-Free: A Ternary LLM Engine on a 130-Dollar FPGA"
date: 2026-06-09
tags: [FPGA, LLM Inference, BitNet, Hardware, Energy Efficiency]
cover_image: /assets/images/ternfpga/fpga-ballout.jpg
---

On a batch-1 decode of a two-billion-parameter language model, an RTX 3060 turns in **3.67 joules per token**. A six-year-old desktop CPU does **4.62** — almost as well. For a 400-dollar GPU against a commodity processor, that is a rout that isn't: the GPU is barely winning, and it is actually *slower* (23.5 tokens/second versus the CPU's 28.4).

The reason is the punchline of this whole post. The model's weights are **ternary** — every number is $-1$, $0$, or $+1$ — and the GPU has no idea what to do with them. It dequantizes those 1.58-bit weights back into 16-bit floats, fills 4.87 GB of memory it didn't need to, and runs its tensor cores at a rounding-error of their potential. The 1.58-bit representation, the whole point of the model, evaporates the moment it hits the hardware.

A 130-dollar FPGA does not throw that away. This post is the story of building one that doesn't — `ternfpga`, a multiply-free ternary LLM-inference engine on a **Xilinx Arty A7-35T**, benchmarked head-to-head against that same RTX 3060 in the same machine. By the end, the FPGA system does the same work at an estimated **1.6 joules per token — roughly 2.3× less energy than the GPU** — by refusing to do the one thing everyone assumes a neural network must do: multiply.

I am going to be exact about the word *estimated*. This is a hardware project, and hardware invites overclaiming the way optimization benchmarks do. So every number below carries a tag: **silicon-measured** (I read it off the board), **derived** (composed from silicon-measured primitives), or **projected** (a forward-looking estimate). The headline energy number is derived; three of its four ingredients are silicon-measured. I'll show you which is which, and there's a full ledger near the end.

## What This Post Covers

- **Why batch-1 LLM decode is a memory problem, not a math problem** — and why that single fact cracks the door open for a board that costs less than a tank of the GPU's electricity.
- **What "ternary" means in actual silicon** — the multiply that collapses into a 6-input lookup table, and the ninety hardware multipliers it leaves completely unused.
- **The sparsity a GPU structurally cannot exploit** — per-token, unstructured, ~60% of an FFN's activations, and the gather engine that skips fetching them.
- **The nine-phase build, on real hardware** — bit-exact at every step, with the three bugs that cost the most time and the one phase where the FPGA was *1.2× worse* than the GPU.
- **The honest ledger** — what I measured on the board, what I derived, and what is still a projection waiting on a bigger board.

This post assumes you know roughly what a neural network and a matrix multiply are. It does **not** assume you know what an FPGA, a DSP slice, or a roofline is — we'll build those up. If you're a hardware person, skim Part I.

A preview of the result, for the impatient:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 680 460" width="680" height="460" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <!-- Title -->
  <text x="340" y="26" text-anchor="middle" font-size="13" font-weight="600" fill="currentColor">Energy per token — BitNet-2B-4T, batch-1 decode</text>
  <text x="340" y="43" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.6">FPGA system energy as transformer "glue" moves onto the fabric, one stage at a time. Lower is better.</text>

  <!-- Y axis -->
  <line x1="90" y1="50" x2="90" y2="380" stroke="currentColor" stroke-width="1.2"/>
  <!-- gridlines + ticks at 0..5 J -->
  <g stroke="currentColor" stroke-width="0.5" opacity="0.18">
    <line x1="90" y1="380" x2="640" y2="380"/>
    <line x1="90" y1="314" x2="640" y2="314"/>
    <line x1="90" y1="248" x2="640" y2="248"/>
    <line x1="90" y1="182" x2="640" y2="182"/>
    <line x1="90" y1="116" x2="640" y2="116"/>
    <line x1="90" y1="50"  x2="640" y2="50"/>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.6">
    <text x="82" y="384" text-anchor="end">0</text>
    <text x="82" y="318" text-anchor="end">1</text>
    <text x="82" y="252" text-anchor="end">2</text>
    <text x="82" y="186" text-anchor="end">3</text>
    <text x="82" y="120" text-anchor="end">4</text>
    <text x="82" y="54"  text-anchor="end">5</text>
  </g>
  <text x="30" y="215" font-size="11" fill="currentColor" opacity="0.75" transform="rotate(-90 30 215)" text-anchor="middle">joules / token</text>

  <!-- GPU reference line at 3.67 -> y = 380 - 3.67*66 = 137.8 -->
  <line x1="90" y1="137.8" x2="640" y2="137.8" stroke="var(--primary, #94452b)" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.9"/>
  <text x="636" y="131" text-anchor="end" font-size="10.5" font-weight="600" fill="var(--primary, #94452b)">RTX 3060 — 3.67 J/tok · the line to beat</text>

  <!-- CPU reference (faint) at 4.62 -> y = 75.1 -->
  <line x1="90" y1="75.1" x2="640" y2="75.1" stroke="currentColor" stroke-width="1" stroke-dasharray="2,3" opacity="0.35"/>
  <text x="636" y="69" text-anchor="end" font-size="9.5" fill="currentColor" opacity="0.5">CPU 5950X — 4.62</text>

  <!-- Bars: centers 170,310,450,590 ; width 70 ; baseline y=380 ; y(v)=380-66v -->
  <!-- Bar 1: host-split 4.32 -> top 94.9 (ABOVE the GPU line: loses) -->
  <rect x="135" y="94.9" width="70" height="285.1" rx="3" fill="var(--error, #a64542)" opacity="0.78"/>
  <text x="170" y="88" text-anchor="middle" font-size="12" font-weight="700" fill="var(--error, #a64542)">4.32</text>
  <text x="170" y="398" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">host-split</text>
  <text x="170" y="411" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.55">(naive)</text>
  <text x="170" y="426" text-anchor="middle" font-size="9.5" font-weight="600" fill="var(--error, #a64542)">1.2× WORSE</text>

  <!-- Bar 2: +attention 1.99 -> top 248.7 -->
  <rect x="275" y="248.7" width="70" height="131.3" rx="3" fill="var(--primary, #94452b)" opacity="0.55"/>
  <text x="310" y="242" text-anchor="middle" font-size="12" font-weight="700" fill="var(--primary, #94452b)">1.99</text>
  <text x="310" y="398" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">+ on-fabric</text>
  <text x="310" y="411" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.55">attention</text>
  <text x="310" y="426" text-anchor="middle" font-size="9.5" font-weight="600" fill="#0a8a3f">1.8× under</text>

  <!-- Bar 3: +ffn glue 1.62 -> top 273.1 -->
  <rect x="415" y="273.1" width="70" height="106.9" rx="3" fill="var(--primary, #94452b)" opacity="0.78"/>
  <text x="450" y="266" text-anchor="middle" font-size="12" font-weight="700" fill="var(--primary, #94452b)">1.62</text>
  <text x="450" y="398" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">+ on-fabric</text>
  <text x="450" y="411" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.55">FFN glue</text>
  <text x="450" y="426" text-anchor="middle" font-size="9.5" font-weight="600" fill="#0a8a3f">2.3× under</text>

  <!-- Bar 4: engine bound 1.47 -> top 283 (dashed = target) -->
  <rect x="555" y="283" width="70" height="97" rx="3" fill="var(--primary, #94452b)" stroke="var(--primary, #94452b)" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.32"/>
  <text x="590" y="276" text-anchor="middle" font-size="12" font-weight="700" fill="var(--primary, #94452b)">1.47</text>
  <text x="590" y="398" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.8">engine</text>
  <text x="590" y="411" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.55">bound</text>
  <text x="590" y="426" text-anchor="middle" font-size="9.5" font-weight="600" fill="var(--primary, #94452b)" opacity="0.7">2.5× (proj.)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The arc of the whole project in one figure. The naive design (left, red) loses to the GPU by 1.2×. Each subsequent bar moves a piece of non-ternary "glue" computation from the slow host CPU onto the FPGA fabric; the system drops below the GPU line and keeps falling toward the engine's own 1.47 J/token floor. The first three FPGA bars are derived from silicon-measured cycle counts; the rightmost is a projection.</p>

The shape of that figure — a bad start, a decisive correction, and a steady climb toward a hard floor — *is* the project. But to see why any of the bars are where they are, we have to start with a fact about how language models actually run.

---

## Part I — Why a 130-Dollar Board Can Win

### Decode is a memory problem, not a math problem

When a language model generates text, it does so one token at a time, and each token requires a full pass over the model's weights. In the **batch-1** case — one user, one stream, the dominant case for local and edge inference — there is no batching to amortize anything. To produce a single token, the hardware reads every weight in the model from memory **exactly once** and does a couple of arithmetic operations with it.

That ratio — arithmetic operations per byte read from memory — is called **arithmetic intensity**, and for batch-1 decode it is brutally low. A weight is fetched, multiplied, accumulated, and discarded. Two floating-point operations for every two bytes of FP16 weight: an intensity around **1 FLOP/byte**. Modern accelerators are built for the opposite regime — training and large-batch serving, where each weight gets reused across hundreds of examples and intensity is in the hundreds.

This is what a **roofline** model makes visible:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 660 380" width="660" height="380" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <marker id="rl-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="currentColor"/>
    </marker>
  </defs>
  <!-- Axes -->
  <line x1="80" y1="330" x2="620" y2="330" stroke="currentColor" stroke-width="1.2" marker-end="url(#rl-arrow)"/>
  <line x1="80" y1="330" x2="80" y2="40" stroke="currentColor" stroke-width="1.2" marker-end="url(#rl-arrow)"/>
  <text x="350" y="365" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75" font-style="italic">arithmetic intensity  (operations per byte) →</text>
  <text x="26" y="185" text-anchor="middle" font-size="11" fill="currentColor" opacity="0.75" font-style="italic" transform="rotate(-90 26 185)">attainable throughput →</text>

  <!-- Memory-bound diagonal roof: from (80,330) to knee (330,100) -->
  <line x1="80" y1="330" x2="330" y2="100" stroke="var(--primary, #94452b)" stroke-width="2.5"/>
  <!-- Compute-bound flat roof: from knee (330,100) to (620,100) -->
  <line x1="330" y1="100" x2="620" y2="100" stroke="var(--primary, #94452b)" stroke-width="2.5"/>
  <text x="150" y="232" font-size="10" fill="var(--primary, #94452b)" font-style="italic" transform="rotate(-43 150 232)">memory-bandwidth roof</text>
  <text x="470" y="92" text-anchor="middle" font-size="10" fill="var(--primary, #94452b)" font-style="italic">compute roof (tensor cores)</text>

  <!-- Shaded memory-bound region -->
  <path d="M 80 330 L 330 100 L 330 330 Z" fill="var(--primary, #94452b)" opacity="0.07"/>
  <line x1="330" y1="100" x2="330" y2="330" stroke="currentColor" stroke-width="0.8" stroke-dasharray="3,3" opacity="0.4"/>
  <text x="205" y="320" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">memory-bound</text>
  <text x="475" y="320" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.45">compute-bound</text>

  <!-- batch-1 decode marker (far left, low on the diagonal) -->
  <circle cx="135" cy="280.6" r="6" fill="var(--primary, #94452b)" stroke="white" stroke-width="2"/>
  <text x="148" y="272" font-size="10.5" font-weight="600" fill="var(--primary, #94452b)">batch-1 decode</text>
  <text x="148" y="286" font-size="9.5" fill="currentColor" opacity="0.7">one token = stream every weight once</text>
  <text x="148" y="299" font-size="9.5" fill="currentColor" opacity="0.7">≈ 1 FLOP / byte</text>

  <!-- training/batched marker near the knee -->
  <circle cx="360" cy="100" r="5" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <text x="372" y="96" font-size="10" fill="currentColor" opacity="0.7">training / large batch</text>
  <text x="372" y="110" font-size="9.5" fill="currentColor" opacity="0.5">reuse weights → compute-bound</text>

  <!-- Annotation: the wasted ceiling -->
  <line x1="135" y1="270" x2="135" y2="112" stroke="currentColor" stroke-width="0.8" stroke-dasharray="2,2" opacity="0.4"/>
  <text x="135" y="62" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.55">all this compute</text>
  <text x="135" y="74" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.55">sits idle ↓</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">A roofline diagram. Performance is capped either by memory bandwidth (the rising diagonal) or by raw compute (the flat ceiling), whichever is lower at a given arithmetic intensity. Batch-1 decode lives far to the left, pinned to the bandwidth roof — the GPU's enormous compute ceiling is irrelevant, because the work is gated entirely by how fast weights arrive from memory.</p>

This is the crack in the GPU's armor. When you are decode-bound, **the expensive part of the GPU — the tensor cores — is idle**, waiting on memory. You are paying for, and powering, silicon you cannot use. What matters is bytes-per-second of weight traffic and the energy spent moving them. And *that* is a contest a small, low-power device can enter — provided it can cut the bytes.

There are two ways to cut the bytes that a GPU structurally cannot follow. The whole project is built on them.

### Escape 1: stop multiplying

The first escape is the model itself. **BitNet b1.58** is a family of large language models whose weights are constrained, during training, to just three values: $-1$, $0$, and $+1$. "b1.58" is the information content of one such weight: $\log_2 3 \approx 1.58$ bits, against the 16 bits of an FP16 weight or the 8 of INT8. The activations stay in normal integer precision; only the weights are ternary. Remarkably, this barely dents quality — the 2-billion-parameter BitNet-2B-4T is competitive with same-size FP16 models — because the network is *trained* in this regime rather than quantized after the fact.

Here is the part that matters for hardware. A matrix-multiply is a sea of multiply-accumulates, $\text{acc} \mathrel{+}= w \cdot a$. When $w \in \lbrace -1, 0, +1 \rbrace$, that multiply is not a multiply at all:

$$
w \cdot a \;=\;
\begin{cases}
+a & w = +1 \\
\phantom{+}0 & w = 0 \\
-a & w = -1
\end{cases}
$$

There is nothing to multiply. You either pass the activation through, zero it, or negate it. In digital logic that is a three-way select — a few gates, a single small lookup table — and it is the entire arithmetic core of the engine:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 600 280" width="600" height="280" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <marker id="ts-arrow" markerWidth="7" markerHeight="5" refX="6.5" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="currentColor"/>
    </marker>
  </defs>

  <!-- activation input -->
  <rect x="30" y="120" width="86" height="40" rx="5" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.5"/>
  <text x="73" y="140" text-anchor="middle" dominant-baseline="central" font-size="12" font-weight="600" fill="var(--primary, #94452b)">activation a</text>
  <text x="73" y="176" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6">int8</text>

  <!-- selector box -->
  <rect x="210" y="60" width="120" height="160" rx="8" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.5"/>
  <text x="270" y="48" text-anchor="middle" font-size="10.5" font-weight="600" fill="currentColor" opacity="0.8">sign / zero select</text>

  <!-- three outputs inside -->
  <text x="270" y="92"  text-anchor="middle" font-size="12" fill="currentColor">w = +1  →  +a</text>
  <text x="270" y="140" text-anchor="middle" font-size="12" fill="currentColor">w =  0  →   0</text>
  <text x="270" y="188" text-anchor="middle" font-size="12" fill="currentColor">w = −1  →  −a</text>

  <!-- a -> selector -->
  <line x1="116" y1="140" x2="210" y2="140" stroke="currentColor" stroke-width="1.4" marker-end="url(#ts-arrow)"/>

  <!-- w control input from below -->
  <rect x="210" y="244" width="120" height="30" rx="5" fill="none" stroke="currentColor" stroke-width="1.2" stroke-dasharray="3,2"/>
  <text x="270" y="259" text-anchor="middle" dominant-baseline="central" font-size="11" fill="currentColor">ternary weight w (2 bits)</text>
  <line x1="270" y1="244" x2="270" y2="220" stroke="currentColor" stroke-width="1.4" marker-end="url(#ts-arrow)"/>

  <!-- selector -> accumulator -->
  <line x1="330" y1="140" x2="424" y2="140" stroke="currentColor" stroke-width="1.4" marker-end="url(#ts-arrow)"/>

  <!-- accumulator -->
  <circle cx="470" cy="140" r="34" fill="var(--surface-container, #f3f0eb)" stroke="var(--primary, #94452b)" stroke-width="1.8"/>
  <text x="470" y="135" text-anchor="middle" font-size="20" font-weight="600" fill="var(--primary, #94452b)">Σ</text>
  <text x="470" y="154" text-anchor="middle" font-size="8.5" fill="currentColor" opacity="0.7">accumulate</text>
  <text x="470" y="192" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6">int32</text>

  <!-- callout -->
  <text x="470" y="244" text-anchor="middle" font-size="11" font-weight="600" fill="#0a8a3f">one 6-LUT · zero DSP</text>
  <text x="470" y="259" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6">no hardware multiplier touched</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The entire "multiply" in a ternary network: a sign-and-zero select, implementable as a single six-input lookup table. The chip's dedicated hardware multipliers are never touched. A GPU, by contrast, has no ternary datapath — it must expand each weight back into an INT8 or FP16 value and run a real multiply through its arithmetic units.</p>

The bytes-saved story compounds with a packing trick. Three ternary values have $3^3 = 27$ combinations; five have $3^5 = 243$, which still fits in a single byte ($243 < 256$). So you can store **five ternary weights per byte — 1.6 bits each**, within a whisker of the 1.585-bit theoretical optimum, and 5× denser than INT8. Since decode is bandwidth-bound, that packing is a direct, multiplicative cut to the traffic that bottlenecks everything. A small combinational decoder turns one byte back into five weight codes on the fly, feeding the sign-select lanes straight from a memory burst.

The GPU cannot follow here. Its tensor cores have no $\lbrace -1,0,+1 \rbrace$ mode; the most efficient thing it can do is *dequantize* the ternary weights up to INT8 or FP16 and run ordinary multiplies. It pays the full memory traffic of the wide format and the full energy of real multipliers — for weights that carry 1.58 bits of information. That is exactly why, in the benchmark that opens this post, the 3060 barely beats a CPU: it is doing the expensive version of an inexpensive problem.

### Escape 2: stop fetching zeros

The second escape is **activation sparsity**. BitNet's feed-forward blocks use a squared-ReLU nonlinearity, which forces a large fraction of intermediate activations to exactly zero on every token. When an activation is zero, the entire column of weights it would have multiplied **never needs to be fetched** — those bytes contribute nothing to the result.

I measured this on BitNet-2B-4T directly, hooking all thirty `down_proj` layers over diverse text: **59.8% of activations are zero per token** (ranging 42–79% by depth). That is below the 85–95% that relu-fied models like ProSparse reach — I'll return to that honesty point later — but it is real, and it is a ~2.5× reduction in `down_proj` weight traffic that comes for free with the model.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 640 300" width="640" height="300" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <marker id="sp-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="var(--primary, #94452b)"/>
    </marker>
  </defs>

  <!-- left: dense activation vector with zeros -->
  <text x="120" y="34" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor" opacity="0.8">activation vector h</text>
  <text x="120" y="48" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.55">~60% exactly zero (per token)</text>
  <!-- 10 cells, ~6 zero -->
  <g font-size="11" text-anchor="middle">
    <rect x="92" y="60"  width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="120" y="75" fill="var(--primary, #94452b)" font-weight="600">3</text>
    <rect x="92" y="84"  width="56" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.35"/><text x="120" y="99" fill="currentColor" opacity="0.4">0</text>
    <rect x="92" y="108" width="56" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.35"/><text x="120" y="123" fill="currentColor" opacity="0.4">0</text>
    <rect x="92" y="132" width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="120" y="147" fill="var(--primary, #94452b)" font-weight="600">−2</text>
    <rect x="92" y="156" width="56" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.35"/><text x="120" y="171" fill="currentColor" opacity="0.4">0</text>
    <rect x="92" y="180" width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="120" y="195" fill="var(--primary, #94452b)" font-weight="600">5</text>
    <rect x="92" y="204" width="56" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.35"/><text x="120" y="219" fill="currentColor" opacity="0.4">0</text>
    <rect x="92" y="228" width="56" height="22" rx="3" fill="none" stroke="currentColor" stroke-width="0.8" opacity="0.35"/><text x="120" y="243" fill="currentColor" opacity="0.4">0</text>
  </g>

  <!-- arrow: gather -->
  <line x1="160" y1="155" x2="270" y2="155" stroke="var(--primary, #94452b)" stroke-width="1.6" marker-end="url(#sp-arrow)"/>
  <text x="215" y="146" text-anchor="middle" font-size="9.5" fill="var(--primary, #94452b)" font-weight="600">gather</text>
  <text x="215" y="170" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">nonzeros only</text>

  <!-- middle: compacted -->
  <text x="320" y="48" text-anchor="middle" font-size="10.5" fill="currentColor" opacity="0.7">compacted</text>
  <g font-size="11" text-anchor="middle">
    <rect x="292" y="108" width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="320" y="123" fill="var(--primary, #94452b)" font-weight="600">3</text>
    <rect x="292" y="132" width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="320" y="147" fill="var(--primary, #94452b)" font-weight="600">−2</text>
    <rect x="292" y="156" width="56" height="22" rx="3" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.2"/><text x="320" y="171" fill="var(--primary, #94452b)" font-weight="600">5</text>
  </g>

  <!-- arrow: fetch matching columns -->
  <line x1="360" y1="140" x2="455" y2="140" stroke="var(--primary, #94452b)" stroke-width="1.6" marker-end="url(#sp-arrow)"/>
  <text x="407" y="131" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">fetch only</text>
  <text x="407" y="156" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">these columns</text>

  <!-- right: weight matrix with only 3 columns fetched -->
  <text x="540" y="34" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor" opacity="0.8">weight columns in DRAM</text>
  <g>
    <!-- 8 columns, 3 highlighted -->
    <rect x="470" y="60" width="16" height="190" rx="2" fill="var(--primary, #94452b)" opacity="0.8"/>
    <rect x="490" y="60" width="16" height="190" rx="2" fill="none" stroke="currentColor" stroke-width="0.7" opacity="0.25"/>
    <rect x="510" y="60" width="16" height="190" rx="2" fill="none" stroke="currentColor" stroke-width="0.7" opacity="0.25"/>
    <rect x="530" y="60" width="16" height="190" rx="2" fill="var(--primary, #94452b)" opacity="0.8"/>
    <rect x="550" y="60" width="16" height="190" rx="2" fill="none" stroke="currentColor" stroke-width="0.7" opacity="0.25"/>
    <rect x="570" y="60" width="16" height="190" rx="2" fill="var(--primary, #94452b)" opacity="0.8"/>
    <rect x="590" y="60" width="16" height="190" rx="2" fill="none" stroke="currentColor" stroke-width="0.7" opacity="0.25"/>
    <rect x="610" y="60" width="16" height="190" rx="2" fill="none" stroke="currentColor" stroke-width="0.7" opacity="0.25"/>
  </g>
  <text x="540" y="272" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6">solid = fetched · faint = skipped entirely</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Activation-sparse gather. The zeros in the activation vector mean the corresponding weight columns are never read from memory. At the measured 60% sparsity, <code>down_proj</code> fetches ~44% of the dense bytes — bit-exact, because skipping a column that gets multiplied by zero changes nothing.</p>

Once again, the GPU cannot follow. Its hardware accelerates only **2:4 structured** sparsity — a rigid pattern where exactly two of every four weights are zero, fixed at model-compression time. What BitNet produces is **unstructured and per-token**: a different ~60% of activations are zero on every single token, in no fixed pattern. A GPU faced with this either ignores it (and fetches everything) or pays so much overhead chasing irregular indices that it loses. An FPGA, whose datapath you design yourself, can build a gather that issues a memory read *only* for the nonzero columns — and skips the rest at full speed.

### The FPGA's unfair advantage: zero DSPs, sub-watt

So far this is an argument about *bytes*. The reason it converts into an *energy* win is the FPGA itself.

A field-programmable gate array is a sheet of reconfigurable logic: hundreds of thousands of lookup tables (LUTs) and flip-flops you wire into whatever digital circuit you want, plus a few hundred dedicated hardware multipliers called **DSP slices**. On the Arty A7-35T — the cheapest useful Artix-7 board, around 130 dollars — there are about 20,800 LUTs, 41,600 flip-flops, 90 DSP slices, and 50 small blocks of on-chip memory (BRAM).

A conventional matrix engine spends those 90 DSP slices doing multiplies, and 90 multiplies is not very many — it is the heart of why a small FPGA loses on raw throughput. But a *ternary* engine does not multiply. The sign-select core synthesizes entirely into LUTs and the chip's carry chains; I confirmed in synthesis that it uses **zero DSP slices** all the way up to a 2048-wide datapath. All 90 multipliers sit unused. The engine is not competing for the scarce resource; it sidesteps it.

And it does so at a power level a GPU cannot approach. The whole system-on-chip — the ternary engine, a RISC-V CPU, and the DDR3 memory controller — draws an estimated **0.489 W**. The ternary engine alone is about **0.06 W**. The RTX 3060, doing the same decode, measured **86.4 W**. That is a ~175× power gap at the system level, and it is the entire denominator of "energy per token." Even running far slower, a device that sips power while cutting memory traffic can spend less *energy* getting to the same token.

Let me be honest about the other side of the ledger, because it matters: **the FPGA loses on raw throughput, by design and by a lot.** With ~280× less memory bandwidth than the 3060, it will generate tokens more slowly. This project never claims a "40× faster" headline — that would be dishonest. The claim is narrower and, I think, more interesting: on **energy per token**, **batch-1 latency**, and a **capability the GPU lacks** (native ternary, per-token unstructured sparsity), a 130-dollar board can beat a 400-dollar GPU. Those are the axes that matter at the edge, where there is no datacenter to batch your requests and the power budget is a wall, not a line item.

That is the case on paper. The rest of this post is what happened when I tried to build it.

---

## Part II — Building It, One Honest Measurement at a Time

I built `ternfpga` as a strict test-driven loop, because hardware bugs are expensive and silicon bugs are *very* expensive. Every module got a NumPy "golden" reference and a cocotb testbench that checked the RTL **bit-exact against it before anything was synthesized**, let alone flashed. The development cycle ran across two machines: I authored on a laptop, an `rsync` pushed the tree to a Linux box with the FPGA physically attached, and that box ran Verilator + cocotb for simulation, Vivado for synthesis, and `openFPGALoader` to flash the board. Results — including the board's own UART output — streamed back. And I kept an append-only build log, dated, including the dead-ends. Most of the good anecdotes below are lifted straight from it.

What follows is nine phases. They are not all glamorous. One of them is the FPGA *losing*.

### Phase 0 — the multiply-free core, from a unit test to silicon

The first thing to exist was the NumPy golden for a ternary dot product, then its cocotb test, then the SystemVerilog. The RTL is the sign-select from Part I, eight lanes wide, summed in an adder tree. **6 directed edge cases plus 2,000 randomized dot products, bit-exact, zero mismatches** — before a single gate was synthesized.

Then synthesis, on the real part, to check the central claim:

| module | LUTs | FFs | **DSP48** | Fmax (synth est.) |
|---|---:|---:|---:|---:|
| `ternary_dot` | 233 (1.1%) | 0 | **0** | combinational |
| `ternary_gemv` | 384 (1.9%) | 582 | **0** | ~104 MHz |
| `ternary_gemv_sparse` | 521 (2.5%) | 664 | **0** | ~116 MHz |

**Zero DSP slices on every module.** Vivado confirmed the ternary "multiply" is pure lookup-table sign-select plus carry-chain adders — all 90 hardware multipliers free. A later three-stage pipeline lifted the dot from ~104 MHz to **~280 MHz** while *shrinking* to 149 LUTs (registers break the long adder tree into cheaper pieces), still 0 DSP. The full memory-to-compute datapath looks like this:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 720 270" width="720" height="270" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <defs>
    <marker id="dp-arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="currentColor"/>
    </marker>
  </defs>

  <!-- activation BRAM (top, feeds the dot) -->
  <rect x="305" y="22" width="140" height="42" rx="6" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.4"/>
  <text x="375" y="40" text-anchor="middle" font-size="11" font-weight="600" fill="var(--primary, #94452b)">activation x</text>
  <text x="375" y="55" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.65">in on-chip BRAM</text>
  <line x1="375" y1="64" x2="375" y2="96" stroke="currentColor" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- main row, y=100..164 -->
  <!-- DDR3 -->
  <rect x="14" y="100" width="116" height="64" rx="6" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
  <text x="72" y="124" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">DDR3</text>
  <text x="72" y="140" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">weights, base-3</text>
  <text x="72" y="152" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">5 trits / byte</text>
  <line x1="130" y1="132" x2="160" y2="132" stroke="currentColor" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- unpack -->
  <rect x="162" y="100" width="116" height="64" rx="6" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
  <text x="220" y="124" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">unpack</text>
  <text x="220" y="140" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">1 byte →</text>
  <text x="220" y="152" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">5 weight codes</text>
  <line x1="278" y1="132" x2="308" y2="132" stroke="currentColor" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- ternary dot (hero) -->
  <rect x="310" y="100" width="140" height="64" rx="6" fill="var(--surface-container, #f3f0eb)" stroke="var(--primary, #94452b)" stroke-width="2"/>
  <text x="380" y="122" text-anchor="middle" font-size="11" font-weight="600" fill="var(--primary, #94452b)">sign-select dot</text>
  <text x="380" y="138" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.7">K lanes + adder tree</text>
  <text x="380" y="151" text-anchor="middle" font-size="9" font-weight="600" fill="#0a8a3f">0 DSP</text>
  <line x1="450" y1="132" x2="480" y2="132" stroke="currentColor" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- accumulate -->
  <rect x="482" y="100" width="104" height="64" rx="6" fill="var(--surface-container, #f3f0eb)" stroke="currentColor" stroke-width="1.2"/>
  <text x="534" y="124" text-anchor="middle" font-size="11" font-weight="600" fill="currentColor">accumulate</text>
  <text x="534" y="140" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">over NT tiles</text>
  <text x="534" y="152" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">int32</text>
  <line x1="586" y1="132" x2="616" y2="132" stroke="currentColor" stroke-width="1.4" marker-end="url(#dp-arrow)"/>

  <!-- y out -->
  <rect x="618" y="100" width="86" height="64" rx="6" fill="var(--primary-container, #fceee9)" stroke="var(--primary, #94452b)" stroke-width="1.4"/>
  <text x="661" y="128" text-anchor="middle" font-size="11" font-weight="600" fill="var(--primary, #94452b)">y = W·x</text>
  <text x="661" y="144" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.65">BRAM</text>

  <!-- measured callout -->
  <text x="360" y="200" text-anchor="middle" font-size="11" font-weight="600" fill="var(--primary, #94452b)">1.00 cycle / tile — measured on silicon</text>
  <text x="360" y="216" text-anchor="middle" font-size="9.5" fill="currentColor" opacity="0.6">8 ternary MACs/cycle = 800 M MAC/s at 100 MHz, sustained, bit-exact</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The engine datapath. Weights stream from DRAM in the dense base-3 format, a combinational decoder expands each byte into five weight codes, the sign-select lanes consume them against the BRAM-resident activation, and partial sums accumulate across tiles into the output. Nowhere in this path is there a hardware multiplier.</p>

The milestone of Phase 0 was getting that datapath onto the physical board. I wrote a tiny top-level design — the ternary dot of a running counter against a fixed weight vector, streamed out over the USB serial port — synthesized it to a bitstream, and flashed the Arty. The board was supposed to print `y = 2·counter` on every line. It printed `y = 0`.

The counter was incrementing correctly, so the serial path was fine; the bug was in the logic. I wrote a full integration simulation that **reproduced** the `y = 0`, then probed the internals: the dot's inputs were arriving correctly, but its output was zero. The root cause was embarrassing and instructive. My "weight constant" `0xAA55` packed to four $+1$s and four $-1$s — which sum to **zero**, not the $+2$ I intended. The correct constant was `0xA955`. Three separate unit tests had passed; all three used the *real* test vectors and never exercised that specific hard-coded demo constant. **The integration test earned its keep.** One character fixed, rebuilt, reflashed:

```
16/16 UART lines: y == 2·counter
```

The multiply-free engine, computing correctly in fabric: **105 LUTs, 0 DSP, 100 MHz met, ~63 mW on-chip** by Vivado's power estimate. That 63 mW — more than a thousandfold below the GPU's draw — is the number the whole energy argument rests on.

### The baseline triad — and a very awkward number for the GPU

You cannot claim an energy win without measuring what you're beating. So I stood up two baselines on the same machine that hosts the FPGA. The CPU baseline used `bitnet.cpp`, the official ternary inference runtime, with energy read from the processor's own RAPL counters. The GPU baseline took a maintenance window — I did a live driver swap from `nouveau` to NVIDIA without a reboot (a second display GPU kept the console alive, so the only network path to the box never dropped), then measured decode throughput and `nvidia-smi` power.

| platform | path | tok/s | power | J / token |
|---|---|---:|---:|---:|
| CPU 5950X | native ternary (`i2_s`) | 28.4 | ~121 W | 4.62 |
| **RTX 3060** | **bf16 (dequantized)** | 23.5 | 86.4 W | **3.67** |
| FPGA Arty | ternary, 0 DSP | *(building)* | ~0.06–0.5 W | *(the rest of this post)* |

There is the awkward number. The RTX 3060, faced with BitNet, **has no ternary datapath**, so it dequantizes the weights to bf16 — inflating a model that should occupy a few hundred megabytes into 4.87 GB — and runs ordinary tensor-core multiplies. The result: **3.67 J/token, barely better than the CPU's native-ternary 4.62, and actually slower**. A $400 GPU, extracting almost no value from the 1.58-bit weights it was handed. That gap is the entire opportunity, quantified. The FPGA's job is to *not* throw the ternary structure away.

### Phase 1 — DDR3 and a RISC-V CPU, on the board

To run anything model-sized, the engine needs to stream weights from the board's DRAM, and it needs a host to sequence the layers. I brought up a LiteX system-on-chip on the Arty: a VexRiscv RISC-V CPU, the LiteDRAM controller driving the board's 256 MB DDR3, and the ternary engine wired in as a memory-mapped peripheral. The hardest part of any FPGA project like this is DRAM calibration, and it came up green — `Memtest OK`, read leveling calibrated. Then the firmware drove the engine and checked it:

```
=== ternfpga on-board streaming GEMV (K=8, M=16) ===
GEMV_ONBOARD_PASS  (16 rows bit-exact vs golden)
```

The full chain — CPU writes the activation, streams packed weight bytes, the engine unpacks and does the multiply-free dot, the CPU reads the result — **bit-exact, on silicon, in a real SoC.** The two scariest integration risks (DRAM calibration and the CPU↔engine interface) were now retired.

### Phase 2 — the pivot, where honesty changed the plan

This is the phase where the project nearly got more ambitious and instead got more honest. Before committing many sessions to "scale the engine to a full model," I ran a structured literature review. It returned a verdict that was partly humbling and entirely useful:

- **The 0-DSP LUT approach is the field standard.** The best published ternary FPGA engines all store the $\lbrace -1,0,+1 \rbrace$ product in lookup tables — a DSP slice's 25×18 multiplier is wasted on a pass/negate/zero. I had independently built the right datapath.
- **A full BitNet 0.73B does not fit a 35T.** The closest state-of-the-art result — a full ternary BitNet on FPGA — runs on a ~$300 Zynq board, and its ternary core *alone* is bigger than my entire LUT budget. The host-split it uses (CPU for the non-ternary glue, FPGA for the matmuls) validated my VexRiscv design.

So I re-scoped, explicitly: from "a full model on the board" down to **one real-width transformer block**, streamed from DRAM, with the non-ternary glue (normalization, attention softmax, the LM head) running on the host CPU, and the headline being **energy per token versus the RTX 3060 on identical numerics**. Nobody had built an LLM datapath on an Artix-7-class board; that white space was the point.

Two de-risking measurements followed, both before writing the block. The first was a place-and-route fit sweep — how wide can the datapath get on a 35T before it stops fitting?

| datapath width | LUT | % LUT | FF | % FF | DSP |
|--:|--:|--:|--:|--:|--:|
| 32 | 565 | 2.7% | 865 | 2.1% | **0** |
| 1024 | 10,234 | 49% | 24,675 | 59% | **0** |
| 2048 | 11,013 | 53% | 32,961 | **79%** | **0** |

**Zero DSP holds all the way to width 2048** — the multiply-free property, proven at real scale. But notice the flip-flops: 79% at width 2048. The wall isn't compute; it's **keeping operands in registers**. The lesson, which shaped the entire microarchitecture, was: the scalable engine must be **BRAM-centric** — operands live in block RAM and stream through sequentially — not a giant flat array of registers. (Moving operands from flip-flops to BRAM later collapsed the flip-flop usage ~90× *and* met timing where the register-resident version had failed by 5.9 ns. I'll spare you the third recurrence of the underlying bug until it bites again below.)

The second measurement was sparsity. Direction "skip the zeros" needs zeros to skip, and I could find no published figure for BitNet b1.58's FFN sparsity — so I measured it: **59.8% of activations zero per token**, averaged over diverse text across all thirty layers. That is real and GPU-unmatchable, but it is *below* the 85–95% that relu-fied models reach. I corrected the project's own README to match the measurement rather than the hope. Honest beats optimistic; this theme recurs.

Phase 2 also turned up a small piece of mathematics I find genuinely lovely. The feed-forward block computes, per channel, $\text{relu}(\text{gate})^2 \cdot \text{up}$, normalizes it, and feeds the result — requantized to int8 — into the final projection. That requantization looks like it needs floating-point: dequantize the integer matmul outputs by their per-token scales, apply the RMSNorm divide, then re-quantize. But the int8 value that actually reaches the next matmul is

$$
h_{q,i} \;=\; \text{round}\!\left(\frac{127 \, N_i}{\max_j |N_j|}\right), \qquad N_i = \text{relu}(g_i)^2 \cdot u_i \cdot w_i
$$

where $g_i$ and $u_i$ are the *integer* gate and up outputs and $w_i$ is a fixed-point norm weight. The requantization is a **ratio** — $N_i$ over the maximum $|N|$ — and every per-token dequant scale and the entire RMSNorm normalizer are common positive factors that appear in both the numerator and the denominator. **They cancel, exactly.** The "hard" floating-point glue between the matmuls is, on-chip, pure integer arithmetic plus a single reciprocal. I verified this against the validated reference: with floating-point norm weights it is a **100.00% exact match**; with 16-bit fixed-point weights, 99.99% (off by at most 1). That identity is what later makes an on-fabric glue unit clean instead of nightmarish.

### Phase 3 — the projection, and two risks named out loud

With the block datapath validated against PyTorch (the full decoder layer reproduced the real model at cosine similarity 1.000000), I could compose a full-model energy estimate from silicon-measured primitives — the engine's 1.00 cycle/tile, the measured DRAM bandwidth, the measured power. It came out to roughly **1.47 J/token of engine compute, about 2.5× under the GPU**. But a projection is only as honest as the risks it confronts, and the review had named two.

**Risk 1: maybe single-channel DDR3 is too slow to matter.** I measured the actual read roofline with a hardware DMA engine and a cycle counter: **1,423 MB/s sustained, 89% of the memory port's theoretical peak.** That caps a 0.7B model at about 8 tokens/second — slow, as promised, but the *energy* floor is ~60 mJ/token, still tens of times under the GPU. The engine itself only demands 200 MB/s, so it is compute-bound, not bandwidth-starved; the path to using the full channel is a wider datapath, not faster memory. **Risk 1 survives.**

**Risk 2: maybe the sparsity is fake** — if those 60% zeros fell in a fixed or regular pattern, a GPU's structured-sparsity hardware could capture them and the differentiator would evaporate. So I measured the *structure*: **93.9% of channels are data-dependent** (only ~4% are statically zero), the active set changes so much token-to-token that two tokens share less than half their nonzeros, and a static structured mask captures only 69% of the zeros. **The sparsity is genuinely unstructured** — exactly the kind a GPU cannot exploit and an FPGA gather can. **Risk 2 holds.**

### Phase 4 — the fully-measured verdict: the FPGA loses

Here is the phase I most want to keep in the post, because it is the one that almost every writeup would quietly delete.

Up to now, the engine was silicon-measured but the *system* energy was a projection. To make it real, I rewrote all the transformer "glue" — the normalization, the rotary position embedding, the attention scores and softmax — as pure integer code (lookup tables for the transcendentals, the cancellation identity for the rest) and **measured it running on the board**:

```
norm = 0.54M   rope = 0.08M   attention = 16.2M   ffn-glue = 2.58M
GLUE_INT_PER_LAYER = 19.42 M cycles
```

And then the arithmetic stopped being kind. The engine is 8.68M cycles per layer; the glue is **19.42M** — more than twice the engine. A full token is $30 \times 28.1\text{M} + \text{LM head} = 884\text{M}$ cycles, which at 100 MHz and 0.489 W is **4.32 J/token**. The engine alone would be 2.5× under the GPU — but the naive host-split system was **1.2× *worse* than the GPU it was supposed to beat.**

That is a real result, measured on real silicon, and it said the design was wrong. The diagnosis was unambiguous: **83% of the glue was attention** — the scores, the softmax, the weighted sum over the value cache — running on a cacheless soft RISC-V core, bottlenecked on DRAM latency. The CPU was a terrible place to do attention.

(A debugging confession from this phase, because it's the kind of thing that eats a day: an earlier version of the glue firmware halted after printing about six characters over serial. I blamed the soft-float library, the timer, the heavy math. It was none of those. LiteX's serial output is interrupt-driven, and I had omitted the one line that enables interrupts. The transmit buffer filled and stalled. The right move would have been to check the I/O path *before* the soft-float rabbit hole. Lesson re-learned: suspect the boring thing first.)

The losing number was the most valuable measurement in the project. It converted an architectural opinion — "attention should be on the fabric" — into a quantified necessity, and it set the agenda for everything that followed. The next three phases are the climb back up the first figure in this post.

### Phases 5 & 6 — attention onto the fabric, and the verdict flips

If the CPU is a bad place to do attention, the fix is to build attention in hardware. The `attention_unit` keeps the key and value cache in on-chip BRAM and, for one query, computes the scores (an int16 multiply-accumulate against each cached key), turns them into a softmax, and produces the weighted sum over the values. The softmax is the interesting part, because a naive softmax wants floating-point exponentials and a division. I avoided both:

- The exponential becomes a **lookup table**. The score is reduced by a programmable right-shift (the softmax temperature, chosen as a power of two so it's a shift, not a multiply) and used to index a Q15 fixed-point `exp` table. No multiplier, no `expf`.
- The division is **deferred**. Instead of computing $\text{softmax}(s)_i = e^{s_i} / \sum_j e^{s_j}$ and then the weighted sum, the unit outputs the unnormalized weighted sum $\text{num}_d = \sum_i e^{s_i} v_{i,d}$ and the normalizer $\sum_i e^{s_i}$ separately. The single divide happens once, downstream — the on-fabric unit stays divider-free.

Bit-exact against its integer oracle, the unit runs at about **one multiply-accumulate per cycle**, which makes attention roughly **98× faster** than the host version. (Two bugs en route, both worth the warning: the `exp` table's first entry was 32768, one past the signed-16-bit maximum, and silently wrapped negative — clipped to 32767. And the key/value memories, when their *read* port lived in a block with an asynchronous reset, synthesized as a vast array of flip-flops with an address decoder instead of as BRAM, blowing LUT usage to 90%. Moving the read into a clock-only block with an explicit `ram_style="block"` hint fixed it. **Hold that thought** — it happens again.)

What does that do to the first figure in this post? Replace the 16.2M-cycle host attention with the on-fabric ~0.33M, and the glue per layer collapses from 19.4M to 3.5M cycles. The layer drops from 28.1M to 12.2M, the token from 884M to ~407M, and the energy from 4.32 J to **1.99 J/token**. The system goes from **1.2× worse than the GPU to roughly 1.8× better.** The engine is now 71% of the layer's work — the 0-DSP ternary advantage finally shows up at the *system* level, not just the kernel level. That is the single most important transition in the project, and it is the second bar in the opening figure.

Phase 6 put it on silicon. Wrapped as a peripheral, built, flashed, and run on the physical board:

```
ATTN_ONBOARD_PASS  (128 num + sum_e bit-exact)
MEASURED attention cycles/query = 16456  (T=64, D=128)
```

**Bit-exact on real hardware**, at one MAC per cycle confirmed — a ~49× collapse versus host attention at this cache depth. Attention now had the full **PyTorch → simulation → silicon** chain, so the flipped verdict rested on a measured term, not a synthesized estimate. (Timing honesty, since this matters: the static analyzer reported the design missing 100 MHz by 1.27 ns at the worst-case corner. It ran bit-exact at 100 MHz anyway, because the worst-case model is pessimistic and the room is air-conditioned — but the honest fix, a pipeline register on the critical path, is noted as owed. I am not going to pretend a negative slack number is a positive one.)

### Phases 7 & 8 — the last big glue term, and the bug that came back a third time

With attention handled, the largest remaining host term was the FFN inter-projection glue — the $\text{relu}(\text{gate})^2 \cdot \text{up} \cdot w$ and int8 requantization from Phase 2, 2.58M cycles a layer on the host. The `ffn_glue_unit` does it on-fabric in two passes: compute every $N_i$ and track the running maximum $|N|$, then requantize. The requantization needs a divide by $\max|N|$ per channel, which I did *not* want to instantiate 6,912 times. The trick: compute **one** reciprocal per call — $\text{recip} = (127 \ll R) / \max|N|$ with $R$ chosen so the reciprocal lands in a fixed 32-bit window — using a single restoring divider, then each channel's requant is a multiply and a shift, $h_{q,i} = (|N_i| \cdot \text{recip} \gg R) \cdot \text{sign}$. One divide, reused across all channels.

Bit-exact against the oracle, ~165× faster than the host glue. And then synthesis came back at **115% of the LUTs and 134% of the flip-flops** — it didn't fit at all. The cause was the bug from Phase 2 and Phase 5, for the third time: the output memory `hqmem` had its *write* in one always-block and its *read* in another, so the tool couldn't infer a block RAM and instead built a 6,912-deep array of flip-flops plus an address decoder. Putting the write and read in the **same clock-only block** turned it into a proper simple-dual-port BRAM, and usage dropped to **7% LUT / 1% FF / 40% BRAM / 19 DSP**. Three times this exact mistake cost me a synthesis run; it is now the first thing I check when utilization looks insane. (The lesson, stated generally: *a memory whose read and write live in different procedural blocks will not infer as BRAM.* Tattoo it somewhere.)

The unit's single-cycle compute path was far too long for 100 MHz — a negative slack of 42.7 ns. Pipelining it was mechanical once I committed to one operation per stage: eight stages, with a valid bit and the channel index and operands all travelling together, closed it in three principled steps (−42.7 → −8.9 once pipelined → −3.2 once I registered the requant scale so a priority encoder left the per-cycle path → −1.9 ns once I split a 122-bit add from a barrel shift). Same cycle count, same bit-exact result. Then, on silicon:

```
FFNGLUE_ONBOARD_PASS  (6912 h_q + max|N| bit-exact)
MEASURED ffn-glue cycles/layer = 13974  (F=6912)  →  184× vs host
```

**Bit-exact on the board at the real FFN width**, a 184× collapse. (The measured 13,974 is *more* trustworthy than the simulation's earlier 15.6K extrapolation, which had double-counted a per-pass drain — measuring the real thing beat extrapolating from a small one.) The system energy fell to **1.62 J/token, ~2.3× under the GPU** — the third bar in the opening figure. With both big terms on the fabric, the engine is now **90% of the layer**, and the largest thing left is the pair of RMSNorm operations at 0.54M cycles — the gap between the 1.62 bar and the 1.47 engine floor.

The bookkeeping milestone of Phase 8: **three of the system's four cycle terms — the engine, attention, and the FFN glue — are now silicon-measured.** Only the RMSNorm and the fully-integrated loop remain projections.

### Phase 9 — two accelerators on one board, cooperating

Every silicon result so far measured *one* unit in isolation. The capstone was to put two of them in the same chip and make them hand data to each other: the ternary engine computes the gate and up projections, its integer outputs feed the FFN-glue unit, and the requantized result comes back — the actual data flow of an FFN, end to end, on real hardware.

```
engine gate/up GEMV: ok (0 row mismatches)
COMBINED_ONBOARD_PASS  (engine -> ffn_glue, 32 h_q bit-exact, ffn-glue 214 cyc)
```

**Bit-exact, end to end, on the board** — the first multi-accelerator computation in the project. It also pinned the honest frontier, which is the most important thing this phase has to say:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 560 380" width="560" height="380" xmlns="http://www.w3.org/2000/svg" style="display: block; margin: 0 auto; font-family: 'Inter', sans-serif; max-width: 100%;">
  <!-- Title -->
  <text x="280" y="24" text-anchor="middle" font-size="12.5" font-weight="600" fill="currentColor">On-chip memory (BRAM) is the wall</text>
  <text x="280" y="41" text-anchor="middle" font-size="10" fill="currentColor" opacity="0.6">Arty A7-35T has 50 block-RAM tiles. A pair of accelerators fits; all three don't.</text>

  <!-- Y axis: 0..70 tiles, baseline y=330, top y=55. scale 4px/tile. y(t)=330-4t -->
  <line x1="90" y1="55" x2="90" y2="330" stroke="currentColor" stroke-width="1.1"/>
  <g font-size="9.5" fill="currentColor" opacity="0.6">
    <text x="83" y="334" text-anchor="end">0</text>
    <text x="83" y="294" text-anchor="end">10</text>
    <text x="83" y="254" text-anchor="end">20</text>
    <text x="83" y="214" text-anchor="end">30</text>
    <text x="83" y="174" text-anchor="end">40</text>
    <text x="83" y="134" text-anchor="end">50</text>
    <text x="83" y="94"  text-anchor="end">60</text>
  </g>
  <text x="34" y="195" font-size="10.5" fill="currentColor" opacity="0.7" transform="rotate(-90 34 195)" text-anchor="middle">BRAM tiles used</text>

  <!-- 50-tile budget line: y = 330 - 200 = 130 -->
  <line x1="90" y1="130" x2="520" y2="130" stroke="var(--error, #a64542)" stroke-width="1.6" stroke-dasharray="7,4"/>
  <text x="516" y="123" text-anchor="end" font-size="10.5" font-weight="600" fill="var(--error, #a64542)">50-tile budget</text>

  <!-- Bar 1: the proven pair (45). center x=200, width 110. SoC+engine 27, ffn-glue 18 -->
  <!-- SoC+engine 0..27: y 330..222, h=108 -->
  <rect x="145" y="222" width="110" height="108" fill="var(--primary, #94452b)" opacity="0.45" stroke="var(--primary,#94452b)" stroke-width="0.8"/>
  <text x="200" y="280" text-anchor="middle" font-size="9.5" fill="currentColor">SoC + engine</text>
  <text x="200" y="293" text-anchor="middle" font-size="9" fill="currentColor" opacity="0.6">~27</text>
  <!-- ffn-glue 27..45: y 222..150, h=72 -->
  <rect x="145" y="150" width="110" height="72" fill="var(--primary, #94452b)" opacity="0.78" stroke="var(--primary,#94452b)" stroke-width="0.8"/>
  <text x="200" y="190" text-anchor="middle" font-size="9.5" fill="white">FFN-glue</text>
  <text x="200" y="203" text-anchor="middle" font-size="9" fill="white" opacity="0.85">~18</text>
  <text x="200" y="142" text-anchor="middle" font-size="11" font-weight="700" fill="#0a8a3f">45 ✓</text>
  <text x="200" y="350" text-anchor="middle" font-size="10" font-weight="600" fill="currentColor">engine + FFN-glue</text>
  <text x="200" y="364" text-anchor="middle" font-size="9" fill="#0a8a3f">built, bit-exact on silicon</text>

  <!-- Bar 2: the trio (63). center x=400, width 110 -->
  <rect x="345" y="222" width="110" height="108" fill="var(--primary, #94452b)" opacity="0.45" stroke="var(--primary,#94452b)" stroke-width="0.8"/>
  <text x="400" y="282" text-anchor="middle" font-size="9.5" fill="currentColor">SoC + engine</text>
  <rect x="345" y="150" width="110" height="72" fill="var(--primary, #94452b)" opacity="0.78" stroke="var(--primary,#94452b)" stroke-width="0.8"/>
  <text x="400" y="190" text-anchor="middle" font-size="9.5" fill="white">FFN-glue</text>
  <!-- attention 45..63: y 150..78, h=72 -> pokes above 50-line (y=130) -->
  <rect x="345" y="78" width="110" height="72" fill="var(--error, #a64542)" opacity="0.8" stroke="var(--error,#a64542)" stroke-width="0.8"/>
  <text x="400" y="114" text-anchor="middle" font-size="9.5" fill="white">attention</text>
  <text x="400" y="127" text-anchor="middle" font-size="9" fill="white" opacity="0.9">~18</text>
  <text x="400" y="70" text-anchor="middle" font-size="11" font-weight="700" fill="var(--error, #a64542)">63 ✗</text>
  <text x="400" y="350" text-anchor="middle" font-size="10" font-weight="600" fill="currentColor">all three</text>
  <text x="400" y="364" text-anchor="middle" font-size="9" fill="var(--error, #a64542)">over budget — needs tiling / bigger board</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The frontier, measured. The ternary engine and the full-width FFN-glue unit, plus the supporting CPU/DRAM system, fit a 35T at 45 of its 50 block-RAM tiles — built and verified. Adding the attention unit's ~18 tiles would need 63, over the budget. The full three-accelerator decode loop wants either FFN tiling or a board with more on-chip memory (a 250-dollar A7-100T, or a Zynq KV260).</p>

The pair fits at **45 of 50 BRAM tiles**; adding attention's ~18 would need 63. So the *fully integrated* three-accelerator decode loop does **not** fit a single 35T — it needs either FFN tiling (narrowing the glue/attention to share memory) or a board with more on-chip RAM. I want to be unambiguous about this, because it is the one place the "single 130-dollar board" framing has an asterisk: each accelerator is silicon-proven, a pair co-resides and cooperates on silicon, but the whole loop in one bitstream is the step that needs a bigger board or a tiling pass. The *energy* argument is built from cycle counts and holds regardless of which board the loop ultimately runs on; what Phase 9 proves is that the accelerators share a die and hand off data for real.

---

## The Evidence Ledger

Hardware projects, like benchmark papers, are easy to oversell. So here is every load-bearing claim in this post, sorted by how I actually know it. **Silicon-measured** means I read it off the physical board. **Derived** means I composed it from silicon-measured primitives. **Projected** means it's a forward estimate I have not yet built.

| Claim | Value | Evidence |
|---|---|---|
| The ternary "multiply" uses no hardware multipliers | **0 DSP** up to datapath width 2048 | **silicon-measured** (synthesis + on-board) |
| Engine throughput | **1.00 cycle/tile**, bit-exact (800 M MAC/s) | **silicon-measured** |
| DDR3 read bandwidth (the decode bottleneck) | **1,423 MB/s** (89% of port peak) | **silicon-measured** |
| Host-CPU transformer glue | **19.42 M cycles/layer** | **silicon-measured** |
| On-fabric attention | **16,456 cycles/query**, bit-exact | **silicon-measured** |
| On-fabric FFN glue | **13,974 cycles/layer**, bit-exact (184×) | **silicon-measured** |
| Engine + FFN-glue co-resident and cooperating | bit-exact end-to-end, **45/50 BRAM** | **silicon-measured** |
| CPU / GPU energy baselines | **4.62 / 3.67 J/token** | measured (RAPL / `nvidia-smi`) |
| FFN activation sparsity, and that it's unstructured | **59.8%**, 94% data-dependent | measured (model analysis) |
| System energy, host-split (the loss) | **4.32 J/token** (1.2× worse) | **derived** (measured cycles × est. power) |
| System energy, +on-fabric attention | **1.99 J/token** (~1.8× under GPU) | **derived** |
| System energy, +on-fabric FFN glue | **1.62 J/token** (~2.3× under GPU) | **derived** |
| Engine-bound floor (RMSNorm also on fabric) | **1.47 J/token** (~2.5× under GPU) | **projected** |
| Full three-accelerator loop in one bitstream | — | **projected** (doesn't fit a 35T; needs tiling/bigger board) |
| Relu-fied sparsity upside (10–20× on the FFN) | 85–95% sparse | **projected** (needs a fine-tune) |

One caveat deserves to be stated loudly, because it touches every energy number: **power is the one quantity I did not meter.** The 0.489 W I multiply cycles by is Vivado's vectorless post-route estimate, not a reading from a current probe. So every joule-per-token here is honestly *"measured cycle counts times an estimated wattage."* The cycle path is silicon-measured end-to-end; closing the loop with a metered watt is the single highest-value thing left to do, and I'd treat the energy ratios as good-to-~20% until then.

With that said: the load-bearing surprise — that a GPU extracts almost nothing from ternary weights (3.67 J/token, barely beating a CPU) — is directly measured, and the engine differentiator (0 DSP, 1 cycle/tile, sub-watt) is directly measured. The ~2.3× system win is derived from those, not asserted.

## What I Learned

1. **Memory is the cost, not arithmetic.** Every advantage in this project comes from moving *fewer bytes* — ternary weights, skipped sparse columns — not from doing math faster. The roofline said batch-1 decode is bandwidth-bound before I wrote a line of RTL, and it was right the whole way down. If you take one thing from this post, let it be that the interesting lever in edge LLM inference is the memory system, and that's a lever a tiny device can pull.

2. **The honest pivot beat the optimistic projection.** The most valuable measurement I made was the one where the FPGA *lost* — 4.32 J/token, 1.2× worse than the GPU, in Phase 4. A projection would have quietly rounded that away. Measuring it converted "attention should probably be on the fabric" into a quantified necessity and set the agenda for the three phases that produced the actual win. Build the thing that can embarrass you.

3. **Label every number by how you got it.** Silicon-measured, derived, projected. Keeping those tiers separate — in the build log, in the README, in this post — is the entire difference between a result and a press release. It also makes the open work obvious: the projected rows above *are* the to-do list.

4. **A memory's read and write must live in the same clocked block,** or the synthesis tool builds a flip-flop array with an address decoder instead of a block RAM and detonates your resource budget. This bug cost me a wasted synthesis run three separate times — on three different modules — before it finally stuck. Some lessons you learn once; this one I had to learn thrice.

5. **Integration tests earn their keep.** Three unit tests passed and all three missed the one hard-coded constant (`0xAA55` summing to zero) that made the first on-board run print garbage. The bug only existed at the seam the unit tests didn't cover. On hardware, where a wrong bitstream is a ten-minute round-trip, the test that exercises the whole path is worth ten that exercise pieces.

6. **Suspect the boring thing first.** I lost the better part of a day to "soft-float math is too slow" when the real cause was a single missing line that enables interrupts, stalling the serial port. The exotic explanation is seductive precisely because it's interesting. Check the plumbing before the theory.

## What I'd Build Next

In rough order of value:

- **RMSNorm and quantization on the fabric.** It's the last glue term — 0.54M cycles, the gap between the measured 1.62 J/token and the 1.47 engine floor. The cancellation identity from Phase 2 already shows most of it is integer arithmetic; the unit should be small.
- **The full three-accelerator decode loop, end to end.** This is the one that needs the decision Phase 9 surfaced: either tile the FFN/attention to share BRAM on the 35T, or move to a board with more on-chip memory (an A7-100T, ~$250, or a Zynq KV260). The reward is a *measured* full-token J/token instead of a derived one.
- **A metered watt.** Replace the Vivado power estimate with a current-probe reading on the board's supply rails, and the energy numbers stop having an asterisk.
- **A relu-fied ternary fine-tune.** Pushing FFN sparsity from the measured 60% toward the 85–95% of ProSparse-style models roughly doubles the gather's payoff — the full ternary × sparsity stack, where the FPGA's advantage is largest. The datapath to exploit it already exists; the model does not yet.
- **A wider, DMA-fed engine.** The K=8 engine is compute-bound at 200 MB/s against a 1.4 GB/s channel. Widening it and feeding it by DMA would convert the spare bandwidth into throughput (tokens/second) — the axis this project concedes by design, but one worth a real number.
- **Independent reproduction.** The repo is built for it — test-driven, with a one-command simulation harness — but the real bar is someone else flashing their own Arty and reading the same `PASS` lines. If that someone is you, the build log is the map.

## Acknowledgments

This project stands on a specific stack of prior work: the **BitNet b1.58** line that made ternary LLMs trainable and good, and the FPGA-inference papers — **TerEffic**, **TeLLMe**, **ProSparse** — that established the LUT-based ternary datapath and the sparsity numbers I measured against. The open toolchain did the heavy lifting: **LiteX** and **LiteDRAM** for the SoC and the DDR3 controller, **VexRiscv** for the soft CPU, **cocotb** and **Verilator** for test-driven simulation, **openFPGALoader** for flashing, and AMD's **Vivado** for synthesis. None of this would be approachable on a hobbyist budget without them.

The full source is on [GitHub](https://github.com/Neumann-Labs/ternfpga) under Apache-2.0, including the dated build log that every anecdote here came from, and the reproduction scripts. If you'd argue with any number in this post — especially the energy ratios, until that metered watt exists — open an issue. That's how the next iteration gets honest.
