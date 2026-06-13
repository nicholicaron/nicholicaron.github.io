---
layout: post
title: "Training a Language Model on Machines That Keep Dying"
date: 2026-06-13
tags: [Distributed Training, Fault Tolerance, DiLoCo, torchft, LLM]
cover_image: /assets/images/ft-diloco/cover.png
---

Thirty-two copies of a language model were training on a single desktop. Every twenty-nine seconds, on a schedule, something went wrong on purpose: a process `kill -9`'d out of existence, or frozen mid-step with a `SIGSTOP`, or cut off from the network entirely. Over half an hour, **twenty-seven of them were killed outright** — and partitioned, and stalled, sixty-two faults in all. Nobody touched a keyboard. At the end, the model's loss had fallen **10.8 → 4.3** in a clean monotonic descent, and the cluster had retained **97.7% of the throughput** it would have had if nothing ever broke.

<div style="margin: 2rem 0; padding: 0.75rem; background: var(--surface-container); border: 1px solid var(--outline-variant); border-radius: 0.5rem;">
<img class="ftd-gif ftd-gif-light" src="/assets/images/ft-diloco/storm_light.gif" alt="A 32-cell grid of replicas changing color as faults hit, with live quorum and loss charts, reconstructed from telemetry" style="max-width:100%; display:block; margin:0 auto; border-radius:0.25rem;">
<img class="ftd-gif ftd-gif-dark" src="/assets/images/ft-diloco/storm_dark.gif" alt="A 32-cell grid of replicas changing color as faults hit, with live quorum and loss charts, reconstructed from telemetry" style="max-width:100%; display:block; margin:0 auto; border-radius:0.25rem;">
</div>
<style>.ftd-gif-dark{display:none} html.dark .ftd-gif-light{display:none} html.dark .ftd-gif-dark{display:block}</style>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The whole project in twenty-six seconds. Each cell is one of 32 replica groups, colored by what it's doing — training, committing a sync, killed, frozen, partitioned, or healing back from a peer. The left chart tracks the live quorum, the right chart the eval loss, and the red ticks mark every kill. Thirty minutes of real wall-clock is compressed here; the animation isn't a screen recording but a reconstruction from the timestamped JSONL each replica wrote, which is why the time axis and the encoding are mine to choose. The thing to watch is that the chaos on the grid never stops the loss on the right from falling. (This figure ships in two renders — one tuned for each — and swaps with the light/dark toggle at the top of the page.)</p>

That this works at all is the easy part of the story, and it isn't mine — it's [DiLoCo](https://arxiv.org/abs/2311.08105) and Meta's [torchft](https://github.com/meta-pytorch/torchft). The interesting part is what you have to discover to make it work *honestly*. torchft's headline result is fault tolerance at the scale you'd expect it: ~300 production GPUs, a real training job, machines failing about once a minute. The question that started this project was smaller and more suspicious — **does any of that survive on the kind of hardware you'd actually scrounge?** A gaming GPU, a couple of old boxes, a home network you can throttle to a crawl.

The answer is yes, but the yes comes with two distinctions I didn't have words for when I started, and that turned out to *be* the project:

- **Connectivity ≠ coordination.** Two machines that can reach each other are not therefore training together. I watched a "cluster" silently dissolve into independent solo runs while every node reported perfect health.
- **Liveness ≠ participation.** At scale, "alive" and "contributing to this sync" are different numbers, and the gap between them is a real, measurable tax that no dashboard was showing me.

This is the build log of getting there — across six milestones, on a Ryzen desktop and an 8-core box and about three dollars of rented cloud — including the run where the loss *regressed* while every throughput metric looked perfect, which is the actual research finding buried in here.

I'll be disciplined about provenance, because distributed-systems claims oversell as easily as benchmark numbers. Every load-bearing figure below is tagged: **measured** (read directly from a run's telemetry), **derived** (composed from measured primitives), or **single-run** (a one-shot result I haven't repeated — the cloud experiments, mostly). There's a full ledger near the end. The animation above, and every chart in this post, is reconstructed from the JSONL each run wrote — the timestamps are the ground truth.

## What This Post Covers

- **Why you'd train a model with rare syncs at all** — the assumption baked into normal distributed training, and the one DiLoCo relaxes to make commodity hardware viable.
- **What torchft actually guarantees** — quorum, peer-to-peer recovery, commit/rollback — and the open question it leaves, which is the one I set out to answer empirically: does the *outer optimizer's momentum* survive a machine leaving and coming back?
- **The six-milestone build, with the negative results kept in** — the silent eval-loss regression, the cluster that dissolved over a healthy network, and the participation ceiling that turned out to be a CPU-scheduling artifact.
- **The honest ledger** — what I measured, what I derived, and what's a single un-repeated run.

This post assumes you know roughly what it means to train a neural network with gradient descent. It does **not** assume you know what DiLoCo, a quorum, or an all-reduce is — we'll build those up. If you do distributed training for a living, skim Part I.

---

## Part I — Why Train This Way at All

### The expensive-cluster assumption

The default way to train a model on many machines is **data-parallel**: every machine holds a full copy of the model, each processes a different slice of the batch, and after *every single step* they average their gradients so all copies stay identical. That average is an **all-reduce** — every machine sends its gradient to every other, billions of numbers, hundreds of times a second.

This is why "distributed training" is a synonym for "expensive interconnect." Syncing every step only makes sense if the network between your machines is nearly as fast as the memory inside them — NVLink, InfiniBand, a datacenter fabric. The moment your machines are connected by something ordinary — an office LAN, the open internet, a home router — per-step syncing collapses, because the machines spend all their time waiting on the wire instead of computing.

DiLoCo's move is to relax *when* you synchronize, not just how much. Each worker trains entirely on its own for **H steps** — a hundred, five hundred — using a normal inner optimizer (AdamW). Only then do the workers compare notes, and what they exchange isn't a per-step gradient but a **pseudo-gradient**: the total drift of each worker's parameters over those H steps. A second, *outer* optimizer (Nesterov-momentum SGD) treats that averaged drift as a single gradient and takes one big step. Then everyone broadcasts the updated weights and runs another H steps alone.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 780 320" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <defs>
    <marker id="ah1" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L7,3 L0,6 Z" fill="var(--primary,#94452b)"/>
    </marker>
  </defs>
  <!-- inner loop boxes -->
  <g>
    <rect x="32" y="44" width="246" height="74" rx="8" fill="var(--primary-container,#fceee9)" stroke="var(--primary,#94452b)" stroke-width="1.3"/>
    <text x="46" y="66" font-size="12.5" font-weight="600" fill="currentColor">Replica A — inner loop</text>
    <text x="46" y="84" font-size="10.5" fill="currentColor" opacity="0.65">AdamW, H steps, fully local (no network)</text>
    <g fill="var(--primary,#94452b)">
      <circle cx="52" cy="104" r="3"/><circle cx="70" cy="104" r="3"/><circle cx="88" cy="104" r="3"/><circle cx="106" cy="104" r="3"/><circle cx="124" cy="104" r="3"/><circle cx="142" cy="104" r="3"/><circle cx="160" cy="104" r="3"/>
    </g>
    <text x="178" y="108" font-size="10.5" fill="currentColor" opacity="0.6">… ×H</text>

    <rect x="32" y="200" width="246" height="74" rx="8" fill="var(--primary-container,#fceee9)" stroke="var(--primary,#94452b)" stroke-width="1.3"/>
    <text x="46" y="222" font-size="12.5" font-weight="600" fill="currentColor">Replica B — inner loop</text>
    <text x="46" y="240" font-size="10.5" fill="currentColor" opacity="0.65">AdamW, H steps, fully local (no network)</text>
    <g fill="var(--primary,#94452b)">
      <circle cx="52" cy="260" r="3"/><circle cx="70" cy="260" r="3"/><circle cx="88" cy="260" r="3"/><circle cx="106" cy="260" r="3"/><circle cx="124" cy="260" r="3"/><circle cx="142" cy="260" r="3"/><circle cx="160" cy="260" r="3"/>
    </g>
    <text x="178" y="264" font-size="10.5" fill="currentColor" opacity="0.6">… ×H</text>
  </g>
  <!-- pseudo-gradient arrows to the average -->
  <path d="M278,80 C320,80 330,140 372,150" fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.7" marker-end="url(#ah1)"/>
  <path d="M278,238 C320,238 330,176 372,166" fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.7" marker-end="url(#ah1)"/>
  <text x="300" y="74" font-size="10.5" fill="var(--primary,#94452b)" font-weight="600">Δ&#8202;A</text>
  <text x="300" y="252" font-size="10.5" fill="var(--primary,#94452b)" font-weight="600">Δ&#8202;B</text>
  <text x="312" y="128" font-size="9.5" fill="currentColor" opacity="0.6" text-anchor="middle">pseudo-grad</text>
  <text x="312" y="140" font-size="9.5" fill="currentColor" opacity="0.6" text-anchor="middle">Δ = θ&#8320; − θ&#8336;</text>
  <!-- average box -->
  <rect x="382" y="128" width="118" height="62" rx="8" fill="var(--surface-container,#f3f0eb)" stroke="currentColor" stroke-width="1" stroke-opacity="0.5"/>
  <text x="441" y="153" font-size="11.5" font-weight="600" fill="currentColor" text-anchor="middle">all-reduce</text>
  <text x="441" y="170" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.65">average the Δ's</text>
  <text x="441" y="183" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.5">(once per H steps)</text>
  <!-- outer step box -->
  <path d="M500,159 L520,159" fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.7" marker-end="url(#ah1)"/>
  <rect x="524" y="128" width="132" height="62" rx="8" fill="var(--primary-container,#fceee9)" stroke="var(--primary,#94452b)" stroke-width="1.3"/>
  <text x="590" y="151" font-size="11.5" font-weight="600" fill="currentColor" text-anchor="middle">outer step</text>
  <text x="590" y="168" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.7">Nesterov SGD</text>
  <text x="590" y="181" font-size="9" fill="currentColor" text-anchor="middle" opacity="0.5">on the averaged Δ</text>
  <!-- broadcast loop back -->
  <path d="M656,170 C710,178 716,295 360,295 L150,295 C70,295 56,290 56,200 L56,196"
        fill="none" stroke="var(--primary,#94452b)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#ah1)" opacity="0.85"/>
  <path d="M56,118 L56,124" fill="none" stroke="var(--primary,#94452b)" stroke-width="1.4" stroke-dasharray="5 3" marker-end="url(#ah1)" opacity="0.85"/>
  <text x="430" y="289" font-size="10.5" fill="var(--primary,#94452b)" text-anchor="middle" font-weight="600">broadcast updated params θ, repeat</text>
  <!-- headline annotation -->
  <text x="700" y="120" font-size="10.5" fill="currentColor" opacity="0.6" text-anchor="end">~H× less traffic than</text>
  <text x="700" y="133" font-size="10.5" fill="currentColor" opacity="0.6" text-anchor="end">syncing every step</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The DiLoCo loop. Each replica runs H inner AdamW steps with no communication at all, then the replicas exchange one number per parameter — the pseudo-gradient Δ, the drift over those H steps — average it, and the outer Nesterov optimizer takes a single step on that average before broadcasting the new weights. The communication that data-parallel training does every step happens here once every H steps, so the wire traffic drops by roughly a factor of H. The price is that the workers diverge for H steps before being pulled back together, and the quality of the final model depends on H and the outer learning rate. That this trade is *smooth* — that you can dial H up and watch quality degrade gracefully — is what makes it a knob rather than a cliff, and Part II measures exactly where it bends.</p>

The consequence is the whole reason this is interesting on cheap hardware: **communication drops by roughly a factor of H.** If you sync every 100 steps instead of every step, you move ~100× less data. Suddenly a home gigabit link — or a 50 Mbps DSL connection, or the open internet between two cities — is fast enough to train collaboratively. And because the syncs are rare and coarse, the system can tolerate a worker vanishing between them, which is where fault tolerance comes in.

### What torchft adds, and the question it leaves open

DiLoCo is an algorithm; you still need machinery to run it across machines that fail. That's [torchft](https://github.com/meta-pytorch/torchft). It contributes three things. A **lighthouse** — a small coordinator process — tracks which workers are alive and forms a **quorum** (the set of workers participating in a given sync). Each worker runs a **manager** that handles the membership protocol. When a worker dies and a replacement starts, torchft does **peer-to-peer recovery**: the newcomer pulls the current model state directly from a living peer, no shared checkpoint required. And every sync is a **commit/rollback transaction** — if a worker drops mid-sync, the others roll back cleanly rather than corrupting the average.

The torchft team [demonstrated](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synchronizations/) this at scale: a Llama-3 1B model, ~300 L40S GPUs in 30 replica groups, a failure injected roughly once a minute, sustaining **82.3% step efficiency** through it. That's the number to beat — or rather, to see whether anything like it holds three orders of magnitude down the hardware ladder.

There's a specific open question underneath all this, raised in torchft's own [issue #171](https://github.com/meta-pytorch/torchft/issues/171) on semi-synchronous training. DiLoCo's outer optimizer has **momentum** — a running memory of past pseudo-gradients, essential to convergence. When a worker is killed and rebuilt from a peer, does it recover that outer-optimizer momentum *exactly*, or does recovery quietly reset it and degrade the run? Peer-to-peer recovery of model weights is one thing; bit-exact recovery of optimizer state through a real `kill -9` is the empirical question I most wanted to answer, because if it doesn't hold, none of the rest matters. (torchft is moving fast; I pinned commit `4157be16` for every run here so the numbers are reproducible.)

### The rig

Everything below runs on deliberately ordinary hardware: a **Ryzen 9 5950X with a single RTX 3060 (12 GB)** as the GPU trainer, an **8-core box** as the lighthouse, gigabit home ethernet with `tc/netem` standing in for a worse WAN, and — for the two cross-region experiments — about **\$3 of rented cloud**. The model is a small GPT (51M parameters for the convergence work, a 3.3M "micro" model for the largest-scale chaos) trained on TinyStories. None of this is a cluster. That's the point.

---

## Part II — Building It, One Honest Measurement at a Time

### M0 — A baseline, and a memory surprise

You cannot claim fault tolerance preserves quality without a quality bar to compare against, so the first run was the most boring possible: a single GPU, no faults, three random seeds. The 51M model settles at an eval loss of **1.6773 ± 0.0009** — tight enough across seeds that any later degradation will be visible against it. *[measured]*

The only surprise here was a memory one, and it's worth a sentence because it bites everyone once. The first training run OOM'd at a batch size the GPU should have handled with room to spare. The culprit wasn't the weights — it was the **cross-entropy loss over a 50,000-token vocabulary**, which briefly materializes a logits tensor of `batch × sequence × vocab` floats. The model is tiny; its *output distribution* is enormous. The fix was a micro-batch with gradient accumulation, but the lesson generalizes: in a language model, profile the loss, not just the parameters.

### M0.5 — Does any of this actually work on my hardware?

Before building anything elaborate, I wanted the Part I question answered on the real rig: kill a worker mid-training and see what survives. Two replicas, training; then `kill -9` on one. The survivor's next sync committed **5.0 seconds** later with the quorum cleanly shrunk from two to one — no stall, no manual intervention. The killed worker was relaunched, and torchft logged exactly what I'd hoped: `healing is required`, then a peer-to-peer state transfer from the survivor. **54 seconds** after the kill it had rejoined and committed as a full member again — most of that time process and CUDA startup, not the transfer itself. *[measured]*

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 780 260" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <defs>
    <marker id="ah2" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L7,3 L0,6 Z" fill="var(--primary,#94452b)"/>
    </marker>
  </defs>
  <!-- lane labels -->
  <text x="14" y="84" font-size="12" font-weight="600" fill="currentColor">Replica A</text>
  <text x="14" y="100" font-size="9.5" fill="currentColor" opacity="0.55">survivor</text>
  <text x="14" y="184" font-size="12" font-weight="600" fill="currentColor">Replica B</text>
  <text x="14" y="200" font-size="9.5" fill="currentColor" opacity="0.55">killed</text>
  <!-- time axis -->
  <line x1="92" y1="232" x2="740" y2="232" stroke="currentColor" stroke-width="1" opacity="0.25"/>
  <text x="740" y="248" font-size="10" fill="currentColor" opacity="0.5" text-anchor="end">wall-clock time →</text>
  <!-- lane A (survives the whole way) -->
  <line x1="92" y1="80" x2="740" y2="80" stroke="var(--primary,#94452b)" stroke-width="2.2"/>
  <!-- lane B: train, killed, gap, relaunched -->
  <line x1="92" y1="176" x2="300" y2="176" stroke="var(--primary,#94452b)" stroke-width="2.2"/>
  <line x1="300" y1="176" x2="452" y2="176" stroke="currentColor" stroke-width="1.4" stroke-dasharray="3 4" opacity="0.4"/>
  <line x1="452" y1="176" x2="740" y2="176" stroke="var(--primary,#94452b)" stroke-width="2.2"/>
  <!-- kill marker -->
  <circle cx="300" cy="176" r="9" fill="none" stroke="var(--error,#a64542)" stroke-width="2"/>
  <line x1="295" y1="171" x2="305" y2="181" stroke="var(--error,#a64542)" stroke-width="2"/>
  <line x1="305" y1="171" x2="295" y2="181" stroke="var(--error,#a64542)" stroke-width="2"/>
  <text x="300" y="150" font-size="11" font-weight="600" fill="var(--error,#a64542)" text-anchor="middle">kill -9</text>
  <!-- survivor solo commit -->
  <circle cx="352" cy="80" r="5" fill="var(--primary,#94452b)"/>
  <text x="352" y="64" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.8">commits solo</text>
  <text x="352" y="52" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.6">quorum 2 → 1</text>
  <!-- relaunch -->
  <circle cx="452" cy="176" r="4" fill="currentColor" opacity="0.5"/>
  <text x="452" y="200" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.65">relaunch</text>
  <!-- P2P heal arrow A -> B -->
  <path d="M500,88 L500,166" fill="none" stroke="var(--primary,#94452b)" stroke-width="1.6" marker-end="url(#ah2)"/>
  <text x="512" y="128" font-size="10" fill="var(--primary,#94452b)" font-weight="600">P2P recovery</text>
  <text x="512" y="141" font-size="9.5" fill="currentColor" opacity="0.65">params + outer momentum</text>
  <!-- joint 2-participant commit -->
  <line x1="588" y1="80" x2="588" y2="176" stroke="currentColor" stroke-width="1" opacity="0.35"/>
  <circle cx="588" cy="80" r="5" fill="var(--primary,#94452b)"/>
  <circle cx="588" cy="176" r="5" fill="var(--primary,#94452b)"/>
  <text x="588" y="64" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.8">2-participant commit</text>
  <!-- digest callout -->
  <rect x="630" y="108" width="138" height="40" rx="7" fill="var(--surface-container,#f3f0eb)" stroke="currentColor" stroke-opacity="0.4"/>
  <text x="699" y="126" font-size="10.5" font-weight="600" fill="var(--primary,#94452b)" text-anchor="middle">digests match</text>
  <text x="699" y="140" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.65">bit-identical, no checkpoint</text>
  <!-- timing brackets -->
  <line x1="300" y1="216" x2="352" y2="216" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <text x="326" y="212" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.6">T_resume 5s</text>
  <line x1="300" y1="228" x2="588" y2="228" stroke="currentColor" stroke-width="1" opacity="0.4"/>
  <text x="444" y="224" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.6">T_rejoin 54s (mostly process + CUDA startup)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">A single kill-and-recover, the mechanism the whole project rests on. Replica B is killed mid-run; the quorum shrinks two-to-one and the survivor (A) keeps committing solo — fault tolerance means the cluster doesn't wait for the dead. When B is relaunched it doesn't restart from scratch: it pulls the live model state directly from A over a peer-to-peer transfer and rejoins at the cluster's current step. The number that mattered to me is the one in the box: I logged a SHA-256 digest of the parameters *and* the outer Nesterov momentum at every post-recovery sync, and every single one matched bit-for-bit across both replicas. The outer optimizer's momentum — the thing issue #171 worries about — survives a real `kill -9` exactly, with no checkpoint involved. That's the foundation; if this digest hadn't matched, nothing later would be trustworthy.</p>

The headline from M0.5 is that box: at every sync after recovery, I logged a digest of the parameters and the outer momentum buffers, and **they were bit-identical across replicas, every time.** The outer optimizer's momentum survives a kill-and-rejoin through live recovery alone — no durable checkpoint needed. That's the #171 question answered in the affirmative, at least in the easy regime of a single failure with a healthy survivor. (M3 will find the regime where "a healthy survivor" stops being a safe assumption.)

A footnote for anyone trying this: torchft's standalone path has sharp edges that aren't well documented. The manager hard-requires torchrun-style environment variables and a `TCPStore` it expects something else to host; run it bare and it blocks forever with no error. I ended up hosting the store myself and steering around a couple of known live bugs (`use_async_quorum=False`, HTTP transport for recovery). Reproducing this on your own hardware is very doable, but budget an afternoon for the plumbing.

### M1 — Parity, and the communication win

With recovery proven, the next question is the core DiLoCo trade: how much quality do you give up for syncing rarely, and how much communication do you actually save? I ran a sweep — sync every H = 25, 50, 100, 200, 500 steps — against the M0 baseline at equal total tokens.

| sync every H steps | eval loss | vs baseline | communication |
|---|---|---|---|
| **25** | 1.724 | +2.8% | 25× less |
| **50** | 1.756 | +4.7% | 50× less |
| **100** | 1.783 | +6.3% | 100× less |
| **200** | 1.801 | +7.4% | 200× less |
| **500** | 1.836 | +9.4% | 500× less |

The trade is smooth and monotonic, exactly as DiLoCo's authors found — which is what makes H a usable dial. *[measured; single seed per H, and the outer learning rate is left untuned across H, which honestly favors small H — a per-H tune would flatten the right end of this curve.]* The communication reduction is exact: H = 100 really does move 100× fewer bytes.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Communication volume per replica — DiLoCo vs syncing every step</text>
  <line x1="64.0" y1="332.3" x2="738.0" y2="332.3" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="336.3" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">1 GB</text>
  <line x1="64.0" y1="284.8" x2="738.0" y2="284.8" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="288.8" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">3 GB</text>
  <line x1="64.0" y1="232.6" x2="738.0" y2="232.6" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="236.6" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">10 GB</text>
  <line x1="64.0" y1="185.1" x2="738.0" y2="185.1" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="189.1" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">30 GB</text>
  <line x1="64.0" y1="132.9" x2="738.0" y2="132.9" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="136.9" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">100 GB</text>
  <line x1="64.0" y1="85.4" x2="738.0" y2="85.4" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="89.4" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">300 GB</text>
  <text transform="rotate(-90 16 188)" x="16.0" y="188.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">GB / replica (whole run)</text>
  <rect x="80.8" y="54.3" width="33.7" height="287.7" rx="2" fill="var(--error,#a64542)" opacity="1.0"/>
  <rect x="114.6" y="193.7" width="33.7" height="148.3" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <rect x="148.3" y="192.8" width="33.7" height="149.2" rx="2" fill="currentColor" opacity="0.85"/>
  <text x="131.4" y="187.7" font-size="11" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">25×</text>
  <rect x="215.7" y="54.3" width="33.7" height="287.7" rx="2" fill="var(--error,#a64542)" opacity="1.0"/>
  <rect x="249.4" y="223.7" width="33.7" height="118.3" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <rect x="283.1" y="222.0" width="33.7" height="120.0" rx="2" fill="currentColor" opacity="0.85"/>
  <text x="266.2" y="217.7" font-size="11" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">50×</text>
  <rect x="350.4" y="54.3" width="33.7" height="287.7" rx="2" fill="var(--error,#a64542)" opacity="1.0"/>
  <rect x="384.1" y="253.7" width="33.7" height="88.3" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <rect x="417.8" y="250.5" width="33.7" height="91.5" rx="2" fill="currentColor" opacity="0.85"/>
  <text x="401.0" y="247.7" font-size="11" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">100×</text>
  <rect x="485.3" y="54.3" width="33.7" height="287.7" rx="2" fill="var(--error,#a64542)" opacity="1.0"/>
  <rect x="519.0" y="283.8" width="33.7" height="58.2" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <rect x="552.7" y="277.5" width="33.7" height="64.5" rx="2" fill="currentColor" opacity="0.85"/>
  <text x="535.8" y="277.8" font-size="11" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">200×</text>
  <rect x="620.0" y="54.3" width="33.7" height="287.7" rx="2" fill="var(--error,#a64542)" opacity="1.0"/>
  <rect x="653.8" y="323.4" width="33.7" height="18.6" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <rect x="687.5" y="309.4" width="33.7" height="32.6" rx="2" fill="currentColor" opacity="0.85"/>
  <text x="670.6" y="317.4" font-size="11" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">500×</text>
  <text x="131.4" y="360.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">H=25</text>
  <text x="266.2" y="360.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">H=50</text>
  <text x="401.0" y="360.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">H=100</text>
  <text x="535.8" y="360.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">H=200</text>
  <text x="670.6" y="360.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">H=500</text>
  <text x="401.0" y="390.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">inner steps between syncs (H)</text>
  <rect x="72.0" y="36.0" width="11.0" height="11.0" rx="2" fill="var(--error,#a64542)" opacity="0.9"/>
  <text x="88.0" y="45.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.8" font-weight="400">sync every step (DDP)</text>
  <rect x="244.6" y="36.0" width="11.0" height="11.0" rx="2" fill="var(--primary,#94452b)" opacity="0.9"/>
  <text x="260.6" y="45.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.8" font-weight="400">DiLoCo</text>
  <rect x="318.2" y="36.0" width="11.0" height="11.0" rx="2" fill="currentColor" opacity="0.7"/>
  <text x="334.2" y="45.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.8" font-weight="400">DiLoCo measured (veth)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Communication volume per replica over a whole run, on a log scale. The red bars are what data-parallel training would have cost on the same model — sync every step — and they tower over everything, flat across H because per-step syncing doesn't care how you've set H. The brown bars are DiLoCo's analytic cost, dropping by exactly the factor labeled above each pair: 25×, 50×, on up to 500×. The third bar in each group is what I actually measured off the virtual ethernet counters, and it tracks the analytic number to within a few percent — except the gap widens at large H, which is the tell. That residual is a roughly constant ~0.5 GB-per-run floor of control-plane chatter — the lighthouse heartbeats and quorum messages — that only becomes visible once the pseudo-gradient payload itself shrinks small enough to stop dominating. It's a reminder that "communication" in a fault-tolerant system isn't only the data plane; there's a coordinator quietly talking the whole time.</p>

The measured bytes track the analytic prediction within a few percent — until H gets large, where a constant ~0.5 GB-per-run floor of lighthouse heartbeats and quorum traffic starts to show through. It never matters for the data plane, but it foreshadows M4, where that same control-plane traffic becomes the thing that breaks.

### M2 — The money shot: killing a node on camera

M0.5 proved recovery works; M2 was about proving it works under a *scripted* sequence of real faults, and instrumenting it well enough to put numbers on recovery. I built a small chaos harness that injects genuine OS-level faults — not cooperative shutdowns but actual `kill -9`, `SIGSTOP` for stragglers, and link-down via `iptables` for partitions — each one logged with a timestamp to a ground-truth `chaos.jsonl`. A six-fault scenario (kill, relaunch, partition, heal, stall a straggler, resume) ran headless and finished within **+0.6%** of the fault-free loss, with **84 of 84** post-recovery digests matching. *[measured]*

The lesson I took from M2 is one I keep relearning: **the demo is half the deliverable.** A claim that "fault tolerance works" is worth far less than a recording of a node dying and the system shrugging it off, and building the telemetry to *reconstruct* that recording faithfully — every fault and commit timestamped — is what later let me animate a thirty-two-node storm from data alone, with no live screen-capture at all.

### M3 — Failure storms, and the negative result that's the actual research

Single scripted faults are a warm-up. A real test is a **storm**: faults arriving on a Poisson schedule, a supervisor automatically relaunching the dead, no human in the loop, for forty-five minutes straight. I ran two — one averaging a kill every 120 seconds, one every 60 — and measured step efficiency the way torchft does: committed training steps per second under chaos, as a fraction of the fault-free rate.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 640 400" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Step efficiency under failure storms (2 replicas, ~45 min each)</text>
  <line x1="64.0" y1="342.0" x2="618.0" y2="342.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="346.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">0%</text>
  <line x1="64.0" y1="280.4" x2="618.0" y2="280.4" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="284.4" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">20%</text>
  <line x1="64.0" y1="218.8" x2="618.0" y2="218.8" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="222.8" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">40%</text>
  <line x1="64.0" y1="157.2" x2="618.0" y2="157.2" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="161.2" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">60%</text>
  <line x1="64.0" y1="95.6" x2="618.0" y2="95.6" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="99.6" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">80%</text>
  <line x1="64.0" y1="34.0" x2="618.0" y2="34.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="38.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">100%</text>
  <text transform="rotate(-90 16 188)" x="16.0" y="188.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">% of fault-free throughput</text>
  <line x1="64.0" y1="34.0" x2="618.0" y2="34.0" stroke="currentColor" stroke-width="1" opacity="0.3" stroke-dasharray="4 3"/>
  <text x="614.0" y="29.0" font-size="10" fill="currentColor" text-anchor="end" opacity="0.55" font-weight="400">fault-free = 100%</text>
  <line x1="64.0" y1="88.5" x2="618.0" y2="88.5" stroke="var(--primary,#94452b)" stroke-width="1.2" opacity="0.7" stroke-dasharray="5 3"/>
  <rect x="144.3" y="70.3" width="116.3" height="271.7" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <text x="202.5" y="63.3" font-size="14" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">88.2%</text>
  <text x="202.5" y="360.0" font-size="11.5" fill="currentColor" text-anchor="middle" opacity="0.85" font-weight="500">storm k120</text>
  <text x="202.5" y="375.0" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.6" font-weight="400">69 faults/hr</text>
  <rect x="421.3" y="80.3" width="116.3" height="261.7" rx="2" fill="var(--primary,#94452b)" opacity="1.0"/>
  <text x="479.5" y="73.3" font-size="14" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">85.0%</text>
  <text x="479.5" y="360.0" font-size="11.5" fill="currentColor" text-anchor="middle" opacity="0.85" font-weight="500">storm k60</text>
  <text x="479.5" y="375.0" font-size="10" fill="currentColor" text-anchor="middle" opacity="0.6" font-weight="400">85 faults/hr</text>
  <text x="341.0" y="82.5" font-size="10.5" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">torchft: 82.3%</text>
  <text x="341.0" y="100.5" font-size="9" fill="var(--primary,#94452b)" text-anchor="middle" opacity="0.8" font-weight="400">(Llama-3 1B, 300 GPUs)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Step efficiency under two failure storms, against the references that matter. The dashed line at the top is fault-free throughput — the ceiling. The dashed line at 82.3% is torchft's published large-scale result on 300 GPUs, the number I most wanted to not embarrass myself against. Both of my storms — 69 and 85 executed faults per hour, heavier fault rates than the reference, on two consumer replicas — clear it: 88.2% and 85.0% of fault-free throughput. I want to be careful about what this does and doesn't claim. It is not "commodity hardware beats a datacenter," because the workloads and scales are completely different; it's that the *coordination machinery* doesn't fall apart at small scale, and that the efficiency cost of a fault is comparable. The throughput story, in other words, looked great. Which is exactly why the next figure matters.</p>

Both storms cleared 85% — above torchft's large-scale bar, at higher fault rates, on two consumer replicas. *[derived: committed-step rate under chaos ÷ the fault-free M1 rate.]* I was ready to call M3 a win on the strength of that number. Then I looked at the eval loss.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Throughput looked healthy while the model rotted — and the fix</text>
  <line x1="64.0" y1="288.4" x2="738.0" y2="288.4" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="292.4" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">2</text>
  <line x1="64.0" y1="173.2" x2="738.0" y2="173.2" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="177.2" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">3</text>
  <line x1="64.0" y1="57.9" x2="738.0" y2="57.9" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="61.9" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">4</text>
  <line x1="64.0" y1="-57.3" x2="738.0" y2="-57.3" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="-53.3" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">5</text>
  <text transform="rotate(-90 16 190)" x="16.0" y="190.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">global eval loss</text>
  <text x="64.0" y="364.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">0</text>
  <text x="401.0" y="364.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">23</text>
  <text x="738.0" y="364.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">47</text>
  <text x="401.0" y="390.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">storm time (min)</text>
  <polyline points="112.6,92.7 167.4,186.5 202.8,229.5 223.5,243.6 332.3,133.8 360.2,182.5 406.2,99.9 448.0,195.4 478.6,226.3 504.1,236.3 550.0,57.1 601.9,185.5 601.9,185.5 634.6,225.1 720.4,100.2 737.8,132.0" fill="none" stroke="var(--error,#a64542)" stroke-width="2.4" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="112.8,92.5 125.2,92.5 152.2,92.5 167.6,186.5 203.1,229.7 224.0,243.8 249.5,236.7 263.9,257.5 284.2,255.2 310.1,268.2 332.1,281.1 360.2,280.9 364.2,283.3 407.5,293.4 417.4,295.2 422.8,288.7 469.5,300.2 469.5,300.2 513.5,302.4 516.4,297.5 538.8,302.4 554.4,299.6 558.9,297.3 601.7,300.7 626.1,306.1 633.6,302.5 634.0,301.7 667.9,307.9 682.1,303.7 694.3,307.9 735.0,309.1 738.0,299.2" fill="none" stroke="var(--primary,#94452b)" stroke-width="2.4" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="737.8" y="124.0" font-size="11" fill="var(--error,#a64542)" text-anchor="end" opacity="1" font-weight="600">no checkpoints: regresses</text>
  <text x="738.0" y="315.2" font-size="11" fill="var(--primary,#94452b)" text-anchor="end" opacity="1" font-weight="600">commit-coupled checkpoints: holds</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The most important figure in this post, and the one I almost didn't make. The red curve is the eval loss during a storm where throughput read a healthy 87% the entire time — and the model is getting *worse*, oscillating up toward 4.0 after having reached 2.4. The system was committing syncs at a great rate; the syncs were poisoning the model. The mechanism is subtle and specific to small replica counts: a kill landing while the only other member is alive-but-not-yet-healed leaves a freshly-restarted worker as a one-member quorum, and that worker's near-random initial weights silently *become* the cluster's official state — I caught the survivors healing from a donor at step zero. Live peer-to-peer recovery, the thing M0.5 proved works, is necessary but not sufficient under restart churn. The brown curve is the same storm after the fix: each replica also persists state every few commits, so a wiped quorum resumes from durable ground truth instead of from noise, and a kill is only ever injected when a *healthy* donor exists. The loss descends and holds. Throughput was never the thing to watch.</p>

This is the research result, and it's a negative one I'm glad I kept. Throughput is a seductive metric because it's always green: as long as syncs are committing, the dashboard looks healthy. But a sync can commit *the wrong thing*. At small replica counts, a kill that lands while the cluster's only other member hasn't finished healing leaves a fresh-init worker alone in a quorum, and torchft faithfully makes that worker's random weights the official state — the survivors then "recover" from step zero. Throughput stayed at 87%; the model rotted from 2.4 back up toward 4.0.

The fix has two parts: **commit-coupled checkpoints** (each replica also persists durable state every few commits, so a wiped quorum resumes from real progress instead of noise) and an experiment-hygiene rule that a kill only fires when a healthy, recently-committed donor exists. With both in place, the second run's loss descended monotonically and held. torchft's 30-group setup makes this failure mode practically unreachable — but the cross-datacenter, few-big-members regime that #171 is actually about hits it head-on, which is why it's worth documenting rather than patching over.

### M4 — The WAN, and "connectivity ≠ coordination"

Everything so far ran on a fast LAN. The premise of the whole project is *unreliable, slow* links, so M4 made the network bad on purpose with `netem`, then made it real with rented cloud.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 760 410" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Throughput vs link speed — DiLoCo holds, per-step sync collapses</text>
  <line x1="64.0" y1="354.0" x2="738.0" y2="354.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="358.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">0k</text>
  <line x1="64.0" y1="274.0" x2="738.0" y2="274.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="278.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">10k</text>
  <line x1="64.0" y1="194.0" x2="738.0" y2="194.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="198.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">20k</text>
  <line x1="64.0" y1="114.0" x2="738.0" y2="114.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="118.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">30k</text>
  <line x1="64.0" y1="34.0" x2="738.0" y2="34.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="38.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">40k</text>
  <text transform="rotate(-90 16 194)" x="16.0" y="194.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">aggregate throughput (k tok/s)</text>
  <text x="94.0" y="372.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">10</text>
  <text x="310.5" y="372.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">50</text>
  <text x="403.7" y="372.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">100</text>
  <text x="713.5" y="372.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">1000</text>
  <text x="401.0" y="400.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">link bandwidth (Mbps), 20 ms RTT</text>
  <rect x="64.0" y="34.0" width="75.3" height="320.0" rx="0" fill="var(--error,#a64542)" opacity="0.07"/>
  <text x="94.0" y="48.0" font-size="9.5" fill="var(--error,#a64542)" text-anchor="middle" opacity="0.8" font-weight="400">control-plane</text>
  <text x="94.0" y="60.0" font-size="9.5" fill="var(--error,#a64542)" text-anchor="middle" opacity="0.8" font-weight="400">starvation</text>
  <polyline points="310.5,135.1 403.7,98.0 713.5,54.5" fill="none" stroke="var(--primary,#94452b)" stroke-width="2.4" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="94.0" y="334.8" font-size="10" fill="var(--primary,#94452b)" text-anchor="middle" opacity="1" font-weight="600">DNF</text>
  <circle cx="310.5" cy="135.1" r="3.2" fill="var(--primary,#94452b)"/>
  <circle cx="403.7" cy="98.0" r="3.2" fill="var(--primary,#94452b)"/>
  <circle cx="713.5" cy="54.5" r="3.2" fill="var(--primary,#94452b)"/>
  <polyline points="310.5,347.1 403.7,341.4 713.5,311.7" fill="none" stroke="var(--error,#a64542)" stroke-width="2.4" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <text x="94.0" y="334.8" font-size="10" fill="var(--error,#a64542)" text-anchor="middle" opacity="1" font-weight="600">DNF</text>
  <circle cx="310.5" cy="347.1" r="3.2" fill="var(--error,#a64542)"/>
  <circle cx="403.7" cy="341.4" r="3.2" fill="var(--error,#a64542)"/>
  <circle cx="713.5" cy="311.7" r="3.2" fill="var(--error,#a64542)"/>
  <rect x="538.0" y="38.0" width="11.0" height="11.0" rx="2" fill="var(--primary,#94452b)" opacity="1"/>
  <text x="554.0" y="47.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.85" font-weight="400">DiLoCo (H=100)</text>
  <rect x="538.0" y="56.0" width="11.0" height="11.0" rx="2" fill="var(--error,#a64542)" opacity="1"/>
  <text x="554.0" y="65.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.85" font-weight="400">sync every step</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Throughput as the link degrades from gigabit down to 10 Mbps, at a fixed 20 ms of latency. The lower curve is per-step syncing (data-parallel's communication pattern): even at gigabit it's already paying ~2 seconds per step to move the model, and it falls off a cliff as bandwidth shrinks. The upper curve is DiLoCo at H = 100 — flat across two orders of magnitude of bandwidth, because it amortizes that same transfer over a hundred local steps. The shaded zone on the left is where both die, and *how* they die is the interesting part: at 10 Mbps the ~200-second pseudo-gradient all-reduce so completely saturates the link that the lighthouse's own heartbeats can't get through, the quorum times out mid-transfer, and the cluster cascades into failure. The data plane starved the control plane — the same ~0.5 GB of coordinator chatter from the M1 figure, now fatal because it's competing for a pipe that's already full. The documented fix direction is quantized or streamed syncs; that's on the to-do list, not in this post.</p>

DiLoCo holds flat from gigabit down to 50 Mbps while per-step syncing collapses — exactly the regime DiLoCo exists for. *[measured]* The failure at 10 Mbps is the instructive part: the pseudo-gradient transfer saturates the link so thoroughly that torchft's *own coordination traffic* can't get through, and the quorum times out mid-sync. A fault-tolerant system has a control plane that needs bandwidth too, and starving it is its own failure mode.

Then I rented a GPU. A home RTX 3060 and a Virginia RTX 4090, joined over a Tailscale mesh across the open internet, trained as one DiLoCo cluster — and the model digests came out **bit-identical across the WAN**, the first time I'd seen the whole stack work between two machines in different states. Total spend across every cloud experiment, including the false starts: **\$3.12.** *[single-run.]* But scaling that up surfaced the finding that names the milestone.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 780 340" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <defs>
    <marker id="ah8" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="userSpaceOnUse">
      <path d="M0,0 L7,3 L0,6 Z" fill="var(--primary,#94452b)"/>
    </marker>
  </defs>
  <!-- ===== top panel: aligned ===== -->
  <text x="20" y="30" font-size="12.5" font-weight="600" fill="var(--primary,#94452b)">Aligned sync boundaries → one shared average</text>
  <text x="20" y="60" font-size="11" font-weight="600" fill="currentColor">A</text>
  <text x="20" y="100" font-size="11" font-weight="600" fill="currentColor">B</text>
  <!-- lanes with step ticks reaching the SAME boundary x=360 -->
  <line x1="40" y1="56" x2="360" y2="56" stroke="var(--primary,#94452b)" stroke-width="2"/>
  <line x1="40" y1="96" x2="360" y2="96" stroke="var(--primary,#94452b)" stroke-width="2"/>
  <g fill="currentColor" opacity="0.45">
    <circle cx="80" cy="56" r="2.4"/><circle cx="130" cy="56" r="2.4"/><circle cx="180" cy="56" r="2.4"/><circle cx="230" cy="56" r="2.4"/><circle cx="280" cy="56" r="2.4"/><circle cx="330" cy="56" r="2.4"/>
    <circle cx="80" cy="96" r="2.4"/><circle cx="130" cy="96" r="2.4"/><circle cx="180" cy="96" r="2.4"/><circle cx="230" cy="96" r="2.4"/><circle cx="280" cy="96" r="2.4"/><circle cx="330" cy="96" r="2.4"/>
  </g>
  <!-- shared barrier connector -->
  <line x1="360" y1="56" x2="360" y2="96" stroke="var(--primary,#94452b)" stroke-width="1.4"/>
  <circle cx="360" cy="56" r="5" fill="var(--primary,#94452b)"/>
  <circle cx="360" cy="96" r="5" fill="var(--primary,#94452b)"/>
  <text x="360" y="124" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.6">H-boundary, same wall-time</text>
  <path d="M372,76 L470,76" fill="none" stroke="var(--primary,#94452b)" stroke-width="1.5" marker-end="url(#ah8)"/>
  <rect x="476" y="54" width="120" height="44" rx="7" fill="var(--primary-container,#fceee9)" stroke="var(--primary,#94452b)" stroke-width="1.2"/>
  <text x="536" y="73" font-size="10.5" font-weight="600" fill="currentColor" text-anchor="middle">shared all-reduce</text>
  <text x="536" y="88" font-size="9.5" fill="currentColor" text-anchor="middle" opacity="0.65">both participate</text>
  <text x="616" y="71" font-size="11" fill="var(--primary,#94452b)" font-weight="600">→ identical model</text>
  <text x="616" y="86" font-size="9.5" fill="currentColor" opacity="0.65">digests match (a573c3de)</text>
  <!-- divider -->
  <line x1="20" y1="160" x2="760" y2="160" stroke="currentColor" stroke-width="1" opacity="0.18"/>
  <!-- ===== bottom panel: skewed ===== -->
  <text x="20" y="196" font-size="12.5" font-weight="600" fill="var(--error,#a64542)">Skewed boundaries → two solo runs (same network!)</text>
  <text x="20" y="232" font-size="11" font-weight="600" fill="currentColor">A</text>
  <text x="20" y="280" font-size="11" font-weight="600" fill="currentColor">B</text>
  <!-- A reaches boundary early (x=300), B late (x=430) -->
  <line x1="40" y1="228" x2="300" y2="228" stroke="var(--primary,#94452b)" stroke-width="2"/>
  <line x1="40" y1="276" x2="430" y2="276" stroke="var(--primary,#94452b)" stroke-width="2"/>
  <g fill="currentColor" opacity="0.45">
    <circle cx="80" cy="228" r="2.4"/><circle cx="125" cy="228" r="2.4"/><circle cx="170" cy="228" r="2.4"/><circle cx="215" cy="228" r="2.4"/><circle cx="260" cy="228" r="2.4"/>
    <circle cx="80" cy="276" r="2.4"/><circle cx="135" cy="276" r="2.4"/><circle cx="190" cy="276" r="2.4"/><circle cx="245" cy="276" r="2.4"/><circle cx="300" cy="276" r="2.4"/><circle cx="355" cy="276" r="2.4"/>
  </g>
  <!-- A commits solo -->
  <circle cx="300" cy="228" r="5" fill="var(--error,#a64542)"/>
  <text x="300" y="216" font-size="9.5" fill="var(--error,#a64542)" text-anchor="middle" font-weight="600">commits alone</text>
  <!-- B commits solo later -->
  <circle cx="430" cy="276" r="5" fill="var(--error,#a64542)"/>
  <text x="430" y="298" font-size="9.5" fill="var(--error,#a64542)" text-anchor="middle" font-weight="600">commits alone, later</text>
  <text x="600" y="240" font-size="11" fill="var(--error,#a64542)" font-weight="600" text-anchor="middle">→ two different models</text>
  <text x="600" y="255" font-size="9.5" fill="currentColor" opacity="0.65" text-anchor="middle">digests diverge ✗</text>
  <text x="600" y="278" font-size="9.5" fill="currentColor" opacity="0.55" text-anchor="middle">no peer is waiting at</text>
  <text x="600" y="290" font-size="9.5" fill="currentColor" opacity="0.55" text-anchor="middle">the barrier (min_replica_size=1)</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Connectivity is not coordination — the same network produces both of these outcomes. DiLoCo workers sync after every H *local* steps, which means they sync at whatever wall-clock moment they happen to finish those steps. Top: when the workers are well-matched and started together, their H-boundaries line up, both arrive at the barrier at the same time, and you get one shared all-reduce and one identical model — matching digests. Bottom: when the workers start minutes apart or run at different speeds (a 4090 is ~4× a 3060), each one reaches its boundary alone, finds no peer waiting, and — with the minimum-quorum size set to one — commits *by itself*. The cluster silently decays into N independent solo runs that each converge fine, but to different optima; the cross-region digests did not match. Both nodes reported perfect health the entire time. Nothing was down. They simply weren't training together, and no liveness check would ever tell you so. Genuine collaboration needs either aligned starts and matched speeds, or a hard barrier that makes fast workers wait — which has its own cost, measured next.</p>

This is the one that changed how I think about the whole problem. With the minimum quorum size set to one, two healthy, fully-connected nodes that hit their sync boundaries at different wall-clock times each commit *alone*, and the "cluster" quietly becomes two independent solo runs converging to different models — while every health check stays green. **Connectivity is not coordination.** The fix is a real barrier (require at least two participants per sync), but a barrier forces fast nodes to wait for slow ones, and that cost is exactly what the last milestone had to confront at scale.

### M5 — 32 replicas, one desktop, and "liveness ≠ participation"

torchft's framing is ~30 replica groups. I wanted to match that scale — but the unknowns at N = 32 weren't about geography, which M4 had already settled; they were about *coordination*. Does the lighthouse stay sane managing 32 members? Does a 32-way all-reduce form and commit? Does the barrier from M4 hold without falling over? Every one of those questions answers on a single machine, for **\$0**, with chaos I control precisely and no cloud nodes to herd. So M5 ran 32 replica groups as CPU processes on one Ryzen desktop, each in its own network namespace, with a 3.3M-parameter model — small because thirty-two full models don't fit commodity RAM, and the storm exercises the coordination machinery, which doesn't care how big the model is.

I built up to it on a de-risk ladder — N = 4, then 8, then 32 — and it paid for itself immediately. The "barrier OOM" that had spooked me in M4 turned out to be an artifact: orphaned processes from repeated dirty relaunches piling up on a machine I hadn't cleaned, not a real limit. On a clean box the barrier runs in about 6 GB. The lighthouse coordinated all 32 managers without complaint. The ring formed and committed. So far, so good — and then the median quorum came back at **16 of 32**, and stayed there even with no faults at all.

That number is the whole milestone, so I chased it. Sixteen of thirty-two participating, fault-free, is not a fault-tolerance story — it's a clue. The cause was **CPU oversubscription**: thirty-two compute-heavy processes time-slicing across sixteen physical cores get scheduled unevenly, so their step times drift apart, and at any given sync boundary only about half have arrived. Pinning each process to a dedicated core fixed it — but only after discovering that `taskset` silently *doesn't* work here, because PyTorch and its math libraries reset CPU affinity when they import. The pin has to be set from inside the process after the libraries load, and re-asserted, because the collective operations reset it again. With that, fault-free participation jumped to the full **32**.

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 820 380" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Liveness vs per-sync participation under 125 faults/hr</text>
  <line x1="64.0" y1="328.0" x2="798.0" y2="328.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="332.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">0</text>
  <line x1="64.0" y1="256.7" x2="798.0" y2="256.7" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="260.7" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">8</text>
  <line x1="64.0" y1="185.5" x2="798.0" y2="185.5" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="189.5" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">16</text>
  <line x1="64.0" y1="114.2" x2="798.0" y2="114.2" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="118.2" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">24</text>
  <line x1="64.0" y1="42.9" x2="798.0" y2="42.9" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="46.9" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">32</text>
  <text transform="rotate(-90 16 181)" x="16.0" y="181.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">replicas</text>
  <text x="64.0" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">0</text>
  <text x="433.3" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">15</text>
  <text x="802.6" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">30</text>
  <text x="431.0" y="370.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">storm time (min)</text>
  <line x1="64.0" y1="42.9" x2="798.0" y2="42.9" stroke="currentColor" stroke-width="1" opacity="0.3" stroke-dasharray="3 3"/>
  <text x="68.0" y="37.9" font-size="10" fill="currentColor" text-anchor="start" opacity="0.55" font-weight="400">cluster size N=32</text>
  <polygon points="64.0,328.0 64.0,42.9 70.1,42.9 76.2,42.9 82.4,60.7 88.5,51.8 94.6,51.8 100.7,60.7 106.8,51.8 112.9,42.9 119.0,51.8 125.2,51.8 131.3,42.9 137.4,42.9 143.5,42.9 149.6,42.9 155.8,42.9 161.9,42.9 168.0,42.9 174.1,42.9 180.2,42.9 186.3,42.9 192.5,51.8 198.6,51.8 204.7,69.6 210.8,78.5 216.9,69.6 223.0,69.6 229.1,60.7 235.3,60.7 241.4,51.8 247.5,42.9 253.6,51.8 259.7,51.8 265.9,42.9 272.0,42.9 278.1,42.9 284.2,42.9 290.3,42.9 296.4,42.9 302.5,51.8 308.7,51.8 314.8,51.8 320.9,42.9 327.0,42.9 333.1,42.9 339.2,42.9 345.4,42.9 351.5,51.8 357.6,51.8 363.7,60.7 369.8,69.6 376.0,42.9 382.1,42.9 388.2,42.9 394.3,42.9 400.4,42.9 406.5,51.8 412.6,51.8 418.8,51.8 424.9,42.9 431.0,51.8 437.1,60.7 443.2,78.5 449.4,78.5 455.5,78.5 461.6,69.6 467.7,69.6 473.8,60.7 479.9,51.8 486.0,42.9 492.2,51.8 498.3,42.9 504.4,42.9 510.5,42.9 516.6,51.8 522.8,60.7 528.9,60.7 535.0,60.7 541.1,51.8 547.2,42.9 553.3,42.9 559.5,60.7 565.6,60.7 571.7,60.7 577.8,51.8 583.9,60.7 590.0,51.8 596.1,51.8 602.3,51.8 608.4,42.9 614.5,51.8 620.6,60.7 626.7,51.8 632.9,51.8 639.0,51.8 645.1,60.7 651.2,51.8 657.3,51.8 663.4,51.8 669.5,60.7 675.7,60.7 681.8,42.9 687.9,51.8 694.0,42.9 700.1,42.9 706.2,42.9 712.4,42.9 718.5,42.9 724.6,51.8 730.7,51.8 736.8,42.9 743.0,42.9 749.1,42.9 755.2,51.8 761.3,51.8 767.4,51.8 773.5,60.7 779.6,69.6 785.8,51.8 791.9,42.9 798.0,51.8 798.0,328.0" fill="var(--primary,#94452b)" opacity="0.18"/>
  <polyline points="64.0,42.9 70.1,42.9 76.2,42.9 82.4,60.7 88.5,51.8 94.6,51.8 100.7,60.7 106.8,51.8 112.9,42.9 119.0,51.8 125.2,51.8 131.3,42.9 137.4,42.9 143.5,42.9 149.6,42.9 155.8,42.9 161.9,42.9 168.0,42.9 174.1,42.9 180.2,42.9 186.3,42.9 192.5,51.8 198.6,51.8 204.7,69.6 210.8,78.5 216.9,69.6 223.0,69.6 229.1,60.7 235.3,60.7 241.4,51.8 247.5,42.9 253.6,51.8 259.7,51.8 265.9,42.9 272.0,42.9 278.1,42.9 284.2,42.9 290.3,42.9 296.4,42.9 302.5,51.8 308.7,51.8 314.8,51.8 320.9,42.9 327.0,42.9 333.1,42.9 339.2,42.9 345.4,42.9 351.5,51.8 357.6,51.8 363.7,60.7 369.8,69.6 376.0,42.9 382.1,42.9 388.2,42.9 394.3,42.9 400.4,42.9 406.5,51.8 412.6,51.8 418.8,51.8 424.9,42.9 431.0,51.8 437.1,60.7 443.2,78.5 449.4,78.5 455.5,78.5 461.6,69.6 467.7,69.6 473.8,60.7 479.9,51.8 486.0,42.9 492.2,51.8 498.3,42.9 504.4,42.9 510.5,42.9 516.6,51.8 522.8,60.7 528.9,60.7 535.0,60.7 541.1,51.8 547.2,42.9 553.3,42.9 559.5,60.7 565.6,60.7 571.7,60.7 577.8,51.8 583.9,60.7 590.0,51.8 596.1,51.8 602.3,51.8 608.4,42.9 614.5,51.8 620.6,60.7 626.7,51.8 632.9,51.8 639.0,51.8 645.1,60.7 651.2,51.8 657.3,51.8 663.4,51.8 669.5,60.7 675.7,60.7 681.8,42.9 687.9,51.8 694.0,42.9 700.1,42.9 706.2,42.9 712.4,42.9 718.5,42.9 724.6,51.8 730.7,51.8 736.8,42.9 743.0,42.9 749.1,42.9 755.2,51.8 761.3,51.8 767.4,51.8 773.5,60.7 779.6,69.6 785.8,51.8 791.9,42.9 798.0,51.8" fill="none" stroke="var(--primary,#94452b)" stroke-width="1.0" opacity="0.45" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.8,176.5 76.9,176.5 76.9,176.5 76.9,176.5 76.9,176.5 76.9,176.5 76.9,176.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.6,185.5 99.7,185.5 99.7,185.5 99.7,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.4,185.5 125.5,185.5 125.5,185.5 125.5,185.5 125.5,185.5 125.5,185.5 125.5,185.5 125.6,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 148.0,185.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 175.6,176.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 195.0,185.5 228.4,185.5 228.4,185.5 228.4,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 228.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 245.5,185.5 281.8,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 281.9,176.5 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 301.2,194.4 338.5,176.5 338.5,176.5 338.5,176.5 338.5,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 338.6,176.5 359.6,194.4 359.6,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 359.7,194.4 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.7,185.5 384.8,185.5 384.8,185.5 384.8,185.5 384.8,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 422.3,185.5 430.0,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 430.1,185.5 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 460.8,194.4 487.9,176.5 487.9,176.5 487.9,176.5 487.9,176.5 487.9,176.5 487.9,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 488.0,176.5 511.2,185.5 511.2,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 511.3,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 538.2,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.1,185.5 562.2,185.5 562.2,185.5 562.2,185.5 562.2,185.5 562.2,185.5 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 585.7,194.4 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 611.9,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.6,185.5 636.7,185.5 636.7,185.5 636.8,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 670.6,185.5 686.5,185.5 686.5,185.5 686.5,185.5 686.5,185.5 686.5,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 686.6,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.5,185.5 735.6,185.5 735.6,185.5 735.6,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5 775.2,185.5" fill="none" stroke="var(--primary,#94452b)" stroke-width="2.2" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <line x1="638.8" y1="328.0" x2="638.8" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="117.5" y1="328.0" x2="117.5" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="443.5" y1="328.0" x2="443.5" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="559.0" y1="328.0" x2="559.0" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="720.4" y1="328.0" x2="720.4" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="200.8" y1="328.0" x2="200.8" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="230.3" y1="328.0" x2="230.3" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="426.7" y1="328.0" x2="426.7" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="558.5" y1="328.0" x2="558.5" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="298.8" y1="328.0" x2="298.8" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="201.0" y1="328.0" x2="201.0" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="616.2" y1="328.0" x2="616.2" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="462.5" y1="328.0" x2="462.5" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="644.7" y1="328.0" x2="644.7" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="442.0" y1="328.0" x2="442.0" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="529.4" y1="328.0" x2="529.4" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="81.4" y1="328.0" x2="81.4" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="581.9" y1="328.0" x2="581.9" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="215.2" y1="328.0" x2="215.2" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="516.6" y1="328.0" x2="516.6" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="440.2" y1="328.0" x2="440.2" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="798.0" y1="328.0" x2="798.0" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="95.4" y1="328.0" x2="95.4" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="754.9" y1="328.0" x2="754.9" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="252.4" y1="328.0" x2="252.4" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="349.3" y1="328.0" x2="349.3" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="431.6" y1="328.0" x2="431.6" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <line x1="566.2" y1="328.0" x2="566.2" y2="321.0" stroke="var(--error,#a64542)" stroke-width="1" opacity="0.6"/>
  <text x="794.0" y="56.7" font-size="10.5" fill="var(--primary,#94452b)" text-anchor="end" opacity="0.75" font-weight="400">alive (training / healing)</text>
  <text x="794.0" y="179.5" font-size="10.5" fill="var(--primary,#94452b)" text-anchor="end" opacity="1" font-weight="600">in each sync's quorum</text>
  <text x="68.0" y="316.0" font-size="9.5" fill="var(--error,#a64542)" text-anchor="start" opacity="0.8" font-weight="400">red ticks = kills</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">The distinction that names this milestone, measured over the full storm. The filled band is *liveness* — how many of the 32 replicas are alive and training at each moment — and it rides high, around 30, dipping only briefly when a cluster of faults lands. The bold line below it is *participation* — how many replicas actually made it into each sync's quorum — and it sits at about 16, half the cluster, the entire run. The red ticks along the bottom are the kills. The gap between the band and the line is the finding: under a steady fault rate, "alive" and "contributing to this sync" are simply different numbers. Almost everyone is up and working; only about half are phase-aligned enough at any given barrier to be counted in it, because every fault reconfigures the quorum and knocks the survivors' sync timing out of step, and at 125 faults an hour the cluster never fully re-aligns. The honest way to report a run like this is all three numbers — cluster size (32), live replicas (~30), per-sync participation (~16) — because collapsing them into one hides the tax that churn imposes on coordination.</p>

That gap — ~30 alive, ~16 participating — is the second distinction, and it's the one I'd most want a reader to take away. Under a steady fault rate, the workers stay overwhelmingly *alive*, but each fault reconfigures the quorum and jostles the survivors' sync timing out of phase, and at 125 faults an hour they never fully settle back into lockstep. So a snapshot finds most of them up and training, but only about half aligned at any given barrier. Cluster size, live count, and per-sync participation are three different numbers, and a single "healthy: 32/32" would have lied to me about all of it. (Pinning lifts the fault-free participation to 32 and the commit reliability to 97%, but under the storm, participation is governed by churn, not the scheduler — so it sits near 16 regardless.)

None of which stopped the cluster from doing its job:

<div style="overflow-x: auto; margin: 2rem 0;">
<svg viewBox="0 0 680 380" xmlns="http://www.w3.org/2000/svg" style="font-family:'Inter',sans-serif; max-width:100%; display:block; margin:0 auto; height:auto;">
  <text x="64.0" y="18.0" font-size="13" fill="currentColor" text-anchor="start" opacity="1.0" font-weight="600">Recovery latency across all 27 kills (CDF)</text>
  <line x1="64.0" y1="328.0" x2="658.0" y2="328.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="332.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">0%</text>
  <line x1="64.0" y1="254.5" x2="658.0" y2="254.5" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="258.5" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">25%</text>
  <line x1="64.0" y1="181.0" x2="658.0" y2="181.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="185.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">50%</text>
  <line x1="64.0" y1="107.5" x2="658.0" y2="107.5" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="111.5" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">75%</text>
  <line x1="64.0" y1="34.0" x2="658.0" y2="34.0" stroke="currentColor" stroke-width="0.8" opacity="0.12"/>
  <text x="56.0" y="38.0" font-size="11" fill="currentColor" text-anchor="end" opacity="0.65" font-weight="400">100%</text>
  <text transform="rotate(-90 16 181)" x="16.0" y="181.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">% of kills ≤ t</text>
  <text x="64.0" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">0</text>
  <text x="361.0" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">201</text>
  <text x="658.0" y="346.0" font-size="11" fill="currentColor" text-anchor="middle" opacity="0.7" font-weight="400">402</text>
  <text x="361.0" y="370.0" font-size="12" fill="currentColor" text-anchor="middle" opacity="0.8" font-weight="400">seconds after kill</text>
  <polyline points="64.0,328.0 72.8,328.0 72.8,317.5 76.2,317.5 76.2,307.0 77.4,307.0 77.4,296.5 77.9,296.5 77.9,286.0 79.2,286.0 79.2,275.5 95.7,275.5 95.7,265.0 101.5,265.0 101.5,254.5 118.8,254.5 118.8,244.0 126.4,244.0 126.4,233.5 129.6,233.5 129.6,223.0 131.7,223.0 131.7,212.5 134.2,212.5 134.2,202.0 138.4,202.0 138.4,191.5 142.0,191.5 142.0,181.0 155.7,181.0 155.7,170.5 160.2,170.5 160.2,160.0 162.2,160.0 162.2,149.5 169.2,149.5 169.2,139.0 173.2,139.0 173.2,128.5 174.0,128.5 174.0,118.0 178.3,118.0 178.3,107.5 214.8,107.5 214.8,97.0 224.2,97.0 224.2,86.5 225.1,86.5 225.1,76.0 232.4,76.0 232.4,65.5 239.9,65.5 239.9,55.0 317.3,55.0 317.3,44.5 356.9,44.5 356.9,34.0" fill="none" stroke="var(--error,#a64542)" stroke-width="2.2" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <polyline points="64.0,328.0 170.3,328.0 170.3,317.1 232.6,317.1 232.6,306.2 235.9,306.2 235.9,295.3 250.0,295.3 250.0,284.4 253.3,284.4 253.3,273.6 254.5,273.6 254.5,262.7 256.4,262.7 256.4,251.8 260.0,251.8 260.0,240.9 261.2,240.9 261.2,230.0 261.4,230.0 261.4,219.1 266.8,219.1 266.8,208.2 273.3,208.2 273.3,197.3 283.1,197.3 283.1,186.4 284.6,186.4 284.6,175.6 303.8,175.6 303.8,164.7 304.4,164.7 304.4,153.8 312.9,153.8 312.9,142.9 313.3,142.9 313.3,132.0 317.6,132.0 317.6,121.1 320.1,121.1 320.1,110.2 336.8,110.2 336.8,99.3 350.9,99.3 350.9,88.4 355.0,88.4 355.0,77.6 355.3,77.6 355.3,66.7 356.2,66.7 356.2,55.8 405.2,55.8 405.2,44.9 629.7,44.9 629.7,34.0" fill="none" stroke="var(--primary,#94452b)" stroke-width="2.2" opacity="1.0" stroke-linejoin="round" stroke-linecap="round"/>
  <rect x="74.0" y="40.0" width="11.0" height="11.0" rx="2" fill="var(--error,#a64542)" opacity="1"/>
  <text x="90.0" y="50.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.85" font-weight="400">T_resume — median 57s</text>
  <rect x="74.0" y="58.0" width="11.0" height="11.0" rx="2" fill="var(--primary,#94452b)" opacity="1"/>
  <text x="90.0" y="68.0" font-size="10.5" fill="currentColor" text-anchor="start" opacity="0.85" font-weight="400">T_back — median 149s</text>
</svg>
</div>
<p style="text-align: center; font-style: italic; color: var(--on-surface-variant); font-size: 0.9rem; margin-top: -0.5rem;">Recovery latency across all 27 executed kills, as cumulative distributions. The left curve, T_resume, is how long until the *survivors* commit their next sync after a kill — a median of 57 seconds, the cluster barely breaking stride. The right curve, T_back, is the harder bar: how long until the killed replica is fully relaunched, has pulled fresh state from a peer, and is committing as a member again — a median of 149 seconds, with the long tail being processes unlucky enough to be relaunching into the teeth of the oversubscribed CPU while everything else competes for the same cores. Every one of the 27 kills recovered; none required a human. Read together with the previous figure, this is the actual fault-tolerance claim: the cluster stays alive and keeps making correct progress through a fault every twenty-nine seconds, and the loss — measured separately — falls 10.8 to 4.3 straight through all of it, with zero out-of-memory deaths across the entire run.</p>

The final tally for the canonical pinned run: **97.7% step efficiency** against the matched fault-free baseline, **97% of sync attempts committing**, **all 27 kills recovered** (T_back median 149 s), the loss descending **10.8 → 4.3** monotonically through 28 kills, and **zero** out-of-memory deaths over the whole thirty minutes. *[measured / derived.]* That's the animation at the top of this post — every cell a replica, color-coded by state, the quorum and the loss tracked live, all of it reconstructed from the run's own telemetry.

### Process lessons (the stuff between the milestones)

A build log owes you the texture, not just the results. The things that actually cost time:

- **Assume the control channel is unreliable, including the one you're typing into.** My laptop-to-trainer link flapped under load and eventually died outright. Everything long-running lives in `tmux` on the worker; every transfer is an `rsync` in a retry loop; and I built a runner that retries across link flaps and falls back to a second network path. Never hand-drive a 45-minute job over a bare SSH session.
- **`pkill -f` is a loaded gun.** A pattern that matches your training command also matches the SSH session running it — I killed my own workers with diagnostic commands more than once. Scope kill patterns to a run ID and put them in a script, never the command line.
- **At scale, watch the aggregate, not the nodes.** The thing that made the cloud milestone expensive in attention was hand-inspecting individual machines. The thirty-two-node storm was only tractable because I monitored one summary — derived from the lighthouse and the telemetry — instead of thirty-two terminals.
- **Cost discipline is a habit, not an afterthought.** Every cloud experiment, including a dozen false-start instances, totaled **\$3.12**, because instances get destroyed the moment they're done and a ledger tracks every cent.

---

## The Evidence Ledger

Distributed-systems claims oversell as easily as benchmarks, so here is every load-bearing number, sorted by how I actually know it. **Measured** means read directly from a run's telemetry. **Derived** means composed from measured primitives. **Single-run** means one un-repeated result.

| Claim | Value | Evidence |
|---|---|---|
| Outer-optimizer momentum recovers through `kill -9` | **bit-identical digests**, every post-recovery sync | **measured** (M0.5, M2) |
| Single-fault recovery latency | **T_resume 5.0 s, T_rejoin 54 s** | **measured** (M2) |
| DiLoCo parity vs baseline | **+2.8% (H=25) … +9.4% (H=500)** loss | **measured** (single seed/H, outer-lr untuned) |
| Communication reduction | **exactly H-fold**, measured within a few % | **measured** (veth counters) |
| Storm step efficiency (2 replicas) | **88.2% / 85.0%** at 69 / 85 faults·hr⁻¹ | **derived** (chaos rate ÷ fault-free rate) |
| Throughput-healthy eval regression, and its fix | **2.4 → 4.0 then fixed** to monotonic | **measured** (M3, with/without checkpoints) |
| WAN throughput, DiLoCo vs per-step | **27–37k vs 5.3k→0.9k tok/s**; both DNF at 10 Mbps | **measured** (netem sweep) |
| Cross-region cloud trains as one cluster | **bit-identical digests over the internet** | **single-run** ($3.12 total cloud) |
| Connectivity ≠ coordination | solo commits, **divergent digests**, all nodes healthy | **measured** (M4) |
| N=32 step efficiency under 125 faults·hr⁻¹ | **97.7%** of fault-free; **27/27** kills recovered | **measured / derived** (M5) |
| Liveness vs participation under churn | **~30/32 alive, ~16/32 per sync** | **measured** (M5) |
| Loss through the N=32 storm | **10.8 → 4.3 monotonic**, 0 OOM | **measured** (M5) |

A caveat worth stating loudly: the convergence numbers (M0, M1) are single-seed where they'd ideally be multi-seed, and the cloud results (M4) are single runs, not distributions — they're tagged accordingly. The fault-tolerance and coordination findings — the digests, the regression, the participation gap — are the load-bearing ones, and those are measured directly off telemetry I can hand you. I also kept a separate log of the torchft friction I hit along the way (an address-binding bug that breaks recovery behind NAT, a too-short default quorum timeout, the undocumented standalone setup); those are candidate upstream contributions I intend to raise with the project after this writeup, framed as questions rather than claims.

## What I Learned

1. **Momentum survives recovery — and everything depended on that holding.** The single most consequential measurement in the project was the earliest: that the outer optimizer's momentum comes back bit-exact through a real kill, with no checkpoint. If that digest hadn't matched, every later result would have been built on sand. When you're testing a system, find the load-bearing assumption and attack it first.

2. **Measure progress, not throughput — they diverge silently, and the divergence is the research.** The eval-loss regression in M3 was invisible to every operational metric: syncs were committing, efficiency was 87%, the cluster looked perfectly healthy while the model rotted. Throughput tells you the machine is busy; it does not tell you the work is good. The most valuable thing I built was the telemetry that let me watch the model's *quality* in real time, because that's where the failure hid.

3. **Cluster size, live replicas, and per-sync participation are three different numbers.** Collapsing them into one "healthy: N/N" hides the entire cost that churn imposes on coordination. At scale, almost everyone is alive almost always — and only some of them are actually in any given sync. If you report one number, report the one that's doing the work: participation.

## What I'd Build Next

In rough order of value:

- **An alignment mechanism for the participation gap.** A short quorum-gather window, or re-phasing replicas' sync boundaries after a reconfiguration, so churn stops scattering them out of step. This is the direct lever on the ~16/32 number, and it's the most interesting open problem the project surfaced.
- **Quantized or streamed syncs, against the 10 Mbps cliff.** The control-plane starvation that kills the cluster at 10 Mbps is a bandwidth problem with a known shape: shrink the pseudo-gradient payload (it's full fp32 today) or stream it in fragments so heartbeats can interleave. The reward is collaborative training over genuinely bad links.
- **Larger N on real multi-host hardware.** M5's 32 replicas shared one desktop's cores, which is what produced the oversubscription story. The same harness across several physical machines would separate the coordination findings from the scheduling artifact and push toward torchft's actual scale.
- **The torchft friction, upstream.** The address-binding bug, the quorum-timeout default, the standalone-setup gap — written up as issues and, where I have a clean fix, pull requests. After this post, and as questions first.
- **Independent reproduction.** The repo is built for it — every run writes the JSONL these figures are reconstructed from, and the de-risk ladder is scripted. The real bar is someone else killing their own nodes and reading the same recovered digests.

## Acknowledgments

This stands on specific prior work: the **DiLoCo** paper that made low-communication training a real algorithm, and Meta's **torchft**, which turned it into running fault-tolerant infrastructure and whose [issue #171](https://github.com/meta-pytorch/torchft/issues/171) framed the question I spent six milestones answering. The model is a nanoGPT-class network trained on **TinyStories**; the animations and live dashboards lean on **Rich**, **plotext**, **asciinema**, and **agg**; the WAN realism comes from Linux `tc/netem` and **Tailscale**; the cloud GPUs were rented from **Vast.ai** for the price of a sandwich.

The full source — every milestone's run data, the chaos harness, the telemetry pipeline these figures are built from, and the dated build log every anecdote here came from — is on [GitHub](https://github.com/Neumann-Labs/ft-diloco). If you'd argue with any number in this post, the JSONL is right there to argue with. That's how the next iteration gets honest.
