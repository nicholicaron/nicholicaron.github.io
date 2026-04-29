---
layout: post
title: "CHOP: A Research Diary in Learning to Branch"
date: 2026-04-28 17:00:00 -0400
tags: [Optimization, GNNs, Reinforcement Learning, AI, MILP]
cover_image: /assets/images/chop/benchmark_all.png
---

In Part 1 we ended on a question: when the textbook node-selection heuristic — best-bound — explores roughly twice as many nodes as the alternatives on Set Cover, can we *learn* a heuristic that beats all of them? This post is the story of trying. It's a research diary in the literal sense — eight architectures, three trainers, four ablations, two genuine wins, and several honest losses, all assembled on a laptop CPU over the course of a single research push. Code is on [GitHub](https://github.com/nicholicaron/chop).

If you haven't read [Part 1]({{ site.url }}/2026/04/28/branch-and-bound-from-first-principles.html), I'd start there — it explains MILP, branch-and-bound, and why the question is interesting in the first place.

## What This Post Covers

- **The CHOP environment** — how branch-and-bound becomes a Gymnasium-style RL env, and the action-design bug that nearly sank the project
- **The architecture progression** — MLP → GCN over the B&B tree → Tree-GNN → Bipartite-GCN (Gasse 2019) → Bipartite-GCN with cross-candidate attention. What each can see, what each learned
- **The training stack** — REINFORCE vs PPO, single-task vs multi-task, imitation warm-start
- **Honest negative results** — the naive ensemble that hurt, the hybrid encoder that added parameters but no signal, and the longer training run that overfit
- **The robustness finding that changed how I read every other number in this post** — and the meta-lesson on single-seed evaluation
- **What I'd build next, given more compute**

The complete leaderboard, for those who'd like to skip to the punchline:

![Final benchmark: all 16 approaches on SetCover(50e × 80s d=0.10), n=40 held-out](/assets/images/chop/benchmark_all.png)

The top seven bars are all learned policies. Depth-first — the strongest classical heuristic on this distribution — is eighth. Best-bound is dead last (rightmost gray bar).

---

## The Problem in RL Form

A branch-and-bound solver makes a sequence of decisions. Modeled as a Markov decision process, each decision is an *action*; the *state* is the current shape of the search tree, the LP solutions, and the incumbent; the *reward* is some signal about whether you're solving the problem efficiently. For node selection, the action set at each step is "the open frontier" — the list of sub-problems that haven't been expanded yet.

That's the picture. Implementing it as a real Gymnasium env is where the work is. A few design questions answer themselves quickly:

- **Reward.** I went with `-1 per node expanded`, plus a small `+5` bonus when a new incumbent is found, plus a `+50` terminal bonus when the queue empties (the problem is proven optimal). The MDP wants to *minimize* nodes; this signal pushes that.
- **Action space type.** Variable-size action spaces are a pain for standard RL libraries. The open frontier can have hundreds of nodes. So I went with a fixed top-K formulation: at each step the env exposes the *K=16 best-LP-bound candidates* and the agent emits 16 real-valued scores. Argmax picks which candidate to expand. Padded slots get a separate `is_real` flag and are masked out before softmax.
- **Episode length.** One episode = one MILP instance. Episode terminates when the open queue empties (problem solved) or when a step/time budget is hit (truncated).

What turned out to be subtle was the *causal* link between the action and the search.

### The Bug That Broke Everything First

The original env wrapped the solver's internal priority heap and tried to inject the agent's action as a *perturbation* to every queued node's priority. The thinking was: best-bound is just `priority = LP value`, so we'd let the agent add or subtract some learned amount from each LP value. The action was a single scalar.

The thinking was wrong. **Adding a uniform constant to every priority does not change the heap's order.** If you bump every node's priority by `+0.7`, the same node still pops first. The action was a no-op. The agent's policy, no matter what it learned, had zero causal effect on which node B&B expanded.

I didn't catch this immediately because the env *runs*. It returns observations, rewards, episode endings, all the right shapes. The bug was that the agent's outputs flowed into the env and dissipated. Several training runs later I noticed that "best-bound agent", "depth-first agent", and "random agent" all produced *identical* node counts. That shouldn't be possible if the action mattered.

Once I knew, the fix was an env redesign: the env now manages the open node list itself (sidestepping the solver's heap entirely) and the action vector directly chooses which candidate to pop. The original test that caught this is now the first regression test in the repo:

```python
def test_rl_env_action_actually_changes_search():
    """Different heuristic agents MUST produce different node counts.
    This test would have caught the original "action is a no-op" bug."""
    counts = {}
    for mode in ("best_bound", "breadth_first", "random"):
        ns = []
        for ep in range(3):
            env = _knapsack_factory(seed=100 + ep)
            ...  # run agent for one episode
            ns.append(info["nodes_explored"])
        counts[mode] = float(np.mean(ns))

    bb, bf, rd = counts["best_bound"], counts["breadth_first"], counts["random"]
    assert bb < bf, f"best_bound ({bb}) should beat breadth_first ({bf})"
    assert bb < rd, f"best_bound ({bb}) should beat random ({rd})"
```

A boring assertion, but the kind of boring that saves you a week of confusion. The lesson is older than RL — *test your causal interfaces, not just your output shapes* — but the urgency increases when the symptom of a broken interface is "training proceeds normally and produces nothing."

### Per-candidate features

For each of the K=16 candidates exposed at every step, the env emits seven scalar features. The exact list is in the codebase, but the gist:

| Feature       | Meaning                                                   |
|---------------|-----------------------------------------------------------|
| `rel_bound`   | LP value of this candidate, normalized by the root LP value |
| `depth_norm`  | Tree depth of this candidate, capped and rescaled         |
| `frac_share`  | Fraction of variables still fractional in this LP         |
| `best_frac`   | Closest-to-0.5 fractionality among any var (the "ripest" candidate for branching) |
| `gap`         | Normalized gap between this candidate's LP and the incumbent |
| `can_improve` | 1 if this candidate's bound exceeds the incumbent, else 0 |
| `is_real`     | Mask: 1 for a real candidate, 0 for a padded slot         |

Plus six global features: step count, queue size, normalized incumbent gap, whether an incumbent exists yet, total nodes created, and elapsed wall-clock fraction. The full observation is `Box(K * 7 + 6,) = Box(118,)` — a fixed-shape vector that any RL library can consume.

This is what classical heuristics get to work with too. Best-bound looks at `rel_bound` and picks the highest. Depth-first looks at `depth_norm` and picks the highest. The MLP policy in the next section starts from the same input.

---

## Architecture 1 — MLP

The simplest thing that could work: a 2-layer MLP (118 → 64 → 64 → K=16) with `tanh` activations and a softmax output, trained with REINFORCE-with-baseline. It scores the K candidates independently and picks the argmax (with a Boltzmann sample at training time for exploration).

Trained on `Knapsack(n_items=25, medium difficulty)` for 800 episodes — about 50 seconds on a laptop CPU. Held-out evaluation on 40 fresh instances:

| Policy           | Nodes (mean ± std) | vs. learned |
|------------------|--------------------|-------------|
| **Learned (MLP)** | **66.7 ± 52.9**   | 1.00x       |
| BestBound        | 66.7 ± 52.9        | 1.00x       |
| DepthFirst       | 86.1 ± 61.3        | 1.29x       |
| BreadthFirst     | 250.3 ± 69.2       | 3.75x       |
| Random           | 207.1 ± 60.2       | 3.10x       |

Identical to BestBound. The means agree to one decimal; the standard deviations agree exactly; even the per-instance trajectories appear to coincide. **The policy converged to the same node ordering as BestBound on every test instance.** This is not luck — the gradient signal pulled it toward the highest-`rel_bound` candidate, which is exactly best-bound's choice.

That is, on Knapsack, best-bound is essentially optimal. The integrality gap is small, the LP guides B&B faithfully, and there's no surplus of nodes for a learned policy to save. REINFORCE picked up on this and recovered the right behavior from scratch in a minute of CPU.

A nice sanity check, but not the win we want. The win we want is on a problem where best-bound is *bad*.

### A small generalization aside

The same trained policy generalizes across problem sizes it was never trained on:

![Same policy, evaluated on Knapsack instances of different sizes](/assets/images/chop/generalization_across_sizes.png)

Across `n_items ∈ {15, 20, 25, 30}`, the learned policy and BestBound agree to within 0.1 nodes on every size — the purple "Learned" line is hidden directly underneath the green "BestBound" line. A nice negative space — the policy has learned a *function*, not memorized the training distribution.

---

## Architecture 2 — GCN over the B&B Tree

The B&B enumeration tree is, naturally, a graph. Each node carries features (depth, bound, fractionality flags); each edge represents a parent-child branching decision. A graph convolution lets the policy aggregate information from a candidate's neighborhood — its parent, siblings, ancestors — when scoring it.

I built a basic two-layer GCN over the tree, scoring the K candidate nodes by their final embeddings. Trained on `SetCover(50e × 80s, density 0.10)` — the regime from Part 1 where best-bound is provably weak.

Stochastic-mode evaluation (sampling from the softmax): **12.3 ± 10.3 nodes vs best-bound's 19.1.** A 1.55x improvement.

Deterministic-mode evaluation (taking the argmax): **19.1 ± 16.1 — exactly matching best-bound's mean and standard deviation.** Suspicious. Identical std to three significant figures suggests the policies were taking *the exact same trajectory* on every test instance.

Diagnosis took about ten minutes. I dumped the policy's per-step scores on a held-out instance:

```
Step 1: queue=2, n_real=2
  scores: [0.0768, 0.0772]
  argmax = 1
Step 2: queue=3, n_real=3
  scores: [0.0522, 0.1562, 0.1566]
  argmax = 2
```

The scores at the top are nearly tied. The GCN had learned that all the highly-LP-bounded candidates are roughly equivalent, but `argmax` always picks the index of the maximum, and PyTorch's `argmax` returns the *first* tied index. The first-by-LP-bound index is exactly best-bound's choice. The collapse was a tiebreak artifact, not a representation failure.

The fix was four lines. At eval time, instead of pure argmax, sample from a Boltzmann distribution at low temperature ($T = 0.05$):

```python
if deterministic:
    cool_logits = full_logits / 0.05
    cool_probs = F.softmax(cool_logits, dim=-1)
    choice = torch.distributions.Categorical(probs=cool_probs).sample()
```

This concentrates the distribution sharply on the genuine top scorers but breaks ties by sampling. After the fix, deterministic eval came back at **14.0 ± 10.8 — beating best-bound 1.36x.**

The GCN-over-the-tree was never a strong architecture (the tree topology underspecifies what the policy actually needs to know), but the tiebreak bug taught me something more general: any time a learned policy's deterministic evaluation looks suspiciously identical to a classical baseline's, suspect tiebreaking before suspecting representation failure.

---

## Architecture 3 — Tree-GNN with Bottom-Up Message Passing

The 2024 paper [*Reinforcement Learning for Node Selection in Branch-and-Bound*](https://arxiv.org/html/2310.00112v2) pointed at a more principled way to read the tree: K iterations of *bottom-up message passing*, where each parent node aggregates its children's embeddings:

$$h_{t+1}(\text{parent}) = h_t(\text{parent}) + \text{emb}\!\left(\frac{1}{|\text{children}|} \sum_{c \in \text{children}} h_t(c)\right)$$

After $K$ iterations, each node's embedding summarizes its $K$-deep subtree. Per-node features include a *histogram of fractional-variable parts* (10 buckets) — a constant-size summary of the LP solution at that node, regardless of problem dimension.

I implemented this directly. Stochastic eval: **9.7 ± 8.1 nodes (1.97x best-bound).** Deterministic: **10.7 ± 9.2 (1.79x).** Both clear wins. The first time a learned policy meaningfully beat best-bound on this distribution.

What's interesting is *what the tree-GNN sees*. The bottom-up aggregation means that scoring a candidate node propagates information from the entire subtree below it — the open frontier under that candidate, the integer-feasible leaves found there, the structure of past branching decisions. Best-bound sees only the candidate's own LP value. The tree-GNN sees its *neighborhood*.

---

## Architecture 4 — Bipartite-GCN over the LP

The canonical "GNN for B&B" architecture is from the 2019 NeurIPS paper [*Exact Combinatorial Optimization with Graph Convolutional Neural Networks*](https://arxiv.org/abs/1906.01629) by Maxime Gasse and collaborators. They were doing *variable selection* (branching), not node selection, but the encoder applies almost without modification.

The idea is to read the *LP problem* itself as a graph. For each candidate B&B node, build a bipartite graph:

- One side: ILP variables, with features like the LP value at this candidate, fractionality, objective coefficient
- Other side: constraints, with features like RHS, slack, binding flag
- Edges: variable $j$ connects to constraint $i$ whenever $A_{ij} \neq 0$, with edge feature equal to the (normalized) coefficient

Two interleaved half-convolutions:

$$\mathbf{c}_i \leftarrow f_C\!\left(\mathbf{c}_i,\; \sum_{(i,j) \in \mathcal{E}} g_C(\mathbf{c}_i, \mathbf{v}_j, \mathbf{e}_{i,j})\right)$$

$$\mathbf{v}_j \leftarrow f_V\!\left(\mathbf{v}_j,\; \sum_{(i,j) \in \mathcal{E}} g_V(\mathbf{c}_i^{\text{new}}, \mathbf{v}_j, \mathbf{e}_{i,j})\right)$$

where $f_*$ and $g_*$ are 2-layer perceptrons with ReLU. The first pass aggregates variable info into constraints; the second pass aggregates the updated constraint embeddings back into variables.

For our task, after the convolution we mean-pool the per-variable embeddings to a single per-candidate vector, concat a couple of scalar features (LP value, depth), and project to a score. K candidates means K bipartite-graph forward passes per env step.

Critically, Gasse identifies one architectural trick as essential: **prenorm**. Instead of the usual batch-norm-after-update, they apply an empirical-stats normalization *after the message sum, before the update MLP*. The paper attributes most of the cross-instance generalization to this. I implemented it as a `BatchNorm1d` in the same position. The first run already worked.

**Bipartite-GCN result on n=40 held-out: 9.3 ± 7.4 nodes — 2.05x better than best-bound.** New top of the leaderboard at the time. With the same REINFORCE trainer, on the same instance distribution, just by switching the encoder. The architecture matters more than the training time — earlier, longer training of the MLP (1500 episodes vs 600) had actually gotten *worse* (13.9 vs 10.8). But switching to the LP-aware encoder cut nodes by 14% on the same training budget.

The intuition is clean: the LP graph captures *problem structure* — which variables are intertwined through which constraints. A candidate node whose constraints have a small "neighborhood" of fractional variables can probably be resolved quickly; one whose constraints sprawl across many fractional variables won't. Best-bound, comparing only LP values, is blind to this structural difference. The bipartite GCN learns to read it.

---

## Architecture 5 — Bipartite-GCN with Cross-Candidate Attention

Two thoughts about the bipartite GCN bothered me:

1. The score head projects each candidate's embedding to a scalar *independently*. But ranking is a comparative task — to score candidate $i$ well, you should know what candidates $j \neq i$ look like.
2. Pointer-network and learning-to-rank literature handles this with *self-attention over the candidate set*. Each token attends to the others; the score depends on the others; the inductive bias matches.

I added a 2-layer transformer encoder on top of the bipartite GCN: stack the K bipartite-GCN per-candidate vectors as tokens, prepend a global-features token, run self-attention with a padding-aware mask, project each candidate token to a score.

**First training run (seed=0, 400 episodes): 8.9 ± 6.4 nodes — 2.15x better than best-bound.** New champion. The mean improvement over the bare bipartite GCN (9.3) is small (~4%), but the standard deviation also tightened from 7.4 to 6.4, suggesting the policy's rankings were not just slightly better on average but *more consistent*.

This is the architecture that ships in the repo as the headline. But you should hold on a paragraph before believing it.

---

## What Didn't Work

The honest part. Three architectures and several training tricks I tried that *didn't* improve over the baselines — sometimes by a clear margin.

### Failure 1 — Naive Ensemble

Hypothesis: the bipartite GCN reads LP structure; the tree-GNN reads search-tree structure. The signals are orthogonal. An ensemble that averages their action votes should do better than either alone.

Implementation: take the two best individual checkpoints, evaluate them on the same instance, average their action vectors (which are `+1.0` in the chosen slot, `-1.0` elsewhere), pick the argmax of the average. With weight $1.0$ each, you get a soft majority vote.

Result on n=30 with the same eval seeds:

| Policy | Nodes (mean ± std) |
|--------|--------------------|
| Bipartite-GCN alone | 10.4 ± 8.0 |
| Tree-GNN alone | 11.2 ± 9.5 |
| Ensemble (1:1) | 10.97 ± 9.50 — *worse than alone* |
| Ensemble (2:1, 3:1, 5:1 favoring Bipartite) | 10.37 ± 8.04 — *recovers Bipartite-alone* |

A soft 1:1 vote *hurts* the better member. Skewing the weight toward Bipartite-GCN doesn't help; it just recovers what Bipartite-GCN was already doing. Conclusion: **the two architectures make highly correlated mistakes** on this distribution. Averaging doesn't add information; it just dilutes the better signal with the marginally worse one. The orthogonality I assumed wasn't real.

### Failure 2 — Hybrid Joint Encoder

If the naive ensemble fails because the architectures were trained *independently* and learned similar features, maybe joint training would force them to specialize on complementary signals?

Implementation: a `HybridGNNPolicy` that runs both the Bipartite-GCN encoder and the Tree-GNN encoder in the forward pass, concatenates their per-candidate embeddings, and feeds the concatenation to a shared score head. End-to-end REINFORCE training, same hyperparameters as the individual runs.

Result: **10.6 ± 10.4 nodes — slightly worse than the bare Bipartite-GCN (9.3).**

The shared score head presumably learned to weight the bipartite component higher and the tree component lower, but the extra parameters added noise rather than signal. Joint training didn't fix the correlated-mistake problem; it produced a model with more parameters and the same effective representation.

### Failure 3 — Imitation + REINFORCE Fine-Tune

A natural recipe from the literature: distill the strongest classical heuristic (best-bound) into the policy with cross-entropy supervised learning, then fine-tune with RL to surpass it.

Implementation: collect 100 best-bound rollouts on training instances, record `(observation, choice)` pairs, train the MLP policy with cross-entropy. Then switch to REINFORCE for 400 more episodes.

Post-imitation eval: **18.2 ± 14.2 nodes ≈ best-bound's 19.1.** Good — the policy learned to imitate the expert.

Post-RL fine-tune eval: **17.7 ± 13.7.** Marginal improvement but the per-episode trace during training shows the policy *destabilizing* — it spent dozens of episodes in the 18-22 node range before the fine-tune ended. REINFORCE with on-policy data and no clipping is high-variance, and starting from a near-optimal policy makes that variance more dangerous than helpful. PPO fine-tune (with its trust region) would almost certainly be the right algorithm here, but I didn't have the budget to retest.

### Failure 4 — Long Training of the MLP

If training for 600 episodes gets the MLP to 10.8 nodes, surely training for 1500 episodes gets it lower? It does not. The 1500-episode run came in at **13.9 ± 8.8** — about 30% *worse* than the 600-episode run, on the same eval seeds. REINFORCE on a fixed problem distribution can drift away from a good policy if the learning rate isn't decayed; the gradient is high-variance and will happily unlearn what it learned.

### Failure 5 — PPO

PPO has a clipped surrogate loss, GAE for variance reduction, and minibatch updates. On longer-horizon RL problems it routinely beats REINFORCE. Surely it'd help here?

It did not. **PPO+MLP came in at 16.2 ± 11.3** — better than best-bound (1.18x) but worse than every single REINFORCE-trained architecture. My read: this regime has very short episodes (~10 nodes per episode), so PPO's sample-efficiency edge — which comes from extracting more updates per rollout — has nothing to bite on. PPO also adds value-function variance that in this regime hurts more than it helps.

---

## The Robustness Finding

After the cross-candidate attention architecture took the headline at 8.9 nodes, I retrained it with a different seed (seed=7 instead of seed=0) and a longer schedule (800 episodes instead of 400). The plan was to confirm that the 2.15x improvement was robust.

**Result: 9.7 ± 7.9 nodes (1.97x better than best-bound).** Worse than the first run by a clear margin — and worse than the bare Bipartite-GCN's 9.3 ± 7.4. The "champion" architecture, on a different training seed, was no longer the champion.

What happened: **single-seed training-time noise on this benchmark is comparable to the architectural differences between policies in the top tier.** The instance-level variance is around ± 8 nodes on means of ~10. Two different training seeds explore different parts of policy space; either can land in a slightly better or worse local minimum. With $n=40$ test instances and standard error roughly $\sigma/\sqrt{n} \approx 1.2$, the apparent difference between 8.9 and 9.3 is well within noise.

I rewrote the README to report both numbers, with the caveat:

> The headline "2.15x best_bound" for Bipartite-Attn is real on that seed, but it's not a robust improvement over Bipartite-GCN's 2.05x. To claim that confidently, you'd need multi-seed evaluation (mean-of-means over 5+ training runs).

The finding is *more* publishable than the original headline. It's a concrete empirical version of advice that's easy to ignore: report multi-seed runs, not single-seed best-of. The single-seed best-of was tempting because the training run was fast enough to repeat with different seeds in minutes, but I'd already moved on. The right move would have been to do five training runs from the start and report the median.

This is the kind of thing that decides whether a "2.15x improvement" survives peer review. It is also the kind of thing every junior researcher does once and then internalizes for life. Consider the lesson internalized.

---

## The Final Leaderboard

Comprehensive evaluation on 40 held-out SetCover(50e × 80s, d=0.10) instances:

| Rank | Approach                  | Nodes (mean ± std) | vs. best_bound |
|-----:|---------------------------|--------------------|----------------|
| 1    | Bipartite-GCN + self-attention (seed=0) | **8.9 ± 6.4** | **2.15x** |
| 1b   | Bipartite-GCN + self-attention (seed=7, 800 ep) | 9.7 ± 7.9 | 1.97x |
| 2    | Bipartite-GCN             | 9.3 ± 7.4          | 2.05x         |
| 3    | Tree-GNN (stochastic)     | 9.3 ± 7.3          | 2.05x         |
| 4    | Multi-task MLP            | 10.0 ± 8.9         | 1.91x         |
| 5    | Hybrid (Bipartite+Tree)   | 10.6 ± 10.4        | 1.80x         |
| 6    | Tree-GNN (deterministic)  | 10.7 ± 9.2         | 1.79x         |
| 7    | REINFORCE + MLP           | 10.8 ± 9.0         | 1.77x         |
| 8    | depth_first (heuristic)   | 10.9 ± 10.5        | 1.75x         |
| 9    | breadth_first (heuristic) | 11.7 ± 9.8         | 1.63x         |
| 10   | random (heuristic)        | 13.2 ± 9.8         | 1.45x         |
| 11   | GCN over B&B tree (stoch) | 13.3 ± 9.8         | 1.43x         |
| 12   | REINFORCE + MLP-long      | 13.9 ± 8.8         | 1.37x         |
| 13   | GCN over B&B tree (det)   | 14.0 ± 10.8        | 1.36x         |
| 14   | PPO + MLP                 | 16.2 ± 11.3        | 1.18x         |
| 15   | Imitation + RL + MLP      | 18.8 ± 15.9        | 1.02x         |
| 16   | best_bound (heuristic)    | 19.1 ± 16.1        | 1.00x         |

The honest "stable best" is the bare Bipartite-GCN (Gasse architecture, faithfully reimplemented from the 2019 NeurIPS paper), at **2.05x best_bound** on 400 episodes of CPU training. That is the number I'd put in a paper.

A few honest ties:

- **Tree-GNN stochastic (9.3) ties Bipartite-GCN (9.3) on the same seed.** Both architectures look at very different signals — one reads the LP, one reads the search tree — and both arrive at the same node count.
- **Multi-task MLP (10.0) beats single-task MLP (10.8) with a less expressive architecture.** This is a different kind of generalization: train one model on a 50/50 mix of Knapsack and Set Cover, evaluate on each separately. Specialist with same architecture got 10.8; generalist got 10.0 *and* matches best-bound on Knapsack.

The five top performers are all learned policies. The strongest classical heuristic (depth-first, on this distribution) ranks eighth.

---

## What I Learned, As a System

A few things I'd do differently from the start, in order of regret:

1. **Test the action's causal effect first.** Before training anything, write a regression test that asserts different agents produce different node counts. The "uniform constant on every priority is a no-op" bug burned a week of confused training runs.

2. **Multi-seed by default.** I treated single-seed runs as the unit of "experiment" because training was fast. The robustness retrain showed how much that cost in interpretive confidence. Three seeds per architecture would have cost an extra ~10 minutes total and made every claim more honest.

3. **Architecture beats training time within bounds.** Long-training the MLP made it *worse*. Switching to a structurally-aware encoder (bipartite GCN) made it better. On a fixed budget, spend the marginal compute on a better architecture before spending it on more episodes.

4. **Negative results are research output too.** The naive ensemble didn't work because the architectures' errors were correlated. The hybrid encoder didn't work because shared-head pooling can't manufacture orthogonality from non-orthogonal signals. Those facts are worth knowing — both for anyone tempted to try them, and for understanding the limits of the architectures we *do* have.

5. **Report the variance.** Means on benchmarks like this can swing several percent on a single random seed. The honest comparison is mean-of-means with standard errors. The fast-and-loose comparison is the "champion" you happened to get.

---

## What I'd Build Next

A short list, in rough order of expected return on compute:

- **Multi-seed retraining of the top 4 architectures.** Run each architecture 5+ times with different seeds; report the median and the IQR, not just the best. This is the cheapest, highest-priority next experiment.
- **Tree-MDP credit assignment** ([Scavuzzo et al. 2022](https://arxiv.org/abs/2205.11107)). The standard temporal MDP gives every node-selection action equal weight in the gradient. The tree MDP formulation assigns credit *down the path* the action led to, which empirically improves sample efficiency. Plumbing it into our REINFORCE trainer is a few hundred lines.
- **Bipartite-GCN with batched per-candidate evaluation, then multi-task training.** Combining the two best ideas (Bipartite encoder + multi-task) didn't converge in our budget because each Knapsack episode runs 16 bipartite-GCN forward passes per step. Batching all K candidates into one PyG `Batch.from_data_list` call would cut per-step compute roughly 5x, making multi-task tractable.
- **PPO fine-tune from imitation warm-start.** REINFORCE fine-tune destabilized the imitation-trained policy. PPO's trust region should fix that. We have both ingredients in the repo; just haven't connected them.
- **Strong-branching imitation for the variable-selection step.** We currently use `MostFractional` as the branching rule. Strong branching is the gold standard but expensive; Gasse showed it can be imitated with a single bipartite GCN forward pass. Would let us learn *both* node selection and variable selection.
- **Bigger problem sizes.** Everything here is at SetCover(50e × 80s). At SetCover(100e × 200s), the learned-vs-classical gap should widen, and PPO's edge should start to materialize. Also opens the door to MIPLIB benchmarks.

---

## Acknowledgments

CHOP started as a college research project under Dr. Misha Lavrov at Kennesaw State, who patiently tolerated the early iterations of this codebase when it could barely solve a 5-variable LP. The 2019 NeurIPS paper from Gasse, Chetelat, Ferroni, Charlin, and Lodi is the technical north star — most of what works in CHOP is a faithful re-implementation of architectural ideas they pioneered. The 2024 paper on RL for node selection (arxiv 2310.00112) clarified the right way to think about per-node features and tree-structured GNNs. The [SCIP solver](https://www.scipopt.org/) and the [Ecole library](https://www.ecole.ai/) are the open-source pillars that make any of this approachable.

The full code is on [GitHub](https://github.com/nicholicaron/chop), under MIT license, with a README that should let any reader reproduce the headline numbers in under five minutes.

If you read this far, thanks. If you'd argue with anything in this post, please open an issue on the repo and I will read it carefully — that, more than the experiments, is how the next iteration of these ideas gets better.
