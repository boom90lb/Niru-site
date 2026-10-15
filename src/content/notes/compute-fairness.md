---
title: "Compute fairness is a control contract"
description: "What a comparison must declare before its ranking means anything, and how matched compute, pairing and bounded promotion serve a comparison of architectures under one recipe."
eyebrow: "Method note / 01"
published: 2026-08-16
updated: 2026-10-14
status: "Method note"
---

An architecture search does two things at once. It compares candidates, and
it decides how much evidence each candidate may collect. The second decision
shapes the first. How compute is allocated is therefore part of the method,
not a detail of the implementation.

## The comparison problem

A ranking is a claim about candidates. It is only that claim if nothing else
differed. Suppose one candidate trained a little longer than another, or drew
a kinder learning rate, or met an easier threshold on its way up. The ranking now measures the search as much as the candidates, and nothing in the final table says so. Published architecture search has been caught this way. [Li and Talwalkar](https://arxiv.org/abs/1902.07638) found random search with early stopping competitive with leading methods on standard benchmarks. [Yang, Esperança and Carlucci](https://arxiv.org/abs/1912.12522) found that the training protocol accounted for more of the reported gains than the searched architectures did. [Lindauer and Hutter](https://arxiv.org/abs/1909.02453) set out a checklist so that results can be compared at all.

## Three kinds of comparison

What a ranking can say depends on what the comparison held fixed. AMLO's
design names three kinds, and each answers a different question.

- **A frozen-recipe comparison** changes the architecture and keeps one
  training recipe fixed, with matched training compute. It asks whether this
  architecture is better under this recipe. The examples on this site are of
  this kind.
- **An architecture with its own tuning** gives each architecture a recipe
  tuned by a declared procedure, and it charges the tuning work to the
  comparison. It asks whether this architecture, with the tuning it needs,
  is better.
- **A deployed-system comparison** compares a model with its tools,
  retrieval, verifiers and inference budget. Its resources include inference
  tokens, tool calls and latency. It asks which system gives the most quality
  for what it spends.

Equal parameters, equal tokens, equal training compute and equal tuning
effort are different contracts. A comparison states which one it uses, and
a tuned system is not reported as an effect of its architecture alone.

For a frozen-recipe comparison, the design sets four conditions.

1. The compute is matched, and the match is enforced.
2. The randomness and the evaluation data are shared.
3. The promotion rule is statistical and was stated in advance.
4. The decision record is complete.

## Matched compute

Each evaluation tier has one compute target, $C$, in floating-point
operations. For a candidate that spends $M$ operations on each training token,
with batch size $B$ and sequence length $L$, the planner sets the step count as follows.

$$\text{steps} = \left\lfloor \frac{C}{M \cdot B \cdot L} \right\rfloor$$

The planner refuses a plan whose step count is zero, or that misses the
target by more than 1%, or whose estimated memory use exceeds 90% of the
device. In the design, the step count alone stops the training loop, because a
budget that only records compute is not a budget. The receipt reports planned
and measured compute separately. The current code already sets the step count
from the plan. It cannot yet measure the compute in each step, so its receipts record counted compute, which is the steps run times the planned compute per token.

Resources are a vector, not a scalar. Floating-point operations, tokens and
memory each have their own target and their own check, and no conversion
between them is assumed.

## Paired comparison

Every candidate in a tier trains from the same seed and is scored on the same
examples in the same order. The reason is a line from first-year probability.

$$\operatorname{var}(x - y) = \operatorname{var}(x) + \operatorname{var}(y) - 2 \operatorname{cov}(x, y)$$

Two losses measured on different draws are independent, and their difference carries both noises in full. Two losses measured on the same draw share much of their noise, the covariance is large, and the difference cancels what is shared. Pairing costs no extra evaluations; it only analyzes the same ones together.

Pairing also narrows what the interval covers, to these trained candidates on these examples. A claim about new training runs needs independent runs as
its units, and an equal integer seed does not by itself prove that two runs
are coupled. A later tier changes the scale, the budget and the data
together, so it makes a new comparison, not a repeat of the old one.

## Bounded promotion

Once the search has a current best, the gate takes the per-example
difference in loss between each candidate and that best, resamples the differences at least 2,000 times, and forms a bias-corrected and accelerated (BCa) bootstrap interval for the mean. The bootstrap seed is a hash of the evidence, so the same evidence
always gives the same verdict.

A candidate advances only when the lower bound of its interval is above
zero and its confidence is certified. The share of resamples at or below zero
is only an estimate of the bootstrap's tail. The gate passes when an exact
binomial bound on that tail is within the threshold, and refuses when the
bound shows the tail is above it. In between, it draws more resamples, up to
128,000, and if the tail is still unresolved it refuses. The Monte Carlo error
of this rule is at most 1% per comparison. This guards against an old trap. The best of many noisy scores is biased upward, because the noise helped choose it. The gate does not ask
which candidate scored highest. It asks which candidates are shown to be
better. In the design, a Benjamini–Hochberg correction over every candidate
tested at a tier holds the expected fraction of false promotions. The
current code applies it only to the candidates that passed their own gate,
which does not yet give that guarantee. At Tier 1, a
candidate with fewer than 20 evaluated examples does not advance, and an interval that crosses zero is recorded as inconclusive, not as a loss.

## Evidence that steers is not evidence that confirms

Every look at a held-out set spends a little of its validity. A set that has ranked a thousand candidates has, in a small way, been fit to; [Dwork and colleagues](https://doi.org/10.1126/science.aaa9375) made that cost precise and showed how to bound it. The direction of
the program is to separate the two uses. The evidence that steers the search
stays visible to it, and the evidence that confirms a result is sealed,
opened only after the selection is frozen, and held by an Auditor that can
deny a claim but cannot choose the next candidate. The Auditor keeps custody
on every path, and how it validates an outcome depends on the outcome. It may use a held-out metric fixed with the analysis, a trusted formal check, or an
independent judge for a broad measure of capability.

## Why start on one device

A reference path on one device has fewer moving parts, so the accounting
contract can be made to hold in a setting where every term is visible. The
next step is not more devices. It is a trained candidate inside the search,
with measured compute and checked recovery, and then one credible comparison
from end to end. Distributed training comes when a study needs it. The same
contract must then hold across many devices, with their topology,
concurrency, failures and wall-clock effects, and the receipts must say
where each term was measured.
