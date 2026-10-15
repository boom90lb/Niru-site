---
title: "Spectral diagnostics in a toy model"
description: "An exploratory toy study of learning-rate transfer. Its hidden-layer averages show why a useful diagnostic is still short of a transfer certificate."
eyebrow: "Measured study / 03"
published: 2026-10-10
updated: 2026-10-14
status: "Measured study"
---

A cheap proxy is useful only when what it teaches survives the move to a larger model.
This exploratory study asks what a few measurements can tell us about that move, and what
they leave out. It sweeps learning rates in three parameterizations of one small classifier.
The central lesson is about the check. An average of hidden layers can look stable while the
input or output layer moves outside the same tolerance band.

These are measured NumPy runs, separate from AMLO's model construction and training path.
They do not qualify transfer for AMLO, establish an advantage for one parameterization, or
replicate the transformer experiments in the u-μP paper.

## What ran

A fixed random teacher network classifies strings of eight tokens into ten classes. The student
has an embedding table, two square ReLU hidden matrices of width $n$ and a readout. It trains
with Adam for 1,000 steps on fresh token strings at each step. The loss is the cross-entropy on
8,192 held-out strings after the last step.

The sweep has 1,095 training runs, over three parameterizations, five widths from 64 to 1024, 23 to
27 learning rates for each parameterization and three seeds. The training uses NumPy on the CPU
in float32. The operator norms come from float64 singular value decompositions.

## Three parameterizations

- **SP** is the standard parameterization.
- **u-μP rules in this toy** use the per-layer factors in
  [Blake et al. (2024)](https://arxiv.org/abs/2407.17465), Table 2. The input-layer learning-rate factor is
  $1/\sqrt{\text{fan-out}}$.
- **The input-rate variant** is the same unit-scaled model with μP's width-independent learning
  rate for the input layer. It was added to the sweep after the first trial runs showed the best
  learning rate of u-μP moving with width. It is an exploratory, post-inspection choice,
  not a preregistered comparison or a confirmed improvement.

## What the runs show

$\eta^{\ast}$ is the best learning rate at width 64, from a parabola through the lowest point of
the mean loss and its two neighbours.

- **SP.** At width 1024, the best learning rate is $\eta^{\ast}/11$. At the rate tuned at width
  64, the operator norms of $W$ and $\Delta W$ leave their band from width 128. The activation
  RMS leaves its band at width 1024.
- **u-μP rules in this toy.** At width 1024, the fitted best learning rate is
  $1.9 \eta^{\ast}$. The two hidden-layer norm summaries stay inside their band.
  The second hidden layer's activation-RMS ratio is 0.7975, against a lower boundary of 0.80. The mean is outside the band by 0.0025, about 0.3% of that boundary. Its scientific meaning is inconclusive, because the data file does not
  retain the per-seed activation values needed to assess seed uncertainty.
- **The input-rate variant.** The best learning rate stays within a factor of 1.2 of
  $\eta^{\ast}$, and the plotted hidden-layer summaries stay inside the band. The readout does not. Its norm and endpoint displacement both leave the band at widths 512 and 1024.

<!-- figure:loss -->

<!-- figure:certificate -->

The band is a descriptive check of the recorded means, not a confidence interval or a
statistical test. The saved loss curves include individual seed losses, but the spectral norms
and activation measurements retain only means over successful seeds. No interval for those
measurements can be recovered from this file. A near-boundary value therefore supports a
question for a new study, not a claim that a named scaling method fails.

A run fails when its final loss is at or above $\ln 10$, the loss of a zero output. A failed run
counts at $\ln 10$ in the loss curves. Over the whole sweep, 33 of 405 SP runs failed, 92 of 345
u-μP runs and 124 of 345 runs of the variant. At the rates used for the diagnostic, one SP seed
failed, at width 1024.

## What the plotted rows measure

- $\Delta W$ is the total change of a weight over training, the weight at the end minus the
  weight at the start. It is not the update of one step.
- The two norm rows average the seeds that trained and the two hidden matrices. They exclude
  the embedding and readout. Layer values in the data file are also seed means; they are not
  raw per-seed observations.
- The activation RMS is the RMS of the second hidden layer only.
- The diagnostic runs use, at every width, the grid learning rate nearest to the fitted best
  rate at width 64.
- A ratio is a value divided by the same value at width 64. The descriptive band is 0.80 to 1.25.

Before taking ratios, each layer's operator norm is divided by
$\sqrt{\text{fan-out}/\text{fan-in}}$. That factor is one for the square hidden matrices.
Applying the same calculation to the layers left out of the plots gives these examples;
values here are rounded to three decimals. The tables below give every layer at every width.

At width 1024, the u-μP toy's embedding has a normalized weight-norm ratio of
0.334 and displacement ratio of 0.250; its readout ratios are 1.730 and 1.747. The
input-rate variant's readout ratios are 1.345 and 1.334 at width 512, then 1.455 and 1.450
at width 1024.

Every value in these examples lies outside 0.80–1.25. Neither parameterization keeps every layer
inside this band. The variant's hidden-layer summary therefore cannot be called a whole-model
pass, and this exploratory comparison cannot establish that the variant transfers better.

AMLO's design asks for more than these rows before it transfers a setting. It also asks for
evidence about the differentiated program, the context length, the numerics and the training
horizon. This study does not measure any of those. It also does not test depth, other tasks or
other optimizers.

## Shared key and value heads

Figure 3 is computed from random matrices. Nothing in it is trained. In
grouped-query attention, one key and value projection serves $r$ query heads. The projection
acts as $r$ stacked copies of one block. With the initialization of an ordinary hidden layer,
the copies keep the activation RMS within 1% of its value at $r = 1$. The operator norm grows
with $r$, and at $r = 16$ it is 2.49 times its value at $r = 1$. The random-matrix limit of this
ratio is $(1 + \sqrt{r})/2$. A check of the activation RMS alone would pass this projection.
This is a mathematical control, not a trained study of grouped-query attention.

<!-- figure:kv -->

## What a follow-up would need

A confirmatory study would freeze the parameterizations, layers, metrics, tolerance bands,
seed plan and analysis before collecting new runs. It would save per-seed observations,
report uncertainty for the width ratios, and check input and output layers as well as hidden
layers. Tasks, architectures and training horizons would need their own declared scope.
This artifact supplies exploratory observations for that design; it supplies no such
confirmation.

## Provenance

`scripts/spectral-certificate.py` writes the data file `src/data/spectral-certificate.json`. The
file records the SHA-256 of the script that wrote it, the Python and NumPy versions and the run
time. Every run is seeded, and each worker uses one BLAS thread, so the sweep repeats bit for
bit on one machine with one NumPy version. A full sweep takes about 76 minutes on 6 CPU cores, with one worker for each core.
The site's tests check the saved grid, failure counts, ratio arithmetic and source provenance
without a new sweep. They do not recover missing observations or validate scientific transfer.

First published October 10, 2026; interpretation and layer reporting revised October 14, 2026.
This revision does not change the measured data or rerun the experiment. The exact historical
generator is kept under `scripts/spectral-history/`, named by the digest recorded in the data;
the current generator's comments describe the narrower scope above.
