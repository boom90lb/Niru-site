---
title: "The three expansions of the search"
description: "AMLO's design widens what it optimizes in three stages, from architectures to training programs to the deployed model with its harness."
eyebrow: "Program note / 02"
published: 2026-09-30
updated: 2026-10-14
status: "Program note"
---

Most architecture search holds the training recipe fixed and leaves the
deployed system out of scope. In AMLO's design, the architecture is only the
first object of the search. The program widens that object in three stages,
with the same accounting at each.

## Architecture

The first stage is an evolutionary search over architectures. Each candidate
is a typed graph of components, and what the graph means is fixed before it
runs. Before training, a candidate gets a structural check and an analytic
cost estimate. It then trains under a fixed compute budget and advances
through the evaluation tiers only when the evidence supports it. This stage
is the reference path. Parts of its foundation are built, including the checks before training, the compute planner and the promotion statistics. Candidate training inside
the search is the next milestone.

## Training programs

The second stage puts the training program into the search, with its data mixtures
and their schedules, filters, the order of the data, objectives, optimizers
and the phases of training. These choices can move results as much as the architecture does; [one study of architecture search](https://arxiv.org/abs/1912.12522) found the training protocol mattered more. They are also where the most specialist knowledge sits. A
program is a candidate in its own right, with its own identity and record.
Once the search covers it under the same budgets and gates, it can aim at an
outcome rather than at a network. A training-program study can run at one
scale; it needs a test of transfer only when its claim is about another
scale.

## The deployed model and its harness

The third stage widens the boundary once more, to the system a user meets, which is the model with its tools, memory, retrieval, verifiers and the inference
budget it is allowed to spend. At this stage the search optimizes a working
system, and it can include the loop that trains that system.

## Where the search runs

Distributed training is a separate line of work. It changes where the search
runs, not what it optimizes. Placement and sharding become recorded choices,
and the identities, budgets and gates stay the same.

## Why staged, not joint

The three stages are coupled. An architecture can change which curriculum is
best. A curriculum can change what a harness is worth. A deployment context
can invalidate a result from an earlier stage. AMLO neither ignores these
effects nor searches everything at once. It optimizes in stages, gives each
stage a typed contract with the next, and measures the interactions directly.
A pipeline is trusted only inside the region where that measurement supports
it, and where the interaction is large, the stages are searched together with
their own budget. Each stage keeps the full record of the stage before it.

## Beyond the three stages

The long-term aim is a system that applies the same procedure to its own method. It proposes a change to how it searches, and keeps the change only
when a sealed comparison supports it. This is a research aim, not a stage of
the current plan. It needs its own evidence that each change to the method
helps, measured against a reference the method cannot alter.

The distinctive bet is in the combination of an editable representation of the whole learning
program, compositional meaning and cost, small-scale experiments with tested transfer, and
independent confirmation under declared resources. A research system could then learn which
experiments are worth running, while its proposed changes to the research method face the same
comparison discipline as changes to a model.

The potential advantage is cumulative. An improvement in how research is conducted can help
find subsequent improvements in the learner, training program, and working system. That is a
reason to pursue this line of attack, not evidence that the loop will be stable or that its benefits will compound without limit. Each of those claims needs its
own experiments. Confirmation must remain outside the process being optimized; a method cannot
improve its result by rewriting the test that judges it.
