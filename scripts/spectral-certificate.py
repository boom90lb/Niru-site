#!/usr/bin/env python3
"""
Writes src/data/spectral-certificate.json, the data of the measured toy study
at /research/spectral-diagnostics/ on the spectral obligations of a transfer
certificate.

Acts 1 and 2 are real training runs. A small model reads eight tokens through
an embedding table, passes them through two square ReLU hidden matrices and a
readout, and learns to classify them as a fixed random teacher network does.
It trains with Adam at five widths, over one learning-rate grid, in three
parameterizations: the standard parameterization (SP), u-μP (Blake et al.
2024, arXiv:2407.17465, Table 2) and a variant of u-μP with μP's
width-independent learning rate for the input layer. The toy implementation
uses these per-layer rules: unit initialisation everywhere;
forward multipliers 1 (embedding), 1/√fan_in
(hidden) and 1/fan_in (readout); Adam learning-rate factors 1/√fan_out
(embedding), 1/√fan_in (hidden) and 1 (readout); and a unit-scaled ReLU.
Adam is invariant to the scale of each gradient up to ε, so u-μP's backward
scale factors do not change these runs and are left out. After training, the
script measures spectral and coordinate diagnostics on every run. The saved
spectral and activation values are seed means, not per-seed observations;
the hidden-layer summary is not a certificate for every layer. This script
does not use or qualify AMLO's training path. The input-rate variant was
chosen after inspecting initial trials and is an exploratory comparison.

Act 3 is linear algebra, not training. A grouped-query K/V projection with r
repeats is r row-stacked copies of one (n/r) × n block (GQA-μP, Chickering et
al. 2026, arXiv:2605.15290, Lemma 1). With the hidden-layer initialisation
that ignores the repetition, the script draws the block, stacks it and
measures the activation RMS and the operator norm of the stacked matrix.

Everything is seeded. Each run uses one BLAS thread, so the output repeats
bit for bit on one machine and numpy version. The pool runs one worker per
physical core: two hyperthreads share a core's vector units, so a second
worker per core slows this matrix-bound work.

Usage: npm run spectral, or
       uv run --no-project --python 3.15 --with numpy==2.5.4 -- \
           python -I scripts/spectral-certificate.py [--jobs N]
       (the diagnostic flags are listed by --help)
"""

from __future__ import annotations

import os

# One BLAS thread per worker: a fixed order of sums, and the pool does the rest.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "data" / "spectral-certificate.json"

CONFIG = {
    "task": {
        "kind": "tokens",
        "positions": 8,
        "vocab": 32,
        "outputs": 10,
        "teacher_embed": 32,
        "teacher_hidden": [128, 128],
        "teacher_seed": 0,
        "eval_inputs": 8192,
        "eval_seed": 7,
    },
    "model": {
        "layers": ["embedding", "hidden_1", "hidden_2", "readout"],
        "activation": "relu",
    },
    "widths": [64, 128, 256, 512, 1024],
    "steps": 1000,
    "batch": 256,
    "warmup_fraction": 0.1,
    "final_lr_fraction": 0.1,
    "adam": {"beta1": 0.9, "beta2": 0.999, "eps": 1e-12},
    "seeds": [1, 2, 3],
    # log2 of the base learning rate η, in half-octave steps.
    "lr_grid": {
        "sp": [x / 2 for x in range(-32, -5)],
        "umup": [x / 2 for x in range(-10, 13)],
        "mup": [x / 2 for x in range(-10, 13)],
    },
    # The certificate's tolerance band, as a ratio to the proxy's value.
    "band": [0.8, 1.25],
    "act3": {
        "width": 512,
        "query_heads": 16,
        "repeats": [1, 2, 4, 8, 16],
        "seed": 11,
        "probe_inputs": 4096,
    },
}

# Diagnostic: the same classification on dense Gaussian inputs, where the
# first layer is a projection rather than an embedding.
DENSE = {
    "kind": "dense",
    "inputs": 16,
    "outputs": 10,
    "teacher_hidden": [128, 128],
    "teacher_seed": 0,
}

PILOT = {
    "widths": [64, 256, 1024],
    "steps": 600,
    "seeds": [1],
    "lr_grid": {
        "sp": [float(x) for x in range(-16, -2)],
        "umup": [float(x) for x in range(-7, 8)],
        "mup": [float(x) for x in range(-7, 8)],
    },
}


# ---------------------------------------------------------------------------
# Task: label inputs as a fixed random teacher network does.


def relu(z):
    return np.maximum(z, 0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


class Task:
    """Inputs, labels and the cross-entropy loss for one teacher."""

    def __init__(self, cfg):
        t = cfg["task"]
        self.t = t
        self.kind = t["kind"]
        rng = np.random.default_rng(t["teacher_seed"])
        if self.kind == "tokens":
            self.inputs = t["positions"] * t["vocab"]
            # The teacher embeds the tokens with its own table.
            self.embed = rng.standard_normal((self.inputs, t["teacher_embed"])).astype(np.float32)
            dims = [t["teacher_embed"], *t["teacher_hidden"], t["outputs"]]
        else:
            self.inputs = t["inputs"]
            self.embed = None
            dims = [t["inputs"], *t["teacher_hidden"], t["outputs"]]
        self.w = [
            (rng.standard_normal((dims[i + 1], dims[i])) * math.sqrt(2 / dims[i])).astype(
                np.float32
            )
            for i in range(len(dims) - 1)
        ]
        self.outputs = t["outputs"]

    def sample(self, rng, count):
        """Model inputs. Tokens are a multi-hot row scaled by 1/√positions, so
        that x @ Eᵀ is the unit-scaled sum of the tokens' embedding rows."""
        if self.kind == "tokens":
            p, v = self.t["positions"], self.t["vocab"]
            tok = rng.integers(0, v, size=(count, p))
            x = np.zeros((count, p * v), dtype=np.float32)
            x[np.arange(count)[:, None], np.arange(p)[None, :] * v + tok] = 1 / math.sqrt(p)
            return x
        return rng.standard_normal((count, self.inputs)).astype(np.float32)

    def label(self, x):
        h = x @ self.embed if self.embed is not None else x
        for w in self.w[:-1]:
            h = relu(h @ w.T)
        y = h @ self.w[-1].T
        return np.eye(self.outputs, dtype=np.float32)[np.argmax(y, axis=1)]

    @staticmethod
    def loss(y, out):
        return float(-(y * log_softmax(out.astype(np.float64))).sum(axis=1).mean())

    @staticmethod
    def grad(y, out):
        return ((softmax(out) - y) / np.float32(len(y))).astype(np.float32)

    def trivial(self):
        """The loss of a zero output: uniform class probabilities."""
        return math.log(self.outputs)


# ---------------------------------------------------------------------------
# Parameterizations. The effective weight is W = A · w, w the stored tensor.


def parameterization(name, width, task, cfg):
    """Shapes, init std, forward multiplier A, Adam LR factor C, ReLU gain."""
    d_in = task.inputs
    k = task.outputs
    n = width
    emb = task.kind == "tokens"
    shapes = [(n, d_in), (n, n), (n, n), (k, n)]
    if name == "sp":
        # Standard practice: N(0, 1) embeddings (or He init for a dense input
        # layer), He init for the hidden layers, LeCun init for the readout,
        # no multipliers and one global Adam learning rate.
        init = [
            1.0 if emb else math.sqrt(2 / d_in),
            math.sqrt(2 / n),
            math.sqrt(2 / n),
            math.sqrt(1 / n),
        ]
        mult = [1.0, 1.0, 1.0, 1.0]
        lr = [1.0, 1.0, 1.0, 1.0]
        gain = 1.0
    elif name == "umup":
        # u-μP: unit init; A = 1 (embedding) or 1/√fan_in (dense input),
        # 1/√fan_in (hidden), 1/fan_in (readout); Adam LR factor 1/√fan_out
        # (input), 1/√fan_in (hidden), 1 (readout). The unit-scaled ReLU,
        # √2 · relu(z), has unit RMS for unit-normal z.
        init = [1.0, 1.0, 1.0, 1.0]
        mult = [1.0 if emb else 1 / math.sqrt(d_in), 1 / math.sqrt(n), 1 / math.sqrt(n), 1 / n]
        lr = [1 / math.sqrt(n), 1 / math.sqrt(n), 1 / math.sqrt(n), 1.0]
        if cfg.get("input_lr_rule") == "mup":
            lr[0] = 1.0  # diagnostic: μP's width-independent Adam LR for input weights
        gain = math.sqrt(2.0)
    elif name == "mup":
        # μP in the same unit-scaled form: u-μP with μP's own rule for the
        # input layer, a width-independent Adam LR, in place of u-μP's
        # embedding rule 1/√fan_out. By abc-symmetry its dynamics are μP's.
        init = [1.0, 1.0, 1.0, 1.0]
        mult = [1.0 if emb else 1 / math.sqrt(d_in), 1 / math.sqrt(n), 1 / math.sqrt(n), 1 / n]
        lr = [1.0, 1 / math.sqrt(n), 1 / math.sqrt(n), 1.0]
        gain = math.sqrt(2.0)
    else:
        raise ValueError(name)
    return shapes, init, mult, lr, gain


def forward(ws, mult, gain, x, first_act, keep=False):
    """Input layer, two ReLU hidden layers, readout. An embedding input feeds
    the first hidden matrix directly; a dense input layer has its own ReLU."""
    acts = [x]
    pre = []
    h = x
    for i in range(3):
        z = (h @ ws[i].T) * np.float32(mult[i])
        pre.append(z)
        h = np.maximum(z, 0) * np.float32(gain) if (i > 0 or first_act) else z
        acts.append(h)
    out = (h @ ws[3].T) * np.float32(mult[3])
    return (out, acts, pre) if keep else out


def lr_at(step, steps, cfg):
    """Linear warmup, then a cosine decay to a fraction of the peak."""
    warm = max(1, int(cfg["warmup_fraction"] * steps))
    if step < warm:
        return (step + 1) / warm
    u = (step - warm) / max(1, steps - warm - 1)
    end = cfg["final_lr_fraction"]
    return end + (1 - end) * 0.5 * (1 + math.cos(math.pi * min(u, 1.0)))


def op_norm(m):
    return float(np.linalg.svd(m.astype(np.float64), compute_uv=False)[0])


def rms(a):
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


_TASK = None
_EVAL = None


def _setup(cfg):
    global _TASK, _EVAL
    if _TASK is None:
        _TASK = Task(cfg)
        rng = np.random.default_rng(cfg["task"]["eval_seed"])
        x = _TASK.sample(rng, cfg["task"]["eval_inputs"])
        _EVAL = (x, _TASK.label(x))


def evaluate(ws, mult, gain, first_act):
    x, y = _EVAL
    total = 0.0
    for i in range(0, len(x), 2048):
        out = forward(ws, mult, gain, x[i : i + 2048], first_act)
        total += _TASK.loss(y[i : i + 2048], out) * len(out)
    return total / len(x)


def activation_rms(ws, mult, gain, first_act, n_probe=1024):
    _, acts, _ = forward(ws, mult, gain, _EVAL[0][:n_probe], first_act, keep=True)
    return [rms(a) for a in acts[1:]]


def train(job):
    name, width, log2lr, seed, cfg = job
    _setup(cfg)
    task = _TASK
    first_act = task.kind != "tokens"
    shapes, init, mult, lrf, gain = parameterization(name, width, task, cfg)
    rng = np.random.default_rng([seed, width, 1])
    ws = [(rng.standard_normal(s) * init[i]).astype(np.float32) for i, s in enumerate(shapes)]
    w0 = [w.copy() for w in ws]
    ms = [np.zeros_like(w) for w in ws]
    vs = [np.zeros_like(w) for w in ws]
    tmp = [np.empty_like(w) for w in ws]
    data = np.random.default_rng([seed, 2])  # the same batches at every width and LR
    b1, b2, eps = cfg["adam"]["beta1"], cfg["adam"]["beta2"], cfg["adam"]["eps"]
    base = 2.0**log2lr
    steps, batch = cfg["steps"], cfg["batch"]
    act0 = activation_rms(ws, mult, gain, first_act)
    diverged = False
    grad_rms = [math.inf] * 4
    for step in range(steps):
        x = task.sample(data, batch)
        y = task.label(x)
        out, acts, pre = forward(ws, mult, gain, x, first_act, keep=True)
        if not np.all(np.isfinite(out)):
            diverged = True
            break
        g_out = task.grad(y, out)
        grads = [None] * 4
        grads[3] = (g_out.T @ acts[3]) * np.float32(mult[3])
        g_h = (g_out @ ws[3]) * np.float32(mult[3])
        for i in (2, 1, 0):
            g_z = g_h * (pre[i] > 0) * np.float32(gain) if (i > 0 or first_act) else g_h
            grads[i] = (g_z.T @ acts[i]) * np.float32(mult[i])
            if i > 0:
                g_h = (g_z @ ws[i]) * np.float32(mult[i])
        lr_t = base * lr_at(step, steps, cfg)
        c1 = 1 - b1 ** (step + 1)
        c2 = 1 - b2 ** (step + 1)
        for i in range(4):
            g, m, v, tm = grads[i], ms[i], vs[i], tmp[i]
            if step == steps - 1:
                grad_rms[i] = rms(g)
            m *= b1
            np.multiply(g, 1 - b1, out=tm)
            m += tm
            v *= b2
            np.multiply(g, g, out=tm)
            tm *= 1 - b2
            v += tm
            # w ← w − η_t · C · m̂ / (√v̂ + ε)
            np.sqrt(v, out=tm)
            tm *= np.float32(1 / math.sqrt(c2))
            tm += np.float32(eps)
            np.divide(m, tm, out=tm)
            tm *= np.float32(lr_t * lrf[i] / c1)
            ws[i] -= tm
    loss = evaluate(ws, mult, gain, first_act) if not diverged else math.nan
    trivial = task.trivial()
    # A run that ends no better than the zero output has failed: it diverged
    # or its units died.
    if not math.isfinite(loss) or loss >= trivial:
        diverged = True
    rec = {
        "p": name,
        "width": width,
        "log2_lr": log2lr,
        "seed": seed,
        "loss": loss if not diverged else None,
        "loss_capped": min(loss, trivial) if math.isfinite(loss) else trivial,
        "diverged": diverged,
    }
    if not diverged:
        eff = [w * np.float32(mult[i]) for i, w in enumerate(ws)]
        eff0 = [w * np.float32(mult[i]) for i, w in enumerate(w0)]
        rec["layers"] = {}
        for i, layer in enumerate(cfg["model"]["layers"]):
            fo, fi = shapes[i]
            d = eff[i] - eff0[i]
            rec["layers"][layer] = {
                "fan_out": fo,
                "fan_in": fi,
                "W_op": op_norm(eff[i]),
                "W0_op": op_norm(eff0[i]),
                "dW_op": op_norm(d),
                "dW_fro": float(np.linalg.norm(d.astype(np.float64))),
                "W_fro": float(np.linalg.norm(eff[i].astype(np.float64))),
            }
        rec["act_rms_init"] = act0
        rec["act_rms"] = activation_rms(ws, mult, gain, first_act)
        rec["grad_rms_last_step"] = grad_rms
    return rec


# ---------------------------------------------------------------------------
# Act 3: a K/V projection reused across query groups.


def act3(cfg):
    a = cfg["act3"]
    n = a["width"]
    rng = np.random.default_rng(a["seed"])
    x = rng.standard_normal((a["probe_inputs"], n))
    rows = []
    for r in a["repeats"]:
        # The block that a hidden-layer rule initialises: σ = 1/√fan_in.
        block = rng.standard_normal((n // r, n)) / math.sqrt(n)
        stacked = np.vstack([block] * r)
        s_block = np.linalg.svd(block, compute_uv=False)
        s_stack = np.linalg.svd(stacked, compute_uv=False)
        rows.append(
            {
                "r": r,
                "kv_heads": a["query_heads"] // r,
                "block_shape": [n // r, n],
                "stacked_shape": [n, n],
                "block_rms": rms(block),
                "stacked_rms": rms(stacked),
                "block_op": float(s_block[0]),
                "stacked_op": float(s_stack[0]),
                "stacked_rank": int(np.sum(s_stack > s_stack[0] * 1e-10)),
                "act_rms": rms(x @ stacked.T),
                # Bai–Yin edge of the stack: √r · (1/√n)(√(n/r) + √n) = 1 + √r.
                "bai_yin_op": 1 + math.sqrt(r),
            }
        )
    ref = rows[0]
    for row in rows:
        row["coord_ratio"] = row["act_rms"] / ref["act_rms"]
        row["spectral_ratio"] = row["stacked_op"] / ref["stacked_op"]
        row["identity_error"] = (
            abs(row["stacked_op"] - math.sqrt(row["r"]) * row["block_op"]) / row["stacked_op"]
        )
        row["coord_inside"] = cfg["band"][0] <= row["coord_ratio"] <= cfg["band"][1]
        row["spectral_inside"] = cfg["band"][0] <= row["spectral_ratio"] <= cfg["band"][1]
    return rows


# ---------------------------------------------------------------------------
# Summaries.


def fit_minimum(xs, ys):
    """Vertex of the parabola through the grid minimum and its two neighbours."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pts) < 3:
        return None
    i = min(range(len(pts)), key=lambda k: pts[k][1])
    if i == 0 or i == len(pts) - 1:
        return {"log2_lr": pts[i][0], "loss": pts[i][1], "grid_log2_lr": pts[i][0], "edge": True}
    (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    c = y1 - a * x1 * x1 - b * x1
    xv = min(max(-b / (2 * a) if a > 0 else x1, x0), x2)
    return {"log2_lr": xv, "loss": a * xv * xv + b * xv + c, "grid_log2_lr": x1, "edge": False}


def summarise(records, cfg):
    out = {}
    for p in [q for q in ("sp", "umup", "mup") if any(r["p"] == q for r in records)]:
        curves = []
        for width in cfg["widths"]:
            pts = []
            for lr in cfg["lr_grid"][p]:
                rs = [
                    r
                    for r in records
                    if r["p"] == p and r["width"] == width and r["log2_lr"] == lr
                ]
                pts.append(
                    {
                        "log2_lr": lr,
                        # A failed seed counts at the zero-output loss.
                        "loss": float(np.mean([r["loss_capped"] for r in rs])),
                        "failed": sum(r["diverged"] for r in rs),
                        "seeds": [r["loss"] for r in rs],
                    }
                )
            fit = fit_minimum(
                [q["log2_lr"] for q in pts], [q["loss"] if q["failed"] == 0 else None for q in pts]
            )
            curves.append({"width": width, "points": pts, "minimum": fit})
        out[p] = curves
    return out


def slope(curves):
    """Least-squares slope of the fitted minimum, in octaves of η per octave of width."""
    xs = np.log2([c["width"] for c in curves])
    ys = np.array([c["minimum"]["log2_lr"] for c in curves])
    return float(np.polyfit(xs, ys, 1)[0])


def certificate(records, curves, cfg):
    """The certificate at the learning rate tuned at the proxy (the smallest width)."""
    band = cfg["band"]
    hidden = ["hidden_1", "hidden_2"]
    res = {}
    for p in curves:
        lr = curves[p][0]["minimum"]["grid_log2_lr"]
        rows = []
        for width in cfg["widths"]:
            rs = [r for r in records if r["p"] == p and r["width"] == width and r["log2_lr"] == lr]
            ok = [r for r in rs if not r["diverged"]]
            row = {"width": width, "log2_lr": lr, "failed": len(rs) - len(ok)}
            if ok:

                def mean(f):
                    return float(np.mean([f(r, h) for r in ok for h in hidden]))

                def spec(layer):
                    return math.sqrt(layer["fan_out"] / layer["fan_in"])

                row["dW_op_norm"] = mean(
                    lambda r, h: r["layers"][h]["dW_op"] / spec(r["layers"][h])
                )
                row["W_op_norm"] = mean(lambda r, h: r["layers"][h]["W_op"] / spec(r["layers"][h]))
                row["rel_update_fro"] = mean(
                    lambda r, h: r["layers"][h]["dW_fro"] / r["layers"][h]["W_fro"]
                )
                row["act_rms"] = float(np.mean([r["act_rms"][2] for r in ok]))
                row["loss"] = float(np.mean([r["loss"] for r in ok]))
                row["layers"] = {
                    layer: {
                        "dW_op_norm": float(
                            np.mean(
                                [
                                    r["layers"][layer]["dW_op"] / spec(r["layers"][layer])
                                    for r in ok
                                ]
                            )
                        ),
                        "W_op_norm": float(
                            np.mean(
                                [r["layers"][layer]["W_op"] / spec(r["layers"][layer]) for r in ok]
                            )
                        ),
                        "W0_op_norm": float(
                            np.mean(
                                [
                                    r["layers"][layer]["W0_op"] / spec(r["layers"][layer])
                                    for r in ok
                                ]
                            )
                        ),
                    }
                    for layer in cfg["model"]["layers"]
                }
                row["act_rms_layers"] = [
                    float(np.mean([r["act_rms"][k] for r in ok])) for k in range(3)
                ]
            rows.append(row)
        ref = rows[0]
        for row in rows:
            for key in ("dW_op_norm", "W_op_norm", "act_rms"):
                if key in row and key in ref:
                    row[key + "_ratio"] = row[key] / ref[key]
                    row[key + "_inside"] = band[0] <= row[key + "_ratio"] <= band[1]
        res[p] = {"proxy_width": cfg["widths"][0], "log2_lr": lr, "rows": rows}
    return res


DEFINITIONS = {
    "task": "Eight tokens, each one of 32 values, are classified into 10 classes by a fixed random teacher (its own embedding table, then two ReLU layers of width 128, then the argmax). Every step draws 256 fresh token strings.",
    "model": "An embedding table (width n), two n × n ReLU hidden matrices and a readout to 10 logits.",
    "loss": "Cross-entropy against the teacher's labels on 8192 fixed held-out token strings after the last step, averaged over seeds. A run that ends at or above ln 10, the loss of a zero output, has failed and counts at ln 10.",
    "W": "The effective weight of a layer, W = A · w: the stored tensor w times its forward multiplier A (A = 1 in SP).",
    "dW": "ΔW = W_T − W_0, the total change of the effective weight over training.",
    "op_norm": "‖M‖op, the largest singular value of M, from a float64 SVD.",
    "spectral_normalisation": "‖W‖op and ‖ΔW‖op are divided by √(fan_out/fan_in) (Yang, Simon & Bernstein 2023, arXiv:2310.17813). For the square hidden matrices this factor is 1.",
    "act_rms": "Root mean square of the second hidden layer's activations on 1024 held-out token strings after training.",
    "ratio": "A certificate value divided by the same value at the proxy width, the smallest.",
    "band": "A value passes when its ratio to the proxy lies inside the band.",
    "transfer": "The certificate runs use, at every width, the grid learning rate nearest to the proxy's fitted minimum.",
    "minimum": "Vertex of the parabola through the grid minimum of the mean loss and its two neighbours, in log2 η.",
    "slope": "Least-squares slope of the fitted minima, in octaves of η per octave of width.",
    "act3": "A grouped-query K/V projection with r repeats is the row stack of r copies of one (n/r) × n block (GQA-μP Lemma 1). The block has i.i.d. N(0, 1/n) entries: the hidden-layer rule, which ignores the repetition. act_rms is the RMS of the stacked projection applied to N(0, I) inputs; stacked_op is its largest singular value; the ratios are to r = 1. Nothing here is trained.",
}


def physical_cores():
    """The pool's workers: one per physical core this process may use."""
    logical = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    try:
        smt = Path("/sys/devices/system/cpu/smt/active").read_text().strip() == "1"
    except OSError:
        smt = False
    return max(1, (logical or 4) // (2 if smt else 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=physical_cores())
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--pilot", action="store_true", help="diagnostic: coarse grid, one seed")
    ap.add_argument(
        "--dense", action="store_true", help="diagnostic: dense Gaussian inputs instead of tokens"
    )
    ap.add_argument(
        "--input-lr",
        choices=["umup", "mup"],
        default="umup",
        help="diagnostic: u-μP input-layer LR rule",
    )
    ap.add_argument("--widths", default=None, help="diagnostic: comma-separated widths")
    ap.add_argument("--params", default="sp,umup,mup", help="diagnostic: parameterizations to run")
    args = ap.parse_args()
    cfg = json.loads(json.dumps(CONFIG))
    diagnostic = (
        args.pilot
        or args.dense
        or args.input_lr != "umup"
        or args.widths
        or args.params != "sp,umup,mup"
    )
    if args.pilot:
        cfg.update(PILOT)
    if args.dense:
        cfg["task"].update(DENSE)
        cfg["model"]["layers"][0] = "input"
    if args.input_lr != "umup":
        cfg["input_lr_rule"] = args.input_lr
    if args.widths:
        cfg["widths"] = [int(w) for w in args.widths.split(",")]
    if diagnostic and args.out == str(OUT):
        sys.exit("diagnostic runs need --out outside src/data")
    jobs = [
        (p, w, lr, s, cfg)
        for w in sorted(cfg["widths"], reverse=True)
        for p in args.params.split(",")
        for lr in cfg["lr_grid"][p]
        for s in cfg["seeds"]
    ]
    t0 = time.time()
    records = []
    with Pool(args.jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(train, jobs, chunksize=1)):
            records.append(rec)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(
                    f"{i + 1}/{len(jobs)} runs, {time.time() - t0:.0f} s",
                    file=sys.stderr,
                    flush=True,
                )
    runtime = time.time() - t0
    records.sort(key=lambda r: (r["p"], r["width"], r["log2_lr"], r["seed"]))
    curves = summarise(records, cfg)
    slopes = {p: slope(curves[p]) for p in curves}
    for p in curves:
        for c in curves[p]:
            m = c["minimum"]
            print(
                f"{p:5s} width {c['width']:5d}  min log2 η {m['log2_lr']:+.2f}  loss {m['loss']:.4f}{'  (edge)' if m['edge'] else ''}"
            )
        print(f"{p:5s} slope {slopes[p]:+.3f} octaves of η per octave of width")
    _setup(cfg)
    cert = certificate(records, curves, cfg)
    for p in cert:
        for row in cert[p]["rows"]:
            print(
                p, row["width"], {k: round(v, 3) for k, v in row.items() if isinstance(v, float)}
            )
    a3 = act3(cfg)
    for row in a3:
        print(
            "act3 r",
            row["r"],
            "coord",
            round(row["coord_ratio"], 4),
            "spectral",
            round(row["spectral_ratio"], 4),
            "identity err",
            f"{row['identity_error']:.1e}",
        )
    source = Path(__file__).read_bytes()
    doc = {
        "schema": "niru.spectral-certificate/1",
        "generator": {
            "script": "scripts/spectral-certificate.py",
            "sha256": hashlib.sha256(source).hexdigest(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "runtime_seconds": round(runtime),
            "jobs": args.jobs,
        },
        "settings": cfg,
        "definitions": DEFINITIONS,
        "loss_zero_output": _TASK.trivial(),
        "curves": curves,
        "slopes": slopes,
        "certificate": cert,
        "act3": a3,
    }
    if diagnostic:
        doc["records"] = records
    Path(args.out).write_text(
        json.dumps(doc, indent=1, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print("wrote", args.out, f"in {runtime:.0f} s", file=sys.stderr)


if __name__ == "__main__":
    main()
