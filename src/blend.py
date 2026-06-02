"""Caruana greedy ensemble selection — multiclass / balanced-accuracy variant.

Ported in spirit from rogii src/blend.py (which was 1D-regression/RMSE). Here each member
is an (n, n_classes) probability matrix and the objective is balanced accuracy on the argmax
of the blended probabilities. Greedy add-with-replacement, bagged over the model library,
sorted-init — the overfit-RESISTANT alternative to weight grids / Nelder-Mead (which burned us
in S6E5 v51). This is the endgame technique our notes flagged as a gap.

Selection scores on plain argmax balanced accuracy (fast); the per-class decision-threshold
tuning (metrics.tune_class_weights) is applied ONCE to the final blend by the caller — Caruana
picks the probability mix, the threshold layer is the post-hoc decision rule.

Optimize on the LABELED OOF only, never the LB/holdout.
"""
from __future__ import annotations

import numpy as np

from .metrics import competition_score


def _blend(members: list[np.ndarray], counts: np.ndarray) -> np.ndarray:
    """Weighted-average probability matrices by selection counts."""
    w = counts / counts.sum()
    out = np.zeros_like(members[0])
    for wi, P in zip(w, members):
        if wi:
            out += wi * P
    return out


def caruana_select_proba(
    oof: dict[str, np.ndarray],
    y_int,
    *,
    n_bag: int = 20,
    bag_frac: float = 0.5,
    max_iters: int = 50,
    seed: int = 42,
) -> tuple[dict[str, float], dict]:
    """Greedy bagged selection over probability-matrix members → weights summing to 1.

    `oof[name]` = (n, n_classes) OOF probabilities; `y_int` = integer labels. Score =
    balanced accuracy of argmax(blend). Returns (weights, info).
    """
    names = list(oof)
    members = [np.asarray(oof[n], dtype=float) for n in names]
    y = np.asarray(y_int)
    n = len(names)
    if n == 1:
        return {names[0]: 1.0}, {"n_members": 1}

    def score(P):  # balanced accuracy of the argmax — higher better
        return competition_score(y, np.argmax(P, axis=1))

    solo = np.array([score(P) for P in members])
    rng = np.random.default_rng(seed)
    total = np.zeros(n)

    for _ in range(n_bag):
        k = min(n, max(2, int(np.ceil(bag_frac * n))))
        lib = rng.choice(n, size=k, replace=False)
        order = lib[np.argsort(-solo[lib])]           # sorted init: best first
        counts = np.zeros(n)
        counts[order[0]] += 1.0
        cur_sum = members[order[0]].copy()
        cur_n = 1.0
        best = score(cur_sum / cur_n)
        for _ in range(max_iters):
            cand = [(score((cur_sum + members[j]) / (cur_n + 1)), j) for j in lib]
            s, j = max(cand, key=lambda t: t[0])
            if s <= best + 1e-12:
                break
            counts[j] += 1.0
            cur_sum += members[j]
            cur_n += 1.0
            best = s
        total += counts / counts.sum()

    w = total / total.sum()
    weights = dict(zip(names, w.tolist()))
    blend = _blend(members, np.array([weights[n_] for n_ in names]))
    bi = int(np.argmax(solo))
    info = {
        "n_members": n, "n_bag": n_bag,
        "best_single": names[bi], "best_single_score": float(solo[bi]),
        "simple_mean_score": float(score(np.mean(members, axis=0))),
        "caruana_argmax_score": float(score(blend)),
    }
    return weights, info


def blend_proba(oof: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Apply a weight dict to probability-matrix members → blended (n, n_classes)."""
    names = list(oof)
    return _blend([np.asarray(oof[n], dtype=float) for n in names],
                  np.array([weights[n] for n in names]))
