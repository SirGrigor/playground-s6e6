"""Multiclass AUC diagnostics (OvR / OvO / single scalar).

IMPORTANT: AUC is NOT the competition metric (that's accuracy — see config.METRIC).
These are *diagnostics*: threshold-independent separability probes that stay informative
when accuracy is saturated near the ceiling. We trend the single scalar in the diary
alongside accuracy; the per-pair OvO is the one that matters most here because the whole
comp is the GALAXY↔QSO boundary.

Usage::

    from src.metrics import multiclass_auc_report
    rep = multiclass_auc_report(y_true, proba)        # proba: (n, n_classes) aligned to CLASSES
    rep["scalar"]          # headline number to track (macro-OvO by default)
    rep["ovr_per_class"]   # {class: auc}
    rep["ovo_per_pair"]    # {"GALAXY|QSO": auc, ...}  ← the crux pair lives here
    # pass rep["scalar"] into Experiment.record(extra={"auc_ovo_macro": rep["scalar"]})
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .config import CLASSES


def multiclass_auc_report(y_true, proba, *, labels=None, headline: str = "ovo_macro") -> dict:
    """Compute OvR + OvO multiclass AUC every way, plus one scalar headline.

    Returns a dict with:
      ovr_per_class, ovr_macro, ovr_weighted, ovr_micro,
      ovo_per_pair, ovo_macro,
      scalar  (= the `headline` flavor; default macro-OvO, imbalance-robust per Hand & Till)
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize

    classes = list(labels) if labels is not None else list(CLASSES)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    Y = label_binarize(y_true, classes=classes)

    out: dict = {}

    # --- OvR: each class vs the rest -------------------------------------
    ovr_per_class = {}
    for k, c in enumerate(classes):
        # guard against a class absent from this subset (e.g. a tiny holdout slice)
        if Y[:, k].min() == Y[:, k].max():
            ovr_per_class[c] = float("nan")
        else:
            ovr_per_class[c] = float(roc_auc_score(Y[:, k], proba[:, k]))
    out["ovr_per_class"] = ovr_per_class
    # Aggregate from the per-class values we already computed correctly. We deliberately
    # do NOT call sklearn's multi_class roc_auc_score: it assumes proba columns are in
    # SORTED label order and rejects an unsorted `labels=` — our columns follow CLASSES.
    # The macro/weighted average of the OvR per-class AUCs IS the standard definition.
    valid = {c: v for c, v in ovr_per_class.items() if not np.isnan(v)}
    support = {c: int((y_true == c).sum()) for c in valid}
    tot = sum(support.values()) or 1
    out["ovr_macro"] = float(np.mean(list(valid.values()))) if valid else float("nan")
    out["ovr_weighted"] = float(sum(valid[c] * support[c] for c in valid) / tot) if valid else float("nan")
    out["ovr_micro"] = float(roc_auc_score(Y.ravel(), proba.ravel()))

    # --- OvO: each pair, restricted to rows of those two classes ---------
    ovo_per_pair = {}
    for a, b in combinations(range(len(classes)), 2):
        mask = np.isin(y_true, [classes[a], classes[b]])
        if mask.sum() == 0:
            continue
        ya = (y_true[mask] == classes[a]).astype(int)
        if ya.min() == ya.max():
            continue
        # score = P(a) renormalized over the pair, so it's a proper 2-class ranking
        pa, pb = proba[mask, a], proba[mask, b]
        denom = np.clip(pa + pb, 1e-12, None)
        ovo_per_pair[f"{classes[a]}|{classes[b]}"] = float(roc_auc_score(ya, pa / denom))
    out["ovo_per_pair"] = ovo_per_pair
    out["ovo_macro"] = float(np.mean(list(ovo_per_pair.values()))) if ovo_per_pair else float("nan")

    out["scalar"] = out.get(headline, out["ovo_macro"])
    out["headline_flavor"] = headline
    return out


def format_auc_report(rep: dict) -> str:
    """One-line-per-section text summary for logs / the live dashboard."""
    lines = [f"AUC scalar ({rep.get('headline_flavor', 'ovo_macro')}): {rep['scalar']:.5f}"]
    lines.append("  OvR: " + ", ".join(f"{c} {v:.4f}" for c, v in rep["ovr_per_class"].items())
                 + f"  | macro {rep['ovr_macro']:.4f} micro {rep['ovr_micro']:.4f}")
    if rep["ovo_per_pair"]:
        lines.append("  OvO pairs: " + ", ".join(f"{p} {v:.4f}" for p, v in rep["ovo_per_pair"].items())
                     + f"  | macro {rep['ovo_macro']:.4f}")
    return "\n".join(lines)
