"""Multiclass diagnostic visualizations for S6E6 (GALAXY / STAR / QSO).

The competition is a near-ceiling accuracy problem: redshift alone separates most
rows; the whole game is the galaxy↔QSO confusion cell at moderate redshift. So the
confusion matrix is the centerpiece, not an afterthought.

Every function:
  - takes plain arrays (no model objects),
  - saves a PNG to `reports/figs/` and returns the matplotlib Figure,
  - is wrapped so a plotting failure never aborts a training run (cosmetic-only,
    same discipline as src/dashboard.py).

Core entry point — call once per experiment with OOF (or holdout) predictions::

    from src.viz import render_all_panels
    paths = render_all_panels(
        y_true, y_pred, proba,            # proba: (n, n_classes) aligned to CLASSES
        fold_scores={"v3": [0.974, ...]}, # optional, for the fold boxplot
        prefix="v3",
    )
    # paths: dict[str, Path] of every PNG written — feed to dashboard.html

`render_all_panels` is what the dashboard embeds. Individual panels are public too.
"""
from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import CLASSES, FIGS, METRIC

# Consistent class → color across every panel.
CLASS_COLORS = {"GALAXY": "#4c72b0", "STAR": "#dd8452", "QSO": "#55a868"}


def _safe(fn):
    """Plotting is cosmetic — never let it abort a run. Returns None on failure."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"  [viz] {fn.__name__} skipped: {type(e).__name__}: {e}")
            return None
    return wrapper


def _classes(labels: Sequence[str] | None) -> list[str]:
    return list(labels) if labels is not None else list(CLASSES)


def _figpath(prefix: str, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    return FIGS / f"{prefix}_{name}.png"


def _save(fig, path: Path):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- panels
@_safe
def confusion_matrix_panel(y_true, y_pred, *, labels=None, prefix="model", normalize=True):
    """Confusion matrix — counts AND row-normalized, side by side.

    The galaxy↔QSO off-diagonal cells are where the competition is won; they get
    annotated in red when row-rate exceeds 1%.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    classes = _classes(labels)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, title, fmt in (
        (axes[0], cm, "Confusion — counts", "d"),
        (axes[1], cmn, "Confusion — row-normalized", ".3f"),
    ):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=(cm.max() if fmt == "d" else 1.0))
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = mat[i, j]
                txt = f"{v:{fmt}}"
                off_diag = i != j
                rate = cmn[i, j]
                color = ("crimson" if (off_diag and rate > 0.01) else
                         ("white" if im.norm(v) > 0.5 else "black"))
                weight = "bold" if (off_diag and rate > 0.01) else "normal"
                ax.text(j, i, txt, ha="center", va="center", color=color, fontweight=weight, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"{prefix} — confusion (red = off-diagonal >1%)", fontweight="bold")
    return _save(fig, _figpath(prefix, "confusion"))


@_safe
def roc_panel(y_true, proba, *, labels=None, prefix="model"):
    """One-vs-rest ROC per class + micro & macro average AUC."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    classes = _classes(labels)
    proba = np.asarray(proba)
    Y = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    aucs = {}
    for k, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
        a = auc(fpr, tpr); aucs[cls] = a
        ax.plot(fpr, tpr, color=CLASS_COLORS.get(cls), lw=2, label=f"{cls} (AUC {a:.4f})")
    # micro
    fpr_micro, tpr_micro, _ = roc_curve(Y.ravel(), proba.ravel())
    a_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color="black", lw=1.5, ls="--", label=f"micro (AUC {a_micro:.4f})")
    a_macro = float(np.mean(list(aucs.values())))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(f"{prefix} — one-vs-rest ROC  (macro AUC {a_macro:.4f})")
    ax.legend(loc="lower right", fontsize=9)
    return _save(fig, _figpath(prefix, "roc"))


@_safe
def pr_panel(y_true, proba, *, labels=None, prefix="model"):
    """Per-class precision-recall curves + average precision (matters under imbalance)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    classes = _classes(labels)
    proba = np.asarray(proba)
    Y = label_binarize(y_true, classes=classes)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for k, cls in enumerate(classes):
        prec, rec, _ = precision_recall_curve(Y[:, k], proba[:, k])
        ap = average_precision_score(Y[:, k], proba[:, k])
        ax.plot(rec, prec, color=CLASS_COLORS.get(cls), lw=2, label=f"{cls} (AP {ap:.4f})")
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{prefix} — precision-recall (one-vs-rest)")
    ax.legend(loc="lower left", fontsize=9)
    return _save(fig, _figpath(prefix, "pr"))


@_safe
def classification_report_panel(y_true, y_pred, *, labels=None, prefix="model"):
    """Per-class precision/recall/F1 heatmap + accuracy & macro/weighted F1 in the title."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    classes = _classes(labels)
    rep = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    metrics = ["precision", "recall", "f1-score"]
    mat = np.array([[rep[c][m] for m in metrics] for c in classes])

    fig, ax = plt.subplots(figsize=(6, 0.9 * len(classes) + 2))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.8, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=10)
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0)
    ax.set_title(f"{prefix} — accuracy {acc:.4f} | macro-F1 {macro:.4f} | weighted-F1 {weighted:.4f}",
                 fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="score")
    return _save(fig, _figpath(prefix, "report"))


@_safe
def class_balance_panel(y_true, y_pred=None, *, labels=None, prefix="model"):
    """True (and optionally predicted) class distribution — catches prior shift."""
    import matplotlib.pyplot as plt

    classes = _classes(labels)
    y_true = np.asarray(y_true)
    true_counts = [int((y_true == c).sum()) for c in classes]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(classes))
    w = 0.38 if y_pred is not None else 0.6
    ax.bar(x - (w / 2 if y_pred is not None else 0), true_counts, w,
           label="true", color=[CLASS_COLORS.get(c) for c in classes])
    if y_pred is not None:
        y_pred = np.asarray(y_pred)
        pred_counts = [int((y_pred == c).sum()) for c in classes]
        ax.bar(x + w / 2, pred_counts, w, label="predicted", color="grey", alpha=0.7)
        ax.legend()
    ax.set_xticks(x, classes); ax.set_ylabel("count")
    ax.set_title(f"{prefix} — class balance")
    return _save(fig, _figpath(prefix, "class_balance"))


@_safe
def fold_score_boxplot(fold_scores: dict[str, Sequence[float]], *, prefix="pool"):
    """Per-fold score spread across models — surfaces fold collapse / instability.

    fold_scores: {model_name: [fold0, fold1, ...]}
    """
    import matplotlib.pyplot as plt

    names = list(fold_scores.keys())
    data = [list(fold_scores[n]) for n in names]
    order = np.argsort([np.median(d) for d in data])
    names = [names[i] for i in order]; data = [data[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names) + 3), 5))
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4c72b0"); patch.set_alpha(0.5)
    for i, d in enumerate(data):
        ax.scatter([i + 1] * len(d), d, color="black", s=18, zorder=3)
    ax.set_ylabel(f"per-fold {METRIC}")
    ax.set_title(f"{prefix} — per-fold {METRIC} (median-sorted)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, _figpath(prefix, "fold_scores"))


@_safe
def ovo_auc_panel(y_true, proba, *, labels=None, prefix="model"):
    """One-vs-one pairwise AUC bars — the GALAXY↔QSO pair (the crux) highlighted.

    Complements roc_panel (which is one-vs-rest). OvO is imbalance-robust (Hand & Till)
    and directly measures the pairwise separability that decides this competition.
    """
    import matplotlib.pyplot as plt

    from .metrics import multiclass_auc_report

    rep = multiclass_auc_report(y_true, proba, labels=labels)
    pairs = list(rep["ovo_per_pair"].items())
    if not pairs:
        return None
    pairs.sort(key=lambda kv: kv[1])
    names = [p for p, _ in pairs]
    vals = [v for _, v in pairs]
    colors = ["crimson" if {"GALAXY", "QSO"} == set(n.split("|")) else "#4c72b0" for n in names]

    fig, ax = plt.subplots(figsize=(6.5, 0.7 * len(names) + 2))
    ax.barh(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.4f}", va="center", fontsize=9)
    ax.set_xlim(min(0.9, min(vals) - 0.01), 1.001)
    ax.set_xlabel("one-vs-one AUC")
    ax.set_title(f"{prefix} — OvO pairwise AUC (red = GALAXY↔QSO) | macro {rep['ovo_macro']:.4f}",
                 fontsize=10, fontweight="bold")
    return _save(fig, _figpath(prefix, "ovo_auc"))


@_safe
def proba_rho_matrix(probas: dict[str, np.ndarray], *, klass_index=0, prefix="pool"):
    """Rank-correlation heatmap between models' P(class) vectors — ensemble diversity.

    probas: {model_name: (n, n_classes) array}. Compares the column `klass_index`
    (default class 0). Low ρ pairs = the diversity Caruana can exploit.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import rankdata

    names = list(probas.keys())
    cols = [np.asarray(probas[n])[:, klass_index] for n in names]
    n = len(names)
    R = np.eye(n)
    for a in range(n):
        for b in range(a + 1, n):
            r = float(np.corrcoef(rankdata(cols[a]), rankdata(cols[b]))[0, 1])
            R[a, b] = R[b, a] = r
    fig, ax = plt.subplots(figsize=(0.6 * n + 3, 0.6 * n + 3))
    im = ax.imshow(R, cmap="viridis", vmin=0.9, vmax=1.0)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center",
                    color="white" if R[i, j] < 0.97 else "black", fontsize=8)
    cls = CLASSES[klass_index] if klass_index < len(CLASSES) else f"class{klass_index}"
    ax.set_title(f"{prefix} — P({cls}) rank-ρ between models")
    plt.colorbar(im, ax=ax, shrink=0.75, label="ρ")
    return _save(fig, _figpath(prefix, "rho_matrix"))


# --------------------------------------------------------------------------- bundle
def render_all_panels(y_true, y_pred, proba, *, labels=None, fold_scores=None, prefix="model") -> dict[str, Path]:
    """Render the standard per-experiment panel set. Returns {panel: png_path}."""
    out: dict[str, Path] = {}
    jobs = [
        ("confusion", lambda: confusion_matrix_panel(y_true, y_pred, labels=labels, prefix=prefix)),
        ("report", lambda: classification_report_panel(y_true, y_pred, labels=labels, prefix=prefix)),
        ("roc", lambda: roc_panel(y_true, proba, labels=labels, prefix=prefix)),
        ("ovo_auc", lambda: ovo_auc_panel(y_true, proba, labels=labels, prefix=prefix)),
        ("pr", lambda: pr_panel(y_true, proba, labels=labels, prefix=prefix)),
        ("class_balance", lambda: class_balance_panel(y_true, y_pred, labels=labels, prefix=prefix)),
    ]
    if fold_scores:
        jobs.append(("fold_scores", lambda: fold_score_boxplot(fold_scores, prefix=prefix)))
    for name, job in jobs:
        fig = job()
        if fig is not None:
            out[name] = _figpath(prefix, name)
    return out
