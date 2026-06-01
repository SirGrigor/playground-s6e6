"""Error-cell + error-predictability diagnostics — name the missing signal at a plateau.

The classification analog of the signal_hunt / residual diagnostic (feedback_signal-hunt-
before-in-paradigm, feedback_discovery-first). Two questions, both printed:

1. error_cell_report: WHERE are the errors? Per-class recall (the balanced-accuracy
   components) + the ranked (true→pred) confusion cells carrying the error mass.
2. error_predictability: are the errors STRUCTURED or NOISE? Fit a fresh GBDT to predict
   "did the model get this row wrong?" from the features, OOF AUC. >0.55 → structure exists,
   top features name the FE to build. ~0.50 → errors are irreducible → the ceiling is real,
   stop adding features and pivot (ensemble / decision rule).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def error_cell_report(y_true, y_pred, classes: list[str]) -> str:
    """Confusion (counts + row-normalized), per-class recall, ranked error cells."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rowsum = cm.sum(axis=1, keepdims=True).clip(1, None)
    recall = (cm.diagonal() / rowsum.ravel())
    bal = balanced_accuracy_score(y_true, y_pred)

    lines = [f"[error-cell] balanced-acc {bal:.4f} = mean per-class recall:"]
    for c, r, n in zip(classes, recall, rowsum.ravel()):
        lines.append(f"   {c:<8} recall {r:.4f}  (support {int(n):,})  ← {'DRAG' if r < bal else 'ok'}")
    lines.append("   confusion (rows=true, cols=pred):")
    header = "        " + "".join(f"{c:>10}" for c in classes)
    lines.append(header)
    for i, c in enumerate(classes):
        lines.append(f"   {c:<6}" + "".join(f"{cm[i, j]:>10,}" for j in range(len(classes))))
    # ranked off-diagonal cells by error mass
    cells = [(classes[i], classes[j], int(cm[i, j]), cm[i, j] / rowsum[i, 0])
             for i in range(len(classes)) for j in range(len(classes)) if i != j and cm[i, j] > 0]
    cells.sort(key=lambda t: -t[2])
    lines.append("   top error cells (true→pred, count, row-rate):")
    for tc, pc, n, rate in cells[:6]:
        lines.append(f"     {tc}→{pc}: {n:,} ({rate*100:.2f}% of {tc})")
    return "\n".join(lines)


@dataclass
class ErrorPredictability:
    auc: float
    top_features: list[tuple[str, float]]

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.auc):
            return "INSUFFICIENT errors to judge (too few/many misclassified rows) — rerun on full data."
        if self.auc < 0.52:
            return ("NOISE — errors are not predictable from features. Local CEILING is real; "
                    "stop adding features, pivot to ensemble / decision-rule / different signal source.")
        if self.auc < 0.58:
            return "WEAK structure — minor exploitable pattern; check the top features below."
        return "STRUCTURED — errors ARE predictable; the top features name the FE to build."

    def __str__(self) -> str:
        top = "  ".join(f"{f}({g:.0f})" for f, g in self.top_features[:8])
        return f"[error-predict] wrong-flag OOF AUC = {self.auc:.4f}\n   → {self.verdict}\n   top: {top}"


def error_predictability(X: pd.DataFrame, y_true_int, oof_proba, n_splits: int = 5,
                         seed: int = 42) -> ErrorPredictability:
    """Fit a fresh GBDT to predict the model's own errors. OOF AUC = is there structure?"""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    wrong = (np.argmax(oof_proba, axis=1) != np.asarray(y_true_int)).astype(int)
    if wrong.sum() < 50 or wrong.mean() > 0.99:
        return ErrorPredictability(float("nan"), [])
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(X))
    imp = np.zeros(X.shape[1])
    for tr, va in skf.split(X, wrong):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                               min_child_samples=100, verbose=-1, random_state=seed)
        m.fit(X.iloc[tr], wrong[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        imp += m.booster_.feature_importance(importance_type="gain") / n_splits
    auc = roc_auc_score(wrong, oof)
    order = np.argsort(imp)[::-1]
    top = [(X.columns[i], float(imp[i])) for i in order]
    return ErrorPredictability(float(auc), top)
