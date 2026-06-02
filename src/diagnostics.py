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


def label_noise_report(y_int, pred_probs, classes: list[str]) -> tuple[str, float]:
    """Cleanlab confident learning — is the 'error' actually LABEL NOISE in the synthetic data?

    Uses OOF probs we already have. Returns (text, estimated_noise_rate). If the flagged rate
    ≈ our error rate AND concentrates in GALAXY↔STAR, then 0.9664 IS the synthetic noise floor.
    """
    from cleanlab.count import compute_confident_joint
    from cleanlab.filter import find_label_issues

    y_int = np.asarray(y_int)
    issues = find_label_issues(y_int, np.asarray(pred_probs, dtype=float),
                               return_indices_ranked_by=None)  # boolean mask
    n_iss = int(np.asarray(issues).sum())
    rate = n_iss / len(y_int)
    cj = compute_confident_joint(y_int, np.asarray(pred_probs, dtype=float))  # (K,K): given×true
    lines = [f"[label-noise] cleanlab flags {n_iss:,} likely-mislabeled rows = {rate*100:.2f}% "
             f"(compare to our OOF error rate ~3.5%)"]
    lines.append("   confident joint (rows=given label, cols=cleanlab's inferred true):")
    lines.append("        " + "".join(f"{c:>9}" for c in classes))
    for i, c in enumerate(classes):
        lines.append(f"   {c:<6}" + "".join(f"{int(cj[i, j]):>9,}" for j in range(len(classes))))
    # which off-diagonal pair carries the most inferred noise
    off = [(classes[i], classes[j], int(cj[i, j])) for i in range(len(classes))
           for j in range(len(classes)) if i != j]
    off.sort(key=lambda t: -t[2])
    lines.append("   top inferred-noise cells (given→true): " +
                 ", ".join(f"{a}→{b}:{n:,}" for a, b, n in off[:3]))
    return "\n".join(lines), rate


def knn_bayes_error(X_num: pd.DataFrame, y_int, k: int = 10, sample: int = 50000,
                    seed: int = 42) -> tuple[str, float]:
    """kNN local label-disagreement = an estimate of the irreducible Bayes-error floor.

    Fraction of each point's k nearest neighbours (in standardized numeric feature space) with
    a different label. If ≈ our error rate, 0.9664 is the feature-space overlap floor.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y_int)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(sample, len(y)), replace=False)
    Xs = StandardScaler().fit_transform(X_num.iloc[idx].fillna(X_num.median()).to_numpy())
    ys = y[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
    _, nbr = nn.kneighbors(Xs)
    nbr = nbr[:, 1:]                                   # drop self
    disagree = (ys[nbr] != ys[:, None]).mean()
    txt = (f"[bayes-floor] kNN(k={k}) local label-disagreement = {disagree*100:.2f}% "
           f"on {len(idx):,}-row sample → irreducible error floor estimate")
    return txt, float(disagree)


def error_confidence_report(y_int, oof_proba) -> str:
    """Are errors high-confidence-wrong (noise/systematic) or low-confidence (genuine overlap)?"""
    y = np.asarray(y_int)
    P = np.asarray(oof_proba, dtype=float)
    pred = np.argmax(P, axis=1)
    conf = P.max(axis=1)
    wrong = pred != y
    if wrong.sum() == 0:
        return "[confidence] no errors."
    hi = (conf[wrong] > 0.9).mean()
    return (f"[confidence] errors: mean confidence {conf[wrong].mean():.3f} "
            f"(correct {conf[~wrong].mean():.3f}); {hi*100:.1f}% of errors are high-confidence (>0.9). "
            f"Low conf → genuine overlap (irreducible); high conf → label noise / systematic bias.")


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
