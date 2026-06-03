"""CV'd seed-bagged logistic-regression stacker — the robust meta-layer (our own impl of the technique).

Our v15/v16 blend fit a SINGLE LogReg on the full OOF → it over-fit (v16 regressed on LB despite a
higher holdout). The fix (from Deotte's GPU-LogReg-stacker pattern, reimplemented as technique, not his
outputs): make the META-MODEL itself cross-validated + seed-bagged, on **log-odds** features, with a
**class-weighted** objective for balanced accuracy. Differences from v15's `_logreg_stack`:
  - log-odds `log(p/(1-p))` (clipped ±30), not log(p)
  - 5-seed × 5-fold CV'd meta-model (25 fits averaged) → meta-OOF is out-of-fold → resists meta-overfit
  - class_weight='balanced' + stronger C (0.1)

`cv_logreg_stack(oofs, y, tests)` → (oof_stack_proba, test_stack_proba, cv_balanced_acc). Tiny linear
model → sklearn/CPU is plenty (no GPU needed). Reused by the conveyor (stack harvested OOFs) and re-blends.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-15
LOGIT_CLIP = 30.0


def logodds(P) -> np.ndarray:
    P = np.clip(np.asarray(P, dtype=np.float64), EPS, 1.0 - EPS)
    return np.clip(np.log(P / (1.0 - P)), -LOGIT_CLIP, LOGIT_CLIP).astype(np.float32)


def cv_logreg_stack(oofs: dict, y_int, apply: dict, *, n_folds: int = 5, n_seeds: int = 5,
                    C: float = 0.1, class_weight="balanced", seed0: int = 42):
    """Seed-bagged, CV'd, class-weighted log-odds LogReg stacker.

    oofs[leg] = (N, n_classes) dev-OOF probabilities (the meta-features). `apply` = named sets to
    transform with the fitted stacker, each a dict {leg: (rows, n_classes)} over the SAME legs —
    e.g. {"hold": {leg: holdout_preds}, "test": {leg: test_preds}}.

    Returns (oof_stack, {name: stacked_preds}, cv_balanced_acc). The meta-model is trained out-of-fold
    so its OOF score is an honest estimate (the property our v15 single-fit stacker lacked)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    names = list(oofs)
    y = np.asarray(y_int)
    N, nc = oofs[names[0]].shape
    Xo = np.concatenate([logodds(oofs[n]) for n in names], axis=1)
    Xa = {k: np.concatenate([logodds(s[n]) for n in names], axis=1) for k, s in apply.items()}

    oof_sum = np.zeros((N, nc))
    a_sum = {k: np.zeros((len(X), nc)) for k, X in Xa.items()}
    for s in range(seed0, seed0 + n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
        for tr, va in skf.split(Xo, y):
            clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=3000)
            clf.fit(Xo[tr], y[tr])
            oof_sum[va] += clf.predict_proba(Xo[va])           # each row scored once per seed
            for k in Xa:
                a_sum[k] += clf.predict_proba(Xa[k]) / n_folds  # fold-averaged per seed
    oof_stack = (oof_sum / n_seeds).astype(np.float32)
    a_stack = {k: (v / n_seeds).astype(np.float32) for k, v in a_sum.items()}
    cv = float(balanced_accuracy_score(y, np.argmax(oof_stack, axis=1)))
    return oof_stack, a_stack, cv
