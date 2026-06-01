"""Reusable model fitting. v1 inlined the fold loop; from v2 on, share it here."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, MODEL_SEED, N_FOLDS
from .cv import stratified_folds

LGB_PARAMS = dict(
    objective="multiclass", num_class=len(CLASSES), n_estimators=2000,
    learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, class_weight="balanced", random_state=MODEL_SEED, n_jobs=-1, verbose=-1,
)


def lgb_oof(Xdev: pd.DataFrame, ydev, Xhold: pd.DataFrame, Xte: pd.DataFrame,
            n_folds: int = N_FOLDS, params: dict | None = None, on_fold=None):
    """StratifiedKFold OOF on dev; fold-averaged proba for holdout + test.

    Returns (oof_proba, hold_proba, test_proba, fold_val_indices). `on_fold(i, n)` is an
    optional callback (e.g. dashboard.training) fired before each fold.
    """
    import lightgbm as lgb
    p = {**LGB_PARAMS, **(params or {})}
    oof = np.zeros((len(Xdev), len(CLASSES)))
    hold = np.zeros((len(Xhold), len(CLASSES)))
    test = np.zeros((len(Xte), len(CLASSES)))
    fold_va = []
    folds = list(stratified_folds(ydev, n_folds))
    for i, (tr, va) in enumerate(folds, 1):
        if on_fold:
            on_fold(i, n_folds)
        m = lgb.LGBMClassifier(**p)
        m.fit(Xdev.iloc[tr], np.asarray(ydev)[tr], eval_set=[(Xdev.iloc[va], np.asarray(ydev)[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(Xdev.iloc[va])
        hold += m.predict_proba(Xhold) / n_folds
        test += m.predict_proba(Xte) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va
