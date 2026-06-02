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


def _cat_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if str(X[c].dtype) == "category"]


def cat_oof(Xdev, ydev, Xhold, Xte, n_folds: int = N_FOLDS, on_fold=None):
    """CatBoost OOF (natural distribution — v5 showed decision-rule is post-hoc). Native
    categorical via cat_features; category cols → string so CatBoost ingests them + NaN."""
    from catboost import CatBoostClassifier, Pool
    cats = _cat_cols(Xdev)

    def prep(df):
        d = df.copy()
        for c in cats:
            d[c] = d[c].astype("string").fillna("NA")
        return d
    Xd, Xh, Xt = prep(Xdev), prep(Xhold), prep(Xte)
    oof = np.zeros((len(Xd), len(CLASSES)))
    hold = np.zeros((len(Xh), len(CLASSES)))
    test = np.zeros((len(Xt), len(CLASSES)))
    fold_va = []
    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        m = CatBoostClassifier(loss_function="MultiClass", iterations=2000, learning_rate=0.05,
                               depth=8, l2_leaf_reg=3.0, random_seed=MODEL_SEED, verbose=0,
                               early_stopping_rounds=100, cat_features=cats)
        m.fit(Pool(Xd.iloc[tr], np.asarray(ydev)[tr], cat_features=cats),
              eval_set=Pool(Xd.iloc[va], np.asarray(ydev)[va], cat_features=cats))
        oof[va] = m.predict_proba(Xd.iloc[va])
        hold += m.predict_proba(Xh) / n_folds
        test += m.predict_proba(Xt) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va


def tabicl_oof(Xdev, ydev, Xhold, Xte, n_folds: int = N_FOLDS, on_fold=None,
               train_cap: int = 100_000):
    """TabICLv2 (tabular foundation model) OOF — the NEW-paradigm, scale-appropriate non-GBDT.

    In-context-learning transformer (auto-downloads the v2 checkpoint). Genuinely different from
    GBDT *and* RealMLP → the strong-AND-decorrelated member we lacked. Categoricals → integer
    codes (TabICL takes numeric); numeric NaN → median. `train_cap` stratified-subsamples the
    fold-train context (in-context models need plenty but not all rows; caps T4 memory)."""
    from tabicl import TabICLClassifier
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    cats = _cat_cols(Xdev)
    num = [c for c in Xdev.columns if c not in cats]
    medians = Xdev[num].astype(float).median()
    # align categorical integer codes across splits
    cat_maps = {c: {v: i for i, v in enumerate(Xdev[c].astype("string").fillna("NA").unique())} for c in cats}

    def prep(df):
        d = df.copy()
        for c in cats:
            s = d[c].astype("string").fillna("NA")
            d[c] = s.map(cat_maps[c]).fillna(-1).astype(int)
        for c in num:
            d[c] = d[c].astype(float).fillna(medians[c])
        return d.to_numpy(dtype=float)
    Xd, Xh, Xt = prep(Xdev), prep(Xhold), prep(Xte)
    y = np.asarray(ydev)

    oof = np.zeros((len(Xd), len(CLASSES)))
    hold = np.zeros((len(Xh), len(CLASSES)))
    test = np.zeros((len(Xt), len(CLASSES)))
    fold_va = []
    rng = np.random.default_rng(MODEL_SEED)
    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        if train_cap and len(tr) > train_cap:        # stratified subsample of fold-train context
            from sklearn.model_selection import train_test_split
            tr, _ = train_test_split(tr, train_size=train_cap, random_state=MODEL_SEED, stratify=y[tr])
        m = TabICLClassifier(device=device, random_state=MODEL_SEED, n_jobs=-1)
        m.fit(Xd[tr], y[tr])
        oof[va] = m.predict_proba(Xd[va])
        hold += m.predict_proba(Xh) / n_folds
        test += m.predict_proba(Xt) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va


def nn_oof(Xdev, ydev, Xhold, Xte, n_folds: int = N_FOLDS, on_fold=None):
    """RealMLP (pytabkit) OOF — the NON-GBDT paradigm for decorrelation.

    Neural net with PLR embeddings + learned categorical embeddings → genuinely different
    decision boundary than trees → (hopefully) ρ<0.9 vs the GBDTs → ensemble lift. GPU if
    available. Categoricals via cat_col_names (cast to string); numeric NaN filled with the
    dev-fold median (RealMLP's scaling transforms don't tolerate NaN like GBDTs do).
    """
    from pytabkit import RealMLP_TD_Classifier
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    cats = _cat_cols(Xdev)
    num = [c for c in Xdev.columns if c not in cats]
    medians = Xdev[num].astype(float).median()

    def prep(df):
        d = df.copy()
        for c in cats:
            d[c] = d[c].astype("string").fillna("NA").astype(object)
        for c in num:
            d[c] = d[c].astype(float).fillna(medians[c])
        return d
    Xd, Xh, Xt = prep(Xdev), prep(Xhold), prep(Xte)

    oof = np.zeros((len(Xd), len(CLASSES)))
    hold = np.zeros((len(Xh), len(CLASSES)))
    test = np.zeros((len(Xt), len(CLASSES)))
    fold_va = []
    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        m = RealMLP_TD_Classifier(device=device, random_state=MODEL_SEED, n_cv=1, verbosity=0)
        m.fit(Xd.iloc[tr], np.asarray(ydev)[tr], cat_col_names=cats)
        oof[va] = m.predict_proba(Xd.iloc[va])
        hold += m.predict_proba(Xh) / n_folds
        test += m.predict_proba(Xt) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va


def xgb_oof(Xdev, ydev, Xhold, Xte, n_folds: int = N_FOLDS, on_fold=None, seed: int = MODEL_SEED):
    """XGBoost OOF (natural distribution). enable_categorical uses pandas category dtype.
    `seed` varies the model RNG (folds stay fixed) for seed-bagging."""
    import xgboost as xgb
    oof = np.zeros((len(Xdev), len(CLASSES)))
    hold = np.zeros((len(Xhold), len(CLASSES)))
    test = np.zeros((len(Xte), len(CLASSES)))
    fold_va = []
    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        m = xgb.XGBClassifier(objective="multi:softprob", num_class=len(CLASSES),
                              n_estimators=2000, learning_rate=0.05, max_depth=8, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                              enable_categorical=True, eval_metric="mlogloss",
                              early_stopping_rounds=100, random_state=seed, n_jobs=-1)
        m.fit(Xdev.iloc[tr], np.asarray(ydev)[tr],
              eval_set=[(Xdev.iloc[va], np.asarray(ydev)[va])], verbose=False)
        oof[va] = m.predict_proba(Xdev.iloc[va])
        hold += m.predict_proba(Xhold) / n_folds
        test += m.predict_proba(Xte) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va
