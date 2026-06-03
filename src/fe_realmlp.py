"""Rich feature engineering (Vladimir @yekenot's recipe) + the per-fold lgb-vs-realmlp harness.

v13 showed our RealMLP recipe works (+0.0023 over v7) but loses to the GBDT on bare features
(0.9609 vs 0.9657) — the +0.0073 to the public 0.968 is the FEATURE ENGINEERING the public 0.968
baseline includes and we omitted. This module reproduces that FE as TECHNIQUE (ideas are doctrine-OK
to adopt, like Deotte forking Vladimir — we are NOT blending notebook outputs):

- ratios g/redshift, i/redshift   - colors u-g, u-r
- every base numeric floored → a categorical (feeds the NN embedding layers)
- KBins(delta) at 100 & 500 quantile bins
- crossed categoricals alpha_cat×delta_cat, u_cat×z_cat  → the combos that get target-encoded
- per-class OOF target encoding of the combos (fit per-fold, leak-safe) — the strong NN feature

`race_oof` runs the per-fold loop training BOTH LGBM and RealMLP on identical folds, so we measure
(a) does FE close RealMLP to ~0.968, and (b) does the same FE also lift the GBDT. Static FE (factorize/
KBins — target-independent) is fit once on full train; only the target encoding is per-fold.

NN gets the full rich set (embeds the high-card derived cats); LGBM gets numerics + TE + the native
cats as int codes (the high-card numeric→cat / combos are NN-specific and only overfit a tree — v3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, MODEL_SEED

BASE_NUM = ["u", "g", "r", "i", "z", "redshift", "alpha", "delta"]
RATIOS = [("g", "redshift"), ("i", "redshift")]
COLOR_PAIRS = [("u", "g"), ("u", "r")]
KBINS = {"delta": [100, 500]}
COMBOS = [("alpha_cat_", "delta_cat_"), ("u_cat_", "z_cat_")]
NATIVE_CATS = ["spectral_type", "galaxy_population"]
SENTINEL = -9999.0


def build_rich_features(df: pd.DataFrame, *, fit: bool, state: dict | None = None):
    """Static (target-independent) FE. Fit factorize/KBins on train (fit=True), reuse on test.

    Returns (X, info, state). info = {cat_cols, native_cat_cols, combo_cols, num_cols}."""
    from sklearn.preprocessing import KBinsDiscretizer
    st = {} if state is None else state
    out = pd.DataFrame(index=df.index)

    base = {}
    for c in BASE_NUM:
        if c in df.columns:
            col = df[c].astype("float32").replace(SENTINEL, np.nan)
            base[c] = col
            out[c] = col
    for a, b in RATIOS:
        if a in base and b in base:
            out[f"_{a}_o_{b}"] = (base[a] / (base[b] + 1e-6)).astype("float32")
    for a, b in COLOR_PAIRS:
        if a in base and b in base:
            out[f"_{a}-{b}"] = (base[a] - base[b]).astype("float32")

    def _cat_from_codes(codes):
        return pd.Series(np.asarray(codes), index=df.index).astype("category")

    native = []
    for c in NATIVE_CATS:
        if c not in df.columns:
            continue
        native.append(c)
        if fit:
            codes, uniq = pd.factorize(df[c]); st[f"nat::{c}"] = uniq
        else:
            cmap = {v: i for i, v in enumerate(st[f"nat::{c}"])}
            codes = df[c].map(cmap).fillna(-1).astype("int64").to_numpy()
        out[c] = _cat_from_codes(codes)

    for c in BASE_NUM:
        if c not in base:
            continue
        floored = np.floor(base[c])
        if fit:
            codes, uniq = pd.factorize(floored); st[f"numcat::{c}"] = uniq
        else:
            cmap = {v: i for i, v in enumerate(st[f"numcat::{c}"])}
            codes = pd.Series(floored).map(cmap).fillna(-1).astype("int64").to_numpy()
        out[f"{c}_cat_"] = _cat_from_codes(codes)

    for c, bins_list in KBINS.items():
        if c not in base:
            continue
        for nb in bins_list:
            name = f"{c}_{nb}_qbin_"
            if fit:
                med = float(base[c].median())
                kb = KBinsDiscretizer(n_bins=nb, encode="ordinal", strategy="quantile", subsample=None)
                binned = kb.fit_transform(base[c].fillna(med).to_frame()).ravel().astype("int64")
                st[f"kb::{name}"] = (kb, med)
            else:
                kb, med = st[f"kb::{name}"]
                binned = kb.transform(base[c].fillna(med).to_frame()).ravel().astype("int64")
            out[name] = _cat_from_codes(binned)

    combo_cols = []
    for cols in COMBOS:
        if not all(cc in out.columns for cc in cols):
            continue
        name = "X".join(cols)
        combo_cols.append(name)
        s = out[cols[0]].astype(str)
        for cc in cols[1:]:
            s = s + "_" + out[cc].astype(str)
        if fit:
            codes, uniq = pd.factorize(s); st[f"combo::{name}"] = uniq
        else:
            cmap = {v: i for i, v in enumerate(st[f"combo::{name}"])}
            codes = s.map(cmap).fillna(-1).astype("int64").to_numpy()
        out[name] = _cat_from_codes(codes)

    cat_cols = [c for c in out.columns if str(out[c].dtype) == "category"]
    num_cols = [c for c in out.columns if c not in cat_cols]
    return out, {"cat_cols": cat_cols, "native_cat_cols": native,
                 "combo_cols": combo_cols, "num_cols": num_cols}, st


def _fold_te(X_tr, y_tr, evals: dict, combo_cols, n_classes, seed):
    """Per-class OOF target encoding of the combos. Leak-safe: internal CV on train, full-train on eval."""
    from sklearn.preprocessing import TargetEncoder
    enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=seed, target_type="multiclass")
    names = [f"_TE_{col}_c{cls}" for col in combo_cols for cls in range(n_classes)]
    tr = pd.DataFrame(enc.fit_transform(X_tr[combo_cols], y_tr).astype("float32"), columns=names, index=X_tr.index)
    ev = {k: pd.DataFrame(enc.transform(Xe[combo_cols]).astype("float32"), columns=names, index=Xe.index)
          for k, Xe in evals.items()}
    return tr, ev, names


def _lgb_fit_predict(X_tr, y_tr, evals: dict, seed):
    """LGBM on numerics + TE + native cats (category cols → int codes; high-card derived cats dropped)."""
    import lightgbm as lgb
    from .models import LGB_PARAMS
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    def prep(X):
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].cat.codes.astype("int32")
        return X
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "random_state": seed})
    m.fit(prep(X_tr), np.asarray(y_tr))
    return {k: m.predict_proba(prep(Xe)) for k, Xe in evals.items()}


def _cat_fit_predict(X_tr, y_tr, evals: dict, info, seed, gpu):
    """CatBoost on numerics + TE + categoricals NATIVE (ordered target stats — its strength), the
    raw high-card combos dropped (their per-class TE is already a feature). 3rd, algo-diverse leg."""
    from catboost import CatBoostClassifier
    keep = [c for c in X_tr.columns if c not in info["combo_cols"]]
    cat_feats = [c for c in keep if str(X_tr[c].dtype) == "category"]
    def prep(X):
        X = X[keep].copy()
        for c in cat_feats:
            X[c] = X[c].cat.codes.astype("int32")
        return X
    m = CatBoostClassifier(iterations=1500, learning_rate=0.05, depth=6, loss_function="MultiClass",
                           random_seed=seed, verbose=0, task_type=("GPU" if gpu else "CPU"))
    m.fit(prep(X_tr), np.asarray(y_tr), cat_features=cat_feats)
    return {k: m.predict_proba(prep(Xe)).astype("float32") for k, Xe in evals.items()}


def race_oof(Xdev, ydev, Xhold, Xte, info, n_folds, cfg, on_fold=None, seed=MODEL_SEED,
             with_cat=False, gpu=None, rm_seeds=None, with_tabm=False, tabm_cfg=None, extra_nn=None):
    """Per-fold loop: inject per-fold TE, train LGBM + RealMLP (+ CatBoost if with_cat) on identical
    folds. Returns dict with oof/hold/test per model + fva. NN sees the full rich set; LGBM sees
    numerics + TE + native cats (high-card derived cats are NN-specific — v3); CatBoost sees the cats
    native (its ordered-TS strength) minus the raw mega-combos.

    rm_seeds: None → single RealMLP per fold (exact v14/v15 behaviour). A list → SEED-BAG the RealMLP
    (train once per seed, average) for variance reduction on our strongest leg (v17)."""
    from .cv import stratified_folds
    from .realmlp import realmlp_fit_predict
    if gpu is None:
        try:
            import torch; gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
    nc = len(CLASSES); ydev = np.asarray(ydev)
    combo_cols, native, num_cols = info["combo_cols"], info["native_cat_cols"], info["num_cols"]
    lgb_static = num_cols + native  # + per-fold TE names

    z = lambda n: np.zeros((n, nc), "float32")
    R = {"lgb_oof": z(len(Xdev)), "lgb_hold": z(len(Xhold)), "lgb_test": z(len(Xte)),
         "rm_oof": z(len(Xdev)), "rm_hold": z(len(Xhold)), "rm_test": z(len(Xte)), "fva": []}
    if with_cat:
        R.update({"cat_oof": z(len(Xdev)), "cat_hold": z(len(Xhold)), "cat_test": z(len(Xte))})
    if with_tabm:
        R.update({"tabm_oof": z(len(Xdev)), "tabm_hold": z(len(Xhold)), "tabm_test": z(len(Xte))})
    extra_nn = extra_nn or []
    for e in extra_nn:
        R.update({f"{e['name']}_oof": z(len(Xdev)), f"{e['name']}_hold": z(len(Xhold)), f"{e['name']}_test": z(len(Xte))})

    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        Xtr, Xva = Xdev.iloc[tr], Xdev.iloc[va]
        te_tr, te_ev, te_names = _fold_te(Xtr, ydev[tr], {"va": Xva, "hold": Xhold, "test": Xte},
                                          combo_cols, nc, seed + i)
        cat = lambda X, t: pd.concat([X.reset_index(drop=True), t.reset_index(drop=True)], axis=1)
        rm_tr, rm_va = cat(Xtr, te_tr), cat(Xva, te_ev["va"])
        rm_hold, rm_te = cat(Xhold, te_ev["hold"]), cat(Xte, te_ev["test"])

        # RealMLP — full rich features (optionally seed-bagged: train per seed, average)
        seeds_f = [seed + i] if rm_seeds is None else [seed + i * 100 + s for s in rm_seeds]
        vp = np.zeros((len(va), nc), "float32"); hp = z(len(Xhold)); tp = z(len(Xte))
        for s in seeds_f:
            val_p, ev = realmlp_fit_predict(rm_tr, ydev[tr], rm_va, ydev[va],
                                            {"hold": rm_hold, "test": rm_te}, {**cfg, "seed": s})
            vp += val_p / len(seeds_f); hp += ev["hold"] / len(seeds_f); tp += ev["test"] / len(seeds_f)
        R["rm_oof"][va] = vp; R["rm_hold"] += hp / n_folds; R["rm_test"] += tp / n_folds

        # LGBM — numerics + TE + native cats
        lcols = lgb_static + te_names
        lp = _lgb_fit_predict(rm_tr[lcols], ydev[tr],
                              {"va": rm_va[lcols], "hold": rm_hold[lcols], "test": rm_te[lcols]}, seed + i)
        R["lgb_oof"][va] = lp["va"]; R["lgb_hold"] += lp["hold"] / n_folds; R["lgb_test"] += lp["test"] / n_folds

        # TabM — decorrelated NN paradigm (shared weight + rank-1 ensembling)
        if with_tabm:
            from .realmlp import TABM_CFG
            tval, tev = realmlp_fit_predict(rm_tr, ydev[tr], rm_va, ydev[va], {"hold": rm_hold, "test": rm_te},
                                            {**TABM_CFG, **(tabm_cfg or {}), "seed": seed + i})
            R["tabm_oof"][va] = tval; R["tabm_hold"] += tev["hold"] / n_folds; R["tabm_test"] += tev["test"] / n_folds

        # extra NN legs (fleet diversity) — each {"name", "base":"realmlp"|"tabm", "cfg":{...}}
        for e in extra_nn:
            from .realmlp import DEFAULT_CFG, TABM_CFG
            base = TABM_CFG if e.get("base") == "tabm" else DEFAULT_CFG
            ev_p, eev = realmlp_fit_predict(rm_tr, ydev[tr], rm_va, ydev[va], {"hold": rm_hold, "test": rm_te},
                                            {**base, **(e.get("cfg") or {}), "seed": seed + i})
            nm = e["name"]
            R[f"{nm}_oof"][va] = ev_p; R[f"{nm}_hold"] += eev["hold"] / n_folds; R[f"{nm}_test"] += eev["test"] / n_folds

        # CatBoost — native categoricals (3rd algo-diverse leg)
        if with_cat:
            cp = _cat_fit_predict(rm_tr, ydev[tr], {"va": rm_va, "hold": rm_hold, "test": rm_te},
                                  info, seed + i, gpu)
            R["cat_oof"][va] = cp["va"]; R["cat_hold"] += cp["hold"] / n_folds; R["cat_test"] += cp["test"] / n_folds
        R["fva"].append(va)
    return R


def single_oof(Xdev, ydev, Xhold, Xte, info, model, n_folds, on_fold=None, seed=MODEL_SEED):
    """Per-fold loop for ONE model — the parallel-fleet unit (one model per Kaggle kernel).

    model = {"type":"lgb"} | {"type":"nn","base":"realmlp"|"tabm","cfg":{...}}. Uses the SAME folds +
    per-fold TE as race_oof (fixed seeds) so OOFs from separate kernels ALIGN and can be stacked after.
    Returns (oof, hold, test)."""
    from .cv import stratified_folds
    from .realmlp import DEFAULT_CFG, TABM_CFG, realmlp_fit_predict
    nc = len(CLASSES); ydev = np.asarray(ydev)
    combo_cols, native, num_cols = info["combo_cols"], info["native_cat_cols"], info["num_cols"]
    z = lambda n: np.zeros((n, nc), "float32")
    oof, hold, test = z(len(Xdev)), z(len(Xhold)), z(len(Xte))
    for i, (tr, va) in enumerate(stratified_folds(ydev, n_folds), 1):
        if on_fold:
            on_fold(i, n_folds)
        Xtr, Xva = Xdev.iloc[tr], Xdev.iloc[va]
        te_tr, te_ev, te_names = _fold_te(Xtr, ydev[tr], {"va": Xva, "hold": Xhold, "test": Xte},
                                          combo_cols, nc, seed + i)
        cat = lambda X, t: pd.concat([X.reset_index(drop=True), t.reset_index(drop=True)], axis=1)
        rm_tr, rm_va = cat(Xtr, te_tr), cat(Xva, te_ev["va"])
        rm_hold, rm_te = cat(Xhold, te_ev["hold"]), cat(Xte, te_ev["test"])
        if model["type"] == "lgb":
            lcols = num_cols + native + te_names
            lp = _lgb_fit_predict(rm_tr[lcols], ydev[tr],
                                  {"va": rm_va[lcols], "hold": rm_hold[lcols], "test": rm_te[lcols]}, seed + i)
            oof[va] = lp["va"]; hold += lp["hold"] / n_folds; test += lp["test"] / n_folds
        else:  # nn (realmlp / tabm)
            base = TABM_CFG if model.get("base") == "tabm" else DEFAULT_CFG
            vp, ev = realmlp_fit_predict(rm_tr, ydev[tr], rm_va, ydev[va], {"hold": rm_hold, "test": rm_te},
                                         {**base, **(model.get("cfg") or {}), "seed": seed + i})
            oof[va] = vp; hold += ev["hold"] / n_folds; test += ev["test"] / n_folds
    return oof, hold, test
