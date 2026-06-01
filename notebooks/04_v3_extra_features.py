"""v3 — use the Playground-added spectral_type + galaxy_population columns.

The v2 recon revealed train has `spectral_type` and `galaxy_population` (NOT in SDSS17
original) that v1 silently dropped. They're almost certainly strong class signals. This
experiment: (1) EDA their relationship to class (distribution + nullity per class — could be
near-perfect indicators); (2) retrain the v1 pipeline with them added (LGBM-native categorical);
(3) score vs v1's 0.9655 with the per-class threshold tuning.

Run locally:  PYTHONPATH=. python notebooks/04_v3_extra_features.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.config import CLASSES, EXTRA_COLS, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, training, verdict
from src.diary import render_all
from src.features import build_features
from src.metrics import (competition_score, format_auc_report, multiclass_auc_report,
                         tune_class_weights, weighted_predict)
from src.models import lgb_oof
from src.observer import Experiment
from src.viz import render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)


def eda_extra(train: pd.DataFrame) -> None:
    """Show how each extra column relates to class — distribution + nullity per class."""
    for c in EXTRA_COLS:
        if c not in train.columns:
            print(f"[eda:{c}] ABSENT from train"); continue
        col = train[c]
        print(f"[eda:{c}] dtype={col.dtype} cardinality={col.nunique(dropna=True)} "
              f"overall_null={col.isna().mean()*100:.1f}%")
        # null rate per class — reveals "only stars have a spectral_type" type structure
        null_by_cls = train.groupby(TARGET, observed=True)[c].apply(lambda s: s.isna().mean())
        print("   null-rate by class: " + ", ".join(f"{k} {v*100:.1f}%" for k, v in null_by_cls.items()))
        # if low cardinality, show the cross-tab (which value → which class)
        if col.nunique(dropna=True) <= 20:
            ct = pd.crosstab(train[c], train[TARGET])
            print("   value × class:\n" + ct.to_string().replace("\n", "\n   "))


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return (full.iloc[:6000].reset_index(drop=True),
            full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True)


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape} test={test.shape}")
    eda_extra(train)

    X_all, feats = build_features(train, use_extra=True)
    Xte, _ = build_features(test, use_extra=True)
    # align category dtypes across train/test so LGBM sees a consistent encoding
    for c in feats:
        if str(X_all[c].dtype) == "category":
            cats = X_all[c].cat.categories
            Xte[c] = pd.Categorical(Xte[c], categories=cats)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    print(f"[features] {len(feats)} cols (incl extras): {feats}")

    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = X_all.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = X_all.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]

    goal_banner("v3", "add spectral_type + galaxy_population", "Playground extras carry strong class signal",
                best=0.9655)
    oof, hold_proba, test_proba, fold_va = lgb_oof(
        Xdev, ydev, Xhold, Xte, on_fold=lambda i, n: training(f"lgb fold {i}", i, n))

    oof_argmax = competition_score(ydev, np.argmax(oof, axis=1))
    weights, oof_tuned = tune_class_weights(oof, ydev, labels=list(range(len(CLASSES))))
    per_fold = [competition_score(ydev[va], np.argmax(oof[va], axis=1)) for va in fold_va]
    hold_pred = weighted_predict(hold_proba, weights, labels=list(range(len(CLASSES))))
    hold_score = competition_score(yhold, hold_pred)
    print(f"[tune] OOF bal-acc: argmax {oof_argmax:.4f} → weighted {oof_tuned:.4f}  weights={np.round(weights,3)}")

    names = INT2CLS
    ydev_names = names[ydev]
    auc_rep = multiclass_auc_report(ydev_names, oof, labels=CLASSES)

    exp = Experiment.start(
        version="v3", parent="v1",
        hypothesis="Adding spectral_type + galaxy_population (LGBM-native categorical) lifts "
                   "balanced accuracy well above v1 — they are near-direct class indicators.",
        predicted_delta=0.005, confidence="high",
        feature_changes=["+ spectral_type (cat)", "+ galaxy_population (cat)"],
        pipeline_changes=["LGBM native categorical"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=oof_tuned, oof_score_per_fold=per_fold, holdout_score=hold_score,
               runtime_sec=time.time() - t0,
               extra={"oof_argmax": oof_argmax, "class_weights": list(np.round(weights, 4)),
                      "auc_ovo_macro": auc_rep["scalar"]})
    exp.commit(); render_all()

    panels = render_all_panels(ydev_names, weighted_predict(oof, weights, labels=CLASSES), oof,
                               labels=CLASSES, fold_scores={"v3": per_fold}, prefix="v3")
    print("[viz] panels:", sorted(panels))
    print("[auc] " + format_auc_report(auc_rep).replace("\n", "\n      "))
    render_html(prefix="v3", title="S6E6 v3 — + spectral_type + galaxy_population")
    verdict("v3", hold_score, time.time() - t0, best=0.9655, oof=oof_tuned)

    PROBS.mkdir(parents=True, exist_ok=True); SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(test_proba, weights, labels=CLASSES)})
    sub.to_csv(SUBMISSIONS / "v3_extra.csv", index=False)
    print(f"[submission] v3_extra.csv  dist={sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
