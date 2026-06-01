"""v5 — decision-rule rework (single clean change vs v3: drop class_weight='balanced').

v4 diagnosis: errors are STRUCTURED (AUC 0.92), STAR recall (0.9538) is the drag, GALAXY↔STAR
confusion in redshift space. Prime suspect = double-handling imbalance (class_weight=balanced in
training AND post-hoc decision weights) → mis-placed boundary.

This experiment: train on the NATURAL distribution (class_weight=None → calibrated probs), then
let per-class decision-threshold tuning be the SOLE imbalance handling, optimized directly for
balanced accuracy. Isolates the class_weight effect (everything else = v3). Prints per-class
recall before/after tuning so we see whether STAR recall lifts.

Run locally:  PYTHONPATH=. python notebooks/06_v5_decision_rule.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, training, verdict
from src.diagnostics import error_cell_report
from src.diary import render_all
from src.features import build_features
from src.metrics import (competition_score, format_auc_report, multiclass_auc_report,
                         tune_class_weights, weighted_predict)
from src.models import lgb_oof
from src.observer import Experiment
from src.viz import render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape}")

    X_all, feats = build_features(train, use_extra=True)
    Xte, _ = build_features(test, use_extra=True)
    for c in feats:
        if str(X_all[c].dtype) == "category":
            Xte[c] = pd.Categorical(Xte[c], categories=X_all[c].cat.categories)
    y_all = train[TARGET].map(CLS2INT).to_numpy()

    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = X_all.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = X_all.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]

    goal_banner("v5", "decision-rule rework (class_weight=None + threshold opt)",
                "natural calibrated probs + direct bal-acc threshold tuning beats double-handled imbalance",
                best=0.9657)
    # THE single change: class_weight=None (natural distribution, calibrated probabilities)
    oof, hold_proba, test_proba, fold_va = lgb_oof(
        Xdev, ydev, Xhold, Xte, params={"class_weight": None},
        on_fold=lambda i, n: training(f"lgb fold {i}", i, n))

    ydev_names = INT2CLS[ydev]
    # before tuning: natural argmax (expected to under-serve STAR recall)
    argmax_pred = INT2CLS[np.argmax(oof, axis=1)]
    oof_argmax = competition_score(ydev_names, argmax_pred)
    print("--- per-class recall BEFORE tuning (natural argmax) ---")
    print(error_cell_report(ydev_names, argmax_pred, CLASSES))

    # the sole imbalance handling: optimize per-class decision weights for balanced accuracy
    weights, oof_tuned = tune_class_weights(oof, ydev, labels=list(range(len(CLASSES))))
    tuned_pred = weighted_predict(oof, weights, labels=CLASSES)
    print(f"\n[tune] OOF bal-acc: argmax {oof_argmax:.4f} → tuned {oof_tuned:.4f}  weights={np.round(weights,3)}")
    print("--- per-class recall AFTER tuning ---")
    print(error_cell_report(ydev_names, tuned_pred, CLASSES))

    per_fold = [competition_score(ydev_names[va], weighted_predict(oof[va], weights, labels=CLASSES))
                for va in fold_va]
    hold_pred = weighted_predict(hold_proba, weights, labels=CLASSES)
    hold_score = competition_score(INT2CLS[yhold], hold_pred)
    auc_rep = multiclass_auc_report(ydev_names, oof, labels=CLASSES)

    exp = Experiment.start(
        version="v5", parent="v3",
        hypothesis="Dropping class_weight='balanced' (calibrated natural probs) + direct per-class "
                   "threshold tuning for balanced accuracy lifts STAR recall and beats v3's "
                   "double-handled imbalance.",
        predicted_delta=0.0015, confidence="medium",
        feature_changes=[], pipeline_changes=["- class_weight=balanced (natural distribution)",
                                              "decision-threshold tuning = sole imbalance handling"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=oof_tuned, oof_score_per_fold=per_fold, holdout_score=hold_score,
               runtime_sec=time.time() - t0,
               extra={"oof_argmax": oof_argmax, "class_weights": list(np.round(weights, 4)),
                      "auc_ovo_macro": auc_rep["scalar"]})
    exp.commit(); render_all()

    panels = render_all_panels(ydev_names, tuned_pred, oof, labels=CLASSES,
                               fold_scores={"v5": per_fold}, prefix="v5")
    print("\n[viz] panels:", sorted(panels))
    print("[auc] " + format_auc_report(auc_rep).replace("\n", "\n      "))
    render_html(prefix="v5", title="S6E6 v5 — decision-rule rework")
    verdict("v5", hold_score, time.time() - t0, best=0.9657, oof=oof_tuned)

    PROBS.mkdir(parents=True, exist_ok=True); SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(test_proba, weights, labels=CLASSES)})
    sub.to_csv(SUBMISSIONS / "v5_decision.csv", index=False)
    print(f"[submission] v5_decision.csv  dist={sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
