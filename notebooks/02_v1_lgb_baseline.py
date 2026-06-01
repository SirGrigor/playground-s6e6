"""v1 — LGBM multiclass baseline (balanced-accuracy native).

Pipeline: features (colors + sentinel→NaN) → sacred 20% holdout → StratifiedKFold(5) OOF on
the dev set → LGBM per fold (class_weight balanced, early stopping) → per-class decision-weight
tuning on OOF (the balanced-accuracy lever; argmax is suboptimal) → honest holdout score →
test submission. Everything scored through metrics.competition_score (balanced accuracy).

Instrumented: observer diary (hypothesis + predicted Δ), live dashboard scoreboard + verdict,
full viz panels on OOF, HTML dashboard. Runs on Colab via bootstrap; locally it falls back to
the synthetic SDSS-shaped frame so the pipeline is smoke-testable with no download.

Run locally:  PYTHONPATH=. python notebooks/02_v1_lgb_baseline.py
"""
from __future__ import annotations

import time

import numpy as np

from src import data
from src.config import CLASSES, ID, MODEL_SEED, N_FOLDS, PROBS, SUBMISSIONS, TARGET
from src.cv import stratified_folds
from src.dashboard import goal_banner, render_html, scoreboard, training, verdict
from src.diary import render_all
from src.features import build_features
from src.metrics import (competition_score, format_auc_report, multiclass_auc_report,
                         tune_class_weights, weighted_predict)
from src.observer import Experiment
from src.viz import render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
LGB_PARAMS = dict(
    objective="multiclass", num_class=len(CLASSES), n_estimators=2000,
    learning_rate=0.05, num_leaves=63, subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, class_weight="balanced", random_state=MODEL_SEED, n_jobs=-1, verbose=-1,
)


def _load():
    raw = data.load_raw()
    if raw is not None:
        train, test = raw
        return train, test, False
    print("⚠ data/raw empty — synthetic SDSS-shaped fallback (smoke test only).")
    full = data.synthetic_fallback(n=8000)
    train = full.iloc[:6000].reset_index(drop=True)
    test = full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True)
    return train, test, True


def _fit_fold(Xtr, ytr, Xva, yva):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(**LGB_PARAMS)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    return m


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape} test={test.shape}")

    X_all, feats = build_features(train)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xte, _ = build_features(test)
    print(f"[features] {len(feats)} cols: {feats}")

    exp = Experiment.start(
        version="v1",
        parent=None,
        hypothesis="LGBM on redshift+bands+colors with class-balanced weights and OOF-tuned "
                   "per-class decision weights beats the redshift baseline on balanced accuracy.",
        predicted_delta=0.0,  # no parent — baseline anchor
        confidence="high",
        feature_changes=["+ redshift", "+ bands(u,g,r,i,z)", "+ colors(u-g,g-r,r-i,i-z)", "+ alpha,delta"],
        pipeline_changes=["LGBM multiclass", "StratifiedKFold(5)", "sacred 20% holdout",
                          "class_weight=balanced", "OOF per-class decision-weight tuning"],
        cloud_or_local="cloud" if not synthetic else "local",
    )

    # sacred holdout (stratified) carved off train; CV runs on the dev remainder
    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = X_all.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = X_all.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]
    print(f"[split] dev={len(dev_idx)} holdout={len(hold_idx)}")

    oof = np.zeros((len(Xdev), len(CLASSES)))
    hold_proba = np.zeros((len(Xhold), len(CLASSES)))
    test_proba = np.zeros((len(Xte), len(CLASSES)))
    fold_scores: list[float] = []

    goal_banner("v1", "LGBM baseline (balanced-acc, OOF-tuned weights)",
                "redshift+bands+colors carry the 3-class signal")

    for i, (tr, va) in enumerate(stratified_folds(ydev, N_FOLDS), 1):
        training(f"lgb fold {i}", i, N_FOLDS)
        m = _fit_fold(Xdev.iloc[tr], ydev[tr], Xdev.iloc[va], ydev[va])
        oof[va] = m.predict_proba(Xdev.iloc[va])
        hold_proba += m.predict_proba(Xhold) / N_FOLDS
        test_proba += m.predict_proba(Xte) / N_FOLDS
        # per-fold balanced accuracy (plain argmax) — feeds the fold-collapse detector
        fold_scores.append(competition_score(ydev[va], np.argmax(oof[va], axis=1)))

    # plain-argmax OOF vs tuned: the balanced-accuracy lever
    oof_argmax = competition_score(ydev, np.argmax(oof, axis=1))
    weights, oof_tuned = tune_class_weights(oof, ydev, labels=list(range(len(CLASSES))))
    print(f"[tune] OOF balanced-acc: argmax {oof_argmax:.4f} → weighted {oof_tuned:.4f}  weights={np.round(weights,3)}")

    # honest holdout with the OOF-tuned weights
    hold_pred = weighted_predict(hold_proba, weights, labels=list(range(len(CLASSES))))
    hold_score = competition_score(yhold, hold_pred)

    # AUC diagnostics computed ONCE in class-name space (proba cols align to CLASSES)
    names = np.asarray(CLASSES)
    ydev_names = names[ydev]
    auc_rep = multiclass_auc_report(ydev_names, oof, labels=CLASSES)

    exp.record(oof_score_mean=oof_tuned, oof_score_per_fold=fold_scores,
               holdout_score=hold_score, runtime_sec=time.time() - t0,
               extra={"oof_argmax": oof_argmax, "class_weights": list(np.round(weights, 4)),
                      "auc_ovo_macro": auc_rep["scalar"]})
    exp.commit()
    render_all()

    panels = render_all_panels(ydev_names, weighted_predict(oof, weights, labels=CLASSES), oof,
                               labels=CLASSES, fold_scores={"v1": fold_scores}, prefix="v1")
    print("[viz] panels:", sorted(panels))
    print("[auc] " + format_auc_report(auc_rep).replace("\n", "\n      "))
    render_html(prefix="v1", title="S6E6 v1 — LGBM baseline")
    verdict("v1", hold_score, time.time() - t0, oof=oof_tuned)

    # submission (test labels via tuned weights)
    import pandas as pd
    PROBS.mkdir(parents=True, exist_ok=True); SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(test_proba, weights, labels=CLASSES)})
    sub_path = SUBMISSIONS / "v1_lgb.csv"
    sub.to_csv(sub_path, index=False)
    np.save(PROBS / "v1_test_proba.npy", test_proba)
    print(f"[submission] {sub_path}  ({len(sub)} rows)  dist={sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
