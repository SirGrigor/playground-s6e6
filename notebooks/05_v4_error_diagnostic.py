"""v4 — error-cell + error-predictability diagnostic (NOT a model change).

We plateaued at ~0.9655 (v1=v3; extra cols redundant). Before building more, name the gap:
WHERE the errors are (per-class recall + confusion cells) and WHETHER they're structured
(can a fresh GBDT predict the model's own mistakes from the features?) or irreducible noise.
Everything PRINTS so the verdict lands in the run output. No submission; records the verdict
to the diary so the decision is logged.

Run locally:  PYTHONPATH=. python notebooks/05_v4_error_diagnostic.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.config import CLASSES, TARGET
from src.diagnostics import error_cell_report, error_predictability
from src.diary import render_all
from src.features import build_features
from src.metrics import competition_score, multiclass_auc_report
from src.models import lgb_oof
from src.observer import Experiment

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

    print("[run] generating OOF for the diagnostic (LGBM, 5-fold)…")
    oof, hold_proba, _, fold_va = lgb_oof(Xdev, ydev, Xhold, Xte)
    oof_pred = INT2CLS[np.argmax(oof, axis=1)]
    ydev_names = INT2CLS[ydev]

    # --- the diagnostic ---
    print(error_cell_report(ydev_names, oof_pred, CLASSES))
    ep = error_predictability(Xdev, ydev, oof)
    print(ep)

    # log the verdict to the diary (score unchanged — this is a probe, predicted_delta 0)
    oof_score = competition_score(ydev_names, oof_pred)
    hold_score = competition_score(INT2CLS[yhold], INT2CLS[np.argmax(hold_proba, axis=1)])
    per_fold = [competition_score(ydev_names[va], oof_pred[va]) for va in fold_va]
    exp = Experiment.start(
        version="v4", parent="v3",
        hypothesis="DIAGNOSTIC: are the ~0.9655 plateau errors structured (feature pattern → "
                   "exploitable FE) or irreducible noise? Names the next lever.",
        predicted_delta=0.0, confidence="medium",
        feature_changes=[], pipeline_changes=["error-cell + error-predictability probe (no model change)"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=oof_score, oof_score_per_fold=per_fold, holdout_score=hold_score,
               runtime_sec=time.time() - t0,
               extra={"error_predict_auc": ep.auc, "verdict": ep.verdict,
                      "top_error_features": [f for f, _ in ep.top_features[:8]],
                      "auc_ovo_macro": multiclass_auc_report(ydev_names, oof, labels=CLASSES)["scalar"]})
    exp.note(f"error-predict AUC={ep.auc:.4f} → {ep.verdict}")
    exp.commit(); render_all()
    print(f"\n[verdict logged to diary] error-predict AUC {ep.auc:.4f}")


if __name__ == "__main__":
    main()
