"""v9 — ceiling diagnostic: is 0.9664 the data's irreducible floor (label noise / overlap)?

The question that has haunted v4-v8. Three independent probes, all on a fresh LGBM OOF:
  1. label_noise_report (cleanlab confident learning) — estimated label-noise rate + which
     class-pair the inferred noise concentrates in. If ≈ our 3.5% error AND in GALAXY↔STAR,
     0.9664 IS the synthetic generator's label-noise floor.
  2. knn_bayes_error — local label disagreement among feature-space neighbours = irreducible
     overlap floor. If ≈ our error rate, the classes genuinely overlap in feature space.
  3. error_confidence_report — high-confidence-wrong (noise/systematic) vs low-confidence (overlap).

Converging verdict settles whether to bother with v10 (TabICLv2) or declare the real ceiling.
No submission, no model change. Logs the verdict to the diary.

Run locally:  PYTHONPATH=. python notebooks/10_v9_ceiling.py
"""
from __future__ import annotations

import time

import numpy as np

from src import data
from src.config import CLASSES, COORD_COLS, PHOTOMETRIC_BANDS, TARGET
from src.diagnostics import (error_confidence_report, knn_bayes_error, label_noise_report)
from src.diary import render_all
from src.features import build_features
from src.metrics import competition_score
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
            import pandas as pd
            Xte[c] = pd.Categorical(Xte[c], categories=X_all[c].cat.categories)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = X_all.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = X_all.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]

    print("[run] generating LGBM OOF for the probes…")
    oof, hold_proba, _, fold_va = lgb_oof(Xdev, ydev, Xhold, Xte)
    err_rate = float((np.argmax(oof, axis=1) != ydev).mean())
    print(f"[baseline] OOF argmax error rate = {err_rate*100:.2f}%  (balanced-acc {competition_score(ydev, np.argmax(oof,axis=1)):.4f})")

    # 1. label noise (cleanlab)
    txt_noise, noise_rate = label_noise_report(ydev, oof, CLASSES)
    print(txt_noise)
    # 2. Bayes floor (kNN on the FULL feature space — numeric + one-hot cats)
    txt_bayes, bayes = knn_bayes_error(Xdev, ydev)
    print(txt_bayes)
    # 3. confidence split
    txt_conf = error_confidence_report(ydev, oof)
    print(txt_conf)

    # converging verdict
    print("\n=== CEILING VERDICT ===")
    print(f"  our error rate {err_rate*100:.2f}%  |  cleanlab noise {noise_rate*100:.2f}%  |  kNN Bayes {bayes*100:.2f}%")
    close = abs(noise_rate - err_rate) < 0.01 and abs(bayes - err_rate) < 0.015
    if close:
        print("  → CONVERGE: error rate ≈ label-noise ≈ Bayes floor. 0.9664 is the DATA'S irreducible")
        print("    ceiling (synthetic label noise + feature-space overlap). No model can beat it. v10 unlikely to help.")
    else:
        print("  → DIVERGENT: error rate exceeds the noise/Bayes floors → residual structure may be")
        print("    extractable. v10 (TabICLv2, new paradigm) is justified.")

    exp = Experiment.start(
        version="v9", parent="v8",
        hypothesis="DIAGNOSTIC: is 0.9664 the data's irreducible floor? cleanlab label-noise + kNN "
                   "Bayes-error + error-confidence converge to settle the ceiling.",
        predicted_delta=0.0, confidence="medium",
        feature_changes=[], pipeline_changes=["cleanlab confident-learning + kNN Bayes-floor + confidence probe"],
        cloud_or_local="cloud" if not synthetic else "local")
    s = competition_score(ydev, np.argmax(oof, axis=1))
    per_fold = [competition_score(ydev[va], np.argmax(oof[va], axis=1)) for va in fold_va]
    exp.record(oof_score_mean=s, oof_score_per_fold=per_fold,
               holdout_score=competition_score(yhold, np.argmax(hold_proba, axis=1)),
               runtime_sec=time.time() - t0,
               extra={"err_rate": err_rate, "cleanlab_noise_rate": noise_rate, "knn_bayes": bayes,
                      "converged_ceiling": bool(close)})
    exp.note(f"err {err_rate:.4f} | cleanlab {noise_rate:.4f} | bayes {bayes:.4f} | converged={close}")
    exp.commit(); render_all()
    print(f"\n[verdict logged] converged_ceiling={close}")


if __name__ == "__main__":
    main()
