"""v6 — model zoo (LGBM + CatBoost + XGBoost) + ensemble.

v2/v3/v5 eliminated leak, obvious features, and the decision rule (Pareto-exhausted). The only
remaining lever is better PROBABILITIES → decorrelated models + ensemble. This trains three
GBDTs (natural distribution, per-v5), reports each one's threshold-tuned balanced accuracy, the
ρ-decorrelation between them (the make-or-break — L54: same inputs may give ρ→1, no room), and a
coarse weight-searched ensemble. Builds the model zoo the endgame Caruana blend needs.

Run locally:  PYTHONPATH=. python notebooks/07_v6_model_zoo.py
"""
from __future__ import annotations

import time
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import data
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, scoreboard, training, verdict
from src.diary import render_all
from src.features import build_features
from src.metrics import competition_score, tune_class_weights, weighted_predict
from src.models import cat_oof, lgb_oof, xgb_oof
from src.observer import Experiment
from src.viz import proba_rho_matrix, render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def tuned_score(oof, y, labels):
    w, s = tune_class_weights(oof, y, labels=labels)
    return s, w


def mean_rho(a, b):
    """Mean rank-correlation across the class probability columns."""
    return float(np.mean([np.corrcoef(rankdata(a[:, k]), rankdata(b[:, k]))[0, 1]
                          for k in range(a.shape[1])]))


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
    ydev_names, yhold_names = INT2CLS[ydev], INT2CLS[yhold]
    ints = list(range(len(CLASSES)))

    goal_banner("v6", "model zoo: LGBM + CatBoost + XGBoost + ensemble",
                "decorrelated models lift the probabilities the decision rule can't", best=0.9657)

    runners = {"lgb": lgb_oof, "cat": cat_oof, "xgb": xgb_oof}
    oofs, holds, tests, scoreboard_rows = {}, {}, {}, []
    for name, fn in runners.items():
        print(f"\n=== training {name} ===")
        oof, hold, tst, fold_va = fn(Xdev, ydev, Xhold, Xte,
                                     on_fold=lambda i, n, nm=name: training(f"{nm} fold {i}", i, n))
        oofs[name], holds[name], tests[name] = oof, hold, tst
        s_oof, w = tuned_score(oof, ydev, ints)
        s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
        print(f"[{name}] tuned OOF {s_oof:.4f} | holdout {s_hold:.4f}")
        scoreboard_rows.append({"name": name, "oof": s_oof, "holdout": s_hold})
        oofs[name + "_lastfold"] = fold_va  # keep for per-fold ensemble scoring
    scoreboard(scoreboard_rows, best=0.9657)

    # decorrelation between model OOF probabilities (the make-or-break for ensembling)
    names = list(runners)
    print("\n[rho] pairwise mean rank-correlation of OOF probabilities:")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            print(f"   {names[a]}~{names[b]}: {mean_rho(oofs[names[a]], oofs[names[b]]):.4f}")
    proba_rho_matrix({n: oofs[n] for n in names}, prefix="v6")

    # coarse ensemble weight search on OOF (simplex grid, step 0.25) — limit overfit
    best_w, best_s = None, -1
    grid = [w for w in product(np.arange(0, 1.01, 0.25), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    for ws in grid:
        ens = sum(w * oofs[n] for w, n in zip(ws, names))
        _, s = tune_class_weights(ens, ydev, labels=ints)
        if s > best_s:
            best_s, best_w = s, ws
    print(f"\n[ensemble] best OOF weights {dict(zip(names, np.round(best_w, 2)))} → tuned OOF {best_s:.4f}")

    ens_oof = sum(w * oofs[n] for w, n in zip(best_w, names))
    ens_hold = sum(w * holds[n] for w, n in zip(best_w, names))
    ens_test = sum(w * tests[n] for w, n in zip(best_w, names))
    dw, _ = tune_class_weights(ens_oof, ydev, labels=ints)
    fold_va = oofs["lgb_lastfold"]
    per_fold = [competition_score(ydev_names[va], weighted_predict(ens_oof[va], dw, labels=CLASSES)) for va in fold_va]
    ens_oof_score = competition_score(ydev_names, weighted_predict(ens_oof, dw, labels=CLASSES))
    ens_hold_score = competition_score(yhold_names, weighted_predict(ens_hold, dw, labels=CLASSES))

    exp = Experiment.start(
        version="v6", parent="v5",
        hypothesis="A decorrelated CatBoost+XGBoost+LGBM ensemble lifts the probabilities (and thus "
                   "balanced accuracy) past what any single model + decision rule reached.",
        predicted_delta=0.0010, confidence="low",
        feature_changes=[], pipeline_changes=["+ CatBoost", "+ XGBoost", "+ weight-searched ensemble"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=ens_oof_score, oof_score_per_fold=per_fold, holdout_score=ens_hold_score,
               runtime_sec=time.time() - t0,
               extra={"single": {r["name"]: r["holdout"] for r in scoreboard_rows},
                      "ensemble_weights": dict(zip(names, [float(x) for x in best_w])),
                      "rho": {f"{names[a]}~{names[b]}": mean_rho(oofs[names[a]], oofs[names[b]])
                              for a in range(len(names)) for b in range(a + 1, len(names))}})
    exp.commit(); render_all()

    render_all_panels(ydev_names, weighted_predict(ens_oof, dw, labels=CLASSES), ens_oof,
                      labels=CLASSES, fold_scores={"v6_ens": per_fold}, prefix="v6")
    render_html(prefix="v6", title="S6E6 v6 — model zoo + ensemble")
    verdict("v6", ens_hold_score, time.time() - t0, best=0.9657, oof=ens_oof_score)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(ens_test, dw, labels=CLASSES)})
    sub.to_csv(SUBMISSIONS / "v6_ensemble.csv", index=False)
    for n in names:
        np.save(PROBS / f"v6_{n}_test_proba.npy", tests[n])
    print(f"[submission] v6_ensemble.csv  dist={sub[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
