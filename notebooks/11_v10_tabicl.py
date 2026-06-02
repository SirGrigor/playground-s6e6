"""v10 — TabICLv2 (tabular foundation model) + GBDT ensemble.

The 2026 new-paradigm lever the investigation surfaced. v7's RealMLP decorrelated (ρ0.72) but
was too WEAK (0.9586) to help. TabICLv2 is an in-context-learning transformer that benchmarks
ABOVE XGBoost and scales to our size — the candidate to be both strong AND decorrelated, the one
combination that can lift the GBDT ensemble past 0.9664. Trains LGBM + XGB + TabICLv2, reports
each tuned bal-acc, ρ matrix (is tabicl <0.95 AND competitive?), weight-searched ensemble.

TabICL is try/except-wrapped (download/OOM safe → GBDT ensemble still ships). Run only if v9 said
the ceiling has resolvable structure; if v9 converged on the noise floor, this is confirmatory only.

Run locally:  PYTHONPATH=. python notebooks/11_v10_tabicl.py   (tabicl skipped if absent)
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
from src.models import lgb_oof, tabicl_oof, xgb_oof
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


def mean_rho(a, b):
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

    goal_banner("v10", "TabICLv2 (foundation model) + GBDT ensemble",
                "a strong AND decorrelated non-GBDT — the combo RealMLP wasn't", best=0.9664)

    runners = {"lgb": lgb_oof, "xgb": xgb_oof, "tabicl": tabicl_oof}
    oofs, holds, tests, rows, fold_va = {}, {}, {}, [], None
    for name, fn in runners.items():
        print(f"\n=== training {name} ===")
        try:
            oof, hold, tst, fva = fn(Xdev, ydev, Xhold, Xte,
                                     on_fold=lambda i, n, nm=name: training(f"{nm} fold {i}", i, n))
        except Exception as e:  # noqa: BLE001
            print(f"⚠ {name} FAILED ({type(e).__name__}: {e}) — skipping, continuing")
            continue
        oofs[name], holds[name], tests[name], fold_va = oof, hold, tst, fva
        w, s_oof = tune_class_weights(oof, ydev, labels=ints)
        s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
        rows.append({"name": name, "oof": s_oof, "holdout": s_hold})
        print(f"[{name}] tuned OOF {s_oof:.4f} | holdout {s_hold:.4f}")
    scoreboard(rows, best=0.9664)

    names = list(oofs)
    print("\n[rho] pairwise mean rank-correlation (↓ = ensemble room):")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            print(f"   {names[a]}~{names[b]}: {mean_rho(oofs[names[a]], oofs[names[b]]):.4f}")
    if len(names) >= 2:
        proba_rho_matrix({n: oofs[n] for n in names}, prefix="v10")

    grid = [w for w in product(np.arange(0, 1.01, 0.25), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    best_w, best_s = None, -1
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
    per_fold = [competition_score(ydev_names[va], weighted_predict(ens_oof[va], dw, labels=CLASSES)) for va in fold_va]
    ens_oof_s = competition_score(ydev_names, weighted_predict(ens_oof, dw, labels=CLASSES))
    ens_hold_s = competition_score(yhold_names, weighted_predict(ens_hold, dw, labels=CLASSES))

    exp = Experiment.start(
        version="v10", parent="v8",
        hypothesis="TabICLv2 (foundation model) is strong AND decorrelated from GBDT, so the ensemble "
                   "finally lifts past 0.9664.",
        predicted_delta=0.0010, confidence="low",
        feature_changes=[], pipeline_changes=["+ TabICLv2 (tabular foundation model)", "GBDT+TabICL ensemble"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=ens_oof_s, oof_score_per_fold=per_fold, holdout_score=ens_hold_s,
               runtime_sec=time.time() - t0,
               extra={"single": {r["name"]: r["holdout"] for r in rows},
                      "ensemble_weights": dict(zip(names, [float(x) for x in best_w])),
                      "rho": {f"{names[a]}~{names[b]}": mean_rho(oofs[names[a]], oofs[names[b]])
                              for a in range(len(names)) for b in range(a + 1, len(names))},
                      "tabicl_trained": "tabicl" in oofs})
    exp.commit(); render_all()

    render_all_panels(ydev_names, weighted_predict(ens_oof, dw, labels=CLASSES), ens_oof,
                      labels=CLASSES, fold_scores={"v10_ens": per_fold}, prefix="v10")
    render_html(prefix="v10", title="S6E6 v10 — TabICLv2 + GBDT ensemble")
    verdict("v10", ens_hold_s, time.time() - t0, best=0.9664, oof=ens_oof_s)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(ens_test, dw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v10_tabicl_ensemble.csv", index=False)
    for n in names:
        np.save(PROBS / f"v10_{n}_test_proba.npy", tests[n])
    print(f"[submission] v10_tabicl_ensemble.csv  dist={pd.read_csv(SUBMISSIONS/'v10_tabicl_ensemble.csv')[TARGET].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
