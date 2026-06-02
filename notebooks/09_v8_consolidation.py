"""v8 — endgame consolidation: seed-bag → Caruana selection → 2 final submissions.

0.9662 is an exhaustively-demonstrated ceiling (v2-v7). This is the disciplined close
(feedback_endgame-discipline-and-caruana): seed-bag LGBM+XGB for variance reduction, select
the blend with Caruana greedy (overfit-resistant, the technique we previously lacked), then
lock the TWO final submissions with DIFFERENT failure modes:
  #1 CV anchor   = Caruana-selected seed-bag blend (optimized on OOF)
  #2 risk pick   = simple equal-weight mean of all seeds (un-optimized, won't overfit OOF)

All selection on the LABELED dev OOF; CV↔LB is calibrated (±0.0002) so we trust it.

Run locally:  PYTHONPATH=. python notebooks/09_v8_consolidation.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.blend import blend_proba, caruana_select_proba
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, scoreboard, training, verdict
from src.diary import render_all
from src.features import build_features
from src.metrics import competition_score, tune_class_weights, weighted_predict
from src.models import lgb_oof, xgb_oof
from src.observer import Experiment
from src.viz import proba_rho_matrix, render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
SEEDS = [7, 17, 27]


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _tuned(oof, hold, test, ydev, yhold_names, ints):
    w, s_oof = tune_class_weights(oof, ydev, labels=ints)
    s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
    return w, s_oof, s_hold


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape}  seeds={SEEDS}")

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

    goal_banner("v8", f"seed-bag (LGBM+XGB ×{len(SEEDS)}) → Caruana → 2 final subs",
                "variance reduction + overfit-resistant selection; the disciplined close", best=0.9662)

    oofs, holds, tests, rows, fold_va = {}, {}, {}, [], None
    for seed in SEEDS:
        for tag, fn, kw in (("lgb", lgb_oof, {"params": {"random_state": seed}}),
                            ("xgb", xgb_oof, {"seed": seed})):
            name = f"{tag}_s{seed}"
            print(f"\n=== {name} ===")
            oof, hold, tst, fva = fn(Xdev, ydev, Xhold, Xte,
                                     on_fold=lambda i, n, nm=name: training(f"{nm} fold {i}", i, n), **kw)
            oofs[name], holds[name], tests[name], fold_va = oof, hold, tst, fva
            _, s_oof, s_hold = _tuned(oof, hold, tst, ydev, yhold_names, ints)
            rows.append({"name": name, "oof": s_oof, "holdout": s_hold})
            print(f"[{name}] tuned OOF {s_oof:.4f} | holdout {s_hold:.4f}")
    scoreboard(rows, best=0.9662)
    proba_rho_matrix(oofs, prefix="v8")

    # ---- FINAL #1: Caruana-selected seed-bag blend (CV anchor) ----
    weights, info = caruana_select_proba(oofs, ydev, seed=42)
    print(f"\n[caruana] best_single={info['best_single']} ({info['best_single_score']:.4f}) "
          f"| simple_mean {info['simple_mean_score']:.4f} | caruana_argmax {info['caruana_argmax_score']:.4f}")
    print("[caruana] weights: " + ", ".join(f"{k} {v:.2f}" for k, v in weights.items() if v > 0.01))
    car_oof = blend_proba(oofs, weights)
    car_hold = blend_proba(holds, weights)
    car_test = blend_proba(tests, weights)
    dw, car_oof_s, car_hold_s = _tuned(car_oof, car_hold, car_test, ydev, yhold_names, ints)

    # ---- FINAL #2: simple equal-weight mean-bag (different failure mode) ----
    eq = {n: 1.0 / len(oofs) for n in oofs}
    mean_oof, mean_hold, mean_test = blend_proba(oofs, eq), blend_proba(holds, eq), blend_proba(tests, eq)
    mw, mean_oof_s, mean_hold_s = _tuned(mean_oof, mean_hold, mean_test, ydev, yhold_names, ints)

    print(f"\n=== FINAL SUBMISSION SELECTION ===")
    print(f"  #1 CV anchor (Caruana blend): OOF {car_oof_s:.4f} | holdout {car_hold_s:.4f}")
    print(f"  #2 risk pick (mean-bag)     : OOF {mean_oof_s:.4f} | holdout {mean_hold_s:.4f}")
    print(f"  (v6/v7 best so far = 0.9662 LB)")

    exp = Experiment.start(
        version="v8", parent="v6",
        hypothesis=f"Seed-bagging LGBM+XGB (×{len(SEEDS)}) + Caruana selection reduces variance "
                   "for a small, robust lift and gives two different-failure-mode final subs.",
        predicted_delta=0.0003, confidence="low",
        feature_changes=[], pipeline_changes=[f"seed-bag ×{len(SEEDS)}", "Caruana greedy selection",
                                              "2-final-sub selection (anchor + risk)"],
        cloud_or_local="cloud" if not synthetic else "local")
    per_fold = [competition_score(ydev_names[va], weighted_predict(car_oof[va], dw, labels=CLASSES)) for va in fold_va]
    exp.record(oof_score_mean=car_oof_s, oof_score_per_fold=per_fold, holdout_score=car_hold_s,
               runtime_sec=time.time() - t0,
               extra={"caruana_weights": {k: round(v, 3) for k, v in weights.items()},
                      "final1_caruana_holdout": car_hold_s, "final2_meanbag_holdout": mean_hold_s,
                      "best_single": info["best_single_score"], "n_seeds": len(SEEDS)})
    exp.commit(); render_all()

    render_all_panels(ydev_names, weighted_predict(car_oof, dw, labels=CLASSES), car_oof,
                      labels=CLASSES, fold_scores={"v8_caruana": per_fold}, prefix="v8")
    render_html(prefix="v8", title="S6E6 v8 — endgame consolidation (seed-bag + Caruana)")
    verdict("v8", car_hold_s, time.time() - t0, best=0.9662, oof=car_oof_s)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(car_test, dw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v8_final1_caruana.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(mean_test, mw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v8_final2_meanbag.csv", index=False)
    print("[submissions] v8_final1_caruana.csv (anchor) + v8_final2_meanbag.csv (risk pick) written")


if __name__ == "__main__":
    main()
