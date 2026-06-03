"""v17 — seed-bag the RealMLP (variance reduction on our strongest leg), re-blend with LGBM.

v16 verdict: CatBoost (redundant GBDT, ρ0.936 w/ LGBM, weakest single) HURT on LB (0.96924 < v15
0.96933) — a holdout-overfit that didn't generalize. v17 takes the opposite, generalizing lever:
seed-bag the RealMLP (our best single, 0.9690) — train it with 3 seeds per fold and average. Unlike
the CatBoost holdout-overfit, variance reduction GENERALIZES (v8 proved it on LB). Then 2-way blend
lgb+FE & rm_bagged (CatBoost dropped). Endgame discipline: sub-0.0002 holdout deltas are noise here —
trust the LB, and only adopt v17 if it beats v15's 0.96933.

Run locally:  PYTHONPATH=. python notebooks/17_v17_realmlp_seedbag.py   (synthetic, tiny cfg)
GPU REQUIRED (~40 min: RealMLP × 3 seeds × 5 folds + lgb).
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
from src.fe_realmlp import build_rich_features, race_oof
from src.metrics import competition_score, multiclass_auc_report, tune_class_weights, weighted_predict
from src.observer import Experiment
from src.viz import render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
SMOKE_CFG = {"n_ens": 2, "hidden_dims": [32], "epochs": 1, "eval_bs": 4096,
             "pbld_hidden_dim": 8, "pbld_out_dim": 3}
RM_SEEDS = [1, 2, 3]   # 3-seed bag of the RealMLP


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _logits(P):
    return np.log(np.clip(np.asarray(P, dtype=float), 1e-6, 1.0))


def _logreg_stack(oof_d, hold_d, test_d, ydev):
    from sklearn.linear_model import LogisticRegression
    feats = lambda d: np.concatenate([_logits(d[k]) for k in d], axis=1)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(feats(oof_d), ydev)
    return clf.predict_proba(feats(oof_d)), clf.predict_proba(feats(hold_d)), clf.predict_proba(feats(test_d))


def _eval(boof, bhold, ydev, yhold_names, ydev_names):
    w, s_oof = tune_class_weights(boof, ydev, labels=INTS)
    s_hold = competition_score(yhold_names, weighted_predict(bhold, w, labels=CLASSES))
    return w, s_oof, s_hold


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape}")
    cfg = SMOKE_CFG if synthetic else {}

    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte_rich, _, _ = build_rich_features(test, fit=False, state=state)
    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]
    ydev_names, yhold_names = INT2CLS[ydev], INT2CLS[yhold]

    goal_banner("v17", f"seed-bagged RealMLP (×{len(RM_SEEDS)}) + LGBM, 2-way blend",
                "does variance-reducing our best leg beat v15 0.96933 on LB?", best=0.9694)

    print(f"\n=== race: lgb + RealMLP seed-bag ×{len(RM_SEEDS)} (CatBoost dropped) ===")
    R = race_oof(Xdev, ydev, Xhold, Xte_rich, info, n_folds=(3 if synthetic else 5), cfg=cfg,
                 with_cat=False, rm_seeds=RM_SEEDS,
                 on_fold=lambda i, n: training(f"fold {i} (lgb+rm×{len(RM_SEEDS)})", i, n))
    oof_d = {"lgb": R["lgb_oof"], "rm": R["rm_oof"]}
    hold_d = {"lgb": R["lgb_hold"], "rm": R["rm_hold"]}
    test_d = {"lgb": R["lgb_test"], "rm": R["rm_test"]}

    PROBS.mkdir(parents=True, exist_ok=True)
    for k in oof_d:
        np.save(PROBS / f"v17_{k}_oof.npy", oof_d[k]); np.save(PROBS / f"v17_{k}_test.npy", test_d[k])

    cands = {}
    for k in oof_d:
        cands[f"{k}_single"] = (oof_d[k], hold_d[k], test_d[k]) + _eval(oof_d[k], hold_d[k], ydev, yhold_names, ydev_names)
    cw, _ = caruana_select_proba(oof_d, ydev)
    cands["caruana"] = (blend_proba(oof_d, cw), blend_proba(hold_d, cw), blend_proba(test_d, cw)) + \
        _eval(blend_proba(oof_d, cw), blend_proba(hold_d, cw), ydev, yhold_names, ydev_names)
    mw = {k: 0.5 for k in oof_d}
    cands["mean"] = (blend_proba(oof_d, mw), blend_proba(hold_d, mw), blend_proba(test_d, mw)) + \
        _eval(blend_proba(oof_d, mw), blend_proba(hold_d, mw), ydev, yhold_names, ydev_names)
    so, sh, st = _logreg_stack(oof_d, hold_d, test_d, ydev)
    cands["logreg"] = (so, sh, st) + _eval(so, sh, ydev, yhold_names, ydev_names)

    scoreboard([{"name": k, "oof": v[3], "holdout": v[4]} for k, v in cands.items()], best=0.9694)
    blends = {k: v for k, v in cands.items() if k in ("caruana", "mean", "logreg")}
    best_name = max(blends, key=lambda k: blends[k][4])
    boof, bhold, btest, bw, b_oof, b_hold = cands[best_name]
    rm_single = cands["rm_single"][4]
    print(f"\n[seed-bag effect] rm_single now {rm_single:.4f} (vs v14 single 0.9690)")
    print(f"[winner] {best_name}: OOF {b_oof:.4f} | holdout {b_hold:.4f}  (v15 2-way 0.9694 / LB 0.96933)")
    print(f"[note] sub-0.0002 holdout deltas are NOISE at this ceiling — trust the LB on v17 vs v15")
    b_ovo = multiclass_auc_report(ydev_names, boof, labels=CLASSES)["ovo_per_pair"].get("GALAXY|STAR", float("nan"))

    per_fold = [competition_score(ydev_names[va], weighted_predict(boof[va], bw, labels=CLASSES)) for va in R["fva"]]
    note = (f"seed-bag×{len(RM_SEEDS)}: rm_single {rm_single:.4f} (v14 0.9690); blend winner={best_name} "
            f"holdout {b_hold:.4f} (v15 0.9694) — DECIDE on LB vs 0.96933, not holdout (sub-noise)")
    exp = Experiment.start(
        version="v17", parent="v15",
        hypothesis="Seed-bagging the RealMLP (×3) reduces its variance for a small LB gain that GENERALIZES "
                   "(unlike v16 CatBoost's holdout-overfit), lifting the 2-way blend past v15's 0.96933.",
        predicted_delta=0.0002, confidence="low",
        feature_changes=[], pipeline_changes=[f"RealMLP seed-bag ×{len(RM_SEEDS)}", "drop CatBoost", "2-way blend"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=b_oof, oof_score_per_fold=per_fold, holdout_score=b_hold,
               runtime_sec=time.time() - t0,
               extra={"winner": best_name, "rm_single_bagged": rm_single, "rm_single_v14": 0.9690,
                      "lgb_single": cands["lgb_single"][4], "n_seeds": len(RM_SEEDS),
                      "caruana_holdout": cands["caruana"][4], "logreg_holdout": cands["logreg"][4],
                      "mean_holdout": cands["mean"][4], "ovo_galaxy_star": b_ovo})
    exp.note(note)
    exp.commit(); render_all()

    render_all_panels(ydev_names, weighted_predict(boof, bw, labels=CLASSES), boof, labels=CLASSES,
                      fold_scores={f"v17_{best_name}": per_fold}, prefix="v17")
    render_html(prefix="v17", title="S6E6 v17 — seed-bagged RealMLP blend")
    verdict("v17", b_hold, time.time() - t0, best=0.9694, oof=b_oof)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for name in ("caruana", "mean", "logreg"):
        _, _, tp, w, _, _ = cands[name]
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(tp, w, labels=CLASSES)}).to_csv(
            SUBMISSIONS / f"v17_blend_{name}.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(btest, bw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v17_final.csv", index=False)
    print(f"\n[submission] v17_final.csv (={best_name}) + v17_blend_{{caruana,mean,logreg}}.csv")
    print(f"[verdict] {note}")


if __name__ == "__main__":
    main()
