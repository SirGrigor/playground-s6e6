"""v13 — RealMLP, the paradigm break. Does a proper NN beat our GBDT on the SAME features?

The 0.9664 "ceiling" was a GBDT-architecture artifact: a public RealMLP hits ~0.968 from the same
columns (+0.0025 over our best GBDT). Our v7 dismissed RealMLP after one under-tuned pytabkit run.
This runs OUR reimplementation (src/realmlp.py — PBLD periodic embeddings, n_ens internal bag,
balanced-softmax metric-aware loss, EMA, per-group LRs) head-to-head against the LGBM baseline on the
SAME build_features set, so the delta is pure architecture (clean attribution — FE is a later lever).

If RealMLP > GBDT here, the ceiling is broken and the next move is a GBDT+RealMLP diversity blend.

Run locally:  PYTHONPATH=. python notebooks/13_v13_realmlp.py   (synthetic fallback uses a tiny cfg)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import data
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, scoreboard, training, verdict
from src.diagnostics import error_cell_report
from src.diary import render_all
from src.features import build_features
from src.metrics import competition_score, multiclass_auc_report, tune_class_weights, weighted_predict
from src.models import lgb_oof
from src.observer import Experiment
from src.realmlp import realmlp_oof
from src.viz import proba_rho_matrix, render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))

# Tiny config so the synthetic smoke test exercises the full pipeline in seconds (CPU).
SMOKE_CFG = {"n_ens": 2, "hidden_dims": [32], "epochs": 1, "eval_bs": 4096,
             "pbld_hidden_dim": 8, "pbld_out_dim": 3}


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _mean_rho(a, b):
    return float(np.mean([np.corrcoef(rankdata(a[:, k]), rankdata(b[:, k]))[0, 1] for k in range(a.shape[1])]))


def _score(oof, hold, ydev, yhold_names, ydev_names, fva):
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
    per_fold = [competition_score(ydev_names[va], weighted_predict(oof[va], w, labels=CLASSES)) for va in fva]
    ovo = multiclass_auc_report(ydev_names, oof, labels=CLASSES)["ovo_per_pair"].get("GALAXY|STAR", float("nan"))
    return w, s_oof, s_hold, per_fold, ovo


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape}")
    cfg = SMOKE_CFG if synthetic else None

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

    goal_banner("v13", "RealMLP (PBLD + n_ens bag + balanced-softmax + EMA) vs GBDT baseline",
                "does a proper NN beat GBDT on the SAME features? = is the 0.9664 ceiling real?", best=0.9664)

    # --- LGBM baseline (head-to-head reference) ---
    print("\n=== LGBM baseline ===")
    lgb_oof_p, lgb_hold, lgb_test, fva = lgb_oof(Xdev, ydev, Xhold, Xte,
                                                 on_fold=lambda i, n: training(f"lgb fold {i}", i, n))
    lw, lgb_s_oof, lgb_s_hold, lgb_pf, lgb_ovo = _score(lgb_oof_p, lgb_hold, ydev, yhold_names, ydev_names, fva)
    print(f"[lgb] tuned OOF {lgb_s_oof:.4f} | holdout {lgb_s_hold:.4f}")

    # --- RealMLP ---
    print("\n=== RealMLP ===")
    rm_oof, rm_hold, rm_test, fva2 = realmlp_oof(Xdev, ydev, Xhold, Xte, cfg=cfg,
                                                 on_fold=lambda i, n: training(f"realmlp fold {i}", i, n))
    rw, rm_s_oof, rm_s_hold, rm_pf, rm_ovo = _score(rm_oof, rm_hold, ydev, yhold_names, ydev_names, fva2)
    print(f"[realmlp] tuned OOF {rm_s_oof:.4f} | holdout {rm_s_hold:.4f}")

    scoreboard([{"name": "lgb_baseline", "oof": lgb_s_oof, "holdout": lgb_s_hold},
                {"name": "realmlp", "oof": rm_s_oof, "holdout": rm_s_hold}], best=0.9664)
    rho = _mean_rho(lgb_oof_p, rm_oof)
    d_oof, d_hold = rm_s_oof - lgb_s_oof, rm_s_hold - lgb_s_hold
    print(f"\n[delta] RealMLP vs LGBM: OOF {d_oof:+.5f}  holdout {d_hold:+.5f}  "
          f"| GALAXY|STAR OvO {lgb_ovo:.4f} → {rm_ovo:.4f}  | rho(lgb,realmlp) {rho:.3f}")
    print("\n[per-class recall — RealMLP OOF]")
    print(error_cell_report(ydev_names, weighted_predict(rm_oof, rw, labels=CLASSES), CLASSES))

    ceiling_broken = rm_s_hold > 0.9664
    note = ("RealMLP BEAT the GBDT ceiling — paradigm shift confirmed, next = GBDT+RealMLP blend"
            if ceiling_broken else
            f"RealMLP holdout {rm_s_hold:.4f} (vs GBDT {lgb_s_hold:.4f}); needs FE/tuning to reach public ~0.968")
    exp = Experiment.start(
        version="v13", parent="v8",
        hypothesis="A properly-built RealMLP (PBLD periodic embeddings, n_ens internal bag, balanced-softmax "
                   "metric-aware loss, EMA) beats our GBDTs on the SAME features — the 0.9664 'ceiling' was a "
                   "GBDT-architecture artifact, not the competition's ceiling (v7 dismissed RealMLP after one "
                   "under-tuned run).",
        predicted_delta=0.0025, confidence="medium",
        feature_changes=[], pipeline_changes=["+ RealMLP (custom PyTorch, EXP-549 recipe)", "head-to-head vs LGBM"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=rm_s_oof, oof_score_per_fold=rm_pf, holdout_score=rm_s_hold,
               runtime_sec=time.time() - t0,
               extra={"lgb_oof": lgb_s_oof, "lgb_holdout": lgb_s_hold,
                      "delta_oof_vs_lgb": d_oof, "delta_holdout_vs_lgb": d_hold,
                      "rho_lgb_realmlp": rho, "ovo_galaxy_star": {"lgb": lgb_ovo, "realmlp": rm_ovo},
                      "ceiling_broken": bool(ceiling_broken)})
    exp.note(note)
    exp.commit(); render_all()

    if len(rm_oof) and len(lgb_oof_p):
        proba_rho_matrix({"lgb": lgb_oof_p, "realmlp": rm_oof}, prefix="v13")
    render_all_panels(ydev_names, weighted_predict(rm_oof, rw, labels=CLASSES), rm_oof, labels=CLASSES,
                      fold_scores={"v13_realmlp": rm_pf}, prefix="v13")
    render_html(prefix="v13", title="S6E6 v13 — RealMLP vs GBDT (is the ceiling real?)")
    verdict("v13", rm_s_hold, time.time() - t0, best=0.9664, oof=rm_s_oof)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(rm_test, rw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v13_realmlp.csv", index=False)
    np.save(PROBS / "v13_realmlp_test_proba.npy", rm_test)
    np.save(PROBS / "v13_lgb_test_proba.npy", lgb_test)
    print(f"\n[submission] v13_realmlp.csv  dist="
          f"{pd.read_csv(SUBMISSIONS/'v13_realmlp.csv')[TARGET].value_counts().to_dict()}")
    print(f"[verdict] {note}")


if __name__ == "__main__":
    main()
