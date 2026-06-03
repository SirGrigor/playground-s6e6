"""v14 — RealMLP + the rich feature engineering. Does FE close the gap to the public ~0.968?

v13 verdict: our RealMLP recipe works (+0.0023 over v7) but loses to GBDT on bare features
(0.9609 vs 0.9657); the +0.0073 to the public 0.968 is Vladimir's FEATURE ENGINEERING, which the
public 0.968 baseline includes. This adds it (src/fe_realmlp.py: ratios, numeric→categorical,
KBins, crossed cats, per-class OOF target-encoded combos) and races LGBM vs RealMLP on identical
folds — answering (a) does FE lift RealMLP to ~0.968, (b) does the same FE also lift the GBDT
(if yes, the ceiling breaks regardless of which model wins).

Static FE is fit once on full train; the target encoding is per-fold (leak-safe). NN sees the full
rich set; LGBM sees numerics + TE + native cats. Clean attribution: FE-only (domain |b|/L = v15).

Run locally:  PYTHONPATH=. python notebooks/14_v14_realmlp_fe.py   (synthetic fallback, tiny cfg)
GPU strongly recommended (RealMLP × 5 folds with rich embeddings).
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
from src.fe_realmlp import build_rich_features, race_oof
from src.metrics import competition_score, multiclass_auc_report, tune_class_weights, weighted_predict
from src.observer import Experiment
from src.viz import proba_rho_matrix, render_all_panels

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
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
    cfg = SMOKE_CFG if synthetic else {}

    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte_rich, _, _ = build_rich_features(test, fit=False, state=state)
    print(f"[fe] {Xrich.shape[1]} features  | {len(info['cat_cols'])} categorical "
          f"| {len(info['combo_cols'])} combos (per-fold TE) | {len(info['num_cols'])} numeric")

    dev_idx, hold_idx = data.holdout_split(train)
    Xdev = Xrich.iloc[dev_idx].reset_index(drop=True); ydev = y_all[dev_idx]
    Xhold = Xrich.iloc[hold_idx].reset_index(drop=True); yhold = y_all[hold_idx]
    ydev_names, yhold_names = INT2CLS[ydev], INT2CLS[yhold]

    goal_banner("v14", "RealMLP + rich FE (ratios, num→cat, crossed cats, per-class TE) vs LGBM+FE",
                "does FE close RealMLP to public ~0.968? does it also lift the GBDT?", best=0.9664)

    print("\n=== per-fold race: LGBM vs RealMLP on rich FE ===")
    R = race_oof(Xdev, ydev, Xhold, Xte_rich, info, n_folds=(3 if synthetic else 5), cfg=cfg,
                 on_fold=lambda i, n: training(f"fold {i} (lgb+realmlp)", i, n))

    lw, l_oof, l_hold, l_pf, l_ovo = _score(R["lgb_oof"], R["lgb_hold"], ydev, yhold_names, ydev_names, R["fva"])
    rw, r_oof, r_hold, r_pf, r_ovo = _score(R["rm_oof"], R["rm_hold"], ydev, yhold_names, ydev_names, R["fva"])
    print(f"[lgb+FE]     tuned OOF {l_oof:.4f} | holdout {l_hold:.4f}")
    print(f"[realmlp+FE] tuned OOF {r_oof:.4f} | holdout {r_hold:.4f}")

    scoreboard([{"name": "lgb+FE", "oof": l_oof, "holdout": l_hold},
                {"name": "realmlp+FE", "oof": r_oof, "holdout": r_hold}], best=0.9664)
    rho = _mean_rho(R["lgb_oof"], R["rm_oof"])
    print(f"\n[delta vs prior bests] lgb+FE {l_hold-0.9657:+.4f} vs v13-lgb 0.9657 | "
          f"realmlp+FE {r_hold-0.9609:+.4f} vs v13-realmlp 0.9609")
    print(f"[delta] realmlp+FE vs lgb+FE: holdout {r_hold-l_hold:+.5f} | "
          f"GALAXY|STAR OvO lgb {l_ovo:.4f} / realmlp {r_ovo:.4f} | rho {rho:.3f}")
    best_name, best_oof, best_hold, best_pf, best_w, best_oofp, best_testp, best_ovo = (
        ("realmlp+FE", r_oof, r_hold, r_pf, rw, R["rm_oof"], R["rm_test"], r_ovo) if r_hold >= l_hold
        else ("lgb+FE", l_oof, l_hold, l_pf, lw, R["lgb_oof"], R["lgb_test"], l_ovo))
    print(f"\n[per-class recall — {best_name} OOF]")
    print(error_cell_report(ydev_names, weighted_predict(best_oofp, best_w, labels=CLASSES), CLASSES))

    broke = best_hold > 0.9664
    note = (f"FE {'BROKE the ceiling' if broke else 'did not break the ceiling'}: best={best_name} "
            f"holdout {best_hold:.4f} (lgb+FE {l_hold:.4f}, realmlp+FE {r_hold:.4f}; v13 lgb 0.9657)")
    exp = Experiment.start(
        version="v14", parent="v13",
        hypothesis="Vladimir's feature engineering (ratios, numeric→categorical, crossed cats, per-class "
                   "OOF target-encoded combos) closes RealMLP to the public ~0.968 and/or lifts the GBDT — "
                   "the v13 gap was FE, not architecture.",
        predicted_delta=0.0050, confidence="low",
        feature_changes=["+ratios", "+numeric→categorical", "+KBins(delta)", "+crossed cats", "+per-class TE combos"],
        pipeline_changes=["per-fold target encoding", "lgb-vs-realmlp race on rich FE"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=best_oof, oof_score_per_fold=best_pf, holdout_score=best_hold,
               runtime_sec=time.time() - t0,
               extra={"lgb_fe_oof": l_oof, "lgb_fe_holdout": l_hold,
                      "realmlp_fe_oof": r_oof, "realmlp_fe_holdout": r_hold,
                      "rho_lgb_realmlp": rho, "ovo_galaxy_star": {"lgb": l_ovo, "realmlp": r_ovo},
                      "n_features": int(Xrich.shape[1]), "ceiling_broken": bool(broke), "winner": best_name})
    exp.note(note)
    exp.commit(); render_all()

    proba_rho_matrix({"lgb_fe": R["lgb_oof"], "realmlp_fe": R["rm_oof"]}, prefix="v14")
    render_all_panels(ydev_names, weighted_predict(best_oofp, best_w, labels=CLASSES), best_oofp,
                      labels=CLASSES, fold_scores={f"v14_{best_name}": best_pf}, prefix="v14")
    render_html(prefix="v14", title="S6E6 v14 — RealMLP + rich FE vs LGBM (does FE break the ceiling?)")
    verdict("v14", best_hold, time.time() - t0, best=0.9664, oof=best_oof)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    for nm, tp, w in [("realmlp", R["rm_test"], rw), ("lgb", R["lgb_test"], lw)]:
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(tp, w, labels=CLASSES)}).to_csv(
            SUBMISSIONS / f"v14_{nm}_fe.csv", index=False)
        np.save(PROBS / f"v14_{nm}_fe_test_proba.npy", tp)
    print(f"\n[submission] v14_realmlp_fe.csv + v14_lgb_fe.csv written")
    print(f"[verdict] {note}")


if __name__ == "__main__":
    main()
