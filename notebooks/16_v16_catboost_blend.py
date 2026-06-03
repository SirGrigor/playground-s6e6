"""v16 — add CatBoost+FE as a 3rd leg, 3-way blend. Push the last 0.0006 to clear 0.9700.

v15: LogReg-on-logits stacker of lgb+FE & realmlp+FE → holdout 0.9694 / LB 0.96933 (Caruana
degenerated to rm-only; the smooth stacker extracted LGBM's signal). v16 adds CatBoost+FE (native
categoricals = its ordered-TS strength → algo diversity) as a 3rd member and re-blends 3-way.

Honest prior: CatBoost is another GBDT (v6 ρ0.9 with LGBM) → marginal NEW diversity; the LogReg
stacker will down/zero-weight it if redundant, so LOW RISK (gain or flat). If it underwhelms, the
higher-diversity levers are seed-bagged RealMLP + the RealMLP Optuna sweep. Selection on holdout
(CV↔LB confirmed tight: v15 0.9694→0.96933).

Run locally:  PYTHONPATH=. python notebooks/16_v16_catboost_blend.py   (synthetic fallback, tiny cfg)
GPU REQUIRED (~30-35 min: RealMLP race + CatBoost × 5 folds).
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


def _mean_rho(a, b):
    from scipy.stats import rankdata
    return float(np.mean([np.corrcoef(rankdata(a[:, k]), rankdata(b[:, k]))[0, 1] for k in range(a.shape[1])]))


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

    goal_banner("v16", "3-leg blend: lgb+FE + realmlp+FE + CatBoost+FE (LogReg stacker)",
                "does a 3rd algo-diverse leg clear the 0.9700 target?", best=0.9694)

    print("\n=== 3-leg race: lgb + realmlp + catboost on rich FE ===")
    R = race_oof(Xdev, ydev, Xhold, Xte_rich, info, n_folds=(3 if synthetic else 5), cfg=cfg,
                 with_cat=True, on_fold=lambda i, n: training(f"fold {i} (lgb+rm+cat)", i, n))
    oof_d = {"lgb": R["lgb_oof"], "rm": R["rm_oof"], "cat": R["cat_oof"]}
    hold_d = {"lgb": R["lgb_hold"], "rm": R["rm_hold"], "cat": R["cat_hold"]}
    test_d = {"lgb": R["lgb_test"], "rm": R["rm_test"], "cat": R["cat_test"]}

    PROBS.mkdir(parents=True, exist_ok=True)
    for k in oof_d:
        np.save(PROBS / f"v16_{k}_fe_oof.npy", oof_d[k])
        np.save(PROBS / f"v16_{k}_fe_test.npy", test_d[k])

    # singles + pairwise rho (is cat decorrelated, or redundant with lgb?)
    cands = {}
    for k in oof_d:
        w, so, sh = _eval(oof_d[k], hold_d[k], ydev, yhold_names, ydev_names)
        cands[f"{k}_single"] = (oof_d[k], hold_d[k], test_d[k], w, so, sh)
    print(f"[rho] lgb~cat {_mean_rho(R['lgb_oof'], R['cat_oof']):.3f} | "
          f"rm~cat {_mean_rho(R['rm_oof'], R['cat_oof']):.3f} | lgb~rm {_mean_rho(R['lgb_oof'], R['rm_oof']):.3f}")

    cw, cinfo = caruana_select_proba(oof_d, ydev)
    print(f"[caruana] weights {({k: round(v,3) for k,v in cw.items()})}")
    cands["caruana"] = (blend_proba(oof_d, cw), blend_proba(hold_d, cw), blend_proba(test_d, cw)) + \
        _eval(blend_proba(oof_d, cw), blend_proba(hold_d, cw), ydev, yhold_names, ydev_names)
    mw = {k: 1 / 3 for k in oof_d}
    cands["mean"] = (blend_proba(oof_d, mw), blend_proba(hold_d, mw), blend_proba(test_d, mw)) + \
        _eval(blend_proba(oof_d, mw), blend_proba(hold_d, mw), ydev, yhold_names, ydev_names)
    so, sh_, st = _logreg_stack(oof_d, hold_d, test_d, ydev)
    cands["logreg"] = (so, sh_, st) + _eval(so, sh_, ydev, yhold_names, ydev_names)

    scoreboard([{"name": k, "oof": v[4], "holdout": v[5]} for k, v in cands.items()], best=0.9694)
    blends = {k: v for k, v in cands.items() if k in ("caruana", "mean", "logreg")}
    best_name = max(blends, key=lambda k: blends[k][5])
    boof, bhold, btest, bw, b_oof, b_hold = cands[best_name]
    best_single = max(cands[f"{k}_single"][5] for k in oof_d)
    print(f"\n[winner] {best_name}: OOF {b_oof:.4f} | holdout {b_hold:.4f}  "
          f"(best single {best_single:.4f}; v15 logreg-2way 0.9694)")
    print(f"[blend lift] {b_hold - best_single:+.5f} over best single | {b_hold - 0.9694:+.5f} vs v15")
    b_ovo = multiclass_auc_report(ydev_names, boof, labels=CLASSES)["ovo_per_pair"].get("GALAXY|STAR", float("nan"))

    per_fold = [competition_score(ydev_names[va], weighted_predict(boof[va], bw, labels=CLASSES)) for va in R["fva"]]
    cleared = b_hold >= 0.9700
    note = (f"3-leg blend winner={best_name} holdout {b_hold:.4f} "
            f"({'CLEARED 0.9700' if cleared else 'short of 0.9700'}); vs v15 2-way {b_hold-0.9694:+.5f}")
    exp = Experiment.start(
        version="v16", parent="v15",
        hypothesis="A 3rd algo-diverse leg (CatBoost+FE, native categoricals) adds enough decorrelated "
                   "signal for the LogReg stacker to clear the 0.9700 target.",
        predicted_delta=0.0004, confidence="low",
        feature_changes=[], pipeline_changes=["+ CatBoost+FE (native cats) 3rd leg", "3-way blend"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=b_oof, oof_score_per_fold=per_fold, holdout_score=b_hold,
               runtime_sec=time.time() - t0,
               extra={"winner": best_name, "singles": {k: cands[f"{k}_single"][5] for k in oof_d},
                      "caruana_holdout": cands["caruana"][5], "logreg_holdout": cands["logreg"][5],
                      "mean_holdout": cands["mean"][5], "rho_lgb_cat": _mean_rho(R["lgb_oof"], R["cat_oof"]),
                      "rho_rm_cat": _mean_rho(R["rm_oof"], R["cat_oof"]),
                      "ovo_galaxy_star": b_ovo, "cleared_target": bool(cleared)})
    exp.note(note)
    exp.commit(); render_all()

    proba_rho_matrix(oof_d, prefix="v16")
    render_all_panels(ydev_names, weighted_predict(boof, bw, labels=CLASSES), boof, labels=CLASSES,
                      fold_scores={f"v16_{best_name}": per_fold}, prefix="v16")
    render_html(prefix="v16", title="S6E6 v16 — 3-leg blend (toward 0.9700)")
    verdict("v16", b_hold, time.time() - t0, best=0.9694, oof=b_oof)

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for name in ("caruana", "mean", "logreg"):
        _, _, tp, w, _, _ = cands[name]
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(tp, w, labels=CLASSES)}).to_csv(
            SUBMISSIONS / f"v16_blend_{name}.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(btest, bw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v16_final.csv", index=False)
    print(f"\n[submission] v16_final.csv (={best_name}) + v16_blend_{{caruana,mean,logreg}}.csv")
    print(f"[verdict] {note}")


if __name__ == "__main__":
    main()
