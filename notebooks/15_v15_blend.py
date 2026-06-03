"""v15 — supervised blend of lgb+FE and realmlp+FE. Push past 0.9690 toward the 0.9700 target.

v14 broke the ceiling: realmlp+FE 0.9690, lgb+FE 0.9682, ρ=0.85 (decorrelated AND both strong) —
the textbook blend setup. This re-runs the deterministic v14 race (reproduces both OOF+test — v14
only persisted test probas), then fits three supervised blends on the LABELED dev OOF and selects on
the held-out holdout:
  - Caruana greedy (src/blend.py) — overfit-resistant, our endgame tool
  - LogReg-on-logits stacker — Deotte's exact 2nd-place move (cuML there, sklearn here)
  - simple mean — the un-optimized baseline that beat Caruana in S6E5/v8 (objective-misalignment)

Selection is on the holdout (large, ~92K; ~3 candidates → negligible optimism) — the OOF favors the
in-sample-fit stacker, so holdout is the honest comparator. OOF is persisted (probs/v15_*_oof.npy) so
later blends skip the 24-min re-run. FE-only inputs; domain |b|/L = v16 if the blend stalls.

Run locally:  PYTHONPATH=. python notebooks/15_v15_blend.py   (synthetic fallback, tiny cfg)
GPU REQUIRED (re-runs the RealMLP race, ~25 min).
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
    """Deotte's move: multinomial LogReg on the concatenated per-model logits (level-2 stacker)."""
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

    goal_banner("v15", "supervised blend of lgb+FE & realmlp+FE (Caruana / LogReg-logits / mean)",
                "ρ=0.85, both strong → can a blend clear the 0.9700 target?", best=0.9690)

    print("\n=== re-run v14 race (reproduces lgb+FE / realmlp+FE OOF + test) ===")
    R = race_oof(Xdev, ydev, Xhold, Xte_rich, info, n_folds=(3 if synthetic else 5), cfg=cfg,
                 on_fold=lambda i, n: training(f"fold {i} (lgb+realmlp)", i, n))
    oof_d = {"lgb": R["lgb_oof"], "rm": R["rm_oof"]}
    hold_d = {"lgb": R["lgb_hold"], "rm": R["rm_hold"]}
    test_d = {"lgb": R["lgb_test"], "rm": R["rm_test"]}

    # persist OOF + test so future blends skip the re-run
    PROBS.mkdir(parents=True, exist_ok=True)
    for k in oof_d:
        np.save(PROBS / f"v15_{k}_fe_oof.npy", oof_d[k])
        np.save(PROBS / f"v15_{k}_fe_hold.npy", hold_d[k])
        np.save(PROBS / f"v15_{k}_fe_test.npy", test_d[k])

    # candidate blends (+ the two singles for reference)
    cands = {}
    lw, ls_o, ls_h = _eval(R["lgb_oof"], R["lgb_hold"], ydev, yhold_names, ydev_names)
    rw, rs_o, rs_h = _eval(R["rm_oof"], R["rm_hold"], ydev, yhold_names, ydev_names)
    cands["lgb_single"] = (R["lgb_oof"], R["lgb_hold"], R["lgb_test"], lw, ls_o, ls_h)
    cands["rm_single"] = (R["rm_oof"], R["rm_hold"], R["rm_test"], rw, rs_o, rs_h)

    cw, cinfo = caruana_select_proba(oof_d, ydev)
    print(f"[caruana] weights {({k: round(v,3) for k,v in cw.items()})} | {cinfo}")
    cands["caruana"] = (blend_proba(oof_d, cw), blend_proba(hold_d, cw), blend_proba(test_d, cw)) + \
        _eval(blend_proba(oof_d, cw), blend_proba(hold_d, cw), ydev, yhold_names, ydev_names)

    mw = {"lgb": 0.5, "rm": 0.5}
    cands["mean"] = (blend_proba(oof_d, mw), blend_proba(hold_d, mw), blend_proba(test_d, mw)) + \
        _eval(blend_proba(oof_d, mw), blend_proba(hold_d, mw), ydev, yhold_names, ydev_names)

    s_oofp, s_holdp, s_testp = _logreg_stack(oof_d, hold_d, test_d, ydev)
    cands["logreg"] = (s_oofp, s_holdp, s_testp) + _eval(s_oofp, s_holdp, ydev, yhold_names, ydev_names)

    # scoreboard + pick winner by HOLDOUT (honest comparator; OOF favors the in-sample stacker)
    rows = [{"name": k, "oof": v[4], "holdout": v[5]} for k, v in cands.items()]
    scoreboard(rows, best=0.9690)
    blends = {k: v for k, v in cands.items() if k in ("caruana", "mean", "logreg")}
    best_name = max(blends, key=lambda k: blends[k][5])
    boof, bhold, btest, bw, b_oof, b_hold = cands[best_name]
    print(f"\n[winner] {best_name}: OOF {b_oof:.4f} | holdout {b_hold:.4f}  "
          f"(vs realmlp+FE single {rs_h:.4f}, lgb+FE single {ls_h:.4f})")
    print(f"[blend lift] best blend {b_hold - max(rs_h, ls_h):+.5f} over the best single")
    b_ovo = multiclass_auc_report(ydev_names, boof, labels=CLASSES)["ovo_per_pair"].get("GALAXY|STAR", float("nan"))

    per_fold = [competition_score(ydev_names[va], weighted_predict(boof[va], bw, labels=CLASSES)) for va in R["fva"]]
    cleared = b_hold >= 0.9700
    note = (f"blend winner={best_name} holdout {b_hold:.4f} ({'CLEARED' if cleared else 'short of'} 0.9700 "
            f"target); +{b_hold-max(rs_h,ls_h):.5f} over best single (rm {rs_h:.4f} / lgb {ls_h:.4f})")
    exp = Experiment.start(
        version="v15", parent="v14",
        hypothesis="A supervised blend of the decorrelated (ρ0.85) lgb+FE and realmlp+FE beats either "
                   "alone and pushes toward the 0.9700 target.",
        predicted_delta=0.0008, confidence="medium",
        feature_changes=[], pipeline_changes=[f"blend: caruana/logreg/mean → winner {best_name}"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=b_oof, oof_score_per_fold=per_fold, holdout_score=b_hold,
               runtime_sec=time.time() - t0,
               extra={"winner": best_name, "lgb_single_holdout": ls_h, "rm_single_holdout": rs_h,
                      "caruana_holdout": cands["caruana"][5], "logreg_holdout": cands["logreg"][5],
                      "mean_holdout": cands["mean"][5], "caruana_weights": cw,
                      "ovo_galaxy_star": b_ovo, "cleared_target": bool(cleared)})
    exp.note(note)
    exp.commit(); render_all()

    render_all_panels(ydev_names, weighted_predict(boof, bw, labels=CLASSES), boof, labels=CLASSES,
                      fold_scores={f"v15_{best_name}": per_fold}, prefix="v15")
    render_html(prefix="v15", title="S6E6 v15 — supervised blend (toward 0.9700)")
    verdict("v15", b_hold, time.time() - t0, best=0.9690, oof=b_oof)

    # write every blend candidate + the winner as v15_final (endgame: anchor=blend, hedge=rm single)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    for name in ("caruana", "mean", "logreg"):
        _, _, tp, w, _, _ = cands[name]
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(tp, w, labels=CLASSES)}).to_csv(
            SUBMISSIONS / f"v15_blend_{name}.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(btest, bw, labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v15_final.csv", index=False)
    print(f"\n[submission] v15_final.csv (={best_name}) + v15_blend_{{caruana,mean,logreg}}.csv")
    print(f"[verdict] {note}")


if __name__ == "__main__":
    main()
