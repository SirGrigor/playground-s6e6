"""v11 — domain-knowledge features: probe-first, then a leakage-safe ablation.

The 2026-06-03 domain-research swarm NAMED our 0.9664 ceiling as a MISSING-CHANNEL problem
(morphology / mid-IR absent, not latent in ugriz) and left a short list of features derivable
from OUR columns that an axis-aligned GBDT cannot trivially reconstruct: Galactic latitude |b|
(spatial prior orthogonal to photometry+z), the stellar-locus distance L (distance-to-a-curve),
the −9999 dropout PATTERN (Lyman-break signal build_features throws away), and long-baseline
colors. The swarm was emphatic these only help IF the synthetic generator preserved the physics.

So this notebook is discovery-first (feedback_discovery-first-diagnose-before-build):
  1. PROBES T1–T4 print confirm-or-kill verdicts on whether the physics survived generation.
  2. ABLATION: LGBM on BASELINE features vs BASELINE+DOMAIN, same folds/holdout, reporting the
     bal-acc delta, per-class recall, and the crux GALAXY|STAR OvO for both. The stellar locus
     is fit on DEV STARs only (held-out holdout never enters the fit).

Honest prior (our 0.9664 ceiling + 4× over-prediction history): predicted_delta 0.0005, low conf.
The deliverable is the VERDICT (did domain knowledge find signal outside the raw bands?), logged
to the diary — not a guaranteed lift. A v11_domain.csv is written so a positive result is submittable.

Run locally:  PYTHONPATH=. python notebooks/12_v11_domain.py   (synthetic fallback if no data/raw)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
from src.dashboard import goal_banner, render_html, scoreboard, training, verdict
from src.diagnostics import error_cell_report
from src.diary import render_all
from src.domain import (add_domain_features, categorical_color_probe, fit_stellar_locus,
                        latitude_class_probe, sentinel_class_probe, stellar_locus_tightness)
from src.features import build_features
from src.metrics import (competition_score, multiclass_auc_report, tune_class_weights,
                         weighted_predict)
from src.models import lgb_oof
from src.observer import Experiment

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("⚠ data/raw empty — synthetic fallback (smoke test).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _run_lgb(Xdev, ydev, Xhold, yhold, ydev_names, yhold_names, fold_va_holder, Xte, tag):
    """OOF+holdout LGBM, tuned for balanced accuracy. Returns dict of scores + probas."""
    oof, hold, tst, fva = lgb_oof(Xdev, ydev, Xhold, Xte,
                                  on_fold=lambda i, n, t=tag: training(f"{t} fold {i}", i, n))
    fold_va_holder.append(fva)
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    pred_hold = weighted_predict(hold, w, labels=CLASSES)
    s_hold = competition_score(yhold_names, pred_hold)
    per_fold = [competition_score(ydev_names[va], weighted_predict(oof[va], w, labels=CLASSES)) for va in fva]
    ovo = multiclass_auc_report(ydev_names, oof, labels=CLASSES)["ovo_per_pair"].get("GALAXY|STAR", float("nan"))
    return {"tag": tag, "oof": s_oof, "holdout": s_hold, "per_fold": per_fold, "w": w,
            "oof_proba": oof, "hold_proba": hold, "test_proba": tst,
            "pred_oof": weighted_predict(oof, w, labels=CLASSES), "ovo_gs": ovo}


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape}  test={test.shape}")

    # base feature frame (v6/v10 baseline) + categorical alignment to test
    Xbase, feats = build_features(train, use_extra=True)
    Xte_base, _ = build_features(test, use_extra=True)
    for c in feats:
        if str(Xbase[c].dtype) == "category":
            Xte_base[c] = pd.Categorical(Xte_base[c], categories=Xbase[c].cat.categories)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    y_names_all = INT2CLS[y_all]

    goal_banner("v11", "domain features (Galactic |b|, stellar-locus L, dropout flags, long colors)",
                "did domain knowledge find signal OUTSIDE the raw bands? probe-first, then ablate", best=0.9664)

    # ---------- PROBES T1–T4 (confirm-or-kill, run on full train w/ labels) ----------
    print("\n" + "=" * 78 + "\n  CONFIRM-OR-KILL PROBES — did the synthetic generator keep the physics?\n" + "=" * 78)
    t1_txt, t1_rho = latitude_class_probe(Xbase, y_names_all, CLASSES); print(t1_txt)
    t2_txt, t2_ratio = stellar_locus_tightness(Xbase, y_names_all); print(t2_txt)
    t3_txt, t3_l1 = sentinel_class_probe(train, y_names_all, CLASSES); print(t3_txt)
    t4_txt, t4_mono = categorical_color_probe(Xbase, train, y_names_all); print(t4_txt)

    # ---------- holdout split + stellar-locus ridge fit on DEV STARs only ----------
    dev_idx, hold_idx = data.holdout_split(train)
    ridge = fit_stellar_locus(Xbase.iloc[dev_idx], (y_all[dev_idx] == CLS2INT["STAR"]))
    # build the augmented frames (locus applied everywhere from the dev-STAR ridge)
    Xdom_all, new_feats = add_domain_features(Xbase, train, ridge)
    Xte_dom, _ = add_domain_features(Xte_base, test, ridge)
    print(f"\n[domain] +{len(new_feats)} features: {new_feats}")

    Xb_dev, Xb_hold = Xbase.iloc[dev_idx].reset_index(drop=True), Xbase.iloc[hold_idx].reset_index(drop=True)
    Xd_dev, Xd_hold = Xdom_all.iloc[dev_idx].reset_index(drop=True), Xdom_all.iloc[hold_idx].reset_index(drop=True)
    ydev, yhold = y_all[dev_idx], y_all[hold_idx]
    ydev_names, yhold_names = INT2CLS[ydev], INT2CLS[yhold]

    # ---------- ABLATION: baseline vs +domain (same folds, same holdout) ----------
    print("\n" + "=" * 78 + "\n  ABLATION — LGBM baseline vs baseline+domain\n" + "=" * 78)
    fva_holder = []
    base = _run_lgb(Xb_dev, ydev, Xb_hold, yhold, ydev_names, yhold_names, fva_holder, Xte_base, "baseline")
    dom = _run_lgb(Xd_dev, ydev, Xd_hold, yhold, ydev_names, yhold_names, fva_holder, Xte_dom, "domain")

    scoreboard([{"name": base["tag"], "oof": base["oof"], "holdout": base["holdout"]},
                {"name": dom["tag"], "oof": dom["oof"], "holdout": dom["holdout"]}], best=0.9664)
    d_oof, d_hold = dom["oof"] - base["oof"], dom["holdout"] - base["holdout"]
    print(f"\n[delta] OOF {d_oof:+.5f}  holdout {d_hold:+.5f}  "
          f"| GALAXY|STAR OvO {base['ovo_gs']:.4f} → {dom['ovo_gs']:.4f} ({dom['ovo_gs']-base['ovo_gs']:+.4f})")
    print("\n[per-class recall — domain model OOF]")
    print(error_cell_report(ydev_names, dom["pred_oof"], CLASSES))

    # ---------- record verdict to the diary ----------
    verdict_txt = ("domain features LIFTED the holdout" if d_hold > 0.0003 else
                   "domain features NET-NEUTRAL (≤ +0.0003) — ceiling is a missing channel, not latent in ugriz")
    exp = Experiment.start(
        version="v11", parent="v8",
        hypothesis="Domain-knowledge features derivable from our columns (Galactic |b|, stellar-locus L, "
                   "−9999 dropout flags, long-baseline colors) extract class signal a GBDT misses — IF the "
                   "synthetic generator preserved the physics (probes T1–T4 decide).",
        predicted_delta=0.0005, confidence="low",
        feature_changes=[f"+{c}" for c in new_feats],
        pipeline_changes=["domain probes T1–T4 (confirm-or-kill)", "baseline-vs-domain ablation",
                          "stellar-locus ridge fit on DEV STARs only"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=dom["oof"], oof_score_per_fold=dom["per_fold"], holdout_score=dom["holdout"],
               runtime_sec=time.time() - t0,
               extra={"baseline_oof": base["oof"], "baseline_holdout": base["holdout"],
                      "delta_oof": d_oof, "delta_holdout": d_hold,
                      "ovo_galaxy_star": {"baseline": base["ovo_gs"], "domain": dom["ovo_gs"]},
                      "probes": {"T1_latitude_spearman": t1_rho, "T2_locus_ratio": t2_ratio,
                                 "T3_sentinel_L1": t3_l1, "T4_spectral_g_r_range": t4_mono},
                      "new_features": new_feats})
    exp.note(f"probes: T1 rho={t1_rho:+.3f} T2 ratio={t2_ratio:.3f} T3 L1={t3_l1:.3f}; "
             f"ablation holdout {d_hold:+.5f} → {verdict_txt}")
    exp.commit(); render_all()

    # ---------- dashboard + submittable artifact ----------
    from src.viz import render_all_panels
    render_all_panels(ydev_names, dom["pred_oof"], dom["oof_proba"], labels=CLASSES,
                      fold_scores={"v11_domain": dom["per_fold"]}, prefix="v11")
    render_html(prefix="v11", title="S6E6 v11 — domain features (probe-first ablation)")
    verdict("v11", dom["holdout"], time.time() - t0, best=0.9664, oof=dom["oof"])

    SUBMISSIONS.mkdir(parents=True, exist_ok=True); PROBS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(dom["test_proba"], dom["w"], labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v11_domain.csv", index=False)
    np.save(PROBS / "v11_domain_test_proba.npy", dom["test_proba"])
    print(f"\n[submission] v11_domain.csv  dist="
          f"{pd.read_csv(SUBMISSIONS/'v11_domain.csv')[TARGET].value_counts().to_dict()}")
    print(f"[verdict] {verdict_txt}")


if __name__ == "__main__":
    main()
