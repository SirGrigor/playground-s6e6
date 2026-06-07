"""v19 — original-SDSS17 CONCAT augmentation on the RealMLP+FE leg (the PRIMARY ceiling-break lever).

See docs/LADDER.md. v18 confirms the augmentation AXIS cheaply on LGBM; v19 applies it to our strong
leg — the v14 RealMLP+rich-FE single (0.96901) we want to push toward >0.9696 (Gemini's prediction;
the top public single RealMLP hits 0.96980 with this exact lever).

Faithful to v14: SAME static rich FE + per-fold per-class OOF target encoding (`_fold_te`). The only
additions are (1) the original rows concatenated into each fold's RealMLP training set, TE-encoded by
the SAME fold-train mapping (leak-safe — TE is fit on competition fold-train only, then applied to
the appended original rows and to val/hold/test); (2) per-row sample_weight (comp=1.0, original=w_c
per-class); (3) the balanced-softmax prior_counts pinned to the COMPETITION fold-train counts so the
metric-aware loss still targets the competition class balance, not the augmented mix.

Runs baseline (no aug) AND augmented on identical folds → clean isolated delta.

GATE → v20 iff: augmented OOF clears ~0.9696 AND holds on the holdout (vs baseline). Flat → the
axis is weaker than the community single claims; reassess before the artifact-feature rung.

Run locally (smoke, tiny cfg, no original → baseline only):
  PYTHONPATH=. python notebooks/19_v19_origaug_realmlp.py
Cloud: GPU REQUIRED (~40–80 min: RealMLP × 5 folds × {baseline, aug}). bootstrap downloads both files.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.augment import per_class_weights, prep_original
from src.config import CLASSES, ID, MODEL_SEED, PROBS, SUBMISSIONS, TARGET
from src.cv import stratified_folds
from src.fe_realmlp import _fold_te, build_rich_features
from src.metrics import competition_score, tune_class_weights, weighted_predict
from src.observer import Experiment
from src.realmlp import realmlp_fit_predict

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
NC = len(CLASSES)
N_FOLDS = 5
SEED = MODEL_SEED
# v19 retest 2026-06-07: prior-corrected weighting HURT RealMLP (-0.0005), likely double-correcting
# with our balanced-softmax loss. Test the EXACT proven public recipe = flat 0.35/row (no class
# correction; let balanced-softmax own the prior). flat|prior.
WEIGHT_MODE = "flat"
SMOKE_CFG = {"n_ens": 2, "hidden_dims": [32], "epochs": 1, "eval_bs": 4096,
             "pbld_hidden_dim": 8, "pbld_out_dim": 3}


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], data.load_original(), False
    print("WARN data/raw empty -> synthetic fallback (smoke; no augmentation).")
    full = data.synthetic_fallback(n=6000)
    tr = full.iloc[:4500].reset_index(drop=True)
    te = full.iloc[4500:].drop(columns=[TARGET]).reset_index(drop=True)
    return tr, te, None, True


def _z(n):
    return np.zeros((n, NC), "float32")


def _run(Xdev, ydev, Xhold, Xte, combo_cols, cfg, *, aug=None, on_fold=None):
    """Per-fold RealMLP+FE OOF. If aug=(Xa_rich, ya, wa): append original rows to each fold-train
    (TE-encoded by the fold-train mapping), sample-weighted, prior pinned to competition counts."""
    oof, hold, test, fva = _z(len(Xdev)), _z(len(Xhold)), _z(len(Xte)), []
    cat = lambda X, t: pd.concat([X.reset_index(drop=True), t.reset_index(drop=True)], axis=1)
    for i, (tr, va) in enumerate(stratified_folds(ydev, N_FOLDS), 1):
        if on_fold:
            on_fold(i, N_FOLDS)
        Xtr, Xva = Xdev.iloc[tr], Xdev.iloc[va]
        evals = {"va": Xva, "hold": Xhold, "test": Xte}
        if aug is not None:
            evals["aug"] = aug[0]
        te_tr, te_ev, _ = _fold_te(Xtr, ydev[tr], evals, combo_cols, NC, SEED + i)
        rm_tr, rm_va = cat(Xtr, te_tr), cat(Xva, te_ev["va"])
        rm_hold, rm_te = cat(Xhold, te_ev["hold"]), cat(Xte, te_ev["test"])

        ytr = ydev[tr]
        sw = None
        prior_counts = np.bincount(ytr, minlength=NC)        # pin balanced-softmax to comp prior
        if aug is not None:
            Xa_rich, ya, wa = aug
            rm_aug = cat(Xa_rich, te_ev["aug"])
            rm_tr = pd.concat([rm_tr, rm_aug], ignore_index=True)
            ytr = np.concatenate([ytr, ya])
            sw = np.concatenate([np.ones(len(tr), dtype=float), wa])

        val_p, ev = realmlp_fit_predict(rm_tr, ytr, rm_va, ydev[va], {"hold": rm_hold, "test": rm_te},
                                        {**cfg, "seed": SEED + i}, sample_weight=sw, prior_counts=prior_counts)
        oof[va] = val_p
        hold += ev["hold"] / N_FOLDS
        test += ev["test"] / N_FOLDS
        fva.append(va)
    return oof, hold, test, fva


def _recall(y_names, pred_names):
    from sklearn.metrics import recall_score
    r = recall_score(y_names, pred_names, labels=CLASSES, average=None, zero_division=0)
    return {c: float(v) for c, v in zip(CLASSES, r)}


def _score(oof, hold, ydev, yhold_names):
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
    return w, s_oof, s_hold


def main() -> None:
    t0 = time.time()
    train, test, orig, synthetic = _load()
    cfg = SMOKE_CFG if synthetic else {}
    print(f"[load] train={train.shape} test={test.shape} original={None if orig is None else orig.shape}")

    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte_rich, _, _ = build_rich_features(test, fit=False, state=state)
    combo_cols = info["combo_cols"]
    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]
    yhold_names, ydev_names = INT2CLS[yhold], INT2CLS[ydev]

    aug = None
    orig_prep = prep_original(orig)
    if orig_prep is not None:
        ya = orig_prep[TARGET].map(CLS2INT).to_numpy()
        Xa, _, _ = build_rich_features(orig_prep, fit=False, state=state)
        Xa = Xa[Xdev.columns].reset_index(drop=True)
        wa = per_class_weights(ydev, ya, mode=WEIGHT_MODE)
        aug = (Xa, ya, wa)
        print(f"[aug] mode={WEIGHT_MODE}  original rows={len(ya):,}  "
              f"weight uniq={np.round(np.unique(wa),4).tolist()}  mean={wa.mean():.3f}")
    else:
        print("[aug] no original data -> baseline only (smoke/local).")

    print("\n=== baseline RealMLP+FE (no augmentation) ===")
    b_oof, b_hold, b_test, fva = _run(Xdev, ydev, Xhold, Xte_rich, combo_cols, cfg,
                                      aug=None, on_fold=lambda i, n: print(f"  baseline fold {i}/{n}"))
    bw, b_s_oof, b_s_hold = _score(b_oof, b_hold, ydev, yhold_names)
    b_rec = _recall(ydev_names, weighted_predict(b_oof, bw, labels=CLASSES))
    print(f"[baseline] OOF {b_s_oof:.5f} | holdout {b_s_hold:.5f} | recall {b_rec}")

    a_s_oof = a_s_hold = float("nan"); a_rec = {}; a_oof = a_hold = a_test = None; aw = bw
    if aug is not None:
        print("\n=== augmented RealMLP+FE (original concat, per-class weighted, comp-prior loss) ===")
        a_oof, a_hold, a_test, _ = _run(Xdev, ydev, Xhold, Xte_rich, combo_cols, cfg,
                                        aug=aug, on_fold=lambda i, n: print(f"  aug fold {i}/{n}"))
        aw, a_s_oof, a_s_hold = _score(a_oof, a_hold, ydev, yhold_names)
        a_rec = _recall(ydev_names, weighted_predict(a_oof, aw, labels=CLASSES))
        print(f"[augmented] OOF {a_s_oof:.5f} | holdout {a_s_hold:.5f} | recall {a_rec}")

    d_oof = (a_s_oof - b_s_oof) if aug is not None else float("nan")
    d_hold = (a_s_hold - b_s_hold) if aug is not None else float("nan")
    qso_d = (a_rec.get("QSO", 0) - b_rec.get("QSO", 0)) if aug is not None else float("nan")
    star_d = (a_rec.get("STAR", 0) - b_rec.get("STAR", 0)) if aug is not None else float("nan")
    clears = (aug is not None) and (a_s_oof >= 0.9696)
    holds = (aug is not None) and (d_hold >= -0.0001)
    gate_pass = clears and holds and (qso_d >= -0.0005) and (star_d >= -0.0005)
    note = (f"rm aug: baseline OOF {b_s_oof:.5f} -> aug {a_s_oof:.5f} (delta {d_oof:+.5f}, hold {d_hold:+.5f}); "
            f"QSO {qso_d:+.4f}, STAR {star_d:+.4f}; "
            f"GATE {'PASS -> v20 (artifact features)' if gate_pass else 'FAIL -> reassess axis'}")
    print(f"\n[GATE] {note}")

    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(PROBS / "v19_rm_baseline_oof.npy", b_oof); np.save(PROBS / "v19_rm_baseline_test.npy", b_test)
    if a_oof is not None:
        np.save(PROBS / "v19_rm_aug_oof.npy", a_oof); np.save(PROBS / "v19_rm_aug_test.npy", a_test)

    use_oof = a_oof if (aug is not None) else b_oof
    use_w = aw if (aug is not None) else bw
    best_s_oof = a_s_oof if (aug is not None) else b_s_oof
    best_s_hold = a_s_hold if (aug is not None) else b_s_hold
    per_fold = [competition_score(ydev_names[va], weighted_predict(use_oof[va], use_w, labels=CLASSES))
                for va in fva]

    exp = Experiment.start(
        version="v19", parent="v15",
        hypothesis=f"RETEST [{WEIGHT_MODE}]: prior-corrected concat augmentation HURT RealMLP "
                   f"(-0.0005, double-correcting with balanced-softmax). The proven public recipe is "
                   f"flat 0.35/row; let balanced-softmax own the prior. Test whether flat-weighted "
                   f"original-SDSS17 concat lifts the v14 RealMLP+FE single (0.96896) past it.",
        predicted_delta=0.0004, confidence="low",
        feature_changes=[], pipeline_changes=[
            f"original-SDSS17 concat into each RealMLP fold-train (leak-safe TE; val=comp rows only)",
            f"per-row sample_weight mode={WEIGHT_MODE} (mean 0.35/row) via custom RealMLP loss",
            "balanced-softmax prior_counts pinned to competition fold-train"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=best_s_oof, oof_score_per_fold=per_fold, holdout_score=best_s_hold,
               runtime_sec=time.time() - t0,
               extra={"baseline_oof": b_s_oof, "baseline_hold": b_s_hold, "baseline_recall": b_rec,
                      "aug_oof": a_s_oof, "aug_hold": a_s_hold, "aug_recall": a_rec,
                      "delta_oof": d_oof, "delta_hold": d_hold, "qso_recall_delta": qso_d,
                      "star_recall_delta": star_d, "gate_pass": gate_pass,
                      "n_original": (0 if aug is None else int(len(aug[1])))})
    exp.note(note)
    exp.commit()
    try:
        from src.diary import render_all
        render_all()
    except Exception as e:
        print(f"[warn] render_all skipped: {e}")

    if a_test is not None:
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(a_test, aw, labels=CLASSES)}).to_csv(
            SUBMISSIONS / "v19_rm_aug.csv", index=False)
        print("[submission] v19_rm_aug.csv (augmented single leg)")
    print(f"\n[verdict] {note}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
