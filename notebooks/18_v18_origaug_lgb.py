"""v18 — original-SDSS17 CONCAT augmentation, AXIS confirmation on LGBM (cheap go/no-go).

See docs/LADDER.md. We locked at ~0.9698 calling it a real ceiling — half wrong: the architecture
axis is exhausted, but we never tried adding the ORIGINAL fedesoriano SDSS17 file as down-weighted
extra training rows. Our v2 killed the *leak-match* (0% exact join) — but the community CONCATS
(~100K real labeled rows, sample_weight 0.35), which is different and lifts the top public single
model to 0.96980.

This rung proves the AXIS is real + balanced-accuracy-safe on the cheap LGBM leg (native
sample_weight) BEFORE we plumb per-row weights into the custom RealMLP (v19, the primary lever).

Gemini guardrail: a naive global 0.35 weight down-weights OOD rows but does NOT correct the prior
shift (original class balance != competition's), which poisons per-class recall under balanced
accuracy. Fix = PER-CLASS weight  w_c = (N_c_comp / N_c_orig) * 0.35  so the added per-class
weight-mass matches competition proportions. Validation is on COMPETITION rows only (original is
appended to every TRAIN fold, never to val) — leak-safe.

GATE → v19 iff: OOF lift >= +0.0003 AND no per-class (QSO/STAR) recall regression vs the no-aug
baseline. Else STOP and revisit the weighting.

Run locally (smoke, no original → baseline only):  PYTHONPATH=. python notebooks/18_v18_origaug_lgb.py
Cloud (real): bootstrap downloads comp + fedesoriano SDSS17 into data/raw + data/external.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.augment import BASE_W, per_class_weights, prep_original, unify_categories
from src.config import (CLASSES, ID, MODEL_SEED, PROBS, SUBMISSIONS, TARGET)
from src.cv import stratified_folds
from src.fe_realmlp import build_rich_features
from src.metrics import competition_score, tune_class_weights, weighted_predict
from src.models import LGB_PARAMS
from src.observer import Experiment

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
N_FOLDS = 5


def _load():
    """Returns (train, test, original_or_None, synthetic_flag)."""
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], data.load_original(), False
    print("WARN data/raw empty -> synthetic fallback (smoke; no augmentation possible).")
    full = data.synthetic_fallback(n=8000)
    tr = full.iloc[:6000].reset_index(drop=True)
    te = full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True)
    return tr, te, None, True


def _lgb_oof(Xdev, ydev, Xhold, Xte, *, aug=None):
    """StratifiedKFold OOF on comp dev rows. If aug=(Xa, ya, wa), append it to EVERY train fold with
    sample_weight (comp rows weight 1.0); validation stays on comp rows only. Returns oof/hold/test
    proba + fold val indices."""
    import lightgbm as lgb
    p = {**LGB_PARAMS}
    oof = np.zeros((len(Xdev), len(CLASSES)))
    hold = np.zeros((len(Xhold), len(CLASSES)))
    test = np.zeros((len(Xte), len(CLASSES)))
    fva = []
    ydev = np.asarray(ydev)
    for i, (tr, va) in enumerate(stratified_folds(ydev, N_FOLDS), 1):
        Xtr, ytr = Xdev.iloc[tr], ydev[tr]
        w = np.ones(len(tr), dtype=float)
        if aug is not None:
            Xa, ya, wa = aug
            Xtr = pd.concat([Xtr, Xa], ignore_index=True)
            ytr = np.concatenate([ytr, ya])
            w = np.concatenate([w, wa])
        print(f"  fold {i}/{N_FOLDS}  train={len(Xtr):,} (aug={'on' if aug else 'off'})  val={len(va):,}")
        m = lgb.LGBMClassifier(**p)
        m.fit(Xtr, ytr, sample_weight=w,
              eval_set=[(Xdev.iloc[va], ydev[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(Xdev.iloc[va])
        hold += m.predict_proba(Xhold) / N_FOLDS
        test += m.predict_proba(Xte) / N_FOLDS
        fva.append(va)
    return oof, hold, test, fva


def _recall(y_names, pred_names) -> dict:
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
    print(f"[load] train={train.shape} test={test.shape} original={None if orig is None else orig.shape}")

    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte_rich, _, _ = build_rich_features(test, fit=False, state=state)

    # --- augmentation frame, built BEFORE the holdout slice so categories can be unified across
    # ALL frames (LightGBM requires the concatenated train frame's category sets to match its val).
    aug_src = None
    orig_prep = prep_original(orig)
    if orig_prep is not None and len(orig_prep) > 0:
        ya = orig_prep[TARGET].map(CLS2INT).to_numpy()
        Xa, _, _ = build_rich_features(orig_prep, fit=False, state=state)   # SAME pipeline, aligned cols
        Xa = Xa[Xrich.columns]                                             # identical column order
        unify_categories([Xrich, Xte_rich, Xa])                           # shared category sets (LGB fix)
        aug_src = (Xa.reset_index(drop=True), ya)

    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]
    yhold_names, ydev_names = INT2CLS[yhold], INT2CLS[ydev]

    have_aug = aug_src is not None
    if have_aug:
        Xa, ya = aug_src
        n_comp = np.bincount(ydev, minlength=3); n_orig = np.bincount(ya, minlength=3)
        print(f"[aug] original rows={len(ya):,}  N comp={dict(zip(CLASSES, n_comp.tolist()))}  "
              f"orig={dict(zip(CLASSES, n_orig.tolist()))}")
    else:
        print("[aug] no original data -> baseline only (axis test cannot run; smoke/local).")

    # --- baseline (no aug) ---
    print("\n=== baseline LGBM (no augmentation) ===")
    b_oof, b_hold, b_test, fva = _lgb_oof(Xdev, ydev, Xhold, Xte_rich, aug=None)
    bw, b_s_oof, b_s_hold = _score(b_oof, b_hold, ydev, yhold_names)
    b_rec = _recall(ydev_names, weighted_predict(b_oof, bw, labels=CLASSES))
    print(f"[baseline] OOF {b_s_oof:.5f} | holdout {b_s_hold:.5f} | recall {b_rec}")

    # --- augmented arms: flat (proven public recipe) vs prior (balanced-acc corrected). Both have
    # mean original-row weight = BASE_W, so neither over-weights OOD data (the v18-v1 mistake). ---
    arms = {}
    if have_aug:
        for mode in ("flat", "prior"):
            wa = per_class_weights(ydev, ya, mode=mode)
            print(f"\n=== augmented LGBM [{mode}]  mean_w={wa.mean():.3f}  uniq={np.round(np.unique(wa),3).tolist()} ===")
            o, h, te, _ = _lgb_oof(Xdev, ydev, Xhold, Xte_rich, aug=(Xa, ya, wa))
            w_, s_o, s_h = _score(o, h, ydev, yhold_names)
            rec = _recall(ydev_names, weighted_predict(o, w_, labels=CLASSES))
            arms[mode] = dict(oof=o, hold=h, test=te, w=w_, s_oof=s_o, s_hold=s_h, rec=rec)
            print(f"[{mode}] OOF {s_o:.5f} (d{s_o-b_s_oof:+.5f}) | holdout {s_h:.5f} (d{s_h-b_s_hold:+.5f}) | recall {rec}")

    # --- pick best arm by OOF + GATE ---
    best_mode = max(arms, key=lambda m: arms[m]["s_oof"]) if arms else None
    if best_mode:
        A = arms[best_mode]
        a_oof, a_hold, a_test, aw = A["oof"], A["hold"], A["test"], A["w"]
        a_s_oof, a_s_hold, a_rec = A["s_oof"], A["s_hold"], A["rec"]
    else:
        a_oof = a_hold = a_test = None; aw = bw; a_s_oof = a_s_hold = float("nan"); a_rec = {}
    d_oof = (a_s_oof - b_s_oof) if best_mode else float("nan")
    qso_drop = (a_rec.get("QSO", 0) - b_rec.get("QSO", 0)) if best_mode else float("nan")
    star_drop = (a_rec.get("STAR", 0) - b_rec.get("STAR", 0)) if best_mode else float("nan")
    recall_safe = bool(best_mode) and (qso_drop >= -0.0005) and (star_drop >= -0.0005)
    gate_pass = bool(best_mode) and (d_oof >= 0.0003) and recall_safe
    note = (f"aug-axis: baseline OOF {b_s_oof:.5f} -> best[{best_mode}] {a_s_oof:.5f} (delta {d_oof:+.5f}); "
            f"QSO {qso_drop:+.4f}, STAR {star_drop:+.4f}; GATE "
            f"{'PASS -> v19 (RealMLP aug, mode='+str(best_mode)+')' if gate_pass else 'FAIL -> LGBM saturated or axis weak; decide v19 on community evidence'}")
    print(f"\n[GATE] {note}")

    # --- persist + diary ---
    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(PROBS / "v18_lgb_baseline_oof.npy", b_oof); np.save(PROBS / "v18_lgb_baseline_test.npy", b_test)
    if a_oof is not None:
        np.save(PROBS / "v18_lgb_aug_oof.npy", a_oof); np.save(PROBS / "v18_lgb_aug_test.npy", a_test)

    has_aug = a_oof is not None
    use_oof = a_oof if has_aug else b_oof
    use_w = aw if has_aug else bw
    best_s_oof = a_s_oof if has_aug else b_s_oof
    best_s_hold = a_s_hold if has_aug else b_s_hold
    per_fold = [competition_score(ydev_names[va], weighted_predict(use_oof[va], use_w, labels=CLASSES))
                for va in fva]

    exp = Experiment.start(
        version="v18", parent="v15",
        hypothesis="Concatenating ~100K original SDSS17 rows into each LGBM training fold (val on "
                   "competition rows only), with mean original-row weight = BASE_W (proven recipe "
                   "magnitude) under flat vs balanced-acc-corrected schemes, lifts the LGBM leg's OOF "
                   "balanced-acc — the cheap axis probe for the original-data CONCAT lever.",
        predicted_delta=0.0004, confidence="medium",
        feature_changes=[], pipeline_changes=[
            "original-SDSS17 concat augmentation into each train fold (val=comp rows only, leak-safe)",
            "two weight arms: flat (public 0.35/row) vs prior (class-corrected, mean=0.35/row)",
            "LGBM axis-confirmation leg (native sample_weight)"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=best_s_oof, oof_score_per_fold=per_fold, holdout_score=best_s_hold,
               runtime_sec=time.time() - t0,
               extra={"baseline_oof": b_s_oof, "baseline_hold": b_s_hold, "baseline_recall": b_rec,
                      "best_mode": best_mode, "aug_oof": a_s_oof, "aug_hold": a_s_hold, "aug_recall": a_rec,
                      "arms": {m: {"oof": A["s_oof"], "hold": A["s_hold"], "rec": A["rec"]} for m, A in arms.items()},
                      "delta_oof": d_oof, "qso_recall_delta": qso_drop, "star_recall_delta": star_drop,
                      "gate_pass": gate_pass, "n_original": (int(len(ya)) if have_aug else 0),
                      "base_w": BASE_W})
    exp.note(note)
    exp.commit()
    try:
        from src.diary import render_all
        render_all()
    except Exception as e:  # viz must never lose the experiment record
        print(f"[warn] render_all skipped: {e}")

    if a_test is not None:
        SUBMISSIONS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: test[ID], TARGET: weighted_predict(a_test, aw, labels=CLASSES)}).to_csv(
            SUBMISSIONS / "v18_lgb_aug.csv", index=False)
        print("[submission] v18_lgb_aug.csv (axis-test single leg; not a finals candidate)")
    print(f"\n[verdict] {note}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
