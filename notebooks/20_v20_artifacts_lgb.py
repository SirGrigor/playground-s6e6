"""v20 — generator-artifact features, AXIS confirmation on LGBM (cheap go/no-go).

See docs/LADDER.md. Augmentation (v18/v19) is refuted. v20 tests the ORTHOGONAL lever: synthetic-
generator quantization fingerprints in the low-order decimal digits of u/g/r/i/z/redshift (digit/mod/
decimal features). These are tree-friendly (categorical splits), so LGBM is both the cheap probe AND
a directly useful stack leg — and they add no external rows, so they dodge the real-vs-synthetic
distribution gap that sank v19.

Runs baseline (rich FE) vs rich FE + artifacts on identical folds -> clean isolated delta.
GATE -> v20-RealMLP / v21 iff: OOF lift >= +0.0003 AND no QSO/STAR recall regression.

Run locally (smoke): PYTHONPATH=. python notebooks/20_v20_artifacts_lgb.py
Cloud: CPU only (LGBM); bootstrap mounts the competition data.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.artifacts import add_artifact_features
from src.config import CLASSES, ID, PROBS, SUBMISSIONS, TARGET
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
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("WARN data/raw empty -> synthetic fallback (smoke).")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _lgb_oof(Xdev, ydev, Xhold, Xte):
    import lightgbm as lgb
    oof = np.zeros((len(Xdev), len(CLASSES))); hold = np.zeros((len(Xhold), len(CLASSES)))
    test = np.zeros((len(Xte), len(CLASSES))); fva = []
    ydev = np.asarray(ydev)
    for i, (tr, va) in enumerate(stratified_folds(ydev, N_FOLDS), 1):
        print(f"  fold {i}/{N_FOLDS}")
        m = lgb.LGBMClassifier(**LGB_PARAMS)
        m.fit(Xdev.iloc[tr], ydev[tr], eval_set=[(Xdev.iloc[va], ydev[va])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[va] = m.predict_proba(Xdev.iloc[va])
        hold += m.predict_proba(Xhold) / N_FOLDS
        test += m.predict_proba(Xte) / N_FOLDS
        fva.append(va)
    return oof, hold, test, fva


def _recall(y_names, pred_names):
    from sklearn.metrics import recall_score
    r = recall_score(y_names, pred_names, labels=CLASSES, average=None, zero_division=0)
    return {c: float(v) for c, v in zip(CLASSES, r)}


def _score(oof, hold, ydev, yhold_names):
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    return w, s_oof, competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))


def main() -> None:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[load] train={train.shape} test={test.shape}")

    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte_rich, _, _ = build_rich_features(test, fit=False, state=state)
    art_tr = add_artifact_features(train)
    art_te = add_artifact_features(test)
    Xfull = pd.concat([Xrich, art_tr], axis=1)
    Xte_full = pd.concat([Xte_rich, art_te], axis=1)
    print(f"[artifacts] +{art_tr.shape[1]} features ({list(art_tr.columns[:6])}...)")

    dev_idx, hold_idx = data.holdout_split(train)
    yhold_names = INT2CLS[y_all[hold_idx]]; ydev_names = INT2CLS[y_all[dev_idx]]
    ydev, yhold = y_all[dev_idx], y_all[hold_idx]

    def arm(X, Xte_, label):
        Xd, Xh = X.iloc[dev_idx].reset_index(drop=True), X.iloc[hold_idx].reset_index(drop=True)
        print(f"\n=== {label} ({X.shape[1]} feats) ===")
        o, h, te, fva = _lgb_oof(Xd, ydev, Xh, Xte_)
        w, s_o, s_h = _score(o, h, ydev, yhold_names)
        rec = _recall(ydev_names, weighted_predict(o, w, labels=CLASSES))
        print(f"[{label}] OOF {s_o:.5f} | holdout {s_h:.5f} | recall {rec}")
        return dict(oof=o, hold=h, test=te, w=w, s_oof=s_o, s_hold=s_h, rec=rec, fva=fva)

    base = arm(Xrich, Xte_rich, "baseline (rich FE)")
    art = arm(Xfull, Xte_full, "rich FE + artifacts")

    d_oof = art["s_oof"] - base["s_oof"]; d_hold = art["s_hold"] - base["s_hold"]
    qso_d = art["rec"]["QSO"] - base["rec"]["QSO"]; star_d = art["rec"]["STAR"] - base["rec"]["STAR"]
    recall_safe = (qso_d >= -0.0005) and (star_d >= -0.0005)
    gate_pass = (d_oof >= 0.0003) and recall_safe
    note = (f"artifacts: baseline OOF {base['s_oof']:.5f} -> +art {art['s_oof']:.5f} (delta {d_oof:+.5f}, "
            f"hold {d_hold:+.5f}); QSO {qso_d:+.4f}, STAR {star_d:+.4f}; GATE "
            f"{'PASS -> v20-RealMLP / v21' if gate_pass else 'FAIL -> artifacts add no LGBM signal'}")
    print(f"\n[GATE] {note}")

    PROBS.mkdir(parents=True, exist_ok=True)
    np.save(PROBS / "v20_lgb_base_oof.npy", base["oof"]); np.save(PROBS / "v20_lgb_art_oof.npy", art["oof"])
    np.save(PROBS / "v20_lgb_art_test.npy", art["test"])
    per_fold = [competition_score(ydev_names[va], weighted_predict(art["oof"][va], art["w"], labels=CLASSES))
                for va in art["fva"]]

    exp = Experiment.start(
        version="v20", parent="v15",
        hypothesis="Generator-artifact features (low-order decimal digit/mod/frac of u/g/r/i/z/redshift) "
                   "add synthetic-quantization signal the rich physical FE misses; tree-friendly, so the "
                   "LGBM leg's OOF balanced-acc lifts. Orthogonal to the refuted augmentation (adds "
                   "columns, not rows -> no distribution gap).",
        predicted_delta=0.0004, confidence="low",
        feature_changes=[f"+{art_tr.shape[1]} artifact features (digit k=2/4/6, mod10/100, frac)"],
        pipeline_changes=["LGBM artifact axis-probe (rich FE vs rich FE + artifacts, identical folds)"],
        cloud_or_local="cloud" if not synthetic else "local")
    exp.record(oof_score_mean=art["s_oof"], oof_score_per_fold=per_fold, holdout_score=art["s_hold"],
               runtime_sec=time.time() - t0,
               extra={"baseline_oof": base["s_oof"], "baseline_hold": base["s_hold"], "baseline_recall": base["rec"],
                      "art_oof": art["s_oof"], "art_hold": art["s_hold"], "art_recall": art["rec"],
                      "delta_oof": d_oof, "delta_hold": d_hold, "qso_recall_delta": qso_d,
                      "star_recall_delta": star_d, "gate_pass": gate_pass, "n_artifact_feats": int(art_tr.shape[1])})
    exp.note(note); exp.commit()
    try:
        from src.diary import render_all
        render_all()
    except Exception as e:
        print(f"[warn] render_all skipped: {e}")

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: test[ID], TARGET: weighted_predict(art["test"], art["w"], labels=CLASSES)}).to_csv(
        SUBMISSIONS / "v20_lgb_art.csv", index=False)
    print(f"\n[verdict] {note}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
