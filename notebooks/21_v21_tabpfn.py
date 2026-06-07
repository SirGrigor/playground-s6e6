"""v21 — TabPFN-v2 leg (the one paradigm missing from the fleet: foundation-model in-context learner).

See docs/LADDER.md. Restacking the existing 19 legs did NOT beat the 0.96979 anchor (holdout 0.96957)
— every leg is an rm/tabm/tree variant with rho>0.8, so decorrelation is exhausted within our paradigm
set. The only remaining widen lever is a genuinely NEW strong+decorrelated paradigm. TabPFN is it.

THE L3 BAR (both required, else this joins the FT/TabICL/kNN "decorrelated-but-weak -> 0 weight" graveyard):
  - single-leg balanced-acc >= ~0.968 (competitive)
  - rho < ~0.80 vs the rm/tabm legs (decorrelated)

TabPFN's native regime is <=10K rows; our dev is ~462K. Strategy: per CV fold, BAG several TabPFN fits
each on a class-stratified 10K subsample of the fold-train, average probs. Produces an aligned OOF leg
(same HOLDOUT_SEED + CV_SEED as the rest of the fleet) -> stack locally with fleet_stack afterwards.
Compact NUMERIC feature view (bands + redshift + coords + colors) — TabPFN's strength and an extra
decorrelation source vs the rich-FE legs.

Outputs (harvest dir): oof_tabpfn.npy / hold_tabpfn.npy / test_tabpfn.npy + y_dev/y_hold/test_ids.

Cloud: GPU REQUIRED. The kernel installs tabpfn + the P100 torch fix.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src import data
from src.config import CLASSES, ID, MODEL_SEED, TARGET
from src.cv import stratified_folds
from src.metrics import competition_score, tune_class_weights, weighted_predict

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
NC = len(CLASSES)
N_FOLDS = 5
SEED = MODEL_SEED
SUBSAMPLE = 10000        # TabPFN context cap
BAGS = 4                 # subsample bags per fold (variance reduction)
PRED_BATCH = 20000       # batched predict to bound GPU memory
BANDS = ["u", "g", "r", "i", "z"]
SENTINEL = -9999.0


def _load():
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("WARN data/raw empty -> synthetic fallback (smoke).")
    full = data.synthetic_fallback(n=4000)
    return full.iloc[:3000].reset_index(drop=True), full.iloc[3000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _features(df: pd.DataFrame) -> np.ndarray:
    """Compact numeric view for TabPFN: bands + redshift + coords + adjacent/long colors."""
    out = pd.DataFrame(index=df.index)
    base = {}
    for c in BANDS + ["redshift", "alpha", "delta"]:
        if c in df.columns:
            base[c] = df[c].astype("float32").replace(SENTINEL, np.nan)
            out[c] = base[c]
    for a, b in [("u", "g"), ("g", "r"), ("r", "i"), ("i", "z"), ("u", "r"), ("g", "i")]:
        if a in base and b in base:
            out[f"{a}_{b}"] = (base[a] - base[b]).astype("float32")
    return out.to_numpy(dtype=np.float32)


def _strat_subsample(y, n, rng):
    """Class-stratified subsample of indices (proportional, capped at n)."""
    idx_by_c = [np.where(y == c)[0] for c in range(NC)]
    take = [max(1, int(round(n * len(ix) / len(y)))) for ix in idx_by_c]
    pick = [rng.choice(ix, size=min(t, len(ix)), replace=False) for ix, t in zip(idx_by_c, take)]
    out = np.concatenate(pick); rng.shuffle(out); return out


def _predict_batched(clf, X):
    return np.concatenate([clf.predict_proba(X[i:i + PRED_BATCH]) for i in range(0, len(X), PRED_BATCH)], axis=0)


def main() -> None:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] device={dev} torch={torch.__version__}")

    train, test, synthetic = _load()
    print(f"[load] train={train.shape} test={test.shape}")
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    Xall = _features(train); Xte = _features(test)
    print(f"[feat] {Xall.shape[1]} numeric features")

    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = Xall[dev_idx], y_all[dev_idx]
    Xhold, yhold = Xall[hold_idx], y_all[hold_idx]
    ydev_names, yhold_names = INT2CLS[ydev], INT2CLS[yhold]

    oof = np.zeros((len(Xdev), NC), "float32")
    hold = np.zeros((len(Xhold), NC), "float32")
    tst = np.zeros((len(Xte), NC), "float32")
    sub = SUBSAMPLE if not synthetic else 500
    bags = BAGS if not synthetic else 2

    for i, (tr, va) in enumerate(stratified_folds(ydev, N_FOLDS), 1):
        ft = time.time()
        vp = np.zeros((len(va), NC), "float32"); hp = np.zeros((len(Xhold), NC), "float32"); tp = np.zeros((len(Xte), NC), "float32")
        for b in range(bags):
            rng = np.random.default_rng(SEED + 100 * i + b)
            si = tr[_strat_subsample(ydev[tr], sub, rng)]
            clf = TabPFNClassifier(device=dev, ignore_pretraining_limits=True, random_state=SEED + b)
            clf.fit(Xdev[si], ydev[si])
            vp += _predict_batched(clf, Xdev[va]) / bags
            hp += _predict_batched(clf, Xhold) / bags
            tp += _predict_batched(clf, Xte) / bags
        oof[va] = vp; hold += hp / N_FOLDS; tst += tp / N_FOLDS
        fs = competition_score(ydev_names[va], INT2CLS[np.argmax(vp, 1)])
        print(f"  fold {i}/{N_FOLDS}  bal-acc {fs:.5f}  ({time.time()-ft:.0f}s, {bags} bags x {sub})")

    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    s_hold = competition_score(yhold_names, weighted_predict(hold, w, labels=CLASSES))
    from sklearn.metrics import recall_score
    rec = {c: float(v) for c, v in zip(CLASSES, recall_score(ydev_names, weighted_predict(oof, w, labels=CLASSES), labels=CLASSES, average=None, zero_division=0))}
    print(f"\n[tabpfn] single-leg OOF {s_oof:.5f} | holdout {s_hold:.5f} | recall {rec}")
    print(f"[L3 bar] competitive>=~0.968: {'PASS' if s_oof >= 0.968 else 'FAIL'} (rho vs fleet computed locally on harvest)")

    out = "/kaggle/working" if not synthetic else "/tmp"
    np.save(f"{out}/oof_tabpfn.npy", oof); np.save(f"{out}/hold_tabpfn.npy", hold); np.save(f"{out}/test_tabpfn.npy", tst)
    np.save(f"{out}/y_dev.npy", ydev); np.save(f"{out}/y_hold.npy", yhold)
    pd.DataFrame({ID: test[ID]}).to_csv(f"{out}/test_ids.csv", index=False)
    print(f"[saved] oof/hold/test_tabpfn.npy + labels -> {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
