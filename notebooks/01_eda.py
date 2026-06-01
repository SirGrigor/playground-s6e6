"""Phase 0 — EDA / recon. Runs as the ACTIVE script via colab/bootstrap.py.

What it does (no model yet):
  1. Load competition data (or a synthetic fallback so the pipeline renders anywhere).
  2. Column-parity check vs the SDSS17 original schema (config.py) — flags drift.
  3. Class balance + redshift separation by class (the dominant signal).
  4. Sentinel (-9999) audit on u/g/z.
  5. A redshift-only baseline (3-bin rule) → renders the standard viz panels so the
     dashboard has a confusion matrix / ROC / F1 from row one, and sets the FLOOR.

Run locally:  PYTHONPATH=. python notebooks/01_eda.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import data
from src.config import (CLASSES, ID, ID_LIKE_COLS, PHOTOMETRIC_BANDS, SENTINEL_COLS,
                        SENTINEL_VALUE, TARGET, color_features)
from src.dashboard import render_html
from src.viz import render_all_panels


def column_parity(train: pd.DataFrame) -> None:
    expected = set([ID, TARGET, "redshift", "alpha", "delta", *PHOTOMETRIC_BANDS, *ID_LIKE_COLS])
    got = set(train.columns)
    missing, extra = expected - got, got - expected
    print(f"[parity] columns: {len(got)} present")
    if missing:
        print(f"  ⚠ expected-but-missing (schema drift — update config.py): {sorted(missing)}")
    if extra:
        print(f"  ⚠ present-but-unexpected: {sorted(extra)}")
    if not (missing or extra):
        print("  ✓ schema matches SDSS17 original exactly")


def class_balance(train: pd.DataFrame) -> None:
    if TARGET not in train.columns:
        print("[balance] no target column — skipping"); return
    vc = train[TARGET].value_counts()
    print("[balance] class distribution:")
    for c in CLASSES:
        n = int(vc.get(c, 0))
        print(f"  {c:<8} {n:>8}  ({100 * n / len(train):.2f}%)")


def redshift_separation(train: pd.DataFrame) -> None:
    if TARGET not in train.columns or "redshift" not in train.columns:
        return
    print("[redshift] median by class (the dominant separator):")
    for c in CLASSES:
        s = train.loc[train[TARGET] == c, "redshift"]
        if len(s):
            print(f"  {c:<8} median={s.median():+.5f}  p10={s.quantile(.1):+.4f}  p90={s.quantile(.9):+.4f}")


def sentinel_audit(train: pd.DataFrame) -> None:
    print(f"[sentinel] {SENTINEL_VALUE} count in {SENTINEL_COLS}:")
    for c in SENTINEL_COLS:
        if c in train.columns:
            n = int((train[c] == SENTINEL_VALUE).sum())
            print(f"  {c}: {n}")


def redshift_baseline(train: pd.DataFrame):
    """Cheap 3-bin redshift rule → first confusion matrix + FLOOR estimate.

    Thresholds are derived from the class medians, not hand-tuned — this is a
    sanity baseline to populate the dashboard, not a submission.
    """
    if TARGET not in train.columns or "redshift" not in train.columns:
        return None
    z = train["redshift"].to_numpy()
    # STAR ~ 0, GALAXY low-moderate, QSO high. Pick split points between class medians.
    pred = np.where(z < 0.002, "STAR", np.where(z < 0.5, "GALAXY", "QSO"))
    # crude one-hot "proba" so ROC/PR panels render (real models give calibrated proba)
    proba = np.zeros((len(z), len(CLASSES)))
    for k, c in enumerate(CLASSES):
        proba[:, k] = (pred == c).astype(float)
    # nudge with redshift so ROC isn't degenerate
    qso_k = CLASSES.index("QSO")
    proba[:, qso_k] = np.clip(z / (z.max() + 1e-9), 0, 1)
    proba = proba / proba.sum(axis=1, keepdims=True).clip(1e-9)
    from src.metrics import competition_score
    y = train[TARGET].to_numpy()
    bal = competition_score(y, pred)
    acc = float((pred == y).mean())
    print(f"[baseline] redshift 3-bin rule: BAL-ACC={bal:.4f} (metric) acc={acc:.4f}  → set BAL-ACC as dashboard FLOOR")
    return y, pred, proba, bal


def main() -> None:
    raw = data.load_raw()
    if raw is None:
        print("⚠ data/raw empty — using synthetic SDSS-shaped fallback (smoke test only).")
        train = data.synthetic_fallback()
    else:
        train, test = raw
        print(f"[load] train={train.shape}  test={test.shape}")
        column_parity(train)

    class_balance(train)
    redshift_separation(train)
    sentinel_audit(train)

    base = redshift_baseline(train)
    if base is not None:
        y_true, y_pred, proba, acc = base
        panels = render_all_panels(y_true, y_pred, proba, prefix="baseline_redshift")
        print(f"[viz] wrote {len(panels)} panels: {sorted(panels)}")
        # AUC diagnostics (OvR + OvO + scalar) — separability, NOT the comp metric.
        from src.metrics import format_auc_report, multiclass_auc_report
        print("[auc] " + format_auc_report(multiclass_auc_report(y_true, proba)).replace("\n", "\n      "))

    out = render_html(prefix="baseline_redshift", title="S6E6 — Phase 0 recon")
    print(f"[dashboard] {out}")

    print("\ncolor features to build next:", [f"{a}-{b}" for a, b in color_features()])


if __name__ == "__main__":
    main()
