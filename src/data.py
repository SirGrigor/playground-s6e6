"""Data loading + the holdout split. Single place that touches data/raw.

`load_raw()` reads the competition CSVs. `load_original()` reads the fedesoriano
SDSS17 dataset (leak-match + augmentation source). `synthetic_fallback()` makes a
tiny SDSS-shaped frame so the viz + dashboard pipeline can be smoke-tested with no
data download — used by notebooks/01_eda.py when data/raw is empty.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CLASSES, HOLDOUT_FRAC, HOLDOUT_SEED, ID, PHOTOMETRIC_BANDS,
                     RAW, EXTERNAL, TARGET)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(train, test) from data/raw, or None if not downloaded yet."""
    tr, te = RAW / "train.csv", RAW / "test.csv"
    if not (tr.exists() and te.exists()):
        return None
    return pd.read_csv(tr), pd.read_csv(te)


def load_original() -> pd.DataFrame | None:
    """fedesoriano SDSS17 original from data/external, or None."""
    csvs = list(EXTERNAL.glob("*.csv"))
    return pd.read_csv(csvs[0]) if csvs else None


def holdout_split(df: pd.DataFrame):
    """Stratified holdout (frac=HOLDOUT_FRAC) — returns (dev_idx, holdout_idx).

    Stratify on TARGET so the rare QSO/STAR classes are represented in the holdout.
    """
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    strat = df[TARGET] if TARGET in df.columns else None
    dev, hold = train_test_split(idx, test_size=HOLDOUT_FRAC, random_state=HOLDOUT_SEED, stratify=strat)
    return np.sort(dev), np.sort(hold)


def synthetic_fallback(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    """Tiny SDSS-shaped frame for smoke-testing the pipeline without a download.

    Encodes the real separation structure: STAR ~ redshift 0, GALAXY low-moderate,
    QSO high — so the confusion matrix / ROC actually look meaningful in a smoke run.
    NOT for modeling; purely to exercise viz + dashboard end-to-end.
    """
    rng = np.random.default_rng(seed)
    probs = [0.59, 0.22, 0.19]  # GALAXY / STAR / QSO
    y = rng.choice(CLASSES, size=n, p=probs)
    redshift = np.where(y == "STAR", rng.normal(0.0, 0.0005, n),
                np.where(y == "GALAXY", rng.normal(0.15, 0.08, n),
                                        rng.normal(1.5, 0.6, n)))
    redshift = np.clip(redshift, -0.01, 7.0)
    df = {ID: np.arange(n), "redshift": redshift, TARGET: y}
    base = {"u": 22, "g": 21, "r": 20, "i": 20, "z": 20}
    for b in PHOTOMETRIC_BANDS:
        df[b] = rng.normal(base[b], 1.5, n) + (y == "QSO") * rng.normal(0.5, 0.3, n)
    for c in ("alpha", "delta"):
        df[c] = rng.uniform(0, 360, n) if c == "alpha" else rng.uniform(-20, 80, n)
    return pd.DataFrame(df)
