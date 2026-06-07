"""Generator-artifact features (ladder v20, see docs/LADDER.md).

S6E6 train+test are synthetically generated from SDSS17 by the same deep model. Such generators leave
QUANTIZATION / ROUNDING fingerprints in the low-order decimal digits of continuous columns — structure
that correlates with the generative process (and, via it, sometimes the class) but is invisible to
physics-based FE. Because the SAME generator produced train and the private test, these fingerprints
generalize (Gemini: risk ~ 0). Our "72-approach signal matrix" was scoped to physical re-expressions
and missed this meta axis entirely.

This is ORTHOGONAL to the (refuted) original-data augmentation: it adds no external rows, only new
columns derived from the existing values — so it is immune to the real-vs-synthetic distribution gap
that sank v19.

`add_artifact_features(df)` returns a frame of digit/mod/decimal features (categorical where low-card)
to concat onto the rich-FE matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ARTIFACT_COLS = ["u", "g", "r", "i", "z", "redshift"]
SENTINEL = -9999.0


def add_artifact_features(df: pd.DataFrame, cols: list | None = None) -> pd.DataFrame:
    """Per continuous column: fractional part (numeric) + the k-th decimal digits (k=2,4,6) and the
    mod-10 / mod-100 of the 6-decimal-scaled integer (categorical). Sentinel/NaN -> code -1.

    Digits are extracted from |v| so the (rare) negative redshifts don't flip them; the integer part
    is already captured by build_rich_features' floor-factorize, so it is not repeated here."""
    cols = cols or ARTIFACT_COLS
    out = pd.DataFrame(index=df.index)

    def _cat(arr, mask):
        return pd.Series(np.where(mask, -1, arr).astype("int64"), index=df.index).astype("category")

    for c in cols:
        if c not in df.columns:
            continue
        v = df[c].astype("float64").replace(SENTINEL, np.nan)
        nan = v.isna().to_numpy()
        av = np.abs(v.to_numpy())
        out[f"{c}_frac"] = (v - np.floor(v)).astype("float32")          # numeric mantissa
        for k in (2, 4, 6):
            digit = np.floor(np.where(nan, 0.0, av) * (10 ** k)) % 10
            out[f"{c}_d{k}"] = _cat(digit, nan)                          # k-th decimal digit (0-9)
        i6 = np.round(np.where(nan, 0.0, av) * 1e6)
        out[f"{c}_m10"] = _cat(np.mod(i6, 10), nan)
        out[f"{c}_m100"] = _cat(np.mod(i6, 100), nan)
    return out
