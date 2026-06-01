"""Feature engineering. v1 = minimal & defensible (clean attribution for the diary).

Keeps: redshift (the dominant signal), the 5 photometric bands, adjacent-band COLORS
(u-g, g-r, r-i, i-z — classic astronomical separators), sky coords (alpha, delta).
Drops: row-unique IDs (obj_ID, spec_obj_ID) that only invite overfit, and the constant
rerun_ID. Survey IDs (run/cam/field/plate/MJD/fiber) are LEFT OUT of v1 on purpose — they
can carry leak-y structure but muddy attribution; add them as a deliberate later experiment.
Sentinel −9999 in u/g/z → NaN (LGBM handles NaN natively).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (COORD_COLS, ID, PHOTOMETRIC_BANDS, SENTINEL_COLS, SENTINEL_VALUE,
                     TARGET, color_features)

# Columns dropped from v1 (kept here so the choice is explicit + greppable).
DROP_ALWAYS = ["obj_ID", "spec_obj_ID", "rerun_ID"]
SURVEY_IDS_HELD_OUT_V1 = ["run_ID", "cam_col", "field_ID", "plate", "MJD", "fiber_ID"]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names). Does NOT touch the target or id columns."""
    out = pd.DataFrame(index=df.index)

    # sentinel → NaN on the affected magnitude bands
    bands = {}
    for b in PHOTOMETRIC_BANDS:
        if b in df.columns:
            col = df[b].astype(float).replace(SENTINEL_VALUE, np.nan)
            bands[b] = col
            out[b] = col

    # colors = adjacent-band differences (NaN-propagating, which is correct)
    for a, c in color_features():
        if a in bands and c in bands:
            out[f"{a}_{c}"] = bands[a] - bands[c]

    if "redshift" in df.columns:
        out["redshift"] = df["redshift"].astype(float)

    for c in COORD_COLS:
        if c in df.columns:
            out[c] = df[c].astype(float)

    return out, list(out.columns)
