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

from .config import (COORD_COLS, EXTRA_COLS, ID, PHOTOMETRIC_BANDS, SENTINEL_COLS,
                     SENTINEL_VALUE, TARGET, color_features)

# Columns dropped from v1 (kept here so the choice is explicit + greppable).
DROP_ALWAYS = ["obj_ID", "spec_obj_ID", "rerun_ID"]
SURVEY_IDS_HELD_OUT_V1 = ["run_ID", "cam_col", "field_ID", "plate", "MJD", "fiber_ID"]


def build_features(df: pd.DataFrame, *, use_extra: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Return (X, feature_names). Does NOT touch the target or id columns.

    `use_extra` toggles the Playground-added spectral_type / galaxy_population columns
    (v3+). Object / low-cardinality extras become pandas `category` so LGBM uses them
    natively; numeric extras pass through. NaN is preserved (LGBM-native, and a missing
    spectral_type is itself informative — e.g. only stars have one)."""
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

    if use_extra:
        for c in EXTRA_COLS:
            if c not in df.columns:
                continue
            col = df[c]
            # treat as categorical when it's object/string or has few distinct values
            if col.dtype == object or str(col.dtype) == "string" or col.nunique(dropna=True) <= 50:
                out[c] = col.astype("category")
            else:
                out[c] = col.astype(float)

    return out, list(out.columns)
