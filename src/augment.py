"""Original-SDSS17 concat-augmentation helpers (ladder v18+, see docs/LADDER.md).

The lever our locked ceiling missed: our v2 killed the *leak-match* (0% exact join), but the top
public single model CONCATS the fedesoriano SDSS17 file as ~100K down-weighted extra training rows.
These helpers normalise that frame to the competition base schema (so it flows through the SAME
feature pipeline) and compute the prior-shift-correcting PER-CLASS weights.

Gemini guardrail: a naive global down-weight does NOT correct the prior shift (original class balance
!= competition's) — under balanced accuracy that poisons per-class recall. `per_class_weights` makes
the added per-class weight-mass proportional to the competition class proportions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CLASSES, TARGET
from .fe_realmlp import BASE_NUM, NATIVE_CATS

BASE_W = 0.35   # community down-weight; the per-class factor multiplies this


def prep_original(orig: pd.DataFrame | None) -> pd.DataFrame | None:
    """Normalise the fedesoriano SDSS17 frame to the competition base schema. Keeps BASE_NUM + TARGET;
    adds the playground-only NATIVE_CATS as NA (→ factorize -1 = 'unknown', a legitimate missing-cat
    signal on original rows). Returns None if `orig` is None or has no usable rows."""
    if orig is None:
        return None
    df = orig.copy()
    if TARGET not in df.columns:
        for cand in ("class", "Class", "label"):
            if cand in df.columns:
                df = df.rename(columns={cand: TARGET})
                break
    if TARGET not in df.columns:
        return None
    df[TARGET] = df[TARGET].astype(str).str.upper().str.strip()
    df = df[df[TARGET].isin(CLASSES)].copy()
    if len(df) == 0:
        return None
    keep = [c for c in BASE_NUM if c in df.columns] + [TARGET]
    df = df[keep].copy()
    for c in NATIVE_CATS:
        df[c] = pd.array([None] * len(df), dtype="string")
    return df.reset_index(drop=True)


def per_class_weights(y_comp_int: np.ndarray, y_orig_int: np.ndarray, base_w: float = BASE_W) -> np.ndarray:
    """w_c = (N_c_comp / N_c_orig) * base_w for each original row's class — neutralises prior shift.

    Per class c the original weight-mass = N_c_orig * w_c = base_w * N_c_comp, i.e. the augmentation
    adds base_w x the competition mass PER CLASS, so the effective class distribution stays comp's."""
    n_comp = np.bincount(np.asarray(y_comp_int), minlength=len(CLASSES)).astype(float)
    n_orig = np.bincount(np.asarray(y_orig_int), minlength=len(CLASSES)).astype(float)
    factor = np.where(n_orig > 0, (n_comp / np.maximum(n_orig, 1.0)) * base_w, 0.0)
    return factor[np.asarray(y_orig_int)]


def unify_categories(frames: list) -> list:
    """Give every pandas-`category` column an IDENTICAL category set (the union across `frames`),
    in place. Required by LightGBM for concat-augmentation: it compares the categorical_feature
    definition of the train Dataset against the valid Dataset, so a train frame built by
    pd.concat([fold_train, original]) (whose categories union) must still match its validation slice.
    The RealMLP path does not need this (`_encode_frames` already shares a vocabulary)."""
    if not frames:
        return frames
    cat_cols = [c for c in frames[0].columns if str(frames[0][c].dtype) == "category"]
    for c in cat_cols:
        cats = pd.Index(sorted(set().union(*[set(pd.Series(f[c]).dropna().unique()) for f in frames])))
        for f in frames:
            f[c] = pd.Categorical(f[c], categories=cats)
    return frames
