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


def per_class_weights(y_comp_int: np.ndarray, y_orig_int: np.ndarray, base_w: float = BASE_W,
                      mode: str = "prior") -> np.ndarray:
    """Per-original-row sample weights. base_w is the AVERAGE original row weight (matches the proven
    public recipe magnitude — original contributes ~base_w x its row-count of effective mass, i.e.
    ~6-7% of competition mass at base_w=0.35), NOT a per-class mass fraction.

    mode="flat": every original row = base_w (the public recipe, no prior correction).
    mode="prior": relative weights ∝ N_c_comp / N_c_orig (so the WEIGHTED original class distribution
      matches the competition's — the balanced-accuracy prior-shift correction), scaled so the mean
      original row weight is still base_w. Derivation: w_c = (N_c_comp/N_c_orig) * base_w * N_orig/N_comp
      → sum_c N_c_orig*w_c = base_w*N_orig → mean row weight = base_w.

    v18 (2026-06-07) lesson: the earlier per-class-MASS form over-weighted OOD original data ~5x and
    the LGBM axis test washed out; this magnitude fix keeps the prior correction at the proven scale.
    """
    y_orig = np.asarray(y_orig_int)
    if mode == "flat":
        return np.full(len(y_orig), float(base_w))
    n_comp = np.bincount(np.asarray(y_comp_int), minlength=len(CLASSES)).astype(float)
    n_orig = np.bincount(y_orig, minlength=len(CLASSES)).astype(float)
    k = base_w * len(y_orig) / max(len(np.asarray(y_comp_int)), 1)        # = base_w * N_orig/N_comp
    factor = np.where(n_orig > 0, (n_comp / np.maximum(n_orig, 1.0)) * k, 0.0)
    return factor[y_orig]


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
