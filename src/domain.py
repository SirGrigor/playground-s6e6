"""Domain-knowledge features + confirm-or-kill probes for the SDSS stellar-class problem.

Born from the 2026-06-03 domain-research swarm (docs/strategy.md). The swarm NAMED our 0.9664
ceiling as a MISSING-CHANNEL problem (morphology / mid-IR are absent, not latent in ugriz) and
surfaced a short list of features that are (a) derivable from OUR columns alone and (b) NOT
trivially reconstructable by an axis-aligned GBDT. This module builds them and — crucially —
ships the cheap EDA PROBES that decide whether the synthetic generator preserved the physics
BEFORE we trust any of them (discovery-first, feedback_discovery-first-diagnose-before-build).

Features (all leakage-safe except the stellar locus, which is fit on TRAIN STARs only):
- galactic_coords:  b, |b|, sin b, l  — exact equatorial→Galactic rotation of (alpha, delta).
  Spatial prior ORTHOGONAL to photometry+redshift (stars crowd the disk, |b| low). The one axis
  that could move the pinned 0.9958 STAR↔GALAXY OvO. A GBDT can split raw alpha/delta but the
  class gradient runs DIAGONALLY along |b|; handing it |b| aligns one monotone split with physics.
- sentinel_flags:  is_{u,g,z}_sentinel, n_sentinels  — a u-band non-detection is the Lyman-break
  / u-dropout signature of high-z objects, NOT random missingness. build_features() converts
  −9999→NaN and THROWS THE PATTERN AWAY; this recovers it.
- long_baseline_colors:  u_r (Strateva red-sequence/blue-cloud split ~2.22), g_i (stellar-locus
  temperature ordinate, and the parameter the locus is binned on). Weakly non-reconstructable.
- stellar_locus L:  perpendicular distance to the STAR color manifold (Covey-style). Single stars
  collapse onto a tight 1-D curve in 4-D color space; galaxies/QSOs sit OFF it. "Distance to a
  curve" is exactly what axis-aligned splits cannot represent — the one transform that hands the
  tree the manifold residual directly.

Probes (each PRINTS a verdict; returns (text, metric)):
- latitude_class_probe (T1)     → did class-conditional sky structure survive generation?
- stellar_locus_tightness (T2)  → did the STAR locus stay tight, or get smeared into a blob?
- sentinel_class_probe (T3)     → are sentinels class/redshift-correlated or uniform noise?
- categorical_color_probe (T4)  → are spectral_type/galaxy_population just color re-bins (→ ~0)?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SENTINEL_VALUE

# Equatorial (J2000/ICRS) → Galactic rotation constants, degrees.
_ALPHA_NGP = 192.85948
_DELTA_NGP = 27.12825
_L_NCP = 122.93192

LOCUS_COLORS = ["u_g", "g_r", "r_i", "i_z"]


# --------------------------------------------------------------------------- features
def galactic_coords(alpha, delta) -> pd.DataFrame:
    """Exact equatorial→Galactic transform. NO external data — pure trig on (alpha, delta)."""
    a = np.radians(np.asarray(alpha, dtype=float))
    d = np.radians(np.asarray(delta, dtype=float))
    aG, dG, lNCP = np.radians(_ALPHA_NGP), np.radians(_DELTA_NGP), np.radians(_L_NCP)
    sin_b = np.clip(np.sin(dG) * np.sin(d) + np.cos(dG) * np.cos(d) * np.cos(a - aG), -1.0, 1.0)
    b = np.arcsin(sin_b)
    l = lNCP - np.arctan2(np.cos(d) * np.sin(a - aG),
                          np.cos(dG) * np.sin(d) - np.sin(dG) * np.cos(d) * np.cos(a - aG))
    l = np.mod(l, 2 * np.pi)
    b_deg = np.degrees(b)
    return pd.DataFrame({"gal_b": b_deg, "gal_abs_b": np.abs(b_deg),
                         "gal_sin_b": sin_b, "gal_l": np.degrees(l)})


def sentinel_flags(raw_df: pd.DataFrame, cols=("u", "g", "z")) -> pd.DataFrame:
    """Recover the −9999 dropout PATTERN that build_features() collapses to NaN."""
    out = pd.DataFrame(index=raw_df.index)
    present = [c for c in cols if c in raw_df.columns]
    for c in present:
        out[f"is_{c}_sentinel"] = (raw_df[c].astype(float) == SENTINEL_VALUE).astype("int8")
    out["n_sentinels"] = (out[[f"is_{c}_sentinel" for c in present]].sum(axis=1).astype("int8")
                          if present else np.int8(0))
    return out


def long_baseline_colors(Xbands: pd.DataFrame) -> pd.DataFrame:
    """u−r and g−i from the (already sentinel→NaN) bands in the base feature frame."""
    out = pd.DataFrame(index=Xbands.index)
    if {"u", "r"}.issubset(Xbands.columns):
        out["u_r"] = Xbands["u"].astype(float) - Xbands["r"].astype(float)
    if {"g", "i"}.issubset(Xbands.columns):
        out["g_i"] = Xbands["g"].astype(float) - Xbands["i"].astype(float)
    return out


def fit_stellar_locus(Xbase: pd.DataFrame, is_star: np.ndarray, n_bins: int = 40) -> dict:
    """Fit the STAR color ridge: per g−i bin, robust median + IQR-σ of each color.

    MUST be fit on TRAIN STARs only (the caller restricts to dev-set stars). Returns a ridge
    dict {edges, med[color], sig[color]} applied unchanged to val/holdout/test by locus_distance.
    """
    g_i = (Xbase["g"].astype(float) - Xbase["i"].astype(float)).to_numpy()
    star = np.asarray(is_star, dtype=bool)
    gi_star = g_i[star]
    finite = gi_star[np.isfinite(gi_star)]
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.nanquantile(finite, qs)) if finite.size else np.array([0.0, 1.0])
    if edges.size < 3:
        edges = np.array([np.nanmin(finite) - 1, np.nanmean(finite), np.nanmax(finite) + 1]) \
            if finite.size else np.array([-1.0, 0.0, 1.0])
    inner = edges[1:-1]
    nb = len(inner) + 1
    bin_star = np.clip(np.digitize(gi_star, inner), 0, nb - 1)
    med, sig = {}, {}
    for col in LOCUS_COLORS:
        if col not in Xbase.columns:
            continue
        v = Xbase[col].astype(float).to_numpy()[star]
        m = np.full(nb, np.nan); s = np.full(nb, np.nan)
        for bi in range(nb):
            vals = v[bin_star == bi]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 20:
                m[bi] = np.median(vals)
                q1, q3 = np.percentile(vals, [25, 75])
                s[bi] = max((q3 - q1) / 1.349, 1e-3)   # IQR→σ, floored
        # fill empty bins by forward/back fill so every bin has a reference
        m = pd.Series(m).ffill().bfill().to_numpy()
        s = pd.Series(s).ffill().bfill().fillna(1.0).to_numpy()
        med[col], sig[col] = m, s
    return {"edges_inner": inner, "n_bins": nb, "med": med, "sig": sig}


def locus_distance(Xbase: pd.DataFrame, ridge: dict) -> pd.DataFrame:
    """Apply a fitted ridge: L = sqrt(Σ z_color²) + signed per-color residuals (z-scores)."""
    g_i = (Xbase["g"].astype(float) - Xbase["i"].astype(float)).to_numpy()
    inner, nb = ridge["edges_inner"], ridge["n_bins"]
    bin_idx = np.clip(np.digitize(np.nan_to_num(g_i, nan=np.nanmedian(g_i)), inner), 0, nb - 1)
    out = pd.DataFrame(index=Xbase.index)
    z2 = np.zeros(len(Xbase)); ncol = np.zeros(len(Xbase))
    for col in LOCUS_COLORS:
        if col not in ridge["med"]:
            continue
        v = Xbase[col].astype(float).to_numpy()
        z = (v - ridge["med"][col][bin_idx]) / ridge["sig"][col][bin_idx]
        out[f"locus_res_{col}"] = z
        good = np.isfinite(z)
        z2[good] += z[good] ** 2
        ncol[good] += 1
    out["locus_L"] = np.where(ncol > 0, np.sqrt(z2 * (len(ridge["med"]) / np.maximum(ncol, 1))), np.nan)
    return out


def add_domain_features(Xbase: pd.DataFrame, raw_df: pd.DataFrame, ridge: dict | None = None,
                        *, parts=("galactic", "sentinel", "longcolor", "locus")
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Augment a base feature frame (output of build_features) with the domain features.

    `ridge` (from fit_stellar_locus on dev STARs) is required for the 'locus' part; pass None to
    skip it. Returns (X_augmented, new_feature_names)."""
    pieces = [Xbase]
    if "galactic" in parts:
        pieces.append(galactic_coords(Xbase["alpha"], Xbase["delta"]))
    if "sentinel" in parts:
        pieces.append(sentinel_flags(raw_df))
    if "longcolor" in parts:
        pieces.append(long_baseline_colors(Xbase))
    if "locus" in parts and ridge is not None:
        pieces.append(locus_distance(Xbase, ridge))
    aug = pd.concat(pieces, axis=1)
    new = [c for c in aug.columns if c not in Xbase.columns]
    return aug, new


# --------------------------------------------------------------------------- probes (T1–T4)
def latitude_class_probe(Xbase: pd.DataFrame, y_names: np.ndarray, classes: list[str],
                         n_bins: int = 10) -> tuple[str, float]:
    """T1 — per-class fraction across |b| quantile bins. Monotone STAR trend ⇒ structure survived."""
    from scipy.stats import spearmanr
    gal = galactic_coords(Xbase["alpha"], Xbase["delta"])
    absb = gal["gal_abs_b"].to_numpy()
    y = np.asarray(y_names)
    edges = np.unique(np.nanquantile(absb, np.linspace(0, 1, n_bins + 1)))
    binid = np.clip(np.digitize(absb, edges[1:-1]), 0, len(edges) - 2)
    lines = ["[T1 latitude] per-class fraction vs |b| quantile (low |b| = Galactic disk):",
             f"   {'bin':>4} {'|b|_mid':>8} " + "".join(f"{c:>9}" for c in classes) + f"{'n':>9}"]
    star_frac, mids = [], []
    for bi in range(len(edges) - 1):
        m = binid == bi
        if not m.any():
            continue
        mid = float(np.nanmedian(absb[m])); mids.append(mid)
        fr = {c: float((y[m] == c).mean()) for c in classes}
        star_frac.append(fr.get("STAR", np.nan))
        lines.append(f"   {bi:>4} {mid:>8.1f} " + "".join(f"{fr[c]:>9.3f}" for c in classes)
                     + f"{int(m.sum()):>9,}")
    rho = float(spearmanr(mids, star_frac).correlation) if len(mids) > 2 else float("nan")
    survived = np.isfinite(rho) and abs(rho) > 0.3
    lines.append(f"   → Spearman(|b|_mid, STAR_frac) = {rho:+.3f}  "
                 f"⇒ {'SPATIAL STRUCTURE SURVIVED — build gal_b/|b| (#2/#3)' if survived else 'FLAT — drop spatial features for class (keep only for error-routing)'}")
    return "\n".join(lines), rho


def stellar_locus_tightness(Xbase: pd.DataFrame, y_names: np.ndarray, n_bins: int = 30
                            ) -> tuple[str, float]:
    """T2 — STAR locus width / inter-class color spread. Ratio « 1 ⇒ tight curve survived ⇒ L lives."""
    y = np.asarray(y_names)
    star = y == "STAR"
    g_i = (Xbase["g"].astype(float) - Xbase["i"].astype(float)).to_numpy()
    finite = g_i[star & np.isfinite(g_i)]
    if finite.size < 100:
        return "[T2 locus] too few STARs to judge.", float("nan")
    edges = np.unique(np.nanquantile(finite, np.linspace(0, 1, n_bins + 1)))
    binid = np.clip(np.digitize(g_i, edges[1:-1]), 0, len(edges) - 2)
    widths, spreads = [], []
    for col in LOCUS_COLORS:
        if col not in Xbase.columns:
            continue
        v = Xbase[col].astype(float).to_numpy()
        wb = []
        for bi in range(len(edges) - 1):
            vals = v[star & (binid == bi)]
            vals = vals[np.isfinite(vals)]
            if vals.size >= 20:
                q1, q3 = np.percentile(vals, [25, 75]); wb.append((q3 - q1) / 1.349)
        if wb:
            widths.append(np.median(wb))
            allv = v[np.isfinite(v)]
            spreads.append(np.std(allv))
    width = float(np.mean(widths)) if widths else float("nan")
    spread = float(np.mean(spreads)) if spreads else float("nan")
    ratio = width / spread if spread else float("nan")
    survived = np.isfinite(ratio) and ratio < 0.5
    txt = (f"[T2 locus] STAR locus width (median IQR-σ over g−i bins) = {width:.4f} mag vs "
           f"inter-class color spread {spread:.4f} → ratio {ratio:.3f}\n"
           f"   → {'TIGHT LOCUS SURVIVED — L (#1) has contrast, build it' if survived else 'SMEARED/blob — L is noise, skip #1'}")
    return txt, ratio


def sentinel_class_probe(raw_df: pd.DataFrame, y_names: np.ndarray, classes: list[str]
                         ) -> tuple[str, float]:
    """T3 — class mix + median redshift in sentinel vs non-sentinel rows. Skew ⇒ dropout signal real."""
    flags = sentinel_flags(raw_df)
    n_sent = int((flags["n_sentinels"] > 0).sum()) if "n_sentinels" in flags else 0
    if n_sent == 0:
        return "[T3 sentinel] no −9999 sentinels present (synthetic fallback or clean split) — flags are hygiene-only.", 0.0
    y = np.asarray(y_names); sent = (flags["n_sentinels"] > 0).to_numpy()
    base = {c: float((y == c).mean()) for c in classes}
    smix = {c: float((y[sent] == c).mean()) for c in classes}
    l1 = float(np.sum(np.abs([smix[c] - base[c] for c in classes])))
    lines = [f"[T3 sentinel] {n_sent:,} rows with ≥1 sentinel ({n_sent/len(y)*100:.2f}%):",
             "   class fraction  overall → sentinel-rows:"]
    for c in classes:
        lines.append(f"     {c:<8} {base[c]:.3f} → {smix[c]:.3f}")
    if "redshift" in raw_df.columns:
        rz = raw_df["redshift"].astype(float).to_numpy()
        lines.append(f"   median redshift: non-sentinel {np.nanmedian(rz[~sent]):.3f} | "
                     f"sentinel {np.nanmedian(rz[sent]):.3f}")
    survived = l1 > 0.05
    lines.append(f"   → class-mix shift L1={l1:.3f} ⇒ "
                 f"{'DROPOUT SIGNAL SURVIVED — keep flags + redshift interaction (#4)' if survived else 'uniform — flags are nuisance, keep NaN-hygiene only'}")
    return "\n".join(lines), l1


def categorical_color_probe(Xbase: pd.DataFrame, raw_df: pd.DataFrame, y_names: np.ndarray
                            ) -> tuple[str, float]:
    """T4 — are spectral_type / galaxy_population just color re-bins (→ a GBDT already has them, ~0)?"""
    lines = ["[T4 categoricals] are spectral_type/galaxy_population color-derived (→ expect ~0)?"]
    g_r = (Xbase["g"].astype(float) - Xbase["r"].astype(float)).to_numpy() if {"g", "r"}.issubset(Xbase.columns) else None
    u = Xbase["u"].astype(float).to_numpy() if "u" in Xbase.columns else None
    r = Xbase["r"].astype(float).to_numpy() if "r" in Xbase.columns else None
    mono = float("nan")
    if "spectral_type" in raw_df.columns and g_r is not None:
        grp = pd.Series(g_r).groupby(raw_df["spectral_type"].astype("string").to_numpy()).median()
        grp = grp.dropna().sort_values()
        lines.append("   median g−r by spectral_type (expect monotone O/B<A/F<G/K<M if temperature re-bin):")
        lines.append("     " + ", ".join(f"{k}:{v:+.2f}" for k, v in grp.items()))
        mono = float(grp.iloc[-1] - grp.iloc[0]) if len(grp) > 1 else float("nan")
    if "galaxy_population" in raw_df.columns and u is not None and r is not None:
        u_r = u - r
        grp = pd.Series(u_r).groupby(raw_df["galaxy_population"].astype("string").to_numpy()).median()
        grp = grp.dropna()
        lines.append("   median u−r by galaxy_population (expect a clean split ~2.22 if red-sequence re-bin):")
        lines.append("     " + ", ".join(f"{k}:{v:+.2f}" for k, v in grp.items()))
    lines.append("   → if labels are clean monotone color re-bins, they add ~0 to a GBDT (v3 confirmed +0.0002); "
                 "value is only as a decorrelated input to a NN leg.")
    return "\n".join(lines), mono
