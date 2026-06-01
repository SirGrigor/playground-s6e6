"""Multiclass original-dataset leak matching for S6E6.

synth-decoder's build_match is BINARY (averages labels, ROC-AUC). Our target is 3-class,
so we need a multiclass-aware matcher: join on a high-precision fingerprint key, recover the
true `class` label (mode on collisions), and GATE it — does the recovered label actually
beat the model on matched rows (balanced accuracy), or is it a coverage/illusion trap?

The discipline is synth-decoder's: coverage is not value. A leak is worth using ONLY if it
beats the model on the rows it touches. (S6E5 lesson: a 32%-coverage "96% accurate" leak was
a majority-class illusion worth 0 as a ranker.)

TODO upstream: generalize synth-decoder.matching.build_match to multiclass and replace this.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_key(df: pd.DataFrame, cols: list[str], decimals: int = 5) -> pd.Series:
    """High-precision fingerprint: round continuous cols and join into one string key.

    Rounding absorbs float round-trip noise from the generator while staying near-unique
    on continuous photometry/redshift. Lower `decimals` = looser match (more coverage,
    more collisions). NaN-safe.
    """
    parts = [df[c].round(decimals).astype("string").fillna("NA") for c in cols]
    return parts[0].str.cat(parts[1:], sep="|")


@dataclass
class LeakMatch:
    key_cols: list[str]
    decimals: int
    n_orig_keys: int
    collision_rate: float          # fraction of orig keys mapping to >1 row
    ambiguous_rate: float          # collisions with MIXED class labels (unreliable)
    coverage: dict[str, float]     # split → fraction matched

    def __str__(self) -> str:
        cov = "  ".join(f"{k}={v*100:.2f}%" for k, v in self.coverage.items())
        return (f"key={self.key_cols} @ {self.decimals}dp\n"
                f"  orig keys {self.n_orig_keys:,} | collisions {self.collision_rate*100:.2f}% "
                f"(label-ambiguous {self.ambiguous_rate*100:.2f}%)\n"
                f"  coverage: {cov}")


def match_multiclass(orig: pd.DataFrame, splits: dict[str, pd.DataFrame], key_cols: list[str],
                     target_col: str, decimals: int = 5) -> tuple[dict[str, pd.DataFrame], LeakMatch]:
    """Recover `target_col` from `orig` into each split via the fingerprint key.

    Adds `leak_label` (recovered class, NaN if unmatched) + `leak_matched` (0/1) to each
    split copy. Collisions resolve to the MODE class; mixed-label collisions are counted.
    """
    o = orig.copy()
    o["_key"] = make_key(o, key_cols, decimals)
    grp = o.groupby("_key")[target_col]
    n_unique = grp.ngroups
    sizes = grp.size()
    mode = grp.agg(lambda s: s.mode().iloc[0])           # majority class per key
    nuniq = grp.nunique()
    collide = sizes > 1
    collision_rate = float(collide.mean()) if n_unique else 0.0
    ambiguous_rate = float((nuniq[collide] > 1).mean()) if collide.any() else 0.0
    lookup = mode.rename("leak_label").reset_index()

    out, coverage = {}, {}
    for name, df in splits.items():
        d = df.copy()
        d["_key"] = make_key(d, key_cols, decimals)
        d = d.merge(lookup, on="_key", how="left").drop(columns="_key")
        d["leak_matched"] = d["leak_label"].notna().astype("int8")
        out[name] = d
        coverage[name] = float(d["leak_matched"].mean())
    return out, LeakMatch(key_cols, decimals, n_unique, collision_rate, ambiguous_rate, coverage)


@dataclass
class LeakGate:
    coverage: float
    leak_bal_acc: float            # balanced acc of recovered label on matched rows
    model_bal_acc: float           # balanced acc of the model on those same rows
    verdict: str                   # USE / SKIP

    def __str__(self) -> str:
        return (f"GATE on matched rows (cov {self.coverage*100:.2f}%): "
                f"leak {self.leak_bal_acc:.4f} vs model {self.model_bal_acc:.4f} → {self.verdict}")


def gate_leak(y_true, leak_label, model_pred, matched) -> LeakGate:
    """Does the recovered leak label beat the model on the rows it covers?

    All args are label arrays over a LABELED split; `matched` is the 0/1 mask. USE only if
    the leak's balanced accuracy on matched rows strictly beats the model's there.
    """
    from .metrics import competition_score
    matched = np.asarray(matched).astype(bool)
    y_true = np.asarray(y_true)
    cov = float(matched.mean())
    if matched.sum() == 0 or len(np.unique(y_true[matched])) < 2:
        return LeakGate(cov, float("nan"), float("nan"), "SKIP (no/degenerate matches)")
    leak_acc = competition_score(y_true[matched], np.asarray(leak_label)[matched])
    model_acc = competition_score(y_true[matched], np.asarray(model_pred)[matched])
    verdict = "USE" if leak_acc > model_acc else "SKIP (illusion — model already wins)"
    return LeakGate(cov, leak_acc, model_acc, verdict)
