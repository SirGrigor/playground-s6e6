"""Experiment observer — hypothesis-before-result discipline enforced.

Metric-agnostic generalization of the S6E5 observer (which was AUC-named).
Here `score` is whatever `config.METRIC` is (accuracy / macro-F1 / logloss);
direction handled via `config.GREATER_IS_BETTER`.

Usage::

    from src.observer import Experiment

    exp = Experiment.start(
        version="v3",
        parent="v2",
        hypothesis="redshift x (g-r) interaction lifts holdout accuracy +0.001-0.002",
        predicted_delta=0.0015,
        confidence="medium",
        feature_changes=["+ redshift_x_gr"],
        config_changes={},
        pipeline_changes=[],
        cloud_or_local="local",
    )

    # ... train + eval ...

    exp.record(
        oof_score_mean=0.9741,
        oof_score_per_fold=[0.9738, 0.9745, 0.9739, 0.9742, 0.9741],
        holdout_score=0.9740,
        runtime_sec=183,
    )
    exp.commit()

`Experiment.start()` enforces non-empty hypothesis + predicted_delta.
`exp.commit()` runs the 7 auto-flag detectors before appending to
`experiments.jsonl`.

`experiments.jsonl` is the source of truth; `docs/diary.md` is rendered
from it by `src.diary` (read-only).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from statistics import stdev
from typing import Any

from .config import GREATER_IS_BETTER, METRIC, ROOT

JSONL_PATH = ROOT / "experiments.jsonl"

# Detector thresholds — tuned for near-ceiling accuracy (deltas in 1e-3..1e-2).
FOLD_COLLAPSE = 0.01
LEAK_GAP = 0.005
REGRESSION_DROP = 0.001
FOLD_INSTABILITY = 0.005


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _load_jsonl() -> list[dict]:
    if not JSONL_PATH.exists():
        return []
    out = []
    for line in JSONL_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _find_entry(version: str) -> dict | None:
    for entry in _load_jsonl():
        if entry.get("version") == version:
            return entry
    return None


def _improved(delta: float) -> bool:
    """Did `delta` move the score in the good direction?"""
    return delta > 0 if GREATER_IS_BETTER else delta < 0


@dataclass
class Experiment:
    # required pre-run
    version: str
    parent: str | None
    hypothesis: str
    predicted_delta: float
    confidence: str
    feature_changes: list[str]
    config_changes: dict[str, Any]
    pipeline_changes: list[str]
    cloud_or_local: str

    # auto-captured
    metric: str = METRIC
    created_at: str = field(default_factory=_now_iso)
    git_sha: str | None = field(default_factory=_git_sha)

    # post-run (record())
    completed_at: str | None = None
    oof_score_mean: float | None = None
    oof_score_per_fold: list[float] | None = None
    holdout_score: float | None = None
    runtime_sec: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    # post-commit (auto-fill)
    flags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    actual_delta: float | None = None
    parent_holdout_score: float | None = None

    @classmethod
    def start(
        cls,
        *,
        version: str,
        parent: str | None,
        hypothesis: str,
        predicted_delta: float,
        confidence: str = "medium",
        feature_changes: list[str] | None = None,
        config_changes: dict[str, Any] | None = None,
        pipeline_changes: list[str] | None = None,
        cloud_or_local: str = "local",
    ) -> "Experiment":
        if not hypothesis or not hypothesis.strip():
            raise ValueError("Experiment.start() requires a non-empty hypothesis.")
        if predicted_delta is None:
            raise ValueError("Experiment.start() requires predicted_delta (use 0.0 if truly none).")
        _COERCE = {"medium-high": "medium", "high-medium": "medium",
                   "low-medium": "low", "medium-low": "low",
                   "very high": "high", "very low": "low"}
        if confidence in _COERCE:
            print(f"  [observer] coercing confidence {confidence!r} → {_COERCE[confidence]!r}")
            confidence = _COERCE[confidence]
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(f"confidence must be low/medium/high (got {confidence!r}); "
                             f"recognized coercions: {sorted(_COERCE)}")
        if _find_entry(version) is not None:
            raise ValueError(
                f"Experiment {version!r} already exists in {JSONL_PATH.name}. "
                "Choose a new version name."
            )
        return cls(
            version=version,
            parent=parent,
            hypothesis=hypothesis.strip(),
            predicted_delta=float(predicted_delta),
            confidence=confidence,
            feature_changes=feature_changes or [],
            config_changes=config_changes or {},
            pipeline_changes=pipeline_changes or [],
            cloud_or_local=cloud_or_local,
        )

    def record(
        self,
        *,
        oof_score_mean: float,
        oof_score_per_fold: list[float],
        holdout_score: float,
        runtime_sec: float,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.oof_score_mean = float(oof_score_mean)
        self.oof_score_per_fold = [float(x) for x in oof_score_per_fold]
        self.holdout_score = float(holdout_score)
        self.runtime_sec = float(runtime_sec)
        self.completed_at = _now_iso()
        if extra:
            self.extra.update(extra)

    def note(self, text: str) -> None:
        if not text.strip():
            return
        self.notes.append(f"[{_now_iso()}] {text.strip()}")

    def _autoflag(self) -> None:
        f = self.flags

        if self.oof_score_per_fold is None or self.holdout_score is None or self.oof_score_mean is None:
            return

        # 1. Fold collapse — a fold far worse than the mean (in the bad direction).
        if self.oof_score_per_fold:
            worst = min(self.oof_score_per_fold) if GREATER_IS_BETTER else max(self.oof_score_per_fold)
            if abs(worst - self.oof_score_mean) > FOLD_COLLAPSE and not _improved(worst - self.oof_score_mean):
                f.append(f"fold_collapse(worst={worst:.5f}, mean={self.oof_score_mean:.5f})")

        # 2. Methodology leak — OOF and holdout disagree too much.
        gap = abs(self.oof_score_mean - self.holdout_score)
        if gap > LEAK_GAP:
            f.append(f"methodology_leak(|oof-holdout|={gap:.5f})")

        # 3. Silent regression vs parent holdout.
        if self.parent:
            parent_entry = _find_entry(self.parent)
            if parent_entry and parent_entry.get("holdout_score") is not None:
                self.parent_holdout_score = float(parent_entry["holdout_score"])
                self.actual_delta = self.holdout_score - self.parent_holdout_score
                if (not _improved(self.actual_delta)) and abs(self.actual_delta) > REGRESSION_DROP:
                    f.append(f"silent_regression(Δhold={self.actual_delta:+.5f} vs {self.parent})")

        # 4. Fold instability.
        if len(self.oof_score_per_fold) >= 2:
            fold_std = stdev(self.oof_score_per_fold)
            if fold_std > FOLD_INSTABILITY:
                f.append(f"fold_instability(std={fold_std:.5f})")

        # 5/6. Prediction calibration vs the stated predicted_delta.
        if self.actual_delta is not None and self.predicted_delta:
            pred = self.predicted_delta
            act = self.actual_delta
            same_dir = _improved(pred) == _improved(act) and pred != 0 and act != 0
            if same_dir:
                abs_ratio = abs(act) / abs(pred)
                if abs_ratio < 0.5:
                    f.append(f"prediction_undershot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
                elif abs_ratio > 2.0:
                    f.append(f"prediction_overshot(actual={act:+.5f} vs pred={pred:+.5f}, ratio={abs_ratio:.2f})")
            elif pred != 0 and act != 0:
                f.append(f"prediction_sign_mismatch(actual={act:+.5f} vs pred={pred:+.5f})")

        # 7. Multiple changes → attribution ambiguous.
        n_changes = (
            len(self.feature_changes)
            + len(self.pipeline_changes)
            + len(self.config_changes)
        )
        if n_changes > 1:
            f.append(f"multiple_changes(n={n_changes}) — attribution ambiguous, consider ablation")

    def commit(self) -> None:
        if self.oof_score_mean is None or self.holdout_score is None:
            raise RuntimeError(
                "Experiment.commit() requires .record() to have been called first."
            )
        self._autoflag()
        JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with JSONL_PATH.open("a") as fp:
            fp.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


def add_note(version: str, text: str) -> None:
    """Append a human note to an existing experiment (rewrites experiments.jsonl)."""
    if not text.strip():
        return
    entries = _load_jsonl()
    found = False
    for entry in entries:
        if entry.get("version") == version:
            entry.setdefault("notes", []).append(f"[{_now_iso()}] {text.strip()}")
            found = True
            break
    if not found:
        raise ValueError(f"No experiment {version!r} in {JSONL_PATH.name}.")
    with JSONL_PATH.open("w") as fp:
        for entry in entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
