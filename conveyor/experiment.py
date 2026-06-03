"""Config-driven experiment runner — runs ON Kaggle (or locally). The conveyor's unit of work.

A Kaggle kernel clones this repo, installs the GPU-compatible torch, then runs:
    python conveyor/experiment.py <config.json>
which builds the rich FE, runs the requested experiment (reusing src/fe_realmlp, realmlp, blend),
and writes `result.json` (harvested by the orchestrator from the kernel output) + prints a
`RESULT_JSON <...>` line for stdout parsing.

Config schema (Phase 1):
    {
      "id": "hpo_001",
      "kind": "realmlp_race",        # rich FE, race lgb + realmlp (+ catboost if with_cat)
      "rm_cfg": {"pbld_freq_scale": 2.5, "dropout": 0.05},   # RealMLP hyperparameter overrides
      "n_folds": 5,
      "rm_seeds": null,              # or [1,2,3] to seed-bag
      "with_cat": false
    }
This is the search unit for RealMLP HPO (vary rm_cfg) and FE/model search (later kinds).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# repo root on sys.path whether run from /kaggle/working/<repo> or the repo dir
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.config import CLASSES, TARGET  # noqa: E402
from src.fe_realmlp import build_rich_features, race_oof  # noqa: E402
from src.metrics import competition_score, tune_class_weights, weighted_predict  # noqa: E402

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))


def _load():
    """Kaggle competition data if present, else local data/raw, else synthetic fallback."""
    import glob
    import pandas as pd
    tr = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    te = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    if tr and te:
        print(f"[data] kaggle: {tr[0]}")
        return pd.read_csv(tr[0]), pd.read_csv(te[0]), False
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("[data] synthetic fallback")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def run(config: dict) -> dict:
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[run] id={config.get('id')} kind={config.get('kind')} train={train.shape}")

    Xrich, info, state = build_rich_features(train, fit=True)
    Xte, _, _ = build_rich_features(test, fit=False, state=state)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    dev_idx, hold_idx = data.holdout_split(train)
    Xdev, ydev = Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx]
    Xhold, yhold = Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx]
    yhold_names = INT2CLS[yhold]

    R = race_oof(Xdev, ydev, Xhold, Xte, info,
                 n_folds=int(config.get("n_folds", 5)),
                 cfg=config.get("rm_cfg", {}),
                 with_cat=bool(config.get("with_cat", False)),
                 rm_seeds=config.get("rm_seeds"))

    result = {"id": config.get("id"), "kind": config.get("kind", "realmlp_race"),
              "config": config, "synthetic": synthetic, "n_features": int(Xrich.shape[1])}
    for name in (["lgb", "rm"] + (["cat"] if config.get("with_cat") else [])):
        w, s_oof = tune_class_weights(R[f"{name}_oof"], ydev, labels=INTS)
        s_hold = competition_score(yhold_names, weighted_predict(R[f"{name}_hold"], w, labels=CLASSES))
        result[f"{name}_oof"] = round(s_oof, 5)
        result[f"{name}_holdout"] = round(s_hold, 5)
        print(f"[{name}] OOF {s_oof:.5f} | holdout {s_hold:.5f}")
    result["runtime_sec"] = round(time.time() - t0, 1)

    out = Path("/kaggle/working/result.json")
    out = out if out.parent.exists() else Path("result.json")
    out.write_text(json.dumps(result))
    print("RESULT_JSON", json.dumps(result))
    return result


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {"id": "smoke", "n_folds": 3}
    run(cfg)
