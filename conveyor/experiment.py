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
import pandas as pd

# repo root on sys.path whether run from /kaggle/working/<repo> or the repo dir
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.config import CLASSES, ID, TARGET  # noqa: E402
from src.fe_realmlp import build_rich_features, race_oof, single_oof  # noqa: E402
from src.metrics import competition_score, tune_class_weights, weighted_predict  # noqa: E402
from src.stacker import cv_logreg_stack  # noqa: E402

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


def _prep(train, test):
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte, _, _ = build_rich_features(test, fit=False, state=state)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    dev_idx, hold_idx = data.holdout_split(train)
    return (Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx],
            Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx], Xte, info)


def _outdir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def run_single(config: dict) -> dict:
    """ONE model per kernel (parallel fleet). Persists aligned oof/hold/test + labels for a later stack."""
    t0 = time.time()
    train, test, synthetic = _load()
    spec = config["single"]   # {"name","type":"lgb"|"nn","base":...,"cfg":...}
    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    print(f"[single] id={config.get('id')} model={spec['name']} train={train.shape}")
    oof, hold, test_p = single_oof(Xdev, ydev, Xhold, Xte, info, spec, int(config.get("n_folds", 5)))
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    s_hold = competition_score(INT2CLS[yhold], weighted_predict(hold, w, labels=CLASSES))
    od = _outdir(); name = spec["name"]
    np.save(od / f"oof_{name}.npy", oof); np.save(od / f"hold_{name}.npy", hold); np.save(od / f"test_{name}.npy", test_p)
    np.save(od / "y_dev.npy", ydev); np.save(od / "y_hold.npy", yhold)
    test[[ID]].to_csv(od / "test_ids.csv", index=False)
    result = {"id": config.get("id"), "kind": "single", "model": name, "synthetic": synthetic,
              "oof": round(s_oof, 5), "holdout": round(s_hold, 5), "runtime_sec": round(time.time() - t0, 1)}
    (od / "result.json").write_text(json.dumps(result))
    print(f"[single] {name} OOF {s_oof:.5f} | holdout {s_hold:.5f}")
    print("RESULT_JSON", json.dumps(result))
    return result


def run(config: dict) -> dict:
    if config.get("single"):
        return run_single(config)
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[run] id={config.get('id')} kind={config.get('kind')} train={train.shape}")

    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    Xrich = Xdev  # (n_features reported below from the dev frame)
    yhold_names = INT2CLS[yhold]

    extra_nn = config.get("extra_nn") or []
    R = race_oof(Xdev, ydev, Xhold, Xte, info,
                 n_folds=int(config.get("n_folds", 5)),
                 cfg=config.get("rm_cfg", {}),
                 with_cat=bool(config.get("with_cat", False)),
                 with_tabm=bool(config.get("with_tabm", False)),
                 tabm_cfg=config.get("tabm_cfg"),
                 rm_seeds=config.get("rm_seeds"),
                 extra_nn=extra_nn)

    legs = (["lgb", "rm"] + (["tabm"] if config.get("with_tabm") else [])
            + [e["name"] for e in extra_nn] + (["cat"] if config.get("with_cat") else []))
    result = {"id": config.get("id"), "kind": config.get("kind", "realmlp_race"),
              "config": config, "synthetic": synthetic, "n_features": int(Xrich.shape[1]), "legs": legs}
    for name in legs:
        w, s_oof = tune_class_weights(R[f"{name}_oof"], ydev, labels=INTS)
        s_hold = competition_score(yhold_names, weighted_predict(R[f"{name}_hold"], w, labels=CLASSES))
        result[f"{name}_oof"] = round(s_oof, 5)
        result[f"{name}_holdout"] = round(s_hold, 5)
        print(f"[{name}] OOF {s_oof:.5f} | holdout {s_hold:.5f}")

    # CV'd seed-bagged log-odds stacker (the v16-overfit fix) — the conveyor's headline number
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    if len(legs) >= 2:
        oof_d = {l: R[f"{l}_oof"] for l in legs}
        stack_oof, stacked, stack_cv = cv_logreg_stack(
            oof_d, ydev, {"hold": {l: R[f"{l}_hold"] for l in legs}, "test": {l: R[f"{l}_test"] for l in legs}})
        sw, _ = tune_class_weights(stack_oof, ydev, labels=INTS)
        stack_hold = competition_score(yhold_names, weighted_predict(stacked["hold"], sw, labels=CLASSES))
        result["stack_cv"] = round(stack_cv, 5)
        result["stack_holdout"] = round(stack_hold, 5)
        print(f"[stack] CV {stack_cv:.5f} | holdout {stack_hold:.5f}  (legs={legs})")
        # write the stacked submission + persist OOFs (for re-blends without re-running)
        sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(stacked["test"], sw, labels=CLASSES)})
        sub.to_csv(out_dir / "submission.csv", index=False)
        for l in legs:
            np.save(out_dir / f"oof_{l}.npy", R[f"{l}_oof"]); np.save(out_dir / f"test_{l}.npy", R[f"{l}_test"])

    result["runtime_sec"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(result))
    print("RESULT_JSON", json.dumps(result))
    return result


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {"id": "smoke", "n_folds": 3}
    run(cfg)
