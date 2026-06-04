"""Modal backend for the conveyor fleet — 10-wide parallel, friction-free.

Why: Kaggle Kernels capped us at 2 concurrent GPU + rapid-push corruption + wave-batching. Modal's
Starter tier gives 10 GPU concurrency + $30/mo free credits + modern GPUs (A10G sm_86 → NO P100 torch
pin needed). `run_model.map(configs)` fires the WHOLE fleet in parallel and returns the OOFs directly —
all the Kaggle orchestration (push/poll/harvest/corruption/waves) just vanishes. The modeling stack
(src/single_oof) and the decorrelation layer (conveyor/fleet_stack) are reused UNCHANGED.

One-time setup:
    pip install modal
    modal setup                                          # browser auth
    modal secret create kaggle KAGGLE_USERNAME=iljagrigorjev KAGGLE_KEY=<your-key>
    modal run conveyor/modal_app.py::setup_data          # download comp data into a Volume (once)

Run a fleet (configs = a JSON list of single-model configs, same schema as the Kaggle path):
    modal run conveyor/modal_app.py --configs-json /tmp/fleet.json
    # → writes aligned OOFs to /tmp/conveyor/<id>/ locally, then:
    PYTHONPATH=. python conveyor/fleet_stack.py <id1> <id2> ...   # local stack (rho + greedy)
"""
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
DATA = "/data"
COMP = "playground-series-s6e6"

app = modal.App("s6e6-fleet")

# Modern GPUs (T4 sm_75 / A10G sm_86) run current torch — no torch==2.5.1 P100 pin required.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2.0", "pandas", "scikit-learn", "scipy", "lightgbm", "torch", "kaggle")
    .add_local_dir(str(REPO / "src"), "/root/src")        # our modeling stack, reused unchanged
)
data_vol = modal.Volume.from_name("s6e6-data", create_if_missing=True)


@app.function(image=image, volumes={DATA: data_vol}, secrets=[modal.Secret.from_name("kaggle")])
def setup_data():
    """Download the competition data into the Volume once (every model fn then reuses it)."""
    import os
    import subprocess
    os.makedirs(DATA, exist_ok=True)
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-p", DATA, "--unzip"], check=True)
    data_vol.commit()
    print("data ready:", sorted(os.listdir(DATA)))


@app.function(image=image, gpu="A10G", volumes={DATA: data_vol}, timeout=7200)
def run_model(config: dict) -> dict:
    """ONE model (single_oof) on the cached data → return aligned oof/hold/test + labels + scores.
    Same fixed-seed folds as the Kaggle path, so OOFs from every Modal call ALIGN for the local stack."""
    import sys
    import numpy as np
    import pandas as pd
    sys.path.insert(0, "/root")
    from src import data as datamod
    from src.config import CLASSES, TARGET
    from src.fe_realmlp import build_rich_features, single_oof
    from src.metrics import competition_score, tune_class_weights, weighted_predict

    cls2int = {c: i for i, c in enumerate(CLASSES)}
    int2cls = np.asarray(CLASSES); ints = list(range(len(CLASSES)))
    train = pd.read_csv(f"{DATA}/train.csv"); test = pd.read_csv(f"{DATA}/test.csv")
    Xr, info, st = build_rich_features(train, fit=True)
    Xte, _, _ = build_rich_features(test, fit=False, state=st)
    y = train[TARGET].map(cls2int).to_numpy()
    dev, hold = datamod.holdout_split(train)
    Xdev, ydev = Xr.iloc[dev].reset_index(drop=True), y[dev]
    Xhold, yhold = Xr.iloc[hold].reset_index(drop=True), y[hold]

    oof, h, t = single_oof(Xdev, ydev, Xhold, Xte, info, config["single"], int(config.get("n_folds", 5)))
    w, s_oof = tune_class_weights(oof, ydev, labels=ints)
    s_hold = competition_score(int2cls[yhold], weighted_predict(h, w, labels=CLASSES))
    print(f"[{config['single']['name']}] OOF {s_oof:.5f} | holdout {s_hold:.5f}")
    return {"id": config["id"], "model": config["single"]["name"], "oof": oof, "hold": h, "test": t,
            "y_dev": ydev, "y_hold": yhold, "test_ids": test["id"].to_numpy(),
            "holdout": round(float(s_hold), 5), "oof_score": round(float(s_oof), 5)}


@app.local_entrypoint()
def main(configs_json: str):
    """Fire the whole fleet in parallel on Modal, save aligned OOFs locally for fleet_stack."""
    import json
    import numpy as np
    import pandas as pd
    configs = json.load(open(configs_json))
    print(f"firing {len(configs)} models in parallel on Modal (up to 10 GPUs)…")
    for cfg, r in zip(configs, run_model.map(configs)):
        d = Path("/tmp/conveyor") / cfg["id"]; d.mkdir(parents=True, exist_ok=True)
        nm = r["model"]
        np.save(d / f"oof_{nm}.npy", r["oof"]); np.save(d / f"hold_{nm}.npy", r["hold"])
        np.save(d / f"test_{nm}.npy", r["test"])
        np.save(d / "y_dev.npy", r["y_dev"]); np.save(d / "y_hold.npy", r["y_hold"])
        pd.DataFrame({"id": r["test_ids"]}).to_csv(d / "test_ids.csv", index=False)
        print(f"  {cfg['id']:<10} {nm:<12} holdout {r['holdout']}")
    ids = " ".join(c["id"] for c in configs)
    print(f"\nall done → stack locally:\n  PYTHONPATH=. python conveyor/fleet_stack.py {ids}")
