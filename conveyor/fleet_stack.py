"""Local fleet-stack — combine harvested single-model OOFs into a CV'd-stacked submission.

The parallel-fleet completion step. Each single-model kernel (run_single) persisted aligned
oof_<name>.npy / hold_<name>.npy / test_<name>.npy + y_dev.npy / y_hold.npy / test_ids.csv to its
harvest dir (/tmp/conveyor/<id>/). Because every kernel uses the same fixed holdout split + CV folds,
the OOFs ALIGN — so we stack them here LOCALLY (sklearn LogReg, CPU, instant), validate on the held-out
20%, and write the submission. No GPU/Kaggle needed for the stack → iterate the blend freely.

Usage:  PYTHONPATH=. python conveyor/fleet_stack.py <id1> <id2> ...   (harvest-dir ids under /tmp/conveyor)
        → writes /tmp/conveyor/fleet_submission.csv, prints stack CV + holdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CLASSES, ID, TARGET  # noqa: E402
from src.metrics import competition_score, tune_class_weights, weighted_predict  # noqa: E402
from src.stacker import cv_logreg_stack  # noqa: E402

INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))
ROOT = Path("/tmp/conveyor")


def _model_name(d: Path) -> str:
    f = next(d.glob("oof_*.npy"))
    return f.stem[len("oof_"):]


def fleet_stack(ids: list, out_csv: Path = ROOT / "fleet_submission.csv"):
    dirs = [ROOT / i for i in ids]
    labels_dir = next((d for d in dirs if (d / "y_dev.npy").exists()), None)
    if labels_dir is None:
        raise SystemExit("no y_dev.npy in any harvest dir — were the single kernels run?")
    ydev = np.load(labels_dir / "y_dev.npy"); yhold = np.load(labels_dir / "y_hold.npy")
    test_ids = pd.read_csv(labels_dir / "test_ids.csv")

    oofs, holds, tests = {}, {}, {}
    for d in dirs:
        nm = _model_name(d)
        oofs[nm] = np.load(d / f"oof_{nm}.npy")
        holds[nm] = np.load(d / f"hold_{nm}.npy")
        tests[nm] = np.load(d / f"test_{nm}.npy")
        ho = competition_score(INT2CLS[yhold],
                               weighted_predict(holds[nm], tune_class_weights(oofs[nm], ydev, labels=INTS)[0], labels=CLASSES))
        print(f"  leg {nm:<10} holdout {ho:.5f}")

    stack_oof, stacked, cv = cv_logreg_stack(oofs, ydev, {"hold": holds, "test": tests})
    w, _ = tune_class_weights(stack_oof, ydev, labels=INTS)
    stack_hold = competition_score(INT2CLS[yhold], weighted_predict(stacked["hold"], w, labels=CLASSES))
    print(f"\n[fleet-stack] {len(oofs)} legs {list(oofs)} → CV {cv:.5f} | holdout {stack_hold:.5f}")

    pd.DataFrame({ID: test_ids[ID], TARGET: weighted_predict(stacked["test"], w, labels=CLASSES)}).to_csv(out_csv, index=False)
    print(f"[fleet-stack] submission → {out_csv}")
    return stack_hold


if __name__ == "__main__":
    fleet_stack(sys.argv[1:])
