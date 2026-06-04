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


def _rho(a, b):
    from scipy.stats import rankdata
    return float(np.mean([np.corrcoef(rankdata(a[:, k]), rankdata(b[:, k]))[0, 1] for k in range(a.shape[1])]))


def _rho_matrix(oofs: dict):
    names = list(oofs)
    print("\n[rho] pairwise rank-correlation (lower = more decorrelated):")
    print("        " + "".join(f"{n[:7]:>8}" for n in names))
    for a in names:
        print(f"  {a[:7]:>6} " + "".join(f"{_rho(oofs[a], oofs[b]):>8.3f}" for b in names))


def greedy_select(oofs: dict, ydev, min_gain=1e-5):
    """Forward selection on the CV'd-stack OOF score — add the leg that most improves the STACK,
    not the best single. Auto-rejects correlated-but-redundant members (e.g. rm_b)."""
    remaining, selected, best = set(oofs), [], -1.0
    while remaining:
        scored = []
        for n in remaining:
            sub = selected + [n]
            _, _, cv = cv_logreg_stack({k: oofs[k] for k in sub}, ydev, {}, n_seeds=1, n_folds=3)  # cheap for search
            scored.append((cv, n))
        cv, n = max(scored)
        if selected and cv <= best + min_gain:
            break
        best, selected = cv, selected + [n]; remaining.discard(n)
        print(f"  + {n:<10} stack-CV {cv:.5f}")
    return selected, best


def fleet_stack(ids: list, out_csv: Path = ROOT / "fleet_submission.csv"):
    dirs = [ROOT / i for i in ids]
    labels_dir = next((d for d in dirs if (d / "y_dev.npy").exists()), None)
    if labels_dir is None:
        raise SystemExit("no y_dev.npy in any harvest dir — were the single kernels run?")
    ydev = np.load(labels_dir / "y_dev.npy"); yhold = np.load(labels_dir / "y_hold.npy")
    test_ids = pd.read_csv(labels_dir / "test_ids.csv")

    oofs, holds, tests = {}, {}, {}
    for d in dirs:
        try:                                              # skip incomplete harvests (Gemini audit: robustness)
            nm = _model_name(d)
            oofs[nm] = np.load(d / f"oof_{nm}.npy")
            holds[nm] = np.load(d / f"hold_{nm}.npy")
            tests[nm] = np.load(d / f"test_{nm}.npy")
        except (StopIteration, FileNotFoundError):
            print(f"  skip {d.name}: incomplete artifacts"); continue
        ho = competition_score(INT2CLS[yhold],
                               weighted_predict(holds[nm], tune_class_weights(oofs[nm], ydev, labels=INTS)[0], labels=CLASSES))
        print(f"  leg {nm:<10} holdout {ho:.5f}")

    _rho_matrix(oofs)

    def _stack(names):
        so, ap, cv = cv_logreg_stack({k: oofs[k] for k in names}, ydev,
                                     {"hold": {k: holds[k] for k in names}, "test": {k: tests[k] for k in names}})
        w, _ = tune_class_weights(so, ydev, labels=INTS)
        h = competition_score(INT2CLS[yhold], weighted_predict(ap["hold"], w, labels=CLASSES))
        return cv, h, ap["test"], w

    cv_all, h_all, test_all, w_all = _stack(list(oofs))   # cache (Gemini audit: was double-computed)
    print(f"\n[stack-all] {len(oofs)} legs → CV {cv_all:.5f} | holdout {h_all:.5f}")

    print("\n[greedy] forward-selecting the complementary subset (on stack-CV):")
    chosen, cv_best = greedy_select(oofs, ydev)
    cv_g, h_g, test_g, w_g = _stack(chosen)
    print(f"[greedy] chosen {chosen} → CV {cv_g:.5f} | holdout {h_g:.5f}")

    # submit the better of all-legs vs greedy subset (decision on the pristine holdout)
    use_all = h_all >= h_g
    test_f, w_f = (test_all, w_all) if use_all else (test_g, w_g)
    pd.DataFrame({ID: test_ids[ID], TARGET: weighted_predict(test_f, w_f, labels=CLASSES)}).to_csv(out_csv, index=False)
    print(f"\n[fleet-stack] WINNER = {'all '+str(len(oofs))+' legs' if use_all else 'greedy '+str(chosen)} "
          f"→ holdout {max(h_all, h_g):.5f} | submission → {out_csv}")
    return max(h_all, h_g)


if __name__ == "__main__":
    fleet_stack(sys.argv[1:])
