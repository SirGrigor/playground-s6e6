"""Cross-validation splits. Stratified by class so the rare QSO/STAR folds are honest
(critical under balanced accuracy, where rare-class recall counts equally)."""
from __future__ import annotations

import numpy as np

from .config import CV_SEED, N_FOLDS


def stratified_folds(y, n_splits: int = N_FOLDS, seed: int = CV_SEED):
    """Yield (train_idx, val_idx) for StratifiedKFold over labels `y`."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = np.asarray(y)
    for tr, va in skf.split(np.zeros(len(y)), y):
        yield tr, va
