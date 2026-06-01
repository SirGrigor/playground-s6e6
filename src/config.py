"""Centralized configuration: seeds, paths, constants.

Single source of truth. Imported by every notebook and module so that
seeds and paths cannot drift across the project.

NOTE (2026-06-01): schema below is the KNOWN fedesoriano SDSS17 original.
The Playground synthetic train.csv must be diffed against this on first load
(notebooks/01_eda.py asserts column parity). Anything tagged `# VERIFY` is
inferred, not yet confirmed on the live competition page.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"          # fedesoriano SDSS17 original lives here
SPLITS = DATA / "splits"
PROBS = ROOT / "probs"
SUBMISSIONS = ROOT / "submissions"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figs"

# Seeds — locked across the entire competition.
CV_SEED = 42
HOLDOUT_SEED = 11
MODEL_SEED = 7
N_FOLDS = 5
HOLDOUT_FRAC = 0.20

# Problem definition --------------------------------------------------------
TARGET = "class"
ID = "id"
METRIC = "accuracy"                   # VERIFY on /evaluation tab (vs macro-F1 / logloss)
GREATER_IS_BETTER = True              # accuracy/F1 → True; logloss → flip to False
TASK = "multiclass"
CLASSES = ["GALAXY", "STAR", "QSO"]   # SDSS17 label set; confirm exact strings in train.csv
N_CLASSES = 3

# Schema (SDSS17 original) --------------------------------------------------
# The dominant separator. Stars ~0, galaxies low-moderate, QSOs high.
PHOTOMETRIC_BANDS = ["u", "g", "r", "i", "z"]        # magnitudes
STRONG_NUMERIC = ["redshift"]
COORD_COLS = ["alpha", "delta"]                       # RA / Dec
# Survey identifiers — mostly junk, but plate/MJD/fiber can leak via spectroscopy
# observation structure. rerun_ID is constant (=301) in the original.
ID_LIKE_COLS = ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID",
                "spec_obj_ID", "plate", "MJD", "fiber_ID"]

# Known data quirk: the original has -9999 sentinel error values in u, g, z
# for a handful of rows. Treat explicitly, do not let them pollute scaling.
SENTINEL_VALUE = -9999.0
SENTINEL_COLS = ["u", "g", "z"]

# Color features — classic astronomical separators (adjacent-band differences).
def color_features(bands=PHOTOMETRIC_BANDS):
    """Adjacent-band color terms: u-g, g-r, r-i, i-z."""
    return [(bands[k], bands[k + 1]) for k in range(len(bands) - 1)]

# External original dataset (for leak-match + extra-row augmentation).
SDSS17_KAGGLE_DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
