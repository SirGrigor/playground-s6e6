# S6E6 Strategy — Predicting Stellar Class

**Hard rule (Ilja, 2026-06-01):** solo-built top-50. No public-notebook blends until the
final 2 submissions. Forums for signals/opinions/ideas ✓; copying notebook outputs into a
blend ✗ until finals — and we aim to *not need them*.

## The problem in one paragraph
Multiclass (GALAXY / STAR / QSO) on SDSS17-derived synthetic data. `redshift` separates
most rows (STAR ≈ 0, GALAXY low-moderate, QSO high) → near-ceiling accuracy. The synthetic
redshift 3-bin baseline already hits ~0.97. The competition is the last 1–2%: the
**galaxy↔QSO confusion cell** at moderate redshift + the **−9999 sentinel** rows. This is an
error-cell / residual-diagnostic problem.

## Phases (each gated by the experiment diary — hypothesis + predicted Δ before any fit)

**Phase 0 — Recon (no fits).** `notebooks/01_eda.py`. Confirm metric/target/deadline on the
page; column parity vs SDSS17; class balance; redshift separation; sentinel audit; redshift
baseline → sets the dashboard FLOOR. Compute-parity audit of top-5 (Deotte risk → expectation
calibration, `feedback_compute-parity-calibration`).

**Phase 1 — Decode the generator (synth-decoder, vendored).** adversarial_validation (train
vs test shift) → match synthetic↔original SDSS17 (the join gives true labels on test rows =
our **labeled CV signal** for endgame blend weights, `feedback_supervised-blend-not-lb-probing`)
→ quantization / −9999 fingerprint → **gate everything** (does recovered structure beat the
model on affected rows? — the majority-class-illusion guard).

**Phase 2 — signal_revelation.** Measure feature→target *shape* before FE. Build colors
(u-g, g-r, r-i, i-z), redshift×color interactions; handle sentinels explicitly.

**Phase 3 — Solo GBDT zoo.** LGBM / XGB / CatBoost from day 1, multi-seed, stratified
GroupKFold-safe CV, per-fold accuracy logged via observer. No HPO until signal exists
(`feedback_hpo-needs-signal`).

**Phase 4 — Diagnose the wall.** When levers go flat: error-cell matrix on the galaxy↔QSO
regime + `signal_hunt.py` (Bayes-floor + residual-boost R² + adv-AUC + SHAP) → *name* the
missing axis, don't guess (`feedback_discovery-first`, `feedback_signal-hunt`). Pull the LB
spread: razor-pack = benign floor; breakaway = extractable signal we missed.

**Phase 5 — Endgame.** Caruana greedy selection (rogii `src/blend.py`) over OUR model zoo,
weights supervised on leak-matched labels — NOT Nelder-Mead, NOT public-LB grids. Ship 2
finals: #1 best-CV anchor, #2 explainable risk pick with a different failure mode. Track
CV-vs-LB correlation every submission (`feedback_endgame-discipline-and-caruana`).

## UI / traction (the two surfaces)
- **Live in-cell** (`src/dashboard.py`): rich render-on-update scoreboard + ascent goal banner
  (FLOOR → best → top-50 TARGET) + per-run verdict. Colab-safe (no rich.Live).
- **Persistent HTML** (`reports/dashboard.html`): confusion / ROC / PR / per-class F1 + diary
  timeline, base64-inlined → opens from Drive. Auto-rendered by bootstrap after each run.

## Infra
Colab-native (`colab/bootstrap.py` single source of truth + thin `colab_runner.ipynb` +
`SPRINT_ACTIVE.txt`). Own toolkits **vendored** (fresh clone + PYTHONPATH), never pip-from-git
(`feedback_colab-vendor-not-pip-own-code`). Artifacts + diary sync to Drive + git each run.

## Open items
- **Metric unconfirmed** — almost certainly Accuracy; confirm on /evaluation. If logloss:
  flip `config.GREATER_IS_BETTER`, calibration becomes first-class.
- FLOOR/TARGET in `dashboard.py` are placeholders → set after Phase-0 baseline + LB audit.
- synth-decoder GitHub remote: confirm it's pushed before the bootstrap clones it.
