# playground-s6e6 — Predicting Stellar Class

Kaggle Playground Series **S6E6**: multiclass classification of Sloan Digital Sky Survey
objects into **GALAXY / STAR / QSO** (based on fedesoriano's SDSS17 dataset).

**Hard rule:** solo-built. No public-notebook blends until the final 2 submissions.
Target: top-50 on our own. Forums for signals/opinions ✓; copying notebook outputs ✗.

## Why this comp is what it is
`redshift` alone separates most rows (STAR ≈ 0, GALAXY low-moderate, QSO high), so this is
a **near-ceiling accuracy problem**. The whole game is the last 1–2%: the **galaxy↔QSO
confusion cell** at moderate redshift, and the **−9999 sentinel** rows in u/g/z. That's an
error-cell / residual-diagnostic problem — the muscle built in S6E5 + rogii.

## Layout
```
src/
  config.py      seeds, paths, SDSS17 schema, metric (VERIFY: accuracy)
  data.py        load_raw / load_original / holdout_split / synthetic_fallback
  observer.py    experiment diary — hypothesis+predicted-Δ gate, 7 detectors
  diary.py       read views over experiments.jsonl → docs/diary.md
  viz.py         multiclass panels: confusion, ROC, PR, per-class F1, balance, ρ-matrix
  dashboard.py   live in-cell rich scoreboard (ascent) + portable HTML dashboard
notebooks/
  01_eda.py      Phase-0 recon (ACTIVE) — parity, balance, redshift sep, baseline + dashboard
  colab_runner.ipynb   thin launcher — mount Drive, secrets, fresh-clone, run bootstrap
colab/bootstrap.py     single source of truth for a Colab run (install→data→vendor→run→dashboard→sync)
vendor/                fresh-cloned own toolkits (synth-decoder); reached via PYTHONPATH, never pip'd
SPRINT_ACTIVE.txt      the one script bootstrap runs; switch experiment = edit + push
```

## Run
Local smoke test (synthetic fallback, no download needed):
```
PYTHONPATH=. uv run python notebooks/01_eda.py   # writes reports/dashboard.html
```
Colab: open `notebooks/colab_runner.ipynb` → Run all. Switch experiment by editing
`SPRINT_ACTIVE.txt` and pushing.

## Method spine
Phase 0 recon → synth-decoder decode (adversarial → original-leak match for labeled CV →
fingerprint → **gate**) → signal_revelation on redshift/colors → solo GBDT zoo (multi-seed,
day 1) → error-cell + signal_hunt diagnostic at the wall → **Caruana endgame supervised on
leak-matched labels**, 2 finals (CV anchor + explainable risk pick). No public blends until finals.

See `docs/strategy.md` for the full plan and `docs/diary.md` for the live experiment log.
