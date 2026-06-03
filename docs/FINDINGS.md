# S6E6 — Findings & Lessons (Predicting Stellar Class)

**Competition:** Kaggle Playground Series S6E6 — multiclass `GALAXY / STAR / QSO`, synthetically
generated from fedesoriano's SDSS17. Metric: **balanced accuracy**. ~577K train rows, 12 columns.
Swag/portfolio comp → **methodology is the deliverable.**

**The arc in one line:** `0.96641 → 0.96933` LB — by catching a *false ceiling* (a GBDT-architecture
artifact we had "proven 4 ways"), rebuilding RealMLP, discovering **feature engineering** was the real
lever, and stacking decorrelated models. Passed the public single-model SOTA (Demidov 0.96920); within
~0.0002 of the 2nd-place blend (Deotte 0.96949). Solo, 3 models.

---

## 1. Version history

| Ver | Parent | Change | Holdout | LB | Verdict |
|-----|--------|--------|---------|----|---------|
| v1 | — | LGBM (redshift+bands+colors), class-weights + per-class threshold tuning | 0.96548 | — | baseline |
| v2 | v1 | original-SDSS17 leak-match (phot+z join) | 0.96393 | — | **dead** (0% join, adv-AUC 0.499 → genuinely generated) |
| v3 | v1 | + `spectral_type`,`galaxy_population` (LGBM-native cats) | 0.96570 | — | redundant (+0.0002 — color re-bins) |
| v4 | v3 | error-cell + error-predictability diagnostic | — | — | errors *structured* (predict-own-errors AUC 0.92) |
| v5 | v3 | drop class_weight + direct bal-acc threshold tuning | 0.96574 | — | Pareto-exhausted decision rule |
| v6 | v5 | LGBM+XGB+CatBoost zoo | 0.96606 | — | ρ0.94–0.96, CatBoost worst |
| v7 | v6 | RealMLP (pytabkit, **under-tuned**, 256 ep) | 0.96606 | — | decorrelated (ρ0.72) but **weak 0.9586 → 0 weight** ← *the costly mistake* |
| v8 | v6 | seed-bag LGBM+XGB ×3 + Caruana | 0.96561 | **0.96641** | best-at-the-time; 2 finals locked |
| v9 | v8 | cleanlab + kNN-Bayes ceiling diagnostic | — | — | "0.9% noise + 2.4% overlap" ← *measured with GBDT probs (flawed, see L1)* |
| v10 | v8 | TabICLv2 foundation model | 0.96606 | — | decorrelated (ρ0.89) but weak → 0 weight |
| v11 | v8 | domain feats (Galactic \|b\|, locus L, dropout flags) | 0.96613 | — | +0.0004 real but tiny; physics *survived* generation (T1 ρ=−0.72) |
| **v13** | v8 | **proper RealMLP** (PBLD, n_ens bag, balanced-softmax, EMA), bare features | 0.96087 | — | **LOST to GBDT** — architecture alone insufficient |
| **v14** | v13 | **+ rich FE** (ratios, num→cat, crossed cats, per-class OOF TE) | 0.96901 | **0.96893** (rm) / 0.96845 (lgb) | **BROKE THE CEILING** — FE was the gap |
| **v15** | v14 | **LogReg-on-logits blend** (lgb+FE & rm+FE, ρ0.85) | 0.96940 | **0.96933** | **best** — diversity cashed out |
| v16 | v15 | + CatBoost+FE 3rd leg | 0.96951 | 0.96924 | **regressed** — CatBoost redundant (ρ0.936 w/ lgb), holdout-overfit |
| v17 | v15 | seed-bag RealMLP ×3 | — | 0.96930 | no gain — `n_ens=8` already did the variance reduction |

*(no "v12" — notebook 12 holds v11; version labels skip 12.)*

---

## 2. The big lessons (the real deliverable)

**L1 — The "proven ceiling" was a GBDT-architecture artifact, not the competition's.**
We declared 0.9664 a hard ceiling and "proved it 4 ways" (v9: cleanlab label-noise + kNN-Bayes-floor +
error-confidence + OvO pin). All four were computed on **GBDT** probabilities and a **kNN** floor — both
axis-aligned-ish learners. A properly-built RealMLP extracted **+0.0025 more** from the *same columns*.
→ **A Bayes-error / "irreducible overlap" estimate is only as good as the model class you measure it
with.** Our own `discovery-first` rule ("a flat lever = OUR method saturated, NOT a proven ceiling")
applied to the *ceiling estimate itself*.

**L2 — Feature engineering was the lever, not architecture.**
v13 (proper RealMLP, bare features) *lost* to the GBDT (0.9609 vs 0.9657). v14 (same RealMLP + rich FE)
hit 0.9690. The decomposition: v7 pytabkit-default 0.9586 → v13 our-recipe 0.9609 (+0.0023 from training
refinements) → v14 0.9690 (+0.0081 from FE). **The public 0.968 was FE + architecture together**, and on
bare features RealMLP < GBDT. The strong FE (per-class **target-encoded combos**, numeric→categorical
embeddings) is NN-centric — it feeds the embedding layers our minimal set starved.

**L3 — "Decorrelation is necessary but NOT sufficient." (three strikes)**
A blend member must be decorrelated **AND competitive**. A diverse-but-weak model loses every argmax tie
→ 0 ensemble weight. Confirmed three times: NN (v7, 0.9586), TabICL (v10, 0.9603), CatBoost (v16, 0.9662).

**L4 — Stacking (LogReg-on-logits) beat Caruana for this metric.**
Caruana optimizes *argmax* balanced-accuracy and repeatedly **degenerated** (dropped LGBM in v15, dropped
CatBoost in v16) because those legs didn't help the argmax objective. The smooth **LogReg stacker**
extracted the probability-level signal Caruana threw away → won every blend. Objective-misalignment, same
lesson as S6E5/v8 but now the proper stacker wins.

**L5 — At the ceiling, sub-0.0002 holdout deltas are NOISE.**
v16 (holdout 0.96951 > v15 0.96940) scored *lower* on LB (0.96924 < 0.96933) — the first holdout↔LB rank
disagreement, both inside the ±0.0002 CV↔LB gap. → Stop chasing sub-noise holdout gains; decide on LB.

**L6 — Variance-reduction levers can be pre-exhausted.** Seed-bagging the RealMLP (v17) gave nothing
because the model *already* has `n_ens=8` internal bagging — the variance was squeezed. Know what your
model already does before adding an outer wrapper.

**L7 — CV↔LB was calibrated tight (±0.0001–0.0003), including for the NN** (v14 rm 0.9690→0.96893,
v15 0.9694→0.96933). → trust CV, iterate on the holdout, no LB-chasing needed.

**L8 — Calibrate predictions down at a saturated signal.** We over-predicted lifts 4–5× running early
(v3/v5/v6/v7/v13). The first *calibrated* prediction was v11 (+0.0005 predicted, +0.0005 actual). Default
to predicting ~0 for any re-expression once the signal is saturated.

---

## 3. What worked vs what didn't

**Worked (the +0.0029):**
- **Rich FE** — per-class OOF target-encoded crossed-cats, numeric→categorical embeddings, ratios. The
  single biggest lever (v14, +0.008).
- **Proper RealMLP recipe** — PBLD periodic embeddings, `n_ens=8` internal bag, **balanced-softmax
  metric-aware loss** (`loss_prior_power`, not class-weights), EMA, per-parameter-group LRs.
- **LogReg-on-logits stacking** of decorrelated legs (ρ0.85).

**Didn't move it:**
- Leak-match (v2), domain features (v11, +0.0004), CatBoost 3rd leg (v16, regressed), seed-bag (v17, 0).
- **Signal-extraction matrix (72 approaches, swarm-audited):** every one either reconstructable by a
  GBDT, mechanism-dead by closed form, or synthetic-severed. *Feature construction was genuinely
  exhausted* — but the **paradigm** (GBDT→NN) and **FE** were the real levers, not more features.

---

## 4. Final standing & submissions
- **Finals: anchor = `v15_final.csv` (LB 0.96933, 2-way LogReg blend); hedge = `v14_realmlp_fe.csv`
  (0.96893, single NN — different failure mode).**
- Superseded the old v8 0.96641 locked finals.

---

## 5. Reusable assets built (portfolio capabilities, comp-agnostic)
- `src/realmlp.py` — custom PyTorch RealMLP (PBLD + n_ens bag + balanced-softmax + EMA) + `realmlp_oof`
  / `realmlp_fit_predict`. **Banked for every future tabular comp where GBDT plateaus.**
- `src/fe_realmlp.py` — rich FE (ratios, num→cat, KBins, crossed cats) + leak-safe **per-fold target
  encoding** + the lgb/realmlp/catboost **race harness** (with `with_cat`, `rm_seeds` seed-bagging).
- `src/blend.py` — Caruana greedy + (notebook) **LogReg-on-logits stacker**.
- `src/domain.py` — domain feature builders + confirm-or-kill EDA probes (T1–T4).
- **Signal-extraction matrix** — a 72-approach coverage map (tried/untried × reconstructable × EV).
- **Kaggle-Kernels conveyor harness** — validated headless push→poll→pull (see §6).

---

## 6. Autonomous conveyor — feasibility (proven)
Goal: an autonomous experiment loop (generate → run on Kaggle GPU → harvest → decide → repeat), the way
Deotte's Codex ran 218 models. **Feasibility PROVEN** via two probe kernels (`iljagrigorjev/s6e6-conveyor-probe`):
- ✅ **Headless loop** — `kaggle kernels push`→RUNNING→COMPLETE in ~40s; poll + output-pull work; driven end-to-end.
- ✅ **Competition data** — native at `/kaggle/input/competitions/playground-series-s6e6/`, zero download.
- ✅ **GPU** — Kaggle assigns a **P100 (sm_60)** that its torch 2.10+cu128 can't drive; fix validated:
  `pip install torch==2.5.1 --index-url .../cu121` → GPU works (install ~178s/kernel, optimizable via a
  pre-built-wheel dataset). GBDT/FE kernels need no torch.
- **Constraints:** ~30 GPU-h/week quota; real experiments ~30–40 min each → the conveyor runs over days,
  not hours; semi-autonomous with budget guardrails is the pragmatic model.
- **Next (Phase 1):** the loop — config → render notebook (reuse the modules above) → push → poll → pull
  → parse CV/holdout → leaderboard → decide. Phase 2 = search space (RealMLP Optuna, FE ops, TabM/2nd-NN).

---

## 7. Methodology meta-lesson
The win came from a **discovery-first loop**, not luck:
1. **Name the false ceiling** (an external result — Deotte's writeup — exposed it).
2. **Isolate the variable** (v13: RealMLP on the *same* features → architecture alone).
3. **Diagnose the gap** (v13 lost → the gap is FE, not architecture).
4. **Fix and measure** (v14: add FE → ceiling broke).
5. **Compose** (v15: stack the decorrelated legs).

And the hardest lesson, worth more than the score: **we declared a ceiling we hadn't actually proven —
we'd proven our *method* was saturated.** The reusable guard: an "irreducible error" claim is only valid
relative to the model class that measured it. Change the paradigm before declaring physics.
