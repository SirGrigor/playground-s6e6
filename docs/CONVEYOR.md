# The Autonomous Conveyor — architecture, methodology, results

An autonomous ML experimentation loop on **headless Kaggle compute**: generate a config → render a
kernel → push to Kaggle GPU → poll → harvest → stack → leaderboard, **with no human in the loop**.
Built when hand-tuning plateaued at v16 (0.96924); the remaining gains were search-heavy (model
diversity, HPO) — exactly what an autonomous loop is for. Inspired by Deotte's Codex `--yolo` fleet.

**Result: the conveyor's first *original* experiment beat our hand-tuned best**, and the approach took
S6E6 from 0.96641 → **0.96979** (LB). The reusable machine + the validated decorrelation methodology are
the bigger deliverable than the score.

---

## 1. Architecture

| Piece | Runs | Role |
|---|---|---|
| `conveyor/experiment.py` | **on Kaggle** | config-driven: build rich FE → train model(s) → CV'd stack → `result.json` + persisted OOFs |
| `conveyor/orchestrator.py` | **locally** | render config→kernel (repo-clone + GPU-fix torch), push, poll, harvest → `leaderboard.jsonl` |
| `conveyor/fleet_stack.py` | **locally** | combine harvested single-model OOFs (sklearn CPU, instant): ρ matrix + greedy select + CV'd stack → submission |
| `src/stacker.py` | both | the CV'd seed-bagged class-weighted **log-odds LogReg stacker** (Deotte technique — fixes the v16 meta-overfit) |

**Two experiment modes:**
- **bundled race** — `lgb + rm + tabm [+cat] [+extra_nn]` in one kernel (config: `with_tabm`, `extra_nn`).
- **single-model** (`config["single"]`) — *one model per kernel*, persists aligned OOF/hold/test + labels →
  the **parallel fleet** (run many models concurrently, stack locally afterward).

**Model legs available** (`single_oof`): `lgb`, `rm` (RealMLP), `tabm` (TabM), and CPU paradigms
`extratrees`, `rf`, `histgb`, `logreg`.

---

## 2. The GPU recipe (validated, Phase 0.5)
Kaggle assigns a **P100 (sm_60)**; the preinstalled `torch 2.10+cu128` *can't drive it* ("no kernel image").
**Fix, baked into every GPU kernel:** `pip install torch==2.5.1 --index-url .../cu121` (~178s). CPU legs
(GBDT/sklearn) need no GPU and no torch. Competition data is **native** at `/kaggle/input` — zero download.

## 3. Kaggle-Kernels gotchas (hard-won, all now handled)
- **MAX 2 concurrent GPU sessions** ("Maximum batch GPU session count of 2 reached") → fire GPU kernels in **waves of 2**.
- **Rapid new-kernel creation → "Notebook not found"** *and corrupts the slug* (un-revivable) → **stagger pushes ~15s**, and on failure **retry with a FRESH id**.
- **Status `404 ...Client Error` contains "Error"** → don't mistake it for the terminal ERROR status. `_is_done()` requires the real `KernelWorkerStatus.` enum.
- **Transient DNS blips** mid-harvest → just retry the harvest (the kernel already completed).

## 4. The parallel-fleet pattern
`single_oof` uses the **fixed** holdout split (`HOLDOUT_SEED`) + CV folds (`CV_SEED`), so OOFs from
*separate kernels align* and can be stacked. One model per kernel ⇒ **parallel** (subject to the 2-GPU cap)
**+ free local re-stacking** (add a leg, re-stack in seconds, no GPU). CPU paradigm legs dodge the GPU cap entirely.

---

## 5. The decorrelation methodology (the core lesson)

**Stack value ≈ strength × (1 − correlation).** Two models help an ensemble only when they make
*different errors* — which comes from different **inductive bias** and **information**, not from re-rolling
the same model.

**Decorrelation levers, ranked:** paradigm/algorithm 🔥🔥🔥 > feature set 🔥🔥 > objective/data 🔥 >
encoding 🔥 > **hyperparameters** (weak) > **seeds** (~zero, that's *bagging*).

**Empirical proof (ρ = rank-correlation of OOF probabilities):**
- **Variants are near-clones** — `rm`↔`rm_b` = **0.93**, `tabm`↔`tabm_b` = **0.92**. Hyperparameter tweaks
  do NOT decorrelate. (This is why adding `rm_b`/`tabm_b` barely moved the stack.)
- **CPU paradigms genuinely decorrelate** — `extratrees` is the lowest-ρ leg in the whole roster (ρ 0.66
  with `rm`!), `logreg` ρ ~0.76–0.85. *But they are weak* (0.95–0.967).
- **`histgb` ≈ `lgb`** (ρ 0.91) — both GBDTs, redundant.

**Greedy forward-selection** (on the CV'd-stack score, cheap 1-seed search) *automates complementarity*:
on the 9-leg roster it **kept** `[tabm_b, lgb, extratrees, tabm_c, logreg]` — i.e. it **kept the weak-but-
decorrelated paradigms and dropped the correlated clones** (`rm`, `rm_b`, `histgb`) — beating the naive
all-9 stack (0.96943) by +0.00027. **Decorrelation, measured and selected, beats "more legs."**

---

## 6. Results (the conveyor's autonomous contributions)

| Run | Composition | LB / holdout | Note |
|---|---|---|---|
| v15 (hand) | lgb+rm 2-way stack | LB 0.96933 | pre-conveyor best |
| **tabm1** | lgb+rm+**tabm** 3-leg | **LB 0.96955** | conveyor's 1st original exp — **beat hand-tuning** |
| **fleet1** | +`tabm_b` 4-leg | **LB 0.96979** | **best** (a lucky, complementary bundled draw) |
| 9-leg greedy | `[tabm_b,lgb,extratrees,tabm_c,logreg]` | holdout 0.96970 | decorrelation validated; *matched* fleet1 from a less-lucky single-model draw |

**CV↔LB tracked within ±0.0001 across every submission** → fully trust the holdout.
**Plateau ~0.9697–0.9698.** Best = fleet1 (0.96979).

---

## 7. Lessons
1. **The conveyor beat hand-tuning on its first original run** — the thesis (autonomous search > hand-iteration at a plateau) held.
2. **Decorrelation = different paradigm + features, NOT variants/seeds** — proven via the ρ matrix.
3. **Greedy selection automates the strength×decorrelation trade-off** — keeps diverse legs, rejects clones.
4. **The plateau's cause:** the decorrelated paradigms (ExtraTrees, LogReg) are *too weak* to push past the
   strong NN core — they recover a bad draw but don't exceed the best. → **the next lever is DEPTH**:
   make the legs *stronger* (HPO, seed-averaging) so they're decorrelated **and** strong.

## 8. Reusability
The whole thing is **comp-agnostic** — a Kaggle experiment factory for any tabular competition: drop in a
new dataset + model legs, and the conveyor runs the fleet, measures decorrelation, and selects the best
complementary stack. That capability outlasts any single leaderboard.
