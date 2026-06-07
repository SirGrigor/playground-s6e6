# S6E6 — Ceiling-Break Ladder (reopened 2026-06-07)

**Why reopened.** We locked at ~0.96979 calling it a "real ceiling" (three paradigms — RealMLP /
TabM / LGBM / FT — all converge to ~0.9697). That verdict was **half wrong**: the *architecture* axis
is exhausted, but we committed our own **L1 error a second time** — mistook OUR-method-saturation for
the competition ceiling. Live LB + community evidence (pulled 2026-06-07) shows two real signal axes,
**orthogonal to architecture**, that we never tried. This is the Rogii discipline applied: *a flat
lever = our method saturated, not a proven ceiling — name the missing signal axis.*

**Evidence (2026-06-07).**
- LB #1 Deotte **0.97131**; top-20 spans 0.97131→0.97106 = **0.00025** → tight noise ceiling, NO
  breakaway → no hidden key/leak. Our ~0.9698 sits ~0.0015 behind: an *accumulation* gap.
- Top public **single** model (RealMLP **0.96980**) = original-SDSS17 **CONCAT augmentation**
  (sample_weight 0.35, OOF folds over comp rows only) — NOT the row-match our v2 correctly killed
  (0% exact join). **We conflated "leak-match dead" with "original data useless."**
- Same model uses **generator-artifact** features (digit/decimal/mod per magnitude) — a *meta* axis
  our "72-approach signal matrix" (scoped to physics) missed entirely.
- LB leader's recipe = multinomial LogReg-on-logits stacker over **~12 decorrelated legs** = exactly
  our L4 stacker, just wider (we run 2–3 legs).
- Physics longshot (distance modulus M=m−μ(z)) → **Gemini + LB say skip**: v11 showed physics +0.0004;
  RealMLP color×z embeddings already approximate it.

**Gemini-hardened guardrail (the one trap).** Naive global weight 0.35 down-weights OOD rows but does
NOT correct the **prior shift** — original class balance ≠ competition's, and under **balanced
accuracy** that poisons per-class recall. **Fix:** per-class weight `w_c = (N_c_comp / N_c_orig) ×
0.35` so the added per-class weight-mass matches competition proportions. Coupling caveat: axes are
NOT independent — re-test the "three paradigms agree" ceiling AFTER augmentation, on the richer set.

---

## The ladder (each rung hypothesis-first, gated on the prior)

Targets are vs our clean reproducible best **v15 = 0.96933 LB / 0.96940 holdout** and the fleet anchor
**~0.96979**. Each rung writes its diary entry BEFORE training (`feedback_experiment-diary-required`).

### v18 — Augmentation AXIS confirmation on LGBM  *(cheap go/no-go)* — RUNNABLE
- **Why LGBM first:** native `sample_weight`; our custom RealMLP uses balanced-softmax, not per-row
  weights (`realmlp.py:16`) → plumbing deferred to v19. v18 proves the *axis* is real + prior-safe
  for a few minutes of CPU before we invest in the NN change.
- **Hypothesis:** concatenating ~100K original SDSS17 rows into each training fold, per-class weighted
  `w_c=(N_c_comp/N_c_orig)×0.35`, val on comp rows only, lifts the LGBM leg's OOF balanced-acc without
  degrading per-class (esp. QSO/STAR) recall.
- **Predicted Δ:** +0.0004 on the lgb single (≈0.9674 → ≈0.9678). Confidence: medium.
- **GATE → v19 iff:** OOF lift ≥ +0.0003 **AND** no per-class recall regression (QSO/STAR). If ≤0 or
  QSO recall drops → augmentation hurts under balanced-acc → STOP, revisit the weighting before NN.

### v19 — Augmentation on RealMLP  *(the primary lever)*
- **Prereq:** plumb per-row `sample_weight` into the custom RealMLP loss (`realmlp.py`), keeping
  `loss_prior_power` counts on the **competition** prior (metric-aware loss still targets comp balance).
- **Hypothesis:** per-class-weighted original concat lifts v14 RealMLP+FE single from **0.96901**
  toward **>0.9696** (Gemini's prediction; matches public single 0.96980).
- **Predicted Δ:** +0.0006. Confidence: medium.
- **GATE → v20 iff** OOF clears ~0.9696 and holds on holdout. Flat → axis weaker than community claims.

### v20 — Generator-artifact digit/decimal features
- Add per-magnitude `first_digit, last_digit, decimal_part, mod10, mod100, floor().factorize()`
  (+ redshift), on top of v19's augmented best.
- **Hypothesis:** generator quantization fingerprints add signal orthogonal to physics; same generator
  makes train + private test → generalizes (Gemini: risk ≈ 0). Lifts single +0.0003–0.0008.
- **Predicted Δ:** +0.0005. Confidence: medium.
- **GATE → v21 iff** positive and holds on holdout. Sub-noise → drop (cheap, harmless).

### v21 — Widen the decorrelated stack + LogReg-on-logits  *(accumulation → ~0.971)*
- Legs (all trained with v19 aug + v20 artifacts): RealMLP (seed-bag), TabM, TabPFN-v3, TabICLv2,
  XGB, LGBM, CatBoost. Existing `src/stacker.py` LogReg-on-logits (= LB leader's recipe).
- **Hypothesis:** 8–12 strengthened decorrelated legs accumulate to **~0.9705–0.9710** holdout,
  matching the top-of-LB accumulation pattern.
- **Predicted Δ:** +0.0008–0.0012 over best single. Confidence: medium. Track CV↔LB every submission.

### v22 (contingent, low priority) — distance modulus M=m−μ(z) single kill-test
- Only if v19–v21 underperform. Gemini/LB say skip; kept as a documented fallback, not on the path.

---

## Endgame (2 finals, `feedback_endgame-discipline-and-caruana`)
- **Anchor:** best-CV wide-stack (v21).
- **Hedge:** best augmented single (v19+v20) — different failure mode.
- Selection: 1 safe-CV anchor + 1 explainable risk pick; trust CV with tracked CV↔LB correlation.

## Honest calibration
+~0.0015 on a tightly-bunched **swag-only** LB — the *score* is minor. The deliverable is the
**methodology lesson** (caught the L1 error twice) + three banked, comp-agnostic capabilities:
per-class-weighted external-data concat, synthetic-generator artifact FE, and confirmation our stacker
== the LB leader's. "Methodology is the deliverable."
