"""Config-driven experiment runner — runs ON Kaggle (or locally). The conveyor's unit of work.

A Kaggle kernel clones this repo, installs the GPU-compatible torch, then runs:
    python conveyor/experiment.py <config.json>
which builds the rich FE, runs the requested experiment (reusing src/fe_realmlp, realmlp, blend),
and writes `result.json` (harvested by the orchestrator from the kernel output) + prints a
`RESULT_JSON <...>` line for stdout parsing.

Config schema (Phase 1):
    {
      "id": "hpo_001",
      "kind": "realmlp_race",        # rich FE, race lgb + realmlp (+ catboost if with_cat)
      "rm_cfg": {"pbld_freq_scale": 2.5, "dropout": 0.05},   # RealMLP hyperparameter overrides
      "n_folds": 5,
      "rm_seeds": null,              # or [1,2,3] to seed-bag
      "with_cat": false
    }
This is the search unit for RealMLP HPO (vary rm_cfg) and FE/model search (later kinds).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# repo root on sys.path whether run from /kaggle/working/<repo> or the repo dir
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.config import CLASSES, ID, TARGET  # noqa: E402
from src.fe_realmlp import build_rich_features, race_oof, single_oof  # noqa: E402
from src.metrics import competition_score, tune_class_weights, weighted_predict  # noqa: E402
from src.stacker import cv_logreg_stack  # noqa: E402

CLS2INT = {c: i for i, c in enumerate(CLASSES)}
INT2CLS = np.asarray(CLASSES)
INTS = list(range(len(CLASSES)))


def _load():
    """Kaggle competition data if present, else local data/raw, else synthetic fallback."""
    import glob
    import pandas as pd
    tr = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    te = glob.glob("/kaggle/input/**/test.csv", recursive=True)
    if tr and te:
        print(f"[data] kaggle: {tr[0]}")
        return pd.read_csv(tr[0]), pd.read_csv(te[0]), False
    raw = data.load_raw()
    if raw is not None:
        return raw[0], raw[1], False
    print("[data] synthetic fallback")
    full = data.synthetic_fallback(n=8000)
    return full.iloc[:6000].reset_index(drop=True), full.iloc[6000:].drop(columns=[TARGET]).reset_index(drop=True), True


def _prep(train, test):
    Xrich, info, state = build_rich_features(train, fit=True)
    Xte, _, _ = build_rich_features(test, fit=False, state=state)
    y_all = train[TARGET].map(CLS2INT).to_numpy()
    dev_idx, hold_idx = data.holdout_split(train)
    return (Xrich.iloc[dev_idx].reset_index(drop=True), y_all[dev_idx],
            Xrich.iloc[hold_idx].reset_index(drop=True), y_all[hold_idx], Xte, info)


def _outdir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def run_single(config: dict) -> dict:
    """ONE model per kernel (parallel fleet). Persists aligned oof/hold/test + labels for a later stack."""
    t0 = time.time()
    train, test, synthetic = _load()
    spec = config["single"]   # {"name","type":"lgb"|"nn","base":...,"cfg":...}
    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    print(f"[single] id={config.get('id')} model={spec['name']} train={train.shape}")
    oof, hold, test_p = single_oof(Xdev, ydev, Xhold, Xte, info, spec, int(config.get("n_folds", 5)))
    w, s_oof = tune_class_weights(oof, ydev, labels=INTS)
    s_hold = competition_score(INT2CLS[yhold], weighted_predict(hold, w, labels=CLASSES))
    od = _outdir(); name = spec["name"]
    np.save(od / f"oof_{name}.npy", oof); np.save(od / f"hold_{name}.npy", hold); np.save(od / f"test_{name}.npy", test_p)
    np.save(od / "y_dev.npy", ydev); np.save(od / "y_hold.npy", yhold)
    test[[ID]].to_csv(od / "test_ids.csv", index=False)
    result = {"id": config.get("id"), "kind": "single", "model": name, "synthetic": synthetic,
              "oof": round(s_oof, 5), "holdout": round(s_hold, 5), "runtime_sec": round(time.time() - t0, 1)}
    (od / "result.json").write_text(json.dumps(result))
    print(f"[single] {name} OOF {s_oof:.5f} | holdout {s_hold:.5f}")
    print("RESULT_JSON", json.dumps(result))
    return result


def run_diagnose(config: dict) -> dict:
    """Error-cell diagnostic — IS 0.9697 the data's floor, or is there a signal we're missing?

    The decorrelation lever exhausted (kNN-view ρ0.766 added +0.00004). This asks the mechanism
    question directly: (1) WHERE the errors are (confusion), (2) the kNN-Bayes floor in full feature
    space, (3) is local label-disagreement HIGHER on our error rows (→ genuine overlap), (4) the
    local-ORACLE ceiling: label each row by its neighbours' TRUE majority — if that can't beat 0.9697,
    no model can. (5) coords loose-end: do alpha/delta carry signal we excluded?"""
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    sys.stdout.reconfigure(line_buffering=True)   # unbuffered: survive a crash with output intact
    t0 = time.time()
    train, test, synthetic = _load()
    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    nc = len(CLASSES); ydev = np.asarray(ydev)
    print(f"[diagnose] dev={Xdev.shape} synthetic={synthetic}", flush=True)

    # (1) strong proxy for the consensus error cell: the actual lgb leg's OOF (ρ~0.8 w/ the fleet)
    oof, _, _ = single_oof(Xdev, ydev, Xhold, Xte, info, {"name": "lgb", "type": "lgb"}, 5)
    pred = oof.argmax(1)
    err = pred != ydev
    base_ba = balanced_accuracy_score(ydev, pred)
    cm = confusion_matrix(ydev, pred)
    print(f"\n[1] lgb-proxy OOF balanced-acc {base_ba:.5f} | error rate {err.mean()*100:.2f}%")
    print("    confusion (rows=true, cols=pred) over", CLASSES)
    for i, c in enumerate(CLASSES):
        print(f"      {c:<7} " + "  ".join(f"{cm[i, j]:>7d}" for j in range(nc)))
    print("    per-true-class error rate: " + ", ".join(
        f"{c} {(1 - cm[i, i] / cm[i].sum()) * 100:.2f}%" for i, c in enumerate(CLASSES)))

    # full feature space, standardized, sampled. One-hot only LOW-card cats (≤15 levels) to avoid an
    # OOM blow-up — high-card crossed cats are already captured numerically by the per-class TE columns.
    cat_all = [c for c in Xdev.columns if str(Xdev[c].dtype) == "category"]
    cat = [c for c in cat_all if Xdev[c].nunique() <= 15]
    num = [c for c in Xdev.columns if c not in cat_all]
    Xn = Xdev[num].apply(pd.to_numeric, errors="coerce"); Xn = Xn.fillna(Xn.median())
    Xfull = Xn if not cat else pd.concat(
        [Xn, pd.get_dummies(Xdev[cat].astype("string").fillna("NA"), dummy_na=False).astype("float32")], axis=1)
    print(f"[space] {Xfull.shape[1]} dims ({len(num)} numeric + {len(cat)} low-card cats one-hot; "
          f"dropped {len(cat_all) - len(cat)} high-card cats — TE-encoded numerically)", flush=True)
    rng = np.random.default_rng(42)
    samp = rng.choice(len(ydev), size=min(120_000, len(ydev)), replace=False)

    def floor_and_oracle(cols, tag):
        Xs = StandardScaler().fit_transform(Xfull[cols].iloc[samp].to_numpy())
        ys = ydev[samp]; es = err[samp]
        nn = NearestNeighbors(n_neighbors=31, n_jobs=-1).fit(Xs)
        _, nbr = nn.kneighbors(Xs); nbr = nbr[:, 1:]            # drop self
        disagree = (ys[nbr] != ys[:, None]).mean(1)             # per-row neighbour disagreement
        # local ORACLE: predict each row by neighbours' TRUE majority label
        votes = np.zeros((len(ys), nc))
        for c in range(nc):
            votes[:, c] = (ys[nbr] == c).sum(1)
        oracle = votes.argmax(1)
        oracle_ba = balanced_accuracy_score(ys, oracle)
        print(f"\n[{tag}] kNN(k=30) full-space floor:")
        print(f"      neighbour label-disagreement: overall {disagree.mean()*100:.2f}% | "
              f"on ERROR rows {disagree[es].mean()*100:.2f}% | on CORRECT rows {disagree[~es].mean()*100:.2f}%")
        print(f"      LOCAL-ORACLE balanced-acc (neighbour true-majority) = {oracle_ba:.5f}  "
              f"(vs our {base_ba:.5f})")
        return oracle_ba

    o_noc = floor_and_oracle([c for c in Xfull.columns if c not in ("alpha", "delta")], "2/3/4 no-coords")
    o_c = floor_and_oracle(list(Xfull.columns), "5 with-coords")
    print(f"\n[coords] local-oracle Δ from adding alpha/delta: {o_c - o_noc:+.5f}  "
          f"({'coords carry signal' if o_c - o_noc > 0.001 else 'coords ~ noise (as expected for synthetic)'})")

    print(f"\n[verdict] if local-oracle ≈ our {base_ba:.4f} → at the feature-space floor (real ceiling). "
          f"if oracle ≫ ours → recoverable local signal we're missing.")
    result = {"id": config.get("id"), "kind": "diagnose", "synthetic": synthetic,
              "base_ba": round(float(base_ba), 5), "oracle_noc": round(float(o_noc), 5),
              "oracle_coords": round(float(o_c), 5), "err_rate": round(float(err.mean()), 5),
              "runtime_sec": round(time.time() - t0, 1)}
    out = _outdir(); (out / "result.json").write_text(json.dumps(result))
    print("\nRESULT_JSON", json.dumps(result))
    return result


def run_adv_validation(config: dict) -> dict:
    """Adversarial validation on the CURRENT rich FE (audit #1 — the gate).

    Our 'CV is trustworthy / ceiling is real' edifice rests on adv-AUC=0.499 measured at v2 on RAW BANDS,
    before every winning feature. This re-measures it on build_rich_features (ratios/colors/num->cat/KBins/
    crossed cats) — the space the models actually see. A binary GBDT tries to tell train from test under CV.
    AUC~0.5 = i.i.d., CV trustworthy for finals. AUC>>0.5 = a train/test shift; the top features NAME it."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    import numpy as np
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    t0 = time.time()
    train, test, synthetic = _load()
    Xtr, info, state = build_rich_features(train, fit=True)
    Xte, _, _ = build_rich_features(test, fit=False, state=state)
    common = [c for c in Xtr.columns if c in Xte.columns and c not in (ID, TARGET)]
    Xtr, Xte = Xtr[common].copy(), Xte[common].copy()
    print(f"[advval] rich FE: {len(common)} features | train={len(Xtr)} test={len(Xte)} synthetic={synthetic}", flush=True)

    cat = [c for c in common if str(Xtr[c].dtype) == "category"]
    for c in cat:                                  # unify category codes across train+test
        cats = pd.api.types.union_categoricals([Xtr[c], Xte[c]]).categories
        Xtr[c] = Xtr[c].cat.set_categories(cats).cat.codes
        Xte[c] = Xte[c].cat.set_categories(cats).cat.codes
    X = pd.concat([Xtr, Xte], ignore_index=True)
    yadv = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]

    aucs, imp = [], np.zeros(len(common))
    for i, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, yadv), 1):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                               subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1)
        m.fit(X.iloc[tr], yadv[tr])
        a = roc_auc_score(yadv[va], m.predict_proba(X.iloc[va])[:, 1])
        aucs.append(a); imp += m.feature_importances_ / 5
        print(f"  fold {i}: adv-AUC {a:.4f}", flush=True)
    auc = float(np.mean(aucs))
    order = np.argsort(imp)[::-1][:15]
    print(f"\n[advval] mean adv-AUC = {auc:.4f}  (0.50 = i.i.d. → CV trustworthy; >0.55 = train/test SHIFT)", flush=True)
    print("  top train/test-distinguishing features (potential shift/leak sources):", flush=True)
    for j in order:
        print(f"    {common[j]:40s} imp {imp[j]:8.1f}", flush=True)
    verdict = ("i.i.d. — CV trustworthy for finals" if auc <= 0.52 else
               "MILD shift — watch the top features" if auc <= 0.55 else
               "SHIFT — CV may mislead; regularize the named features")
    print(f"\n[verdict] {verdict}", flush=True)
    result = {"id": config.get("id"), "kind": "advval", "synthetic": synthetic,
              "adv_auc": round(auc, 4), "n_features": len(common),
              "top_features": [common[j] for j in order], "runtime_sec": round(time.time() - t0, 1)}
    out = _outdir(); (out / "result.json").write_text(json.dumps(result))
    print("\nRESULT_JSON", json.dumps(result), flush=True)
    return result


def run_pl(config: dict) -> dict:
    """Transductive pseudo-labeling (audit/domain lever #1). Stage 1: normal CV → test preds. Stage 2:
    hard-label ALL test rows (argmax), append to EACH train fold, re-fit; OOF scored on REAL rows only
    (pseudo rows NEVER enter OOF/CV — the #1 PL leak). Pre-registered kill: GALAXY recall must rise."""
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, recall_score
    t0 = time.time()
    train, test, synthetic = _load()
    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    ydev = np.asarray(ydev); spec = config["single"]; nf = int(config.get("n_folds", 5))

    def report(tag, oof):
        pred = oof.argmax(1)
        ba = balanced_accuracy_score(ydev, pred)
        rec = recall_score(ydev, pred, average=None, labels=list(range(len(CLASSES))))
        print(f"  [{tag}] OOF bal-acc {ba:.5f} | recall " +
              " ".join(f"{c}={r:.4f}" for c, r in zip(CLASSES, rec)), flush=True)
        return ba, rec

    print(f"[pl] stage 1: base {spec.get('name')} (no PL)", flush=True)
    oof1, hold1, test1 = single_oof(Xdev, ydev, Xhold, Xte, info, spec, nf)
    ba1, rec1 = report("stage1", oof1)

    pseudo_y = test1.argmax(1).astype(int)
    from collections import Counter
    print(f"[pl] stage 2: TRANSDUCTIVE PL — {len(pseudo_y)} test rows pseudo-labeled "
          f"{dict((CLASSES[k], v) for k, v in sorted(Counter(pseudo_y).items()))}", flush=True)
    oof2, hold2, test2 = single_oof(Xdev, ydev, Xhold, Xte, info, spec, nf, pseudo_y=pseudo_y)
    ba2, rec2 = report("stage2-PL", oof2)

    gal = CLASSES.index("GALAXY")
    d_gal, d_ba = rec2[gal] - rec1[gal], ba2 - ba1
    verdict = ("PL HELPS — GALAXY recall up, pursue" if d_gal > 0.0005 and d_ba >= -0.0001 else
               "PL FLAT/HARMFUL — GAL<->STAR confirmed morphology-capped" if d_gal <= 0.0005 else "MIXED")
    print(f"\n[pl-verdict] ΔGALAXY-recall {d_gal:+.5f}  Δbal-acc {d_ba:+.5f}  → {verdict}", flush=True)

    out = _outdir(); name = spec.get("name", "leg")
    np.save(out / f"oof_{name}.npy", oof2); np.save(out / f"hold_{name}.npy", hold2)
    np.save(out / f"test_{name}.npy", test2)
    np.save(out / "y_dev.npy", ydev); np.save(out / "y_hold.npy", np.asarray(yhold))
    pd.DataFrame({ID: test[ID]}).to_csv(out / "test_ids.csv", index=False)
    result = {"id": config.get("id"), "kind": "pl", "model": name, "synthetic": synthetic,
              "ba_stage1": round(float(ba1), 5), "ba_stage2_pl": round(float(ba2), 5),
              "gal_recall_stage1": round(float(rec1[gal]), 5), "gal_recall_stage2_pl": round(float(rec2[gal]), 5),
              "delta_gal_recall": round(float(d_gal), 5), "verdict": verdict,
              "runtime_sec": round(time.time() - t0, 1)}
    (out / "result.json").write_text(json.dumps(result))
    print("\nRESULT_JSON", json.dumps(result), flush=True)
    return result


def run_priors(config: dict) -> dict:
    """Original-SDSS17 BIN-LEVEL class-rate priors (domain lever #4) with the DECISIVE kill-test.
    P(class | g-r bin, redshift bin) from 100K REAL external labels (a higher-fidelity estimate than any
    synthetic row carries) — NOT a row-join (v2 proved that dead). KILL-TEST: do EXTERNAL-label priors beat
    the SAME binning target-encoded from our OWN train labels? If external <= own, it is byte-redundant with
    redshift (v3 lesson) -> kill. 3 arms on identical folds: baseline / +external-prior / +own-prior."""
    import sys, glob
    sys.stdout.reconfigure(line_buffering=True)
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score
    t0 = time.time()
    train, test, synthetic = _load()
    orig = None
    for p in glob.glob("/kaggle/input/*/star_classification.csv") + glob.glob("/kaggle/input/**/star_classification.csv", recursive=True):
        orig = pd.read_csv(p); print(f"[priors] external SDSS17: {p} {orig.shape}", flush=True); break
    if orig is None:
        orig = data.load_original()
    if orig is None:
        raise RuntimeError("no star_classification.csv — pass dataset_sources=['fedesoriano/stellar-classification-dataset-sdss17']")

    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    ydev = np.asarray(ydev); yhold = np.asarray(yhold); nf = int(config.get("n_folds", 5))

    def cellcol(df):  # the (g-r bin x redshift bin) cell — Deotte's dominant cross
        gr = (df["g"] - df["r"]).clip(-2, 4); z = df["redshift"].clip(-0.01, 1.0)
        return (np.floor(gr * 2).astype("int32").astype(str) + "_" + np.floor(z * 10).astype("int32").astype(str))

    # external prior P(class|cell) from the 100K REAL rows, smoothed toward the global external prior
    oy = orig["class"].map(CLS2INT).to_numpy()
    od = pd.DataFrame({"cell": cellcol(orig).to_numpy(), "y": oy})
    pg = np.bincount(oy, minlength=len(CLASSES)).astype("float64"); pg /= pg.sum()
    ct = od.groupby("cell")["y"].value_counts().unstack(fill_value=0).reindex(columns=range(len(CLASSES)), fill_value=0)
    sm = 20.0
    P = (ct.to_numpy() + sm * pg) / (ct.to_numpy().sum(1, keepdims=True) + sm)
    emap = {c: P[i] for i, c in enumerate(ct.index)}
    extf = lambda df: np.array([emap.get(x, pg) for x in cellcol(df).to_numpy()], dtype="float32")
    edev, ehold, ete = extf(Xdev), extf(Xhold), extf(Xte)
    print(f"[priors] external cells: {len(emap)} | train cell coverage "
          f"{np.mean([x in emap for x in cellcol(Xdev).to_numpy()[:50000]]):.3f}", flush=True)

    def arm(kind_):
        X, Xh, Xt, inf = Xdev.copy(), Xhold.copy(), Xte.copy(), dict(info)
        if kind_ == "ext":
            for j, cl in enumerate(CLASSES):
                X[f"extp_{cl}"], Xh[f"extp_{cl}"], Xt[f"extp_{cl}"] = edev[:, j], ehold[:, j], ete[:, j]
            inf["num_cols"] = info["num_cols"] + [f"extp_{cl}" for cl in CLASSES]
        elif kind_ == "own":   # same cell, target-encoded per-fold OOF from OUR labels (leak-safe via _fold_te)
            X["gr_z_cell"] = cellcol(Xdev).astype("category")
            Xh["gr_z_cell"] = cellcol(Xhold).astype("category")
            Xt["gr_z_cell"] = cellcol(Xte).astype("category")
            inf["combo_cols"] = info["combo_cols"] + ["gr_z_cell"]
        oof, hold, _ = single_oof(X, ydev, Xh, Xt, inf, {"name": "lgb", "type": "lgb"}, nf)
        from src.metrics import tune_class_weights, weighted_predict, competition_score
        w, oof_ba = tune_class_weights(oof, ydev, labels=list(range(len(CLASSES))))
        hold_ba = competition_score(np.asarray(CLASSES)[yhold], weighted_predict(hold, w, labels=CLASSES))
        print(f"  [{kind_:8}] OOF {oof_ba:.5f} | holdout {hold_ba:.5f}", flush=True)
        return round(float(oof_ba), 5), round(float(hold_ba), 5)

    print("[priors] 3-arm ablation (identical folds):", flush=True)
    base = arm("base"); ext = arm("ext"); own = arm("own")
    d_ext, d_own = ext[1] - base[1], own[1] - base[1]
    ext_beats_own = ext[1] > own[1] + 0.0001
    verdict = ("EXTERNAL PRIOR ADDS NEW SIGNAL — pursue (beats own-label binning)" if d_ext > 0.0002 and ext_beats_own else
               "REDUNDANT — external prior ~= own-label binning ~= redshift; kill" if not ext_beats_own else
               "MARGINAL — external edges own but near noise")
    print(f"\n[priors-verdict] Δext {d_ext:+.5f}  Δown {d_own:+.5f}  ext-beats-own={ext_beats_own}  → {verdict}", flush=True)
    result = {"id": config.get("id"), "kind": "priors", "synthetic": synthetic,
              "base_holdout": base[1], "ext_holdout": ext[1], "own_holdout": own[1],
              "delta_ext": round(d_ext, 5), "delta_own": round(d_own, 5), "verdict": verdict,
              "runtime_sec": round(time.time() - t0, 1)}
    out = _outdir(); (out / "result.json").write_text(json.dumps(result))
    print("\nRESULT_JSON", json.dumps(result), flush=True)
    return result


def run(config: dict) -> dict:
    if config.get("kind") == "priors":
        return run_priors(config)
    if config.get("kind") == "pl":
        return run_pl(config)
    if config.get("kind") == "advval":
        return run_adv_validation(config)
    if config.get("kind") == "diagnose":
        return run_diagnose(config)
    if config.get("single"):
        return run_single(config)
    t0 = time.time()
    train, test, synthetic = _load()
    print(f"[run] id={config.get('id')} kind={config.get('kind')} train={train.shape}")

    Xdev, ydev, Xhold, yhold, Xte, info = _prep(train, test)
    Xrich = Xdev  # (n_features reported below from the dev frame)
    yhold_names = INT2CLS[yhold]

    extra_nn = config.get("extra_nn") or []
    R = race_oof(Xdev, ydev, Xhold, Xte, info,
                 n_folds=int(config.get("n_folds", 5)),
                 cfg=config.get("rm_cfg", {}),
                 with_cat=bool(config.get("with_cat", False)),
                 with_tabm=bool(config.get("with_tabm", False)),
                 tabm_cfg=config.get("tabm_cfg"),
                 rm_seeds=config.get("rm_seeds"),
                 extra_nn=extra_nn)

    legs = (["lgb", "rm"] + (["tabm"] if config.get("with_tabm") else [])
            + [e["name"] for e in extra_nn] + (["cat"] if config.get("with_cat") else []))
    result = {"id": config.get("id"), "kind": config.get("kind", "realmlp_race"),
              "config": config, "synthetic": synthetic, "n_features": int(Xrich.shape[1]), "legs": legs}
    for name in legs:
        w, s_oof = tune_class_weights(R[f"{name}_oof"], ydev, labels=INTS)
        s_hold = competition_score(yhold_names, weighted_predict(R[f"{name}_hold"], w, labels=CLASSES))
        result[f"{name}_oof"] = round(s_oof, 5)
        result[f"{name}_holdout"] = round(s_hold, 5)
        print(f"[{name}] OOF {s_oof:.5f} | holdout {s_hold:.5f}")

    # CV'd seed-bagged log-odds stacker (the v16-overfit fix) — the conveyor's headline number
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    if len(legs) >= 2:
        oof_d = {l: R[f"{l}_oof"] for l in legs}
        stack_oof, stacked, stack_cv = cv_logreg_stack(
            oof_d, ydev, {"hold": {l: R[f"{l}_hold"] for l in legs}, "test": {l: R[f"{l}_test"] for l in legs}})
        sw, _ = tune_class_weights(stack_oof, ydev, labels=INTS)
        stack_hold = competition_score(yhold_names, weighted_predict(stacked["hold"], sw, labels=CLASSES))
        result["stack_cv"] = round(stack_cv, 5)
        result["stack_holdout"] = round(stack_hold, 5)
        print(f"[stack] CV {stack_cv:.5f} | holdout {stack_hold:.5f}  (legs={legs})")
        # write the stacked submission + persist OOFs (for re-blends without re-running)
        sub = pd.DataFrame({ID: test[ID], TARGET: weighted_predict(stacked["test"], sw, labels=CLASSES)})
        sub.to_csv(out_dir / "submission.csv", index=False)
        for l in legs:
            np.save(out_dir / f"oof_{l}.npy", R[f"{l}_oof"]); np.save(out_dir / f"test_{l}.npy", R[f"{l}_test"])

    result["runtime_sec"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(result))
    print("RESULT_JSON", json.dumps(result))
    return result


if __name__ == "__main__":
    cfg = json.loads(Path(sys.argv[1]).read_text()) if len(sys.argv) > 1 else {"id": "smoke", "n_folds": 3}
    run(cfg)
