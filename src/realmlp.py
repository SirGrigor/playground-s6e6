"""RealMLP — a proper tabular NN that BREAKS the GBDT 0.9664 ceiling (2026-06-03).

Our v7 dismissed RealMLP after ONE under-tuned pytabkit run (0.9586). That was the campaign's
biggest error: a public RealMLP (Vladimir @yekenot, Codex-tuned by Deotte) hits CV 0.9688 / LB
0.9695 — +0.0025 over our best GBDT from the SAME columns. The 0.9664 "ceiling" (and v9's
kNN-Bayes "genuine overlap") was a GBDT-ARCHITECTURE artifact: a Bayes-error estimate is only as
good as the model class you measure it with, and axis-aligned trees can't carve the curved
redshift/color boundary a periodic-embedding MLP can.

This is OUR reimplementation of the load-bearing recipe (architecture + training are public ideas;
doctrine-OK to rebuild — we are NOT copying notebook outputs into a blend, which stays barred until
the final 2 subs). The pieces that matter, none of which v7 had:
- PBLD periodic numerical embeddings (cos-basis) — fine, non-monotonic boundaries on redshift/colors.
- n_ens internal bag (vectorised) — each fold trains N MLPs at once and averages.
- balanced-softmax loss via `loss_prior_power` — METRIC-AWARE (optimizes balanced accuracy directly),
  not class_weight (class_weight_power=0.0; sklearn weights were found to HURT).
- EMA weights, per-parameter-group LRs, flat-cos schedule, label smoothing, decaying dropout.

`realmlp_oof(Xdev, ydev, Xhold, Xte, ...)` mirrors lgb_oof so it drops into the existing notebooks,
dashboard and diary. Categoricals (pandas `category`) are code-converted internally; numeric NaN is
median-imputed (trees handle NaN natively — an NN cannot). torch is imported lazily so the module
loads without it.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CLASSES, MODEL_SEED, N_FOLDS
from .cv import stratified_folds

# Best single-model recipe (Deotte EXP-549 on S6E6). cfg-overridable for smoke tests.
DEFAULT_CFG = {
    "n_ens": 8, "embed_dim": 7, "onehot_thresh": 10, "hidden_dims": [512, 512, 512],
    "dropout": 0.041, "p_drop_sched": "expm4t", "activation": "gelu", "add_front_scale": True,
    "pbld_hidden_dim": 16, "pbld_out_dim": 5, "pbld_freq_scale": 2.33, "pbld_activation": "prelu",
    "pbld_lr_factor": 0.115,
    "lr": 0.01, "mom": 0.9, "sq_mom": 0.98, "lr_sched": "flat_cos", "flat_ratio": 0.175,
    "first_layer_lr_factor": 0.5, "first_layer_wd_factor": 0.1, "lr_scale_mult": 10.0,
    "lr_bias_mult": 0.1, "weight_decay": 0.013, "wd_scale_mult": 0.1, "wd_bias_mult": 0.5,
    "grad_clip": 1.0, "class_weight_power": 0.0, "loss_prior_power": 1.075,
    "ls_eps": 0.04, "ls_eps_sched": "cos", "tfms": ["median_center", "robust_scale"],
    "epochs": 6, "train_bs": 256, "eval_bs": 10240, "ema_decay": 0.99775, "seed": MODEL_SEED,
    "arch": "realmlp",   # "realmlp" (NTP independent-weight ensemble) | "tabm" (shared weight + rank-1 scalings)
}

# TabM (Don @donmarch14 recipe): parameter-efficient ensemble → n_ens=32 cheap, wider, dropout=0
# (the rank-1 scalings regularize), SiLU. Architecturally DECORRELATED from RealMLP — the diversity leg.
TABM_CFG = {**DEFAULT_CFG, "arch": "tabm", "n_ens": 32, "hidden_dims": [1024, 512, 512],
            "dropout": 0.0, "p_drop_sched": "constant", "activation": "silu",
            "pbld_freq_scale": 5.0, "pbld_lr_factor": 0.093}

# FT-Transformer (attention paradigm — strong AND decorrelated from MLP/tree legs). Architecture in
# src/fttransformer.py; reuses _fit_one (balanced-softmax loss + EMA). n_ens=1 (single attention model).
# Gemini-drafted config, Claude-tuned: lr 1e-4 + cosine (transformers need it), no weight-decay on biases.
FT_CFG = {**DEFAULT_CFG, "arch": "ft", "n_ens": 1, "lr": 1e-4, "weight_decay": 1e-5,
          "lr_sched": "cos", "flat_ratio": 0.0, "epochs": 15, "train_bs": 512, "ls_eps": 0.0,
          "lr_bias_mult": 1.0, "wd_bias_mult": 0.0,
          "d_token": 192, "n_blocks": 3, "n_heads": 8, "attention_dropout": 0.2,
          "ffn_dropout": 0.1, "ffn_factor": 4 / 3}


def _act(name):
    import torch.nn as nn
    return {"gelu": nn.GELU, "prelu": nn.PReLU, "relu": nn.ReLU, "silu": nn.SiLU}[name]


def _schedule(init, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init
    if sched == "cos":
        return init * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init * (math.cos(math.pi * t) + 1) / 2
    if sched == "expm4t":
        return init * math.exp(-4 * progress)
    raise ValueError(f"unknown schedule {sched!r}")


def _build_modules():
    """Define the torch nn.Modules lazily (so the file imports without torch)."""
    import torch
    import torch.nn as nn

    class PBLDEmbedding(nn.Module):
        """Periodic basis (cos) numerical embedding — the RealMLP signature layer."""
        def __init__(self, n_ens, n_features, hidden_dim, out_dim, freq_scale, activation):
            super().__init__()
            self.out_dim = out_dim
            self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
            self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
            self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
            self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
            self.act = activation()
            nn.init.uniform_(self.b1, -math.pi, math.pi)

        def forward(self, x):
            periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
            transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
            return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)

    class CategoricalLayer(nn.Module):
        def __init__(self, n_ens, cat_dims, embed_dim, onehot_thresh):
            super().__init__()
            self.n_ens, self.cat_dims = n_ens, cat_dims
            self.onehot_features, self.embed_layers, self._embed_idx = [], nn.ModuleList(), []
            for i, dim in enumerate(cat_dims):
                if dim <= onehot_thresh:
                    self.onehot_features.append(i)
                else:
                    self.embed_layers.append(nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)]))
                    self._embed_idx.append(i)

        def forward(self, x):
            b, n_ens, _ = x.shape
            feats = []
            if self.onehot_features:
                oh = x[:, :, self.onehot_features]
                dims = [self.cat_dims[i] for i in self.onehot_features]
                enc = torch.zeros(b, n_ens, sum(dims), device=x.device)
                start = 0
                for idx, dim in enumerate(dims):
                    enc.scatter_(2, oh[:, :, idx:idx + 1].long() + start, 1.0)
                    start += dim
                feats.append(enc)
            for emb_list, fi in zip(self.embed_layers, self._embed_idx):
                e = [emb_list[m](x[:, m, fi:fi + 1].long()) for m in range(self.n_ens)]
                feats.append(torch.cat(e, dim=1))
            return torch.cat(feats, dim=2) if feats else torch.zeros(b, n_ens, 0, device=x.device)

    class ScalingLayer(nn.Module):
        def __init__(self, n_ens, n_features):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(n_ens, n_features))

        def forward(self, x):
            return x * self.scale[None, :, :]

    class NTPLinear(nn.Module):
        def __init__(self, n_ens, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
            self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

        def forward(self, x):
            x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
            return x + self.bias if self.bias is not None else x

    class TabMLinear(nn.Module):
        """TabM: parameter-efficient ensemble — ONE shared weight + per-member rank-1 scalings.
        Decorrelates from NTPLinear (independent weights) → the diversity leg."""
        def __init__(self, n_ens, in_features, out_features, bias=True):
            super().__init__()
            self.in_features = in_features
            self.weight = nn.Parameter(torch.randn(in_features, out_features))      # SHARED
            self.scale_in = nn.Parameter(torch.ones(n_ens, in_features))            # rank-1 per member
            self.scale_out = nn.Parameter(torch.ones(n_ens, out_features))
            self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
            self.bias_ens = nn.Parameter(torch.zeros(n_ens, out_features)) if bias else None

        def forward(self, x):
            x = x * self.scale_in.unsqueeze(0)
            x = torch.einsum("bki,io->bko", x, self.weight) / math.sqrt(self.in_features)
            x = x * self.scale_out.unsqueeze(0)
            if self.bias is not None:
                x = x + self.bias.view(1, 1, -1) + self.bias_ens.unsqueeze(0)
            return x

    class RealMLPNet(nn.Module):
        def __init__(self, output_dim, cat_dims, n_numerical, cfg):
            super().__init__()
            self.n_ens = cfg["n_ens"]
            self.cate = CategoricalLayer(self.n_ens, cat_dims, cfg["embed_dim"], cfg["onehot_thresh"])
            self.num_embed = PBLDEmbedding(self.n_ens, n_numerical, cfg["pbld_hidden_dim"],
                                           cfg["pbld_out_dim"], cfg["pbld_freq_scale"], _act(cfg["pbld_activation"]))
            num_dim = n_numerical * cfg["pbld_out_dim"]
            cat_dim = sum(c if c <= cfg["onehot_thresh"] else cfg["embed_dim"] for c in cat_dims)
            total = num_dim + cat_dim
            act = _act(cfg["activation"])
            Linear = TabMLinear if cfg.get("arch") == "tabm" else NTPLinear
            layers, self._drops = [], []
            if cfg["add_front_scale"]:
                layers.append(ScalingLayer(self.n_ens, total))
            in_dim = total
            for i, h in enumerate(cfg["hidden_dims"]):
                lin = Linear(self.n_ens, in_dim, h)
                if i == 0:
                    self.first_linear = lin
                drop = nn.Dropout(cfg["dropout"])
                self._drops.append(drop)
                layers += [lin, act(), drop]
                in_dim = h
            self.hidden = nn.Sequential(*layers)
            self.output_layer = Linear(self.n_ens, in_dim, output_dim)

        def forward(self, x_num, x_cat):
            x_num = self.num_embed(x_num.unsqueeze(1).expand(-1, self.n_ens, -1))
            x_cat = self.cate(x_cat.unsqueeze(1).expand(-1, self.n_ens, -1))
            x = self.hidden(torch.cat([x_num, x_cat], dim=2))
            return torch.softmax(self.output_layer(x), dim=2)

    return RealMLPNet


def _param_groups(model, p):
    first_id = id(model.first_linear.weight) if hasattr(model, "first_linear") else None
    scale, pbld, first_w, other_w, bias = [], [], [], [], []
    for name, par in model.named_parameters():
        if "num_embed" in name:
            pbld.append(par)
        elif "scale" in name:
            scale.append(par)
        elif first_id is not None and id(par) == first_id:
            first_w.append(par)
        elif "bias" in name:
            bias.append(par)
        else:
            other_w.append(par)
    lr, wd = p["lr"], p["weight_decay"]
    return [
        {"params": scale, "lr": lr * p["lr_scale_mult"], "weight_decay": wd * p["wd_scale_mult"]},
        {"params": pbld, "lr": lr * p["pbld_lr_factor"], "weight_decay": wd},
        {"params": first_w, "lr": lr * p["first_layer_lr_factor"], "weight_decay": wd * p["first_layer_wd_factor"]},
        {"params": other_w, "lr": lr, "weight_decay": wd},
        {"params": bias, "lr": lr * p["lr_bias_mult"], "weight_decay": wd * p["wd_bias_mult"]},
    ]


def _smooth_ce(y_true, y_pred, ls, prior_mult, w=None):
    """Balanced-softmax (prior_mult) + label-smoothed cross-entropy. The metric-aware loss.

    `w` (optional per-row weights, same length as the flattened batch) → weighted mean instead of
    plain mean. Used by concat-augmentation (ladder v19) to down-weight appended original rows."""
    import torch
    n = y_pred.size(1)
    if prior_mult is not None:
        y_pred = y_pred * prior_mult[None, :]
        y_pred = y_pred / y_pred.sum(dim=1, keepdim=True).clamp_min(1e-15)
    y_smooth = torch.full_like(y_pred, ls / n)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n)
    per_row = -(y_smooth * torch.log(y_pred.clamp(1e-15, 1))).sum(dim=1)
    if w is not None:
        return (per_row * w).sum() / w.sum().clamp_min(1e-15)
    return per_row.mean()


class _Preprocessor:
    """median_center + robust_scale (IQR) on numerics. Fit on fold-train only."""
    def __init__(self, tfms):
        self.tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip")]

    def fit(self, X):
        self.median = np.median(X, axis=0)
        q = np.quantile(X, 0.75, axis=0) - np.quantile(X, 0.25, axis=0)
        bad = q == 0.0
        q[bad] = 0.5 * (X.max(axis=0)[bad] - X.min(axis=0)[bad])
        self.iqr = 1.0 / (q + 1e-30)
        self.iqr[q == 0.0] = 0.0
        return self

    def transform(self, X):
        X = X.copy().astype(np.float32)
        for t in self.tfms:
            if t == "median_center":
                X -= self.median[None, :]
            elif t == "robust_scale":
                X *= self.iqr[None, :]
            elif t == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
        return X


def _fit_one(Xn_tr, Xc_tr, y_tr, Xn_va, Xc_va, y_va, cat_dims, cfg, sw_tr=None, prior_counts=None):
    """Train a single RealMLP (n_ens internal bag), EMA, return best-val proba predictor closure.

    sw_tr: optional per-train-row sample weights (concat-augmentation, v19) → weighted loss.
    prior_counts: optional class counts driving the balanced-softmax prior_mult. When augmenting,
    pass the COMPETITION fold-train counts so the metric-aware loss still targets the competition
    class balance (not the augmented mix). Defaults to bincount(y_tr) (unchanged behaviour)."""
    import torch
    from sklearn.metrics import balanced_accuracy_score

    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = int(max(y_tr.max(), y_va.max())) + 1

    pre = _Preprocessor(cfg["tfms"]).fit(Xn_tr)
    Xn_tr, Xn_va = pre.transform(Xn_tr), pre.transform(Xn_va)
    swt = torch.as_tensor(np.asarray(sw_tr), dtype=torch.float32, device=dev) if sw_tr is not None else None

    if cfg.get("arch") == "ft":
        from .fttransformer import build_fttransformer
        Net = build_fttransformer()
    else:
        Net = _build_modules()
    model = Net(n_classes, cat_dims, Xn_tr.shape[1], cfg).to(dev)

    # balanced-softmax prior multipliers (the metric-aware term)
    prior_mult = None
    if cfg["loss_prior_power"] != 0.0:
        counts = (np.asarray(prior_counts, dtype="float64") if prior_counts is not None
                  else np.bincount(y_tr, minlength=n_classes).astype("float64"))
        counts = counts / np.exp(np.log(counts).mean())
        prior_mult = torch.as_tensor(np.power(counts, cfg["loss_prior_power"]), dtype=torch.float32, device=dev)

    groups = _param_groups(model, cfg)
    for g in groups:
        g["lr_base"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(cfg["mom"], cfg["sq_mom"]))

    Xtn = torch.as_tensor(Xn_tr, dtype=torch.float32, device=dev)
    Xtc = torch.as_tensor(Xc_tr, dtype=torch.long, device=dev)
    ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
    Xvn = torch.as_tensor(Xn_va, dtype=torch.float32, device=dev)
    Xvc = torch.as_tensor(Xc_va, dtype=torch.long, device=dev)

    n_ens, bs, ebs, epochs = cfg["n_ens"], cfg["train_bs"], cfg["eval_bs"], cfg["epochs"]
    total_steps = max(1, epochs * len(y_tr))
    order = np.arange(len(y_tr))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()} if cfg["ema_decay"] > 0 else None
    best_score, best_state = -np.inf, None

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(order)
        for start in range(0, len(y_tr), bs):
            progress = (epoch * len(y_tr) + start) / total_steps
            idx = order[start:start + bs]
            for g in opt.param_groups:
                g["lr"] = _schedule(g["lr_base"], progress, cfg["lr_sched"], cfg["flat_ratio"])
            opt.zero_grad()
            pred = model(Xtn[idx], Xtc[idx])  # (b, n_ens, C)
            ls_val = _schedule(cfg["ls_eps"], progress, cfg["ls_eps_sched"], cfg["flat_ratio"])
            drop_val = _schedule(cfg["dropout"], progress, cfg["p_drop_sched"], cfg["flat_ratio"])
            for dm in model._drops:
                dm.p = drop_val
            wb = swt[idx].repeat_interleave(n_ens) if swt is not None else None
            loss = _smooth_ce(ytt[idx].repeat_interleave(n_ens), pred.reshape(-1, n_classes), ls_val, prior_mult, w=wb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if torch.is_floating_point(v):
                            ema[k].mul_(cfg["ema_decay"]).add_(v.detach(), alpha=1.0 - cfg["ema_decay"])
                        else:
                            ema[k].copy_(v)

        # validate on EMA weights
        model.eval()
        live = None
        if ema is not None:
            live = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema, strict=True)
        with torch.no_grad():
            vp = np.concatenate([model(Xvn[s:s + ebs], Xvc[s:s + ebs]).mean(dim=1).cpu().numpy()
                                 for s in range(0, len(y_va), ebs)], axis=0)
        if live is not None:
            model.load_state_dict(live, strict=True)
        score = balanced_accuracy_score(y_va, np.argmax(vp, axis=1))
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in (ema or model.state_dict()).items()}

    model.load_state_dict(best_state, strict=True)
    model.eval()

    def predict(Xn, Xc):
        Xn = pre.transform(Xn)
        xn = torch.as_tensor(Xn, dtype=torch.float32, device=dev)
        xc = torch.as_tensor(Xc, dtype=torch.long, device=dev)
        with torch.no_grad():
            return np.concatenate([model(xn[s:s + ebs], xc[s:s + ebs]).mean(dim=1).cpu().numpy()
                                   for s in range(0, len(Xn), ebs)], axis=0)

    return predict, best_score


def _encode(Xdev, Xhold, Xte):
    """Split into numeric (NaN preserved) + categorical (aligned int codes, NaN->0). Returns arrays + cat_dims."""
    cat_cols = [c for c in Xdev.columns if str(Xdev[c].dtype) == "category"]
    num_cols = [c for c in Xdev.columns if c not in cat_cols]

    def num_arr(X):
        return X[num_cols].astype("float32").to_numpy() if num_cols else np.zeros((len(X), 0), "float32")

    cat_dims = []
    cat_dev, cat_hold, cat_te = [], [], []
    for c in cat_cols:
        cats = pd.Categorical(Xdev[c]).categories
        def code(X):
            return (pd.Categorical(X[c], categories=cats).codes.astype("int64") + 1)  # NaN(-1)->0
        cat_dev.append(code(Xdev)); cat_hold.append(code(Xhold)); cat_te.append(code(Xte))
        cat_dims.append(len(cats) + 1)
    stack = lambda cols, n: np.stack(cols, axis=1) if cols else np.zeros((n, 0), "int64")
    return (num_cols, cat_cols, cat_dims,
            num_arr(Xdev), stack(cat_dev, len(Xdev)),
            num_arr(Xhold), stack(cat_hold, len(Xhold)),
            num_arr(Xte), stack(cat_te, len(Xte)))


def realmlp_oof(Xdev: pd.DataFrame, ydev, Xhold: pd.DataFrame, Xte: pd.DataFrame,
                n_folds: int = N_FOLDS, cfg: dict | None = None, on_fold=None):
    """StratifiedKFold OOF on dev; fold-averaged proba for holdout + test. Mirrors lgb_oof.

    Returns (oof_proba, hold_proba, test_proba, fold_val_indices). Numeric NaN is median-imputed
    per fold-train (an NN cannot ingest NaN); categoricals become aligned int codes.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    ydev = np.asarray(ydev)
    nc = len(CLASSES)
    (_, _, cat_dims, Xn_dev, Xc_dev, Xn_hold, Xc_hold, Xn_te, Xc_te) = _encode(Xdev, Xhold, Xte)

    oof = np.zeros((len(Xdev), nc), "float32")
    hold = np.zeros((len(Xhold), nc), "float32")
    test = np.zeros((len(Xte), nc), "float32")
    fold_va = []
    folds = list(stratified_folds(ydev, n_folds))
    for i, (tr, va) in enumerate(folds, 1):
        if on_fold:
            on_fold(i, n_folds)
        # median-impute numerics on fold-train stats
        med = np.nanmedian(Xn_dev[tr], axis=0) if Xn_dev.shape[1] else np.zeros(0)
        imp = lambda A: np.where(np.isnan(A), med[None, :], A).astype("float32") if A.shape[1] else A
        predict, _ = _fit_one(imp(Xn_dev[tr]), Xc_dev[tr], ydev[tr],
                              imp(Xn_dev[va]), Xc_dev[va], ydev[va], cat_dims,
                              {**cfg, "seed": cfg["seed"] + i})
        oof[va] = predict(imp(Xn_dev[va]), Xc_dev[va])
        hold += predict(imp(Xn_hold), Xc_hold) / n_folds
        test += predict(imp(Xn_te), Xc_te) / n_folds
        fold_va.append(va)
    return oof, hold, test, fold_va


def _encode_frames(frames, cat_cols, num_cols):
    """Encode a LIST of frames with a SHARED category vocabulary (so codes align across them).

    cats → aligned int codes (NaN→0); numerics → float, NaN median-imputed from frames[0] (the
    train frame). Returns (num_arrays, cat_arrays, cat_dims), parallel to `frames`."""
    per_codes = [[] for _ in frames]
    cat_dims = []
    for c in cat_cols:
        cats = pd.Index(sorted({v for f in frames for v in pd.Series(f[c]).dropna().unique()}))
        for fi, f in enumerate(frames):
            per_codes[fi].append(pd.Categorical(f[c], categories=cats).codes.astype("int64") + 1)
        cat_dims.append(len(cats) + 1)
    cat_arrays = [np.stack(cc, axis=1) if cc else np.zeros((len(frames[i]), 0), "int64")
                  for i, cc in enumerate(per_codes)]
    num_arrays = [f[num_cols].astype("float32").to_numpy() if num_cols else np.zeros((len(f), 0), "float32")
                  for f in frames]
    if num_cols:
        med = np.nanmedian(num_arrays[0], axis=0)
        num_arrays = [np.where(np.isnan(A), med[None, :], A).astype("float32") for A in num_arrays]
    return num_arrays, cat_arrays, cat_dims


def realmlp_fit_predict(X_tr, y_tr, X_val, y_val, evals: dict, cfg: dict | None = None,
                        sample_weight=None, prior_counts=None):
    """Train ONE RealMLP (n_ens internal bag); X_val drives best-epoch; predict X_val + each frame
    in `evals`. Returns (val_proba, {name: proba}). For per-fold harnesses that inject fold-specific
    features (e.g. target encoding) before training — the OOF folding is the caller's job.

    sample_weight: optional per-train-row weights (concat-augmentation, v19). prior_counts: optional
    class counts for the balanced-softmax prior (pass competition-only counts when augmenting)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype) == "category"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]
    keys = list(evals.keys())
    nums, cats, cat_dims = _encode_frames([X_tr, X_val] + [evals[k] for k in keys], cat_cols, num_cols)
    predict, _ = _fit_one(nums[0], cats[0], np.asarray(y_tr), nums[1], cats[1], np.asarray(y_val), cat_dims,
                          cfg, sw_tr=sample_weight, prior_counts=prior_counts)
    return predict(nums[1], cats[1]), {k: predict(nums[2 + i], cats[2 + i]) for i, k in enumerate(keys)}
