"""Diary — read views over experiments.jsonl.

Asymmetric interface:
- Claude reads JSONL programmatically.
- Human reads markdown rendered by `render_all()` into docs/diary.md.
- Both can use the CLI: `python -m src.diary <command> [args]`.

Metric-agnostic (S6E6): scores are `config.METRIC` (accuracy by default).

Commands::

    python -m src.diary timeline             # table of all experiments
    python -m src.diary compare v_a v_b      # diff between two versions
    python -m src.diary report v_n           # per-version markdown
    python -m src.diary regressions          # flagged silent-regression versions
    python -m src.diary flag v_n "note"      # add human note to v_n
    python -m src.diary render               # regenerate docs/diary.md + per-version files
"""
from __future__ import annotations

import sys

from .config import DOCS, METRIC
from .observer import _load_jsonl, add_note

M = METRIC  # short label for headers


def _entries() -> list[dict]:
    return _load_jsonl()


def _by_version(version: str) -> dict | None:
    for e in _entries():
        if e.get("version") == version:
            return e
    return None


# --------------------------------------------------------------------------- views
def timeline() -> str:
    rows = _entries()
    if not rows:
        return "(no experiments yet — run an experiment via src.observer.Experiment)\n"
    lines = []
    lines.append(
        f"{'ver':<6} {'parent':<8} {'pred':>9} {'actual':>9} {'hold':>9} "
        f"{'oof':>9} {'gap':>7}  hypothesis  [flags]"
    )
    lines.append("-" * 110)
    for e in rows:
        ver = str(e.get("version", "?"))
        par = str(e.get("parent") or "—")
        pred = e.get("predicted_delta")
        act = e.get("actual_delta")
        hold = e.get("holdout_score")
        oof = e.get("oof_score_mean")
        gap = (hold - oof) if (hold is not None and oof is not None) else None
        hyp = (e.get("hypothesis") or "")
        if len(hyp) > 50:
            hyp = hyp[:47] + "..."
        flags = e.get("flags") or []
        flag_str = "  ⚠ " + ", ".join(s.split("(")[0] for s in flags) if flags else ""
        lines.append(
            f"{ver:<6} {par:<8} "
            f"{(f'{pred:+.4f}' if pred is not None else '—'):>9} "
            f"{(f'{act:+.4f}' if act is not None else '—'):>9} "
            f"{(f'{hold:.5f}' if hold is not None else '—'):>9} "
            f"{(f'{oof:.5f}' if oof is not None else '—'):>9} "
            f"{(f'{gap:+.4f}' if gap is not None else '—'):>7}  "
            f"{hyp}{flag_str}"
        )
    return "\n".join(lines) + "\n"


def compare(v_a: str, v_b: str) -> str:
    ea = _by_version(v_a)
    eb = _by_version(v_b)
    if not ea:
        return f"unknown version: {v_a}\n"
    if not eb:
        return f"unknown version: {v_b}\n"
    lines = [f"# compare {v_a}  →  {v_b}"]

    a_feats = set(ea.get("feature_changes") or [])
    b_feats = set(eb.get("feature_changes") or [])
    if (b_feats - a_feats) or (a_feats - b_feats):
        lines.append("\n## Feature changes")
        for f in sorted(b_feats - a_feats):
            lines.append(f"  only in {v_b}: {f}")
        for f in sorted(a_feats - b_feats):
            lines.append(f"  only in {v_a}: {f}")
    else:
        lines.append("\n## Feature changes: (none)")

    a_pipe = set(ea.get("pipeline_changes") or [])
    b_pipe = set(eb.get("pipeline_changes") or [])
    if (b_pipe - a_pipe) or (a_pipe - b_pipe):
        lines.append("\n## Pipeline changes")
        for p in sorted(b_pipe - a_pipe):
            lines.append(f"  only in {v_b}: {p}")
        for p in sorted(a_pipe - b_pipe):
            lines.append(f"  only in {v_a}: {p}")

    a_cfg = ea.get("config_changes") or {}
    b_cfg = eb.get("config_changes") or {}
    diff_cfg = {k: (a_cfg.get(k), b_cfg.get(k)) for k in set(a_cfg) | set(b_cfg) if a_cfg.get(k) != b_cfg.get(k)}
    if diff_cfg:
        lines.append("\n## Config changes")
        for k, (va, vb) in diff_cfg.items():
            lines.append(f"  {k}: {va!r} → {vb!r}")

    lines.append("\n## Metrics")
    for k in ("oof_score_mean", "holdout_score", "runtime_sec"):
        va, vb = ea.get(k), eb.get(k)
        if va is not None and vb is not None:
            delta = vb - va if isinstance(va, (int, float)) else None
            extra = f"  (Δ={delta:+.5f})" if delta is not None else ""
            lines.append(f"  {k}: {va} → {vb}{extra}")

    pred_b = eb.get("predicted_delta")
    act_b = eb.get("actual_delta")
    if pred_b is not None and act_b is not None:
        ratio = (abs(act_b) / abs(pred_b)) if pred_b else 0
        match = "✓" if 0.5 <= ratio <= 2.0 else "✗"
        lines.append(f"\n## Hypothesis check for {v_b}")
        lines.append(f"  Predicted: {pred_b:+.5f}")
        lines.append(f"  Actual:    {act_b:+.5f}")
        lines.append(f"  Match:     {match}  (ratio={ratio:.2f})")

    fa, fb = ea.get("flags") or [], eb.get("flags") or []
    if fa or fb:
        lines.append("\n## Flags")
        if fa:
            lines.append(f"  {v_a}: {fa}")
        if fb:
            lines.append(f"  {v_b}: {fb}")

    return "\n".join(lines) + "\n"


def regressions() -> str:
    out = [e for e in _entries() if any("silent_regression" in f for f in (e.get("flags") or []))]
    if not out:
        return "no silent regressions detected.\n"
    lines = ["# silent regressions (holdout drop vs parent)"]
    for e in out:
        lines.append(
            f"  {e['version']} (parent={e['parent']}): holdout {e['parent_holdout_score']:.5f} → "
            f"{e['holdout_score']:.5f} (Δ={e['actual_delta']:+.5f})  hyp={e['hypothesis']!r}"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- rendering
def _md_per_version(e: dict) -> str:
    ver = e["version"]
    parent = e.get("parent") or "—"
    flags = e.get("flags") or []
    notes = e.get("notes") or []

    pred = e.get("predicted_delta")
    act = e.get("actual_delta")
    match_str = "—"
    if pred is not None and act is not None:
        ratio = (abs(act) / abs(pred)) if pred else 0
        if pred == 0 and act == 0:
            match_str = "n/a"
        elif (pred > 0 and act > 0) or (pred < 0 and act < 0):
            match_str = "✓ matched" if 0.5 <= ratio <= 2.0 else "⚠ off"
        else:
            match_str = "✗ sign mismatch"

    lines = [f"# {ver} — hypothesis check"]
    lines.append("")
    lines.append(f"- **Parent**: `{parent}`")
    lines.append(f"- **Created**: {e.get('created_at', '?')}")
    lines.append(f"- **Completed**: {e.get('completed_at', '?')}")
    lines.append(f"- **Cloud or local**: {e.get('cloud_or_local', '?')}")
    lines.append(f"- **Git SHA**: `{e.get('git_sha') or '?'}`")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(f"> {e.get('hypothesis', '?')}")
    lines.append("")
    lines.append(f"- **Predicted Δ holdout**: `{pred:+.5f}`" if pred is not None else "- **Predicted Δ**: ?")
    lines.append(f"- **Actual Δ holdout**: `{act:+.5f}`" if act is not None else "- **Actual Δ**: (no parent or no result yet)")
    lines.append(f"- **Match**: {match_str}")
    lines.append(f"- **Confidence stated**: {e.get('confidence', '?')}")
    lines.append("")
    lines.append(f"## Metrics ({M})")
    lines.append(f"- **OOF {M}**: `{e.get('oof_score_mean'):.5f}`" if e.get("oof_score_mean") is not None else f"- OOF {M}: ?")
    folds = e.get("oof_score_per_fold") or []
    if folds:
        lines.append(f"- **Per-fold {M}**: " + ", ".join(f"`{x:.5f}`" for x in folds))
    lines.append(f"- **Holdout {M}**: `{e.get('holdout_score'):.5f}`" if e.get("holdout_score") is not None else f"- Holdout {M}: ?")
    if e.get("oof_score_mean") is not None and e.get("holdout_score") is not None:
        gap = e["holdout_score"] - e["oof_score_mean"]
        lines.append(f"- **Gap holdout−oof**: `{gap:+.5f}`")
    if e.get("runtime_sec") is not None:
        lines.append(f"- **Runtime**: `{e['runtime_sec']:.1f}s`")
    lines.append("")
    lines.append("## Changes from parent")
    feats = e.get("feature_changes") or []
    pipe = e.get("pipeline_changes") or []
    cfg = e.get("config_changes") or {}
    if feats:
        lines.append("**Features:**")
        for f in feats:
            lines.append(f"  - {f}")
    if pipe:
        lines.append("**Pipeline:**")
        for p in pipe:
            lines.append(f"  - {p}")
    if cfg:
        lines.append("**Config:**")
        for k, v in cfg.items():
            lines.append(f"  - `{k}` = `{v!r}`")
    if not (feats or pipe or cfg):
        lines.append("(none recorded)")
    lines.append("")
    lines.append("## Flags")
    if flags:
        for f in flags:
            lines.append(f"- ⚠ {f}")
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Human notes")
    if notes:
        for n in notes:
            lines.append(f"- {n}")
    else:
        lines.append(f"_None yet. Add with:_  `python -m src.diary flag {ver} \"...\"`")
    lines.append("")
    return "\n".join(lines)


def _md_index() -> str:
    rows = _entries()
    lines = [
        "# S6E6 Experiment Diary — Predicting Stellar Class",
        "",
        f"Metric: **{M}**. Source of truth: `experiments.jsonl` (git-tracked, append-only). "
        "Auto-regenerated by `python -m src.diary render`.",
        "",
        "## Timeline",
        "",
        f"| ver | parent | predicted Δ | actual Δ | holdout {M} | OOF {M} | gap | hypothesis | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in rows:
        ver = e.get("version", "?")
        par = e.get("parent") or "—"
        pred = e.get("predicted_delta")
        act = e.get("actual_delta")
        hold = e.get("holdout_score")
        oof = e.get("oof_score_mean")
        gap = (hold - oof) if (hold is not None and oof is not None) else None
        hyp = e.get("hypothesis") or ""
        if len(hyp) > 70:
            hyp = hyp[:67] + "..."
        flags = e.get("flags") or []
        flag_str = ("⚠ " + ", ".join(f.split("(")[0] for f in flags)) if flags else ""
        lines.append(
            f"| [{ver}](versions/{ver}.md) | `{par}` | "
            f"{(f'{pred:+.5f}' if pred is not None else '—')} | "
            f"{(f'{act:+.5f}' if act is not None else '—')} | "
            f"{(f'{hold:.5f}' if hold is not None else '—')} | "
            f"{(f'{oof:.5f}' if oof is not None else '—')} | "
            f"{(f'{gap:+.5f}' if gap is not None else '—')} | "
            f"{hyp} | {flag_str} |"
        )
    lines.append("")
    lines.append("## Read more")
    lines.append("- Per-version write-ups: `docs/versions/<vN>.md`")
    lines.append("- Strategy: `docs/strategy.md`")
    lines.append("- HTML dashboard: `reports/dashboard.html`")
    lines.append("")
    lines.append("## Commands")
    lines.append("```")
    lines.append("python -m src.diary timeline       # table")
    lines.append("python -m src.diary compare A B    # diff")
    lines.append("python -m src.diary report vN      # one version (text)")
    lines.append("python -m src.diary regressions    # flagged holdout drops")
    lines.append("python -m src.diary flag vN 'note' # add human note")
    lines.append("python -m src.diary render         # regenerate this file + per-version pages")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def render_all() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "versions").mkdir(parents=True, exist_ok=True)
    (DOCS / "diary.md").write_text(_md_index())
    for e in _entries():
        path = DOCS / "versions" / f"{e['version']}.md"
        path.write_text(_md_per_version(e))


# --------------------------------------------------------------------------- CLI
def _cli(argv: list[str]) -> int:
    if not argv:
        print(__doc__.splitlines()[0])
        print("commands: timeline | compare A B | report vN | regressions | flag vN \"note\" | render")
        return 2
    cmd = argv[0]
    args = argv[1:]
    if cmd == "timeline":
        print(timeline())
    elif cmd == "compare" and len(args) == 2:
        print(compare(args[0], args[1]))
    elif cmd == "report" and len(args) == 1:
        e = _by_version(args[0])
        if not e:
            print(f"unknown version: {args[0]}")
            return 1
        print(_md_per_version(e))
    elif cmd == "regressions":
        print(regressions())
    elif cmd == "flag" and len(args) >= 2:
        add_note(args[0], " ".join(args[1:]))
        print(f"note added to {args[0]}.")
    elif cmd == "render":
        render_all()
        print(f"rendered {DOCS / 'diary.md'} and per-version pages.")
    else:
        print(f"unknown command or wrong args: {cmd} {args}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
