"""Model-process UI for S6E6 — two surfaces (per the locked design):

1. **Live in-cell** (rich, render-on-update): goal banner → per-model scoreboard →
   end-of-run verdict, printed into the Colab cell so you get *traction while it runs*.
2. **Persistent HTML** (`render_html`): one portable file embedding the viz panels
   (confusion / ROC / PR / F1) + the experiment-diary timeline, base64-inlined so it
   opens straight from Drive. Reviewable after the run.

Design notes (inherited from rogii src/dashboard.py — hard-won):
- **Cosmetic only.** Every entry point is try/except wrapped; a render failure can
  NEVER abort training. The diary + experiments.jsonl are what matter.
- **Render-on-update, not rich.Live.** Live's cursor-movement codes render glitchy
  through Colab's subprocess pipe. We re-print a fresh table instead; `force_terminal=True`
  emits ANSI colours that Colab's cell output *does* render.
- **Ascent, not descent.** Unlike rogii (minimize RMSE), here we MAXIMIZE accuracy:
  baseline floor → current best → TARGET (top-50 LB bar). Higher is better.

FLOOR / TARGET are placeholders until Phase-0 sets them:
  - FLOOR  = redshift-only baseline accuracy (fill after notebooks/01_eda.py).
  - TARGET = top-50 public-LB accuracy (fill after the compute-parity audit).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

from .config import METRIC, REPORTS, ROOT

# --- ascent anchors (balanced accuracy; update after Phase 0) ---------------
FLOOR = 0.950    # redshift-only BAL-ACC baseline guess — REPLACE with measured value
TARGET = 0.980   # top-50 LB balanced accuracy — REPLACE after LB audit
HIGHER_BETTER = True

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _C = Console(force_terminal=True)
    _RICH = True
except Exception:
    _RICH = False
    _C = None


def _pct(x: float) -> float:
    """Fraction of floor→target distance covered (clamped 0..1)."""
    if TARGET <= FLOOR:
        return 0.0
    return max(0.0, min(1.0, (x - FLOOR) / (TARGET - FLOOR)))


def _bar(x: float, width: int = 22) -> str:
    n = int(round(_pct(x) * width))
    return "█" * n + "░" * (width - n)


def _style_for(delta_vs_best: float) -> tuple[str, str]:
    """(style, marker) by improvement over current best: ≥0 good, ≥-0.001 ok, else bad."""
    if delta_vs_best >= 0.0:
        return "green", "✅"
    if delta_vs_best >= -0.001:
        return "yellow", "≈"
    return "red", "✗"


# --------------------------------------------------------------------------- live
def goal_banner(ver: str, run_desc: str, test_desc: str, best: float = FLOOR) -> None:
    """Run-start panel: floor → best → TARGET, where we are, what this run tests."""
    try:
        if _RICH:
            body = Text()
            body.append(f"floor {FLOOR:.4f}", style="dim"); body.append("    ")
            body.append(f"best {best:.4f}", style="cyan"); body.append("    ")
            body.append(f"◆ TARGET {TARGET:.4f}\n", style="bold green")
            body.append(f"{FLOOR:.3f} "); body.append(_bar(best), style="yellow")
            body.append(f" {TARGET:.3f}", style="dim")
            body.append(f"   {_pct(best) * 100:.0f}% there (at best)\n", style="dim")
            body.append("this run: ", style="dim"); body.append(f"{run_desc}\n", style="bold")
            body.append("testing : ", style="dim"); body.append(test_desc, style="italic")
            _C.print(Panel(body, title=f"[bold]S6E6 · Stellar Class · {METRIC}[/]  ·  {ver}",
                           border_style="green", box=box.ROUNDED))
        else:
            print(f"\n=== S6E6 · {METRIC} · {ver} ===")
            print(f"floor {FLOOR:.4f} | best {best:.4f} | TARGET {TARGET:.4f}  ({_pct(best)*100:.0f}% there)")
            print(f"this run: {run_desc}\ntesting : {test_desc}\n")
    except Exception:
        pass


def training(name: str, i: int, n: int) -> None:
    try:
        msg = f"▶ training {name}  ({i}/{n})…"
        _C.print(msg, style="bold blue") if _RICH else print(msg)
    except Exception:
        pass


def scoreboard(rows: list[dict], best: float | None = None) -> None:
    """Re-print the per-model table. rows: [{name, oof, holdout}, ...]."""
    try:
        if best is None:
            best = max((r["holdout"] for r in rows), default=FLOOR)
        if _RICH:
            t = Table(box=box.SIMPLE_HEAVY, title="models so far", title_style="dim")
            t.add_column("model"); t.add_column("oof", justify="right")
            t.add_column("holdout", justify="right"); t.add_column("vs best", justify="right")
            t.add_column(f"→{TARGET:.3f}")
            for r in rows:
                d = r["holdout"] - best
                style, mark = _style_for(d)
                t.add_row(r["name"], f"{r['oof']:.4f}", f"[{style}]{r['holdout']:.4f}[/]",
                          f"[{style}]{d:+.4f} {mark}[/]", _bar(r["holdout"], 10))
            _C.print(t)
        else:
            for r in rows:
                print(f"  {r['name']}: oof {r['oof']:.4f} | holdout {r['holdout']:.4f} "
                      f"({r['holdout'] - best:+.4f} vs best)")
    except Exception:
        pass


def verdict(ver: str, holdout: float, runtime_sec: float, best: float = FLOOR,
            oof: float | None = None) -> None:
    """End-of-run panel: holdout score, delta vs best + target, runtime, ascent bar."""
    try:
        d_best = holdout - best
        d_target = holdout - TARGET
        if holdout >= TARGET:
            v_txt = "✅ beat target!"
        elif d_best > 0.0005:
            v_txt = f"📈 improved (+{d_best:.4f} vs best)"
        elif abs(d_best) <= 0.0005:
            v_txt = "≈ matched best"
        else:
            v_txt = f"⚠ below best ({d_best:+.4f})"
        if _RICH:
            body = Text()
            body.append(f"holdout {METRIC}: {holdout:.4f}", style="bold")
            if oof is not None:
                body.append(f"   (oof {oof:.4f})", style="dim")
            body.append("\n")
            body.append(f"vs best {best:.4f} : {d_best:+.4f}\n",
                        style="green" if d_best >= 0 else "yellow")
            body.append(f"vs target {TARGET:.4f}: {d_target:+.4f}",
                        style="green" if d_target >= 0 else "cyan")
            body.append(f"   ({max(0.0, -d_target):.4f} to go)\n", style="dim")
            body.append(f"runtime      : {runtime_sec / 60:.1f} min\n", style="dim")
            body.append(f"ascent: {FLOOR:.3f} "); body.append(_bar(holdout), style="bold green")
            body.append(f" {TARGET:.3f}  ({_pct(holdout) * 100:.0f}% there)\n", style="dim")
            body.append("VERDICT: ", style="dim"); body.append(v_txt, style="bold")
            _C.print(Panel(body, title=f"[bold]{ver} — RESULT[/]",
                           border_style="green" if d_best >= 0 else "yellow", box=box.DOUBLE))
        else:
            extra = f" | oof {oof:.4f}" if oof is not None else ""
            print(f"\n=== {ver} RESULT: holdout {holdout:.4f}{extra} | vs best {d_best:+.4f} | "
                  f"to target {max(0.0, -d_target):.4f} | {runtime_sec/60:.1f} min | {v_txt} ===")
    except Exception:
        pass


# --------------------------------------------------------------------------- HTML
def _img_tag(path: Path) -> str:
    if not path.exists():
        return f"<div class='missing'>missing: {path.name}</div>"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"<img src='data:image/png;base64,{b64}' alt='{path.name}'/>"


def _diary_rows() -> list[dict]:
    p = ROOT / "experiments.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def render_html(panels: dict[str, Path] | None = None, *, prefix: str = "latest",
                title: str = "S6E6 — Predicting Stellar Class") -> Path:
    """Render the portable HTML dashboard: viz panels (base64-inlined) + diary timeline.

    `panels` = the dict returned by viz.render_all_panels(); if None, embeds whatever
    PNGs exist for `prefix` in reports/figs/. Returns the written HTML path.
    """
    REPORTS.mkdir(parents=True, exist_ok=True)
    figs_dir = REPORTS / "figs"
    if panels is None:
        panels = {p.stem.replace(f"{prefix}_", ""): p
                  for p in sorted(figs_dir.glob(f"{prefix}_*.png"))}

    # diary timeline table
    rows = _diary_rows()
    diary_html = ["<table><tr><th>ver</th><th>parent</th><th>pred Δ</th><th>actual Δ</th>"
                  f"<th>holdout {METRIC}</th><th>oof</th><th>hypothesis</th><th>flags</th></tr>"]
    for e in rows:
        flags = ", ".join(f.split("(")[0] for f in (e.get("flags") or []))
        flag_cls = " class='flag'" if flags else ""
        def fmt(v, p=".5f"):
            return f"{v:{p}}" if isinstance(v, (int, float)) else "—"
        diary_html.append(
            f"<tr><td>{e.get('version','?')}</td><td>{e.get('parent') or '—'}</td>"
            f"<td>{fmt(e.get('predicted_delta'), '+.5f')}</td>"
            f"<td>{fmt(e.get('actual_delta'), '+.5f')}</td>"
            f"<td>{fmt(e.get('holdout_score'))}</td><td>{fmt(e.get('oof_score_mean'))}</td>"
            f"<td class='hyp'>{(e.get('hypothesis') or '')[:90]}</td>"
            f"<td{flag_cls}>{flags}</td></tr>"
        )
    diary_html.append("</table>")

    panel_order = ["confusion", "report", "roc", "ovo_auc", "pr", "class_balance", "fold_scores", "rho_matrix"]
    ordered = [k for k in panel_order if k in panels] + [k for k in panels if k not in panel_order]
    panels_html = "".join(
        f"<figure><figcaption>{k}</figcaption>{_img_tag(panels[k])}</figure>" for k in ordered
    )

    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:24px;background:#0f1117;color:#e6e6e6}}
 h1{{font-weight:600}} h2{{border-bottom:1px solid #333;padding-bottom:4px;margin-top:32px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}}
 figure{{margin:0;background:#171a23;border:1px solid #262b38;border-radius:8px;padding:10px}}
 figcaption{{font-size:13px;color:#8aa;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}}
 img{{width:100%;border-radius:4px}} .missing{{color:#a55;font-style:italic}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #2a2f3c;padding:5px 8px;text-align:right}}
 th{{background:#1b1f2a;color:#9bb}} td.hyp{{text-align:left;color:#bcd}} td.flag{{color:#e88;font-weight:600}}
 .meta{{color:#778;font-size:13px}}
</style></head><body>
<h1>{title}</h1>
<p class='meta'>metric: <b>{METRIC}</b> · prefix: <b>{prefix}</b> · floor {FLOOR:.4f} → target {TARGET:.4f}</p>
<h2>Diagnostics — {prefix}</h2>
<div class='grid'>{panels_html}</div>
<h2>Experiment timeline</h2>
{''.join(diary_html)}
</body></html>"""
    out = REPORTS / "dashboard.html"
    out.write_text(html)
    return out
