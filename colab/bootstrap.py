"""Colab bootstrap — the single source of truth for a S6E6 Colab run (idempotent).

All evolving Colab logic lives HERE, version-controlled, so a fresh clone always runs
the latest correct flow. The notebook is a thin, stable launcher that only does the
Colab-specific bits (mount Drive, read secrets into env, fresh-clone) then calls this.
Change the run flow → edit this file + push; the notebook never changes. Switch the
experiment → edit SPRINT_ACTIVE.txt + push, re-run the Run cell.

Steps: install deps → GPU report → get data (kaggle comp + fedesoriano SDSS17 original
+ Drive sync) → pull prior artifacts → run the active script from SPRINT_ACTIVE.txt →
render HTML dashboard → push artifacts+dashboard back to Drive → persist diary to git.

Vendoring (per feedback_colab-vendor-not-pip-own-code): our own toolkits (synth-decoder,
S6E5 diagnostics, rogii Caruana) are FRESH-CLONED into vendor/ and reached via PYTHONPATH —
never pip-installed-from-git, which version-caches a stale wheel.

Run on Colab from the repo root:  python colab/bootstrap.py
Env: DRIVE_ROOT (artifact/external sync), KAGGLE_API_TOKEN (data), GH_TOKEN (diary push).
Use --dry-run to print the plan without installing/training (local sanity).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMP = "playground-series-s6e6"
GH_REPO = "SirGrigor/playground-s6e6"
ORIGINAL_DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
DEPS = ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow",
        "lightgbm", "xgboost", "catboost",
        "pytabkit", "torch",          # v7 RealMLP (neural net) — Colab has torch+GPU preinstalled
        "cleanlab",                    # v9 confident-learning (label-noise / ceiling diagnostic)
        "tabicl",                      # v10 TabICLv2 tabular foundation model (auto-downloads ckpt)
        "matplotlib", "seaborn", "rich", "joblib"]
ARTIFACT_DIRS = ("probs", "submissions", "reports")
# Our own toolkits — fresh-cloned, NOT pip'd. (repo_url, vendor_subdir)
VENDOR = [
    ("https://github.com/SirGrigor/synth-decoder.git", "synth-decoder"),
]
DRY = "--dry-run" in sys.argv


def sh(cmd: str, check: bool = True) -> None:
    print(f"  $ {cmd}")
    if not DRY:
        subprocess.run(cmd, shell=True, check=check)


def active_script() -> str:
    for line in (ROOT / "SPRINT_ACTIVE.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise RuntimeError("SPRINT_ACTIVE.txt has no active script line")


def install() -> None:
    print("[1] deps (fixed union — idempotent)")
    sh("pip install -q -U kaggle")            # Colab's preinstalled kaggle is old
    sh(f"pip install -q {' '.join(DEPS)}")


def gpu_report() -> None:
    """GPU is OPTIONAL for S6E6 — GBDT (LGBM/XGB/CatBoost) run fine on CPU. Report only."""
    print("=" * 64)
    has_gpu = os.path.exists("/proc/driver/nvidia/version")
    if not DRY:
        subprocess.run("nvidia-smi -L 2>/dev/null || echo '[GPU] none'", shell=True, check=False)
    print(f"[GPU] present={has_gpu} → GBDT works either way; GPU only speeds large sweeps.")
    print("=" * 64)


def vendor_toolkits() -> None:
    print("[2] vendor own toolkits (fresh clone, PYTHONPATH — never pip-from-git)")
    vdir = ROOT / "vendor"
    vdir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("GH_TOKEN")  # synth-decoder is a PRIVATE repo → clone needs auth
    for url, sub in VENDOR:
        dst = vdir / sub
        auth_url = url.replace("https://", f"https://{token}@") if token else url
        if DRY:
            print(f"  clone {url} -> {dst}"); continue
        if dst.exists():
            subprocess.run("git pull -q", shell=True, cwd=str(dst), check=False)
            print(f"  pulled {sub}")
        else:
            r = subprocess.run(f"git clone -q --depth 1 {auth_url} {dst}", shell=True)
            if r.returncode == 0:
                print(f"  cloned {sub}")
            else:
                print(f"  ⚠ clone failed for {sub} (private repo? set GH_TOKEN with repo read scope)")


def _copy_into(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        return
    if DRY:
        print(f"  sync {src} -> {dst}"); return
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src, dst); print(f"  copied {label}")


def get_data() -> None:
    print("[3] competition + original (SDSS17) data")
    for d in ("data/raw", "data/external", "probs", "submissions", "reports/figs"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    drive = os.environ.get("DRIVE_ROOT")
    # competition data
    if not (ROOT / "data" / "raw" / "train.csv").exists():
        sh(f"kaggle competitions download -c {COMP} -p data/raw", check=False)
        sh("cd data/raw && (unzip -o -q '*.zip' 2>/dev/null; rm -f *.zip) || true", check=False)
    if not (ROOT / "data" / "raw" / "train.csv").exists() and drive:
        for fn in ("train.csv", "test.csv", "sample_submission.csv"):
            _copy_into(Path(drive) / "data" / "raw" / fn, ROOT / "data" / "raw" / fn, f"raw/{fn}")
    # original SDSS17 — leak-match + augmentation source
    if not list((ROOT / "data" / "external").glob("*.csv")):
        sh(f"kaggle datasets download -d {ORIGINAL_DATASET} -p data/external", check=False)
        sh("cd data/external && (unzip -o -q '*.zip' 2>/dev/null; rm -f *.zip) || true", check=False)
    if not DRY:
        sh("ls -la data/raw data/external data/splits 2>/dev/null || true", check=False)


def pull_artifacts() -> None:
    drive = os.environ.get("DRIVE_ROOT")
    print(f"[4] pull prior artifacts from Drive ({drive or 'DRIVE_ROOT unset — skip'})")
    if not drive:
        return
    for sub in ARTIFACT_DIRS:
        _copy_into(Path(drive) / sub, ROOT / sub, sub)


def run_script() -> None:
    script = active_script()
    print(f"[5] RUN  {script}  (PYTHONPATH=repo root + vendored toolkits)")
    # add both repo root and its src/ (synth-decoder is a src-layout package → import synth_decoder)
    vendor_paths = []
    for _, sub in VENDOR:
        base = ROOT / "vendor" / sub
        vendor_paths.append(str(base))
        if (base / "src").is_dir():
            vendor_paths.append(str(base / "src"))
    pythonpath = os.pathsep.join([str(ROOT), *vendor_paths, os.environ.get("PYTHONPATH", "")])
    if DRY:
        print(f"  $ PYTHONPATH={pythonpath} python {script}"); return
    env = {**os.environ, "PYTHONPATH": pythonpath}
    subprocess.run(f"python {script}", shell=True, check=True, env=env, cwd=str(ROOT))


def render_dashboard() -> None:
    """Regenerate reports/dashboard.html from whatever figs+diary exist (best-effort)."""
    print("[6] render HTML dashboard")
    if DRY:
        print("  $ python -c 'from src.dashboard import render_html; render_html()'"); return
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-c", "from src.dashboard import render_html; print(render_html())"],
                   cwd=str(ROOT), env=env, check=False)


def push_artifacts() -> None:
    drive = os.environ.get("DRIVE_ROOT")
    print(f"[7] push artifacts + dashboard to Drive ({drive or 'DRIVE_ROOT unset — skip'})")
    if not drive:
        return
    for sub in ("probs", "submissions", "reports"):
        _copy_into(ROOT / sub, Path(drive) / sub, sub)
    src = ROOT / "experiments.jsonl"
    if src.exists() and not DRY:
        shutil.copy(src, Path(drive) / "experiments.jsonl")


def push_diary_to_git() -> None:
    """Persist experiments.jsonl + docs/diary.md to git (needs GH_TOKEN, Contents:RW PAT)."""
    token = os.environ.get("GH_TOKEN")
    print(f"[8] persist diary to git ({'GH_TOKEN set' if token else 'no GH_TOKEN — Drive only; skip'})")
    if not token or DRY:
        return
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    subprocess.run([sys.executable, "-m", "src.diary", "render"], cwd=str(ROOT), env=env, check=False)
    for c in ('git config user.email "colab@s6e6.bootstrap"',
              'git config user.name "s6e6-colab-bootstrap"',
              "git add experiments.jsonl docs/diary.md docs/versions 2>/dev/null",
              'git diff --cached --quiet || git commit -q -m "diary: Colab run"',
              # stash any other run-dirtied tracked files so the rebase can't abort (v10 log lesson)
              "git stash -q 2>/dev/null || true",
              "git pull --rebase -q origin master 2>/dev/null || git pull --rebase -q origin main 2>/dev/null || true",
              "git stash pop -q 2>/dev/null || true"):
        subprocess.run(c, shell=True, cwd=str(ROOT), check=False)
    r = subprocess.run(["git", "push", "-q", f"https://{token}@github.com/{GH_REPO}.git", "HEAD"],
                       cwd=str(ROOT), capture_output=True, text=True)
    print("  ✓ diary pushed" if r.returncode == 0 else f"  ⚠ diary push failed: {r.stderr.strip()[:160]}")


def main() -> None:
    print(f"=== S6E6 Colab bootstrap (root={ROOT}{' DRY-RUN' if DRY else ''}) ===")
    install()
    gpu_report()
    vendor_toolkits()
    get_data()
    pull_artifacts()
    script_failed = None
    try:
        run_script()
    except subprocess.CalledProcessError as e:
        # Even on failure, render + push whatever exists — don't lose compute on disconnect.
        script_failed = e
        print(f"\n⚠ script failed (rc={e.returncode}) — rendering + pushing artifacts anyway...")
    render_dashboard()
    push_artifacts()
    push_diary_to_git()
    if script_failed:
        print(f"=== done (with script error rc={script_failed.returncode}) — artifacts synced where present ===")
        raise SystemExit(script_failed.returncode)
    print("=== done ===")


if __name__ == "__main__":
    main()
