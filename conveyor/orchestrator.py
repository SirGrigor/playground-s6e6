"""Conveyor orchestrator — runs LOCALLY (driven by Claude). One config → one harvested result.

The autonomous loop's engine. For a given experiment config it:
  1. renders a Kaggle kernel (metadata + entrypoint that clones the repo, installs the GPU-compatible
     torch, then runs conveyor/experiment.py in a fresh process so the pinned torch loads),
  2. `kaggle kernels push`,
  3. polls `kaggle kernels status` to completion,
  4. `kaggle kernels output` → parses result.json,
  5. appends to conveyor/leaderboard.jsonl.

Phase 1 = prove this cycle on one config. Phase 2 wraps it in a search policy (generate configs:
RealMLP Optuna knobs, FE ops, model types) + a decision loop. Phase 3 = cron/budget-guarded autonomy.

Usage:  python conveyor/orchestrator.py path/to/config.json
GPU recipe baked in: the kernel pip-installs torch==2.5.1 (cu121) — the validated P100 fix.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

USER = "iljagrigorjev"
REPO_URL = "https://github.com/SirGrigor/playground-s6e6.git"
COMPETITION = "playground-series-s6e6"
HERE = Path(__file__).resolve().parent
LEADERBOARD = HERE / "leaderboard.jsonl"

KERNEL_TEMPLATE = '''\
import json, subprocess, sys, os
CONFIG = json.loads({config_json})  # parse JSON (handles false/true/null), don't inline as Python
print("=== CONVEYOR KERNEL", CONFIG.get("id"), "===")
subprocess.run(["git", "clone", "-q", "--depth", "1", "{repo_url}"], check=True)
REPO = os.path.join(os.getcwd(), "playground-s6e6")
if CONFIG.get("needs_gpu", True):
    # validated P100 fix: Kaggle's torch can't drive sm_60; pin a Pascal-compatible build
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
                    "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
json.dump(CONFIG, open("/kaggle/working/config.json", "w"))
# run in a FRESH process so the pinned torch loads
r = subprocess.run([sys.executable, os.path.join(REPO, "conveyor", "experiment.py"),
                    "/kaggle/working/config.json"])
sys.exit(r.returncode)
'''


def _kaggle(*args, timeout=600):
    out = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    return (out.stdout + out.stderr).replace("Warning: Looks like you're using an outdated", "").strip()


def _is_done(st: str) -> bool:
    """A kernel is terminal only on a real KernelWorkerStatus — NOT a transient 404/'Client Error'
    (which contains 'Error' and would otherwise be mistaken for the ERROR status)."""
    up = st.upper()
    return "KERNELWORKERSTATUS." in up and any(k in up for k in ("COMPLETE", "ERROR", "CANCEL"))


def render(config: dict, workdir: Path) -> str:
    """Write kernel-metadata.json + main.py for `config`. Returns the kernel slug."""
    workdir.mkdir(parents=True, exist_ok=True)
    slug = f"{USER}/s6e6-cv-{config['id']}".replace("_", "-").lower()
    (workdir / "main.py").write_text(
        KERNEL_TEMPLATE.format(config_json=repr(json.dumps(config)), repo_url=REPO_URL))
    (workdir / "kernel-metadata.json").write_text(json.dumps({
        "id": slug, "title": f"s6e6 cv {config['id']}"[:50], "code_file": "main.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": bool(config.get("needs_gpu", True)), "enable_internet": True,
        "competition_sources": [COMPETITION],
        "dataset_sources": config.get("dataset_sources", []), "kernel_sources": [],
    }, indent=2))
    return slug


def submit_and_wait(slug: str, workdir: Path, poll_s=30, max_polls=240) -> str:  # ~120 min cap
    print(f"[push] {slug}")
    print(_kaggle("kernels", "push", "-p", str(workdir)))
    for i in range(max_polls):
        st = _kaggle("kernels", "status", slug)
        done = _is_done(st)
        print(f"[poll {i + 1}] {st.splitlines()[-1] if st else st}")
        if done:
            return st
        time.sleep(poll_s)
    return "TIMEOUT"


def harvest(slug: str, workdir: Path) -> dict | None:
    print(_kaggle("kernels", "output", slug, "-p", str(workdir)))
    rj = workdir / "result.json"
    if rj.exists():
        return json.loads(rj.read_text())
    # fallback: parse the RESULT_JSON line from the kernel log
    for log in workdir.glob("*.log"):
        try:
            entries = json.loads(log.read_text())
            text = "".join(e.get("data", "") for e in entries if e.get("stream_name") == "stdout")
            for line in text.splitlines():
                if line.startswith("RESULT_JSON "):
                    return json.loads(line[len("RESULT_JSON "):])
        except Exception:  # noqa: BLE001
            continue
    return None


def run_one(config: dict) -> dict | None:
    workdir = Path("/tmp/conveyor") / config["id"]
    slug = render(config, workdir)
    submit_and_wait(slug, workdir)
    result = harvest(slug, workdir)
    if result:
        result["_slug"] = slug
        with LEADERBOARD.open("a") as f:
            f.write(json.dumps(result) + "\n")
        best = max((result.get(f"{m}_holdout", 0) for m in ("rm", "lgb", "cat")), default=0)
        print(f"[harvest] id={result['id']} best_holdout={best:.5f} → appended to leaderboard.jsonl")
    else:
        print("[harvest] NO RESULT — check the kernel log in", workdir)
    return result


def fire_parallel(configs: list, poll_s=45, max_wait_min=360) -> dict:
    """Push N kernels at once, poll all to completion, harvest each. Returns {id: result}.

    Kaggle may cap concurrent GPU sessions (excess queue, still complete) — degrades gracefully.
    Each kernel writes its own output to /tmp/conveyor/<id>/ (harvested independently)."""
    slugs = {}
    for cfg in configs:
        wd = Path("/tmp/conveyor") / cfg["id"]
        slug = render(cfg, wd)
        print(f"[push] {slug}"); print(_kaggle("kernels", "push", "-p", str(wd)))
        slugs[cfg["id"]] = (slug, wd)
    pending = dict(slugs)
    for _ in range(int(max_wait_min * 60 / poll_s)):
        if not pending:
            break
        done = []
        for cid, (slug, wd) in pending.items():
            st = _kaggle("kernels", "status", slug)
            if _is_done(st):
                print(f"[done] {cid}: {st.splitlines()[-1] if st else st}"); done.append(cid)
        for cid in done:
            del pending[cid]
        if pending:
            time.sleep(poll_s)
    results = {}
    for cid, (slug, wd) in slugs.items():
        r = harvest(slug, wd)
        results[cid] = r
        if r:
            with LEADERBOARD.open("a") as f:
                f.write(json.dumps({**r, "_slug": slug}) + "\n")
            print(f"[harvest] {cid}: {({k: r[k] for k in ('model', 'holdout') if k in r})}")
        else:
            print(f"[harvest] {cid}: NO RESULT")
    return results


if __name__ == "__main__":
    arg = json.loads(Path(sys.argv[1]).read_text())
    # a list of configs → parallel fleet; a single config → one run
    fire_parallel(arg) if isinstance(arg, list) else run_one(arg)
