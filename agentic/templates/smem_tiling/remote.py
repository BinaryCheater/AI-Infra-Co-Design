"""Run a generated kernel on a GPU host through the runner (SSH alias as parameter; no address here)."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[3] / "calibration" / "runner"
REMOTE_DIR = "~/work/research/AI-Infra-Co-Design-calib/runner"


def run_remote(host: str, source: str, m: int, n: int, k: int, reps: int = 50, arch: str = "sm_121", tol: float = 1e-4,
               amplify_warp: int = -1, amplify_cycles: int = 20000, ncu: bool = False, contract: str | None = None) -> dict:
    """contract: path of a measured contract YAML; shipped to the host so the runner derives warm-up (B22) and tolerance (B26.5)."""
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {REMOTE_DIR}"], check=True, capture_output=True)
    subprocess.run(["rsync", "-az", f"{RUNNER}/", f"{host}:{REMOTE_DIR}/"], check=True, capture_output=True)
    with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as f:
        f.write(source); local = f.name
    subprocess.run(["rsync", "-az", local, f"{host}:{REMOTE_DIR}/kernel.cu"], check=True, capture_output=True)
    if contract:
        subprocess.run(["rsync", "-az", str(contract), f"{host}:{REMOTE_DIR}/contract.yaml"], check=True, capture_output=True)
    cmd = (f"export PATH=$PATH:/usr/local/cuda/bin; cd {REMOTE_DIR} && PY=$HOME/.venv/bin/python; [ -x $PY ] || PY=python3; $PY agentic_runner.py --kernel kernel.cu "
           f"--m {m} --n {n} --k {k} --reps {reps} --arch {arch} --tol {tol} --amplify-warp {amplify_warp} --amplify-cycles {amplify_cycles}"
           + (" --ncu" if ncu else "") + (" --contract contract.yaml" if contract else ""))
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("{")), None)
    if line is None:
        # the runner prints an indented JSON object; take everything from the first "{"
        i = r.stdout.find("{")
        if i < 0:
            raise RuntimeError(f"runner produced no JSON: {r.stderr[-800:]}")
        return json.loads(r.stdout[i:])
    return json.loads(r.stdout[r.stdout.find("{"):])
