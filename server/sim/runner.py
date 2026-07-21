"""sim/runner.py — orchestre les jobs de simulation (lance cocotb_runner ou
iverilog_runner en subprocess, suit le status, localise le VCD).

Jobs en mémoire (Phase B simple ; pas de persistance). Le subprocess hérite du
PATH du process uvicorn (oss-cad-suite bin+lib via environment.ps1 au lancement
du serveur + env.setup() dans chaque runner) -> évite le piège DLL 0xC0000139.
"""
from __future__ import annotations
import subprocess
import sys
import threading
import time
import uuid
from typing import Optional

from .. import config
from . import tb_registry

_jobs: dict = {}
_vcd_cache: dict = {}  # job_id -> VcdFile


def run_sim(test_module: str, mode: str = "") -> dict:
    spec = tb_registry.find(test_module)
    if spec is None:
        raise ValueError(f"test inconnu: {test_module}")
    if not mode:
        mode = spec.get("mode", "cocotb")

    runner_module = "server.sim.iverilog_runner" if mode == "iverilog" else "server.sim.cocotb_runner"
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "test": test_module,
        "mode": mode,
        "started": time.time(),
        "vcd_path": None,
        "log": "",
        "error": None,
    }

    def work() -> None:
        job = _jobs[job_id]
        try:
            proc = subprocess.run(
                [sys.executable, "-m", runner_module, test_module],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            job["log"] = (proc.stdout or "") + (proc.stderr or "")
            for line in (proc.stdout or "").splitlines():
                if line.startswith("VCD:"):
                    job["vcd_path"] = line[4:].strip()
            job["status"] = "done" if job["vcd_path"] else "error"
            if not job["vcd_path"]:
                job["error"] = "VCD non produit (iverilog/cocotb a échoué — voir log)"
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            job["log"] = job.get("log", "") + f"\nexception: {e}"
        job["finished"] = time.time()

    threading.Thread(target=work, daemon=True).start()
    return _jobs[job_id]


def status(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)


def get_vcd(job_id: str):
    """Retourne le VcdFile (cache) pour le job, ou None si pas prêt."""
    job = _jobs.get(job_id)
    if not job or not job.get("vcd_path"):
        return None
    if job_id in _vcd_cache:
        return _vcd_cache[job_id]
    from ..vcd.parser import open_vcd
    v = open_vcd(job["vcd_path"])
    _vcd_cache[job_id] = v
    return v


def list_jobs() -> list:
    return list(_jobs.values())