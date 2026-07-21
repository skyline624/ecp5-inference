"""env.py — prépare l'environnement subprocess (PATH oss-cad-suite + Graphviz).

Piège documenté (mémoire oss-cad-build-env) : lancer yosys/iverilog avec juste
`bin/` sur le PATH -> DLL manquante 0xC0000139. Il faut `bin/` ET `lib/` (sous
Windows oss-cad-suite met les DLL dans `lib/`).

Graphviz (binaire `dot`) est installé via winget sous
`C:\\Program Files\\Graphviz\\bin` (Gate 0) -- absent du PATH global.

Ce module ajoute les deux au PATH du process courant (et donc des subprocess
héritent du PATH). À importer au démarrage du serveur (dans main.py).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

from . import config


def _prepend(path: str) -> None:
    """Ajoute `path` en tête du PATH s'il existe et n'y est pas déjà."""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        return
    cur = os.environ.get("PATH", "")
    parts = cur.split(os.pathsep)
    if path not in parts:
        os.environ["PATH"] = path + os.pathsep + cur


def setup() -> None:
    """Configure le PATH pour yosys/iverilog/nextpnr + dot. Idempotent."""
    # oss-cad-suite : bin/ ET lib/ (les DLL sont dans lib/).
    _prepend(str(Path(config.OSS_CAD) / "bin"))
    _prepend(str(Path(config.OSS_CAD) / "lib"))
    # Graphviz `dot` (winget install, Gate 0).
    _prepend(config.GRAPHVIZ_BIN)


def python_exe() -> str:
    """Python.org 3.12 (PAS le Store) -- cf mémoire python-env-setup."""
    return sys.executable


if __name__ == "__main__":
    setup()
    import shutil
    for tool in ("yosys", "iverilog", "vvp", "dot", "nextpnr-ecp5"):
        print(f"{tool:14} -> {shutil.which(tool) or 'ABSENT'}")