"""config.py — chemins du projet et variables d'environnement pour le serveur.

Toutes les paths sont absolues (le serveur peut être lancé depuis n'importe
quel cwd). La racine du dépôt est le parent du dossier `server/`.
"""
from __future__ import annotations
import os
from pathlib import Path

# Racine du dépôt = parent de server/
ROOT = Path(__file__).resolve().parent.parent

SRC_DIR = ROOT / "src"
BOARD_DIR = ROOT / "board"
SIM_DIR = ROOT / "sim"
BUILD_DIR = ROOT / "build"
LUT_DIR = ROOT / "lut"
WEB_DIR = ROOT / "web"

# Workdir du serveur (artefacts générés : VCD, index, cluster_top.v)
WORK_DIR = ROOT / "server" / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# oss-cad-suite (yosys / iverilog / nextpnr). Le plan documente le piège : PATH
# seul = DLL manquante 0xC0000139 -> il faut bin/ ET lib/ sur le PATH.
OSS_CAD = os.environ.get("OSS_CAD_SUITE", r"C:\oss-cad-suite")

# Graphviz installé via winget (Gate 0) -- pas sur le PATH global par défaut.
GRAPHVIZ_BIN = r"C:\Program Files\Graphviz\bin"

# Fichiers Verilog sources du graphe RTL (Phase A) : src/*.v + board/gen_seq_silicon.v.
# top.v est la version LEGACY (2959 lignes) ; node_top.v est la version courante
# (3252 lignes). Les deux définissent `module top` -> on exclut top.v pour que le
# graphe reflète le design réel (node_top.v, qui instancie flash_boot/wcache_ffn/mac18).
RTL_FILES = [p for p in sorted(SRC_DIR.glob("*.v")) if p.name != "top.v"] + [
    BOARD_DIR / "gen_seq_silicon.v",
]

# Top synthétisable (Phase A "silicium") et top de sim (Phase B).
TOP_SILICON = "gen_seq_silicon"
TOP_SIM = "gen_seq_top"