"""sim/cocotb_runner.py — reproduit sim/run_cocotb.py avec waves=True (VCD).

Lancé en subprocess par server/sim/runner.py pour isoler cocotb (qui peut
clasher le process). Reprend exactement les SOURCES / .hex copy / LINK_SS de
sim/run_cocotb.py (cf plan : wrapper sans modifier l'original).

Usage : python -m server.sim.cocotb_runner <test_module>
Imprime `VCD:<path>` sur stdout (récupéré par runner.py). Même si le test cocotb
échoue (assert), le VCD est déjà écrit -> on l'imprime et exit 0 (la visu a
juste besoin de la trace, pas du pass/fail).
"""
from __future__ import annotations
import glob
import os
import shutil
import sys
from pathlib import Path

# Configure le PATH (oss-cad-suite bin+lib, dot) AVANT cocotb/iverilog. Sans ça,
# iverilog crash (0xC0000005/0xC0000139 : DLL manquante -- piège oss-cad).
from .. import env
env.setup()

HERE = Path(__file__).resolve().parent
RTL = HERE.parent.parent / "src"
LUT = HERE.parent.parent / "lut"
SIM = HERE.parent.parent / "sim"
MODELS = SIM / "models"
BUILD = SIM / "sim_build"

# cocotb doit pouvoir importer le test_module (test_gen_A.py vit dans sim/, pas
# dans BUILD). run_cocotb.py marche car lance depuis sim/ (CWD sur sys.path) ;
# ici on tourne depuis la racine du depot -> on ajoute sim/ explicitement.
sys.path.insert(0, str(SIM))

TOPLEVEL = "gen_seq_top"

SOURCES = [
    str(MODELS / "ECP5_PLL_sim.v"),
    str(MODELS / "MULTALU18X18.v"),
    str(MODELS / "sdram_chip.v"),
    str(RTL / "mac18.v"),
    str(RTL / "rmsnorm_op.v"),
    str(RTL / "silu_op.v"),
    str(RTL / "rope_op.v"),
    str(RTL / "softmax_op.v"),
    str(RTL / "attention_head_op.v"),
    str(RTL / "sdram.v"),
    str(RTL / "async_fifo.v"),
    str(RTL / "uart_bridge.v"),
    str(RTL / "vec_alu2.v"),
    str(RTL / "node_top.v"),
    str(RTL / "gen_seq.v"),
    str(RTL / "gen_seq_top.v"),
]


def main() -> int:
    test_module = sys.argv[1] if len(sys.argv) > 1 else "test_gen_A"
    BUILD.mkdir(parents=True, exist_ok=True)
    # Copy .hex LUTs (le piège $readmemh cwd) vers BUILD ET SIM. cocotb/icarus
    # lance vvp avec cwd=test_dir (=SIM), pas build_dir -> $readmemh("xxx.hex")
    # cherche dans SIM. On copie donc dans les deux pour couvrir les deux CWD.
    for f in glob.glob(str(LUT / "*.hex")):
        shutil.copy(f, BUILD)
        shutil.copy(f, SIM)
    # Image SDRAM préchargée (poids du modèle) pour les tests de génération.
    sdram_hex = SIM / "sdram_preload.hex"
    if sdram_hex.exists():
        shutil.copy(sdram_hex, BUILD)

    from cocotb_tools.runner import get_runner
    r = get_runner("icarus")
    try:
        r.build(
            sources=SOURCES,
            hdl_toplevel=TOPLEVEL,
            build_dir=str(BUILD),
            always=True,
            timescale=("1ns", "1ps"),
            includes=[str(RTL)],
            build_args=["-DLINK_SS", "-DSDRAM_PRELOAD"],
        )
        # vvp résout $readmemh("xxx.hex") relativement à son CWD. cocotb/icarus
        # lance vvp en héritant du CWD du process -> chdir vers BUILD (où les
        # .hex ont été copiés) AVANT r.test(), sinon les LUT sont introuvables
        # (rmsnorm/silu/rope/attention/exp/inv -> X -> forward nul -> argmax 0).
        os.chdir(BUILD)
        r.test(
            hdl_toplevel=TOPLEVEL,
            test_module=test_module,
            build_dir=str(BUILD),
            test_dir=str(SIM),
            waves=True,
        )
        print("RESULT:PASS")
    except Exception as e:
        # Le test peut fail (assert) mais le VCD est déjà là.
        print(f"RESULT:FAIL {e}", file=sys.stderr)

    # Localise le VCD produit (cocotb/icarus -> waves.vcd ou <top>.vcd dans BUILD).
    vcds = sorted(glob.glob(str(BUILD / "**/*.vcd"), recursive=True), key=os.path.getmtime)
    if vcds:
        print(f"VCD:{vcds[-1]}")
        return 0
    print("VCD:NONE", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())