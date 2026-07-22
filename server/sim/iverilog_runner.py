"""sim/iverilog_runner.py — compile+run un TB Verilog pur avec iverilog/vvp (VCD).

Usage : python -m server.sim.iverilog_runner <tb_name>
Imprime `VCD:<path>` sur stdout (récupéré par runner.py).

Champs d'une entrée TBS :
- sources : fichiers .v relatifs à src/ (RTL du design).
- models  : fichiers .v relatifs à sim/models/ (mocks : sdram_chip, PLL, ...).
- tb      : fichier .v du TB, relatif à server/sim/.
- top     : nom du module TB (pour -s).
- defines : liste de macros -D (ex. ["LINK_SS","SDRAM_PRELOAD"]).
- vcd_name: nom du VCD produit (défaut "dump_tb.vcd").

Pour les TB avec SDRAM_PRELOAD, le runner copie sim/sdram_preload.hex + lut/*.hex
vers WORK avant le run vvp (vvp résout $readmemh relativement à son cwd = WORK).
"""
from __future__ import annotations
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import env, config

env.setup()

HERE = Path(__file__).resolve().parent
RTL = config.SRC_DIR
MODELS = config.SIM_DIR / "models"
LUT = config.LUT_DIR
WORK = config.WORK_DIR

# TB Verilog purs disponibles.
TBS = {
    "dump_tb_wcache": {
        "sources": ["wcache_ffn.v"],
        "tb": "dump_tb.v",
        "top": "dump_tb",
        "vcd_name": "dump_tb.vcd",
    },
    "tb_node": {
        "sources": [
            "mac18.v", "rmsnorm_op.v", "silu_op.v", "rope_op.v", "softmax_op.v",
            "attention_head_op.v", "sdram.v", "async_fifo.v", "uart_bridge.v",
            "vec_alu2.v", "node_top.v", "gen_seq.v", "gen_seq_top.v",
        ],
        "models": ["ECP5_PLL_sim.v", "MULTALU18X18.v", "sdram_chip.v"],
        "tb": "tb_node.v",
        "top": "tb_node",
        "defines": ["LINK_SS", "SDRAM_PRELOAD"],
        "vcd_name": "tb_node.vcd",
    },
}


def _src(p: str) -> str:
    return str(RTL / p)


def _model(p: str) -> str:
    return str(MODELS / p)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TBS:
        print(f"usage: iverilog_runner <{'|'.join(TBS)}>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    spec = TBS[name]
    WORK.mkdir(parents=True, exist_ok=True)

    tb_path = HERE / spec["tb"]
    sources = (
        [_src(s) for s in spec["sources"]]
        + [_model(s) for s in spec.get("models", [])]
        + [str(tb_path)]
    )
    vvp = WORK / f"{name}.vvp"

    # 1) compile iverilog (icarus, SystemVerilog 2012)
    defines = ["-D" + d for d in spec.get("defines", [])]
    cmd = ["iverilog", "-g2012", "-o", str(vvp)] + defines + sources
    comp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if comp.returncode != 0:
        print("IVERILOG:COMPILE_FAIL", file=sys.stderr)
        sys.stderr.write(comp.stdout + comp.stderr)
        return 1

    # 1b) copie les .hex (LUT + sdram_preload) vers WORK (cwd de vvp) si le TB
    # utilise $readmemh. Indispensable pour SDRAM_PRELOAD (poids du modèle) et
    # les LUT rmsnorm/silu/rope/attention. vvp résout $readmemh("x.hex")
    # relativement à son cwd = WORK.
    for f in glob.glob(str(LUT / "*.hex")):
        shutil.copy(f, WORK)
    sdram_hex = config.SIM_DIR / "sdram_preload.hex"
    if sdram_hex.exists():
        shutil.copy(sdram_hex, WORK)

    # 2) run vvp depuis WORK pour que $dumpfile écrive le VCD dans WORK
    sim = subprocess.run(
        ["vvp", str(vvp)],
        cwd=str(WORK),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.stdout.write(sim.stdout)
    sys.stderr.write(sim.stderr)

    vcd = WORK / spec.get("vcd_name", "dump_tb.vcd")
    if vcd.exists():
        print(f"VCD:{vcd}")
        return 0
    print("VCD:NONE", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())