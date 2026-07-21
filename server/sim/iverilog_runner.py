"""sim/iverilog_runner.py — compile+run un TB Verilog pur avec iverilog/vvp (VCD).

Alternative légère au runner cocotb (qui depends du node complet, cassé en icarus
depuis les étapes 2a/3 : forward refs mac_*/eff_ref). On instancie ici des modules
feuilles qui compilent en icarus strict (wcache_ffn, mac18, rmsnorm_op, ...).

Usage : python -m server.sim.iverilog_runner <tb_name>
Imprime `VCD:<path>` sur stdout (récupéré par runner.py).
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

from .. import env, config

env.setup()

HERE = Path(__file__).resolve().parent
RTL = config.SRC_DIR
WORK = config.WORK_DIR

# TB Verilog purs disponibles. Chaque entrée : sources relatives à src/, top du TB.
TBS = {
    "dump_tb_wcache": {
        "sources": ["wcache_ffn.v"],
        "tb": "dump_tb.v",
        "top": "dump_tb",
    },
}


def _src(p: str) -> str:
    return str(RTL / p)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in TBS:
        print(f"usage: iverilog_runner <{'|'.join(TBS)}>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    spec = TBS[name]
    WORK.mkdir(parents=True, exist_ok=True)

    tb_path = HERE / spec["tb"]
    sources = [_src(s) for s in spec["sources"]] + [str(tb_path)]
    vvp = WORK / f"{name}.vvp"

    # 1) compile iverilog (icarus, SystemVerilog 2012)
    cmd = ["iverilog", "-g2012", "-o", str(vvp)] + sources
    comp = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if comp.returncode != 0:
        print("IVERILOG:COMPILE_FAIL", file=sys.stderr)
        sys.stderr.write(comp.stdout + comp.stderr)
        return 1

    # 2) run vvp depuis WORK pour que $dumpfile("dump_tb.vcd") écrive dans WORK
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

    vcd = WORK / "dump_tb.vcd"
    if vcd.exists():
        print(f"VCD:{vcd}")
        return 0
    print("VCD:NONE", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())