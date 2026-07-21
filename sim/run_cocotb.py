"""run_cocotb.py -- drive cocotb+Icarus WITHOUT make (no `make` installed here).
Mirrors ecp5/sim/Makefile : same RTL sources, LINK_SS define, .hex auto-copied
into the build dir (the $readmemh relative-path trap).

Usage (from ecp5/sim/):
    python run_cocotb.py test_gen_A        # quick smoke test
    python run_cocotb.py test_gen_F        # full 17-token (~2.4h)
"""
import os, sys, shutil, glob
from cocotb_tools.runner import get_runner

HERE = os.path.dirname(os.path.abspath(__file__))
RTL  = os.path.join(HERE, "..", "src")
LUT  = os.path.join(HERE, "..", "lut")
MODELS = os.path.join(HERE, "models")
BUILD = os.path.join(HERE, "sim_build")

TOPLEVEL = "gen_seq_top"
MODULE = sys.argv[1] if len(sys.argv) > 1 else "test_gen_A"

SOURCES = [
    os.path.join(MODELS, "ECP5_PLL_sim.v"),     # provides ecp5_pll (and Gowin_rPLL) sim stub
    os.path.join(MODELS, "MULTALU18X18.v"),
    os.path.join(MODELS, "sdram_chip.v"),
    os.path.join(RTL, "mac18.v"),
    os.path.join(RTL, "rmsnorm_op.v"),
    os.path.join(RTL, "silu_op.v"),
    os.path.join(RTL, "rope_op.v"),
    os.path.join(RTL, "softmax_op.v"),
    os.path.join(RTL, "attention_head_op.v"),
    os.path.join(RTL, "sdram.v"),
    os.path.join(RTL, "async_fifo.v"),
    os.path.join(RTL, "uart_bridge.v"),          # rx8_link/tx8_link (node LINK_SS PHY)
    os.path.join(RTL, "vec_alu2.v"),
    os.path.join(RTL, "node_top.v"),             # the LINK_SS compute node (module top)
    os.path.join(RTL, "gen_seq.v"),
    os.path.join(RTL, "gen_seq_top.v"),
]

# Copy .hex LUTs into BUILD so $readmemh("silu_lut.hex") etc. resolve there.
os.makedirs(BUILD, exist_ok=True)
for f in glob.glob(os.path.join(LUT, "*.hex")):
    shutil.copy(f, BUILD)

r = get_runner("icarus")
r.build(
    sources=SOURCES,
    hdl_toplevel=TOPLEVEL,
    build_dir=BUILD,
    always=True,                       # rebuild each run (cheap, avoids stale)
    timescale=("1ns","1ps"),
    includes=[RTL],
    build_args=["-DLINK_SS"],
)
r.test(
    hdl_toplevel=TOPLEVEL,
    test_module=MODULE,
    build_dir=BUILD,
    test_dir=BUILD,
    waves=False,
)