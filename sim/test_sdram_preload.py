"""Diagnostic : preload_model ecrit-il vraiment dans dut.u_sdram.mem ?

Preload le dump (comme test_gen_F2) puis relit dut.u_sdram.mem[0x4000]
(= SDRAM octet 0x010000 >> 2 = rms_att layer 0). Si la valeur lue == les 4
premiers octets de m['layers'][0]['rms_att'] (packed LE), cocotb ecrit bien.
Sinon, le preload VPI sur le grand tableau `reg [31:0] mem [0:2M]` echoue.

  python -m server.sim.cocotb_runner test_sdram_preload
"""
import json
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "gen_model_dump.json")


@cocotb.test()
async def test_sdram_preload(dut):
    from test_gen_F2 import preload, preload_model
    with open(DUMP) as f:
        m = json.load(f)

    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0
    for s in ("start", "gen_mode", "x_in", "sx_in", "sw_rms", "swq", "swk", "swv",
              "swo", "sw_rmsf", "sw1", "sw3", "sw2", "sw_rmsfinal", "sw_emb", "pos"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    # rms_att layer 0 seul (rapide) : 64 octets @ 0x010000
    preload(dut.u_sdram, 0x010000, m['layers'][0]['rms_att'])

    wb = 0x010000 >> 2  # = 0x4000
    got = dut.u_sdram.mem[wb].value
    dut._log.info(f"mem[0x4000] = {got!r}")
    try:
        w0 = int(got) & 0xFFFFFFFF
    except Exception as e:
        dut._log.info(f"int(got) failed: {e}; trying signed...")
        w0 = int(got.signed_integer) & 0xFFFFFFFF

    ra = m['layers'][0]['rms_att']
    exp_w0 = ((ra[0] & 0xFF) | ((ra[1] & 0xFF) << 8) |
              ((ra[2] & 0xFF) << 16) | ((ra[3] & 0xFF) << 24))
    dut._log.info(f"expected  = {exp_w0:08x} (rms_att[0:4]={ra[0:4]})")
    ok = (w0 == exp_w0)
    dut._log.info(f"preload write {'OK' if ok else 'FAIL'}")
    assert ok, f"preload FAIL: mem[0x4000]={w0:08x} expected={exp_w0:08x}"
    dut._log.info("STEP PRELOAD PASS: cocotb ecrit bien dans dut.u_sdram.mem.")