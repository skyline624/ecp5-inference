"""Diagnostic rapide : confirme via $display RTL (dans sdram_chip.v) que
l'image SDRAM sdram_preload.hex est chargée au temps 0. Ne fait AUCUN forward
(10 cycles puis exit) -> build+sim en ~2-3 min au lieu de 9.

  python -m server.sim.cocotb_runner test_preload_check
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles


@cocotb.test()
async def test_preload_check(dut):
    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 10)
    dut._log.info("preload_check: 10 cycles faits (les $display sdram_chip.v "
                  "sont dans le log ci-dessus).")