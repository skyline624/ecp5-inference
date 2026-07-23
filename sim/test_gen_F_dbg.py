"""test_gen_F_dbg -- debug rtl=0: trace gen_seq<->node + forward node_top non-FLASH_BOOT
to locate where 0/X appears. Short sim (3M cycles, ~30s real): periodic peek of
FSM/states, ss_link activity, sd_addr/dout, obuf/xbuf samples, argmax.
Goal: pinpoint the bug in the non-FLASH_BOOT path (memories sim-path-vs-silicon-path /
cocotb-path-sim-refute).
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import json, os

N_TOK = 17
D, H, KH, HS, HID, NL, VOCAB = 64, 8, 4, 8, 172, 5, 512
LBASE, LSTRIDE = 0x010000, 0x010000
A_RMSFINAL, A_EMB = 0x060000, 0x000000
OFF = dict(rms_att=0x0000, wq=0x0100, wk=0x1100, wv=0x1900, wo=0x2100,
           rms_ffn=0x3100, w1=0x3200, w3=0x6200, w2=0x9200)
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "gen_model_dump.json")


def pack(vals): return sum((v & 0xFF) << (i*8) for i, v in enumerate(vals))


def drive_shifts(dut, m):
    key = dict(sw_rms='rms_att', swq='wq', swk='wk', swv='wv', swo='wo',
               sw_rmsf='rms_ffn', sw1='w1', sw3='w3', sw2='w2')
    for port, wk in key.items():
        getattr(dut, port).value = pack([m['layers'][l][wk + '_s'] for l in range(NL)])
    dut.sw_rmsfinal.value = m['rms_final_s'] & 0xFF
    dut.sw_emb.value = m['tok_emb_s'] & 0xFF


def gv(dut, path, bits=64):
    # safe peek of a signal by hierarchical path; returns int or -1 if inaccessible
    try:
        h = dut
        for p in path.split("."):
            h = getattr(h, p)
        v = int(h.value)
        return v & ((1 << bits) - 1) if bits < 64 else v
    except Exception:
        return -1


@cocotb.test()
async def test_gen_F_dbg(dut):
    with open(DUMP) as f:
        m = json.load(f)
    cocotb.start_soon(Clock(dut.clk, 37, unit="ns").start())
    dut.rst_n.value = 0
    for s in ("start", "gen_mode", "x_in", "sx_in", "sw_rms", "swq", "swk", "swv", "swo",
              "sw_rmsf", "sw1", "sw3", "sw2", "sw_rmsfinal", "sw_emb", "pos"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)
    drive_shifts(dut, m)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.gen_mode.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # sanity: is sdram_chip.mem loaded? (BOS = tok_emb[1] @ A_EMB + 1*64)
    bos_word = gv(dut, "u_sdram.mem[64]", 32)
    dut._log.info(f"[init] sdram_chip.mem[64] (BOS byte0) = {bos_word:#x}  (expect non-0: tok_emb[1][0])")
    sw_rms = gv(dut, "sw_rms", 64)
    dut._log.info(f"[init] sw_rms[0..7] = {sw_rms:#x}  (expect non-0: 5-layer shifts)")

    for step in range(120):  # 120 * 25k = 3M cycles (~111M ns)
        await ClockCycles(dut.clk, 25000)
        seq_st   = gv(dut, "u_seq.st", 6)
        seq_ph   = gv(dut, "u_seq.phase", 4)
        seq_cur  = gv(dut, "u_seq.cur_tok", 10)
        cmd_wr   = gv(dut, "u_seq.s_cmd_wr", 1)
        resp_em  = gv(dut, "u_seq.s_resp_empty", 1)
        n_state  = gv(dut, "u_n.state", 9)
        n_rxp    = gv(dut, "u_n.rx_pending", 1)
        n_sdaddr = gv(dut, "u_n.sd_addr", 23)
        n_sddout = gv(dut, "u_n.sd_dout", 8)
        try:
            ob0 = int(dut.u_n.obuf[0].value)
        except Exception:
            ob0 = -2
        try:
            xb0 = int(dut.u_n.xbuf[0].value)
        except Exception:
            xb0 = -2
        tok  = gv(dut, "token", 10)
        tokv = gv(dut, "token_valid", 1)
        dut._log.info(f"step{step:3d}: seq.st={seq_st:2d} ph={seq_ph} cur={seq_cur:3d} "
                      f"cmdwr={cmd_wr} respem={resp_em} | n.state={n_state:3d} rxp={n_rxp} "
                      f"sd_addr={n_sdaddr:#x} sd_dout={n_sddout:3d} | obuf[0]={ob0:4d} "
                      f"xbuf[0]={xb0:4d} | tok={tok:3d} tv={tokv}")
        if tokv:
            dut._log.info(f"[token_valid] token0 = {tok}  (trace stopped)")
            break
    dut._log.info("[dbg] end of trace (3M cycle budget exhausted or token0 out)")