"""test_gen_F_fb -- FLASH_BOOT mock sim: validate the REAL silicium path bit-exact.

gen_seq_top is compiled with -DFLASH_BOOT + models/flash_boot_sim.v (drop-in mock of
flash_boot). The mock pulses boot_done ~10 cycles after node_top auto-pulses
boot_start at reset (SDRAM is already preloaded via $readmemh SDRAM_PRELOAD, so the
mock skips the DMA). The cache-fill FSM in node_top then wakes on boot_done and
DMA's the NC FFN layers SDRAM -> BRAM (the silicium path, cache ACTIF).

Flow:
  1. reset, drive_shifts (step-1 weight shifts via the sw_* ports).
  2. wait for dut.u_n.wcache_fill_done ( RisingEdge ) -> cache BRAM filled.
  3. pulse gen_seq start (gen_mode=1, prompt_len=0 -> BOS prefix hardwired in gen_seq).
  4. collect 17 tokens on RisingEdge(dut.token_valid) (fast : no per-cycle python loop).
  5. compare vs m['node_tokens'] (oracle prove_gen_node.py bit-exact with silicium RTL).

Two watchdogs (fill + gen) abort with a clear message if the sim stalls, instead of
hanging the whole run. ASCII-only (no accents) to avoid iverilog/cocotb encoding pitfalls.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, First
import json, os

N_TOK = int(os.environ.get("NTOK", "17"))
NL = 5   # surchargé depuis m['cfg']['n_layers'] au chargement du dump
DUMP = os.environ.get("DUMP", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "host", "gen_model_dump.json"))

FILL_TIMEOUT_CYC = 4_000_000   # ~4M cyc budget for cache fill (NC*36864 SDRAM reads)
GEN_TIMEOUT_CYC = 35_000_000   # ~35M cyc budget for 17 tokens (~1.65M cyc/tok)


def pack(vals):
    return sum((v & 0xFF) << (i * 8) for i, v in enumerate(vals))


def drive_shifts(dut, m):
    key = dict(sw_rms='rms_att', swq='wq', swk='wk', swv='wv', swo='wo',
               sw_rmsf='rms_ffn', sw1='w1', sw3='w3', sw2='w2')
    for port, wk in key.items():
        getattr(dut, port).value = pack([m['layers'][l][wk + '_s'] for l in range(NL)])
    dut.sw_rmsfinal.value = m['rms_final_s'] & 0xFF
    dut.sw_emb.value = m['tok_emb_s'] & 0xFF


def _peek_dbg(dut, idx):
    # at token_valid: peek argmax state + a few logits + XB/XN samples to locate the
    # zero-token bug (logits all 0 -> upstream XN/embed/SDRAM ; logits non-zero ->
    # argmax or shift re-align). Called only for the first 3 tokens (limit log noise).
    def g(name):
        try:
            return int(getattr(dut.u_seq, name).value)
        except Exception:
            return None
    def glog(i):
        for expr in (lambda: int(dut.u_seq.logits[i].value),
                     lambda: int(dut.u_seq.logits[i]),
                     lambda: int(dut._id("logits", extended=False)[i].value)):
            try:
                return expr()
            except Exception:
                pass
        return None
    def gvfile(slot, i):
        idx = slot * 64 + i
        for expr in (lambda: int(dut.u_seq.vfile[idx].value),
                     lambda: int(dut.u_seq.vfile[idx]),
                     lambda: int(dut._id("vfile", extended=False)[idx].value)):
            try:
                return expr()
            except Exception:
                pass
        return None
    best_val = g("best_val")
    best_idx = g("best_idx")
    sref_lm = g("sref_lm")
    sx0 = g("sx0")
    sxn = g("sxn")
    sxb = g("sxb")
    logits = [glog(i) for i in (0, 1, 2, 63, 100, 200, 261, 268, 298, 403, 414, 511)]
    xb = [gvfile(0, i) for i in (0, 1, 2, 3, 4, 5, 6, 7)]   # SB_XB=0
    xn = [gvfile(1, i) for i in (0, 1, 2, 3, 4, 5, 6, 7)]   # SB_XN=1
    def gcsh(i):
        for expr in (lambda: int(dut.u_seq.csh[i].value),
                     lambda: int(dut.u_seq.csh[i]),
                     lambda: int(dut.u_seq._id("csh", extended=False)[i].value)):
            try:
                return expr()
            except Exception:
                pass
        return None
    csh = [gcsh(i) for i in range(8)]
    def s8(v):
        return None if v is None else (v - 256 if v & 0x80 else v)
    # logits re-alignés (v = logits[i] >> (sref_lm - csh[i>>6])) comme l'oracle argmax
    def ralign(tok_i):
        lg = logits[[0,1,2,63,100,200,261,268,298,403,414,511].index(tok_i)]
        ch = csh[tok_i >> 6]
        if lg is None or ch is None or sref_lm is None:
            return None
        sh = sref_lm - ch
        return s8(lg) if sh == 0 else (s8(lg) >> sh if sh > 0 else s8(lg) << (-sh))
    dut._log.info(f"[dbg tok{idx}] best_val={best_val} best_idx={best_idx} "
                  f"sref_lm={sref_lm} sx0={s8(sx0)} sxn={s8(sxn)} sxb={s8(sxb)}")
    dut._log.info(f"[dbg tok{idx}] logits@{[0,1,2,63,100,200,261,268,298,403,414,511]} = "
                  f"{[s8(v) for v in logits]}")
    dut._log.info(f"[dbg tok{idx}] csh={csh}  ralign[268]={ralign(268)} ralign[298]={ralign(298)} "
                  f"ralign[261]={ralign(261)} ralign[414]={ralign(414)}")
    dut._log.info(f"[dbg tok{idx}] XB[0:8]={[s8(v) for v in xb]}  "
                  f"XN[0:8]={[s8(v) for v in xn]}  (XN=0 -> lm_head logits=0 -> token 0)")


@cocotb.test()
async def test_gen_F_fb(dut):
    global NL
    with open(DUMP) as f:
        m = json.load(f)
    NL = int(m['cfg']['n_layers'])
    expect = m['node_tokens'][:N_TOK]
    assert len(expect) == N_TOK, f"oracle has {len(m['node_tokens'])} tokens, need {N_TOK}"

    cocotb.start_soon(Clock(dut.clk, 37, unit="ns").start())
    dut.rst_n.value = 0
    for s in ("start", "gen_mode", "x_in", "sx_in", "sw_rms", "swq", "swk", "swv",
              "swo", "sw_rmsf", "sw1", "sw3", "sw2", "sw_rmsfinal", "sw_emb", "pos"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)
    drive_shifts(dut, m)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 20)

    # sanity : boot_done from the mock should rise within ~15 cyc after reset
    boot_done = -1
    try:
        boot_done = int(dut.u_n.boot_done.value)
    except Exception:
        pass
    fill_done = -1
    try:
        fill_done = int(dut.u_n.wcache_fill_done.value)
    except Exception:
        pass
    dut._log.info(f"[init] boot_done={boot_done} wcache_fill_done={fill_done} "
                  f"(expect boot_done=0, fill_done=0 right after reset)")

    # ---- phase 1 : wait for the cache FFN BRAM fill (the silicium path) ----
    dut._log.info("[fill] waiting for dut.u_n.wcache_fill_done (cache FFN BRAM fill)...")

    async def fill_watchdog():
        await ClockCycles(dut.clk, FILL_TIMEOUT_CYC)
        return "fill_timeout"

    wd = cocotb.start_soon(fill_watchdog())
    await First(RisingEdge(dut.u_n.wcache_fill_done), wd.join())
    wd.kill()
    fill_done = -1
    try:
        fill_done = int(dut.u_n.wcache_fill_done.value)
    except Exception:
        pass
    if fill_done != 1:
        dut._log.error(f"[fill] TIMEOUT after {FILL_TIMEOUT_CYC} cyc, fill_done={fill_done}")
        assert False, "wcache_fill_done never asserted (cache fill did not complete)"
    dut._log.info("[fill] cache FFN BRAM filled -> silicium path active, starting gen")

    # ---- phase 2 : start generation (BOS prefix hardwired in gen_seq for prompt_len=0) ----
    dut.gen_mode.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    # ---- phase 3 : collect 17 tokens on RisingEdge(dut.token_valid) ----
    got = []

    async def collect_all():
        while len(got) < N_TOK:
            await RisingEdge(dut.token_valid)
            tok = int(dut.token.value) & 0x3FF
            got.append(tok)
            idx = len(got) - 1
            tag = "OK" if tok == expect[idx] else "MISMATCH"
            dut._log.info(f"[tok {idx:2d}] = {tok:4d}  (expect {expect[idx]:4d})  {tag}")
            if idx <= 9:
                _peek_dbg(dut, idx)

    async def gen_watchdog():
        await ClockCycles(dut.clk, GEN_TIMEOUT_CYC)
        return "gen_timeout"

    cw = cocotb.start_soon(collect_all())
    gw = cocotb.start_soon(gen_watchdog())
    await First(cw.join(), gw.join())
    cw.kill(); gw.kill()

    dut._log.info(f"[result] got = {got}")
    dut._log.info(f"[result] exp = {expect}")
    assert len(got) == N_TOK, f"only {len(got)}/{N_TOK} tokens produced (gen timeout?)"
    for i, (g, e) in enumerate(zip(got, expect)):
        assert g == e, f"token {i}: got {g} expect {e}"
    dut._log.info("[PASS] 17/17 bit-exact vs node_tokens (FLASH_BOOT silicium path)")