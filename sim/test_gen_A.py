"""Step A gate : gen_seq embedding -> first generated token (prefill 1 token).

The old step-A drove an internal `cur_tok` port and read `xb_out` (vfile[XB]) to
isolate the embedding lookup. Both signals were internalised (cur_tok is now a
reg, vfile is LUT-RAM) and the embedding is reached only through the prefill
path : gen_mode=1 with a prompt of length 1 -> the sequencer embeds prompt[0]
(EE reads SDRAM[tok*64], tok_emb preloaded via $readmemh), runs the 5 layers +
lm_head + argmax, and emits the first generated token.

So this gate now validates the FULL path embed -> 5 layers -> lm_head -> argmax
on ONE token (fast smoke of the prefill mechanism + the prompt port), not the
embedding in isolation. Token-by-token isolation is covered by test_gen_F/F2.

Poids préchargés via $readmemh("sdram_preload.hex") (define -DSDRAM_PRELOAD) ;
reference = dump['node_tokens'][0] (bit-exact node model host/prove_gen_node.py).

  make MODULE=test_gen_A
"""
import json
import os
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, with_timeout

N_TOK = 1
D, H, KH, HS, HID, NL, VOCAB = 64, 8, 4, 8, 172, 5, 512
NPR = 16
LBASE, LSTRIDE = 0x010000, 0x010000
DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "host", "gen_model_dump.json")


def pack(vals): return sum((v & 0xFF) << (i*8) for i, v in enumerate(vals))


def drive_shifts(dut, m):
    key = dict(sw_rms='rms_att', swq='wq', swk='wk', swv='wv', swo='wo',
               sw_rmsf='rms_ffn', sw1='w1', sw3='w3', sw2='w2')
    for port, wk in key.items():
        getattr(dut, port).value = pack([m['layers'][l][wk + '_s'] for l in range(NL)])
    dut.sw_rmsfinal.value = m['rms_final_s'] & 0xFF
    dut.sw_emb.value = m['tok_emb_s'] & 0xFF


@cocotb.test()
async def test_gen_A(dut):
    with open(DUMP) as f:
        m = json.load(f)
    expected = m['node_tokens'][:N_TOK]          # node-faithful reference (prove_gen_node)

    cocotb.start_soon(Clock(dut.clk, 37, units="ns").start())
    dut.rst_n.value = 0
    for s in ("start", "gen_mode", "x_in", "sx_in", "sw_rms", "swq", "swk", "swv", "swo",
              "sw_rmsf", "sw1", "sw3", "sw2", "sw_rmsfinal", "sw_emb", "pos",
              "prompt", "prompt_len"):
        getattr(dut, s).value = 0
    await ClockCycles(dut.clk, 10)

    # Poids préchargés via $readmemh("sdram_preload.hex") dans sdram_chip.v.
    drive_shifts(dut, m)

    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 5)
    dut.gen_mode.value = 1
    # Prefill 1 token = BOS (token 1). Prompt port exercé : prompt_len=1, prompt[0]=1.
    # (prompt_len=0 -> BOS hardcodé côté gen_seq donne le même token 0.)
    dut.prompt_len.value = 1
    dut.prompt.value = 1          # BOS, dans les 10 bits de poids faible
    dut.start.value = 1
    await RisingEdge(dut.clk); dut.start.value = 0

    got = []
    for i in range(N_TOK):
        await with_timeout(RisingEdge(dut.token_valid), 400_000_000, "ns")
        got.append(int(dut.token.value))
        dut._log.info(f"  token {i} : rtl={got[-1]}  expected={expected[i]}")

    dut._log.info(f"  got={got}  expected={expected}")
    assert got == expected, f"step A token mismatch : got={got} expected={expected}"
    dut._log.info("STEP A PASS: prefill 1 token (embed -> 5 layers -> lm_head -> argmax) "
                  "via le port prompt ; premier token == reference node.")