"""sim/tb_registry.py — liste les testbenches disponibles (cocotb + Verilog pur).

Deux familles :
- iverilog : TB Verilog pur (dump_tb.v, tb_node.v) -> compile en icarus strict,
  VCD natif via $dumpvars. dump_tb_wcache = module feuille (VCD simple, Phase B) ;
  tb_node = gen_seq_top node complet (VCD overlay animé Phase C).
- cocotb : sim/test_*.py (test_gen_A..F2, attention, ring, ...) via cocotb_runner.
  La compile icarus du node est rétablie depuis e1b668e ; test_gen_F/F2 tournent
  (rtl=0 en non-FLASH_BOOT, cf mémoire sim-path-vs-silicon-path).
"""
from __future__ import annotations
from typing import List

from .. import config

SIM_DIR = config.SIM_DIR


# TB Verilog purs (iverilog_runner). Compilent en icarus strict, VCD natif.
IVERILOG_TBS = [
    {
        "name": "dump_tb_wcache",
        "mode": "iverilog",
        "desc": "wcache_ffn write/read (16 octets) — VCD simple, module feuille",
        "duration": "instantané",
    },
    {
        "name": "tb_node",
        "mode": "iverilog",
        "desc": "gen_seq_top node complet — VCD du parcours du signal (overlay animé Phase C)",
        "duration": "~60k cycles (quelques sec)",
    },
]


# Tests cocotb (cocotb_runner). Compile icarus rétablie (e1b668e) ; test_gen_F/F2
# tournent (rtl=0 en non-FLASH_BOOT, path != path silicium — cf sim-path-vs-silicon-path).
COCOTB_TESTS = [
    {"name": "test_gen_A", "mode": "cocotb", "desc": "embedding EE (tok_emb) — très court", "duration": "court"},
    {"name": "test_gen_B", "mode": "cocotb", "desc": "RMSNorm FFN", "duration": "court"},
    {"name": "test_gen_C", "mode": "cocotb", "desc": "matmul FFN W1", "duration": "court"},
    {"name": "test_gen_D", "mode": "cocotb", "desc": "SiLU + W2", "duration": "court"},
    {"name": "test_gen_E", "mode": "cocotb", "desc": "attention", "duration": "court"},
    {"name": "test_gen_F", "mode": "cocotb", "desc": "full 17 tokens", "duration": "long (~2.4h)"},
    {"name": "test_vec_alu2", "mode": "cocotb", "desc": "ALU elementwise", "duration": "court"},
    {"name": "test_attn_tp", "mode": "cocotb", "desc": "attention tensor-parallel", "duration": "court"},
    {"name": "test_ring", "mode": "cocotb", "desc": "ring cluster all-gather", "duration": "court"},
    {"name": "test_ss_link", "mode": "cocotb", "desc": "ss_link inter-FPGA", "duration": "court"},
]


def list_cocotb_tests() -> List[dict]:
    available = {p.stem for p in SIM_DIR.glob("test_*.py")}
    out = []
    for t in COCOTB_TESTS:
        t = dict(t)
        t["available"] = t["name"] in available
        out.append(t)
    return out


def list_iverilog_tbs() -> List[dict]:
    return [dict(t) for t in IVERILOG_TBS]


def list_all() -> List[dict]:
    return list_iverilog_tbs() + list_cocotb_tests()


def find(test_name: str):
    for t in list_all():
        if t["name"] == test_name:
            return t
    return None


def is_known(test_name: str) -> bool:
    return find(test_name) is not None