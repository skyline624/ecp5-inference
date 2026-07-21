"""sim/tb_registry.py — liste les testbenches disponibles (cocotb + Verilog pur).

Deux familles :
- cocotb : sim/test_*.py (test_gen_A..F2, attention, ring, ...) via cocotb_runner
  (waves=True). NB : le node complet est cassé en icarus depuis les étapes 2a/3
  (forward refs mac_*/eff_ref que yosys tolère mais pas icarus). Ces tests
  peuvent donc échouer à la compile — régression à corriger séparément.
- iverilog : TB Verilog pur (dump_tb.v) instanciant des modules feuilles
  (wcache_ffn) -> compile en icarus strict, VCD simple. Recommandé pour la Phase B/C.
"""
from __future__ import annotations
from typing import List

from .. import config

SIM_DIR = config.SIM_DIR


# TB Verilog purs (iverilog_runner). Recommandés (compilent en icarus strict).
IVERILOG_TBS = [
    {
        "name": "dump_tb_wcache",
        "mode": "iverilog",
        "desc": "wcache_ffn write/read (16 octets) — VCD simple, module feuille",
        "duration": "instantané",
    },
]


# Tests cocotb (cocotb_runner). ATTENTION : peuvent échouer à la compile icarus
# (régression étapes 2a/3 sur node_top.v). Exposés pour quand la régression sera
# corrigée.
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