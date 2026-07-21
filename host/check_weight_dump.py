#!/usr/bin/env python3
# compare_weight_dump.py -- Compare les octets SDRAM dumpes en silicium (diag_g5_readback)
# aux valeurs int8 attendues du modele stories260K, pour determiner si le bug
# "17 tokens = 0" vient d'un SDRAM corrompu (load/shadow-word/refresh) ou d'une
# lecture/fetch/calcul errone.
#
# Lit le modele, quantize tok_emb et wq[0] (to_i8_shift), et affiche les 16
# premiers octets @ 0x000000 (tok_emb) et @ 0x010100 (wq couche 0). Comparer aux
# readback silicium.

import os, sys, struct
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, MODEL
from v4_quant import to_i8_shift

# Readback silicium (issu de diag_g5_readback.py sur la i9, apres LL load poids)
SILICON_TOK_EMB_0 = bytes.fromhex("ed240b16f90f0102030ef4e012ecf312")  # @ 0x000000
SILICON_WQ_L0     = bytes.fromhex("0305ff0102fd02fff2edfe010808f602")  # @ 0x010100


def main():
    m = load_model(MODEL)
    cfg = m['cfg']
    print(f"modele : dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}\n")

    # tok_emb [V, D] -> i8 -> flatten -> 16 premiers octets @ 0x000000
    tok_i8, sh_emb = to_i8_shift(m['tok_emb'])
    exp_tok = bytes((tok_i8.reshape(-1)[:16].astype(np.int8).astype(np.uint8)).tolist())
    print(f"Attendu tok_emb @ 0x000000 : {exp_tok.hex()}")
    print(f"Silicon tok_emb @ 0x000000 : {SILICON_TOK_EMB_0.hex()}")
    print(f"  match {min(16,len(SILICON_TOK_EMB_0))}/16 : {exp_tok[:16] == SILICON_TOK_EMB_0[:16]}")
    # detail octet par octet
    for i in range(min(16, len(SILICON_TOK_EMB_0))):
        e = exp_tok[i]; s = SILICON_TOK_EMB_0[i]
        tag = "OK" if e == s else "DIFF"
        ei = e - 256 if e >= 128 else e
        si = s - 256 if s >= 128 else s
        print(f"    [{i:2d}] attendu={ei:4d} silicon={si:4d}  {tag}")
    print()

    # wq couche 0 [Hh*HSh, D] -> i8 -> flatten -> 16 premiers octets @ 0x010100
    wq_i8, sh_q = to_i8_shift(m['wq'][0])
    exp_wq = bytes((wq_i8.reshape(-1)[:16].astype(np.int8).astype(np.uint8)).tolist())
    print(f"Attendu wq L0  @ 0x010100 : {exp_wq.hex()}  (sh_q={sh_q})")
    print(f"Silicon wq L0  @ 0x010100 : {SILICON_WQ_L0.hex()}")
    print(f"  match {min(16,len(SILICON_WQ_L0))}/16 : {exp_wq[:16] == SILICON_WQ_L0[:16]}")
    for i in range(min(16, len(SILICON_WQ_L0))):
        e = exp_wq[i]; s = SILICON_WQ_L0[i]
        tag = "OK" if e == s else "DIFF"
        ei = e - 256 if e >= 128 else e
        si = s - 256 if s >= 128 else s
        print(f"    [{i:2d}] attendu={ei:4d} silicon={si:4d}  {tag}")


if __name__ == "__main__":
    main()