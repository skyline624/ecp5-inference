#!/usr/bin/env python3
# =============================================================================
# diag_g5_readback.py - Diagnostique silicium du bug "17 tokens = 0" de G5'.
#
# Hypothese : sur silicium, gen_seq tourne (~0.15 s/token) mais argmax=0 -> logits
# nuls. Soit (a) les ecritures LL ne collent pas en SDRAM (SDRAM vide -> fetch=0),
# soit (b) les ecritures collent mais gen_seq ne lit pas la SDRAM correctement.
#
# Ce script localise :
#   1. Roundtrip connu : LL d'un motif connu -> CC readback -> compare.  Valide le
#      chemin LL+CC du wrapper gen_seq_silicon sur silicium (pass-through + drain).
#   2. Poids reels : charge stories260K via quantize_and_load_weights (LL), puis
#      CC readback de quelques adresses de poids connues -> compare aux octets
#      attendus.  Si OK, l'ecriture SDRAM colle -> le bug vient de la lecture gen_seq.
#
# Reutilise sd_load / sd_dump (CC readback) de test_sdram_diag.py : le wrapper
# forward LL/CC en pass-through et draine 2 (LK) ou 2+N (CK+data), donc sd_dump
# fonctionne a travers le wrapper tel quel.
# =============================================================================

import os, sys, time
import numpy as np
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_sdram_diag import sd_load, sd_dump, addr_bytes, n_bytes, i8
from infer_fpga import load_model, quantize_and_load_weights, MODEL

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000


def main():
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # ---- 1. Roundtrip motif connu (valide LL+CC a travers le wrapper) ----
    print("=== Test 1 : roundtrip motif connu LL -> CC (64 octets) ===")
    pattern = bytes((i * 7 + 3) & 0xFF for i in range(64))   # motif non trivial
    ok = sd_load(ser, 0x100000, pattern)
    print(f"  LL load ack LK : {ok}")
    got = sd_dump(ser, 0x100000, 64)
    if got is None:
        print("  FAIL : CC readback pas de reponse CK")
    else:
        match = sum(1 for a, b in zip(got, pattern) if a == b)
        print(f"  CC readback : {match}/64 octets correspondent")
        if got == pattern:
            print("  [OK] chemin LL+CC du wrapper valide sur silicium")
        else:
            print(f"  got[:16] = {got[:16].hex()}")
            print(f"  exp[:16] = {pattern[:16].hex()}")
    print()

    # ---- 2. Poids reels : charge stories260K puis readback ----
    print("=== Test 2 : charge poids stories260K (LL) puis CC readback ===")
    print(f"  Chargement modele : {MODEL}")
    m = load_model(MODEL)
    t0 = time.time()
    w = quantize_and_load_weights(ser, m)   # charge via LL + enregistre les shifts
    print(f"  -> poids charges en {time.time()-t0:.1f}s")

    # quantize_and_load_weights a charge les poids i8 en SDRAM. On readback
    # quelques zones : tok_emb ligne 0 (addr 0x000000), et wq couche 0
    # (0x010000 + 0x100 = 0x010100). On compare aux octets i8 attendus.
    # NB : les octets attendus sont ceux que quantize_and_load_weights a ecrit ;
    # on les recupere depuis les donnees quantisees qu'il a utilisees. Faute d'un
    # retour direct, on readback et on verifie juste qu'ils sont non nuls et
    # stables (deux readbacks identiques).
    print("  Readback tok_emb[0..15] @ 0x000000 :")
    g1 = sd_dump(ser, 0x000000, 16)
    print(f"    {g1.hex() if g1 else 'AUCUNE REPONSE'}")
    print("  Readback wq couche 0[0..15] @ 0x010100 :")
    g2 = sd_dump(ser, 0x010100, 16)
    print(f"    {g2.hex() if g2 else 'AUCUNE REPONSE'}")

    # stabilite : deux readbacks identiques ?
    g1b = sd_dump(ser, 0x000000, 16)
    g2b = sd_dump(ser, 0x010100, 16)
    print(f"  stabilite tok_emb : {'OK' if g1 == g1b else 'INSTABLE'}")
    print(f"  stabilite wq       : {'OK' if g2 == g2b else 'INSTABLE'}")

    if g1 and g2 and any(b != 0 for b in g1) and any(b != 0 for b in g2):
        print("\n  [OK] SDRAM non vide -> ecriture LL colle. Bug token=0 vient de")
        print("       la lecture/fetch par gen_seq, pas de l'ecriture.")
    else:
        print("\n  [FAIL] SDRAM read back a 0/vide -> ecriture LL ne colle pas a")
        print("         travers le wrapper (ou readback casse).")

    ser.close()


if __name__ == "__main__":
    main()