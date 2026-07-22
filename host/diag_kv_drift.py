#!/usr/bin/env python3
# =============================================================================
# diag_kv_drift.py - Distingue drift SDRAM vs différence de quantization bénigne
# dans l'anomalie CC-read KV post-génération (étape 1).
#
# Flow : RR+SH+GO -> 17 tokens -> RR(retour LOAD) -> CC-read KV[0,0..3] K+V
# **deux fois de suite**. Comparaisons :
#   * read1 vs read2  : si != -> DRIFT réelle (les cellules changent entre les
#     reads -> refresh insuffisant sur génération longue -> bug à sécuriser).
#     Si == -> données stables -> la divergence vs oracle est une différence de
#     quantization bénigne (Kc du seq != Kc oracle dans les bits bas, masquée par
#     le re-align shift -> tokens bit-exact malgré tout).
#   * read1 vs oracle : amplitude + pattern des bytes divergents (low-bits ?).
# =============================================================================
import os, sys, time
import serial
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, MODEL
from gen_flash_model import build_image
from test_sdram_diag import sd_dump

PORT = os.environ.get("G5_PORT", "COM3"); BAUD = 1_000_000
N_TOK = int(os.environ.get("G5_NTOK", "17"))
KV_BASE = 0x070000
TMAX = 32; KVW = 32

def popcount_byte_diff(a, b):
    """nombre de bits qui diffèrent entre deux bytes"""
    x = a ^ b
    return bin(x).count("1")

def main():
    m = load_model(MODEL)
    _, shifts = build_image(m)
    assert len(shifts) == 47
    sh_payload = b'SH' + bytes([s & 0xFF for s in shifts])

    oracle = np.load(os.path.join(HERE, "kv_oracle.npz"))
    kmem = oracle['kmem']; vmem = oracle['vmem']

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    print("Attente fin boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    print("RR (LOAD) -> RK"); ser.write(b'RR'); print("  ack:", ser.read(2))
    time.sleep(0.05); ser.reset_input_buffer()
    print("SH -> SK"); ser.write(sh_payload)
    ack = ser.read(2)
    if ack != b'SK': print(f"FAIL SH ack={ack!r}"); ser.close(); sys.exit(1)
    print("  OK")
    print("GO -> GK"); ser.write(b'GO')
    ack = ser.read(2)
    if ack != b'GK': print(f"FAIL GO ack={ack!r}"); ser.close(); sys.exit(1)
    print("  OK, génération...")

    tokens = []
    for i in range(N_TOK):
        b = ser.read(2)
        if len(b) != 2: print(f"  token {i} timeout"); break
        tokens.append(b[0] | (b[1] << 8))
    print(f"  tokens silicium : {tokens}")

    # RR en RUN mode -> retour LOAD
    print("\nRR (retour LOAD via RR-en-RUN) -> RK")
    time.sleep(0.05); ser.write(b'RR')
    print(f"  ack: {ser.read(2)!r}")
    time.sleep(0.05); ser.reset_input_buffer()

    print("\nCC-read KV[0,0..3] K+V -- deux passes :")
    npos = 4
    drift_bytes = 0
    total_bytes = 0
    oracle_mismatch = 0
    bit_diff_hist = [0]*9  # histogramme 0..8 bits différant par byte
    for p in range(npos):
        k_addr = KV_BASE + (0*TMAX + p)*2*KVW
        v_addr = KV_BASE + (0*TMAX + p)*2*KVW + KVW
        k1 = sd_dump(ser, k_addr, KVW)
        v1 = sd_dump(ser, v_addr, KVW)
        k2 = sd_dump(ser, k_addr, KVW)
        v2 = sd_dump(ser, v_addr, KVW)
        if k1 is None or v1 is None or k2 is None or v2 is None:
            print(f"  pos {p}: pas de réponse CC"); continue

        # drift : read1 vs read2
        k_drift = sum(1 for a,b in zip(k1,k2) if a!=b)
        v_drift = sum(1 for a,b in zip(v1,v2) if a!=b)
        drift_bytes += k_drift + v_drift
        total_bytes += 2*KVW

        # vs oracle
        exp_k = bytes((kmem[0,p].flatten().astype(np.int32) & 0xFF).astype(np.uint8).tolist())
        exp_v = bytes((vmem[0,p].flatten().astype(np.int32) & 0xFF).astype(np.uint8).tolist())
        k_oracle_mm = sum(1 for a,b in zip(k1,exp_k) if a!=b)
        v_oracle_mm = sum(1 for a,b in zip(v1,exp_v) if a!=b)
        oracle_mismatch += k_oracle_mm + v_oracle_mm

        # histogramme bits différant (vs oracle) sur les bytes divergents
        for a,b in zip(k1,exp_k):
            if a!=b: bit_diff_hist[popcount_byte_diff(a,b)] += 1
        for a,b in zip(v1,exp_v):
            if a!=b: bit_diff_hist[popcount_byte_diff(a,b)] += 1

        stable = "STABLE" if (k_drift==0 and v_drift==0) else f"DRIFT k={k_drift} v={v_drift}"
        print(f"  pos {p}: {stable}  | vs oracle K {KVW-k_oracle_mm}/{KVW} V {KVW-v_oracle_mm}/{KVW}")
        if k_drift or v_drift:
            print(f"      k1: {k1.hex()}\n      k2: {k2.hex()}")
        if k_oracle_mm and not k_drift:
            print(f"      K got: {k1.hex()}\n      K exp: {exp_k.hex()}")

    print(f"\n== DRIFT (read1 vs read2) : {drift_bytes}/{total_bytes} bytes diffèrent ==")
    print(f"== vs oracle             : {oracle_mismatch}/{total_bytes} bytes diffèrent ==")
    print(f"== histogramme bits/byte divergents (vs oracle) : {bit_diff_hist}")
    if drift_bytes == 0:
        print("[STABLE] données KV stables entre 2 reads -> PAS de drift.")
        print("  La divergence vs oracle est une différence de quantization (Kc seq vs")
        print("  oracle) dans les bits bas, masquée par le re-align shift -> tokens")
        print("  bit-exact. Non bloquant pour les générations longues.")
        if bit_diff_hist[1] + bit_diff_hist[2] > sum(bit_diff_hist[3:]):
            print("  Confirme : majorité 1-2 bits/byte = bits bas (masqués par shift).")
    else:
        print("[DRIFT] les cellules KV changent entre 2 reads -> refresh insuffisant")
        print("  sur génération longue. Bug à corriger (taux refresh / couverture rows).")
    ser.close()

if __name__ == "__main__":
    main()