#!/usr/bin/env python3
# =============================================================================
# diag_kv_after_gen.py - Localise le bug étape 1 (KV SDRAM) en CC-lisant le KV
# écrit par le seq (via CDC) APRÈS une génération, via le path sd_addr du node
# (proven correct par diag_kv_region). Distinction :
#   * CC-read == oracle  -> écriture KV (CDC write path) correcte -> bug = READ
#     path (PH_MM : CDC read ou FSM seq de lecture).
#   * CC-read != oracle  -> écriture KV corrompue -> bug = CDC write path.
#
# Flow : RR + SH + GO -> lit 17 tokens -> RR (retour LOAD, nouveau RR-en-RUN) ->
# CC-read KV[layer=0, pos=0..3, K+V] (256 octets) -> compare à kv_oracle.npz.
# La SDRAM volatile est préservée (gap < 64ms, refresh auto au 1er CC op).
# =============================================================================
import os, sys, time
import serial
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, decode, MODEL
from gen_flash_model import build_image
from test_sdram_diag import sd_dump

PORT = os.environ.get("G5_PORT", "COM3"); BAUD = 1_000_000
N_TOK = 17
KV_BASE = 0x070000
TMAX = 32; KVW = 32  # KH*HS

def main():
    # --- 47 shifts pour SH (identiques au trailer flash) ---
    m = load_model(MODEL)
    _, shifts = build_image(m)
    assert len(shifts) == 47
    sh_payload = b'SH' + bytes([s & 0xFF for s in shifts])

    oracle = np.load(os.path.join(HERE, "kv_oracle.npz"))
    kmem = oracle['kmem']; vmem = oracle['vmem']
    node_tokens = oracle['node_tokens']
    print(f"Oracle KV : kmem shape={kmem.shape}  node_tokens[:6]={node_tokens[:6].tolist()}")

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    print("Attente fin boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    print("RR (LOAD) -> RK"); ser.write(b'RR')
    print("  ack:", ser.read(2)); time.sleep(0.05); ser.reset_input_buffer()
    print("SH -> SK"); ser.write(sh_payload)
    ack = ser.read(2)
    if ack != b'SK': print(f"FAIL SH ack={ack!r}"); ser.close(); sys.exit(1)
    print("  OK")
    print("GO -> GK"); ser.write(b'GO')
    ack = ser.read(2)
    if ack != b'GK': print(f"FAIL GO ack={ack!r}"); ser.close(); sys.exit(1)
    print("  OK, génération en cours...")

    tokens = []
    for i in range(N_TOK):
        b = ser.read(2)
        if len(b) != 2: print(f"  token {i} timeout"); break
        tokens.append(b[0] | (b[1] << 8))
    print(f"  tokens silicium : {tokens}")
    print(f"  oracle node     : {node_tokens.tolist()}")
    match = sum(1 for i in range(min(len(tokens),17)) if i<len(node_tokens) and tokens[i]==int(node_tokens[i]))
    print(f"  match leading : {match}/{min(len(tokens),17)}")

    # --- RR en RUN mode -> retour LOAD (nouveau path debug étape 1) ---
    print("\nRR (retour LOAD via RR-en-RUN) -> RK")
    time.sleep(0.05)
    ser.write(b'RR')
    ack = ser.read(2)
    print(f"  ack: {ack!r}  (attendu b'RK')")
    time.sleep(0.05); ser.reset_input_buffer()

    # --- CC-read KV[layer=0, pos=0..3, K+V] via sd_addr (proven) ---
    print("\nCC-read KV[layer=0, pos=0..3, K+V] via sd_addr :")
    npos_dump = 4
    k_ok = [0]*npos_dump; v_ok = [0]*npos_dump
    all_ok = True
    for p in range(npos_dump):
        k_addr = KV_BASE + (0*TMAX + p)*2*KVW          # K offset 0..31
        v_addr = KV_BASE + (0*TMAX + p)*2*KVW + KVW     # V offset 32..63
        got_k = sd_dump(ser, k_addr, KVW)
        got_v = sd_dump(ser, v_addr, KVW)
        exp_k = bytes((kmem[0, p].flatten().astype(np.int32) & 0xFF).astype(np.uint8).tolist())
        exp_v = bytes((vmem[0, p].flatten().astype(np.int32) & 0xFF).astype(np.uint8).tolist())
        if got_k is None or got_v is None:
            print(f"  pos {p}: pas de réponse CC"); all_ok = False; continue
        k_ok[p] = sum(1 for a,b in zip(got_k, exp_k) if a==b)
        v_ok[p] = sum(1 for a,b in zip(got_v, exp_v) if a==b)
        kok = (got_k == exp_k); vok = (got_v == exp_v)
        all_ok = all_ok and kok and vok
        print(f"  pos {p}: K {k_ok[p]}/{KVW} {'[OK]' if kok else '[MISMATCH]'}  V {v_ok[p]}/{KVW} {'[OK]' if vok else '[MISMATCH]'}")
        if not kok: print(f"      K got: {got_k.hex()}\n      K exp: {exp_k.hex()}")
        if not vok: print(f"      V got: {got_v.hex()}\n      V exp: {exp_v.hex()}")

    print("\n" + ("[PASS] KV écriture correcte (CDC write OK) -> bug = READ path (PH_MM / CDC read)"
                 if all_ok else
                 "[FAIL] KV écriture corrompue -> bug = CDC write path (seq KVWR -> SDRAM)"))
    ser.close()

if __name__ == "__main__":
    main()