#!/usr/bin/env python3
# =============================================================================
# diag_kv_region.py - Test isolé de la région KV SDRAM (0x070000) via le NODE.
# LL-écrit un motif connu puis CC-lit -> valide SDRAM + contrôleur + shadow-word
# pour la zone KV, SANS impliquer la CDC du seq.
#   * si PASS : la SDRAM/contrôleur/shadow-word sont OK pour la zone KV ->
#               le bug étape 1 est dans la CDC ou la FSM read/write du seq.
#   * si FAIL : la zone KV elle-même est corrompue (shadow-word, row/bank,
#               chevauchement avec le modèle boot, etc.).
# =============================================================================
import os, sys, time
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_sdram_diag import sd_load, sd_dump

PORT = os.environ.get("G5_PORT", "COM3"); BAUD = 1_000_000
KV_BASE = 0x070000

def main():
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    print("Attente fin boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    # RR pour armer un état propre (au cas où gen_seq aurait démarré).
    ser.write(b'RR'); print("  RR ack:", ser.read(2))
    time.sleep(0.05); ser.reset_input_buffer()

    all_ok = True
    for off, n in [(0, 64), (64, 64), (128, 64), (192, 64), (0, 256)]:
        addr = KV_BASE + off
        pattern = bytes([(i*7 + off*3 + 0xA5) & 0xFF for i in range(n)])
        ok_w = sd_load(ser, addr, pattern)
        got = sd_dump(ser, addr, n)
        if got is None:
            print(f"  @0x{addr:06x} +{n}: pas de réponse CC"); all_ok = False; continue
        match = sum(1 for a,b in zip(got, pattern) if a==b)
        ok = (got == pattern) and ok_w
        all_ok = all_ok and ok
        print(f"  @0x{addr:06x} +{n}: write={'OK' if ok_w else 'FAIL'} readback {match}/{n} {'[OK]' if ok else '[MISMATCH]'}")
        if not ok:
            print(f"      got : {got.hex()}")
            print(f"      exp : {pattern.hex()}")

    # aussi tester l'écriture byte-par-byte (comme le seq KVWR le fait) sur 1 mot
    # 32-bit puis 4 bytes contigus -> shadow-word.
    print("\n  Test shadow-word : 4 bytes contigus (mot 32-bit) @0x070100")
    for i in range(4):
        ok = sd_load(ser, 0x070100 + i, bytes([(0x10+i)]))
        if not ok: print(f"    write byte {i} FAIL")
    got4 = sd_dump(ser, 0x070100, 4)
    exp4 = bytes([0x10,0x11,0x12,0x13])
    print(f"    readback 4 bytes: {got4.hex()} exp {exp4.hex()} {'[OK]' if got4==exp4 else '[MISMATCH]'}")
    all_ok = all_ok and (got4 == exp4)

    print("\n" + ("[PASS] région KV SDRAM OK via node (SDRAM+contrôleur+shadow-word sains)" if all_ok
                  else "[FAIL] région KV corrompue -> bug SDRAM/shadow-word/chevauchement"))
    ser.close()

if __name__ == "__main__":
    main()