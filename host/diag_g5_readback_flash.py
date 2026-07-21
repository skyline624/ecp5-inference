#!/usr/bin/env python3
# =============================================================================
# diag_g5_readback_flash.py - Validate the autonomous flash->SDRAM boot on silicon.
#
# Pre-requisite : the bitstream is flashed at offset 0 AND model_flash.bin is
# flashed at offset 0x120000, then the board is power-cycled. At power-on the
# FPGA's `flash_boot` module DMAs the model into the SDRAM; the wrapper keeps the
# UART gated (rx_valid && !booting) until that finishes.
#
# This script sends NO `LL` (the model is already in SDRAM from flash). It just
# waits for the boot to finish, then `CC`-readbacks three SDRAM zones and compares
# them byte-for-byte to model_flash.bin (the exact image that was flashed). If the
# three zones match, the flash DMA + refresh cohabitation are correct.
#
#   zone A : tok_emb[0..31]      @ SDRAM 0x000000  (flash image offset 0x000000)
#   zone B : wq layer 0[0..31]   @ SDRAM 0x010100  (flash image offset 0x010100)
#   zone C : rms_final[0..31]    @ SDRAM 0x060000  (flash image offset 0x060000)
# =============================================================================

import os, sys, time
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_sdram_diag import sd_dump

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000
IMG = os.path.join(HERE, "model_flash.bin")

ZONES = [
    ("tok_emb[0..31]   ", 0x000000, 32),
    ("wq layer0[0..31] ", 0x010100, 32),
    ("rms_final[0..31] ", 0x060000, 32),
]


def main():
    with open(IMG, "rb") as f:
        img = f.read()
    print(f"Reference : {IMG} ({len(img)} octets)\n")

    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    # flash boot (~0.5-0.6 s) + POR + SDRAM init ; the wrapper gates the UART
    # while booting. Wait well past it before issuing the first CC.
    print("Attente fin du boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    all_ok = True
    for name, addr, n in ZONES:
        got = sd_dump(ser, addr, n)
        if got is None:
            print(f"  {name} @ 0x{addr:06x} : AUCUNE REPONSE (CC readback casse)")
            all_ok = False
            continue
        exp = img[addr:addr + n]
        match = sum(1 for a, b in zip(got, exp) if a == b)
        ok = (got == exp)
        all_ok = all_ok and ok
        tag = "[OK]" if ok else "[MISMATCH]"
        print(f"  {name} @ 0x{addr:06x} : {match}/{n} octets  {tag}")
        if not ok:
            print(f"      got : {got.hex()}")
            print(f"      exp : {exp.hex()}")

    print()
    if all_ok:
        print("[PASS] boot flash valide : SDRAM peuplee bit-exact depuis la flash.")
        print("       Lancer infer_g5silicon_flash.py pour la generation 17 tokens.")
    else:
        print("[FAIL] SDRAM != flash -> le DMA flash_boot est casse (ou readback).")
    ser.close()


if __name__ == "__main__":
    main()