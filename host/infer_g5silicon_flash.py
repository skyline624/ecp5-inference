#!/usr/bin/env python3
# =============================================================================
# infer_g5silicon_flash.py - G5' inference, STEP 1 of the autonomous flash boot.
#
# The model weights are NO LONGER loaded by the PC : at power-on the FPGA's
# `flash_boot` module DMAs model_flash.bin (offset 0x120000 in the W25Q64) into
# the SDRAM. So the PC only sends `SH` (the 47 shift bytes, still over UART) then
# `GO`, and reads 17 tokens. If they match the oracle, the flash->SDRAM boot +
# refresh cohabitation are correct on silicon.
#
# The 47 shifts are taken from gen_flash_model.build_image -- the EXACT same bytes
# that the step-2 flash trailer holds -- so the UART `SH` and the flash trailer
# cannot diverge. Step 2 (infer_g5autonomous.py) will drop the `SH` entirely.
# =============================================================================

import os, sys, time
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, load_tok, decode, MODEL, TOK
from gen_flash_model import build_image

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000
N_TOK = 17

# Oracle (float v4sim, node-faithful : tokens 0..8 match, divergence @9 accepted)
ORACLE = [1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298]


def main():
    print(f"Chargement modele (host-side, pour les shifts) : {MODEL}")
    m = load_model(MODEL); cfg = m['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}\n")
    vocab = load_tok(TOK, cfg['vocab_size'])

    # Build the flash image just to harvest the 47 shifts (sh_packet order).
    # Identical bytes to the trailer openFPGALoader flashed at 0x181000.
    _, shifts = build_image(m)
    assert len(shifts) == 47
    sh_payload = b'SH' + bytes([s & 0xFF for s in shifts])
    print(f"  shifts (47) : {[s for s in shifts]}\n")

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    # The flash boot (~0.5-0.6 s) + POR + SDRAM init must finish before the
    # wrapper un-gates the UART (rx_valid && !booting). Wait it out.
    print("Attente fin du boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    # RR : reset gen_seq for 1 cycle -> expect 'RK'. Guards against gen_seq having
    # started spuriously during the flash boot (the wrapper host FSM is NOT gated by
    # `booting`, only the node is -- a stray byte at port-open could look like GO).
    print("RR : reset gen_seq -> attend RK")
    ser.write(b'RR')
    ack = ser.read(2)
    if ack != b'RK':
        print(f"  (RR ack = {ack!r}, continue quand meme)")
    else:
        print("  OK : RK recu")
    time.sleep(0.05); ser.reset_input_buffer()

    # SH : latch the 47 shift bytes -> expect 'SK'
    print("SH : latch des shifts -> attend SK")
    ser.write(sh_payload)
    ack = ser.read(2)
    if ack != b'SK':
        print(f"FAIL: SH ack = {ack!r} (attendu b'SK')"); ser.close(); sys.exit(1)
    print("  OK : SK recu\n")

    # GO : start gen_seq, switch to RUN -> expect 'GK'
    print("GO : demarrage gen_seq (mode RUN, PC read-only) -> attend GK")
    ser.write(b'GO')
    ack = ser.read(2)
    if ack != b'GK':
        print(f"FAIL: GO ack = {ack!r} (attendu b'GK')"); ser.close(); sys.exit(1)
    print("  OK : GK recu, gen_seq tourne\n")

    # Read 17 tokens (2 bytes each : low, {6'b0, high[9:8]})
    print(f"Lecture de {N_TOK} tokens (PC read-only)...")
    tokens = []
    text = b""
    for i in range(N_TOK):
        t_step = time.time()
        b = ser.read(2)
        if len(b) != 2:
            print(f"FAIL: token {i} timeout, recu {len(b)} octets"); break
        tok = b[0] | (b[1] << 8)
        tokens.append(tok)
        prev = tokens[-2] if len(tokens) >= 2 else 1
        p = decode(prev, tok, vocab); text += p
        dt = time.time() - t_step
        tag = "  *" if i < len(ORACLE) and tok == ORACLE[i] else ""
        print(f"  tok {i:2d} = {tok:4d}  ({dt:5.2f}s){tag}  {repr(p.decode('utf-8','replace'))}")

    print(f"\nTexte FPGA : {text.decode('utf-8','replace')!r}")
    print(f"Tokens    : {tokens}")
    print(f"Oracle    : {ORACLE}")
    # The BOS token (1) may or may not be emitted first depending on whether gen_seq
    # was reset (RR) before GO. Align the oracle to the first emitted token, then
    # count the leading bit-exact run. 9 consecutive matches = the same "Once upon
    # a time, there was a little" as the LL silicon run (divergence @9 accepted).
    if not tokens:
        print("[FAIL] aucun token recu"); ser.close(); sys.exit(1)
    align = 0 if tokens[0] == ORACLE[0] else (1 if tokens[0] == ORACLE[1] else None)
    if align is None:
        print(f"[FAIL] premier token {tokens[0]} != oracle[0]={ORACLE[0]} ni oracle[1]={ORACLE[1]}")
        ser.close(); sys.exit(1)
    n = min(len(tokens), len(ORACLE) - align)
    match = 0
    for i in range(n):
        if tokens[i] == ORACLE[align + i]: match += 1
        else: break
    print(f"Alignement oracle[{align}:] ; match leading : {match}/{n}")
    if match >= 9:
        print("[PASS] boot flash etape 1 valide : modele auto-charge, 9 tokens bit-exact")
    else:
        print("[FAIL] divergence prematuree (boot flash SDRAM suspect)")
    ser.close()


if __name__ == "__main__":
    main()