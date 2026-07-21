#!/usr/bin/env python3
# =============================================================================
# infer_g5autonomous.py - G5' inference, STEP 2 of the autonomous flash boot.
#
# 100% autonomous : at power-on the FPGA loads BOTH the model weights AND the 47
# per-layer/lm-head shifts from the W25Q64 flash (model_flash.bin @ 0x120000,
# trailer @ 0x181000) into the SDRAM and the sw_* regs. The PC only pulses `RR`
# (reset gen_seq, guard against a stray start during boot) then `GO`, and reads
# 17 tokens. No `LL`, no `SH` -- the PC is now just a read-only observer.
#
# The model/shifts never come from the PC, so this script does NOT load the model
# host-side ; it only loads the tokenizer to decode the emitted tokens.
# =============================================================================

import os, sys, time
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_tok, decode, TOK

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000
N_TOK = 17
VOCAB = 512   # stories260K

# Oracle (float v4sim, node-faithful : tokens 0..8 match, divergence @9 accepted).
# The BOS token (1) is not emitted after RR, so align on oracle[1:].
ORACLE = [1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298]


def main():
    vocab = load_tok(TOK, VOCAB)
    print("Mode autonome : modele + shifts auto-charges depuis la flash (RR + GO seul)\n")

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    # flash boot (~0.5-0.6 s) + POR + SDRAM init ; wrapper gates UART while booting.
    print("Attente fin du boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    # RR : reset gen_seq for 1 cycle -> expect 'RK'. Guards against gen_seq having
    # started spuriously during the flash boot (wrapper host FSM is NOT gated by
    # `booting`, only the node is -- a stray byte at port-open could look like GO).
    print("RR : reset gen_seq -> attend RK")
    ser.write(b'RR')
    ack = ser.read(2)
    if ack != b'RK':
        print(f"  (RR ack = {ack!r}, continue quand meme)")
    else:
        print("  OK : RK recu")
    time.sleep(0.05); ser.reset_input_buffer()

    # GO : start gen_seq, switch to RUN -> expect 'GK'  (no SH -- shifts from flash)
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
    if not tokens:
        print("[FAIL] aucun token recu"); ser.close(); sys.exit(1)
    # BOS (1) is not emitted after RR -> align on oracle[1:].
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
        print("[PASS] boot flash etape 2 valide : 100% autonome, GO seul, 9 tokens bit-exact")
    else:
        print("[FAIL] divergence prematuree (shifts flash trailer suspect)")
    ser.close()


if __name__ == "__main__":
    main()