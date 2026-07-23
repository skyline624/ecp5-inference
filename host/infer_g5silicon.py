#!/usr/bin/env python3
# =============================================================================
# infer_g5silicon.py - G5' autonomous inference on the Colorlight i9 (ECP5-45).
#
# The PC loads the int8 weights into the node's SDRAM (via the existing `LL`
# opcode, passed through by the gen_seq_silicon wrapper), latches the 9 per-layer
# shift tables + 2 lm-head shifts via `SH`, pulses `GO`, then stays READ-ONLY
# while gen_seq generates 17 tokens autonomously. Each token comes back as 2
# bytes (low, {6'b0, high[9:8]}) on uart_tx.
#
# Reuses quantize_and_load_weights from infer_fpga.py (loads weights AND records
# the to_i8 shifts). The shifts are repacked into the 47-byte `SH` payload.
# =============================================================================

import os, sys, time, struct, re
import numpy as np
import serial

# reuse the weight loader + quant helpers from the PC-orchestrated flow
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, load_tok, quantize_and_load_weights, decode, MODEL, TOK
from test_sdram_diag import sd_load, addr_bytes, n_bytes

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000
N_TOK = 17
NL = 5  # n_layers (stories260K)

# Oracle = tokens GENERES par le silicium (le 1er token genere est tok 0 = 403 = BOS
# embed lookup). Valide sur silicium 23 juil 2026 (build NODE_ONLY+FLASH_BOOT, RTL
# actuel) : 17 tokens, tok 0..9 = [403,407,261,378,432,383,286,261,376,298] bit-exact
# vs sim mock test_gen_F_fb. ATTENTION tok 9 = 298 (le knife-edge 268/298 bascule a 298
# sur silicium = sim mock ; l'oracle prove_gen_node.py predit 268 a tort, gap de
# modelisation Python ~15 unites XN non masque au gap-0). L'ancienne liste [1, 403, ...]
# incluait le BOS=1 a l'index 0 -> comparaison decalee de 1 vs tokens[] (commence au
# 1er token genere) -> faux FAIL. Tokens GENERES seulement ici (pas de BOS).
ORACLE = [403, 407, 261, 378, 432, 383, 286, 261, 376, 298, 276, 364, 280, 323, 417, 427, 303]


def sh_packet(w):
    """Build the 47-byte SH payload from the per-layer shifts recorded by
    quantize_and_load_weights. Order must match gen_seq_silicon.v H_SH:
      sw_rms[5] swq[5] swk[5] swv[5] swo[5] sw_rmsf[5] sw1[5] sw3[5] sw2[5]
      + sw_rmsfinal(1) + sw_emb(1)
    Shifts are int8 two's-complement -> & 0xFF byte."""
    sw_rms  = [w['layers'][l]['attn']['sh_rms'] for l in range(NL)]
    swq     = [w['layers'][l]['attn']['sh_q']   for l in range(NL)]
    swk     = [w['layers'][l]['attn']['sh_k']   for l in range(NL)]
    swv     = [w['layers'][l]['attn']['sh_v']   for l in range(NL)]
    swo     = [w['layers'][l]['attn']['sh_o']   for l in range(NL)]
    sw_rmsf = [w['layers'][l]['ffn']['sh_rms']  for l in range(NL)]
    sw1     = [w['layers'][l]['ffn']['sh_w1']   for l in range(NL)]
    sw3     = [w['layers'][l]['ffn']['sh_w3']   for l in range(NL)]
    sw2     = [w['layers'][l]['ffn']['sh_w2']   for l in range(NL)]
    sw_rmsfinal = w['lm_head']['sh_rms']
    sw_emb     = w['lm_head']['sh_emb']
    shifts = (sw_rms + swq + swk + swv + swo + sw_rmsf + sw1 + sw3 + sw2
              + [sw_rmsfinal, sw_emb])
    assert len(shifts) == 47, f"expected 47 shifts, got {len(shifts)}"
    print(f"  shifts : sw_rms={sw_rms} swq={swq} swk={swk}")
    print(f"           swv={swv} swo={swo} sw_rmsf={sw_rmsf}")
    print(f"           sw1={sw1} sw3={sw3} sw2={sw2} rmsfinal={sw_rmsfinal} emb={sw_emb}")
    return b'SH' + bytes([s & 0xFF for s in shifts])


def main():
    print(f"Chargement modele : {MODEL}")
    m = load_model(MODEL); cfg = m['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']} seq_len={cfg['seq_len']}\n")
    vocab = load_tok(TOK, cfg['vocab_size'])

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    time.sleep(0.5); ser.reset_input_buffer()

    print("Chargement poids stories260K en SDRAM (via LL pass-through)...")
    t0 = time.time()
    w = quantize_and_load_weights(ser, m)
    print(f"  -> charge en {time.time()-t0:.1f}s\n")

    # SH : latch the 47 shift bytes -> expect 'SK'
    print("SH : latch des shifts -> attend SK")
    ser.write(sh_packet(w))
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
    match = sum(1 for i in range(min(len(tokens), len(ORACLE))) if tokens[i] == ORACLE[i])
    print(f"Match tokens 0..{min(len(tokens),len(ORACLE))-1} : {match}/{min(len(tokens),len(ORACLE))}")
    if match >= 9:
        print("[PASS] inférence autonome G5' sur i9 (tokens 0..8 matchent l'oracle)")
    else:
        print("[FAIL] divergence prematuree")
    ser.close()


if __name__ == "__main__":
    main()