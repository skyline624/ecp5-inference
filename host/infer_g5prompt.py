#!/usr/bin/env python3
# =============================================================================
# infer_g5prompt.py - G5' inference with PREFILL (prompt -> response).
#
# The FPGA is now a black box : at power-on it loads the model + 47 shifts from
# the W25Q64 flash (validated étapes 1+2). This script sends a PROMPT (a list of
# token IDs) via the new `PR` opcode, then `GO`, and reads the generated tokens.
# gen_seq runs the prompt through the transformer WITHOUT emitting (prefill :
# embed + KV-fill for the P-1 first tokens), then emits argmax(last prompt token)
# = the first generated token, and decodes the rest autonomously.
#
# Host byte protocol (added) :
#   PR len t0_lo t0_hi t1_lo t1_hi ...  -> ack PK   (latch prompt_tokens/prompt_len)
#   len = number of prompt tokens (<= 16). For each token : low byte then
#   {6'b0, token[9:8]} (same byte order as the token packer emits).
#
# BOS convention (host-side, the hardware is agnostic) : the oracle was produced
# WITH BOS=1 at pos 0. To reproduce bit-exact, prefix BOS so the prompt is
# [1, <tok0>, <tok1>, ...] (default --bos). With --no-bos the prompt is bare and
# the first generated token is NOT guaranteed to match the oracle (model may be
# BOS-agnostic -- diagnostic, not a PASS criterion).
#
# Emissions count : P=0 -> NPOS (BOS path, backward compat). P>=1 -> NPOS-P+1
# (P-1 prefill passes without emit, then NPOS-P+1 passes with emit).
#
# Workflow (run after `openFPGALoader -m build\gen_seq_silicon.bit` so the flash
# boot restarts clean and run_mode=0) :
#   $env:G5_PORT="COM3"
#   py-3.12 host\infer_g5prompt.py                       # default : BOS+"Once upon a time"
#   py-3.12 host\infer_g5prompt.py --ids 1 403 407 261 378
#   py-3.12 host\infer_g5prompt.py --text "Once upon a time"
#   py-3.12 host\infer_g5prompt.py --ids 403 407 261 378 --no-bos   # experimental
# =============================================================================

import os, sys, time, argparse
import serial

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_tok, decode, TOK

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000
NPOS = 17          # gen_seq token loop length (total prefill + generation)
NPR  = 16          # max prompt tokens (wrapper clamp)
VOCAB = 512        # stories260K

# Oracle (float v4sim, node-faithful). BOS=1 is the seed at pos 0 (not emitted in
# the BOS path). oracle[5:] = [432, 383, 286, 261, 376, 298] = ", there was a little".
ORACLE = [1, 403, 407, 261, 378, 432, 383, 286, 261, 376, 298]

# Default test prompt : BOS + "Once upon a time" -> reproduces the oracle context
# (BOS@0, Once@1, upon@2, a@3, time@4), argmax@4 = 432 guaranteed.
DEFAULT_IDS = [1, 403, 407, 261, 378]


def encode(text, vocab, max_len=16):
    """Greedy longest-match encoder over the word-level vocab (tokens carry a
    leading space, e.g. b' Once'). Prefixes a leading space so the first word
    matches b' Once' just like the reference tokenizer after BOS."""
    b2i = {}
    for i, v in enumerate(vocab):
        if v:
            b2i.setdefault(v, i)   # first ID wins if a byte string repeats
    buf = text.encode('utf-8')
    if not buf.startswith(b' '):
        buf = b' ' + buf
    ids, pos = [], 0
    while pos < len(buf):
        best = None
        for L in range(min(max_len, len(buf) - pos), 0, -1):
            c = buf[pos:pos + L]
            if c in b2i:
                best = (b2i[c], L); break
        if best is None:
            print(f"  (encode: unknown byte {buf[pos]:#x} at pos {pos}, skipped)")
            pos += 1; continue
        ids.append(best[0]); pos += best[1]
    return ids


def send_pr(ser, ids):
    """PR len t0_lo t0_hi ... -> expect 'PK'."""
    assert len(ids) <= NPR, f"prompt too long : {len(ids)} > NPR={NPR}"
    pkt = b'PR' + bytes([len(ids) & 0xFF])
    for t in ids:
        pkt += bytes([t & 0xFF, (t >> 8) & 0x03])
    ser.write(pkt)
    return ser.read(2)


def main():
    ap = argparse.ArgumentParser(description="G5' prefill inference (PR + GO).")
    ap.add_argument("--ids", type=int, nargs="*", default=None,
                    help="prompt token IDs (default : BOS + 'Once upon a time').")
    ap.add_argument("--text", type=str, default=None,
                    help="prompt text (greedy encode over the vocab).")
    ap.add_argument("--no-bos", action="store_true",
                    help="do NOT prefix BOS=1 (experimental ; oracle not guaranteed).")
    args = ap.parse_args()

    # build the prompt ID list
    vocab = load_tok(TOK, VOCAB)
    if args.ids is not None:
        ids = list(args.ids)
    elif args.text is not None:
        ids = encode(args.text, vocab)
        print(f"  encode({args.text!r}) -> {ids}")
        if not ids:
            print("[FAIL] encode produced no tokens"); sys.exit(1)
    else:
        ids = list(DEFAULT_IDS)
    if not args.no_bos and ids and ids[0] != 1:
        ids = [1] + ids
    P = len(ids)
    if P > NPR:
        print(f"[FAIL] prompt length {P} > NPR={NPR}"); sys.exit(1)
    if P > NPOS - 1:
        print(f"[FAIL] prompt length {P} >= NPOS={NPOS} (no room to generate)"); sys.exit(1)

    n_emit = NPOS if P == 0 else NPOS - P + 1
    print(f"Prompt IDs ({P} tokens) : {ids}")
    print(f"  -> {b''.join(decode(ids[i-1] if i>0 else 1, ids[i], vocab) for i in range(len(ids))).decode('utf-8','replace')!r}")
    print(f"  -> will read {n_emit} generated tokens\n")

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    print("Attente fin du boot flash (~2 s)...")
    time.sleep(2.0); ser.reset_input_buffer()

    # RR : reset gen_seq (guards against a stray start during boot). Not strictly
    # required after a clean reconfigure, but harmless and matches autonomous.py.
    print("RR : reset gen_seq -> attend RK")
    ser.write(b'RR')
    ack = ser.read(2)
    print(f"  RR ack = {ack!r}" + (" (OK)" if ack == b'RK' else " (continue)"))
    time.sleep(0.05); ser.reset_input_buffer()

    # PR : load the prompt (skip if P==0 -> BOS path, no PR needed)
    if P > 0:
        print(f"PR : charge {P} tokens -> attend PK")
        ack = send_pr(ser, ids)
        if ack != b'PK':
            print(f"FAIL: PR ack = {ack!r} (attendu b'PK'). "
                  f"run_mode probablement colle (GO parasite) -> reconfigurez : "
                  f"openFPGALoader -m build\\gen_seq_silicon.bit")
            ser.close(); sys.exit(1)
        print("  OK : PK recu")
        time.sleep(0.05); ser.reset_input_buffer()

    # GO : start gen_seq -> expect 'GK'
    print("GO : demarrage gen_seq -> attend GK")
    ser.write(b'GO')
    ack = ser.read(2)
    if ack != b'GK':
        print(f"FAIL: GO ack = {ack!r} (attendu b'GK')"); ser.close(); sys.exit(1)
    print("  OK : GK recu, gen_seq tourne\n")

    # read the generated tokens
    print(f"Lecture de {n_emit} tokens...")
    tokens, text = [], b""
    for i in range(n_emit):
        b = ser.read(2)
        if len(b) != 2:
            print(f"FAIL: token {i} timeout, recu {len(b)} octets"); break
        tok = b[0] | (b[1] << 8)
        tokens.append(tok)
        prev = tokens[-2] if len(tokens) >= 2 else (ids[-1] if ids else 1)
        p = decode(prev, tok, vocab); text += p
        print(f"  tok {i:2d} = {tok:4d}  {repr(p.decode('utf-8','replace'))}")

    print(f"\nPrompt   : {ids}")
    print(f"Genere   : {tokens}")
    print(f"Texte    : {text.decode('utf-8','replace')!r}")
    if not tokens:
        print("[FAIL] aucun token recu"); ser.close(); sys.exit(1)

    # alignment + PASS criterion
    # The model's nominal divergence point is oracle[10]=298 (same as the BOS
    # path validated in étape 2, where 9 leading tokens match then it diverges
    # at the 10th). For a prompt == oracle[:P], the slice oracle[P:] reaches that
    # same absolute point at relative index (10-P), so the expected leading match
    # is (10 - max(P,1)) tokens. P=0 -> 9 (BOS path), P=5 -> 5 (", there was a little").
    if P == 0:
        ref = ORACLE[1:]; label = "oracle[1:]"
    elif ids == ORACLE[:P]:
        ref = ORACLE[P:]; label = f"oracle[{P}:]"
    else:
        ref = None; label = "(prompt hors oracle -> pas de critere PASS strict)"
    if ref is not None:
        n = min(len(tokens), len(ref)); match = 0
        for i in range(n):
            if tokens[i] == ref[i]: match += 1
            else: break
        threshold = 10 - max(P, 1)
        print(f"Alignement {label} ; match leading : {match}/{n} (seuil nominal {threshold}, "
              f"divergence connue @ oracle[10]=298)")
        if match >= threshold:
            print(f"[PASS] prefill valide : {match} tokens bit-exact ({label}) -- "
                  f"comportement nominal (divergence @298 = point connu du modele)")
        else:
            print(f"[FAIL] divergence prematuree (avant le point nominal @298)")
    else:
        print(label)
    ser.close()


if __name__ == "__main__":
    main()