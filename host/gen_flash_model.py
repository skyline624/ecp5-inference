#!/usr/bin/env python3
# =============================================================================
# gen_flash_model.py - Build model_flash.bin for the W25Q64 autonomous boot.
#
# Produces a single binary that the FPGA's `flash_boot` module DMAs from the SPI
# flash (offset 0x120000) into the SDRAM at power-on. The image is a byte-for-byte
# mirror of the SDRAM layout that infer_fpga.py:quantize_and_load_weights would
# otherwise load over UART with `LL`, plus a 47-byte shift trailer at offset
# 0x061000 (step 2 replaces the UART `SH` command).
#
# Flash layout (8 MB W25Q64 on the Colorlight i9) :
#   0x000000   bitstream (written separately by openFPGALoader, ~1.02 MB)
#   0x120000   this image : SDRAM mirror [0x000000..0x060040) + trailer @0x061000
#
# SDRAM mirror layout (identical to infer_fpga.py / src/gen_seq.v OFF_*) :
#   0x000000 : tok_emb [512, 64] chunked ([64,64] blocks, row-major, zero-padded)
#   0x010000 + l*0x10000 : layer l
#       +0x0000 rms_att [64]            +0x0100 wq [64,64]   +0x1100 wk [32,64]
#       +0x1900 wv [32,64]              +0x2100 wo [64,64]   +0x3100 rms_ffn [64]
#       +0x3200 w1 [172,64] chunked     +0x6200 w3 [172,64]  +0x9200 w2 [64,172]
#   0x060000 : rms_final [64]
#   0x061000 : 47-byte shift trailer (sh_packet order, see infer_g5silicon.py)
#
# Usage :
#   python gen_flash_model.py                     # -> host/model_flash.bin
#   python gen_flash_model.py --hex-out model_flash.hex   # also dump a $readmemh hex
#   python gen_flash_model.py --out /path/to/model_flash.bin
# Then flash :
#   openFPGALoader -b colorlight-i9 --write-flash --address 0x120000 \
#       --write-bin model_flash.bin --verify
# =============================================================================
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_fpga import load_model, MODEL
from v4_quant import to_i8_shift
from transformer_ops import D, FQ_MAX_N, FQ_K   # D=64, FQ_MAX_N=64, FQ_K=64

MODEL_END  = 0x060040      # last useful SDRAM byte (rms_final @0x060000 + 64)
IMG_SIZE   = 0x070000      # image size (pads past the trailer @0x061000 + 47)
SHIFT_OFF  = 0x061000      # shift trailer offset within the image
FLASH_BASE = 0x120000      # flash offset where openFPGALoader writes this image


def put(buf, addr, data):
    """Write raw bytes into the image buffer at a byte address."""
    b = bytes(data)
    assert addr >= 0 and addr + len(b) <= len(buf), \
        f"put @ {addr:#x} len {len(b)} overflows image ({len(buf)})"
    buf[addr:addr + len(b)] = b


def put_chunked(buf, addr, W_i8, N, K):
    """Replica of transformer_ops.sd_load_matrix_chunked but writing into the
    image buffer instead of streaming over UART. Stores W [N,K] as a sequence of
    [FQ_MAX_N, FQ_K] = [64,64] zero-padded blocks, indexed
    (k_chunk_idx * n_chunks_per_row + n_chunk_idx) * 64*64."""
    n_chunks_per_row = (N + FQ_MAX_N - 1) // FQ_MAX_N
    k_chunks         = (K + FQ_K - 1) // FQ_K
    for k_idx in range(k_chunks):
        k0 = k_idx * FQ_K
        k1 = min(k0 + FQ_K, K)
        for n_idx in range(n_chunks_per_row):
            n0 = n_idx * FQ_MAX_N
            n1 = min(n0 + FQ_MAX_N, N)
            block = np.zeros((FQ_MAX_N, FQ_K), dtype=np.int8)
            block[:n1 - n0, :k1 - k0] = W_i8[n0:n1, k0:k1]
            block_idx = k_idx * n_chunks_per_row + n_idx
            put(buf, addr + block_idx * FQ_MAX_N * FQ_K, block.tobytes())


def build_image(m):
    cfg = m['cfg']
    L   = cfg['n_layers']
    HID = cfg['hidden_dim']
    V   = cfg['vocab_size']

    # rework dim (e21d94b) : layout SDRAM CUMULATIF. DOIT matcher src/model_cfg.vh
    # CFG_OFF_* (identique a host/gen_sdram_preload_hex.py). Les anciens offsets
    # paddés en magic numbers (0x100, 0x1100, 0x1900...) etaient valides AVANT le
    # rework dim ; depuis e21d94b le RTL lit aux offsets cumulatifs (OFF_WQ=D=0x40,
    # OFF_WK=D+D*D=0x1040...) -> si on flashe encore l'image paddée, le RTL lit du
    # padding (wq=0) -> attention cassee -> tokens "ow" constants sur silicium.
    Dm  = cfg['dim']
    H   = cfg['n_heads']
    KH  = cfg['n_kv_heads']
    HS  = cfg['head_size']
    KVW = KH * HS
    HIDPAD = ((HID + Dm - 1) // Dm) * Dm
    NCHUNK = HIDPAD // Dm
    MAT_SZ = NCHUNK * Dm * Dm
    OFF_RMS_ATT = 0
    OFF_WQ      = Dm
    OFF_WK      = Dm + Dm*Dm
    OFF_WV      = Dm + Dm*Dm + KVW*Dm
    OFF_WO      = Dm + Dm*Dm + 2*KVW*Dm
    OFF_RMSFF   = Dm + 2*Dm*Dm + 2*KVW*Dm
    OFF_W1      = 2*Dm + 2*Dm*Dm + 2*KVW*Dm     # = OFF_RMSFF + Dm
    OFF_W3      = OFF_W1 + MAT_SZ
    OFF_W2      = OFF_W3 + MAT_SZ
    # LBASE doit etre >= la taille de tok_emb (VOCAB*D octets, place a A_EMB=0), sinon
    # les poids de la couche 0 ecrasent la fin de l'embedding. Fige a 0x010000, ce plan
    # memoire cassait dim256 (512*256 = 0x20000 > 0x10000) : le chunk 1 du lm_head lisait
    # rms_att/WQ au lieu de tok_emb[256..511]. Plancher a 0x010000 pour ne rien changer a
    # dim64/dim128 (valides bit-exact sur silicium). Doit rester identique a
    # src/model_cfg.vh CFG_LBASE.
    _EMB_SZ = VOCAB * D
    LBASE   = 0x010000 if _EMB_SZ <= 0x010000 else (1 << (_EMB_SZ - 1).bit_length())
    LSTRIDE = 1 << (OFF_W1 + 3*MAT_SZ - 1).bit_length()  # next pow2 >= layer_sz
    A_EMB      = 0x000000
    A_RMSFINAL = LBASE + L * LSTRIDE
    assert Dm == D, f"cfg dim {Dm} != transformer_ops D {D} (gen_flash_model est dim64-spec)"

    buf = bytearray(IMG_SIZE)
    sh  = {}  # collected shifts, keyed like sh_packet

    # tok_emb [V, D] chunked @ A_EMB (put_chunked K=D<=FQ_K => row-major-equivalent)
    tok_i8, sh_emb = to_i8_shift(m['tok_emb'])
    put_chunked(buf, A_EMB, tok_i8, V, D)
    sh['emb'] = sh_emb

    # rms_final [D] @ A_RMSFINAL
    rf_i8, sh_rf = to_i8_shift(m['rms_final'])
    put(buf, A_RMSFINAL, rf_i8.tobytes())
    sh['rmsfinal'] = sh_rf

    # per-layer weights
    sw_rms, swq, swk, swv, swo = [], [], [], [], []
    sw_rmsf, sw1, sw3, sw2 = [], [], [], []
    for l in range(L):
        base = LBASE + l * LSTRIDE

        rms_att_i8, s_ra = to_i8_shift(m['rms_att'][l])
        wq_i8, s_q       = to_i8_shift(m['wq'][l])
        wk_i8, s_k       = to_i8_shift(m['wk'][l])
        wv_i8, s_v       = to_i8_shift(m['wv'][l])
        wo_i8, s_o       = to_i8_shift(m['wo'][l])
        rms_ffn_i8, s_rf = to_i8_shift(m['rms_ffn'][l])
        w1_i8, s_w1      = to_i8_shift(m['w1'][l])
        w3_i8, s_w3      = to_i8_shift(m['w3'][l])
        w2_i8, s_w2      = to_i8_shift(m['w2'][l])

        put(buf, base + OFF_RMS_ATT, rms_att_i8.tobytes())
        put(buf, base + OFF_WQ, wq_i8.reshape(-1).tobytes())
        put(buf, base + OFF_WK, wk_i8.reshape(-1).tobytes())
        put(buf, base + OFF_WV, wv_i8.reshape(-1).tobytes())
        put(buf, base + OFF_WO, wo_i8.reshape(-1).tobytes())
        put(buf, base + OFF_RMSFF, rms_ffn_i8.tobytes())
        put_chunked(buf, base + OFF_W1, w1_i8, HID, D)
        put_chunked(buf, base + OFF_W3, w3_i8, HID, D)
        put_chunked(buf, base + OFF_W2, w2_i8, D, HID)

        sw_rms.append(s_ra);  swq.append(s_q);  swk.append(s_k)
        swv.append(s_v);      swo.append(s_o);  sw_rmsf.append(s_rf)
        sw1.append(s_w1);     sw3.append(s_w3); sw2.append(s_w2)

    # 47-byte shift trailer @ 0x061000, in sh_packet order (infer_g5silicon.py:39-56)
    shifts = (sw_rms + swq + swk + swv + swo + sw_rmsf + sw1 + sw3 + sw2
              + [sh['rmsfinal'], sh['emb']])
    assert len(shifts) == 47, f"expected 47 shifts, got {len(shifts)}"
    put(buf, SHIFT_OFF, bytes([s & 0xFF for s in shifts]))
    return buf, shifts


def to_hex(buf, hex_path):
    """Dump the image as a $readmemh hex file (one byte per line, 2 hex digits)."""
    with open(hex_path, 'w') as f:
        for b in buf:
            f.write(f"{b:02x}\n")
    print(f"  wrote {hex_path} ({len(buf)} lines)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "model_flash.bin"))
    ap.add_argument("--hex-out", default=None,
                    help="also dump a $readmemh hex file (for iverilog flash mock)")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    print(f"Chargement modele : {args.model}")
    m = load_model(args.model)
    cfg = m['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}")

    print("Construction de l'image flash (mirror SDRAM + trailer 47 shifts)...")
    buf, shifts = build_image(m)

    with open(args.out, 'wb') as f:
        f.write(buf)
    print(f"  wrote {args.out} ({len(buf)} bytes = {len(buf)/1024:.1f} KiB)")
    print(f"  SDRAM mirror : [0x000000..0x{MODEL_END:06x})")
    print(f"  shift trailer @ 0x{SHIFT_OFF:06x} (47 bytes, sh_packet order)")
    print(f"  flash --address 0x{FLASH_BASE:06x} --write-bin {os.path.basename(args.out)}")

    print(f"  shifts (47) : {[s for s in shifts]}")

    if args.hex_out:
        to_hex(buf, args.hex_out)


if __name__ == "__main__":
    main()