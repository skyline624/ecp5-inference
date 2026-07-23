#!/usr/bin/env python3
# =============================================================================
# gen_sdram_preload_hex.py -- Produit sim/sdram_preload.hex : image mémoire de la
# SDRAM mock (sim/models/sdram_chip.v) au format $readmemh, pour charger les
# poids du modèle stories260K SANS passer par cocotb VPI (qui ne persiste pas
# l'écriture dans le grand `reg [31:0] mem [0:2M]` sous icarus -- cf diagnostic
# sim/test_sdram_preload.py).
#
# Reproduit EXACTEMENT la logique de preload_model (sim/test_gen_F2.py) : mêmes
# offsets, même packing little-endian, même chunking w1/w3 (172 -> 192 pad) et
# w2 (3 blocs de 64 colonnes). Le mock sdram_chip indexe `mem[idx]` avec
# idx = {BA, row, col} = byte_addr>>2 (linéaire 21-bit) -> l'adresse dans le hex
# = byte_addr>>2.
#
# Format $readmemh : `@<8-hex-addr>` reset l'index courant, puis un mot 32-bit
# hex (8 chiffres) par ligne. Les régions non écrites restent X (comme le
# silicium : seules les régions de poids sont chargées, le reste est "vide").
#
# Branchement RTL : sim/models/sdram_chip.v charge ce fichier en simulation :
#   `ifdef SDRAM_PRELOAD
#     initial $readmemh("sdram_preload.hex", mem);
#   `endif
# cocotb_runner.py ajoute -DSDRAM_PRELOAD au build et copie sdram_preload.hex
# vers BUILD + SIM (vvp résout $readmemh relativement à son CWD = test_dir=SIM).
# =============================================================================
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.environ.get("DUMP", os.path.join(HERE, "gen_model_dump.json"))
OUT_HEX = os.environ.get("OUT_HEX", os.path.join(HERE, "..", "sim", "sdram_preload.hex"))

# Shape lue depuis le dump (m['cfg'], lui-même issu du header modèle llama2.c) ->
# plus de magic numbers hardcodés. Les offsets OFF/LBASE/LSTRIDE restent fixés pour
# le layout dim 64 (généralisation à dim 128+ = TODO, voir model_cfg.vh).
LBASE, LSTRIDE = 0x010000, 0x010000
A_RMSFINAL, A_EMB = 0x060000, 0x000000
OFF = dict(rms_att=0x0000, wq=0x0100, wk=0x1100, wv=0x1900, wo=0x2100,
           rms_ffn=0x3100, w1=0x3200, w3=0x6200, w2=0x9200)


def rows(M, a, b, cols):
    return [M[r][k] for r in range(a, b) for k in range(cols)]


def pack_le4(vals):
    """Liste d'octets -> liste de mots 32-bit little-endian (avec pad)."""
    b = bytes((v & 0xFF) for v in vals)
    if len(b) % 4:
        b += bytes(4 - len(b) % 4)
    return [b[4*w] | (b[4*w+1] << 8) | (b[4*w+2] << 16) | (b[4*w+3] << 24)
            for w in range(len(b) // 4)]


def emit_region(lines, byte_addr, vals):
    """Émet un bloc `@addr` + mots 32-bit pour une région contiguë."""
    words = pack_le4(vals)
    idx = byte_addr >> 2
    lines.append(f"@{idx:08x}")
    for w in words:
        lines.append(f"{w & 0xFFFFFFFF:08x}")


def main():
    with open(DUMP) as f:
        m = json.load(f)
    cfg = m['cfg']
    D   = cfg['dim']
    HID = cfg['hidden_dim']
    NL  = cfg['n_layers']
    H   = cfg['n_heads']
    KH  = cfg['n_kv_heads']
    HS  = cfg['head_size']
    VOCAB = cfg['vocab_size']
    # FFN chunking : HID padded au multiple de D supérieur, chunké en blocs [D,D]
    # (identique au RTL gen_seq.v / model_cfg.vh : 172 -> 192 = 3 chunks).
    HIDPAD = ((HID + D - 1) // D) * D
    NCHUNK = HIDPAD // D

    lines = []
    # Embedding table @ 0x000000 (row-major [VOCAB,D])
    emit_region(lines, A_EMB, rows(m['tok_emb'], 0, VOCAB, D))

    for l in range(NL):
        w = m['layers'][l]
        base_l = LBASE + l * LSTRIDE
        emit_region(lines, base_l + OFF['rms_att'], w['rms_att'])
        emit_region(lines, base_l + OFF['wq'], rows(w['wq'], 0, H*HS, D))
        emit_region(lines, base_l + OFF['wk'], rows(w['wk'], 0, KH*HS, D))
        emit_region(lines, base_l + OFF['wv'], rows(w['wv'], 0, KH*HS, D))
        emit_region(lines, base_l + OFF['wo'], rows(w['wo'], 0, D, H*HS))
        emit_region(lines, base_l + OFF['rms_ffn'], w['rms_ffn'])
        # w1/w3 : HID lignes pad à HIDPAD (zéro) -> NCHUNK chunks de D
        w1 = w['w1'] + [[0]*D for _ in range(HIDPAD - HID)]
        w3 = w['w3'] + [[0]*D for _ in range(HIDPAD - HID)]
        emit_region(lines, base_l + OFF['w1'], rows(w1, 0, HIDPAD, D))
        emit_region(lines, base_l + OFF['w3'], rows(w3, 0, HIDPAD, D))
        # w2 [D,HID] : NCHUNK blocs de D colonnes, chaque bloc [D,D] (zéro au-delà de HID)
        for c in range(NCHUNK):
            blk = [[(w['w2'][r][c*D+k] if c*D+k < HID else 0) for k in range(D)]
                   for r in range(D)]
            emit_region(lines, base_l + OFF['w2'] + c*D*D, rows(blk, 0, D, D))

    # rms_final @ 0x060000
    emit_region(lines, A_RMSFINAL, m['rms_final'])

    os.makedirs(os.path.dirname(OUT_HEX), exist_ok=True)
    with open(OUT_HEX, 'w') as f:
        f.write("\n".join(lines) + "\n")
    size = os.path.getsize(OUT_HEX)
    nwords = sum(1 for ln in lines if not ln.startswith("@"))
    print(f"Écrit {OUT_HEX} : {size} octets, {nwords} mots 32-bit, "
          f"{len([ln for ln in lines if ln.startswith('@')])} régions.")


if __name__ == "__main__":
    main()