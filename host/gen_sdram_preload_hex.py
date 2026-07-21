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
DUMP = os.path.join(HERE, "gen_model_dump.json")
OUT_HEX = os.path.join(HERE, "..", "sim", "sdram_preload.hex")

D, H, KH, HS, HID, NL, VOCAB = 64, 8, 4, 8, 172, 5, 512
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

    lines = []
    # Embedding table @ 0x000000 (row-major [512,64])
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
        # w1/w3 : 172 lignes pad à 192 (zéro) -> 3 chunks de 64
        w1 = w['w1'] + [[0]*D for _ in range(192 - HID)]
        w3 = w['w3'] + [[0]*D for _ in range(192 - HID)]
        emit_region(lines, base_l + OFF['w1'], rows(w1, 0, 192, D))
        emit_region(lines, base_l + OFF['w3'], rows(w3, 0, 192, D))
        # w2 [64,172] : 3 blocs de 64 colonnes, chaque bloc [64,64] (zéro au-delà de 172)
        for c in range(3):
            blk = [[(w['w2'][r][c*64+k] if c*64+k < HID else 0) for k in range(D)]
                   for r in range(D)]
            emit_region(lines, base_l + OFF['w2'] + c*0x1000, rows(blk, 0, D, D))

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