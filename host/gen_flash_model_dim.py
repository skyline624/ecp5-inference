#!/usr/bin/env python3
"""gen_flash_model_dim.py -- image flash du modele pour N'IMPORTE QUEL palier dim.

gen_flash_model.py est verrouille sur stories260K/dim64 (il quantifie du float et
importe D depuis transformer_ops, avec un assert explicite "dim64-spec"). Consequence :
le boot flash autonome n'existait qu'a dim64 ; dim128/256 devaient charger la SDRAM par
UART a chaque essai (~60 s a dim256).

Ce script part d'un dump DEJA int8 (gen_synthetic_dim.py --dim N, ou tout dump au meme
format) et reutilise build_regions() / sh_packet() de infer_silicon_dim128.py, qui sont
dim-generiques et valides sur silicium. L'image produite est le miroir exact de ce que
le RTL attend en SDRAM, plus le trailer des 47 shifts.

Layout (doit rester identique a src/model_cfg.vh) :
    [0 .. FLASH_LEN)      miroir SDRAM : tok_emb @0, couches @LBASE+l*LSTRIDE, rms_final
    FLASH_LEN = A_RMSFINAL + D
    SHIFT_OFF = FLASH_LEN arrondi au multiple de 4 Ko    -> 47 octets de shifts
Le KV-cache n'est PAS dans l'image : il est calcule a l'execution.

L'image se flashe a CFG_FLASH_BASE = 0x200000 :
    openFPGALoader -b colorlight-i9 -f -o 0x200000 --file-type bin --verify <out>

Usage :
    python gen_flash_model_dim.py --dump gen_model_dump_dim256.json --out model_dim256.bin
"""
import os, sys, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from infer_silicon_dim128 import build_regions, sh_packet

FLASH_SIZE = 8 * 1024 * 1024      # W25Q64
FLASH_BASE = 0x200000             # = CFG_FLASH_BASE (le bitstream occupe 0..~0x140000)


def build_flash_image(m):
    """Retourne (buf, meta). buf = miroir SDRAM + trailer shifts, pret a flasher."""
    regions, cfg, (LBASE, LSTRIDE, A_RMSFINAL) = build_regions(m)
    D = cfg['dim']
    flash_len = A_RMSFINAL + D                          # fin de rms_final
    shift_off = ((flash_len + 0xFFF) // 0x1000) * 0x1000  # trailer aligne 4 Ko
    img_sz    = shift_off + 47

    buf = bytearray(img_sz)
    for addr, vals in regions:
        b = bytes((v & 0xFF) for v in vals)
        assert addr + len(b) <= flash_len, \
            f"region @0x{addr:06x} ({len(b)} o) depasse FLASH_LEN=0x{flash_len:06x}"
        buf[addr:addr + len(b)] = b

    trailer = sh_packet(m)[2:]                          # sh_packet prefixe 'SH'
    assert len(trailer) == 47, f"trailer {len(trailer)} octets, attendu 47"
    buf[shift_off:shift_off + 47] = trailer

    return buf, dict(D=D, LBASE=LBASE, LSTRIDE=LSTRIDE, A_RMSFINAL=A_RMSFINAL,
                     FLASH_LEN=flash_len, SHIFT_OFF=shift_off, IMG_SZ=img_sz,
                     n_regions=len(regions))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="dump JSON int8 (gen_synthetic_dim.py)")
    ap.add_argument("--out",  default=None, help="fichier .bin de sortie")
    args = ap.parse_args()

    with open(args.dump) as f:
        m = json.load(f)
    cfg = m['cfg']
    out = args.out or os.path.join(HERE, f"model_flash_dim{cfg['dim']}.bin")

    print(f"Dump : {args.dump}")
    print(f"  dim={cfg['dim']} HID={cfg['hidden_dim']} NL={cfg['n_layers']} "
          f"VOCAB={cfg['vocab_size']}")

    buf, meta = build_flash_image(m)
    print(f"  {meta['n_regions']} regions -> image {meta['IMG_SZ']} octets "
          f"({meta['IMG_SZ']/1024/1024:.2f} Mo)")
    print(f"  LBASE=0x{meta['LBASE']:06x} LSTRIDE=0x{meta['LSTRIDE']:06x} "
          f"A_RMSFINAL=0x{meta['A_RMSFINAL']:06x}")
    print(f"  FLASH_LEN=0x{meta['FLASH_LEN']:06x}  trailer @0x{meta['SHIFT_OFF']:06x}")

    fin = FLASH_BASE + meta['IMG_SZ']
    if fin > FLASH_SIZE:
        print(f"\n  ERREUR : l'image finit a 0x{fin:06x} > flash 0x{FLASH_SIZE:06x} "
              f"({fin/1024/1024:.1f} Mo > 8 Mo).")
        print( "  Ce palier ne tient pas en flash -> chargement UART "
               "(infer_silicon_dim128.py --dump ...).")
        sys.exit(1)

    with open(out, "wb") as f:
        f.write(buf)
    print(f"\n  ecrit : {out}")
    print(f"  flash : openFPGALoader -b colorlight-i9 -f -o 0x{FLASH_BASE:06x} "
          f"--file-type bin --verify {os.path.basename(out)}")
    print(f"  occupe 0x{FLASH_BASE:06x}..0x{fin:06x} sur 8 Mo")


if __name__ == "__main__":
    main()
