#!/usr/bin/env python3
# Genere les LUT RoPE freq_cis_cos.hex / freq_cis_sin.hex pour le module gen_seq.
#
# RoPE "a la Llama2 stories260K" :
#   theta = 10000.0
#   freqs[k] = 1 / theta ** (2k / hs)         pour k = 0..hs/2-1
#   angles[pos, k] = pos * freqs[k]
#   cos_rom[pos*hs/2 + k] = round(cos(angles[pos,k]) * 32768)   (Q15, clamp int16)
#   sin_rom[pos*hs/2 + k] = round(sin(angles[pos,k]) * 32768)
#
# Parametres par defaut (stories260K) : NPOS=17, HS=8, theta=10000.
#   -> 4 paires (hs/2) x 17 positions = 68 entrees par fichier.
#   -> freqs = [1, 0.1, 0.01, 0.001]  (car 10000^[0, 0.25, 0.5, 0.75] = [1,10,100,1000])
#
# Ces ROMs remplacent les anciens ports cos_q15/sin_q15 (1088-bit) du design
# Gowin d'origine : passage host-serie -> BSRAM ROM interne pour eviter le
# mux 68:1 + bus high-fanout qui congestionnait le GW2AR-18 (Session 6b, lever #1).
#
# Consommees par : src/gen_seq.v (cos_rom / sin_rom, $readmemh lignes 227-229).
# Sortie : ../lut/freq_cis_cos.hex, ../lut/freq_cis_sin.hex
#         (format $readmemh, NPOS*(HS/2) lignes hex 16-bit signed Q15).
#
# NOTE : le calcul est fait en float32 (np.float32) pour reproduire
# bit-exact les .hex existants (valides 17/17 tokens bit-exact en sim).

import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NPOS = int(os.environ.get("NPOS", 17))
HS   = int(os.environ.get("HS", 8))
THETA = float(os.environ.get("THETA", 10000.0))
SUFFIX = os.environ.get("SUFFIX", "")   # ex "_hs16" pour generer freq_cis_cos_hs16.hex

OUT_COS = os.path.join(HERE, "..", "lut", f"freq_cis_cos{SUFFIX}.hex")
OUT_SIN = os.path.join(HERE, "..", "lut", f"freq_cis_sin{SUFFIX}.hex")


def make_freq_cis(seq_len, hs, theta):
    """RoPE frequencies a la Llama2 stories260K. Retourne (cos, sin) float32."""
    freqs = 1.0 / (theta ** (np.arange(0, hs, 2).astype(np.float32) / hs))
    t = np.arange(seq_len).astype(np.float32)
    angles = np.outer(t, freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def enc_q15(v):
    """Quantifie en Q15 signed 16-bit (deux-complement), clamp [-32768, 32767]."""
    q = int(round(float(v) * 32768))
    return max(-32768, min(32767, q))


def write_rom(path, mat, npos, half):
    with open(path, "w") as f:
        f.write(f"// RoPE {os.path.basename(path).replace('.hex','')} Q15, "
                f"NPOS={npos} HS={HS} theta={THETA} ({npos}*{half}={npos*half} entrees)\n")
        f.write(f"// index = pos*{half} + k,  val = round({'cos' if 'cos' in path else 'sin'}(pos*freqs[k]) * 32768)\n")
        f.write(f"// freqs[k] = 1/{THETA}^(2k/{HS})\n")
        for pos in range(npos):
            for k in range(half):
                q = enc_q15(mat[pos, k])
                if q < 0:
                    q += 65536
                f.write(f"{q:04x}\n")


def main():
    half = HS // 2
    cos_a, sin_a = make_freq_cis(NPOS, HS, THETA)
    write_rom(OUT_COS, cos_a, NPOS, half)
    write_rom(OUT_SIN, sin_a, NPOS, half)
    freqs = 1.0 / (THETA ** (np.arange(0, HS, 2).astype(np.float32) / HS))
    print(f"RoPE freq_cis generees : {OUT_COS} + {OUT_SIN}")
    print(f"  NPOS={NPOS} HS={HS} theta={THETA}  -> {NPOS*half} entrees/fichier")
    print(f"  freqs = {list(np.round(freqs, 6))}")
    print(f"  cos(pos=0) = 0x7fff (1.0) ; sin(pos=0) = 0x0000 (0.0)")


if __name__ == "__main__":
    main()