#!/usr/bin/env python3
# Genere la LUT 1/sqrt(x) pour le module rmsnorm_op.
# 256 entrees, x in [1, 2), y in (sqrt(0.5), 1], stocke en Q1.15.
#
# La constante SQRT2_Q15 = 0xb505 (= round(sqrt(2)*32768)) est hardcoded dans
# src/rmsnorm_op.v (correction de parite : si shift_amt est impair, on multiplie
# le resultat LUT par sqrt(2) pour couvrir la moitie d'intervalle impaire).
#
# Consommee par : src/rmsnorm_op.v (rsqrt_lut).
# Sortie : ../lut/rsqrt_lut.hex  (format $readmemh, 256 lignes hex 16-bit).

import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lut", "rsqrt_lut.hex")
SQRT2_Q15 = int(round(np.sqrt(2.0) * 32768))   # = 46341 = 0xb505

with open(OUT, "w") as f:
    f.write(f"// LUT 1/sqrt(x) pour x in [1, 2), Q1.15.\n")
    f.write(f"// SQRT2 (pour parite impaire) = 0x{SQRT2_Q15:04x}\n")
    for i in range(256):
        x = (256 + i) / 256.0          # x in [1, 2)
        y = 1.0 / np.sqrt(x)           # y in (0.707, 1]
        y_q15 = int(round(y * 32768))
        y_q15 = min(y_q15, 32767)      # cap a Q1.15 max
        f.write(f"{y_q15:04x}\n")

print(f"LUT generee dans {OUT}")
print(f"  256 entrees, Q1.15, x in [1, 2)")
print(f"  SQRT2 Q1.15 = 0x{SQRT2_Q15:04x} (constante a hardcoder dans le RTL)")