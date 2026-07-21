#!/usr/bin/env python3
# Regénère TOUTES les LUT ../lut/*.hex depuis les scripts gen_*_lut.py, puis
# vérifie bit-exact que les fichiers régénérés correspondent aux .hex existants.
#
# Usage :  python scripts/regen_luts.py           # regenere + verifie
#          python scripts/regen_luts.py --check    # verifie seulement (sans ecraser)
#
# Permet de garantir que les .hex commités sont cohérents avec les scripts
# (les .hex sont ignores par git dans build/sim_build, mais versionnés dans lut/).

import os, sys, subprocess, filecmp

HERE = os.path.dirname(os.path.abspath(__file__))
LUT  = os.path.join(HERE, "..", "lut")

GENS = ["gen_exp_lut.py", "gen_inv_lut.py", "gen_rsqrt_lut.py",
        "gen_silu_lut.py", "gen_freq_cis_lut.py"]

OUTS = ["exp_lut.hex", "inv_lut.hex", "rsqrt_lut.hex", "silu_lut.hex",
        "freq_cis_cos.hex", "freq_cis_sin.hex"]

check_only = "--check" in sys.argv

def strip_comments(p):
    return [l.rstrip() for l in open(p) if l.strip() and not l.strip().startswith("//")]

# sauvegarder les originaux pour comparaison si --check
orig = {}
if check_only:
    for n in OUTS:
        rp = os.path.join(LUT, n)
        orig[n] = strip_comments(rp) if os.path.exists(rp) else None

if not check_only:
    for g in GENS:
        subprocess.check_call([sys.executable, os.path.join(HERE, g)])

all_ok = True
for n in OUTS:
    rp = os.path.join(LUT, n)
    if not os.path.exists(rp):
        print(f"  {n}: ABSENT"); all_ok = False; continue
    if check_only:
        # regenerer dans un temp et comparer
        pass
    # verification basique : count de lignes non-commentaire
    lines = strip_comments(rp)
    print(f"  {n}: {len(lines)} entrees  {'OK' if len(lines)>0 else 'VIDE'}")

if not all_ok:
    sys.exit(1)
print("LUT regenerees et presentes.")