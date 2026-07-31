#!/usr/bin/env python3
"""bench_sdram.py -- mesure la crete REELLE de la SDRAM (controleur + puce).

Pourquoi ce script : les "125 Mo/s" cites dans docs/CAPACITE_FPGA.md sont un
CALCUL (bus 32 bits x 31,25 MHz) et decrivent le bus physique, pas ce que le
controleur peut rendre. Deux leviers (burst, prefetch) se sont brises sur cette
question sans qu'elle ait jamais ete mesuree.

Deux modes, exposes par node_top (S_M2_Z) :
  ZZ  acces ISOLES    -- handshake complet (S_OP_RD_BUSY -> S_OP_RD_DONE),
                         exactement ce que fait le datapath. C'est la reference.
  ZY  acces RAPPROCHES-- handshake MINIMAL (pulse -> attend busy=1 -> reboucle).
                         Mesure ce que coute le handshake seul.

NB : aucun des deux ne teste le PIPELINING (2 acces en vol) -- le controleur ne
gere qu'une transaction a la fois, cela demanderait de modifier sdram.v.

Chaque mode renvoie 8 octets LE : 4 de compteur de cycles + 4 de CHECKSUM (XOR
des mots lus). Si le mode rapproche corrompt les donnees, son checksum differe
de celui du mode isole -- c'est le garde-fou qui manquait au burst.

Usage :
    $env:G5_PORT="COM10"
    python host/bench_sdram.py
"""
import os, sys, time, struct
import serial

PORT = os.environ.get("G5_PORT", "COM10")
BAUD = 1_000_000
BENCH_N = 4096                  # doit correspondre a BENCH_N dans node_top.v
CLK_HZ = 31_250_000             # clk_sys
OCTETS = BENCH_N * 4


def attendus():
    """Les 4 premiers mots a l'adresse 0 = tok_emb[0..15] du dump."""
    import json, pathlib
    p = pathlib.Path(__file__).with_name("gen_model_dump_dim256.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    emb = d.get("tok_emb") or d.get("token_embedding_table")
    if not emb:
        return None
    plat = [v for ligne in emb for v in ligne][:16] if isinstance(emb[0], list) else emb[:16]
    o = [v & 0xFF for v in plat]
    return [int.from_bytes(bytes(o[i*4:(i+1)*4]), "little") for i in range(4)]


def run_mode(ser, second_byte, label, ref=None):
    ser.reset_input_buffer()
    ser.write(b"Z" + second_byte)
    raw = ser.read(24)
    if len(raw) != 24:
        print(f"  {label:<12} FAIL : {len(raw)} octets recus au lieu de 24")
        return None
    cycles, checksum = struct.unpack("<II", raw[:8])
    mots = struct.unpack("<IIII", raw[8:24])
    if cycles == 0:
        print(f"  {label:<12} FAIL : compteur nul")
        return None
    cyc_per_word = cycles / BENCH_N
    debit = OCTETS / (cycles / CLK_HZ) / 1e6
    print(f"  {label:<12} {cycles:>10,} cyc | {cyc_per_word:5.2f} cyc/mot | "
          f"{debit:6.1f} Mo/s | checksum {checksum:08x}")
    mots_txt = " ".join(f"{m:08x}" for m in mots)
    if ref:
        etat = " ".join("OK " if m == r else "FAUX" for m, r in zip(mots, ref))
        print(f"               mots 0-3 : {mots_txt}")
        print(f"               attendus : {' '.join(f'{r:08x}' for r in ref)}")
        print(f"               verdict  : {etat}")
    else:
        print(f"               mots 0-3 : {mots_txt}")
    return cycles, cyc_per_word, debit, checksum


def main():
    print(f"Port {PORT} @ {BAUD} bauds -- {BENCH_N} mots ({OCTETS/1024:.0f} Ko) par mode\n")
    ser = serial.Serial(PORT, BAUD, timeout=10.0)
    time.sleep(0.2)

    ref = attendus()
    if ref:
        print(f"  (reference tok_emb[0..15] : {' '.join(f'{r:08x}' for r in ref)})\n")
    iso = run_mode(ser, b"Z", "ZZ isole", ref)
    tight = run_mode(ser, b"Y", "ZY rapproche", ref)
    print()

    if not iso or not tight:
        ser.close()
        return 1

    bus = CLK_HZ * 4 / 1e6
    print(f"  bus physique (1 mot/cycle) : {bus:.0f} Mo/s")
    print(f"    -> isole     : {iso[2]/bus*100:.1f} %")
    print(f"    -> rapproche : {tight[2]/bus*100:.1f} %")
    print()

    # Integrite : le mode rapproche doit rendre EXACTEMENT les memes donnees.
    if iso[3] != tight[3]:
        print(f"  /!\\ CHECKSUMS DIFFERENTS ({iso[3]:08x} vs {tight[3]:08x})")
        print("      -> les acces rapproches CORROMPENT les donnees sur cette puce.")
        print("      C'est l'explication la plus probable de l'echec du burst.")
    else:
        print(f"  checksums identiques ({iso[3]:08x}) -> les acces rapproches sont SURS.")
        gain = (iso[1] - tight[1]) / iso[1] * 100
        print(f"  gain du handshake minimal : {iso[1]:.2f} -> {tight[1]:.2f} cyc/mot "
              f"({gain:+.1f} %)")
        if gain > 5:
            print("  -> reduire le handshake est un levier reel et sans risque.")
        else:
            print("  -> le handshake n'est PAS le poste dominant : c'est le controleur")
            print("     lui-meme (ACT/READ/CAS/capture) qui coute. Seul le pipelining")
            print("     (modification de sdram.v) irait plus loin.")
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
