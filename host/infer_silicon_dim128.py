# infer_silicon_dim128.py -- Test silicium dim128 (profile DIM128, NC=0 stream).
#
# Méthode tok9 ([[drift-tok9-investigation]]) : le silicium exécute le MÊME RTL que
# le sim Verilator. Si silicium == sim != oracle -> le RTL est cohérent, l'oracle
# Python a un gap (RTL right). Si silicium == oracle != sim -> bug RTL. On observe
# seulement les tokens (UART) -- pas l'XB (sondes dbg_* SIM-only).
#
# Contrairement à infer_g5silicon.py (stories260K float -> quantize_and_load_weights),
# ici le dump synthétique dim128 est DÉJÀ int8 (gen_synthetic_dim.py). On charge
# directement les régions SDRAM via sd_load (LL pass-through) avec les offsets
# paramétriques CFG_OFF_* (mêmes formules que gen_sdram_preload_hex.py / model_cfg.vh),
# puis SH (47 shifts du dump) + GO + lecture 17 tokens.
#
# Build requis : build_silicon.ps1 -Dim128 (NODE_ONLY+FLASH_BOOT+CLUSTER_NODE+DIM128).
# Usage : G5_PORT=COM3 python infer_silicon_dim128.py [--dump <dump.json>] [--ntok N]
import os, sys, json, time, argparse
import serial
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_sdram_diag import sd_load, addr_bytes, n_bytes

PORT = os.environ.get("G5_PORT", "COM6"); BAUD = 1_000_000

# Références de comparaison (dump dim128 synthétique, 17 tokens).
# SIM_TOKENS = sim Verilator fixé (après fix RoPE ROM gen_seq.v : charge freq_cis_*_hs16.hex
#   quand HS==16, pas seulement sous HS16_PROFILE). Le sim fixé == oracle 17/17.
#   L'ANCIEN sim (buggé, ROM HS=8 chargé en dim128) donnait [385,...,60,324,79] (3 mismatches
#   p14-16) — silicium pré-fix aussi (ROM baked identique sim/silicium).
# Oracle (prove_gen_node.py node_tokens du dump) = référence correcte (lit le bon fichier _hs16).
SIM_TOKENS = [385, 385, 385, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 60]
ORACLE     = [385, 385, 385, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 60]


def rows(M, a, b, cols):
    """Aplatit M[a:b] row-major (chaque ligne `cols` éléments)."""
    return [int(M[r][k]) for r in range(a, b) for k in range(cols)]


def sd_load_split(ser, addr, data):
    """sd_load avec découpage : le packet LL a un champ len 16-bit (max 65535).
    tok_emb = VOCAB*D = 65536 à dim128 -> dépasse. On découpe en chunks < 65535."""
    MAX = 60000
    off = 0
    while off < len(data):
        n = min(MAX, len(data) - off)
        if not sd_load(ser, addr + off, data[off:off + n]):
            return False
        off += n
    return True


def build_regions(m):
    """Construit la liste (addr, vals) des régions SDRAM -- DOIT matcher
    gen_sdram_preload_hex.py byte-pour-byte (ce que le RTL lit)."""
    cfg = m['cfg']
    D = cfg['dim']; HID = cfg['hidden_dim']; NL = cfg['n_layers']
    H = cfg['n_heads']; KH = cfg['n_kv_heads']; HS = cfg['head_size']; VOCAB = cfg['vocab_size']
    HIDPAD = ((HID + D - 1) // D) * D
    NCHUNK = HIDPAD // D
    KVW = KH * HS
    OFF = {
        'rms_att': 0,
        'wq':      D,
        'wk':      D + D * D,
        'wv':      D + D * D + KVW * D,
        'wo':      D + D * D + 2 * KVW * D,
        'rms_ffn': D + 2 * D * D + 2 * KVW * D,
        'w1':      2 * D + 2 * D * D + 2 * KVW * D,            # = rms_ffn + D
        'w3':      2 * D + 2 * D * D + 2 * KVW * D + NCHUNK * D * D,
        'w2':      2 * D + 2 * D * D + 2 * KVW * D + 2 * NCHUNK * D * D,
    }
    layer_sz = OFF['w1'] + 3 * NCHUNK * D * D
    # LBASE doit etre >= la taille de tok_emb (VOCAB*D octets, place a A_EMB=0), sinon
    # les poids de la couche 0 ecrasent la fin de l'embedding. Fige a 0x010000, ce plan
    # memoire cassait dim256 (512*256 = 0x20000 > 0x10000) : le chunk 1 du lm_head lisait
    # rms_att/WQ au lieu de tok_emb[256..511]. Plancher a 0x010000 pour ne rien changer a
    # dim64/dim128 (valides bit-exact sur silicium). Doit rester identique a
    # src/model_cfg.vh CFG_LBASE.
    _EMB_SZ = VOCAB * D
    LBASE = 0x010000 if _EMB_SZ <= 0x010000 else (1 << (_EMB_SZ - 1).bit_length())
    LSTRIDE = 1 << (layer_sz - 1).bit_length() if layer_sz > 1 else 1   # next pow2 >= layer_sz
    A_EMB = 0x000000
    A_RMSFINAL = LBASE + NL * LSTRIDE

    regions = []
    regions.append((A_EMB, rows(m['tok_emb'], 0, VOCAB, D)))
    for l in range(NL):
        w = m['layers'][l]; base = LBASE + l * LSTRIDE
        regions.append((base + OFF['rms_att'], [int(x) for x in w['rms_att']]))
        regions.append((base + OFF['wq'], rows(w['wq'], 0, H * HS, D)))
        regions.append((base + OFF['wk'], rows(w['wk'], 0, KH * HS, D)))
        regions.append((base + OFF['wv'], rows(w['wv'], 0, KH * HS, D)))
        regions.append((base + OFF['wo'], rows(w['wo'], 0, D, H * HS)))
        regions.append((base + OFF['rms_ffn'], [int(x) for x in w['rms_ffn']]))
        w1 = [list(r) for r in w['w1']] + [[0] * D for _ in range(HIDPAD - HID)]
        w3 = [list(r) for r in w['w3']] + [[0] * D for _ in range(HIDPAD - HID)]
        regions.append((base + OFF['w1'], rows(w1, 0, HIDPAD, D)))
        regions.append((base + OFF['w3'], rows(w3, 0, HIDPAD, D)))
        for c in range(NCHUNK):
            blk = [[(int(w['w2'][r][c * D + k]) if c * D + k < HID else 0) for k in range(D)]
                   for r in range(D)]
            regions.append((base + OFF['w2'] + c * D * D, rows(blk, 0, D, D)))
    regions.append((A_RMSFINAL, [int(x) for x in m['rms_final']]))
    return regions, cfg, (LBASE, LSTRIDE, A_RMSFINAL)


def sh_packet(m):
    """47-byte SH payload. Ordre = gen_seq_silicon.v H_SH :
       per-layer (9): rms_att_s wq_s wk_s wv_s wo_s rms_ffn_s w1_s w3_s w2_s
       + rms_final_s + tok_emb_s. Shifts int8 -> & 0xFF."""
    cfg = m['cfg']; NL = cfg['n_layers']
    shifts = []
    for l in range(NL):
        w = m['layers'][l]
        shifts += [w['rms_att_s'], w['wq_s'], w['wk_s'], w['wv_s'], w['wo_s'],
                   w['rms_ffn_s'], w['w1_s'], w['w3_s'], w['w2_s']]
    shifts += [m['rms_final_s'], m['tok_emb_s']]
    assert len(shifts) == 47, f"expected 47 shifts, got {len(shifts)}"
    print(f"  shifts (47): {shifts}")
    return b'SH' + bytes([s & 0xFF for s in shifts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "gen_model_dump_dim128.json"))
    ap.add_argument("--ntok", type=int, default=17)
    args = ap.parse_args()

    with open(args.dump) as f:
        m = json.load(f)
    cfg = m['cfg']
    print(f"Dump : {args.dump}")
    print(f"  dim={cfg['dim']} HID={cfg['hidden_dim']} NL={cfg['n_layers']} "
          f"H={cfg['n_heads']} KH={cfg['n_kv_heads']} HS={cfg['head_size']} "
          f"VOCAB={cfg['vocab_size']} score_sh={cfg.get('score_sh', 0)}")

    regions, cfg, (LBASE, LSTRIDE, A_RMSFINAL) = build_regions(m)
    total = sum(len(v) for _, v in regions)
    print(f"  {len(regions)} régions SDRAM, {total} octets "
          f"(LSTRIDE={LSTRIDE:#x} A_RMSFINAL={A_RMSFINAL:#x})")

    ser = serial.Serial(PORT, BAUD, timeout=30.0)
    time.sleep(0.5); ser.reset_input_buffer()

    # ---- Chargement SDRAM via LL pass-through ----
    print(f"Chargement {len(regions)} régions en SDRAM via LL...")
    t0 = time.time()
    for i, (addr, vals) in enumerate(regions):
        b = bytes((v & 0xFF) for v in vals)
        if not sd_load_split(ser, addr, b):
            print(f"FAIL sd_load région {i} @ {addr:#08x} ({len(b)} octets)")
            ser.close(); sys.exit(1)
    print(f"  {len(regions)} régions chargées en {time.time()-t0:.1f}s")

    # ---- SH : latch les 47 shifts -> attend SK ----
    print("SH : latch des shifts -> attend SK")
    ser.write(sh_packet(m))
    ack = ser.read(2)
    if ack != b'SK':
        print(f"FAIL: SH ack = {ack!r} (attendu b'SK')"); ser.close(); sys.exit(1)
    print("  SK OK")

    # ---- GO : démarre gen_seq (RUN, PC read-only) -> attend GK ----
    print("GO : démarrage gen_seq (mode RUN) -> attend GK")
    ser.write(b'GO')
    ack = ser.read(2)
    if ack != b'GK':
        print(f"FAIL: GO ack = {ack!r} (attendu b'GK')"); ser.close(); sys.exit(1)
    print("  GK OK")

    # ---- Lecture des tokens (2 octets chacun : low, {6'b0, high[9:8]}) ----
    print(f"Lecture de {args.ntok} tokens (PC read-only)...")
    tokens = []
    for i in range(args.ntok):
        b = ser.read(2)
        if len(b) < 2:
            print(f"FAIL: token {i} timeout, reçu {len(b)} octets"); break
        tok = b[0] | (b[1] << 8)
        tokens.append(tok)
    ser.close()

    # Oracle : priorite au node_tokens du dump (calcule par gen_synthetic_dim.py pour CE
    # palier) plutot qu'aux constantes ci-dessus, figees sur dim128. Rend le script
    # utilisable tel quel a n'importe quel dim -- c'est deja le cas du reste du code, qui
    # derive tous ses offsets de cfg (D/HID/NL/H/KH/HS/VOCAB).
    oracle = m.get('node_tokens') or ORACLE
    sim_ref = SIM_TOKENS if oracle is ORACLE else oracle
    print(f"\nSilicium tokens : {tokens}")
    print(f"RTL (sim)      : {sim_ref}")
    print(f"Oracle         : {oracle}")
    SIM_TOKENS_L, ORACLE_L = sim_ref, oracle
    n = min(len(tokens), len(SIM_TOKENS_L), len(ORACLE_L))
    ms = sum(1 for i in range(n) if tokens[i] == SIM_TOKENS_L[i])
    mo = sum(1 for i in range(n) if tokens[i] == ORACLE_L[i])
    print(f"\nMatch silicium vs sim    : {ms}/{n}")
    print(f"Match silicium vs oracle : {mo}/{n}")
    # Verdict (sur les 3 mismatches p14-16, ou match global)
    if ms == n and n == len(SIM_TOKENS_L):
        print("VERDICT: silicium == sim -> RTL right (oracle a un gap, comme tok9)")
    elif mo == n and n == len(ORACLE_L):
        print("VERDICT: silicium == oracle -> bug RTL (sim/RTL diverge du modèle)")
    else:
        print("VERDICT: silicium diffère de sim ET oracle -> investiguer (race timing ?)")


if __name__ == "__main__":
    main()