#!/usr/bin/env python3
# =============================================================================
# gen_synthetic_dim.py -- Dump SYNTHÉTIQUE dim-paramétrable pour la preuve du rework
# dim (Phase 7 : preuve bit-exact par palier 128/256/512).
#
# Stratégie A (poids aléatoires, cf plan bright-fluttering-key Phase 7) : on NE réutilise
# PAS stories260K (dim=64, mauvaise shape) — on génère des poids int8 aléatoires uniformes
# aux shapes du palier, shifts forcés à 0 (poids déjà int8, pas de re-quant). La preuve est
# MÉCANIQUE : RTL=oracle par construction (mêmes opérations int8 bit-exact), peu importe les
# valeurs — l'objectif est d'exercer le datapath dim-paramétrique (largeurs $clog2(D), chunks
# NCHUNK_F=3, lm_head NCHUNK_LM, layout SDRAM cumulatif) à D>64.
#
# Shape palier (cohérent src/model_cfg.vh `DIM128`) : D=128, HID=3*D=384 (NCHUNK_F=3 FSM GG
# hardcodé), HS=16, H=8, KH=2, KVW=KH*HS=32, NREP=H/KH=4, NL=5, VOCAB=512, score_sh=1 (HS=16).
# D=H*HS=8*16=128. À dim=64 la preuve vient de stories260K (non-régression), pas d'ici.
#
# Sortie : host/gen_model_dump_dim{D}.json (consommé par sim/run_verilator_fb.ps1 -Dump
# + -Dim128 qui définit DIM128 -> model_cfg.vh profile dim128). La preuve = DUT sim == node_tokens.
# =============================================================================
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def build_synth_m(D, HID, NL, H, KH, HS, VOCAB, seed=42):
    """Construit le dict m (int8 + shifts=0) pour un palier dim. Poids int8 uniformes
    [-WMAX, WMAX] (modérés pour éviter overflow int32 dans les accumulations ; le FQ lb-6
    RTL+oracle gère mais on garde de la marge). Shifts forcés à 0 (poids déjà int8)."""
    rng = np.random.default_rng(seed)
    WMAX = 16
    def r(*shape): return rng.integers(-WMAX, WMAX + 1, size=shape, dtype=np.int8)
    KVW = KH * HS
    cfg = {
        'dim': D, 'hidden_dim': HID, 'n_layers': NL, 'n_heads': H,
        'n_kv_heads': KH, 'vocab_size': VOCAB, 'seq_len': 512,
        'head_size': HS, 'score_sh': 1, 't_max': 32,   # score_sh=1 (HS=16), t_max=32 (différé)
    }
    m = {
        'cfg': cfg,
        'tok_emb':      r(VOCAB, D),
        'tok_emb_s':    0,
        'rms_final':    r(D),
        'rms_final_s':  0,
        'layers': [],
    }
    for _ in range(NL):
        m['layers'].append({
            'rms_att': r(D),         'rms_att_s': 0,
            'rms_ffn': r(D),         'rms_ffn_s': 0,
            'wq': r(H * HS, D),       'wq_s': 0,      # [D, D] (H*HS=D)
            'wk': r(KVW, D),          'wk_s': 0,
            'wv': r(KVW, D),          'wv_s': 0,
            'wo': r(D, H * HS),       'wo_s': 0,      # [D, D]
            'w1': r(HID, D),          'w1_s': 0,
            'w2': r(D, HID),          'w2_s': 0,
            'w3': r(HID, D),          'w3_s': 0,
        })
    return m


def main():
    import argparse
    from prove_gen_node import forward_node
    from gen_model_dump import dump_to_dict

    ap = argparse.ArgumentParser()
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--nl', type=int, default=5)
    ap.add_argument('--vocab', type=int, default=512)
    ap.add_argument('--ntok', type=int, default=17)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    D = args.dim
    # Palier dim : HID=3*D (NCHUNK_F=3 FSM GG hardcodé), HS=16, KVW=32 fixe (évite rework KV :
    # attn_kv_stride[5:0] + BSRAM K/V dimensionnés pour KVW=32). D=H*HS, KVW=KH*HS=32.
    # À HS=16 : KH=KVW/HS=2, H=D/HS, NREP=H/KH. D=128->H=8 KH=2 NREP=4 ; D=256->H=16 NREP=8 ;
    # D=512->H=32 NREP=16. (stories260K HS=8 : H=8 KH=4 NREP=2 — KH change à dim>=128, NREP
    # aussi, mais les deux sont paramétriques via CFG_LOG2_NREP, cf model_cfg.vh DIM128.)
    HS = 16
    KVW = 32
    KH = KVW // HS                     # 2 (HS=16)
    H = D // HS                        # 8 (D=128), 16 (D=256), 32 (D=512)
    HID = 3 * D
    NL = args.nl
    VOCAB = args.vocab
    assert H * HS == D, f"H*HS={H*HS} != D={D} (palier exige D multiple de HS=16)"
    assert KH * HS == KVW, f"KVW=KH*HS={KH*HS} != {KVW}"
    assert H % KH == 0, f"H={H} non multiple de KH={KH} (NREP=H/KH doit etre entier)"

    print(f"Palier synthétique : dim={D} HID={HID} HS={HS} H={H} KH={KH} "
          f"KVW={KH*HS} NL={NL} VOCAB={VOCAB} seed={args.seed}")
    m = build_synth_m(D, HID, NL, H, KH, HS, VOCAB, args.seed)
    cfg = m['cfg']
    print(f"  cfg : {cfg}")

    print(f"Forward nœud int8 bit-exact ({args.ntok} tokens)...")
    node_tokens, _, _, _, _ = forward_node(m, args.ntok)
    print(f"  node_tokens (dim={D}) = {node_tokens}")

    d = dump_to_dict(m)
    d['node_tokens'] = [int(t) for t in node_tokens]
    # 'tokens' au format [BOS] + node_tokens pour compat TB (gen_tb_hex.py).
    d['tokens'] = [1] + [int(t) for t in node_tokens]

    out = os.path.join(HERE, f"gen_model_dump_dim{D}.json")
    with open(out, 'w') as f:
        json.dump(d, f)
    print(f"  dump écrit : {out} ({os.path.getsize(out)} octets)")
    print(f"  ==> preuve = DUT sim dim={D} == node_tokens (bit-exact).")


if __name__ == "__main__":
    main()