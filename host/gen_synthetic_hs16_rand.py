#!/usr/bin/env python3
# =============================================================================
# gen_synthetic_hs16_rand.py -- Dump HS=16 Stratégie A (poids aléatoires, shifts=0)
# pour une preuve bit-exact PROPRE du rework HS-paramétrable.
#
# Pourquoi Stratégie A vs B (reshape stories260K) :
#   - Stratégie B réutilise les poids stories260K (mêmes bytes) mais hérite du
#     gap prove_gen_node (~15 unités XN au knife-edge, cf drift-tok9-investigation)
#     -> la sim DUT peut diverger de l'oracle node_tokens à un knife-edge, et sans
#     oracle silicium HS=16 on ne peut pas distinguer bug rework vs gap oracle.
#   - Stratégie A : poids int8 aléatoires uniformes + SHIFTS=0.
#       * shifts=0 -> re-alignement KV (clip8(mem >>> (ref-sh))) devient no-op
#         (ref-sh=0) -> élimine la source principale du gap oracle.
#       * poids aléatoires -> grands écarts de logits (pas de near-tie) -> aucun
#         knife-edge ne flippo même si un résidu de gap subsiste.
#     -> DUT sim HS=16 == node_tokens bit-exact 17/17 (preuve mécanique propre).
#
# Shapes (D=H*HS=64, KVW=KH*HS=32 inchangés vs HS=8) :
#   tok_emb [512,64], rms_final [64]
#   per layer : rms_att [64], rms_ffn [64], wq [64,64], wk [32,64], wv [32,64],
#               wo [64,64], w1 [172,64], w3 [172,64], w2 [64,172]  (tous shifts=0)
#
# Sortie : host/gen_model_dump_hs16_rand.json + sim/sdram_preload_hs16_rand.hex
# (via DUMP=... OUT_HEX=... host/gen_sdram_preload_hex.py).
# =============================================================================
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "gen_model_dump_hs16_rand.json")

CFG = dict(dim=64, hidden_dim=172, n_layers=5, n_heads=4, n_kv_heads=2,
           head_size=16, vocab_size=512, score_sh=1)
SEED = 1234


def rand_i8(shape, rng):
    """int8 uniforme [-127,127] (évite -128 pour ne pas edge-case le round)."""
    return rng.integers(-127, 128, size=shape).astype(np.int8)


def build_random_model(cfg, seed):
    rng = np.random.default_rng(seed)
    D, HID, NL = cfg['dim'], cfg['hidden_dim'], cfg['n_layers']
    H, KH, HS = cfg['n_heads'], cfg['n_kv_heads'], cfg['head_size']
    VOCAB = cfg['vocab_size']
    m = {
        'cfg': dict(cfg),
        'tok_emb': rand_i8((VOCAB, D), rng),
        'tok_emb_s': 0,
        'rms_final': rand_i8((D,), rng),
        'rms_final_s': 0,
        'layers': [],
    }
    for _ in range(NL):
        lay = {}
        lay['rms_att'] = rand_i8((D,), rng); lay['rms_att_s'] = 0
        lay['rms_ffn'] = rand_i8((D,), rng); lay['rms_ffn_s'] = 0
        lay['wq'] = rand_i8((H * HS, D), rng); lay['wq_s'] = 0   # [64,64]
        lay['wk'] = rand_i8((KH * HS, D), rng); lay['wk_s'] = 0  # [32,64]
        lay['wv'] = rand_i8((KH * HS, D), rng); lay['wv_s'] = 0  # [32,64]
        lay['wo'] = rand_i8((D, H * HS), rng); lay['wo_s'] = 0   # [64,64]
        lay['w1'] = rand_i8((HID, D), rng); lay['w1_s'] = 0      # [172,64]
        lay['w3'] = rand_i8((HID, D), rng); lay['w3_s'] = 0      # [172,64]
        lay['w2'] = rand_i8((D, HID), rng); lay['w2_s'] = 0      # [64,172]
        m['layers'].append(lay)
    return m


def main():
    from prove_gen_node import forward_node
    from gen_model_dump import dump_to_dict

    print(f"Construction modèle aléatoire HS=16 (seed={SEED}, shifts=0)...")
    print(f"  HS=16 H={CFG['n_heads']} KH={CFG['n_kv_heads']} D={CFG['dim']} "
          f"KVW={CFG['n_kv_heads']*CFG['head_size']} score_sh={CFG['score_sh']}")
    m = build_random_model(CFG, SEED)

    print("Forward nœud int8 HS=16 (17 tokens)...")
    node_tokens, _, _, _, _ = forward_node(m, 17)
    print(f"  node_tokens (HS=16 rand) = {node_tokens}")

    d = dump_to_dict(m)
    d['node_tokens'] = [int(t) for t in node_tokens]
    d['tokens'] = [1] + [int(t) for t in node_tokens]
    # marker pour distinguer du dump Stratégie B au cas où
    d['profile'] = 'hs16_rand'

    print(f"Écriture {DUMP} ...")
    with open(DUMP, 'w') as f:
        json.dump(d, f)
    size = os.path.getsize(DUMP)
    print(f"  -> {size} octets ({size / 1024:.1f} KiB)")
    print("  ==> dump HS=16 rand généré.")
    print("  ==> Pour le preload : "
          f"DUMP=\"{DUMP}\" OUT_HEX=\"{os.path.join(HERE, '..', 'sim', 'sdram_preload_hs16_rand.hex')}\" "
          f"python host/gen_sdram_preload_hex.py")
    print("  ==> Pour la sim : DUMP=\"{DUMP}\" + -DHS16_PROFILE, copier "
          "sdram_preload_hs16_rand.hex -> sim/sdram_preload.hex")


if __name__ == "__main__":
    main()