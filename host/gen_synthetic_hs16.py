#!/usr/bin/env python3
# =============================================================================
# gen_synthetic_hs16.py -- Dump HS=16 pour la preuve du rework HS-paramétrable.
#
# Stratégie B (reshape stories260K) : les matrices wq[64,64]/wk[64,32]/wv[64,32]
# ont la MÊME shape pour HS=8 (H=8/KH=4) et HS=16 (H=4/KH=2) car D=H*HS=64 et
# KVW=KH*HS=32 sont inchangés. On réutilise les poids stories260K (mêmes bytes
# int8 + shifts) en surchargeant juste le découpage head via cfg :
#   n_heads=4, n_kv_heads=2, head_size=16, score_sh=1
# forward_node réinterprète alors Q.reshape(4,16) au lieu de (8,8) et l'attention
# opère sur des heads de 16 éléments -> exerce tout le datapath HS-paramétrable
# (i_idx/d_idx 5 bits, Q_reg depth 16, head-loop H=4, kv_offset (h>>1)<<4,
# pre-shift score >>>1, RoPE HS=16 cos/sin ROM 136 entries).
#
# Sortie : host/gen_model_dump_hs16.json (consommé par sim/test_gen_F_fb.py avec
# DUMP=host/gen_model_dump_hs16.json +define+HS16_PROFILE). La preuve = DUT sim
# HS=16 == node_tokens du dump (bit-exact).
# =============================================================================
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "gen_model_dump_hs16.json")


def main():
    from infer_v5gen_ref import load_model, MODEL
    from prove_gen_node import build_quantized_model, forward_node
    from gen_model_dump import dump_to_dict

    print(f"Chargement modele : {MODEL} (reshape HS=16)")
    m_float = load_model(MODEL)
    m = build_quantized_model(m_float)

    # Override head-shape : HS=8 (H=8/KH=4) -> HS=16 (H=4/KH=2).
    # D=H*HS=64 et KVW=KH*HS=32 inchangés -> poids identiques (mêmes bytes),
    # seul le découpage head change. score_sh=1 (pre-shift score, HS=16).
    cfg = m['cfg']
    cfg['n_heads'] = 4
    cfg['n_kv_heads'] = 2
    cfg['head_size'] = 16
    cfg['score_sh'] = 1
    print(f"  HS=16 reshape : H={cfg['n_heads']} KH={cfg['n_kv_heads']} "
          f"HS={cfg['head_size']} D={cfg['dim']} KVW={cfg['n_kv_heads']*cfg['head_size']} "
          f"score_sh={cfg['score_sh']}")

    print("Forward nœud int8 bit-exact HS=16 (17 tokens)...")
    node_tokens, _, _, _, _ = forward_node(m, 17)
    print(f"  node_tokens (HS=16) = {node_tokens}")

    d = dump_to_dict(m)
    d['node_tokens'] = [int(t) for t in node_tokens]
    # Pas d'oracle float HS=16 nécessaire : la preuve est DUT == node (int8 bit-exact).
    # On garde 'tokens' au format [BOS] + node_tokens pour compat test_gen_F_fb.
    d['tokens'] = [1] + [int(t) for t in node_tokens]

    print(f"Écriture {DUMP} ...")
    with open(DUMP, 'w') as f:
        json.dump(d, f)
    size = os.path.getsize(DUMP)
    print(f"  -> {size} octets ({size / 1024:.1f} KiB)")
    print("  ==> dump HS=16 généré (preuve = DUT sim HS=16 == node_tokens).")


if __name__ == "__main__":
    main()