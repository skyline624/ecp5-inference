#!/usr/bin/env python3
# =============================================================================
# gen_model_dump.py -- Produit host/gen_model_dump.json : le dump int8+shifts
# du modèle stories260K + les références (node_tokens[17] bit-exact, tokens[18]
# oracle float) consommé par sim/test_gen_F.py (preuve RTL 17/17) et
# sim/test_gen_F2.py (gate 2 tokens).
#
# Étapes :
#   1. load_model(MODEL) -> tenseurs float
#   2. build_quantized_model -> dict m int8 + shifts _s (par tenseur)
#   3. forward_node(m) -> node_tokens[17]  (modèle bit-exact du nœud RTL)
#   4. forward_ref(m_float, 17) -> oracle ; tokens = [1] + oracle -> [18]
#   5. json.dump (numpy -> list via .tolist(), shifts -> int)
#
# Le dump est l'ARTEFACT DE PREUVE : committé dans le repo, il rend la preuve
# reproductible sans numpy ni stories260K.bin (absents du sim Docker). w1/w3
# gardés à 172 lignes (preload_model padde à 192), w2 à [64,172] (preload
# découpe en 3 blocs de 64 colonnes).
# =============================================================================
import os
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "gen_model_dump.json")


def _to_list(a):
    """numpy int8 array -> nested list de signed ints (le & 0xFF est fait dans preload)."""
    return np.asarray(a).astype(np.int32).tolist()


def dump_to_dict(m):
    """Convertit le dict m (numpy arrays + shifts) en dict JSON-sérialisable."""
    cfg = m['cfg']
    out = {
        'cfg': {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in cfg.items()},
        'tok_emb': _to_list(m['tok_emb']),
        'tok_emb_s': int(m['tok_emb_s']),
        'rms_final': _to_list(m['rms_final']),
        'rms_final_s': int(m['rms_final_s']),
        'layers': [],
    }
    for lay in m['layers']:
        d = {}
        for key in ('rms_att', 'rms_ffn', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'):
            d[key] = _to_list(lay[key])
        for key in ('rms_att_s', 'rms_ffn_s', 'wq_s', 'wk_s', 'wv_s',
                    'wo_s', 'w1_s', 'w2_s', 'w3_s'):
            d[key] = int(lay[key])
        out['layers'].append(d)
    return out


def main():
    from infer_v5gen_ref import load_model, MODEL, forward_ref
    from prove_gen_node import build_quantized_model, forward_node

    print(f"Chargement modele : {MODEL}")
    m_float = load_model(MODEL)
    cfg = m_float['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}")

    print("Quantification int8...")
    m = build_quantized_model(m_float)

    print("Forward nœud int8 bit-exact (17 tokens)...")
    node_tokens = forward_node(m, 17)
    print(f"  node_tokens = {node_tokens}")

    print("Forward oracle float (17 tokens)...")
    oracle = forward_ref(m_float, 17)
    tokens = [1] + oracle
    print(f"  tokens (BOS + 17) = {tokens}")

    d = dump_to_dict(m)
    d['node_tokens'] = [int(t) for t in node_tokens]
    d['tokens'] = [int(t) for t in tokens]

    print(f"Écriture {DUMP} ...")
    with open(DUMP, 'w') as f:
        json.dump(d, f)
    size = os.path.getsize(DUMP)
    print(f"  -> {size} octets ({size / 1024:.1f} KiB)")

    # ---- auto-vérifications (gate 2) ----
    ok = True
    exp9 = [403, 407, 261, 378, 432, 383, 286, 261, 376]
    if d['node_tokens'][:9] != exp9:
        print(f"  FAIL : node_tokens[:9] = {d['node_tokens'][:9]} != {exp9}"); ok = False
    if d['node_tokens'][9] != 268:
        print(f"  FAIL : node_tokens[9] = {d['node_tokens'][9]} != 268"); ok = False
    if d['tokens'][1:10] != d['node_tokens'][:9]:
        print(f"  FAIL : tokens[1:10] != node_tokens[:9]"); ok = False
    if len(d['node_tokens']) != 17 or len(d['tokens']) != 18:
        print(f"  FAIL : longueurs node_tokens/tokens"); ok = False
    if ok:
        print("  ==> GATE 2 PASS : dump généré, node_tokens[9]==268, "
              "tokens[1:10]==node_tokens[:9].")
    else:
        print("  ==> GATE 2 FAIL.")


if __name__ == "__main__":
    main()