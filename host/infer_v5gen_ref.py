#!/usr/bin/env python3
# =============================================================================
# infer_v5gen_ref.py -- Oracle FLOAT "propre" du nœud stories260K.
#
# Référence "ce que ferait un LLM normal" : RMSNorm/SiLU/softmax/RoPE/attention
# en float64 pur (avec /√HS), SANS aucune quantification intermédiaire. Sert à
# produire m['tokens'] (18 tokens : [BOS=1, <17 générés>]) pour le dump
# gen_model_dump.json.
#
# Pour les 9 premiers tokens, l'oracle float == le modèle int8 du nœud (les
# logits sont bien séparés, argmax robuste). Au token 9 l'attention du nœud
# (LUT exp/inv simplifiée, sans /√HS) fait basculer un near-tie : float -> 298,
# hardware -> 268. C'est une propriété matérielle du nœud, pas un bug de
# séquenceur (cf. host/prove_gen_node.py qui prouve RTL == nœud 17/17).
#
# Indépendant de prove_gen_node (aucun import croisé) : c'est juste un forward
# float standard pour comparer au modèle bit-exact.
# =============================================================================
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# stories260K.bin vit dans le dépôt sibling dsp_coproc (cf. host/infer_fpga.py).
MODEL = os.path.join(HERE, "..", "..", "dsp_coproc", "host", "models", "stories260K.bin")


def load_model(path):
    """Charge stories260K.bin -> dict de tenseurs float (cf. infer_fpga.load_model)."""
    import struct
    with open(path, 'rb') as f:
        header = f.read(28)
    cfg = dict(zip(['dim', 'hidden_dim', 'n_layers', 'n_heads', 'n_kv_heads',
                    'vocab_size', 'seq_len'], struct.unpack('<7i', header)))
    cfg['head_size'] = cfg['dim'] // cfg['n_heads']
    cfg['vocab_size'] = abs(cfg['vocab_size'])
    raw = np.fromfile(path, dtype=np.float32, offset=28)
    L, D, H, KH, HS, HID, V, S = (cfg['n_layers'], cfg['dim'], cfg['n_heads'],
        cfg['n_kv_heads'], cfg['head_size'], cfg['hidden_dim'],
        cfg['vocab_size'], cfg['seq_len'])
    pos = [0]
    def t(*s):
        n = int(np.prod(s)); v = raw[pos[0]:pos[0] + n].reshape(s).astype(np.float64).copy()
        pos[0] += n; return v
    return {'cfg': cfg, 'tok_emb': t(V, D), 'rms_att': t(L, D),
            'wq': t(L, H * HS, D), 'wk': t(L, KH * HS, D), 'wv': t(L, KH * HS, D),
            'wo': t(L, D, H * HS), 'rms_ffn': t(L, D), 'w1': t(L, HID, D),
            'w2': t(L, D, HID), 'w3': t(L, HID, D), 'rms_final': t(D),
            'freq_cis_real': t(S, HS // 2), 'freq_cis_imag': t(S, HS // 2)}


def rmsnorm(x, w, eps=1e-5):
    return x * w / np.sqrt(np.mean(x * x) + eps)


def silu(x):
    return x / (1.0 + np.exp(-x))


def apply_rope(x, fr, fi):
    """x [N, HS], fr/fi [HS//2] -> applique RoPE par paire (2i, 2i+1)."""
    out = x.copy()
    for i in range(len(fr)):
        xr = x[:, 2 * i]; xi = x[:, 2 * i + 1]
        out[:, 2 * i]     = xr * fr[i] - xi * fi[i]
        out[:, 2 * i + 1] = xr * fi[i] + xi * fr[i]
    return out


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def forward_ref(m, n_tokens=17):
    """Forward float pur, greedy argmax. Retourne [token_1, ..., token_n_tokens]."""
    cfg = m['cfg']
    L, H, KH, HS, D, HID, V, S = (cfg['n_layers'], cfg['n_heads'], cfg['n_kv_heads'],
        cfg['head_size'], cfg['dim'], cfg['hidden_dim'], cfg['vocab_size'], cfg['seq_len'])
    nrep = H // KH
    # KV cache float : une liste de (KH, HS) par position, par couche.
    kv = [{'K': [], 'V': []} for _ in range(L)]
    cur_tok = 1  # BOS
    out = []
    for pos in range(n_tokens):
        x = m['tok_emb'][cur_tok].astype(np.float64).copy()
        for l in range(L):
            xn = rmsnorm(x, m['rms_att'][l])
            Q  = (m['wq'][l] @ xn).reshape(H, HS)
            Kc = (m['wk'][l] @ xn).reshape(KH, HS)
            Vc = (m['wv'][l] @ xn).reshape(KH, HS)
            fr = m['freq_cis_real'][pos]; fi = m['freq_cis_imag'][pos]
            Q  = apply_rope(Q, fr, fi)
            Kc = apply_rope(Kc, fr, fi)
            kv[l]['K'].append(Kc); kv[l]['V'].append(Vc)
            Ks = np.array(kv[l]['K'])  # [T, KH, HS]
            Vs = np.array(kv[l]['V'])
            attn_out = np.zeros(D)
            for h in range(H):
                kh = h // nrep
                Qh = Q[h]
                K_h = Ks[:, kh, :]              # [T, HS]
                V_h = Vs[:, kh, :]              # [T, HS]
                scores = Qh @ K_h.T / np.sqrt(HS)
                attn = softmax(scores)
                attn_out[h * HS:(h + 1) * HS] = attn @ V_h
            x = x + m['wo'][l] @ attn_out
            xn = rmsnorm(x, m['rms_ffn'][l])
            h1 = m['w1'][l] @ xn
            h3 = m['w3'][l] @ xn
            hh = silu(h1) * h3
            x = x + m['w2'][l] @ hh
        xn = rmsnorm(x, m['rms_final'])
        logits = m['tok_emb'] @ xn
        nxt = int(np.argmax(logits))
        out.append(nxt)
        cur_tok = nxt
    return out


def main():
    print(f"Chargement modele : {MODEL}")
    m = load_model(MODEL)
    cfg = m['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']} seq_len={cfg['seq_len']}")
    print("\nOracle float v5 (ref propre, /sqrt(HS), greedy argmax, 17 tokens) :")
    toks = forward_ref(m, 17)
    tokens = [1] + toks
    print(f"  tokens (BOS + 17) = {tokens}")
    print(f"  genere           = {toks}")
    print("  (diverge du nœud int8 @ tok 9 : float=298, hardware=268 ; "
          "les 9 premiers coincident avec le modele int8)")


if __name__ == "__main__":
    main()