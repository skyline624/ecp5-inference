# diag_dim128_pos14.py -- classe la divergence tok[14] (near-tie vs bug RTL).
# Rejoue forward_node (oracle dim128) en capturant logits+csh+XB(cur_tok) a chaque pos,
# dump le top-8 argmax (v shifted) pour voir la marge entre le token oracle et 60.
# Si marge(79,60) = 1-2 unites -> near-tie (RTL "right" comme tok9). Si grande -> bug RTL.
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_gen_node import (rmsnorm_lut, fq_matvec, rope_q15, attention_head_lut,
                            vec_alu2_add, ffn_chunked_lut, lm_head_chunked_lut,
                            argmax_lm_head, clip8, COS_ROM_16, SIN_ROM_16, COS_ROM, SIN_ROM)

def load_m_from_dump(path):
    d = json.load(open(path))
    cfg = d['cfg']
    m = {'cfg': cfg, 'tok_emb': np.array(d['tok_emb'], dtype=np.int8),
         'tok_emb_s': d['tok_emb_s'], 'rms_final': np.array(d['rms_final'], dtype=np.int8),
         'rms_final_s': d['rms_final_s']}
    layers = []
    for lay in d['layers']:
        layers.append({k: (np.array(v, dtype=np.int8) if isinstance(v, list) else v) for k, v in lay.items()})
    m['layers'] = layers
    return m

def forward_dbg(m, n_tokens=17, capture_pos=14):
    cfg = m['cfg']
    NL = cfg['n_layers']; H = cfg['n_heads']; KH = cfg['n_kv_heads']
    HS = cfg['head_size']; D = cfg['dim']
    TMAX = int(cfg.get('t_max', 32)); KVW = KH * HS
    HALF = HS // 2; NREP = H // KH
    score_sh = int(cfg.get('score_sh', 0))
    cos_rom = COS_ROM_16 if (HS == 16 and COS_ROM_16 is not None) else COS_ROM
    sin_rom = SIN_ROM_16 if (HS == 16 and SIN_ROM_16 is not None) else SIN_ROM
    kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    ksh = np.full((NL, TMAX), -128, dtype=np.int8)
    vsh = np.full((NL, TMAX), -128, dtype=np.int8)
    cur_tok = 1
    out = []
    for pos in range(n_tokens):
        cos_q15 = cos_rom[pos * HALF:pos * HALF + HALF]
        sin_q15 = sin_rom[pos * HALF:pos * HALF + HALF]
        XB = m['tok_emb'][cur_tok].astype(np.int8).copy()
        sxb = m['tok_emb_s']
        for l in range(NL):
            w = m['layers'][l]
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
            Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])
            Kc, sK = fq_matvec(w['wk'], XN, sxn, w['wk_s'])
            Vc, sV = fq_matvec(w['wv'], XN, sxn, w['wv_s'])
            Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
            for h in range(H):
                Q[h], _ = rope_q15(Q[h], sQ, cos_q15, sin_q15)
            for kh in range(KH):
                Kc[kh], _ = rope_q15(Kc[kh], sK, cos_q15, sin_q15)
            Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
            kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
            ksh[l, pos] = sK; vsh[l, pos] = sV
            sKref = int(ksh[l, 0:pos + 1].max())
            sVref = int(vsh[l, 0:pos + 1].max())
            ATT = np.zeros(D, dtype=np.int8)
            for h in range(H):
                kh = h // NREP
                Qh = Q[h * HS:(h + 1) * HS]
                T = pos + 1
                shK = sKref - ksh[l, 0:T].astype(np.int64)
                shV = sVref - vsh[l, 0:T].astype(np.int64)
                K_h = clip8((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]))
                V_h = clip8((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]))
                ATT[h * HS:(h + 1) * HS] = attention_head_lut(Qh, K_h, V_h, T, score_sh)
            sA = sVref
            OUTV, sOut = fq_matvec(w['wo'], ATT, sA, w['wo_s'])
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sOut)
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
            OUTV, sov = ffn_chunked_lut(w, XN, sxn)
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sov)
        XN, sxn = rmsnorm_lut(XB, sxb, m['rms_final'], m['rms_final_s'])
        logits, csh = lm_head_chunked_lut(m, XN, sxn)
        best_idx, _ = argmax_lm_head(logits, csh, D, cfg['vocab_size'])
        # top-1 v (pour comparaison au best_v du RTL a chaque token)
        sref_lm = max(csh); log2D = D.bit_length() - 1
        vtop = int(logits[best_idx]) >> (sref_lm - csh[best_idx >> log2D])
        print(f"pos{pos}: oracle best_idx={best_idx} best_v={vtop}")
        if pos == capture_pos:
            print(f"=== pos {pos} (cur_tok input={cur_tok}) ===")
            print(f"  csh={csh}")
            sref_lm = max(csh)
            log2D = D.bit_length() - 1
            vals = []
            for i in range(cfg['vocab_size']):
                chunk = i >> log2D
                v = int(logits[i]) >> (sref_lm - csh[chunk])
                vals.append((v, i))
            vals.sort(reverse=True)
            print(f"  top-8 (v, idx): {vals[:8]}")
            # marge entre 79 et 60
            v79 = int(logits[79]) >> (sref_lm - csh[79 >> log2D])
            v60 = int(logits[60]) >> (sref_lm - csh[60 >> log2D])
            print(f"  v[79]={v79}  v[60]={v60}  delta={v79-v60}")
            print(f"  oracle best_idx={best_idx}")
        out.append(int(best_idx))
        cur_tok = int(best_idx)
    return out

if __name__ == '__main__':
    path = os.path.join(HERE, 'gen_model_dump_dim128.json')
    m = load_m_from_dump(path)
    print("cfg:", m['cfg'])
    out = forward_dbg(m, 17, 14)
    print("node_tokens:", out)