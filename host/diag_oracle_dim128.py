#!/usr/bin/env python3
# diag_oracle_dim128.py -- Localiser le gap oracle vs RTL sur le dump dim128.
#
# dump XB (résiduel 128 octets) à chaque token, au MÊME format hex que le TB sim
# (sim_main.cpp : 32 mots uint32 %08x, byte0=LSB). Permet de diff oracle vs RTL
# token-par-token pour confirmer p0 exact / p1 diverge + isoler le pattern.
#
# Option --layers : dump aussi XB après CHAQUE couche à la position P donnée
# (--pos P), pour localiser la couche/stade qui introduit la divergence.
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prove_gen_node as P


def xb_hex(xb_i8):
    """128 int8 -> 32 mots uint32 %08x sep par espace (match sim_main.cpp XB=)."""
    x = np.asarray(xb_i8, dtype=np.int8).astype(np.uint8)
    words = []
    for w in range(32):
        b0, b1, b2, b3 = int(x[4*w]), int(x[4*w+1]), int(x[4*w+2]), int(x[4*w+3])
        val = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        words.append(f"{val:08x}")
    return " ".join(words)


def forward_diag(m, n_tokens=17, dump_layers_at=None):
    """Comme forward_node mais dump XB/token (+ XB par couche à dump_layers_at)."""
    cfg = m['cfg']
    NL = cfg['n_layers']; H = cfg['n_heads']; KH = cfg['n_kv_heads']
    HS = cfg['head_size']; D = cfg['dim']
    TMAX = int(cfg.get('t_max', 32)); KVW = KH * HS
    HALF = HS // 2; NREP = H // KH
    score_sh = int(cfg.get('score_sh', 0))
    cos_rom = P.COS_ROM_16 if (HS == 16 and P.COS_ROM_16 is not None) else P.COS_ROM
    sin_rom = P.SIN_ROM_16 if (HS == 16 and P.SIN_ROM_16 is not None) else P.SIN_ROM
    kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    ksh = np.full((NL, TMAX), -128, dtype=np.int8)
    vsh = np.full((NL, TMAX), -128, dtype=np.int8)
    cur_tok = 1; out = []; xb_per_tok = []
    for pos in range(n_tokens):
        cos_q15 = cos_rom[pos*HALF:pos*HALF+HALF]
        sin_q15 = sin_rom[pos*HALF:pos*HALF+HALF]
        XB = m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb = m['tok_emb_s']
        for l in range(NL):
            w = m['layers'][l]
            XN, sxn = P.rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
            Q, sQ = P.fq_matvec(w['wq'], XN, sxn, w['wq_s'])
            Kc, sK = P.fq_matvec(w['wk'], XN, sxn, w['wk_s'])
            Vc, sV = P.fq_matvec(w['wv'], XN, sxn, w['wv_s'])
            Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
            for h in range(H): Q[h], _ = P.rope_q15(Q[h], sQ, cos_q15, sin_q15)
            for kh in range(KH): Kc[kh], _ = P.rope_q15(Kc[kh], sK, cos_q15, sin_q15)
            Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
            kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
            ksh[l, pos] = sK; vsh[l, pos] = sV
            sKref = int(ksh[l, 0:pos+1].max()); sVref = int(vsh[l, 0:pos+1].max())
            ATT = np.zeros(D, dtype=np.int8)
            for h in range(H):
                kh = h // NREP; Qh = Q[h*HS:(h+1)*HS]; T = pos+1
                shK = sKref - ksh[l, 0:T].astype(np.int64)
                shV = sVref - vsh[l, 0:T].astype(np.int64)
                K_h = P.clip8((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]))
                V_h = P.clip8((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]))
                ATT[h*HS:(h+1)*HS] = P.attention_head_lut(Qh, K_h, V_h, T, score_sh)
            sA = sVref
            OUTV, sOut = P.fq_matvec(w['wo'], ATT, sA, w['wo_s'])
            XB, sxb = P.vec_alu2_add(XB, sxb, OUTV, sOut)
            XN, sxn = P.rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
            OUTV, sov = P.ffn_chunked_lut(w, XN, sxn)
            XB, sxb = P.vec_alu2_add(XB, sxb, OUTV, sov)
            if dump_layers_at == pos:
                print(f"  pos{pos} L{l:<1} post-FFN XB sxb={sxb:>3} : {xb_hex(XB)}")
        XN, sxn = P.rmsnorm_lut(XB, sxb, m['rms_final'], m['rms_final_s'])
        logits, csh = P.lm_head_chunked_lut(m, XN, sxn)
        best_idx, sref = P.argmax_lm_head(logits, csh, D, cfg['vocab_size'])
        out.append(int(best_idx)); xb_per_tok.append((XB.copy(), int(sxb)))
        cur_tok = int(best_idx)
    return out, xb_per_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=os.path.join(HERE, "gen_model_dump_dim128.json"))
    ap.add_argument("--ntok", type=int, default=17)
    ap.add_argument("--layers", type=int, default=None, help="dump XB par couche à cette pos")
    args = ap.parse_args()
    with open(args.dump) as f: m = json.load(f)
    # JSON -> numpy int8 (le dump est déjà int8 + shifts, comme attendu par forward_node)
    def to_i8(v): return np.array(v, dtype=np.int8)
    m['tok_emb'] = to_i8(m['tok_emb']); m['rms_final'] = to_i8(m['rms_final'])
    for l in range(len(m['layers'])):
        for k in ('rms_att','rms_ffn','wq','wk','wv','wo','w1','w2','w3'):
            m['layers'][l][k] = to_i8(m['layers'][l][k])
    print(f"# oracle diag dim128 (dump XB par token, format = TB sim_main.cpp)")
    if args.layers is not None:
        print(f"# XB par couche à pos={args.layers} :")
    out, xb_per_tok = forward_diag(m, args.ntok, dump_layers_at=args.layers)
    print(f"# oracle tokens = {out}")
    print(f"# RTL    tokens = [385, 385, 385, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 60, 324, 79]")
    print(f"#")
    for i, (xb, sxb) in enumerate(xb_per_tok):
        print(f"tok[{i:>2}] sxb={sxb:>3} XB={xb_hex(xb)}")


if __name__ == "__main__":
    main()