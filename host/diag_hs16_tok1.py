#!/usr/bin/env python3
# diag_hs16_tok1.py -- Forward HS=16 (Strat B) 2 tokens propre, dump par-head au tok 1.
# But : determiner si l'oracle a un bug (heads 0,1 identiques ?) ou si le DUT a un bug.
import os
import numpy as np
from prove_gen_node import (build_quantized_model, rmsnorm_lut, fq_matvec, rope_q15,
                            attention_head_lut, vec_alu2_add, ffn_chunked_lut,
                            COS_ROM_16, SIN_ROM_16, clip8, EXP_LUT, INV_LUT, leading_bit)


def main():
    from infer_v5gen_ref import load_model, MODEL
    from prove_gen_node import lm_head_chunked_lut, argmax_lm_head
    m_float = load_model(MODEL)
    base = build_quantized_model(m_float)
    cfg0 = base['cfg']
    cfg0['n_heads'] = 4; cfg0['n_kv_heads'] = 2; cfg0['head_size'] = 16
    NL = cfg0['n_layers']; H = 4; KH = 2; HS = 16; D = 64
    TMAX = 32; KVW = KH * HS; HALF = HS // 2; NREP = H // KH
    cos_rom, sin_rom = COS_ROM_16, SIN_ROM_16

    for score_sh in (1, 0):
        import copy
        m = copy.deepcopy(base)
        m['cfg']['score_sh'] = score_sh
        kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
        vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
        ksh = np.full((NL, TMAX), -128, dtype=np.int8)
        vsh = np.full((NL, TMAX), -128, dtype=np.int8)
        cur_tok = 1
        for pos in range(2):
            cos_q15 = cos_rom[pos*HALF:pos*HALF+HALF]
            sin_q15 = sin_rom[pos*HALF:pos*HALF+HALF]
            XB = m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb = m['tok_emb_s']
            for l in range(NL):
                w = m['layers'][l]
                XN, sxn = rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
                if pos == 1 and l == 0 and score_sh == 1:
                    print(f"ORACLE XN[0:8]={list(int(x) for x in XN[:8])} sxn={sxn}  (XB=tok_emb[{cur_tok}][:8]={list(int(x) for x in XB[:8])} sxb={sxb})")
                Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])
                Kc, sK = fq_matvec(w['wk'], XN, sxn, w['wk_s'])
                Vc, sV = fq_matvec(w['wv'], XN, sxn, w['wv_s'])
                Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
                if pos == 1 and l == 0 and score_sh == 1:
                    print(f"ORACLE_PRE_ROPE Q[0,:8] ={list(int(x) for x in Q[0,:8])}  Kc[0,:8]={list(int(x) for x in Kc[0,:8])}  sQ={sQ} sK={sK}")
                    print(f"ORACLE cos_q15={list(int(x) for x in cos_q15)}  sin_q15={list(int(x) for x in sin_q15)}")
                for h in range(H): Q[h], _ = rope_q15(Q[h], sQ, cos_q15, sin_q15)
                for kh in range(KH): Kc[kh], _ = rope_q15(Kc[kh], sK, cos_q15, sin_q15)
                Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
                kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
                ksh[l, pos] = sK; vsh[l, pos] = sV
                sKref = int(ksh[l, 0:pos+1].max()); sVref = int(vsh[l, 0:pos+1].max())
                ATT = np.zeros(D, dtype=np.int8)
                for h in range(H):
                    kh = h // NREP
                    Qh = Q[h*HS:(h+1)*HS]; T = pos+1
                    shK = sKref - ksh[l, 0:T].astype(np.int64)
                    shV = sVref - vsh[l, 0:T].astype(np.int64)
                    K_h = np.clip((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]), -128, 127).astype(np.int8)
                    V_h = np.clip((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]), -128, 127).astype(np.int8)
                    out_h = attention_head_lut(Qh, K_h, V_h, T, score_sh)
                    ATT[h*HS:(h+1)*HS] = out_h
                    if pos == 1 and l == 0 and score_sh == 1:
                        yraw = Qh.astype(np.int64) @ K_h.astype(np.int64).T
                        sc = np.clip((yraw >> score_sh), -32768, 32767).astype(np.int64)
                        mx = int((sc >> 8).max())
                        print(f"ORACLEDGB T={T} sQ={sQ} sK={sK} sV={sV} stride={KVW} off={kh*HS} | Q0..3={int(Qh[0])} {int(Qh[1])} {int(Qh[2])} {int(Qh[3])} | s0={int(sc[0])} sT={int(sc[T-1])} max={mx} | out0..3={int(out_h[0])} {int(out_h[1])} {int(out_h[2])} {int(out_h[3])}")
                sA = sVref
                OUTV, sOut = fq_matvec(w['wo'], ATT, sA, w['wo_s'])
                XB, sxb = vec_alu2_add(XB, sxb, OUTV, sOut)
                XN, sxn = rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
                OUTV, sov = ffn_chunked_lut(w, XN, sxn)
                XB, sxb = vec_alu2_add(XB, sxb, OUTV, sov)
            XNf, sxnf = rmsnorm_lut(XB, sxb, m['rms_final'], m['rms_final_s'])
            logits, csh = lm_head_chunked_lut(m, XNf, sxnf)
            best, _ = argmax_lm_head(logits, csh)
            if pos == 1:
                sref = int(max(csh))
                li = {}
                for t in (403, 407, 261, 430, 414):
                    sh = sref - csh[t>>6]
                    li[t] = int(logits[t] >> sh) if sh >= 0 else int(logits[t] << (-sh))
                print(f"[score_sh={score_sh}] oracle tok1 argmax={best} XB[0:8]={list(XB[:8])} sxb={sxb} logits={li}")
            cur_tok = int(best)
    print("DUT tok1 : best_idx=403 best_val=98  logits[403]=98 [261]=48 [414]=48  XB[0:8]=[-12,-40,23,-51,14,16,27,9] sxb=-4")


if __name__ == "__main__":
    main()