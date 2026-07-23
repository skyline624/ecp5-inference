#!/usr/bin/env python3
# diag_xn_pos01.py -- Dump oracle XN[0:8] a pos=0 et pos=1 L=0 (HS=16 Strat B)
# pour comparer avec XNDGB du DUT.
import numpy as np
from prove_gen_node import build_quantized_model, rmsnorm_lut, fq_matvec, rope_q15
from prove_gen_node import COS_ROM_16, SIN_ROM_16
from infer_v5gen_ref import load_model, MODEL


def main():
    m_float = load_model(MODEL)
    base = build_quantized_model(m_float)
    cfg0 = base['cfg']
    cfg0['n_heads'] = 4; cfg0['n_kv_heads'] = 2; cfg0['head_size'] = 16; cfg0['score_sh'] = 1
    NL = cfg0['n_layers']; H = 4; KH = 2; HS = 16; D = 64
    TMAX = 32; KVW = KH * HS; HALF = HS // 2; NREP = H // KH
    cos_rom, sin_rom = COS_ROM_16, SIN_ROM_16

    import copy
    m = copy.deepcopy(base)
    kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    ksh = np.full((NL, TMAX), -128, dtype=np.int8)
    vsh = np.full((NL, TMAX), -128, dtype=np.int8)
    cur_tok = 1
    for pos in range(2):
        cos_q15 = cos_rom[pos*HALF:pos*HALF+HALF]
        sin_q15 = sin_rom[pos*HALF:pos*HALF+HALF]
        XB = m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb = m['tok_emb_s']
        if pos <= 1:
            print(f"ORACLE pos={pos} cur_tok={cur_tok} XB[0:8]={list(int(x) for x in XB[:8])} sxb={sxb}")
        for l in range(NL):
            w = m['layers'][l]
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
            if pos <= 1 and l == 0:
                print(f"ORACLE pos={pos} L=0 sxn={sxn} XN[0:32]={list(int(x) for x in XN[:32])}")
                Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])
                print(f"ORACLE pos={pos} L=0 sQ={sQ} Q[0:8]={list(int(x) for x in Q[:8])}")
                # raw dot y = wq@XN (int64), max_abs, leading_bit, shift_out (cf RTL DOTDBG/SHFTDBG)
                from prove_gen_node import leading_bit, fq_requant_lb6
                yraw = w['wq'].astype(np.int64) @ XN.astype(np.int64)
                y8, sh = fq_requant_lb6(yraw, sxn + w['wq_s'])
                mx = int(np.max(np.abs(yraw)))
                print(f"ORACLE pos={pos} L=0 yraw[0:4]={list(int(x) for x in yraw[:4])} "
                      f"max_abs={mx} lead={leading_bit(mx)} shift_out={sh - (sxn + w['wq_s'])} "
                      f"shift_total={sh} (sxn+wq_s={sxn + w['wq_s']})")
                print(f"ORACLE pos={pos} L=0 wq[0][0:32]={list(int(x) for x in w['wq'][0,:32])}")
            Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])
            Kc, sK = fq_matvec(w['wk'], XN, sxn, w['wk_s'])
            Vc, sV = fq_matvec(w['wv'], XN, sxn, w['wv_s'])
            Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
            for h in range(H): Q[h], _ = rope_q15(Q[h], sQ, cos_q15, sin_q15)
            for kh in range(KH): Kc[kh], _ = rope_q15(Kc[kh], sK, cos_q15, sin_q15)
            Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
            kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
            ksh[l, pos] = sK; vsh[l, pos] = sV
            # skip full attention for speed (only need XN/Q at L0)
            from prove_gen_node import attention_head_lut, vec_alu2_add, ffn_chunked_lut
            sKref = int(ksh[l, 0:pos+1].max()); sVref = int(vsh[l, 0:pos+1].max())
            ATT = np.zeros(D, dtype=np.int8)
            for h in range(H):
                kh = h // NREP; Qh = Q[h*HS:(h+1)*HS]; T = pos+1
                shK = sKref - ksh[l, 0:T].astype(np.int64)
                shV = sVref - vsh[l, 0:T].astype(np.int64)
                K_h = np.clip((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]), -128, 127).astype(np.int8)
                V_h = np.clip((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]), -128, 127).astype(np.int8)
                ATT[h*HS:(h+1)*HS] = attention_head_lut(Qh, K_h, V_h, T, 1)
            OUTV, sOut = fq_matvec(w['wo'], ATT, sVref, w['wo_s'])
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sOut)
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
            OUTV, sov = ffn_chunked_lut(w, XN, sxn)
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sov)
        from prove_gen_node import lm_head_chunked_lut, argmax_lm_head, rmsnorm_lut as _r
        XNf, sxnf = _r(XB, sxb, m['rms_final'], m['rms_final_s'])
        logits, csh = lm_head_chunked_lut(m, XNf, sxnf)
        best, _ = argmax_lm_head(logits, csh)
        print(f"ORACLE pos={pos} argmax={best}")
        cur_tok = int(best)


if __name__ == "__main__":
    main()