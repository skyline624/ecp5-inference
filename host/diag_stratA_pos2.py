#!/usr/bin/env python3
# diag_stratA_pos2.py -- Forward oracle HS=16 Strat A (random, shifts=0) pour pos=0,1,2.
# Dump par-head (Q pre/post-RoPE, K, scores, attn out) + XN/dot/shift pour comparer
# avec les $display debug du DUT (XNDGB/FQ/DOTDBG/SHFTDBG/ATTNDBG) a tok2.
import os, json
import numpy as np
from prove_gen_node import (build_quantized_model, rmsnorm_lut, fq_matvec, rope_q15,
                            attention_head_lut, vec_alu2_add, ffn_chunked_lut,
                            leading_bit, COS_ROM_16, SIN_ROM_16,
                            lm_head_chunked_lut, argmax_lm_head)

DUMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_model_dump_hs16_rand.json")


def main():
    with open(DUMP) as f:
        d = json.load(f)
    cfg = d['cfg']
    NL = cfg['n_layers']; H = cfg['n_heads']; KH = cfg['n_kv_heads']; HS = cfg['head_size']
    D = cfg['dim']; TMAX = cfg.get('t_max', 32); KVW = KH * HS; HALF = HS // 2
    NREP = H // KH; score_sh = cfg.get('score_sh', 1)
    cos_rom, sin_rom = COS_ROM_16, SIN_ROM_16
    print(f"cfg: H={H} KH={KH} HS={HS} D={D} KVW={KVW} NREP={NREP} score_sh={score_sh} NL={NL}")

    # reconstruire m depuis le dump (dump_to_dict format)
    m = {'cfg': dict(cfg), 'tok_emb': np.array(d['tok_emb'], dtype=np.int8),
         'tok_emb_s': d['tok_emb_s'], 'rms_final': np.array(d['rms_final'], dtype=np.int8),
         'rms_final_s': d['rms_final_s'], 'layers': []}
    for lay in d['layers']:
        L = {}
        for k in ('rms_att', 'rms_ffn'):
            L[k] = np.array(lay[k], dtype=np.int8); L[k + '_s'] = lay[k + '_s']
        for k in ('wq', 'wk', 'wv', 'wo', 'w1', 'w3', 'w2'):
            L[k] = np.array(lay[k], dtype=np.int8); L[k + '_s'] = lay[k + '_s']
        m['layers'].append(L)

    kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    ksh = np.full((NL, TMAX), -128, dtype=np.int8)
    vsh = np.full((NL, TMAX), -128, dtype=np.int8)
    cur_tok = 1  # BOS prefix hardwired -> cur_tok=1 au pos=0
    for pos in range(3):
        cos_q15 = cos_rom[pos * HALF:pos * HALF + HALF]
        sin_q15 = sin_rom[pos * HALF:pos * HALF + HALF]
        XB = m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb = m['tok_emb_s']
        for l in range(NL):
            w = m['layers'][l]
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
            Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])
            Kc, sK = fq_matvec(w['wk'], XN, sxn, w['wk_s'])
            Vc, sV = fq_matvec(w['wv'], XN, sxn, w['wv_s'])
            Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
            if pos >= 1 and l == 0:
                print(f"\n=== ORACLE pos={pos} L=0 cur_tok={cur_tok} sxn={sxn} ===")
                print(f"  XN[0:16]={list(int(x) for x in XN[:16])}")
                print(f"  wq[0][0:16]={list(int(x) for x in w['wq'][0,:16])}")
                yraw = w['wq'].astype(np.int64) @ XN.astype(np.int64)
                print(f"  yraw[0:4]={list(int(x) for x in yraw[:4])} max_abs={int(np.max(np.abs(yraw)))}")
                print(f"  Q_pre_rope[0][0:8]={list(int(x) for x in Q[0,:8])} sQ={sQ}")
            # RoPE
            for h in range(H): Q[h], _ = rope_q15(Q[h], sQ, cos_q15, sin_q15)
            for kh in range(KH): Kc[kh], _ = rope_q15(Kc[kh], sK, cos_q15, sin_q15)
            if pos >= 1 and l == 0:
                print(f"  Q_post_rope[0][0:8]={list(int(x) for x in Q[0,:8])}")
                print(f"  Kc[0][0:8]={list(int(x) for x in Kc[0,:8])} sK={sK}")
                print(f"  cos_q15[0:8]={list(int(x) for x in cos_q15)}")
                print(f"  sin_q15[0:8]={list(int(x) for x in sin_q15)}")
            Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
            kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
            ksh[l, pos] = sK; vsh[l, pos] = sV
            sKref = int(ksh[l, 0:pos + 1].max()); sVref = int(vsh[l, 0:pos + 1].max())
            ATT = np.zeros(D, dtype=np.int8)
            for h in range(H):
                kh = h // NREP; Qh = Q[h * HS:(h + 1) * HS]; T = pos + 1
                shK = sKref - ksh[l, 0:T].astype(np.int64)
                shV = sVref - vsh[l, 0:T].astype(np.int64)
                K_h = np.clip((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]), -128, 127).astype(np.int8)
                V_h = np.clip((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]), -128, 127).astype(np.int8)
                out_h = attention_head_lut(Qh, K_h, V_h, T, score_sh)
                ATT[h * HS:(h + 1) * HS] = out_h
                if pos == 2:
                    yraw = Qh.astype(np.int64) @ K_h.astype(np.int64).T
                    sc = np.clip((yraw >> score_sh), -32768, 32767).astype(np.int64)
                    if h == 0:
                        print(f"  L{l} ATTN h={h} kh={kh} T={T} sQ={sQ} sK={sK} sV={sV} sKref={sKref} sVref={sVref} "
                              f"Q0..3={int(Qh[0])} {int(Qh[1])} {int(Qh[2])} {int(Qh[3])} "
                              f"s0={int(sc[0])} sT={int(sc[T-1])} out0..15={list(int(x) for x in out_h[:16])}")
                    elif h == 1:
                        print(f"  L{l} ATTN h={h} kh={kh} T={T} sKref={sKref} sVref={sVref} "
                              f"Q0..3={int(Qh[0])} {int(Qh[1])} {int(Qh[2])} {int(Qh[3])} "
                              f"s0={int(sc[0])} sT={int(sc[T-1])} out0..15={list(int(x) for x in out_h[:16])}")
            OUTV, sOut = fq_matvec(w['wo'], ATT, sVref, w['wo_s'])
            if pos == 2:
                print(f"  L{l} WO sOut={sOut} OUTV0..7={list(int(x) for x in OUTV[:8])}")
                if l == 0:
                    yraw_wo = w['wo'].astype(np.int64) @ ATT.astype(np.int64)
                    print(f"  L0 WO yraw0..7={list(int(x) for x in yraw_wo[:8])} max_abs={int(np.max(np.abs(yraw_wo)))}")
                    print(f"  L0 ATT full0..15={list(int(x) for x in ATT[:16])}")
                    print(f"  L0 ATT 16..31={list(int(x) for x in ATT[16:32])}")
                    print(f"  L0 ATT 32..47={list(int(x) for x in ATT[32:48])}")
                    print(f"  L0 ATT 48..63={list(int(x) for x in ATT[48:64])}")
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sOut)
            if pos == 2:
                print(f"  L{l} afterWO_XB sxb={sxb} XB0..7={list(int(x) for x in XB[:8])}")
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
            OUTV, sov = ffn_chunked_lut(w, XN, sxn)
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sov)
            if pos == 2:
                print(f"  L{l} afterFFN_XB sxb={sxb} XB0..7={list(int(x) for x in XB[:8])}")
        XNf, sxnf = rmsnorm_lut(XB, sxb, m['rms_final'], m['rms_final_s'])
        logits, csh = lm_head_chunked_lut(m, XNf, sxnf)
        best, _ = argmax_lm_head(logits, csh)
        print(f"\n  ORACLE pos={pos} argmax={best}  XB[0:8]={list(int(x) for x in XB[:8])}")
        cur_tok = int(best)


if __name__ == "__main__":
    main()