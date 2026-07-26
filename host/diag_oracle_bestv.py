# diag_oracle_bestv.py -- dump oracle best_idx + best_v (top-1 v après re-align) par position
# pour comparer au dbg_best_v/dbg_best_i du RTL (sim_main.cpp). Usage : python diag_oracle_bestv.py <dump.json>
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prove_gen_node import (forward_node, lm_head_chunked_lut, argmax_lm_head, rmsnorm_lut,
                            fq_matvec, rope_q15, attention_head_lut, vec_alu2_add,
                            ffn_chunked_lut, clip8, COS_ROM_16, SIN_ROM_16, COS_ROM, SIN_ROM)

def load_m(path):
    d = json.load(open(path))
    m = {'cfg': d['cfg'],
         'tok_emb': np.array(d['tok_emb'], dtype=np.int8), 'tok_emb_s': d['tok_emb_s'],
         'rms_final': np.array(d['rms_final'], dtype=np.int8), 'rms_final_s': d['rms_final_s']}
    m['layers'] = [{k: (np.array(v, dtype=np.int8) if isinstance(v, list) else v) for k, v in lay.items()} for lay in d['layers']]
    return m

def forward_bestv(m, n_tokens=17):
    cfg = m['cfg']
    NL=cfg['n_layers']; H=cfg['n_heads']; KH=cfg['n_kv_heads']; HS=cfg['head_size']; D=cfg['dim']
    TMAX=int(cfg.get('t_max',32)); KVW=KH*HS; HALF=HS//2; NREP=H//KH
    score_sh=int(cfg.get('score_sh',0))
    cos_rom = COS_ROM_16 if (HS==16 and COS_ROM_16 is not None) else COS_ROM
    sin_rom = SIN_ROM_16 if (HS==16 and SIN_ROM_16 is not None) else SIN_ROM
    kmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8); vmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8)
    ksh=np.full((NL,TMAX),-128,dtype=np.int8); vsh=np.full((NL,TMAX),-128,dtype=np.int8)
    cur_tok=1; out=[]
    for pos in range(n_tokens):
        cos_q15=cos_rom[pos*HALF:pos*HALF+HALF]; sin_q15=sin_rom[pos*HALF:pos*HALF+HALF]
        XB=m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb=m['tok_emb_s']
        for l in range(NL):
            w=m['layers'][l]
            XN,sxn=rmsnorm_lut(XB,sxb,w['rms_att'],w['rms_att_s'])
            Q,sQ=fq_matvec(w['wq'],XN,sxn,w['wq_s']); Kc,sK=fq_matvec(w['wk'],XN,sxn,w['wk_s']); Vc,sV=fq_matvec(w['wv'],XN,sxn,w['wv_s'])
            Q=Q.reshape(H,HS); Kc=Kc.reshape(KH,HS)
            for h in range(H): Q[h],_=rope_q15(Q[h],sQ,cos_q15,sin_q15)
            for kh in range(KH): Kc[kh],_=rope_q15(Kc[kh],sK,cos_q15,sin_q15)
            Q=Q.reshape(D); Kc=Kc.reshape(KVW); Vc=Vc.reshape(KVW)
            kmem[l,pos]=Kc.reshape(KH,HS); vmem[l,pos]=Vc.reshape(KH,HS); ksh[l,pos]=sK; vsh[l,pos]=sV
            sKref=int(ksh[l,0:pos+1].max()); sVref=int(vsh[l,0:pos+1].max())
            ATT=np.zeros(D,dtype=np.int8)
            for h in range(H):
                kh=h//NREP; Qh=Q[h*HS:(h+1)*HS]; T=pos+1
                shK=sKref-ksh[l,0:T].astype(np.int64); shV=sVref-vsh[l,0:T].astype(np.int64)
                K_h=clip8((kmem[l,0:T,kh].astype(np.int64)>>shK[:,None])); V_h=clip8((vmem[l,0:T,kh].astype(np.int64)>>shV[:,None]))
                ATT[h*HS:(h+1)*HS]=attention_head_lut(Qh,K_h,V_h,T,score_sh)
            sA=sVref; OUTV,sOut=fq_matvec(w['wo'],ATT,sA,w['wo_s'])
            XB,sxb=vec_alu2_add(XB,sxb,OUTV,sOut)
            XN,sxn=rmsnorm_lut(XB,sxb,w['rms_ffn'],w['rms_ffn_s'])
            OUTV,sov=ffn_chunked_lut(w,XN,sxn)
            XB,sxb=vec_alu2_add(XB,sxb,OUTV,sov)
        XN,sxn=rmsnorm_lut(XB,sxb,m['rms_final'],m['rms_final_s'])
        logits,csh=lm_head_chunked_lut(m,XN,sxn)
        best_idx,_=argmax_lm_head(logits,csh,D,cfg['vocab_size'])
        sref_lm=max(csh); log2D=D.bit_length()-1
        vtop=int(logits[best_idx])>>(sref_lm-csh[best_idx>>log2D])
        # dump XB (result) : 32 mots uint32 little-endian par octet (match RTL result[ldi*8+:8])
        xb_bytes = [int(b) & 0xFF for b in XB[:D]]
        words = []
        for w in range(32):
            v=0
            for b in range(4):
                bi=w*4+b
                if bi < len(xb_bytes): v |= (xb_bytes[bi] & 0xFF) << (b*8)
            words.append(v & 0xFFFFFFFF)
        hexw = " ".join(f"{w:08x}" for w in words)
        # sxb signé -> uint8
        sxb_u = int(sxb) & 0xFF
        print(f"pos{pos}: best_idx={best_idx} best_v={vtop} sh={sxb_u} XB={hexw}")
        out.append(int(best_idx)); cur_tok=int(best_idx)
    return out

if __name__=='__main__':
    path = sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,'gen_model_dump.json')
    m=load_m(path); print("cfg:",m['cfg']); forward_bestv(m,17)