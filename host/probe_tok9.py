#!/usr/bin/env python3
# probe_tok9.py -- dump logits/XN/KV au tok 9 (reference oracle hardware) pour
# comparer au sim mock FLASH_BOOT (qui drift a 298 au lieu de 268).
# Reutilise les helpers de prove_gen_node (modeles int8 bit-exact RTL).
import os, numpy as np
import prove_gen_node as P

def forward_probe(m, n_tokens=10):
    cfg = m['cfg']
    NL=cfg['n_layers']; H=cfg['n_heads']; KH=cfg['n_kv_heads']; HS=cfg['head_size']; D=cfg['dim']
    TMAX=32; KVW=KH*HS
    kmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8); vmem=np.zeros_like(kmem)
    ksh=np.full((NL,TMAX),-128,dtype=np.int8); vsh=np.full_like(ksh,-128)
    cur_tok=1; probes={}
    for pos in range(n_tokens):
        cos_q15=P.COS_ROM[pos*4:pos*4+4]; sin_q15=P.SIN_ROM[pos*4:pos*4+4]
        XB=m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb=m['tok_emb_s']
        for l in range(NL):
            w=m['layers'][l]
            XN,sxn=P.rmsnorm_lut(XB,sxb,w['rms_att'],w['rms_att_s'])
            Q,sQ=P.fq_matvec(w['wq'],XN,sxn,w['wq_s'])
            Kc,sK=P.fq_matvec(w['wk'],XN,sxn,w['wk_s'])
            Vc,sV=P.fq_matvec(w['wv'],XN,sxn,w['wv_s'])
            Q=Q.reshape(H,HS); Kc=Kc.reshape(KH,HS)
            for h in range(H): Q[h],_=P.rope_q15(Q[h],sQ,cos_q15,sin_q15)
            for kh in range(KH): Kc[kh],_=P.rope_q15(Kc[kh],sK,cos_q15,sin_q15)
            Q=Q.reshape(D); Kc=Kc.reshape(KVW); Vc=Vc.reshape(KVW)
            kmem[l,pos]=Kc.reshape(KH,HS); vmem[l,pos]=Vc.reshape(KH,HS)
            ksh[l,pos]=sK; vsh[l,pos]=sV
            sKref=int(ksh[l,0:pos+1].max()); sVref=int(vsh[l,0:pos+1].max())
            ATT=np.zeros(D,dtype=np.int8)
            for h in range(H):
                kh=h//2; Qh=Q[h*HS:(h+1)*HS]; T=pos+1
                shK=sKref-ksh[l,0:T].astype(np.int64); shV=sVref-vsh[l,0:T].astype(np.int64)
                K_h=P.clip8((kmem[l,0:T,kh].astype(np.int64)>>shK[:,None]))
                V_h=P.clip8((vmem[l,0:T,kh].astype(np.int64)>>shV[:,None]))
                ATT[h*HS:(h+1)*HS]=P.attention_head_lut(Qh,K_h,V_h,T)
            sA=sVref
            OUTV,sOut=P.fq_matvec(w['wo'],ATT,sA,w['wo_s'])
            XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sOut)
            XN,sxn=P.rmsnorm_lut(XB,sxb,w['rms_ffn'],w['rms_ffn_s'])
            OUTV,sov=P.ffn_chunked_lut(w,XN,sxn)
            XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sov)
        XN,sxn=P.rmsnorm_lut(XB,sxb,m['rms_final'],m['rms_final_s'])
        logits,csh=P.lm_head_chunked_lut(m,XN,sxn)
        best_idx,best_val=P.argmax_lm_head(logits,csh)
        if pos in (8,9):
            probes[pos]=dict(XN=XN.copy(),sxn=sxn,XB=XB.copy(),sxb=sxb,
                logits=logits.copy(),csh=csh.copy(),best_idx=int(best_idx),best_val=int(best_val),
                kmem_pos=kmem[:,pos].copy(),vmem_pos=vmem[:,pos].copy(),
                ksh_pos=ksh[:,pos].copy(),vsh_pos=vsh[:,pos].copy())
        cur_tok=int(best_idx)
    return probes

def main():
    from infer_v5gen_ref import load_model, MODEL
    m_float=load_model(MODEL); m=P.build_quantized_model(m_float)
    probes=forward_probe(m,10)
    for pos in (8,9):
        pr=probes[pos]
        lg=pr['logits']; csh=pr['csh']
        sref_lm=max(csh)
        v=lambda i: int(lg[i])>>(sref_lm-csh[i>>6]) if (sref_lm-csh[i>>6])>=0 else int(lg[i])<<(csh[i>>6]-sref_lm)
        print(f"=== tok {pos} : best_idx={pr['best_idx']} best_val={pr['best_val']} sxn={pr['sxn']} sxb={pr['sxb']} ===")
        print(f"  XN[0:8]={list(pr['XN'][:8])}")
        print(f"  XB[0:8]={list(pr['XB'][:8])}")
        print(f"  logits[268]={v(268)} logits[298]={v(298)} logits[261]={v(261)} logits[403]={v(403)} logits[414]={v(414)}")
        print(f"  csh={list(csh)}")
        print(f"  K (layer0,head0)[0:8]={list(pr['kmem_pos'][0,0,:8])}  sK={pr['ksh_pos'][0]}")
        print(f"  V (layer0,head0)[0:8]={list(pr['vmem_pos'][0,0,:8])}  sV={pr['vsh_pos'][0]}")
    np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),'tok9_probe.npz'),
             **{f"tok{p}_{k}":v for p,pr in probes.items() for k,v in pr.items()})
    print("dump tok9_probe.npz")

if __name__=="__main__":
    main()