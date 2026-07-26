import numpy as np, sys, json
HERE = r'C:\Users\User\Downloads\ecp5-inference\host'
sys.path.insert(0, HERE)
import prove_gen_node as P

with open(HERE + r'\gen_model_dump_dim128.json') as f: m = json.load(f)
def to_i8(v): return np.array(v, dtype=np.int8)
m['tok_emb'] = to_i8(m['tok_emb']); m['rms_final'] = to_i8(m['rms_final'])
for l in range(len(m['layers'])):
    for k in ('rms_att','rms_ffn','wq','wk','wv','wo','w1','w2','w3'):
        m['layers'][l][k] = to_i8(m['layers'][l][k])

cfg = m['cfg']; NL=cfg['n_layers']; H=cfg['n_heads']; KH=cfg['n_kv_heads']
HS=cfg['head_size']; D=cfg['dim']; TMAX=int(cfg.get('t_max',32)); KVW=KH*HS
HALF=HS//2; NREP=H//KH; score_sh=int(cfg.get('score_sh',0))
cos_rom = P.COS_ROM_16 if (HS==16 and P.COS_ROM_16 is not None) else P.COS_ROM
sin_rom = P.SIN_ROM_16 if (HS==16 and P.SIN_ROM_16 is not None) else P.SIN_ROM
kmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8); vmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8)
ksh=np.full((NL,TMAX),-128,dtype=np.int8); vsh=np.full((NL,TMAX),-128,dtype=np.int8)
cur_tok=1
for pos in range(2):  # p0 puis p1
    cos=cos_rom[pos*HALF:pos*HALF+HALF]; sin=sin_rom[pos*HALF:pos*HALF+HALF]
    XB=m['tok_emb'][cur_tok].astype(np.int8).copy(); sxb=m['tok_emb_s']
    for l in range(NL):
        w=m['layers'][l]
        XN,sxn=P.rmsnorm_lut(XB,sxb,w['rms_att'],w['rms_att_s'])
        Q,sQ=P.fq_matvec(w['wq'],XN,sxn,w['wq_s'])
        Kc,sK=P.fq_matvec(w['wk'],XN,sxn,w['wk_s'])
        Vc,sV=P.fq_matvec(w['wv'],XN,sxn,w['wv_s'])
        Q=Q.reshape(H,HS); Kc=Kc.reshape(KH,HS)
        for h in range(H): Q[h],_=P.rope_q15(Q[h],sQ,cos,sin)
        for kh in range(KH): Kc[kh],_=P.rope_q15(Kc[kh],sK,cos,sin)
        Q=Q.reshape(D); Kc=Kc.reshape(KVW); Vc=Vc.reshape(KVW)
        kmem[l,pos]=Kc.reshape(KH,HS); vmem[l,pos]=Vc.reshape(KH,HS)
        ksh[l,pos]=sK; vsh[l,pos]=sV
        sKref=int(ksh[l,0:pos+1].max()); sVref=int(vsh[l,0:pos+1].max())
        if pos==1 and l==0:
            print(f"=== p1 L0 : KV shifts ===")
            print(f"  ksh[0:2]={list(ksh[l,0:2])} vsh[0:2]={list(vsh[l,0:2])}")
            print(f"  sKref={sKref} sVref={sVref}")
            shK=sKref-ksh[l,0:2].astype(np.int64); shV=sVref-vsh[l,0:2].astype(np.int64)
            print(f"  shK (re-align K)={list(shK)}  shV={list(shV)}")
            print(f"  re-align actif ? K={shK.any()} V={shV.any()}")
            print(f"  Kc raw[0] (pos1, kh0, first 8)={list(Kc.reshape(KH,HS)[0,:8])}")
            print(f"  K[0] raw (kh0 first8)={list(kmem[l,0,0,:8])} sh={int(ksh[l,0])}")
            print(f"  K[1] raw (kh0 first8)={list(kmem[l,1,0,:8])} sh={int(ksh[l,1])}")
            K0=P.clip8((kmem[l,0,0].astype(np.int64)>>shK[0]))
            K1=P.clip8((kmem[l,1,0].astype(np.int64)>>shK[1]))
            print(f"  K[0] realign first8={list(K0[:8])} (shK={int(shK[0])})")
            print(f"  K[1] realign first8={list(K1[:8])} (shK={int(shK[1])})")
        ATT=np.zeros(D,dtype=np.int8)
        for h in range(H):
            kh=h//NREP; Qh=Q[h*HS:(h+1)*HS]; T=pos+1
            shK=sKref-ksh[l,0:T].astype(np.int64); shV=sVref-vsh[l,0:T].astype(np.int64)
            K_h=P.clip8((kmem[l,0:T,kh].astype(np.int64)>>shK[:,None]))
            V_h=P.clip8((vmem[l,0:T,kh].astype(np.int64)>>shV[:,None]))
            ATT[h*HS:(h+1)*HS]=P.attention_head_lut(Qh,K_h,V_h,T,score_sh)
        if pos==1 and l==0:
            print(f"  ATT[0:8]={list(ATT[:8])}  sA=sVref={sVref}")
        sA=sVref; OUTV,sOut=P.fq_matvec(w['wo'],ATT,sA,w['wo_s'])
        XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sOut)
        XN,sxn=P.rmsnorm_lut(XB,sxb,w['rms_ffn'],w['rms_ffn_s'])
        OUTV,sov=P.ffn_chunked_lut(w,XN,sxn)
        XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sov)
    XN,sxn=P.rmsnorm_lut(XB,sxb,m['rms_final'],m['rms_final_s'])
    logits,csh=P.lm_head_chunked_lut(m,XN,sxn)
    bi,_=P.argmax_lm_head(logits,csh,D,cfg['vocab_size'])
    print(f"p{pos} : token={bi} sxb={sxb}")
    cur_tok=int(bi)