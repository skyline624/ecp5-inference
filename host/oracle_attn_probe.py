import numpy as np, sys, json
HERE = r'C:\Users\User\Downloads\ecp5-inference\host'
sys.path.insert(0, HERE)
import prove_gen_node as P

with open(HERE + r'\gen_model_dump_dim128.json') as f: m = json.load(f)
def to_i8(v): return np.array(v, dtype=np.int8)
m['tok_emb']=to_i8(m['tok_emb']); m['rms_final']=to_i8(m['rms_final'])
for l in range(len(m['layers'])):
    for k in ('rms_att','rms_ffn','wq','wk','wv','wo','w1','w2','w3'):
        m['layers'][l][k]=to_i8(m['layers'][l][k])

cfg=m['cfg']; NL=cfg['n_layers']; H=cfg['n_heads']; KH=cfg['n_kv_heads']
HS=cfg['head_size']; D=cfg['dim']; TMAX=int(cfg.get('t_max',32)); KVW=KH*HS
HALF=HS//2; NREP=H//KH; score_sh=int(cfg.get('score_sh',0))
cos_rom=P.COS_ROM_16 if (HS==16 and P.COS_ROM_16 is not None) else P.COS_ROM
sin_rom=P.SIN_ROM_16 if (HS==16 and P.SIN_ROM_16 is not None) else P.SIN_ROM
kmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8); vmem=np.zeros((NL,TMAX,KH,HS),dtype=np.int8)
ksh=np.full((NL,TMAX),-128,dtype=np.int8); vsh=np.full((NL,TMAX),-128,dtype=np.int8)

def attn_dump(Q_i8, K_i8, V_i8, T, score_sh, label=""):
    HS2=Q_i8.shape[0]
    y=Q_i8.astype(np.int64)@K_i8.astype(np.int64).T
    y=y>>score_sh; scores=np.clip(y,-32768,32767)
    proxy=(scores>>8).astype(np.int64); max_score=int(proxy.max())
    diff=scores-(max_score<<8); diff_sc=diff>>10
    idx=np.clip(diff_sc+256,0,255).astype(np.int64)
    ev=P.EXP_LUT[idx]; sum_exp=int(ev.sum())
    p_sum=P.leading_bit(sum_exp) if sum_exp>0 else 0
    shift_inv=8-p_sum
    sum_norm=sum_exp<<shift_inv if shift_inv>=0 else sum_exp>>(-shift_inv)
    inv_idx=sum_norm&0xFF; inv_q15=int(P.INV_LUT[inv_idx])
    norm_shift=15-shift_inv; attn_prod=ev*inv_q15
    if norm_shift>0: attn=(attn_prod+(1<<(norm_shift-1)))>>norm_shift
    elif norm_shift<0: attn=attn_prod<<(-norm_shift)
    else: attn=attn_prod
    attn=attn&0xFFFF
    acc=attn.astype(np.int64)@V_i8.astype(np.int64)
    out=P.clip8((acc+128)>>8)
    print(f"ORACLE {label} ATTN T={T} HS={HS2} max_score={max_score} sum_acc={sum_exp} inv_sum={inv_q15}")
    for t in range(T):
        print(f"  t={t} score={int(scores[t])} ev={int(ev[t])} attn={int(attn[t])}")
    for d in range(HS2):
        print(f"  OUT d={d} out_acc={int(acc[d])} out_clip={int(out[d])}")
    return out

cur_tok=1
for pos in range(2):
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
        ATT=np.zeros(D,dtype=np.int8)
        for h in range(H):
            kh=h//NREP; Qh=Q[h*HS:(h+1)*HS]; T=pos+1
            shK=sKref-ksh[l,0:T].astype(np.int64); shV=sVref-vsh[l,0:T].astype(np.int64)
            K_h=P.clip8((kmem[l,0:T,kh].astype(np.int64)>>shK[:,None]))
            V_h=P.clip8((vmem[l,0:T,kh].astype(np.int64)>>shV[:,None]))
            if pos==1 and l==0 and h==0:
                print(f"ORACLE Qh (16)={list(int(x) for x in Qh)}")
                for t in range(T):
                    print(f"ORACLE K_h[t={t}] (16)={list(int(x) for x in K_h[t])}")
                ATT[h*HS:(h+1)*HS]=attn_dump(Qh,K_h,V_h,T,score_sh,f"L0h0p1")
            else:
                ATT[h*HS:(h+1)*HS]=P.attention_head_lut(Qh,K_h,V_h,T,score_sh)
        sA=sVref; OUTV,sOut=P.fq_matvec(w['wo'],ATT,sA,w['wo_s'])
        XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sOut)
        XN,sxn=P.rmsnorm_lut(XB,sxb,w['rms_ffn'],w['rms_ffn_s'])
        OUTV,sov=P.ffn_chunked_lut(w,XN,sxn)
        XB,sxb=P.vec_alu2_add(XB,sxb,OUTV,sov)
    XN,sxn=P.rmsnorm_lut(XB,sxb,m['rms_final'],m['rms_final_s'])
    logits,csh=P.lm_head_chunked_lut(m,XN,sxn)
    bi,_=P.argmax_lm_head(logits,csh,D,cfg['vocab_size'])
    print(f"p{pos} token={bi}")
    cur_tok=int(bi)