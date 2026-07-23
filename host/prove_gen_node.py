#!/usr/bin/env python3
# =============================================================================
# prove_gen_node.py -- Modele int8 BIT-EXACT du nœud RTL (gen_seq + opérateurs).
#
# Reproduit en Python/numpy exactement l'arithmétique entière du nœud hardware
# (src/gen_seq.v, src/rmsnorm_op.v, src/attention_head_op.v, src/vec_alu2.v,
#  src/silu_op.v, src/rope_op.v, opérateur FQ du node_top) pour produire les 17
# tokens de génération autonome (BOS=1, pos 0..16). Sert de RÉFÉRENCE au test
# cocotb test_gen_F (preuve RTL == nœud 17/17).
#
# Convention numérique : (int8, shift), real = int8 * 2**shift. Toute l'arithmétique
# en int (jamais float dans les chemins bit-exact). `>>` Python = arithmétique vers
# -inf (idem Verilog `>>>` sur signed).
#
# Source de vérité RTL : gen_seq.v (ordre forward, signaux sKref/sVref/csh/
# sref_lm/lraw/ksh/vsh, clip8, >>>). Les LUTs sont chargées depuis lut/*.hex
# (bit-identiques au $readmemh du RTL).
#
# Divergence @token 9 : le nœud calcule un softmax SIMPLIFIÉ (pas de /√HS,
# score>>10, LUT exp/inv) qui fait basculer un near-tie gap-3 : float oracle -> 298,
# hardware/nœud -> 268. Les 9 premiers tokens coïncident avec l'oracle float.
# =============================================================================
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LUT = os.path.join(HERE, "..", "lut")

# ---- Quantification helpers (adaptés de dsp_inference/host/v4_quant.py) --------
def to_i8_shift(x_float):
    """float -> (int8, shift) tel que real ~= int8 * 2**shift."""
    max_abs = float(np.max(np.abs(x_float)))
    if max_abs == 0.0:
        return np.zeros_like(x_float, dtype=np.int8), 0
    shift = int(np.ceil(np.log2(max_abs / 127.0)))
    return np.clip(np.round(x_float / (2.0 ** shift)), -128, 127).astype(np.int8), shift


def from_i8_shift(x_i8, shift):
    return x_i8.astype(np.float64) * (2.0 ** shift)


def clip8(x):
    """Clip signed vers int8 [-128, 127] (clip8 du RTL)."""
    return np.clip(x, -128, 127).astype(np.int8)


def leading_bit(v):
    """Index du bit de tête de |v| (msb_idx du RTL), 0 si v==0."""
    a = int(abs(int(v)))
    return a.bit_length() - 1 if a > 0 else 0


# ---- Chargement des LUT .hex (bit-identiques au $readmemh du RTL) -------------
def load_hex(name, signed=False):
    vals = []
    with open(os.path.join(LUT, name)) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            v = int(line, 16)
            if signed and v >= 32768:
                v -= 65536
            vals.append(v)
    return np.array(vals, dtype=np.int64)


# LUTs chargées une fois au module load (mêmes .hex que le RTL).
RSQRT_LUT = load_hex("rsqrt_lut.hex", signed=False)   # 256 Q1.15 unsigned
EXP_LUT   = load_hex("exp_lut.hex",   signed=False)   # 256 Q15   unsigned
INV_LUT   = load_hex("inv_lut.hex",   signed=False)   # 256 Q15   unsigned
SILU_LUT  = load_hex("silu_lut.hex",  signed=True)    # 256 Q4.11 signed
COS_ROM   = load_hex("freq_cis_cos.hex", signed=True) # 68 Q15 signed (17 pos × 4, HS=8)
SIN_ROM   = load_hex("freq_cis_sin.hex", signed=True)
# ROM HS=16 (rework HS) : 136 entries (17 pos × 8). Présentes si lut/freq_cis_*_hs16.hex.
def _try_load(name):
    p = os.path.join(LUT, name)
    return load_hex(name, signed=True) if os.path.exists(p) else None
COS_ROM_16 = _try_load("freq_cis_cos_hs16.hex")
SIN_ROM_16 = _try_load("freq_cis_sin_hs16.hex")
SQRT2_Q15 = 0xB505


# ---- FQ : requantize lb-6 (pivot, utilisé pour TOUS les matmuls) -------------
def fq_requant_lb6(y_i32, shift_in):
    """int32 (shift_in) -> (int8, shift_out). Formule exacte du FQ node + vec_alu2 REQ."""
    max_abs = int(np.max(np.abs(y_i32))) if y_i32.size else 0
    lb = leading_bit(max_abs)
    sh = max(0, lb - 6)
    rounding = (1 << (sh - 1)) if sh > 0 else 0
    out = clip8((y_i32 + rounding) >> sh)   # >> arith
    return out, shift_in + sh


# ---- Matvecs chunkés bit-exact (NE PAS réaligner en float) -------------------
def fq_matvec(W_i8, x_i8, sx, sw):
    """Matvec simple (N≤64, K≤64) : y = W@x int64 puis FQ lb-6."""
    y = W_i8.astype(np.int64) @ x_i8.astype(np.int64)
    return fq_requant_lb6(y, sx + sw)


def fq_matvec_chunked_N(W_i8, x_i8, sx, sw, n_chunks=3, chunk=64):
    """W1/W3 (N=172 paddé à 192, 3 chunks de 64). FQ lb-6 PAR chunk (shifts par chunk)."""
    N = W_i8.shape[0]
    # pad à 192 (20 lignes zéro) -> 3 chunks de 64 propres, comme le RTL.
    Wpad = np.zeros((n_chunks * chunk, W_i8.shape[1]), dtype=np.int8)
    Wpad[:N] = W_i8
    out = np.zeros(n_chunks * chunk, dtype=np.int8)
    shifts = []
    for c in range(n_chunks):
        a = c * chunk; b = a + chunk
        y = Wpad[a:b].astype(np.int64) @ x_i8.astype(np.int64)
        y8, sh = fq_requant_lb6(y, sx + sw)
        out[a:b] = y8
        shifts.append(sh)
    return out, shifts


def fq_matvec_w2(w2_i8, hg_i8, shc, sw2_s):
    """W2 [D,HID] 3 blocs de 64 colonnes. FQ lb-6 par chunk, reduce via vec_alu2_add."""
    D, HID = w2_i8.shape
    n_chunks = 3; chunk = 64
    # pad colonnes à 192 (hg_i8 est déjà [192] padded par fq_matvec_chunked_N)
    Wpad = np.zeros((D, n_chunks * chunk), dtype=np.int8)
    Wpad[:, :HID] = w2_i8
    hpad = hg_i8
    OUTV = None; sov = None
    for c in range(n_chunks):
        a = c * chunk; b = a + chunk
        y = Wpad[:, a:b].astype(np.int64) @ hpad[a:b].astype(np.int64)   # [D]
        P, spc = fq_requant_lb6(y, shc[c] + sw2_s)
        if c == 0:
            OUTV, sov = P.copy(), spc
        else:
            OUTV, sov = vec_alu2_add(OUTV, sov, P, spc)
    return OUTV, sov


# ---- vec_alu2 : MUL et ADD (adaptés de src/vec_alu2.v) ------------------------
def vec_alu2_mul(a_i8, sa, b_i8, sb):
    """MUL : acc=a*b, FQ lb-6. out_sh = (sa+sb) + add_sh."""
    acc = a_i8.astype(np.int64) * b_i8.astype(np.int64)
    max_abs = int(np.max(np.abs(acc))) if acc.size else 0
    lb = leading_bit(max_abs)
    add_sh = max(0, lb - 6)
    rounding = (1 << (add_sh - 1)) if add_sh > 0 else 0
    out = clip8((acc + rounding) >> add_sh)
    return out, (sa + sb) + add_sh


def vec_alu2_add(a_i8, sa, b_i8, sb):
    """ADD : aligne sur MIN(sa,sb) (base_sh), a<<da + b<<db_, FQ lb-6. out_sh=min+add_sh."""
    base_sh = min(sa, sb)
    da = sa - base_sh   # >= 0
    db_ = sb - base_sh  # >= 0
    add_val = (a_i8.astype(np.int64) << da) + (b_i8.astype(np.int64) << db_)
    max_abs = int(np.max(np.abs(add_val))) if add_val.size else 0
    lb = leading_bit(max_abs)
    add_sh = max(0, lb - 6)
    rounding = (1 << (add_sh - 1)) if add_sh > 0 else 0
    out = clip8((add_val + rounding) >> add_sh)
    return out, base_sh + add_sh


# ---- RMSNorm (adapté de src/rmsnorm_op.v + test_rmsnorm.expected_intermediates)
def rmsnorm_lut(x_i8, sx, w_i8, sw):
    """RMSNorm LUT rsqrt. shift_out = sw (shift_x IGNORÉ côté shift_out, cf rmsnorm_op.v:202)."""
    x64 = x_i8.astype(np.int64)
    acc = int((x64 * x64).sum())                          # int (24-bit en RTL)
    p = leading_bit(acc) if acc > 0 else 0
    shift_amt = 8 - p                                     # signed 6-bit
    if shift_amt >= 0:
        acc_norm = acc << shift_amt
    else:
        acc_norm = acc >> (-shift_amt)                    # acc > 0 -> logical = arith
    index = acc_norm & 0xFF
    raw_inv = int(RSQRT_LUT[index])
    if shift_amt & 1:
        raw_inv = (raw_inv * SQRT2_Q15) >> 15             # parité √2
    apply_shift = (shift_amt >> 1) - 16                   # arith (Python >> signed = -inf = Verilog >>>)
    prod = x64 * w_i8.astype(np.int64) * raw_inv          # signed int64
    if apply_shift >= 0:
        shifted = prod << apply_shift
    else:
        # apply_shift < 0 : rounding 0 si -1 (troncature pure), sinon 1<<(-apply_shift-1)
        rounding = 0 if apply_shift == -1 else (1 << (-apply_shift - 1))
        shifted = (prod + rounding) >> (-apply_shift)     # arith
    return clip8(shifted), sw


# ---- SiLU (adapté de src/silu_op.v + test_silu.silu_ref) ---------------------
def silu_lut(x_i8, sx):
    """SiLU LUT Q4.11. shift_out = sx."""
    shx_p4 = sx + 4
    x64 = x_i8.astype(np.int64)
    if shx_p4 >= 0:
        x16 = x64 << shx_p4
    else:
        x16 = x64 >> (-shx_p4)                             # arith
    idx = np.clip(x16 + 128, 0, 255).astype(np.int64)
    silu_int = SILU_LUT[idx]                              # Q4.11 signed
    out_shift = 11 + sx
    if out_shift > 0:
        o = (silu_int + (1 << (out_shift - 1))) >> out_shift
    elif out_shift < 0:
        o = silu_int << (-out_shift)
    else:
        o = silu_int
    return clip8(o), sx


# ---- RoPE (adapté de src/rope_op.v + test_rope.rope_ref) ---------------------
def rope_q15(x_i8, sx, cos_q15, sin_q15):
    """RoPE Q15 sur HS éléments (HALF paires). shift_out = sx. HS dérivé de x_i8."""
    HS = x_i8.shape[0]; HALF = HS // 2
    out = np.zeros(HS, dtype=np.int8)
    for i in range(HALF):
        xr = int(x_i8[2 * i]); xi = int(x_i8[2 * i + 1])
        c = int(cos_q15[i]); s = int(sin_q15[i])
        nr = xr * c - xi * s
        ni = xr * s + xi * c
        out[2 * i]     = int(np.clip((nr + 16384) >> 15, -128, 127))
        out[2 * i + 1] = int(np.clip((ni + 16384) >> 15, -128, 127))
    return out, sx


# ---- Attention single-head (adapté de src/attention_head_op.v) ---------------
# IMPORTANT : score 16-bit COMPLET (pas tronqué low byte), contrairement à
# attn_ref_fpga (test_attention.py) qui tronque. Le RTL (attention_head_op.v:142)
# utilise score_curr = score_at_idx (16-bit) - max_ext_lshift, donc garde le low
# byte. C'est crucial pour le near-tie @token 9.
def attention_head_lut(Q_i8, K_i8, V_i8, T, score_sh=0):
    """Attention 1 head, HS dynamique, T positions. Retourne out_i8[HS]. shift_out = sV (appelant).
    score_sh = pre-shift du score avant clip int16 (calibrage HS, 0=HS=8, 1=HS=16)."""
    HS = Q_i8.shape[0]
    # 1) scores = (Q.K[t] >>> score_sh) clipé int16
    y = Q_i8.astype(np.int64) @ K_i8.astype(np.int64).T   # [T] int64
    y = y >> score_sh                                      # arith pre-shift (0 = HS=8 inchangé)
    scores = np.clip(y, -32768, 32767)                   # int16 signed
    # 2) max sur high-byte (score>>8 int8 signed)
    proxy = (scores >> 8).astype(np.int64)               # arith, = signed high byte
    max_score = int(proxy.max())
    # 3) LUT exp (score COMPLET, pas tronqué)
    diff = scores - (max_score << 8)                      # int, garde le low byte
    diff_sc = diff >> 10                                  # arith
    idx = np.clip(diff_sc + 256, 0, 255).astype(np.int64)
    ev = EXP_LUT[idx]                                    # [T] unsigned Q15
    sum_exp = int(ev.sum())
    # 4) LUT inv sur sum_exp
    p_sum = leading_bit(sum_exp) if sum_exp > 0 else 0
    shift_inv = 8 - p_sum
    if shift_inv >= 0:
        sum_norm = sum_exp << shift_inv
    else:
        sum_norm = sum_exp >> (-shift_inv)               # sum_exp > 0
    inv_idx = sum_norm & 0xFF
    inv_q15 = int(INV_LUT[inv_idx])
    # 5) attn[t] = exp[t] * inv >> norm_shift, Q15 unsigned
    norm_shift = 15 - shift_inv
    attn_prod = ev * inv_q15                             # unsigned
    if norm_shift > 0:
        attn = (attn_prod + (1 << (norm_shift - 1))) >> norm_shift
    elif norm_shift < 0:
        attn = attn_prod << (-norm_shift)
    else:
        attn = attn_prod
    attn = attn & 0xFFFF                                 # 16-bit (positif)
    # 6) out[d] = clip8(sum_t attn[t]*V[t,d] + 128 >> 8)
    acc = attn.astype(np.int64) @ V_i8.astype(np.int64)  # [HS] signed (V signed)
    out = clip8((acc + 128) >> 8)                         # arith
    return out


# ---- Argmax lm_head (adapté de gen_seq.v AMAX_INIT/AMAX/AMAXC) ----------------
def argmax_lm_head(logits_i8, csh):
    """sref_lm = max(csh). v = logits[i] >>> (sref_lm - csh[i>>6]). ties = premier index."""
    sref_lm = max(csh)
    best_val = -2147483647
    best_idx = 0
    for i in range(len(logits_i8)):
        chunk = i >> 6
        v = int(logits_i8[i]) >> (sref_lm - csh[chunk])  # sref_lm>=csh -> shift >= 0
        if v > best_val:
            best_val = v
            best_idx = i
    return best_idx, sref_lm


# ---- FFN chunké (W1/W3 3 chunks, silu/MUL par chunk, W2 3 chunks + reduce) ---
def ffn_chunked_lut(w, XN_i8, sxn):
    H1, s1c = fq_matvec_chunked_N(w['w1'], XN_i8, sxn, w['w1_s'])   # [192], [3]
    H3, s3c = fq_matvec_chunked_N(w['w3'], XN_i8, sxn, w['w3_s'])
    # silu par chunk
    SG = np.zeros(192, dtype=np.int8); ssgc = [0, 0, 0]
    for c in range(3):
        a = c * 64; b = a + 64
        sg_c, _ = silu_lut(H1[a:b], s1c[c])
        SG[a:b] = sg_c
        ssgc[c] = s1c[c]                                # silu shift_out = sx
    # MUL par chunk
    HG = np.zeros(192, dtype=np.int8); shc = [0, 0, 0]
    for c in range(3):
        a = c * 64; b = a + 64
        att_c, sh = vec_alu2_mul(SG[a:b], ssgc[c], H3[a:b], s3c[c])
        HG[a:b] = att_c
        shc[c] = sh
    # W2 (3 chunks K) + reduce
    OUTV, sov = fq_matvec_w2(w['w2'], HG, shc, w['w2_s'])
    return OUTV, sov


# ---- LM head chunké (8 chunks de 64, csh[8]) --------------------------------
def lm_head_chunked_lut(m, XN_i8, sxn):
    tok_emb = m['tok_emb']                              # [512, 64] int8
    logits = np.zeros(512, dtype=np.int8)
    csh = [0] * 8
    for c in range(8):
        a = c * 64; b = a + 64
        y = tok_emb[a:b].astype(np.int64) @ XN_i8.astype(np.int64)
        y8, sh = fq_requant_lb6(y, sxn + m['tok_emb_s'])
        logits[a:b] = y8
        csh[c] = sh
    return logits, csh


# ---- Construction du modèle quantisé (m_float -> m int8 + shifts) ------------
def build_quantized_model(m_float):
    """Quantize chaque tenseur (to_i8_shift) et restructure en dict m int8+shifts
    compatible avec forward_node ET preload_model de test_gen_F."""
    cfg = m_float['cfg']
    m = {
        'cfg': cfg,
        'tok_emb': to_i8_shift(m_float['tok_emb'])[0].astype(np.int8),
        'tok_emb_s': int(to_i8_shift(m_float['tok_emb'])[1]),
        'rms_final': to_i8_shift(m_float['rms_final'])[0].astype(np.int8),
        'rms_final_s': int(to_i8_shift(m_float['rms_final'])[1]),
        'layers': [],
    }
    for l in range(cfg['n_layers']):
        lay = {}
        for key in ('rms_att', 'rms_ffn'):
            q, s = to_i8_shift(m_float[key][l])
            lay[key] = q.astype(np.int8); lay[key + '_s'] = int(s)
        for key in ('wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3'):
            q, s = to_i8_shift(m_float[key][l])
            lay[key] = q.astype(np.int8); lay[key + '_s'] = int(s)
        m['layers'].append(lay)
    return m


# ---- Orchestrateur : forward 17 tokens (ordre forward exact de gen_seq.v) -----
def forward_node(m, n_tokens=17):
    cfg = m['cfg']
    NL = cfg['n_layers']; H = cfg['n_heads']; KH = cfg['n_kv_heads']
    HS = cfg['head_size']; D = cfg['dim']
    TMAX = 32; KVW = KH * HS
    HALF = HS // 2; NREP = H // KH
    score_sh = int(cfg.get('score_sh', 0))   # pre-shift score (0=HS=8, 1=HS=16)
    # ROM RoPE selon HS (HS=16 -> fichiers _hs16, sinon HS=8)
    cos_rom = COS_ROM_16 if (HS == 16 and COS_ROM_16 is not None) else COS_ROM
    sin_rom = SIN_ROM_16 if (HS == 16 and SIN_ROM_16 is not None) else SIN_ROM
    # KV cache persistant (seed -128 pour les shifts, cf gen_seq pitfall #2)
    kmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    vmem = np.zeros((NL, TMAX, KH, HS), dtype=np.int8)
    ksh = np.full((NL, TMAX), -128, dtype=np.int8)
    vsh = np.full((NL, TMAX), -128, dtype=np.int8)

    cur_tok = 1  # BOS (prompt_len=0)
    out = []
    for pos in range(n_tokens):
        cos_q15 = cos_rom[pos * HALF:pos * HALF + HALF]
        sin_q15 = sin_rom[pos * HALF:pos * HALF + HALF]
        # ---- EE : embedding ----
        XB = m['tok_emb'][cur_tok].astype(np.int8).copy()
        sxb = m['tok_emb_s']
        for l in range(NL):
            w = m['layers'][l]
            # ---- FN rmsnorm(XB, rms_att) -> XN ----
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_att'], w['rms_att_s'])
            # ---- FQ wq/wk/wv ----
            Q, sQ = fq_matvec(w['wq'], XN, sxn, w['wq_s'])      # [H*HS]
            Kc, sK = fq_matvec(w['wk'], XN, sxn, w['wk_s'])     # [KH*HS]
            Vc, sV = fq_matvec(w['wv'], XN, sxn, w['wv_s'])
            # ---- ROPE Q (H heads) + K (KH heads) ----
            Q = Q.reshape(H, HS); Kc = Kc.reshape(KH, HS)
            for h in range(H):
                Q[h], _ = rope_q15(Q[h], sQ, cos_q15, sin_q15)
            for kh in range(KH):
                Kc[kh], _ = rope_q15(Kc[kh], sK, cos_q15, sin_q15)
            Q = Q.reshape(D); Kc = Kc.reshape(KVW); Vc = Vc.reshape(KVW)
            # ---- KVWR ----
            kmem[l, pos] = Kc.reshape(KH, HS); vmem[l, pos] = Vc.reshape(KH, HS)
            ksh[l, pos] = sK; vsh[l, pos] = sV
            # ---- SCAN max shifts ----
            sKref = int(ksh[l, 0:pos + 1].max())
            sVref = int(vsh[l, 0:pos + 1].max())
            # ---- MM : H heads (GQA nrep=NREP) ----
            ATT = np.zeros(D, dtype=np.int8)
            for h in range(H):
                kh = h // NREP
                Qh = Q[h * HS:(h + 1) * HS]
                T = pos + 1
                # re-align K/V par position (clip8(mem >>> (ref-sh)))
                shK = sKref - ksh[l, 0:T].astype(np.int64)      # [T] >= 0
                shV = sVref - vsh[l, 0:T].astype(np.int64)
                K_h = clip8((kmem[l, 0:T, kh].astype(np.int64) >> shK[:, None]))
                V_h = clip8((vmem[l, 0:T, kh].astype(np.int64) >> shV[:, None]))
                ATT[h * HS:(h + 1) * HS] = attention_head_lut(Qh, K_h, V_h, T, score_sh)
            sA = sVref
            # ---- FQ wo(ATT) -> OUTV ----
            OUTV, sOut = fq_matvec(w['wo'], ATT, sA, w['wo_s'])
            # ---- residual ADD(XB, OUTV) -> XB ----
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sOut)
            # ---- FN2 rmsnorm(XB, rms_ffn) -> XN ----
            XN, sxn = rmsnorm_lut(XB, sxb, w['rms_ffn'], w['rms_ffn_s'])
            # ---- FFN 3 chunks -> OUTV ----
            OUTV, sov = ffn_chunked_lut(w, XN, sxn)
            # ---- residual ADD(XB, OUTV) -> XB ----
            XB, sxb = vec_alu2_add(XB, sxb, OUTV, sov)
        # ---- FNF rmsnorm(XB, rms_final) -> XN ----
        XN, sxn = rmsnorm_lut(XB, sxb, m['rms_final'], m['rms_final_s'])
        # ---- LM head 8 chunks ----
        logits, csh = lm_head_chunked_lut(m, XN, sxn)
        # ---- argmax ----
        best_idx, _ = argmax_lm_head(logits, csh)
        out.append(int(best_idx))
        cur_tok = int(best_idx)
    return out, kmem, vmem, ksh, vsh


# ---- Main : gate 1 offline ---------------------------------------------------
def main():
    from infer_v5gen_ref import load_model, MODEL, forward_ref
    print(f"Chargement modele : {MODEL}")
    m_float = load_model(MODEL)
    cfg = m_float['cfg']
    print(f"  dim={cfg['dim']} hidden={cfg['hidden_dim']} layers={cfg['n_layers']} "
          f"heads={cfg['n_heads']}/{cfg['n_kv_heads']} vocab={cfg['vocab_size']}")
    print("\nQuantification int8 du modele...")
    m = build_quantized_model(m_float)

    print("Forward nœud int8 bit-exact (17 tokens)...")
    node_tokens, kmem, vmem, ksh, vsh = forward_node(m, 17)
    print(f"  node_tokens = {node_tokens}")

    oracle = forward_ref(m_float, 17)
    print(f"  oracle float = {oracle}")

    expected_9 = [403, 407, 261, 378, 432, 383, 286, 261, 376]
    ok_9 = node_tokens[:9] == expected_9
    tok9 = node_tokens[9] if len(node_tokens) > 9 else None
    print(f"\n  9 premiers == oracle float {[403,407,261,378,432,383,286,261,376]} : {ok_9}")
    print(f"  token 9 (hardware) = {tok9}  (float oracle = 298, attendu nœud = 268)")
    if ok_9 and tok9 == 268:
        print("  ==> GATE 1 PASS : nœud bit-exact, 9 premiers == oracle, token 9 == 268.")
    else:
        print("  ==> GATE 1 FAIL : revoir les operateurs.")

    # Dump KV oracle (kmem/vmem/ksh/vsh) pour comparaison silicium étape 1
    # (CC-read du KV SDRAM via sd_addr après génération + RR).
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_oracle.npz")
    np.savez(out_path, kmem=kmem, vmem=vmem, ksh=ksh, vsh=vsh, node_tokens=np.array(node_tokens))
    print(f"  KV oracle dumpé : {out_path}  (kmem/vmem shape={kmem.shape})")


if __name__ == "__main__":
    main()