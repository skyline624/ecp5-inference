// =============================================================================
// model_cfg.vh -- Source de vérité unique de la shape modèle + cache manager.
//
// Inclu via `include "model_cfg.vh" dans les modules qui en ont besoin
// (node_top.v, wcache_ffn.v, ...). Noms préfixés CFG_ pour éviter les collisions
// avec les parameter/localparam D/NL/... déjà présents dans les modules.
//
// AVANT synthèse : éditer ce fichier (ou le surcharger via un model_cfg_<profile>.vh)
// pour changer la shape (D/HID/NL) ou la granularité du cache (CFG_NC).
//
//   CFG_NC (NC_LAYERS_CACHED) = combien des premières couches FFN sont cachées
//   en BRAM au boot (STATIC : rempli une fois, réutilisé pour tous les tokens) :
//     0  = stream pur (pas de BRAM FFN utile, fallback SDRAM pour tout)
//     k  = k couches en BRAM, (NL-k) streamées depuis SDRAM à chaque token
//     NL = cache-all (comportement historique stories260K, 90 blocs DP16KD)
//
//   Rationale STATIC (pas de cache roulant) : re-fill 1 couche FFN depuis SDRAM
//   prend ~3*HID*D octets à 1 octet/cycle, tandis que la calculer prend ~2*HID*D
//   macs à 1 mac/cycle -> le re-fill coûte ~3x le calcul. Un cache roulant est donc
//   systématiquement perdant pour le FFN. Le cache BRAM n'est rentable que statique
//   (rempli une fois au boot, réutilisé sur tous les tokens).
// =============================================================================

// dim paramétrable (rework dim) : CFG_D/HID/NC surchargeables via DIM128 (+ à venir
// DIM256/DIM512 après extension des largeurs HS>16, cf rework dim Phase 2/5). Défaut =
// stories260K dim=64. À dim>=128 : NC=0 stream SDRAM (libère 90 blocs BRAM, le FFN est
// streamé -> "plus de boucles" au lieu de cacher en BRAM ; cf rework dim Phase 5).
// HS=16 forcé à dim=128 (H=8 KH=2 KVW=32 inchangés vs stories260K, déjà prouvé bit-exact
// rework HS-param Phase 4 ; NREP=H/KH=4 nouveau mais paramétrique via CFG_LOG2_NREP).
// head_size paramétrable (rework HS) : à dim=64, HS/H/KH/SCORE_SH surchargeables via
// `HS16_PROFILE pour la preuve HS=16 (D=H*HS=64, KVW=KH*HS=32 inchangés -> isole
// le rework HS). Défaut = stories260K HS=8. CFG_SCORE_SH = pre-shift du score
// d'attention avant le clip int16 (calibrage ALU dépendant de HS, pas un weight
// shift) : 0 pour HS=8 (non-régression bit-exact par construction), 1 pour HS=16.
`ifdef DIM128
localparam CFG_D    = 128;    // dim (palier 1 rework dim)
// HID = 3*D (NCHUNK_F=3) : la FSM GG FFN hardcode 3 chunks (états
// S_GG_SETUP_W1/W3_CH0/CH1/CH2 + offsets k*D). Rendre NCHUNK_F>3 paramétrique = rework
// FSM (extension différée). Pour la preuve bit-exact par palier, 3*D suffit (poids
// synthétiques, HID exact sans importance tant que RTL=oracle). dim=64 reste HID=172
// (HIDPAD=192=3*64, NCHUNK_F=3 aussi -> cohérent).
localparam CFG_HID  = 3*CFG_D; // hidden_dim (384 à dim 128)
localparam CFG_NC   = 0;      // stream SDRAM : FFN streamé, BRAM libérée (90 blocs)
localparam CFG_HS   = 16;     // head_size (H=8 KH=2 KVW=32 inchangés, NREP=4)
localparam CFG_H    = 8;      // n_heads (D=H*HS=8*16=128)
localparam CFG_KH   = 2;      // n_kv_heads (KVW=KH*HS=2*16=32 inchangé)
localparam CFG_SCORE_SH = 1;  // pre-shift score (HS=16)
`else
localparam CFG_D    = 64;     // dim
localparam CFG_HID  = 172;    // hidden_dim (champ header modèle llama2.c)
`ifdef NC0_PROFILE
// Phase 5 rework dim : force stream SDRAM à dim=64 pour valider le path FFN stream
// bit-exact (is_ffn_addr=0 -> S_FM_RD lit sd_dout au lieu de wcache_dout). Indépendant
// du dump dim128 (utilise stories260K). NC=0 = comportement DIM128 (FFN streamé, BRAM libérée).
localparam CFG_NC   = 0;      // stream pur (test Phase 5 : path FFN stream bit-exact)
`else
localparam CFG_NC   = 5;      // cache-all (perf 14 tok/s stories260K)
`endif
`ifdef HS16_PROFILE
localparam CFG_HS   = 16;     // head_size (preuve rework HS, H*HS=64 inchangé)
localparam CFG_H    = 4;      // n_heads  (D=H*HS=4*16=64)
localparam CFG_KH   = 2;      // n_kv_heads (KVW=KH*HS=2*16=32 inchangé, NREP=2)
localparam CFG_SCORE_SH = 1;  // pre-shift score (HS=16: raw dot 258k >>>1 = 129k, clip comme HS=8)
`else
localparam CFG_HS   = 8;      // head_size
localparam CFG_H    = 8;      // n_heads
localparam CFG_KH   = 4;      // n_kv_heads (GQA, n_rep = H/KH = 2)
localparam CFG_SCORE_SH = 0;  // pre-shift score (0 = comportement historique HS=8)
`endif
`endif
localparam CFG_NL   = 5;      // n_layers
localparam CFG_VOCAB= 512;
localparam CFG_NN   = 1;      // tensor-parallel degree (première livraison : single-node)

// --- dérivés HS/head-shape (retire les magic numbers 8/4/32/64 du datapath attention) ---
localparam CFG_KVW    = CFG_KH*CFG_HS;                       // 32 (KV width par couche)
localparam CFG_NREP   = CFG_H/CFG_KH;                        // 2 (GQA repeat)
localparam CFG_HALF   = CFG_HS/2;                            // 4 (paires RoPE, entries cos/sin_reg)
localparam CFG_LOG2_HS   = $clog2(CFG_HS);                   // 3 (slice i_idx/d_idx, shift << HS)
localparam CFG_LOG2_HALF = $clog2(CFG_HALF);                 // 2 (index cos/sin_reg)
localparam CFG_LOG2_NREP = $clog2(CFG_NREP);                 // 1 (mh_h >> log2(NREP) pour kv_offset)
localparam CFG_HBITS   = $clog2(CFG_H);                      // 3 (largeur mh_h ; 4 pour H<=16)
localparam CFG_DBITS   = $clog2(CFG_D);                      // 6 (index Q_packed D-byte)
localparam CFG_KVWBITS = $clog2(CFG_KVW);                    // 5 (index gg_k/v_packed)
localparam CFG_NPOS    = 17;                                 // generation length (token loop)
localparam CFG_TMAX    = 32;                                 // max sequence length (attention T)
localparam CFG_NCHUNK  = CFG_VOCAB/CFG_D;                    // 8 (lm_head chunks)
localparam CFG_NPR     = 16;                                 // max prompt tokens (prefill)
// FFN layout : HID padded au multiple de D supérieur, chunké en blocs [D,D]
// (cohérent avec gen_seq.v "HID chunked 3x64" : 172 -> 192 = 3*64).
localparam CFG_HIDPAD  = ((CFG_HID+CFG_D-1)/CFG_D)*CFG_D;     // 172 -> 192
localparam CFG_NCHUNK_F= CFG_HIDPAD/CFG_D;                   // 3
localparam CFG_MAT_SZ  = CFG_NCHUNK_F*CFG_D*CFG_D;           // 12288 (1 matrice [HIDPAD,D])
localparam CFG_WCACHE_LAYER_SZ = 3*CFG_MAT_SZ;                // 36864 (W1+W3+W2 par couche)
// BRAM utile : NC couches. Garde NC=0 -> NFFN=1 (évite un tableau BRAM vide
// `reg [7:0] mem [0:-1]` illégal ; yosys élimine la BRAM 1-octet, le fill FSM reste
// idle car wcache_fill_done_r forcé à 1 au reset).
localparam CFG_NFFN   = (CFG_NC == 0) ? 1 : CFG_NC*CFG_WCACHE_LAYER_SZ;
localparam CFG_AW     = 20;     // largeur ports addr wcache (jusqu'à 1 Mo, couvre dim 256 NC=1)
// Offsets intra-couche (layout cumulatif PARAMÉTRIQUE, dérivé de CFG_D/CFG_KVW). RTL
// (gen_seq.v) et host (gen_sdram_preload_hex.py) DOIVENT utiliser ces mêmes formules
// (sinon mismatch SDRAM). Cumulatif sans padding : rms_att=D, WQ=D*D, WK/WV=KVW*D,
// WO=D*D, rms_ffn=D, W1/W3/W2=NCHUNK_F*D*D (=MAT_SZ). À dim 64 : LSTRIDE=0x10000 et
// A_RMSFINAL=0x60000 / KV_BASE=0x70000 inchangés vs layout paddé historique (seuls les
// offsets intra-couche passent de paddés à cumulatifs, ex WQ 0x100->0x40) -> non-régression
// bit-exact car RTL+host changent ensemble (mêmes poids, autres adresses).
localparam CFG_OFF_RMS   = 0;
localparam CFG_OFF_WQ    = CFG_D;                             // rms_att = D bytes
localparam CFG_OFF_WK    = CFG_OFF_WQ  + CFG_D*CFG_D;          // WQ [H*HS,D] = D*D
localparam CFG_OFF_WV    = CFG_OFF_WK  + CFG_D*CFG_KVW;       // WK [KVW,D] = KVW*D
localparam CFG_OFF_WO    = CFG_OFF_WV  + CFG_D*CFG_KVW;       // WV [KVW,D] = KVW*D
localparam CFG_OFF_RMSFF = CFG_OFF_WO  + CFG_D*CFG_D;         // WO [D,H*HS] = D*D
localparam CFG_OFF_W1    = CFG_OFF_RMSFF + CFG_D;             // rms_ffn = D ; W1 [HIDPAD,D]
localparam CFG_OFF_W3    = CFG_OFF_W1 + CFG_MAT_SZ;
localparam CFG_OFF_W2    = CFG_OFF_W3 + CFG_MAT_SZ;
localparam CFG_FFN_LO  = CFG_OFF_W1;
localparam CFG_FFN_HI  = CFG_OFF_W1 + CFG_WCACHE_LAYER_SZ;
// layout SDRAM global (paramétrique) : embedding@0, LBASE au-dessus, RMSFINAL+KV après
localparam CFG_LBASE   = 'h010000;
// LSTRIDE = next pow2 >= taille couche (OFF_W1 + 3*MAT_SZ) -> 0x10000 dim 64, 0x40000 dim 128
localparam CFG_LSTRIDE = 1 << $clog2(CFG_OFF_W1 + 3*CFG_MAT_SZ);
localparam CFG_A_EMB      = 'h000000;                          // tok_emb @ 0 (VOCAB*D octets)
localparam CFG_A_RMSFINAL = CFG_LBASE + CFG_NL*CFG_LSTRIDE;    // après les NL couches (0x60000 dim 64)
localparam CFG_KV_BASE    = CFG_A_RMSFINAL + 'h10000;          // KV cache après rmsfinal (0x70000 dim 64)
// adresse FFN couche 0 dans SDRAM (début du fill DMA boot)
localparam CFG_WCACHE_FFN_BASE = CFG_LBASE + CFG_OFF_W1;