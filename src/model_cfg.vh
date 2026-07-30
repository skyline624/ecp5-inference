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
`ifdef DIM512
localparam CFG_D    = 512;    // dim (palier 3 rework dim, Phase 7)
// HID = 3*D (NCHUNK_F=3 FSM GG hardcodé), HS=16, KVW=32 fixe (évite rework KV :
// attn_kv_stride/BSRAM K/V dimensionnés pour KVW=32). D=H*HS, KH=KVW/HS=2, H=D/HS,
// NREP=H/KH. D=512 -> H=32 KH=2 NREP=16 (mh_h largeur $clog2(CFG_H):0 côté node_top).
localparam CFG_HID  = 3*CFG_D;
localparam CFG_NC   = 0;      // stream SDRAM (FFN streamé, BRAM libérée)
localparam CFG_HS   = 16;
localparam CFG_H    = 32;     // n_heads (D=H*HS=32*16=512)
localparam CFG_KH   = 2;      // n_kv_heads (KVW=KH*HS=2*16=32 inchangé)
localparam CFG_SCORE_SH = 1;  // pre-shift score (HS=16)
`elsif DIM256
localparam CFG_D    = 256;    // dim (palier 2 rework dim, Phase 7)
localparam CFG_HID  = 3*CFG_D; // D=256 -> H=16 KH=2 NREP=8
localparam CFG_NC   = 0;
localparam CFG_HS   = 16;
localparam CFG_H    = 16;     // n_heads (D=H*HS=16*16=256)
localparam CFG_KH   = 2;
localparam CFG_SCORE_SH = 1;
`elsif DIM128
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
// vocab paramétrable (Phase 6) : CFG_VOCAB surchargeable via `VOCAB32000 (cible
// tinyllamas). Défaut 512 = stories260K (non-régression). La table d'embedding
// SDRAM (VOCAB*D) et le scan argmax (VOCAB cycles/token) passent à 32000 -> SDRAM
// ~4 Mo à dim=128 (déborde la i9, faut carte custom/cluster) + argmax ~62x plus
// lent. RTL: token TBITS=16 bits, NCHUNK=ceil(VOCAB/D) (500/250/125 à dim 64/128/256).
`ifdef VOCAB32000
localparam CFG_VOCAB = 32000;
`else
localparam CFG_VOCAB = 512;
`endif
localparam CFG_NN   = 1;      // tensor-parallel degree (première livraison : single-node)

// --- dérivés HS/head-shape (retire les magic numbers 8/4/32/64 du datapath attention) ---
localparam CFG_KVW    = CFG_KH*CFG_HS;                       // 32 (KV width par couche)
localparam CFG_NREP   = CFG_H/CFG_KH;                        // 2 (GQA repeat)
localparam CFG_HALF   = CFG_HS/2;                            // 4 (paires RoPE, entries cos/sin_reg)
localparam CFG_LOG2_HS   = $clog2(CFG_HS);                   // 3 (slice i_idx/d_idx, shift << HS)
localparam CFG_LOG2_HALF = $clog2(CFG_HALF);                 // 2 (index cos/sin_reg)
localparam CFG_LOG2_NREP = $clog2(CFG_NREP);                 // 1 (mh_h >> log2(NREP) pour kv_offset)
localparam CFG_LOG2_H    = $clog2(CFG_H);                    // 3..5 (mh_h largeur [$clog2(H):0], Phase 7 H=16/32)
localparam CFG_HBITS   = $clog2(CFG_H);                      // 3 (largeur mh_h ; 4 pour H<=16)
localparam CFG_DBITS   = $clog2(CFG_D);                      // 6 (index Q_packed D-byte)
localparam CFG_KVWBITS = $clog2(CFG_KVW);                    // 5 (index gg_k/v_packed)
localparam CFG_NPOS    = 17;                                 // generation length (token loop)
localparam CFG_TMAX    = 32;                                 // max sequence length (attention T)
localparam CFG_NCHUNK  = (CFG_VOCAB + CFG_D - 1)/CFG_D;      // ceil(VOCAB/D) lm_head chunks
   // (Phase 6 vocab-paramétrable : ceil au lieu de floor -> évite la troncature
   //  silencieuse de logits quand VOCAB%D != 0, ex dim=512/vocab=32000 = 62.5.
   //  Integer à tous les paliers dim 64/128/256 avec vocab=32000 : 500/250/125.
   //  lmchunk/scan_c largeur $clog2(CFG_NCHUNK):0 côté gen_seq.)
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
//
// ⚠ CFG_LBASE était figé à 'h010000 alors que tok_emb occupe VOCAB*D octets DEPUIS 0 :
//     dim64  : 512*64  = 0x08000  <  0x10000   ok
//     dim128 : 512*128 = 0x10000  == 0x10000   ok (finit pile où LBASE commence)
//     dim256 : 512*256 = 0x20000  >  0x10000   COLLISION : les poids de la couche 0
//              écrasaient la 2e moitié de tok_emb, donc exactement le chunk 1 du
//              lm_head (logits[256..511] calculés sur rms_att/WQ au lieu de tok_emb).
//     dim512 : 512*512 = 0x40000  >  0x10000   idem, pire.
// Symptôme observé à dim256 : sim ET silicium retournaient le max du chunk 0 (l'oracle
// Python, lui, n'a pas de layout SDRAM -> il était correct). Le RTL lisait fidèlement
// une SDRAM mal remplie ; ni le RTL ni l'oracle n'étaient en cause, mais ce plan mémoire.
// LBASE dérive donc maintenant de la taille réelle de l'embedding, arrondie à la
// puissance de 2 supérieure, avec un plancher à 0x010000 pour ne RIEN changer à dim64
// et dim128 (tous deux validés bit-exact sur silicium).
localparam CFG_EMB_SZ  = CFG_VOCAB*CFG_D;                     // octets de tok_emb @ A_EMB=0
localparam CFG_LBASE   = (CFG_EMB_SZ <= 'h010000) ? 'h010000
                                                  : (1 << $clog2(CFG_EMB_SZ));
// LSTRIDE = next pow2 >= taille couche (OFF_W1 + 3*MAT_SZ) -> 0x10000 dim 64, 0x40000 dim 128
localparam CFG_LSTRIDE = 1 << $clog2(CFG_OFF_W1 + 3*CFG_MAT_SZ);
localparam CFG_A_EMB      = 'h000000;                          // tok_emb @ 0 (VOCAB*D octets)
localparam CFG_A_RMSFINAL = CFG_LBASE + CFG_NL*CFG_LSTRIDE;    // après les NL couches (0x60000 dim 64)
localparam CFG_KV_BASE    = CFG_A_RMSFINAL + 'h10000;          // KV cache après rmsfinal (0x70000 dim 64)
// adresse FFN couche 0 dans SDRAM (début du fill DMA boot)
localparam CFG_WCACHE_FFN_BASE = CFG_LBASE + CFG_OFF_W1;

// ---- image flash (boot autonome) : dérivée du profil, plus hardcodée ----
// flash_boot.v DMA [0 .. FLASH_LEN) de la flash vers la SDRAM au power-up, puis lit un
// trailer de 47 shifts à SHIFT_BASE. Ces deux valeurs étaient figées à 24'h060040 et
// 24'h261000 dans board/gen_seq_silicon.v — la géométrie dim64 — ce qui rendait le boot
// autonome INUTILISABLE au-delà (dim128/256 devaient charger la SDRAM par UART, ~60 s).
// Le mirror va de 0 à la fin de rms_final ; le trailer suit, aligné sur 4 Ko.
// Vérifié : à dim64 ces formules redonnent EXACTEMENT 0x060040 / 0x061000 (profil validé
// silicium, donc bit-identique). Le KV (CFG_KV_BASE) n'est PAS dans l'image : il est
// calculé à l'exécution, jamais chargé.
// FLASH_BASE : le bitstream occupe 0..~0x140000 (cache FFN 90 BRAM), d'où 0x200000.
localparam CFG_FLASH_BASE = 'h200000;
localparam CFG_FLASH_LEN = CFG_A_RMSFINAL + CFG_D;                       // 0x060040 dim64
localparam CFG_SHIFT_OFF = ((CFG_FLASH_LEN + 'hFFF)/'h1000)*'h1000;      // 0x061000 dim64
localparam CFG_SHIFT_BASE = CFG_FLASH_BASE + CFG_SHIFT_OFF;              // 0x261000 dim64
// Taille totale occupée en flash depuis FLASH_BASE (mirror + trailer 47 octets).
// À FLASH_BASE=0x200000 sur un W25Q64 (8 Mo) : dim64/128/256 tiennent, dim512 non
// (0x1441000 + 0x200000 = 23 Mo) -> dim512 restera en chargement UART.
localparam CFG_FLASH_IMG_SZ = CFG_SHIFT_OFF + 47;