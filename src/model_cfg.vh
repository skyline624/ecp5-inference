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

localparam CFG_D    = 64;     // dim
localparam CFG_HID  = 172;    // hidden_dim (champ header modèle llama2.c)
localparam CFG_NL   = 5;      // n_layers
localparam CFG_HS   = 8;      // head_size
localparam CFG_H    = 8;      // n_heads
localparam CFG_KH   = 4;      // n_kv_heads (GQA, n_rep = H/KH = 2)
localparam CFG_VOCAB= 512;
localparam CFG_NN   = 1;      // tensor-parallel degree (première livraison : single-node)
localparam CFG_NC   = 5;      // NC_LAYERS_CACHED : 0=stream .. NL=cache-all (STATIC)

// --- dérivés (retire les magic numbers 12288/36864/184320/0xC200/0x3200/...) ---
localparam CFG_KVW    = CFG_KH*CFG_HS;                       // 32 (KV width par couche)
localparam CFG_NREP   = CFG_H/CFG_KH;                        // 2 (GQA repeat)
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
// offsets intra-couche : W1 base fixe (dépend du layout rms/wq/.. avant, TODO dériver pour
// dim 128) ; W3/W2 dérivés (1 matrice [HIDPAD,D] chunkée = MAT_SZ octets d'écart).
localparam CFG_OFF_W1  = 'h3200;
localparam CFG_OFF_W3  = CFG_OFF_W1 + CFG_MAT_SZ;             // 0x6200 dim 64
localparam CFG_OFF_W2  = CFG_OFF_W3 + CFG_MAT_SZ;             // 0x9200 dim 64
localparam CFG_FFN_LO  = CFG_OFF_W1;                          // 0x3200
localparam CFG_FFN_HI  = CFG_OFF_W1 + CFG_WCACHE_LAYER_SZ;   // 0xC200 dim 64
// layout SDRAM
localparam CFG_LBASE   = 'h010000;
// LSTRIDE = next pow2 >= taille couche (OFF_W1 + 3*MAT_SZ) -> 0x10000 dim 64, 0x40000 dim 128
localparam CFG_LSTRIDE = 1 << $clog2(CFG_OFF_W1 + 3*CFG_MAT_SZ);
// adresse FFN couche 0 dans SDRAM (début du fill DMA boot)
localparam CFG_WCACHE_FFN_BASE = CFG_LBASE + CFG_OFF_W1;      // 0x013200