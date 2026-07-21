#!/usr/bin/env python3
"""test_wcache_mapping.py -- valide la cohérence du cache FFN BRAM (etape 3).

Reproduit exactement la logique Verilog de src/node_top.v :
  - decodeur is_ffn_addr(sd_addr)  (ff_layer_ok && ff_off_ok && wcache_fill_done)
  - mappeur cache_addr = ff_layer*36864 + (sd_addr[15:0]-0x3200)
  - remplissage FSM wcache_fill : cache[layer*36864+off] = SDRAM[0x013200+layer*0x10000+off]

Utilise une SDRAM factice (SDRAM[addr] = addr & 0xFF) pour valider le mapping
INDEPENDAMMENT des valeurs du modele reel. Verifie :
  1. is_ffn_addr pour toutes les plages FFN (5 couches x 36864) == True, et miss
     pour rms/wq/wk/wv/wo/rms_ffn/tok_emb/rms_final + gaps inter-couches.
  2. cache[cache_addr] == SDRAM[sd_addr] pour les 184320 octets FFN (coherence
     remplissage FSM <-> lecture decodeur).
  3. Frontieres (debut/fin de chaque matrice/couche, gap, rms_final, tok_emb).
"""
import random

SDRAM_SZ = 0x070000
FFN_BASE = 0x013200          # debut FFN couche 0
LBASE    = 0x010000          # base couche 0
STRIDE   = 0x10000           # 65536 entre couches
OFF_W1   = 0x3200
LAYER_SZ = 36864             # 0x9000 = 3 matrices x 3 chunks x 4096
NLAYERS  = 5
NFFN     = NLAYERS * LAYER_SZ   # 184320


def sdram_byte(addr):
    """SDRAM factice : addr & 0xFF."""
    return addr & 0xFF


def is_ffn_addr(sd_addr, fill_done=True):
    """Reproduit le decodeur Verilog (fill_done = wcache_fill_done_r)."""
    if not fill_done:
        return False
    layer_field = (sd_addr >> 16) & 0x1F          # sd_addr[20:16]
    off         = sd_addr & 0xFFFF                # sd_addr[15:0]
    ff_layer_ok = (layer_field >= 1) and (layer_field <= 5)
    ff_off_ok   = (off >= 0x3200) and (off < 0xC200)
    return ff_layer_ok and ff_off_ok


def cache_addr(sd_addr):
    """Reproduit le mappeur Verilog. Supose is_ffn_addr == True."""
    ff_layer = ((sd_addr >> 16) & 0x1F) - 1       # sd_addr[20:16] - 1
    off      = sd_addr & 0xFFFF
    return ff_layer * LAYER_SZ + (off - OFF_W1)


def fill_cache():
    """Reproduit la FSM wcache_fill : copie SDRAM -> cache (compact par couche)."""
    cache = bytearray(NFFN)
    for layer in range(NLAYERS):
        base_l = LBASE + layer * STRIDE           # 0x010000 + L*0x10000
        src = FFN_BASE + layer * STRIDE           # 0x013200 + L*0x10000
        for off in range(LAYER_SZ):
            cache[layer * LAYER_SZ + off] = sdram_byte(src + off)
    return cache


def main():
    random.seed(20260721)
    cache = fill_cache()
    errors = []

    # 1. Tous les octets FFN : is_ffn_addr True + coherence cache vs SDRAM
    n_hit = 0
    for layer in range(NLAYERS):
        src = FFN_BASE + layer * STRIDE
        for off in range(LAYER_SZ):
            sd = src + off
            if not is_ffn_addr(sd):
                errors.append(f"MISS attendu hit  sd=0x{sd:06x} (layer {layer} off {off})")
                continue
            ca = cache_addr(sd)
            if cache[ca] != sdram_byte(sd):
                errors.append(f"cache mismatch sd=0x{sd:06x} ca={ca} "
                              f"cache={cache[ca]} sdram={sdram_byte(sd)}")
            n_hit += 1
    print(f"[1] octets FFN balayes : {n_hit} (attendu {NFFN})")

    # 2. Miss : balayer l'espace hors FFN et verifier is_ffn_addr False
    #    + echantillonner densement les plages attn/rms_ffn/gap/rms_final/tok_emb
    miss_ranges = [
        ("tok_emb",      0x000000, 0x000000 + 512*64),         # [512,64]@0
        ("rms_att L0",   0x010000, 0x010100),
        ("wq L0",        0x010100, 0x011100),
        ("wk L0",        0x011100, 0x011900),
        ("wv L0",        0x011900, 0x012100),
        ("wo L0",        0x012100, 0x013100),
        ("rms_ffn L0",   0x013100, 0x013200),                  # finit juste avant W1
        ("gap L0->L1",   0x01C200, 0x023200),                  # fin W2 L0 .. debut W1 L1
        ("rms_final",    0x060000, 0x060040),
    ]
    n_miss_checked = 0
    for name, lo, hi in miss_ranges:
        for sd in range(lo, hi):
            if is_ffn_addr(sd):
                errors.append(f"HIT inattendu dans {name} sd=0x{sd:06x}")
            n_miss_checked += 1
    print(f"[2] octets hors-FFN verifies (plages critiques) : {n_miss_checked}")

    # 3. Echantillon aleatoire sur tout l'espace SDRAM
    n_rand = 20000
    for _ in range(n_rand):
        sd = random.randrange(0, SDRAM_SZ)
        hit = is_ffn_addr(sd)
        # un sd est hit ssi il tombe dans [FFN_BASE+L*STRIDE, +LAYER_SZ) pour un L
        expected_hit = any(
            (FFN_BASE + L * STRIDE) <= sd < (FFN_BASE + L * STRIDE + LAYER_SZ)
            for L in range(NLAYERS)
        )
        if hit != expected_hit:
            errors.append(f"rand mismatch sd=0x{sd:06x} is_ffn={hit} expected={expected_hit}")
        if hit:
            ca = cache_addr(sd)
            if cache[ca] != sdram_byte(sd):
                errors.append(f"rand cache mismatch sd=0x{sd:06x}")
    print(f"[3] echantillon aleatoire : {n_rand}")

    # 4. Frontieres specifiques
    borders = [
        (0x013200, True,  0),         # debut W1 L0 -> hit, cache_addr 0
        (0x016200, True,  12288),     # debut W3 L0 -> hit, cache_addr 12288
        (0x019200, True,  24576),     # debut W2 L0 -> hit, cache_addr 24576
        (0x01C1FF, True,  36863),     # dernier octet W2 L0 -> hit, cache_addr 36863
        (0x01C200, False, None),      # debut gap -> miss
        (0x023200, True,  36864),     # debut W1 L1 -> hit, cache_addr 36864
        (0x053200, True,  147456),    # debut W1 L4 -> hit, cache_addr 147456
        (0x05C1FF, True,  184319),    # dernier octet W2 L4 -> hit, cache_addr 184319
        (0x05C200, False, None),      # fin FFN L4 -> miss
        (0x010000, False, None),      # rms_att L0 -> miss
        (0x013100, False, None),      # rms_ffn L0 (off 0x3100 < 0x3200) -> miss
        (0x060000, False, None),      # rms_final (layer_field 6) -> miss
        (0x000000, False, None),      # tok_emb (layer_field 0) -> miss
    ]
    for sd, exp_hit, exp_ca in borders:
        hit = is_ffn_addr(sd)
        if hit != exp_hit:
            errors.append(f"border hit sd=0x{sd:06x} is_ffn={hit} expected={exp_hit}")
        if exp_hit:
            ca = cache_addr(sd)
            if ca != exp_ca:
                errors.append(f"border cache_addr sd=0x{sd:06x} ca={ca} expected={exp_ca}")
            if cache[ca] != sdram_byte(sd):
                errors.append(f"border cache mismatch sd=0x{sd:06x}")
    print(f"[4] frontieres : {len(borders)}")

    # 5. wcache_fill_done=0 -> aucun hit (cache inactif)
    for sd in [0x013200, 0x053200, 0x019200]:
        if is_ffn_addr(sd, fill_done=False):
            errors.append(f"hit alors que fill_done=0 sd=0x{sd:06x}")

    print()
    if errors:
        print(f"FAIL : {len(errors)} erreurs (10 premieres) :")
        for e in errors[:10]:
            print("  " + e)
        raise SystemExit(1)
    print(f"PASS : mapping cache FFN coherent ({NFFN} octets FFN, "
          f"{n_miss_checked} octets hors-FFN, {n_rand} rand, {len(borders)} frontieres)")


if __name__ == "__main__":
    main()