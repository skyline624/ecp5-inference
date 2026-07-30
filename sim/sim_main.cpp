// sim_main.cpp -- TB C++ pour Verilator UNTIMED (top = gen_seq_top, le DUT direct).
// Mode rapide : le C++ drive clk (toggle + eval), poll sim_fill_done, pulse start,
// capture les fronts token_valid, compare aux tokens attendus (lus en C++ depuis
// tb_expected.hex). Aucun --timing (coroutines) -> pas d'overhead scheduler -> 50x.
//
// Hex (gen_tb_hex.py) lus en C++ : tb_meta.hex (NTOK), tb_shifts.hex (11 ports),
// tb_expected.hex (tokens). Lancer depuis sim/ (chemins $readmemh + hex relatifs).
#include "Vgen_seq_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

double sc_time_stamp() { return 0; }

// nombre de mots uint32 du port result (W*D/32). 16 dim=64, 32 dim=128.
// Passé via /DXB_WORDS=32 par le wrapper (build_verilator_fb.bat %CXXDEFS%) en dim128.
#ifndef XB_WORDS
#define XB_WORDS 16
#endif

static Vgen_seq_top* top;
static void tick() {            // 1 cycle : front descendant puis front montant
    top->clk = 0; top->eval();
    top->clk = 1; top->eval();
}

// lit des mots hex (hex, sep par whitespace) dans out[], renvoie le nombre lus
static int read_hex(const char* fn, uint64_t* out, int maxn) {
    FILE* f = fopen(fn, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", fn); exit(1); }
    int n = 0; char buf[80];
    while (n < maxn && fscanf(f, "%79s", buf) == 1)
        out[n++] = strtoull(buf, nullptr, 16);
    fclose(f);
    return n;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vgen_seq_top;

    uint64_t meta[1];     read_hex("tb_meta.hex", meta, 1);
    int NTOK = (int)meta[0];
    uint64_t shifts[11];   read_hex("tb_shifts.hex", shifts, 11);
    uint64_t expect[256];  int ne = read_hex("tb_expected.hex", expect, 256);
    if (ne < NTOK) { fprintf(stderr, "expected: %d mots, besoin NTOK=%d\n", ne, NTOK); exit(1); }

    // ---- reset + drives (x_in/prompt laisses a 0 : path BOS prompt_len=0) ----
    top->clk = 0; top->rst_n = 0; top->start = 0; top->gen_mode = 0;
    top->sx_in = 0; top->pos = 0; top->prompt_len = 0;
    top->sw_rms = shifts[0]; top->swq = shifts[1]; top->swk = shifts[2];
    top->swv   = shifts[3]; top->swo = shifts[4]; top->sw_rmsf = shifts[5];
    top->sw1   = shifts[6]; top->sw3  = shifts[7]; top->sw2    = shifts[8];
    top->sw_rmsfinal = (uint8_t)(shifts[9] & 0xFF);
    top->sw_emb      = (uint8_t)(shifts[10] & 0xFF);
    top->eval();

    for (int i = 0; i < 20; i++) tick();     // reset
    top->rst_n = 1;

    // ---- attend le remplissage cache FFN BRAM (sonde sim_fill_done) ----
    uint64_t guard = 0;
    while (!top->sim_fill_done) {
        tick();
        if (++guard > 5'000'000) { printf("FAIL: fill timeout\n"); top->final(); return 1; }
    }
    tick();
    top->gen_mode = 1; top->start = 1; tick(); top->start = 0;

    // ---- capture sur front montant token_valid ----
    int got = 0, mismatch = 0;
    uint8_t tv_d = 1;   // pas de faux front au demarrage
    uint64_t wd = 0;
    // Cap cycles anti-hang. Defaut 40M (calibré NC=5 cache BRAM). NC=0 stream SDRAM
    // est ~3x plus lent (FFN streamé a chaque token) -> 17 tokens dépassent 40M. Le
    // wrapper run_verilator_fb.ps1 passe SIM_CYCLE_CAP plus élevé via -StreamFFN.
    uint64_t cap = 40'000'000;
    if (const char* e = std::getenv("SIM_CYCLE_CAP")) cap = (uint64_t)strtoull(e, nullptr, 0);
    while (got < NTOK) {
        tick(); wd++;
        uint8_t tv = top->token_valid & 1;
        if (tv && !tv_d) {
            int tok = (int)(top->token & 0xFFFF);      // TBITS=16 (vocab-param Phase 6)
            int bv = (int)(top->dbg_best_v & 0xFFFFFFFF);
            int bi = (int)(top->dbg_best_i & 0xFFFF);   // couvre VOCAB<=65536 (32000)
            int sh = (int)(top->result_sh & 0xFF);
            // XB (result) : XB_WORDS mots uint32 little-endian par octet (match oracle diag).
            // Le separateur etait conditionne par `w<31`, hardcode pour 32 mots : au-dela
            // (XB_WORDS=64 en dim256, 128 en dim512) les mots 32+ sortaient CONCATENES sans
            // espace, donc impossibles a reparser -> tout diagnostic XB au-dela de dim128
            // comparait en fait la mauvaise moitie du vecteur.
            char hexbuf[XB_WORDS*9+1]; hexbuf[0]=0;
            for (int w=0; w<XB_WORDS; w++) {
                uint32_t wd = top->result[w];
                char tmp[9]; snprintf(tmp,9,"%08x",(unsigned)wd); strcat(hexbuf,tmp);
                if (w+1 < XB_WORDS) strcat(hexbuf," ");
            }
            // Sondes lm_head : csh[0]/csh[1] (shifts de requantification par chunk) et
            // sref_lm=max(csh). L'argmax realigne par (sref_lm - csh[chunk]) : un csh
            // errone ecrase tout un chunk et l'empeche de gagner, sans que XB bouge.
            int c0 = (int)(int8_t)(top->dbg_csh0 & 0xFF);
            int c1 = (int)(int8_t)(top->dbg_csh1 & 0xFF);
            int sr = (int)(int8_t)(top->dbg_sref_lm & 0xFF);
            int bv1 = (int)top->dbg_bv1; int bi1 = (int)(top->dbg_bi1 & 0xFFFF);
            // Adresse SDRAM du DERNIER matmul FQ = dernier chunk du lm_head. A dim256 le
            // chunk 1 doit valoir 0x010000 (A_EMB + 1*D*D). Toute autre valeur explique
            // directement des logits "valides mais faux" sur ce chunk.
            unsigned fma = (unsigned)(top->dbg_fm_addr & 0x7FFFFF);
            unsigned wr0 = (unsigned)top->dbg_wrow0;   // 4 premiers poids de la ligne 0
            unsigned rda = (unsigned)(top->dbg_rdaddr & 0x7FFFFF);  // sd_addr effectif
            if (tok != (int)expect[got]) {
                printf("MISMATCH tok[%d]=%d expect=%d  (rtl best_i=%d best_v=%d) sh=%d csh=[%d,%d] sref=%d chunk1_max=%d@%d fm_addr=0x%06x rdaddr=0x%06x wrow0=%08x XB=%s\n",
                       got, tok, (int)expect[got], bi, bv, sh, c0, c1, sr, bv1, bi1, fma, rda, wr0, hexbuf);
                mismatch++;
            } else {
                printf("tok[%d]=%d OK  (best_i=%d best_v=%d) sh=%d csh=[%d,%d] sref=%d chunk1_max=%d@%d fm_addr=0x%06x rdaddr=0x%06x wrow0=%08x XB=%s\n",
                       got, tok, bi, bv, sh, c0, c1, sr, bv1, bi1, fma, rda, wr0, hexbuf);
            }
            got++;
        }
        tv_d = tv;
        // debug hang : print le FSM state a CHAQUE changement (montre la séquence avant
        // figeage). Le dernier print avant silence = state où le seq hang.
        uint8_t stc = top->dbg_st & 0x3F;
        static uint8_t st_d = 0xFF;
        uint16_t nst = top->dbg_node_state & 0x1FF;
        if (stc != st_d) {
            printf("wd=%llu st=%u ph=%u idx=%llu nst=%u (was %u)\n", (unsigned long long)wd,
                   (unsigned)stc, (unsigned)(top->dbg_phase & 0xF),
                   (unsigned long long)(top->dbg_idx & 0xFFF), (unsigned)nst, (unsigned)st_d);
            st_d = stc;
        }
        if (wd > cap) { printf("FAIL: gen timeout got=%d/%d (cap=%llu) st=%u ph=%u idx=%llu nst=%u\n",
                       got, NTOK, (unsigned long long)cap, (unsigned)(top->dbg_st & 0x3F),
                       (unsigned)(top->dbg_phase & 0xF), (unsigned long long)(top->dbg_idx & 0xFFF),
                       (unsigned)(top->dbg_node_state & 0x1FF));
                       top->final(); return 1; }
    }
    if (mismatch == 0)
        printf("PASS: %d/%d bit-exact vs node_tokens (Verilator untimed C++ TB)\n", got, NTOK);
    else
        printf("FAIL: %d mismatch sur %d tokens\n", mismatch, NTOK);
    top->final();
    delete top;
    return mismatch ? 1 : 0;
}