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
    while (got < NTOK) {
        tick(); wd++;
        uint8_t tv = top->token_valid & 1;
        if (tv && !tv_d) {
            int tok = (int)(top->token & 0x3FF);
            if (tok != (int)expect[got]) {
                printf("MISMATCH tok[%d]=%d expect=%d\n", got, tok, (int)expect[got]);
                mismatch++;
            } else {
                printf("tok[%d]=%d OK\n", got, tok);
            }
            got++;
        }
        tv_d = tv;
        if (wd > 40'000'000) { printf("FAIL: gen timeout got=%d/%d\n", got, NTOK); top->final(); return 1; }
    }
    if (mismatch == 0)
        printf("PASS: %d/%d bit-exact vs node_tokens (Verilator untimed C++ TB)\n", got, NTOK);
    else
        printf("FAIL: %d mismatch sur %d tokens\n", mismatch, NTOK);
    top->final();
    delete top;
    return mismatch ? 1 : 0;
}