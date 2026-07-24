`timescale 1ns/1ps
// gen_seq_top - one node (top.v LINK_SS) + the generation sequencer over ss_link.
module gen_seq_top #(
    // D/NL/NPOS : largeurs des ports x_in[W*D-1:0] / result[W*D-1:0] / sw_*[NL*8-1:0].
    // L'instance gen_seq interne utilise CFG_D/CFG_NL/CFG_NPOS (localparams model_cfg.vh,
    // inclus dans le body). Les PORTS, eux, sont parsés AVANT model_cfg.vh -> ils ne
    // voient pas CFG_D. Il faut donc que le défaut du paramètre D suive CFG_D selon le
    // profile build-time. Sinon (bug historique dim128) result ne fait que 64 octets
    // alors que gen_seq produit 128 octets -> troncature -> le probe lit 32 mots uint32
    // mais les 16 hauts (octets 64..127) sont hors-borne = garbage (artefact de probe,
    // pas un bug de compute). DIM128/HS16_PROFILE = flags -D passés au build.
    parameter W = 8,
`ifdef DIM512
    parameter D = 512,
`elsif DIM256
    parameter D = 256,
`elsif DIM128
    parameter D = 128,
`else
    parameter D = 64,
`endif
    parameter HS = 8, parameter NL = 5, parameter NPOS = 17
) (
    input  wire           clk, rst_n, start, gen_mode,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx_in,
    input  wire [NL*8-1:0] sw_rms, swq, swk, swv, swo,
    input  wire [NL*8-1:0] sw_rmsf, sw1, sw3, sw2,
    input  wire signed [7:0] sw_rmsfinal, sw_emb,
    input  wire [5:0]     pos,
    // prefill : prompt token IDs (packed, NPR*TBITS = 16*16 bits) + count. prompt_len=0
    // -> BOS hardcodé côté gen_seq. Exposé ici pour piloter le token initial en gen_mode
    // (test_gen_A et tests de prefill) ; génère à partir du dernier token du prompt.
    // TBITS=16 (vocab-paramétrable Phase 6, couvre VOCAB <= 65536) côté gen_seq.
    input  wire [16*16-1:0] prompt,
    input  wire [6:0]        prompt_len,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire [15:0]    token,
    output wire           token_valid,
    output wire           done,
`ifdef SIM
    output wire [5:0]    dbg_st,           // debug SIM-only : gen_seq FSM state (localise hang)
    output wire [3:0]    dbg_phase,        // debug SIM-only : gen_seq phase (PH_*)
    output wire [11:0]   dbg_idx,          // debug SIM-only : gen_seq idx (octet paquet / rcnt)
    output wire [8:0]    dbg_node_state,   // debug SIM-only : node FSM state (localise hang)
    output wire [31:0]   dbg_best_v,       // debug SIM-only : argmax best_val final (v du token émis)
    output wire [15:0]   dbg_best_i,       // debug SIM-only : argmax best_idx final (= token émis)
`endif
    // sonde sim-only : cache FFN BRAM rempli (path silicium FLASH_BOOT). Permet au
    // TB Verilog pur / au TB C++ Verilator d'attendre le fill SANS ref hierarchique
    // (dut.u_n.wcache_fill_done). Sempre 1 hors FLASH_BOOT (pas de cache fill).
    output wire           sim_fill_done
);
    `include "model_cfg.vh"
    // gen_seq shape : HID/LSTRIDE/OFF_*/LBASE/H/KH/HS passés via CFG_* (rework dim Phase 1).
    // D/NL/NPOS restent des parameter propagés à gen_seq (.D(D)) ; le PARENT (TB ou
    // gen_seq_silicon) doit surcharger .D(CFG_D)/.NL(CFG_NL)/.NPOS(CFG_NPOS) pour que les
    // ports [W*D-1:0]/[NL*8-1:0] matchent CFG_D/CFG_NL. dim paramétrable via `DIM128.
    wire [7:0] s_cmd_data;  wire s_cmd_wr;  wire s_cmd_full;
    wire [7:0] s_resp_data; wire s_resp_empty; wire s_resp_rd;

    // ---- KV cache SDRAM (étape 1) : 2e contrôleur SDRAM dédié au KV du seq ----
    // En sim gen_seq_top (clk unique) : pas de CDC, 2e sdram.v + sdram_chip séparés
    // (le seq écrit/lit sa propre mémoire KV). En silicium, 1.1d fait le 4e maître
    // du mux node_top (une seule SDRAM physique). Le seq pulse kv_refresh lui-même
    // (~15µs) ; routé vers le refresh du 2e contrôleur (mock sdram_chip = pas de
    // perte en sim, mais la FSM seq l'exerce comme en silicium).
    wire [22:0] kv_addr;  wire kv_rd, kv_wr;  wire [7:0] kv_din;
    wire [7:0]  kv_dout;  wire kv_busy;        wire kv_refresh;
    wire o_kv_sdram_clk, o_kv_sdram_cas_n, o_kv_sdram_ras_n, o_kv_sdram_wen_n;
    wire [10:0] o_kv_sdram_addr;  wire [1:0] o_kv_sdram_ba;  wire [31:0] io_kv_sdram_dq;

`ifdef FLASH_BOOT
    // ---- FLASH_BOOT mock (path silicium en sim) ----
    // node_top instancie `flash_boot` (module) ; on fournit le mock flash_boot_sim.v
    // (drop-in) a la compile -> node_top l'utilise. Le mock signale boot_done apres
    // qques cycles (SDRAM deja prechargee via SDRAM_PRELOAD), ce qui reveille la FSM
    // wcache_fill -> remplit la BRAM cache depuis SDRAM = path silicium (cache ACTIF).
    // shifts_out/valid laisses non connectes : drive_shifts fournit sw_* via les ports
    // gen_seq (step-1). flash_sck/mosi/cs_n outputs node -> wires unused en sim (pas
    // de W25Q64). wcache_fill_done_w : le test attend dut.u_n.wcache_fill_done avant
    // de pulser start (cache rempli ~1M cycles apres boot_done).
    wire        flash_sck_w, flash_mosi_w, flash_cs_n_w, flash_sck_tsen_w;
    wire [47*8-1:0] shifts_out_w;
    wire        shifts_valid_w;
    wire        wcache_fill_done_w;
    assign      sim_fill_done = wcache_fill_done_w;   // sonde sim-only (cache rempli)
`else
    assign      sim_fill_done = 1'b1;                 // pas de cache fill hors FLASH_BOOT
`endif
`ifdef SIM
    wire [5:0] dbg_st_w;                              // debug : gen_seq FSM state
    wire [3:0] dbg_phase_w;                           // debug : gen_seq phase
    wire [11:0] dbg_idx_w;                            // debug : gen_seq idx
    wire [8:0] dbg_node_state_w;                      // debug : node FSM state
    wire [31:0] dbg_best_v_w;                         // debug : argmax best_val final
    wire [15:0]  dbg_best_i_w;                         // debug : argmax best_idx final
    assign dbg_st = dbg_st_w;                         // sonde SIM-only (localise hang)
    assign dbg_phase = dbg_phase_w;
    assign dbg_idx = dbg_idx_w;
    assign dbg_node_state = dbg_node_state_w;
    assign dbg_best_v = dbg_best_v_w;
    assign dbg_best_i = dbg_best_i_w;
`endif

    // rework dim Phase 1 : .D(CFG_D)/.NL(CFG_NL)/.NPOS(CFG_NPOS) (était .D(D) local défaut 64
    // -> seq restait à dim 64 même sous DIM128, mismatch avec node CFG_D=128 -> deadlock FIFO).
    gen_seq #(.W(W), .D(CFG_D), .NL(CFG_NL), .NPOS(CFG_NPOS),
              .H(CFG_H), .KH(CFG_KH), .HS(CFG_HS),
              .VOCAB(CFG_VOCAB), .NCHUNK(CFG_NCHUNK), .TMAX(CFG_TMAX),
              .HID(CFG_HID), .LSTRIDE(CFG_LSTRIDE), .LBASE(CFG_LBASE),
              .OFF_RMS(CFG_OFF_RMS), .OFF_WQ(CFG_OFF_WQ), .OFF_WK(CFG_OFF_WK),
              .OFF_WV(CFG_OFF_WV), .OFF_WO(CFG_OFF_WO), .OFF_RMSFF(CFG_OFF_RMSFF),
              .OFF_W1(CFG_OFF_W1), .OFF_W3(CFG_OFF_W3), .OFF_W2(CFG_OFF_W2),
              .A_RMSFINAL(CFG_A_RMSFINAL), .A_EMB(CFG_A_EMB), .KV_BASE(CFG_KV_BASE)) u_seq (
        .clk(clk), .rst_n(rst_n), .start(start), .gen_mode(gen_mode),
        .x_in(x_in), .sx_in(sx_in),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .sw_rmsf(sw_rmsf), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .sw_rmsfinal(sw_rmsfinal), .sw_emb(sw_emb),
        .pos(pos), .base(23'd0),
        .prompt(prompt), .prompt_len(prompt_len),
        .lk_cmd_data(s_cmd_data), .lk_cmd_wr(s_cmd_wr), .lk_cmd_full(s_cmd_full),
        .lk_resp_data(s_resp_data), .lk_resp_empty(s_resp_empty), .lk_resp_rd(s_resp_rd),
        .result(result), .result_sh(result_sh), .token(token),
        .token_valid(token_valid), .done(done),
        // KV SDRAM maître direct (étape 1)
        .kv_addr(kv_addr), .kv_rd(kv_rd), .kv_wr(kv_wr), .kv_din(kv_din),
        .kv_dout(kv_dout), .kv_busy(kv_busy), .kv_refresh(kv_refresh)
`ifdef SIM
        ,.dbg_st(dbg_st_w)
        ,.dbg_phase(dbg_phase_w)
        ,.dbg_idx(dbg_idx_w)
        ,.dbg_best_v(dbg_best_v_w)
        ,.dbg_best_i(dbg_best_i_w)
`endif
        );

    wire [7:0] c_rdata; wire c_empty; wire c_rd;
    async_fifo #(.DW(8), .AW(4)) u_cmd (
        .wclk(clk), .wrst_n(rst_n), .winc(s_cmd_wr), .wdata(s_cmd_data), .wfull(s_cmd_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(c_rd), .rdata(c_rdata), .rempty(c_empty));
    wire [7:0] r_wdata; wire r_wr; wire r_full;
    async_fifo #(.DW(8), .AW(4)) u_resp (
        .wclk(clk), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(s_resp_rd), .rdata(s_resp_data), .rempty(s_resp_empty));

    wire o_sdram_clk, o_sdram_cas_n, o_sdram_ras_n, o_sdram_wen_n;
    wire [10:0] o_sdram_addr; wire [1:0] o_sdram_ba; wire [31:0] io_sdram_dq;
    top u_n (
        .clk(clk),
        .lk_rx_data(c_rdata), .lk_rx_empty(c_empty), .lk_rx_rd(c_rd),
        .lk_tx_data(r_wdata), .lk_tx_wr(r_wr), .lk_tx_full(r_full), .led(),
        .O_sdram_clk(o_sdram_clk),
        .O_sdram_cas_n(o_sdram_cas_n), .O_sdram_ras_n(o_sdram_ras_n), .O_sdram_wen_n(o_sdram_wen_n),
        .IO_sdram_dq(io_sdram_dq), .O_sdram_addr(o_sdram_addr), .O_sdram_ba(o_sdram_ba),
        // KV maître : en sim, le seq a sa 2e SDRAM dédiée -> node SDRAM inchangée.
        .kv_addr(23'd0), .kv_rd(1'b0), .kv_wr(1'b0), .kv_din(8'd0), .kv_refresh(1'b0),
        .kv_dout(), .kv_busy()
`ifdef SIM
        ,.dbg_node_state(dbg_node_state_w)
`endif
`ifdef FLASH_BOOT
        ,// FLASH_BOOT mock : flash_* (SPI pins, unused en sim), shifts (drive_shifts
         // fournit sw_*), wcache_fill_done (le test attend ce signal avant start).
         .flash_sck(flash_sck_w), .flash_mosi(flash_mosi_w), .flash_miso(1'b0),
         .flash_cs_n(flash_cs_n_w), .flash_sck_tsen(flash_sck_tsen_w),
         .shifts_out(shifts_out_w), .shifts_valid(shifts_valid_w),
         .wcache_fill_done(wcache_fill_done_w)
`endif
        );
    // Colorlight i9 : CKE=VCC, nCS=GND, DQM=GND hardwired on PCB (no FPGA pin).
    sdram_chip u_sdram (
        .SDRAM_CLK(o_sdram_clk), .SDRAM_CKE(1'b1), .SDRAM_nCS(1'b0),
        .SDRAM_nRAS(o_sdram_ras_n), .SDRAM_nCAS(o_sdram_cas_n), .SDRAM_nWE(o_sdram_wen_n),
        .SDRAM_BA(o_sdram_ba), .SDRAM_A(o_sdram_addr), .SDRAM_DQM(4'b0000),
        .SDRAM_DQ(io_sdram_dq));

    // KV cache SDRAM (étape 1) : 2e contrôleur dédié au seq. clk_sdram = ~clk
    // (on réutilise o_sdram_clk du node, déjà décalée 180°). En sim le KV n'est pas
    // préchargé (le seq l'écrit en KVWR puis le lit en PH_MM).
    sdram u_kv_sdram (
        .clk(clk), .clk_sdram(o_sdram_clk), .resetn(rst_n),
        .rd(kv_rd), .wr(kv_wr), .refresh(kv_refresh),
        .addr(kv_addr), .din(kv_din), .dout(kv_dout), .busy(kv_busy),
        // SDRAM-side : pilotent le mock u_kv_chip. AVANT ces outputs n'étaient PAS
        // connectés -> u_kv_chip recevait des wires non drivés (z) -> read SDRAM KV
        // = z -> PH_MM lit K/V=z -> scores=X -> cascade X -> token 0 (cf
        // flash-boot-mock-sim). En silicium le seq passe par le mux node_top (1
        // SDRAM physique), ces wires sim-only reproduisent le path.
        .SDRAM_DQ(io_kv_sdram_dq), .SDRAM_A(o_kv_sdram_addr),
        .SDRAM_BA(o_kv_sdram_ba), .SDRAM_nWE(o_kv_sdram_wen_n),
        .SDRAM_nRAS(o_kv_sdram_ras_n), .SDRAM_nCAS(o_kv_sdram_cas_n),
        .SDRAM_CLK(o_kv_sdram_clk), .SDRAM_DQM(),
        .SDRAM_nCS(), .SDRAM_CKE());
    sdram_chip u_kv_chip (
        .SDRAM_CLK(o_kv_sdram_clk), .SDRAM_CKE(1'b1), .SDRAM_nCS(1'b0),
        .SDRAM_nRAS(o_kv_sdram_ras_n), .SDRAM_nCAS(o_kv_sdram_cas_n),
        .SDRAM_nWE(o_kv_sdram_wen_n),
        .SDRAM_BA(o_kv_sdram_ba), .SDRAM_A(o_kv_sdram_addr), .SDRAM_DQM(4'b0000),
        .SDRAM_DQ(io_kv_sdram_dq));
endmodule
