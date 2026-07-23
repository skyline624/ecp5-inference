`timescale 1ns/1ps
// gen_seq_top - one node (top.v LINK_SS) + the generation sequencer over ss_link.
module gen_seq_top #(
    parameter W = 8, parameter D = 64, parameter HS = 8, parameter NL = 5, parameter NPOS = 17
) (
    input  wire           clk, rst_n, start, gen_mode,
    input  wire [W*D-1:0] x_in,
    input  wire signed [7:0] sx_in,
    input  wire [NL*8-1:0] sw_rms, swq, swk, swv, swo,
    input  wire [NL*8-1:0] sw_rmsf, sw1, sw3, sw2,
    input  wire signed [7:0] sw_rmsfinal, sw_emb,
    input  wire [5:0]     pos,
    // prefill : prompt token IDs (packed, NPR*10 bits) + count. prompt_len=0 -> BOS
    // hardcodé côté gen_seq. Exposé ici pour piloter le token initial en gen_mode
    // (test_gen_A et tests de prefill) ; génère à partir du dernier token du prompt.
    input  wire [16*10-1:0] prompt,
    input  wire [6:0]        prompt_len,
    output wire [W*D-1:0] result,
    output wire signed [7:0] result_sh,
    output wire [9:0]     token,
    output wire           token_valid,
    output wire           done
);
    `include "model_cfg.vh"
    // gen_seq shape from model_cfg.vh (source unique) : HID/LSTRIDE/OFF_*/LBASE.
    // D/NL/NPOS restent des parameter (defaults dim 64 = CFG_). Pour dim 128+ :
    // surcharger ces parameter ET éditer model_cfg.vh ensemble (TODO unification).
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
`endif

    gen_seq #(.W(W), .D(D), .NL(NL), .NPOS(NPOS),
              .HID(CFG_HID), .LSTRIDE(CFG_LSTRIDE), .LBASE(CFG_LBASE),
              .OFF_W1(CFG_OFF_W1), .OFF_W3(CFG_OFF_W3), .OFF_W2(CFG_OFF_W2)) u_seq (
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
        .kv_dout(kv_dout), .kv_busy(kv_busy), .kv_refresh(kv_refresh));

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
