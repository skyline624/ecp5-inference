`timescale 1ns/1ps
// =============================================================================
// kv_cdc -- CDC clk (seq 25 MHz) <-> clk_sys (node 31.25 MHz) pour le maître KV
// SDRAM (étape 1 KV-cache BRAM->SDRAM, plan §2.2). Le seq pulse kv_rd/kv_wr/
// kv_refresh 1 cycle clk puis attend kv_busy (pattern v3). Le contrôleur SDRAM
// du node est sur clk_sys et attend un pulse 1-cycle en IDLE (un niveau maintenu
// re-déclencherait l'op). D'où :
//   * requête (clk -> clk_sys) : toggle synchronizer (pulse 1c -> pulse 1c) +
//     latch op_h/addr_h/din_h stables côté clk (échantillonnés direct côté clk_sys
//     au cycle du pulse sync, stable depuis plusieurs clk_sys -> pas de métastab).
//   * réponse (clk_sys -> clk) : 2-FF sur sd_busy + 2-FF sur sd_dout (staging
//     égal : sd_dout valide 1 cyc clk_sys avant sd_busy tombe -> dout_s2 valide
//     le cycle où s_kv_busy passe 0 -> E_MDONE capture le bon byte).
// Reset commun rst_n (por[15]), comme async_fifo.
// =============================================================================
module kv_cdc (
    input  wire        clk,
    input  wire        clk_sys,
    input  wire        rst_n,
    // seq side (clk) -- master
    input  wire [22:0] s_kv_addr,
    input  wire        s_kv_rd,
    input  wire        s_kv_wr,
    input  wire [7:0]  s_kv_din,
    input  wire        s_kv_refresh,
    output wire [7:0]  s_kv_dout,
    output wire        s_kv_busy,
    // node side (clk_sys) -- drives the 4th mux entry in node_top
    output wire [22:0] n_kv_addr,
    output wire        n_kv_rd,
    output wire        n_kv_wr,
    output wire [7:0]  n_kv_din,
    output wire        n_kv_refresh,
    input  wire [7:0]  n_sd_dout,
    input  wire        n_sd_busy
);
    // ---- requête : clk -> clk_sys (toggle synchronizer) ----
    // rd|wr|refresh = états FSM mutuellement exclusifs -> exactement un pulse par
    // cycle clk. On bascule req_tog et on latche op+addr+din au même front ; stables
    // jusqu'à la requête suivante (bien après la consommation côté clk_sys).
    reg        req_tog;
    reg [22:0] addr_h;
    reg [7:0]  din_h;
    reg [1:0]  op_h;        // 2'b01=rd, 2'b10=wr, 2'b11=refresh, 2'b00=none
    wire       req_in = s_kv_rd | s_kv_wr | s_kv_refresh;
    always @(posedge clk) begin
        if (!rst_n) begin
            req_tog <= 1'b0; addr_h <= 23'd0; din_h <= 8'd0; op_h <= 2'b00;
        end else if (req_in) begin
            req_tog <= ~req_tog;
            addr_h  <= s_kv_addr;
            if (s_kv_wr) din_h <= s_kv_din;
            if (s_kv_rd)         op_h <= 2'b01;
            else if (s_kv_wr)    op_h <= 2'b10;
            else                 op_h <= 2'b11;   // refresh
        end
    end

    // clk_sys : 2-FF sync du toggle + edge detect -> pulse 1-cycle.
    reg [2:0] tog_s;   // tog_s[1] = 2-FF synced, tog_s[2] = 1 cyc plus tard (edge)
    always @(posedge clk_sys) begin
        if (!rst_n) tog_s <= 3'b0;
        else        tog_s <= {tog_s[1:0], req_tog};
    end
    wire req_pulse = tog_s[2] ^ tog_s[1];

    assign n_kv_addr    = addr_h;       // stable (clk-domain) échantillonné direct
    assign n_kv_din     = din_h;
    assign n_kv_rd      = req_pulse & (op_h == 2'b01);
    assign n_kv_wr      = req_pulse & (op_h == 2'b10);
    assign n_kv_refresh = req_pulse & (op_h == 2'b11);

    // ---- réponse : clk_sys -> clk (2-FF busy + 2-FF dout, staging égal) ----
    reg [1:0] busy_s;
    reg [7:0] dout_s1, dout_s2;
    always @(posedge clk) begin
        if (!rst_n) begin
            busy_s <= 2'b0; dout_s1 <= 8'd0; dout_s2 <= 8'd0;
        end else begin
            busy_s  <= {busy_s[0], n_sd_busy};
            dout_s1 <= n_sd_dout;
            dout_s2 <= dout_s1;
        end
    end
    assign s_kv_busy = busy_s[1];
    assign s_kv_dout = dout_s2;
endmodule