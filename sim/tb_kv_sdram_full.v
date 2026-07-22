`timescale 1ns/1ps
// =============================================================================
// tb_kv_sdram_full -- path complet kv_* : kv_cdc (clk 25 / clk_sys 31.25) +
// VRAI sdram.v + sdram_chip (DQM hardwired 0 comme la i9). Teste le pattern
// INTERLEAVED de KVWR (K[ldi]@off ldi puis V[ldi]@off 32+ldi) vs CONTIGU.
//
// Hypothèse : le shadow-word (sdram.v l.176-181) reset sh_word quand addr[22:2]
// change. En interleaved, K et V sont dans des mots différents (off ldi vs
// 32+ldi) -> chaque write reset sh_word -> seul le dernier byte du mot survit,
// les 3 autres clobberés à 0 (DQM=0 écrit tout le mot). Contigu = OK (accumule).
// =============================================================================
module tb_kv_sdram_full;
    reg clk = 0, clk_sys = 0, rst_n = 0;
    always #20  clk     = ~clk;       // 25 MHz
    always #16  clk_sys = ~clk_sys;   // 31.25 MHz
    wire clk_sdram = ~clk_sys;

    // seq side (clk)
    reg  [22:0] s_kv_addr = 0;
    reg         s_kv_rd = 0, s_kv_wr = 0, s_kv_refresh = 0;
    reg  [7:0]  s_kv_din = 0;
    wire [7:0]  s_kv_dout;
    wire        s_kv_busy;
    // node side (clk_sys) -> mux eff_*
    wire [22:0] n_kv_addr;
    wire        n_kv_rd, n_kv_wr, n_kv_refresh;
    wire [7:0]  n_kv_din;
    wire [7:0]  sd_dout;
    wire        sd_busy;

    kv_cdc dut (
        .clk(clk), .clk_sys(clk_sys), .rst_n(rst_n),
        .s_kv_addr(s_kv_addr), .s_kv_rd(s_kv_rd), .s_kv_wr(s_kv_wr),
        .s_kv_din(s_kv_din), .s_kv_refresh(s_kv_refresh),
        .s_kv_dout(s_kv_dout), .s_kv_busy(s_kv_busy),
        .n_kv_addr(n_kv_addr), .n_kv_rd(n_kv_rd), .n_kv_wr(n_kv_wr),
        .n_kv_din(n_kv_din), .n_kv_refresh(n_kv_refresh),
        .n_sd_dout(sd_dout), .n_sd_busy(sd_busy));

    // mux eff_* : kv_firing only (pas de node master dans ce TB isolé).
    wire kv_firing = n_kv_rd | n_kv_wr;
    wire [22:0] eff_addr = n_kv_addr;
    wire        eff_rd   = n_kv_rd;
    wire        eff_wr   = n_kv_wr;
    wire [7:0]  eff_din  = n_kv_din;
    wire        eff_ref  = n_kv_refresh;

    wire [31:0] SDRAM_DQ;
    wire [10:0] SDRAM_A;
    wire [1:0]  SDRAM_BA;
    wire        SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE, SDRAM_CLK;

    sdram #(.FREQ(31_250_000)) u_ctrl (
        .clk(clk_sys), .clk_sdram(clk_sdram), .resetn(rst_n),
        .rd(eff_rd), .wr(eff_wr), .refresh(eff_ref),
        .addr(eff_addr), .din(eff_din), .dout(sd_dout), .dout32(),
        .data_ready(), .busy(sd_busy),
        .SDRAM_DQ(SDRAM_DQ), .SDRAM_A(SDRAM_A), .SDRAM_BA(SDRAM_BA),
        .SDRAM_nCS(), .SDRAM_nWE(SDRAM_nWE),
        .SDRAM_nRAS(SDRAM_nRAS), .SDRAM_nCAS(SDRAM_nCAS),
        .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(), .SDRAM_DQM());

    sdram_chip u_chip (
        .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(1'b1), .SDRAM_nCS(1'b0),
        .SDRAM_nRAS(SDRAM_nRAS), .SDRAM_nCAS(SDRAM_nCAS), .SDRAM_nWE(SDRAM_nWE),
        .SDRAM_BA(SDRAM_BA), .SDRAM_A(SDRAM_A), .SDRAM_DQM(4'b0000),
        .SDRAM_DQ(SDRAM_DQ));

    // ---- driver seq-like (clk domain) : pulse 1c, wait busy=1 puis =0 ----
    task do_write(input [22:0] a, input [7:0] d);
        begin
            @(posedge clk); s_kv_addr <= a; s_kv_din <= d; s_kv_wr <= 1;
            @(posedge clk); s_kv_wr <= 0;
            wait(s_kv_busy); @(posedge clk);
            while (s_kv_busy) @(posedge clk);
        end
    endtask
    task do_read(input [22:0] a, output [7:0] rd);
        begin
            @(posedge clk); s_kv_addr <= a; s_kv_rd <= 1;
            @(posedge clk); s_kv_rd <= 0;
            wait(s_kv_busy); @(posedge clk);
            while (s_kv_busy) @(posedge clk);
            rd = s_kv_dout;
        end
    endtask

    localparam [22:0] KV_BASE = 23'h070000;
    localparam KVW = 32;
    integer errors = 0, i;
    reg [7:0] rd;
    reg [7:0] Kexp [0:31], Vexp [0:31];

    // peek chip mem à addr[22:2] (mot 32-bit) pour voir le clobbering direct
    task peek_word(input [22:0] a, input [31:0] exp, input [255:0] msg);
        reg [31:0] got;
        begin
            got = u_chip.mem[a[22:2]];
            if (got !== exp) begin
                errors = errors + 1;
                $display("  FAIL %0s: got %h exp %h", msg, got, exp);
            end else $display("  OK   %0s: %h", msg, got);
        end
    endtask

    initial begin
        for (i=0;i<32;i=i+1) begin Kexp[i] = 8'hA0 + i[7:0]; Vexp[i] = 8'hB0 + i[7:0]; end

        repeat (20) @(posedge clk);
        rst_n = 1;
        // attendre fin init SDRAM 200us (~12500 cyc clk_sys) : busy tombe
        wait(sd_busy == 1'b0);
        repeat (200) @(posedge clk_sys);
        $display("SDRAM init done. Test INTERLEAVED (pattern KVWR seq) :");

        // ---- pattern INTERLEAVED exact du seq KVWR : K[ldi]@off ldi puis V[ldi]@32+ldi ----
        for (i=0;i<KVW;i=i+1) do_write(KV_BASE + i,      Kexp[i]);  // all K contigu (offsets 0..31)
        for (i=0;i<KVW;i=i+1) do_write(KV_BASE + 32 + i,   Vexp[i]);  // all V contigu (offsets 32..63)

        // peek mots K : mot0 (off 0..3) ne devrait contenir QUE K[3] si clobbering
        $display("  Peek chip apres interleaved (mot0=off0..3, mot8=off32..35) :");
        peek_word(KV_BASE + 0, {Kexp[3],Kexp[2],Kexp[1],Kexp[0]}, "mot0 K (contigu -> mot complet)");
        peek_word(KV_BASE + 4, {Kexp[7],Kexp[6],Kexp[5],Kexp[4]}, "mot1 K");
        peek_word(KV_BASE + 32, {Vexp[3],Vexp[2],Vexp[1],Vexp[0]}, "mot8 V");

        // readback K[0..31] via le path complet (kv_cdc read)
        $display("  Readback K[0..31] via kv_cdc :");
        for (i=0;i<KVW;i=i+1) begin
            do_read(KV_BASE + i, rd);
            // si clobbering : rd = 0 sauf i%4==3 (Kexp[i])
            if (rd !== Kexp[i]) begin
                errors = errors + 1;
                if (errors < 8) $display("    FAIL K[%0d] lu %0h exp %0h", i, rd, Kexp[i]);
            end
        end
        if (errors == 0) $display("    K readback integre (pas de clobbering)");

        $display("== RESULT : %0d errors ==", errors);
        if (errors == 0) $display("PASS (interleaved OK -> shadow-word robuste)");
        else $display("FAIL (interleaved clobber -> BUG SHADOW-WORD confirme)");
        $finish;
    end

    initial begin #8000000; $display("TIMEOUT"); $finish; end
endmodule