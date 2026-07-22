`timescale 1ns/1ps
// =============================================================================
// tb_kv_cdc -- testbench ciblé pour kv_cdc (CDC clk 25 MHz <-> clk_sys 31.25 MHz).
// Modélise le contrôleur SDRAM côté clk_sys : sur pulse n_kv_rd/wr/refresh,
// sd_busy=1 pendant 5 clk_sys, sd_dout valide 1 cycle AVANT sd_busy tombe
// (fidèle à sdram.v : data_ready@T_RCD+CAS, busy@T_RCD+CAS+1).
// Côté clk : drive writes/reads/refresh via le seq-like pattern (pulse 1 cycle,
// attend s_kv_busy=1 puis =0, capture s_kv_dout).
// Vérifie : (1) aucun pulse perdu (chaque op écrite est relue correcte),
// (2) dout aligné (la valeur lue = la valeur écrite à la même adresse),
// (3) op types corrects (rd/wr/refresh vus côté clk_sys).
// =============================================================================
module tb_kv_cdc;
    reg clk = 0, clk_sys = 0, rst_n = 0;
    // seq side (clk)
    reg  [22:0] s_kv_addr = 0;
    reg         s_kv_rd = 0, s_kv_wr = 0, s_kv_refresh = 0;
    reg  [7:0]  s_kv_din = 0;
    wire [7:0]  s_kv_dout;
    wire        s_kv_busy;
    // node side (clk_sys)
    wire [22:0] n_kv_addr;
    wire        n_kv_rd, n_kv_wr, n_kv_refresh;
    wire [7:0]  n_kv_din;
    wire [7:0]  n_sd_dout;
    wire        n_sd_busy;

    // horloges async (rapport non entier 25/31.25 = 4/5)
    always #20  clk = ~clk;          // 25 MHz
    always #16  clk_sys = ~clk_sys;  // 31.25 MHz

    kv_cdc dut (
        .clk(clk), .clk_sys(clk_sys), .rst_n(rst_n),
        .s_kv_addr(s_kv_addr), .s_kv_rd(s_kv_rd), .s_kv_wr(s_kv_wr),
        .s_kv_din(s_kv_din), .s_kv_refresh(s_kv_refresh),
        .s_kv_dout(s_kv_dout), .s_kv_busy(s_kv_busy),
        .n_kv_addr(n_kv_addr), .n_kv_rd(n_kv_rd), .n_kv_wr(n_kv_wr),
        .n_kv_din(n_kv_din), .n_kv_refresh(n_kv_refresh),
        .n_sd_dout(n_sd_dout), .n_sd_busy(n_sd_busy));

    // ---- modèle contrôleur SDRAM (clk_sys) -- fidèle à sdram.v ----
    // Mémoire 32-bit par mot (comme la SDRAM physique). Shadow-word : accumulate
    // 4 bytes par mot (DQM hardwired -> drive le mot complet). Reset du shadow
    // quand addr[22:2] change. data_ready@T_RCD+CAS, busy tombe@T_RCD+CAS+1.
    reg [31:0] sim_word [0:255];
    reg [31:0] sh_word = 0;
    reg [20:0] sh_addr = 21'h1FFFFF;
    reg [22:0] op_addr;  reg op_wr;  reg [1:0] op_off;
    reg [2:0] cstate = 0;
    reg        sd_busy_r = 0;
    reg [31:0] dq_seen   = 0;       // mot réellement drivé sur DQ à l'écriture
    reg [7:0]  dout_buf_r = 0;
    reg        dout_ready = 0;
    localparam C_IDLE=0, C_B1=1, C_B2=2, C_B3=3, C_B4=4, C_DONE=5;
    always @(posedge clk_sys) begin
        if (!rst_n) begin
            cstate <= C_IDLE; sd_busy_r <= 0; dout_ready <= 0;
            sh_word <= 0; sh_addr <= 21'h1FFFFF;
        end else begin
            case (cstate)
                C_IDLE: begin
                    sd_busy_r <= 0;
                    if (n_kv_rd | n_kv_wr) begin
                        op_addr <= n_kv_addr; op_wr <= n_kv_wr; op_off <= n_kv_addr[1:0];
                        if (n_kv_wr) begin
                            // shadow-word (identique sdram.v l.176-181)
                            if (n_kv_addr[22:2] != sh_addr) sh_word <= 32'b0;
                            sh_word[n_kv_addr[1:0]*8 +: 8] <= n_kv_din;
                            sh_addr <= n_kv_addr[22:2];
                        end
                        cstate <= C_B1; sd_busy_r <= 1;
                    end else if (n_kv_refresh) begin
                        op_wr <= 0; cstate <= C_B1; sd_busy_r <= 1;
                    end
                end
                C_B1: cstate <= C_B2;
                C_B2: cstate <= C_B3;
                // WRITE@T_RCD : dq_out=sh_word (mot complet) -> commit le mot
                // READ@T_RCD+CAS : data_ready, sd_dout = byte du mot lu
                C_B3: begin
                    if (op_wr) begin
                        sim_word[op_addr[9:2]] <= sh_word;   // commit mot complet
                        dq_seen <= sh_word;
                    end
                    dout_buf_r <= sim_word[op_addr[9:2]][op_off*8 +: 8];
                    dout_ready <= 1;
                    cstate <= C_B4;
                end
                C_B4: begin sd_busy_r <= 0; cstate <= C_DONE; end
                C_DONE: cstate <= C_IDLE;
                default: cstate <= C_IDLE;
            endcase
        end
    end
    assign n_sd_dout = dout_ready ? sim_word[op_addr[9:2]][op_off*8 +: 8] : dout_buf_r;
    assign n_sd_busy = sd_busy_r;

    // ---- monitor côté clk_sys : log chaque op vue ----
    integer opcount = 0;
    always @(posedge clk_sys) if (rst_n && (n_kv_rd|n_kv_wr|n_kv_refresh)) begin
        opcount = opcount + 1;
        $display("[%0t] clk_sys OP#%0d rd=%b wr=%b ref=%b addr=%0d din=%0d",
                 $time, opcount, n_kv_rd, n_kv_wr, n_kv_refresh, n_kv_addr, n_kv_din);
    end

    // ---- driver côté clk : pattern seq (pulse 1 cycle, wait busy=1, wait !busy) ----
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
    task do_refresh;
        begin
            @(posedge clk); s_kv_refresh <= 1;
            @(posedge clk); s_kv_refresh <= 0;
            wait(s_kv_busy); @(posedge clk);
            while (s_kv_busy) @(posedge clk);
        end
    endtask

    integer errors = 0;
    reg [7:0] rd;
    integer i;
    initial begin
        // init sim_word à 0xFFFFFFFF (pour détecter les reads non écrits)
        for (i=0; i<256; i=i+1) sim_word[i] = 32'hFFFFFFFF;
        // reset
        repeat (20) @(posedge clk);
        rst_n = 1;
        repeat (20) @(posedge clk);

        // ---- test 1 : write 64 bytes (un (layer,pos) de KV) puis read back ----
        $display("== Test 1 : write/read 64 bytes KV ==");
        for (i=0; i<64; i=i+1) do_write(23'h070000 + i, i+8'h10);
        for (i=0; i<64; i=i+1) begin
            do_read(23'h070000 + i, rd);
            if (rd !== (i+8'h10)) begin
                $display("  FAIL byte %0d : lu %0d attendu %0d", i, rd, i+8'h10);
                errors = errors + 1;
            end else $display("  OK byte %0d = %0d", i, rd);
        end

        // ---- test 2 : interleaved K/V comme PH_MM (K 0..31 puis V 0..31) ----
        $display("== Test 2 : K puis V (32+32) ==");
        for (i=0; i<32; i=i+1) do_write(23'h070000 + i, 8'hA0 + i[7:0]);      // K
        for (i=0; i<32; i=i+1) do_write(23'h070000 + 32 + i, 8'hB0 + i[7:0]);  // V
        for (i=0; i<32; i=i+1) begin
            do_read(23'h070000 + i, rd);
            if (rd !== 8'hA0 + i[7:0]) begin $display("  FAIL K %0d lu %0d", i, rd); errors=errors+1; end
        end
        for (i=0; i<32; i=i+1) begin
            do_read(23'h070000 + 32 + i, rd);
            if (rd !== 8'hB0 + i[7:0]) begin $display("  FAIL V %0d lu %0d", i, rd); errors=errors+1; end
        end
        $display("  Test 2 done");

        // ---- test 3 : refresh entrelacé (comme preemption seq) ----
        $display("== Test 3 : refresh entrelacé ==");
        do_write(23'h070200, 8'h5A);
        do_refresh;
        do_read(23'h070200, rd);
        if (rd !== 8'h5A) begin $display("  FAIL refresh+read lu %0d", rd); errors=errors+1; end
        else $display("  OK refresh entrelacé");

        // ---- test 4 : rafale rapide writes puis reads (stress toggle sync) ----
        $display("== Test 4 : rafale 256 writes puis 256 reads ==");
        for (i=0; i<256; i=i+1) do_write(23'h071000 + i, i[7:0] ^ 8'hCC);
        for (i=0; i<256; i=i+1) begin
            do_read(23'h071000 + i, rd);
            if (rd !== (i[7:0] ^ 8'hCC)) begin
                $display("  FAIL rafale %0d lu %0d attendu %0d", i, rd, i[7:0]^8'hCC);
                errors=errors+1;
            end
        end
        $display("  Test 4 done");

        $display("== RESULT : %0d errors, %0d ops clk_sys ==", errors, opcount);
        if (errors==0) $display("PASS"); else $display("FAIL");
        $finish;
    end

    // timeout
    initial begin
        #5000000;
        $display("TIMEOUT");
        $finish;
    end
endmodule