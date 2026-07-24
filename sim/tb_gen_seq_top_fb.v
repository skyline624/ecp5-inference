// =============================================================================
// tb_gen_seq_top_fb.v -- testbench Verilog PUR pour gen_seq_top (path FLASH_BOOT).
//
// Remplace le flot cocotb (test_gen_F_fb.py) pour accelerer la sim :
//   - clock free-running en C/Verilog (always #18.5 clk=~clk) -> PLUS de callback
//     Python par cycle (le clock-driver cocotb = ~50M callbacks Timer etait le
//     bottleneck dominant, independant de -SimDebug).
//   - watchdog en C (compteur cyc, $finish) -> plus de ClockCycles(35M) Python.
//   - capture sur front montant token_valid, compare vs $readmemh tb_expected.hex.
//
// Marche sur iverilog (maintenant) et le simulateur compile --timing (TB pret pour
// ce simulateur ; acces hierarchique dut.u_n.wcache_fill_done -> ajouter
// --public-flat-rw). Aucun cocotb/Python/VPI.
//
// Fichiers hex (gen_tb_hex.py) dans sim/ : tb_meta.hex (NTOK), tb_shifts.hex
// (11 ports sw_*), tb_expected.hex (tokens attendus = node_tokens).
//
// Lancement iverilog (depuis sim/) :
//   iverilog -o tb_gen_seq_top_fb.vvp -s tb_gen_seq_top_fb -g2012 \
//     -DLINK_SS -DSDRAM_PRELOAD -DSIM -DFLASH_BOOT -I../src \
//     tb_gen_seq_top_fb.v models/ECP5_PLL_sim.v models/MULTALU18X18.v \
//     models/sdram_chip.v models/flash_boot_sim.v \
//     ../src/node_top.v ../src/wcache_ffn.v ../src/sdram.v ../src/mac18.v \
//     ../src/rmsnorm_op.v ../src/silu_op.v ../src/rope_op.v ../src/softmax_op.v \
//     ../src/attention_head_op.v ../src/vec_alu2.v ../src/async_fifo.v \
//     ../src/uart_bridge.v ../src/gen_seq.v ../src/gen_seq_top.v
//   vvp tb_gen_seq_top_fb.vvp
// =============================================================================
`timescale 1ns/1ps
module tb_gen_seq_top_fb;
    `include "model_cfg.vh"
    localparam W = 8;

    reg clk = 1'b0;
    always #18.5 clk = ~clk;          // 37ns period ~27 MHz (match cocotb)

    reg rst_n = 1'b0;
    reg start = 1'b0, gen_mode = 1'b0;
    reg [W*CFG_D-1:0]   x_in = 0;
    reg signed [7:0] sx_in = 0;
    reg [CFG_NL*8-1:0]  sw_rms=0, swq=0, swk=0, swv=0, swo=0;
    reg [CFG_NL*8-1:0]  sw_rmsf=0, sw1=0, sw3=0, sw2=0;
    reg signed [7:0] sw_rmsfinal=0, sw_emb=0;
    reg [5:0]       pos = 0;
    reg [16*10-1:0] prompt = 0;
    reg [6:0]       prompt_len = 0;

    wire [W*CFG_D-1:0]    result;
    wire signed [7:0] result_sh;
    wire [9:0]       token;
    wire             token_valid, done;
    wire             sim_fill_done;

    gen_seq_top #(.W(W), .D(CFG_D), .NL(CFG_NL), .NPOS(CFG_NPOS)) dut (
        .clk(clk), .rst_n(rst_n), .start(start), .gen_mode(gen_mode),
        .x_in(x_in), .sx_in(sx_in),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .sw_rmsf(sw_rmsf), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .sw_rmsfinal(sw_rmsfinal), .sw_emb(sw_emb),
        .pos(pos), .prompt(prompt), .prompt_len(prompt_len),
        .result(result), .result_sh(result_sh), .token(token),
        .token_valid(token_valid), .done(done),
        .sim_fill_done(sim_fill_done)
    );

    // ---- entrees hex ----
    reg [31:0]      meta_mem [0:0];
    reg [CFG_NL*8-1:0]  shifts_mem [0:10];
    reg [9:0]       expect_tok [0:63];
    integer NTOK;
    integer got_count;
    integer mismatch;

    // ---- watchdog (en C, rapide) : budget 40M cyc (fill ~1M + 17 tok ~1.5M) ----
    integer cyc = 0;
    always @(posedge clk) begin
        cyc <= cyc + 1;
        if (cyc > 40_000_000) begin
            $display("FAIL: watchdog timeout apres %0d cyc, got=%0d/%0d", cyc, got_count, NTOK);
            $finish;
        end
    end

    // ---- heartbeat : progression visible (où en est la sim) ----
    integer hb = 0;
    always @(posedge clk) begin
        hb <= hb + 1;
        if (hb >= 500_000) begin
            hb <= 0;
            $display("[hb] cyc=%0d got=%0d/%0d fill_done=%b done=%b tv=%b",
                     cyc, got_count, NTOK, sim_fill_done, done, token_valid);
            $fflush;
        end
    end

    // ---- capture sur front montant token_valid ----
    reg tv_d = 1'b1;                  // init a 1 -> pas de faux front pendant le reset
    always @(posedge clk) begin
        tv_d <= token_valid;
        if (rst_n && token_valid && !tv_d && (got_count < NTOK)) begin
            if (token !== expect_tok[got_count]) begin
                $display("MISMATCH tok[%0d]=%0d expect=%0d", got_count, token, expect_tok[got_count]);
                mismatch <= mismatch + 1;
            end else begin
                $display("tok[%0d]=%0d OK (expect %0d)", got_count, token, expect_tok[got_count]);
            end
            got_count <= got_count + 1;
        end
        if (rst_n && (got_count == NTOK)) begin
            if (mismatch == 0)
                $display("PASS: %0d/%0d bit-exact vs node_tokens (FLASH_BOOT, TB Verilog pur)", got_count, NTOK);
            else
                $display("FAIL: %0d mismatch sur %0d tokens", mismatch, NTOK);
            $finish;
        end
    end

    initial begin
        $readmemh("tb_meta.hex", meta_mem);
        NTOK = meta_mem[0];
        $readmemh("tb_shifts.hex", shifts_mem);
        $readmemh("tb_expected.hex", expect_tok);
        got_count = 0;
        mismatch = 0;

        // shifts (layer0 -> [7:0], identique a cocotb drive_shifts)
        sw_rms      = shifts_mem[0];
        swq         = shifts_mem[1];
        swk         = shifts_mem[2];
        swv         = shifts_mem[3];
        swo         = shifts_mem[4];
        sw_rmsf     = shifts_mem[5];
        sw1         = shifts_mem[6];
        sw3         = shifts_mem[7];
        sw2         = shifts_mem[8];
        sw_rmsfinal = shifts_mem[9][7:0];
        sw_emb      = shifts_mem[10][7:0];

        rst_n = 1'b0;
        repeat (20) @(posedge clk);
        rst_n = 1'b1;

        // attend le remplissage du cache FFN BRAM (path silicium)
        wait (sim_fill_done === 1'b1);
        @(posedge clk);
        gen_mode = 1'b1;
        start   = 1'b1;
        @(posedge clk);
        start   = 1'b0;
        // la generation proceede ; le bloc always ci-dessus collecte les tokens
    end
endmodule