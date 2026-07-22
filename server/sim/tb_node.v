// tb_node.v — TB Verilog pur instanciant gen_seq_top (le node complet) pour la
// Phase C du Packet Tracer : produire un VCD du parcours du signal (FSM states,
// opcodes, reads SDRAM, data path NN) à animer en overlay sur le graphe RTL.
//
// Compilé via server/sim/iverilog_runner.py (mode iverilog, -DLINK_SS
// -DSDRAM_PRELOAD). La SDRAM mock (sim/models/sdram_chip.v) précharge les poids
// via $readmemh("sdram_preload.hex") ; le runner copie le .hex + les LUT vers
// WORK (cwd de vvp).
//
// NB : le forward donne rtl=0 en mode non-FLASH_BOOT (path != path silicium
// FLASH_BOOT, cf mémoire sim-path-vs-silicon-path) — mais l'ACTIVITÉ des modules
// (FSM, reads, rmsnorm, attention) suffit pour l'animation. L'objectif de la
// Phase C est le PARCOURS, pas la valeur du token.
//
// $dumpvars restreint à u_seq + u_n (le sequencer + le data path NN + contrôleur
// SDRAM). n'inclut PAS u_dut.u_sdram (sdram_chip, 2M entrées -> VCD géant) ni
// u_dut.u_cmd/u_resp (FIFOS peu visuelles). u_n.u_wc (wcache_ffn BRAM 184320) est
// inclus mais n'est pas écrit en non-FLASH_BOOT -> init seulement, gérable.
`timescale 1ns/1ps
module tb_node;
    // gen_seq_top defaults : W=8, D=64, HS=8, NL=5, NPOS=17
    localparam W = 8, D = 64, NL = 5;
    localparam WD = W*D;            // 512
    localparam NLS = NL*8;          // 40

    reg         clk = 1'b0;
    reg         rst_n = 1'b0;
    reg         start = 1'b0;
    reg         gen_mode = 1'b0;
    reg  [WD-1:0]  x_in = 0;
    reg  signed [7:0] sx_in = 0;
    // Shifts int8 (signés, 8-bit) hardcodés depuis host/gen_model_dump.json (_s).
    // Layer 0 = byte 0 (LSB). 0xFB=-5, 0xFA=-6, 0xF8=-8, 0xF9=-7, 0xFC=-4.
    reg  [NLS-1:0] sw_rms    = 40'hFBFBFBFBFB;  // rms_att_s : -5 par couche
    reg  [NLS-1:0] swq       = 40'hFAFAFAFAFA;  // wq_s : -6
    reg  [NLS-1:0] swk       = 40'hFAFAF9FAFA;  // wk_s : -6 (L2=-7)
    reg  [NLS-1:0] swv       = 40'hF8F8F8F7F8;  // wv_s : -8 (L1=-9)
    reg  [NLS-1:0] swo       = 40'hF9F8F8F8F9;  // wo_s : -7/-8
    reg  [NLS-1:0] sw_rmsf   = 40'hFAFAFAFAF9;  // rms_ffn_s : -6/-7
    reg  [NLS-1:0] sw1       = 40'hF9F9F9F9F9;  // w1_s : -7
    reg  [NLS-1:0] sw3       = 40'hF9F9F9F9F9;  // w3_s : -7
    reg  [NLS-1:0] sw2       = 40'hF9F9F9F9F9;  // w2_s : -7
    reg  signed [7:0] sw_rmsfinal = 8'hFC;      // rms_final_s : -4
    reg  signed [7:0] sw_emb      = 8'hFA;      // tok_emb_s : -6
    reg  [5:0]   pos = 6'd0;

    wire [WD-1:0] result;
    wire signed [7:0] result_sh;
    wire [9:0]   token;
    wire         token_valid;
    wire         done;

    gen_seq_top #(.W(W), .D(D), .NL(NL)) u_dut (
        .clk(clk), .rst_n(rst_n), .start(start), .gen_mode(gen_mode),
        .x_in(x_in), .sx_in(sx_in),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .sw_rmsf(sw_rmsf), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .sw_rmsfinal(sw_rmsfinal), .sw_emb(sw_emb), .pos(pos),
        .result(result), .result_sh(result_sh),
        .token(token), .token_valid(token_valid), .done(done));

    // clock 100 MHz (la fréquence réelle n'importe pas pour la visu)
    always #5 clk = ~clk;

    integer i;
    initial begin
        $dumpfile("tb_node.vcd");
        // Restreint : sequencer + node (data path NN + ctrl SDRAM).
        // Exclut u_dut.u_sdram (sdram_chip 2M), u_dut.u_cmd/u_resp.
        $dumpvars(0, tb_node.u_dut.u_seq);
        $dumpvars(0, tb_node.u_dut.u_n);

        // reset
        rst_n = 1'b0;
        for (i = 0; i < 20; i = i + 1) @(negedge clk);
        rst_n = 1'b1;
        for (i = 0; i < 4; i = i + 1) @(negedge clk);

        // mode génération autonome + pulse start
        gen_mode = 1'b1;
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;

        // laisse tourner ~60k cycles (EE + rmsnorm + début attention pour la visu)
        for (i = 0; i < 60000; i = i + 1) @(posedge clk);
        $display("TB_NODE: t=%0t ns token=%0d token_valid=%b done=%b",
                  $time, token, token_valid, done);
        $finish;
    end

    // watchdog
    initial begin
        #2000000;  // 2 ms sim max
        $display("TB_NODE: TIMEOUT");
        $finish;
    end
endmodule