// =============================================================================
// sdram_test.v -- Gate 2b (M2): SDRAM self-test on Colorlight i9 v7.2.
//
// Writes 4 distinct bytes per 32-bit word (filling the controller's shadow-word
// accumulator), then reads back the full dout32 and compares, over 1024 words
// x 2 pattern passes. Exercises all 32 DQ lines (each lane carries a distinct
// byte that varies across words) and 4 SDRAM rows (A[10:8]) in bank 0.
//
// LED D2:  slow blink (~1.5 Hz) = PASS (every word matched both passes)
//          steady              = FAIL (at least one mismatch)
//
// Why 4 bytes per word: on the i9 v7.2, DQM[3:0]=GND is HARDWIRED on the PCB
// (no FPGA pin), so the controller's byte-write mask is ineffective -- every
// write stores the full 32-bit DQ into all 4 lanes. The controller therefore
// accumulates byte writes into a shadow-word (sdram.v) and drives the full
// word on DQ. To store 4 distinct bytes in a word we must write them as 4
// contiguous byte writes (offsets 0..3) so the shadow builds {b3,b2,b1,b0}.
//
// Refresh: intentionally NOT issued during this short test (~2 ms total <<
// 64 ms retention). The controller's CONFIG does 2 auto-refreshes at init,
// giving enough margin for the test duration. The real design MUST interleave
// refresh every ~15 us via the sdram.v 'refresh' input.
//
// Build:  .\build.ps1 -Top sdram_test -Extra ../src/ecp5_ecp5pll.v,../src/sdram.v -Flash
// =============================================================================
module sdram_test (
    input  wire        clk25,
    output wire        led,
    inout  wire [31:0] SDRAM_DQ,
    output wire [10:0] SDRAM_A,
    output wire [1:0]  SDRAM_BA,
    output wire        SDRAM_nWE,
    output wire        SDRAM_nRAS,
    output wire        SDRAM_nCAS,
    output wire        SDRAM_CLK
);
    wire clk_sys, clk_sdram, lock;
    ecp5_pll u_pll (
        .clkin(clk25), .reset(1'b0),
        .clkout(clk_sys), .clkoutp(clk_sdram), .lock(lock)
    );

    // SDRAM controller. FREQ retuned to 25 MHz (sets the 200 us init counter).
    // nCS / CKE / DQM have NO FPGA pin on the i9 (hardwired on PCB) -> left
    // unconnected. refresh tied low (see note above).
    wire        rd, wr, data_ready, busy;
    wire [22:0] addr;
    wire [7:0]  din, dout;
    wire [31:0] dout32;
    sdram #(.FREQ(25_000_000)) u_sdram (
        .SDRAM_DQ(SDRAM_DQ), .SDRAM_A(SDRAM_A), .SDRAM_BA(SDRAM_BA),
        .SDRAM_nCS(), .SDRAM_nWE(SDRAM_nWE), .SDRAM_nRAS(SDRAM_nRAS),
        .SDRAM_nCAS(SDRAM_nCAS), .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(),
        .SDRAM_DQM(),
        .clk(clk_sys), .clk_sdram(clk_sdram), .resetn(lock),
        .rd(rd), .wr(wr), .refresh(1'b0), .addr(addr), .din(din),
        .dout(dout), .dout32(dout32), .data_ready(data_ready), .busy(busy)
    );

    // ----- self-test FSM -----
    localparam [2:0] S_WAIT=0, S_W_ISSUE=1, S_W_WAIT=2,
                     S_R_ISSUE=3, S_R_WAIT=4, S_DONE=5;
    reg [2:0]  st = S_WAIT;
    reg [9:0] word_idx;       // 0..1023 (1024 words = 4 rows in bank 0)
    reg [1:0] byte_idx;       // 0..3 byte lane within the current word
    reg       pass_b;         // 0 = pass A, 1 = pass B
    reg       fail;
    reg [31:0] expected_word;

    // 4 distinct per-lane XOR masks -> 4 distinct bytes per word, varying
    // across words; pass B inverts the index (~) for a second, independent pass.
    wire [7:0] widx8 = word_idx[7:0];
    wire [7:0] b0 = (pass_b ? ~widx8 : widx8) ^ 8'h5A;
    wire [7:0] b1 = (pass_b ? ~widx8 : widx8) ^ 8'hA5;
    wire [7:0] b2 = (pass_b ? ~widx8 : widx8) ^ 8'h3C;
    wire [7:0] b3 = (pass_b ? ~widx8 : widx8) ^ 8'hC3;
    wire [31:0] pat = {b3, b2, b1, b0};

    // Command pulses are combinational on state: each ISSUE state lasts
    // exactly 1 cycle and is only entered when busy=0, so wr/rd are clean
    // 1-cycle pulses (the controller latches addr/din on that edge).
    assign wr   = (st == S_W_ISSUE);
    assign rd   = (st == S_R_ISSUE);
    assign din  = pat[byte_idx*8 +: 8];
    assign addr = {11'b0, word_idx, byte_idx};   // byte address (lane = byte_idx)

    always @(posedge clk_sys) begin
        case (st)
            S_WAIT: if (lock && !busy) begin
                word_idx <= 10'd0; byte_idx <= 2'd0; pass_b <= 1'b0;
                fail <= 1'b0; st <= S_W_ISSUE;
            end
            S_W_ISSUE: st <= S_W_WAIT;                       // 1-cycle wr pulse
            S_W_WAIT: if (!busy) begin
                if (byte_idx == 2'd3) begin
                    if (word_idx == 10'd1023) begin
                        word_idx <= 10'd0; byte_idx <= 2'd0;
                        st <= S_R_ISSUE;                      // write phase done -> read
                    end else begin
                        word_idx <= word_idx + 10'd1; byte_idx <= 2'd0;
                        st <= S_W_ISSUE;                      // next word
                    end
                end else begin
                    byte_idx <= byte_idx + 2'd1; st <= S_W_ISSUE; // next byte
                end
            end
            S_R_ISSUE: begin expected_word <= pat; st <= S_R_WAIT; end // 1-cycle rd
            S_R_WAIT: if (data_ready) begin
                if (dout32 !== expected_word) fail <= 1'b1;
                if (word_idx == 10'd1023) begin
                    if (!pass_b) begin
                        pass_b <= 1'b1; word_idx <= 10'd0; byte_idx <= 2'd0;
                        st <= S_W_ISSUE;                      // pass B: rewrite all
                    end else st <= S_DONE;
                end else begin
                    word_idx <= word_idx + 10'd1; st <= S_R_ISSUE; // next read
                end
            end
            S_DONE: st <= S_DONE;                             // hold result
        endcase
    end

    // LED: slow blink = pass, steady = fail (unambiguous either polarity).
    reg [24:0] ctr = 25'd0;
    always @(posedge clk_sys) ctr <= ctr + 25'd1;
    assign led = fail ? 1'b1 : ctr[24];
endmodule