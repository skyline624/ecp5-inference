// =============================================================================
// matmul_test.v -- Gate 3 silicon test: 1 matmul bit-exact vs Python ref.
//
// Computes y[m] = sum_k W[m][k] * x[k]  (int8 x int8 -> int32) for m=0..7,
// K=8, using the mac18 DSP brick. W, x and the expected y (computed offline
// in Python, ecp5/sim/gen_matmul_vec.py) are hardcoded on-chip via the
// included initial block (matmul_data.vh). After all 8 rows, the FSM latches
// each result and compares to expected[m]; any mismatch sets `fail`.
//
// LED D2:  slow blink (~1.5 Hz) = PASS (all 8 rows match the Python ref)
//          steady              = FAIL (at least one mismatch)
//
// Pipeline (mac18 = 3-cycle): FEED K cycles (load=1 at k=0 to start the row,
// load=0 to accumulate), then DRAIN ~6 cycles (ce=1, a=b=0, load=0 so the
// accumulator holds), then CHECK latches result[31:0] vs expected[m].
//
// Validated in sim (iverilog, tb_matmul.v) before silicon (G1).
// Build:  .\build.ps1 -Top matmul_test -Extra ../src/ecp5_ecp5pll.v,../src/mac18.v -Flash
// =============================================================================
module matmul_test (
    input  wire clk25,
    output wire led
);
    wire clk_sys, clk_sdram, lock;
    ecp5_pll u_pll (
        .clkin(clk25), .reset(1'b0),
        .clkout(clk_sys), .clkoutp(clk_sdram), .lock(lock)
    );

    // on-chip test data (int8 weights/activations, int32 expected outputs)
    reg signed [7:0]  W [0:63];          // W[8][8] flattened: W[m*8+k]
    reg signed [7:0]  x [0:7];
    reg signed [31:0] expected [0:7];
    `include "../build/matmul_data.vh"   // initial block (auto-generated)

    localparam integer M = 8, K = 8;
    localparam [2:0] S_IDLE=3'd0, S_FEED=3'd1, S_DRAIN=3'd2,
                     S_CHECK=3'd3, S_DONE=3'd4;
    reg [2:0] st    = S_IDLE;
    reg [3:0] m     = 4'd0;
    reg [3:0] k     = 4'd0;
    reg [3:0] drain = 4'd0;
    reg       fail  = 1'b0;

    // combinational mac18 drive: feed W/x during FEED, zeros during DRAIN.
    wire signed [7:0] W_mk = W[m*K + k];
    wire signed [7:0] x_k  = x[k];
    wire signed [17:0] mac_a = (st == S_FEED) ? {{10{W_mk[7]}}, W_mk} : 18'sd0;
    wire signed [17:0] mac_b = (st == S_FEED) ? {{10{x_k[7]}},  x_k}  : 18'sd0;
    wire        mac_ce   = (st == S_FEED) || (st == S_DRAIN);
    wire        mac_load = (st == S_FEED) && (k == 4'd0);

    wire signed [53:0] result;
    mac18 u_mac (
        .clk(clk_sys), .rst(~lock), .ce(mac_ce), .load(mac_load),
        .a(mac_a), .b(mac_b), .result(result)
    );

    always @(posedge clk_sys) begin
        case (st)
            S_IDLE: if (lock) begin
                m <= 4'd0; k <= 4'd0; drain <= 4'd0; fail <= 1'b0; st <= S_FEED;
            end
            S_FEED: begin
                if (k == K-1) begin st <= S_DRAIN; drain <= 4'd0; end
                else k <= k + 4'd1;
            end
            S_DRAIN: begin
                if (drain == 4'd6) st <= S_CHECK; else drain <= drain + 4'd1;
            end
            S_CHECK: begin
                if (result[31:0] != expected[m]) fail <= 1'b1;
                if (m == M-1) st <= S_DONE;
                else begin m <= m + 4'd1; k <= 4'd0; st <= S_FEED; end
            end
            S_DONE: st <= S_DONE;
        endcase
    end

    reg [24:0] ctr = 25'd0;
    always @(posedge clk_sys) ctr <= ctr + 25'd1;
    assign led = fail ? 1'b1 : ctr[24];
endmodule