// =============================================================================
// mac18.v -- ECP5 MAC wrapper (replaces the Gowin MULTALU18X18 version).
//
// Signed 18x18 -> 54-bit accumulator, 3-cycle pipeline, bit-exact behavioral
// copy of the Gowin mac18 (see ecp5/sim/models/MULTALU18X18.v -- the proven
// reference model). Ports are IDENTICAL to src/mac18.v so top.v is unchanged.
//
// Semantics (matches src/mac18.v + MULTALU18X18 model exactly):
//   load=1 -> ACC <= A*B            (charge / overwrite)
//   load=0 -> ACC <= ACC + A*B      (accumulate)
//   accload = ~load  (polarity inverted vs the primitive's ACCLOAD)
//   3-cycle latency: AREG/BREG -> PIPE_REG -> OUT_REG
//   ce=0 -> all pipeline regs HOLD (frozen, no spurious accumulation)
//   rst  -> sync reset to 0 (active high)
//
// Why pure RTL instead of instantiating an ECP5 DSP primitive:
//   The open toolchain (yosys / trellis / nextpnr) exposes MULT18X18D (18x18
//   multiply, no accumulator) and ALU54B (54-bit ALU+acc). There is NO
//   "ALU54A" in cells_bb.v (that name is Lattice Diamond's). Modeling ALU54B's
//   exact pipeline behaviorally for sim is error-prone. Pure RTL that mirrors
//   the proven Gowin pipeline is bit-exact BY CONSTRUCTION; yosys synth_ecp5
//   infers MULT18X18D for the a_r*b_r product (and ALU54B / fabric for the
//   accumulator). At 25 MHz this fits with large margin (the node is small).
//   Correctness first (Gate 3 = bit-exact); DSP packing can be tuned later.
//
// Validated bit-exact vs the Gowin mac18 in sim (iverilog, 4000 randomized
// LFSR samples incl. ce=0 and load transitions): see ecp5/sim/tb_mac18.v.
// =============================================================================
module mac18 (
    input  wire                clk,
    input  wire                rst,
    input  wire                ce,
    input  wire                load,
    input  wire signed [17:0]  a,
    input  wire signed [17:0]  b,
    output wire signed [53:0]  result
);
    reg signed [17:0]  a_r, b_r;     // AREG / BREG (stage 1)
    reg signed [35:0]  prod_r;       // PIPE_REG: registered product (stage 2)
    reg                acl0, acl1;   // ACCLOAD_REG0 / REG1: accload = ~load, 2-stage
    reg signed [53:0]  acc;          // OUT_REG: accumulator (stage 3)

    always @(posedge clk) begin
        if (rst) begin
            a_r    <= 18'sd0;
            b_r    <= 18'sd0;
            prod_r <= 36'sd0;
            acl0   <= 1'b0;
            acl1   <= 1'b0;
            acc    <= 54'sd0;
        end else if (ce) begin
            a_r    <= a;
            b_r    <= b;
            prod_r <= a_r * b_r;                 // 36-bit signed product
            acl0   <= ~load;                     // accload polarity inverted
            acl1   <= acl0;
            acc    <= (acl1 ? acc : 54'sd0)      // 1=accumulate, 0=load(overwrite)
                      + {{18{prod_r[35]}}, prod_r};
        end
    end

    assign result = acc;
endmodule