// =============================================================================
// ECP5_PLL_sim.v -- behavioral sim stub for the ECP5 PLL wrapper (step 3).
//
// For now the sim uses models/Gowin_rPLL.v (a behavioral pass-through) because
// ecp5/src/top.v still instantiates `Gowin_rPLL`. When step 3 swaps top.v to
// instantiate the real ECP5 PLL wrapper (ecp5_ecp5pll.v, EHXPLLL), point the
// sim at this stub instead -- same interface, behavioral (no timing).
//
// Interface matches Gowin_rPLL so the swap is 1 line in the Makefile:
//   VERILOG_SOURCES += models/ECP5_PLL_sim.v   (instead of models/Gowin_rPLL.v)
// =============================================================================
`timescale 1ns/1ps
module Gowin_rPLL (
    output clkout,
    output clkoutp,
    output reg lock,
    input  reset,
    input  clkin
);
    // clk_sys = clkin (25 MHz), aligned.
    assign clkout  = clkin;
    // clk_sdram = clkin shifted ~180 deg (half period) for the SDRAM capture.
    // A simple inverted clock models the 180 deg phase at behavioral level.
    assign clkoutp = ~clkin;

    initial lock = 1'b0;
    always @(posedge clkin) begin
        if (!reset) lock <= 1'b1;
        else        lock <= 1'b0;
    end
endmodule

// =============================================================================
// ecp5_pll -- behavioral sim stub matching ecp5/src/ecp5_ecp5pll.v interface.
// Use in iverilog sim INSTEAD of the real EHXPLLL wrapper (no EHXPLLL model
// needed). clkout=clkin, clkoutp=~clkin (180 deg by inversion), lock rises 1
// cycle after reset deassertion. For synth, use ecp5/src/ecp5_ecp5pll.v.
// =============================================================================
module ecp5_pll (
    input  wire clkin,
    input  wire reset,
    output wire clkout,
    output wire clkoutp,
    output reg  lock
);
    assign clkout  = clkin;
    assign clkoutp = ~clkin;
    initial lock = 1'b0;
    always @(posedge clkin) lock <= reset ? 1'b0 : 1'b1;
endmodule