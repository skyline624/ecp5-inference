// =============================================================================
// blink.v -- Gate 1 (Hello-ECP5) for the Colorlight i9 v7.2 (LFE5U-45F).
// Minimal LED blink to validate the open toolchain (yosys + nextpnr-ecp5 +
// ecppack + openFPGALoader) end-to-end on real silicon.
//
// 25 MHz oscillator on pin P3 -> 25-bit counter -> LED D2 on pin L2.
// 25e6 / 2^25 ~= 0.74 Hz toggle -> ~1.5 Hz visible blink.
// =============================================================================
module blink (
    input  wire clk25,   // 25 MHz from pin P3
    output wire led      // LED D2 on pin L2 (shared with SODIMM-41)
);
    reg [24:0] ctr = 25'd0;
    always @(posedge clk25) ctr <= ctr + 25'd1;
    assign led = ctr[24];
endmodule