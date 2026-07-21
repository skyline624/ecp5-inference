// =============================================================================
// pll_test.v -- Gate 2a: validate the ECP5 PLL wrapper (ecp5_pll) on silicon.
// The 25 MHz crystal (P3) feeds the PLL; clk_sys drives a counter; LED D2
// blinks ONLY while the PLL reports lock. If the PLL config is rejected or
// fails to lock, the LED stays dark -> the test fails visibly.
//
// Build:  .\build.ps1 -Top pll_test -Flash
// =============================================================================
module pll_test (
    input  wire clk25,   // 25 MHz on P3
    output wire led      // LED D2 on L2
);
    wire clk_sys;
    wire clk_sdram;      // unused here, exercised by sdram_test
    wire lock;

    ecp5_pll u_pll (
        .clkin   (clk25),
        .reset   (1'b0),
        .clkout  (clk_sys),
        .clkoutp (clk_sdram),
        .lock    (lock)
    );

    // Counter now runs on clk_sys (the PLL CLKOP output) to validate that the
    // PLL output clock actually reaches the fabric and clocks logic — not just
    // that lock=1. lock was already confirmed (v4: counter on clk25 blinked).
    // clk_sys = 25 MHz = clk25, so the blink rate is the same (~1.5 Hz); the
    // test is: blink => clk_sys is alive, steady => clk_sys is dead despite
    // lock. Both lock AND clk_sys good => Gate 2a fully green.
    reg [24:0] ctr = 25'd0;
    always @(posedge clk_sys) ctr <= ctr + 25'd1;

    assign led = lock ? ctr[24] : 1'b1;
endmodule