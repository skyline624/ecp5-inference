// =============================================================================
// ecp5_ecp5pll.v -- ECP5 PLL wrapper (replaces gowin_rpll.v).
// Colorlight i9 v7.2 : 25 MHz crystal on pin P3 -> clk_sys 25 MHz (0 deg) +
// clk_sdram 25 MHz (180 deg, for SDRAM capture) + lock.
//
// Interface matches gowin_rpll.v / Gowin_rPLL (clkout, clkoutp, lock, reset,
// clkin) so the SDRAM test top and (later) the ported top.v instantiate it
// the same way the Gowin design did.
//
// EHXPLLL config (port/param names from yosys share/yosys/ecp5/cells_bb.v;
// feedback/CPHASE conventions per litex lattice_ecp5.py):
//   FEEDBK_PATH="INT_OP"  -> INTERNAL feedback from CLKOP divider (no external
//                            CLKFB wire needed; "CLKOP" would expect an external
//                            feedback pin and the PLL would NOT lock).
//   f_VCO = f_CLKI * CLKFB_DIV * CLKOP_DIV / CLKI_DIV = 25 * 1 * 20 = 500 MHz
//   (mid 400-800 MHz band).
//   CLKOP = VCO/20 = 25 MHz @0 deg   (CPHASE = div-1 = 19, litex 0-deg convention)
//   CLKOS = VCO/20 = 25 MHz @180 deg (CPHASE = div-1 + div/2 = 29)
//
// The 180 deg phase correctness is validated indirectly by the SDRAM self-test
// (Gate 2b): if the phase is wrong, SDRAM reads return garbage and the match
// LED stays off. PLL lock + clock validated by pll_test (Gate 2a).
// =============================================================================
module ecp5_pll (
    input  wire clkin,    // 25 MHz from pin P3
    input  wire reset,
    output wire clkout,   // clk_sys  = 25 MHz, 0 deg
    output wire clkoutp,  // clk_sdram = 25 MHz, 180 deg
    output wire lock
);
    // Power-on reset for the PLL: RST is active-high async (Lattice
    // FPGA-TN-02200). The PLL must be reset at power-up, then needs ~16 ms
    // to lock after RST deassertion. Tying RST=0 permanently leaves the PLL
    // in an undefined state -> no lock. Assert RST for the first ~2.6 ms
    // (2^16 clkin cycles), then release. lock rises ~16 ms later.
    reg [16:0] por_cnt = 17'd0;
    wire por_done = por_cnt[16];
    always @(posedge clkin) if (!por_done) por_cnt <= por_cnt + 17'd1;
    wire pll_rst = reset | ~por_done;

    // Synthesis attributes (REQUIRED for analog lock — copied verbatim from
    // litex soc/cores/clock/lattice_ecp5.py do_finalize). ICP_CURRENT and
    // LPF_RESISTOR configure the PLL loop filter / charge pump; without them
    // the loop filter is at default/zero and the PLL NEVER locks, regardless
    // of the divider/phase settings. FREQUENCY_PIN_CLKI tells trellis the
    // input frequency so it picks the right VCO band. This was the missing
    // piece in the v1-v3 wrappers (dividers/CPHASE were already correct).
    (* FREQUENCY_PIN_CLKI     = "25.0",
       ICP_CURRENT            = "6",
       LPF_RESISTOR           = "16",
       MFG_ENABLE_FILTEROPAMP= "1",
       MFG_GMCREF_SEL         = "2",
       FREQUENCY_PIN_CLKOP    = "25.0",
       FREQUENCY_PIN_CLKOS    = "25.0" *)
    EHXPLLL #(
        .CLKI_DIV         (1),
        .CLKFB_DIV        (1),        // VCO = 25 * 1 * 20 = 500 MHz
        .CLKOP_DIV        (20),       // 500/20 = 25 MHz
        .CLKOS_DIV        (20),       // 500/20 = 25 MHz
        .CLKOS2_DIV       (8),
        .CLKOS3_DIV       (8),
        .CLKOP_ENABLE     ("ENABLED"),
        .CLKOS_ENABLE     ("ENABLED"),
        .CLKOS2_ENABLE    ("DISABLED"),
        .CLKOS3_ENABLE    ("DISABLED"),
        .CLKOP_CPHASE     (19),       // 0 deg   (litex: div-1)
        .CLKOS_CPHASE     (29),       // 180 deg (litex: phase>>3 + div-1)
        .CLKOS2_CPHASE    (0),
        .CLKOS3_CPHASE    (0),
        .CLKOP_FPHASE     (0),
        .CLKOS_FPHASE     (0),
        .FEEDBK_PATH      ("INT_OP"), // litex: INT_O{n_to_l[clkfb]} = INT_OP
        .CLKOP_TRIM_POL   ("RISING"),
        .CLKOP_TRIM_DELAY (0),
        .CLKOS_TRIM_POL   ("RISING"),
        .CLKOS_TRIM_DELAY (0),
        .OUTDIVIDER_MUXA  ("DIVA"),
        .OUTDIVIDER_MUXB  ("DIVB"),
        .OUTDIVIDER_MUXC  ("DIVC"),
        .OUTDIVIDER_MUXD  ("DIVD"),
        .STDBY_ENABLE     ("DISABLED"),
        .REFIN_RESET      ("DISABLED"),
        .INT_LOCK_STICKY  ("ENABLED"),
        .DPHASE_SOURCE    ("DISABLED"),
        .PLLRST_ENA       ("DISABLED")
    ) u_pll (
        .CLKI         (clkin),
        // CLKFB left OPEN for INT_OP feedback (litex does not connect it).
        .PHASESEL1    (1'b0),
        .PHASESEL0    (1'b0),
        .PHASEDIR     (1'b0),
        .PHASESTEP    (1'b0),
        .PHASELOADREG (1'b0),
        .STDBY        (1'b0),
        .PLLWAKESYNC  (1'b0),
        .RST          (pll_rst),
        .ENCLKOP      (1'b1),
        .ENCLKOS      (1'b1),
        .ENCLKOS2     (1'b0),
        .ENCLKOS3     (1'b0),
        .CLKOP        (clkout),
        .CLKOS        (clkoutp),
        .CLKOS2       (),
        .CLKOS3       (),
        .LOCK         (lock),
        .INTLOCK      (),
        .REFCLK       (),
        .CLKINTFB     ()
    );
endmodule