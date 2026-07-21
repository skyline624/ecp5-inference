// =============================================================================
// tb_mac18.v -- self-checking sim for the mac18 brick (Gate 3, sub-brick DSP).
//
// Drives the DUT `mac18` with a deterministic LFSR stimulus (signed a/b, ce
// mostly 1 but sometimes 0, load toggling) and logs `result` (54-bit) every
// cycle to mac18_log.txt.
//
// Non-regression (G4): compile TWICE with the SAME tb but different mac18:
//   Run ECP5 : iverilog -o ecp5.vvp  tb_mac18.v ../src/mac18.v
//   Run Gowin: iverilog -o gowin.vvp tb_mac18.v ../../src/mac18.v models/MULTALU18X18.v
// then diff the two mac18_log.txt -> empty diff = bit-exact (ECP5 RTL matches
// the proven Gowin MULTALU18X18 pipeline). No make / no cocotb needed.
//
// Run:  cd ecp5/sim ; iverilog -o ecp5.vvp tb_mac18.v ../src/mac18.v ; vvp ecp5.vvp
// =============================================================================
`timescale 1ns/1ps
module tb_mac18;
    reg        clk = 1'b0;
    reg        rst = 1'b1;
    reg        ce, load;
    reg signed [17:0] a, b;
    wire signed [53:0] result;
    integer i;

    // deterministic LFSR state (same seed -> identical stimulus in both runs)
    reg [17:0] lfsr_a = 18'h1ACE;
    reg [17:0] lfsr_b = 18'h5EED;
    reg [7:0]  lfsr_c = 8'h7B;

    function [17:0] lfsr18(input [17:0] s);
        reg bitx;
        begin
            bitx = s[17] ^ s[5] ^ s[2] ^ s[0];
            lfsr18 = {s[16:0], bitx};
        end
    endfunction
    function [7:0] lfsr8(input [7:0] s);
        reg bitx;
        begin
            bitx = s[7] ^ s[5] ^ s[4] ^ s[3];
            lfsr8 = {s[6:0], bitx};
        end
    endfunction

    mac18 dut (
        .clk(clk), .rst(rst), .ce(ce), .load(load),
        .a(a), .b(b), .result(result)
    );

    always #5 clk = ~clk;   // 10 ns period (100 MHz sim clock)

    integer f;
    initial begin
        f = $fopen("mac18_log.txt", "w");
        ce = 1'b0; load = 1'b0; a = 18'sd0; b = 18'sd0;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);
        for (i = 0; i < 4000; i = i + 1) begin
            @(negedge clk);
            lfsr_a = lfsr18(lfsr_a);
            lfsr_b = lfsr18(lfsr_b);
            lfsr_c = lfsr8(lfsr_c);
            a   = lfsr_a;
            b   = lfsr_b;
            ce  = lfsr_c[7] | lfsr_c[6];   // ~75% ones (some ce=0 drain cycles)
            load = lfsr_c[5];
            @(posedge clk);
            $fwrite(f, "%0d %h\n", i, result);
        end
        $fclose(f);
        $display("DONE %0d samples written to mac18_log.txt", i);
        $finish;
    end
endmodule