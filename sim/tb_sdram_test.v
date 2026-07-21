`timescale 1ns/1ps
// =============================================================================
// tb_sdram_test.v -- sim wrapper for board/sdram_test.v (M2 silicon pre-check).
//
// Instantiates sdram_test (DUT, drives the SDRAM pins) + sdram_chip (wired as
// on the i9: DQM=0, CKE=1, nCS=0 hardwired). Drives clk25, lets the self-test
// FSM run, then peeks the internal `fail` flag once the FSM reaches S_DONE.
// Prints PASS / FAIL. This is a G1 pre-check before flashing the board.
// =============================================================================
module tb_sdram_test;
    reg clk25 = 1'b0;
    always #20 clk25 = ~clk25;            // 25 MHz

    wire        led;
    wire [31:0] SDRAM_DQ;
    wire [10:0] SDRAM_A;
    wire [1:0]  SDRAM_BA;
    wire        SDRAM_nWE, SDRAM_nRAS, SDRAM_nCAS, SDRAM_CLK;

    sdram_test u_dut (
        .clk25(clk25), .led(led),
        .SDRAM_DQ(SDRAM_DQ), .SDRAM_A(SDRAM_A), .SDRAM_BA(SDRAM_BA),
        .SDRAM_nWE(SDRAM_nWE), .SDRAM_nRAS(SDRAM_nRAS),
        .SDRAM_nCAS(SDRAM_nCAS), .SDRAM_CLK(SDRAM_CLK)
    );

    sdram_chip u_chip (
        .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(1'b1), .SDRAM_nCS(1'b0),
        .SDRAM_nRAS(SDRAM_nRAS), .SDRAM_nCAS(SDRAM_nCAS), .SDRAM_nWE(SDRAM_nWE),
        .SDRAM_BA(SDRAM_BA), .SDRAM_A(SDRAM_A), .SDRAM_DQM(4'b0000),
        .SDRAM_DQ(SDRAM_DQ)
    );

    initial begin
        // wait long enough for 200us init + ~2ms test (2 passes x 1024 words)
        #3500000;                          // 3.5 ms
        if (u_dut.st === 3'd5) begin        // S_DONE
            if (u_dut.fail === 1'b0) $display("PASS: sdram_test self-test OK");
            else $display("FAIL: sdram_test mismatch (fail=1)");
        end else begin
            $display("FAIL: test did not finish (st=%0d)", u_dut.st);
        end
        $finish;
    end

    initial begin
        #5000000;                           // 5 ms safety timeout
        $display("TIMEOUT (st=%0d fail=%0d)", u_dut.st, u_dut.fail);
        $finish;
    end
endmodule