// =============================================================================
// tb_matmul.v -- sim driver for matmul_test (Gate 3, G1: sim vert avant P&R).
// Generates clk25, lets the matmul FSM run to completion, then reports whether
// the hardware result matches the Python-computed expected (via the FSM's
// `fail` flag, observed hierarchically).
//
// Run:  cd ecp5/sim
//   iverilog -o matmul.vvp tb_matmul.v ../board/matmul_test.v ../src/mac18.v \
//                                  models/ECP5_PLL_sim.v
//   vvp matmul.vvp
// =============================================================================
`timescale 1ns/1ps
module tb_matmul;
    reg clk25 = 1'b0;
    wire led;
    matmul_test dut(.clk25(clk25), .led(led));

    always #5 clk25 = ~clk25;   // 10 ns period

    integer i;
    initial begin
        for (i = 0; i < 4000; i = i + 1) @(posedge clk25);
        if (dut.st !== 3'd4)          // S_DONE
            $display("SIM FAIL: FSM did not reach DONE (st=%0d)", dut.st);
        else if (dut.fail)
            $display("SIM FAIL: matmul mismatch vs Python reference");
        else
            $display("SIM PASS: matmul bit-exact vs Python reference (8 rows)");
        $finish;
    end
endmodule