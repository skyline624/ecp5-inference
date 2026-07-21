`timescale 1ns/1ps
// =============================================================================
// tb_sdram_shadow.v -- validate the Colorlight i9 shadow-word WRITE path.
//
// The sdram_chip model is wired as on the i9 PCB: SDRAM_DQM=4'b0000, CKE=1,
// nCS=0 hardwired (no FPGA pin). The controller's SDRAM_DQM/CKE/nCS outputs
// are left unconnected. With DQM=0 the chip writes all 4 byte lanes from DQ
// on every write, so the controller MUST drive the full correct 32-bit word
// on DQ -- provided by the shadow-word accumulation in sdram.v.
//
// We write 4 contiguous bytes per word, then peek the chip's mem[] array
// directly (idx = addr[22:2] = {bank,row,col}) to check the stored 32-bit
// word. Peeking avoids the model's read-data timing (which is approximate)
// and directly validates the write path the shadow-word fixes. We also verify
// a second word doesn't clobber the first (shadow resets on word change).
// Self-checking; prints ALL PASS / TOTAL FAILURES.
// =============================================================================
module tb_sdram_shadow;
    reg clk_sys = 1'b0;
    always #20 clk_sys = ~clk_sys;       // 25 MHz, 40ns period
    wire clk_sdram = ~clk_sys;           // 180-degree SDRAM clock

    reg resetn = 1'b0;
    reg rd = 1'b0, wr = 1'b0, refresh = 1'b0;
    reg  [22:0] addr = 23'b0;
    reg  [7:0]  din = 8'b0;
    wire [7:0]  dout;
    wire [31:0] dout32;
    wire        data_ready, busy;

    wire [31:0] SDRAM_DQ;
    wire [10:0] SDRAM_A;
    wire [1:0]  SDRAM_BA;
    wire        SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE, SDRAM_CLK;

    sdram #(.FREQ(25_000_000)) u_ctrl (
        .clk(clk_sys), .clk_sdram(clk_sdram), .resetn(resetn),
        .rd(rd), .wr(wr), .refresh(refresh),
        .addr(addr), .din(din), .dout(dout), .dout32(dout32),
        .data_ready(data_ready), .busy(busy),
        .SDRAM_DQ(SDRAM_DQ), .SDRAM_A(SDRAM_A), .SDRAM_BA(SDRAM_BA),
        .SDRAM_nCS(), .SDRAM_nWE(SDRAM_nWE),
        .SDRAM_nRAS(SDRAM_nRAS), .SDRAM_nCAS(SDRAM_nCAS),
        .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(), .SDRAM_DQM()
    );

    sdram_chip u_chip (
        .SDRAM_CLK(SDRAM_CLK), .SDRAM_CKE(1'b1), .SDRAM_nCS(1'b0),
        .SDRAM_nRAS(SDRAM_nRAS), .SDRAM_nCAS(SDRAM_nCAS), .SDRAM_nWE(SDRAM_nWE),
        .SDRAM_BA(SDRAM_BA), .SDRAM_A(SDRAM_A), .SDRAM_DQM(4'b0000),
        .SDRAM_DQ(SDRAM_DQ)
    );

    // write one byte: pulse wr, wait for op to start then finish
    task do_write(input [22:0] a, input [7:0] d);
        begin
            @(negedge clk_sys); addr = a; din = d; wr = 1'b1;
            @(negedge clk_sys); wr = 1'b0;
            wait(busy == 1'b1);
            wait(busy == 1'b0);
            @(negedge clk_sys);
        end
    endtask

    integer errors = 0;
    // peek the chip mem at word index addr[22:2] and compare
    task peek_check(input [22:0] a, input [31:0] exp, input [127:0] msg);
        reg [31:0] got;
        begin
            @(negedge clk_sys);
            got = u_chip.mem[a[22:2]];
            if (got !== exp) begin
                errors = errors + 1;
                $display("FAIL %0s: got %h exp %h", msg, got, exp);
            end else
                $display("PASS %0s: %h", msg, got);
        end
    endtask

    initial begin
        resetn = 1'b0;
        repeat(10) @(negedge clk_sys);
        resetn = 1'b1;
        wait(busy == 1'b0);          // controller 200us init + config done
        repeat(5) @(negedge clk_sys);

        // Word 0 @ 0x000000: bytes 0x11,0x22,0x33,0x44 -> 0x44332211
        do_write(23'h000000, 8'h11);
        do_write(23'h000001, 8'h22);
        do_write(23'h000002, 8'h33);
        do_write(23'h000003, 8'h44);
        peek_check(23'h000000, 32'h44332211, "word0");

        // Word 1 @ 0x000004: 0xAA,0xBB,0xCC,0xDD -> 0xDDCCBBAA
        do_write(23'h000004, 8'hAA);
        do_write(23'h000005, 8'hBB);
        do_write(23'h000006, 8'hCC);
        do_write(23'h000007, 8'hDD);
        peek_check(23'h000004, 32'hDDCCBBAA, "word1");

        // Word 0 must still be intact (shadow reset on word change, not a clobber)
        peek_check(23'h000000, 32'h44332211, "word0 intact");

        if (errors == 0) $display("ALL PASS");
        else $display("TOTAL FAILURES: %0d", errors);
        $finish;
    end

    initial begin
        #5000000;                     // 5ms safety timeout
        $display("TIMEOUT");
        $finish;
    end
endmodule