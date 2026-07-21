// =============================================================================
// sdram_sink.v -- mock SDRAM controller for the flash_boot testbench.
//
// Mimics the backpressure behaviour of src/sdram.v: each write occupies the
// controller for ~5 cycles during which `sdram_busy` is asserted.  flash_boot
// waits for !sdram_busy before pulsing mem_wr and freezes SCK (F_WR stall)
// in the meantime -- so a mismatch here would surface as a protocol violation
// rather than silent corruption.
//
// Captured bytes land in `ram[mem_addr]` and are compared against the flash
// mock contents at end-of-sim.  ram is sized for the full 0x060040 model
// image; smaller FLASH_LEN overrides simply leave the tail unused.
// =============================================================================
module sdram_sink (
    input  wire        clk,
    input  wire [22:0] mem_addr,
    input  wire [7:0]  mem_din,
    input  wire        mem_wr,
    output reg         sdram_busy
);

    reg [7:0] ram [0:393280];
    reg [2:0] busy_cnt;
    integer   i;

    initial begin
        for (i = 0; i < 393280; i = i + 1) ram[i] = 8'hxx;
        sdram_busy = 1'b0;
        busy_cnt   = 3'd0;
    end

    always @(posedge clk) begin
        if (mem_wr) begin
            ram[mem_addr] <= mem_din;
            busy_cnt      <= 3'd5;
            sdram_busy    <= 1'b1;
        end else if (busy_cnt != 3'd0) begin
            busy_cnt <= busy_cnt - 3'd1;
            if (busy_cnt == 3'd1) sdram_busy <= 1'b0;
        end
    end

endmodule