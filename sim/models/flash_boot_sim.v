// =============================================================================
// flash_boot_sim.v -- behavioral mock of flash_boot for cocotb/icarus sim.
//
// Mocks the autonomous SPI flash -> SDRAM boot loader so the sim can exercise the
// REAL silicium path (FLASH_BOOT) without a W25Q64 model. In sim the SDRAM is
// already preloaded via $readmemh("sdram_preload.hex") (SDRAM_PRELOAD define), so
// this mock SKIPS the DMA : it just pulses `done` (boot_done) a few cycles after
// `start`, leaving the SDRAM image intact. The cache-fill FSM in node_top then
// wakes on boot_done and DMA's the FFN weights SDRAM -> BRAM (the silicium path,
// cache ACTIF). Shifts come from the test via drive_shifts (gen_seq.sw_* ports),
// so shifts_valid stays 0 here (matching step-1 SHIFT_LOAD=0 behavior).
//
// Interface matches src/flash_boot.v exactly (drop-in : same module name + ports).
// =============================================================================
`timescale 1ns/1ps
module flash_boot #(
    parameter [23:0] FLASH_BASE = 24'h0,
    parameter [23:0] FLASH_LEN  = 24'h0,
    parameter [23:0] SHIFT_BASE = 24'h0,
    parameter        SHIFT_LOAD = 0,
    parameter [9:0]  SPI_HALF   = 10'd1
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        start,
    output reg         flash_cs_n,
    output reg         flash_sck,
    output reg         flash_mosi,
    input  wire        flash_miso,
    output reg  [22:0] mem_addr,
    output reg  [7:0]  mem_din,
    output reg         mem_wr,
    input  wire        sdram_busy,
    output wire        drained,
    output reg         done,
    output reg  [15:0] dbg_blk,
    output wire        dbg_ready,
    output reg         dbg_fail,
    output wire        dbg_wpend,
    output reg  [47*8-1:0] shifts_out,
    output reg         shifts_valid
);
    reg [3:0] delay;
    reg       started;
    initial begin
        flash_cs_n  = 1'b1;     // flash idle (deselected)
        flash_sck   = 1'b0;
        flash_mosi  = 1'b0;
        mem_addr    = 23'd0;    // stable (eff_addr=boot_addr during booting)
        mem_din     = 8'd0;
        mem_wr      = 1'b0;     // NO DMA : SDRAM already preloaded
        done        = 1'b0;
        dbg_blk     = 16'd0;
        dbg_fail    = 1'b0;
        shifts_out  = {47*8{1'b0}};
        shifts_valid= 1'b0;
        delay       = 4'd0;
        started     = 1'b0;
    end
    assign drained   = ~mem_wr;   // no write in flight -> refresh may slip through
    assign dbg_ready = done;
    assign dbg_wpend = 1'b0;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            done    <= 1'b0;
            delay   <= 4'd0;
            started <= 1'b0;
        end else begin
            if (start) started <= 1'b1;       // latch the 1-cycle boot_start pulse
            if (started && !done) begin
                if (delay == 4'd10) done <= 1'b1;   // boot_done after ~10 cycles
                else                delay <= delay + 4'd1;
            end
        end
    end
endmodule