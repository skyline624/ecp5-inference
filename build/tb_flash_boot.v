// =============================================================================
// tb_flash_boot.v -- simulation harness for src/flash_boot.v
//
// Validates that the bytes written on mem_addr/mem_din/mem_wr are bit-exact
// the bytes read from the SPI flash mock, and that sdram_busy backpressure
// does not corrupt the stream.
//
// Three DUTs are exercised in parallel so a single vvp run covers the cases
// the task asks for:
//   DUT_SMALL  : SPI_HALF=12, FLASH_LEN=0x1000                 (slow/debug)
//   DUT_FULL   : SPI_HALF=1,  FLASH_LEN=0x10000, SHIFT_LOAD=0  (target speed)
//   DUT_SHIFT  : SPI_HALF=4,  FLASH_LEN=0x1000,  SHIFT_LOAD=1  (step 2 trailer)
//
// FLASH_LEN overrides: DUT_SMALL and DUT_SHIFT use 0x1000 (4 KB) for fast
// iteration; DUT_FULL uses 0x10000 (64 KB) -- enough to exercise the SPI_HALF=1
// target-speed path and the sdram_busy backpressure loop many times over,
// while keeping sim time tractable (~100 ms simulated).  The production
// 0x060040 image is identical per-byte; only the length changes.  See the
// header of gen_flash_test_hex.py for the flash image layout.
//
// Run:
//   iverilog -g2012 -o tb_flash_boot.vvp tb_flash_boot.v \
//       ../src/flash_boot.v w25q64_model.v sdram_sink.v
//   vvp tb_flash_boot.vvp
// =============================================================================
`timescale 1ns/1ps

module tb_flash_boot;

    // ----- clock & reset (25 MHz, 40 ns period) -----
    reg clk = 1'b0;
    reg rstn = 1'b0;
    always #20 clk = ~clk;

    // ===== DUT_SMALL : SPI_HALF=12, FLASH_LEN=0x1000 =====
    localparam SMALL_LEN = 24'h001000;
    localparam SMALL_HALF = 10'd12;

    reg         s_start;
    wire        s_cs_n, s_sck, s_mosi;
    wire        s_miso;
    wire [22:0] s_addr;
    wire [7:0]  s_din;
    wire        s_wr, s_drained, s_done, s_busy;
    wire [15:0] s_blk;
    wire        s_ready, s_fail, s_wpend;
    wire [47*8-1:0] s_shifts;
    wire        s_shifts_valid;

    flash_boot #(
        .FLASH_LEN(SMALL_LEN),
        .SHIFT_LOAD(0),
        .SPI_HALF(SMALL_HALF)
    ) dut_small (
        .clk(clk), .rstn(rstn), .start(s_start),
        .flash_cs_n(s_cs_n), .flash_sck(s_sck),
        .flash_mosi(s_mosi), .flash_miso(s_miso),
        .mem_addr(s_addr), .mem_din(s_din), .mem_wr(s_wr),
        .sdram_busy(s_busy), .drained(s_drained), .done(s_done),
        .dbg_blk(s_blk), .dbg_ready(s_ready),
        .dbg_fail(s_fail), .dbg_wpend(s_wpend),
        .shifts_out(s_shifts), .shifts_valid(s_shifts_valid)
    );

    w25q64_model s_flash (.flash_cs_n(s_cs_n), .flash_sck(s_sck),
                          .flash_mosi(s_mosi), .flash_miso(s_miso));

    sdram_sink s_sink (.clk(clk), .mem_addr(s_addr), .mem_din(s_din),
                       .mem_wr(s_wr), .sdram_busy(s_busy));

    // ===== DUT_FULL : SPI_HALF=1, FLASH_LEN=0x10000, SHIFT_LOAD=0 =====
    localparam FULL_LEN  = 24'h060040;
    localparam FULL_HALF = 10'd1;

    reg         f_start;
    wire        f_cs_n, f_sck, f_mosi;
    wire        f_miso;
    wire [22:0] f_addr;
    wire [7:0]  f_din;
    wire        f_wr, f_drained, f_done, f_busy;
    wire [15:0] f_blk;
    wire        f_ready, f_fail, f_wpend;
    wire [47*8-1:0] f_shifts;
    wire        f_shifts_valid;

    flash_boot #(
        .FLASH_LEN(FULL_LEN),
        .SHIFT_LOAD(0),
        .SPI_HALF(FULL_HALF)
    ) dut_full (
        .clk(clk), .rstn(rstn), .start(f_start),
        .flash_cs_n(f_cs_n), .flash_sck(f_sck),
        .flash_mosi(f_mosi), .flash_miso(f_miso),
        .mem_addr(f_addr), .mem_din(f_din), .mem_wr(f_wr),
        .sdram_busy(f_busy), .drained(f_drained), .done(f_done),
        .dbg_blk(f_blk), .dbg_ready(f_ready),
        .dbg_fail(f_fail), .dbg_wpend(f_wpend),
        .shifts_out(f_shifts), .shifts_valid(f_shifts_valid)
    );

    w25q64_model f_flash (.flash_cs_n(f_cs_n), .flash_sck(f_sck),
                          .flash_mosi(f_mosi), .flash_miso(f_miso));

    sdram_sink f_sink (.clk(clk), .mem_addr(f_addr), .mem_din(f_din),
                       .mem_wr(f_wr), .sdram_busy(f_busy));

    // ===== DUT_SHIFT : SPI_HALF=4, FLASH_LEN=0x1000, SHIFT_LOAD=1 =====
    localparam SH_LEN  = 24'h001000;
    localparam SH_HALF = 10'd4;

    reg         h_start;
    wire        h_cs_n, h_sck, h_mosi;
    wire        h_miso;
    wire [22:0] h_addr;
    wire [7:0]  h_din;
    wire        h_wr, h_drained, h_done, h_busy;
    wire [15:0] h_blk;
    wire        h_ready, h_fail, h_wpend;
    wire [47*8-1:0] h_shifts;
    wire        h_shifts_valid;

    flash_boot #(
        .FLASH_LEN(SH_LEN),
        .SHIFT_LOAD(1),
        .SPI_HALF(SH_HALF)
    ) dut_shift (
        .clk(clk), .rstn(rstn), .start(h_start),
        .flash_cs_n(h_cs_n), .flash_sck(h_sck),
        .flash_mosi(h_mosi), .flash_miso(h_miso),
        .mem_addr(h_addr), .mem_din(h_din), .mem_wr(h_wr),
        .sdram_busy(h_busy), .drained(h_drained), .done(h_done),
        .dbg_blk(h_blk), .dbg_ready(h_ready),
        .dbg_fail(h_fail), .dbg_wpend(h_wpend),
        .shifts_out(h_shifts), .shifts_valid(h_shifts_valid)
    );

    w25q64_model h_flash (.flash_cs_n(h_cs_n), .flash_sck(h_sck),
                          .flash_mosi(h_mosi), .flash_miso(h_miso));

    sdram_sink h_sink (.clk(clk), .mem_addr(h_addr), .mem_din(h_din),
                       .mem_wr(h_wr), .sdram_busy(h_busy));

    // ----- expected flash image (mirror of gen_flash_test_hex.py) -----
    function [7:0] expected_byte;
        input [31:0] abs_addr;
        begin
            if (abs_addr >= 32'h181000 && abs_addr < 32'h181000 + 47)
                expected_byte = (200 + (abs_addr - 32'h181000)) & 8'hFF;
            else if (abs_addr < 32'h180040)
                expected_byte = ((abs_addr - 32'h120000) * 7 + 13) & 8'hFF;
            else
                expected_byte = 8'h00;
        end
    endfunction

    // ----- timing measurement -----
    integer s_start_cycle, f_start_cycle, h_start_cycle;
    integer s_done_cycle,  f_done_cycle,  h_done_cycle;
    integer cyc;
    integer mism, k, shown;
    integer total_errors;
    reg     s_checked, f_checked, h_checked;

    // cycle counter
    always @(posedge clk) cyc <= cyc + 1;

    // ----- stimulus -----
    initial begin
        cyc = 0;
        s_start = 0; f_start = 0; h_start = 0;
        s_checked = 0; f_checked = 0; h_checked = 0;
        s_done_cycle = 0; f_done_cycle = 0; h_done_cycle = 0;
        total_errors = 0;
        // hold reset for 10 cycles
        #200 rstn = 1'b1;
        @(posedge clk);
        // pulse all three starts one cycle after reset release
        s_start = 1; f_start = 1; h_start = 1;
        s_start_cycle = cyc;
        f_start_cycle = cyc;
        h_start_cycle = cyc;
        @(posedge clk);
        s_start = 0; f_start = 0; h_start = 0;
    end

    // ----- per-DUT done capture + check -----
    always @(posedge clk) begin
        if (s_done && !s_checked) begin
            s_done_cycle = cyc;     // blocking so check_small sees it
            s_checked = 1;
            check_small;
        end
        if (f_done && !f_checked) begin
            f_done_cycle = cyc;
            f_checked = 1;
            check_full;
        end
        if (h_done && !h_checked) begin
            h_done_cycle = cyc;
            h_checked = 1;
            check_shift;
        end
        if (s_checked && f_checked && h_checked) begin
            $display("\n========================================");
            if (total_errors == 0) $display("SIM PASS -- all DUTs bit-exact");
            else                   $display("SIM FAIL -- %0d error classes", total_errors);
            $display("========================================");
            if (total_errors > 0) $error("flash_boot simulation failed");
            $finish;
        end
    end

    // ----- watchdog -----
    initial begin
        // SMALL: 4KB @ SPI_HALF=12   ~ 850k cyc ~ 34 ms
        // FULL : 393KB @ SPI_HALF=1  ~ 16M cyc  ~ 640 ms
        // SHIFT: 4KB+47B @ SPI_HALF=4 ~ 350k cyc ~ 14 ms
        #1_500_000_000;   // 1.5 s simulated (covers FULL at production size)
        $display("ERROR: simulation timeout (s_checked=%b f_checked=%b h_checked=%b)",
                 s_checked, f_checked, h_checked);
        $finish;
    end

    task check_small;
        begin
            $display("\n=== DUT_SMALL (SPI_HALF=12, FLASH_LEN=0x%06x) ===", SMALL_LEN);
            mism = 0; shown = 0;
            for (k = 0; k < SMALL_LEN; k = k + 1) begin
                if (s_sink.ram[k] !== expected_byte(32'h120000 + k)) begin
                    mism = mism + 1;
                    if (shown < 10) begin
                        $display("  MISMATCH idx=%0d: sink=0x%02x exp=0x%02x",
                                 k, s_sink.ram[k], expected_byte(32'h120000 + k));
                        shown = shown + 1;
                    end
                end
            end
            $display("  mismatches = %0d / %0d", mism, SMALL_LEN);
            $display("  boot cycles = %0d (%0t ns)", s_done_cycle - s_start_cycle,
                     (s_done_cycle - s_start_cycle) * 40);
            if (mism > 0) total_errors = total_errors + 1;
        end
    endtask

    task check_full;
        begin
            $display("\n=== DUT_FULL (SPI_HALF=1, FLASH_LEN=0x%06x) ===", FULL_LEN);
            mism = 0; shown = 0;
            for (k = 0; k < FULL_LEN; k = k + 1) begin
                if (f_sink.ram[k] !== expected_byte(32'h120000 + k)) begin
                    mism = mism + 1;
                    if (shown < 10) begin
                        $display("  MISMATCH idx=%0d: sink=0x%02x exp=0x%02x",
                                 k, f_sink.ram[k], expected_byte(32'h120000 + k));
                        shown = shown + 1;
                    end
                end
            end
            $display("  mismatches = %0d / %0d", mism, FULL_LEN);
            $display("  boot cycles = %0d (%0t ns)", f_done_cycle - f_start_cycle,
                     (f_done_cycle - f_start_cycle) * 40);
            if (mism > 0) total_errors = total_errors + 1;
        end
    endtask

    task check_shift;
        begin
            integer shift_mism;
            $display("\n=== DUT_SHIFT (SPI_HALF=4, FLASH_LEN=0x%06x, SHIFT_LOAD=1) ===", SH_LEN);
            mism = 0; shown = 0;
            for (k = 0; k < SH_LEN; k = k + 1) begin
                if (h_sink.ram[k] !== expected_byte(32'h120000 + k)) begin
                    mism = mism + 1;
                    if (shown < 10) begin
                        $display("  MISMATCH idx=%0d: sink=0x%02x exp=0x%02x",
                                 k, h_sink.ram[k], expected_byte(32'h120000 + k));
                        shown = shown + 1;
                    end
                end
            end
            $display("  DMA mismatches = %0d / %0d", mism, SH_LEN);
            if (mism > 0) total_errors = total_errors + 1;

            shift_mism = 0;
            for (k = 0; k < 47; k = k + 1) begin
                if (h_shifts[k*8 +: 8] !== expected_byte(32'h181000 + k)) begin
                    $display("  SHIFT MISMATCH idx=%0d: got=0x%02x exp=0x%02x",
                             k, h_shifts[k*8 +: 8], expected_byte(32'h181000 + k));
                    shift_mism = shift_mism + 1;
                end
            end
            if (h_shifts_valid !== 1'b1) begin
                $display("  shifts_valid not asserted!");
                shift_mism = shift_mism + 1;
            end
            $display("  shift trailer: %0d/47 mismatch, shifts_valid=%b",
                     shift_mism, h_shifts_valid);
            if (shift_mism > 0) total_errors = total_errors + 1;
            $display("  boot cycles = %0d (%0t ns)", h_done_cycle - h_start_cycle,
                     (h_done_cycle - h_start_cycle) * 40);
        end
    endtask

endmodule