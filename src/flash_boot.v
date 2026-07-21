// =============================================================================
// flash_boot.v -- Autonomous SPI flash -> SDRAM boot loader for the Colorlight
// i9 v7.2 (W25Q64JVSIQ, 8 MB, 4-wire SPI on pins R2/V2/W2/U3).
//
// At power-on (after PLL lock + SDRAM init), this module streams the int8 model
// weights from the SPI flash into the SDRAM via a byte-by-byte DMA, then (step 2,
// SHIFT_LOAD=1) reads a 47-byte shift trailer and exposes it so the wrapper can
// populate the gen_seq sw_* regs -- replacing the UART SH command.
//
// It replaces the legacy Tang-Nano `sd_boot` (SD-card) skeleton referenced in
// node_top.v: same mem_addr/mem_din/mem_wr/done/drained interface, but driving a
// W25Q64 over 4-wire SPI with the standard 0x03 READ DATA command (no QSPI --
// DQ2/DQ3 not wired on the i9). CS is held low for the whole read burst so the
// flash auto-increments its internal address.
//
// SPI mode 0 (CPOL=0, CPHA=0): SCK low at idle, sample MISO on rising edge,
// shift MOSI on falling edge. The SCK divider counter is FROZEN (not reset) while
// waiting for !sdram_busy in F_WR, so the flash holds the current bit indefinitely
// -- no glitch, no timeout. `drained` lets node_top's bref_cnt slip an SDRAM
// refresh through during the long boot (see node_top.v:183-198).
//
// WARNING: do NOT use the ECP5 USRMCLK primitive for SCK -- it conflicts with the
// hard SPI config controller. Drive flash_sck as a plain GPIO.
//
// The W25Q64 ships in normal mode and the hard config controller leaves it
// active after bitstream load, so release-deep-power-down (0xAB) is not issued.
// If a future power-management scheme sleeps the flash between boots, prepend a
// 0xAB sequence here.
// =============================================================================
module flash_boot #(
    parameter [23:0] FLASH_BASE = 24'h120000,   // flash offset of the model image
    parameter [23:0] FLASH_LEN  = 24'h060040,   // bytes DMA'd into SDRAM (mirror 0..0x060040)
    parameter [23:0] SHIFT_BASE = 24'h181000,   // = FLASH_BASE + 0x061000, 47-byte shift trailer
    parameter        SHIFT_LOAD = 0,            // 0 = step 1 (SH provides shifts), 1 = step 2
    parameter [9:0]  SPI_HALF   = 10'd1         // SCK ~ clk_sys/(2*(SPI_HALF+1)) ~ 6.25 MHz @25 MHz
) (
    input  wire        clk,
    input  wire        rstn,
    input  wire        start,          // 1-cycle pulse from node_top (post SDRAM init)

    // SPI flash pins (LPF: flash_cs_n=R2, flash_miso=V2, flash_mosi=W2, flash_sck=U3)
    output reg         flash_cs_n,
    output reg         flash_sck,
    output reg         flash_mosi,
    input  wire        flash_miso,

    // DMA to the SDRAM controller (routed via the eff_* mux in node_top while booting)
    output reg  [22:0] mem_addr,       // SDRAM byte address (increments +1)
    output reg  [7:0]  mem_din,
    output reg         mem_wr,         // 1-cycle write pulse
    input  wire        sdram_busy,     // backpressure: wait !sdram_busy before pulsing mem_wr

    // Refresh cohabitation (node_top.v:189 gates a refresh on !sd_busy && !boot_wr && boot_drained)
    output wire        drained,        // 1 when no mem_wr is in flight -> refresh may slip through

    // Status / debug (match the legacy u_boot port list so node_top stays minimal)
    output reg         done,
    output reg  [15:0] dbg_blk,
    output wire        dbg_ready,
    output reg         dbg_fail,
    output wire        dbg_wpend,

    // Step 2: 47 shift bytes latched from the flash trailer -> populate sw_*
    output reg  [47*8-1:0] shifts_out,
    output reg             shifts_valid
);

    // ---------------------------------------------------------------------------
    // FSM states
    // ---------------------------------------------------------------------------
    localparam [3:0]
        F_IDLE   = 4'd0,
        F_RECS   = 4'd1,    // deselect flash (CS high) for a few cycles before a new command
        F_CMD    = 4'd2,    // CS low, load cmd shreg {0x03, addr[23:0]}, prep MSB
        F_CMD_TX = 4'd3,    // shift 32 cmd/addr bits out on MOSI (MISO ignored)
        F_DATA   = 4'd4,    // stream FLASH_LEN data bytes from MISO -> SDRAM
        F_WR     = 4'd5,    // wait !sdram_busy, pulse mem_wr, advance byte_idx
        F_SHIFT  = 4'd6,    // (step 2) stream 47 shift bytes from MISO into shifts_out
        F_DONE   = 4'd7;

    reg [3:0]  state = F_IDLE;

    // SPI engine
    reg [9:0]  div_cnt;          // half-period counter (frozen during F_WR stall)
    reg        sck;              // internal SCK (registered -> output flash_sck)
    reg [5:0]  bit_idx;          // bit index within the current phase (0..31 cmd, 0..7 data)
    reg [31:0] cmd_shreg;        // {8'h03, addr[23:0]} shifted out MSB-first
    reg [7:0]  rx_shreg;         // byte shifted in MSB-first
    reg [22:0] byte_idx;         // SDRAM byte offset (0..FLASH_LEN-1) for the model DMA
    reg [6:0]  sh_idx;           // shift byte index (0..46) for the trailer
    reg        phase_shift;      // 0 = model DMA phase, 1 = shift trailer phase
    reg [3:0]  recs_cnt;         // re-CS deselect timer

    // `tick` fires once per SCK half-period, only while the SPI engine is running.
    // F_WR with sdram_busy=1 freezes the engine (stall) so SCK holds its level.
    wire spi_run = (state==F_CMD_TX) || (state==F_DATA) || (state==F_SHIFT);
    wire stall   = (state==F_WR) && sdram_busy;
    wire tick    = spi_run && !stall && (div_cnt == SPI_HALF);

    assign drained   = ~mem_wr;          // refresh is further gated by !sd_busy && !boot_wr upstream
    assign dbg_ready = (state==F_DONE);
    assign dbg_wpend = (state==F_WR);

    // ---------------------------------------------------------------------------
    // Main sequential block
    // ---------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rstn) begin
            state        <= F_IDLE;
            flash_cs_n   <= 1'b1;
            flash_sck    <= 1'b0;
            flash_mosi   <= 1'b0;
            sck          <= 1'b0;
            div_cnt      <= 10'd0;
            bit_idx      <= 6'd0;
            byte_idx     <= 23'd0;
            sh_idx       <= 7'd0;
            phase_shift  <= 1'b0;
            recs_cnt     <= 4'd0;
            cmd_shreg    <= 32'd0;
            rx_shreg     <= 8'd0;
            mem_addr     <= 23'd0;
            mem_din      <= 8'd0;
            mem_wr       <= 1'b0;
            done         <= 1'b0;
            dbg_blk      <= 16'd0;
            dbg_fail     <= 1'b0;
            shifts_out   <= 0;
            shifts_valid <= 1'b0;
        end else begin
            // defaults
            mem_wr       <= 1'b0;
            // Note: shifts_valid is intentionally STICKY (set once in F_SHIFT,
            // held through F_DONE) so the wrapper can sample it any time after
            // boot.  Do NOT clear it here -- a 1-cycle pulse would be easy to
            // miss since `done` is asserted one cycle AFTER shifts_valid.

            // SCK half-period counter (frozen when !spi_run or stalling on sdram_busy)
            if (spi_run && !stall) begin
                if (tick) div_cnt <= 10'd0;
                else      div_cnt <= div_cnt + 10'd1;
            end

            case (state)
                // ------------------------------------------------------------
                F_IDLE: begin
                    flash_cs_n <= 1'b1;
                    flash_sck  <= 1'b0;
                    sck        <= 1'b0;
                    if (start) begin
                        byte_idx    <= 23'd0;
                        sh_idx      <= 7'd0;
                        phase_shift <= 1'b0;
                        done        <= 1'b0;
                        state       <= F_CMD;
                    end
                end

                // ------------------------------------------------------------ deselect between two reads
                F_RECS: begin
                    flash_cs_n <= 1'b1;
                    flash_sck  <= 1'b0;
                    sck        <= 1'b0;
                    if (recs_cnt == 4'd8) begin
                        state <= F_CMD;
                    end else recs_cnt <= recs_cnt + 4'd1;
                end

                // ------------------------------------------------------------ assert CS, load command + addr
                F_CMD: begin
                    flash_cs_n <= 1'b0;
                    div_cnt    <= 10'd0;
                    sck        <= 1'b0;
                    flash_sck  <= 1'b0;
                    bit_idx    <= 6'd0;
                    cmd_shreg  <= phase_shift ? {8'h03, SHIFT_BASE} : {8'h03, FLASH_BASE};
                    flash_mosi <= 1'b0;          // bit 31 = MSB of 0x03 = 0
                    state      <= F_CMD_TX;
                end

                // ------------------------------------------------------------ shift 32 cmd/addr bits (MISO ignored)
                F_CMD_TX: begin
                    flash_cs_n <= 1'b0;
                    if (tick) begin
                        if (!sck) begin                 // rising edge -> slave samples MOSI bit
                            if (bit_idx == 6'd31) begin
                                // 32 bits shifted; SCK is now high (just rose). The flash will
                                // output data bit 0 on the next falling edge -> enter data/shift.
                                bit_idx  <= 6'd0;
                                rx_shreg <= 8'd0;
                                state    <= phase_shift ? F_SHIFT : F_DATA;
                            end else begin
                                bit_idx <= bit_idx + 6'd1;
                            end
                        end else begin                  // falling edge -> shift MOSI to next bit
                            cmd_shreg  <= {cmd_shreg[30:0], 1'b0};
                            flash_mosi <= cmd_shreg[30];
                        end
                        sck       <= ~sck;
                        flash_sck <= ~sck;
                    end
                end

                // ------------------------------------------------------------ stream model bytes MISO -> SDRAM
                F_DATA: begin
                    flash_cs_n <= 1'b0;
                    if (tick) begin
                        if (!sck) begin                 // rising edge -> sample MISO
                            if (bit_idx == 6'd7) begin
                                mem_din <= {rx_shreg[6:0], flash_miso};
                                bit_idx <= 6'd0;
                                state   <= F_WR;
                            end else begin
                                rx_shreg <= {rx_shreg[6:0], flash_miso};
                                bit_idx  <= bit_idx + 6'd1;
                            end
                        end
                        // falling edge: flash advances to the next data bit, nothing to drive
                        sck       <= ~sck;
                        flash_sck <= ~sck;
                    end
                end

                // ------------------------------------------------------------ push one byte into SDRAM
                F_WR: begin
                    flash_cs_n <= 1'b0;
                    // SCK frozen here (spi_run=0) -> flash holds the current bit.
                    if (!sdram_busy) begin
                        mem_wr   <= 1'b1;
                        mem_addr <= byte_idx;
                        byte_idx <= byte_idx + 23'd1;
                        dbg_blk  <= byte_idx[15:0];
                        if (byte_idx == FLASH_LEN - 1) begin
                            // model fully loaded
                            if (SHIFT_LOAD) begin
                                phase_shift <= 1'b1;
                                recs_cnt    <= 4'd0;
                                state       <= F_RECS;
                            end else begin
                                state <= F_DONE;
                            end
                        end else begin
                            state <= F_DATA;            // continue the burst (SCK resumes high)
                        end
                    end
                end

                // ------------------------------------------------------------ (step 2) stream 47 shift bytes
                F_SHIFT: begin
                    flash_cs_n <= 1'b0;
                    if (tick) begin
                        if (!sck) begin                 // rising edge -> sample MISO
                            if (bit_idx == 6'd7) begin
                                shifts_out[sh_idx*8 +: 8] <= {rx_shreg[6:0], flash_miso};
                                if (sh_idx == 7'd46) begin
                                    shifts_valid <= 1'b1;
                                    state        <= F_DONE;
                                end else begin
                                    sh_idx  <= sh_idx + 7'd1;
                                    bit_idx <= 6'd0;
                                end
                            end else begin
                                rx_shreg <= {rx_shreg[6:0], flash_miso};
                                bit_idx  <= bit_idx + 6'd1;
                            end
                        end
                        sck       <= ~sck;
                        flash_sck <= ~sck;
                    end
                end

                // ------------------------------------------------------------
                F_DONE: begin
                    flash_cs_n <= 1'b1;
                    flash_sck  <= 1'b0;
                    sck        <= 1'b0;
                    done       <= 1'b1;
                end

                default: state <= F_IDLE;
            endcase
        end
    end

endmodule