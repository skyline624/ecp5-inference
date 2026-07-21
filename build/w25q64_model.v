// =============================================================================
// w25q64_model.v -- minimal SPI flash slave model for the flash_boot testbench.
//
// Emulates a Winbond W25Q64 (8 MB) responding to the 0x03 READ DATA command
// in 4-wire SPI mode 0: master samples MISO on rising SCK, shifts MOSI on
// falling SCK.  Only the continuous-read (auto-increment) path is modelled --
// enough to validate flash_boot's byte-by-byte DMA.
//
// Storage: 2 MB BRAM-style `reg [7:0] mem [0:0x1FFFFF]`.  That covers the
// model image at 0x120000 (len 0x60040) and the 47-byte shift trailer at
// 0x181000.  Loaded from "model_flash.hex" via $readmemh with start_address
// 0x120000 (iverilog honours the 3rd $readmemh argument).
//
// Protocol timing (matches flash_boot.v):
//   * CS low starts a transaction.  The master clocks 32 bits out on MOSI
//     (8'h03 then 24-bit addr, MSB-first).
//   * Slave samples MOSI on posedge SCK into cmd_addr_sr[31:0].
//   * On the 32nd posedge, the slave latches `addr`, loads out_byte=mem[addr],
//     and enters READ_DATA mode.  No wait state is inserted -- the first
//     data bit (bit 7 of out_byte) is driven on the *next* negedge SCK.
//   * In READ_DATA: on each negedge SCK, drive flash_miso=out_byte[out_bit];
//     after bit 0, auto-increment addr and load the next out_byte.  This
//     continues while CS stays low.
//   * CS high resets the FSM and tri-states MISO.
//
// The cmd->data handoff is the delicate point: flash_boot enters F_DATA with
// sck=1, so the first tick in F_DATA is a *falling* edge.  This model must
// therefore have out_byte ready by the 32nd posedge so the negedge right
// after it can drive bit 7 immediately.
// =============================================================================
module w25q64_model (
    input  wire flash_cs_n,
    input  wire flash_sck,
    input  wire flash_mosi,
    output reg  flash_miso
);

    // 2 MB is enough for FLASH_BASE(0x120000)+model(0x60040)+shift(0x181000+47).
    reg [7:0] mem [0:2097151];

    reg [31:0] cmd_addr_sr;
    reg [23:0] addr;
    reg [7:0]  out_byte;
    reg [2:0]  out_bit;
    reg [5:0]  bit_cnt;       // 0..31 during command phase
    reg        in_data;       // 1 once READ_DATA is active
    integer    i;

    initial begin
        for (i = 0; i < 2097152; i = i + 1) mem[i] = 8'h00;
        // Load the test image at flash offset 0x120000.
        $readmemh("model_flash.hex", mem, 24'h120000);
        flash_miso   = 1'b0;
        cmd_addr_sr  = 32'd0;
        addr         = 24'd0;
        out_byte     = 8'd0;
        out_bit      = 3'd7;
        bit_cnt      = 6'd0;
        in_data      = 1'b0;
    end

    // CS edge: (re)start or abort a transaction.
    always @(flash_cs_n) begin
        if (flash_cs_n === 1'b1) begin
            // Deselect: idle the slave, release MISO.
            bit_cnt     = 6'd0;
            in_data     = 1'b0;
            flash_miso  = 1'b0;
            out_bit     = 3'd7;
        end else begin
            // Select: begin command phase.
            bit_cnt     = 6'd0;
            in_data     = 1'b0;
            flash_miso  = 1'b0;
            out_bit     = 3'd7;
        end
    end

    // Sample MOSI on posedge SCK (mode 0) during the command/addr phase.
    always @(posedge flash_sck) begin
        if (!flash_cs_n && !in_data) begin
            cmd_addr_sr <= {cmd_addr_sr[30:0], flash_mosi};
            if (bit_cnt == 6'd31) begin
                // 32 bits captured: cmd_addr_sr next = {0x03, addr[23:0]}.
                // The bit just shifted in is addr[0]; reconstruct full addr.
                addr     <= {cmd_addr_sr[22:0], flash_mosi};
                out_byte <= mem[{cmd_addr_sr[22:0], flash_mosi}];
                out_bit  <= 3'd7;
                in_data  <= 1'b1;
                bit_cnt  <= 6'd0;
            end else begin
                bit_cnt <= bit_cnt + 6'd1;
            end
        end
    end

    // Drive MISO on negedge SCK during READ_DATA (and don't touch it during
    // command phase -- master is shifting MOSI, MISO is don't-care).
    always @(negedge flash_sck) begin
        if (!flash_cs_n && in_data) begin
            flash_miso <= out_byte[out_bit];
            if (out_bit == 3'd0) begin
                // End of current byte: auto-increment and load next.
                addr     <= addr + 24'd1;
                out_byte <= mem[addr + 24'd1];
                out_bit  <= 3'd7;
            end else begin
                out_bit <= out_bit - 3'd1;
            end
        end
    end

endmodule