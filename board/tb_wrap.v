`timescale 1ns/1ps
// =============================================================================
// tb_wrap.v -- G5' wrapper unit test (iverilog, fast). Validates the gen_seq_silicon
// wrapper FSM : SH latch, GO pulse, LL pass-through + LK ack, token packer (2 B/tok).
// Uses a MOCK gen_seq (emits a few token_valid after start) and a MOCK node `top`
// (parses LL and acks "LK") so the sim is microseconds, not the full 17-token run.
// =============================================================================
`define LINK_SS

// ---- MOCK gen_seq : after a 1-cycle start, emits token_valid with a fixed token ----
// sequence (captures sw_* and prompt on start so the testbench can check SH + PR).
module gen_seq #(parameter W=8, parameter D=64, parameter NL=5, parameter NPOS=17, parameter NPR=16) (
    input  wire           clk, rst_n, start, gen_mode,
    input  wire [W*D-1:0]  x_in,
    input  wire signed [7:0] sx_in,
    input  wire [NL*8-1:0] sw_rms, swq, swk, swv, swo, sw_rmsf, sw1, sw3, sw2,
    input  wire signed [7:0] sw_rmsfinal, sw_emb,
    input  wire [5:0]      pos,
    input  wire [22:0]     base,
    input  wire [NPR*10-1:0] prompt,
    input  wire [6:0]        prompt_len,
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    output reg  [W*D-1:0]  result,
    output reg  signed [7:0] result_sh,
    output reg  [9:0]      token,
    output reg             token_valid,
    output reg             done
);
    // capture shifts + prompt at start (testbench inspects them hierarchically)
    reg [NL*8-1:0] cap_rms, cap_q, cap_k, cap_v, cap_o, cap_rmsf, cap_1, cap_3, cap_2;
    reg signed [7:0] cap_rmsfinal, cap_emb;
    reg [NPR*10-1:0] cap_prompt; reg [6:0] cap_plen;
    reg [10:0] cnt; reg [3:0] emitted; reg running;
    // mock token sequence
    wire [9:0] SEQ [0:4] = '{ 10'd1, 10'd403, 10'd407, 10'd261, 10'd378 };
    always @(posedge clk) begin
        if (!rst_n) begin cnt<=0; emitted<=0; running<=0; token_valid<=0; done<=0; token<=0;
                         cap_prompt<=0; cap_plen<=0; end
        else begin
            token_valid <= 1'b0; done <= 1'b0;
            if (start) begin
                cap_rms<=sw_rms; cap_q<=swq; cap_k<=swk; cap_v<=swv; cap_o<=swo;
                cap_rmsf<=sw_rmsf; cap_1<=sw1; cap_3<=sw3; cap_2<=sw2;
                cap_rmsfinal<=sw_rmsfinal; cap_emb<=sw_emb;
                cap_prompt<=prompt; cap_plen<=prompt_len;
                cnt<=0; emitted<=0; running<=1;
            end else if (running) begin
                // space tokens ~600 cycles so the wrapper's 2-byte/token packer
                // (~520 cycles) never drops one (real gen_seq emits ms apart)
                if (cnt < 11'd600) cnt <= cnt + 11'd1;
                else if (emitted < 4'd5) begin
                    token <= SEQ[emitted]; token_valid <= 1'b1; emitted <= emitted + 4'd1; cnt <= 0;
                end else begin
                    done <= 1'b1; running <= 0;
                end
            end
        end
    end
    assign lk_cmd_data = 8'd0; assign lk_cmd_wr = 1'b0; assign lk_resp_rd = 1'b0;
    assign result = 0; assign result_sh = 0;
endmodule

// ---- MOCK node `top` (LINK_SS) : parses LL (acks "LK") and CC (acks "CK"+N data) ----
// Accepts the flash-boot params the wrapper passes to `top` (ignored by the mock ;
// the mock node never touches the flash). Keeps tb_wrap compatible with the
// FLASH_BOOT-era wrapper instance.
module top #(parameter SD_HALF=135, parameter [15:0] SD_NBLK=769,
             parameter [23:0] FLASH_BASE=24'h120000, parameter [23:0] FLASH_LEN=24'h060040,
             parameter [23:0] SHIFT_BASE=24'h181000, parameter SHIFT_LOAD_EN=1'b1,
             parameter [9:0] SPI_HALF_DIV=10'd1) (
    input  wire        clk,
    input  wire [7:0]  lk_rx_data, input wire lk_rx_empty, output wire lk_rx_rd,
    output wire [7:0]  lk_tx_data, output wire lk_tx_wr, input wire lk_tx_full,
    output wire [5:0]  led,
    output wire        O_sdram_clk, O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n,
    inout  wire [31:0] IO_sdram_dq, output wire [10:0] O_sdram_addr, output wire [1:0] O_sdram_ba);
    // header : op0 op1 a0 a1 a2 n0 n1 (7 bytes), then N data (LL consume / CC none).
    // Clean 3-phase FSM : HDR (collect 7 bytes) -> DATA (LL drain) or CC (push) ->
    // ACK (2-byte "LK"/"CK" + optional CC data). Phases are mutually exclusive.
    reg [2:0] rxcnt = 3'd0;          // 0..6 header byte index
    reg [7:0] op0 = 8'd0;
    reg [7:0] n0c = 8'd0;            // n0 captured at rxcnt==5 (NOT loaded into remaining yet)
    reg [15:0] dleft  = 16'd0;       // LL data bytes still to consume (data phase)
    reg       in_data = 1'b0;        // 1 = LL data-drain phase active
    reg [15:0] cc_left = 16'd0;      // CC data bytes still to push (incl. the 2 ack bytes? no, ack handled separately)
    reg [15:0] cc_idx  = 16'd0;      // index of CC data byte being pushed
    reg [7:0] tx_byte = 8'd0; reg tx_push = 1'b0;
    reg [1:0] tx_state = 2'd0;       // 0 idle,1 push first(C/L),2 push K,3 push CC data
    // lk_rx_rd : consume in HDR or DATA phase, never during CC push / ack push
    assign lk_rx_rd = ~lk_rx_empty & (cc_left==16'd0) & (tx_state==2'd0);
    assign lk_tx_data = tx_byte; assign lk_tx_wr = tx_push & ~lk_tx_full;
    assign led = 6'b0;
    assign O_sdram_clk=0; assign O_sdram_cas_n=1; assign O_sdram_ras_n=1;
    assign O_sdram_wen_n=1; assign O_sdram_addr=0; assign O_sdram_ba=0;
    assign IO_sdram_dq = 32'bz;
    wire [7:0] rb = lk_rx_data;
    always @(posedge clk) begin
        tx_push <= 1'b0;
        if (cc_left != 16'd0) begin
            // CC readback : push "CK" then cc_left data bytes (pattern 0xA0+i)
            if (tx_state==0 & ~lk_tx_full) begin
                tx_byte <= 8'h43; tx_push <= 1'b1; tx_state <= 1;   // 'C'
            end else if (tx_state==1 & ~tx_push & ~lk_tx_full) begin
                tx_byte <= 8'h4B; tx_push <= 1'b1; tx_state <= 2; cc_idx <= 0;  // 'K'
            end else if (tx_state==2 & ~tx_push & ~lk_tx_full) begin
                // push next CC data byte; when this is the last (cc_left 1->0), the
                // outer `if (cc_left!=0)` falls through next cycle and the tx_state==2
                // LL-ack branch returns to header phase (no extra byte pushed).
                tx_byte <= 8'hA0 + cc_idx[7:0]; tx_push <= 1'b1;
                cc_idx <= cc_idx + 16'd1; cc_left <= cc_left - 16'd1;
            end
        end
        else if (tx_state != 2'd0) begin
            // LL ack in progress ("LK") : push 'K' then return to header phase
            if (tx_state==1 & ~tx_push & ~lk_tx_full) begin
                tx_byte <= 8'h4B; tx_push <= 1'b1; tx_state <= 2;
            end else if (tx_state==2 & ~tx_push & ~lk_tx_full) begin
                tx_state <= 0; rxcnt <= 0;
            end
        end
        else if (in_data) begin
            // LL data drain : consume dleft bytes, ack 'L' on the last
            if (~lk_rx_empty) begin
                if (dleft == 16'd1) begin
                    dleft <= 16'd0; in_data <= 1'b0;
                    tx_state <= 1; tx_byte <= 8'h4C; tx_push <= 1'b1;  // 'L'
                end else dleft <= dleft - 16'd1;
            end
        end
        else if (~lk_rx_empty) begin
            // header parse (between commands)
            case (rxcnt)
                3'd0: begin op0 <= rb; rxcnt <= 3'd1; end
                3'd5: begin n0c <= rb; rxcnt <= 3'd6; end
                3'd6: begin
                          rxcnt <= 3'd0;
                          if (op0==8'h43) begin
                              // CC : ack "CK" + N data (no consume)
                              cc_left <= {rb, n0c};
                          end else begin
                              // LL : consume N data, then ack "LK"
                              if (rb==0 && n0c==0) begin
                                  tx_state <= 1; tx_byte <= 8'h4C; tx_push <= 1'b1; // N=0 -> 'L' now
                              end else begin
                                  in_data <= 1'b1;
                                  dleft   <= {rb, n0c};
                              end
                          end
                      end
                default: rxcnt <= rxcnt + 3'd1;
            endcase
        end
    end
endmodule

// ---- UART BFM (1 Mbaud @ 25 MHz, DIV=25, 40ns/cycle -> bit = 25*40 = 1000 ns) ----
module tb_wrap;
    localparam CYC = 40;        // 25 MHz
    localparam DIV = 25;
    localparam BIT = CYC*DIV;   // 1000 ns/bit
    reg clk=0; always #(CYC/2) clk = ~clk;

    reg uart_rx = 1'b1; wire uart_tx; wire [5:0] led;
    wire O_sdram_clk, O_sdram_cas_n, O_sdram_ras_n, O_sdram_wen_n;
    wire [31:0] IO_sdram_dq; wire [10:0] O_sdram_addr; wire [1:0] O_sdram_ba;

    gen_seq_silicon #(.W(8),.D(64),.NL(5),.NPOS(17)) dut (
        .clk(clk), .uart_rx(uart_rx), .uart_tx(uart_tx), .led(led),
        .O_sdram_clk(O_sdram_clk), .O_sdram_cas_n(O_sdram_cas_n), .O_sdram_ras_n(O_sdram_ras_n),
        .O_sdram_wen_n(O_sdram_wen_n), .IO_sdram_dq(IO_sdram_dq),
        .O_sdram_addr(O_sdram_addr), .O_sdram_ba(O_sdram_ba));

    // host -> fpga : send 1 start bit (0), 8 data LSB, 1 stop (1)
    task host_send(input [7:0] b);
        integer i;
        begin
            uart_rx = 1'b0; #(BIT);          // start
            for (i=0; i<8; i=i+1) begin uart_rx = b[i]; #(BIT); end
            uart_rx = 1'b1; #(BIT);           // stop
            #(BIT/2);                          // idle margin
        end
    endtask
    // fpga -> host : CONTINUOUS UART RX monitor (buffers bytes from t=0, like real HW).
    // host_recv pops from the queue. Avoids the send/recv race where the wrapper
    // starts acking before a blocking @(negedge) listener is armed.
    reg [7:0] rxq [0:63];
    integer rq_w = 0, rq_r = 0;
    integer mstate = 0, mcnt = 0, mbit = 0;
    reg [7:0] mbyte;
    always @(posedge clk) begin
        case (mstate)
            0: if (!uart_tx) begin mstate <= 1; mcnt <= 0; end       // falling edge = start
            1: if (mcnt == DIV/2) begin mstate <= 2; mcnt <= 0; mbit <= 0; end
               else mcnt <= mcnt + 1;                                 // mid start
            2: if (mcnt == DIV-1) begin
                   mcnt <= 0; mbyte[mbit] <= uart_tx;                  // sample mid-bit
                   if (mbit == 7) mstate <= 3; else mbit <= mbit + 1;
               end else mcnt <= mcnt + 1;
            3: if (mcnt == DIV-1) begin                                // stop bit
                   mstate <= 0; mcnt <= 0;
                   rxq[rq_w] <= mbyte; rq_w <= (rq_w + 1) % 64;
               end else mcnt <= mcnt + 1;
        endcase
    end
    task host_recv(output [7:0] b);
        begin
            wait (rq_r != rq_w);
            b = rxq[rq_r];
            rq_r = (rq_r + 1) % 64;
        end
    endtask

    integer errors = 0;
    reg [7:0] rb;
    // expected shifts (must match what we send via SH)
    localparam NL = 5;

    // monitor every byte the wrapper transmits (kept for diagnostics)
    always @(posedge clk) if (dut.tx_send_w) $display("TX: %02h run=%b", dut.tx_data_w, dut.run_mode);

    initial begin
        // let the wrapper POR complete naturally (por[15] rises at ~32768 cycles ~ 1.3 ms sim)
        uart_rx = 1'b1;
        #(CYC*40000);
        // ensure rst_n high
        if (dut.rst_n !== 1'b1) begin
            $display("FAIL: rst_n not high after POR"); errors=errors+1;
        end

        // ---- TEST 1 : SH + 47 bytes -> expect "SK" ----
        host_send(8'h53); // 'S'
        host_send(8'h48); // 'H'
        // 47 shift bytes : sw_rms[0..4]=0xFB(-5), swq=0xFA(-6), swk=[FA,FA,F9,FA,FA],
        //   swv=[F8,F7,F8,F8,F8], swo=[F9,F8,F8,F8,F9], sw_rmsf=[F9,FA,FA,FA,FA],
        //   sw1=0xF9*5, sw3=0xF9*5, sw2=0xF9*5, sw_rmsfinal=0xFC(-4), sw_emb=0xFA(-6)
        begin : sh_block
        integer i;
        reg [7:0] sh[0:46];
        sh[0]=8'hFB;sh[1]=8'hFB;sh[2]=8'hFB;sh[3]=8'hFB;sh[4]=8'hFB;       // sw_rms -5
        sh[5]=8'hFA;sh[6]=8'hFA;sh[7]=8'hFA;sh[8]=8'hFA;sh[9]=8'hFA;       // swq -6
        sh[10]=8'hFA;sh[11]=8'hFA;sh[12]=8'hF9;sh[13]=8'hFA;sh[14]=8'hFA;  // swk
        sh[15]=8'hF8;sh[16]=8'hF7;sh[17]=8'hF8;sh[18]=8'hF8;sh[19]=8'hF8;  // swv
        sh[20]=8'hF9;sh[21]=8'hF8;sh[22]=8'hF8;sh[23]=8'hF8;sh[24]=8'hF9;  // swo
        sh[25]=8'hF9;sh[26]=8'hFA;sh[27]=8'hFA;sh[28]=8'hFA;sh[29]=8'hFA;  // sw_rmsf
        sh[30]=8'hF9;sh[31]=8'hF9;sh[32]=8'hF9;sh[33]=8'hF9;sh[34]=8'hF9;  // sw1 -7
        sh[35]=8'hF9;sh[36]=8'hF9;sh[37]=8'hF9;sh[38]=8'hF9;sh[39]=8'hF9;  // sw3 -7
        sh[40]=8'hF9;sh[41]=8'hF9;sh[42]=8'hF9;sh[43]=8'hF9;sh[44]=8'hF9;  // sw2 -7
        sh[45]=8'hFC;  // sw_rmsfinal -4
        sh[46]=8'hFA;  // sw_emb -6
        for (i=0; i<47; i=i+1) host_send(sh[i]);
        end
        host_recv(rb);
        if (rb !== 8'h53) begin $display("FAIL: SH ack byte0 = %02h (exp 53 'S')", rb); errors=errors+1; end
        host_recv(rb);
        if (rb !== 8'h4B) begin $display("FAIL: SH ack byte1 = %02h (exp 4B 'K')", rb); errors=errors+1; end
        if (errors==0) $display("OK: SH latch + SK ack");

        // ---- TEST 2 : LL pass-through + LK ack ----
        host_send(8'h4C); host_send(8'h4C); // "LL"
        host_send(8'h00); host_send(8'h00); host_send(8'h01); // addr 0x010000
        host_send(8'h04); host_send(8'h00); // N=4
        host_send(8'hDE); host_send(8'hAD); host_send(8'hBE); host_send(8'hEF); // 4 data bytes
        host_recv(rb); if (rb !== 8'h4C) begin $display("FAIL: LL ack b0=%02h (exp 4C)", rb); errors=errors+1; end
        host_recv(rb); if (rb !== 8'h4B) begin $display("FAIL: LL ack b1=%02h (exp 4B)", rb); errors=errors+1; end
        if (errors==0) $display("OK: LL pass-through + LK ack");

        // ---- TEST 2b : CC readback -> "CK" + N data bytes (variable drain) ----
        begin : cc_block
        integer i; reg [7:0] expd;
        host_send(8'h43); host_send(8'h43);              // "CC"
        host_send(8'h00); host_send(8'h00); host_send(8'h01); // addr 0x010000
        host_send(8'h04); host_send(8'h00);              // N=4
        host_recv(rb); if (rb !== 8'h43) begin $display("FAIL: CC ack b0=%02h (exp 43 'C')", rb); errors=errors+1; end
        host_recv(rb); if (rb !== 8'h4B) begin $display("FAIL: CC ack b1=%02h (exp 4B 'K')", rb); errors=errors+1; end
        for (i=0; i<4; i=i+1) begin
            host_recv(rb); expd = 8'hA0 + i[7:0];
            if (rb !== expd) begin $display("FAIL: CC data[%0d]=%02h (exp %02h)", i, rb, expd); errors=errors+1; end
        end
        end
        if (errors==0) $display("OK: CC readback (CK + 4 data)");

        // ---- TEST 2c : PR with len=0 -> direct "PK", no data, prompt_len=0 (BOS path) ----
        host_send(8'h50); host_send(8'h52); // "PR"
        host_send(8'd0);                    // len = 0
        host_recv(rb); if (rb !== 8'h50) begin $display("FAIL: PR0 ack b0=%02h (exp 50 'P')", rb); errors=errors+1; end
        host_recv(rb); if (rb !== 8'h4B) begin $display("FAIL: PR0 ack b1=%02h (exp 4B 'K')", rb); errors=errors+1; end
        if (dut.prompt_len !== 7'd0) begin $display("FAIL: PR0 prompt_len=%0d (exp 0)", dut.prompt_len); errors=errors+1; end
        if (errors==0) $display("OK: PR len=0 -> direct PK (BOS backward-compat path)");

        // ---- TEST 2d : PR with len=5 -> "PK", wrapper prompt_tokens/prompt_len correct ----
        // tokens [1, 403, 407, 261, 378] (BOS + "Once upon a time"), lo then hi (same
        // byte order as the token packer emits).
        host_send(8'h50); host_send(8'h52); // "PR"
        host_send(8'd5);                    // len = 5
        host_send(8'h01); host_send(8'h00); // 1     = 0x001
        host_send(8'h93); host_send(8'h01); // 403   = 0x193
        host_send(8'h97); host_send(8'h01); // 407   = 0x197
        host_send(8'h05); host_send(8'h01); // 261   = 0x105
        host_send(8'h7A); host_send(8'h01); // 378   = 0x17A
        host_recv(rb); if (rb !== 8'h50) begin $display("FAIL: PR ack b0=%02h (exp 50 'P')", rb); errors=errors+1; end
        host_recv(rb); if (rb !== 8'h4B) begin $display("FAIL: PR ack b1=%02h (exp 4B 'K')", rb); errors=errors+1; end
        if (dut.prompt_len !== 7'd5) begin $display("FAIL: PR prompt_len=%0d (exp 5)", dut.prompt_len); errors=errors+1; end
        begin : pr_check
        integer i; reg [9:0] exp [0:4]; reg [9:0] got;
        exp[0]=10'd1; exp[1]=10'd403; exp[2]=10'd407; exp[3]=10'd261; exp[4]=10'd378;
        for (i=0;i<5;i=i+1) begin
            got = dut.prompt_tokens[i*10 +: 10];
            if (got !== exp[i]) begin $display("FAIL: prompt_tokens[%0d]=%0d (exp %0d)", i, got, exp[i]); errors=errors+1; end
        end
        end
        if (errors==0) $display("OK: PR len=5 -> PK + prompt_tokens latched (lo/hi bit-order OK)");

        // ---- TEST 3 : GO -> GK, then 5 tokens (2 bytes each) on uart_tx ----
        // Also checks the prompt is passed to gen_seq at GO (cap_prompt/cap_plen).
        host_send(8'h47); host_send(8'h4F); // "GO"
        host_recv(rb); if (rb !== 8'h47) begin $display("FAIL: GO ack b0=%02h (exp 47)", rb); errors=errors+1; end
        host_recv(rb); if (rb !== 8'h4B) begin $display("FAIL: GO ack b1=%02h (exp 4B)", rb); errors=errors+1; end
        // run_mode + start_pulse fire on ack_done (after the GK frame finishes);
        // the host's recv returns just as the frame ends, so settle a few cycles.
        #(CYC*30);
        if (dut.run_mode !== 1'b1) begin $display("FAIL: run_mode not set after GO"); errors=errors+1; end
        if (dut.u_seq.cap_rms[7:0] !== 8'hFB) begin $display("FAIL: cap_rms[0]=%02h (exp FB)", dut.u_seq.cap_rms[7:0]); errors=errors+1; end
        if (dut.u_seq.cap_emb !== 8'hFA) begin $display("FAIL: cap_emb=%02h (exp FA)", dut.u_seq.cap_emb); errors=errors+1; end
        if (dut.u_seq.cap_rmsfinal !== 8'hFC) begin $display("FAIL: cap_rmsfinal=%02h (exp FC)", dut.u_seq.cap_rmsfinal); errors=errors+1; end
        if (dut.u_seq.cap_k[23:16] !== 8'hF9) begin $display("FAIL: cap_k[2]=%02h (exp F9)", dut.u_seq.cap_k[23:16]); errors=errors+1; end
        // prompt is passed through to gen_seq at GO (PR latch reaches u_seq)
        if (dut.u_seq.cap_plen !== 7'd5) begin $display("FAIL: cap_plen=%0d (exp 5)", dut.u_seq.cap_plen); errors=errors+1; end
        if (dut.u_seq.cap_prompt[0*10 +: 10] !== 10'd1)   begin $display("FAIL: cap_prompt[0]=%0d (exp 1)",   dut.u_seq.cap_prompt[0*10 +: 10]); errors=errors+1; end
        if (dut.u_seq.cap_prompt[1*10 +: 10] !== 10'd403) begin $display("FAIL: cap_prompt[1]=%0d (exp 403)", dut.u_seq.cap_prompt[1*10 +: 10]); errors=errors+1; end
        if (dut.u_seq.cap_prompt[4*10 +: 10] !== 10'd378) begin $display("FAIL: cap_prompt[4]=%0d (exp 378)", dut.u_seq.cap_prompt[4*10 +: 10]); errors=errors+1; end
        if (errors==0) $display("OK: GO + GK ack + run_mode + shifts latched + prompt passed to gen_seq");
        // collect 5 tokens (mock emits SEQ=[1,403,407,261,378])
        begin : tok_block
        integer t; reg [9:0] exp [0:4];
        reg [9:0] got; reg [7:0] lo, hi;
        exp[0]=10'd1; exp[1]=10'd403; exp[2]=10'd407; exp[3]=10'd261; exp[4]=10'd378;
        for (t=0; t<5; t=t+1) begin
            host_recv(lo); host_recv(hi);
            got = {hi[1:0], lo};
            if (got !== exp[t]) begin $display("FAIL: token %0d = %0d (exp %0d)", t, got, exp[t]); errors=errors+1; end
            else $display("OK: token %0d = %0d", t, got);
        end
        end

        if (errors==0) $display("PASS: gen_seq_silicon wrapper FSM OK");
        else $display("FAIL: %0d errors", errors);
        $finish(errors);
    end

    // safety timeout
    initial begin
        #(50_000_000); // 50 ms sim
        $display("FAIL: timeout");
        $finish(1);
    end
endmodule