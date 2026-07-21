`timescale 1ns/1ps
`define LINK_SS
// =============================================================================
// gen_seq_silicon.v -- G5' SILICON wrapper : autonomous gen_seq on Colorlight
// i9 v7.2 (ECP5-45). The PC loads int8 weights into the node's SDRAM via the
// existing `LL` opcode, loads the 9 per-layer shift tables + 2 lm-head shifts
// via a new `SH` opcode, then pulses `GO` : the wrapper starts gen_seq and the
// FPGA generates 17 tokens autonomously (zero PC commands during inference).
// Each emitted token is shipped back to the PC as 2 bytes on uart_tx (read-only).
//
// Host byte protocol (8N1, 1 Mbaud, DIV=25 @ 25 MHz; pure byte stream):
//   LL a0 a1 a2 n0 n1 data[N]  -> ack LK   (bulk SDRAM write; N=n0+n1*256, LE addr)
//   SH <47 shift bytes>        -> ack SK   (latch sw_* regs feeding gen_seq)
//   GO                         -> ack GK   (start gen_seq, switch to RUN mode)
//   RR                         -> ack RK   (re-arm LOAD, reset gen_seq) [optional]
//
// `LL` is handled by pass-through with a "boundary" flag : the wrapper forwards
// every host byte to the node's lk_rx while in a command, and counts the 2-byte
// ack the node returns on lk_tx. After 2 ack bytes the next host byte is again
// an opcode boundary, where SH/GO/RR are intercepted (they never reach the node).
// v1 supports `LL` only (2-byte ack); BB/CC (variable ack) are not used.
//
// Modes :
//   LOAD (run_mode=0) : host UART -> cmd FIFO write side; node lk_tx acks -> uart_tx.
//                       gen_seq sits in IDLE (start=0, s_cmd_full forced high).
//   RUN  (run_mode=1) : gen_seq owns the cmd FIFO; node lk_tx -> resp FIFO -> gen_seq;
//                       each token_valid packs 2 bytes to uart_tx. Host is read-only.
// =============================================================================
module gen_seq_silicon #(
    parameter W = 8, parameter D = 64, parameter NL = 5, parameter NPOS = 17,
    parameter NPR = 16        // max prompt tokens for prefill (PR opcode)
) (
    input  wire           clk,                 // 25 MHz, pin P3
    input  wire           uart_rx,             // H18 (host -> FPGA, 1 Mbaud)
    output wire           uart_tx,             // J17 (FPGA -> host)
    output wire [5:0]     led,                 // node heartbeat (led[5] -> L2)
    // SDRAM pins to the external M12L64322A (CKE/CS/DQM hardwired on PCB)
    output wire          O_sdram_clk,
    output wire          O_sdram_cas_n,
    output wire          O_sdram_ras_n,
    output wire          O_sdram_wen_n,
    inout  wire [31:0]   IO_sdram_dq,
    output wire [10:0]   O_sdram_addr,
    output wire [1:0]    O_sdram_ba
`ifdef FLASH_BOOT
    // SPI flash (W25Q64) for the autonomous boot. CS/MOSI/MISO on R2/W2/V2.
    // SCK is NOT a top-level pin: pin U3 is the dedicated MCLK/CCLK pad and is
    // driven via the USRMCLK primitive below (see LPF header comment).
    ,output wire       flash_mosi
    ,input  wire       flash_miso
    ,output wire       flash_cs_n
`endif
);
    // ---- Power-on reset (raw clk domain; the node has its own internal POR) ----
    // Counter saturates at 0xFFFF so rst_n = por[15] rises once and STAYS high.
    // A free-running 16-bit counter would wrap every 65536 cycles (~2.6 ms @ 25 MHz)
    // and drop rst_n again, resetting the whole design periodically.
    reg [15:0] por = 16'd0;
    always @(posedge clk) if (por != 16'hFFFF) por <= por + 16'd1;
    wire rst_n = por[15];
    wire uart_rst = ~rst_n;

    // ---- Host UART (1 Mbaud @ 25 MHz : DIV = 25) ----
    wire [7:0] rx_byte;  wire rx_valid;
    uart_rx_8n1 #(.DIV(25)) u_rx (
        .clk(clk), .rst(uart_rst), .rx(uart_rx), .data(rx_byte), .valid(rx_valid));

    wire [7:0] tx_data_w;  wire tx_send_w;  wire tx_busy;
    uart_tx_8n1 #(.DIV(25)) u_tx (
        .clk(clk), .rst(uart_rst), .data(tx_data_w), .send(tx_send_w), .tx(uart_tx), .busy(tx_busy));

    // ---- Shift register file (47 bytes loaded by `SH`, held for the whole run) ----
    reg [NL*8-1:0] sw_rms, swq, swk, swv, swo, sw_rmsf, sw1, sw3, sw2;
    reg signed [7:0] sw_rmsfinal, sw_emb;

    // ---- Prompt register file (loaded by `PR`, fed to gen_seq at GO) ----
    // prompt_tokens = NPR packed 10-bit token IDs (lo-first over UART : t_lo then t_hi).
    // prompt_len=0 at reset -> gen_seq uses its hardcodé BOS (backward compat, étape 2).
    reg [NPR*10-1:0] prompt_tokens;
    reg [6:0]        prompt_len;
    reg [4:0]        prompt_idx;   // next token slot during PR ingest
    reg [7:0]        pr_lo;        // low byte held between the lo/hi pair

`ifdef FLASH_BOOT
    // 47 shift bytes from the flash trailer (step 2). Latched once into sw_* when
    // shifts_valid_from_flash asserts (end of the flash boot). A later UART `SH`
    // still overwrites sw_* for debug. Order matches gen_flash_model.py trailer /
    // sh_packet / H_SH exactly.
    wire [47*8-1:0] shifts_from_flash;
    wire            shifts_valid_from_flash;
    reg             shifts_latched;

    // SCK to the W25Q64 : driven through the USRMCLK primitive to the dedicated
    // MCLK/CCLK pad (U3). flash_sck_int is the fabric-side clock from u_n.
    // sck_tsen (from u_n) is the tri-state control : 0 = drive SCK (only while
    // flash_boot is actively reading), 1 = HiZ (else) so the hard config controller
    // / JTAG spiOverJtag can reprogram the flash without contention. USRMCLKTS may
    // NOT be a compile-time constant -- it is a real signal from the node.
    wire            flash_sck_int;
    wire            sck_tsen;
    USRMCLK u_flashmclk (
        .USRMCLKI (flash_sck_int),
        .USRMCLKTS(sck_tsen)
    );
`endif

    // ---- Mode + start ----
    reg run_mode = 1'b0;       // 0 = LOAD (host owns lk bus), 1 = RUN (gen_seq owns)
    reg start_pulse = 1'b0;   // 1-cycle pulse to gen_seq.start on GO
    reg seq_rst_n = 1'b1;      // gen_seq reset (toggled low for 1 cycle by RR)

    // ---- gen_seq <-> node link (mirrors gen_seq_top) ----
    wire [7:0] s_cmd_data;  wire s_cmd_wr;
    wire [7:0] s_resp_data; wire s_resp_rd;
    wire [9:0] token;  wire token_valid;  wire done;
    wire [W*D-1:0] result;  wire signed [7:0] result_sh;

    // clk_sys = node system clock (PLL output, 31.25 MHz). The host FSM + UART + gen_seq
    // run on `clk` (25 MHz crystal); the compute node runs on clk_sys. The two async_fifo
    // below cross the clk/clk_sys boundary : the node-side port MUST be on clk_sys so the
    // 1-cycle o_rd/o_wr pulses from rx8_link/tx8_link (clocked by clk_sys inside the node)
    // are sampled on the same domain as the FIFO read/write port. With clk_sys != clk this
    // is a true async crossing handled by the FIFO's Gray-pointer CDC. (At 25 MHz clk==clk_sys
    // it was quasi-synchronous by accident; bumping the PLL makes the separation mandatory.)
    wire clk_sys;

    // cmd FIFO (u_cmd) : write side mux (host LOAD | gen_seq RUN) on clk, read side = node lk_rx on clk_sys
    wire [7:0] cmd_wdata;  wire cmd_wr;  wire cmd_full;
    wire [7:0] c_rdata;    wire c_empty;  wire c_rd;
    // resp FIFO (u_resp) : write side = node lk_tx, read side mux (gen_seq RUN | ack-drain LOAD)
    wire [7:0] r_wdata;    wire r_wr;    wire r_full;
    wire [7:0] resp_rdata; wire resp_empty; wire resp_rd;

    // gen_seq sees its own full flag in RUN; in LOAD it is held full to block it
    wire s_cmd_full = run_mode ? cmd_full : 1'b1;
    wire s_resp_empty = run_mode ? resp_empty : 1'b1;  // gen_seq never reads in LOAD

    gen_seq #(.W(W), .D(D), .NL(NL), .NPOS(NPOS), .NPR(NPR)) u_seq (
        .clk(clk), .rst_n(rst_n & seq_rst_n), .start(start_pulse), .gen_mode(1'b1),
        .x_in({W*D{1'b0}}), .sx_in(8'sd0),
        .sw_rms(sw_rms), .swq(swq), .swk(swk), .swv(swv), .swo(swo),
        .sw_rmsf(sw_rmsf), .sw1(sw1), .sw3(sw3), .sw2(sw2),
        .sw_rmsfinal(sw_rmsfinal), .sw_emb(sw_emb),
        .pos(6'd0), .base(23'd0),
        .prompt(prompt_tokens), .prompt_len(prompt_len),
        .lk_cmd_data(s_cmd_data), .lk_cmd_wr(s_cmd_wr), .lk_cmd_full(s_cmd_full),
        .lk_resp_data(s_resp_data), .lk_resp_empty(s_resp_empty), .lk_resp_rd(s_resp_rd),
        .result(result), .result_sh(result_sh), .token(token),
        .token_valid(token_valid), .done(done));

    async_fifo #(.DW(8), .AW(4)) u_cmd (
        .wclk(clk), .wrst_n(rst_n), .winc(cmd_wr), .wdata(cmd_wdata), .wfull(cmd_full),
        .rclk(clk_sys), .rrst_n(rst_n), .rinc(c_rd), .rdata(c_rdata), .rempty(c_empty));
    async_fifo #(.DW(8), .AW(4)) u_resp (
        .wclk(clk_sys), .wrst_n(rst_n), .winc(r_wr), .wdata(r_wdata), .wfull(r_full),
        .rclk(clk), .rrst_n(rst_n), .rinc(resp_rd), .rdata(resp_rdata), .rempty(resp_empty));

    // ---- Forward latch (host rx -> u_cmd) : 1-deep elastic buffer, never drops ----
    // host_state is 4 bits : H_IDLE..H_R (0..6) + H_PR/H_PRLEN/H_PRDATA (7..9) for the
    // `PR` (prompt load) opcode. The forward-latch and the `host_state != H_R` guard
    // below are width-insensitive.
    localparam H_IDLE=4'd0, H_FWD=4'd1, H_S=4'd2, H_SH=4'd3, H_G=4'd4, H_GK=4'd5, H_R=4'd6,
               H_PR=4'd7, H_PRLEN=4'd8, H_PRDATA=4'd9;
    reg [3:0] host_state = H_IDLE;
    wire is_pt = (rx_byte==8'h4C)|(rx_byte==8'h57)|(rx_byte==8'h42)|(rx_byte==8'h43); // L W B C
    wire fwd_accept = rx_valid & ((host_state==H_FWD) | (host_state==H_IDLE & is_pt));
    reg [7:0] fwd_buf;  reg fwd_pending;
    wire fwd_fire = fwd_pending & ~cmd_full;
    wire fwd_load = fwd_accept & (~fwd_pending | fwd_fire);
    always @(posedge clk) begin
        if (!rst_n) fwd_pending <= 1'b0;
        else if (fwd_load) begin fwd_buf <= rx_byte; fwd_pending <= 1'b1; end
        else if (fwd_fire) fwd_pending <= 1'b0;
    end
    assign cmd_wdata = run_mode ? s_cmd_data : fwd_buf;
    assign cmd_wr    = run_mode ? s_cmd_wr   : fwd_fire;

    // ---- Ack serializer (LOAD) : drains a_drain bytes to uart_tx ----
    // sel=0 : constant pair (a_b0c then a_b1c) -- only used with a_drain=2 (SH/GO/RR).
    // sel=1 : drain a_drain bytes from u_resp -- 2 for LL "LK"/WW "WK", 2+ndata for
    // CC "CK"+data (SDRAM readback diagnostic).
    // Per byte : send (A_T), then busy-then-done handshake on tx (A_WB waits for
    // busy = send accepted, A_WD waits for ~busy = frame finished). Mandatory :
    // tx_busy asserts 1 cycle after send, so checking ~tx_busy immediately drops
    // the next byte.
    localparam A_IDLE=3'd0, A_W0=3'd1, A_T=3'd2, A_WB=3'd3, A_WD=3'd4;
    reg [2:0] a_state = A_IDLE;
    reg       a_sel;          // 0=const, 1=u_resp
    reg       a_sel_next;     // set by host FSM before ack_go (0=const, 1=u_resp)
    reg [15:0] a_drain;       // total bytes to drain (set by host FSM before ack_go)
    reg [15:0] a_left;        // bytes remaining after the one being sent
    reg [7:0] a_b0c, a_b1c;   // constant ack bytes (sel=0)
    reg [7:0] a_byte;         // byte being sent
    reg       ack_go;         // 1-cycle trigger from the host FSM
    reg       ack_done;       // 1-cycle pulse when all a_drain bytes have been sent
    reg [7:0] a_tx_data;  reg a_tx_send;
    wire a_resp_rd = (a_state==A_W0) & ~resp_empty;
    always @(posedge clk) begin
        if (!rst_n) begin a_state<=A_IDLE; ack_done<=1'b0; a_tx_data<=8'd0; a_tx_send<=1'b0; a_byte<=8'd0; a_left<=16'd0; end
        else begin
            a_tx_send <= 1'b0; ack_done <= 1'b0;
            case (a_state)
                A_IDLE: if (ack_go) begin
                    a_sel  <= a_sel_next;
                    a_left <= a_drain - 16'd1;   // bytes remaining after byte1
                    if (a_sel_next==1'b0) begin a_byte <= a_b0c; a_state <= A_T; end
                    else a_state <= A_W0;        // fetch byte1 from u_resp (A_W0 pops)
                end
                A_W0: if (~resp_empty) begin a_byte <= resp_rdata; a_state <= A_T; end
                A_T:  begin a_tx_data <= a_byte; a_tx_send <= 1'b1; a_state <= A_WB; end
                A_WB: if (tx_busy) a_state <= A_WD;            // wait for send to start
                A_WD: if (~tx_busy) begin                      // wait for frame to finish
                          if (a_left > 16'd0) begin
                              a_left <= a_left - 16'd1;
                              if (a_sel==1'b0) begin a_byte <= a_b1c; a_state <= A_T; end
                              else a_state <= A_W0;              // fetch next from u_resp
                          end else begin
                              a_state <= A_IDLE; ack_done <= 1'b1;
                          end
                      end
                default: a_state <= A_IDLE;
            endcase
        end
    end
    // (a_sel_next / a_drain declared with the ack serializer regs above)

    // ---- Host byte FSM ----
    reg [6:0] sh_cnt;
    reg [2:0] sh_lyr;   // 0..4 layer index within a shift table
    reg [3:0] sh_tab;   // 0..8 shift table index (sw_rms..sw2)
    reg [7:0] fwd_op1;   // first byte of a pass-through command (L/W/B/C)
    reg [3:0] fwd_idx;   // byte index within H_FWD (0 = 2nd byte of the command)
    reg [7:0] cc_n0;     // CC low byte of N (for readback drain count)
    always @(posedge clk) begin
        if (!rst_n) begin
            host_state <= H_IDLE; run_mode <= 1'b0; start_pulse <= 1'b0; seq_rst_n <= 1'b1;
            sh_cnt <= 7'd0; sh_lyr <= 3'd0; sh_tab <= 4'd0;
            ack_go <= 1'b0; a_sel_next <= 1'b0; a_b0c <= 8'd0; a_b1c <= 8'd0; a_drain <= 16'd2;
            fwd_op1 <= 8'd0; fwd_idx <= 4'd0; cc_n0 <= 8'd0;
            sw_rms<=0; swq<=0; swk<=0; swv<=0; swo<=0;
            sw_rmsf<=0; sw1<=0; sw3<=0; sw2<=0; sw_rmsfinal<=8'sd0; sw_emb<=8'sd0;
            prompt_len<=7'd0; prompt_idx<=5'd0; prompt_tokens<={NPR*10{1'b0}}; pr_lo<=8'd0;
`ifdef FLASH_BOOT
            shifts_latched <= 1'b0;
`endif
        end else begin
`ifdef FLASH_BOOT
            // Step 2 : latch the 47 shifts from the flash trailer once, when the
            // flash boot signals them valid. A later UART `SH` (H_SH below) still
            // overwrites sw_* for debug. Order matches gen_flash_model.py trailer
            // and the sh_packet / H_SH order exactly.
            if (shifts_valid_from_flash && !shifts_latched) begin
                sw_rms      <= shifts_from_flash[ 0*8 +: 40];
                swq         <= shifts_from_flash[ 5*8 +: 40];
                swk         <= shifts_from_flash[10*8 +: 40];
                swv         <= shifts_from_flash[15*8 +: 40];
                swo         <= shifts_from_flash[20*8 +: 40];
                sw_rmsf     <= shifts_from_flash[25*8 +: 40];
                sw1         <= shifts_from_flash[30*8 +: 40];
                sw3         <= shifts_from_flash[35*8 +: 40];
                sw2         <= shifts_from_flash[40*8 +: 40];
                sw_rmsfinal <= $signed(shifts_from_flash[45*8 +: 8]);
                sw_emb      <= $signed(shifts_from_flash[46*8 +: 8]);
                shifts_latched <= 1'b1;
            end
`endif
            if (run_mode) begin
                // RUN : host FSM idle (host read-only); token packer drives u_tx.
                host_state <= H_IDLE; start_pulse <= 1'b0; ack_go <= 1'b0;
            end else begin
            start_pulse <= 1'b0; ack_go <= 1'b0;
            case (host_state)
                H_IDLE: if (rx_valid) begin
                    case (rx_byte)
                        8'h4C, 8'h57, 8'h42, 8'h43: begin
                            // L W B C -> pass-through; default drain = 2 (LL/WW ack),
                            // overridden to 2+N for CC after parsing N.
                            fwd_op1 <= rx_byte; fwd_idx <= 4'd0; a_drain <= 16'd2;
                            host_state <= H_FWD;
                        end
                        8'h53: host_state <= H_S;    // S -> SH?
                        8'h47: host_state <= H_G;    // G -> GO?
                        8'h52: host_state <= H_R;    // R -> RR?
                        8'h50: host_state <= H_PR;   // P -> PR? (prompt load)
                        default: host_state <= H_IDLE;
                    endcase
                end
                H_FWD: begin
                    // pass-through rx->u_cmd handled by the forward latch; here parse
                    // CC's N field (for the drain count) and launch the ack serializer.
                    if (rx_valid) begin
                        if (fwd_op1==8'h43) begin           // CC : bytes 5,6 = N LE
                            if (fwd_idx==4'd4) cc_n0 <= rx_byte;
                            else if (fwd_idx==4'd5) a_drain <= 16'd2 + {rx_byte, cc_n0};
                        end
                        fwd_idx <= fwd_idx + 4'd1;
                    end
                    if (~resp_empty & a_state==A_IDLE & ~ack_go) begin
                        a_sel_next <= 1'b1; ack_go <= 1'b1;
                    end else if (ack_done) begin
                        host_state <= H_IDLE;       // ack drained -> boundary
                    end
                end
                H_S: if (rx_valid) begin
                    if (rx_byte==8'h48) begin host_state <= H_SH; sh_cnt <= 7'd0; sh_lyr <= 3'd0; sh_tab <= 4'd0; end  // 'H'
                    else host_state <= H_IDLE;
                end
                H_SH: if (rx_valid) begin
                    // 47 bytes : tables 0..8 (5 bytes each, layer 0..4) + sw_rmsfinal + sw_emb
                    if (sh_cnt < 7'd45) begin
                        case (sh_tab)
                            4'd0: sw_rms  [sh_lyr*8 +: 8] <= rx_byte;
                            4'd1: swq    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd2: swk    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd3: swv    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd4: swo    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd5: sw_rmsf[sh_lyr*8 +: 8] <= rx_byte;
                            4'd6: sw1    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd7: sw3    [sh_lyr*8 +: 8] <= rx_byte;
                            4'd8: sw2    [sh_lyr*8 +: 8] <= rx_byte;
                        endcase
                        // advance layer, wrap table every 5
                        if (sh_lyr == 3'd4) begin sh_lyr <= 3'd0; sh_tab <= sh_tab + 4'd1; end
                        else sh_lyr <= sh_lyr + 3'd1;
                    end else if (sh_cnt == 7'd45) sw_rmsfinal <= $signed(rx_byte);
                    else /*46*/ sw_emb <= $signed(rx_byte);
                    if (sh_cnt == 7'd46) begin
                        a_sel_next <= 1'b0; a_drain <= 16'd2; a_b0c <= 8'h53; a_b1c <= 8'h4B; ack_go <= 1'b1; // 'S','K'
                        host_state <= H_IDLE;
                    end else sh_cnt <= sh_cnt + 7'd1;
                end
                H_G: if (rx_valid) begin
                    if (rx_byte==8'h4F) begin  // 'O' -> GO
                        a_sel_next <= 1'b0; a_drain <= 16'd2; a_b0c <= 8'h47; a_b1c <= 8'h4B; ack_go <= 1'b1; // 'G','K'
                        host_state <= H_GK;
                    end else host_state <= H_IDLE;
                end
                H_GK: if (ack_done) begin
                    run_mode <= 1'b1; start_pulse <= 1'b1; host_state <= H_IDLE;
                end
                H_R: if (rx_valid) begin
                    if (rx_byte==8'h52) begin  // 'R' -> RR
                        a_sel_next <= 1'b0; a_drain <= 16'd2; a_b0c <= 8'h52; a_b1c <= 8'h4B; ack_go <= 1'b1; // 'R','K'
                        seq_rst_n <= 1'b0;      // reset gen_seq for 1 cycle
                        host_state <= H_IDLE;
                    end else host_state <= H_IDLE;
                end
                // ---- PR : load prompt tokens (prefill) ----
                // Format : PR len t0_lo t0_hi t1_lo t1_hi ...  (len tokens, lo then hi,
                // same byte order as the token packer emits). Ack 'P''K'. prompt_len is
                // clamped to NPR ; the host MUST send len <= NPR (and <= NPOS-1 to leave
                // room for >=1 generated token). prompt_len=0 -> ack direct, BOS path.
                H_PR: if (rx_valid) begin
                    if (rx_byte==8'h52) host_state <= H_PRLEN;   // 'R' -> PR
                    else host_state <= H_IDLE;
                end
                H_PRLEN: if (rx_valid) begin
                    prompt_len <= (rx_byte > NPR) ? NPR[6:0] : rx_byte[6:0];
                    if (rx_byte==8'd0) begin
                        a_sel_next <= 1'b0; a_drain <= 16'd2; a_b0c <= 8'h50; a_b1c <= 8'h4B; ack_go <= 1'b1; // 'P','K'
                        host_state <= H_IDLE;
                    end else begin
                        prompt_idx <= 5'd0; sh_cnt <= 7'd0;
                        host_state <= H_PRDATA;
                    end
                end
                H_PRDATA: if (rx_valid) begin
                    if (sh_cnt[0]==1'b0) pr_lo <= rx_byte;
                    else begin
                        prompt_tokens[prompt_idx*10 +: 10] <= {rx_byte[1:0], pr_lo};
                        prompt_idx <= prompt_idx + 5'd1;
                    end
                    if (sh_cnt == ({prompt_len,1'b0} - 7'd1)) begin   // 2*prompt_len-1 = last hi byte
                        a_sel_next <= 1'b0; a_drain <= 16'd2; a_b0c <= 8'h50; a_b1c <= 8'h4B; ack_go <= 1'b1; // 'P','K'
                        host_state <= H_IDLE;
                    end else sh_cnt <= sh_cnt + 7'd1;
                end
                default: host_state <= H_IDLE;
            endcase
            if (host_state != H_R) seq_rst_n <= 1'b1;  // release gen_seq reset
            end // close the LOAD (run_mode==0) else-begin
        end     // close the post-reset else-begin
    end

    // ---- u_resp read side mux (RUN = gen_seq, LOAD = ack serializer) ----
    assign resp_rd = run_mode ? s_resp_rd : a_resp_rd;
    // data path : no mux needed -- in LOAD, s_resp_empty=1 blocks gen_seq so it
    // never asserts s_resp_rd / never reads s_resp_data; in RUN, the ack serializer
    // sits in A_IDLE and never asserts a_resp_rd. Both sides read the same FIFO, but
    // only one is ever active, so the read data can be shared directly. Omitting this
    // assign left s_resp_data undriven -> gen_seq read 0x00 for every node response in
    // RUN -> all activations/logits 0 -> argmax 0 -> "17 tokens = 0" on silicon.
    assign s_resp_data = resp_rdata;

    // ---- Token packer (RUN) : 2 bytes per token_valid on u_tx ----
    // Same busy-then-done handshake as the ack serializer : the low and high bytes
    // are back-to-back, so the 2nd must wait for the 1st frame to finish.
    localparam TP_IDLE=3'd0, TP_S0=3'd1, TP_WB0=3'd2, TP_WD0=3'd3, TP_S1=3'd4, TP_WB1=3'd5, TP_WD1=3'd6;
    reg [2:0] tp_state = TP_IDLE;
    reg [9:0] tok_hold;
    reg [7:0] t_tx_data;  reg t_tx_send;
    always @(posedge clk) begin
        if (!rst_n || !run_mode) begin tp_state <= TP_IDLE; t_tx_send <= 1'b0; t_tx_data <= 8'd0; end
        else begin
            t_tx_send <= 1'b0;
            case (tp_state)
                TP_IDLE: if (token_valid) begin tok_hold <= token; tp_state <= TP_S0; end
                TP_S0:   begin t_tx_data <= tok_hold[7:0];      t_tx_send <= 1'b1; tp_state <= TP_WB0; end
                TP_WB0:  if (tx_busy) tp_state <= TP_WD0;       // wait for send to start
                TP_WD0:  if (~tx_busy) tp_state <= TP_S1;        // wait for frame to finish
                TP_S1:   begin t_tx_data <= {6'b0, tok_hold[9:8]}; t_tx_send <= 1'b1; tp_state <= TP_WB1; end
                TP_WB1:  if (tx_busy) tp_state <= TP_WD1;
                TP_WD1:  if (~tx_busy) tp_state <= TP_IDLE;
                default: tp_state <= TP_IDLE;
            endcase
        end
    end

    // ---- u_tx mux : RUN = token packer, LOAD = ack serializer ----
    assign tx_data_w = run_mode ? t_tx_data : a_tx_data;
    assign tx_send_w = run_mode ? t_tx_send : a_tx_send;

    // ---- the LINK_SS compute node (module `top` in node_top.v) ----
    wire [5:0] node_led;
    top #(
        .FLASH_BASE(24'h120000), .FLASH_LEN(24'h060040), .SHIFT_BASE(24'h181000),
        .SHIFT_LOAD_EN(1'b1),    .SPI_HALF_DIV(10'd1)   // step 2: shifts also from flash trailer
    ) u_n (
        .clk(clk),
        .clk_sys_out(clk_sys),
        .lk_rx_data(c_rdata), .lk_rx_empty(c_empty), .lk_rx_rd(c_rd),
        .lk_tx_data(r_wdata), .lk_tx_wr(r_wr), .lk_tx_full(r_full), .led(node_led),
        .O_sdram_clk(O_sdram_clk),
        .O_sdram_cas_n(O_sdram_cas_n), .O_sdram_ras_n(O_sdram_ras_n), .O_sdram_wen_n(O_sdram_wen_n),
        .IO_sdram_dq(IO_sdram_dq), .O_sdram_addr(O_sdram_addr), .O_sdram_ba(O_sdram_ba)
`ifdef FLASH_BOOT
        ,.flash_sck(flash_sck_int), .flash_mosi(flash_mosi),
        .flash_miso(flash_miso), .flash_cs_n(flash_cs_n),
        .flash_sck_tsen(sck_tsen),
        .shifts_out(shifts_from_flash), .shifts_valid(shifts_valid_from_flash)
`endif
    );
    assign led = node_led;
endmodule