`timescale 1ns/1ps
// =============================================================================
// gen_seq - autonomous GENERATION sequencer (mono-card, NN=1), LUT-lean.
//
// Forked from ffn_tp_seq2 (register-file vfile, the optimised -49% LUT style) and
// extended step by step into the full generation orchestrator. Built INCREMENTALLY
// with one validated gate per step (docs/PLAN_GEN_SEQUENCER.md).
//
//   STEP A : embedding                                         [validated]
//   STEP B : causal attention block (this file)                <-- here
//     FN(XB,rms_att) -> XN ; FQ(XN)->Q/Kc/Vc ; RR per head on Q,Kc ;
//     write Kc_roped/Vc into kvmem[pos] ; scan max-shift ; MM(T=pos+1) -> ATT ;
//     FQ(ATT,wo) -> OUTV ; residual XB = XB + OUTV.  Ports the validated
//     attn_causal_seq (Phase 5b) into the vfile style (no 512-bit mux).
//
// The 3 Phase-5b pitfalls are avoided here : (1) rx_rdy pulses via rx_ack,
// (2) sKref/sVref seeded at -128 (never read an uninitialised shift),
// (3) MM K/V streamed by incremental counters (no div/mod).
//
// Slot map (vfile) : XB=0 XN=1 Q=2 Kc=3 Vc=4 ATT=5 OUTV=6 . Each slot is D=64
// bytes (Kc/Vc use only the first KVW=32). KV cache in a separate BSRAM (kvmem).
// =============================================================================
(* nomem2reg *)   // keep logits/vfile as $mem -> DP16KD BRAM (not frontend registers).
                   // (KV cache kmem/vmem a été déplacé en SDRAM externe @ KV_BASE, étape 1.)
                   // gen_seq uses registered reads (no same-cycle read-during-write), so safe.
                   // Without this, the yosys frontend mem2reg-converts these BEFORE the bram
                   // pass, so ram_style="block" never applies -> 58k LUT (vs 17k on Gowin).
module gen_seq #(
    parameter W = 8,
    parameter D = 64,
    parameter H = 8, parameter KH = 4, parameter HS = 8,
    parameter HID = 172,                 // FFN hidden_dim (chunké en NCHUNK_F blocs [D,D])
    parameter TMAX = 32,
    parameter BOOT = 40000,
    parameter NL   = 5,                   // number of transformer layers
    parameter LBASE = 23'h010000,         // layer 0 weight base (base_l = LBASE + layer*LSTRIDE)
    parameter LSTRIDE = 23'h010000,       // per-layer SDRAM stride (>= taille couche, pow2)
    // per-layer weight offsets (stories260K SDRAM layout ; dériver pour dim 128+ = TODO)
    parameter OFF_RMS   = 23'h000000,
    parameter OFF_WQ    = 23'h000100,
    parameter OFF_WK    = 23'h001100,
    parameter OFF_WV    = 23'h001900,
    parameter OFF_WO    = 23'h002100,
    parameter OFF_RMSFF = 23'h003100,
    parameter OFF_W1    = 23'h003200,     // W1 [HIDPAD,D], chunk c @ +c*(D*D)
    parameter OFF_W3    = 23'h006200,     // W3 [HIDPAD,D], chunk c @ +c*(D*D)
    parameter OFF_W2    = 23'h009200,     // W2 chunk c = [D,D] @ +c*(D*D)
    // lm_head (after the 5 layers) : fixed addresses, not per-layer
    parameter VOCAB  = 512,
    parameter NCHUNK = 8,                 // VOCAB/64
    parameter A_RMSFINAL = 23'h060000,    // final rmsnorm weight
    parameter A_EMB      = 23'h000000,    // tok_emb (shared classifier), chunk c @ +c*0x1000
    parameter NPOS  = 17,                 // generation length (token loop)
    parameter NPR   = 16                  // max prompt tokens for prefill (prompt_len=0 -> BOS)
) (
    input  wire            clk, rst_n, start,
    input  wire            gen_mode,      // 0 = single-shot (external x_in), 1 = autonomous generation
    input  wire [W*D-1:0]  x_in,          // step-B: external x (bypasses embed)
    input  wire signed [7:0] sx_in,
    input  wire [NL*8-1:0] sw_rms, swq, swk, swv, swo,      // per-layer weight shifts
    input  wire [NL*8-1:0] sw_rmsf, sw1, sw3, sw2,         // (selected by current layer)
    input  wire signed [7:0] sw_rmsfinal, sw_emb,          // lm_head shifts (single, shared)
    input  wire [5:0]      pos,           // single-shot mode : fixed position
    input  wire [22:0]     base,
    // prefill : prompt token IDs (packed) + count. prompt_len=0 -> BOS hardcodé (backward compat).
    input  wire [NPR*10-1:0] prompt,
    input  wire [6:0]        prompt_len,
    output wire [7:0]      lk_cmd_data,
    output wire            lk_cmd_wr,
    input  wire            lk_cmd_full,
    input  wire [7:0]      lk_resp_data,
    input  wire            lk_resp_empty,
    output wire            lk_resp_rd,
    output reg  [W*D-1:0]  result,        // XB after the 5 layers (captured before lm_head)
    output reg  signed [7:0] result_sh,
    output reg  [9:0]      token,         // argmax of the lm_head logits (= token_out)
    output reg             token_valid,   // 1-cycle pulse when a token is emitted (gen mode)
    output reg             done,
    // ---- KV cache maître SDRAM direct (étape 1 KV→SDRAM, plan §2.2) ----
    // kmem/vmem (BRAM 10 Ko, 6 DP16KD) déplacés en SDRAM @ KV_BASE. ksh/vsh
    // restent on-chip (re-align 1 cyc). Interface byte, non-burst : pulse rd/wr +
    // addr, attendre !busy, dout est buffered jusqu'au prochain read. Latence 4-5
    // cyc/byte, mais PH_MM est ss_link-bound (~2-3 cyc/byte) -> SDRAM naive reste
    // <0.2% du cycle token, pas de prefetch nécessaire (cf note mémoire).
    output reg  [22:0]     kv_addr,
    output reg             kv_rd,
    output reg             kv_wr,
    output reg  [7:0]      kv_din,
    input  wire [7:0]      kv_dout,
    input  wire            kv_busy,
    output reg             kv_refresh      // preemption refresh (étape 1) : le seq compte
                                           // le temps passé sur le bus KV (PH_MM/KVWR)
                                           // et pulse lui-même un refresh SDRAM toutes
                                           // les ~15µs pour ne pas perdre la SDRAM. Le
                                           // node route kv_refresh vers eff_ref (1.1c).
);
    localparam KVW  = KH*HS;              // 32
    localparam NREP = H/KH;
    localparam NSLOTS = 8;
    // FFN chunking : HID padded au multiple de D supérieur, chunké en NCHUNK_F blocs [D,D]
    // (172 -> 192 = 3 chunks pour dim 64). Cohérent avec model_cfg.vh / node_top.wcache.
    localparam HIDPAD  = ((HID+D-1)/D)*D;
    localparam NCHUNK_F= HIDPAD/D;        // 3 pour stories260K
    localparam SW = 3;
    localparam AW = SW + 6;
    // Session 6b lever #2 (vfile fusion SB_P->SB_Vc, 8->7 slots) was tried and REVERTED :
    // it cut cell count (Logic -411, SSRAM -48) but WORSENED routing (789->931 unrouted)
    // by concentrating V/P accesses into one congested distributed-RAM hotspot. The node
    // is congestion-bound (Logic 84%), not capacity-bound, so fusing storage is harmful.
    localparam [SW-1:0] SB_XB=0, SB_XN=1, SB_Q=2, SB_Kc=3, SB_Vc=4, SB_ATT=5, SB_OUTV=6, SB_P=7;
    // FFN reuses the attention slots once attention is done : XN=rmsnorm,
    // H1=Q, H3=Kc, SG=Vc, HG=ATT, W2-accumulator=OUTV, W2-partial=P, residual=XB.

    // ---- register file (LUT-RAM) ----
    reg [7:0] vfile [0:NSLOTS*D-1];
    reg [AW-1:0] vaddr; reg [7:0] vdin; reg vwe;
    function [AW-1:0] vidx; input [SW-1:0] slot; input [5:0] b; vidx = (slot<<6)|b; endfunction

    // ---- KV cache : K/V en SDRAM externe (étape 1, plan §2.2), shifts on-chip ----
    // kmem/vmem (BSRAM 10 Ko, 6 DP16KD) déplacés en SDRAM @ KV_BASE. Le seq est 4e
    // maître du mux eff_* dans node_top (1.1c). Layout SDRAM, K/V contigus par
    // position : KV_BASE + (layer*TMAX+pos)*2*KVW + (mmv?KVW:0) + mmo. Prefetch
    // non nécessaire (cf port kv_* + note mémoire) : SDRAM naive 5 cyc/byte reste
    // <0.2% du cycle token car PH_MM est ss_link-bound. Le re-align
    // clip8(kraw >>> (sKref - ksh[...])) reste dans le seq (ksh/vsh on-chip, 1 cyc).
    localparam [22:0] KV_BASE = 23'h070000;   // après miroir poids 0x000000-0x060040
    // 10 Ko actuels (NL*TMAX*2*KVW). 7.5 Mo libres -> TMAX>>32 possible.
    reg signed [7:0] ksh  [0:NL*TMAX-1];      // per-(layer,pos) K shift (re-align)
    reg signed [7:0] vsh  [0:NL*TMAX-1];      // per-(layer,pos) V shift (re-align)

    // ---- lm_head : VOCAB logits (too big for a vfile slot) + per-chunk shift ----
    // In BSRAM (frees ~256 RAM16 of congested CLS). The argmax read applies a shift
    // (re-align), so like the KV cache it uses a clean registered read (lraw) + a
    // 1-cycle-delayed compare (states AMAXC/AMAXW) -> BSRAM output register usable.
    (* ram_style = "block" *) reg signed [7:0] logits [0:VOCAB-1];
    reg signed [7:0] lraw;                    // registered logits[amax_i] (BSRAM read)
    reg signed [7:0] csh [0:NCHUNK-1];
    reg [3:0] lmchunk;                       // lm_head tok_emb chunk 0..NCHUNK-1
    reg signed [7:0] sref_lm;                // max chunk shift (argmax re-align)
    reg signed [31:0] best_val;
    reg [9:0] amax_i, best_idx;
    reg [3:0] scan_c;

    // ---- ss_link byte adapters ----
    reg  [7:0] tx_data; reg tx_send; wire tx_busy;
    wire [7:0] rx_data; wire rx_valid; wire rx_rdy;
    tx8_link u_tx (.clk(clk), .rst(~rst_n), .data(tx_data), .send(tx_send),
                   .busy(tx_busy), .o_data(lk_cmd_data), .o_wr(lk_cmd_wr), .i_full(lk_cmd_full));
    rx8_link u_rx (.clk(clk), .rst(~rst_n), .data(rx_data), .valid(rx_valid),
                   .rdy(rx_rdy), .i_data(lk_resp_data), .i_empty(lk_resp_empty), .o_rd(lk_resp_rd));

    // ---- serialized elementwise ALU (FFN MUL, W2 reduce ADD, residual ADD) ----
    reg            alu_start, alu_op;
    reg  [1:0]     alu_next;                  // 0=MUL->HG, 1=reduce->OUTV, 2=residual->XB
    reg  [W*D-1:0] alu_a, alu_b; reg signed [7:0] alu_sa, alu_sb;
    wire [W*D-1:0] alu_out; wire signed [7:0] alu_out_sh; wire alu_done;
    reg  [SW-1:0]  alu_aslot, alu_bslot, alu_dslot;
    vec_alu2 #(W, D) u_alu (.clk(clk), .rst_n(rst_n), .start(alu_start), .op(alu_op),
        .a(alu_a), .sa(alu_sa), .b(alu_b), .sb(alu_sb),
        .out(alu_out), .out_sh(alu_out_sh), .done(alu_done));

    reg [7:0] pkt [0:79];
    reg [11:0] pkt_len, resp_len, idx, rcnt;   // MM packet grows to ~1158B at pos=16
    reg [6:0] ldi;
    reg [23:0] bcnt;

    reg signed [7:0] sx0, sxn, sQ, sK, sV, sA, sOut;
    // FFN running/per-chunk shifts + chunk counter
    reg signed [7:0] sxb, s1c, s3c, ssgc, shc, spc, sov;
    reg [2:0] fc;                            // ffn hidden chunk 0..NCHUNK_F-1 (D-wide each)
    reg in_ffn;                              // 0 while attention, 1 during FFN
    reg [22:0] waddr;                        // blocking temp : chunked weight address
    // FFN hidden dim HID is chunked NCHUNK_F blocks of D with W1/W3 rows HID..HIDPAD-1
    // zero-padded (test preload) so every chunk is a clean D-wide op : silu(0)=0, no
    // stale value contaminates the per-chunk output shift, and W2 cols HID..HIDPAD-1 are 0.
    reg [2:0] layer;                         // current transformer layer 0..NL-1
    wire [22:0] base_l = base + LBASE + layer*LSTRIDE;   // = base + LBASE + layer*LSTRIDE
    reg [9:0] cur_tok;                       // current token fed to EMBED (gen mode)
    reg [5:0] pos_reg;                       // effective position (internal counter in gen mode)
    // prefill : prompt_len tokens ingérés sans émission (embed+KVWR), puis émission au
    // dernier token du prompt = premier token généré. prefill_left = tokens prompt
    // restant à traiter APRÈS le courant ; prompt_idx = prochain index dans `prompt`.
    reg [6:0] prefill_left;
    reg [4:0] prompt_idx;
    // per-layer weight shift selected by the current layer
    wire signed [7:0] sw_rms_l  = $signed(sw_rms [layer*8 +: 8]);
    wire signed [7:0] swq_l     = $signed(swq    [layer*8 +: 8]);
    wire signed [7:0] swk_l     = $signed(swk    [layer*8 +: 8]);
    wire signed [7:0] swv_l     = $signed(swv    [layer*8 +: 8]);
    wire signed [7:0] swo_l     = $signed(swo    [layer*8 +: 8]);
    wire signed [7:0] sw_rmsf_l = $signed(sw_rmsf[layer*8 +: 8]);
    wire signed [7:0] sw1_l     = $signed(sw1    [layer*8 +: 8]);
    wire signed [7:0] sw3_l     = $signed(sw3    [layer*8 +: 8]);
    wire signed [7:0] sw2_l     = $signed(sw2    [layer*8 +: 8]);

    function signed [7:0] clip8; input signed [31:0] v;
        clip8 = (v > 127) ? 8'sd127 : (v < -128) ? -8'sd128 : v[7:0]; endfunction

    // phases : attention (FN..WO), FFN (FN2, W1, W3, SS, W2), lm_head (FNF, LM)
    localparam [3:0] PH_FN=0, PH_WQ=1, PH_WK=2, PH_WV=3, PH_MM=4, PH_WO=5,
                     PH_FN2=6, PH_FW1=7, PH_FW3=8, PH_SS=9, PH_FW2=10,
                     PH_FNF=11, PH_LM=12, PH_EE=13;
    reg [3:0] phase;
    reg [3:0] src_slot, dst_slot;
    reg [7:0] Nfq;                          // FQ output count for current phase

    localparam IDLE=0, LOADX=1, BWAIT=2, BUILD=3, E_SET=4, E_BUSY=5, E_DONE=6, RECV=7,
               ROPEQ=8, ROPEK=9, KVWR=10, SCAN=11, RESID=12, DONE_ST=13,
               ALD_A=14, ALD_B=15, ALU_WAIT=16, AST_A=17, FRED_COPY=18,
               AMAX_INIT=19, AMAX=20, E_MMRD=21, AMAXC=22, AMAXW=23,
               // KV SDRAM (étape 1) : read FSM (pulse rd -> busy=1 -> !busy -> E_MMRD)
               // + write FSM KVWR (phase K contigu ldi 0..31 @ off ldi, puis phase V
               //   contigu ldi 0..31 @ off 32+ldi). Contigu obligatoire : shadow-word
               //   sdram.v reset sh_word quand addr[22:2] change + DQM=0 écrit tout le
               //   mot -> interleaved K/V (mots différents) clobberait 3 bytes/mot.
               E_MREQ=24, E_MBUSY=25, E_MDONE=26,
               KVWR_KB=27, KVWR_KD=28, KVWR_VB=29, KVWR_VD=30,
               // preemption refresh : le seq compte le temps sur le bus KV et pulse
               // lui-même un refresh SDRAM (~15µs) aux points de cession E_MREQ/KVWR/V.
               E_MREFRESH=31, E_MREF_BUSY=32, E_MREF_DONE=33, KVWR_V=34;
    reg [5:0] st;              // élargi à 6 bits (35 états > 32)
    reg [1:0] kv_refresh_ret;  // 0 = read (E_MREQ), 1 = write K (KVWR), 2 = write V (KVWR_V)
    // compteur temps bus KV (PH_MM/KVWR) -> refresh auto toutes les ~15µs (375 cyc @25MHz)
    reg [8:0]  kv_refresh_cnt;
    wire       kv_refresh_due = (kv_refresh_cnt >= 9'd375);
    wire       kv_window = (phase==PH_MM) || (st==KVWR) || (st==KVWR_KB) ||
                           (st==KVWR_KD) || (st==KVWR_VB) || (st==KVWR_VD) ||
                           (st==E_MREFRESH) || (st==E_MREF_BUSY) || (st==E_MREF_DONE);

    // rx one-byte-per-ack handshake
    reg rx_ack;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) rx_ack <= 1'b1; else rx_ack <= ~rx_valid;
    assign rx_rdy = (st == RECV) & rx_ack;

    // KV refresh auto (étape 1) : compte le temps passé sur le bus KV (PH_MM/KVWR).
    // Reset après un refresh (kv_refresh pulse) ou hors fenêtre KV. Quand >= 15µs,
    // la FSM cède (E_MREFRESH) avant sa prochaine requête -> refresh SDRAM, sinon la
    // SDRAM perd ses données (PH_MM/KVWR tiennent le bus > 15µs à 25 MHz).
    always @(posedge clk or negedge rst_n)
        if (!rst_n)                       kv_refresh_cnt <= 9'd0;
        else if (kv_refresh)              kv_refresh_cnt <= 9'd0;   // refresh parti
        else if (!kv_window)              kv_refresh_cnt <= 9'd0;   // hors fenêtre
        else if (kv_refresh_cnt != 9'd511) kv_refresh_cnt <= kv_refresh_cnt + 9'd1;

    function [7:0] a0; input [22:0] a; a0=a[7:0]; endfunction
    function [7:0] a1; input [22:0] a; a1=a[15:8]; endfunction
    function [7:0] a2; input [22:0] a; a2=a[22:16]; endfunction

    // rope one head : reads 8 bytes at (slot, h*8), writes them back roped
    reg [6:0] hidx;                          // head index during rope
    reg [3:0] rope_slot;                     // slot being roped (Q or Kc)
    reg signed [7:0] rh [0:HS-1];            // head bytes being roped
    reg [2:0] rcp;                           // rope compute pointer : one rotation/cycle
    reg rope_rd;                             // rope 2-phase : 0=present ROM addr, 1=compute w/ crom/srom

    // max shift scan + MM streaming counters
    reg signed [7:0] sKref, sVref;
    reg [5:0] scan_i;
    reg [5:0] mmp; reg [5:0] mmo; reg mmv;   // MM K/V stream position/offset/region

    // KV SDRAM read (étape 1) : kraw/vraw sont chargés par la FSM read (E_MREQ->
    // E_MBUSY->E_MDONE) depuis kv_dout, au lieu du registered BRAM read. L'adresse
    // du byte courant du stream (mmp,mmo,mmv) ; K et V contigus, V décalé de KVW.
    reg signed [7:0] kraw, vraw;            // byte KV courant (re-align dans E_MMRD)
    // BSRAM read pipeline for logits : lraw lags amax_i by one cycle (see AMAXC/AMAXW).
    always @(posedge clk) lraw <= logits[amax_i];

    // ---- rope freq_cis : two small BSRAM ROMs, addressed by (pos, pair) ----
    // Replaces the flat 1088-bit cos_q15/sin_q15 ports : their variable part-select
    // cos_q15[pos_reg*FW + 16*rcp +: 16] synthesised into two 68:1 muxes + a high-fanout
    // 1088-bit bus = the GW2AR-18C routing congestion (Session 6b, lever #1). idx =
    // pos*(HS/2) + rcp. Registered read (crom/srom) -> Gowin maps to BSRAM ; the rope is
    // already sequenced, so the +1 read latency is a clean 2-phase step (rope_rd) that
    // leaves the roped bytes IDENTICAL to the old combinational read.
    (* ram_style = "block" *) reg signed [15:0] cos_rom [0:NPOS*(HS/2)-1];
    (* ram_style = "block" *) reg signed [15:0] sin_rom [0:NPOS*(HS/2)-1];
    initial begin
        $readmemh("freq_cis_cos.hex", cos_rom);
        $readmemh("freq_cis_sin.hex", sin_rom);
    end
    wire [6:0] rope_ridx = pos_reg*(HS/2) + rcp;
    reg signed [15:0] crom, srom;
    always @(posedge clk) begin crom <= cos_rom[rope_ridx]; srom <= sin_rom[rope_ridx]; end

    integer j;

`ifdef SIM
    // Sim-only : initialise les mémoires interne (vfile/ksh/vsh/logits/csh) a 0 pour
    // matcher l'init matérielle ECP5 (BRAM/LUTRAM default 0). Icarus laisse les
    // `reg [...] mem [...]` a X ; or certaines cases sont lues avant d'être écrites
    // dans la cascade attention->residual->rmsnorm_ffn (cf flash-boot-mock-sim :
    // x0=x a W1, XN=x final -> lm_head logits=X -> token 0). En silicium ces cases
    // default a 0 (ECP5), donc la cascade est stable ; l'init 0 sim reproduit ça.
    integer sim_gi;
    initial begin
        for (sim_gi = 0; sim_gi < NSLOTS*D; sim_gi = sim_gi + 1) vfile[sim_gi] = 8'd0;
        for (sim_gi = 0; sim_gi < NL*TMAX; sim_gi = sim_gi + 1) begin
            ksh[sim_gi] = 8'sd0; vsh[sim_gi] = 8'sd0;
        end
        for (sim_gi = 0; sim_gi < VOCAB; sim_gi = sim_gi + 1) logits[sim_gi] = 8'sd0;
        for (sim_gi = 0; sim_gi < NCHUNK; sim_gi = sim_gi + 1) csh[sim_gi] = 8'sd0;
    end
`endif

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; tx_send<=0; idx<=0; rcnt<=0; bcnt<=0; phase<=PH_FN;
            ldi<=0; vwe<=0; hidx<=0; mmp<=0; mmo<=0; mmv<=0; rcp<=0; rope_rd<=0;
            alu_start<=0; fc<=0; in_ffn<=0; layer<=0; token<=0;
            cur_tok<=10'd1; pos_reg<=6'd0; token_valid<=1'b0;
            prefill_left<=7'd0; prompt_idx<=5'd0;
            kv_rd<=0; kv_wr<=0; kv_addr<=0; kv_din<=0; kv_refresh<=0; kv_refresh_ret<=0;
        end else begin
            tx_send<=0; done<=0; vwe<=0; alu_start<=0; token_valid<=1'b0;
            kv_rd<=0; kv_wr<=0;     // pulses 1-cycle (pattern v3)
            kv_refresh<=0;          // pulse 1-cycle (refresh SDRAM)
            if (vwe) vfile[vaddr] <= vdin;

            case (st)
                // load external x into vfile[XB], then start the attention block
                IDLE: if (start) begin sx0<=sx_in; sxn<=sx_in; sxb<=sx_in;
                        fc<=0; in_ffn<=0; layer<=0; bcnt<=0; ldi<=0;
                        if (prompt_len != 7'd0) begin
                            // prefill : embed the first prompt token, no emission until the last one
                            cur_tok      <= prompt[0*10 +: 10];
                            prefill_left <= prompt_len - 7'd1;
                            prompt_idx   <= 5'd1;
                        end else begin
                            cur_tok      <= 10'd1;            // BOS hardcodé (backward compat)
                            prefill_left <= 7'd0;
                            prompt_idx   <= 5'd0;
                        end
                        if (gen_mode) begin phase<=PH_EE; pos_reg<=6'd0; st<=BWAIT; end  // autonomous : embed first
                        else          begin phase<=PH_FN; pos_reg<=pos;   st<=LOADX; end // single-shot : external x
                    end
                LOADX: begin vaddr<=vidx(SB_XB, ldi[5:0]); vdin<=x_in[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; st<=BWAIT; end else ldi<=ldi+1; end
                BWAIT: if (bcnt==BOOT) st<=BUILD; else bcnt<=bcnt+1;

                BUILD: begin
                    case (phase)
                        PH_FN: begin
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sxb;pkt[3]<=sw_rms_l;
                            pkt[68]<=a0(base_l+OFF_RMS);pkt[69]<=a1(base_l+OFF_RMS);pkt[70]<=a2(base_l+OFF_RMS);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_WQ: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=H*HS;pkt[3]<=sxn;pkt[4]<=swq_l;
                            pkt[69]<=a0(base_l+OFF_WQ);pkt[70]<=a1(base_l+OFF_WQ);pkt[71]<=a2(base_l+OFF_WQ);
                            pkt_len<=72; resp_len<=3+H*HS; src_slot<=SB_XN; dst_slot<=SB_Q; Nfq<=H*HS; end
                        PH_WK: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swk_l;
                            pkt[69]<=a0(base_l+OFF_WK);pkt[70]<=a1(base_l+OFF_WK);pkt[71]<=a2(base_l+OFF_WK);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Kc; Nfq<=KH*HS; end
                        PH_WV: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=KH*HS;pkt[3]<=sxn;pkt[4]<=swv_l;
                            pkt[69]<=a0(base_l+OFF_WV);pkt[70]<=a1(base_l+OFF_WV);pkt[71]<=a2(base_l+OFF_WV);
                            pkt_len<=72; resp_len<=3+KH*HS; src_slot<=SB_XN; dst_slot<=SB_Vc; Nfq<=KH*HS; end
                        PH_MM: begin
                            pkt[0]<="M";pkt[1]<="M";pkt[2]<=sQ;pkt[3]<=sKref;pkt[4]<=sVref;
                            pkt[5]<=({2'd0,pos_reg}+8'd1);
                            // Q from vfile[Q], then K/V streamed from kvmem
                            pkt_len<=6+D+2*(({2'd0,pos_reg}+12'd1)*KVW); resp_len<=3+D;
                            src_slot<=SB_Q; end
                        PH_WO: begin
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sA;pkt[4]<=swo_l;
                            pkt[69]<=a0(base_l+OFF_WO);pkt[70]<=a1(base_l+OFF_WO);pkt[71]<=a2(base_l+OFF_WO);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_ATT; dst_slot<=SB_OUTV; Nfq<=D[7:0]; end
                        // ---- FFN ----
                        PH_FN2: begin  // rmsnorm(XB, rms_ffn) -> XN
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sxb;pkt[3]<=sw_rmsf_l;
                            pkt[68]<=a0(base_l+OFF_RMSFF);pkt[69]<=a1(base_l+OFF_RMSFF);pkt[70]<=a2(base_l+OFF_RMSFF);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_FW1: begin  // W1_chunk(XN) -> H1 (SB_Q), N_out=64 (zero-padded)
                            waddr = base_l + OFF_W1 + fc*(D*D);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw1_l;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_XN; dst_slot<=SB_Q; Nfq<=D[7:0]; end
                        PH_FW3: begin  // W3_chunk(XN) -> H3 (SB_Kc)
                            waddr = base_l + OFF_W3 + fc*(D*D);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw3_l;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_XN; dst_slot<=SB_Kc; Nfq<=D[7:0]; end
                        PH_SS: begin   // silu(H1) -> SG (SB_Vc)
                            pkt[0]<="S";pkt[1]<="S";pkt[2]<=s1c;
                            pkt_len<=67; resp_len<=70; src_slot<=SB_Q; dst_slot<=SB_Vc; Nfq<=D[7:0]; end
                        // ---- lm_head ----
                        PH_FNF: begin  // final rmsnorm(XB, rms_final) -> XN
                            pkt[0]<="F";pkt[1]<="N";pkt[2]<=sxb;pkt[3]<=sw_rmsfinal;
                            pkt[68]<=a0(A_RMSFINAL+base);pkt[69]<=a1(A_RMSFINAL+base);pkt[70]<=a2(A_RMSFINAL+base);
                            pkt_len<=71; resp_len<=75; src_slot<=SB_XB; dst_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_LM: begin   // FQ(XN, tok_emb chunk lmchunk) -> logits chunk
                            waddr = A_EMB + base + lmchunk*(D*D);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=sxn;pkt[4]<=sw_emb;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_XN; Nfq<=D[7:0]; end
                        PH_EE: begin   // embedding lookup : EE(cur_tok) -> XB (resp 'E''K' x[64])
                            pkt[0]<="E";pkt[1]<="E";pkt[2]<=cur_tok[7:0];pkt[3]<={6'd0,cur_tok[9:8]};
                            pkt_len<=4; resp_len<=66; src_slot<=SB_XB; dst_slot<=SB_XB; Nfq<=D[7:0]; end
                        default: begin // PH_FW2 : W2_chunk(HG) -> P (SB_P), N_out=64
                            waddr = base_l + OFF_W2 + fc*(D*D);
                            pkt[0]<="F";pkt[1]<="Q";pkt[2]<=D[7:0];pkt[3]<=shc;pkt[4]<=sw2_l;
                            pkt[69]<=a0(waddr);pkt[70]<=a1(waddr);pkt[71]<=a2(waddr);
                            pkt_len<=72; resp_len<=3+D; src_slot<=SB_ATT; dst_slot<=SB_P; Nfq<=D[7:0]; end
                    endcase
                    idx<=0; st<=E_SET;
                end

                // stream packet; vector bytes read combinationally from vfile
                E_SET: begin
                    // KV cache byte : lancer une requête read SDRAM (E_MREQ) pour le
                    // byte (mmp,mmo,mmv) courant ; le re-align + send se font dans
                    // E_MMRD une fois kraw/vraw chargés. Tout le reste envoie maintenant.
                    if (phase==PH_MM && idx>=(6+D)) begin
                        st<=E_MREQ;
                    end else begin
                        if ((phase==PH_FN||phase==PH_FN2||phase==PH_FNF) && idx>=4 && idx<4+D)
                                                                        tx_data<=vfile[vidx(src_slot, idx[5:0]-4)];
                        else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO||
                                  phase==PH_FW1||phase==PH_FW3||phase==PH_FW2||phase==PH_LM) && idx>=5 && idx<5+D)
                                                                        tx_data<=vfile[vidx(src_slot, idx[5:0]-5)];
                        else if (phase==PH_SS && idx>=3 && idx<3+D)     tx_data<=vfile[vidx(src_slot, idx[5:0]-3)];
                        else if (phase==PH_MM && idx>=6 && idx<6+D)     tx_data<=vfile[vidx(SB_Q, idx[5:0]-6)];
                        else tx_data<=pkt[idx];
                        tx_send<=1; st<=E_BUSY;
                    end
                end
                // ---- KV SDRAM read FSM (étape 1, pattern v3) ----
                // E_MREQ : pulse kv_rd + addr du byte (mmp,mmo,mmv). Layout K/V contigus
                // par position : KV_BASE + (layer*TMAX+mmp)*2*KVW + (mmv?KVW:0) + mmo.
                // Si le compteur KV refresh est dû (~15µs), on cède et on pulse kv_refresh
                // AVANT de lire (E_MREFRESH) pour ne pas perdre la SDRAM.
                E_MREQ: begin
                    if (kv_refresh_due) begin
                        kv_refresh_ret <= 2'd0;     // retour E_MREQ après refresh
                        st <= E_MREFRESH;
                    end else begin
                        kv_addr <= KV_BASE + ((layer*TMAX + mmp) << 6)   // *2*KVW (=64)
                                 + (mmv ? 6'd32 : 6'd0) + mmo;            // +KVW si V
                        kv_rd   <= 1'b1;
                        st      <= E_MBUSY;
                    end
                end
                // E_MREFRESH : pulse kv_refresh (1 cycle) -> le node le route vers eff_ref.
                E_MREFRESH: begin
                    kv_refresh <= 1'b1;
                    st <= E_MREF_BUSY;
                end
                // E_MREF_BUSY / E_MREF_DONE : pattern v3 (busy=1 -> !busy), puis retour.
                E_MREF_BUSY: if (kv_busy) st<=E_MREF_DONE;
                E_MREF_DONE: if (!kv_busy) st <= (kv_refresh_ret==2'd1) ? KVWR :
                                                          (kv_refresh_ret==2'd2) ? KVWR_V : E_MREQ;
                // E_MBUSY : attend que le contrôleur SDRAM prenne la commande (busy=1).
                E_MBUSY: if (kv_busy) st<=E_MDONE;
                // E_MDONE : opération finie (!busy) -> kv_dout est buffered/valide ;
                //           capturer dans kraw (K) ou vraw (V), puis envoyer dans E_MMRD.
                E_MDONE: if (!kv_busy) begin
                    if (!mmv) kraw <= $signed(kv_dout);
                    else      vraw <= $signed(kv_dout);
`ifdef SIM_DEBUG
                    if (mmo==3'd0)
                        $display("[SIM] E_MDONE mmv=%0d mmp=%0d mmo0 kv_dout=%0d",
                                 mmv, mmp, $signed(kv_dout));
`endif
                    st <= E_MMRD;
                end
                // KV cache byte : kraw/vraw now hold the SDRAM byte at the current
                // (mmp,mmo,mmv) -> apply the per-position re-align shift and send.
                E_MMRD: begin
                    if (!mmv) tx_data <= clip8( $signed(kraw) >>> (sKref - ksh[layer*TMAX + mmp]) );
                    else      tx_data <= clip8( $signed(vraw) >>> (sVref - vsh[layer*TMAX + mmp]) );
                    tx_send<=1; st<=E_BUSY;
                end
                E_BUSY: if (tx_busy) st<=E_DONE;
                E_DONE: if (!tx_busy) begin
                    if (idx==pkt_len-1) begin rcnt<=0; st<=RECV; end
                    else begin
                        idx<=idx+1;
                        if (phase==PH_MM && idx>=(6+D)) begin
                            if (mmo==KVW-1) begin mmo<=0;
                                if (mmp==pos_reg) begin mmp<=0; mmv<=1'b1; end else mmp<=mmp+1;
                            end else mmo<=mmo+1;
                        end
                        st<=E_SET;
                    end
                end

                // stream response; write vector bytes into dst_slot
                RECV: if (rx_valid) begin
`ifdef SIM_DEBUG
                    if (rcnt==2 || rcnt==3)
                        $display("[SIM] RECV phase=%0d lmchunk=%0d rcnt=%0d rx_data=%0d",
                                 phase, lmchunk, rcnt, $signed(rx_data));
`endif
                    if ((phase==PH_FN||phase==PH_FN2||phase==PH_FNF) && rcnt>=11 && rcnt<11+D)
                                                                  begin vaddr<=vidx(SB_XN, rcnt[5:0]-11); vdin<=rx_data; vwe<=1'b1; end
                    else if ((phase==PH_WQ||phase==PH_WK||phase==PH_WV||phase==PH_WO||
                              phase==PH_FW1||phase==PH_FW3||phase==PH_FW2) && rcnt>=3 && rcnt<3+Nfq)
                                                                  begin vaddr<=vidx(dst_slot, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_SS && rcnt>=6 && rcnt<6+D) begin vaddr<=vidx(dst_slot, rcnt[5:0]-6); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_MM && rcnt>=3 && rcnt<3+D) begin vaddr<=vidx(SB_ATT, rcnt[5:0]-3); vdin<=rx_data; vwe<=1'b1; end
                    else if (phase==PH_EE && rcnt>=2 && rcnt<2+D) begin vaddr<=vidx(SB_XB, rcnt[5:0]-2); vdin<=rx_data; vwe<=1'b1; end
                    // lm_head chunk logits go to the separate logits[] array (not vfile)
                    if (phase==PH_LM && rcnt>=3 && rcnt<3+D) logits[lmchunk*64 + (rcnt-3)] <= $signed(rx_data);
                    case (phase)
                        PH_FN:  if (rcnt==2) sxn<=$signed(rx_data);
                        PH_WQ:  if (rcnt==2) sQ<=$signed(rx_data);
                        PH_WK:  if (rcnt==2) sK<=$signed(rx_data);
                        PH_WV:  if (rcnt==2) sV<=$signed(rx_data);
                        PH_MM:  if (rcnt==2) sA<=$signed(rx_data);
                        PH_WO:  if (rcnt==2) sOut<=$signed(rx_data);
                        PH_FN2: if (rcnt==2) sxn<=$signed(rx_data);
                        PH_FW1: if (rcnt==2) s1c<=$signed(rx_data);
                        PH_FW3: if (rcnt==2) s3c<=$signed(rx_data);
                        PH_SS:  if (rcnt==2) ssgc<=$signed(rx_data);
                        PH_FW2: if (rcnt==2) spc<=$signed(rx_data);
                        PH_FNF: if (rcnt==2) sxn<=$signed(rx_data);
                        PH_LM:  if (rcnt==2) csh[lmchunk]<=$signed(rx_data);
                    endcase
                    if (rcnt==resp_len-1) begin
                        case (phase)
                            PH_FN:  begin phase<=PH_WQ; st<=BUILD; end
                            PH_WQ:  begin phase<=PH_WK; st<=BUILD; end
                            PH_WK:  begin phase<=PH_WV; st<=BUILD; end
                            PH_WV:  begin hidx<=0; rope_slot<=SB_Q; st<=ROPEQ; end
                            PH_MM:  begin phase<=PH_WO; st<=BUILD; end
                            PH_WO:  begin  // attention residual : XB = XB + OUTV (requantizing add)
                                    alu_aslot<=SB_XB; alu_bslot<=SB_OUTV; alu_dslot<=SB_XB;
                                    alu_op<=1'b1; alu_sa<=sxb; alu_sb<=sOut; alu_next<=2'd2;
                                    ldi<=0; st<=ALD_A; end
                            PH_FN2: begin phase<=PH_FW1; fc<=0; st<=BUILD; end
                            PH_FW1: begin phase<=PH_FW3; st<=BUILD; end
                            PH_FW3: begin phase<=PH_SS; st<=BUILD; end
                            PH_SS:  begin  // MUL : SG(Vc) * H3(Kc) -> HG(ATT)
                                    alu_aslot<=SB_Vc; alu_bslot<=SB_Kc; alu_dslot<=SB_ATT;
                                    alu_op<=1'b0; alu_sa<=ssgc; alu_sb<=s3c; alu_next<=2'd0;
                                    ldi<=0; st<=ALD_A; end
                            PH_EE:  begin sxb<=sw_emb; phase<=PH_FN; st<=BUILD; end   // embed done -> attention
                            PH_FNF: begin phase<=PH_LM; lmchunk<=0; st<=BUILD; end
                            PH_LM:  if (lmchunk==NCHUNK-1) st<=AMAX_INIT;   // all chunks done -> argmax
                                    else begin lmchunk<=lmchunk+1; st<=BUILD; end
                            default: begin  // PH_FW2 : reduce P into OUTV accumulator
                                    if (fc==2'd0) begin ldi<=0; st<=FRED_COPY; end
                                    else begin alu_aslot<=SB_OUTV; alu_bslot<=SB_P; alu_dslot<=SB_OUTV;
                                               alu_op<=1'b1; alu_sa<=sov; alu_sb<=spc; alu_next<=2'd1;
                                               ldi<=0; st<=ALD_A; end
                            end
                        endcase
                    end else rcnt<=rcnt+1;
                end

                // rope Q heads (H) then K heads (KH). Two-phase per head to avoid
                // read-after-write corruption : ldi==0 computes ALL 8 roped bytes into
                // rh from the ORIGINAL vfile bytes (rope preserves position, so writing
                // must not feed back into the compute). ldi 0..7 then write rh serially.
                // The compute reads vfile at hidx*8; the writes at hidx*8+ldi only start
                // taking effect from ldi>=1, but rh already holds the full head -> safe.
                // Compute ONE complex rotation per cycle (rcp=0..HS/2-1) into rh, then
                // write rh->vfile serially (ldi phase). rh is separate from vfile, so
                // spreading the 4 rotations over 4 cycles produces the SAME roped bytes
                // as the old unrolled form -- but the rotation multiplier is now SHARED
                // (one rotation of hardware, reused) instead of 4 unrolled in parallel,
                // which is what keeps the design inside the 12-DSP-block budget.
                ROPEQ: begin
                    if (ldi==0) begin: rqc
                        reg signed [7:0] xr, xi;
                        reg signed [31:0] nr, ni;
                        // 2-phase : rope_rd==0 presents the ROM address (crom/srom latch
                        // cos_rom/sin_rom[rope_ridx] this cycle) ; rope_rd==1 computes with
                        // the registered crom/srom -- byte-identical to the old direct read.
                        if (!rope_rd) rope_rd <= 1'b1;
                        else begin
                            xr = $signed(vfile[vidx(SB_Q, hidx[2:0]*HS + 2*rcp)]);
                            xi = $signed(vfile[vidx(SB_Q, hidx[2:0]*HS + 2*rcp + 1)]);
                            nr = (xr*crom - xi*srom + 32'sd16384) >>> 15;
                            ni = (xr*srom + xi*crom + 32'sd16384) >>> 15;
                            rh[2*rcp]   <= clip8(nr); rh[2*rcp+1] <= clip8(ni);
                            rope_rd <= 1'b0;
                            if (rcp==HS/2-1) begin rcp<=0; ldi<=1; end else rcp<=rcp+1;
                        end
                    end else begin
                        vaddr<=vidx(SB_Q, hidx[2:0]*HS + (ldi-1)); vdin<=rh[ldi-1]; vwe<=1'b1;
                        if (ldi==HS) begin ldi<=0;
                            if (hidx==H-1) begin hidx<=0; st<=ROPEK; end
                            else hidx<=hidx+1;
                        end else ldi<=ldi+1;
                    end
                end
                ROPEK: begin   // same 2-phase registered-ROM scheme as ROPEQ, on Kc
                    if (ldi==0) begin: rkc
                        reg signed [7:0] xr, xi;
                        reg signed [31:0] nr, ni;
                        if (!rope_rd) rope_rd <= 1'b1;
                        else begin
                            xr = $signed(vfile[vidx(SB_Kc, hidx[2:0]*HS + 2*rcp)]);
                            xi = $signed(vfile[vidx(SB_Kc, hidx[2:0]*HS + 2*rcp + 1)]);
                            nr = (xr*crom - xi*srom + 32'sd16384) >>> 15;
                            ni = (xr*srom + xi*crom + 32'sd16384) >>> 15;
                            rh[2*rcp]   <= clip8(nr); rh[2*rcp+1] <= clip8(ni);
                            rope_rd <= 1'b0;
                            if (rcp==HS/2-1) begin rcp<=0; ldi<=1; end else rcp<=rcp+1;
                        end
                    end else begin
                        vaddr<=vidx(SB_Kc, hidx[2:0]*HS + (ldi-1)); vdin<=rh[ldi-1]; vwe<=1'b1;
                        if (ldi==HS) begin ldi<=0;
                            if (hidx==KH-1) begin hidx<=0; st<=KVWR; end
                            else hidx<=hidx+1;
                        end else ldi<=ldi+1;
                    end
                end

                // copy roped Kc / Vc into KV SDRAM[pos], store shifts (étape 1).
                // KVWR phase K : pulse write K[ldi]@off ldi (ldi 0..31 CONTIGU -> les 4
                //   bytes de chaque mot K sont écrits consécutivement, shadow-word
                //   sdram.v accumule correctement). KVWR_KB busy -> KVWR_KD !busy ->
                //   (ldi++ -> KVWR) ou (ldi==31 -> ldi=0 -> KVWR_V).
                // KVWR_V phase V : pulse write V[ldi]@off 32+ldi (ldi 0..31 CONTIGU,
                //   mots V séparés des mots K -> shadow-word reset au 1er V puis accumule).
                //   KVWR_VB busy -> KVWR_VD !busy -> (ldi++ -> KVWR_V) ou SCAN.
                // IMPORTANT : K puis V contigus (PAS interleaved) — interleaved K/V
                //   clobberait le shadow-word (mots différents reset sh_word, DQM=0
                //   écrit tout le mot -> 3 bytes/mot perdus). Validé tb_kv_sdram_full.
                KVWR: begin
                    if (kv_refresh_due) begin
                        kv_refresh_ret <= 2'd1;     // retour KVWR (phase K) après refresh
                        st <= E_MREFRESH;
                    end else begin
                        kv_addr <= KV_BASE + ((layer*TMAX + pos_reg) << 6) + ldi[5:0];
                        kv_din  <= vfile[vidx(SB_Kc, ldi[5:0])];
                        kv_wr   <= 1'b1;
`ifdef SIM_DEBUG
                        if (ldi[5:0]==6'd0)
                            $display("[SIM] KVWR K ldi0 vKc0=%0d vQ0=%0d pos=%0d",
                                     $signed(vfile[vidx(SB_Kc,6'd0)]),
                                     $signed(vfile[vidx(SB_Q,6'd0)]), pos_reg);
`endif
                        st      <= KVWR_KB;
                    end
                end
                KVWR_KB: if (kv_busy) st<=KVWR_KD;
                KVWR_KD: if (!kv_busy) begin
                    if (ldi==KVW-1) begin ldi<=6'd0; st<=KVWR_V; end   // K fini -> phase V
                    else begin ldi<=ldi+1; st<=KVWR; end
                end
                KVWR_V: begin
                    if (kv_refresh_due) begin
                        kv_refresh_ret <= 2'd2;     // retour KVWR_V (phase V) après refresh
                        st <= E_MREFRESH;
                    end else begin
                        kv_addr <= KV_BASE + ((layer*TMAX + pos_reg) << 6) + 6'd32 + ldi[5:0];
                        kv_din  <= vfile[vidx(SB_Vc, ldi[5:0])];
                        kv_wr   <= 1'b1;
                        st      <= KVWR_VB;
                    end
                end
                KVWR_VB: if (kv_busy) st<=KVWR_VD;
                KVWR_VD: if (!kv_busy) begin
                    if (ldi==KVW-1) begin
                        ksh[layer*TMAX + pos_reg]<=sK; vsh[layer*TMAX + pos_reg]<=sV;
                        sKref<=-8'sd128; sVref<=-8'sd128; scan_i<=0; st<=SCAN;
                    end else begin
                        ldi<=ldi+1; st<=KVWR_V;
                    end
                end
                // serial max-shift scan over positions 0..pos of the CURRENT layer (seed -128)
                SCAN: begin
                    if (ksh[layer*TMAX + scan_i] > sKref) sKref<=ksh[layer*TMAX + scan_i];
                    if (vsh[layer*TMAX + scan_i] > sVref) sVref<=vsh[layer*TMAX + scan_i];
                    if (scan_i==pos_reg) begin mmp<=0; mmo<=0; mmv<=1'b0; phase<=PH_MM; st<=BUILD; end
                    else scan_i<=scan_i+1;
                end

                // capture x-after-5-layers into result (for test_gen_D), then lm_head
                DONE_ST: begin
                    result[ldi*W +: W] <= vfile[vidx(SB_XB, ldi[5:0])];
                    if (ldi==D-1) begin result_sh<=sxb; ldi<=0; phase<=PH_FNF; st<=BUILD; end
                    else ldi<=ldi+1;
                end

                // ---- serialized ALU sequence (MUL for SwiGLU, ADD for W2 reduce) ----
                // load alu_a (D bytes, combinational vfile read), then alu_b, pulse start.
                ALD_A: begin alu_a[ldi*W +: W] <= vfile[vidx(alu_aslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; st<=ALD_B; end else ldi<=ldi+1; end
                ALD_B: begin alu_b[ldi*W +: W] <= vfile[vidx(alu_bslot, ldi[5:0])];
                        if (ldi==D-1) begin ldi<=0; alu_start<=1'b1; st<=ALU_WAIT; end else ldi<=ldi+1; end
                ALU_WAIT: if (alu_done) begin
                        case (alu_next)
                            2'd0: shc<=alu_out_sh;    // MUL -> HG shift
                            2'd1: sov<=alu_out_sh;    // reduce -> accumulator shift
                            default: sxb<=alu_out_sh; // residual -> new running XB shift
                        endcase
                        ldi<=0; st<=AST_A;
                    end
                // serial-store alu_out into alu_dslot, then dispatch by alu_next
                AST_A: begin vaddr<=vidx(alu_dslot, ldi[5:0]); vdin<=alu_out[ldi*W +: W]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0;
                            case (alu_next)
                                2'd0: begin phase<=PH_FW2; st<=BUILD; end               // MUL done -> W2
                                2'd1: if (fc==NCHUNK_F-1) begin                        // last reduce -> FFN residual
                                          alu_aslot<=SB_XB; alu_bslot<=SB_OUTV; alu_dslot<=SB_XB;
                                          alu_op<=1'b1; alu_sa<=sxb; alu_sb<=sov; alu_next<=2'd2;
                                          ldi<=0; st<=ALD_A;
                                      end else begin fc<=fc+1; phase<=PH_FW1; st<=BUILD; end
                                default: begin  // residual done
                                          if (!in_ffn) begin in_ffn<=1'b1; phase<=PH_FN2; st<=BUILD; end  // attn -> FFN
                                          else if (layer==NL-1) st<=DONE_ST;                              // last layer
                                          else begin  // next layer : XB carries over, KV persists
                                              layer<=layer+1; in_ffn<=1'b0; fc<=2'd0;
                                              phase<=PH_FN; ldi<=0; st<=BUILD;
                                          end
                                      end
                            endcase
                        end else ldi<=ldi+1; end
                // fc==0 : the first W2 partial simply seeds the accumulator (no add)
                FRED_COPY: begin vaddr<=vidx(SB_OUTV, ldi[5:0]); vdin<=vfile[vidx(SB_P, ldi[5:0])]; vwe<=1'b1;
                        if (ldi==D-1) begin ldi<=0; sov<=spc; fc<=fc+1; phase<=PH_FW1; st<=BUILD; end
                        else ldi<=ldi+1; end

                // ---- lm_head argmax : sref = max(csh), then running max over VOCAB ----
                AMAX_INIT: begin
                    sref_lm <= csh[0]; scan_c <= 1;
                    best_val <= -32'sd2147483647; best_idx <= 0; amax_i <= 0;
                    st <= AMAX;
                end
                AMAX: begin   // scan max(csh) into sref_lm ; amax_i stays 0 -> lraw = logits[0]
                    if (scan_c < NCHUNK) begin
                        if (csh[scan_c] > sref_lm) sref_lm <= csh[scan_c];
                        scan_c <= scan_c + 1;
                    end else st <= AMAXC;
                end
                // running max over VOCAB : lraw holds logits[amax_i] (registered BSRAM
                // read), so compare here then step amax_i and let lraw re-settle (AMAXW).
                AMAXC: begin: sweep
                    reg signed [31:0] v;
                    v = $signed(lraw) >>> (sref_lm - csh[amax_i[9:6]]);
                    if (v > best_val) begin best_val <= v; best_idx <= amax_i; end
                    if (amax_i==VOCAB-1) begin
                        token <= best_idx;
                        if (gen_mode) begin
                            if (prefill_left != 7'd0) begin
                                // prefill intermédiaire : embed prochain token du prompt,
                                // PAS d'émission (token_valid reste 0 par défaut). KV cache
                                // se remplit à pos 0..P-1, masque causal pos_reg+1 respecté.
                                cur_tok      <= prompt[prompt_idx*10 +: 10];
                                prompt_idx   <= prompt_idx + 5'd1;
                                prefill_left <= prefill_left - 7'd1;
                                pos_reg      <= pos_reg + 6'd1;
                                layer<=0; in_ffn<=1'b0; fc<=2'd0;
                                phase<=PH_EE; st<=BUILD;
                            end else begin
                                // dernier token du prompt (prefill_left vient d'atteindre 0)
                                // OU decode normal : émission + feed argmax (comportement actuel).
`ifdef SIM_DEBUG
                                // dump argmax state + XB/XN/logits samples to localize the
                                // zero-token bug (logits all X/0 -> upstream XN/embed/SDRAM).
                                $display("[SIM] emit tok=%0d pos=%0d best_val=%0d best_idx=%0d sref_lm=%0d csh=%0d %0d %0d %0d",
                                         best_idx, pos_reg, best_val, best_idx, sref_lm,
                                         $signed(csh[0]), $signed(csh[1]), $signed(csh[2]), $signed(csh[3]));
                                $display("[SIM] XB[0:7]=%0d %0d %0d %0d %0d %0d %0d %0d",
                                         $signed(vfile[0]),$signed(vfile[1]),$signed(vfile[2]),$signed(vfile[3]),
                                         $signed(vfile[4]),$signed(vfile[5]),$signed(vfile[6]),$signed(vfile[7]));
                                $display("[SIM] XN[0:7]=%0d %0d %0d %0d %0d %0d %0d %0d",
                                         $signed(vfile[64]),$signed(vfile[65]),$signed(vfile[66]),$signed(vfile[67]),
                                         $signed(vfile[68]),$signed(vfile[69]),$signed(vfile[70]),$signed(vfile[71]));
                                $display("[SIM] logits[0:7]=%0d %0d %0d %0d %0d %0d %0d %0d",
                                         $signed(logits[0]),$signed(logits[1]),$signed(logits[2]),$signed(logits[3]),
                                         $signed(logits[4]),$signed(logits[5]),$signed(logits[6]),$signed(logits[7]));
                                $display("[SIM] logits[400:410]=%0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                                         $signed(logits[400]),$signed(logits[401]),$signed(logits[402]),$signed(logits[403]),
                                         $signed(logits[404]),$signed(logits[405]),$signed(logits[406]),$signed(logits[407]),
                                         $signed(logits[408]),$signed(logits[409]),$signed(logits[410]));
`endif
                                token_valid <= 1'b1; cur_tok <= best_idx;   // emit + feed next EMBED
                                if (pos_reg == NPOS-1) begin done<=1; st<=IDLE; end   // last token
                                else begin  // next token : embed again, KV cache persists
                                    pos_reg<=pos_reg+1; layer<=0; in_ffn<=1'b0; fc<=2'd0;
                                    phase<=PH_EE; st<=BUILD;
                                end
                            end
                        end else begin done<=1; st<=IDLE; end
                    end
                    else begin amax_i <= amax_i + 1; st <= AMAXW; end
                end
                AMAXW: st <= AMAXC;   // 1-cycle wait : lraw <= logits[amax_i] settles
            endcase
        end
    end
endmodule
