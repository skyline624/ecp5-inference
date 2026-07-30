`timescale 1ns/1ps
// =============================================================================
// vec_alu2 - SERIALIZED elementwise ALU for the sequencer's FFN glue ops.
//
// Same ops as vec_alu (MUL = a*b on DSP, ADD = shift-align + add, then requantize)
// but every elementwise pass runs ONE element per cycle instead of 64 in parallel.
// The 64-wide parallel adder/max/scaler that cost ~5000 LUT in v1 becomes a tiny
// serial loop. acc lives in a 64x32 LUT-RAM (cheap). Cost target: a few hundred LUT.
// The 64-cycle passes add ~200 cycles/FFN = ~0.7% of the matmul time -> negligible.
//
// NOTE (reset) : synchronous active-low reset (rst_n), consistent with the other
// ops. Combinational intermediates (add_val/mul_abs/fin_rounded/fin_shifted) live
// in a dedicated always @(*) so we never mix blocking and non-blocking assigns
// inside the clocked block.
//
// -----------------------------------------------------------------------------
// INTERFACE ELEMENT-SERIE (rework 2026-07-30) — pourquoi ce n'est plus vectoriel
// -----------------------------------------------------------------------------
// AVANT : les operandes arrivaient en ports plats `a`/`b` de W*D bits et le
// resultat sortait en `out` de W*D bits. Le calcul etant serialise (1 element par
// cycle), chaque acces se faisait par part-select dynamique `a[i*W +: W]` — soit
// DEUX multiplexeurs D:1 en lecture et UN demultiplexeur D:1 en ecriture, plus
// 3*D*8 fils de routage depuis gen_seq, plus les registres plats alu_a/alu_b/
// alu_out cote gen_seq (eux-memes remplis par part-select : 2 demux de plus).
// Mesure : 3 143 LUT4 a dim64 et 5 514 a dim128 (x1,75) — pour une cible annoncee
// de "a few hundred LUT". Le datapath etait serialise, l'interface non.
//
// APRES : le module pilote lui-meme l'index de l'element qu'il veut lire
// (`rd_idx`) et lequel des deux operandes (`rd_sel`), et gen_seq lui presente
// l'octet correspondant depuis `vfile` (lecture combinatoire, 1 port). En sortie,
// le module pilote `wr_idx`/`wr_el`/`wr_we` que gen_seq route vers son port
// d'ecriture vfile. Les 6 structures D:1 disparaissent : l'adressage remplace le
// multiplexage. Le cout ne depend plus de D (seul l'index grandit en $clog2(D)).
//
// Effet de bord : gen_seq n'a plus besoin des passes de recopie ALD_A/ALD_B (2*D
// cycles de prechargement) ni de AST_A (D cycles de rangement) — le module lit et
// ecrit vfile en place. On paie 1 cycle de plus par element (lecture de a puis de
// b sur un port unique) mais on economise 3*D cycles de copie : gain net en
// cycles AUSSI, pas seulement en LUT.
//
// Bit-exactitude : les valeurs calculees sont inchangees (memes operandes, memes
// shifts da/db_/add_sh, meme ordre croissant de i donc meme sequence max_abs).
// =============================================================================
module vec_alu2 #(
    parameter W = 8,
    parameter D = 64
) (
    input  wire            clk,
    input  wire            rst_n,
    input  wire            start,
    input  wire            op,          // 0 = MUL, 1 = ADD
    input  wire signed [7:0] sa,
    input  wire signed [7:0] sb,
    // ---- lecture element-serie des operandes (presentes par gen_seq/vfile) ----
    output reg  [$clog2(D)-1:0] rd_idx,  // index de l'element demande (0..D-1)
    output reg                 rd_sel,   // 0 = operande a, 1 = operande b
    input  wire [W-1:0]        rd_el,    // octet demande, valide le cycle courant
    // ---- ecriture element-serie du resultat (routee par gen_seq vers vfile) ----
    output reg  [$clog2(D)-1:0] wr_idx,
    output reg  [W-1:0]        wr_el,
    output reg                 wr_we,
    output reg  signed [7:0]   out_sh,
    output reg             done
);
    localparam LAT = 4;
    reg signed [17:0] mac_a, mac_b; reg mac_load;
    wire signed [53:0] mac_res;
    mac18 u_mac (.clk(clk), .rst(~rst_n), .ce(1'b1),
                 .load(mac_load), .a(mac_a), .b(mac_b), .result(mac_res));

    reg signed [31:0] acc [0:D-1];        // products / aligned sums (LUT-RAM)
    reg signed [31:0] max_abs;
    reg signed [7:0]  base_sh, da, db_;
    reg [5:0] add_sh;
    reg [$clog2(D):0] i;                  // 0..D element index (etait [6:0] -> caps a 127, hang dim>=256)
    reg [2:0] fcnt;                       // DSP pipeline drain counter
    reg [2:0] ph;                         // sub-phase: 0=MUL/ADD, 1=REQ, 2=FIN
    reg [W-1:0] a_hold;                   // operande a[i] capture au cycle precedent
                                          // (port de lecture unique -> a puis b)

    // ---- combinational intermediates (pure functions of current state) ----
    reg signed [31:0] add_val;     // ADDC : shift-aligned sum a[i]<<<da + b[i]<<<db_
    reg signed [31:0] mul_abs;      // MUL_W: abs(mac_res) for the running max
    reg signed [31:0] add_abs;      // ADDC : abs(add_val)  for the running max
    reg signed [31:0] fin_rounded;  // FIN  : acc[i] + rounding constant
    reg signed [31:0] fin_shifted;  // FIN  : fin_rounded >>> add_sh

    always @(*) begin
        // a_hold = a[i] (lu a l'etat *_A), rd_el = b[i] (presente a l'etat *_B)
        add_val    = ($signed(a_hold) <<< da) + ($signed(rd_el) <<< db_);
        mul_abs    = mac_res[31] ? -mac_res[31:0] : mac_res[31:0];
        add_abs    = add_val[31] ? -add_val : add_val;
        fin_rounded= acc[i] + ((add_sh>0) ? (32'sd1 <<< (add_sh-1)) : 32'sd0);
        fin_shifted= fin_rounded >>> add_sh;
    end

    function [5:0] msb_idx;
        input [31:0] v; integer bb;
        begin msb_idx=0; for (bb=0;bb<32;bb=bb+1) if (v[bb]) msb_idx=bb[5:0]; end
    endfunction

    localparam IDLE=0, MUL_A=1, MUL_B=2, MUL_W=3, ADD_A=4, ADD_B=5, REQ=6, FIN=7;
    reg [2:0] st;

    always @(posedge clk) begin
        if (!rst_n) begin
            st<=IDLE; done<=0; mac_load<=0; i<=0; fcnt<=0; max_abs<=0;
            mac_a<=0; mac_b<=0; add_sh<=0; base_sh<=0; da<=0; db_<=0;
            rd_idx<=0; rd_sel<=0; wr_idx<=0; wr_el<=0; wr_we<=0; a_hold<=0;
        end else begin
            done<=0; mac_load<=0; wr_we<=0;
            case (st)
                IDLE: if (start) begin
                    base_sh <= (op==1'b0) ? (sa+sb) : ((sa<sb)?sa:sb);
                    da  <= (op==1'b0) ? 8'sd0 : (sa - ((sa<sb)?sa:sb));
                    db_ <= (op==1'b0) ? 8'sd0 : (sb - ((sa<sb)?sa:sb));
                    i<=0; max_abs<=0; fcnt<=0;
                    rd_idx<=0; rd_sel<=1'b0;      // demande a[0] : valide au cycle suivant
                    st <= (op==1'b0) ? MUL_A : ADD_A;
                end

                // ---- MUL via DSP : acc[i] = a[i]*b[i], track max_abs serially ----
                // Port de lecture unique : a[i] a l'etat MUL_A, b[i] a l'etat MUL_B.
                MUL_A: begin
                    a_hold <= rd_el;             // a[i]
                    rd_sel <= 1'b1;              // demande b[i]
                    st <= MUL_B;
                end
                MUL_B: begin
                    mac_a <= {{10{a_hold[W-1]}}, a_hold};
                    mac_b <= {{10{rd_el[W-1]}},  rd_el};    // b[i]
                    mac_load <= 1'b1; fcnt<=0;
                    rd_sel <= 1'b0;
                    st<=MUL_W;
                end
                MUL_W: begin
                    mac_a <= 0; mac_b <= 0;   // drain a*b only : zero operands so the
                                              // accumulate cycles don't re-add a*b (2x bug)
                    if (fcnt==LAT) begin
                        acc[i] <= mac_res[31:0];
                        if (mul_abs > max_abs) max_abs <= mul_abs;
                        if (i==D-1) st<=REQ;
                        else begin i<=i+1; rd_idx<=rd_idx+1'b1; st<=MUL_A; end
                    end else fcnt<=fcnt+1;
                end

                // ---- ADD : shift-align + add, one element/cycle, track max_abs ----
                ADD_A: begin
                    a_hold <= rd_el;             // a[i]
                    rd_sel <= 1'b1;              // demande b[i]
                    st <= ADD_B;
                end
                ADD_B: begin
                    acc[i] <= add_val;           // a_hold<<<da + rd_el(=b[i])<<<db_
                    if (add_abs > max_abs) max_abs <= add_abs;
                    rd_sel <= 1'b0;
                    if (i==D-1) st<=REQ;
                    else begin i<=i+1; rd_idx<=rd_idx+1'b1; st<=ADD_A; end
                end

                // ---- requantize pass : scale acc[i] -> out[i], one/cycle ----
                REQ: begin
                    add_sh <= (max_abs==0) ? 6'd0 :
                              (msb_idx(max_abs) > 6) ? (msb_idx(max_abs)-6) : 6'd0;
                    i<=0;                 // restart element index for the requantize pass
                    st<=FIN;
                end
                FIN: begin
                    wr_idx <= i[$clog2(D)-1:0];
                    wr_el  <= (fin_shifted > 32'sd127)  ? 8'sd127  :
                              (fin_shifted < -32'sd128) ? -8'sd128 : fin_shifted[7:0];
                    wr_we  <= 1'b1;
                    if (i==D-1) begin out_sh<=base_sh+$signed({2'b0,add_sh}); done<=1; st<=IDLE; end
                    else i<=i+1;
                end
            endcase
        end
    end
endmodule
