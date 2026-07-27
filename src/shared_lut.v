// =============================================================================
// shared_lut.v -- Module BRAM ROM 256x16 parametrable (fonctions non-lineaires)
//
// Factorisation des tables exp_lut / inv_lut partagees entre softmax_op.v et
// attention_head_op.v (decision utilisateur : 2 BRAM partagees au lieu de 4
// dupliquees). Pattern tire de gen_seq.v:305-324 (cos_rom/sin_rom RoPE) :
//   - (* ram_style = "block" *) force l'inférence BRAM DP16KD par yosys
//   - $readmemh dans initial (yosys embarque le contenu dans le bitstream BRAM)
//   - read registre 1 cycle (port q synchrone) -> necessite 1 etat wait en aval
//     pour absorber la latence (pattern 2-phase, cf. silu_op.v S_LOOKUP_WAIT).
//
// Interface :
//   addr  : adresse 8-bit presentee au cycle N
//   q     : donnee 16-bit disponible au cycle N+1 (registree)
//
// Utilisation : instancier avec parameter HEX="exp_lut.hex" ou "inv_lut.hex".
//
// NOTE : Verilator/yosys ne partage PAS automatiquement une BRAM entre modules
// (chaque instance = sa propre BRAM). Le vrai partage physique (1 BRAM pour
// exp_lut lue par softmax_op ET attention_head_op) exigerait d'instancier ce
// module au niveau node_top.v et de router les ports. Pour la phase 4, on
// instancie shared_lut DANS chaque module (2 instances par module, 4 au total)
// -- le gain LUT vient de la migration LUT-RAM -> BRAM, pas du partage physique
// (les 4 BRAM sont negligeables : 4 blocs sur 208, dont 200 oisifs a dim>=128).
// =============================================================================

module shared_lut #(
    parameter HEX = "exp_lut.hex"
) (
    input  wire        clk,
    input  wire [7:0]  addr,
    output reg  [15:0] q
);
    (* ram_style = "block" *) reg [15:0] mem [0:255];
    initial $readmemh(HEX, mem);

    always @(posedge clk) q <= mem[addr];
endmodule