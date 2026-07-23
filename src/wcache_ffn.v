// wcache_ffn : cache on-chip des poids FFN (w1/w3/w2, 5 couches) en BRAM ECP5.
//
// Capacite : 184320 octets int8 = 5 couches x 36864 octets (3 matrices W1/W3/W2 x
// 3 chunks [64,64] x 4096, layout chunked 1:1 identique a la SDRAM). 90 blocs DP16KD.
//
// Layout (cache_addr = ff_layer*36864 + (sd_addr[15:0]-0x3200), voir decodeur dans
// node_top.v) :
//   couche L (0..4), base_l = 0x010000 + L*0x10000
//   W1 [base_l+0x3200, +0x6200) -> cache_addr [L*36864, L*36864+12288)
//   W3 [base_l+0x6200, +0x9200) -> [..+12288, ..+24576)
//   W2 [base_l+0x9200, +0xC200) -> [..+24576, ..+36864)
// Le zero-pad (fc==2, lignes 44..63) est stocke comme des zeros -> retourne naturellement,
// silu(0)=0 preserve, shifts par-chunk intacts.
//
// Double-port : ecriture 1 octet/cycle (boot DMA SDRAM->BRAM via la FSM wcache_fill),
// lecture 1-cycle registered (FSM FQ S_FM_RD->S_FM_WB). Sur clk_sys.

module wcache_ffn #(
    parameter NFFN = 184320,        // taille BRAM (NC_LAYERS_CACHED * WCACHE_LAYER_SZ)
    parameter AW = 18               // largeur ports addr (>= $clog2(NFFN+1))
) (
    input  wire          clk,
    // port ecriture (boot DMA, 1 octet/cycle)
    input  wire [AW-1:0] waddr,     // 0..NFFN-1
    input  wire [7:0]    wdata,
    input  wire          we,
    // port lecture (FSM FQ, 1-cycle registered)
    input  wire [AW-1:0] raddr,     // 0..NFFN-1
    output wire [7:0]   rdata
);
    (* ram_style = "block" *) reg [7:0] mem [0:NFFN-1];

    // En silicium : la BRAM s'init a 0, puis le boot DMA (FSM wcache_fill) la remplit
    // depuis la SDRAM. En sim : le testbench charge wcache_ffn.hex via reference
    // hierarchique (top.u_n.u_wc.mem) -- pas de $readmemh ici (yosys essaierait
    // d'ouvrir le fichier a la synthese).

    always @(posedge clk) if (we) mem[waddr] <= wdata;

    reg [7:0] rdata_r;
    always @(posedge clk) rdata_r <= mem[raddr];
    assign rdata = rdata_r;
endmodule