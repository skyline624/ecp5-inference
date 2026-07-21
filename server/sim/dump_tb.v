// dump_tb.v — TB Verilog minimal qui instancie wcache_ffn (module étape 3) avec
// stimulus write/read, et dump VCD. Sert de source VCD simple pour la Phase B/C
// (test de la chaîne VCD -> parser -> API -> timeline -> animation) SANS dépendre
// du node complet (cassé en icarus depuis les étapes 2a/3 : forward refs mac_* /
// eff_ref que yosys tolère mais pas icarus -- régression à corriger séparément).
//
// Hiérarchie VCD : dump_tb.clk, .we, .waddr, .wdata, .raddr, .rdata (= rdata_r),
// .u_dut.*. Mapping vers les buses du graphe (Phase C) : waddr->cache_addr,
// rdata->wcache_dout, we->wc_we (via net_map.json).
`timescale 1ns/1ps
module dump_tb;
    reg         clk = 1'b0;
    reg  [17:0] waddr = 18'd0;
    reg  [7:0]  wdata = 8'd0;
    reg         we = 1'b0;
    reg  [17:0] raddr = 18'd0;
    wire [7:0]  rdata;

    wcache_ffn #(.NFFN(184320)) u_dut (
        .clk(clk),
        .waddr(waddr), .wdata(wdata), .we(we),
        .raddr(raddr), .rdata(rdata));

    // clock 100 MHz (sim ; la fréquence réelle n'importe pas pour la visu)
    always #5 clk = ~clk;

    integer i;
    initial begin
        $dumpfile("dump_tb.vcd");
        $dumpvars(0, dump_tb);
        // remplissage séquentiel (imité la FSM wcache_fill : 16 octets)
        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk);
            waddr <= i; wdata <= i + 8'd32; we <= 1'b1;
        end
        @(negedge clk); we <= 1'b0;
        // lecture séquentielle (imité la FSM FQ : 16 octets)
        for (i = 0; i < 16; i = i + 1) begin
            @(negedge clk); raddr <= i;
        end
        @(negedge clk);
        #40 $finish;
    end
endmodule