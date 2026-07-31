// Simple SDRAM controller for Tang 20k
// nand2mario
// 
// 2023.7: add buffers to din, dout and addr for ease-of-use.
// 2023.3: ported to use GW2AR-18's embedded 64Mbit SDRAM.
//         changed to byte-based access.
// 2022.9: iniital version.
//
// This is a byte-based, low-latency and non-bursting controller for the embedded SDRAM
// on Tang Nano 20K. The SDRAM module is 64Mbit 32bit. (2K rows x 256 columns x 4 banks x 32 bits).
//
// Under default settings (max 66.7Mhz):
// - Data read latency is 4 cycles. 
// - Read/write operations take 5 cycles to complete. There's no overlap between
//   reads/writes.
// - All reads/writes are done with auto-precharge. So user does not need to deal with
//   row activations and precharges.
// - SDRAMs need periodic refreshes or they lose data. So they provide an "auto-refresh"
//   function to do one row of refresh. This "auto-refresh" operation is controlled with
//   the 'refresh' input. 4096 or more refreshes should happen in any 64ms for the memory
//   to not lose data. So the main circuit should invoke auto-refresh at least once 
//   **every ~15us**.
//
// Finally you need a 180-degree phase-shifted clock signal (clk_sdram) for SDRAM. 
// This can be generated with PLL's clkoutp output.
//

module sdram
#(
    // Clock frequency, max 66.7Mhz with current set of T_xx/CAS parameters.
    parameter         FREQ = 64_800_000,  
    parameter         DATA_WIDTH = 32,
    parameter         ROW_WIDTH = 11,  // 2K rows
    parameter         COL_WIDTH = 8,   // 256 words per row (1Kbytes)
    parameter         BANK_WIDTH = 2,  // 4 banks

    // Time delays for 66.7Mhz max clock (min clock cycle 15ns)
    // The SDRAM supports max 166.7Mhz (RP/RCD/RC need changes)
    parameter [3:0]   CAS  = 4'd2,     // 2/3 cycles, set in mode register
    parameter [3:0]   T_WR = 4'd2,     // 2 cycles, write recovery
    parameter [3:0]   T_MRD= 4'd2,     // 2 cycles, mode register set
    parameter [3:0]   T_RP = 4'd1,     // 15ns, precharge to active
    parameter [3:0]   T_RCD= 4'd1,     // 15ns, active to r/w
    parameter [3:0]   T_RC = 4'd4      // 60ns, ref/active to ref/active
)
(
    // SDRAM side interface
    inout [DATA_WIDTH-1:0]      SDRAM_DQ,
    output reg [ROW_WIDTH-1:0]  SDRAM_A,
    output reg [BANK_WIDTH-1:0] SDRAM_BA,
    output            SDRAM_nCS,    // not strictly necessary, always 0
    output reg        SDRAM_nWE,
    output reg        SDRAM_nRAS,
    output reg        SDRAM_nCAS,
    output            SDRAM_CLK,
    output            SDRAM_CKE,    // not strictly necessary, always 1
    output reg  [3:0] SDRAM_DQM,
    
    // Logic side interface
    input             clk,
    input             clk_sdram,    // phase shifted from clk (normally 180-degrees)
    input             resetn,
    input             rd,           // command: read
    input             wr,           // command: write
    input             refresh,      // command: auto refresh. 4096 refresh cycles in 64ms. Once per 15us.
    input      [22:0] addr,         // byte address, buffered at rd/wr pulse time
    input       [7:0] din,          // data input, buffered at wr pulse time
    output      [7:0] dout,         // data output, available 4 cycles after rd becomes 1
                                    // output is buffered until next read request
    output [DATA_WIDTH-1:0] dout32, // 32-bit data output (valide au cycle data_ready)
    output [DATA_WIDTH-1:0] dout32b,// 32-bit bufferise (valide quand busy retombe)
    output [127:0]    dout128,      // bloc de 16 octets du dernier burst (idem)
    output reg        data_ready,   // available 6 cycles after wr is set
    output reg        busy          // 0: ready for next command
);

// Tri-state DQ input/output
reg dq_oen;         // 0 means output
reg [DATA_WIDTH-1:0] dq_out;
assign SDRAM_DQ = dq_oen ? 32'bzzzz_zzzz_zzzz_zzzz_zzzz_zzzz_zzzz_zzzz : dq_out;
wire [DATA_WIDTH-1:0] dq_in = SDRAM_DQ;     // DQ input

reg [1:0] off;          // byte offset
reg [7:0] dout_buf;
// Mot 32 bits BUFFERISE : dout32 (=dq_in) n'est valide qu'au cycle data_ready, donc
// inutilisable par un appelant qui echantillonne quand busy retombe. dout32_buf permet
// de consommer les 4 octets d'un seul acces au lieu d'un acces par octet.
reg [31:0] dout32_buf = 32'b0;
wire [7:0] next_dout =  off == 0 ? dq_in[7:0] :
                        off == 1 ? dq_in[15:8] :
                        off == 2 ? dq_in[23:16] : dq_in[31:24];
assign dout = data_ready ? next_dout : dout_buf;
assign dout32 = dq_in;
assign dout32b = dout32_buf;
// (assign dout128 place plus bas : rd_word est declare apres ce point)
assign SDRAM_CLK = clk_sdram;
assign SDRAM_CKE = 1'b1;
assign SDRAM_nCS = 1'b0;

reg [2:0] state;
localparam INIT = 3'd0;
localparam CONFIG = 3'd1;
localparam IDLE = 3'd2;
localparam READ = 3'd3;
localparam WRITE = 3'd4;
localparam REFRESH = 3'd5;
localparam RD_HIT  = 3'd6;   // hit du read shadow-word : pas d'acces SDRAM

// RAS# CAS# WE#
localparam CMD_SetModeReg=3'b000;
localparam CMD_AutoRefresh=3'b001;
localparam CMD_PreCharge=3'b010;
localparam CMD_BankActivate=3'b011;
localparam CMD_Write=3'b100;
localparam CMD_Read=3'b101;
localparam CMD_NOP=3'b111;

// Mode register INCHANGE : burst length 1.
// Le multi-mot est obtenu par 4 CMD_Read CONSECUTIFS dans la ligne ouverte, pas par
// le burst du mode register. Raison : un BURST_LEN=4 programme ici depend de la puce
// pour etre applique, et RIEN dans la simulation ne peut le verifier -- le modele
// comportemental ne decode pas le mode register. Une tentative en BL=4 (31/07) a
// passe la sim 3/3 et bloque le silicium. Avec 4 READ explicites, ce que la sim
// verifie est exactement ce que la puce recoit, et les ecritures restent single
// sans dependre du bit A9.
localparam [2:0] BURST_LEN = 3'b0;      // burst length 1
localparam BURST_MODE = 1'b0;           // sequential
localparam [10:0] MODE_REG = {4'b0, CAS[2:0], BURST_MODE, BURST_LEN};
localparam [3:0]  RD_WORDS = 4'd4;      // mots lus par acces (4 x 32 bits = 16 octets)

reg cfg_now;            // pulse for configuration
reg [3:0] cycle;        // each operation (config/read/write) are max 7 cycles
reg [7:0] din_buf;      // set at wr=1 pulse time
reg [22:0] addr_buf;

// ---- Colorlight i9 shadow-word (DQM hardwired GND on PCB -> can't mask byte
//      lanes; accumulate byte writes into a 32-bit shadow per word, then drive
//      the full word on DQ at write time). Transparent to the host: contiguous
//      byte loads (S_LL) build the correct word; a single byte write to an
//      already-loaded word clobbers siblings (S_WW diagnostic, out of inference
//      path). See plan M2 / kianRiscV m12l64322a_ctrl.v for the RMW alternative. ----
reg [31:0] sh_word = 32'b0;   // accumulated 32-bit word for the current byte addr
reg [20:0] sh_addr = 21'h1FFFFF;  // addr[22:2] of the word currently in sh_word

// ---- Read shadow-word : symetrique du sh_word d'ecriture ----------------------
// La SDRAM renvoie 32 bits par acces (BURST_LEN=1), mais l'interface est OCTET :
// le datapath streame les poids avec addr+1 a chaque octet. Sans cache, 4 octets
// consecutifs = 4 sequences ACTIVATE+READ+precharge completes (~7 cycles chacune)
// alors qu'ils vivent dans le MEME mot. Mesure avant optimisation : 9,3 cycles par
// octet de poids a dim256 (35,7 M cycles/token pour 3,64 Mo lus).
// On garde donc le dernier mot lu : si l'octet demande est dans ce mot, on le sert
// en 2 cycles sans toucher la SDRAM. Invalide des qu'une ecriture passe (le boot
// flash et S_LL ecrivent la SDRAM ; on invalide sans comparer l'adresse, plus sur).
// Depuis le passage en BURST 4 : le cache couvre 16 octets (4 mots) et non plus 4.
// Une activation de ligne sert donc 16 octets consecutifs -- CAS et handshake busy
// sont amortis sur 4x plus de donnees. Mesure du 31/07 avant ce changement :
// 7 cycles d'attente memoire par acces de 4 octets, soit 59 % du budget.
reg [127:0] rd_word  = 128'b0;
reg [18:0]  rd_waddr = 19'h7FFFF;      // addr[22:4] du bloc de 16 octets en cache
reg         rd_valid = 1'b0;
wire        rd_hit   = rd && rd_valid && (addr[22:4] == rd_waddr);
// Mot complet a l'instant de la DERNIERE capture du burst : rd_word[95:0] porte deja
// les mots 0..2, dq_in porte le mot 3 (non encore enregistre).
wire [127:0] rd_full = {dq_in, rd_word[95:0]};
assign dout128 = rd_word;

//
// SDRAM state machine
//
always @(posedge clk) begin
    cycle <= cycle == 4'd15 ? 4'd15 : cycle + 4'd1;
    // defaults
    {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_NOP; 
    casex ({state, cycle})
        // wait 200 us on power-on
        {INIT, 4'bxxxx} : if (cfg_now) begin
            state <= CONFIG;
            cycle <= 0;
        end

        // configuration sequence
        //  cycle  / 0 \___/ 1 \___/ 2 \___/ ... __/ 6 \___/ ...___/10 \___/11 \___/ 12\___
        //  cmd            |PC_All |Refresh|       |Refresh|       |  MRD  |       | _next_
        //                 '-T_RP--`----  T_RC  ---'----  T_RC  ---'------T_MRD----'
        {CONFIG, 4'd0} : begin
            // precharge all
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_PreCharge;
            SDRAM_A[10] <= 1'b1;
        end
        {CONFIG, T_RP} : begin
            // 1st AutoRefresh
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_AutoRefresh;
        end
        {CONFIG, T_RP+T_RC} : begin
            // 2nd AutoRefresh
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_AutoRefresh;
        end
        {CONFIG, T_RP+T_RC+T_RC} : begin
            // set register
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_SetModeReg;
            SDRAM_A[10:0] <= MODE_REG;
        end
        {CONFIG, T_RP+T_RC+T_RC+T_MRD} : begin
            state <= IDLE;
            busy <= 1'b0;              // init&config is done
        end
        
        // read/write/refresh
        // HIT du read shadow-word : l'octet demande est dans le mot deja lu -> on le
        // sert sans acces SDRAM. busy monte quand meme 1 cycle : l'appelant attend la
        // sequence busy=1 puis busy=0 (node_top S_OP_RD_BUSY/S_OP_RD_DONE).
        {IDLE, 4'bxxxx}: if (rd_hit) begin
            dout_buf   <= rd_word[addr[3:0]*8 +: 8];
            dout32_buf <= rd_word[addr[3:2]*32 +: 32];
            addr_buf   <= addr;
            busy     <= 1'b1;
            cycle    <= 4'd1;
            state    <= RD_HIT;
        end else if (rd | wr) begin
            // bank activate
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_BankActivate;
            SDRAM_BA <= addr[ROW_WIDTH+COL_WIDTH+BANK_WIDTH-1+2 : ROW_WIDTH+COL_WIDTH+2];    // bank id
            SDRAM_A <= addr[ROW_WIDTH+COL_WIDTH-1+2:COL_WIDTH+2];      // 12-bit row address
            state <= rd ? READ : WRITE;
            addr_buf <= addr;
            if (wr) begin
                din_buf <= din;
                rd_valid <= 1'b0;         // une ecriture peut toucher le mot cache -> invalide
                // shadow-word: reset on word change, accumulate target byte.
                if (addr[22:2] != sh_addr) sh_word <= 32'b0;
                sh_word[addr[1:0]*8 +: 8] <= din;
                sh_addr <= addr[22:2];
            end
            cycle <= 4'd1;
            busy <= 1'b1;
        end else if (refresh) begin
            // auto-refresh
            // no need for precharge-all b/c all our r/w are done with auto-precharge.
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_AutoRefresh;
            state <= REFRESH;
            cycle <= 4'd1;
            busy <= 1'b1;
        end

        // read sequence
        //  cycle  / 0 \___/ 1 \___/ 2 \___/ 3 \___/ 4 \___/ 5 \___
        //  rd     /       \_______________________________
        //  cmd            |Active | Read  |  NOP  |  NOP  | _Next_
        //  DQ                                     |  Dout |
        //  data_ready ____________________________/       \_______   
        //  busy   ________/                               \_______
        //                 `-T_RCD-'------CAS------'
        // 4 CMD_Read consecutifs dans la ligne ouverte (cycles T_RCD .. T_RCD+3).
        // A10=0 : PAS d'auto-precharge, sinon la ligne se fermerait apres le 1er READ.
        // Le precharge est emis explicitement apres la derniere capture.
        // Colonne ALIGNEE sur 4 mots : on lit le bloc de 16 octets qui CONTIENT l'octet
        // demande, et on l'extrait a l'arrivee.
        // ---- Lecture de 4 mots (16 octets) par 4 CMD_Read consecutifs -------------
        // Calibre pour T_RCD=1, CAS=2 (valeurs de ce controleur). Les emissions et les
        // captures SE CHEVAUCHENT, d'ou une sequence ecrite cycle par cycle plutot
        // qu'avec des expressions en T_RCD/CAS : un casex ne prend QUE la premiere
        // branche qui matche, donc deux branches tombant sur le meme cycle en
        // perdraient une silencieusement.
        //
        //  cycle : 1     2     3     4     5     6     7     8
        //  cmd   : RD0   RD1   RD2   RD3   -     -     PRE   -
        //  DQ    :             w0    w1    w2    w3
        //  latch :                   w0    w1    w2    w3
        //
        // A10=0 sur les READ : PAS d'auto-precharge, sinon la ligne se fermerait des
        // le premier mot. Precharge explicite au cycle 7, puis T_RP avant de rendre
        // la main -- le refresh comme le prochain ACT exigent la ligne fermee.
        {READ, 4'd1}: begin
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_Read;
            SDRAM_A[10]  <= 1'b0;
            // Colonne ALIGNEE sur 4 mots : on lit le bloc de 16 octets CONTENANT
            // l'octet demande, et on l'extrait a l'arrivee.
            SDRAM_A[9:0] <= {1'b0, addr_buf[COL_WIDTH-1+2:4], 2'b00};
            SDRAM_DQM <= 4'b0;
            off <= addr_buf[1:0];
        end
        {READ, 4'd2}: begin
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_Read;
            SDRAM_A[10]  <= 1'b0;
            SDRAM_A[9:0] <= {1'b0, addr_buf[COL_WIDTH-1+2:4], 2'b01};
        end
        {READ, 4'd3}: begin                      // = T_RCD+CAS : 1er mot sur DQ
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_Read;
            SDRAM_A[10]  <= 1'b0;
            SDRAM_A[9:0] <= {1'b0, addr_buf[COL_WIDTH-1+2:4], 2'b10};
            data_ready <= 1'b1;
        end
        {READ, 4'd4}: begin                      // dernier READ + 1ere capture
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_Read;
            SDRAM_A[10]  <= 1'b0;
            SDRAM_A[9:0] <= {1'b0, addr_buf[COL_WIDTH-1+2:4], 2'b11};
            data_ready <= 1'b0;
            rd_word[31:0]  <= dq_in;
        end
        {READ, 4'd5}: rd_word[63:32] <= dq_in;
        {READ, 4'd6}: rd_word[95:64] <= dq_in;
        {READ, 4'd7}: begin
            rd_word[127:96] <= dq_in;
            // rd_full combine les 3 mots deja enregistres et dq_in (le 4e).
            dout_buf   <= rd_full[addr_buf[3:0]*8  +: 8];
            dout32_buf <= rd_full[addr_buf[3:2]*32 +: 32];
            rd_waddr   <= addr_buf[22:4];
            rd_valid   <= 1'b1;
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_PreCharge;
            SDRAM_A[10] <= 1'b1;                 // precharge ALL banks
        end
        {READ, 4'd7+T_RP}: begin
            busy <= 0;
            state <= IDLE;
        end

        // hit cache lecture : dout_buf est deja pose, on relache busy
        {RD_HIT, 4'bxxxx}: begin
            busy  <= 1'b0;
            state <= IDLE;
        end

        // write sequence
        //  cycle / 0 \___/ 1 \___/ 2 \___/ 3 \___/ 4 \___/ 5 \___
        //  wr    /       \_______________________________
        //  cmd           |Active | Write |  NOP  |  NOP  | _Next_
        //  DQ                    | Din   |
        //  busy   _______/                               \_______
        //                `-T_RCD-'-------T_WR+T_RP-------'
        {WRITE, T_RCD}: begin
            {SDRAM_nRAS, SDRAM_nCAS, SDRAM_nWE} <= CMD_Write;
            SDRAM_A[10] <= 1'b1;        // set auto precharge
            SDRAM_A[9:0] <= {1'b0, addr_buf[COL_WIDTH-1+2:2]};  // column address
            SDRAM_DQM <= addr_buf[1:0] == 2'd0 ? 4'b1110 :
                         addr_buf[1:0] == 2'd1 ? 4'b1101 :
                         addr_buf[1:0] == 2'd2 ? 4'b1011 : 4'b0111;     // only write the correct byte
            off <= addr_buf[1:0];
            dq_out <= sh_word;              // i9: DQM hardwired -> drive full shadow word
            dq_oen <= 1'b0;                 // DQ output on
        end
        {WRITE, T_RCD+4'd1}: begin
            dq_oen <= 1'b1;
        end
        {WRITE, T_RCD+T_WR+T_RP}: begin  // 2+2+1
            busy <= 0;
            state <= IDLE;
        end

        // refresh sequence
        //  cycle   / 0 \___/ 1 \___/ 2 \___/ 3 \___/ 4 \___/ 5 \___
        //  refresh /       \_______________________________
        //  cmd             |Refresh|  NOP  |  NOP  |  NOP  | _Next_
        //  busy     _______/                               \_______
        //                  `------------- T_RC ------------'
        {REFRESH, T_RC}: begin
            state <= IDLE;
            busy <= 0;
        end
    endcase

    if (~resetn) begin
        busy <= 1'b1;
        dq_oen <= 1'b1;         // turn off DQ output
        SDRAM_DQM <= 4'b0;
        state <= INIT;
    end
end


//
// Generate cfg_now pulse after initialization delay (normally 200us)
//
reg  [14:0]   rst_cnt;
reg rst_done, rst_done_p1, cfg_busy;
  
always @(posedge clk) begin
    rst_done_p1 <= rst_done;
    cfg_now     <= rst_done & ~rst_done_p1;// Rising Edge Detect

    if (rst_cnt != FREQ / 1000 * 200 / 1000) begin      // count to 200 us
        rst_cnt  <= rst_cnt[14:0] + 1;
        rst_done <= 1'b0;
        cfg_busy <= 1'b1;
    end else begin
        rst_done <= 1'b1;
        cfg_busy <= 1'b0;
    end

    if (~resetn) begin
        rst_cnt  <= 15'd0;
        rst_done <= 1'b0;
        cfg_busy <= 1'b1;
    end
end

endmodule