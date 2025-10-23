// ============================================================
// S T M I C R O E L E C T R O N I C S
// AI Accelerators Hands-on HW Design  - Jul.2025
//
// auther : Mahmoud Abdo
// AGU.sv
// 
// description : Complete system supporting all convolution types + matrix multiply
// ============================================================

module AGU #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 16,
    parameter logic [ADDR_WIDTH-1:0] NULL_ADDR = {ADDR_WIDTH{1'b1}}
)(
    input  logic clk,
    input  logic rst_n,

    // Global control
    output logic                     processing, // busy or not 
    output logic                     AGU_ready, // reay to generate offsets within a given tile 
    input  logic                     tile_req, // ask for tile within an input 
    output logic                     tile_done,
    input  logic                     read_req, //ask for an address within a tile
    output logic                     all_tiles_done,

    // Operation type and parameters
    input  logic [1:0]               op_mode,// 00=regular, 01=pointwise, 10=depthwise, 11=matrix_mul
    input  logic [IDX_WIDTH-1:0]     act_H, act_W, act_CIN,
    input  logic [IDX_WIDTH-1:0]     ker_H, ker_W, out_chs,
    input  logic [IDX_WIDTH-1:0]     padding, stride,

    // Matrix multiply dimensions (for op_mode=matrix_mul)
    input  logic [IDX_WIDTH-1:0]     mat_M, mat_K, mat_N,

    // Tile configuration
    input  logic [IDX_WIDTH-1:0]     TM, TN, TK,

    // Base addresses
    input  logic [ADDR_WIDTH-1:0]    baseA,
    input  logic [ADDR_WIDTH-1:0]    baseB,
    input  logic [ADDR_WIDTH-1:0]    baseC,

    // Memory interface
    output logic [ADDR_WIDTH-1:0]    mem_addr,
    output logic [1:0]               mem_id,      // 00 = A/act, 01 = B/ker, 10 = C/out
    output logic                     mem_valid,
    output logic                     mem_is_null
);

    // ----------------------------
    // Internal signals
    logic [IDX_WIDTH-1:0] i_tile, j_tile, k_tile;
    logic [IDX_WIDTH-1:0] eTM, eTN, eTK;
    logic [IDX_WIDTH-1:0] M, K, N;
    logic start_tile, tile_ack, ready;

    // Direct connections
    assign tile_done = tile_ack;
    assign AGU_ready = ready;
    
    // ----------------------------
    // Tile Indices Generator
    ConvTileIndicesGenerator #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH(IDX_WIDTH)
    ) tile_indices_gen (
        .clk(clk),
        .rst_n(rst_n),
        // handshake & control signals 
        .tile_req(tile_req),
        .AGU_ready(ready),
        .start_tile(start_tile),
        .processing(processing),
        .tile_ack(tile_ack),
        .all_tiles_done(all_tiles_done),

        .op_mode(op_mode),
        .act_H(act_H),
        .act_W(act_W),
        .act_CIN(act_CIN),
        .ker_H(ker_H),
        .ker_W(ker_W),
        .out_chs(out_chs),
        .padding(padding),
        .stride(stride),

        .mat_M(mat_M),
        .mat_K(mat_K),
        .mat_N(mat_N),
        .TM(TM),
        .TN(TN),
        .TK(TK),
        .i_tile(i_tile),
        .j_tile(j_tile),
        .k_tile(k_tile),
        .eTM(eTM),
        .eTN(eTN),
        .eTK(eTK),
        .M(M),
        .K(K),
        .N(N)
    );

    // ----------------------------
    // Offset Generator
    ConvOffsetsAGU #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH(IDX_WIDTH),
        .NULL_ADDR(NULL_ADDR)
    ) offset_gen (
        .clk(clk),
        .rst_n(rst_n),
        .start_tile(start_tile),
        .tile_done(tile_ack),
        .AGU_ready(ready),
        .read_req(read_req),

        .op_mode(op_mode),
        .act_H(act_H),
        .act_W(act_W),
        .act_CIN(act_CIN),
        .ker_H(ker_H),
        .ker_W(ker_W),
        .out_chs(out_chs),
        .padding(padding),
        .stride(stride),

        .mat_M(mat_M),
        .mat_K(mat_K),
        .mat_N(mat_N),

        .i_tile(i_tile),
        .j_tile(j_tile),
        .k_tile(k_tile),

        .eTM(eTM),
        .eTN(eTN),
        .eTK(eTK),
        
        .TM(TM),
        .TN(TN),
        .TK(TK),

        .M(M),
        .N(N),
        .K(K),

        .baseA(baseA),
        .baseB(baseB),
        .baseC(baseC),

        .addr_out(mem_addr),
        .id_out(mem_id),
        .valid_out(mem_valid),
        .is_null_addr(mem_is_null)
    );

initial begin
  $dumpfile("agu_sim.vcd");
  $dumpvars(0);
end


endmodule