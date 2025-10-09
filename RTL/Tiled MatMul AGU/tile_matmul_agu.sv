module tile_matmul_agu #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 8
)(
    input  logic clk,
    input  logic rst,
    input  logic tile_req,
	output logic done_all,
    input  logic read_req,
	
	// global parameters
    input  logic [IDX_WIDTH-1:0] M, N, K,
    input  logic [IDX_WIDTH-1:0] TM_cfg, TN_cfg, TK_cfg,
	
	input  logic [ADDR_WIDTH-1:0] baseA,
    input  logic [ADDR_WIDTH-1:0] baseB,
    input  logic [ADDR_WIDTH-1:0] baseC,

    
    output logic [ADDR_WIDTH-1:0] o_addr,
    output logic [1:0] addr_id,
    output logic valid
);



    // --------------------------------------------------------
    // Internal signals
    // --------------------------------------------------------
    logic start_tile, tile_done, tile_ready;
    logic [ADDR_WIDTH-1:0] baseA_tile, baseB_tile, baseC_tile;
    logic [IDX_WIDTH-1:0] eTM, eTN, eTK;

    // --------------------------------------------------------
    // Instantiate Controller
    // --------------------------------------------------------
    tile_base_controller #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH(IDX_WIDTH)
    ) u_ctrl (
        .clk(clk),
        .rst(rst),
        .tile_req(tile_req),
        .done_all(done_all),

        .M(M), .N(N), .K(K),
        .TM_cfg(TM_cfg), .TN_cfg(TN_cfg), .TK_cfg(TK_cfg),
        .baseA(baseA), .baseB(baseB), .baseC(baseC),

        .start_tile(start_tile),
        .tile_done(tile_done),
        .tile_ready(tile_ready),

        .baseA_tile(baseA_tile),
        .baseB_tile(baseB_tile),
        .baseC_tile(baseC_tile),
        .eTM(eTM), .eTN(eTN), .eTK(eTK)
    );


    // --------------------------------------------------------
    // Instantiate Element AGU
    // --------------------------------------------------------
    tile_offsets_agu #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH(IDX_WIDTH)
    ) u_agu (
        .clk(clk),
        .rst(rst),
        .start_tile(start_tile),
        .read_req(read_req),

        .baseA_tile(baseA_tile),
        .baseB_tile(baseB_tile),
        .baseC_tile(baseC_tile),

        .eTM(eTM),
        .eTN(eTN),
        .eTK(eTK),
        .TM_cfg(TM_cfg),
        .TN_cfg(TN_cfg),
        .TK_cfg(TK_cfg),
        .FULL_K(K),
        .FULL_N(N),

        .o_addr(o_addr),
        .addr_id(addr_id),
        .valid(valid),

        .tile_done(tile_done),
        .tile_ready(tile_ready)
    );

endmodule
