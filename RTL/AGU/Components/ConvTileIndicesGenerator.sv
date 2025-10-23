// ============================================================
// S T M I C R O E L E C T R O N I C S
// AI Accelerators Hands-on HW Design  - Jul.2025
//
// auther : Mahmoud Abdo
// AGU.sv
// 
// description : tile incices generation 
//   supporting all convolution types + matrix multiply
// ============================================================
module ConvTileIndicesGenerator #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 16
)(
    input  logic clk,
    input  logic rst_n,

    // Control handshake
    input  logic tile_req,        // request next tile
    input  logic AGU_ready,       // address generator unit ready
    output logic start_tile,      // to AGU - pulse for one cycle
    input  logic tile_ack,        // current tile done
    output logic processing,      // busy with a conv or matmul operation
    output logic all_tiles_done,  // asserted for 1 clk when last tile done

    // Operation type and parameters
    input  logic [1:0] op_mode,   // 00=regular, 01=pointwise, 10=depthwise, 11=matrix_mul
    input  logic [IDX_WIDTH-1:0] act_H, act_W, act_CIN,
    input  logic [IDX_WIDTH-1:0] ker_H, ker_W, out_chs,
    input  logic [IDX_WIDTH-1:0] padding, stride,

    // Matrix multiply dimensions
    input  logic [IDX_WIDTH-1:0] mat_M, mat_K, mat_N,

    // Tile configuration
    input  logic [IDX_WIDTH-1:0] TM, TN, TK,

    // Output tile indices and dimensions
    output logic [IDX_WIDTH-1:0] i_tile, j_tile, k_tile,
    output logic [IDX_WIDTH-1:0] eTM, eTN, eTK,
    output logic [IDX_WIDTH-1:0] M, K, N
);

    // Operation types
    localparam logic [1:0] OP_REGULAR    = 2'b00;
    localparam logic [1:0] OP_POINTWISE  = 2'b01;
    localparam logic [1:0] OP_DEPTHWISE  = 2'b10;
    localparam logic [1:0] OP_MATRIX_MUL = 2'b11;

    // ----------------------------
    // Derived dimensions
    logic [IDX_WIDTH-1:0] Kh_eff, Kw_eff, K_block;
    logic [IDX_WIDTH-1:0] out_H,out_W ;

    always_comb begin
        case (op_mode)
            OP_MATRIX_MUL: begin
                M = mat_M;
                K = mat_K;
                N = mat_N;
                out_H = 0;
                out_W = 0;
                K_block = 0;
            end

            default: begin
                if (op_mode == OP_POINTWISE) begin
                    Kh_eff = 1;
                    Kw_eff = 1;
                end else begin
                    Kh_eff = ker_H;
                    Kw_eff = ker_W;
                end

                out_H = ((act_H + 2*padding - Kh_eff) / stride) + 1;
                out_W = ((act_W + 2*padding - Kw_eff) / stride) + 1;
                M = out_H * out_W;
                K_block = ker_H * ker_W;

                case (op_mode)
                    OP_REGULAR: begin
                        K = ker_H * ker_W * act_CIN;
                        N = out_chs;
                    end
                    OP_POINTWISE: begin
                        K = act_CIN;
                        N = out_chs;
                    end
                    OP_DEPTHWISE: begin
                        K = K_block * act_CIN;  // 3x3x3 = 27
                        N = act_CIN;           // 3 output channels
                    end
                    default: begin
                        K = ker_H * ker_W * act_CIN;
                        N = out_chs;
                    end
                endcase
            end
        endcase
    end

    // ----------------------------
    // Tile indices registers
    logic [IDX_WIDTH-1:0] i_tile_reg, j_tile_reg, k_tile_reg;
    logic [IDX_WIDTH-1:0] i_tile_next, j_tile_next, k_tile_next;
    logic last_tile;

    // ----------------------------
    // Effective tile size computation - FIXED: Same for all modes including depthwise
    always_comb begin
        eTM = (i_tile_reg < M) ? ((M - i_tile_reg >= TM) ? TM : (M - i_tile_reg)) : 0;
        eTN = (j_tile_reg < N) ? ((N - j_tile_reg >= TN) ? TN : (N - j_tile_reg)) : 0;
        eTK = (k_tile_reg < K) ? ((K - k_tile_reg >= TK) ? TK : (K - k_tile_reg)) : 0;

        last_tile = ((i_tile_reg + TM >= M) && (j_tile_reg + TN >= N) && (k_tile_reg + TK >= K));
    end

    // ----------------------------
    // Next tile computation - FIXED: Same for all modes including depthwise
    always_comb begin
        i_tile_next = i_tile_reg;
        j_tile_next = j_tile_reg;
        k_tile_next = k_tile_reg;

        // Standard tiling pattern for ALL modes
        if (k_tile_reg + TK < K) begin
            k_tile_next = k_tile_reg + TK;
        end else begin
            k_tile_next = 0;
            if (j_tile_reg + TN < N) begin
                j_tile_next = j_tile_reg + TN;
            end else begin
                j_tile_next = 0;
                if (i_tile_reg + TM < M) begin
                    i_tile_next = i_tile_reg + TM;
                end
            end
        end
    end

    // ----------------------------
    // FSM States
    typedef enum logic [2:0] {
        IDLE,
        WAIT_FIRST_TILE_REQ,
        PROCESSING_TILE,
        WAIT_NEXT_TILE_REQ,
        FINAL_TILE_DONE
    } state_t;

    state_t state, next_state;

    // FSM next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (tile_req && AGU_ready) 
                    next_state = WAIT_FIRST_TILE_REQ;
            end

            WAIT_FIRST_TILE_REQ: begin
                next_state = PROCESSING_TILE;
            end

            PROCESSING_TILE: begin
                if (tile_ack) begin
                    if (last_tile)
                        next_state = FINAL_TILE_DONE;
                    else
                        next_state = WAIT_NEXT_TILE_REQ;
                end
            end

            WAIT_NEXT_TILE_REQ: begin
                if (tile_req && AGU_ready)
                    next_state = PROCESSING_TILE;
            end

            FINAL_TILE_DONE: begin
                next_state = IDLE;
            end
        endcase
    end

    // ----------------------------
    // Sequential logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            i_tile_reg <= '0;
            j_tile_reg <= '0;
            k_tile_reg <= '0;
            processing <= 1'b0;
            start_tile <= 1'b0;
            all_tiles_done <= 1'b0;
        end else begin
            state <= next_state;
            
            // Default outputs
            start_tile <= 1'b0;
            all_tiles_done <= 1'b0;

            case (state)
                IDLE: begin
                    processing <= 1'b0;
                    // Reset tile indices when starting new operation
                    if (tile_req && AGU_ready) begin
                        i_tile_reg <= '0;
                        j_tile_reg <= '0;
                        k_tile_reg <= '0;
                        processing <= 1'b1;
                    end
                end

                WAIT_FIRST_TILE_REQ: begin
                    processing <= 1'b1;
                    start_tile <= 1'b1; // Start first tile
                end

                PROCESSING_TILE: begin
                    processing <= 1'b1;
                    
                    // Update tile indices when current tile is done
                    if (tile_ack) begin
                        i_tile_reg <= i_tile_next;
                        j_tile_reg <= j_tile_next;
                        k_tile_reg <= k_tile_next;
                    end
                end

                WAIT_NEXT_TILE_REQ: begin
                    processing <= 1'b1;
                    // Wait here for next tile_req + AGU_ready
                    if (tile_req && AGU_ready) begin
                        start_tile <= 1'b1; // Start next tile
                    end
                end

                FINAL_TILE_DONE: begin
                    processing <= 1'b0;
                    all_tiles_done <= 1'b1;
                end
            endcase
        end
    end

    // ----------------------------
    // Outputs
    assign i_tile = i_tile_reg;
    assign j_tile = j_tile_reg;
    assign k_tile = k_tile_reg;

endmodule