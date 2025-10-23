// ============================================================
// S T M I C R O E L E C T R O N I C S
// AI Accelerators Hands-on HW Design  - Jul.2025
//
// auther : Mahmoud Abdo
// AGU.sv
// 
// description :
// Supports -->  Regular, Pointwise, Depthwise conv + Tiled Matrix Multiply
// With Diagonal Kernel Reading for Depthwise
// ============================================================

module ConvOffsetsAGU #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 16,
    parameter logic [ADDR_WIDTH-1:0] NULL_ADDR = {ADDR_WIDTH{1'b1}}
)(
    input  logic                     clk,
    input  logic                     rst_n,

    // Control handshake
    input  logic                     start_tile,    // pulse to start current tile
    output logic                     tile_done,     // tile finished
    output logic                     AGU_ready,     // AGU ready to receive new tile indices

    // Read request control
    input  logic                     read_req,      // assert to request addresses, deassert to pause

    // Base addresses
    input  logic [ADDR_WIDTH-1:0]    baseA,
    input  logic [ADDR_WIDTH-1:0]    baseB,
    input  logic [ADDR_WIDTH-1:0]    baseC,

    // Operation type and parameters
    input  logic [1:0]               op_mode,       // 00=regular, 01=pointwise, 10=depthwise, 11=matrix_mul
    input  logic [IDX_WIDTH-1:0]     act_H, act_W, act_CIN,
    input  logic [IDX_WIDTH-1:0]     ker_H, ker_W, out_chs,
    input  logic [IDX_WIDTH-1:0]     padding, stride,

    // Matrix multiply dimensions (for op_mode=matrix_mul)
    input  logic [IDX_WIDTH-1:0]     mat_M, mat_K, mat_N,

    // Tile parameters
    input  logic [IDX_WIDTH-1:0]     i_tile, j_tile, k_tile,
    input  logic [IDX_WIDTH-1:0]     eTM, eTN, eTK,
    input  logic [IDX_WIDTH-1:0]     TM, TN, TK,
    input  logic [IDX_WIDTH-1:0]     M, N, K,

    // Streamed offset output
    output logic [ADDR_WIDTH-1:0]    addr_out,
    output logic [1:0]               id_out,  // 00=A, 01=B, 10=C
    output logic                     valid_out,
    output logic                     is_null_addr
);

    // ----------------------------
    // Operation type encoding
    localparam logic [1:0] OP_REGULAR    = 2'b00;
    localparam logic [1:0] OP_POINTWISE  = 2'b01;
    localparam logic [1:0] OP_DEPTHWISE  = 2'b10;
    localparam logic [1:0] OP_MATRIX_MUL = 2'b11;

    // ----------------------------
    // State encoding
    typedef enum logic [2:0] {
        IDLE,
        GEN_A,      // Generate A tile addresses
        GEN_B,      // Generate B tile addresses  
        GEN_C,      // Generate C tile addresses
        TILE_DONE
    } state_t;

    state_t state, next_state;

    // Counters within current tile
    logic [IDX_WIDTH-1:0] i_cnt, j_cnt, k_cnt;

    // Internal registered outputs
    logic [ADDR_WIDTH-1:0] addr_out_reg;
    logic [1:0]           id_out_reg;
    logic                 valid_out_reg;
    logic                 is_null_addr_reg;

    // Internal address computation (combinational)
    logic [ADDR_WIDTH-1:0] addr_out_comb;
    logic [1:0]           id_out_comb;
    logic                 is_null_addr_comb;

    // Read request edge detection
    logic read_req_prev;
    logic read_req_posedge;

    assign read_req_posedge = read_req && !read_req_prev;

    // Derived dimensions
    logic [IDX_WIDTH-1:0] out_H, out_W;
    logic [IDX_WIDTH-1:0] Kh_eff, Kw_eff;
    logic [IDX_WIDTH-1:0] K_block; // For depthwise: Kh * Kw
    
    always_comb begin
        if (op_mode == OP_MATRIX_MUL) begin
            // Matrix multiply: no spatial dimensions
            out_H = 0;
            out_W = 0;
            Kh_eff = 0;
            Kw_eff = 0;
            K_block = 0;
        end else begin
            // Convolution modes
            if (op_mode == OP_POINTWISE) begin
                Kh_eff = 1;
                Kw_eff = 1;
            end else begin
                Kh_eff = ker_H;
                Kw_eff = ker_W;
            end
            
            out_H = ((act_H + 2*padding - Kh_eff) / stride) + 1;
            out_W = ((act_W + 2*padding - Kw_eff) / stride) + 1;
            K_block = ker_H * ker_W;
        end
    end

    // Default outputs - now registered and gated by read_req
    assign valid_out  = valid_out_reg;
    assign addr_out   = addr_out_reg;
    assign id_out     = id_out_reg;
    assign is_null_addr = is_null_addr_reg;
    
    assign tile_done  = (state == TILE_DONE);
    assign AGU_ready  = (state == IDLE);

    // ----------------------------
    // Global position computation
    // ----------------------------
    logic [IDX_WIDTH-1:0] i_global, j_global, k_global;
    assign i_global = i_tile + i_cnt;
    assign j_global = j_tile + j_cnt;
    assign k_global = k_tile + k_cnt;

    // ----------------------------
    // Helper function for input offset (HWC layout)
    function logic [ADDR_WIDTH-1:0] input_offset;
        input logic [IDX_WIDTH-1:0] h;
        input logic [IDX_WIDTH-1:0] w; 
        input logic [IDX_WIDTH-1:0] c;
    begin
        input_offset = baseA + ((h * act_W + w) * act_CIN + c);
    end
    endfunction

    // ----------------------------
    // Kernel size computation for different modes
    function logic [IDX_WIDTH-1:0] get_kernel_size;
    begin
        case (op_mode)
            OP_REGULAR:    get_kernel_size = ker_H * ker_W * act_CIN;
            OP_POINTWISE:  get_kernel_size = act_CIN;
            OP_DEPTHWISE:  get_kernel_size = K_block;
            OP_MATRIX_MUL: get_kernel_size = 1; // Not used for matrix multiply
            default:       get_kernel_size = ker_H * ker_W * act_CIN;
        endcase
    end
    endfunction

    // ----------------------------
    // A matrix address computation
    function logic [ADDR_WIDTH-1:0] compute_input_addr;
        input logic [IDX_WIDTH-1:0] i_global_val;
        input logic [IDX_WIDTH-1:0] k_global_val;
        input logic [IDX_WIDTH-1:0] i_local_val;
        input logic [IDX_WIDTH-1:0] k_local_val;
        logic [IDX_WIDTH-1:0] out_h, out_w;
        logic [IDX_WIDTH-1:0] kh_i, kw_i, c_in;
        logic [IDX_WIDTH-1:0] top_h, left_w, in_h, in_w;
        logic [IDX_WIDTH-1:0] channel_idx, k_block_idx;
    begin
        // Check boundary conditions FIRST
        if (i_local_val >= eTM || k_local_val >= eTK) begin
            compute_input_addr = NULL_ADDR;
        end else begin
            case (op_mode)
                OP_MATRIX_MUL: begin
                    // Matrix multiply: A[i][k] = baseA + i*K + k
                    compute_input_addr = baseA + (i_global_val * K + k_global_val);
                end
                
                OP_DEPTHWISE: begin
                    // Depthwise: fill per-channel blocks
                    out_h = i_global_val / out_W;
                    out_w = i_global_val % out_W;
                    channel_idx = k_global_val / K_block;
                    k_block_idx = k_global_val % K_block;
                    kh_i = k_block_idx / ker_W;
                    kw_i = k_block_idx % ker_W;
                    top_h = out_h * stride - padding;
                    left_w = out_w * stride - padding;
                    in_h = top_h + kh_i;
                    in_w = left_w + kw_i;
                    c_in = channel_idx;
                    
                    if (in_h >= 0 && in_h < act_H && in_w >= 0 && in_w < act_W) begin
                        compute_input_addr = input_offset(in_h, in_w, c_in);
                    end else begin
                        compute_input_addr = NULL_ADDR;
                    end
                end
                
                default: begin // REGULAR or POINTWISE
                    if (op_mode == OP_POINTWISE) begin
                        kh_i = 0;
                        kw_i = 0;
                        c_in = k_global_val;
                    end else begin
                        kh_i = k_global_val / (Kw_eff * act_CIN);
                        kw_i = (k_global_val % (Kw_eff * act_CIN)) / act_CIN;
                        c_in = k_global_val % act_CIN;
                    end
                    
                    out_h = i_global_val / out_W;
                    out_w = i_global_val % out_W;
                    top_h = out_h * stride - padding;
                    left_w = out_w * stride - padding;
                    in_h = top_h + kh_i;
                    in_w = left_w + kw_i;
                    
                    if (in_h >= 0 && in_h < act_H && in_w >= 0 && in_w < act_W) begin
                        compute_input_addr = input_offset(in_h, in_w, c_in);
                    end else begin
                        compute_input_addr = NULL_ADDR;
                    end
                end
            endcase
        end
    end
    endfunction

    // ----------------------------
    // B matrix address computation - DIAGONAL READING ORDER FOR DEPTHWISE
    function logic [ADDR_WIDTH-1:0] compute_kernel_addr;
        input logic [IDX_WIDTH-1:0] j_global_val;
        input logic [IDX_WIDTH-1:0] k_global_val;
        input logic [IDX_WIDTH-1:0] j_local_val;
        input logic [IDX_WIDTH-1:0] k_local_val;
        logic [IDX_WIDTH-1:0] kernel_size;
        logic [IDX_WIDTH-1:0] virtual_row, virtual_col;
        logic [IDX_WIDTH-1:0] kernel_idx_within_row;
    begin
        // Check boundary conditions FIRST
        if (j_local_val >= eTN || k_local_val >= eTK) begin
            compute_kernel_addr = NULL_ADDR;
        end else begin
            case (op_mode)
                OP_MATRIX_MUL: begin
                    // Matrix multiply: B[k][j] = baseB + k*N + j
                    compute_kernel_addr = baseB + (k_global_val * N + j_global_val);
                end
                
                OP_DEPTHWISE: begin
                    // DIAGONAL READING ORDER
                    kernel_size = get_kernel_size(); // K_block = ker_H * ker_W
                    
                    // Calculate position in virtual diagonal matrix
                    virtual_row = k_global_val;  // Which "row" in the virtual matrix we're reading
                    virtual_col = j_global_val;  // Which "column" in the virtual matrix we're reading
                    
                    // Check if we're on the diagonal for this reading position
                    if (virtual_row >= (virtual_col * kernel_size) && 
                        virtual_row < ((virtual_col + 1) * kernel_size)) begin
                        // We're in the valid diagonal region for this kernel
                        kernel_idx_within_row = virtual_row - (virtual_col * kernel_size);
                        compute_kernel_addr = baseB + (virtual_col * kernel_size + kernel_idx_within_row);
                    end else begin
                        // Null region - not on the diagonal
                        compute_kernel_addr = NULL_ADDR;
                    end
                end
                
                default: begin // REGULAR or POINTWISE
                    // Regular/Pointwise: Read kernels in row-major order
                    kernel_size = get_kernel_size();
                    compute_kernel_addr = baseB + (j_global_val * kernel_size + k_global_val);
                end
            endcase
        end
    end
    endfunction

    // ----------------------------
    // C matrix address computation
    function logic [ADDR_WIDTH-1:0] compute_output_addr;
        input logic [IDX_WIDTH-1:0] i_global_val;
        input logic [IDX_WIDTH-1:0] j_global_val;
        input logic [IDX_WIDTH-1:0] i_local_val;
        input logic [IDX_WIDTH-1:0] j_local_val;
    begin
        if (i_local_val >= eTM || j_local_val >= eTN) begin
            compute_output_addr = NULL_ADDR;
        end else begin
            // Same for all modes: C[i][j] = baseC + i*N + j
            compute_output_addr = baseC + (i_global_val * N + j_global_val);
        end
    end
    endfunction

    // ----------------------------
    // Combinational address computation
    always_comb begin
        is_null_addr_comb = 1'b1;
        addr_out_comb = NULL_ADDR;
        id_out_comb = 2'b00;
        
        case (state)
            GEN_A: begin
                addr_out_comb = compute_input_addr(i_global, k_global, i_cnt, k_cnt);
                is_null_addr_comb = (addr_out_comb == NULL_ADDR);
                id_out_comb = 2'b00;
            end
            
            GEN_B: begin
                addr_out_comb = compute_kernel_addr(j_global, k_global, j_cnt, k_cnt);
                is_null_addr_comb = (addr_out_comb == NULL_ADDR);
                id_out_comb = 2'b01;
            end
            
            GEN_C: begin
                addr_out_comb = compute_output_addr(i_global, j_global, i_cnt, j_cnt);
                is_null_addr_comb = (addr_out_comb == NULL_ADDR);
                id_out_comb = 2'b10;
            end
            
            default: begin
                is_null_addr_comb = 1'b1;
                addr_out_comb = NULL_ADDR;
                id_out_comb = 2'b00;
            end
        endcase
    end

    // ----------------------------
    // FSM next state logic - now gated by read_req
    always_comb begin
        next_state = state;
        case (state)
            IDLE:      if (start_tile) next_state = GEN_A;
            GEN_A:     if (read_req && (i_cnt == TM-1) && (k_cnt == TK-1)) next_state = GEN_B;
            GEN_B:     if (read_req && (j_cnt == TN-1) && (k_cnt == TK-1)) next_state = GEN_C;
            GEN_C:     if (read_req && (i_cnt == TM-1) && (j_cnt == TN-1)) next_state = TILE_DONE;
            TILE_DONE: next_state = IDLE;
        endcase
    end

    // ----------------------------
    // Sequential behavior
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            i_cnt <= '0;
            j_cnt <= '0;
            k_cnt <= '0;
            addr_out_reg <= NULL_ADDR;
            id_out_reg <= 2'b00;
            valid_out_reg <= 1'b0;
            is_null_addr_reg <= 1'b1;
            read_req_prev <= 1'b0;
        end else begin
            read_req_prev <= read_req;
            
            state <= next_state;

            // Default: outputs are invalid unless we have a read request
            valid_out_reg <= 1'b0;
            addr_out_reg <= NULL_ADDR;
            id_out_reg <= 2'b00;
            is_null_addr_reg <= 1'b1;

            // Only update outputs and counters when read_req is active
            // if (read_req && (state inside {GEN_A, GEN_B, GEN_C})) begin
            if (read_req && (state == GEN_A || state == GEN_B || state == GEN_C)) begin
                valid_out_reg <= 1'b1;
                addr_out_reg <= addr_out_comb;
                id_out_reg <= id_out_comb;
                is_null_addr_reg <= is_null_addr_comb;
            end

            case (state)
                IDLE: begin
                    i_cnt <= '0;
                    j_cnt <= '0;
                    k_cnt <= '0;
                    // Keep outputs invalid in IDLE
                    valid_out_reg <= 1'b0;
                    addr_out_reg <= NULL_ADDR;
                    id_out_reg <= 2'b00;
                    is_null_addr_reg <= 1'b1;
                end

                GEN_A: begin
                    if (read_req) begin
                        if (k_cnt + 1 < TK) begin
                            k_cnt <= k_cnt + 1;
                        end else begin
                            k_cnt <= '0;
                            if (i_cnt + 1 < TM) begin
                                i_cnt <= i_cnt + 1;
                            end
                        end
                    end
                end

                GEN_B: begin
                    if (read_req) begin
                        if (k_cnt + 1 < TK) begin
                            k_cnt <= k_cnt + 1;
                        end else begin
                            k_cnt <= '0;
                            if (j_cnt + 1 < TN) begin
                                j_cnt <= j_cnt + 1;
                            end
                        end
                    end
                end

                GEN_C: begin
                    if (read_req) begin
                        if (j_cnt + 1 < TN) begin
                            j_cnt <= j_cnt + 1;
                        end else begin
                            j_cnt <= '0;
                            if (i_cnt + 1 < TM) begin
                                i_cnt <= i_cnt + 1;
                            end
                        end
                    end
                end

                TILE_DONE: begin
                    // Ready for next tile - outputs remain invalid
                    valid_out_reg <= 1'b0;
                    addr_out_reg <= NULL_ADDR;
                    id_out_reg <= 2'b00;
                    is_null_addr_reg <= 1'b1;
                end
            endcase

            // Reset counters when changing states
            if (state != next_state) begin
                i_cnt <= '0;
                j_cnt <= '0;
                k_cnt <= '0;
            end
        end
    end

endmodule