module tile_base_controller #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 8
)(
    input  logic clk,
    input  logic rst,

    // single request control signal
    input  logic tile_req,         // request to start tile generation
    output logic done_all,         // all tiles done

    // matrix dimensions
    input  logic [IDX_WIDTH-1:0] M, N, K,
    input  logic [IDX_WIDTH-1:0] TM_cfg, TN_cfg, TK_cfg,

    // base addresses
    input  logic [ADDR_WIDTH-1:0] baseA,
    input  logic [ADDR_WIDTH-1:0] baseB,
    input  logic [ADDR_WIDTH-1:0] baseC,

    // handshake with AGU
    output logic start_tile,       // one-cycle pulse to start AGU
    input  logic tile_done,        // AGU finished
    input  logic tile_ready,       // AGU ready

    // outputs
    output logic [ADDR_WIDTH-1:0] baseA_tile,
    output logic [ADDR_WIDTH-1:0] baseB_tile,
    output logic [ADDR_WIDTH-1:0] baseC_tile,
    output logic [IDX_WIDTH-1:0] eTM, eTN, eTK
);

    // internal tile indices
    logic [IDX_WIDTH-1:0] tile_m, tile_n, tile_k;
    logic [IDX_WIDTH-1:0] next_tile_m, next_tile_n, next_tile_k;

    typedef enum logic [1:0] {ST_IDLE, ST_WAIT_DONE} st_t;
    st_t st, st_next;

    // ============================================================
    // Sequential logic
    // ============================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            st <= ST_IDLE;
            tile_m <= 0;
            tile_n <= 0;
            tile_k <= 0;
        end else begin
            st <= st_next;
            tile_m <= next_tile_m;
            tile_n <= next_tile_n;
            tile_k <= next_tile_k;
        end
    end

    // ============================================================
    // Combinational logic
    // ============================================================
    always_comb begin
        // defaults
        st_next      = st;
        start_tile   = 0;
        done_all     = 0;
        next_tile_m  = tile_m;
        next_tile_n  = tile_n;
        next_tile_k  = tile_k;

        // base addresses
		baseA_tile = baseA + (tile_m * TM_cfg * K) + (tile_k * TK_cfg);
		baseB_tile = baseB + (tile_k * TK_cfg * N) + (tile_n * TN_cfg);
		baseC_tile = baseC + (tile_m * TM_cfg * N) + (tile_n * TN_cfg);

        // effective sizes
        eTM = (M > tile_m * TM_cfg) ? ((M - tile_m * TM_cfg) < TM_cfg ? (M - tile_m * TM_cfg) : TM_cfg) : 0;
        eTN = (N > tile_n * TN_cfg) ? ((N - tile_n * TN_cfg) < TN_cfg ? (N - tile_n * TN_cfg) : TN_cfg) : 0;
        eTK = (K > tile_k * TK_cfg) ? ((K - tile_k * TK_cfg) < TK_cfg ? (K - tile_k * TK_cfg) : TK_cfg) : 0;

        // FSM
        case (st)
            //-------------------------------------------------
            ST_IDLE: begin
                if (tile_req && tile_ready) begin
                    start_tile = 1;
                    st_next = ST_WAIT_DONE;
                end
            end

            //-------------------------------------------------
            ST_WAIT_DONE: begin
                if (tile_done) begin
                    // Move to next tile indices
                    if (tile_k + 1 < (K + TK_cfg - 1) / TK_cfg) begin
                        next_tile_k = tile_k + 1;
                    end else if (tile_n + 1 < (N + TN_cfg - 1) / TN_cfg) begin
                        next_tile_k = 0;
                        next_tile_n = tile_n + 1;
                    end else if (tile_m + 1 < (M + TM_cfg - 1) / TM_cfg) begin
                        next_tile_k = 0;
                        next_tile_n = 0;
                        next_tile_m = tile_m + 1;
                    end else begin
                        done_all = 1;
                        st_next = ST_IDLE;
                    end

                    // Start next tile only if user requests again
                    if (!done_all && tile_req && tile_ready)
                        start_tile = 1;
                end
            end
        endcase
    end

endmodule
