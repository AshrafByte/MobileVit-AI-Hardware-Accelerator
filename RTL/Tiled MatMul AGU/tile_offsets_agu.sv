
module tile_offsets_agu #(
    parameter int ADDR_WIDTH = 32,
    parameter int IDX_WIDTH  = 8,
    parameter logic [ADDR_WIDTH-1:0] NULL_ADDR = 32'd9999_9999
)(
    input  logic clk,
    input  logic rst,
    input  logic start_tile,
    input  logic read_req,

    input  logic [ADDR_WIDTH-1:0] baseA_tile,
    input  logic [ADDR_WIDTH-1:0] baseB_tile,
    input  logic [ADDR_WIDTH-1:0] baseC_tile,

    input  logic [IDX_WIDTH-1:0] eTM, eTN, eTK,
    input  logic [IDX_WIDTH-1:0] TM_cfg, TN_cfg, TK_cfg,
    input  logic [IDX_WIDTH-1:0] FULL_K, FULL_N,

    output logic [ADDR_WIDTH-1:0] o_addr,
    output logic [1:0] addr_id,
    output logic valid,

    output logic tile_done,
    output logic tile_ready
);

    typedef enum logic [1:0] {ST_IDLE, ST_INIT, ST_GEN} st_t;
    typedef enum logic [1:0] {PH_SKIP, PH_A, PH_B, PH_C} phase_t;

    st_t st, st_next;
    phase_t phase, phase_next;

    logic [IDX_WIDTH-1:0] i_idx, j_idx, k_idx;
    logic [IDX_WIDTH-1:0] i_nxt, j_nxt, k_nxt;

    logic [ADDR_WIDTH-1:0] prev_baseA;

    //=====================================================================
    // Sequential logic
    //=====================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            st <= ST_IDLE;
            phase <= PH_SKIP;
            i_idx <= 0;
            j_idx <= 0;
            k_idx <= 0;
            tile_done <= 0;
            prev_baseA <= NULL_ADDR;
        end else begin
            st <= st_next;
            phase <= phase_next;
            i_idx <= i_nxt;
            j_idx <= j_nxt;
            k_idx <= k_nxt;

            tile_done <= (phase == PH_C && i_idx == TM_cfg-1 && j_idx == TN_cfg-1);

            // Capture baseA only when tile done
			if(phase == PH_C)
                prev_baseA <= baseA_tile;
        end
    end

    //=====================================================================
    // Combinational logic
    //=====================================================================
    always_comb begin
        // Defaults
        valid = 1'b0;
        o_addr = NULL_ADDR;
        addr_id = 2'b11;
        tile_ready = (st == ST_IDLE);

        st_next = st;
        phase_next = phase;
        i_nxt = i_idx;
        j_nxt = j_idx;
        k_nxt = k_idx;

        case (st)
            //=========================================================
            ST_IDLE: begin
                if (start_tile) begin
                    st_next = ST_INIT;
                    phase_next = PH_SKIP;
                end
            end

            //=========================================================
            ST_INIT: begin
                st_next = ST_GEN;
                i_nxt = 0;
                j_nxt = 0;
                k_nxt = 0;
                phase_next = PH_SKIP;
            end

            //=========================================================
            ST_GEN: begin
                valid = read_req;

                case (phase)
                    PH_SKIP: begin
                        if (read_req)
                            phase_next = (baseA_tile == prev_baseA) ? PH_B : PH_A;
                    end

                    PH_A: begin
                        addr_id = 2'd0;
                        o_addr = ((i_idx < eTM) && (k_idx < eTK))
                            ? baseA_tile + (i_idx * FULL_K) + k_idx : NULL_ADDR;

                        if (k_idx + 1 < TK_cfg)
                            k_nxt = k_idx + 1;
                        else if (i_idx + 1 < TM_cfg) begin
                            i_nxt = i_idx + 1;
                            k_nxt = 0;
                        end else begin
                            i_nxt = 0;
                            k_nxt = 0;
                            phase_next = PH_B;
                        end
                    end

                    PH_B: begin
                        addr_id = 2'd1;
                        o_addr = ((k_idx < eTK) && (j_idx < eTN))
                            ? baseB_tile + (k_idx * FULL_N) + j_idx : NULL_ADDR;

                        if (k_idx + 1 < TK_cfg)
                            k_nxt = k_idx + 1;
                        else if (j_idx + 1 < TN_cfg) begin
                            j_nxt = j_idx + 1;
                            k_nxt = 0;
                        end else begin
                            j_nxt = 0;
                            k_nxt = 0;
                            phase_next = PH_C;
                        end
                    end

                    PH_C: begin
                        addr_id = 2'd2;
                        o_addr = ((i_idx < eTM) && (j_idx < eTN))
                            ? baseC_tile + (i_idx * FULL_N) + j_idx : NULL_ADDR;

                        if (j_idx + 1 < TN_cfg)
                            j_nxt = j_idx + 1;
                        else if (i_idx + 1 < TM_cfg) begin
                            j_nxt = 0;
                            i_nxt = i_idx + 1;
                        end else begin
                            phase_next = PH_SKIP;
                            st_next = ST_IDLE;
                        end
                    end
                endcase
            end
        endcase
    end
endmodule