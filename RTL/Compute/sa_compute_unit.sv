//==============================================================================
// Module: sa_compute_unit
// Description: Wrapper for Lego Systolic Array with accumulation support
//              Handles multi-tile accumulation for large C_in dimensions
//              Provides clean interface for global controller
//
// Author: MobileViT Accelerator Team
// Date: October 23, 2025
//==============================================================================

module sa_compute_unit 
    import accelerator_common_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    
    //==========================================================================
    // Control Interface
    //==========================================================================
    input  logic         start,           // Start computation
    input  logic         load_weights,    // Load weight phase
    input  sa_type_t     sa_type,         // SA configuration type
    input  logic         transpose_en,    // Enable transpose (for K^T)
    input  logic         accum_en,        // Enable accumulation mode
    
    //==========================================================================
    // Data Interface from Memory Subsystem
    //==========================================================================
    input  logic [ACT_WIDTH-1:0]     act_in[SA_SIZE],      // 64 activation inputs
    input  logic [WEIGHT_WIDTH-1:0]  weight_in[SA_SIZE],   // 64 weight inputs
    input  logic [PSUM_WIDTH-1:0]    psum_in[SA_SIZE],     // Previous partial sums (for accumulation)
    input  logic                     data_valid_in,        // Input data valid
    
    //==========================================================================
    // Data Interface to Memory Subsystem
    //==========================================================================
    output logic [PSUM_WIDTH-1:0]    psum_out[SA_SIZE],    // 64 partial sum outputs
    output logic                     valid_out,            // Output valid
    output logic                     done                  // Computation done
);

    //==========================================================================
    // Internal Signals
    //==========================================================================
    logic [PSUM_WIDTH-1:0] sa_psum_raw[SA_SIZE];  // Raw output from Lego SA
    logic                  sa_valid_raw;           // Raw valid from SA
    
    //==========================================================================
    // Lego Systolic Array Instantiation
    //==========================================================================
    Lego_Systolic_Array #(
        .DATA_W(ACT_WIDTH),       // 8-bit activations/weights
        .DATA_W_OUT(PSUM_WIDTH)   // 32-bit partial sums
    ) u_lego_sa (
        .clk(clk),
        .rst_n(rst_n),
        .valid_in(data_valid_in),
        .act_in(act_in),
        .weight_in(weight_in),
        .TYPE_Lego(sa_type),
        .load_w(load_weights),
        .transpose_en(transpose_en),
        .psum_out(sa_psum_raw),
        .valid_out(sa_valid_raw)
    );
    
    //==========================================================================
    // Accumulation Logic
    // When accum_en=1: add new partial sums to previous partial sums
    // When accum_en=0: use fresh partial sums from SA
    //==========================================================================
    genvar i;
    generate
        for (i = 0; i < SA_SIZE; i++) begin : gen_accumulator
            always_comb begin
                if (accum_en && sa_valid_raw) begin
                    psum_out[i] = sa_psum_raw[i] + psum_in[i];
                end else begin
                    psum_out[i] = sa_psum_raw[i];
                end
            end
        end
    endgenerate
    
    //==========================================================================
    // Output Control
    //==========================================================================
    assign valid_out = sa_valid_raw;
    
    // Done signal: indicate computation complete
    // For now, simply pass through valid_out
    // Can be enhanced with cycle counter based on SA type
    assign done = sa_valid_raw;
    
endmodule
