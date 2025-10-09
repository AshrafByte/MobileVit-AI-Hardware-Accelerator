//==============================================================================
// Package: accelerator_norm_pkg
// Description: Normalization (Batch Norm, Layer Norm) types and parameters
//==============================================================================

package accelerator_norm_pkg;

    import accelerator_common_pkg::*;

    //==========================================================================
    // Normalization Parameters
    //==========================================================================
    parameter int NORM_SCALE_FACTOR = 128;   // Scaling factor for layer norm
    parameter int SQRT_ITERATIONS   = 4;     // Newton-Raphson sqrt iterations
    
    //==========================================================================
    // Layer Normalization FSM States
    //==========================================================================
    typedef enum logic [2:0] {
        LN_IDLE = 3'd0,
        LN_LOAD = 3'd1,
        LN_SUM1 = 3'd3,
        LN_MEAN = 3'd2,
        LN_SUM2 = 3'd6,
        LN_VARI = 3'd7,
        LN_NORM = 3'd5,
        LN_DONE = 3'd4
    } layer_norm_state_t;

endpackage : accelerator_norm_pkg
