//==============================================================================
// Package: accelerator_matmul_pkg  
// Description: Matrix multiplication and tiled AGU types and parameters
//==============================================================================

package accelerator_matmul_pkg;

    import accelerator_common_pkg::*;

    //==========================================================================
    // Matrix ID Type (for addr_id signal)
    //==========================================================================
    typedef enum logic [1:0] {
        MAT_A       = 2'd0,
        MAT_B       = 2'd1,
        MAT_C       = 2'd2,
        MAT_INVALID = 2'd3
    } matrix_id_t;
    
    //==========================================================================
    // Tile Controller FSM States
    //==========================================================================
    typedef enum logic [1:0] {
        TILE_CTRL_IDLE      = 2'b00,
        TILE_CTRL_WAIT_DONE = 2'b01
    } tile_ctrl_state_t;
    
    //==========================================================================
    // Tile AGU FSM States  
    //==========================================================================
    typedef enum logic [1:0] {
        AGU_IDLE = 2'b00,
        AGU_INIT = 2'b01,
        AGU_GEN  = 2'b10
    } tile_agu_state_t;
    
    //==========================================================================
    // Tile AGU Phase
    //==========================================================================
    typedef enum logic [1:0] {
        PHASE_SKIP = 2'b00,
        PHASE_A    = 2'b01,
        PHASE_B    = 2'b10,
        PHASE_C    = 2'b11
    } tile_phase_t;

endpackage : accelerator_matmul_pkg
