//==============================================================================
// Package: accelerator_common_pkg
// Description: Common types and parameters shared across all accelerator modules
//==============================================================================

package accelerator_common_pkg;

    //==========================================================================
    // Width Parameters
    //==========================================================================
    parameter int ADDR_WIDTH = 32;           // Address bus width
    parameter int IDX_WIDTH  = 8;            // Index width for dimensions
    
    //==========================================================================
    // Common Typedefs (for readability)
    //==========================================================================
    
    // Address and index types
    typedef logic [ADDR_WIDTH-1:0] addr_t;   // 32-bit address
    typedef logic [IDX_WIDTH-1:0]  idx_t;    // 8-bit index
    
    // Common data widths
    typedef logic [7:0]   byte_t;            // 8-bit unsigned
    typedef logic [15:0]  word_t;            // 16-bit unsigned
    typedef logic [31:0]  dword_t;           // 32-bit unsigned
    
    typedef logic signed [7:0]   sbyte_t;    // 8-bit signed
    typedef logic signed [15:0]  sword_t;    // 16-bit signed  
    typedef logic signed [31:0]  sdword_t;   // 32-bit signed
    
    //==========================================================================
    // Special Address Values
    //==========================================================================
    parameter addr_t NULL_ADDR = 32'h9999_9999;  // Invalid address marker
    
endpackage : accelerator_common_pkg
