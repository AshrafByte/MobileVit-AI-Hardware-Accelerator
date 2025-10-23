//==============================================================================
// Package: accelerator_common_pkg
// Description: Common types and parameters shared across all accelerator modules
//==============================================================================

package accelerator_common_pkg;

    //==========================================================================
    // Width Parameters
    //==========================================================================
    parameter int ADDR_WIDTH = 32;           // Address bus width
    parameter int IDX_WIDTH  = 16;           // Index width for dimensions (16-bit to match AGU)
    parameter int DATA_WIDTH = 32;           // Data width for memory interface
    parameter int ACT_WIDTH  = 8;            // Activation data width (INT8)
    parameter int WEIGHT_WIDTH = 8;          // Weight data width (INT8)
    parameter int PSUM_WIDTH = 32;           // Partial sum width (INT32)
    
    //==========================================================================
    // Memory Buffer Parameters
    //==========================================================================
    parameter int ACTBUF_DEPTH   = 8192;     // 32KB / 4 bytes = 8K entries
    parameter int WGTBUF_DEPTH   = 8192;     // 32KB / 4 bytes = 8K entries
    parameter int PSUMBUF_DEPTH  = 16384;    // 64KB / 4 bytes = 16K entries
    parameter int NUM_BANKS      = 4;        // Number of memory banks
    
    //==========================================================================
    // Systolic Array Parameters
    //==========================================================================
    parameter int SA_SIZE        = 64;       // Lego SA supports up to 64 elements
    parameter int SA_ROWS        = 16;       // Base unit is 16x16
    parameter int SA_COLS        = 16;
    
    //==========================================================================
    // Common Typedefs (for readability)
    //==========================================================================
    
    // Address and index types
    typedef logic [ADDR_WIDTH-1:0] addr_t;   // 32-bit address
    typedef logic [IDX_WIDTH-1:0]  idx_t;    // 16-bit index
    
    // Common data widths
    typedef logic [7:0]   byte_t;            // 8-bit unsigned
    typedef logic [15:0]  word_t;            // 16-bit unsigned
    typedef logic [31:0]  dword_t;           // 32-bit unsigned
    typedef logic [63:0]  qword_t;           // 64-bit unsigned
    
    typedef logic signed [7:0]   sbyte_t;    // 8-bit signed (INT8 for acts/weights)
    typedef logic signed [15:0]  sword_t;    // 16-bit signed  
    typedef logic signed [31:0]  sdword_t;   // 32-bit signed (INT32 for psums)
    
    //==========================================================================
    // Operation Modes (matching AGU op_mode)
    //==========================================================================
    typedef enum logic [1:0] {
        OP_REGULAR_CONV  = 2'b00,  // Regular convolution (3x3, 5x5, etc.)
        OP_POINTWISE     = 2'b01,  // 1x1 convolution
        OP_DEPTHWISE     = 2'b10,  // Depthwise convolution
        OP_MATMUL        = 2'b11   // Matrix multiply (for attention)
    } op_mode_t;
    
    //==========================================================================
    // Layer Types (from MobileViT architecture)
    //==========================================================================
    typedef enum logic [3:0] {
        LAYER_CONV3X3       = 4'b0000,  // Standard 3x3 convolution
        LAYER_CONV1X1       = 4'b0001,  // 1x1 pointwise convolution
        LAYER_DWCONV3X3     = 4'b0010,  // Depthwise 3x3 convolution
        LAYER_MV2_BLOCK     = 4'b0011,  // MobileNetV2 block
        LAYER_MATMUL_Q      = 4'b0100,  // Q = X * W_Q
        LAYER_MATMUL_K      = 4'b0101,  // K = X * W_K
        LAYER_MATMUL_V      = 4'b0110,  // V = X * W_V
        LAYER_MATMUL_QKT    = 4'b0111,  // Q * K^T
        LAYER_MATMUL_ATTN_V = 4'b1000,  // Attention * V
        LAYER_GLOBAL_POOL   = 4'b1001,  // Global pooling
        LAYER_FC            = 4'b1010,  // Fully connected
        LAYER_MVB           = 4'b1011   // MobileViT block (full transformer)
    } layer_type_t;
    
    //==========================================================================
    // Memory Buffer IDs
    //==========================================================================
    typedef enum logic [1:0] {
        BUF_ACTBUF_A = 2'b00,  // Activation buffer A (ping)
        BUF_ACTBUF_B = 2'b01,  // Activation buffer B (pong)
        BUF_WGTBUF   = 2'b10,  // Weight buffer
        BUF_PSUMBUF  = 2'b11   // Partial sum buffer
    } buffer_id_t;
    
    //==========================================================================
    // Systolic Array Type (Lego SA TYPE_Lego parameter)
    //==========================================================================
    typedef enum logic [1:0] {
        SA_TYPE_16X64 = 2'b00,  // 16x64 configuration
        SA_TYPE_32X32 = 2'b01,  // 32x32 configuration
        SA_TYPE_64X16 = 2'b10,  // 64x16 configuration
        SA_TYPE_16X16 = 2'b11   // 16x16 configuration (not used in Lego)
    } sa_type_t;
    
    //==========================================================================
    // Descriptor Structure (256 bits = 8x 32-bit words)
    //==========================================================================
    typedef struct packed {
        logic [31:0] dram_addr;      // [255:224] Source/dest DRAM address
        logic [15:0] sram_addr;      // [223:208] Target SRAM address offset
        logic [15:0] length;         // [207:192] Transfer length in bytes
        logic [15:0] stride;         // [191:176] Stride for 2D transfers
        logic [111:0] reserved1;     // [175:64]  Reserved for future use
        logic [15:0] tile_h;         // [63:48]   Tile height
        logic [15:0] tile_w;         // [47:32]   Tile width
        logic [15:0] c_in;           // [31:16]   Input channels for this tile
        logic [7:0]  flags;          // [15:8]    Control flags
        logic [7:0]  reserved2;      // [7:0]     Reserved
    } descriptor_t;
    
    // Descriptor flags bit definitions
    parameter int FLAG_IS_WEIGHT     = 0;  // Bit 0: 1=weight data, 0=activation
    parameter int FLAG_IS_LAST_TILE  = 1;  // Bit 1: 1=last tile in sequence
    parameter int FLAG_ENABLE_BN     = 2;  // Bit 2: 1=enable batch norm
    parameter int FLAG_ENABLE_SWISH  = 3;  // Bit 3: 1=enable Swish activation
    parameter int FLAG_ENABLE_LN     = 4;  // Bit 4: 1=enable layer norm
    parameter int FLAG_TRANSPOSE     = 5;  // Bit 5: 1=transpose weights (for K^T)
    parameter int FLAG_ACCUMULATE    = 6;  // Bit 6: 1=accumulate to existing psum
    parameter int FLAG_WRITEBACK     = 7;  // Bit 7: 1=writeback result to DRAM
    
    //==========================================================================
    // Special Address Values
    //==========================================================================
    parameter addr_t NULL_ADDR = 32'hFFFF_FFFF;  // Invalid address marker
    
    //==========================================================================
    // AXI Parameters
    //==========================================================================
    parameter int AXI_ADDR_WIDTH = 32;
    parameter int AXI_DATA_WIDTH = 64;   // 64-bit standard width (compatible with DDR3/DDR4)
    parameter int AXI_ID_WIDTH   = 4;
    parameter int AXI_USER_WIDTH = 1;
    
endpackage : accelerator_common_pkg
