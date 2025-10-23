//==============================================================================
// Module: global_controller
// Description: Top-level FSM that orchestrates the entire accelerator
//              Manages: DMA transfers, AGU address generation, SA computation,
//              post-processing (BN/Swish/LN), and ping-pong buffer management
//
// Author: MobileViT Accelerator Team
// Date: October 23, 2025
//==============================================================================

module global_controller 
    import accelerator_common_pkg::*;
(
    input  logic clk,
    input  logic rst_n,
    
    //==========================================================================
    // Control Interface (from CPU via AXI Slave)
    //==========================================================================
    input  logic         start,              // Start processing
    output logic         busy,               // Accelerator is busy
    output logic         done,               // Processing complete
    output logic         error,              // Error occurred
    
    //==========================================================================
    // Descriptor Interface (from descriptor FIFO)
    //==========================================================================
    input  descriptor_t  descriptor,         // Current descriptor
    input  logic         desc_valid,         // Descriptor is valid
    output logic         desc_ready,         // Ready to accept descriptor
    
    //==========================================================================
    // DMA Control Interface
    //==========================================================================
    output logic         dma_start_read,     // Start DMA read (DRAM → SRAM)
    output logic         dma_start_write,    // Start DMA write (SRAM → DRAM)
    output addr_t        dma_src_addr,       // Source address
    output addr_t        dma_dst_addr,       // Destination address
    output word_t        dma_length,         // Transfer length in bytes
    output buffer_id_t   dma_target_buf,     // Which buffer to write to
    input  logic         dma_done,           // DMA transfer complete
    input  logic         dma_error,          // DMA error
    
    //==========================================================================
    // AGU Control Interface
    //==========================================================================
    output logic         agu_tile_req,       // Request new tile
    input  logic         agu_tile_done,      // Tile generation complete
    input  logic         agu_all_tiles_done, // All tiles processed
    output logic         agu_read_req,       // Request address within tile
    input  logic         agu_ready,          // AGU ready
    input  logic         agu_processing,     // AGU is processing
    
    // AGU Configuration
    output op_mode_t     agu_op_mode,        // Operation mode
    output idx_t         agu_act_H,          // Activation height
    output idx_t         agu_act_W,          // Activation width
    output idx_t         agu_act_CIN,        // Input channels
    output idx_t         agu_ker_H,          // Kernel height
    output idx_t         agu_ker_W,          // Kernel width
    output idx_t         agu_out_chs,        // Output channels
    output idx_t         agu_padding,        // Padding
    output idx_t         agu_stride,         // Stride
    output idx_t         agu_mat_M,          // Matrix M dimension
    output idx_t         agu_mat_K,          // Matrix K dimension
    output idx_t         agu_mat_N,          // Matrix N dimension
    output idx_t         agu_TM,             // Tile M
    output idx_t         agu_TN,             // Tile N
    output idx_t         agu_TK,             // Tile K
    output addr_t        agu_baseA,          // Base address A
    output addr_t        agu_baseB,          // Base address B
    output addr_t        agu_baseC,          // Base address C
    
    //==========================================================================
    // Systolic Array Control Interface
    //==========================================================================
    output logic         sa_start,           // Start computation
    output logic         sa_load_w,          // Load weights
    output sa_type_t     sa_type,            // SA type selection
    output logic         sa_transpose_en,    // Enable transpose for K^T
    input  logic         sa_valid_out,       // Output is valid
    
    //==========================================================================
    // Memory Subsystem Control Interface
    //==========================================================================
    output logic         mem_ping_pong_sel,  // 0=ActBufA, 1=ActBufB
    output logic         mem_accum_mode,     // 1=accumulate psums
    output logic         mem_clear_psum,     // 1=clear psum buffer
    output logic         mem_write_enable,   // Enable psum writeback
    
    //==========================================================================
    // Post-Processing Control Interface
    //==========================================================================
    output logic         pp_bn_enable,       // Enable batch normalization
    output logic         pp_swish_enable,    // Enable Swish activation
    output logic         pp_ln_enable,       // Enable layer normalization
    
    //==========================================================================
    // Status & Debug
    //==========================================================================
    output logic [15:0]  tile_counter,       // Current tile being processed
    output logic [31:0]  cycle_counter,      // Cycle counter for profiling
    output logic         irq                 // Interrupt request
);

    //==========================================================================
    // FSM States
    //==========================================================================
    typedef enum logic [4:0] {
        IDLE,                   // Waiting for start signal
        FETCH_DESCRIPTOR,       // Fetch and decode descriptor
        LOAD_WEIGHTS,           // Load weights into WgtBuf via DMA
        WAIT_WEIGHT_DMA,        // Wait for weight DMA to complete
        WEIGHT_LOAD_SA,         // Load weights from WgtBuf into SA PEs
        CLEAR_PSUM,             // Clear partial sum buffer
        LOAD_ACT_PING,          // Load activations into ActBufA
        WAIT_ACT_PING_DMA,      // Wait for activation DMA
        AGU_SETUP_PING,         // Setup AGU and wait for ready
        COMPUTE_PING,           // Compute using ActBufA
        LOAD_ACT_PONG,          // Load activations into ActBufB (overlap)
        WAIT_ACT_PONG_DMA,      // Wait for activation DMA
        AGU_SETUP_PONG,         // Setup AGU and wait for ready  
        COMPUTE_PONG,           // Compute using ActBufB
        APPLY_POST_PROCESS,     // Apply BN/Swish/LN
        WRITEBACK,              // Write results back to DRAM
        WAIT_WRITEBACK,         // Wait for writeback DMA
        LAYER_DONE,             // Layer processing complete
        ERROR_STATE             // Error occurred
    } state_t;
    
    state_t state, next_state;
    
    //==========================================================================
    // Internal Registers
    //==========================================================================
    descriptor_t desc_reg;              // Latched descriptor
    logic [15:0] tile_count;            // Tiles processed so far
    logic [15:0] total_tiles;           // Total tiles to process
    logic        is_first_tile;         // First tile in sequence
    logic        is_last_tile;          // Last tile in sequence
    logic        weights_loaded;        // Weights already loaded
    logic        ping_active;           // Using ActBufA (ping)
    logic [31:0] cycles;                // Cycle counter
    logic [31:0] act_tile_offset;       // Offset for next activation tile
    logic [7:0]  weight_load_cycles;    // Counter for weight loading phase
    logic        agu_setup_done;        // AGU configuration complete
    
    //==========================================================================
    // FSM Sequential Logic
    //==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    //==========================================================================
    // FSM Combinational Logic
    //==========================================================================
    always_comb begin
        // Default assignments
        next_state = state;
        
        case (state)
            IDLE: begin
                if (start && desc_valid) begin
                    next_state = FETCH_DESCRIPTOR;
                end
            end
            
            FETCH_DESCRIPTOR: begin
                // Always load weights first unless already loaded
                if (!weights_loaded) begin
                    next_state = LOAD_WEIGHTS;
                end else begin
                    next_state = CLEAR_PSUM;
                end
            end
            
            LOAD_WEIGHTS: begin
                next_state = WAIT_WEIGHT_DMA;
            end
            
            WAIT_WEIGHT_DMA: begin
                if (dma_done) begin
                    next_state = WEIGHT_LOAD_SA;
                end else if (dma_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            WEIGHT_LOAD_SA: begin
                // Load weights into SA PEs (takes ~32 cycles for 16x16)
                if (weight_load_cycles >= 8'd32) begin
                    next_state = CLEAR_PSUM;
                end
            end
            
            CLEAR_PSUM: begin
                // Single cycle to clear psum buffer
                next_state = LOAD_ACT_PING;
            end
            
            LOAD_ACT_PING: begin
                next_state = WAIT_ACT_PING_DMA;
            end
            
            WAIT_ACT_PING_DMA: begin
                if (dma_done) begin
                    next_state = AGU_SETUP_PING;
                end else if (dma_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            AGU_SETUP_PING: begin
                // Wait for AGU to be ready and configured
                if (agu_ready && agu_setup_done) begin
                    next_state = COMPUTE_PING;
                end
            end
            
            COMPUTE_PING: begin
                // Wait for tile computation to complete
                if (agu_tile_done) begin
                    if (tile_count < total_tiles - 1) begin
                        // More tiles: overlap loading next tile
                        next_state = LOAD_ACT_PONG;
                    end else begin
                        // Last tile, go to post-processing
                        next_state = APPLY_POST_PROCESS;
                    end
                end
            end
            
            LOAD_ACT_PONG: begin
                next_state = WAIT_ACT_PONG_DMA;
            end
            
            WAIT_ACT_PONG_DMA: begin
                if (dma_done) begin
                    next_state = AGU_SETUP_PONG;
                end else if (dma_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            AGU_SETUP_PONG: begin
                // Wait for AGU to be ready
                if (agu_ready && agu_setup_done) begin
                    next_state = COMPUTE_PONG;
                end
            end
            
            COMPUTE_PONG: begin
                // Wait for tile completion
                if (agu_tile_done) begin
                    if (tile_count < total_tiles - 1) begin
                        // Continue with next tile (back to ping)
                        next_state = LOAD_ACT_PING;
                    end else begin
                        next_state = APPLY_POST_PROCESS;
                    end
                end
            end
            
            APPLY_POST_PROCESS: begin
                // Post-processing takes fixed number of cycles
                // For now, assume 10 cycles
                next_state = WRITEBACK;
            end
            
            WRITEBACK: begin
                if (desc_reg.flags[FLAG_WRITEBACK]) begin
                    next_state = WAIT_WRITEBACK;
                end else begin
                    next_state = LAYER_DONE;
                end
            end
            
            WAIT_WRITEBACK: begin
                if (dma_done) begin
                    next_state = LAYER_DONE;
                end else if (dma_error) begin
                    next_state = ERROR_STATE;
                end
            end
            
            LAYER_DONE: begin
                // Check if more descriptors available
                if (desc_valid) begin
                    next_state = FETCH_DESCRIPTOR;
                end else begin
                    next_state = IDLE;
                end
            end
            
            ERROR_STATE: begin
                // Stay in error until reset
                next_state = ERROR_STATE;
            end
            
            default: next_state = IDLE;
        endcase
    end
    
    //==========================================================================
    // Output Logic
    //==========================================================================
    always_comb begin
        // Default output values
        desc_ready       = 1'b0;
        dma_start_read   = 1'b0;
        dma_start_write  = 1'b0;
        dma_src_addr     = '0;
        dma_dst_addr     = '0;
        dma_length       = '0;
        dma_target_buf   = BUF_ACTBUF_A;
        agu_tile_req     = 1'b0;
        agu_read_req     = 1'b0;
        sa_start         = 1'b0;
        sa_load_w        = 1'b0;
        mem_ping_pong_sel = ping_active;
        mem_accum_mode   = !is_first_tile;  // Accumulate after first tile
        mem_clear_psum   = 1'b0;
        mem_write_enable = 1'b0;
        pp_bn_enable     = 1'b0;
        pp_swish_enable  = 1'b0;
        pp_ln_enable     = 1'b0;
        busy             = (state != IDLE);
        done             = (state == LAYER_DONE);
        error            = (state == ERROR_STATE);
        irq              = (state == LAYER_DONE);
        
        case (state)
            IDLE: begin
                desc_ready = 1'b1;
            end
            
            FETCH_DESCRIPTOR: begin
                desc_ready = 1'b1;  // Latch descriptor
            end
            
            LOAD_WEIGHTS: begin
                dma_start_read  = 1'b1;
                dma_src_addr    = desc_reg.dram_addr;
                dma_target_buf  = BUF_WGTBUF;
                dma_length      = desc_reg.length;
            end
            
            WEIGHT_LOAD_SA: begin
                // Load weights into SA PEs
                sa_load_w      = 1'b1;
                sa_start       = 1'b1;
                agu_read_req   = 1'b1;  // Request weight data from memory
            end
            
            CLEAR_PSUM: begin
                mem_clear_psum = 1'b1;
            end
            
            LOAD_ACT_PING: begin
                dma_start_read  = 1'b1;
                dma_src_addr    = desc_reg.dram_addr + act_tile_offset;
                dma_target_buf  = BUF_ACTBUF_A;
                dma_length      = desc_reg.tile_h * desc_reg.tile_w * desc_reg.c_in;
            end
            
            AGU_SETUP_PING: begin
                agu_tile_req   = 1'b1;  // Request tile configuration
                mem_ping_pong_sel = 1'b0;  // Use ActBufA
            end
            
            COMPUTE_PING: begin
                sa_start       = 1'b1;
                agu_read_req   = 1'b1;  // Request address generation
                mem_ping_pong_sel = 1'b0;  // Use ActBufA
                mem_write_enable  = 1'b1;
            end
            
            LOAD_ACT_PONG: begin
                dma_start_read  = 1'b1;
                dma_src_addr    = desc_reg.dram_addr + act_tile_offset;
                dma_target_buf  = BUF_ACTBUF_B;
                dma_length      = desc_reg.tile_h * desc_reg.tile_w * desc_reg.c_in;
            end
            
            AGU_SETUP_PONG: begin
                agu_tile_req   = 1'b1;  // Request tile configuration
                mem_ping_pong_sel = 1'b1;  // Use ActBufB
            end
            
            COMPUTE_PONG: begin
                sa_start       = 1'b1;
                agu_read_req   = 1'b1;
                mem_ping_pong_sel = 1'b1;  // Use ActBufB
                mem_write_enable  = 1'b1;
            end
            
            APPLY_POST_PROCESS: begin
                pp_bn_enable    = desc_reg.flags[FLAG_ENABLE_BN];
                pp_swish_enable = desc_reg.flags[FLAG_ENABLE_SWISH];
                pp_ln_enable    = desc_reg.flags[FLAG_ENABLE_LN];
            end
            
            WRITEBACK: begin
                dma_start_write = desc_reg.flags[FLAG_WRITEBACK];
                dma_src_addr    = desc_reg.sram_addr;  // Read from PSumBuf
                dma_dst_addr    = desc_reg.dram_addr;  // Write to DRAM
                dma_length      = desc_reg.tile_h * desc_reg.tile_w * desc_reg.c_in;
            end
            
            default: begin
                // All other states use default values
            end
        endcase
    end
    
    //==========================================================================
    // Control Registers Update
    //==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            desc_reg          <= '0;
            tile_count        <= '0;
            total_tiles       <= '0;
            is_first_tile     <= 1'b1;
            is_last_tile      <= 1'b0;
            weights_loaded    <= 1'b0;
            ping_active       <= 1'b0;
            cycles            <= '0;
            act_tile_offset   <= '0;
            weight_load_cycles <= '0;
            agu_setup_done    <= 1'b0;
        end else begin
            cycles <= cycles + 1;
            
            case (state)
                FETCH_DESCRIPTOR: begin
                    if (desc_valid) begin
                        desc_reg <= descriptor;
                        tile_count <= '0;
                        is_first_tile <= 1'b1;
                        // Calculate total tiles based on dimensions
                        // For simplicity, assume sequential tiling
                        total_tiles <= 1;  // Will be computed based on layer params
                    end
                end
                
                WAIT_WEIGHT_DMA: begin
                    if (dma_done) begin
                        weights_loaded <= 1'b1;
                        weight_load_cycles <= '0;  // Reset counter
                    end
                end
                
                WEIGHT_LOAD_SA: begin
                    // Count cycles during weight loading
                    weight_load_cycles <= weight_load_cycles + 1;
                end
                
                AGU_SETUP_PING, AGU_SETUP_PONG: begin
                    // Wait for AGU ready and mark setup as done
                    if (agu_ready && agu_tile_req) begin
                        agu_setup_done <= 1'b1;
                    end
                end
                
                COMPUTE_PING, COMPUTE_PONG: begin
                    agu_setup_done <= 1'b0;  // Reset for next tile
                    if (agu_tile_done) begin
                        tile_count <= tile_count + 1;
                        is_first_tile <= 1'b0;
                        is_last_tile <= (tile_count == total_tiles - 2);
                        act_tile_offset <= act_tile_offset + (desc_reg.tile_h * desc_reg.tile_w * desc_reg.c_in);
                    end
                end
                
                LOAD_ACT_PONG: begin
                    ping_active <= 1'b1;
                end
                
                LOAD_ACT_PING: begin
                    ping_active <= 1'b0;
                end
                
                LAYER_DONE: begin
                    weights_loaded <= 1'b0;  // Reset for next layer
                    act_tile_offset <= '0;
                    agu_setup_done <= 1'b0;
                end
                
                default: begin
                    // No state update for other states
                end
            endcase
        end
    end
    
    //==========================================================================
    // AGU Configuration (static for now, will be set from descriptor)
    //==========================================================================
    assign agu_op_mode   = OP_REGULAR_CONV;  // TODO: derive from descriptor
    assign agu_act_H     = desc_reg.tile_h;
    assign agu_act_W     = desc_reg.tile_w;
    assign agu_act_CIN   = desc_reg.c_in;
    assign agu_ker_H     = 16'd3;  // Default 3x3
    assign agu_ker_W     = 16'd3;
    assign agu_out_chs   = 16'd16; // Default
    assign agu_padding   = 16'd1;
    assign agu_stride    = desc_reg.stride;
    assign agu_mat_M     = 16'd0;
    assign agu_mat_K     = 16'd0;
    assign agu_mat_N     = 16'd0;
    assign agu_TM        = 16'd16;
    assign agu_TN        = 16'd16;
    assign agu_TK        = 16'd16;
    assign agu_baseA     = {16'h0000, desc_reg.sram_addr};  // ActBuf offset
    assign agu_baseB     = 32'h0000_0000;  // WgtBuf offset
    assign agu_baseC     = 32'h0001_0000;  // PSumBuf offset
    
    // SA Configuration
    assign sa_type        = SA_TYPE_16X64;  // Default to 16x64
    assign sa_transpose_en = desc_reg.flags[FLAG_TRANSPOSE];
    
    // Status outputs
    assign tile_counter  = tile_count;
    assign cycle_counter = cycles;
    
endmodule
