//==============================================================================
// Module: dma_wrapper
// Description: Wrapper for Xilinx AXI DMA IP
//              Provides simple control interface for the global controller
//              Handles data width conversion (64-bit AXI → 32-bit memory)
//
// Note: This module interfaces with Xilinx AXI DMA IP (axi_dma v7.1)
//       configured in Direct Register Mode
//
// Author: MobileViT Accelerator Team
// Date: October 23, 2025
//==============================================================================

module dma_wrapper 
    import accelerator_common_pkg::*;
#(
    parameter int BURST_LEN = 16  // Burst length in beats (16 beats × 64 bits = 128 bytes)
)(
    input  logic clk,
    input  logic rst_n,
    
    //==========================================================================
    // Control Interface (from Global Controller)
    //==========================================================================
    input  logic         start_read,      // Start DMA read (DRAM → SRAM)
    input  logic         start_write,     // Start DMA write (SRAM → DRAM)
    input  addr_t        src_addr,        // Source address (DRAM)
    input  addr_t        dst_addr,        // Destination address (DRAM for write)
    input  word_t        length,          // Transfer length in bytes
    input  buffer_id_t   target_buf,      // Target buffer for read
    output logic         done,            // Transfer complete
    output logic         error,           // Transfer error
    output logic         busy,            // DMA busy
    
    //==========================================================================
    // AXI Master Interface (Read Channel - MM2S)
    // Connected to Xilinx AXI DMA IP MM2S (Memory to Stream)
    //==========================================================================
    // AR Channel
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_mm2s_araddr,
    output logic [7:0]                 m_axi_mm2s_arlen,
    output logic [2:0]                 m_axi_mm2s_arsize,
    output logic [1:0]                 m_axi_mm2s_arburst,
    output logic                       m_axi_mm2s_arvalid,
    input  logic                       m_axi_mm2s_arready,
    
    // R Channel
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_mm2s_rdata,
    input  logic [1:0]                 m_axi_mm2s_rresp,
    input  logic                       m_axi_mm2s_rlast,
    input  logic                       m_axi_mm2s_rvalid,
    output logic                       m_axi_mm2s_rready,
    
    //==========================================================================
    // AXI Master Interface (Write Channel - S2MM)
    // Connected to Xilinx AXI DMA IP S2MM (Stream to Memory)
    //==========================================================================
    // AW Channel
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_s2mm_awaddr,
    output logic [7:0]                 m_axi_s2mm_awlen,
    output logic [2:0]                 m_axi_s2mm_awsize,
    output logic [1:0]                 m_axi_s2mm_awburst,
    output logic                       m_axi_s2mm_awvalid,
    input  logic                       m_axi_s2mm_awready,
    
    // W Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_s2mm_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] m_axi_s2mm_wstrb,
    output logic                       m_axi_s2mm_wlast,
    output logic                       m_axi_s2mm_wvalid,
    input  logic                       m_axi_s2mm_wready,
    
    // B Channel
    input  logic [1:0]                 m_axi_s2mm_bresp,
    input  logic                       m_axi_s2mm_bvalid,
    output logic                       m_axi_s2mm_bready,
    
    //==========================================================================
    // Memory Subsystem Interface (for writes during read DMA)
    //==========================================================================
    output logic [31:0]               mem_waddr,
    output logic [DATA_WIDTH-1:0]     mem_wdata,
    output logic                      mem_wen,
    output buffer_id_t                mem_target,
    
    //==========================================================================
    // Memory Subsystem Interface (for reads during write DMA)
    //==========================================================================
    output logic [31:0]               mem_raddr,
    output logic                      mem_ren,
    output buffer_id_t                mem_rsrc,
    input  logic [DATA_WIDTH-1:0]     mem_rdata,
    input  logic                      mem_rvalid
);

    //==========================================================================
    // Internal State Machine
    //==========================================================================
    typedef enum logic [2:0] {
        DMA_IDLE,
        DMA_READ_AR,
        DMA_READ_DATA,
        DMA_WRITE_AW,
        DMA_WRITE_DATA,
        DMA_WRITE_RESP,
        DMA_DONE,
        DMA_ERROR
    } dma_state_t;
    
    dma_state_t state, next_state;
    
    //==========================================================================
    // Internal Registers
    //==========================================================================
    addr_t       transfer_addr;      // Current transfer address
    word_t       bytes_remaining;    // Bytes left to transfer
    word_t       bytes_total;        // Total bytes to transfer
    buffer_id_t  current_target;     // Current target buffer
    logic [31:0] write_addr_offset;  // Offset for memory writes (word address)
    logic [7:0]  beat_count;         // Beat counter for burst
    logic        word_count;         // Word counter within AXI beat (0-1 for 64-bit)
    
    // AXI transaction parameters
    logic [7:0]  arlen_reg, awlen_reg;
    logic [31:0] current_dram_addr;
    
    //==========================================================================
    // FSM Sequential Logic
    //==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= DMA_IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    //==========================================================================
    // FSM Combinational Logic
    //==========================================================================
    always_comb begin
        next_state = state;
        
        case (state)
            DMA_IDLE: begin
                if (start_read) begin
                    next_state = DMA_READ_AR;
                end else if (start_write) begin
                    next_state = DMA_WRITE_AW;
                end
            end
            
            DMA_READ_AR: begin
                if (m_axi_mm2s_arvalid && m_axi_mm2s_arready) begin
                    next_state = DMA_READ_DATA;
                end
            end
            
            DMA_READ_DATA: begin
                if (m_axi_mm2s_rvalid && m_axi_mm2s_rready) begin
                    // Stay in this state until all 2 words written AND last beat
                    if (m_axi_mm2s_rlast && word_count == 1'b1) begin
                        if (bytes_remaining == 0) begin
                            next_state = DMA_DONE;
                        end else begin
                            next_state = DMA_READ_AR;  // More bursts needed
                        end
                    end
                    // else stay in DMA_READ_DATA to write remaining words
                end
            end
            
            DMA_WRITE_AW: begin
                if (m_axi_s2mm_awvalid && m_axi_s2mm_awready) begin
                    next_state = DMA_WRITE_DATA;
                end
            end
            
            DMA_WRITE_DATA: begin
                if (m_axi_s2mm_wvalid && m_axi_s2mm_wready && m_axi_s2mm_wlast) begin
                    next_state = DMA_WRITE_RESP;
                end
            end
            
            DMA_WRITE_RESP: begin
                if (m_axi_s2mm_bvalid && m_axi_s2mm_bready) begin
                    if (m_axi_s2mm_bresp != 2'b00) begin
                        next_state = DMA_ERROR;
                    end else if (bytes_remaining == 0) begin
                        next_state = DMA_DONE;
                    end else begin
                        next_state = DMA_WRITE_AW;  // More bursts needed
                    end
                end
            end
            
            DMA_DONE: begin
                next_state = DMA_IDLE;
            end
            
            DMA_ERROR: begin
                // Stay in error until reset
                next_state = DMA_ERROR;
            end
            
            default: next_state = DMA_IDLE;
        endcase
    end
    
    //==========================================================================
    // Control Registers
    //==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            transfer_addr     <= '0;
            bytes_remaining   <= '0;
            bytes_total       <= '0;
            current_target    <= BUF_ACTBUF_A;
            write_addr_offset <= '0;
            beat_count        <= '0;
            word_count        <= '0;
            arlen_reg         <= '0;
            awlen_reg         <= '0;
            current_dram_addr <= '0;
        end else begin
            case (state)
                DMA_IDLE: begin
                    if (start_read) begin
                        transfer_addr     <= src_addr;
                        bytes_total       <= length;
                        bytes_remaining   <= length;
                        current_target    <= target_buf;
                        write_addr_offset <= '0;
                        current_dram_addr <= src_addr;
                        word_count        <= '0;
                        
                        // Calculate burst length (max 256 beats for AXI3/4)
                        if (length >= (BURST_LEN * (AXI_DATA_WIDTH/8))) begin
                            arlen_reg <= BURST_LEN - 1;
                        end else begin
                            arlen_reg <= (length / (AXI_DATA_WIDTH/8)) - 1;
                        end
                    end else if (start_write) begin
                        transfer_addr     <= dst_addr;
                        bytes_total       <= length;
                        bytes_remaining   <= length;
                        write_addr_offset <= '0;
                        current_dram_addr <= dst_addr;
                        word_count        <= '0;
                        
                        if (length >= (BURST_LEN * (AXI_DATA_WIDTH/8))) begin
                            awlen_reg <= BURST_LEN - 1;
                        end else begin
                            awlen_reg <= (length / (AXI_DATA_WIDTH/8)) - 1;
                        end
                    end
                    beat_count <= '0;
                end
                
                DMA_READ_DATA: begin
                    if (m_axi_mm2s_rvalid && m_axi_mm2s_rready) begin
                        // Increment word counter (0-1 for 2 words per 64-bit beat)
                        word_count <= word_count + 1;
                        write_addr_offset <= write_addr_offset + 1;  // Increment by 1 word (32 bits)
                        
                        // After writing 2 words, move to next beat
                        if (word_count == 1'b1) begin
                            beat_count <= beat_count + 1;
                            word_count <= '0;
                        end
                        
                        if (m_axi_mm2s_rlast && word_count == 1'b1) begin
                            // Update remaining bytes
                            bytes_remaining <= bytes_remaining - ((arlen_reg + 1) * (AXI_DATA_WIDTH/8));
                            current_dram_addr <= current_dram_addr + ((arlen_reg + 1) * (AXI_DATA_WIDTH/8));
                            beat_count <= '0;
                        end
                    end
                end
                
                DMA_WRITE_DATA: begin
                    if (m_axi_s2mm_wvalid && m_axi_s2mm_wready) begin
                        write_addr_offset <= write_addr_offset + 4;
                        beat_count <= beat_count + 1;
                        
                        if (m_axi_s2mm_wlast) begin
                            bytes_remaining <= bytes_remaining - ((awlen_reg + 1) * (AXI_DATA_WIDTH/8));
                            current_dram_addr <= current_dram_addr + ((awlen_reg + 1) * (AXI_DATA_WIDTH/8));
                            beat_count <= '0;
                        end
                    end
                end
                
                default: begin
                    // No update
                end
            endcase
        end
    end
    
    //==========================================================================
    // AXI Read Channel Outputs
    //==========================================================================
    assign m_axi_mm2s_araddr  = current_dram_addr;
    assign m_axi_mm2s_arlen   = arlen_reg;
    assign m_axi_mm2s_arsize  = 3'b011;  // 64 bits = 8 bytes = 2^3
    assign m_axi_mm2s_arburst = 2'b01;   // INCR burst
    assign m_axi_mm2s_arvalid = (state == DMA_READ_AR);
    // Only assert rready after writing all 2 words from current beat
    assign m_axi_mm2s_rready  = (state == DMA_READ_DATA) && (word_count == 1'b1);
    
    //==========================================================================
    // AXI Write Channel Outputs
    //==========================================================================
    assign m_axi_s2mm_awaddr  = current_dram_addr;
    assign m_axi_s2mm_awlen   = awlen_reg;
    assign m_axi_s2mm_awsize  = 3'b011;  // 64 bits = 8 bytes = 2^3
    assign m_axi_s2mm_awburst = 2'b01;   // INCR burst
    assign m_axi_s2mm_awvalid = (state == DMA_WRITE_AW);
    assign m_axi_s2mm_wstrb   = {(AXI_DATA_WIDTH/8){1'b1}};  // All bytes valid
    assign m_axi_s2mm_wlast   = (beat_count == awlen_reg);
    assign m_axi_s2mm_wvalid  = (state == DMA_WRITE_DATA) && mem_rvalid;
    assign m_axi_s2mm_wdata   = {mem_rdata, mem_rdata};  // Replicate 32-bit to 64-bit
    assign m_axi_s2mm_bready  = (state == DMA_WRITE_RESP);
    
    //==========================================================================
    // Memory Subsystem Write Interface (for DMA reads)
    //==========================================================================
    // Convert 64-bit AXI data to 2× 32-bit writes sequentially
    always_comb begin
        mem_wen    = (state == DMA_READ_DATA) && m_axi_mm2s_rvalid;
        mem_target = current_target;
        mem_waddr  = write_addr_offset;
        
        // Select one 32-bit word from 64-bit AXI data based on word_count
        case (word_count)
            1'b0: mem_wdata = m_axi_mm2s_rdata[31:0];    // Lower 32 bits
            1'b1: mem_wdata = m_axi_mm2s_rdata[63:32];   // Upper 32 bits
            default: mem_wdata = m_axi_mm2s_rdata[31:0];
        endcase
    end
    
    //==========================================================================
    // Memory Subsystem Read Interface (for DMA writes)
    //==========================================================================
    assign mem_ren   = (state == DMA_WRITE_DATA) && m_axi_s2mm_wready;
    assign mem_raddr = write_addr_offset;
    assign mem_rsrc  = current_target;
    
    //==========================================================================
    // Status Outputs
    //==========================================================================
    assign done  = (state == DMA_DONE);
    assign error = (state == DMA_ERROR);
    assign busy  = (state != DMA_IDLE) && (state != DMA_DONE) && (state != DMA_ERROR);
    
endmodule
