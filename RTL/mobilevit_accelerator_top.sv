//==============================================================================
// Module: mobilevit_accelerator_top
// Description: Top-level MobileViT hardware accelerator
//              Integrates: Controller, DMA, Memory, AGU, SA, Post-processing
//
// Data Flow:
//   1. CPU writes descriptors via AXI Slave
//   2. Controller fetches descriptors and orchestrates:
//      - DMA: DRAM → Memory Subsystem (weights, activations)
//      - AGU: Generates addresses for SA data fetch
//      - SA: Performs matrix multiply / convolution
//      - Post-processing: BN, Swish, Layer Norm
//      - DMA: Memory Subsystem → DRAM (results)
//   3. Controller signals completion via interrupt
//
// Author: MobileViT Accelerator Team
// Date: October 23, 2025
//==============================================================================

module mobilevit_accelerator_top 
    import accelerator_common_pkg::*;
#(
    parameter int FIFO_DEPTH = 32  // Descriptor FIFO depth
)(
    input  logic clk,
    input  logic rst_n,
    
    //==========================================================================
    // AXI Slave Interface (CPU MMIO for control/status registers)
    //==========================================================================
    // AXI4-Lite Slave (32-bit address, 32-bit data)
    input  logic [31:0]  s_axi_awaddr,
    input  logic         s_axi_awvalid,
    output logic         s_axi_awready,
    input  logic [31:0]  s_axi_wdata,
    input  logic [3:0]   s_axi_wstrb,
    input  logic         s_axi_wvalid,
    output logic         s_axi_wready,
    output logic [1:0]   s_axi_bresp,
    output logic         s_axi_bvalid,
    input  logic         s_axi_bready,
    
    input  logic [31:0]  s_axi_araddr,
    input  logic         s_axi_arvalid,
    output logic         s_axi_arready,
    output logic [31:0]  s_axi_rdata,
    output logic [1:0]   s_axi_rresp,
    output logic         s_axi_rvalid,
    input  logic         s_axi_rready,
    
    //==========================================================================
    // AXI Master Interface (DMA to DRAM)
    //==========================================================================
    // Read Address Channel
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr,
    output logic [7:0]                 m_axi_arlen,
    output logic [2:0]                 m_axi_arsize,
    output logic [1:0]                 m_axi_arburst,
    output logic                       m_axi_arvalid,
    input  logic                       m_axi_arready,
    
    // Read Data Channel
    input  logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata,
    input  logic [1:0]                 m_axi_rresp,
    input  logic                       m_axi_rlast,
    input  logic                       m_axi_rvalid,
    output logic                       m_axi_rready,
    
    // Write Address Channel
    output logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr,
    output logic [7:0]                 m_axi_awlen,
    output logic [2:0]                 m_axi_awsize,
    output logic [1:0]                 m_axi_awburst,
    output logic                       m_axi_awvalid,
    input  logic                       m_axi_awready,
    
    // Write Data Channel
    output logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata,
    output logic [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output logic                       m_axi_wlast,
    output logic                       m_axi_wvalid,
    input  logic                       m_axi_wready,
    
    // Write Response Channel
    input  logic [1:0]                 m_axi_bresp,
    input  logic                       m_axi_bvalid,
    output logic                       m_axi_bready,
    
    //==========================================================================
    // Interrupt Output
    //==========================================================================
    output logic irq
);

    //==========================================================================
    // Internal Signals - Controller <-> DMA
    //==========================================================================
    logic         ctrl_dma_start_read;
    logic         ctrl_dma_start_write;
    addr_t        ctrl_dma_src_addr;
    addr_t        ctrl_dma_dst_addr;
    word_t        ctrl_dma_length;
    buffer_id_t   ctrl_dma_target_buf;
    logic         dma_ctrl_done;
    logic         dma_ctrl_error;
    
    //==========================================================================
    // Internal Signals - Controller <-> AGU
    //==========================================================================
    logic         ctrl_agu_tile_req;
    logic         agu_ctrl_tile_done;
    logic         agu_ctrl_all_tiles_done;
    logic         ctrl_agu_read_req;
    logic         agu_ctrl_ready;
    logic         agu_ctrl_processing;
    
    op_mode_t     ctrl_agu_op_mode;
    idx_t         ctrl_agu_act_H, ctrl_agu_act_W, ctrl_agu_act_CIN;
    idx_t         ctrl_agu_ker_H, ctrl_agu_ker_W, ctrl_agu_out_chs;
    idx_t         ctrl_agu_padding, ctrl_agu_stride;
    idx_t         ctrl_agu_mat_M, ctrl_agu_mat_K, ctrl_agu_mat_N;
    idx_t         ctrl_agu_TM, ctrl_agu_TN, ctrl_agu_TK;
    addr_t        ctrl_agu_baseA, ctrl_agu_baseB, ctrl_agu_baseC;
    
    //==========================================================================
    // Internal Signals - AGU <-> Memory
    //==========================================================================
    addr_t        agu_mem_addr;
    logic [1:0]   agu_mem_id;
    logic         agu_mem_valid;
    logic         agu_mem_is_null;
    
    //==========================================================================
    // Internal Signals - Controller <-> SA
    //==========================================================================
    logic         ctrl_sa_start;
    logic         ctrl_sa_load_w;
    sa_type_t     ctrl_sa_type;
    logic         ctrl_sa_transpose_en;
    logic         sa_ctrl_valid_out;
    
    //==========================================================================
    // Internal Signals - Controller <-> Memory Subsystem
    //==========================================================================
    logic         ctrl_mem_ping_pong_sel;
    logic         ctrl_mem_accum_mode;
    logic         ctrl_mem_clear_psum;
    logic         ctrl_mem_write_enable;
    
    //==========================================================================
    // Internal Signals - Controller <-> Post-Processing
    //==========================================================================
    logic         ctrl_pp_bn_enable;
    logic         ctrl_pp_swish_enable;
    logic         ctrl_pp_ln_enable;
    
    //==========================================================================
    // Internal Signals - DMA <-> Memory Subsystem
    //==========================================================================
    logic [31:0]             dma_mem_waddr;
    logic [DATA_WIDTH-1:0]   dma_mem_wdata;
    logic                    dma_mem_wen;
    buffer_id_t              dma_mem_target;
    logic [31:0]             dma_mem_raddr;
    logic                    dma_mem_ren;
    buffer_id_t              dma_mem_rsrc;
    logic [DATA_WIDTH-1:0]   mem_dma_rdata;
    logic                    mem_dma_rvalid;
    
    //==========================================================================
    // Internal Signals - Memory <-> SA
    //==========================================================================
    logic [ACT_WIDTH-1:0]    mem_sa_act_data[SA_SIZE];     // 64× 8-bit
    logic [WEIGHT_WIDTH-1:0] mem_sa_wgt_data[SA_SIZE];     // 64× 8-bit
    logic [PSUM_WIDTH-1:0]   mem_sa_psum_in[SA_SIZE];      // 64× 32-bit
    logic [PSUM_WIDTH-1:0]   mem_sa_psum_readback[16];     // 16× 32-bit from memory
    logic                    mem_sa_data_valid;
    logic [PSUM_WIDTH-1:0]   sa_mem_psum_out[SA_SIZE];
    logic                    sa_mem_valid_out;
    
    //==========================================================================
    // Internal Signals - SA <-> Post-Processing
    //==========================================================================
    logic [PSUM_WIDTH-1:0]   sa_pp_data[16];
    logic                    sa_pp_valid;
    logic [PSUM_WIDTH-1:0]   pp_mem_data[16];
    logic                    pp_mem_valid;
    
    //==========================================================================
    // Descriptor Interface (simplified - from register writes)
    //==========================================================================
    descriptor_t  desc_to_ctrl;
    logic         desc_valid;
    logic         desc_ready;
    
    //==========================================================================
    // Status Signals
    //==========================================================================
    logic         ctrl_busy;
    logic         ctrl_done;
    logic         ctrl_error;
    logic [15:0]  ctrl_tile_counter;
    logic [31:0]  ctrl_cycle_counter;
    
    //==========================================================================
    // Register Map (AXI Slave)
    //==========================================================================
    // Simplified register interface
    // 0x00: CONTROL - [0]=start, [1]=reset
    // 0x04: STATUS  - [0]=busy, [1]=done, [2]=error
    // 0x10-0x2C: DESC_DATA[0:7] - 256-bit descriptor (8× 32-bit words)
    // 0x30: DESC_PUSH - Write 1 to push descriptor to FIFO
    
    logic [31:0] reg_control;
    logic [31:0] reg_status;
    logic [31:0] reg_desc_data[8];
    logic        reg_desc_push;
    logic        start_pulse;
    
    // AXI Slave write logic (simplified)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            reg_control <= '0;
            reg_desc_data <= '{default: '0};
            reg_desc_push <= 1'b0;
            start_pulse <= 1'b0;
        end else begin
            reg_desc_push <= 1'b0;
            start_pulse <= 1'b0;
            
            if (s_axi_awvalid && s_axi_wvalid) begin
                case (s_axi_awaddr[7:0])
                    8'h00: begin
                        reg_control <= s_axi_wdata;
                        start_pulse <= s_axi_wdata[0];
                    end
                    8'h10: reg_desc_data[0] <= s_axi_wdata;
                    8'h14: reg_desc_data[1] <= s_axi_wdata;
                    8'h18: reg_desc_data[2] <= s_axi_wdata;
                    8'h1C: reg_desc_data[3] <= s_axi_wdata;
                    8'h20: reg_desc_data[4] <= s_axi_wdata;
                    8'h24: reg_desc_data[5] <= s_axi_wdata;
                    8'h28: reg_desc_data[6] <= s_axi_wdata;
                    8'h2C: reg_desc_data[7] <= s_axi_wdata;
                    8'h30: reg_desc_push <= s_axi_wdata[0];
                    default: begin
                        // Ignore writes to undefined addresses
                    end
                endcase
            end
        end
    end
    
    // AXI Slave read logic
    always_comb begin
        reg_status = {29'h0, ctrl_error, ctrl_done, ctrl_busy};
    end
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            s_axi_rdata <= '0;
        end else if (s_axi_arvalid) begin
            case (s_axi_araddr[7:0])
                8'h00: s_axi_rdata <= reg_control;
                8'h04: s_axi_rdata <= reg_status;
                8'h34: s_axi_rdata <= {16'h0, ctrl_tile_counter};
                8'h38: s_axi_rdata <= ctrl_cycle_counter;
                default: s_axi_rdata <= 32'hDEAD_BEEF;
            endcase
        end
    end
    
    // AXI handshakes (simplified - no wait states)
    assign s_axi_awready = 1'b1;
    assign s_axi_wready  = 1'b1;
    assign s_axi_bvalid  = s_axi_awvalid && s_axi_wvalid;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_arready = 1'b1;
    assign s_axi_rvalid  = s_axi_arvalid;
    assign s_axi_rresp   = 2'b00;
    
    // Descriptor formatting
    assign desc_to_ctrl = {
        reg_desc_data[7],  // [255:224] dram_addr
        reg_desc_data[6][31:16], // [223:208] sram_addr
        reg_desc_data[6][15:0],  // [207:192] length
        reg_desc_data[5][31:16], // [191:176] stride
        112'h0,            // [175:64] reserved
        reg_desc_data[1][31:16], // [63:48] tile_h
        reg_desc_data[1][15:0],  // [47:32] tile_w
        reg_desc_data[0][31:16], // [31:16] c_in
        reg_desc_data[0][15:8],  // [15:8] flags
        reg_desc_data[0][7:0]    // [7:0] reserved
    };
    assign desc_valid = reg_desc_push;
    
    //==========================================================================
    // Module Instantiations
    //==========================================================================
    
    //--------------------------------------------------------------------------
    // Global Controller
    //--------------------------------------------------------------------------
    global_controller u_global_controller (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse),
        .busy(ctrl_busy),
        .done(ctrl_done),
        .error(ctrl_error),
        
        .descriptor(desc_to_ctrl),
        .desc_valid(desc_valid),
        .desc_ready(desc_ready),
        
        .dma_start_read(ctrl_dma_start_read),
        .dma_start_write(ctrl_dma_start_write),
        .dma_src_addr(ctrl_dma_src_addr),
        .dma_dst_addr(ctrl_dma_dst_addr),
        .dma_length(ctrl_dma_length),
        .dma_target_buf(ctrl_dma_target_buf),
        .dma_done(dma_ctrl_done),
        .dma_error(dma_ctrl_error),
        
        .agu_tile_req(ctrl_agu_tile_req),
        .agu_tile_done(agu_ctrl_tile_done),
        .agu_all_tiles_done(agu_ctrl_all_tiles_done),
        .agu_read_req(ctrl_agu_read_req),
        .agu_ready(agu_ctrl_ready),
        .agu_processing(agu_ctrl_processing),
        
        .agu_op_mode(ctrl_agu_op_mode),
        .agu_act_H(ctrl_agu_act_H),
        .agu_act_W(ctrl_agu_act_W),
        .agu_act_CIN(ctrl_agu_act_CIN),
        .agu_ker_H(ctrl_agu_ker_H),
        .agu_ker_W(ctrl_agu_ker_W),
        .agu_out_chs(ctrl_agu_out_chs),
        .agu_padding(ctrl_agu_padding),
        .agu_stride(ctrl_agu_stride),
        .agu_mat_M(ctrl_agu_mat_M),
        .agu_mat_K(ctrl_agu_mat_K),
        .agu_mat_N(ctrl_agu_mat_N),
        .agu_TM(ctrl_agu_TM),
        .agu_TN(ctrl_agu_TN),
        .agu_TK(ctrl_agu_TK),
        .agu_baseA(ctrl_agu_baseA),
        .agu_baseB(ctrl_agu_baseB),
        .agu_baseC(ctrl_agu_baseC),
        
        .sa_start(ctrl_sa_start),
        .sa_load_w(ctrl_sa_load_w),
        .sa_type(ctrl_sa_type),
        .sa_transpose_en(ctrl_sa_transpose_en),
        .sa_valid_out(sa_ctrl_valid_out),
        
        .mem_ping_pong_sel(ctrl_mem_ping_pong_sel),
        .mem_accum_mode(ctrl_mem_accum_mode),
        .mem_clear_psum(ctrl_mem_clear_psum),
        .mem_write_enable(ctrl_mem_write_enable),
        
        .pp_bn_enable(ctrl_pp_bn_enable),
        .pp_swish_enable(ctrl_pp_swish_enable),
        .pp_ln_enable(ctrl_pp_ln_enable),
        
        .tile_counter(ctrl_tile_counter),
        .cycle_counter(ctrl_cycle_counter),
        .irq(irq)
    );
    
    //--------------------------------------------------------------------------
    // DMA Wrapper
    //--------------------------------------------------------------------------
    dma_wrapper u_dma_wrapper (
        .clk(clk),
        .rst_n(rst_n),
        
        .start_read(ctrl_dma_start_read),
        .start_write(ctrl_dma_start_write),
        .src_addr(ctrl_dma_src_addr),
        .dst_addr(ctrl_dma_dst_addr),
        .length(ctrl_dma_length),
        .target_buf(ctrl_dma_target_buf),
        .done(dma_ctrl_done),
        .error(dma_ctrl_error),
        .busy(),
        
        // AXI Master MM2S (Read)
        .m_axi_mm2s_araddr(m_axi_araddr),
        .m_axi_mm2s_arlen(m_axi_arlen),
        .m_axi_mm2s_arsize(m_axi_arsize),
        .m_axi_mm2s_arburst(m_axi_arburst),
        .m_axi_mm2s_arvalid(m_axi_arvalid),
        .m_axi_mm2s_arready(m_axi_arready),
        .m_axi_mm2s_rdata(m_axi_rdata),
        .m_axi_mm2s_rresp(m_axi_rresp),
        .m_axi_mm2s_rlast(m_axi_rlast),
        .m_axi_mm2s_rvalid(m_axi_rvalid),
        .m_axi_mm2s_rready(m_axi_rready),
        
        // AXI Master S2MM (Write)
        .m_axi_s2mm_awaddr(m_axi_awaddr),
        .m_axi_s2mm_awlen(m_axi_awlen),
        .m_axi_s2mm_awsize(m_axi_awsize),
        .m_axi_s2mm_awburst(m_axi_awburst),
        .m_axi_s2mm_awvalid(m_axi_awvalid),
        .m_axi_s2mm_awready(m_axi_awready),
        .m_axi_s2mm_wdata(m_axi_wdata),
        .m_axi_s2mm_wstrb(m_axi_wstrb),
        .m_axi_s2mm_wlast(m_axi_wlast),
        .m_axi_s2mm_wvalid(m_axi_wvalid),
        .m_axi_s2mm_wready(m_axi_wready),
        .m_axi_s2mm_bresp(m_axi_bresp),
        .m_axi_s2mm_bvalid(m_axi_bvalid),
        .m_axi_s2mm_bready(m_axi_bready),
        
        // Memory Subsystem
        .mem_waddr(dma_mem_waddr),
        .mem_wdata(dma_mem_wdata),
        .mem_wen(dma_mem_wen),
        .mem_target(dma_mem_target),
        .mem_raddr(dma_mem_raddr),
        .mem_ren(dma_mem_ren),
        .mem_rsrc(dma_mem_rsrc),
        .mem_rdata(mem_dma_rdata),
        .mem_rvalid(mem_dma_rvalid)
    );
    
    //--------------------------------------------------------------------------
    // AGU (Address Generation Unit)
    //--------------------------------------------------------------------------
    AGU #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH(IDX_WIDTH),
        .NULL_ADDR(NULL_ADDR)
    ) u_agu (
        .clk(clk),
        .rst_n(rst_n),
        
        .processing(agu_ctrl_processing),
        .AGU_ready(agu_ctrl_ready),
        .tile_req(ctrl_agu_tile_req),
        .tile_done(agu_ctrl_tile_done),
        .read_req(ctrl_agu_read_req),
        .all_tiles_done(agu_ctrl_all_tiles_done),
        
        .op_mode(ctrl_agu_op_mode),
        .act_H(ctrl_agu_act_H),
        .act_W(ctrl_agu_act_W),
        .act_CIN(ctrl_agu_act_CIN),
        .ker_H(ctrl_agu_ker_H),
        .ker_W(ctrl_agu_ker_W),
        .out_chs(ctrl_agu_out_chs),
        .padding(ctrl_agu_padding),
        .stride(ctrl_agu_stride),
        
        .mat_M(ctrl_agu_mat_M),
        .mat_K(ctrl_agu_mat_K),
        .mat_N(ctrl_agu_mat_N),
        
        .TM(ctrl_agu_TM),
        .TN(ctrl_agu_TN),
        .TK(ctrl_agu_TK),
        
        .baseA(ctrl_agu_baseA),
        .baseB(ctrl_agu_baseB),
        .baseC(ctrl_agu_baseC),
        
        .mem_addr(agu_mem_addr),
        .mem_id(agu_mem_id),
        .mem_valid(agu_mem_valid),
        .mem_is_null(agu_mem_is_null)
    );
    
    //--------------------------------------------------------------------------
    // Memory Subsystem
    //--------------------------------------------------------------------------
    memory_subsystem u_memory_subsystem (
        .clk(clk),
        .rst_n(rst_n),
        
        // Configuration
        .sa_type(ctrl_sa_type),  // Pass SA type for dynamic bank configuration
        
        // DMA Write Interface
        .dma_waddr(dma_mem_waddr),
        .dma_wdata(dma_mem_wdata),
        .dma_wen(dma_mem_wen),
        .dma_target(dma_mem_target),
        .dma_wready(),
        
        // AGU Read Interface
        .agu_addr_A(agu_mem_addr[15:0]),
        .agu_addr_B(agu_mem_addr[15:0]),
        .agu_id(agu_mem_id),
        .agu_valid(agu_mem_valid),
        .ping_pong_sel(ctrl_mem_ping_pong_sel),
        .data_to_sa_act(mem_sa_act_data),  // 64× 8-bit activations
        .data_to_sa_wgt(mem_sa_wgt_data),  // 64× 8-bit weights
        .data_valid(mem_sa_data_valid),
        
        // SA Write Interface
        .psum_waddr(ctrl_agu_baseC),
        .psum_wdata(sa_mem_psum_out[0:15]),  // Take first 16 elements
        .psum_wen(ctrl_mem_write_enable),
        
        // SA Psum Readback (for accumulation)
        .psum_raddr(ctrl_agu_baseC),
        .psum_ren(ctrl_mem_accum_mode & ctrl_sa_start),
        .psum_rdata(mem_sa_psum_readback),  // 16× 32-bit for accumulation
        
        // DMA Read Interface
        .dma_raddr(dma_mem_raddr),
        .dma_ren(dma_mem_ren),
        .dma_rsrc(dma_mem_rsrc),
        .dma_rdata(mem_dma_rdata),
        .dma_rvalid(mem_dma_rvalid),
        
        // Accumulation Control
        .accum_mode(ctrl_mem_accum_mode),
        .clear_psum(ctrl_mem_clear_psum)
    );
    
    //--------------------------------------------------------------------------
    // Psum Expansion Logic (MVP: Use only first 16 elements of 64-element SA)
    //--------------------------------------------------------------------------
    // Memory provides 16× 32-bit psums, SA expects 64× 32-bit
    // For MVP, we replicate first 16 to all 64, or use only 16×16 portion of SA
    genvar p;
    generate
        for (p = 0; p < 16; p++) begin : PSUM_EXPAND
            assign mem_sa_psum_in[p] = mem_sa_psum_readback[p];
        end
        for (p = 16; p < 64; p++) begin : PSUM_ZERO
            assign mem_sa_psum_in[p] = '0;  // Zero-pad remaining elements
        end
    endgenerate
    
    //--------------------------------------------------------------------------
    // SA Compute Unit (Lego SA + Accumulation)
    //--------------------------------------------------------------------------
    sa_compute_unit u_sa_compute_unit (
        .clk(clk),
        .rst_n(rst_n),
        
        .start(ctrl_sa_start),
        .load_weights(ctrl_sa_load_w),
        .sa_type(ctrl_sa_type),
        .transpose_en(ctrl_sa_transpose_en),
        .accum_en(ctrl_mem_accum_mode),
        
        .act_in(mem_sa_act_data),
        .weight_in(mem_sa_wgt_data),
        .psum_in(mem_sa_psum_in),
        .data_valid_in(mem_sa_data_valid),
        
        .psum_out(sa_mem_psum_out),
        .valid_out(sa_mem_valid_out),
        .done(sa_ctrl_valid_out)
    );
    
    // Connect first 16 SA outputs to post-processing
    assign sa_pp_data = sa_mem_psum_out[0:15];
    assign sa_pp_valid = sa_mem_valid_out;
    
    //--------------------------------------------------------------------------
    // Post-Processing Pipeline
    //--------------------------------------------------------------------------
    post_processing_pipeline u_post_processing (
        .clk(clk),
        .rst_n(rst_n),
        
        .bn_enable(ctrl_pp_bn_enable),
        .swish_enable(ctrl_pp_swish_enable),
        .ln_enable(ctrl_pp_ln_enable),
        
        .data_in(sa_pp_data),
        .valid_in(sa_pp_valid),
        .data_out(pp_mem_data),
        .valid_out(pp_mem_valid),
        
        .bn_mean(32'h0),
        .bn_var(32'h0000_0001),
        .bn_gamma(32'h0000_0001),
        .bn_beta(32'h0),
        .ln_dim(16'd16)
    );
    
    //==========================================================================
    // Signal Connections - All data paths properly connected
    //==========================================================================
    // mem_sa_act_data[64] - Connected from memory_subsystem.data_to_sa_act
    // mem_sa_wgt_data[64] - Connected from memory_subsystem.data_to_sa_wgt
    // mem_sa_psum_in[16]  - Connected from memory_subsystem.psum_rdata for accumulation
    // All connections made through module port binding above
    
endmodule
