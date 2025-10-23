module memory_subsystem #(
    parameter NUM_BANKS = 16,        // 16 banks for optimal FPGA mapping
    parameter BANK_WIDTH = 32,       // 32-bit per bank (4× 8-bit elements)
    parameter ACTBUF_BANK_DEPTH = 2048,  // 2048 words × 32-bit = 8KB per bank × 16 = 128KB total (2 buffers)
    parameter WGTBUF_BANK_DEPTH = 2048,  // 32KB for weights
    parameter PSUMBUF_BANK_DEPTH = 4096, // 16KB per bank × 16 = 256KB total for partial sums
    parameter ELEMENTS_PER_BANK = 4  // 4× 8-bit elements per 32-bit word
)(
    input  logic clk, rst_n,
    
    //=================================================================
    // Configuration
    //=================================================================
    input  logic [1:0]              sa_type,         // SA configuration (00=16x64, 01=32x32, 10=64x16)
    
    //=================================================================
    // DMA Write Interface (from AXI DMA to buffers)
    //=================================================================
    input  logic [31:0]             dma_waddr,
    input  logic [31:0]             dma_wdata,       // 32-bit word from DMA
    input  logic                    dma_wen,
    input  logic [1:0]              dma_target,      // 0=ActBufA, 1=ActBufB, 2=WgtBuf, 3=PSumBuf
    output logic                    dma_wready,
    
    //=================================================================
    // AGU Read Interface (for feeding Systolic Array)
    //=================================================================
    input  logic [15:0]             agu_addr_A,      // activation address
    input  logic [15:0]             agu_addr_B,      // weight address
    input  logic [1:0]              agu_id,          // which matrix (A, B, C)
    input  logic                    agu_valid,
    input  logic                    ping_pong_sel,   // 0=ActBufA, 1=ActBufB
    output logic [7:0]              data_to_sa_act[64],  // 64× 8-bit activations
    output logic [7:0]              data_to_sa_wgt[64],  // 64× 8-bit weights
    output logic                    data_valid,
    
    //=================================================================
    // Systolic Array Write Interface (partial sums back to PSumBuf)
    //=================================================================
    input  logic [31:0]             psum_waddr,
    input  logic [31:0]             psum_wdata[16],  // 16× 32-bit psums from SA
    input  logic                    psum_wen,
    
    //=================================================================
    // Systolic Array Psum Readback (for accumulation mode)
    //=================================================================
    input  logic [31:0]             psum_raddr,      // Read address for accumulation
    input  logic                    psum_ren,        // Read enable
    output logic [31:0]             psum_rdata[16],  // 16× 32-bit psums for accumulation
    
    //=================================================================
    // DMA Read Interface (for writeback to DRAM)
    //=================================================================
    input  logic [31:0]             dma_raddr,
    input  logic                    dma_ren,
    input  logic [1:0]              dma_rsrc,        // which buffer to read from
    output logic [31:0]             dma_rdata,       // 32-bit word to DMA
    output logic                    dma_rvalid,
    
    //=================================================================
    // Accumulation Control (for multi-tile C_in)
    //=================================================================
    input  logic                    accum_mode,      // 1 = add to existing psum
    input  logic                    clear_psum       // 1 = zero out PSumBuf
);

//=========================================================================
// Internal Buffers (Banked Architecture for Parallel Access)
//=========================================================================
// 16 banks × 32-bit width = 64× 8-bit elements per cycle
// Each bank: 2048 words × 32-bit = 8KB → 16 banks = 128KB total (ActBufA + ActBufB)

logic [31:0] ActBufA_banks[NUM_BANKS][ACTBUF_BANK_DEPTH];  // 16 banks × 2048 words
logic [31:0] ActBufB_banks[NUM_BANKS][ACTBUF_BANK_DEPTH];  // 16 banks × 2048 words
logic [31:0] WgtBuf_banks[NUM_BANKS][WGTBUF_BANK_DEPTH];   // 16 banks × 2048 words
logic [31:0] PSumBuf_banks[NUM_BANKS][PSUMBUF_BANK_DEPTH]; // 16 banks × 4096 words

// Bank read data
logic [31:0] act_bank_data[NUM_BANKS];  // 16× 32-bit words from activation banks
logic [31:0] wgt_bank_data[NUM_BANKS];  // 16× 32-bit words from weight banks

//=========================================================================
// DMA Write Logic (distribute to banks)
//=========================================================================
// Bank selection: Use lower bits of address for bank interleaving
// Address format: [bank_id : word_addr_within_bank]

logic [3:0]  dma_bank_sel;   // Which bank (0-15)
logic [15:0] dma_word_addr;  // Word address within bank

assign dma_bank_sel  = dma_waddr[3:0];  // Lower 4 bits select bank (0-15)
assign dma_word_addr = dma_waddr[19:4]; // Upper bits are word address

always_ff @(posedge clk) begin
    if (dma_wen) begin
        case (dma_target)
            2'b00: ActBufA_banks[dma_bank_sel][dma_word_addr] <= dma_wdata;
            2'b01: ActBufB_banks[dma_bank_sel][dma_word_addr] <= dma_wdata;
            2'b10: WgtBuf_banks[dma_bank_sel][dma_word_addr]  <= dma_wdata;
            2'b11: PSumBuf_banks[dma_bank_sel][dma_word_addr] <= dma_wdata;
            default: begin end  // Do nothing
        endcase
    end
end

assign dma_wready = 1'b1;  // Always ready

//=========================================================================
// AGU Read Logic (Parallel Bank Access based on SA Type)
//=========================================================================
// Number of active banks depends on SA type:
// sa_type = 00 (16×64): 16 banks active → 64 elements
// sa_type = 01 (32×32): 8 banks active  → 32 elements
// sa_type = 10 (64×16): 4 banks active  → 16 elements

logic [3:0] num_banks_active;
logic [31:0] selected_act_banks[NUM_BANKS][ACTBUF_BANK_DEPTH];

// Determine number of active banks
always_comb begin
    case (sa_type)
        2'b00:   num_banks_active = 4'd16;  // Type 0: 16×64
        2'b01:   num_banks_active = 4'd8;   // Type 1: 32×32
        2'b10:   num_banks_active = 4'd4;   // Type 2: 64×16
        default: num_banks_active = 4'd16;
    endcase
end

// Select ping-pong buffer
assign selected_act_banks = ping_pong_sel ? ActBufB_banks : ActBufA_banks;

// Parallel read from all banks
genvar i;
generate
    for (i = 0; i < NUM_BANKS; i++) begin : BANK_READ
        always_ff @(posedge clk) begin
            if (agu_valid && (i < num_banks_active)) begin
                // Read from selected activation buffer
                act_bank_data[i] <= selected_act_banks[i][agu_addr_A];
                // Read from weight buffer
                wgt_bank_data[i] <= WgtBuf_banks[i][agu_addr_B];
            end else begin
                // Inactive banks output zero
                act_bank_data[i] <= '0;
                wgt_bank_data[i] <= '0;
            end
        end
    end
endgenerate

//=========================================================================
// Data Width Conversion: 16× 32-bit → 64× 8-bit
//=========================================================================
// Unpack 32-bit words into 4× 8-bit bytes
// Each 32-bit word contains [byte3, byte2, byte1, byte0] = [31:24, 23:16, 15:8, 7:0]
logic [7:0] act_unpacked[64];
logic [7:0] wgt_unpacked[64];

genvar j, k;
generate
    for (j = 0; j < NUM_BANKS; j++) begin : BANK_UNPACK
        for (k = 0; k < ELEMENTS_PER_BANK; k++) begin : ELEMENT_UNPACK
            assign act_unpacked[j*4 + k] = act_bank_data[j][k*8 +: 8];
            assign wgt_unpacked[j*4 + k] = wgt_bank_data[j][k*8 +: 8];
        end
    end
endgenerate

// Output assignment
assign data_to_sa_act = act_unpacked;
assign data_to_sa_wgt = wgt_unpacked;

//=========================================================================
// Data Valid Pipeline (match 1-cycle read latency)
//=========================================================================
logic data_valid_q;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_valid_q <= 1'b0;
    end else begin
        data_valid_q <= agu_valid;
    end
end

assign data_valid = data_valid_q;

//=========================================================================
// Accumulation Logic (for PSumBuf)
//=========================================================================
// PSumBuf readback for accumulation
// Read 16 consecutive 32-bit words across banks
always_ff @(posedge clk) begin
    if (psum_ren) begin
        for (int m = 0; m < 16; m++) begin
            logic [3:0] psum_bank;
            logic [15:0] psum_addr;
            psum_bank = (psum_raddr + m) % NUM_BANKS;
            psum_addr = (psum_raddr + m) / NUM_BANKS;
            psum_rdata[m] <= PSumBuf_banks[psum_bank][psum_addr];
        end
    end
end

// PSumBuf write (with optional accumulation)
always_ff @(posedge clk) begin
    if (clear_psum) begin
        // Zero out entire PSumBuf
        for (int j = 0; j < NUM_BANKS; j++) begin
            for (int n = 0; n < PSUMBUF_BANK_DEPTH; n++) begin
                PSumBuf_banks[j][n] <= '0;
            end
        end
    end else if (psum_wen) begin
        for (int k = 0; k < 16; k++) begin
            logic [3:0] psum_bank;
            logic [15:0] psum_addr;
            psum_bank = (psum_waddr + k) % NUM_BANKS;
            psum_addr = (psum_waddr + k) / NUM_BANKS;
            
            if (accum_mode) begin
                // Read-modify-write: add new psum to existing
                PSumBuf_banks[psum_bank][psum_addr] <= PSumBuf_banks[psum_bank][psum_addr] + psum_wdata[k];
            end else begin
                // Fresh write
                PSumBuf_banks[psum_bank][psum_addr] <= psum_wdata[k];
            end
        end
    end
end

//=========================================================================
// DMA Read Logic (for writeback)
//=========================================================================
always_ff @(posedge clk) begin
    if (dma_ren) begin
        logic [3:0] rd_bank;
        logic [15:0] rd_addr;
        rd_bank = dma_raddr[3:0];
        rd_addr = dma_raddr[19:4];
        
        case (dma_rsrc)
            2'b00: dma_rdata <= ActBufA_banks[rd_bank][rd_addr];
            2'b01: dma_rdata <= ActBufB_banks[rd_bank][rd_addr];
            2'b11: dma_rdata <= PSumBuf_banks[rd_bank][rd_addr];
            default: dma_rdata <= '0;
        endcase
        dma_rvalid <= 1'b1;
    end else begin
        dma_rvalid <= 1'b0;
    end
end

endmodule