//==============================================================================
// Testbench: mobilevit_accel_tb
// Description: Basic testbench for MobileViT accelerator top-level
//              Tests simple descriptor push and control flow
//
// Author: MobileViT Accelerator Team
// Date: October 23, 2025
//==============================================================================

`timescale 1ns / 1ps

module mobilevit_accel_tb;
    import accelerator_common_pkg::*;
    
    //==========================================================================
    // Parameters
    //==========================================================================
    localparam CLK_PERIOD = 2.5;  // 400 MHz = 2.5ns period
    
    //==========================================================================
    // DUT Signals
    //==========================================================================
    logic clk;
    logic rst_n;
    
    // AXI Slave (simplified - just addresses and data)
    logic [31:0]  s_axi_awaddr;
    logic         s_axi_awvalid;
    logic         s_axi_awready;
    logic [31:0]  s_axi_wdata;
    logic [3:0]   s_axi_wstrb;
    logic         s_axi_wvalid;
    logic         s_axi_wready;
    logic [1:0]   s_axi_bresp;
    logic         s_axi_bvalid;
    logic         s_axi_bready;
    logic [31:0]  s_axi_araddr;
    logic         s_axi_arvalid;
    logic         s_axi_arready;
    logic [31:0]  s_axi_rdata;
    logic [1:0]   s_axi_rresp;
    logic         s_axi_rvalid;
    logic         s_axi_rready;
    
    // AXI Master (DMA)
    logic [AXI_ADDR_WIDTH-1:0]  m_axi_araddr;
    logic [7:0]                 m_axi_arlen;
    logic [2:0]                 m_axi_arsize;
    logic [1:0]                 m_axi_arburst;
    logic                       m_axi_arvalid;
    logic                       m_axi_arready;
    logic [AXI_DATA_WIDTH-1:0]  m_axi_rdata;
    logic [1:0]                 m_axi_rresp;
    logic                       m_axi_rlast;
    logic                       m_axi_rvalid;
    logic                       m_axi_rready;
    logic [AXI_ADDR_WIDTH-1:0]  m_axi_awaddr;
    logic [7:0]                 m_axi_awlen;
    logic [2:0]                 m_axi_awsize;
    logic [1:0]                 m_axi_awburst;
    logic                       m_axi_awvalid;
    logic                       m_axi_awready;
    logic [AXI_DATA_WIDTH-1:0]  m_axi_wdata;
    logic [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb;
    logic                       m_axi_wlast;
    logic                       m_axi_wvalid;
    logic                       m_axi_wready;
    logic [1:0]                 m_axi_bresp;
    logic                       m_axi_bvalid;
    logic                       m_axi_bready;
    
    logic irq;
    
    //==========================================================================
    // DUT Instantiation
    //==========================================================================
    mobilevit_accelerator_top u_dut (
        .clk(clk),
        .rst_n(rst_n),
        
        // AXI Slave
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        
        // AXI Master
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        
        .irq(irq)
    );
    
    //==========================================================================
    // Clock Generation
    //==========================================================================
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end
    
    //==========================================================================
    // Simple DRAM Model (responds to AXI reads with dummy data)
    //==========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arready <= 1'b0;
            m_axi_rvalid  <= 1'b0;
            m_axi_rdata   <= '0;
            m_axi_rlast   <= 1'b0;
            m_axi_awready <= 1'b0;
            m_axi_wready  <= 1'b0;
            m_axi_bvalid  <= 1'b0;
        end else begin
            // Read Address Channel
            m_axi_arready <= m_axi_arvalid;
            
            // Read Data Channel (return dummy data after 2 cycles)
            if (m_axi_arready && m_axi_arvalid) begin
                m_axi_rvalid <= 1'b1;
                m_axi_rdata  <= {4{32'hA5A5_A5A5}};  // Dummy pattern
                m_axi_rlast  <= 1'b1;  // Single beat for simplicity
            end else if (m_axi_rready) begin
                m_axi_rvalid <= 1'b0;
            end
            
            // Write channels (just acknowledge)
            m_axi_awready <= m_axi_awvalid;
            m_axi_wready  <= m_axi_wvalid;
            m_axi_bvalid  <= m_axi_awready && m_axi_wready;
        end
    end
    
    assign m_axi_rresp = 2'b00;  // OKAY
    assign m_axi_bresp = 2'b00;  // OKAY
    
    //==========================================================================
    // AXI Slave Tasks (for register writes/reads)
    //==========================================================================
    task axi_write(input [31:0] addr, input [31:0] data);
        begin
            @(posedge clk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1'b1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1'b1;
            s_axi_bready  <= 1'b1;
            
            @(posedge clk);
            while (!s_axi_awready || !s_axi_wready) @(posedge clk);
            
            s_axi_awvalid <= 1'b0;
            s_axi_wvalid  <= 1'b0;
            
            @(posedge clk);
            s_axi_bready <= 1'b0;
        end
    endtask
    
    task axi_read(input [31:0] addr, output [31:0] data);
        begin
            @(posedge clk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1'b1;
            s_axi_rready  <= 1'b1;
            
            @(posedge clk);
            while (!s_axi_arready) @(posedge clk);
            
            s_axi_arvalid <= 1'b0;
            
            @(posedge clk);
            while (!s_axi_rvalid) @(posedge clk);
            data = s_axi_rdata;
            
            @(posedge clk);
            s_axi_rready <= 1'b0;
        end
    endtask
    
    //==========================================================================
    // Test Sequence
    //==========================================================================
    initial begin
        // Initialize signals
        rst_n = 0;
        s_axi_awvalid = 0;
        s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_arvalid = 0;
        s_axi_rready = 0;
        
        // Reset
        repeat(10) @(posedge clk);
        rst_n = 1;
        repeat(5) @(posedge clk);
        
        $display("========================================");
        $display("MobileViT Accelerator Testbench");
        $display("========================================");
        $display("Time: %0t ns", $time);
        
        // Test 1: Write and read control register
        $display("\n[TEST 1] Writing to CONTROL register");
        axi_write(32'h00, 32'h0000_0002);  // Soft reset
        #10;
        axi_write(32'h00, 32'h0000_0000);  // Clear reset
        
        // Test 2: Read status register
        logic [31:0] status;
        $display("\n[TEST 2] Reading STATUS register");
        axi_read(32'h04, status);
        $display("  STATUS = 0x%08X (busy=%b, done=%b, error=%b)", 
                 status, status[0], status[1], status[2]);
        
        // Test 3: Write descriptor
        $display("\n[TEST 3] Writing descriptor");
        axi_write(32'h10, 32'h0010_0010);  // DESC_DATA[0]: c_in=16, flags=0x10
        axi_write(32'h14, 32'h0010_0010);  // DESC_DATA[1]: tile_h=16, tile_w=16
        axi_write(32'h18, 32'h0000_0000);  // DESC_DATA[2]: reserved
        axi_write(32'h1C, 32'h0000_0000);  // DESC_DATA[3]: reserved
        axi_write(32'h20, 32'h0000_0000);  // DESC_DATA[4]: reserved
        axi_write(32'h24, 32'h0001_0100);  // DESC_DATA[5]: stride=1, ...
        axi_write(32'h28, 32'h0400_0000);  // DESC_DATA[6]: length=1024, sram_addr=0
        axi_write(32'h2C, 32'h8000_0000);  // DESC_DATA[7]: dram_addr=0x8000_0000
        axi_write(32'h30, 32'h0000_0001);  // DESC_PUSH: push descriptor
        
        $display("  Descriptor pushed");
        
        // Test 4: Start processing
        $display("\n[TEST 4] Starting accelerator");
        axi_write(32'h00, 32'h0000_0001);  // Set START bit
        
        // Wait for busy signal
        repeat(10) @(posedge clk);
        axi_read(32'h04, status);
        $display("  STATUS = 0x%08X (should be busy)", status);
        
        // Wait for completion (or timeout)
        $display("\n[TEST 5] Waiting for completion...");
        repeat(1000) begin
            @(posedge clk);
            if (irq) begin
                $display("  IRQ asserted at time %0t ns", $time);
                break;
            end
        end
        
        // Read final status
        axi_read(32'h04, status);
        $display("  Final STATUS = 0x%08X", status);
        
        // Read performance counters
        logic [31:0] tile_count, cycles;
        axi_read(32'h34, tile_count);
        axi_read(32'h38, cycles);
        $display("  Tiles processed: %0d", tile_count);
        $display("  Cycle count: %0d", cycles);
        
        // Test complete
        $display("\n========================================");
        $display("Test Complete!");
        $display("========================================");
        
        repeat(10) @(posedge clk);
        $finish;
    end
    
    //==========================================================================
    // Waveform Dump
    //==========================================================================
    initial begin
        $dumpfile("mobilevit_accel_sim.vcd");
        $dumpvars(0, mobilevit_accel_tb);
    end
    
    //==========================================================================
    // Timeout Watchdog
    //==========================================================================
    initial begin
        #100000;  // 100 us timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end
    
endmodule
