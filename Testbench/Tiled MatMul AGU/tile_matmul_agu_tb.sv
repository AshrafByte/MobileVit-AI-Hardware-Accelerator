// `timescale 1ns/1ps

// module tb_Tield_MatMul_AGU;

    // localparam int ADDR_WIDTH = 32;
    // localparam int IDX_WIDTH  = 8;

    // logic clk, rst;
    // logic start, start_next, read_req;
    // logic done_all;
    // logic [ADDR_WIDTH-1:0] o_addr;
    // logic [1:0] addr_id;
    // logic valid;
	
	// // --------------------------------------------------------
    // // Tile configuration (example: 5x5 matrices, TM=2, TN=2, TK=5)
    // // --------------------------------------------------------
    // localparam logic [IDX_WIDTH-1:0] M = 5;
    // localparam logic [IDX_WIDTH-1:0] N = 5;
    // localparam logic [IDX_WIDTH-1:0] K = 5;
    // localparam logic [IDX_WIDTH-1:0] TM_cfg = 2;
    // localparam logic [IDX_WIDTH-1:0] TN_cfg = 2;
    // localparam logic [IDX_WIDTH-1:0] TK_cfg = 5;

    // // Base addresses
    // localparam logic [ADDR_WIDTH-1:0] baseA = 32'd0000_0000;
    // localparam logic [ADDR_WIDTH-1:0] baseB = 32'd0000_1000;
    // localparam logic [ADDR_WIDTH-1:0] baseC = 32'd0000_2000;

    // // DUT
    // Tiled_MatMul_AGU #(
        // .ADDR_WIDTH(ADDR_WIDTH),
        // .IDX_WIDTH(IDX_WIDTH)
    // ) DUT (
        // .clk(clk),
        // .rst(rst),
        // .start(start),
        // .start_next(start_next),
        // .read_req(read_req),
        // .done_all(done_all),
        // .o_addr(o_addr),
        // .addr_id(addr_id),
        // .valid(valid)
    // );

    // // Clock
    // always #5 clk = ~clk;

    // // Simulation behavior
    // initial begin
        // $display("---- Simulation Started ----");
        // clk = 0;
        // rst = 1;
        // start = 0;
        // start_next = 0;
        // read_req = 0;

        // #20 rst = 0;

        // // Start first tile
        // #10 start = 1; read_req = 1;
        // #10 start = 0;

        // // Generate "read_req" signal for AGU
        // repeat (300) begin
            // #10;
            // // every few cycles, issue next tile
            // if (DUT.u_ctrl.tile_done && !done_all)
                // start_next = 1;
            // else
                // start_next = 0;

            // // Log activity
            // if (valid)
                // // $display("Time %0t | ID=%0d | Addr=0x%08h", $time, addr_id, o_addr);
				// $display("Time %0t | ID=%0d | Addr=%0d", $time, addr_id, o_addr);

        // end

        // $display("---- Simulation Finished ----");
        // $finish;
    // end

// endmodule




`timescale 1ns/1ps

module tile_matmul_agu_tb;

    // Parameters
    localparam int ADDR_WIDTH = 32;
    localparam int IDX_WIDTH  = 8;

    // Signals driven by TB
    logic clk;
    logic rst;
    logic tile_req;
    logic read_req;

    // Outputs from DUT
    logic done_all;
    logic [ADDR_WIDTH-1:0] o_addr;
    logic [1:0] addr_id;
    logic valid;

    // --------------------------------------------------------
    // Tile configuration (example: 5x5 matrices, TM=2, TN=2, TK=5)
    // --------------------------------------------------------
    localparam logic [IDX_WIDTH-1:0] M_cfg    = 5;
    localparam logic [IDX_WIDTH-1:0] N_cfg    = 5;
    localparam logic [IDX_WIDTH-1:0] K_cfg    = 5;
    localparam logic [IDX_WIDTH-1:0] TM_cfg   = 2;
    localparam logic [IDX_WIDTH-1:0] TN_cfg   = 2;
    localparam logic [IDX_WIDTH-1:0] TK_cfg   = 5;

    // Base addresses (use hex for clarity)
    localparam logic [ADDR_WIDTH-1:0] baseA = 32'd0000_0000;
    localparam logic [ADDR_WIDTH-1:0] baseB = 32'd0000_0100;
    localparam logic [ADDR_WIDTH-1:0] baseC = 32'd0000_0200;

    // DUT instance
    tile_matmul_agu #(
        .ADDR_WIDTH(ADDR_WIDTH),
        .IDX_WIDTH (IDX_WIDTH)
    ) DUT (
        .clk     (clk),
        .rst     (rst),
        .tile_req(tile_req),
        .done_all(done_all),
        .read_req    (read_req),

        // global parameters
        .M       (M_cfg),
        .N       (N_cfg),
        .K       (K_cfg),
        .TM_cfg  (TM_cfg),
        .TN_cfg  (TN_cfg),
        .TK_cfg  (TK_cfg),

        .baseA   (baseA),
        .baseB   (baseB),
        .baseC   (baseC),

        // AGU outputs
        .o_addr  (o_addr),
        .addr_id (addr_id),
        .valid   (valid)
    );
	
	string id_name[3] = '{"A","B","C"};
	
    // Clock: 10 ns period
    initial clk = 0;
    always #5 clk = ~clk;

    // // Drive 'read_req' from 'valid' (one-cycle delayed) so AGU sees a read_req each valid.
    // // Using non-blocking update to align with synchronous DUT behavior.
    // always_ff @(posedge clk or posedge rst) begin
        // if (rst) begin
            // read_req <= 1'b0;
        // end else begin
            // // consume every valid address
            // read_req <= valid;
        // end
    // end
	

    // Stimulus
    initial begin
        $display("=== TB: Tiled MatMul AGU Simulation starting ===");
        // initialize
        rst = 1;
		read_req = 1;
        tile_req = 0;
        #20;

        // release reset, assert tile_req permanently (controller uses tile_read_reqy+tile_req)
        rst = 0;
        tile_req = 1'b1;

        $display("Reset released, tile_req asserted -> controller will generate tiles when read_reqy");

        // run until controller asserts done_all
        wait (done_all == 1'b1);
        #20;
        $display("=== All tiles processed (done_all) at time %0t ===", $time);

        $display("=== Simulation finished ===");
        #10;
        $finish;
    end
	
	
	// // Monitor tile_read_reqy and tile_done dynamically
	// always_ff @(posedge clk) begin
	
		// if (DUT.tile_done)
			// $display("Time %0t | ---- Tile Done ----", $time);
		// if (DUT.tile_read_reqy)
			// $display("Time %0t | --- Tile read_reqy ---", $time);

	// end

    // Monitoring: print addresses when valid
    always_ff @(posedge clk) begin
        if (valid) begin
            // print time (ns), addr id and address in decimal as requested
            $display("ID=%0d Matrix [%s]| Addr=%0d",addr_id,id_name[addr_id], o_addr);
        end
    end

endmodule



