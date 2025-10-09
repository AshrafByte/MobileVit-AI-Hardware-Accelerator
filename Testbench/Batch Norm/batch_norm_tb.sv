`timescale 1ns/1ps

module batch_norm_tb;

  // Parameters
  localparam Data_Width = 32;
  localparam N          = 4;   // demo (can set to 32)

  // DUT signals
  logic                    CLK;
  logic                    RST;
  logic [N*Data_Width-1:0] in_row;
  logic                    INBatch_Valid;
  logic [Data_Width-1:0]   A [0:N-1];
  logic [Data_Width-1:0]   B [0:N-1];
  logic [N*Data_Width-1:0] out_row;
  logic                    OutBatch_Valid;

  // Expected output array
  logic [Data_Width-1:0] expected [0:N-1];

  // Variables for checking
  bit success;    // declared once here
  int actual;     // declared once here

  // Instantiate DUT
  batch_norm #(
    .Data_Width(Data_Width),
    .N(N)
  ) dut (
    .CLK(CLK),
    .RST(RST),
    .in_row(in_row),
    .INBatch_Valid(INBatch_Valid),
    .A(A),
    .B(B),
    .out_row(out_row),
    .OutBatch_Valid(OutBatch_Valid)
  );

  // Clock generation
  initial CLK = 0;
  always #5 CLK = ~CLK; // 100MHz clock

  // Stimulus
  initial begin
    // Init
    RST          = 0;
    INBatch_Valid= 0;
    in_row       = 0;

    // Reset pulse
    #12;
    RST = 1;
    $display("[%0t] Reset released", $time);

    // Set coefficients A and B
    for (int i = 0; i < N; i++) begin
      A[i] = 32'd1;        // scale = 1
      B[i] = 32'(i*2);     // bias = 0,2,4,6...
    end

    // Apply input row x = [1,2,3,4]
    for (int i = 0; i < N; i++) begin
      in_row[i*Data_Width +: Data_Width] = i+1;
      expected[i] = (A[i] * (i+1)) + B[i];
    end

    @(posedge CLK);
    INBatch_Valid = 1;
    @(posedge CLK);
    INBatch_Valid = 0;

    // Wait for DUT output
    @(posedge CLK);
    if (OutBatch_Valid) begin
      success = 1;  // reset flag
      $display("[%0t] === BatchNorm Output ===", $time);

      for (int i = 0; i < N; i++) begin
        actual = out_row[i*Data_Width +: Data_Width];
        $display("y[%0d] = %0d (expected %0d)", i, actual, expected[i]);
        if (actual !== expected[i]) success = 0;
      end

      if (success)
        $display(" TEST CASE PASSED at time %0t", $time);
      else
        $display(" TEST CASE FAILED at time %0t", $time);
    end

    #20;
    $display("[%0t] Simulation finished", $time);
    $finish;
  end

endmodule

