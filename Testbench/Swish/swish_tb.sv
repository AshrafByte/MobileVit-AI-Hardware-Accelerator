`timescale 1ns/1ps

module swish_tb();

    // Parameters
    localparam WIDTH = 8;
    localparam USE_SHIFT_ADD = 0;   

    // DUT signals
    reg  signed [WIDTH-1:0] x;
    wire signed [WIDTH-1:0] y_hw;

    // Instantiate DUT
    swish #(
        .WIDTH(WIDTH),
        .USE_SHIFT_ADD(USE_SHIFT_ADD)
    ) dut (
        .x(x),
        .y(y_hw)
    );

    // Function: ideal swish (integer input/output)
    function signed [WIDTH-1:0] ideal_swish;
        input signed [WIDTH-1:0] xi;
        real xf, yf;
        begin
            xf = xi;                                // convert to real
            yf = xf / (1.0 + $exp(-xf));            // real swish
            ideal_swish = yf;                
        end
    endfunction

    // Test
    integer i;
    initial begin
        $display("x\tideal_swish\tH-Swish_hw");
        for (i = -8; i <= 8; i = i + 1) begin
            x = i;
            #1;  
            $display("%0d\t%0d\t\t%0d", x, ideal_swish(x), y_hw);
        end
        $finish;
    end

endmodule
