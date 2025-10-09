module swish #(
    parameter WIDTH = 8, // bit width, default 8-bit signed int
    parameter USE_SHIFT_ADD = 0 // 0 = exact division, 1 = shift+add approx
)(
    input  signed [WIDTH-1:0] x,
    output reg signed [WIDTH-1:0] y
);

    // Constants in integer domain
    localparam signed [WIDTH-1:0] THREE = 3;
    localparam signed [WIDTH-1:0] SIX   = 6;

    // Step 1: x + 3
    wire signed [WIDTH-1:0] x_plus3;;
    // Step 2: clamp into [0,6] → ReLU6(x+3)
    wire signed [WIDTH-1:0] relu6_val;
    assign x_plus3 = x + THREE;
    assign relu6_val =(x_plus3 < 0)   ? 0   :(x_plus3 > SIX) ? SIX :x_plus3;

    generate
        if (USE_SHIFT_ADD == 0) begin : exact_div
            // ----- Exact division by 6 -----
            always_comb begin
                y = (x * relu6_val) / 6;
            end
        end else begin : shift_add
            // ----- Approximate division using shifts -----
            always_comb begin
                y = x * ((relu6_val >>> 3) + (relu6_val >>> 5));
            end
        end
    endgenerate

endmodule
