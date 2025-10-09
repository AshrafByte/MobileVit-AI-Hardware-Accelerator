`timescale 1ns/1ps

module layer_norm2 #(
    parameter N = 8,                 // vector length
    parameter DATA_WIDTH = 8,        // input/output bit width
    parameter ACC_WIDTH  = 32        // accumulator width
)(
    input  logic signed [DATA_WIDTH-1:0] in_vec [N],  // input vector
    output logic signed [DATA_WIDTH-1:0] out_vec [N]  // normalized outputs
);

    // ======================================================
    // Internal signals
    // ======================================================
    integer sum, mean;
    integer variance, stddev;
    integer diff, norm;

    // ======================================================
    // Mean computation
    // ======================================================
    always_comb begin
        sum = 0;
        for (int i = 0; i < N; i++) begin
            sum += in_vec[i];
        end
        mean = sum / N;
    end

    // ======================================================
    // Variance computation (sum of squared diffs / N)
    // ======================================================
    always_comb begin
        variance = 0;
        for (int i = 0; i < N; i++) begin
            diff = in_vec[i] - mean;
            variance += diff * diff;
        end
        variance = variance / N;
    end

    // ======================================================
    // Newton–Raphson sqrt (integer, 4 iterations)
    // ======================================================
    function integer int_sqrt_hwstyle(input integer variance);
        integer std, k;
        begin
            if (variance == 0) begin
                return 0;
            end

            std = variance >>> 1;  // initial guess
            if (std == 0) std = 1;

            for (k = 0; k < 4; k++) begin
                std = (std + (variance / std)) >>> 1;
            end

            return std;
        end
    endfunction

    always_comb begin
        stddev = int_sqrt_hwstyle(variance);
    end

    // ======================================================
    // Normalization
    // out = (x - mean) * SCALE / stddev
    // ======================================================
    localparam SCALE = 128; // scaling factor to keep values in 8-bit

    always_comb begin
        for (int i = 0; i < N; i++) begin
            if (stddev == 0) begin
                out_vec[i] = '0;
            end
            else begin
                norm = ((in_vec[i] - mean) * SCALE) / stddev;
                // Saturate into 8-bit signed
                if (norm > 127) norm = 127;
                else if (norm < -128) norm = -128;
                out_vec[i] = norm[DATA_WIDTH-1:0];
            end
        end
    end

endmodule
