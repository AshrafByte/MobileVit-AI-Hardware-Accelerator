module batch_norm #(
    parameter Data_Width = 32,
    parameter N = 32
) (
    input  wire                    CLK,
    input  wire                    RST,
    input  wire [N*Data_Width-1:0] in_row,
    input  wire                    INBatch_Valid,
    input  wire [Data_Width-1:0]   A [0:N-1], 
    input  wire [Data_Width-1:0]   B [0:N-1], 
    output reg  [N*Data_Width-1:0] out_row,
    output reg                     OutBatch_Valid
);

    // Internal wires for x and y
    reg [Data_Width-1:0] x [0:N-1];
    reg [Data_Width-1:0] y [0:N-1];

    integer i;

    // Combinational unpack input row & compute BatchNorm
    always_comb begin
         for (i = 0; i < N; i = i + 1) begin
            x[i] = in_row[i*Data_Width +: Data_Width];
            y[i] = A[i] * x[i] + B[i];
        end
    end

    // Sequential output register
    always_ff @(posedge CLK or negedge RST) begin
        if (!RST) begin
            out_row        <= 0;
            OutBatch_Valid <= 0;
        end 
        else begin
            if (INBatch_Valid) begin
                for (i = 0; i < N; i = i + 1) begin
                    out_row[i*Data_Width +: Data_Width] <= y[i];
                end
                OutBatch_Valid <= 1'b1;   // pulse when valid input
            end 
            else begin
                OutBatch_Valid <= 1'b0;   // drop valid flag after 1 cycle
            end
        end
    end

endmodule
