
module PE #(parameter DATA_W = 8, parameter DATA_W_OUT = 32)(
input  logic                    clk         ,
input  logic                    rst_n       ,
input  logic                    valid_in    ,       // input valid signal
input  logic [DATA_W-1:0]       in_act      ,       // input activation from left
input  logic [DATA_W_OUT-1:0]   in_psum     ,       // partial sum from top
input  logic [DATA_W-1:0]       weight_load ,       // new weight (when load_w = 1)
input  logic                    load_w      ,       // load weight enable

output logic [DATA_W-1:0]       out_act     ,       // propagate activation right
output logic [DATA_W_OUT-1:0]   out_psum    ,       // propagate partial sum down
output logic                    valid_out
);

logic [DATA_W-1:0] W_reg;
logic [DATA_W-1:0] act_reg;
logic [DATA_W_OUT-1:0] psum_reg;
logic [2*DATA_W-1:0] mac_mul;
logic [DATA_W_OUT-1:0] mac_res;

assign mac_mul = in_act * W_reg ;
assign mac_res = mac_mul + in_psum ;

// Hold stationary weight
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        W_reg <= '0;
    else if (load_w && valid_in)
        W_reg <= weight_load;
end

// Multiply-Accumulate
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)
    begin
        act_reg <= '0;
        psum_reg <= '0;
    end
    else if (!load_w && valid_in)
    begin
        act_reg <= in_act ;
        psum_reg <= mac_res ;
    end
end

// Output propagation
assign out_act   = act_reg;
assign out_psum  = psum_reg;    // truncate to 16-bit (example)
assign valid_out = valid_in;    // Not Now (with controllel) 

endmodule