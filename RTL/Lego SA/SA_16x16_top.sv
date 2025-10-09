
module SA_16x16_top #(
    parameter DATA_W = 8, parameter DATA_W_OUT = 32, SA_indiv = 1 
)(
input  logic                    clk                 ,
input  logic                    rst_n               ,
input  logic                    load_w              ,   // load weight phase
input  logic [DATA_W-1:0]       act_in  [16]        ,   // left edge activations
input  logic [DATA_W_OUT-1:0]   psum_in [16]        ,   // top input psums
input  logic [DATA_W-1:0]       w_load  [16][16]    ,   // weights for each PE
input  logic                    valid_in            ,           

output logic [DATA_W-1:0]       act_out [16]        ,   // right edge activations
output logic [DATA_W_OUT-1:0]   psum_out[16]        ,   // bottom edge partial sums
output logic                    valid_out
);

logic [DATA_W-1:0]       act_TRSLL_SA  [16]      ;    

TRSLL #(.DATAWIDTH(DATA_W),.N_SIZE(16)) reg_shifted_right(
.clk(clk),
.rst_n(rst_n),
.act_in(act_in),
.act_out(act_TRSLL_SA)
);

SA_16x16 #(.DATA_W(DATA_W), .DATA_W_OUT(DATA_W_OUT), .SA_indiv(SA_indiv)) SA (
.clk(clk),
.rst_n(rst_n),
.load_w(load_w),
.act_in(act_TRSLL_SA),
.psum_in(psum_in),
.w_load(w_load),
.valid_in(valid_in),
.act_out(act_out),
.psum_out(psum_out),
.valid_out(valid_out)
);

endmodule