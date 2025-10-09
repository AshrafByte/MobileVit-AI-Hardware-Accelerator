
module Lego_Systolic_Array #(
    parameter DATA_W = 8, parameter DATA_W_OUT = 32, SA_indiv = 1 
)(
input  logic                    clk                 ,
input  logic                    rst_n               ,
input  logic                    load_w              ,   // load weight phase
input  logic [DATA_W-1:0]       act_in  [64]        ,   // left edge activations
input  logic [DATA_W-1:0]       w_load  [32][32]    ,   // weights for each PE
input  logic                    valid_in            ,      
input  logic [1:0]              TYPE_Lego           ,     

output logic [DATA_W_OUT-1:0]   psum_out[32]        ,   // bottom edge partial sums
output logic                    valid_out
);


SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT),.SA_indiv(SA_indiv)) SA_1 (
.clk(clk),
.rst_n(rst_n),
.load_w(load_w),
.act_in(act_in),
.psum_in(psum_in),
.w_load(w_load),
.valid_in(valid_in),
.act_out(act_out),
.psum_out(psum_out),
.valid_out(valid_out)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT),.SA_indiv(SA_indiv)) SA_2 (
.clk(clk),
.rst_n(rst_n),
.load_w(load_w),
.act_in(act_in),
.psum_in(psum_in),
.w_load(w_load),
.valid_in(valid_in),
.act_out(act_out),
.psum_out(psum_out),
.valid_out(valid_out)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT),.SA_indiv(SA_indiv)) SA_3 (
.clk(clk),
.rst_n(rst_n),
.load_w(load_w),
.act_in(act_in),
.psum_in(psum_in),
.w_load(w_load),
.valid_in(valid_in),
.act_out(act_out),
.psum_out(psum_out),
.valid_out(valid_out)
);

SA_16x16_top #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT),.SA_indiv(SA_indiv)) SA_4 (
.clk(clk),
.rst_n(rst_n),
.load_w(load_w),
.act_in(act_in),
.psum_in(psum_in),
.w_load(w_load),
.valid_in(valid_in),
.act_out(act_out),
.psum_out(psum_out),
.valid_out(valid_out)
);

endmodule