
module SA_16x16_top #(
    parameter DATA_W = 8, parameter DATA_W_OUT = 32
)(
input  logic                    clk                 ,
input  logic                    rst_n               ,
input  logic [DATA_W-1:0]       act_in  [16]        ,     
input  logic [DATA_W-1:0]       weight_in [16]      ,
input  logic                    load_w              ,    
input  logic                    transpose_en        ,    
  
output logic [DATA_W_OUT-1:0]   psum_out[16]        
);

logic [DATA_W-1:0]       act_TRSRL_SA  [16]      ;  
logic [DATA_W_OUT-1:0]   psum [16]               ;
logic [DATA_W_OUT-1:0]   psum_TRSDL_SA [16]      ;
logic [DATA_W_OUT-1:0]   psum_TRSDL_out[16]      ;

TRSRL #(.DATAWIDTH(DATA_W),.N_SIZE(16)) reg_shifted_right(
.clk(clk),
.rst_n(rst_n),
.act_in(act_in),
.act_out(act_TRSRL_SA)
);

SA_16x16 #(.DATA_W(DATA_W), .DATA_W_OUT(DATA_W_OUT)) SA (
.clk(clk),
.rst_n(rst_n),
.act_in(act_TRSRL_SA),
.weight_in(weight_in),
.load_w(load_w),
.transpose_en(transpose_en),
.psum_out(psum)
);

TRSDL #(.DATAWIDTH(DATA_W_OUT),.N_SIZE(16)) reg_shifted_down(
.clk(clk),
.rst_n(rst_n),
.psum_in(psum_TRSDL_SA),
.psum_out(psum_TRSDL_out)
);

genvar i ;
generate;
for (i=0 ; i<16; i++)
begin
    assign psum_TRSDL_SA[i] = psum [15-i] ; 
    assign psum_out[i] = psum_TRSDL_out [15-i] ; 
end
endgenerate




endmodule