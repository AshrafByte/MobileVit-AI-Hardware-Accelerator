
module PE_tb #(parameter DATA_W = 8,parameter DATA_W_OUT = 31)();
logic                    clk         ;
logic                    rst_n       ;
logic                    valid_in    ;
logic [DATA_W-1:0]       in_act      ;       // input activation from left
logic [DATA_W_OUT-1:0]   in_psum     ;       // partial sum from top
logic [DATA_W-1:0]       weight_load ;       // new weight (when load_w = 1)
logic                    load_w      ;       // load weight enable

logic [DATA_W-1:0]       out_act     ;       // propagate activation right
logic [DATA_W_OUT-1:0]   out_psum    ;       // propagate partial sum down
logic                    valid_out   ;

PE #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) PE_mac (
.clk(clk),
.rst_n(rst_n),
.valid_in(valid_in),
.in_act(in_act),
.in_psum(in_psum),
.weight_load(weight_load),
.load_w(load_w),
.out_act(out_act),
.out_psum(out_psum),
.valid_out(valid_out)
);

initial clk = 0 ;
always #5 clk = ~clk;

initial
begin
    rst_n = 0 ;
    #5;
    rst_n = 1;
    valid_in = 1 ;
    load_w = 1 ;
    in_act = 0 ;
    in_psum = 0 ;
    weight_load = 5 ;
    $display("[%0t]Loading Wights",$time);
    #10;
    valid_in = 1 ;
    load_w = 0 ;
    in_act = 9 ;
    in_psum = 55 ;
    weight_load = 5 ;
    $display("[%0t]Computing",$time);
    $display("valid_in = %0d ,W = %0d ,act_reg = %0d,psum_reg = %0d",valid_in,PE_mac.W_reg,PE_mac.act_reg,PE_mac.psum_reg);
    $display("out_act = %0d, out_psum = %0d, valid_out = %0d",out_act,out_psum,valid_out);
    #1;
    $display("[%0t]Computing",$time);
    $display("valid_in = %0d ,W = %0d ,act_reg = %0d,psum_reg = %0d",valid_in,PE_mac.W_reg,PE_mac.act_reg,PE_mac.psum_reg);
    $display("out_act = %0d, out_psum = %0d, valid_out = %0d",out_act,out_psum,valid_out);
    $stop;
end
endmodule