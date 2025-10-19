
module PE_tb #(parameter DATA_W = 8,parameter DATA_W_OUT = 31)();
logic                    clk         ;
logic                    rst_n       ;
logic                    valid_in    ;
logic [DATA_W-1:0]       in_act      ;       // input activation from left
logic [DATA_W_OUT-1:0]   in_psum     ;       // partial sum from top
logic [DATA_W-1:0]       w_in_down   ;  
logic [DATA_W-1:0]       w_in_left   ; 
logic                    load_w      ;       // load weight enable
logic                    transpose_en;

logic [DATA_W-1:0]       out_act     ;       // propagate activation right
logic [DATA_W_OUT-1:0]   out_psum    ;       // propagate partial sum down
logic [DATA_W-1:0]       w_out_up       ;
logic [DATA_W-1:0]       w_out_right     ;
logic                    valid_out   ;

PE #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) PE_mac (
.clk(clk),
.rst_n(rst_n),
.valid_in(valid_in),
.in_act(in_act),
.in_psum(in_psum),
.w_in_down(w_in_down),
.w_in_left(w_in_left),
.load_w(load_w),
.transpose_en(transpose_en),
.out_act(out_act),
.out_psum(out_psum),
.w_out_up(w_out_up),
.w_out_right(w_out_right),
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
    transpose_en = 0 ;
    w_in_down = 5 ;
    w_in_left = 10 ;
    $display("[%0t]Loading Wights",$time);
    #10;
    load_w = 0;
    in_act = 10 ;
    in_psum = 40 ;
    #5;
    $display("[%0t] Computing ",$time);
    $display("out_act = %0d , out_psum = %0d , w_out_up = %0d , w_out_right = %0d , valid_out = %0d ",
    out_act,out_psum,w_out_up,w_out_right,valid_out);
    $display("W_reg = %0d",PE_mac.W_reg);
    #5;

    $stop;
end
endmodule