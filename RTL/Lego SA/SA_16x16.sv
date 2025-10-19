

module SA_16x16 #(
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

// Internal interconnect signals
logic   [DATA_W-1:0]        act_sig             [16][17]          ;       // extra column for right output
logic   [DATA_W-1:0]        weight_D_sig        [17][16]          ;
logic   [DATA_W-1:0]        weight_L_sig        [16][17]          ;
logic   [DATA_W_OUT-1:0]    psum_sig            [17][16]          ;       // extra row for bottom output




genvar k ;
genvar i,j ;

generate;
    for (k=0 ; k<16; k++)
    begin
        assign act_sig[k][0]    = act_in[k] ;
        assign psum_sig[0][k]   = '0    ;
        assign weight_D_sig[16][k] = weight_in[k] ;
        assign weight_L_sig[k][16] = weight_in[k]; 
    end
endgenerate

generate;
    for (i = 0 ;i < 16;i++) begin :Row 
        for (j = 0 ;j < 16; j++) begin :COl
            PE #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                .in_act(act_sig[i][j]),
                .in_psum(psum_sig[i][j]),
                .w_in_down(weight_D_sig[i+1][j]),
                .w_in_left(weight_L_sig[i][j+1]),
                .load_w(load_w),
                .transpose_en(transpose_en),
                .out_act(act_sig[i][j+1]),
                .out_psum(psum_sig[i+1][j]),
                .w_out_up(weight_D_sig[i][j]),
                .w_out_right(weight_L_sig[i][j])
            );
        end
    end

    for (j=0; j<16; j++) assign psum_out[j] = psum_sig[16][j];
endgenerate
endmodule