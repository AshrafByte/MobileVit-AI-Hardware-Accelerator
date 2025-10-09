
module SA_16x16 #(
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

// Internal interconnect signals
logic   [DATA_W-1:0]        act_sig             [16][17]          ;       // extra column for right output
logic   [DATA_W_OUT-1:0]    psum_sig            [17][16]          ;       // extra row for bottom output
logic                       valid_sig           [16][17]            ;

assign valid_sig[0][0] = valid_in;              // Not Now (with controllel)

genvar i,j ;

generate;
    for (i = 0 ;i < 16;i++) begin :Row 
        for (j = 0 ;j < 16; j++) begin :COl
            PE #(.DATA_W(DATA_W),.DATA_W_OUT(DATA_W_OUT)) u_pe (
                .clk(clk),
                .rst_n(rst_n),
                .valid_in(valid_sig[i][j]),     // Not Now (with controllel)
                .in_act((j==0) ? act_in[i] : act_sig[i][j]),
                .in_psum((i==0) ? ((SA_indiv) ? '0 :psum_in[j]) : psum_sig[i][j]),
                .weight_load(w_load[i][j]),
                .load_w(load_w),
                .out_act(act_sig[i][j+1]),
                .out_psum(psum_sig[i+1][j]),
                .valid_out(valid_sig[i][j+1])   // Not Now (with controllel)
            );
        end
    end

    // Right & bottom boundary connections
    for (i=0; i<16; i++) assign act_out[i]  = act_sig[i][16];
    for (j=0; j<16; j++) assign psum_out[j] = psum_sig[16][j];
    assign valid_out = valid_sig[15][16];      // Not Now (with controllel)
endgenerate
endmodule