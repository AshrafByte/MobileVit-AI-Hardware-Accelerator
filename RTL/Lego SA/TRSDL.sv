
// (Triangular Register Shift Down Logic)
module TRSDL #(parameter DATAWIDTH = 32, N_SIZE = 16)(
input  logic                       clk                      ,
input  logic                       rst_n                    ,
input  logic [DATAWIDTH-1:0]       psum_in   [N_SIZE]        ,   
output logic [DATAWIDTH-1:0]       psum_out  [N_SIZE]        
);
localparam NUM_OF_REGS = ((N_SIZE-1)*N_SIZE)/2    ;     // Total number of intermediate registers required for data shifting in triangular reg logic

logic   [DATAWIDTH-1:0]        reg_shifted         [1:NUM_OF_REGS]     ;
logic   [DATAWIDTH-1:0]        psum                 [N_SIZE]            ;

assign  psum[0] = psum_in[0]          ;
assign  psum[1] = reg_shifted[1]     ;

genvar k,i_deptha ;
genvar l ;

generate;
    for (k = 1 ; k<16 ; k++)
    begin
        localparam int base = (k*(k-1))/2;
        always_ff @(posedge clk or negedge rst_n)
        begin : First_col_Resgs
            if(!rst_n)
                reg_shifted[(base)+1] <= 0   ;
            else
                reg_shifted[(base)+1] <= psum_in[k]   ;
        end
        if(k>1)
        begin : DEPTH_LEVEL
            for(i_deptha = (base)+2 ; i_deptha < ((base)+1)+k ; i_deptha++)
            begin
                always_ff @(posedge clk or negedge rst_n)
                begin
                    if (~rst_n)
                    begin
                        reg_shifted[i_deptha] <= 0  ;
                    end
                    else
                    begin
                        reg_shifted[i_deptha] <= reg_shifted[i_deptha-1]   ;
                    end
                end 
            end
        end
        if(k>1)
        begin
            assign psum[k] = reg_shifted[(base)+1+k-1]  ;
        end
    end

endgenerate

generate;
    for (l = 0 ; l < N_SIZE ; l++)
        assign psum_out[l] = psum[l] ;
endgenerate

endmodule