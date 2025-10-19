
module Lego_control_unit #()(
input   logic           clk                 ,
input   logic           rst_n               ,

output  logic [5:0]     count_out
);

localparam N_CYCLES_LOAD_W = 16 ;

logic [5:0]     count ;

always_ff @(posedge clk or negedge rst_n)
begin
    if (~rst_n)
    begin
        count <= 0 ;
    end
    else
    begin
        count <= count + 1 ;
    end
end

assign count_out = count ;


endmodule