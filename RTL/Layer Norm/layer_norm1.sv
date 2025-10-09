module layer_norm1 #(
    parameter DATA_WIDTH = 16,
    parameter EMBED_DIM  = 8
)(
    input  wire                           clk,
    input  wire                           rst_n,
    input  wire                           layernorm_start,
    input  wire signed [DATA_WIDTH-1:0]   activation_in [0:EMBED_DIM-1],
    output reg                            layernorm_done,
    output reg signed [DATA_WIDTH-1:0]    normalized_out [0:EMBED_DIM-1]
);

    // Internal registers
    reg  signed [DATA_WIDTH-1:0] buffer [0:EMBED_DIM-1];
    reg  signed [2*DATA_WIDTH:0] sum, mean, sum_sq, vari;
    reg  signed [2*DATA_WIDTH-1:0] std_dev;
    reg         [2:0] counter;
    wire        counter_max;
    reg         std_en;
    integer i;

    // FSM states
    typedef enum logic [2:0] {
        IDLE = 3'd0,
        LOAD = 3'd1,
        SUM1 = 3'd3,
        MEAN = 3'd2,
        SUM2 = 3'd6,
        VARI = 3'd7,
        NORM = 3'd5,
        DONE = 3'd4
    } state_t;

    state_t current_state, next_state;

    assign counter_max = (counter == 3'd4); // Run 4 Newton iterations

    // Sequential: FSM + accumulators
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            current_state  <= IDLE;
            layernorm_done <= 1'b0;
            sum            <= 'd0;
            mean           <= 'd0;
            sum_sq         <= 'd0;
            vari           <= 'd0;
            std_dev        <= 'd0;
            counter        <= 'd0;
            for (i = 0; i < EMBED_DIM; i = i+1)
                buffer[i] <= 'd0;
        end else begin
            current_state <= next_state;

            if (std_en && !counter_max)
                counter <= counter + 1'd1;
            else if (!std_en)
                counter <= 0;
        end
    end

    // Newton-Raphson sqrt
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            std_dev <= 'd0;
        end else if (std_en && counter == 0) begin
            std_dev <= (vari >>> 1); // initial guess
        end else if (std_en && !counter_max) begin
            if (std_dev != 0)
                std_dev <= (std_dev + vari/std_dev) >>> 1;
        end
    end

    // FSM next-state logic
    always_comb begin
        case (current_state)
            IDLE: next_state = layernorm_start ? LOAD : IDLE;
            LOAD: next_state = SUM1;
            SUM1: next_state = MEAN;
            MEAN: next_state = SUM2;
            SUM2: next_state = VARI;
            VARI: next_state = counter_max ? NORM : VARI;
            NORM: next_state = DONE;
            DONE: next_state = IDLE;
            default: next_state = IDLE;
        endcase
    end

    // FSM datapath
    always_comb begin
        layernorm_done = 1'b0;
        std_en         = 1'b0;

        case (current_state)
            LOAD: begin
                for (i = 0; i < EMBED_DIM; i = i+1)
                    buffer[i] = activation_in[i];
            end

            SUM1: begin
                sum = 0;
                for (i = 0; i < EMBED_DIM; i = i+1)
                    sum = sum + buffer[i];
            end

            MEAN: begin
                // mean = sum / EMBED_DIM  ->  shift right
                mean = sum >>> $clog2(EMBED_DIM);
            end

            SUM2: begin
                sum_sq = 0;
                for (i = 0; i < EMBED_DIM; i = i+1)
                    sum_sq = sum_sq + (buffer[i] - mean) * (buffer[i] - mean);
            end

            VARI: begin
                // vari = sum_sq / EMBED_DIM  ->  shift right
                vari  = sum_sq >>> $clog2(EMBED_DIM);
                std_en = 1'b1; // enable Newton iteration
            end

            NORM: begin
                for (i = 0; i < EMBED_DIM; i = i+1) begin
                    if (std_dev != 0)
                        normalized_out[i] = (buffer[i] - mean) / std_dev;
                    else
                        normalized_out[i] = 0;
                end
            end

            DONE: begin
                layernorm_done = 1'b1;
            end
        endcase
    end
endmodule
