# MobileVit-AI-Hardware-Accelerator

# RTL Modules Documentation

This document provides comprehensive documentation for all RTL modules in the MobileVit AI Hardware Accelerator project.

---

## Table of Contents
1. [Tiled Matrix Multiplication AGU](#1-tiled-matrix-multiplication-agu)
   - [tile_matmul_agu](#tile_matmul_agu)
   - [tile_base_controller](#tile_base_controller)
   - [tile_offsets_agu](#tile_offsets_agu)
2. [Lego Systolic Array](#2-lego-systolic-array)
   - [PE (Processing Element)](#pe-processing-element)
   - [SA_16x16](#sa_16x16)
   - [SA_16x16_top](#sa_16x16_top)
   - [TRSLL (Triangular Register Shift Logic)](#trsll-triangular-register-shift-logic)
   - [Lego_Systolic_Array](#lego_systolic_array)
3. [Normalization Modules](#3-normalization-modules)
   - [layer_norm1](#layer_norm1)
   - [layer_norm2](#layer_norm2)
   - [batch_norm](#batch_norm)
4. [Activation Functions](#4-activation-functions)
   - [swish](#swish)

---

# 1. Tiled Matrix Multiplication AGU

## tile_matmul_agu

> **Purpose:** Top-level wrapper for tiled matrix multiplication address generation. Orchestrates tile iteration and element-level address generation.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ADDR_WIDTH` | int | 32 | Width of address bus in bits |
| `IDX_WIDTH` | int | 8 | Width of matrix dimension indices in bits |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst` | Input | 1 | logic | Asynchronous reset (active high) |
| **Control Signals** |||||
| `tile_req` | Input | 1 | logic | Request to start/continue tile processing |
| `done_all` | Output | 1 | logic | Asserted when all tiles completed |
| `read_req` | Input | 1 | logic | Request to output next address |
| **Matrix Dimensions** |||||
| `M` | Input | `IDX_WIDTH` | logic | Number of rows in matrix A (and C) |
| `N` | Input | `IDX_WIDTH` | logic | Number of columns in matrix B (and C) |
| `K` | Input | `IDX_WIDTH` | logic | Common dimension (cols of A, rows of B) |
| **Tile Configuration** |||||
| `TM_cfg` | Input | `IDX_WIDTH` | logic | Tile size for M dimension |
| `TN_cfg` | Input | `IDX_WIDTH` | logic | Tile size for N dimension |
| `TK_cfg` | Input | `IDX_WIDTH` | logic | Tile size for K dimension |
| **Base Addresses** |||||
| `baseA` | Input | `ADDR_WIDTH` | logic | Base memory address for matrix A |
| `baseB` | Input | `ADDR_WIDTH` | logic | Base memory address for matrix B |
| `baseC` | Input | `ADDR_WIDTH` | logic | Base memory address for matrix C |
| **Address Output** |||||
| `o_addr` | Output | `ADDR_WIDTH` | logic | Generated memory address |
| `addr_id` | Output | 2 | logic | Matrix identifier (0=A, 1=B, 2=C, 3=Invalid) |
| `valid` | Output | 1 | logic | Address output is valid |

### Functional Overview

This module performs **C = A × B** using tiled matrix multiplication:
- Divides large matrices into smaller tiles
- Generates memory addresses sequentially for each tile
- Supports address reuse optimization to skip redundant A-matrix reads
- Iterates through all tiles in K→N→M order

**Operation Sequence:**
1. Set matrix dimensions (M, N, K) and tile sizes (TM_cfg, TN_cfg, TK_cfg)
2. Set base addresses for matrices A, B, C
3. Assert `tile_req` to begin
4. Assert `read_req` each cycle to get next address
5. Monitor `addr_id` to know which matrix the address belongs to
6. Wait for `done_all` to signal completion

> **💡 Note:** The module automatically handles edge tiles (when dimensions aren't perfect multiples of tile sizes) and optimizes by skipping duplicate A-tile reads when the same A-tile multiplies different B-tiles.

### Submodule Hierarchy

```
tile_matmul_agu
├── tile_base_controller (u_ctrl)
│   └── Manages tile-level iteration
└── tile_offsets_agu (u_agu)
    └── Generates element-level addresses
```

---

## tile_base_controller

> **Purpose:** Controls tile-level iteration, calculating base addresses and effective sizes for each tile in the matrix multiplication.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ADDR_WIDTH` | int | 32 | Width of address bus in bits |
| `IDX_WIDTH` | int | 8 | Width of indices in bits |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst` | Input | 1 | logic | Asynchronous reset (active high) |
| **Control Interface** |||||
| `tile_req` | Input | 1 | logic | Request to start tile generation |
| `done_all` | Output | 1 | logic | All tiles completed (1-cycle pulse) |
| **Matrix Dimensions** |||||
| `M` | Input | `IDX_WIDTH` | logic | Total rows in matrix A |
| `N` | Input | `IDX_WIDTH` | logic | Total columns in matrix B |
| `K` | Input | `IDX_WIDTH` | logic | Common dimension |
| **Tile Configuration** |||||
| `TM_cfg` | Input | `IDX_WIDTH` | logic | Configured tile size for M dimension |
| `TN_cfg` | Input | `IDX_WIDTH` | logic | Configured tile size for N dimension |
| `TK_cfg` | Input | `IDX_WIDTH` | logic | Configured tile size for K dimension |
| **Global Base Addresses** |||||
| `baseA` | Input | `ADDR_WIDTH` | logic | Global base address for entire matrix A |
| `baseB` | Input | `ADDR_WIDTH` | logic | Global base address for entire matrix B |
| `baseC` | Input | `ADDR_WIDTH` | logic | Global base address for entire matrix C |
| **AGU Handshake** |||||
| `start_tile` | Output | 1 | logic | One-cycle pulse to start element AGU |
| `tile_done` | Input | 1 | logic | Element AGU finished current tile |
| `tile_ready` | Input | 1 | logic | Element AGU ready for new tile |
| **Current Tile Base Addresses** |||||
| `baseA_tile` | Output | `ADDR_WIDTH` | logic | Base address for current A-tile |
| `baseB_tile` | Output | `ADDR_WIDTH` | logic | Base address for current B-tile |
| `baseC_tile` | Output | `ADDR_WIDTH` | logic | Base address for current C-tile |
| **Effective Tile Sizes** |||||
| `eTM` | Output | `IDX_WIDTH` | logic | Effective M size (handles edge tiles) |
| `eTN` | Output | `IDX_WIDTH` | logic | Effective N size (handles edge tiles) |
| `eTK` | Output | `IDX_WIDTH` | logic | Effective K size (handles edge tiles) |

### FSM States

| State | Encoding | Description |
|-------|----------|-------------|
| `ST_IDLE` | 2'b00 | Idle state, waiting for tile request |
| `ST_WAIT_DONE` | 2'b01 | Processing tile, waiting for completion |

### Address Calculation Formulas

**Tile Base Addresses:**
- `baseA_tile = baseA + (tile_m × TM_cfg × K) + (tile_k × TK_cfg)`
- `baseB_tile = baseB + (tile_k × TK_cfg × N) + (tile_n × TN_cfg)`
- `baseC_tile = baseC + (tile_m × TM_cfg × N) + (tile_n × TN_cfg)`

**Effective Tile Sizes (for edge tiles):**
- `eTM = min(M - tile_m × TM_cfg, TM_cfg)`
- `eTN = min(N - tile_n × TN_cfg, TN_cfg)`
- `eTK = min(K - tile_k × TK_cfg, TK_cfg)`

### Tile Iteration Order

**K (innermost) → N → M (outermost)**

For a matrix multiplication with 3×3×3 tiles, iteration order:
```
(0,0,0) → (0,0,1) → (0,0,2) → 
(0,1,0) → (0,1,1) → (0,1,2) →
(0,2,0) → (0,2,1) → (0,2,2) →
(1,0,0) → (1,0,1) → ... → (2,2,2)
```

> **⚠️ Important:** The innermost K-loop ensures better cache locality by completing all K-tiles for a given (M,N) position before moving to the next N or M tile.

---

## tile_offsets_agu

> **Purpose:** Generates individual element addresses within a single tile. Cycles through phases to read A-matrix, B-matrix, then C-matrix elements.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ADDR_WIDTH` | int | 32 | Width of address bus in bits |
| `IDX_WIDTH` | int | 8 | Width of indices in bits |
| `NULL_ADDR` | logic[31:0] | 32'd9999_9999 | Invalid address marker value |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst` | Input | 1 | logic | Asynchronous reset (active high) |
| **Control** |||||
| `start_tile` | Input | 1 | logic | Start address generation for new tile |
| `read_req` | Input | 1 | logic | Request next address (address valid this cycle) |
| **Tile Base Addresses** |||||
| `baseA_tile` | Input | `ADDR_WIDTH` | logic | Base address of current A-tile |
| `baseB_tile` | Input | `ADDR_WIDTH` | logic | Base address of current B-tile |
| `baseC_tile` | Input | `ADDR_WIDTH` | logic | Base address of current C-tile |
| **Effective Tile Dimensions** |||||
| `eTM` | Input | `IDX_WIDTH` | logic | Actual M size of current tile |
| `eTN` | Input | `IDX_WIDTH` | logic | Actual N size of current tile |
| `eTK` | Input | `IDX_WIDTH` | logic | Actual K size of current tile |
| **Configured Tile Sizes** |||||
| `TM_cfg` | Input | `IDX_WIDTH` | logic | Maximum M tile size |
| `TN_cfg` | Input | `IDX_WIDTH` | logic | Maximum N tile size |
| `TK_cfg` | Input | `IDX_WIDTH` | logic | Maximum K tile size |
| **Full Matrix Dimensions** |||||
| `FULL_K` | Input | `IDX_WIDTH` | logic | Full K dimension (for stride calculation) |
| `FULL_N` | Input | `IDX_WIDTH` | logic | Full N dimension (for stride calculation) |
| **Address Output** |||||
| `o_addr` | Output | `ADDR_WIDTH` | logic | Generated address (NULL_ADDR if invalid) |
| `addr_id` | Output | 2 | logic | Matrix ID: 0=A, 1=B, 2=C, 3=Invalid |
| `valid` | Output | 1 | logic | Address is valid (echoes read_req) |
| **Status** |||||
| `tile_done` | Output | 1 | logic | Current tile complete |
| `tile_ready` | Output | 1 | logic | Ready for new tile (in IDLE) |

### FSM States

| State | Encoding | Description |
|-------|----------|-------------|
| `ST_IDLE` | 2'b00 | Idle, waiting for start_tile signal |
| `ST_INIT` | 2'b01 | Initialize indices to zero |
| `ST_GEN` | 2'b10 | Generate addresses in current phase |

### Phase States

| Phase | Encoding | Description |
|-------|----------|-------------|
| `PH_SKIP` | 2'b00 | Decision phase (check if A-tile can be skipped) |
| `PH_A` | 2'b01 | Generate addresses for A-matrix elements |
| `PH_B` | 2'b10 | Generate addresses for B-matrix elements |
| `PH_C` | 2'b11 | Generate addresses for C-matrix elements |

### Address Generation Patterns

**Phase A - Matrix A (TM × TK elements):**
- Address: `baseA_tile + (i × FULL_K) + k`
- Loop order: for i=0 to TM-1, for k=0 to TK-1
- Only generates if `i < eTM AND k < eTK`

**Phase B - Matrix B (TK × TN elements):**
- Address: `baseB_tile + (k × FULL_N) + j`
- Loop order: for k=0 to TK-1, for j=0 to TN-1
- Only generates if `k < eTK AND j < eTN`

**Phase C - Matrix C (TM × TN elements):**
- Address: `baseC_tile + (i × FULL_N) + j`
- Loop order: for i=0 to TM-1, for j=0 to TN-1
- Only generates if `i < eTM AND j < eTN`

### Optimization: A-Tile Reuse

The module implements an optimization to skip redundant A-matrix reads:
- Stores `prev_baseA` (previous A-tile base address)
- If `baseA_tile == prev_baseA`, skips `PH_A` and goes directly to `PH_B`
- Assumes external logic has cached the A-tile data
- Saves memory bandwidth when same A-tile multiplies multiple B-tiles

> **💡 Optimization Benefit:** For a 64×64 matrix with 8×8 tiles, this optimization reduces A-matrix reads by ~87% (from 8 reads to 1 read per A-tile).

> **⚠️ Critical:** The `o_addr` output is `NULL_ADDR` when element indices exceed effective bounds (eTM, eTN, eTK). Consumer must check for `NULL_ADDR` or only use addresses when `valid=1` and within expected count.

---

# 2. Lego Systolic Array

The Lego Systolic Array is a weight-stationary systolic array architecture designed for efficient matrix multiplication. It features a hierarchical design with configurable array sizes and triangular register shifting for optimal data flow.

## PE (Processing Element)

> **Purpose:** Basic processing element implementing a multiply-accumulate (MAC) operation with weight-stationary dataflow.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_W` | int | 8 | Bit width of input activation and weight |
| `DATA_W_OUT` | int | 32 | Bit width of partial sum output |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst_n` | Input | 1 | logic | Asynchronous reset (active low) |
| **Control** |||||
| `valid_in` | Input | 1 | logic | Input data valid signal |
| `load_w` | Input | 1 | logic | Weight load enable (when high, loads new weight) |
| `valid_out` | Output | 1 | logic | Output data valid signal (echoes valid_in) |
| **Data Inputs** |||||
| `in_act` | Input | `DATA_W` | logic | Input activation from left neighbor PE |
| `in_psum` | Input | `DATA_W_OUT` | logic | Partial sum from top neighbor PE |
| `weight_load` | Input | `DATA_W` | logic | New weight value to load |
| **Data Outputs** |||||
| `out_act` | Output | `DATA_W` | logic | Output activation to right neighbor PE |
| `out_psum` | Output | `DATA_W_OUT` | logic | Output partial sum to bottom neighbor PE |

### Functional Description

**Weight-Stationary MAC Operation:**

The PE implements a stationary weight dataflow where weights remain fixed while activations and partial sums flow through:

1. **Weight Loading Phase (`load_w = 1`):**
   ```
   if (valid_in):
       W_reg ← weight_load
   ```

2. **Computation Phase (`load_w = 0`):**
   ```
   mac_mul = in_act × W_reg
   mac_res = mac_mul + in_psum
   
   if (valid_in):
       act_reg ← in_act
       psum_reg ← mac_res
   ```

3. **Output Propagation:**
   ```
   out_act = act_reg    (propagates right)
   out_psum = psum_reg  (propagates down)
   valid_out = valid_in
   ```

### Operation Modes

| Mode | `load_w` | `valid_in` | Action |
|------|----------|------------|--------|
| **Weight Load** | 1 | 1 | Load new weight into W_reg, no computation |
| **Compute** | 0 | 1 | MAC operation, propagate activation & psum |
| **Idle** | X | 0 | No operation, outputs hold previous values |

### Dataflow Pattern

```
         in_psum
            ↓
in_act → [ PE ] → out_act
            ↓
        out_psum
```

> **💡 Weight-Stationary Advantage:** By keeping weights stationary in each PE, the design minimizes weight memory accesses and power consumption. Weights are loaded once and reused for multiple activations.

> **⚠️ Note:** The PE uses registered outputs, adding 1 cycle of latency. Systolic arrays leverage this pipelining for high throughput.

---

## SA_16x16

> **Purpose:** 16×16 systolic array of PEs implementing matrix multiplication with configurable standalone/chaining mode.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_W` | int | 8 | Bit width of activations and weights |
| `DATA_W_OUT` | int | 32 | Bit width of partial sums |
| `SA_indiv` | int | 1 | Mode: 1=standalone (zero top psum), 0=chainable (use psum_in) |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst_n` | Input | 1 | logic | Asynchronous reset (active low) |
| **Control** |||||
| `load_w` | Input | 1 | logic | Weight loading phase enable |
| `valid_in` | Input | 1 | logic | Input data valid signal |
| `valid_out` | Output | 1 | logic | Output data valid signal |
| **Activation Data** |||||
| `act_in[16]` | Input | `DATA_W` × 16 | logic | Left edge activation inputs (one per row) |
| `act_out[16]` | Output | `DATA_W` × 16 | logic | Right edge activation outputs (one per row) |
| **Partial Sum Data** |||||
| `psum_in[16]` | Input | `DATA_W_OUT` × 16 | logic | Top edge partial sum inputs (one per column) |
| `psum_out[16]` | Output | `DATA_W_OUT` × 16 | logic | Bottom edge partial sum outputs (one per column) |
| **Weight Data** |||||
| `w_load[16][16]` | Input | `DATA_W` × 16 × 16 | logic | 2D array of weights for all PEs |

### Array Structure

The module instantiates a **16×16 grid** of PEs with the following interconnect pattern:

```
Row i, Column j:
- Activation flows: Left → Right (horizontal)
- Partial sum flows: Top → Bottom (vertical)
- Each PE[i][j] receives:
  - act_in[i] (if j=0) OR act_sig[i][j] (from left neighbor)
  - psum_in[j] (if i=0 AND SA_indiv=0) OR psum_sig[i][j] (from top neighbor)
  - w_load[i][j] (unique weight)
```

### Internal Interconnect Signals

| Signal | Dimensions | Description |
|--------|------------|-------------|
| `act_sig[16][17]` | `DATA_W` × 16 × 17 | Activation interconnect (extra column for output) |
| `psum_sig[17][16]` | `DATA_W_OUT` × 17 × 16 | Partial sum interconnect (extra row for output) |
| `valid_sig[16][17]` | 1 × 16 × 17 | Valid signal propagation |

### SA_indiv Mode Behavior

| `SA_indiv` | Top Psum Source | Use Case |
|------------|----------------|----------|
| **1** | Force to 0 | Standalone array, computes full matrix result |
| **0** | Use `psum_in[j]` | Chainable array, accumulates with previous results |

### Matrix Multiplication Mapping

For computing **C = A × B**:
- **A matrix:** Rows fed to `act_in[0:15]`
- **B matrix:** Weights loaded via `w_load[i][j]`
- **C matrix:** Results collected from `psum_out[0:15]` after sufficient cycles

**Computation Formula per PE:**
```
C[i][j] += A[i][k] × B[k][j]
```

Where the systolic array accumulates over K dimension as activations flow through.

### Timing Characteristics

- **Weight Load Time:** 1 cycle (broadcast to all PEs)
- **First Output Latency:** 16 cycles (diagonal wavefront propagation)
- **Peak Throughput:** 16 results per cycle (after initial latency)
- **Total Cycles for 16×16:** ~31 cycles (16 initial + 15 propagation)

> **💡 Performance:** The array can sustain 16×16 = 256 MAC operations per cycle at peak throughput, achieving high computational density.

> **⚠️ Data Alignment:** Inputs must be properly skewed/aligned for correct matrix multiplication. Use with TRSLL module for automatic alignment.

---

## SA_16x16_top

> **Purpose:** Wrapper integrating SA_16x16 with triangular register shifting logic (TRSLL) for proper input data alignment.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_W` | int | 8 | Bit width of activations and weights |
| `DATA_W_OUT` | int | 32 | Bit width of partial sums |
| `SA_indiv` | int | 1 | Standalone (1) or chainable (0) mode |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst_n` | Input | 1 | logic | Asynchronous reset (active low) |
| **Control** |||||
| `load_w` | Input | 1 | logic | Weight loading phase enable |
| `valid_in` | Input | 1 | logic | Input data valid signal |
| `valid_out` | Output | 1 | logic | Output data valid signal |
| **Data Interface** |||||
| `act_in[16]` | Input | `DATA_W` × 16 | logic | Unaligned activation inputs |
| `act_out[16]` | Output | `DATA_W` × 16 | logic | Right edge activation outputs |
| `psum_in[16]` | Input | `DATA_W_OUT` × 16 | logic | Top partial sum inputs |
| `psum_out[16]` | Output | `DATA_W_OUT` × 16 | logic | Bottom partial sum outputs (results) |
| `w_load[16][16]` | Input | `DATA_W` × 16 × 16 | logic | Weight matrix for all PEs |

### Submodule Hierarchy

```
SA_16x16_top
├── TRSLL (reg_shifted_right)
│   └── Triangular shift registers for input alignment
└── SA_16x16 (SA)
    └── 16×16 array of PEs
```

### Functional Overview

This module combines two critical components:

1. **TRSLL (Triangular Register Shift Logic):**
   - Aligns activation inputs with progressive delays
   - Row 0 gets 0-cycle delay, Row 1 gets 1-cycle delay, ..., Row 15 gets 15-cycle delay
   - Creates diagonal wavefront for correct matrix multiplication

2. **SA_16x16 (Systolic Array Core):**
   - Receives aligned activations from TRSLL
   - Performs weight-stationary MAC operations
   - Produces partial sum outputs

### Data Flow

```
act_in[16] → [TRSLL] → act_TRSLL_SA[16] → [SA_16x16] → act_out[16]
                                                ↓
psum_in[16] ────────────────────────────→ psum_out[16]
                                         (vertical flow)
```

### Why Triangular Shifting?

**Problem:** Systolic arrays require temporally aligned data where element A[i][k] must meet weight B[k][j] at the correct PE at the correct time.

**Solution:** TRSLL delays each row by its row index:
- Row 0: No delay (0 cycles)
- Row 1: 1 register (1 cycle delay)
- Row 2: 2 registers (2 cycles delay)
- ...
- Row 15: 15 registers (15 cycles delay)

This creates a diagonal wavefront where data arrives at PE[i][j] at time `i + j`.

> **💡 Alignment Example:** For a 4×4 matrix, TRSLL creates this timing pattern:
> ```
> Time:  t=0  t=1  t=2  t=3
> Row0: [a00][a01][a02][a03]
> Row1:  --  [a10][a11][a12][a13]
> Row2:  --   --  [a20][a21][a22][a23]
> Row3:  --   --   --  [a30][a31][a32][a33]
> ```
> This ensures correct element alignment for matrix multiplication.

---

## TRSLL (Triangular Register Shift Logic)

> **Purpose:** Creates triangular register delay structure for systolic array input alignment.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATAWIDTH` | int | 8 | Bit width of each data element |
| `N_SIZE` | int | 16 | Array size (number of rows) |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst_n` | Input | 1 | logic | Asynchronous reset (active low) |
| **Data Interface** |||||
| `act_in[N_SIZE]` | Input | `DATAWIDTH` × `N_SIZE` | logic | Unaligned input activations |
| `act_out[N_SIZE]` | Output | `DATAWIDTH` × `N_SIZE` | logic | Aligned output activations |

### Register Array Structure

**Total Registers:** `NUM_OF_REGS = (N_SIZE - 1) × N_SIZE / 2`

For `N_SIZE=16`: **120 registers** arranged in triangular pattern:

```
Row 0: 0 registers  → act_out[0] = act_in[0] (no delay)
Row 1: 1 register   → act_out[1] delayed by 1 cycle
Row 2: 2 registers  → act_out[2] delayed by 2 cycles
Row 3: 3 registers  → act_out[3] delayed by 3 cycles
...
Row 15: 15 registers → act_out[15] delayed by 15 cycles
```

### Register Indexing Formula

For row `k` (k=1 to N_SIZE-1):
- **Base index:** `base = k × (k - 1) / 2`
- **First register:** `reg_shifted[base + 1]` (directly from `act_in[k]`)
- **Last register:** `reg_shifted[base + k]` (feeds `act_out[k]`)
- **Chain:** Each register shifts into the next

### Delay Pattern Visualization

For 4×4 array (N_SIZE=4):

```
act_in[0] ────────────────────────────→ act_out[0]  (0 cycles)

act_in[1] ──[reg1]────────────────────→ act_out[1]  (1 cycle)

act_in[2] ──[reg2]──[reg3]────────────→ act_out[2]  (2 cycles)

act_in[3] ──[reg4]──[reg5]──[reg6]────→ act_out[3]  (3 cycles)
```

### Generate Logic Structure

The module uses nested `generate` blocks:

1. **Outer loop (k):** Iterates over rows 1 to N_SIZE-1
2. **First column register:** Captures input from `act_in[k]`
3. **Depth registers:** Chain of shift registers for rows with k>1
4. **Output assignment:** Last register in chain feeds `act_out[k]`

### Register Count by Array Size

| Array Size | Total Registers | Formula |
|------------|----------------|---------|
| 4×4 | 6 | (3×4)/2 = 6 |
| 8×8 | 28 | (7×8)/2 = 28 |
| 16×16 | 120 | (15×16)/2 = 120 |
| 32×32 | 496 | (31×32)/2 = 496 |

> **💡 Hardware Cost:** The triangular structure uses fewer registers than a full rectangular delay buffer. For N=16, this saves 120 vs. 136 registers (if using N registers per row).

> **⚠️ Latency Impact:** The TRSLL adds up to (N_SIZE-1) cycles of latency. For a 16×16 array, this adds 15 additional cycles before the first output appears.

---

## Lego_Systolic_Array

> **Purpose:** Top-level systolic array module instantiating multiple 16×16 arrays for larger matrix operations. *(Currently under development)*

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_W` | int | 8 | Bit width of activations and weights |
| `DATA_W_OUT` | int | 32 | Bit width of partial sums |
| `SA_indiv` | int | 1 | Standalone (1) or chainable (0) mode |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | logic | System clock |
| `rst_n` | Input | 1 | logic | Asynchronous reset (active low) |
| **Control** |||||
| `load_w` | Input | 1 | logic | Weight loading phase enable |
| `valid_in` | Input | 1 | logic | Input data valid signal |
| `TYPE_Lego` | Input | 2 | logic | Configuration type for Lego architecture |
| `valid_out` | Output | 1 | logic | Output data valid signal |
| **Data Interface** |||||
| `act_in[64]` | Input | `DATA_W` × 64 | logic | Left edge activations (64 rows) |
| `w_load[32][32]` | Input | `DATA_W` × 32 × 32 | logic | Weight matrix (32×32) |
| `psum_out[32]` | Output | `DATA_W_OUT` × 32 | logic | Bottom edge partial sum outputs |

### Architecture Overview

The module instantiates **4× SA_16x16_top** instances:
- `SA_1`, `SA_2`, `SA_3`, `SA_4`
- Intended for configurable Lego-style tiling
- Allows construction of larger arrays or different topologies

### Configuration Types (TYPE_Lego)

| `TYPE_Lego` | Configuration | Description |
|-------------|---------------|-------------|
| `2'b00` | Single 16×16 | Use SA_1 only *(planned)* |
| `2'b01` | 2× 16×16 | Use SA_1, SA_2 *(planned)* |
| `2'b10` | 32×16 or 16×32 | Horizontal/vertical tiling *(planned)* |
| `2'b11` | 4× 16×16 | Full 32×32 array *(planned)* |

> **⚠️ Development Status:** This module is currently incomplete. The control logic (`Lego_control_unit`) and proper interconnect between sub-arrays are under development. The `TYPE_Lego` parameter is declared but not yet implemented.

> **💡 Design Intent:** The Lego architecture aims to provide flexibility in array size/shape by dynamically connecting smaller 16×16 building blocks based on matrix dimensions and throughput requirements.

---

# 3. Normalization Modules

## layer_norm1

> **Purpose:** Sequential layer normalization with FSM-based computation. Computes normalized output: `(x - mean) / std_dev` using 8-state pipeline.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `DATA_WIDTH` | int | 16 | Bit width of input/output data |
| `EMBED_DIM` | int | 8 | Embedding dimension (vector length) |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `clk` | Input | 1 | wire | System clock |
| `rst_n` | Input | 1 | wire | Asynchronous reset (active low) |
| **Control** |||||
| `layernorm_start` | Input | 1 | wire | Start layer normalization computation |
| `layernorm_done` | Output | 1 | reg | Computation complete, output valid |
| **Data Interface** |||||
| `activation_in` | Input | `DATA_WIDTH` × `EMBED_DIM` | wire signed | Input activation vector (array) |
| `normalized_out` | Output | `DATA_WIDTH` × `EMBED_DIM` | reg signed | Normalized output vector (array) |

### FSM States

| State | Encoding | Description | Computation |
|-------|----------|-------------|-------------|
| `IDLE` | 3'd0 | Idle, waiting for start | - |
| `LOAD` | 3'd1 | Load input data to buffer | `buffer[i] ← activation_in[i]` |
| `SUM1` | 3'd3 | Sum all elements | `sum = Σ buffer[i]` |
| `MEAN` | 3'd2 | Calculate mean | `mean = sum / EMBED_DIM` (shift right) |
| `SUM2` | 3'd6 | Sum squared differences | `sum_sq = Σ(buffer[i] - mean)²` |
| `VARI` | 3'd7 | Calculate variance, run sqrt | `vari = sum_sq / EMBED_DIM`, sqrt iterations |
| `NORM` | 3'd5 | Normalize each element | `out[i] = (buffer[i] - mean) / std_dev` |
| `DONE` | 3'd4 | Output ready | Assert `layernorm_done` |

### Algorithm

**Step-by-step computation:**

1. **Mean Calculation:**
   ```
   sum = Σ(x[i]) for i=0 to EMBED_DIM-1
   mean = sum >> log2(EMBED_DIM)
   ```

2. **Variance Calculation:**
   ```
   sum_sq = Σ((x[i] - mean)²) for i=0 to EMBED_DIM-1
   variance = sum_sq >> log2(EMBED_DIM)
   ```

3. **Standard Deviation (Newton-Raphson):**
   ```
   std_dev = variance >> 1  (initial guess)
   for k=0 to 3:
       std_dev = (std_dev + variance/std_dev) >> 1
   ```

4. **Normalization:**
   ```
   normalized_out[i] = (x[i] - mean) / std_dev
   ```

### Timing

- **Latency:** ~10-12 clock cycles (depends on EMBED_DIM and sqrt iterations)
- **Throughput:** 1 vector per ~12 cycles
- **Newton-Raphson Iterations:** 4 iterations (hardcoded)

> **⚠️ Note:** Division by zero protection: if `std_dev = 0`, output is forced to 0. For hardware efficiency, division uses power-of-2 (shift) for mean/variance, but exact division for final normalization.

> **💡 Optimization:** For power-of-2 EMBED_DIM values (2, 4, 8, 16), division is replaced with efficient right-shift operations.

---

## layer_norm2

> **Purpose:** Purely combinational layer normalization. Outputs normalized values in same cycle (no clock, no FSM).

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `N` | int | 8 | Vector length (number of elements) |
| `DATA_WIDTH` | int | 8 | Input/output bit width |
| `ACC_WIDTH` | int | 32 | Accumulator width for internal calculations |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Data Interface** |||||
| `in_vec` | Input | `DATA_WIDTH` × `N` | logic signed | Input vector (array of N elements) |
| `out_vec` | Output | `DATA_WIDTH` × `N` | logic signed | Normalized output (array of N elements) |

### Functional Description

**Pure combinational logic** - no clock, no registers:

1. **Mean Computation:**
   ```verilog
   sum = Σ in_vec[i]
   mean = sum / N
   ```

2. **Variance Computation:**
   ```verilog
   variance = Σ (in_vec[i] - mean)² / N
   ```

3. **Standard Deviation (Newton-Raphson):**
   ```verilog
   function int_sqrt_hwstyle(variance):
       std = variance >> 1  // initial guess
       for k=0 to 3:
           std = (std + variance/std) >> 1
       return std
   ```

4. **Normalization with Scaling:**
   ```verilog
   norm = ((in_vec[i] - mean) * SCALE) / stddev
   out_vec[i] = saturate(norm, -128, 127)
   ```

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SCALE` | 128 | Scaling factor to preserve precision in 8-bit output |

### Saturation Logic

Output is saturated to fit in `DATA_WIDTH` signed range:
- If `norm > 127`: clamp to 127
- If `norm < -128`: clamp to -128
- Else: use norm value

> **💡 Key Feature:** Fully combinational design means zero latency but higher critical path. Suitable for low-throughput scenarios or when pipelined externally.

> **⚠️ Warning:** Combinational division is expensive in hardware. This design trades area/power for zero latency. Consider `layer_norm1` for high-frequency designs.

---

## batch_norm

> **Purpose:** Batch normalization performing affine transformation `y = A·x + B` on packed row data.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Data_Width` | int | 32 | Bit width of each element |
| `N` | int | 32 | Number of elements per row/batch |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Clock and Reset** |||||
| `CLK` | Input | 1 | wire | System clock |
| `RST` | Input | 1 | wire | Asynchronous reset (active low) |
| **Data Interface** |||||
| `in_row` | Input | `N × Data_Width` | wire | Packed input row (concatenated elements) |
| `out_row` | Output | `N × Data_Width` | reg | Packed output row (concatenated elements) |
| **Control** |||||
| `INBatch_Valid` | Input | 1 | wire | Input data valid strobe |
| `OutBatch_Valid` | Output | 1 | reg | Output data valid (1-cycle pulse) |
| **Normalization Parameters** |||||
| `A` | Input | `Data_Width` × `N` | wire | Scale factors (array, one per element) |
| `B` | Input | `Data_Width` × `N` | wire | Bias terms (array, one per element) |

### Functional Description

**Computation:** For each element i in [0, N-1]:
```
x[i] = in_row[i*Data_Width +: Data_Width]  // Unpack
y[i] = A[i] × x[i] + B[i]                   // Affine transform
out_row[i*Data_Width +: Data_Width] = y[i]  // Pack
```

### Data Packing Format

**Input/Output Row Format:**
```
in_row = {x[N-1], x[N-2], ..., x[1], x[0]}
         |<- MSB              LSB ->|
```

Each element occupies `Data_Width` bits, concatenated in little-endian order.

### Timing Behavior

- **Latency:** 1 clock cycle
- **Throughput:** 1 row per cycle (when `INBatch_Valid` asserted)
- `OutBatch_Valid` pulses high for 1 cycle after valid input
- If `INBatch_Valid=0`, `OutBatch_Valid` forced to 0

### Usage in Batch Normalization

In standard batch normalization, `A` and `B` are derived from:
```
A[i] = gamma[i] / sqrt(variance[i] + epsilon)
B[i] = beta[i] - (A[i] × mean[i])
```

Where:
- `gamma`, `beta`: Learnable affine parameters
- `mean`, `variance`: Batch statistics (pre-computed)
- `epsilon`: Small constant for numerical stability

> **💡 Note:** This module performs the affine transformation only. Mean and variance computation must be done externally, with A and B pre-calculated.

> **⚠️ Design Note:** The module uses combinational unpacking/computation with registered output. Critical path includes N multiplications and additions.

---

# 4. Activation Functions

## swish

> **Purpose:** Hard-Swish activation function with configurable division method. Implements `y = x × ReLU6(x+3) / 6`.

### Module Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `WIDTH` | int | 8 | Bit width of input/output (signed) |
| `USE_SHIFT_ADD` | int | 0 | Division method: 0=exact division, 1=shift-add approximation |

### Port Signals

| Port Name | Direction | Width | Type | Description |
|-----------|-----------|-------|------|-------------|
| **Data Interface** |||||
| `x` | Input | `WIDTH` | signed | Input value (signed integer) |
| `y` | Output | `WIDTH` | reg signed | Output value (activated) |

### Functional Description

**Hard-Swish Function:**
```
y = x × ReLU6(x + 3) / 6
```

Where:
- `ReLU6(z) = max(0, min(z, 6))` - Clamps z to range [0, 6]

**Step-by-step computation:**

1. **Offset:** `x_plus3 = x + 3`

2. **ReLU6 Clamp:**
   ```
   if (x_plus3 < 0):  relu6_val = 0
   if (x_plus3 > 6):  relu6_val = 6
   else:              relu6_val = x_plus3
   ```

3. **Multiply and Divide:**
   - **Exact Division (`USE_SHIFT_ADD=0`):**
     ```
     y = (x × relu6_val) / 6
     ```
   
   - **Approximate Division (`USE_SHIFT_ADD=1`):**
     ```
     y = x × ((relu6_val >> 3) + (relu6_val >> 5))
     Approximation: 1/6 ≈ 1/8 + 1/32 = 0.15625
     Actual: 1/6 = 0.16667
     Error: ~6.25%
     ```

### Division Method Comparison

| Method | Parameter Value | Operation | Accuracy | Hardware Cost |
|--------|----------------|-----------|----------|---------------|
| Exact | `USE_SHIFT_ADD=0` | Division by 6 | Exact | Higher (divider) |
| Approximate | `USE_SHIFT_ADD=1` | Shifts + add | ~6% error | Lower (shifters + adder) |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `THREE` | 3 | Offset constant for hard-swish |
| `SIX` | 6 | Divisor constant for hard-swish |

### Generate Block

The module uses SystemVerilog `generate` to conditionally instantiate exact or approximate division:

```systemverilog
generate
    if (USE_SHIFT_ADD == 0) begin : exact_div
        // Exact division implementation
    end else begin : shift_add
        // Approximate shift-add implementation
    end
endgenerate
```

### Activation Curve

For 8-bit signed input range [-128, 127]:

| Input x | x+3 | ReLU6(x+3) | Output y (exact) |
|---------|-----|------------|------------------|
| -128 | -125 | 0 | 0 |
| -3 | 0 | 0 | 0 |
| 0 | 3 | 3 | 0 |
| 3 | 6 | 6 | 3 |
| 127 | 130 | 6 | 127 |

> **💡 Hardware Optimization:** The shift-add approximation is recommended for area/power-constrained designs where 6% error is acceptable. Exact division should be used for high-precision requirements.

> **⚠️ Note:** This is a **combinational module** (no clock). Output `y` updates immediately when `x` changes. Register externally if pipelining is needed.

---

# Module Summary Table

| Module | Type | Clocked | Latency | Main Function |
|--------|------|---------|---------|---------------|
| `tile_matmul_agu` | Controller | Yes | N/A | Top-level tile iterator |
| `tile_base_controller` | Controller | Yes | N/A | Tile address calculator |
| `tile_offsets_agu` | AGU | Yes | 0 cycles | Element address generator |
| `PE` | Compute | Yes | 1 cycle | MAC processing element |
| `SA_16x16` | Compute Array | Yes | 16 cycles | 16×16 systolic array core |
| `SA_16x16_top` | Compute Array | Yes | 31 cycles | Array with input alignment |
| `TRSLL` | Data Alignment | Yes | 0-15 cycles | Triangular register shifter |
| `Lego_Systolic_Array` | Compute Array | Yes | TBD | Multi-array configuration *(dev)* |
| `layer_norm1` | Normalization | Yes | ~12 cycles | Sequential layer norm with FSM |
| `layer_norm2` | Normalization | No | 0 cycles | Combinational layer norm |
| `batch_norm` | Normalization | Yes | 1 cycle | Batch norm affine transform |
| `swish` | Activation | No | 0 cycles | Hard-swish activation |

---

# Design Conventions

## Reset Strategy
- **Tiled AGU modules:** Asynchronous active-high reset (`rst`)
- **Normalization modules:** Asynchronous active-low reset (`rst_n` or `RST`)
- **Activation modules:** No reset (purely combinational)

## Naming Conventions
- **Parameters:** UPPERCASE or PascalCase (e.g., `ADDR_WIDTH`, `Data_Width`)
- **Inputs:** lowercase with underscores (e.g., `clk`, `tile_req`, `activation_in`)
- **Outputs:** lowercase with underscores (e.g., `done_all`, `normalized_out`)
- **Internal signals:** lowercase with underscores (e.g., `next_state`, `sum_sq`)

## Data Packing
- **Arrays:** Used for multi-element signals (e.g., `activation_in[0:EMBED_DIM-1]`)
- **Packed vectors:** Used for serialized data (e.g., `in_row[N*Data_Width-1:0]`)
- **Unpacking syntax:** `signal[i*WIDTH +: WIDTH]` extracts element i

---

# Usage Guidelines

## Package Integration

Import the accelerator packages for cleaner code:

```systemverilog
import accelerator_common_pkg::*;
import accelerator_matmul_pkg::*;
import accelerator_norm_pkg::*;
import accelerator_activation_pkg::*;

// Now use typedefs
addr_t memory_address;
matrix_id_t current_matrix;
layer_norm_state_t norm_state;
```

## Typical Integration Example

```systemverilog
// Matrix multiplication setup
tile_matmul_agu #(
    .ADDR_WIDTH(32),
    .IDX_WIDTH(8)
) matmul_agu (
    .clk(clk),
    .rst(rst),
    .M(64), .N(64), .K(64),           // 64×64 matrices
    .TM_cfg(8), .TN_cfg(8), .TK_cfg(8), // 8×8 tiles
    .baseA(32'h0000), .baseB(32'h1000), .baseC(32'h2000),
    .tile_req(start),
    .read_req(addr_req),
    .o_addr(mem_addr),
    .addr_id(mat_sel),
    .valid(addr_valid),
    .done_all(complete)
);

// Layer normalization
layer_norm1 #(
    .DATA_WIDTH(16),
    .EMBED_DIM(8)
) layer_norm (
    .clk(clk),
    .rst_n(rst_n),
    .layernorm_start(start_norm),
    .activation_in(input_vec),
    .normalized_out(output_vec),
    .layernorm_done(norm_done)
);

// Activation function
swish #(
    .WIDTH(8),
    .USE_SHIFT_ADD(1)  // Use approximation
) activation (
    .x(input_data),
    .y(activated_data)
);
```

---

# Additional Resources

- **Package Documentation:** See `Include/README.md` for package details
- **Testbenches:** See `Testbench/` directory for usage examples
- **Quick Reference:** See `Include/QUICK_REFERENCE.md` for type lookup

---

*Document Version: 1.0*  
*Last Updated: October 9, 2025*
