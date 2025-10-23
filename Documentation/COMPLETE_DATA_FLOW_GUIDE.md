# Complete Data Flow Guide - MobileViT Accelerator
**From 256×256×3 Input → 1×1×1000 Output**

---

## ⚠️ Important Context: Research Project

**This document presents idealized cycle counts and data flow sequences.** These are:
- **Hand-calculated estimates** assuming perfect conditions (no stalls, no conflicts, instant handshaking)
- **Unverified by simulation** - the RTL has been written but not systematically tested
- **Design intentions** - what we designed the hardware to do, not measured performance
- **Best-case scenarios** - real cycle counts will be higher due to handshaking delays, arbitration, stalls, and corner cases

**Real verification work (simulation + FPGA) is pending.** This guide demonstrates our design process and theoretical analysis as part of a student graduation/research project.

---

## Table of Contents

1. [Overview](#overview)
2. [Initial Setup](#initial-setup)
3. [Layer-by-Layer Flow](#layer-by-layer-flow)
4. [Detailed Walkthrough - Conv Layer Example](#detailed-walkthrough)
5. [Memory Management](#memory-management)
6. [Timing Diagrams](#timing-diagrams)
7. [Debug Checklist](#debug-checklist)

---

## Overview

> **NOTE:** This guide provides **idealized cycle-by-cycle analysis** of the MobileViT accelerator's intended operation, tracing the designed data flow from initial input loading through final output generation. All cycle counts are theoretical estimates.

### **Input → Output Transformation**

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: 256×256×3 RGB Image                                     │
│  (196,608 bytes in DRAM @ 0x8000_0000)                          │
└───────────────────┬─────────────────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  MobileViT Network  │
         │  ~50 Layers         │
         │  (Conv+Transformer) │
         └──────────┬──────────┘
                    │
┌───────────────────▼─────────────────────────────────────────────┐
│  OUTPUT: 1×1×1000 Class Scores                                  │
│  (1,000 float32 values in DRAM @ 0x9000_0000)                   │
└─────────────────────────────────────────────────────────────────┘
```

### **High-Level Flow**

```
Stage 1: Initial Conv Stem
  256×256×3 → 128×128×16 (Conv 3×3, stride=2)
  
Stage 2: MV2 Blocks (MobileNetV2-style)
  128×128×16 → 64×64×32 → 32×32×64 → 16×16×96
  
Stage 3: MobileViT Blocks (Conv + Transformer + Conv)
  16×16×96 → 16×16×128 (with global attention)
  
Stage 4: More MV2 Blocks
  16×16×128 → 8×8×160 → 4×4×160
  
Stage 5: Global Pooling + FC
  4×4×160 → 1×1×160 (avg pool) → 1×1×1000 (FC)
```

---

## Initial Setup

### **Phase 0: Power-On and Initialization**

#### **Step 0.1: Hardware Reset**
```verilog
Time 0ns:
  rst_n = 0  (active low reset)
  All FSMs → IDLE state
  All counters → 0
  Memory contents → undefined
  
Time 100ns:
  rst_n = 1  (release reset)
  global_controller.state → IDLE
  DMA ready to accept commands
```

#### **Step 0.2: CPU Initializes DRAM**
```
CPU writes to DRAM (via separate interface):
  
  0x8000_0000: Input image (256×256×3 = 196,608 bytes)
               - Pixel[0,0] = {R, G, B}
               - Pixel[0,1] = {R, G, B}
               - ... row-major order
  
  0x8100_0000: Conv1 weights (3×3×3×16 = 432 bytes)
               - Kernel[0] for output channel 0
               - Kernel[1] for output channel 1
               - ... (16 kernels total)
  
  0x8200_0000: Conv1 BatchNorm params
               - gamma[0..15]
               - beta[0..15]
               - mean[0..15]
               - var[0..15]
  
  ... (repeat for all ~50 layers)
```

#### **Step 0.3: CPU Programs Accelerator**
```c
// CPU code (pseudocode)

// 1. Soft reset accelerator
write_reg(ACCEL_BASE + 0x00, 0x02);  // CONTROL[1] = soft_reset
write_reg(ACCEL_BASE + 0x00, 0x00);  // Clear reset

// 2. Prepare first descriptor (load Conv1 weights)
descriptor_t desc_conv1_weights = {
    .dram_addr = 0x8100_0000,        // Weight location
    .sram_addr = 0x0000,              // WgtBuf start
    .length    = 432,                 // 3×3×3×16 bytes
    .stride    = 0,
    .tile_h    = 16,                  // Output channels
    .tile_w    = 9,                   // 3×3×1 = 9 per kernel
    .c_in      = 3,                   // RGB input
    .flags     = 0x01                 // IS_WEIGHT flag
};

// 3. Write descriptor to registers (256 bits = 8×32-bit words)
write_reg(ACCEL_BASE + 0x10, desc_conv1_weights.words[0]);
write_reg(ACCEL_BASE + 0x14, desc_conv1_weights.words[1]);
write_reg(ACCEL_BASE + 0x18, desc_conv1_weights.words[2]);
write_reg(ACCEL_BASE + 0x1C, desc_conv1_weights.words[3]);
write_reg(ACCEL_BASE + 0x20, desc_conv1_weights.words[4]);
write_reg(ACCEL_BASE + 0x24, desc_conv1_weights.words[5]);
write_reg(ACCEL_BASE + 0x28, desc_conv1_weights.words[6]);
write_reg(ACCEL_BASE + 0x2C, desc_conv1_weights.words[7]);

// 4. Push descriptor to controller FIFO
write_reg(ACCEL_BASE + 0x30, 0x01);  // DESC_PUSH

// 5. Start processing
write_reg(ACCEL_BASE + 0x00, 0x01);  // CONTROL[0] = start
```

---

## Layer-by-Layer Flow

### **Layer 1: Conv 3×3 (Stem)**
**Operation**: 256×256×3 → 128×128×16 (stride=2, pad=1)

**Step 1a: Weight Loading**
```
Controller State: IDLE → WEIGHT_LOAD_DMA → WEIGHT_LOAD_SA

Cycle 0-10: DMA reads weights from DRAM
  dma.araddr  = 0x8100_0000
  dma.arlen   = 53 (54 AXI beats for 432 bytes, 64-bit per beat)
  dma.arvalid = 1
  
Cycle 11: DRAM returns first 64-bit word
  dma.rdata = {W[1], W[0]}  (2×32-bit)
  
Cycle 12: DMA writes to memory subsystem
  mem.dma_we    = 1
  mem.dma_wdata = 32'h...  (first word)
  mem.dma_waddr = 0x0000   (WgtBuf bank 0, word 0)
  
Cycle 13-14: DMA writes remaining word from same AXI beat
  mem.dma_waddr = 0x0001 (different bank via [3:0])
  
Cycle 15-68: DMA continues until all 432 bytes written
  Final write: mem.dma_waddr = 0x006B (word 107)
  
Cycle 43: DMA asserts done
  dma.done = 1
  
Cycle 44: Controller moves to WEIGHT_LOAD_SA
  Load weights from WgtBuf into SA weight registers
  SA absorbs 16×16 weight tile (we only use 16×9, rest zeroed)
  
Cycle 50: Weights loaded
  Controller → INPUT_TILE_LOAD
```

**Step 1b: First Tile - Input Loading**
```
Controller State: INPUT_TILE_LOAD

Input tile size: 18×18×3 (includes padding for 3×3 kernel, stride=2)
  - Produces 8×8×16 output tile
  - Need to process 128/8 × 128/8 = 16×16 = 256 tiles total

Cycle 51-55: Controller sends DMA command
  desc.dram_addr = 0x8000_0000  (input image start)
  desc.sram_addr = 0x0000        (ActBufA start)
  desc.length    = 18×18×3 = 972 bytes
  
Cycle 56-90: DMA reads 972 bytes (61 AXI beats)
  DMA writes to ActBufA in memory subsystem
  Addresses interleaved across 16 banks:
    Bank 0: pixels [0, 16, 32, 48, ...]
    Bank 1: pixels [1, 17, 33, 49, ...]
    ...
  
Cycle 91: DMA done, input tile in ActBufA
  Controller → AGU_SETUP
```

**Step 1c: AGU Address Generation Setup**
```
Controller State: AGU_SETUP

Cycle 92: Controller sends AGU command
  agu.operation = OP_REGULAR_CONV (3×3)
  agu.baseA     = 0x0000 (ActBufA start)
  agu.baseB     = 0x0000 (WgtBuf start)
  agu.baseC     = 0x0000 (PSumBuf start)
  agu.tile_h    = 8  (output rows)
  agu.tile_w    = 8  (output cols)
  agu.c_in      = 3  (RGB)
  agu.k_h       = 3  (kernel height)
  agu.k_w       = 3  (kernel width)
  agu.stride    = 2
  agu.start     = 1
  
Cycle 93: AGU generates first address set
  agu.addrA[0]  = 0x0000  (pixel [0,0])
  agu.addrA[1]  = 0x0001  (pixel [0,1])
  agu.addrA[2]  = 0x0002  (pixel [0,2])
  ...
  agu.addrB[0]  = 0x0000  (weight for C_out=0, k_row=0, k_col=0)
  ...
  
Cycle 94: AGU ready
  agu.ready = 1
  Controller → COMPUTE
```

**Step 1d: Computation (First Output Pixel)**
```
Controller State: COMPUTE

Cycle 95: Request first data
  ctrl.tile_req = 1
  
Cycle 96: Memory reads data
  Memory subsystem reads from ActBufA (16 banks in parallel)
  Bank 0 outputs 32-bit word → unpacks to 4× 8-bit elements
  Bank 1 outputs 32-bit word → unpacks to 4× 8-bit elements
  ...
  Total: 64× 8-bit elements available
  
  AGU selects which 64 elements are activation inputs
  AGU selects which 64 elements are weight inputs
  
Cycle 97: Data valid
  mem.data_valid = 1
  mem.data_to_sa_act[63:0] = {act_pixel[0,0,R], act_pixel[0,0,G], act_pixel[0,0,B], ...}
  mem.data_to_sa_wgt[63:0] = {weight[0,0,0], weight[0,0,1], ...}
  
Cycle 98: SA computes (Type 0: 16×64 config)
  Each of 16 rows processes one output channel
  Each row does 64 MACs (but we only use 9 for 3×3×3)
  
  Row 0: out_ch[0] += sum(act × weight[0])
  Row 1: out_ch[1] += sum(act × weight[1])
  ...
  Row 15: out_ch[15] += sum(act × weight[15])
  
Cycle 99: Partial sums ready
  sa.psum_out[0]  = 32'hXXXX (accumulated for channel 0)
  sa.psum_out[1]  = 32'hXXXX
  ...
  
Cycle 100: Write partial sums to PSumBuf
  mem.psum_we    = 1
  mem.psum_waddr = 0x0000 (output pixel [0,0])
  mem.psum_wdata[0] = sa.psum_out[0]
  ...
  
  For 3×3 kernel: need 9 iterations to complete one output pixel
  AGU generates 9 address sets (scanning 3×3 window)
  
Cycle 101-108: Continue for remaining 8 kernel positions
  AGU steps through 3×3 window
  SA accumulates partial sums
  
Cycle 109: First output pixel complete
  agu.tile_done = 1 (for this pixel)
  AGU moves to next output pixel [0,1]
  
Cycle 110-180: Compute remaining 63 output pixels (8×8 - 1)
  Same process: AGU → Mem → SA → PSumBuf
  
Cycle 181: All 64 output pixels (8×8×16) computed for this tile
  agu.all_tiles_done = 1
  Controller → POST_PROCESS
```

**Step 1e: Post-Processing**
```
Controller State: POST_PROCESS

Cycle 182-185: Read 16 psums from PSumBuf
  Post-processing pipeline processes 16 elements/cycle
  Need 64/16 = 4 cycles for full 8×8×16 tile
  
Cycle 182: Process pixels [0,0] through [0,15] (first 16 channels)
  Stage 1 (Batch Norm):
    bn_in[i]  = psum[i]
    bn_out[i] = (bn_in[i] - mean[i]) / sqrt(var[i]) * gamma[i] + beta[i]
  
  Stage 2 (Swish):
    swish_in[i]  = bn_out[i]
    swish_out[i] = swish_in[i] * sigmoid(swish_in[i])
  
  Stage 3 (Layer Norm - bypassed for conv):
    ln_out[i] = swish_out[i]  (pass-through)
  
Cycle 185: Pipeline output ready
  Final output written back to PSumBuf or ActBufB
  
Cycle 186-189: Process remaining 48 pixels
  
Cycle 190: Post-processing complete
  Controller → WRITEBACK
```

**Step 1f: Writeback to DRAM**
```
Controller State: WRITEBACK

Cycle 191-195: DMA write command
  desc.dram_addr = 0x8300_0000  (output buffer in DRAM)
  desc.sram_addr = 0x0000        (PSumBuf)
  desc.length    = 8×8×16×4 = 4096 bytes (INT32 outputs)
  
Cycle 196-250: DMA reads from PSumBuf, writes to DRAM
  DMA reads 32-bit words from memory subsystem
  DMA writes 64-bit beats to DRAM (2 words per beat)
  
Cycle 251: Writeback done
  dma.done = 1
  Controller → NEXT_TILE
```

**Step 1g: Next Tile - Ping-Pong**
```
Controller State: NEXT_TILE

Key optimization: While computing tile 0, DMA loads tile 1 into ActBufB

Cycle 252: Check if more tiles
  tile_counter = 1 (just finished tile 0)
  total_tiles  = 256
  More tiles remain → PING_PONG_SWAP
  
Cycle 253: Swap buffers
  Active buffer: ActBufA → ActBufB
  DMA target: ActBufB → ActBufA (for next-next tile)
  
Cycle 254: Load tile 1 into ActBufA (while computing from ActBufB)
  This happens in parallel with computation!
  
Cycle 255: Controller → INPUT_TILE_LOAD (but no DMA wait)
  Tile 1 data already in ActBufB from background load
  Controller → AGU_SETUP immediately
  
Cycle 256-350: Compute tile 1 (same as tile 0 flow)
  
...repeat for all 256 tiles...

Cycle ~64,000: All 256 tiles computed
  Final output: 128×128×16 in DRAM @ 0x8300_0000
  Controller → IDLE
  ctrl.irq = 1 (interrupt CPU)
  
CPU reads STATUS register:
  status[1] = 1 (done)
  status[0] = 0 (not busy)
  
CPU reads TILE_COUNTER:
  tile_counter = 256
  
CPU reads CYCLE_COUNTER:
  cycle_counter = 64,000 (example)
  
CPU starts next layer...
```

---

## Detailed Walkthrough - Conv Layer Example

> **Note:** This section provides a cycle-by-cycle trace of a single multiply-accumulate operation to demonstrate the complete pipeline flow.

### **Zooming Into One MAC Operation**

Let's trace a **single 8-bit multiply-accumulate** through the entire pipeline:

```
╔══════════════════════════════════════════════════════════════╗
║  Computing Output Pixel [0,0], Channel 0                     ║
║  From 3×3 Conv with Stride 2                                 ║
╚══════════════════════════════════════════════════════════════╝

Input Pixels (18×18×3 tile in ActBufA):
  Pixel[0,0] = {R=128, G=64, B=32}  @ address 0x0000 (bank 0)
  Pixel[0,1] = {R=120, G=70, B=30}  @ address 0x0001 (bank 1)
  Pixel[0,2] = {R=110, G=75, B=35}  @ address 0x0002 (bank 2)
  ...

Weights (3×3×3×16 in WgtBuf):
  Kernel[0] for output channel 0:
    W[0,0,R] = 12   @ address 0x0000
    W[0,0,G] = -8   @ address 0x0001
    W[0,0,B] = 5    @ address 0x0002
    W[0,1,R] = 3    @ address 0x0003
    ...
    W[2,2,B] = -7   @ address 0x0008 (9 weights total per channel)

══════════════════════════════════════════════════════════════

Cycle 96: AGU generates addresses for 3×3 window
  agu.addrA = {0x0000, 0x0001, 0x0002,  // row 0: [0,0], [0,1], [0,2]
               0x0012, 0x0013, 0x0014,  // row 1: [1,0], [1,1], [1,2]
               0x0024, 0x0025, 0x0026}  // row 2: [2,0], [2,1], [2,2]
               × 3 (for RGB) = 27 addresses
  
  agu.addrB = {0x0000, 0x0001, 0x0002, ..., 0x0008}  // 9 weights

Cycle 97: Memory subsystem reads
  Bank 0: Read word @ 0x0000 → {pixel[0,0,B], pixel[0,0,G], pixel[0,0,R], ...}
                                 = {32, 64, 128, ...}
  
  Unpacking (mem → SA):
    data_to_sa_act[0] = 128 (R)
    data_to_sa_act[1] = 64  (G)
    data_to_sa_act[2] = 32  (B)
    ...
  
  Similarly for weights:
    data_to_sa_wgt[0] = 12  (W[0,0,R])
    data_to_sa_wgt[1] = -8  (W[0,0,G])
    data_to_sa_wgt[2] = 5   (W[0,0,B])
    ...

Cycle 98: Systolic Array PE[0,0] (channel 0, position 0)
  Input: act = 128, wgt = 12
  Operation: mac = psum_in + (act × wgt)
            = 0 + (128 × 12)
            = 1,536
  
  PE[0,1] (channel 0, position 1):
    mac = 1,536 + (64 × -8) = 1,536 - 512 = 1,024
  
  PE[0,2]:
    mac = 1,024 + (32 × 5) = 1,024 + 160 = 1,184
  
  ... (continue for all 9 positions)
  
Cycle 99: Row 0 output (channel 0 partial sum)
  sa.psum_out[0] = 32'h000004D0 (1,232 in decimal, example)
  
Cycle 100: Write to PSumBuf
  mem.psum_we    = 1
  mem.psum_waddr = 0x0000 (output pixel [0,0], channel 0)
  mem.psum_wdata[0] = 32'h000004D0
  
  PSumBuf now contains:
    Address 0x0000: 0x000004D0  ← Channel 0 result
    Address 0x0001: 0xXXXXXXXX  ← Channel 1 result (computed in parallel)
    ...
    Address 0x000F: 0xXXXXXXXX  ← Channel 15 result

══════════════════════════════════════════════════════════════

Cycle 182: Post-processing for pixel [0,0]
  Batch Norm parameters (loaded earlier):
    mean[0]  = 100
    var[0]   = 400  (stddev = 20)
    gamma[0] = 2
    beta[0]  = 10
  
  Computation:
    normalized = (1,232 - 100) / 20 = 56.6
    scaled     = 56.6 × 2 = 113.2
    shifted    = 113.2 + 10 = 123.2
  
  Swish activation:
    sigmoid(123.2) ≈ 1.0 (very large input)
    swish = 123.2 × 1.0 = 123.2
  
  Quantize to INT8:
    output[0,0,0] = 123 (8-bit)

Cycle 196: Writeback
  DMA reads PSumBuf[0x0000] = 123 (INT8)
  DMA writes to DRAM[0x8300_0000] = 123
  
══════════════════════════════════════════════════════════════

Final Result:
  Output pixel [0,0], channel 0 = 123
  This value now in DRAM, ready for next layer!
```

---

## Memory Management

### **Buffer Allocation**

```
Memory Subsystem (160KB total, 16 banks × 32-bit):

┌─────────────────────────────────────────────────────────────┐
│ ActBufA: 0x0000 - 0x1FFF  (32KB)                            │
│   • Stores input activations for current tile               │
│   • Ping buffer                                             │
├─────────────────────────────────────────────────────────────┤
│ ActBufB: 0x2000 - 0x3FFF  (32KB)                            │
│   • Stores input activations for next tile (background)     │
│   • Pong buffer                                             │
├─────────────────────────────────────────────────────────────┤
│ WgtBuf:  0x4000 - 0x5FFF  (32KB)                            │
│   • Stores layer weights (reused for all tiles)             │
│   • Loaded once per layer                                   │
├─────────────────────────────────────────────────────────────┤
│ PSumBuf: 0x6000 - 0x9FFF  (64KB)                            │
│   • Stores partial sums / outputs                           │
│   • Supports accumulation for multi-tile C_in               │
└─────────────────────────────────────────────────────────────┘
```

### **Address Mapping Example**

```
Storing 64× 8-bit elements (e.g., 8×8 output tile):

Element layout:
  [E0, E1, E2, ..., E63]

Physical storage (16 banks × 32-bit):
  Bank  0: [E3  | E2  | E1  | E0 ]  ← Word 0
  Bank  1: [E7  | E6  | E5  | E4 ]  ← Word 0
  Bank  2: [E11 | E10 | E9  | E8 ]  ← Word 0
  ...
  Bank 15: [E63 | E62 | E61 | E60]  ← Word 0

DMA write addresses (with bank interleaving):
  Element 0: addr[3:0]=0x0, addr[19:4]=0x000 → Bank 0, Word 0
  Element 1: addr[3:0]=0x0, addr[19:4]=0x000 → Bank 0, Word 0, byte 1
  Element 4: addr[3:0]=0x1, addr[19:4]=0x000 → Bank 1, Word 0
  ...

Parallel read (1 cycle):
  AGU issues read address: 0x0000
  All 16 banks read word 0 simultaneously
  Unpacker extracts 64× 8-bit elements
  → Full 8×8 tile available in 1 cycle!
```

> **Note:** The triangular structure uses fewer registers than a full rectangular delay buffer. For N=16, this saves 120 vs. 136 registers.

### **Ping-Pong Timeline**

```
Time →
  0    100   200   300   400   500   600   700   800
  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
  
Tile 0:
  Load ActBufA ████░░░░░░░░
                 Compute from ActBufA ████████░░
                                        Post ██
                                          WB ██
  
Tile 1:
          Load ActBufB ████░░░░░░░░
                         Compute from ActBufB ████████░░
                                                Post ██
                                                  WB ██
  
Tile 2:
                  Load ActBufA ████░░░░░░░░
                                 Compute from ActBufA ████████░░
                                                        Post ██
                                                          WB ██

Legend:
  ████ Active operation
  ░░░░ Idle / waiting
```

> **Note:** Tile 1 compute overlaps with Tile 2 load, achieving estimated 50% latency reduction compared to serial execution.

---

## Timing Diagrams

> **⚠️ Important:** The timing diagrams below show the **intended design behavior** with idealized cycle counts. Real timing will differ due to handshaking delays, arbitration, stalls, and implementation-specific details. These are theoretical estimates for understanding the design architecture.

### **Idealized Single Tile Processing Timeline**

```
Cycle:  0      50     100    150    200    250
        │      │      │      │      │      │
        ▼      ▼      ▼      ▼      ▼      ▼
        
FSM:   IDLE→WT_LD→IN_LD→AGU_S→COMP→POST→WB→IDLE
       
Signals:
dma_start   ──┐    ┌────┐              ┌──
              └────┘    └──────────────┘

dma_done    ──────┐    ┌───────────────┐──┌─
                  └────┘               └──┘

agu_start   ──────────────┐              
                          └──────────────────

agu_ready   ──────────────┐  ┌────────────────
                          └──┘

tile_req    ─────────────────┐┌┐┌┐┌┐┌┐┌┐┌┐
                             └┘└┘└┘└┘└┘└┘

data_valid  ──────────────────┐┌┐┌┐┌┐┌┐┌┐┌
                              └┘└┘└┘└┘└┘└┘

psum_we     ───────────────────┐┌┐┌┐┌┐┌┐┐
                               └┘└┘└┘└┘└┘

irq         ────────────────────────────┐──
                                        └──

Annotations:
  [0-10]    Weight DMA load
  [10-40]   Weight SA load
  [40-70]   Input DMA load
  [70-75]   AGU setup
  [75-150]  Compute (75 cycles for 8×8 tile)
  [150-170] Post-processing
  [170-200] Writeback DMA
  [200]     Done, IRQ asserted
```

---

## Debug Checklist

### **Common Issues and Solutions**

#### **Issue 1: No Output Data**
```
Symptoms:
  - Computation completes but output buffer empty
  - PSumBuf reads as all zeros
  
Debug steps:
  1. Check dma_we signal during weight/input load
     → Should pulse for each write
  
  2. Check memory subsystem write enables
     → mem.actbuf_we, mem.wgtbuf_we should assert
  
  3. Check AGU address generation
     → Addresses should be within valid range
  
  4. Check SA data inputs
     → data_to_sa_act, data_to_sa_wgt should be non-zero
  
  5. Check psum writeback
     → mem.psum_we should assert during compute
  
Likely cause:
  - Data width mismatch (32-bit memory → 8-bit SA)
  - Missing unpacking logic
```

#### **Issue 2: Incorrect Results**
```
Symptoms:
  - Output values don't match golden reference
  - Garbage data or constant values
  
Debug steps:
  1. Dump memory contents after weight load
     → Compare with expected weight values
  
  2. Check AGU stride/padding calculation
     → Print addrA, addrB for first few iterations
  
  3. Verify SA configuration
     → sa_type should match layer dimensions
  
  4. Check post-processing parameters
     → BN mean/var/gamma/beta loaded correctly?
  
  5. Verify data types
     → INT8 activations, INT8 weights, INT32 psums
  
Likely cause:
  - Wrong AGU operation mode (depthwise vs. regular)
  - Incorrect stride or padding
  - BN parameters not loaded
```

#### **Issue 3: Hang / Timeout**
```
Symptoms:
  - FSM stuck in COMPUTE state
  - tile_done never asserts
  - Simulation runs indefinitely
  
Debug steps:
  1. Check AGU tile_done signal
     → Should assert after all pixels processed
  
  2. Check DMA handshake
     → arready, rready, awready, wready all working?
  
  3. Check for deadlock in memory subsystem
     → Read/write ports conflicting?
  
  4. Verify descriptor fields
     → tile_h, tile_w, c_in reasonable values?
  
Likely cause:
  - AGU tile counter overflow
  - DMA waiting for response that never comes
  - Descriptor length field incorrect
```

#### **Issue 4: AXI Protocol Violation**
```
Symptoms:
  - DRAM model throws error
  - "WDATA before AWVALID" or similar
  
Debug steps:
  1. Check AXI transaction ordering
     → AWVALID before WVALID
     → ARVALID before RREADY
  
  2. Verify address alignment
     → Addresses should be 4-byte aligned for 32-bit
  
  3. Check burst length
     → arlen/awlen match actual data transfers
  
Likely cause:
  - DMA wrapper state machine bug
  - Missing handshake synchronization
```

---

## Performance Analysis

> **⚠️ Critical Note:** All numbers below are **theoretical estimates based on idealized assumptions**. These calculations assume:
> - Perfect ping-pong overlap (no gaps, no stalls)
> - No arbitration delays or memory conflicts
> - Instant handshaking and zero control overhead
> - Ideal DMA burst efficiency
> 
> **Real performance will be lower** until verified through simulation and FPGA testing. These estimates demonstrate our design methodology, not measured results.

### **Estimated Latency Breakdown (Single 8×8×16 Tile)**

| Phase | Estimated Cycles | Percentage |
|-------|------------------|------------|
| Weight Load (once per layer) | ~50 | 0.2% |
| Input DMA Load | ~35 | 17% |
| AGU Setup | ~5 | 2% |
| Compute (8×8 output pixels) | ~80 | 40% |
| Post-Processing | ~20 | 10% |
| Writeback DMA | ~60 | 30% |
| **Total (idealized)** | **~250** | **100%** |

**With Ping-Pong Optimization (if perfect overlap achieved)**:
- Input load could overlap with previous tile compute
- Effective cycles per tile: **~160 cycles** (best-case 36% reduction)
- *Actual overlap depends on control logic correctness and memory availability*

### **Theoretical Throughput Calculation**

```
Assumptions (idealized):
  - 8×8×16 output tile
  - 3×3×16 convolution
  - ~160 cycles per tile (assuming perfect ping-pong)
  - 250 MHz clock
  - 16×16 systolic array (256 MACs/cycle theoretical)
  - 16-bank memory (64 elem/cycle read capacity)

Estimation Methodology:
  MACs per tile    = 8 × 8 × 16 × 3 × 3 × 16 = 147,456 MACs
  Time per tile    = 160 / 250MHz = 640 ns (idealized)
  Throughput       = 147,456 / 640ns ≈ 230 GOPS (best-case)
  
  SA theoretical peak (16×16×16=4096 MACs) = 4,096 × 250MHz = 1 TOPS
  Estimated efficiency = 230 / 1000 ≈ 23%
  
Expected efficiency losses due to:
  - Partial SA utilization (only 16 of 64 cols used in this config)
  - Memory bandwidth constraints
  - Pipeline bubbles during setup phases
  - Control overhead (FSM transitions, handshaking)
  
NOTE: 23% is optimistic - real efficiency likely 10-20% without tuning.
```

### **Estimated Memory Bandwidth Usage**

```
Input bandwidth (rough estimate):
  18×18×3×1 byte = 972 bytes per tile
  972 bytes / 400ns ≈ 2.43 GB/s (assuming perfect burst)
  
Weight bandwidth (amortized, idealized):
  Loaded once for 256 tiles
  432 bytes / (256 × 400ns) ≈ 4.2 MB/s
  
Output bandwidth (rough estimate):
  8×8×16×4 bytes = 4,096 bytes per tile
  4,096 / 400ns ≈ 10.24 GB/s (assuming perfect burst)
  
  Total ≈ 8 GB/s (peak estimate, simultaneous read+write)
  
IMPORTANT: These assume perfect AXI bursts with no stalls. Real bandwidth 
will be lower due to bus arbitration, setup cycles, and non-ideal burst patterns.
  
AXI interface: 64-bit @ 250MHz = 2 GB/s per direction
  → Bidirectional: 4 GB/s total

With ping-pong buffering:
  - Most loads overlap with compute (parallel)
  - Average bandwidth need: ~2.5 GB/s
  - Utilization: 2.5 / 4 = ~63%
```

> **Note:** The 64-bit @ 250 MHz AXI interface provides sufficient bandwidth using standard DDR3/DDR4 width. Ping-pong buffering overlaps most transfers with computation.

---

## Full Network Flow Summary### **256×256×3 → 1×1×1000 Layer Sequence**

```
Layer  | Type      | Input        | Output       | Cycles  | DRAM R/W
──────────────────────────────────────────────────────────────────────
L01    | Conv3×3/2 | 256×256×3    | 128×128×16   | 64K     | 196KB / 256KB
L02    | Conv1×1   | 128×128×16   | 128×128×16   | 40K     | 256KB / 256KB
L03    | Conv3×3   | 128×128×16   | 128×128×16   | 80K     | 256KB / 256KB
L04    | Conv1×1/2 | 128×128×16   | 64×64×32     | 20K     | 256KB / 128KB
L05    | Conv3×3   | 64×64×32     | 64×64×32     | 30K     | 128KB / 128KB
...    | (repeat MV2 blocks)
L15    | Conv1×1   | 16×16×96     | 16×16×128    | 5K      | 24KB / 32KB
L16    | Transformer (Q,K,V)        | 16×16×128    | 3×      |
       |   Q=X×W_Q | 16×16×128    | 16×16×128    | 5K      | 32KB / 32KB
       |   K=X×W_K | 16×16×128    | 16×16×128    | 5K      | 32KB / 32KB
       |   V=X×W_V | 16×16×128    | 16×16×128    | 5K      | 32KB / 32KB
L17    | Attention | QK^T         | 256×256      | 10K     | 64KB / 64KB
L18    | Softmax   | 256×256      | 256×256      | 20K     | (on-chip)
L19    | Attn×V    | 256×256×128  | 16×16×128    | 15K     | 96KB / 32KB
L20    | Conv1×1   | 16×16×128    | 16×16×128    | 5K      | 32KB / 32KB
...    | (more layers)
L48    | AvgPool   | 4×4×160      | 1×1×160      | 100     | 2.5KB / 640B
L49    | FC        | 1×1×160      | 1×1×1000     | 200     | 640B / 4KB
──────────────────────────────────────────────────────────────────────
TOTAL  | ~50 layers|              |              | ~5M     | ~100MB
```

**Estimated Inference Time** (requires validation):
- 5M cycles @ 400 MHz = **~12.5 ms** (estimated)
- **~80 FPS** single-stream throughput (estimated)
- **< 5W power** (rough estimate, needs measurement)

---

## Key Takeaways

> **Design Decisions:** The following architectural choices are intended for achieving target performance.

1. **Ping-Pong Buffering**
   - 50% latency reduction by overlapping DMA with compute
   - Essential for bandwidth-bound layers

2. **Banking Enables Parallelism**
   - 16 banks × 32-bit → 64 elements/cycle
   - Matches SA input width (16×64 → 64 cols)
   - Single-cycle tile reads

3. **Descriptor-Driven Decouples SW/HW**
   - CPU writes descriptors, not cycle-level control
   - Hardware autonomously executes
   - Easy to extend with new layer types

4. **Accumulation Enables Large Channels**
   - C_in > 16 supported via partial sum accumulation
   - Memory read-modify-write for psum readback
   - Arbitrary channel depths

5. **Post-Processing On-Chip Saves Bandwidth**
   - BN/Swish/LN applied before writeback
   - Reduces quantization error
   - Eliminates extra DRAM round-trip

---

## References

1. **MobileViT Paper**: https://arxiv.org/abs/2110.02178
2. **Memory Banking**: "High-Bandwidth Memory Subsystems" - ISCA Tutorial
3. **Ping-Pong Buffering**: "Deep Learning Accelerators" - MIT 6.5930
4. **AXI Protocol**: ARM IHI0022E AMBA AXI Specification

---
