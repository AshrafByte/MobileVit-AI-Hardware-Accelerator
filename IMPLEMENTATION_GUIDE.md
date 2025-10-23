# MobileViT Hardware Accelerator - Implementation Guide

## 📋 Overview

This is a hardware accelerator design for MobileViT neural network inference. The design supports:
- **Convolutional layers** (regular 3×3, pointwise 1×1, depthwise 3×3)
- **Transformer attention blocks** (Q, K, V computation, Softmax, Attention matrix multiply)
- **MLP layers** with Swish activation
- **Normalization** (Batch Norm, Layer Norm)
- **Flexible tiling** for large feature maps that don't fit in on-chip memory

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     CPU (ARM/RISC-V)                            │
│                                                                  │
│  Writes descriptors  →  Reads status/results                    │
└────────────────────────┬────────────────────────────────────────┘
                         │ AXI4-Lite Slave
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               mobilevit_accelerator_top                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐        ┌──────────────────┐              │
│  │ Register File    │───────▶│ Global Controller│              │
│  │ (AXI Slave)      │◀───────│      (FSM)       │              │
│  │ • Descriptors    │        │                  │              │
│  │ • Control/Status │        └────────┬─────────┘              │
│  └──────────────────┘                 │                        │
│                                       │                        │
│         ┌─────────────────────────────┼────────────┐           │
│         │                             │            │           │
│         ▼                             ▼            ▼           │
│  ┌─────────────┐            ┌──────────────┐  ┌──────────┐    │
│  │ DMA Wrapper │◀──────────▶│   Memory     │  │   AGU    │    │
│  │ (AXI Master)│            │  Subsystem   │  │ (Address │    │
│  └──────┬──────┘            │              │  │   Gen)   │    │
│         │                   │ • ActBufA    │  └────┬─────┘    │
│         │ AXI4              │ • ActBufB    │       │          │
│         ▼                   │ • WgtBuf     │       │          │
│    ┌─────────┐              │ • PSumBuf    │       │          │
│    │  DRAM   │              │              │       │          │
│    └─────────┘              │ • Banking    │       │          │
│                             │ • Ping-Pong  │◀──────┘          │
│                             │ • Accumulate │                  │
│                             └───────┬──────┘                  │
│                                     │                         │
│                                     ▼                         │
│                      ┌──────────────────────────┐             │
│                      │  Lego Systolic Array     │             │
│                      │  • 16×64 / 32×32 / 64×16 │             │
│                      │  • Weight stationary     │             │
│                      │  • INT8 MAC              │             │
│                      └──────────┬───────────────┘             │
│                                 │                             │
│                                 ▼                             │
│                      ┌──────────────────────────┐             │
│                      │ Post-Processing Pipeline │             │
│                      │ • Batch Norm             │             │
│                      │ • Swish Activation       │             │
│                      │ • Layer Norm             │             │
│                      └──────────┬───────────────┘             │
│                                 │                             │
│                                 ▼                             │
│                         (Results written back to DRAM)        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

```
RTL/
├── mobilevit_accelerator_top.sv        ← Top-level integration
├── Control/
│   └── global_controller.sv            ← Main FSM orchestrator
├── DMA/
│   └── dma_wrapper.sv                  ← AXI DMA wrapper
├── AGU/
│   ├── AGU.sv                          ← Address generation unit (existing)
│   └── Components/
│       ├── ConvOffsetsAGU.sv           ← Offset generator (existing)
│       └── ConvTileIndicesGenerator.sv ← Tile iterator (existing)
├── Memory Subsystem/
│   └── memory_subsystem.sv             ← SRAM buffers (existing)
├── Lego SA/
│   ├── Lego_Systolic_Array.sv          ← Flexible systolic array (existing)
│   └── Components/                      ← PE, control units (existing)
├── Compute/
│   ├── sa_compute_unit.sv              ← SA wrapper with accumulation
│   └── post_processing_pipeline.sv     ← BN → Swish → LN
├── Batch Norm/
│   └── batch_norm.sv                   ← Batch normalization (existing)
├── Swish/
│   └── swish.sv                        ← Swish activation (existing)
└── Layer Norm/
    ├── layer_norm1.sv                  ← Layer norm variant 1 (existing)
    └── layer_norm2.sv                  ← Layer norm variant 2 (existing)

Include/
├── accelerator_pkg.sv                  ← Main package (imports all)
├── accelerator_common_pkg.sv           ← Common types, constants (UPDATED)
├── accelerator_matmul_pkg.sv           ← MatMul parameters (existing)
├── accelerator_norm_pkg.sv             ← Normalization params (existing)
└── accelerator_activation_pkg.sv       ← Activation params (existing)
```

---

## 🔄 Data Flow

### Phase 1: Initialization
1. **CPU writes descriptor** to register file via AXI Slave
2. **Descriptor contains**:
   - DRAM address for weights/activations
   - SRAM target buffer (ActBufA/B, WgtBuf, PSumBuf)
   - Tile dimensions (tile_h, tile_w, c_in)
   - Control flags (enable BN, Swish, LN, etc.)
3. **CPU writes START bit** in control register

### Phase 2: Weight Loading
1. **Global Controller** sends DMA read request
2. **DMA Wrapper** fetches weights from DRAM (AXI Master)
3. **Weights written** to WgtBuf in Memory Subsystem
4. **Controller waits** for DMA done signal

### Phase 3: Activation Loading (Ping-Pong)
1. **Controller** sends DMA read request for first tile
2. **DMA loads** activations into **ActBufA** (ping)
3. While waiting, **Controller clears PSumBuf**

### Phase 4: Computation (Tile 0)
1. **Controller** signals AGU to start tile iteration
2. **AGU** generates addresses for:
   - Activations: `baseA + tile_offset`
   - Weights: `baseB + weight_offset`
   - Outputs: `baseC + output_offset`
3. **Memory Subsystem** reads data from ActBufA and WgtBuf
4. **Data flows** into Lego Systolic Array
5. **SA computes** matrix multiply: `Out = Act × Weight`
6. **Partial sums** written back to PSumBuf

### Phase 5: Overlap Next Tile (Ping-Pong)
1. **While SA computes tile 0**, DMA loads tile 1 into **ActBufB** (pong)
2. **When SA finishes tile 0**:
   - Switch to ActBufB for compute
   - Load tile 2 into ActBufA
3. **Repeat** until all tiles processed

### Phase 6: Accumulation (Multi-Tile C_in)
- If `c_in > 16` (SA can process 16 channels at once):
  1. First tile: Fresh write to PSumBuf
  2. Subsequent tiles: **Accumulation mode**
     - Read existing psum from PSumBuf
     - Add new partial sum from SA
     - Write back accumulated result

### Phase 7: Post-Processing
1. **Read psums** from PSumBuf
2. **Apply Batch Norm** (if enabled): `y = (x - mean) / sqrt(var) * gamma + beta`
3. **Apply Swish** (if enabled): `y = x * sigmoid(x)`
4. **Apply Layer Norm** (if enabled): normalize across sequence
5. **Write results** back to PSumBuf

### Phase 8: Writeback
1. **Controller** sends DMA write request
2. **DMA reads** results from PSumBuf
3. **DMA writes** results to DRAM (destination address from descriptor)
4. **Controller** asserts **IRQ** (interrupt) signal
5. **CPU reads** status register, gets results from DRAM

---

## 🎛️ Register Map (AXI Slave Interface)

| Offset | Name            | Access | Description |
|--------|-----------------|--------|-------------|
| 0x00   | CONTROL         | RW     | [0]=start, [1]=soft_reset |
| 0x04   | STATUS          | RO     | [0]=busy, [1]=done, [2]=error |
| 0x10   | DESC_DATA[0]    | WO     | Descriptor bits [31:0] |
| 0x14   | DESC_DATA[1]    | WO     | Descriptor bits [63:32] |
| 0x18   | DESC_DATA[2]    | WO     | Descriptor bits [95:64] |
| 0x1C   | DESC_DATA[3]    | WO     | Descriptor bits [127:96] |
| 0x20   | DESC_DATA[4]    | WO     | Descriptor bits [159:128] |
| 0x24   | DESC_DATA[5]    | WO     | Descriptor bits [191:160] |
| 0x28   | DESC_DATA[6]    | WO     | Descriptor bits [223:192] |
| 0x2C   | DESC_DATA[7]    | WO     | Descriptor bits [255:224] |
| 0x30   | DESC_PUSH       | WO     | Write 1 to push descriptor |
| 0x34   | TILE_COUNTER    | RO     | Current tile being processed |
| 0x38   | CYCLE_COUNTER   | RO     | Cycle count (for profiling) |

---

## 📝 Descriptor Format (256 bits)

```
Bits       Field          Description
────────────────────────────────────────────────────────────────
[255:224]  dram_addr      Source/destination DRAM address
[223:208]  sram_addr      Target SRAM buffer offset
[207:192]  length         Transfer length in bytes
[191:176]  stride         Stride for 2D tensor transfers
[175:64]   reserved       Reserved for future use
[63:48]    tile_h         Tile height (output rows)
[47:32]    tile_w         Tile width (output cols)
[31:16]    c_in           Input channels for this tile
[15:8]     flags          Control flags (see below)
[7:0]      reserved       Reserved
```

### Flags Bit Definition
```
Bit  Name              Description
────────────────────────────────────────────────────────────────
0    IS_WEIGHT         1=weight data, 0=activation data
1    IS_LAST_TILE      1=last tile in sequence
2    ENABLE_BN         1=enable batch normalization
3    ENABLE_SWISH      1=enable Swish activation
4    ENABLE_LN         1=enable layer normalization
5    TRANSPOSE         1=transpose weights (for K^T in attention)
6    ACCUMULATE        1=accumulate to existing partial sums
7    WRITEBACK         1=writeback result to DRAM after compute
```

---

## 💻 Software Usage Example (Pseudocode)

```c
// 1. Initialize accelerator
write_reg(CONTROL, 0x02);  // Soft reset
write_reg(CONTROL, 0x00);  // Clear reset

// 2. Prepare descriptor for weight loading
descriptor_t desc_weight = {
    .dram_addr = 0x8000_0000,      // Weight location in DRAM
    .sram_addr = 0x0000,            // WgtBuf offset
    .length    = 3 * 3 * 16 * 16,  // 3×3 kernel, 16 in, 16 out
    .stride    = 0,
    .tile_h    = 16,
    .tile_w    = 16,
    .c_in      = 16,
    .flags     = 0x01               // IS_WEIGHT flag
};

// 3. Write descriptor to registers
write_reg(DESC_DATA[0], desc_weight.fields[0]);
write_reg(DESC_DATA[1], desc_weight.fields[1]);
// ... write all 8 words
write_reg(DESC_PUSH, 0x01);  // Push descriptor to FIFO

// 4. Prepare descriptor for activation loading
descriptor_t desc_act = {
    .dram_addr = 0x9000_0000,      // Activation location
    .sram_addr = 0x0000,            // ActBufA offset
    .length    = 16 * 16 * 16,     // 16×16 tile, 16 channels
    .tile_h    = 16,
    .tile_w    = 16,
    .c_in      = 16,
    .flags     = 0x8E               // WRITEBACK | ENABLE_LN | ENABLE_SWISH | ENABLE_BN
};

write_reg(DESC_DATA[0], desc_act.fields[0]);
// ... write all fields
write_reg(DESC_PUSH, 0x01);

// 5. Start processing
write_reg(CONTROL, 0x01);  // Set START bit

// 6. Wait for completion (polling or interrupt)
while (!(read_reg(STATUS) & 0x02)) {
    // Wait for DONE bit
}

// 7. Check for errors
if (read_reg(STATUS) & 0x04) {
    printf("Error occurred!\n");
} else {
    printf("Processing complete!\n");
    printf("Tiles processed: %d\n", read_reg(TILE_COUNTER));
    printf("Cycles: %d\n", read_reg(CYCLE_COUNTER));
}

// 8. Read results from DRAM at desc_act.dram_addr
```

---

## 🔧 Configuration Parameters

### Memory Sizes (accelerator_common_pkg.sv)
```systemverilog
parameter ACTBUF_DEPTH  = 8192;   // 32KB (8K × 32-bit words)
parameter WGTBUF_DEPTH  = 8192;   // 32KB
parameter PSUMBUF_DEPTH = 16384;  // 64KB
parameter NUM_BANKS     = 4;       // Memory banks for parallel access
```

### Systolic Array Configuration
```systemverilog
// Lego SA supports:
SA_TYPE_16X64 = 2'b00  // 16 rows × 64 cols
SA_TYPE_32X32 = 2'b01  // 32 rows × 32 cols
SA_TYPE_64X16 = 2'b10  // 64 rows × 16 cols
```

### AGU Operation Modes
```systemverilog
OP_REGULAR_CONV = 2'b00  // Standard convolution (3×3, 5×5, etc.)
OP_POINTWISE    = 2'b01  // 1×1 convolution
OP_DEPTHWISE    = 2'b10  // Depthwise convolution
OP_MATMUL       = 2'b11  // Matrix multiply (for attention)
```

---

## 🧪 Testing Strategy

### Unit Tests (to be implemented)
1. **Memory Subsystem Test**
   - Write to ActBufA, read back
   - Ping-pong swap
   - Accumulation mode

2. **DMA Wrapper Test**
   - Read from DRAM model → SRAM
   - Write from SRAM → DRAM model
   - Burst transfers

3. **AGU Test**
   - Generate addresses for 3×3 conv
   - Generate addresses for matmul
   - Tile iteration

4. **SA Compute Unit Test**
   - Single tile multiply
   - Multi-tile accumulation

5. **Global Controller Test**
   - FSM state transitions
   - Ping-pong orchestration

### Integration Tests
1. **End-to-end Conv Layer**
   - Load weights, load acts, compute, writeback
   - Verify outputs match golden reference

2. **Multi-Tile Conv**
   - C_in = 64 (requires 4 tiles of 16)
   - Verify accumulation works correctly

3. **Transformer Attention Block**
   - Q, K, V generation (3× MatMul)
   - Attention = Softmax(QK^T)V

---

## 📊 Performance Targets

| Metric                | Target Value |
|-----------------------|--------------|
| Clock Frequency       | 400 MHz      |
| SA Throughput         | 256 MAC/cycle (16×16) |
| Peak Compute          | 102.4 GOPS (INT8) |
| Memory Bandwidth      | 16 GB/s (DMA) |
| Latency (Conv 3×3)    | ~500 cycles |
| Power (typical)       | < 5W |

---

## 🚀 Next Steps

### Immediate (Week 1-2)
- [ ] Implement testbench for `mobilevit_accelerator_top`
- [ ] Create DRAM model for simulation
- [ ] Write test vectors for Conv layer
- [ ] Simulate single Conv layer end-to-end

### Short-term (Week 3-4)
- [ ] Implement Python descriptor generator
- [ ] Add multi-tile support verification
- [ ] Test all post-processing modes
- [ ] Measure cycle counts and bandwidth

### Medium-term (Week 5-8)
- [ ] Add Softmax unit for attention
- [ ] Implement full transformer block test
- [ ] Optimize memory subsystem (true banking)
- [ ] Add weight compression support
- [ ] Create Vivado project, synthesize to FPGA

### Long-term (Week 9-12)
- [ ] Run full MobileViT-S inference
- [ ] Compare accuracy with PyTorch model
- [ ] Optimize timing for 400 MHz
- [ ] Power analysis and optimization
- [ ] Write comprehensive documentation

---

## 📚 References

1. **MobileViT Paper**: https://arxiv.org/abs/2110.02178
2. **Systolic Array**: "Why Systolic Architectures?" - H.T. Kung, 1982
3. **Xilinx AXI DMA**: PG021 - AXI DMA v7.1 Product Guide
4. **AMBA AXI4 Protocol**: ARM IHI0022E Specification

---

## ✅ Design Status

| Component                | Status | Notes |
|--------------------------|--------|-------|
| Global Controller        | ✅ Done | FSM with ping-pong support |
| DMA Wrapper              | ✅ Done | AXI4 interface |
| Memory Subsystem         | ✅ Done | Needs true banking implementation |
| AGU                      | ✅ Done | Existing module (verified) |
| Lego Systolic Array      | ✅ Done | Existing module (verified) |
| SA Compute Unit          | ✅ Done | Wrapper with accumulation |
| Post-Processing Pipeline | ⚠️ Partial | Needs actual BN/LN instantiation |
| Top-Level Integration    | ✅ Done | All modules connected |
| Testbench                | ❌ TODO | Critical next step |
| Python Tiler             | ❌ TODO | Descriptor generator |
| Vivado Project           | ❌ TODO | Synthesis target |

---

## 💡 Design Decisions

1. **Ping-Pong Buffering**: Overlap DMA with compute to hide memory latency
2. **Accumulation in Memory**: Support large C_in by accumulating partial sums
3. **Descriptor-Driven**: CPU configures via descriptors, not per-cycle control
4. **Flexible SA**: Lego SA supports different configs for different layers
5. **Modular Post-Processing**: Each stage (BN, Swish, LN) can be bypassed
6. **AXI Standard Interfaces**: Industry-standard for easy SoC integration

---