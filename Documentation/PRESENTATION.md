# MobileViT Hardware Accelerator
## Design Presentation

**Target**: MobileViT-XXS Neural Network Inference Acceleration

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Design Objectives](#2-design-objectives)
3. [Architecture Overview](#3-architecture-overview)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Performance Analysis](#5-performance-analysis)
6. [Technical Implementation](#6-technical-implementation)
7. [Verification Strategy](#7-verification-strategy)
8. [Results & Achievements](#8-results--achievements)
9. [Limitations & Future Work](#9-limitations--future-work)
10. [Conclusion](#10-conclusion)

---

## 1. Executive Summary

### **What We Built**

A hardware accelerator design for **MobileViT-XXS** neural network inference, featuring:

- **11 RTL modules** (~8,000 lines of SystemVerilog)
- **Descriptor-driven architecture** (inspired by industry approaches)
- **16-bank memory subsystem** (160 KB SRAM, FPGA-optimized)
- **Systolic array compute engine** (16×16 PEs, Lego reconfigurable)
- **Integrated post-processing** (BN → Swish → LayerNorm pipeline)

### **Design Targets**

| Metric | Target | Estimated (Design Calculation) | Notes |
|--------|--------|-------------------------------|-------|
| **Latency** | <15 ms | **~10.5 ms** | Idealized cycle count ÷ clock freq |
| **Peak Throughput** | >60 GOPS | **64 GOPS** | 256 MACs × 250 MHz (theoretical peak) |
| **FPS** | >60 FPS | **~95 FPS** | Derived from estimated latency |
| **Memory** | <200 KB | **160 KB** | Design specification |
| **Hardware Coverage** | Most MobileViT ops | **~85% HW** | Softmax, Residual in SW |
| **Bandwidth** | Sufficient | **~2.5 GB/s avg** | Rough estimate from data movement |

**Important Notes**:
- All numbers are **theoretical estimates** from design calculations, not measurements
- Cycle counts assume **ideal conditions** (no stalls, no conflicts, perfect scheduling)
- **Actual performance requires** FPGA synthesis, implementation, and testing
- These estimates serve as design targets for future implementation

---

## 2. Design Objectives

### **Primary Goal**

Design a hardware accelerator capable of executing the complete **MobileViT-XXS** neural network for efficient image classification on resource-constrained edge devices.

### **Intended Applications**

- Edge computing (IoT cameras, smart sensors)
- Embedded systems (robotics, drones)
- Real-time vision tasks (object detection, classification)

**Note**: These are potential applications based on the design targets. Actual deployment would require further optimization and validation.

### **Design Constraints**

1. **Performance**: Target >60 GOPS, <15 ms latency (typical for edge AI)
2. **Power**: Aim for <5W (suitable for edge deployment)
3. **Area**: Design to fit in mid-range FPGA (Xilinx Zynq UltraScale+)
4. **Flexibility**: Support major MobileViT layer types
5. **Bandwidth**: Use standard 64-bit DDR interface

### **Why MobileViT?**

- **State-of-the-art**: Combines CNNs and Transformers
- **Efficient**: 2-3× fewer parameters than traditional ViT
- **Accurate**: Competitive accuracy on ImageNet
- **Challenging**: Tests both convolution and attention acceleration

---

## 3. Architecture Overview

### **Block Diagram**

```
┌─────────────────────────────────────────────────────────────────┐
│                    MobileViT Accelerator Top                    │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────┐       │
│  │   AXI Host   │◄──►│     Global Controller (FSM)      │       │
│  │  Interface   │    │  • 16 states                     │       │
│  │ • Registers  │    │  • Ping-pong orchestration       │       │
│  │ • Descriptors│    │  • Layer sequencing              │       │
│  └──────────────┘    └──────────────────────────────────┘       │
│         │                          │                            │
│         ▼                          ▼                            │
│  ┌──────────────┐    ┌─────────────────────────────────────┐    │
│  │  DMA Wrapper │◄──►│    Memory Subsystem (160 KB)        │    │
│  │  AXI Master  │    │  • ActBufA/B (32KB each, ping-pong) │    │
│  │  64-bit bus  │    │  • WgtBuf (32KB)                    │    │
│  │              │    │  • PSumBuf (64KB)                   │    │
│  └──────────────┘    │  • 16-bank × 32-bit architecture    │    │
│                      └─────────────────────────────────────┘    │
│                                   │                             │
│                                   ▼                             │
│              ┌────────────────────────────────────┐             │
│              │    Systolic Array (16×16 PEs)      │             │
│              │  • Lego modes: 16×64, 32×32, 64×16 │             │
│              │  • Weight-stationary dataflow      │             │
│              │  • INT8 MACs, 409.6 GOPS peak      │             │
│              └────────────────────────────────────┘             │
│                                   │                             │
│                                   ▼                             │
│         ┌───────────────────────────────────────┐               │
│         │   Post-Processing Pipeline (3 stages) │               │
│         │   Batch Norm → Swish → Layer Norm     │               │
│         │   (16 elements/cycle throughput)      │               │
│         └───────────────────────────────────────┘               │
│                                   │                             │
│                                   ▼                             │
│                      Writeback to DRAM via DMA                  │
└─────────────────────────────────────────────────────────────────┘
```

### **Key Components**

| Component | Function | Details |
|-----------|----------|---------|
| **Global Controller** | FSM orchestration | 16 states, ping-pong control, layer sequencing |
| **DMA Wrapper** | External memory interface | AXI4 Master, 64-bit @ 250 MHz, 2 GB/s per direction |
| **Memory Subsystem** | On-chip SRAM | 160 KB, 16-bank × 32-bit, ping-pong buffers |
| **Systolic Array** | Compute engine | 16×16 PEs, Lego modes, weight-stationary |
| **Post-Processing** | Activation & normalization | BN, Swish, LayerNorm pipeline |
| **AGU** | Address generation | Tile indices, offsets, bank selection |

---

## 4. Key Design Decisions

### **Decision 1: Descriptor-Driven Architecture**

**Rationale**:
- Decouples software from hardware timing
- Hardware can execute autonomously after descriptor configuration
- Inspired by industry approaches (ARM Mali, NVIDIA DLA - from literature study)
- Simplifies software interface design

**Implementation**:
```
Descriptor Format (256-bit):
┌──────────────────────────────────────────────────────┐
│ Layer Type | Dimensions | Addresses | Flags          │
│  [7:0]    | [95:8]     | [223:96]  | [255:224]       │
└──────────────────────────────────────────────────────┘

CPU writes descriptor → Hardware reads → Executes → Interrupt
```

**Benefits**:
- Reduces CPU involvement once descriptor is configured
- Enables potential layer chaining in future
- Extensible for new layer types

---

### **Decision 2: 16-Bank Memory Subsystem**

**Rationale**:
- **32-bit banks** match industry standards (AXI/AHB/APB)
- **16 banks** provide 64 elements/cycle (matches SA width)
- **FPGA-friendly**: Only 16 read ports (vs 64 in alternative designs)
- **Variable activation**: 16/8/4 banks for different SA modes

**Architecture**:
```
┌──────────────────────────────────────────────────────┐
│  Bank 0   Bank 1   Bank 2   ...   Bank 15            │
│  32-bit   32-bit   32-bit   ...   32-bit             │
│ [E3|E2|  [E7|E6|  [E11|E10| ... [E63|E62|            │ 
│  E1|E0]   E5|E4]   E9|E8]        E61|E60]            │
└──────────────────────────────────────────────────────┘
    4 elem    4 elem    4 elem  ...    4 elem
         = 64 elements in 1 cycle (parallel read)
```

**Performance**:
- Designed for single-cycle tile reads for all SA configurations
- **Expected** throughput improvement over sequential access (pending verification)
- Efficient BRAM utilization (~40 blocks for 160 KB estimated)

---

### **Decision 3: Ping-Pong Buffering**

**Rationale**:
- Designed to overlap DMA load with SA computation
- **Intended** to reduce latency vs serial execution
- Essential for bandwidth-limited layers
- Standard technique in ML accelerators (from literature)

**Timeline**:
```
Without Ping-Pong (Sequential):
[Load Tile 0] → [Compute Tile 0] → [Load Tile 1] → [Compute Tile 1] → ...
      100 cycles      200 cycles       100 cycles      200 cycles
              Total: 300 cycles per tile

With Ping-Pong (Designed Behavior):
[Load Tile 0] → [Load Tile 1 | Compute Tile 0] → [Load Tile 2 | Compute Tile 1] → ...
   100 cycles         200 cycles                      200 cycles
              Expected: 200 cycles per tile (after initial load)
                        = 33% time savings (if overlap is perfect)
```

**Intended Benefits**:
- Hide DMA latency behind computation
- Increase effective compute utilization
- Reduce total inference latency

**Note**: Actual overlap depends on data dependencies and control logic correctness.

---

### **Decision 4: Multi-Tile Accumulation**

**Rationale**:
- Systolic array is **16-wide**, but MobileViT uses **16-384 channels**
- Need to process input channels in tiles and accumulate results
- Alternative: larger SA (expensive) or software stitching (slow)

**Strategy**:
```
For C_in = 48:
  Tile 0: Process channels [0:15]   → Fresh write to PSumBuf
  Tile 1: Process channels [16:31]  → Read PSumBuf, accumulate, write back
  Tile 2: Process channels [32:47]  → Read PSumBuf, accumulate, write back
  
  Result: Accumulated output for all 48 input channels
```

**Implementation**:
- FSM tracks tile index and accumulation mode
- Memory subsystem supports partial sum readback
- Post-processing applied only after final tile

**Benefits**:
- Design supports arbitrary channel depths
- No hardware oversizing needed
- Efficient use of on-chip buffers

---

### **Decision 5: Integrated Post-Processing**

**Rationale**:
- Apply **BN → Swish → LayerNorm** pipeline before writeback
- Intended to save DRAM round-trip (no need to read back, process in CPU, write again)
- Reduces quantization error (keeps higher precision internally until final output)
- Common practice in ML accelerators (from literature)

**Pipeline**:
```
SA Output (INT32) → Batch Norm → Swish → Layer Norm → INT8 → Writeback
      256 bits       16 elem     16 elem    16 elem    128 bits
                    (3-stage pipeline, throughput goal: 1 output per cycle after fill)
```

**Intended Benefits**:
- Eliminates up to 3 DRAM accesses per layer
- Maintains higher precision during computation
- Each stage can be bypassed via descriptor flags

---

## 5. Performance Analysis

**⚠️ Important**: All metrics below are **theoretical estimates** from hand calculations assuming ideal conditions. Actual performance requires FPGA implementation and measurement.

### **5.1 Latency Breakdown**

**Target Network**: MobileViT-XXS (256×256×3 input → 1×1×1000 output)

| Stage | Operations | Cycles (Estimated) | Time @ 250 MHz | % of Total |
|-------|-----------|---------------------|----------------|------------|
| **Stem** | Conv 3×3 | ~36,000 | ~0.14 ms | 1.4% |
| **Stage 1-2** | MV2 blocks ×4 | ~212,000 | ~0.85 ms | 8.1% |
| **Stage 3-5** | MV2 + MobileViT | ~2,345,000 | ~9.38 ms | 89.3% |
| **Classifier** | Global Pool + FC | ~11,300 | ~0.05 ms | 0.4% |
| **Software (Softmax)** | CPU fallback | ~21,872 | ~0.09 ms | 0.8% |
| **TOTAL** | ~150 operations | **~2,626,172** | **~10.5 ms** | **100%** |

**Methodology**:
- Cycle counts estimated from: (Output_H × Output_W × C_out × K_H × K_W × C_in) / (PE_array_width)
- Assumes **ideal** conditions: no stalls, perfect data availability, full SA utilization
- **Real cycle counts will be higher** due to: pipeline fills, bank conflicts, control overhead

**Key Observations**:
- ~89% of time in **transformer blocks** (expected for MobileViT architecture)
- Conv layers relatively fast due to high SA utilization potential
- Softmax overhead minimal if done in software (~0.8%)

**Estimated Throughput**: 10.5 ms per image ≈ **~95 FPS** (theoretical, assuming continuous feed)

---

### **5.2 Compute Efficiency**

**Per-Operation Efficiency** (Estimated SA utilization potential):

| Operation Type | Estimated SA Utilization | Notes |
|---------------|-------------------------|-------|
| **Conv 3×3** | 60-80% (goal) | High potential with good data reuse |
| **Conv 1×1** | 40-60% (goal) | Lower due to less data reuse |
| **Transformer MatMul** | 50-70% (goal) | Q×K^T, Attention×V operations |
| **Depthwise Conv** | 20-40% (goal) | Only 1 PE active per row |

**Overall Efficiency Estimate** (across all operations):

**Important Caveat**: Efficiency estimates below assume best-case scenarios. Real-world efficiency typically much lower due to memory stalls, control overhead, and partial tile fills.

| Batch Size | Estimated Average Efficiency | Notes |
|-----------|----------------------------|-------|
| **Batch = 1** | <10% realistic | Includes ALL cycles: compute, DMA, idle, control |
| **Batch = 4** | ~15-20% (estimate) | Better amortization of overhead |
| **Batch = 16** | ~30-40% (estimate) | Theoretical limit with good scheduling |

> **KEY INSIGHT**: The low average efficiency includes DMA transfers, normalization, software operations, and all idle cycles. Individual convolution operations can achieve higher efficiency during their active execution phase. These are rough estimates that require validation through actual implementation and profiling.

---

### **5.3 Memory Footprint**

**Buffer Allocation**:

| Buffer | Size | Purpose |
|--------|------|---------|
| **ActBufA** | 32 KB | Ping buffer for activations |
| **ActBufB** | 32 KB | Pong buffer for activations |
| **WgtBuf** | 32 KB | Weights (reused per layer) |
| **PSumBuf** | 64 KB | Partial sums (largest outputs) |
| **TOTAL** | **160 KB** | All layers fit with careful management |

**Worst-Case Layer** (Stage 3b MobileViT):
- Input: 48×32×32 = 49,152 elements = **48 KB**
- Weights: 64×48×1×1 = 3,072 elements = **3 KB**
- Output: 48×32×32 = 49,152 elements = **48 KB**
- Transformer Q/K/V: 256×256 = 65,536 elements = **64 KB**
- **Peak**: 163 KB (slightly over, managed with ping-pong)

**Conclusion**: All stages fit with careful buffer management and ping-pong strategy.

---

### **5.4 Bandwidth Analysis**

**Rough Estimation**:
```
Required Bandwidth (simplified calculation):
  Total data: ~150 MB transferred throughout network (estimated)
  Time: 2.626M cycles / 250 MHz = 10.5 ms
  Naive: 150 MB / 10.5 ms = 14.3 GB/s (way too high - unrealistic)
  
With Ping-Pong Parallelism (assumed):
  Loads can overlap with compute (if control logic works correctly)
  Only writebacks are fully serial
  Estimated effective need: ~2-3 GB/s average (rough guess)

Available Bandwidth:
  AXI: 64-bit @ 250 MHz = 2 GB/s per direction (read OR write)
  Bidirectional: 4 GB/s total

Estimated Utilization: 2.5 GB/s / 4 GB/s = ~60-65%
```

**Important Caveats**:
- These are **very rough estimates** based on simplified assumptions
- Actual bandwidth depends on:
  - Memory access patterns (sequential vs random)
  - Ping-pong overlap efficiency (depends on control logic correctness)
  - Burst efficiency on AXI bus
  - Bank conflicts in memory subsystem
- **Real bandwidth measurement** requires:
  - Functional simulation with waveforms
  - Monitoring AXI transaction rates
  - Profiling actual data transfer patterns

**Conclusion**: The 64-bit @ 250 MHz AXI interface (4 GB/s total) **should theoretically be sufficient** based on rough estimates, but this needs verification through simulation and actual measurement.

---

### **5.5 Power Estimation**

**Component-Level Breakdown** (rough estimates):

| Component | Power | Percentage |
|-----------|-------|------------|
| Systolic Array | 2.5W | 50% |
| Memory Subsystem | 1.0W | 20% |
| DMA & Interfaces | 0.8W | 16% |
| Control Logic | 0.4W | 8% |
| Post-Processing | 0.3W | 6% |
| **TOTAL** | **~5W** | **100%** |

**Notes**:
- These are rough estimates based on 28nm FPGA technology and typical power consumption patterns
- Actual power consumption can only be determined through FPGA synthesis and measurement
- Power analysis tools (Vivado Power Analyzer) will provide more accurate estimates post-synthesis

---

## 6. Technical Implementation

### **6.1 RTL Module Summary**

| Module | Lines of Code | Function | 
|--------|--------------|----------|
| `mobilevit_accelerator_top.sv` | 500 | Top-level integration | 
| `global_controller.sv` | 800 | FSM orchestration | 
| `dma_wrapper.sv` | 400 | AXI Master interface | 
| `memory_subsystem.sv` | 600 | 16-bank SRAM | 
| `Lego_Systolic_Array.sv` | 2000 | Compute engine | 
| `post_processing_pipeline.sv` | 300 | BN/Swish/LN | 
| `AGU.sv` | 700 | Address generation | 
| `batch_norm.sv` | 250 | Normalization | 
| `swish.sv` | 200 | Activation | 
| `layer_norm1.sv` / `layer_norm2.sv` | 400 | Normalization | 
| **TOTAL** | **~8,000** | 11 modules | 

---

### **6.2 Global Controller FSM**

**16 States**:

| State | Function | Next State(s) |
|-------|----------|---------------|
| **IDLE** | Wait for descriptor | DECODE |
| **DECODE** | Parse layer type | LOAD_ACT, LOAD_WGT |
| **LOAD_ACT** | DMA load activations | LOAD_WGT (if needed), COMPUTE |
| **LOAD_WGT** | DMA load weights | COMPUTE |
| **COMPUTE** | SA execution | ACCUMULATE, POST_PROC |
| **ACCUMULATE** | Multi-tile accumulation | COMPUTE (next tile), POST_PROC |
| **POST_PROC** | BN/Swish/LN pipeline | WRITEBACK |
| **WRITEBACK** | DMA write results | NEXT_TILE, DONE |
| **NEXT_TILE** | Tile loop control | LOAD_ACT |
| **DONE** | Layer complete | IDLE |
| **(6 more states for special cases)** | - | - |

**Key Features**:
- Ping-pong buffer management
- Multi-tile accumulation tracking
- Post-processing bypass control
- Error handling and timeout

---

### **6.3 Systolic Array Details**

**Base Configuration**: 16×16 PEs

**Lego Modes** (runtime reconfigurable):

| Mode | Array Size | Use Case | Efficiency |
|------|-----------|----------|------------|
| **Type 0** | 16×64 | Conv 3×3, large channels | 85-95% |
| **Type 1** | 32×32 | Balanced convolutions | 75-85% |
| **Type 2** | 64×16 | Conv 1×1, pointwise | 60-75% |

**Dataflow**: Weight-stationary
- Weights loaded once, reused for all spatial positions
- Activations stream through PEs
- Partial sums accumulate vertically

**Peak Performance**:
- 256 MACs/cycle (16×16)
- 250 MHz clock
- **64 GOPS** (INT8)

---

### **6.4 Memory Subsystem**

**16-Bank Architecture**:

```systemverilog
// Bank interleaving on DMA write
bank_id = addr[3:0];     // Lower 4 bits
word_addr = addr[19:4];  // Upper 16 bits

// Parallel read on AGU access
for (i = 0; i < num_banks_active; i++) begin
    data_out[i] = bank[i][addr];  // All banks read simultaneously
end

// Variable bank activation
case (sa_type)
    2'b00: num_banks = 16;  // 64 elements/cycle
    2'b01: num_banks = 8;   // 32 elements/cycle
    2'b10: num_banks = 4;   // 16 elements/cycle
endcase
```

**Benefits**:
- Single-cycle tile reads
- FPGA-optimized (16 BRAM ports)
- Power-efficient (only active banks read)

---

## 7. Verification Strategy

### **Four-Phase Approach**

#### **Phase 1: Unit Testing** 
- Individual module testbenches written
- Basic functional verification of each component
- Corner case testing (edge values, overflows)
- **Status**: Testbench code written, systematic testing in progress

#### **Phase 2: Integration Testing** (Planned)
- Single convolution layer end-to-end simulation
- Multi-tile accumulation verification
- Ping-pong buffer operation validation
- Full MobileViT block (MV2 + transformer) test

#### **Phase 3: System Testing** (Planned)
- Complete MobileViT-XXS network simulation
- Output comparison with PyTorch golden reference
- Actual cycle count measurement (vs estimates)
- Memory footprint verification

#### **Phase 4: FPGA Validation** (Future Work)
- Synthesize to Xilinx Zynq UltraScale+ (or available FPGA)
- Timing closure @ 250 MHz (realistic target for memory interface)
- Real performance measurement (actual GOPS, FPS, power)
- Bottleneck profiling and optimization

---

### **Golden Reference**

**Python Model**: `mobile-vit-acc3_official.py`
- PyTorch implementation of MobileViT-XXS
- Layer-by-layer output capture
- Bit-accurate comparison with hardware
- Validation dataset: ImageNet subset

---

## 8. Design Achievements & Status

### **8.1 Design Completeness**

| Category | Target | Status | Notes |
|----------|--------|--------|-------|
| **RTL Modules** | 11 | ✅ RTL Written | All modules coded |
| **Hardware Coverage** | Major MobileViT ops | ✅ ~85% HW | Softmax, Residual in SW |
| **Documentation** | Comprehensive | ✅ Complete | 7 docs, 15,000+ words |
| **Testbenches** | Unit tests | ⏳ In Progress | 11 testbenches written |
| **Integration Tests** | Full system | ⏳ Pending | Requires simulation |
| **FPGA Synthesis** | 250 MHz target | ⏳ Not Started | Future work |

---

## 9. Limitations & Future Work

### **9.1 Current Limitations**

| Limitation | Impact | Potential Solution | Priority |
|------------|--------|-------------------|----------|
| **Softmax in software** | Estimated +4.3 ms if moved to HW | Could add PWL Softmax unit | MEDIUM |
| **No hardware residual add** | SW stitching needed | Could add simple ALU | LOW |
| **Layer Norm single-unit** | Potential bottleneck | Could add 2nd parallel unit | LOW |
| **Global Pool needs workaround** | Extra control logic | Use SA with all-1s weights | LOW |
| **Verification incomplete** | Unproven functionality | Complete sim & FPGA testing | HIGH |

---

### **9.2 Future Enhancements**

#### **Next Steps for Completion**
1. **Functional Verification**
   - Run RTL simulations with test vectors
   - Validate against PyTorch golden reference
   - Measure actual vs estimated cycle counts
   - Debug and fix issues

2. **FPGA Implementation**
   - Synthesize design for target FPGA
   - Work on timing closure @ 250 MHz
   - Implement on-board testing infrastructure
   - Measure real performance

3. **Performance Optimization**
   - Profile actual bottlenecks (may differ from estimates)
   - Optimize critical paths based on synthesis results
   - Tune memory access patterns
   - Investigate batch processing benefits

#### **Possible Enhancements** (if time permits)
1. **Hardware Softmax**
   - Piecewise linear approximation (PWL)
   - Could potentially reduce CPU overhead

2. **Additional Layer Support**
   - Hardware residual addition
   - More normalization variants

3. **Model Scaling**
   - Support for MobileViT-S, MobileViT-XS
   - Would require memory capacity planning

---

## 10. Conclusion

### **Project Summary**

This graduation/research project presents a **hardware accelerator design** for the MobileViT-XXS neural network, focusing on RTL architecture and design methodology.

### **Key Design Contributions**

1. **Descriptor-Driven Architecture**: Clean hardware-software interface inspired by industrial ML accelerators
2. **16-Bank Memory Subsystem**: FPGA-optimized parallel memory access design
3. **Ping-Pong Buffering Strategy**: Designed for DMA/compute overlap to hide transfer latency
4. **Integrated Post-Processing**: On-chip normalization and activation fusion
5. **Comprehensive Documentation**: Detailed design rationale, performance analysis, and architectural decisions

---

### **Project Status & Outcomes**

**Completed Work**:
- ✅ **RTL Design**: 11 modules
- ✅ **Performance Modeling**: Cycle estimation and bandwidth analysis methodology
- ✅ **Testbench Structure**: Unit test framework for all modules

**Remaining Work**:
- ⏳ **Functional Verification**: RTL simulation and validation against golden reference
- ⏳ **FPGA Synthesis**: Timing closure and resource utilization
- ⏳ **Performance Measurement**: Actual cycle counts, throughput, and power
- ⏳ **Optimization**: Based on profiling real bottlenecks

### **Research Value**

This project demonstrates:
- **Architectural thinking** for ML accelerator design
- **Design methodology** from specification to RTL implementation
- **Performance estimation** techniques for hardware design
- **Documentation skills** for complex digital systems

### **Realistic Assessment**

**Strengths**:
- Comprehensive design with clear architectural decisions
- Well-documented rationale for each major choice
- Modular, potentially synthesizable RTL code
- Realistic targets for student FPGA project

**Limitations**:
- Performance numbers are **theoretical estimates only**
- No functional verification completed yet
- Actual performance may differ significantly from estimates
- Optimization opportunities only identified, not implemented

---

## Appendix: Quick Reference

### **Design Specifications**
- **Target Network**: MobileViT-XXS (256×256×3 → 1×1×1000)
- **Compute Engine**: 16×16 Systolic Array (INT8)
- **On-Chip Memory**: 160 KB SRAM (16-bank architecture)
- **External Interface**: 64-bit AXI @ 250 MHz target
- **Control**: Descriptor-driven (256-bit descriptors)

### **Estimated Performance (Theoretical)**
- **Latency**: ~10.5 ms @ 250 MHz (idealized cycle count)
- **Peak Throughput**: 64 GOPS (256 MACs × 250 MHz)
- **FPS**: ~95 FPS (continuous feed assumption)
- **Memory**: 160 KB on-chip designed capacity
- **Bandwidth**: ~2.5 GB/s average estimated need

**⚠️ Important**: These are design-time estimates assuming ideal conditions. Actual numbers require FPGA implementation and measurement.

### **RTL Module Summary**
- **16-bank memory**: 32-bit width per bank
- **Systolic array**: 16×16 PEs, Lego reconfigurable
- **Post-processing**: BN → Swish → LayerNorm (3-stage pipeline)
- **Interface**: AXI4 Master for external memory access



### **Documentation**
1. **EXECUTIVE_SUMMARY.md** - One-page overview
2. **DOCUMENTATION_GUIDE.md** - Key decisions and justification
3. **IMPLEMENTATION_GUIDE.md** - Complete architecture details
4. **MEMORY_BANKING_ARCHITECTURE.md** - 16-bank design
5. **COMPLETE_DATA_FLOW_GUIDE.md** - Cycle-accurate walkthrough
6. **OPERATIONS_TABLE_AND_HARDWARE_FLOW.md** - All 150+ operations
7. **READING_ORDER.md** - Documentation navigation guide

---

**End of Presentation**

Thank you for reviewing our MobileViT Hardware Accelerator design.

*For detailed technical information, please refer to the complete documentation package.*
