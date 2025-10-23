# MobileViT Hardware Accelerator - Executive Summary

**⚠️ Research Project Disclaimer**

This is a student graduation/research project focused on RTL design and architecture. The design has been documented and coded (~8,000 lines of SystemVerilog) but **not yet functionally verified, synthesized, or implemented on FPGA**. All performance metrics are theoretical estimates that require validation through actual implementation and testing.

---

## Project Objective

> **GOAL:** Design and implement a hardware accelerator architecture for MobileViT neural network inference.

Design a **hardware accelerator** for MobileViT neural network inference, supporting:
- Convolutional layers (regular, depthwise, pointwise)
- Transformer attention blocks (multi-head attention, MLP)
- Normalization layers (batch norm, layer norm)
- Design targets: >60 GOPS, <15 ms latency, <5W power (theoretical)

---

## What Was Designed

> **STATUS:** RTL design and architecture documentation completed. Verification and implementation pending.

### **1. RTL Design** (11 modules, ~8,000 lines)
- Top-level integration (`mobilevit_accelerator_top.sv`)
- Global FSM controller (16 states, ping-pong orchestration)
- DMA wrapper (AXI4 Master, 64-bit interface)
- Memory subsystem (160 KB SRAM, 16-bank architecture)
- Systolic array compute unit (16×16 PEs, INT8 MACs)
- Post-processing pipeline (BN → Swish → LayerNorm)
- Address generation unit (supports major layer types)

**Status**: RTL code written (~3,000 new + ~5,000 integrated)

### **2. Documentation** (7 documents, ~15,000 words)
- Architecture overview and design decisions
- Performance estimation methodology
- Memory subsystem detailed design
- Complete data flow analysis
- Operations mapping and coverage

### **3. Verification Framework** (In Progress)
- Top-level testbench structure (register access, descriptor interface)
- Unit testbenches for major modules (written, testing in progress)
- PyTorch golden reference model
- Verification plan (4-phase approach defined)

---

## Design Targets and Estimates

> **IMPORTANT:** All metrics below are theoretical estimates from design calculations. None have been validated through simulation or FPGA implementation.

| Metric | Design Target | Estimated (Theoretical) | Status |
|--------|--------------|------------------------|--------|
| **Clock Frequency** | 250 MHz | Not yet synthesized | Realistic target for FPGA memory interface |
| **Peak Throughput** | >60 GOPS | ~64 GOPS | 256 MACs × 250 MHz (theoretical maximum) |
| **Inference Latency** | <15 ms | ~10.5 ms | From idealized cycle estimates |
| **Throughput** | >60 FPS | ~95 FPS | Derived from latency estimate |
| **Memory Footprint** | <200 KB | 160 KB | Design specification |
| **Bandwidth Need** | Sufficient | ~2.5 GB/s avg | Rough estimate from data movement |
| **Power** | <5W | ~5W estimate | Very rough guess, needs measurement |
| **Hardware Coverage** | Major ops | ~85% HW | Softmax, Residual need CPU assist |

> **Note:** All performance estimates are based on simplified hand calculations assuming ideal conditions (no stalls, perfect ping-pong overlap, no handshaking delays). Actual performance will differ and requires:
> - Functional simulation to validate correctness
> - FPGA synthesis to verify timing closure
> - Real hardware measurement for actual performance
> 
> These numbers serve as design targets to guide future implementation work.

---

## Architecture Highlights

> **Design Features:** Five key design decisions aimed at optimizing performance and resource utilization.

### **Design Decisions**:

1. **Descriptor-Driven Control** (Inspired by industry approaches)
   - CPU configures 256-bit descriptors, hardware executes autonomously
   - Decouples software timing from hardware pipelining
   - Enables potential layer chaining

2. **16-Bank Memory Subsystem** (FPGA-optimized design)
   - 32-bit banks (standard width)
   - Variable bank activation (16/8/4 banks for different SA configs)
   - Designed for single-cycle tile reads (64 elements/cycle goal)

3. **Ping-Pong Buffering** (Latency hiding strategy)
   - ActBufA/ActBufB alternate: intended to load next while computing current
   - For bandwidth-limited layers
   - Designed to overlap DMA with computation

4. **Multi-Tile Accumulation** (Arbitrary channel depth support)
   - SA is 16-wide, but MobileViT uses 16-384 channels
   - First tile: fresh write, subsequent tiles: accumulate
   - Enables flexible layer sizes without hardware oversizing

5. **Integrated Post-Processing** (On-chip fusion approach)
   - BN→Swish→LayerNorm pipeline before writeback
   - Intended to save DRAM round-trips
   - Each stage bypassable via descriptor flags

---

## Technical Depth Demonstrated

### **System-Level**:
- Top-level architecture (AXI interfaces, clock/reset, memory hierarchy)
- Hardware/software interface (register map, descriptor format, interrupt)
- Performance modeling (cycle estimation methodology, bandwidth analysis)

### **Microarchitecture**:
- Datapath design (systolic array, post-processing pipeline)
- Control logic (16-state FSM, ping-pong orchestration)
- Memory management (banking strategy, tiling, accumulation)

### **RTL Implementation**:
- SystemVerilog design (~8,000 lines, packages, typedefs, enums)
- Modular structure (reusable components, parameterized modules)
- Following common practices (AXI protocol, descriptor-driven interface)

### **Verification Planning**:
- Testbench structure (AXI tasks, DRAM model, stimulus generation)
- Verification methodology (unit tests, integration tests, system tests)
- Golden reference (PyTorch model for output comparison)

---

## Performance Analysis Summary

> **IMPORTANT:** All cycle estimates below are theoretical calculations assuming ideal conditions (no stalls, perfect data availability). Actual performance requires simulation and hardware measurement.

**Estimated Full MobileViT-XXS Network** (256×256×3 input → 1×1×1000 output):

| Stage | Operations | Estimated Cycles | Est. Time @ 250 MHz | Notes |
|-------|-----------|------------------|---------------------|-------|
| Stem | Conv 3×3 | ~36,000 | ~0.14 ms | Hand-calculated |
| Stage 1-2 | MV2 blocks ×4 | ~212,000 | ~0.85 ms | Hand-calculated |
| Stage 3-5 | MV2 + MobileViT | ~2,345,000 | ~9.38 ms | Includes SW Softmax estimate |
| Classifier | Global Pool + FC | ~11,300 | ~0.05 ms | Hand-calculated |
| **TOTAL** | ~150 operations | **~2,626,172** | **~10.5 ms** | Theoretical ideal-case |

**Estimation Methodology**:
- Cycle counts derived from: (Output_size × Kernel_size × Channels) / PE_array_width
- Assumes NO stalls, NO conflicts, perfect data availability every cycle
- Does NOT account for: pipeline fills, bank conflicts, handshaking overhead, real-world delays

**Note**: These are best-case theoretical estimates. Real cycle counts will be higher. Simulation and measurement are needed for validation.

**Potential Optimization**: Hardware Softmax could theoretically reduce latency, but adds significant design complexity.

---

## Known Limitations & Future Work

> **Current Limitations:** Areas identified for future improvement and completion.

| Limitation | Impact | Potential Solution | Priority |
|------------|--------|-------------------|----------|
| **Softmax in software** | Adds CPU overhead | Could add PWL Softmax unit | MEDIUM |
| **No hardware residual add** | Need CPU for residuals | Could add simple ALU | LOW |
| **Layer Norm single-unit** | Potential bottleneck | Could add 2nd parallel unit | LOW |
| **Global Pool workaround** | Extra control complexity | Use SA with all-1s weights | LOW |
| **Verification incomplete** | Unproven functionality | Complete simulation & FPGA testing | **HIGH** |

**Current Status**: Design and RTL complete, functional verification pending

---

## Verification Status

| Test Level | Status | Coverage |
|------------|--------|----------|
| **Unit Tests** | ⏳ In Progress | Testbenches written, systematic testing ongoing |
| **Integration Tests** | ⏳ Pending | Require unit tests completion |
| **System Tests** | ⏳ Pending | Full network, accuracy validation |
| **FPGA Validation** | ⏳ Not Started | Actual performance measurement |

**Current Phase**: RTL design complete, moving to verification phase

---

## Learning Outcomes

Through this research project, we gained hands-on experience in:

1. **Digital System Design**: Accelerator architecture from specification to RTL
2. **Computer Architecture**: Memory hierarchy, pipelining, parallelism concepts
3. **ML Hardware**: Systolic arrays, dataflow architectures, quantization principles
4. **RTL Coding**: SystemVerilog design, modular architecture, interface design
5. **Performance Analysis**: Cycle estimation methodology, bottleneck identification
6. **Technical Documentation**: Architecture documentation, design decision justification

---

## Documentation Package

> **READING ORDER:** Start with DOCUMENTATION_GUIDE.md for recommended sequence.

Design documentation is provided in the repository:

1. **DOCUMENTATION_GUIDE.md** ← **START HERE** (reading order, key decisions)
2. **PRESENTATION.md** (comprehensive overview)
3. **IMPLEMENTATION_GUIDE.md** (detailed architecture, dataflow)
4. **OPERATIONS_TABLE_AND_HARDWARE_FLOW.md** (operations analysis)
5. **COMPLETE_DATA_FLOW_GUIDE.md** (detailed walkthrough)
6. **MEMORY_BANKING_ARCHITECTURE.md** (memory design)
7. **READING_ORDER.md** (documentation structure)

**Total**: ~15,000 words of technical documentation

---

## Summary

> **CONCLUSION:** Student research project presenting hardware accelerator architecture design.

This student graduation project presents a **MobileViT hardware accelerator design** that:

- **Architectural Design**: Complete RTL design (~8,000 lines) supporting major MobileViT operations
- **Design Targets**: Aims for ~95 FPS, ~10.5 ms latency (theoretical estimates requiring validation)
- **Resource Planning**: 160 KB on-chip SRAM, ~100K LUTs estimated for mid-range FPGA
- **Standard Interfaces**: 64-bit AXI (compatible with DDR3/DDR4), descriptor-driven control
- **Current Status**: RTL design complete, functional verification and FPGA implementation pending

**Project Value**:
- Demonstrates understanding of ML accelerator architecture principles
- Shows competency in digital design methodology from spec to RTL
- Provides foundation for future implementation and optimization work

**Important Disclaimer**: This is an educational/research project. All performance metrics are theoretical estimates from hand calculations and require validation through simulation, synthesis, and actual hardware measurement. The design has not yet been functionally verified or implemented on FPGA.
