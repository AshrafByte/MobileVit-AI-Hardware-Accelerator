# Documentation Reading Order

---

## ⚠️ Important Context: Student Research Project

**All documentation presents theoretical design work.** The RTL code has been written, but:
- **No systematic simulation/verification** has been completed yet
- **All performance metrics** are estimates based on idealized hand calculations
- **All cycle counts** assume perfect conditions (no stalls, conflicts, or handshaking delays)
- **All efficiency claims** are design targets, not measured results

This is a **graduation/research project** demonstrating architecture design, RTL coding, and system analysis skills. The documentation emphasizes the **design process and learning outcomes**, not production-ready results.

---

## Quick Reference

> **Recommended Reading Sequence:** Follow this order for best understanding.

| Step | Document | Time | Purpose |
|------|----------|------|---------|
| **1** | **READING_ORDER.md** | 5 min | This document - understand structure |
| **2** | **PRESENTATION.md** | 20 min | Overview presentation of the design |
| **3** | **EXECUTIVE_SUMMARY.md** | 10 min | One-page project summary |
| **4** | **DOCUMENTATION_GUIDE.md** | 15 min | Key design decisions and rationale |
| **5** | **IMPLEMENTATION_GUIDE.md** | 30 min | Architecture and dataflow details |
| **6** | **MEMORY_BANKING_ARCHITECTURE.md** | 20 min | Memory subsystem deep dive |
| **7** | **COMPLETE_DATA_FLOW_GUIDE.md** | 30 min | Idealized cycle-by-cycle execution walkthrough |
| **8** | **OPERATIONS_TABLE_AND_HARDWARE_FLOW.md** | 45 min | 150+ operations analysis |
| **9** | **README.md** | 15 min | RTL module reference |


---

## Recommended Reading Sequence

### **STEP 1: Get the Big Picture** (10 minutes)

**Document**: `EXECUTIVE_SUMMARY.md` (Root directory)

**What You'll Learn**:
- Project objectives and what was designed
- Estimated performance metrics (theoretical targets)
- Architecture highlights (5 key design decisions)
- Known limitations with proposed solutions
- Current verification status

**Why Read First**: Provides context for all other documents. You'll understand what was designed, the rationale, and the realistic project status.

> **Key Metrics** (⚠️ theoretical estimates requiring validation):
> - Latency: ~10.5 ms @ 250 MHz (idealized calculation)
> - Throughput: ~95 FPS (theoretical single-image pipeline)
> - Compute Efficiency: 60-80% for convolutions (goal, not measured)
> - Overall Efficiency: <10% realistic average (pending optimization)
> - Bandwidth: ~63% estimated utilization (~2.5 GB/s of 4 GB/s available)
> - Memory: 160 KB SRAM (design capacity)

---

### **STEP 2: Understand This Guide** (5 minutes)

**Document**: `READING_ORDER.md` (Root directory - this file)

**What You'll Learn**:
- How documentation is organized
- Why this reading order is optimal
- What to look for in each document

**Why Read Second**: Ensures you don't waste time jumping between documents randomly.

---

### **STEP 3: Learn Key Design Decisions** (15 minutes)

**Document**: `Documentation/DOCUMENTATION_GUIDE.md`

**What You'll Learn**:
- **Decision 1**: Why descriptor-driven architecture?
- **Decision 2**: Why 16-bank memory (not 4 or 64)?
- **Decision 3**: Why ping-pong buffering?
- **Decision 4**: How multi-tile accumulation is designed to work
- **Decision 5**: Why integrated post-processing?
- Performance estimation methodology (how cycle counts were calculated)
- Known limitations (Softmax, residual add, global pool)

**Why Read Third**: Understand the "why" behind every major choice before diving into implementation details.

> **CRITICAL SECTION:** Performance Justification
> - Shows theoretical cycle calculations for estimated latency
> - Explains bandwidth estimation methodology
> - Breaks down memory footprint by buffer
> - ⚠️ All based on idealized assumptions requiring verification

---

### **STEP 4: Deep Dive Into Architecture** (30 minutes)

**Document**: `IMPLEMENTATION_GUIDE.md` (Root directory)

**What You'll Learn**:
- Block diagram with all 11 modules
- Global controller FSM (16 states explained)
- Descriptor format (256-bit structure)
- Register map (control, status, descriptor push)
- Memory subsystem organization (160 KB layout)
- Systolic array configuration (16×16 base, Lego modes)
- Post-processing pipeline (BN → Swish → LayerNorm)
- DMA interface (AXI4 Master protocol)
- Address generation unit (AGU) operation

**Why Read This**: Now that you know the "why," learn the "how" - implementation details.

> **FOCUS AREAS:**
> - Section 3: Descriptor Format (how CPU programs hardware)
> - Section 5: Memory Subsystem (ping-pong, accumulation)
> - Section 7: Global Controller FSM (orchestration logic)

---

### **STEP 5: Understand Memory Architecture** (20 minutes)

**Document**: `Documentation/MEMORY_BANKING_ARCHITECTURE.md`

**What You'll Learn**:
- Why 16 banks × 32-bit (not 64 banks × 8-bit)
- How bank interleaving is designed to work (DMA writes)
- How parallel reads are intended to work (64 elements/cycle goal)
- Variable bank activation (16/8/4 banks for different SA types)
- FPGA optimization (only 16 BRAM ports, not 64)
- Expected performance improvements

**Why Read This**: Memory is a critical resource - understand how it was designed for both performance and area efficiency.

> **Key Points:** 
> - 32-bit banks match industry standards (AXI/AHB/APB)
> - Designed for single-cycle tile reads for all SA configurations (goal)
> - Expected throughput improvement over flat array design (pending verification)

---

### **STEP 6: Follow Idealized Data Flow** (30 minutes)

**Document**: `Documentation/COMPLETE_DATA_FLOW_GUIDE.md`

**What You'll Learn**:
- Step-by-step theoretical execution analysis
- Intended ping-pong timeline (load, compute, writeback overlap goals)
- Idealized cycle-by-cycle breakdown of conv 3×3 operation
- How multi-tile accumulation is designed to execute
- Memory banking strategy (which banks should be active when)
- Bottleneck estimation (where time is expected to be spent)

**Why Read This**: See the intended hardware flow. This document walks through the **design goals** for execution, showing how pieces are intended to work together.

> **⚠️ IMPORTANT:**
> - All cycle counts are **idealized estimates**
> - Assumes no stalls, conflicts, or handshaking delays
> - Real execution will differ (verification pending)
> - Demonstrates design thinking, not verified performance

---

### **STEP 7: Verify Network Coverage** (45 minutes)

**Document**: `Documentation/OPERATIONS_TABLE_AND_HARDWARE_FLOW.md`

**What You'll Learn**:
- MobileViT-XXS architecture (150+ operations analyzed)
- Every layer from input (256×256×3) to output (1×1×1000)
- Which operations are intended for hardware support (~85%)
- Which operations need software assist (~15%)
- Designed hardware flow for complex MobileViT block (transformers)
- Theoretical performance breakdown by stage
- Estimated bandwidth analysis
- Expected bottlenecks

**Why Read This**: Shows how the hardware is intended to execute the full network. Every operation is analyzed (theoretically).

> **⚠️ CRITICAL NOTES:**
> - All performance numbers are **theoretical estimates**
> - Hardware support percentages are **design intentions**
> - Cycle breakdowns assume **idealized conditions**
> - Bandwidth calculations assume **perfect bursts**
> - Bottleneck identification is **preliminary analysis**

---

### **STEP 8: Check Implementation Status** (10 minutes)

**Document**: `CHECKLIST.md` (Root directory, if exists)

**What You'll Learn**:
- What RTL modules have been written
- What testbenches exist (unit tests status)
- What verification is pending (integration, FPGA)
- Which features are MVP vs future enhancements
- Known issues or limitations

**Why Read Eighth**: Verify implementation status. Transparent assessment of what has been completed and what remains.

---

### **STEP 9: RTL Module Reference** (15 minutes)

**Document**: `README.md` (Root directory)

**What You'll Learn**:
- Every SystemVerilog module listed
- Interface definitions (inputs/outputs)
- Quick description of each module's function
- File structure in repository
- How to run simulations
- How to synthesize for FPGA

**Why Read Last**: Technical reference for looking up specific modules during RTL code review.

---

## For Time-Constrained Review

### **30-Minute Quick Review**
1. **EXECUTIVE_SUMMARY.md** (10 min) - Complete overview
2. **DOCUMENTATION_GUIDE.md** (15 min) - Key decisions
3. **CHECKLIST.md** (5 min) - Verify implementation status

### **1-Hour Review**
1. **EXECUTIVE_SUMMARY.md** (10 min)
2. **DOCUMENTATION_GUIDE.md** (15 min)
3. **IMPLEMENTATION_GUIDE.md** (25 min) - Skim architecture sections
4. **CHECKLIST.md** (10 min)

### **2-Hour Review**
1. **EXECUTIVE_SUMMARY.md** (10 min)
2. **DOCUMENTATION_GUIDE.md** (15 min)
3. **IMPLEMENTATION_GUIDE.md** (30 min)
4. **MEMORY_BANKING_ARCHITECTURE.md** (20 min)
5. **OPERATIONS_TABLE_AND_HARDWARE_FLOW.md** (35 min) - Focus on Section 5
6. **CHECKLIST.md** (10 min)

---

## What to Look For During Review

### **Architecture Quality**
- [ ] Modular design (clean interfaces between blocks)
- [ ] Industry-inspired practices (AXI protocol, descriptor-driven)
- [ ] Proper hierarchy (top-level, subsystems, units)
- [ ] Parameterization (configurable for different designs)

### **Performance Analysis**
- [ ] Theoretical cycle calculations (methodology explained)
- [ ] Bottleneck identification (where time is expected to be spent)
- [ ] Bandwidth estimation (with assumptions stated)
- [ ] Resource estimation (160 KB SRAM, ~100K LUTs target)

### **Technical Depth**
- [ ] FSM design (16 states with clear transitions)
- [ ] Memory management (banking, ping-pong, accumulation)
- [ ] Address generation (AGU designed to handle all layer types)
- [ ] Hardware/software co-design (descriptor interface)

### **Completeness**
- [ ] RTL modules written (11 main modules)
- [ ] All 150+ operations analyzed
- [ ] Known limitations documented with proposed solutions
- [ ] Verification plan defined (even if not yet executed)

### **Professionalism**
- [ ] Clear writing (explanations for technical choices)
- [ ] Consistent formatting (tables, diagrams, code blocks)
- [ ] References to industry practices (with "inspired by" caveats)
- [ ] Honest about limitations (transparent about unverified status)

---

## Common Questions Answered

### **Q1: How do I know the cycle counts are accurate?**
**A**: The cycle counts are **theoretical estimates**, not verified measurements. See `COMPLETE_DATA_FLOW_GUIDE.md` for detailed methodology showing how hand calculations were performed. Real cycle counts will differ and require simulation/FPGA validation.

### **Q2: Why is SA utilization only 4.5%?**
**A**: This was an **overly optimistic estimate**. More realistic average utilization is <10% when accounting for all operations (DMA, normalization, control overhead, software operations). Individual convolutions might achieve 60-80% with proper optimization. See `OPERATIONS_TABLE_AND_HARDWARE_FLOW.md` for analysis.

### **Q3: Why is bandwidth 47-63%?**
**A**: These are **rough estimates** assuming idealized ping-pong overlap and perfect AXI bursts. With ping-pong architecture, reads could overlap with compute, but actual overlap depends on control logic correctness and memory arbitration. Real bandwidth usage requires measurement.

### **Q4: What's not implemented in hardware?**
**A**: Softmax (15% of operations), Residual Add (simple ALU needed), Global Pool (could use SA). See `DOCUMENTATION_GUIDE.md`, Known Limitations section for proposed solutions. These would need to be added for a complete implementation.

### **Q5: Is the design synthesizable?**
**A**: The RTL has been written following synthesizable coding practices. Estimated resources: 160 KB BRAM, ~100K LUTs for Xilinx ZCU102. However, **synthesis has not been run yet**, so these are projections. Timing closure at 250 MHz is a target requiring validation.

### **Q6: What's the verification status?**
**A**: RTL modules have been written and some unit testbenches exist. **Systematic integration testing and FPGA validation are pending**. This is honest disclosure - the design phase is complete, but verification work remains as future work.

---

## Consistency Verification

> **IMPORTANT:** All documents have been updated for consistency and honesty. Key points:

**Performance Metrics** (⚠️ Theoretical estimates across all documents):
- Latency: **~10.5 ms** @ 250 MHz (idealized calculation)
- Throughput: **~95 FPS** (theoretical single-image pipeline)
- Cycles: **~2.6M** estimated for complete network
- Compute Efficiency: **60-80% goal** for convolutions (not measured)
- Overall Efficiency: **<10% realistic** average (pending optimization)

**Bandwidth Analysis** (Rough estimates with caveats):
- Estimated Required: **~2.5 GB/s** (assuming perfect ping-pong)
- Available: **4 GB/s** (bidirectional AXI, 64-bit @ 250 MHz)
- Estimated Utilization: **~63%** (highly dependent on control logic)

**Memory Footprint** (Design capacity):
- Total: **160 KB** SRAM on-chip (design)
- ActBufA: 32 KB, ActBufB: 32 KB
- WgtBuf: 32 KB, PSumBuf: 64 KB
- Banking: **16 banks × 32-bit** architecture

**Resource Coverage** (Design intentions):
- Hardware-intended: **~85%** of operations
- Software-assist needed: **~15%** (primarily Softmax)
- Expected bottleneck: Softmax (pending implementation)

---

## Conclusion

Following this reading order will give you:
1. **Context** (EXECUTIVE_SUMMARY) - What was designed and why
2. **Rationale** (DOCUMENTATION_GUIDE) - Why each design decision was made
3. **Implementation** (IMPLEMENTATION_GUIDE) - How everything is intended to work
4. **Analysis** (COMPLETE_DATA_FLOW_GUIDE, OPERATIONS_TABLE) - Theoretical analysis methodology
5. **Status** (CHECKLIST, README) - What's done (RTL) and what's next (verification)

The documentation demonstrates:
- **System-level thinking** (architecture, interfaces, theoretical performance)
- **RTL implementation skills** (modules written, FSMs, datapaths)
- **Analysis capability** (idealized timing calculations, resource estimation)
- **Professional practices** (verification plan defined, honest about unverified status)
- **Learning outcomes** (design process, trade-off analysis, technical writing)

**This is a student research/graduation project** - the value is in the design process, architectural thinking, and RTL implementation, not in verified production-ready performance.

---
