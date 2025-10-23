# MobileViT Hardware Accelerator

## Overview

This project is a student design effort to create a hardware accelerator for MobileViT neural network inference. The design targets FPGA implementation and includes RTL modules for convolution, transformer attention, and normalization operations.

## Project Structure

```
RTL/                          - SystemVerilog hardware modules
├── mobilevit_accelerator_top.sv  - Top-level integration
├── Control/                  - FSM controller
├── DMA/                      - Memory interface
├── Compute/                  - Systolic array and processing
├── AGU/                      - Address generation
├── Memory Subsystem/         - On-chip SRAM buffers
├── Lego SA/                  - Systolic array components
├── Batch Norm/               - Normalization modules
├── Layer Norm/               - Layer normalization
└── Swish/                    - Activation functions

Include/                      - SystemVerilog packages and types
Testbench/                    - Simulation testbenches
Documentation/                - Design documentation
Python modelling/             - Reference models
```

## Documentation

For detailed information, please refer to:

- **`Documentation/PRESENTATION.md`** - Comprehensive overview and design explanation
- **`Documentation/EXECUTIVE_SUMMARY.md`** - High-level summary
- **`Documentation/IMPLEMENTATION_GUIDE.md`** - Technical implementation details
- **`Documentation/READING_ORDER.md`** - Suggested order for understanding the design

## Design Targets

The accelerator design aims for:
- **Clock**: 400 MHz (target, needs synthesis verification)
- **Throughput**: ~100 GOPS (INT8, theoretical)
- **Memory**: 160 KB on-chip SRAM
- **Power**: Target <5W (estimated)

> **Note**: All performance numbers are design targets based on cycle-accurate calculations. Actual performance requires FPGA implementation and measurement.

## Current Status

This is an academic project currently in the design and verification phase. Key components implemented:
- RTL modules for all major operations
- Control FSM for layer sequencing
- Memory subsystem with ping-pong buffering
- Systolic array compute engine
- Post-processing pipeline

## Requirements

- **Synthesis**: Xilinx Vivado (targeting UltraScale+ or similar)
- **Simulation**: ModelSim, Vivado Simulator, or Verilator
- **Language**: SystemVerilog (IEEE 1800-2017)
- **Target Device**: Xilinx Zynq UltraScale+ FPGA

## Getting Started

1. **Read the documentation**: Start with `Documentation/PRESENTATION.md`
2. **Explore the RTL**: Review `RTL/mobilevit_accelerator_top.sv`
3. **Run simulations**: Use testbenches in `Testbench/`
4. **Synthesis**: Follow instructions in `Documentation/IMPLEMENTATION_GUIDE.md`

## License

This is an academic project developed for educational purposes.

## Contact

For questions about this project, please refer to the documentation or contact the development team.

---

*This is a student project. All claims and performance estimates require validation through synthesis and FPGA implementation.*
