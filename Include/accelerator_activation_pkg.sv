//==============================================================================
// Package: accelerator_activation_pkg
// Description: Activation functions (Swish, ReLU, etc.) parameters
//==============================================================================

package accelerator_activation_pkg;

    import accelerator_common_pkg::*;

    //==========================================================================
    // Swish Activation Constants
    //==========================================================================
    parameter int SWISH_CONST_THREE = 3;     // Offset for hard-swish
    parameter int SWISH_CONST_SIX   = 6;     // Divisor for hard-swish

endpackage : accelerator_activation_pkg
