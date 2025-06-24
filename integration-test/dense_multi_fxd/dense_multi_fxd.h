#ifndef DENSE_MULTI_FXD_DENSE_MULTI_FXD_H
#define DENSE_MULTI_FXD_DENSE_MULTI_FXD_H

// typedef int in_int_t;
// typedef int out_int_t;
// typedef int inout_int_t;

typedef _BitInt(16) default_precision_t;

// ====================================================================
// -------------------- Goes into the config obj ----------------------

// Used for the input
#define NUMBER_BITS_L1   16
#define FRACTION_BITS_L1 10
#define INTEGER_BITS_L1  NUMBER_BITS_L1 - FRACTION_BITS_L1 

// Used for truncation
#define NUMBER_BITS_L2   16
#define FRACTION_BITS_L2 10
#define INTEGER_BITS_L2  NUMBER_BITS_L2 - FRACTION_BITS_L2 

// Input dimensions
#define INPUT_D1 4
#define INPUT_D2 3
#define INPUT_SIZE INPUT_D1 * INPUT_D2

// Layer 1 dimensions
#define IN_L1 INPUT_D2
#define OUT_L1 7 
#define NB_DOT_L1 NUMBER_BITS_L1 + NUMBER_BITS_L1 + IN_L1 - 1
#define FRAC_DOT_L1 FRACTION_BITS_L1 + FRACTION_BITS_L1 

// Layer 2 dimensions
#define IN_L2 OUT_L1
#define OUT_L2 2 
#define NB_DOT_L2 NUMBER_BITS_L2 + NUMBER_BITS_L2 + IN_L2 - 1
#define FRAC_DOT_L2 FRACTION_BITS_L2 + FRACTION_BITS_L2 

// Output dimensions
#define OUTPUT_SIZE INPUT_D1 * OUT_L2


#endif // DENSE_MULTI_FXD_DENSE_MULTI_FXD_H