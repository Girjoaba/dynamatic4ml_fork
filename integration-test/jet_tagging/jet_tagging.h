#ifndef DENSE_MULTI_DENSE_MULTI_H
#define DENSE_MULTI_DENSE_MULTI_H


#include <stdint.h>

// typedef int in_int_t;
// typedef int out_int_t;
// typedef int inout_int_t;

// hls-fpga-machine-learning insert layer precision
#define NB_DEFAULT 16
#define INT_DEFAULT 6
#define FRAC_DEFAULT NB_DEFAULT - INT_DEFAULT

#define NB_ACC  64
#define INT_ACC 44
#define FRAC_ACC NB_ACC - INT_ACC

typedef int16_t input_t;

typedef int64_t dense_accum_t;
typedef int16_t weight_t;
typedef int16_t bias_t;
typedef int16_t layer_t;

typedef int64_t dense_1_accum_t;
typedef int16_t weight_1_t;
typedef int16_t bias_1_t;
typedef int16_t layer_1_t;


// hls-fpga-machine-learning insert numbers
// Used for the input
#define NUMBER_BITS_L1   16
#define FRACTION_BITS_L1 10
#define INTEGER_BITS_L1  NUMBER_BITS_L1 - FRACTION_BITS_L1 

// Used for truncation
#define NUMBER_BITS_L2   16
#define FRACTION_BITS_L2 10
#define INTEGER_BITS_L2  NUMBER_BITS_L2 - FRACTION_BITS_L2 

// Input dimensions
#define INPUT_D 3
#define INPUT_SIZE INPUT_D

// Layer 1 dimensions
#define IN_L1 INPUT_D
#define OUT_L1 7 

// Layer 2 dimensions
#define IN_L2 OUT_L1
#define OUT_L2 2 

// Output dimensions
#define OUTPUT_SIZE OUT_L2


#endif // DENSE_MULTI_FXD_DENSE_MULTI_FXD_H