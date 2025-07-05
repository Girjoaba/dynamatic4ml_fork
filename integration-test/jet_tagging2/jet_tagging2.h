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

#define NB_ACC 64
#define INT_ACC 44
#define FRAC_ACC NB_ACC - INT_ACC

typedef int16_t default_t;
typedef int64_t dense_accum_t;

typedef int16_t input_t;

typedef int64_t dense_0_accum_t;
typedef int16_t weight_0_t;
typedef int16_t bias_0_t;
typedef int16_t layer_0_t;

typedef int64_t dense_1_accum_t;
typedef int16_t weight_1_t;
typedef int16_t bias_1_t;
typedef int16_t layer_1_t;


// Input dimensions
#define INPUT_D 3
#define INPUT_SIZE INPUT_D

// ------------------- Layer 1 dimensions
#define IN_L1 INPUT_D
#define OUT_L1 7 

// ------------------- Layer 2 dimensions
#define IN_L2 OUT_L1
#define OUT_L2 2 

// ------------------- Layer 3 dimensions
#define IN_L3 OUT_L2
#define OUT_L3 32 

// ------------------- Layer 4 dimensions
#define IN_L4 OUT_L3
#define OUT_L4 5 


// Output dimensions
#define OUTPUT_SIZE OUT_L4


#endif // DENSE_MULTI_FXD_DENSE_MULTI_FXD_H