#ifndef JET_TAGGING_W_FIX_JET_TAGGING_W_FIX_H
#define JET_TAGGING_W_FIX_JET_TAGGING_W_FIX_H


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

typedef int16_t default_t;
typedef int64_t accum_t;

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


#endif // JET_TAGGING_W_FIX_FXD_JET_TAGGING_W_FIX_FXD_H