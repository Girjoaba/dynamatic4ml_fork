#ifndef DENSE_MULTI_DENSE_MULTI_H
#define DENSE_MULTI_DENSE_MULTI_H

// typedef int in_int_t;
// typedef int out_int_t;
// typedef int inout_int_t;

typedef _BitInt(16) common_fxd_t;

// Input dimensions
#define INPUT_D1 4
#define INPUT_D2 3
#define INPUT_SIZE INPUT_D1 * INPUT_D2

// Layer 1 dimensions
#define IN_L1 INPUT_D2
#define OUT_L1 7 

// Layer 2 dimensions
#define IN_L2 OUT_L1
#define OUT_L2 2 

// Output dimensions
#define OUTPUT_SIZE INPUT_D1 * OUT_L2


#endif // DENSE_MULTI_DENSE_MULTI_H