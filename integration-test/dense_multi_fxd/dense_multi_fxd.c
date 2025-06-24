
#include "dense_multi_fxd.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

// ================================================================
// ----------------------- Fixed Point Lib ------------------------

// NB = number bits
// Accumulates the result in the rhs. The rhs must already have the out type.
// #define FIXED_ADD(FRAC_IN, NB_OUT, FRAC_OUT, lhs, rhs)                                           \
//     aligned_lhs = (_BitInt(NB_OUT))lhs << (int)(FRAC_OUT - FRAC_IN);           \
//     rhs += aligned_lhs;                               

// Performs a fused-multiply-add and stores the result in c (add rhs operand).
// We must
// #define FIXED_FMADD(NB_OUT, FRAC_OUT, a, b, c)                                      \
//     c = (((_BitInt(NB_OUT))a * (_BitInt(NB_OUT))b) << (int)(FRAC_OUT - FRAC_IN))    \
//         + (_BitInt(NB_OUT))c;

// ================================================================
// ----------------------- NN Implementation ----------------------

#define DOT_PROD(x, y, vec_sz, result)      \
    result = 0;                             \
    for (int i = 0; i < vec_sz; i++) {      \
        result += x[i] * y[i];              \
    }   

#define DENSE(x, w, b, y, batch_idx, in_sz, out_sz)         \
    for (int j = 0; j < out_sz; j++) {                      \
        acc = 0;                                            \
        for (int i = 0; i < in_sz; i++) {                   \
            acc += x[batch_idx*in_sz+i] * w[j][i];          \
        }                                                   \
        acc += b[j];                                        \
        y[batch_idx*out_sz+j] = acc;                        \
    }

#define RELU(y, z, batch_idx, out_sz)                   \
    for (int j = 0; j < out_sz; j++) {                  \
        tmp1 = y[batch_idx*out_sz+j];                   \
        if (tmp1 > 0) {                                 \
            tmp2 = tmp1;                                \
        } else {                                        \
            tmp2 = 0;                                   \
        }                                               \
        z[batch_idx*out_sz+j] = tmp2;                   \
    }   

void dense_multi_fxd(default_precision_t input[INPUT_SIZE],
                     default_precision_t w1[OUT_L1][IN_L1],
                     default_precision_t b1[OUT_L1],
                     default_precision_t y1[INPUT_D1*OUT_L1],
                     default_precision_t z1[INPUT_D1*OUT_L1],
                     default_precision_t w2[OUT_L2][IN_L2],
                     default_precision_t b2[OUT_L2],
                     default_precision_t y2[INPUT_D1*OUT_L2],
                     default_precision_t output[OUTPUT_SIZE]) {
            
    default_precision_t tmp1 = 0, tmp2 = 0, acc;
    
    // ------------------ Layer 1
    _BitInt(NB_DOT_L1) acc_l1 = 0;
    for (int batch_idx = 0; batch_idx < INPUT_D1; batch_idx++) {
        DENSE(input, w1, b1, y1, batch_idx, IN_L1, OUT_L1);
        RELU(y1, z1, batch_idx, OUT_L1);
    }

    // ----------------- Layer 2
    _BitInt(NB_DOT_L2) acc_l2 = 0;
    for (int batch_idx = 0; batch_idx < INPUT_D1; batch_idx++) {
        DENSE(z1, w2, b2, y2, batch_idx, IN_L2, OUT_L2);
        RELU(y2, output, batch_idx, OUT_L2);
    }
}

int main(void) {
    // Input
    default_precision_t input[INPUT_SIZE];
    // Layer 1
    default_precision_t w1[OUT_L1][IN_L1];
    default_precision_t b1[OUT_L1];
    default_precision_t y1[INPUT_D1*OUT_L1];
    default_precision_t z1[INPUT_D1*OUT_L1];
    // Layer 2
    default_precision_t w2[OUT_L2][IN_L2];
    default_precision_t b2[OUT_L2];
    default_precision_t y2[INPUT_D1*OUT_L2];
    default_precision_t output[OUTPUT_SIZE];

    // ----------- Initialization
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1;
    }
    // Layer 1
    for (int j = 0; j < OUT_L1; j++) {
        for (int i = 0; i < IN_L1; i++) {
            w1[j][i] = 1;
        }
        b1[j] = 1;
    }
    for (int j = 0; j < INPUT_D1*OUT_L1; j++) {
        y1[j] = w1[0][0] * IN_L1 * input[0] + b1[0];
        z1[j] = w1[0][0] * IN_L1 * input[0] + b1[0];
    }
    // Layer 2
    for (int j = 0; j < OUT_L2; j++) {
        for (int i = 0; i < IN_L2; i++) {
            w2[j][i] = 1;
        }
        b2[j] = 1;
    }
    for (int j = 0; j < INPUT_D1*OUT_L2; j++) {
        y2[j] = w2[0][0] * IN_L2 * z1[0] + b2[0];
    }
    // // Output
    // for (int j = 0; j < OUTPUT_SIZE; j++) {
    //     output[j] = 29;
    // }

    CALL_KERNEL(dense_multi_fxd,
        input, 
        w1, b1, y1, z1,
        w2, b2, y2, output
    );
    return 0;
}