
#include "jet_tagging_w_fix.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdint.h>


#define DENSE_RELU(x, w, b, z, in_sz, out_sz)           \
for (int j = 0; j < out_sz; j++) {                  \
    acc = 0;                                        \
    for (int i = 0; i < in_sz; i++) {               \
        acc += x[i] * w[in_sz*j + i];               \
    }                                               \
    acc += (accum_t)(b[j] << (FRAC_DEFAULT));       \
    /* TRUNCATE */                                  \
    acc = acc >> (FRAC_DEFAULT);                    \
    tmp1 = (default_t)acc;                          \
    /* RELU ACTIVATION */                           \
    if (tmp1 > 0) {                                 \
        tmp2 = tmp1;                                \
    } else {                                        \
        tmp2 = 0;                                   \
    }                                               \
    z[j] = tmp2;                                    \
}

void jet_tagging_w_fix(
    default_t input[INPUT_SIZE],
    default_t b1[OUT_L1],
    default_t z1[OUT_L1],
    default_t w2[OUT_L2*IN_L2],
    default_t b2[OUT_L2],
    default_t output[OUTPUT_SIZE]) {
        default_t w1[OUT_L1 * IN_L1] = {
            1024, 1024, 1024,
            1024, 1024, 1024,
            1024, 1024, 1024,
            1024, 1024, 1024,
            1024, 1024, 1024,
            1024, 1024, 1024,
            1024, 1024, 1024
        };
        
        accum_t acc;
        default_t tmp1 = 0, tmp2 = 0;
        
        DENSE_RELU(input, w1, b1, z1, IN_L1, OUT_L1);
        DENSE_RELU(z1, w2, b2, output, IN_L2, OUT_L2);
    }
    
// ===========================================================
// -------------------------- Initialize ---------------------
int main(void) {
    // --------------------------------- Input
    default_t input[INPUT_SIZE] = {1024, 1024, 1024};

    // --------------------------------- Layer 1
    default_t   b1[OUT_L1] = {
        -2048, -2048, -2048, -2048, -2048, -2048, -2048
    };
    default_t  z1[OUT_L1];

    // --------------------------------- Layer 2
    default_t w2[OUT_L2 * IN_L2] = {
        1024, 1024, 1024, 1024, 1024, 1024, 1024,
        1024, 1024, 1024, 1024, 1024, 1024, 1024
    };
    default_t   b2[OUT_L2] = {
        -6144, -6144
    };
    default_t  output[OUTPUT_SIZE];

    CALL_KERNEL(jet_tagging_w_fix,
        input, 
        b1, z1,
        w2, b2, output
    );
    return 0;
}