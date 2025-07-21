
#include "jet_tagging.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdint.h>

#define TRUNCATE_LAYER(y, idx)                         \
    acc = acc >> (FRAC_DEFAULT);                       \
    y[idx] = (layer_1_t)acc;                           

#define DOT_PROD(x, y, vec_sz)                          \
    acc = 0;                                            \
    for (int i = 0; i < vec_sz; i++) {                  \
        acc += x[i] * y[i];                             \
    }   

    
#define RELU(y, z, out_sz)                              \
    for (int j = 0; j < out_sz; j++) {                  \
        tmp1 = y[j];                                    \
        if (tmp1 > 0) {                                 \
            tmp2 = tmp1;                                \
        } else {                                        \
            tmp2 = 0;                                   \
        }                                               \
        z[j] = tmp2;                                    \
    }   

#define DENSE_RELU(x, w, b, z, in_sz, out_sz)           \
    for (int j = 0; j < out_sz; j++) {                  \
        DOT_PROD(x, w[j], in_sz);                       \
        acc += (dense_accum_t)(b[j] << 10);             \
        /* TRUNCATE */                                  \
        acc = acc >> (10);                              \
        tmp1 = (layer_1_t)acc;                          \
        /* RELU ACTIVATION */                           \
        if (tmp1 > 0) {                                 \
            tmp2 = tmp1;                                \
        } else {                                        \
            tmp2 = 0;                                   \
        }                                               \
        z[j] = tmp2;                                    \
    }

void jet_tagging(input_t    input[INPUT_SIZE],
                 weight_t   w1[OUT_L1][IN_L1],
                 bias_t     b1[OUT_L1],
                 layer_t    z1[OUT_L1],
                 weight_1_t w2[OUT_L2][IN_L2],
                 bias_1_t   b2[OUT_L2],
                 layer_1_t  output[OUTPUT_SIZE]) {
    dense_accum_t acc;
    layer_t tmp1 = 0, tmp2 = 0;

    DENSE_RELU(input, w1, b1, z1, IN_L1, OUT_L1);
    DENSE_RELU(z1, w2, b2, output, IN_L2, OUT_L2);
}

// ===========================================================
// -------------------------- Initialize ---------------------
int main(void) {
    // --------------------------------- Input
    input_t input[INPUT_SIZE] = {1024, 1024, 1024};

    // --------------------------------- Layer 1
    weight_t w1[OUT_L1][IN_L1] = {
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024}
    };
    bias_t   b1[OUT_L1] = {
        -2048, -2048, -2048, -2048, -2048, -2048, -2048
    };
    layer_t  z1[OUT_L1];

    // --------------------------------- Layer 2
    weight_1_t w2[OUT_L2][IN_L2] = {
        {1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024}
    };
    bias_1_t   b2[OUT_L2] = {
        -6144, -6144
    };
    layer_1_t  output[OUTPUT_SIZE];

    CALL_KERNEL(jet_tagging,
        input, 
        w1, b1, z1,
        w2, b2, output
    );
    return 0;
}