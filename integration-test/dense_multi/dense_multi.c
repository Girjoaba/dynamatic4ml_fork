
#include "dense_multi.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>

#define TRUNCATE_LAYER(y, idx)                         \
    acc = acc >> (10);                                 \
    y[idx] = (layer_1_t)acc;                           \
    // printf("y[%d] = %d\n", idx, y[idx]);

#define DOT_PROD(x, batch_idx, y, vec_sz)               \
    acc = 0;                                            \
    for (int i = 0; i < vec_sz; i++) {                  \
        acc += x[batch_idx + i] * y[i];                 \
    }   

#define DENSE(x, w, b, y, batch_idx, in_sz, out_sz)     \
    for (int j = 0; j < out_sz; j++) {                  \
        DOT_PROD(x, batch_idx*in_sz, w[j], in_sz); \
        acc += b[j];                                    \
        TRUNCATE_LAYER(y, batch_idx*out_sz+j);     \
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

void dense_multi(input_t    input[INPUT_SIZE],
                 weight_t   w1[OUT_L1][IN_L1],
                 bias_t     b1[OUT_L1],
                 layer_t    y1[INPUT_D1*OUT_L1],
                 layer_t    z1[INPUT_D1*OUT_L1],
                 weight_1_t w2[OUT_L2][IN_L2],
                 bias_1_t   b2[OUT_L2],
                 layer_1_t  y2[INPUT_D1*OUT_L2],
                 layer_1_t  output[OUTPUT_SIZE]) {
            
    dense_accum_t acc;
    layer_t tmp1 = 0, tmp2 = 0;
    for (int batch_idx = 0; batch_idx < INPUT_D1; batch_idx++) {
        DENSE(input, w1, b1, y1, batch_idx, IN_L1, OUT_L1);
        RELU(y1, z1, batch_idx, OUT_L1);
    }
    for (int batch_idx = 0; batch_idx < INPUT_D1; batch_idx++) {
        DENSE(z1, w2, b2, y2, batch_idx, IN_L2, OUT_L2);
        RELU(y2, output, batch_idx, OUT_L2);
    }
}

int main(void) {
    // Input
    input_t input[INPUT_SIZE];
    // Layer 1
    weight_t w1[OUT_L1][IN_L1];
    bias_t b1[OUT_L1];
    layer_t y1[INPUT_D1*OUT_L1];
    layer_t z1[INPUT_D1*OUT_L1];
    // Layer 2
    weight_1_t w2[OUT_L2][IN_L2];
    bias_1_t b2[OUT_L2];
    layer_1_t y2[INPUT_D1*OUT_L2];
    layer_1_t output[OUTPUT_SIZE];

    // ----------- Initialization
    // default_precision_t o = 1;
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1024;            // 1 in <16, 6>
    }
    // Layer 1
    for (int j = 0; j < OUT_L1; j++) {
        for (int i = 0; i < IN_L1; i++) {
            w1[j][i] = 1024;        // 1 in <16, 6>
        }
        b1[j] = 0;       // -2 in <64, 44> = -2097152
    }
    for (int j = 0; j < INPUT_D1*OUT_L1; j++) {
        y1[j] = 1024;               // 1 in <16, 6>
        z1[j] = 1024;               // 1 in <16, 6>
    }
    // Layer 2
    for (int j = 0; j < OUT_L2; j++) {
        for (int i = 0; i < IN_L2; i++) {
            w2[j][i] = 1024;        // 1 in <16, 6>
        }
        b2[j] = 0;           // -6 in <64, 44> = -6291456
    }
    for (int j = 0; j < INPUT_D1*OUT_L2; j++) {
        y2[j] = 21504;      // 21 in <16, 6> = 21504
    }
    // // Output
    // for (int j = 0; j < OUTPUT_SIZE; j++) {
    //     output[j] = 29;
    // }

    CALL_KERNEL(dense_multi,
        input, 
        w1, b1, y1, z1,
        w2, b2, y2, output
    );
    return 0;
}