
#include "jet_tagging2.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdint.h>

    // printf("MAX_VAL=%d\n", max_val); 
    // printf("ARGMAX index: %d\n", i);  

#define ARGMAX(y, z, vec_sz)                                \
    for (int i = 0; i < vec_sz; i++) {                   \
        max_val = (max_val > y[i]) ? (max_val) : (y[i]); \
    }                                                    \
    for (int i = 0; i < vec_sz; i++) {                   \
        if (y[i] == max_val) {                           \
            z[i] = 1 << (FRAC_DEFAULT);                  \
        } else {                                         \
            z[i] = 0;                                    \
        }                                                \
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

#define TRUNCATE_LAYER(y, idx)                         \
    acc = acc >> (FRAC_DEFAULT);                       \
    y[idx] = (layer_1_t)acc;                           \
    // printf("y[%d] = %d\n", idx, y[idx]);

#define DOT_PROD(x, y, vec_sz)                          \
    acc = 0;                                            \
    for (int i = 0; i < vec_sz; i++) {                  \
        acc += x[i] * y[i];                             \
    }   

#define DENSE(x, w, b, y, in_sz, out_sz)                \
    for (int j = 0; j < out_sz; j++) {                  \
        DOT_PROD(x, w[j], in_sz);                       \
        acc += (dense_accum_t)(b[j] << (FRAC_DEFAULT)); \
        TRUNCATE_LAYER(y, j);                           \
    }



void jet_tagging2(default_t input[INPUT_SIZE],
                 default_t  w0[OUT_L1][IN_L1],
                 default_t  b0[OUT_L1],
                 default_t  y0[OUT_L1],
                 default_t  z0[OUT_L1],
                 default_t  w1[OUT_L2][IN_L2],
                 default_t  b1[OUT_L2],
                 default_t  y1[OUT_L2],
                 default_t  z1[OUT_L2]) {
            
    dense_accum_t acc;
    default_t tmp1 = 0, tmp2 = 0;
    default_t max_val = -(1 << (NB_DEFAULT-1));

    DENSE(input, w0, b0, y0, IN_L1, OUT_L1);
    RELU(y0, z0, OUT_L1);

    DENSE(z0, w1, b1, y1, IN_L2, OUT_L2);

    // for (int i = 0; i < OUT_L2; i++) {
    //     printf("y1[%d] = %d\n", i, y1[i]);
    // }
    // RELU(y1, z1, OUT_L2);
    ARGMAX(y1, z1, OUT_L2);
    // for (int i = 0; i < OUT_L2; i++) {
    //     printf("z1[%d] = %d\n", i, z1[i]);
    // }
}

int main(void) {
    // Input
    default_t input[INPUT_SIZE];
    // Layer 1
    default_t w0[OUT_L1][IN_L1];
    default_t  b0[OUT_L1];
    default_t  y0[OUT_L1];
    default_t  z0[OUT_L1];
    // Layer 2
    default_t  w1[OUT_L2][IN_L2];
    default_t  b1[OUT_L2];
    default_t  y1[OUT_L2];
    default_t  z1[OUT_L2];

    // ----------- Initialization
    // default_precision_t o = 1;
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1024;            // 1 in <16, 6>
    }
    // Layer 1
    for (int j = 0; j < OUT_L1; j++) {
        for (int i = 0; i < IN_L1; i++) {
            w0[j][i] = 1024;        // 1 in <16, 6>
        }
        b0[j] = -2048;       // -2 in <64, 44> = -2097152
    }
    for (int j = 0; j < OUT_L1; j++) {
        y0[j] = 1024;               // 1 in <16, 6>
        z0[j] = 1024;               // 1 in <16, 6>
    }
    // Layer 2
    for (int j = 0; j < OUT_L2; j++) {
        for (int i = 0; i < IN_L2; i++) {
            w1[j][i] = 1024 + (j << (FRAC_DEFAULT));        // 1 in <16, 6>
        }
        b1[j] = -6144;           // -6 in <64, 44> = -6291456
    }

    // for (int j = 0; j < OUT_L2; j++) {
    //     y1[j] = 0;      // 21 in <16, 6> = 21504
    // }
    // for (int j = 0; j < OUT_L2; j++) {
    //     z1[j] = 0;      // 21 in <16, 6> = 21504
    // }
    y1[0] = 1024;
    y1[1] = 8096;
    z1[0] = 0;
    z1[1] = 1024;


    CALL_KERNEL(jet_tagging2,
        input, 
        w0, b0, y0, z0,
        w1, b1, y1, z1
    );
    return 0;
}