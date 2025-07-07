
#include "jet_tagging2.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdint.h>


#define ARGMAX(z, vec_sz, tmp_argmax)                               \
    for (int i = 0; i < vec_sz; i++) {                              \
        z[i] = (tmp_argmax == z[i]) ? (1 << (FRAC_DEFAULT)) : 0;    \
    }

#define DOT_PROD(x, y, vec_sz, acc_dot)   \
    acc_dot = 0;                          \
    for (int i = 0; i < vec_sz; i++) {    \
        acc_dot += x[i] * y[i];           \
    }   

#define DENSE_ARGMAX(x, w, b, y, in_sz, out_sz, acc_dense, tmp_act)                           \
    for (int j = 0; j < out_sz; j++) {                                                 \
        DOT_PROD(x, w[j], in_sz, acc_dense);                                           \
        acc_dense += (dense_accum_t)(b[j] << (FRAC_DEFAULT));                          \
        /* TRUNCATE */                                                                 \
        acc_dense = acc_dense >> (FRAC_DEFAULT);                                       \
        tmp_act = ((default_t)acc_dense > tmp_act) ? (default_t)acc_dense : tmp_act;   \
        y[j] = (default_t)acc_dense;                                                   \
    }
    
#define DENSE_RELU(x, w, b, z, in_sz, out_sz, acc_dense, tmp_relu)    \
    for (int j = 0; j < out_sz; j++) {                                \
        DOT_PROD(x, w[j], in_sz, acc_dense);                          \
        acc_dense += (dense_accum_t)(b[j] << (FRAC_DEFAULT));         \
        /* TRUNCATE */                                                \
        acc_dense = acc_dense >> (FRAC_DEFAULT);                      \
        tmp_relu = (default_t)acc_dense;                              \
        /* RELU ACTIVATION */                                         \
        z[j] = tmp_relu > 0 ? tmp_relu : 0;                           \
    }

void jet_tagging2(default_t  input[INPUT_SIZE],
                 default_t w1[OUT_L1][IN_L1],
                 default_t b1[OUT_L1],
                 default_t z1[OUT_L1],
                 default_t w2[OUT_L2][IN_L2],
                 default_t b2[OUT_L2],
                 default_t z2[OUT_L2]) {
                     
    // Layer 1:
    dense_accum_t acc0;
    default_t tmp_relu0 = 0;
    DENSE_RELU(input, w1, b1, z1, IN_L1, OUT_L1, acc0, tmp_relu0);

    // Layer 2:
    dense_accum_t acc1;
    default_t tmp_max1 = -(1 << (NB_DEFAULT - 1));
    DENSE_ARGMAX(z1, w2, b2, z2, IN_L2, OUT_L2, acc1, tmp_max1);
    ARGMAX(z2, OUT_L2, tmp_max1);
}

// ===========================================================
// -------------------------- Initialize ---------------------
int main(void) {
    // --------------------------------- Input
    default_t input[INPUT_SIZE] = {1024, 1024, 1024};

    // --------------------------------- Layer 1
    default_t w1[OUT_L1][IN_L1] = {
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024},
        {1024, 1024, 1024}
    };
    default_t   b1[OUT_L1] = {
        -2048, -2048, -2048, -2048, -2048, -2048, -2048
    };
    default_t  z1[OUT_L1];

    // --------------------------------- Layer 2
    default_t w2[OUT_L2][IN_L2] = {
        {1024, 1024, 1024, 1024, 1024, 1024, 1024},
        {1024, 1024, 1024, 1024, 1024, 1024, 1024}
    };
    default_t   b2[OUT_L2] = {
        0, -6144
    };
    default_t  z2[OUT_L2];

    CALL_KERNEL(jet_tagging2,
        input, 
        w1, b1, z1,
        w2, b2, z2
    );
    return 0;
}