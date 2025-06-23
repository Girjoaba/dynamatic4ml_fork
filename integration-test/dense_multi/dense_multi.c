
#include "dense_multi.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>


#define DENSE(x, w, b, y, batch_idx, in_sz, out_sz)     \
    for (int j = 0; j < out_sz; j++) {                 \
        acc = 0;                                       \
        for (int i = 0; i < in_sz; i++) {              \
            acc += x[batch_idx*in_sz+i] * w[j][i];    \
        }                                              \
        acc += b[j];                                   \
        y[batch_idx*out_sz+j] = acc;                   \
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

void dense_multi(int input[INPUT_SIZE],
                 int w1[OUT_L1][IN_L1],
                 int b1[OUT_L1],
                 int y1[INPUT_D1*OUT_L1],
                 int z1[INPUT_D1*OUT_L1],
                 int w2[OUT_L2][IN_L2],
                 int b2[OUT_L2],
                 int y2[INPUT_D1*OUT_L2],
                 int output[OUTPUT_SIZE]) {
            
    int tmp1 = 0, tmp2 = 0, acc;
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
    int input[INPUT_SIZE];
    // Layer 1
    int    w1[OUT_L1][IN_L1];
    int    b1[OUT_L1];
    int y1[INPUT_D1*OUT_L1];
    int z1[INPUT_D1*OUT_L1];
    // Layer 2
    int    w2[OUT_L2][IN_L2];
    int    b2[OUT_L2];
    int y2[INPUT_D1*OUT_L2];
    int output[OUTPUT_SIZE];

    // ----------- Initialization
    for (int i = 0; i < INPUT_SIZE; i++) {
        input[i] = 1 % 4;
    }
    // Layer 1
    for (int j = 0; j < OUT_L1; j++) {
        for (int i = 0; i < IN_L1; i++) {
            w1[j][i] = rand() % 2;
        }
        b1[j] = rand() % 3;
        y1[j] = 0;
        z1[j] = 0;
    }
    // Layer 2
    for (int j = 0; j < OUT_L2; j++) {
        for (int i = 0; i < IN_L2; i++) {
            w2[j][i] = rand() % 2;
        }
        b2[j] = rand() % 3;
        y2[j] = 0;
    }
    // Output
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        output[j] = 0;
    }

    CALL_KERNEL(dense_multi,
        input, 
        w1, b1, y1, z1,
        w2, b2, y2, output
    );
    return 0;
}