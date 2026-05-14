#include "conv2d_accelerator.h"

void conv2d_accelerator(
    float input[IMG_HEIGHT][IMG_WIDTH],
    float kernel[KERNEL_SIZE][KERNEL_SIZE],
    float output[OUT_HEIGHT][OUT_WIDTH]
) {
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=kernel offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=output offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=input bundle=control
#pragma HLS INTERFACE s_axilite port=kernel bundle=control
#pragma HLS INTERFACE s_axilite port=output bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < OUT_HEIGHT; i++) {
        for (int j = 0; j < OUT_WIDTH; j++) {
#pragma HLS PIPELINE II=1

            float sum = 0.0f;

            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
                for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }

            output[i][j] = sum;
        }
    }
}