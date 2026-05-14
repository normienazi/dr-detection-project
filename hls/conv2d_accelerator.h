#ifndef CONV2D_ACCELERATOR_H
#define CONV2D_ACCELERATOR_H

#define IMG_HEIGHT 224
#define IMG_WIDTH 224
#define KERNEL_SIZE 3

#define OUT_HEIGHT (IMG_HEIGHT - KERNEL_SIZE + 1)
#define OUT_WIDTH (IMG_WIDTH - KERNEL_SIZE + 1)

void conv2d_accelerator(
    float input[IMG_HEIGHT][IMG_WIDTH],
    float kernel[KERNEL_SIZE][KERNEL_SIZE],
    float output[OUT_HEIGHT][OUT_WIDTH]
);

#endif