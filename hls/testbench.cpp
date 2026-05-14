#include <iostream>
#include "conv2d_accelerator.h"

int main() {
    static float input[IMG_HEIGHT][IMG_WIDTH];
    static float kernel[KERNEL_SIZE][KERNEL_SIZE];
    static float output[OUT_HEIGHT][OUT_WIDTH];

    // Create artificial image:
    // Left half dark, right half bright
    for (int i = 0; i < IMG_HEIGHT; i++) {
        for (int j = 0; j < IMG_WIDTH; j++) {
            if (j < IMG_WIDTH / 2) {
                input[i][j] = 0.0f;
            } else {
                input[i][j] = 1.0f;
            }
        }
    }

    // Edge detection kernel
    float temp_kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        { -1, -1, -1 },
        { -1,  8, -1 },
        { -1, -1, -1 }
    };

    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            kernel[i][j] = temp_kernel[i][j];
        }
    }

    conv2d_accelerator(input, kernel, output);

    std::cout << "Convolution completed successfully." << std::endl;

    std::cout << "\nSample output around edge region:" << std::endl;

    int edge_col = (IMG_WIDTH / 2) - 3;

    for (int i = 100; i < 105; i++) {
        for (int j = edge_col; j < edge_col + 8; j++) {
            std::cout << output[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}