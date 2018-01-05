//
// Created by yuchen on 18-1-3.
//

#ifndef CLNET_UTILITY_GPU_H
#define CLNET_UTILITY_GPU_H

#define GROUP_SIZE 64 // Mali G71 MP8 has 8 parallel compute units. Here, 64 = 512 / 8
#define WIDTH_G 512
#define HEIGHT_G 512

void matrixMul(int *C, const int *A, const int *B, int hA, int wA, int wB);

void fillRandom(int *data, unsigned int width, unsigned height, unsigned long seed);

bool compare(cl_int *gpuMatC, cl_int *matA, cl_int *matB, int heightA, int widthA, int widthB);

#endif //CLNET_UTILITY_GPU_H
