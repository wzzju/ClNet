//
// Created by yuchen on 18-1-3.
//
#include <CL/cl.h>
#include <random>
#include "helper.h"
#include "utility_gpu.h"

using namespace std;

void fillRandom(int *data, unsigned int width, unsigned height, unsigned long seed) {
    int *iptr = data;

    uniform_int_distribution<int> distribution(0, 100);
    default_random_engine generator(seed);


    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            iptr[i + j * width] = distribution(generator);
//            LOGD("Seed %lu : %d", seed, iptr[i + j * width]);
        }
    }
}

void matrixMul(int *C,
               const int *A,
               const int *B,
               int hA,
               int wA,
               int wB) {
    int tmp = 0;
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j) {
            tmp = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                tmp += A[i * wA + k] * B[k * wB + j];
            }
            C[i * wB + j] = tmp;
        }
}

bool compare(cl_int *gpuMatC, cl_int *matA, cl_int *matB, int heightA, int widthA, int widthB) {

    cl_int *cpuMat = new cl_int[widthB * heightA]();
    SCOPE_EXIT(delete[] cpuMat);
    {
        CostTimeHelper timeHelper("cpu matmul");
        // cpu matmul
        matrixMul(cpuMat, matA, matB, heightA, widthA, widthB);
    }
    size_t length = heightA * widthB;
    for (int i = 0; i < length; ++i) {
//        LOGD("cpu[%d] vs gpu[%d]\n", cpuMat[i], gpuMatC[i]);
        if (cpuMat[i] != gpuMatC[i]) {
            LOGD("cpuMat[%d] != gpuMatC[%d], %d != %d", i, i, cpuMat[i], gpuMatC[i]);
            return false;
        }
    }
    return true;
}
