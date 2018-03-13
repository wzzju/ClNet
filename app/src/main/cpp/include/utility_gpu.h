//
// Created by yuchen on 18-1-3.
//

#ifndef CLNET_UTILITY_GPU_H
#define CLNET_UTILITY_GPU_H

#include <sstream>

class cl_objects;

bool compare_layer(cl_objects &clObject, cl::Buffer &gpu_mat, float *cpu_mat, int size);

void test_relu(cl_objects &clObject, std::stringstream &strs);

void test_inner(cl_objects &clObject, std::stringstream &strs);

void test_inner_plus_b(cl_objects &clObject, std::stringstream &strs);

void test_im2col(cl_objects &clObject, std::stringstream &strs);

void test_max_pool(cl_objects &clObject, std::stringstream &strs);

void test_spmv(cl_objects &clObject, std::stringstream &strs);

#endif //CLNET_UTILITY_GPU_H
