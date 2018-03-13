//
// Created by yuchen on 17-12-29.
//
#include <cassert>
#include <algorithm>
#include <CL/cl.hpp>
#include <opencl/cl_log.h>
#include <utility_cpu.h>
#include <layers/fc_layer.h>

using namespace std;
using namespace cl;
#if DENSE
fc_layer::fc_layer(int num_output, int num_input, float *_W, float *_bias, bool use_gpu,
                   cl_objects &clObject) :
        num_output(num_output), num_input(num_input), W(nullptr), bias(nullptr) {
    int size_w = num_output * num_input;
    if (use_gpu) {
        try {
            cl_W = Buffer(clObject.getContexts()[0],
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          size_w * sizeof(float),
                          _W);
            cl_bias = Buffer(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             num_output * sizeof(float), _bias);
        } catch (cl::Error err) {
            LOGE("ERROR: %s\n", err.what());
        }
    } else {
        W = new float[size_w];
        bias = new float[num_output];
        copy(_W, _W + size_w, W);
        copy(_bias, _bias + num_output, bias);
        // memcpy(W, _W, size_w * sizeof(float));
        // memcpy(bias, _bias, num_output * sizeof(float));
    }
}
#endif

#if SPARSE

fc_layer::fc_layer(int num_output, int num_input, float *values, int *cols, int *ptr, float *_bias,
                   bool use_gpu, cl_objects &clObject) :
        num_output(num_output), num_input(num_input), W_val(nullptr), W_col(nullptr),
        W_ptr(nullptr), bias(nullptr) {
    int num_nonzero = ptr[num_output];
    if (use_gpu) {
        try {
            cl_W_val = Buffer(clObject.getContexts()[0],
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              num_nonzero * sizeof(float),
                              values);
            cl_W_col = Buffer(clObject.getContexts()[0],
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              num_nonzero * sizeof(int),
                              cols);
            cl_W_ptr = Buffer(clObject.getContexts()[0],
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                              (num_output + 1) * sizeof(int),
                              ptr);
            cl_bias = Buffer(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             num_output * sizeof(float), _bias);
        } catch (cl::Error err) {
            LOGE("ERROR: %s\n", err.what());
        }
    } else {
        W_val = new float[num_nonzero];
        W_col = new int[num_nonzero];
        W_ptr = new int[num_output + 1];
        bias = new float[num_output];
        copy(values, values + num_nonzero, W_val);
        copy(cols, cols + num_nonzero, W_col);
        copy(ptr, ptr + num_output + 1, W_ptr);
        copy(_bias, _bias + num_output, bias);
    }
}

#endif

fc_layer::~fc_layer() {
#if DENSE
    delete[] W;
#endif
#if SPARSE
    delete[] W_val;
    delete[] W_col;
    delete[] W_ptr;
#endif
    delete[] bias;
}

#if DENSE
void fc_layer::forward_cpu(float *input, float *fced_res) {
    assert(input != nullptr && fced_res != nullptr);
    inner_plus_b_cpu(W, num_output, num_input, input, num_input, 1, bias, fced_res);
}
#endif
#if SPARSE

void fc_layer::forward_cpu(float *input, float *fced_res) {
    assert(input != nullptr && fced_res != nullptr);
    spmv_csr_cpu(num_output, W_ptr, W_col, W_val, bias, input, fced_res);
}

#endif

void fc_layer::forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &fced_res) {
#if DENSE
    clObject.getInner_plus_b().kernel.setArg(0, cl_W);
    clObject.getInner_plus_b().kernel.setArg(1, input);
    clObject.getInner_plus_b().kernel.setArg(2, num_input);
    clObject.getInner_plus_b().kernel.setArg(3, cl_bias);
    clObject.getInner_plus_b().kernel.setArg(4, fced_res);
    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getInner_plus_b().kernel,
                                                    NullRange,
                                                    NDRange(num_output, 1),
                                                    NullRange,
                                                    nullptr,
                                                    nullptr);
    clObject.getQueues()[0][0].flush();
    clObject.getQueues()[0][0].finish();
#endif
#if SPARSE
    clObject.getSpMV().kernel.setArg(0, cl_W_val);
    clObject.getSpMV().kernel.setArg(1, input);
    clObject.getSpMV().kernel.setArg(2, cl_W_col);
    clObject.getSpMV().kernel.setArg(3, cl_W_ptr);
    clObject.getSpMV().kernel.setArg(4, cl_bias);
    clObject.getSpMV().kernel.setArg(5, num_output);
    clObject.getSpMV().kernel.setArg(6, fced_res);
//    std::size_t maxLocal = clObject.getSpMV().kernel_max_workgroup_size;
//    std::size_t localWorkSize = VECTOR_SIZE;
//    while (localWorkSize + VECTOR_SIZE <= maxLocal &&
//           localWorkSize + VECTOR_SIZE <= BLOCK_SIZE) {
//        localWorkSize += VECTOR_SIZE;
//    }
//    const std::size_t globalWorkSize = num_output * VECTOR_SIZE; // 1 warp per row
//    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getSpMV().kernel,
//                                                    NullRange,
//                                                    NDRange(globalWorkSize),
//                                                    NDRange(localWorkSize),
//                                                    nullptr,
//                                                    nullptr);
    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getSpMV().kernel,
                                                    NullRange,
                                                    NDRange(num_output),
                                                    NullRange,
                                                    nullptr,
                                                    nullptr);
    clObject.getQueues()[0][0].flush();
    clObject.getQueues()[0][0].finish();
#endif
}