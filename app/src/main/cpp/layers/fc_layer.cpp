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

fc_layer::~fc_layer() {
    delete[] bias;
    delete[] W;
}

void fc_layer::forward_cpu(float *input, float *fced_res) {
    assert(input != nullptr && fced_res != nullptr);
    inner_plus_b_cpu(W, num_output, num_input, input, num_input, 1, bias, fced_res);
}

void fc_layer::forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &fced_res) {
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
}