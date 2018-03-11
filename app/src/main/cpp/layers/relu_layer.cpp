//
// Created by yuchen on 17-12-29.
//

#include <cassert>
#include <CL/cl.hpp>
#include <utility_cpu.h>
#include <layers/relu_layer.h>

using namespace cl;

relu_layer::relu_layer(int count) : count(count) {}

void relu_layer::forward_cpu(float *input, float *result) {
    assert(input != nullptr);
    activation_relu_cpu(input, count);
}

void relu_layer::forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &dummy) {
    clObject.getRelu().kernel.setArg(0, input);
    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getRelu().kernel,
                                                    NullRange,
                                                    NDRange(count),
                                                    NullRange,
                                                    nullptr,
                                                    nullptr);
    clObject.getQueues()[0][0].flush();
    clObject.getQueues()[0][0].finish();
}