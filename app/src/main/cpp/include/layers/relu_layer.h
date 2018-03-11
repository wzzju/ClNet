//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_RELU_LAYER_H
#define CLNET_RELU_LAYER_H

#include <CL/cl.hpp>
#include <opencl/cl_objects.h>
#include "layer.h"

class relu_layer : public layer {
    friend class net;

public:
    relu_layer(int count);

    void forward_cpu(float *input, float *result = nullptr);

    void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &dummy);

private:
    int count;
};


#endif //CLNET_RELU_LAYER_H
