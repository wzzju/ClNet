//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_LAYER_H
#define CLNET_LAYER_H

#include <CL/cl.hpp>
#include <opencl/cl_objects.h>
typedef enum {
    LAYER_TYPE_UNKNOWN = 0,
    LAYER_TYPE_CONV,
    LAYER_TYPE_FULLY_CONNECTED,
    LAYER_TYPE_POOL,
    LAYER_TYPE_ACTIVATION,
    LAYER_TYPE_SOFTMAX,
} layer_type;

class layer {

public:
    virtual void forward_cpu(float *input, float *res = nullptr)=0;

    virtual void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &output)=0;

    virtual ~layer() = default;

    layer_type type;
};


#endif //CLNET_LAYER_H
