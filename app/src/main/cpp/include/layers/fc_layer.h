//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_FC_LAYER_H
#define CLNET_FC_LAYER_H

#include <opencl/cl_objects.h>
#include "layer.h"

class fc_layer : public layer {
    friend class net;

public:
    fc_layer(int num_output, int num_input, float *_W, float *_bias, bool use_gpu,
             cl_objects &clObject);

    virtual ~fc_layer();

    void forward_cpu(float *input, float *fced_res = nullptr);

    void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &fced_res);

private:
    int num_output;
    int num_input;
    float *W;
    float *bias;
    cl::Buffer cl_W;
    cl::Buffer cl_bias;
};


#endif //CLNET_FC_LAYER_H
