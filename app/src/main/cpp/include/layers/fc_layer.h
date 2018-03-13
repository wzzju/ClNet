//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_FC_LAYER_H
#define CLNET_FC_LAYER_H

#include <opencl/cl_objects.h>
#include <utility_cpu.h>
#include "layer.h"

class fc_layer : public layer {
    friend class net;

public:
#if DENSE
    fc_layer(int num_output, int num_input, float *_W, float *_bias, bool use_gpu,
             cl_objects &clObject);

#endif
#if SPARSE

    fc_layer(int num_output, int num_input, float *values, int *cols, int *ptr,
             float *_bias, bool use_gpu, cl_objects &clObject);

#endif

    virtual ~fc_layer();

    void forward_cpu(float *input, float *fced_res = nullptr);

    void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &fced_res);

private:
    int num_output;
    int num_input;
    float *bias;
#if DENSE
    float *W;
    cl::Buffer cl_W;
#endif
#if SPARSE
    float *W_val;
    int *W_col;
    int *W_ptr;
    cl::Buffer cl_W_val;
    cl::Buffer cl_W_col;
    cl::Buffer cl_W_ptr;
#endif
    cl::Buffer cl_bias;
};


#endif //CLNET_FC_LAYER_H
