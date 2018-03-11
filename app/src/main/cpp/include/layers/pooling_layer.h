//
// Created by yuchen on 17-12-28.
//

#ifndef CLNET_POOLING_LAYER_H
#define CLNET_POOLING_LAYER_H

#include <CL/cl.hpp>
#include <opencl/cl_objects.h>
#include "layer.h"

class pooling_layer : public layer {
    friend class net;

public:

    pooling_layer(int channels,
                  int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w,
                  int pad_h,
                  int pad_w);

    void forward_cpu(float *input, float *pooled_res = nullptr);
    void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &pooled_res);

private:
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int channels;
    int input_h, input_w;
    //pooling之后的高度、宽度
    int pooled_h, pooled_w;
};


#endif //CLNET_POOLING_LAYER_H
