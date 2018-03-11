//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_SOFTMAX_LAYER_H
#define CLNET_SOFTMAX_LAYER_H

#include "layer.h"

class softmax_layer : public layer {
    friend class net;

public:
    softmax_layer(int count);

    void forward_cpu(float *input, float *result = nullptr);

    void forward_gpu(cl_objects &clObject, cl::Buffer &input, cl::Buffer &output);

private:
    int count;
};


#endif //CLNET_SOFTMAX_LAYER_H
