//
// Created by yuchen on 17-12-29.
//

#ifndef CAFFE_REF_SOFTMAX_LAYER_H
#define CAFFE_REF_SOFTMAX_LAYER_H


#include "layer.h"

class softmax_layer : public layer {
    friend class net;

public:
    softmax_layer(int count);

    virtual ~softmax_layer() = default;

    void forward(float *input, float *result = nullptr);

private:
    int count;
};


#endif //CAFFE_REF_SOFTMAX_LAYER_H
