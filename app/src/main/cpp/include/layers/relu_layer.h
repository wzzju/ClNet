//
// Created by yuchen on 17-12-29.
//

#ifndef CAFFE_REF_RELU_LAYER_H
#define CAFFE_REF_RELU_LAYER_H


#include "layer.h"

class relu_layer : public layer {
    friend class net;

public:
    relu_layer(int count);

    virtual ~relu_layer() = default;

    void forward(float *input, float *result = nullptr);

private:
    int count;
};


#endif //CAFFE_REF_RELU_LAYER_H
