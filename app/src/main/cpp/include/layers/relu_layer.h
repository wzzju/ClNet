//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_RELU_LAYER_H
#define CLNET_RELU_LAYER_H


#include "layer.h"

class relu_layer : public layer {
    friend class net;

public:
    relu_layer(int count);

    void forward(float *input, float *result = nullptr);

private:
    int count;
};


#endif //CLNET_RELU_LAYER_H
