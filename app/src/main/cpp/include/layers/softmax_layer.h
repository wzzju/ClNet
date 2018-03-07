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

    void forward(float *input, float *result = nullptr);

private:
    int count;
};


#endif //CLNET_SOFTMAX_LAYER_H
