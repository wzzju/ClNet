//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_FC_LAYER_H
#define CLNET_FC_LAYER_H


#include "layer.h"

class fc_layer : public layer {
    friend class net;

public:
    fc_layer(int num_output, int num_input, float *_W, float *_bias);

    virtual ~fc_layer();

    void forward(float *input, float *fced_res = nullptr);

private:
    int num_output;
    int num_input;
    float *W;
    float *bias;
};


#endif //CLNET_FC_LAYER_H
