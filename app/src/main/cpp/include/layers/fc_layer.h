//
// Created by yuchen on 17-12-29.
//

#ifndef CAFFE_REF_FC_LAYER_H
#define CAFFE_REF_FC_LAYER_H


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


#endif //CAFFE_REF_FC_LAYER_H
