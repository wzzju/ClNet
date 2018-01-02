//
// Created by yuchen on 17-12-27.
//

#ifndef CAFFE_REF_CONV_LAYER_H
#define CAFFE_REF_CONV_LAYER_H


#include <cstddef>
#include <vector>
#include "layer.h"

class conv_layer : public layer {
    friend class net;

public:

    conv_layer(int conved_c, int input_c, int input_h, int input_w, int kernel_h, int kernel_w, int stride_h,
               int stride_w, int pad_h, int pad_w, float *_W, float *_bias);
    virtual ~conv_layer();

    void forward(float *input, float *conved_res = nullptr);

private:
    int conved_c; // output channel
    int conved_h;//卷积之后的高度,需计算
    int conved_w;//卷积之后的宽度，需计算

    int input_c; // input_data channel
    int input_h;
    int input_w;

    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;

    float *W;
    float *bias;
};


#endif //CAFFE_REF_CONV_LAYER_H
