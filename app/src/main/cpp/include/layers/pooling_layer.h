//
// Created by yuchen on 17-12-28.
//

#ifndef CAFFE_REF_POOLING_LAYER_H
#define CAFFE_REF_POOLING_LAYER_H

#include "layer.h"

class pooling_layer : public layer {
    friend class net;

public:

    pooling_layer(int channels,
                  int input_h, int input_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h,
                  int pad_w);

    virtual ~pooling_layer() = default;

    void forward(float *input, float *pooled_res = nullptr);

private:
    int kernel_h, kernel_w;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int channels;
    int input_h, input_w;
    //pooling之后的高度、宽度
    int pooled_h, pooled_w;
};


#endif //CAFFE_REF_POOLING_LAYER_H
