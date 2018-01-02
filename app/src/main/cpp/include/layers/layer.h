//
// Created by yuchen on 17-12-29.
//

#ifndef CAFFE_REF_LAYER_H
#define CAFFE_REF_LAYER_H


typedef enum {
    LAYER_TYPE_UNKNOWN = 0,
    LAYER_TYPE_CONV,
    LAYER_TYPE_FULLY_CONNECTED,
    LAYER_TYPE_POOL,
    LAYER_TYPE_ACTIVATION,
    LAYER_TYPE_SOFTMAX,
} layer_type;

class layer {

public:
    virtual void forward(float *input, float *fced_res = nullptr)=0;

    layer_type type;
};


#endif //CAFFE_REF_LAYER_H
