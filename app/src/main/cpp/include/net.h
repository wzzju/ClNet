//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_NET_H
#define CLNET_NET_H

#include <vector>

#define LENET 0
#define ALEXNET 1

#if LENET
#define INPUT_C 1
#define INPUT_H 28
#define INPUT_W 28
#endif

#if ALEXNET
#define INPUT_C 3
#define INPUT_H 227
#define INPUT_W 227
#endif

class conv_layer;

class pooling_layer;

class fc_layer;

class relu_layer;

class softmax_layer;

class net {
public:
    net(std::string weight_path, std::string cl_path, bool use_gpu);

    ~net();

    std::vector<float> forward(float *input_data);

private:
    bool use_gpu;
    std::string cl_path;
#if LENET
    conv_layer *conv1;
    pooling_layer *pool1;
    conv_layer *conv2;
    pooling_layer *pool2;
    fc_layer *fc1;
    relu_layer *relu1;
    fc_layer *fc2;
    softmax_layer *softmax;
#endif
#if ALEXNET
    conv_layer *conv1;
    relu_layer *relu1;
    pooling_layer *pool1;
    conv_layer *conv2;
    relu_layer *relu2;
    pooling_layer *pool2;
    conv_layer *conv3;
    relu_layer *relu3;
    conv_layer *conv4;
    relu_layer *relu4;
    conv_layer *conv5;
    relu_layer *relu5;
    pooling_layer *pool5;
    fc_layer *fc6;
    relu_layer *relu6;
    fc_layer *fc7;
    relu_layer *relu7;
    fc_layer *fc8;
    softmax_layer *softmax;
#endif
};


#endif //CLNET_NET_H
