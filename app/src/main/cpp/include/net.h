//
// Created by yuchen on 17-12-29.
//

#ifndef CLNET_NET_H
#define CLNET_NET_H

#include <vector>

#define INPUT_C 3
#define INPUT_H 28
#define INPUT_W 28

class conv_layer;

class pooling_layer;

class fc_layer;

class relu_layer;

class softmax_layer;

class net {
public:
    net();

    ~net();

    void init(std::string weight_path, std::string cl_path, bool use_gpu);

    std::vector<float> forward(float *input_data);

private:
    bool use_gpu;
    std::string cl_path;
    conv_layer *conv1;
    pooling_layer *pool1;
    conv_layer *conv2;
    pooling_layer *pool2;
    fc_layer *fc1;
    relu_layer *relu1;
    fc_layer *fc2;
    softmax_layer *softmax;

};


#endif //CLNET_NET_H
