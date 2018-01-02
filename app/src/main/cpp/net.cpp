//
// Created by yuchen on 17-12-29.
//
#include <iostream>
#include <vector>
#include <cassert>
#include "cnpy.h"
#include "utilities.h"
#include "layers/conv_layer.h"
#include "layers/pooling_layer.h"
#include "layers/softmax_layer.h"
#include "layers/fc_layer.h"
#include "layers/relu_layer.h"
#include "net.h"

using namespace std;

net::net(string proto_path) {
    /****************************卷积层1****************************/
    string path_conv1_w = proto_path + "Convolution1_w.npy";
    cnpy::NpyArray conv1_w = cnpy::npy_load(path_conv1_w);
    float *conv1_w_data = conv1_w.data<float>();

    string path_conv1_b = proto_path + "Convolution1_b.npy";
    cnpy::NpyArray conv1_b = cnpy::npy_load(path_conv1_b);
    float *conv1_b_data = conv1_b.data<float>();

    conv1 = new conv_layer(20, INPUT_C, INPUT_H, INPUT_W, 5, 5, 1, 1, 0, 0, conv1_w_data, conv1_b_data);
    conv1->type = LAYER_TYPE_CONV;
    /****************************池化层1****************************/
    pool1 = new pooling_layer(conv1->conved_c, conv1->conved_h, conv1->conved_w,
                              2, 2, 2, 2, 0, 0);
    pool1->type = LAYER_TYPE_POOL;
    /****************************卷积层2****************************/
    string path_conv2_w = proto_path + "Convolution2_w.npy";
    cnpy::NpyArray conv2_w = cnpy::npy_load(path_conv2_w);
    float *conv2_w_data = conv2_w.data<float>();

    string path_conv2_b = proto_path + "Convolution2_b.npy";
    cnpy::NpyArray conv2_b = cnpy::npy_load(path_conv2_b);
    float *conv2_b_data = conv2_b.data<float>();

    conv2 = new conv_layer(50, pool1->channels, pool1->pooled_h, pool1->pooled_w, 5, 5, 1, 1, 0, 0, conv2_w_data,
                           conv2_b_data);
    conv2->type = LAYER_TYPE_CONV;
    /****************************池化层2****************************/
    pool2 = new pooling_layer(conv2->conved_c, conv2->conved_h, conv2->conved_w,
                              2, 2, 2, 2, 0, 0);
    pool2->type = LAYER_TYPE_POOL;
    /****************************全连接层1****************************/
    string path_fc1_w = proto_path + "InnerProduct1_w.npy";
    cnpy::NpyArray fc1_w = cnpy::npy_load(path_fc1_w);
    float *fc1_w_data = fc1_w.data<float>();

    string path_fc1_b = proto_path + "InnerProduct1_b.npy";
    cnpy::NpyArray fc1_b = cnpy::npy_load(path_fc1_b);
    float *fc1_b_data = fc1_b.data<float>();

    fc1 = new fc_layer(500, pool2->channels * pool2->pooled_h * pool2->pooled_w, fc1_w_data, fc1_b_data);
    fc1->type = LAYER_TYPE_FULLY_CONNECTED;
    /****************************RELU激活层1****************************/
    relu1 = new relu_layer(fc1->num_output);
    relu1->type = LAYER_TYPE_ACTIVATION;
    /****************************全连接层2****************************/
    string path_fc2_w = proto_path + "InnerProduct2_w.npy";
    cnpy::NpyArray fc2_w = cnpy::npy_load(path_fc2_w);
    float *fc2_w_data = fc2_w.data<float>();

    string path_fc2_b = proto_path + "InnerProduct2_b.npy";
    cnpy::NpyArray fc2_b = cnpy::npy_load(path_fc2_b);
    float *fc2_b_data = fc2_b.data<float>();

    fc2 = new fc_layer(10, relu1->count, fc2_w_data, fc2_b_data);
    fc2->type = LAYER_TYPE_FULLY_CONNECTED;
    /****************************Softmax层****************************/
    softmax = new softmax_layer(fc2->num_output);
    softmax->type = LAYER_TYPE_SOFTMAX;
}

net::~net() {
    delete softmax;
    delete fc2;
    delete relu1;
    delete fc1;
    delete pool2;
    delete conv2;
    delete pool1;
    delete conv1;
}

vector<float> net::forward(float *input_data) {
    assert(input_data != nullptr);
    vector<float> output_A, output_B;
    /****************************卷积层1****************************/
    int conved_size = conv1->conved_c * conv1->conved_h * conv1->conved_w;
    // 此处必须初始化为0，内积加中使用的是+=
    output_A.resize(conved_size);
    output_A.assign(conved_size, 0.0f);
    conv1->forward(input_data, output_A.data());
//    cerr << conved_size << endl;
//    for (int i = 0; i < conved_size; ++i) {
//        cerr << output_A[i] << endl;
//    }
    /****************************池化层1****************************/
    int pooled_size = pool1->channels * pool1->pooled_h * pool1->pooled_w;
    output_B.resize(pooled_size);
    output_B.assign(pooled_size, MINUS_FLT_MIN);
    pool1->forward(output_A.data(), output_B.data());
//    cerr << pooled_size << endl;
//    for (int i = 0; i < pooled_size; ++i) {
//        cerr << output_B[i] << endl;
//    }
    /****************************卷积层2****************************/
    conved_size = conv2->conved_c * conv2->conved_h * conv2->conved_w;
    // 此处必须初始化为0，内积加中使用的是+=
    output_A.resize(conved_size);
    output_A.assign(conved_size, 0.0f);
    conv2->forward(output_B.data(), output_A.data());
//    cerr << conved_size << endl;
//    for (int i = 0; i < conved_size; ++i) {
//        cerr << output_A[i] << endl;
//    }
    /****************************池化层2****************************/
    pooled_size = pool2->channels * pool2->pooled_h * pool2->pooled_w;
    output_B.resize(pooled_size);
    output_B.assign(pooled_size, MINUS_FLT_MIN);
    pool2->forward(output_A.data(), output_B.data());
//    cerr << pooled_size << endl;
//    for (int i = 0; i < pooled_size; ++i) {
//        cerr << output_B[i] << endl;
//    }
    /****************************全连接层1****************************/
    // 此处必须初始化为0，内积加中使用的是+=
    output_A.resize(fc1->num_output);
    output_A.assign(fc1->num_output, 0.0f);
    fc1->forward(output_B.data(), output_A.data());
//    cerr << fc1->num_output << endl;
//    for (int i = 0; i < fc1->num_output; ++i) {
//        cerr << output_A[i] << endl;
//    }
    /****************************RELU激活层1****************************/
    relu1->forward(output_A.data());
//    cerr << relu1->count << endl;
//    for (int i = 0; i < relu1->count; ++i) {
//        cerr << output_A[i] << endl;
//    }
    /****************************全连接层2****************************/
    output_B.resize(fc2->num_output);
    output_B.assign(fc2->num_output, 0.0f);
    fc2->forward(output_A.data(), output_B.data());
//    cerr << fc2->num_output << endl;
//    for (int i = 0; i < fc2->num_output; ++i) {
//        cerr << output_B[i] << endl;
//    }
    /****************************Softmax层****************************/
    softmax->forward(output_B.data());
//    cerr << softmax->count << endl;
//    for (int i = 0; i < softmax->count; ++i) {
//        cerr << output_B[i] << endl;
//    }
    return output_B;
}