//
// Created by yuchen on 17-12-29.
//
#include <cassert>
#include <algorithm>
#include "utilities.h"
#include "layers/fc_layer.h"

using namespace std;

fc_layer::fc_layer(int num_output, int num_input, float *_W, float *_bias) :
        num_output(num_output), num_input(num_input) {
    int size_w = num_output * num_input;
    W = new float[size_w];
    bias = new float[num_output];
    copy(_W, _W + size_w, W);
    copy(_bias, _bias + num_output, bias);
//    memcpy(W, _W, size_w * sizeof(float));
//    memcpy(bias, _bias, num_output * sizeof(float));
}

fc_layer::~fc_layer() {
    delete[] bias;
    delete[] W;
}

void fc_layer::forward(float *input, float *fced_res) {
    assert(input != nullptr && fced_res != nullptr);
    inner_plus_b_cpu(W, num_output, num_input, input, num_input, 1, bias, fced_res);
}