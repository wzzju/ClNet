//
// Created by yuchen on 17-12-29.
//
#include <cmath>
#include <cassert>
#include "utilities.h"
#include "layers/softmax_layer.h"

softmax_layer::softmax_layer(int count) : count(count) {}

void softmax_layer::forward(float *input, float *result) {
    assert(input != nullptr);
    float scaled = MINUS_FLT_MIN;
    for (int i = 0; i < count; ++i) {
        scaled = max(scaled, input[i]);
    }
    for (int i = 0; i < count; ++i) {
        input[i] -= scaled;
    }
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += std::exp(input[i]);
    }
    for (int i = 0; i < count; ++i) {
        input[i] = float(std::exp(input[i]) / sum);
    }
}