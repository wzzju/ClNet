//
// Created by yuchen on 17-12-29.
//

#include <cassert>
#include "utility_cpu.h"
#include "layers/relu_layer.h"


relu_layer::relu_layer(int count) : count(count) {}

void relu_layer::forward(float *input, float *result) {
    assert(input != nullptr);
    activation_relu_cpu(input, count);
}