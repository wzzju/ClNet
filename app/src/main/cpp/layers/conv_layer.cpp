//
// Created by yuchen on 17-12-27.
//
#include <cassert>
#include <algorithm>
#include "utilities.h"
#include "layers/conv_layer.h"

using namespace std;

conv_layer::conv_layer(int conved_c, int input_c, int input_h, int input_w, int kernel_h, int kernel_w, int stride_h,
                       int stride_w, int pad_h, int pad_w, float *_W, float *_bias) : conved_c(conved_c),
                                                                                      input_c(input_c),
                                                                                      input_h(input_h),
                                                                                      input_w(input_w),
                                                                                      kernel_h(kernel_h),
                                                                                      kernel_w(kernel_w),
                                                                                      stride_h(stride_h),
                                                                                      stride_w(stride_w), pad_h(pad_h),
                                                                                      pad_w(pad_w) {
    conved_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    conved_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int size_w = conved_c * input_c * kernel_h * kernel_w;
    W = new float[size_w];
    bias = new float[conved_c];
    copy(_W, _W + size_w, W);
    copy(_bias, _bias + conved_c, bias);
//    memcpy(W, _W, size_w * sizeof(float));
//    memcpy(bias, _bias, conved_c * sizeof(float));
}

conv_layer::~conv_layer() {
    delete[] bias;
    delete[] W;
}

//conv_res = new float[conved_c * output_channel_size];
void conv_layer::forward(float *input, float *conved_res) {
    assert(input != nullptr && conved_res != nullptr);
    // [channel*kernel_h*kernel_w] x [卷积之后的图像的长和宽相乘 ]
    const int col_size = input_c * kernel_h * kernel_w * conved_h * conved_w;
    float *data_col = new float[col_size];
    //进行img2col转换
    im2col_cpu(input, input_c, input_h, input_w,
               kernel_h, kernel_w,
               pad_h, pad_w,
               stride_h, stride_w, data_col);

    int same_dim = input_c * kernel_h * kernel_w;
    int output_channel_size = conved_h * conved_w;
    inner_plus_b_cpu(W, conved_c, same_dim,
                     data_col, same_dim, output_channel_size,
                     bias, conved_res);

    delete[] data_col;
}
