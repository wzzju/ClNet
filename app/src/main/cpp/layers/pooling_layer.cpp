//
// Created by yuchen on 17-12-28.
//
#include <cmath>
#include <cassert>
#include "utility_cpu.h"
#include "layers/pooling_layer.h"

using namespace std;

pooling_layer::pooling_layer(int channels, int input_h, int input_w,
                             int kernel_h, int kernel_w, int stride_h,
                             int stride_w, int pad_h, int pad_w) : channels(channels),
                                                                   input_h(input_h),
                                                                   input_w(input_w),
                                                                   kernel_h(kernel_h),
                                                                   kernel_w(kernel_w),
                                                                   stride_h(stride_h),
                                                                   stride_w(stride_w),
                                                                   pad_h(pad_h), pad_w(pad_w) {
    // 计算pooling之后得到的高度和宽度
    //static_cast 显示强制转换 ceil:返回大于或者等于指定表达式的最小整数
    pooled_h = static_cast<int>(ceil(static_cast<float>(
                                             input_h + 2 * pad_h - kernel_h) / stride_h)) + 1;
    pooled_w = static_cast<int>(ceil(static_cast<float>(
                                             input_w + 2 * pad_w - kernel_w) / stride_w)) + 1;

    if (pad_h || pad_w) {
        // 存在padding的时候，确保最后一个pooling区域开始的地方是在图像内，否则去掉最后一部分
        if ((pooled_h - 1) * stride_h >= input_h + pad_h) {
            --pooled_h;
        }
        if ((pooled_w - 1) * stride_w >= input_w + pad_w) {
            --pooled_w;
        }
    }
}

// pooled_res矩阵应该初始化为一个小值，如：MINUS_FLT_MIN
void pooling_layer::forward(float *input, float *pooled_res) {
    assert(input != nullptr && pooled_res != nullptr);
    max_pool_cpu(input, channels, input_h, input_w, pad_h, pad_w, kernel_h, kernel_w, stride_h,
                 stride_w, pooled_h, pooled_w, pooled_res);
}
