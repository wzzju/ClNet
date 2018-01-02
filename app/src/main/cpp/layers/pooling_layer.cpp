//
// Created by yuchen on 17-12-28.
//
#include <cmath>
#include <cassert>
#include "utilities.h"
#include "layers/pooling_layer.h"

using namespace std;

pooling_layer::pooling_layer(int channels, int input_h, int input_w,
                             int kernel_h, int kernel_w, int stride_h,
                             int stride_w, int pad_h, int pad_w) : channels(channels),
                                                                   input_h(input_h), input_w(input_w),
                                                                   kernel_h(kernel_h), kernel_w(kernel_w),
                                                                   stride_h(stride_h), stride_w(stride_w),
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
    for (int c = 0; c < channels; ++c) {
        for (int ph = 0; ph < pooled_h; ++ph) {
            for (int pw = 0; pw < pooled_w; ++pw) {
                // 要pooling的窗口
                int hstart = ph * stride_h - pad_h;
                int wstart = pw * stride_w - pad_w;
                int hend = min(hstart + kernel_h, input_h);
                int wend = min(wstart + kernel_w, input_w);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                //对每张图片来说
                const int pool_index = ph * pooled_w + pw;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                        const int index = h * input_w + w;
                        if (input[index] > pooled_res[pool_index]) {
                            // 循环求得最大值
                            pooled_res[pool_index] = input[index];
                        }
                    }
                }
            }
        }
        // 计算偏移量，进入下一张图的index起始地址
        input += input_w * input_h;
        pooled_res += pooled_h * pooled_w;
    }
}
