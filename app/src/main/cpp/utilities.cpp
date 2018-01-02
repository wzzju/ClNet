//
// Created by yuchen on 17-12-27.
//
#include <iostream>
#include "cnpy.h"
#include "utilities.h"

using namespace std;

/**
 * conv: 卷积权重矩阵的形状: output_channle x [channel*kernel_h*kernel_w]
 * im2col处理之后的原始图像矩阵的形状： [channel*kernel_h*kernel_w] x [卷积之后的图像的长和宽相乘 ]
 * @param data_im 输入图像矩阵
 * @param channels 输入图像矩阵的channel
 * @param height  输入图像矩阵的height
 * @param width 输入图像矩阵的width
 * @param kernel_h
 * @param kernel_w
 * @param pad_h
 * @param pad_w
 * @param stride_h
 * @param stride_w
 * @param data_col
 */
void im2col_cpu(const float *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                float *data_col) {
    //计算输出的size
    const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    //channel_size是每个输入feature map的size
    const int channel_size = height * width;
    //data_im是输入数据的指针，每遍历一次就移动channel_size的位移
    for (int channel = channels; channel--; data_im += channel_size) {
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                //逐行遍历卷积窗口的输入数据
                int input_row = -pad_h + kernel_row;
                //逐行遍历输出数据
                for (int output_rows = output_h; output_rows; output_rows--) {
                    //如果坐标超出输入数据的界限，一般出现这种情况是因为pad!=0
                    if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                        //逐列遍历输出数据，由于输入数据的行超出界限（补0)，对应的输出为0
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    } else {
                        //逐列遍历卷积窗口的输入数据
                        int input_col = -pad_w + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            //输入数据的行坐标和列坐标均没有超过界限
                            if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                //那么输出的值便等于输入的值
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                //如果输入列坐标超过界限，便置0
                                *(data_col++) = 0;
                            }
                            //输出列坐标移动（下一个卷积窗口了）
                            input_col += stride_w;
                        }
                    }
                    //输入行坐标移动（下一个卷积窗口了）
                    input_row += stride_h;
                }
            }
        }
    }
}

/**
 * 内积函数
 * @param mat_left
 * @param row_left
 * @param col_left
 * @param mat_right
 * @param row_right
 * @param col_right
 * @param result
 */
void inner_cpu(float *mat_left, int row_left, int col_left,
               float *mat_right, int row_right, int col_right,
               float *result) {
    assert(col_left == row_right);
    for (int i = 0; i < row_left; i++) {
        for (int j = 0; j < col_right; ++j) {
            for (int k = 0; k < col_left; ++k) {
                result[i * col_right + j] +=
                        mat_left[i * col_left + k] * mat_right[k * col_right + j];//+=，所以result必须要初始化为0
            }
        }
    }
}

void activation_relu(float *input, int count) {
    for (int i = 0; i < count; ++i) {
        input[i] = max(input[i], 0.0f);
    }
}

/**
 * 带偏置的内积函数
 * @param mat_left
 * @param row_left
 * @param col_left
 * @param mat_right
 * @param row_right
 * @param col_right
 * @param bias
 * @param result
 */
void inner_plus_b_cpu(float *mat_left, int row_left, int col_left,
                      float *mat_right, int row_right, int col_right,
                      float *bias, float *result) {
    assert(col_left == row_right);
    for (int i = 0; i < row_left; i++) {
        for (int j = 0; j < col_right; ++j) {
            for (int k = 0; k < col_left; ++k) {
                result[i * col_right + j] +=
                        mat_left[i * col_left + k] * mat_right[k * col_right + j];//+=，所以result必须要初始化为0
            }
            result[i * col_right + j] += bias[i];
        }
    }
}
