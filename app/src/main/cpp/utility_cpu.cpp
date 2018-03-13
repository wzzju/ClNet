//
// Created by yuchen on 17-12-27.
//
#include <iostream>
#include <cmath>
#include <cnpy.h>
#include <utility_cpu.h>

using namespace std;

void spmv_csr_cpu(const int num_rows,
                  const int *ptr,
                  const int *indices,
                  const float *data,
                  const float *bias,
                  const float *vec, float *out) {
    for (int row = 0; row < num_rows; row++) {
        float temp = 0;
        int start_row = ptr[row];
        int end_row = ptr[row + 1];
        for (int j = start_row; j < end_row; j++)
            temp += data[j] * vec[indices[j]];
        out[row] = temp + bias[row];
    }
}

/**
 * im2col_cpu将c个通道的卷积层输入图像转化为c个通道的矩阵，矩阵的行值为卷积核高*卷积核宽，
 * 也就是说，矩阵的单列表征了卷积核操作一次处理的小窗口图像信息；而矩阵的列值为卷积层输出
 * 单通道图像高*卷积层输出单通道图像宽，表示一共要处理多少个小窗口。
 * im2col_cpu接收13个参数，分别为输入数据指针(data_im)，卷积操作处理的一个卷积组的通道
 * 数(channels)，输入图像的高(height)与宽(width)，原始卷积核的高(kernel_h)与宽(kernel_w)，
 * 输入图像高(pad_h)与宽(pad_w)方向的pad，卷积操作高(stride_h)与宽(stride_w)方向的步长，
 * 输出矩阵数据指针(data_col)。
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
    const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;//计算卷积层输出图像的高
    const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;//计算卷积层输出图像的宽
    //channel_size是每个输入feature map的size
    const int channel_size = height * width;//计算卷积层输入单通道图像的数据容量
    /*第一个for循环表示输出的矩阵通道数和卷积层输入图像通道是一样的，每次处理一个输入通道的信息*/
    //data_im是输入数据的指针，每遍历一次就移动channel_size的位移
    for (int channel = channels; channel--; data_im += channel_size) {
        /*第二个和第三个for循环表示了输出单通道矩阵的某一列，同时体现了输出单通道矩阵的行数*/
        for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
                //逐行遍历卷积窗口的输入数据
                int input_row = -pad_h + kernel_row;//在这里找到卷积核中的某一行在输入图像中的第一个操作区域的行索引
                //逐行遍历输出数据
                /*第四个和第五个for循环表示了输出单通道矩阵的某一行，同时体现了输出单通道矩阵的列数*/
                for (int output_rows = output_h; output_rows; output_rows--) {
                    //如果坐标超出输入数据的界限，一般出现这种情况是因为pad!=0
                    if (!is_a_ge_zero_and_a_lt_b(input_row,
                                                 height)) {//如果计算得到的输入图像的行值索引小于零或者大于输入图像的高(该行为pad)
                        //逐列遍历输出数据，由于输入数据的行超出界限（补0)，对应的输出为0
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;//那么将该行在输出的矩阵上的位置置为0
                        }
                    } else {
                        //逐列遍历卷积窗口的输入数据
                        int input_col = -pad_w + kernel_col;//在这里找到卷积核中的某一列在输入图像中的第一个操作区域的列索引
                        for (int output_col = output_w; output_col; output_col--) {
                            //输入数据的行坐标和列坐标均没有超过界限
                            if (is_a_ge_zero_and_a_lt_b(input_col,
                                                        width)) {//如果计算得到的输入图像的列值索引大于等于于零或者小于输入图像的宽(该列不是pad)
                                //那么输出的值便等于输入的值
                                *(data_col++) = data_im[input_row * width +
                                                        input_col];//将输入特征图上对应的区域放到输出矩阵上
                            } else {//否则，计算得到的输入图像的列值索引小于零或者大于输入图像的宽(该列为pad)
                                *(data_col++) = 0;//将该行该列在输出矩阵上的位置置为0
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

void max_pool_cpu(float *input,
                  int channels,
                  int input_h, int input_w,
                  int pad_h, int pad_w,
                  int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int pooled_h, int pooled_w,
                  float *pooled_res) {
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
                pooled_res[pool_index] = input[hstart * input_w + wstart];
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
                        mat_left[i * col_left + k] *
                        mat_right[k * col_right + j];//+=，所以result必须要初始化为0
//                result[i * col_right + j] = isnan(result[i * col_right + j]) ? 0 : result[
//                        i * col_right + j];
            }
        }
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
                        mat_left[i * col_left + k] *
                        mat_right[k * col_right + j];//+=，所以result必须要初始化为0
            }
            result[i * col_right + j] += bias[i];
//            result[i * col_right + j] = isnan(result[i * col_right + j]) ? 0 : result[
//                    i * col_right + j];
        }
    }
}

void activation_relu_cpu(float *input, int count) {
    for (int i = 0; i < count; ++i) {
        input[i] = max(input[i], 0.0f);
    }
}