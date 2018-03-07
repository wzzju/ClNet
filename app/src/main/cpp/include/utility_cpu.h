//
// Created by yuchen on 17-12-27.
//

#ifndef CLNET_UTILITY_CPU_H
#define CLNET_UTILITY_CPU_H

#include <string>

#define MINUS_FLT_MIN         -3.40282e+38f

// inline 函数对编译器而言必须是可见的，以便它能够在调用点内展开该函数。
// 与非inline函数不同的是，inline函数必须在调用该函数的每个文本文件中定义。
// 建议把inline函数的定义放到头文件中。
inline float min(float lhs, float rhs) {
    return lhs < rhs ? lhs : rhs;
}

inline float max(float lhs, float rhs) {
    return lhs > rhs ? lhs : rhs;
}

inline int min(int lhs, int rhs) {
    return lhs < rhs ? lhs : rhs;
}

inline int max(int lhs, int rhs) {
    return lhs > rhs ? lhs : rhs;
}

/**
 * true: a >= 0 && a > b
 * @param a
 * @param b
 * @return
 */
inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

void im2col_cpu(const float *data_im, const int channels,
                const int height, const int width, const int kernel_h, const int kernel_w,
                const int pad_h, const int pad_w,
                const int stride_h, const int stride_w,
                float *data_col);

void max_pool_cpu(float *input,
                  int channels,
                  int input_h, int input_w,
                  int pad_h, int pad_w,
                  int kernel_h, int kernel_w,
                  int stride_h, int stride_w,
                  int pooled_h, int pooled_w,
                  float *pooled_res);

void inner_plus_b_cpu(float *mat_left, int row_left, int col_left,
                      float *mat_right, int row_right, int col_right,
                      float *bias, float *result);

void inner_cpu(float *mat_left, int row_left, int col_left,
               float *mat_right, int row_right, int col_right,
               float *result);

void activation_relu_cpu(float *input, int count);

#endif //CLNET_UTILITY_CPU_H
