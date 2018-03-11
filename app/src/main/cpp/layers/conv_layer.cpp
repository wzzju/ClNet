//
// Created by yuchen on 17-12-27.
//
#include <cassert>
#include <algorithm>
#include <CL/cl.hpp>
#include <opencl/cl_log.h>
#include <utility_cpu.h>
#include <layers/conv_layer.h>

using namespace std;
using namespace cl;

conv_layer::conv_layer(int conved_c, int input_c, int input_h, int input_w, int kernel_h,
                       int kernel_w, int stride_h,
                       int stride_w, int pad_h, int pad_w, float *_W, float *_bias, bool use_gpu,
                       cl_objects &clObject)
        : conved_c(conved_c),
          input_c(input_c),
          input_h(input_h),
          input_w(input_w),
          kernel_h(kernel_h),
          kernel_w(kernel_w),
          stride_h(stride_h),
          stride_w(stride_w), pad_h(pad_h),
          pad_w(pad_w), W(nullptr), bias(nullptr) {
    conved_h = (input_h + 2 * pad_h - kernel_h) / stride_h + 1;
    conved_w = (input_w + 2 * pad_w - kernel_w) / stride_w + 1;
    int size_w = conved_c * input_c * kernel_h * kernel_w;
    if (use_gpu) {
        try {
            cl_W = Buffer(clObject.getContexts()[0],
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          size_w * sizeof(float),
                          _W);
            cl_bias = Buffer(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             conved_c * sizeof(float), _bias);
        } catch (cl::Error err) {
            LOGE("ERROR: %s\n", err.what());
        }
    } else {
        W = new float[size_w];
        bias = new float[conved_c];
        copy(_W, _W + size_w, W);
        copy(_bias, _bias + conved_c, bias);
        // memcpy(W, _W, size_w * sizeof(float));
        // memcpy(bias, _bias, conved_c * sizeof(float));
    }
}

conv_layer::~conv_layer() {
    delete[] bias;
    delete[] W;
}

//conv_res = new float[conved_c * output_channel_size];
void conv_layer::forward_cpu(float *input, float *conved_res) {
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

void conv_layer::forward_gpu(cl_objects &clObject, Buffer &input, Buffer &conved_res) {
    const int col_size = input_c * kernel_h * kernel_w * conved_h * conved_w;
    Buffer colMemObj(clObject.getContexts()[0],
                     CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                     col_size * sizeof(float),
                     nullptr);
    clObject.getImg2col().kernel.setArg(0, input);
    clObject.getImg2col().kernel.setArg(1, input_c);
    clObject.getImg2col().kernel.setArg(2, input_h);
    clObject.getImg2col().kernel.setArg(3, input_w);
    clObject.getImg2col().kernel.setArg(4, kernel_h);
    clObject.getImg2col().kernel.setArg(5, kernel_w);
    clObject.getImg2col().kernel.setArg(6, pad_h);
    clObject.getImg2col().kernel.setArg(7, pad_w);
    clObject.getImg2col().kernel.setArg(8, stride_h);
    clObject.getImg2col().kernel.setArg(9, stride_w);
    clObject.getImg2col().kernel.setArg(10, conved_h);
    clObject.getImg2col().kernel.setArg(11, conved_w);
    clObject.getImg2col().kernel.setArg(12, colMemObj);
    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getImg2col().kernel,
                                                    NullRange,
                                                    NDRange(input_c * conved_h * conved_w),
                                                    NDRange(clObject.getImg2col().kernel_max_workgroup_size),
                                                    nullptr,
                                                    nullptr);
    clObject.getQueues()[0][0].flush();
    clObject.getQueues()[0][0].finish();

    int same_dim = input_c * kernel_h * kernel_w;
    int output_channel_size = conved_h * conved_w;
    clObject.getInner_plus_b().kernel.setArg(0, cl_W);
    clObject.getInner_plus_b().kernel.setArg(1, colMemObj);
    clObject.getInner_plus_b().kernel.setArg(2, same_dim);
    clObject.getInner_plus_b().kernel.setArg(3, cl_bias);
    clObject.getInner_plus_b().kernel.setArg(4, conved_res);
    clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getInner_plus_b().kernel,
                                                    NullRange,
                                                    NDRange(conved_c, output_channel_size),
                                                    NullRange,
                                                    nullptr,
                                                    nullptr);
    clObject.getQueues()[0][0].flush();
    clObject.getQueues()[0][0].finish();
}
