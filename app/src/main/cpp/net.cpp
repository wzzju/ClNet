//
// Created by yuchen on 17-12-29.
//
#include <iostream>
#include <vector>
#include <cassert>
#include <CL/cl.hpp>
#include <utility_gpu.h>
#include <helper.h>
#include <opencl/cl_objects.h>
#include <opencl/cl_log.h>
#include <cnpy.h>
#include <utility_cpu.h>
#include <layers/conv_layer.h>
#include <layers/pooling_layer.h>
#include <layers/softmax_layer.h>
#include <layers/fc_layer.h>
#include <layers/relu_layer.h>
#include <net.h>

using namespace std;
using namespace cl;
#if LENET
net::net(string weight_path, string cl_path, bool use_gpu) : softmax(nullptr), fc2(nullptr),
                                                             relu1(nullptr), fc1(nullptr),
                                                             pool2(nullptr), conv2(nullptr),
                                                             pool1(nullptr), conv1(nullptr),
                                                             use_gpu(use_gpu), cl_path(cl_path) {
    LOGD("Init the net ...");
    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, cl_path.c_str());
/****************************卷积层1****************************/
    string path_conv1_w = weight_path + "conv1_w.npy";
    cnpy::NpyArray conv1_w = cnpy::npy_load(path_conv1_w);
    float *conv1_w_data = conv1_w.data<float>();

    string path_conv1_b = weight_path + "conv1_b.npy";
    cnpy::NpyArray conv1_b = cnpy::npy_load(path_conv1_b);
    float *conv1_b_data = conv1_b.data<float>();

    conv1 = new conv_layer(20, INPUT_C, INPUT_H, INPUT_W, 5, 5, 1, 1, 0, 0, conv1_w_data,
                           conv1_b_data, use_gpu, clObject);
    conv1->type = LAYER_TYPE_CONV;
/****************************池化层1****************************/
    pool1 = new pooling_layer(conv1->conved_c, conv1->conved_h, conv1->conved_w,
                              2, 2, 2, 2, 0, 0);
    pool1->type = LAYER_TYPE_POOL;
/****************************卷积层2****************************/
    string path_conv2_w = weight_path + "conv2_w.npy";
    cnpy::NpyArray conv2_w = cnpy::npy_load(path_conv2_w);
    float *conv2_w_data = conv2_w.data<float>();

    string path_conv2_b = weight_path + "conv2_b.npy";
    cnpy::NpyArray conv2_b = cnpy::npy_load(path_conv2_b);
    float *conv2_b_data = conv2_b.data<float>();

    conv2 = new conv_layer(50, pool1->channels, pool1->pooled_h, pool1->pooled_w, 5, 5, 1, 1, 0, 0,
                           conv2_w_data, conv2_b_data, use_gpu, clObject);
    conv2->type = LAYER_TYPE_CONV;
/****************************池化层2****************************/
    pool2 = new pooling_layer(conv2->conved_c, conv2->conved_h, conv2->conved_w,
                              2, 2, 2, 2, 0, 0);
    pool2->type = LAYER_TYPE_POOL;
/****************************全连接层1****************************/
    string path_fc1_w = weight_path + "ip1_w.npy";
    cnpy::NpyArray fc1_w = cnpy::npy_load(path_fc1_w);
    float *fc1_w_data = fc1_w.data<float>();

    string path_fc1_b = weight_path + "ip1_b.npy";
    cnpy::NpyArray fc1_b = cnpy::npy_load(path_fc1_b);
    float *fc1_b_data = fc1_b.data<float>();

    fc1 = new fc_layer(500, pool2->channels * pool2->pooled_h * pool2->pooled_w, fc1_w_data,
                       fc1_b_data, use_gpu, clObject);
    fc1->type = LAYER_TYPE_FULLY_CONNECTED;
/****************************RELU激活层1****************************/
    relu1 = new relu_layer(fc1->num_output);
    relu1->type = LAYER_TYPE_ACTIVATION;
/****************************全连接层2****************************/
    string path_fc2_w = weight_path + "ip2_w.npy";
    cnpy::NpyArray fc2_w = cnpy::npy_load(path_fc2_w);
    float *fc2_w_data = fc2_w.data<float>();

    string path_fc2_b = weight_path + "ip2_b.npy";
    cnpy::NpyArray fc2_b = cnpy::npy_load(path_fc2_b);
    float *fc2_b_data = fc2_b.data<float>();

    fc2 = new fc_layer(10, relu1->count, fc2_w_data, fc2_b_data, use_gpu, clObject);
    fc2->type = LAYER_TYPE_FULLY_CONNECTED;
/****************************Softmax层****************************/
    softmax = new softmax_layer(fc2->num_output);
    softmax->type = LAYER_TYPE_SOFTMAX;
}
#endif

#if ALEXNET

net::net(std::string weight_path, std::string cl_path, bool use_gpu) : conv1(nullptr),
                                                                       relu1(nullptr),
                                                                       pool1(nullptr),
                                                                       conv2(nullptr),
                                                                       relu2(nullptr),
                                                                       pool2(nullptr),
                                                                       conv3(nullptr),
                                                                       relu3(nullptr),
                                                                       conv4(nullptr),
                                                                       relu4(nullptr),
                                                                       conv5(nullptr),
                                                                       relu5(nullptr),
                                                                       pool5(nullptr), fc6(nullptr),
                                                                       relu6(nullptr), fc7(nullptr),
                                                                       relu7(nullptr), fc8(nullptr),
                                                                       softmax(nullptr),
                                                                       use_gpu(use_gpu),
                                                                       cl_path(cl_path) {
    LOGD("Init the net ...");
    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, cl_path.c_str());
/****************************卷积层1****************************/
    string path_conv1_w = weight_path + "conv1_w.npy";
    cnpy::NpyArray conv1_w = cnpy::npy_load(path_conv1_w);
    float *conv1_w_data = conv1_w.data<float>();

    string path_conv1_b = weight_path + "conv1_b.npy";
    cnpy::NpyArray conv1_b = cnpy::npy_load(path_conv1_b);
    float *conv1_b_data = conv1_b.data<float>();

    conv1 = new conv_layer(96, INPUT_C, INPUT_H, INPUT_W, 11, 11, 4, 4, 0, 0, conv1_w_data,
                           conv1_b_data, use_gpu, clObject);
    conv1->type = LAYER_TYPE_CONV;
    /****************************RELU激活层1****************************/
    int count = conv1->conved_c * conv1->conved_h * conv1->conved_w;
    relu1 = new relu_layer(count);
    relu1->type = LAYER_TYPE_ACTIVATION;
    /****************************池化层1****************************/
    pool1 = new pooling_layer(conv1->conved_c, conv1->conved_h, conv1->conved_w,
                              3, 3, 2, 2, 0, 0);
    pool1->type = LAYER_TYPE_POOL;
    /****************************卷积层2****************************/
    string path_conv2_w = weight_path + "conv2_w.npy";
    cnpy::NpyArray conv2_w = cnpy::npy_load(path_conv2_w);
    float *conv2_w_data = conv2_w.data<float>();

    string path_conv2_b = weight_path + "conv2_b.npy";
    cnpy::NpyArray conv2_b = cnpy::npy_load(path_conv2_b);
    float *conv2_b_data = conv2_b.data<float>();

    conv2 = new conv_layer(256, pool1->channels, pool1->pooled_h, pool1->pooled_w, 5, 5, 1, 1, 2, 2,
                           conv2_w_data, conv2_b_data, use_gpu, clObject);
    conv2->type = LAYER_TYPE_CONV;
    /****************************RELU激活层2****************************/
    count = conv2->conved_c * conv2->conved_h * conv2->conved_w;
    relu2 = new relu_layer(count);
    relu2->type = LAYER_TYPE_ACTIVATION;
    /****************************池化层2****************************/
    pool2 = new pooling_layer(conv2->conved_c, conv2->conved_h, conv2->conved_w,
                              3, 3, 2, 2, 0, 0);
    pool2->type = LAYER_TYPE_POOL;
    /****************************卷积层3****************************/
    string path_conv3_w = weight_path + "conv3_w.npy";
    cnpy::NpyArray conv3_w = cnpy::npy_load(path_conv3_w);
    float *conv3_w_data = conv3_w.data<float>();

    string path_conv3_b = weight_path + "conv3_b.npy";
    cnpy::NpyArray conv3_b = cnpy::npy_load(path_conv3_b);
    float *conv3_b_data = conv3_b.data<float>();

    conv3 = new conv_layer(384, pool2->channels, pool2->pooled_h, pool2->pooled_w, 3, 3, 1, 1, 1, 1,
                           conv3_w_data, conv3_b_data, use_gpu, clObject);
    conv3->type = LAYER_TYPE_CONV;
    /****************************RELU激活层3****************************/
    count = conv3->conved_c * conv3->conved_h * conv3->conved_w;
    relu3 = new relu_layer(count);
    relu3->type = LAYER_TYPE_ACTIVATION;
    /****************************卷积层4****************************/
    string path_conv4_w = weight_path + "conv4_w.npy";
    cnpy::NpyArray conv4_w = cnpy::npy_load(path_conv4_w);
    float *conv4_w_data = conv4_w.data<float>();

    string path_conv4_b = weight_path + "conv4_b.npy";
    cnpy::NpyArray conv4_b = cnpy::npy_load(path_conv4_b);
    float *conv4_b_data = conv4_b.data<float>();

    conv4 = new conv_layer(384, conv3->conved_c, conv3->conved_h, conv3->conved_w, 3, 3, 1, 1, 1, 1,
                           conv4_w_data, conv4_b_data, use_gpu, clObject);
    conv4->type = LAYER_TYPE_CONV;
    /****************************RELU激活层4****************************/
    count = conv4->conved_c * conv4->conved_h * conv4->conved_w;
    relu4 = new relu_layer(count);
    relu4->type = LAYER_TYPE_ACTIVATION;
    /****************************卷积层5****************************/
    string path_conv5_w = weight_path + "conv5_w.npy";
    cnpy::NpyArray conv5_w = cnpy::npy_load(path_conv5_w);
    float *conv5_w_data = conv5_w.data<float>();

    string path_conv5_b = weight_path + "conv5_b.npy";
    cnpy::NpyArray conv5_b = cnpy::npy_load(path_conv5_b);
    float *conv5_b_data = conv5_b.data<float>();

    conv5 = new conv_layer(256, conv4->conved_c, conv4->conved_h, conv4->conved_w, 3, 3, 1, 1, 1, 1,
                           conv5_w_data, conv5_b_data, use_gpu, clObject);
    conv5->type = LAYER_TYPE_CONV;
    /****************************RELU激活层5****************************/
    count = conv5->conved_c * conv5->conved_h * conv5->conved_w;
    relu5 = new relu_layer(count);
    relu5->type = LAYER_TYPE_ACTIVATION;
    /****************************池化层5****************************/
    pool5 = new pooling_layer(conv5->conved_c, conv5->conved_h, conv5->conved_w,
                              3, 3, 2, 2, 0, 0);
    pool5->type = LAYER_TYPE_POOL;
    /****************************全连接层6****************************/
    string path_fc6_w = weight_path + "fc6_w.npy";
    cnpy::NpyArray fc6_w = cnpy::npy_load(path_fc6_w);
    float *fc6_w_data = fc6_w.data<float>();

    string path_fc6_b = weight_path + "fc6_b.npy";
    cnpy::NpyArray fc6_b = cnpy::npy_load(path_fc6_b);
    float *fc6_b_data = fc6_b.data<float>();

    fc6 = new fc_layer(4096, pool5->channels * pool5->pooled_h * pool5->pooled_w, fc6_w_data,
                       fc6_b_data, use_gpu, clObject);
    fc6->type = LAYER_TYPE_FULLY_CONNECTED;
    /****************************RELU激活层6****************************/
    relu6 = new relu_layer(fc6->num_output);
    relu6->type = LAYER_TYPE_ACTIVATION;
    /****************************全连接层7****************************/
    string path_fc7_w = weight_path + "fc7_w.npy";
    cnpy::NpyArray fc7_w = cnpy::npy_load(path_fc7_w);
    float *fc7_w_data = fc7_w.data<float>();

    string path_fc7_b = weight_path + "fc7_b.npy";
    cnpy::NpyArray fc7_b = cnpy::npy_load(path_fc7_b);
    float *fc7_b_data = fc7_b.data<float>();

    fc7 = new fc_layer(4096, relu6->count, fc7_w_data,
                       fc7_b_data, use_gpu, clObject);
    fc7->type = LAYER_TYPE_FULLY_CONNECTED;
    /****************************RELU激活层7****************************/
    relu7 = new relu_layer(fc7->num_output);
    relu7->type = LAYER_TYPE_ACTIVATION;
    /****************************全连接层8****************************/
    string path_fc8_w = weight_path + "fc8_w.npy";
    cnpy::NpyArray fc8_w = cnpy::npy_load(path_fc8_w);
    float *fc8_w_data = fc8_w.data<float>();

    string path_fc8_b = weight_path + "fc8_b.npy";
    cnpy::NpyArray fc8_b = cnpy::npy_load(path_fc8_b);
    float *fc8_b_data = fc8_b.data<float>();

    fc8 = new fc_layer(1000, relu7->count, fc8_w_data,
                       fc8_b_data, use_gpu, clObject);
    fc8->type = LAYER_TYPE_FULLY_CONNECTED;
    /****************************Softmax层****************************/
    softmax = new softmax_layer(fc8->num_output);
    softmax->type = LAYER_TYPE_SOFTMAX;

}

#endif

net::~net() {
    LOGD("Clean the net ...");
#if LENET
    delete softmax;
    delete fc2;
    delete relu1;
    delete fc1;
    delete pool2;
    delete conv2;
    delete pool1;
    delete conv1;
#endif
#if ALEXNET
    delete conv1;
    delete relu1;
    delete pool1;
    delete conv2;
    delete relu2;
    delete pool2;
    delete conv3;
    delete relu3;
    delete conv4;
    delete relu4;
    delete conv5;
    delete relu5;
    delete pool5;
    delete fc6;
    delete relu6;
    delete fc7;
    delete relu7;
    delete fc8;
#endif
}

vector<float> net::forward(float *input_data) {
    assert(input_data != nullptr);
#if LENET
    if (use_gpu) {
        CostTimeHelper timeHelper("GPU Inference");
        try {
            cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, cl_path.c_str());
            /****************************卷积层1 GPU****************************/
            int input_size = conv1->input_c * conv1->input_h * conv1->input_w;
            Buffer input_conv1(clObject.getContexts()[0],
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               input_size * sizeof(float),
                               input_data);
            int conved_size = conv1->conved_c * conv1->conved_h * conv1->conved_w;
            Buffer output_conv1(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv1->forward_gpu(clObject, input_conv1, output_conv1);
            /****************************池化层1 GPU****************************/
            int pooled_size = pool1->channels * pool1->pooled_h * pool1->pooled_w;
            Buffer output_pool1(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                pooled_size * sizeof(float),
                                nullptr);
            pool1->forward_gpu(clObject, output_conv1, output_pool1);

            /****************************卷积层2 GPU****************************/
            conved_size = conv2->conved_c * conv2->conved_h * conv2->conved_w;
            Buffer output_conv2(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv2->forward_gpu(clObject, output_pool1, output_conv2);
            /****************************池化层2 GPU****************************/
            pooled_size = pool2->channels * pool2->pooled_h * pool2->pooled_w;
            Buffer output_pool2(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                pooled_size * sizeof(float),
                                nullptr);
            pool2->forward_gpu(clObject, output_conv2, output_pool2);
            /****************************全连接层1 GPU****************************/
            Buffer output_fc1(clObject.getContexts()[0],
                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              fc1->num_output * sizeof(float),
                              nullptr);
            fc1->forward_gpu(clObject, output_pool2, output_fc1);
            /****************************RELU激活层1 GPU****************************/
            relu1->forward_gpu(clObject, output_fc1, output_fc1);
            /****************************全连接层2 GPU****************************/
            Buffer output_fc2(clObject.getContexts()[0],
                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              fc2->num_output * sizeof(float),
                              nullptr);
            fc2->forward_gpu(clObject, output_fc1, output_fc2);
            /****************************Softmax层****************************/
            float *mapped_memory = (float *) clObject.getQueues()[0][0].enqueueMapBuffer(output_fc2,
                                                                                         CL_TRUE,
                                                                                         CL_MAP_READ,
                                                                                         0,
                                                                                         fc2->num_output *
                                                                                         sizeof(float));

            vector<float> result(mapped_memory, mapped_memory + fc2->num_output);

            clObject.getQueues()[0][0].enqueueUnmapMemObject(output_fc2, mapped_memory);

            softmax->forward_cpu(result.data());
            return result;
        } catch (cl::Error err) {
            LOGE("ERROR: %s\n", err.what());
        }
    } else {
        CostTimeHelper timeHelper("CPU Inference");
        vector<float> output_A, output_B;
        /****************************卷积层1****************************/
        int conved_size = conv1->conved_c * conv1->conved_h * conv1->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv1->forward_cpu(input_data, output_A.data());
        /****************************池化层1****************************/
        int pooled_size = pool1->channels * pool1->pooled_h * pool1->pooled_w;
        output_B.resize(pooled_size);
        pool1->forward_cpu(output_A.data(), output_B.data());
        /****************************卷积层2****************************/
        conved_size = conv2->conved_c * conv2->conved_h * conv2->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv2->forward_cpu(output_B.data(), output_A.data());
        /****************************池化层2****************************/
        pooled_size = pool2->channels * pool2->pooled_h * pool2->pooled_w;
        output_B.resize(pooled_size);
        pool2->forward_cpu(output_A.data(), output_B.data());
        /****************************全连接层1****************************/
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(fc1->num_output);
        output_A.assign(fc1->num_output, 0.0f);
        fc1->forward_cpu(output_B.data(), output_A.data());
        /****************************RELU激活层1****************************/
        relu1->forward_cpu(output_A.data());
        /****************************全连接层2****************************/
        output_B.resize(fc2->num_output);
        output_B.assign(fc2->num_output, 0.0f);
        fc2->forward_cpu(output_A.data(), output_B.data());
        /****************************Softmax层****************************/
        softmax->forward_cpu(output_B.data());
        return output_B;
    }
#endif

#if ALEXNET
    if (use_gpu) {
        CostTimeHelper timeHelper("GPU Inference");
        try {
            cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, cl_path.c_str());
            /****************************卷积层1 GPU****************************/
            int input_size = conv1->input_c * conv1->input_h * conv1->input_w;
            Buffer input_conv1(clObject.getContexts()[0],
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               input_size * sizeof(float),
                               input_data);
            int conved_size = conv1->conved_c * conv1->conved_h * conv1->conved_w;
            Buffer output_conv1(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv1->forward_gpu(clObject, input_conv1, output_conv1);
            /****************************RELU激活层1 GPU****************************/
            relu1->forward_gpu(clObject, output_conv1, output_conv1);
            /****************************池化层1 GPU****************************/
            int pooled_size = pool1->channels * pool1->pooled_h * pool1->pooled_w;
            Buffer output_pool1(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                pooled_size * sizeof(float),
                                nullptr);
            pool1->forward_gpu(clObject, output_conv1, output_pool1);
            /****************************卷积层2 GPU****************************/
            conved_size = conv2->conved_c * conv2->conved_h * conv2->conved_w;
            Buffer output_conv2(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv2->forward_gpu(clObject, output_pool1, output_conv2);
            /****************************RELU激活层2 GPU****************************/
            relu2->forward_gpu(clObject, output_conv2, output_conv2);
            /****************************池化层2 GPU****************************/
            pooled_size = pool2->channels * pool2->pooled_h * pool2->pooled_w;
            Buffer output_pool2(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                pooled_size * sizeof(float),
                                nullptr);
            pool2->forward_gpu(clObject, output_conv2, output_pool2);
            /****************************卷积层3 GPU****************************/
            conved_size = conv3->conved_c * conv3->conved_h * conv3->conved_w;
            Buffer output_conv3(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv3->forward_gpu(clObject, output_pool2, output_conv3);
            /****************************RELU激活层3 GPU****************************/
            relu3->forward_gpu(clObject, output_conv3, output_conv3);
            /****************************卷积层4 GPU****************************/
            conved_size = conv4->conved_c * conv4->conved_h * conv4->conved_w;
            Buffer output_conv4(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv4->forward_gpu(clObject, output_conv3, output_conv4);
            /****************************RELU激活层4 GPU****************************/
            relu4->forward_gpu(clObject, output_conv4, output_conv4);
            /****************************卷积层5 GPU****************************/
            conved_size = conv5->conved_c * conv5->conved_h * conv5->conved_w;
            Buffer output_conv5(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                conved_size * sizeof(float),
                                nullptr);
            conv5->forward_gpu(clObject, output_conv4, output_conv5);
            /****************************RELU激活层5 GPU****************************/
            relu5->forward_gpu(clObject, output_conv5, output_conv5);
            /****************************池化层5 GPU****************************/
            pooled_size = pool5->channels * pool5->pooled_h * pool5->pooled_w;
            Buffer output_pool5(clObject.getContexts()[0],
                                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                pooled_size * sizeof(float),
                                nullptr);
            pool5->forward_gpu(clObject, output_conv5, output_pool5);
            /****************************全连接层6 GPU****************************/
            Buffer output_fc6(clObject.getContexts()[0],
                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              fc6->num_output * sizeof(float),
                              nullptr);
            fc6->forward_gpu(clObject, output_pool5, output_fc6);
            /****************************RELU激活层6 GPU****************************/
            relu6->forward_gpu(clObject, output_fc6, output_fc6);
            /****************************全连接层7 GPU****************************/
            Buffer output_fc7(clObject.getContexts()[0],
                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              fc7->num_output * sizeof(float),
                              nullptr);
            fc7->forward_gpu(clObject, output_fc6, output_fc7);
            /****************************RELU激活层7 GPU****************************/
            relu7->forward_gpu(clObject, output_fc7, output_fc7);
            /****************************全连接层8 GPU****************************/
            Buffer output_fc8(clObject.getContexts()[0],
                              CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                              fc8->num_output * sizeof(float),
                              nullptr);
            fc8->forward_gpu(clObject, output_fc7, output_fc8);
            /****************************Softmax层****************************/
            float *mapped_memory = (float *) clObject.getQueues()[0][0].enqueueMapBuffer(output_fc8,
                                                                                         CL_TRUE,
                                                                                         CL_MAP_READ,
                                                                                         0,
                                                                                         fc8->num_output *
                                                                                         sizeof(float));

            vector<float> result(mapped_memory, mapped_memory + fc8->num_output);

            clObject.getQueues()[0][0].enqueueUnmapMemObject(output_fc8, mapped_memory);

            softmax->forward_cpu(result.data());
            return result;
        } catch (cl::Error err) {
            LOGE("ERROR: %s\n", err.what());
        }
    } else {
        CostTimeHelper timeHelper("CPU Inference");
        vector<float> output_A, output_B;
        /****************************卷积层1****************************/
        int conved_size = conv1->conved_c * conv1->conved_h * conv1->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv1->forward_cpu(input_data, output_A.data());
        /****************************RELU激活层1****************************/
        relu1->forward_cpu(output_A.data());
        /****************************池化层1****************************/
        int pooled_size = pool1->channels * pool1->pooled_h * pool1->pooled_w;
        output_B.resize(pooled_size);
        pool1->forward_cpu(output_A.data(), output_B.data());
        /****************************卷积层2****************************/
        conved_size = conv2->conved_c * conv2->conved_h * conv2->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv2->forward_cpu(output_B.data(), output_A.data());
        /****************************RELU激活层2****************************/
        relu2->forward_cpu(output_A.data());
        /****************************池化层2****************************/
        pooled_size = pool2->channels * pool2->pooled_h * pool2->pooled_w;
        output_B.resize(pooled_size);
        pool2->forward_cpu(output_A.data(), output_B.data());
        /****************************卷积层3****************************/
        conved_size = conv3->conved_c * conv3->conved_h * conv3->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv3->forward_cpu(output_B.data(), output_A.data());
        /****************************RELU激活层3****************************/
        relu3->forward_cpu(output_A.data());
        /****************************卷积层4****************************/
        conved_size = conv4->conved_c * conv4->conved_h * conv4->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_B.resize(conved_size);
        output_B.assign(conved_size, 0.0f);
        conv4->forward_cpu(output_A.data(), output_B.data());
        /****************************RELU激活层4****************************/
        relu4->forward_cpu(output_B.data());
        /****************************卷积层5****************************/
        conved_size = conv5->conved_c * conv5->conved_h * conv5->conved_w;
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(conved_size);
        output_A.assign(conved_size, 0.0f);
        conv5->forward_cpu(output_B.data(), output_A.data());
        /****************************RELU激活层5****************************/
        relu5->forward_cpu(output_A.data());
        /****************************池化层5****************************/
        pooled_size = pool5->channels * pool5->pooled_h * pool5->pooled_w;
        output_B.resize(pooled_size);
        pool5->forward_cpu(output_A.data(), output_B.data());
        /****************************全连接层6****************************/
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(fc6->num_output);
        output_A.assign(fc6->num_output, 0.0f);
        fc6->forward_cpu(output_B.data(), output_A.data());
        /****************************RELU激活层6****************************/
        relu6->forward_cpu(output_A.data());
        /****************************全连接层7****************************/
        // 此处必须初始化为0，内积加中使用的是+=
        output_B.resize(fc7->num_output);
        output_B.assign(fc7->num_output, 0.0f);
        fc7->forward_cpu(output_A.data(), output_B.data());
        /****************************RELU激活层7****************************/
        relu7->forward_cpu(output_B.data());
        /****************************全连接层8****************************/
        // 此处必须初始化为0，内积加中使用的是+=
        output_A.resize(fc8->num_output);
        output_A.assign(fc8->num_output, 0.0f);
        fc8->forward_cpu(output_B.data(), output_A.data());
        /****************************Softmax层****************************/
        softmax->forward_cpu(output_A.data());
        return output_A;
    }
#endif
}
