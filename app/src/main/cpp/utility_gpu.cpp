//
// Created by yuchen on 18-1-3.
//
#include <sstream>
#include <random>
#include <CL/cl.hpp>
#include <utility_cpu.h>
#include <helper.h>
#include <opencl/cl_objects.h>
#include <utility_gpu.h>

using namespace std;
using namespace cl;

void fillRandom(float *data, unsigned int width, unsigned height, unsigned long seed) {
    float *iptr = data;

    uniform_real_distribution<float> distribution(0, 100);
    default_random_engine generator(seed);


    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            iptr[j + i * width] = distribution(generator);
        }
    }
}

bool compare_layer(cl_objects &clObject, Buffer &gpu_mat, float *cpu_mat, int size) {
    bool cmp = false;
    float *mapped_memory = (float *) clObject.getQueues()[0][0].enqueueMapBuffer(gpu_mat,
                                                                                 CL_TRUE,
                                                                                 CL_MAP_READ, 0,
                                                                                 size *
                                                                                 sizeof(float));
    int i = 0;
    for (; i < size; ++i) {
        if (mapped_memory[i] != cpu_mat[i]) {
            LOGD("cpu_mat[%d] != gpu_mat[%d], %f != %f", i, i,
                 cpu_mat[i], mapped_memory[i]);
            break;
        }
    }
    if (i == size) {
        LOGD("Passed!");
        cmp = true;
        for (int j = 0; j < 10; ++j) {
            LOGD("cpu_mat[%d] == gpu_mat[%d], %f == %f", j, j,
                 cpu_mat[j], mapped_memory[j]);
        }
    } else {
        cmp = false;
        LOGD("Failed!");
    }
    clObject.getQueues()[0][0].enqueueUnmapMemObject(gpu_mat, mapped_memory);
    return cmp;
}

bool compare(float *gpuMatC, float *matA, float *matB, int heightA, int widthA, int widthB) {

    float *cpuMat = new float[widthB * heightA]();
    SCOPE_EXIT(delete[] cpuMat);
    {
        CostTimeHelper timeHelper("cpu inner");
        // cpu inner
        inner_cpu(matA, heightA, widthA, matB, widthA, widthB, cpuMat);
    }
    std::size_t length = heightA * widthB;
    for (int i = 0; i < length; ++i) {
        if (cpuMat[i] != gpuMatC[i]) {
            LOGD("cpuMat[%d] != gpuMatC[%d], %d != %d", i, i, cpuMat[i], gpuMatC[i]);
            return false;
        }
    }
    return true;
}

bool compare_plus_b(float *gpuMatC, float *matA, float *matB, float *bias, int heightA, int widthA,
                    int widthB) {

    float *cpuMat = new float[widthB * heightA]();
    SCOPE_EXIT(delete[] cpuMat);
    {
        CostTimeHelper timeHelper("cpu inner_pus_b");
        // cpu inner
        inner_plus_b_cpu(matA, heightA, widthA, matB, widthA, widthB, bias, cpuMat);
    }
    std::size_t length = heightA * widthB;
    for (int i = 0; i < length; ++i) {
        if (cpuMat[i] != gpuMatC[i]) {
            LOGD("cpuMat[%d] != gpuMatC[%d], %d != %d", i, i, cpuMat[i], gpuMatC[i]);
            return false;
        }
    }
    return true;
}

void test_relu(cl_objects &clObject, stringstream &strs) {
    cl_uint num = 64 * 64;
    float *input = new float[num]();
    SCOPE_EXIT(delete[] input);
    float *input_cpu = new float[num]();
    SCOPE_EXIT(delete[] input_cpu);

    fillRandom(input, num, 1, 643);
    copy(input, input + num, input_cpu);

    try {
        Buffer inputMemObj(clObject.getContexts()[0],
                           CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                           num * sizeof(float),
                           input);

        clObject.getRelu().kernel.setArg(0, inputMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;
        // NDRange(clObject.getInner().kernel_max_workgroup_size / 32, 32)
        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getRelu().kernel,
                                                        NullRange,
                                                        NDRange(num),
                                                        NullRange,
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();

        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The relu on GPU took %f s.\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

        clObject.getQueues()[0][0].enqueueReadBuffer(inputMemObj, CL_TRUE, 0,
                                                     num * sizeof(float), input);
    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

    /***************compare*******************/
    {
        CostTimeHelper timeHelper("cpu relu");
        activation_relu_cpu(input_cpu, num);
    }
    int i = 0;
    for (; i < num; ++i) {
        if (input_cpu[i] != input[i]) {
            LOGD("relued_cpu[%d] != relued_gpu[%d], %f != %f", i, i,
                 input_cpu[i], input[i]);
            break;
        }
    }
    if (i == num) {
        strs << "Passed!" << endl;
        LOGD("Passed!");
    } else {
        LOGD("Failed!");
        strs << "Failed!" << endl;
    }

}

void test_inner(cl_objects &clObject, stringstream &strs) {
    cl_uint heightA = 256;
    cl_uint widthA = 1024;
    cl_uint heightB = 1024;
    cl_uint widthB = 512;
    float *matrixA = new float[widthA * heightA]();
    SCOPE_EXIT(delete[] matrixA);
    float *matrixB = new float[widthB * heightB]();
    SCOPE_EXIT(delete[] matrixB);
    float *matrixC = new float[widthB * heightA]();
    SCOPE_EXIT(delete[] matrixC);

    fillRandom(matrixA, widthA, heightA, 643);
    fillRandom(matrixB, widthB, heightB, 991);

    try {
        Buffer matrixAMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthA * heightA * sizeof(float),
                             matrixA);
        Buffer matrixBMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthB * heightB * sizeof(float),
                             matrixB);

        Buffer matrixCMemObj(clObject.getContexts()[0],
                             CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             widthB * heightA * sizeof(float),
                             nullptr);

        clObject.getInner().kernel.setArg(0, matrixAMemObj);
        clObject.getInner().kernel.setArg(1, matrixBMemObj);
        clObject.getInner().kernel.setArg(2, widthA);
        clObject.getInner().kernel.setArg(3, matrixCMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;
        // NDRange(clObject.getInner().kernel_max_workgroup_size / 32, 32)
        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getInner().kernel,
                                                        NullRange,
                                                        NDRange(heightA, widthB),
                                                        NullRange,
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();

        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The inner on GPU took %f s.\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

        clObject.getQueues()[0][0].enqueueReadBuffer(matrixCMemObj, CL_TRUE, 0,
                                                     heightA * widthB * sizeof(float), matrixC);
    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

    if (compare(matrixC, matrixA, matrixB, heightA, widthA, widthB)) {
        LOGD("Passed!");
        strs << "Passed!" << endl;
    } else {
        LOGD("Failed!");
        strs << "Failed!" << endl;
    }
}

void test_inner_plus_b(cl_objects &clObject, stringstream &strs) {
    cl_uint heightA = 256;
    cl_uint widthA = 1024;
    cl_uint heightB = 1024;
    cl_uint widthB = 512;
    float *matrixA = new float[widthA * heightA]();
    SCOPE_EXIT(delete[] matrixA);
    float *matrixB = new float[widthB * heightB]();
    SCOPE_EXIT(delete[] matrixB);
    float *bias = new float[heightA]();
    SCOPE_EXIT(delete[] bias);
//    float *matrixC = new float[widthB * heightA]();
//    SCOPE_EXIT(delete[] matrixC);
    float *mapped_memory; // 内存映射方式读取Buffer

    fillRandom(matrixA, widthA, heightA, 643);
    fillRandom(matrixB, widthB, heightB, 991);
    fillRandom(bias, heightA, 1, 235);

    try {
        Buffer matrixAMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthA * heightA * sizeof(float),
                             matrixA);
        Buffer matrixBMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthB * heightB * sizeof(float),
                             matrixB);
        Buffer biasMemObj(clObject.getContexts()[0],
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          heightA * sizeof(float),
                          bias);

        Buffer matrixCMemObj(clObject.getContexts()[0],
                             CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             widthB * heightA * sizeof(float),
                             nullptr);

        clObject.getInner_plus_b().kernel.setArg(0, matrixAMemObj);
        clObject.getInner_plus_b().kernel.setArg(1, matrixBMemObj);
        clObject.getInner_plus_b().kernel.setArg(2, widthA);
        clObject.getInner_plus_b().kernel.setArg(3, biasMemObj);
        clObject.getInner_plus_b().kernel.setArg(4, matrixCMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;
        // NDRange(clObject.getInner().kernel_max_workgroup_size / 32, 32)
        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getInner_plus_b().kernel,
                                                        NullRange,
                                                        NDRange(heightA, widthB),
                                                        NullRange,
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();

        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The inner_pus_b on GPU took %f s.\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

//        clObject.getQueues()[0][0].enqueueReadBuffer(matrixCMemObj, CL_TRUE, 0,
//                                                     heightA * widthB * sizeof(float), matrixC);

        /***************Map the Buffer*******************/
        mapped_memory = (float *) clObject.getQueues()[0][0].enqueueMapBuffer(matrixCMemObj,
                                                                              CL_TRUE,
                                                                              CL_MAP_READ, 0,
                                                                              heightA * widthB *
                                                                              sizeof(float));
        /***************compare*******************/
        if (compare_plus_b(mapped_memory, matrixA, matrixB, bias, heightA, widthA, widthB)) {
            LOGD("Passed!");
            strs << "Passed!" << endl;
        } else {
            LOGD("Failed!");
            strs << "Failed!" << endl;
        }
        /***************Unmap the Buffer*******************/
        clObject.getQueues()[0][0].enqueueUnmapMemObject(matrixCMemObj, mapped_memory);
    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

//    if (compare_plus_b(matrixC, matrixA, matrixB, bias, heightA, widthA, widthB)) {
//        LOGD("Passed!");
//        strs << "Passed!" << endl;
//    } else {
//        LOGD("Failed!");
//        strs << "Failed!" << endl;
//    }
}

void test_im2col(cl_objects &clObject, stringstream &strs) {
    int channels = 6;
    int height = 64;
    int width = 64;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int kernel_h = 5;
    int kernel_w = 5;

    const int conved_h = (height + 2 * pad_h - kernel_h)
                         / stride_h + 1;
    const int conved_w = (width + 2 * pad_w - kernel_w)
                         / stride_w + 1;
    const int col_size = channels * kernel_h * kernel_w * conved_h * conved_w;

    float *data_im = new float[channels * height * width];
    SCOPE_EXIT(delete[] data_im);
//    float *data_col = new float[col_size];
//    SCOPE_EXIT(delete[] data_col);
    float *mapped_memory; // 内存映射方式读取Buffer

    fillRandom(data_im, channels * height, width, 257);

    /***************cpu im2col*******************/
    float *data_col_cpu = new float[col_size];
    SCOPE_EXIT(delete[] data_col_cpu);
    {
        CostTimeHelper timeHelper("cpu img2col");
        im2col_cpu(data_im, channels,
                   height, width, kernel_h, kernel_w,
                   pad_h, pad_w,
                   stride_h, stride_w,
                   data_col_cpu);
    }

    /***************gpu im2col*******************/
    try {
        Buffer imgMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         channels * height * width * sizeof(float),
                         data_im);

        Buffer colMemObj(clObject.getContexts()[0],
                         CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         col_size * sizeof(float),
                         nullptr);

        clObject.getImg2col().kernel.setArg(0, imgMemObj);
        clObject.getImg2col().kernel.setArg(1, channels);
        clObject.getImg2col().kernel.setArg(2, height);
        clObject.getImg2col().kernel.setArg(3, width);
        clObject.getImg2col().kernel.setArg(4, kernel_h);
        clObject.getImg2col().kernel.setArg(5, kernel_w);
        clObject.getImg2col().kernel.setArg(6, pad_h);
        clObject.getImg2col().kernel.setArg(7, pad_w);
        clObject.getImg2col().kernel.setArg(8, stride_h);
        clObject.getImg2col().kernel.setArg(9, stride_w);
        clObject.getImg2col().kernel.setArg(10, conved_h);
        clObject.getImg2col().kernel.setArg(11, conved_w);
        clObject.getImg2col().kernel.setArg(12, colMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;

        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getImg2col().kernel,
                                                        NullRange,
                                                        NDRange(channels * conved_h * conved_w),
                                                        NDRange(clObject.getImg2col().kernel_max_workgroup_size),
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();

        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The im2col on GPU took %f s.\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

//        clObject.getQueues()[0][0].enqueueReadBuffer(colMemObj, CL_TRUE, 0,
//                                                     col_size * sizeof(float), data_col);
        /***************Map the Buffer*******************/
        mapped_memory = (float *) clObject.getQueues()[0][0].enqueueMapBuffer(colMemObj,
                                                                              CL_TRUE,
                                                                              CL_MAP_READ, 0,
                                                                              col_size *
                                                                              sizeof(float));
        /***************compare*******************/
        int i = 0;
        for (; i < col_size; ++i) {
            if (mapped_memory[i] != data_col_cpu[i]) {
                LOGD("data_col_cpu[%d] != data_col_gpu[%d], %f != %f", i, i,
                     data_col_cpu[i], mapped_memory[i]);
                break;
            }
        }
        if (i == col_size) {
            strs << "Passed!" << endl;
            LOGD("Passed!");
        } else {
            LOGD("Failed!");
            strs << "Failed!" << endl;
        }
        /***************Unmap the Buffer*******************/
        clObject.getQueues()[0][0].enqueueUnmapMemObject(colMemObj, mapped_memory);

    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

//    int i = 0;
//    for (; i < col_size; ++i) {
//        if (data_col[i] != data_col_cpu[i]) {
//            LOGD("data_col_cpu[%d] != data_col[%d], %f != %f", i, i,
//                 data_col_cpu[i], data_col[i]);
//            break;
//        }
//    }
//    if (i == col_size) {
//        strs << "Passed!" << endl;
//        LOGD("Passed!");
//    } else {
//        LOGD("Failed!");
//        strs << "Failed!" << endl;
//    }
}


void test_max_pool(cl_objects &clObject, stringstream &strs) {
    int channels = 6;
    int height = 64;
    int width = 64;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 1;
    int stride_w = 1;
    int kernel_h = 5;
    int kernel_w = 5;

    int pooled_h =
            static_cast<int>(ceil(static_cast<float>(height + 2 * pad_h - kernel_h) / stride_h)) +
            1;
    int pooled_w =
            static_cast<int>(ceil(static_cast<float>(width + 2 * pad_w - kernel_w) / stride_w)) + 1;

    int pooled_size = channels * pooled_h * pooled_w;

    float *data_im = new float[channels * height * width];
    SCOPE_EXIT(delete[] data_im);
    float *data_pooled = new float[pooled_size];
    SCOPE_EXIT(delete[] data_pooled);

    fillRandom(data_im, channels * height, width, 123);

    try {
        Buffer imgMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         channels * height * width * sizeof(float),
                         data_im);

        Buffer pooledMemObj(clObject.getContexts()[0],
                            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                            pooled_size * sizeof(float),
                            nullptr);

        clObject.getMaxPool().kernel.setArg(0, pooled_size);
        clObject.getMaxPool().kernel.setArg(1, imgMemObj);
        clObject.getMaxPool().kernel.setArg(2, channels);
        clObject.getMaxPool().kernel.setArg(3, height);
        clObject.getMaxPool().kernel.setArg(4, width);
        clObject.getMaxPool().kernel.setArg(5, pad_h);
        clObject.getMaxPool().kernel.setArg(6, pad_w);
        clObject.getMaxPool().kernel.setArg(7, kernel_h);
        clObject.getMaxPool().kernel.setArg(8, kernel_w);
        clObject.getMaxPool().kernel.setArg(9, stride_h);
        clObject.getMaxPool().kernel.setArg(10, stride_w);
        clObject.getMaxPool().kernel.setArg(11, pooled_h);
        clObject.getMaxPool().kernel.setArg(12, pooled_w);
        clObject.getMaxPool().kernel.setArg(13, pooledMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;

        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getMaxPool().kernel,
                                                        NullRange,
                                                        NDRange(pooled_h * pooled_w),
                                                        NDRange(clObject.getImg2col().kernel_max_workgroup_size),
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();

        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The max_pool on GPU took %f s.\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

        clObject.getQueues()[0][0].enqueueReadBuffer(pooledMemObj, CL_TRUE, 0,
                                                     pooled_size * sizeof(float), data_pooled);
    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

    /***************compare*******************/
    float *data_pooled_cpu = new float[pooled_size];
    SCOPE_EXIT(delete[] data_pooled_cpu);
    {
        CostTimeHelper timeHelper("cpu max_pool");
        max_pool_cpu(data_im, channels,
                     height, width,
                     pad_h, pad_w, kernel_h, kernel_w,
                     stride_h, stride_w,
                     pooled_h, pooled_w,
                     data_pooled_cpu);
    }
    int i = 0;
    for (; i < pooled_size; ++i) {
        if (data_pooled[i] != data_pooled_cpu[i]) {
            LOGD("data_pooled_cpu[%d] != data_pooled[%d], %f != %f", i, i,
                 data_pooled_cpu[i], data_pooled[i]);
            break;
        }
    }
    if (i == pooled_size) {
        strs << "Passed!" << endl;
        LOGD("Passed!");
    } else {
        LOGD("Failed!");
        strs << "Failed!" << endl;
    }
}