#include <jni.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
#include <opencl/cl_objects.h>
#include "utility_gpu.h"
#include "helper.h"
#include "cnpy.h"
#include "opencl/cl_log.h"
#include "opencl/cl_objects.h"
#include "net.h"
#include "clnet.h"

using namespace std;

/****************************准备网络****************************/
static net m_net("/data/local/tmp/lenet/");

JNIEXPORT jfloatArray JNICALL
CLNET(inference)(JNIEnv *env, jobject instance,
                 jfloatArray data_) {
    jfloat *data = env->GetFloatArrayElements(data_, NULL);

    /****************************前向推断****************************/
    vector<float> result;
    {
        CostTimeHelper timeHelper("inference");
        result = m_net.forward(data);
    }
    jfloatArray resultArr = env->NewFloatArray(result.size());
    env->SetFloatArrayRegion(resultArr, 0, result.size(), result.data());

    env->ReleaseFloatArrayElements(data_, data, 0);

    return resultArr;
}

JNIEXPORT jstring JNICALL
CLNET(runCL)(JNIEnv *env, jobject instance, jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);

    stringstream strs;
    strs << endl << "/*" << __FUNCTION__ << "*/" << endl;

    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, path);
    /****************************Begin to est utility_gpu.cpp****************************/
    cl_uint heightA = HEIGHT_G;
    cl_uint widthA = WIDTH_G;
    cl_uint heightB = HEIGHT_G;
    cl_uint widthB = WIDTH_G;
    // allocate memory for input and output matrices
    // based on whatever matrix theory i know.
    cl_int *matrixA = new cl_int[widthA * heightA]();
    SCOPE_EXIT(delete[] matrixA);
    cl_int *matrixB = new cl_int[widthB * heightB]();
    SCOPE_EXIT(delete[] matrixB);
    cl_int *matrixC = new cl_int[widthB * heightA]();
    SCOPE_EXIT(delete[] matrixC);

    fillRandom(matrixA, widthA, heightA, 643);
    fillRandom(matrixB, widthB, heightB, 991);

    cl_int err;
    cl_mem matrixAMemObj = clCreateBuffer(clObject.getContexts()[0],
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          widthA * heightA * sizeof(cl_int),
                                          matrixA,
                                          &err);
    SCOPE_EXIT(clReleaseMemObject(matrixAMemObj));
    cl_mem matrixBMemObj = clCreateBuffer(clObject.getContexts()[0],
                                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          widthB * heightB * sizeof(cl_int),
                                          matrixB,
                                          &err);
    SCOPE_EXIT(clReleaseMemObject(matrixBMemObj));
    cl_mem matrixCMemObj = clCreateBuffer(clObject.getContexts()[0],
                                          CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                          widthB * heightA * sizeof(cl_int),
                                          0,
                                          &err);
    SCOPE_EXIT(clReleaseMemObject(matrixCMemObj));

    clSetKernelArg(clObject.getMatmul().kernel, 0, sizeof(cl_int), (void *) &widthA);
    clSetKernelArg(clObject.getMatmul().kernel, 1, sizeof(cl_int), (void *) &widthB);
    clSetKernelArg(clObject.getMatmul().kernel, 2, sizeof(cl_mem), (void *) &matrixAMemObj);
    clSetKernelArg(clObject.getMatmul().kernel, 3, sizeof(cl_mem), (void *) &matrixBMemObj);
    clSetKernelArg(clObject.getMatmul().kernel, 4, sizeof(cl_mem), (void *) &matrixCMemObj);
    size_t globalThreads[] = {heightA, widthB};
    size_t localThreads[] = {clObject.getMatmul().kernel_max_workgroup_size / 16, 16};
    cl_event exeEvt;
    cl_ulong executionStart, executionEnd;

    err = clEnqueueNDRangeKernel(clObject.getQueues()[0][0],
                                 clObject.getMatmul().kernel,
                                 2,
                                 NULL,
                                 globalThreads,
                                 localThreads,
                                 0,
                                 NULL,
                                 &exeEvt);
    clWaitForEvents(1, &exeEvt);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    // let's understand how long it took?
    clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_START, sizeof(executionStart),
                            &executionStart, NULL);
    clGetEventProfilingInfo(exeEvt, CL_PROFILING_COMMAND_END, sizeof(executionEnd), &executionEnd,
                            NULL);
    clReleaseEvent(exeEvt);
    LOGD("Execution the matrix-matrix multiplication took %lu.%lu s\n",
         (executionEnd - executionStart) / 1000000000,
         (executionEnd - executionStart) % 1000000000);
    clEnqueueReadBuffer(clObject.getQueues()[0][0],
                        matrixCMemObj,
                        CL_TRUE,
                        0,
                        heightA * widthB * sizeof(cl_int),
                        matrixC,
                        0,
                        NULL,
                        NULL);
    if (compare(matrixC, matrixA, matrixB, heightA, widthA, widthB)) {
        LOGD("Passed!");
        strs << "Passed!" << endl;
    } else {
        LOGD("Failed!");
        strs << "Failed!" << endl;
    }
    /****************************End to est utility_gpu.cpp****************************/

    env->ReleaseStringUTFChars(path_, path);

    return env->NewStringUTF(strs.str().c_str());
}

JNIEXPORT void JNICALL
CLNET(runNpy)(JNIEnv *env, jobject instance, jstring dir_) {
    const char *dir = env->GetStringUTFChars(dir_, 0);
    ostringstream os;
    os << dir << "layer1-conv1_weight_0.npy";
    string path = os.str();
    os.str("");//清空os的stream内容

    cnpy::NpyArray arr = cnpy::npy_load(path);
    LOGD("Element size: %zu", arr.word_size);
    LOGD("Shape size: %zu", arr.shape.size());

    float *loaded_data = arr.data<float>();
    for (auto i = 0; i < arr.shape.size(); ++i) {
        LOGD("Dim %d: %zu", i, arr.shape[i]);
    }
    for (auto i = 0; i < arr.shape[0]; ++i) {
        for (auto j = 0; j < arr.shape[1]; ++j) {
            for (auto k = 0; k < arr.shape[2]; ++k) {
                for (auto l = 0; l < arr.shape[3]; ++l) {
                    auto index = i * arr.shape[1] * arr.shape[2] * arr.shape[3] +
                                 j * arr.shape[2] * arr.shape[3] + k * arr.shape[3] + l;
                    LOGD("%lu : %f", index, loaded_data[index]);
                }
            }
        }
    }

    os << dir << "layer1-conv1_bias_0.npy";
    path = os.str();
    os.str("");

    arr = cnpy::npy_load(path);
    LOGD("Element size: %zu", arr.word_size);
    LOGD("Shape size: %zu", arr.shape.size());

    loaded_data = arr.data<float>();
    for (auto i = 0; i < arr.shape.size(); ++i) {
        LOGD("Dim %d: %zu", i, arr.shape[i]);
    }

    for (auto i = 0; i < arr.shape[0]; ++i) {
        LOGD("%d : %f", i, loaded_data[i]);
    }

    env->ReleaseStringUTFChars(dir_, dir);
}

JNIEXPORT void JNICALL
CLNET(deviceQuery)(JNIEnv *env, jobject instance) {
    vector<cl_platform_id> platforms;
    cl_uint num_platforms;
    cl_int err;
    err = clGetPlatformIDs(5, nullptr, &num_platforms);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    LOGD("Detect %d platform(s).\n", num_platforms);

    platforms.resize(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    CHECK_ERRORS(err, __FILE__, __LINE__);

    vector<char> ext_data;
    size_t ext_size;
    cl_int platform_index = -1;
    const char icd_ext[] = "cl_khr_icd";
    for (cl_int i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, nullptr, &num_devices);
        CHECK_ERRORS(err, __FILE__, __LINE__);
        LOGD("The platform %d has %u devices(CPUs&GPUs).\n", i, num_devices);

        err = clGetPlatformInfo(platforms[i],
                                CL_PLATFORM_EXTENSIONS, 0, nullptr, &ext_size);
        CHECK_ERRORS(err, __FILE__, __LINE__);
        LOGD("The size of extension data is: %zu\n", ext_size);

        ext_data.resize(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,
                          ext_size, ext_data.data(), nullptr);
        LOGD("Platform %d supports extensions: %s\n", i, ext_data.data());

        size_t name_size;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          ext_size, nullptr, &name_size);
        vector<char> name(name_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          ext_size, name.data(), nullptr);
        LOGD("Platform %d name: %s\n", i, name.data());

        size_t vendor_size;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                          ext_size, nullptr, &vendor_size);
        vector<char> vendor(vendor_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                          ext_size, vendor.data(), nullptr);
        LOGD("Platform %d vendor: %s\n", i, vendor.data());

        size_t version_size;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
                          ext_size, nullptr, &version_size);
        vector<char> version(version_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
                          ext_size, version.data(), nullptr);
        LOGD("Platform %d version: %s\n", i, version.data());

        size_t profile_size;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,
                          ext_size, nullptr, &profile_size);
        vector<char> profile(profile_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,
                          ext_size, profile.data(), nullptr);
        LOGD("Platform %d full profile or embeded profile? : %s\n", i, profile.data());

        /* Look for ICD extension */
        string ext_str(ext_data.begin(), ext_data.end());
        if (ext_str.find(icd_ext) != string::npos)
            platform_index = i;
//        if (strstr(ext_data.data(), icd_ext) != nullptr)
//            platform_index = i;
        LOGD("Platform_index = %d", platform_index);
        /* Display whether ICD extension is supported */
        if (platform_index > -1)
            LOGD("Platform %d supports the %s extension.\n",
                 platform_index, icd_ext);
    }

    if (platform_index <= -1)
        LOGD("No platforms support the %s extension.\n", icd_ext);

}