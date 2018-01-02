#include <jni.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
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
    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, path);

    stringstream strs;
    strs << endl << "/*" << __FUNCTION__ << "*/" << endl;;

    /* Data and buffers */
    float mat[16], vec[4], result[4];
    float correct[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    /* Initialize data to be processed by the kernel */
    for (int i = 0; i < 16; i++) {
        mat[i] = i * 2.0f;
    }

    for (int i = 0; i < 4; i++) {
        vec[i] = i * 3.0f;
        correct[0] += mat[i] * vec[i];
        correct[1] += mat[i + 4] * vec[i];
        correct[2] += mat[i + 8] * vec[i];
        correct[3] += mat[i + 12] * vec[i];
    }

    cl_int err;
    /* Create CL buffers to hold input and output data */
    cl_mem mat_buff = clCreateBuffer(clObject.getContexts()[0], CL_MEM_READ_ONLY |
                                                                CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * 16, mat, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem vec_buff = clCreateBuffer(clObject.getContexts()[0], CL_MEM_READ_ONLY |
                                                                CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * 4, vec, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem res_buff = clCreateBuffer(clObject.getContexts()[0], CL_MEM_WRITE_ONLY,
                                     sizeof(float) * 4, nullptr, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(clObject.getMatvec().kernel, 0, sizeof(cl_mem), &mat_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    err = clSetKernelArg(clObject.getMatvec().kernel, 1, sizeof(cl_mem), &vec_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    err = clSetKernelArg(clObject.getMatvec().kernel, 2, sizeof(cl_mem), &res_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Enqueue the command queue to the device */
    {
        CostTimeHelper timeHelper("cl_test");
        size_t work_units_per_kernel = 4; /* 4 work-units per kernel */
        err = clEnqueueNDRangeKernel(clObject.getQueues()[0][0], clObject.getMatvec().kernel, 1,
                                     nullptr, &work_units_per_kernel,
                                     nullptr, 0, nullptr, nullptr);
        CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

        /* Read the result */
        err = clEnqueueReadBuffer(clObject.getQueues()[0][0], res_buff, CL_TRUE, 0,
                                  sizeof(float) * 4,
                                  result, 0, nullptr, nullptr);
        CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    }

    /* Test the result */
    if ((result[0] == correct[0]) && (result[1] == correct[1])
        && (result[2] == correct[2]) && (result[3] == correct[3])) {
        LOGD("Matrix-vector multiplication is executed successfully!\n");
        strs << "Matrix-vector multiplication is executed successfully!" << endl;
    } else {
        LOGD("Fail to execute matrix-vector multiplication!\n");
        strs << "Fail to execute matrix-vector multiplication!" << endl;
    }

    /* Deallocate resources */
    err = clReleaseMemObject(mat_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseMemObject(vec_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseMemObject(res_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    env->ReleaseStringUTFChars(path_, path);
    // strs.str("");// 清空stringstream内容
    // strs.clear();// 清空stream的状态（比如出错状态）
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
        LOGD("%lu : %f", i, loaded_data[i]);
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
        if (strstr(ext_data.data(), icd_ext) != nullptr)
            platform_index = i;
        LOGD("Platform_index = %d", platform_index);
        /* Display whether ICD extension is supported */
        if (platform_index > -1)
            LOGD("Platform %d supports the %s extension.\n",
                 platform_index, icd_ext);
    }

    if (platform_index <= -1)
        LOGD("No platforms support the %s extension.\n", icd_ext);

}