#include <jni.h>
#include <string>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <vector>
#include <CL/cl.hpp>
#include "utility_gpu.h"
#include "helper.h"
#include "cnpy.h"
#include "opencl/cl_log.h"
#include "opencl/cl_objects.h"
#include "net.h"
#include "clnet.h"

using namespace std;
using namespace cl;
/****************************准备网络****************************/
static net m_net("/data/local/tmp/lenet/");

void test_matmul(cl_objects &clObject, stringstream &strs) {
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

    try {
        Buffer matrixAMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthA * heightA * sizeof(cl_int),
                             matrixA);
        Buffer matrixBMemObj(clObject.getContexts()[0],
                             CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             widthB * heightB * sizeof(cl_int),
                             matrixB);

        Buffer matrixCMemObj(clObject.getContexts()[0],
                             CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                             widthB * heightA * sizeof(cl_int),
                             nullptr);

        clObject.getMatmul().kernel.setArg(0, widthA);
        clObject.getMatmul().kernel.setArg(1, widthB);
        clObject.getMatmul().kernel.setArg(2, matrixAMemObj);
        clObject.getMatmul().kernel.setArg(3, matrixBMemObj);
        clObject.getMatmul().kernel.setArg(4, matrixCMemObj);

        Event exeEvt;
        cl_ulong executionStart, executionEnd;

        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getMatmul().kernel,
                                                        NullRange,
                                                        NDRange(heightA, widthB),
                                                        NDRange(clObject.getMatmul().kernel_max_workgroup_size /
                                                                32, 32),
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();
        // let's understand how long it took?
        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("Execution the matrix-matrix multiplication took %lu.%lu s\n",
             (executionEnd - executionStart) / 1000000000,
             (executionEnd - executionStart) % 1000000000);

        clObject.getQueues()[0][0].enqueueReadBuffer(matrixCMemObj, CL_TRUE, 0,
                                                     heightA * widthB * sizeof(cl_int), matrixC);
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
    SCOPE_EXIT(env->ReleaseStringUTFChars(path_, path));

    stringstream strs;
    strs << endl << "/*" << __FUNCTION__ << "*/" << endl;

    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, path);
    /****************************Begin to test matmul****************************/
    test_matmul(clObject, strs);
    /****************************End to test matmul****************************/

    return env->NewStringUTF(strs.str().c_str());
}

JNIEXPORT void JNICALL
CLNET(runNpy)(JNIEnv *env, jobject instance, jstring dir_) {
    const char *dir = env->GetStringUTFChars(dir_, 0);
    ostringstream os;
    os << dir << "Convolution1_w.npy";
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

    os << dir << "Convolution1_b.npy";
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
    try {
        vector<Platform> platforms;
        cl::Platform::get(&platforms);
        LOGD("Detect %zu platform(s).\n", platforms.size());
        cl_int platform_index = -1;
        const char icd_ext[] = "cl_khr_icd";
        for (cl_int i = 0; i < platforms.size(); i++) {
            vector<Device> devices;
            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            LOGD("The platform %d has %zu devices(CPUs&GPUs).\n", i, devices.size());
            for (cl_int j = 0; j < devices.size(); j++) {
                string device_name = devices[j].getInfo<CL_DEVICE_NAME>();
                LOGD("Platform %d, device %d name: %s", i, j, device_name.c_str());
            }
            string p_name = platforms[i].getInfo<CL_PLATFORM_NAME>();
            LOGD("Platform %d name: %s\n", i, p_name.c_str());
            string vendor = platforms[i].getInfo<CL_PLATFORM_VENDOR>();
            LOGD("Platform %d vendor: %s\n", i, vendor.c_str());
            string version = platforms[i].getInfo<CL_PLATFORM_VERSION>();
            LOGD("Platform %d version: %s\n", i, version.c_str());
            string profile = platforms[i].getInfo<CL_PLATFORM_PROFILE>();
            LOGD("Platform %d full profile or embeded profile? : %s\n", i, profile.c_str());
            string ext_data = platforms[i].getInfo<CL_PLATFORM_EXTENSIONS>();
            LOGD("The size of extension data is: %zu\n", ext_data.size());
            LOGD("Platform %d supports extensions: %s\n", i, ext_data.c_str());
            /* Look for ICD extension */
            if (ext_data.find(icd_ext) != string::npos)
                platform_index = i;
            /* Display whether ICD extension is supported */
            if (platform_index > -1)
                LOGD("Platform %d supports the %s extension.\n",
                     platform_index, icd_ext);
        }
        if (platform_index <= -1)
            LOGD("No platforms support the %s extension.\n", icd_ext);

    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

}