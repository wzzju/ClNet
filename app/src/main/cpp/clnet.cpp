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
static net m_net;

JNIEXPORT void JNICALL
CLNET(initNet)(JNIEnv *env, jobject instance, jstring weightPath_,
               jstring clPath_, jboolean useGPU) {
    const char *weightPath = env->GetStringUTFChars(weightPath_, 0);
    const char *clPath = env->GetStringUTFChars(clPath_, 0);

    m_net.init(weightPath, clPath, useGPU);

    env->ReleaseStringUTFChars(weightPath_, weightPath);
    env->ReleaseStringUTFChars(clPath_, clPath);
}

JNIEXPORT jfloatArray JNICALL
CLNET(inference)(JNIEnv *env, jobject instance,
                 jfloatArray data_) {
    jfloat *data = env->GetFloatArrayElements(data_, NULL);
    SCOPE_EXIT(env->ReleaseFloatArrayElements(data_, data, 0));
    /****************************前向推断****************************/
    vector<float> result;
    {
        CostTimeHelper timeHelper("inference");
        result = m_net.forward(data);
    }
    jfloatArray resultArr = env->NewFloatArray(result.size());
    env->SetFloatArrayRegion(resultArr, 0, result.size(), result.data());

    return resultArr;
}

JNIEXPORT jstring JNICALL
CLNET(runCL)(JNIEnv *env, jobject instance, jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);
    SCOPE_EXIT(env->ReleaseStringUTFChars(path_, path));

    stringstream strs;
    strs << endl << "/*" << __FUNCTION__ << "*/" << endl;

    cl_objects &clObject = cl_objects::getCLObject(CL_DEVICE_TYPE_GPU, path);
    /****************************Begin to test OpenCL kernels****************************/
    test_relu(clObject, strs);
    test_inner(clObject, strs);
    test_inner_plus_b(clObject, strs);
    test_im2col(clObject, strs);
    test_max_pool(clObject, strs);
    /****************************End to test OpenCL kernels****************************/

    return env->NewStringUTF(strs.str().c_str());
}

JNIEXPORT void JNICALL
CLNET(runNpy)(JNIEnv *env, jobject instance, jstring dir_) {
    const char *dir = env->GetStringUTFChars(dir_, 0);
    SCOPE_EXIT(env->ReleaseStringUTFChars(dir_, dir));
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