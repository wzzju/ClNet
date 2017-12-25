#include <jni.h>
#include <string>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <CL/cl.h>
#include "clnet.h"
#include "cl_log.h"

using namespace std;

/**
 * 根据OpenCL程序路径返回其字符串源码。
 * @param inputPath
 * @return
 */
inline string load_program(const char *inputPath) {
    ifstream programFile(inputPath);
    string programString{istreambuf_iterator<char>(programFile),
                         istreambuf_iterator<char>()};
    return programString;
}

JNIEXPORT jstring JNICALL
CLNET(runCL)(JNIEnv *env, jobject instance, jstring path_) {
    const char *path = env->GetStringUTFChars(path_, 0);
    LOGD("The OpenCL program path: %s", path);
    string res_str = "\n**************************END**************************\n";

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

    /* Identify a platform */
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, nullptr);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Access a device */
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create the context */
    cl_context context;
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Read program file and place content into buffer */
    const char *program_buffer = load_program(path).c_str();

    /* Create program from file */
    cl_program program;
    program = clCreateProgramWithSource(context, 1,
                                        &program_buffer, 0, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    /* Build program */
    err = clBuildProgram(program, 0, 0, "-O3 -cl-mad-enable -cl-fast-relaxed-math", 0,
                         0);

    /* Get OpenCL program build log */
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_length = 0;

        err = clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                0,
                0,
                &log_length);
        CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

        vector<char> log_buf = vector<char>(log_length);

        err = clGetProgramBuildInfo(
                program,
                device,
                CL_PROGRAM_BUILD_LOG,
                log_length,
                (void *) log_buf.data(),
                0);
        CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

        LOGE("Failed to build the OpenCL program!\nBuild log: %s", log_buf.data());

        return nullptr;
    }

    /* Create kernel for the mat_vec_mult function */
    cl_kernel kernel = clCreateKernel(program, "matvec_mult", &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create CL buffers to hold input and output data */
    cl_mem mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                              CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, mat, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                              CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, vec, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * 4, nullptr, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create kernel arguments from the CL buffers */
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &mat_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &vec_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &res_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create a CL command queue for the device*/
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Enqueue the command queue to the device */
    size_t work_units_per_kernel = 4; /* 4 work-units per kernel */
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &work_units_per_kernel,
                                 nullptr, 0, nullptr, nullptr);

    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Read the result */
    err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float) * 4,
                              result, 0, nullptr, nullptr);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Test the result */
    if ((result[0] == correct[0]) && (result[1] == correct[1])
        && (result[2] == correct[2]) && (result[3] == correct[3])) {
        LOGD("Matrix-vector multiplication is executed successfully!\n");
    } else {
        LOGD("Fail to execute matrix-vector multiplication!\n");
    }

    /* Deallocate resources */
    err = clReleaseMemObject(mat_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseMemObject(vec_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseMemObject(res_buff);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseKernel(kernel);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseCommandQueue(queue);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseProgram(program);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);
    err = clReleaseContext(context);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    LOGD("OpenCL test is finished!");

    env->ReleaseStringUTFChars(path_, path);

    return env->NewStringUTF(res_str.c_str());
}

JNIEXPORT void JNICALL
CLNET(deviceQuery)(JNIEnv *env, jobject instance) {
    vector<cl_platform_id> platforms;
    cl_uint num_platforms;
    cl_int err;
    err = clGetPlatformIDs(5, nullptr, &num_platforms);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    LOGD("I have platforms: %d\n", num_platforms); //本人计算机上显示为2，有intel和nvidia两个平台  

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
        LOGD("The size of extension data is: %d\n", ext_size);

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