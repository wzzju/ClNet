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
 */
inline string load_program(const char *inputPath) {
    ifstream programFile(inputPath);
    string programString((istreambuf_iterator<char>(programFile)),
                         istreambuf_iterator<char>());

    return programString;
}

JNIEXPORT jstring JNICALL
CLNET(testCL)(JNIEnv *env, jobject /* this */) {
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
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Access a device */
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create the context */
    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Read program file and place content into buffer */
    const char *program_buffer = load_program(PROGRAM_FILE).c_str();

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

        return NULL;
    }

    /* Create kernel for the mat_vec_mult function */
    cl_kernel kernel = clCreateKernel(program, KERNEL_FUNC, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Create CL buffers to hold input and output data */
    cl_mem mat_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                              CL_MEM_COPY_HOST_PTR, sizeof(float) * 16, mat, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem vec_buff = clCreateBuffer(context, CL_MEM_READ_ONLY |
                                              CL_MEM_COPY_HOST_PTR, sizeof(float) * 4, vec, &err);
    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    cl_mem res_buff = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * 4, NULL, &err);
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
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &work_units_per_kernel,
                                 NULL, 0, NULL, NULL);

    CHECK_ERRORS_WITH_NULL_RETURN(err, __FILE__, __LINE__);

    /* Read the result */
    err = clEnqueueReadBuffer(queue, res_buff, CL_TRUE, 0, sizeof(float) * 4,
                              result, 0, NULL, NULL);
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

    return env->NewStringUTF(res_str.c_str());
}

JNIEXPORT void JNICALL
CLNET(deviceQuery)(JNIEnv *env, jobject instance) {
    /* Host data structures */
    cl_platform_id *platforms;
    // 每一个cl_platform_id 结构表示一个在主机上的OpenCL执行平台，
    // 就是指电脑中支持OpenCL的硬件，如nvidia显卡，intel CPU和显卡，AMD显卡和CPU等
    cl_uint num_platforms;
    cl_int i, err, platform_index = -1;

    /* Extension data */
    char *ext_data;
    size_t ext_size;
    const char icd_ext[] = "cl_khr_icd";

    // 要使platform工作，需要两个步骤:
    // 1) 需要为cl_platform_id结构分配内存空间。
    // 2) 需要调用clGetPlatformIDs初始化这些数据结构。
    // 一般还需要步骤0)：询问主机上有多少platforms

    /* Find number of platforms */
    // 返回值如果为-1就说明调用函数失败，如果为0标明成功
    // 第二个参数为NULL代表要咨询主机上有多少个platform，并使用num_platforms取得实际flatform数量。
    // 第一个参数为1，代表我们需要取最多1个platform。可以改为任意大如：INT_MAX整数最大值。
    // 但是据说0，否则会报错，实际测试好像不会报错。
    // 下面是步骤0)：询问主机有多少platforms
    err = clGetPlatformIDs(5, NULL, &num_platforms);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    LOGD("I have platforms: %d\n", num_platforms); //本人计算机上显示为2，有intel和nvidia两个平台  

    /* Access all installed platforms */
    // 步骤1 创建cl_platform_id，并分配空间
    platforms = (cl_platform_id *)
            malloc(sizeof(cl_platform_id) * num_platforms);
    // 步骤2 第二个参数用指针platforms存储platform
    clGetPlatformIDs(num_platforms, platforms, NULL);

    /* Find extensions of all platforms */
    // 获取额外的平台信息。上面已经取得了平台id了，那么就可以进一步获取更加详细的信息了。
    // 一个for循环获取所有的主机上的platforms信息:
    for (i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 1, NULL, &num_devices);
        CHECK_ERRORS(err, __FILE__, __LINE__);
        LOGD("The platform %d has %u devices(CPUs&GPUs).\n", i, num_devices);
        /* Find size of extension data */
        // 也是和前面一样，先设置第三和第四个参数为0和NULL，然后就可以用第五个参数ext_size获取额外信息的长度了。
        err = clGetPlatformInfo(platforms[i],
                                CL_PLATFORM_EXTENSIONS, 0, NULL, &ext_size);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        LOGD("The size of extension data is: %d\n", ext_size);//我的计算机显示224.  

        /* Access extension data */
        // 这里的ext_data相当于一个缓存，存储相关信息。
        ext_data = (char *) malloc(ext_size);
        // 这个函数就是获取相关信息的函数，第二个参数指明了需要什么样的信息，
        // 如这里的CL_PLATFORM_EXTENSIONS表示是opencl支持的扩展功能信息。
        // 我计算机输出一大串，机器比较新（专门为了学图形学而购置的电脑），支持的东西比较多。
        clGetPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,
                          ext_size, ext_data, NULL);
        LOGD("Platform %d supports extensions: %s\n", i, ext_data);

        // 这里是输出生产商的名字，比如我显卡信息是：NVIDIA CUDA
        char *name = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                          ext_size, name, NULL);
        LOGD("Platform %d name: %s\n", i, name);

        // 这里是供应商信息，我显卡信息：NVIDIA Corporation
        char *vendor = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,
                          ext_size, vendor, NULL);
        LOGD("Platform %d vendor: %s\n", i, vendor);

        // 最高支持的OpenCL版本，本机显示：OpenCL1.1 CUDA 4.2.1
        char *version = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION,
                          ext_size, version, NULL);
        LOGD("Platform %d version: %s\n", i, version);

        // 这个只有两个值：full profile 和 embeded profile
        char *profile = (char *) malloc(ext_size);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,
                          ext_size, profile, NULL);
        LOGD("Platform %d full profile or embeded profile?: %s\n", i, profile);

        /* Look for ICD extension */
        // 如果支持ICD这一扩展功能的platform，输出显示，本机的Intel和Nvidia都支持这一扩展功能
        if (strstr(ext_data, icd_ext) != NULL)
            platform_index = i;
        LOGD("Platform_index = %d", platform_index);
        /* Display whether ICD extension is supported */
        if (platform_index > -1)
            LOGD("Platform %d supports the %s extension.\n",
                 platform_index, icd_ext);

        // 释放空间
        free(ext_data);
        free(name);
        free(vendor);
        free(version);
        free(profile);
    }

    if (platform_index <= -1)
        LOGD("No platforms support the %s extension.\n", icd_ext);

    /* Deallocate resources */
    free(platforms);

}