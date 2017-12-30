//
// Created by yuchen on 17-12-25.
//
#include <CL/cl.h>
#include <opencl/cl_log.h>
#include <vector>
#include "opencl/cl_objects.h"

using namespace std;

cl_objects &cl_objects::getCLObject(cl_device_type required_device_type, const char *path) {
    static cl_objects clObject(required_device_type, path);
    return clObject;
}

cl_objects::cl_objects(cl_device_type required_device_type, const char *path) {
    cl_int err;
    err = clGetPlatformIDs(5, nullptr, &num_of_platforms);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    LOGD("Detect %d platform(s).\n", num_of_platforms);

    platforms.resize(num_of_platforms);
    err = clGetPlatformIDs(num_of_platforms, platforms.data(), nullptr);
    CHECK_ERRORS(err, __FILE__, __LINE__);

    num_of_devices.resize(num_of_platforms);
    devices.resize(num_of_platforms);
    contexts.resize(num_of_platforms);
    queues.resize(num_of_platforms);
    for (int i = 0; i < num_of_platforms; i++) {
        err = clGetDeviceIDs(platforms[i], required_device_type, 1, nullptr, &num_of_devices[i]);
        LOGD("Platform %d has %d required device(s)", i, num_of_devices[i]);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        devices[i].resize(num_of_devices[i]);
        err = clGetDeviceIDs(platforms[i], required_device_type, num_of_devices[i],
                             devices[i].data(), nullptr);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        contexts[i] = clCreateContext(nullptr, 1, devices[i].data(), nullptr, nullptr, &err);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        queues[i].resize(num_of_devices[i]);
        for (cl_uint j = 0; j < num_of_devices[i]; ++j) {
            queues[i][j] = clCreateCommandQueue(contexts[i], devices[i][j], 0, &err);
            CHECK_ERRORS(err, __FILE__, __LINE__);
        }
    }

    const char *program_buffer = load_program(path).c_str();
    program = clCreateProgramWithSource(contexts[0], 1,
                                        &program_buffer, 0, &err);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    err = clBuildProgram(program, 0, 0, "-O3 -cl-mad-enable -cl-fast-relaxed-math", 0,
                         0);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_length = 0;

        err = clGetProgramBuildInfo(
                program,
                devices[0][0],
                CL_PROGRAM_BUILD_LOG,
                0,
                0,
                &log_length);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        vector<char> log_buf = vector<char>(log_length);

        err = clGetProgramBuildInfo(
                program,
                devices[0][0],
                CL_PROGRAM_BUILD_LOG,
                log_length,
                (void *) log_buf.data(),
                0);
        CHECK_ERRORS(err, __FILE__, __LINE__);

        LOGE("Failed to build the OpenCL program!\nBuild log: %s", log_buf.data());
    }

    matvec.kernel = clCreateKernel(program, "matvec_mult", &err);
    CHECK_ERRORS(err, __FILE__, __LINE__);

    clGetKernelWorkGroupInfo(
            matvec.kernel,
            devices[0][0],
            CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t),
            &matvec.kernel_max_workgroup_size,
            nullptr
    );
    LOGD("WORK_GROUP_SIZE : %zu ", matvec.kernel_max_workgroup_size);
}

cl_objects::~cl_objects() {
    cl_int err = clReleaseKernel(matvec.kernel);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    err = clReleaseProgram(program);
    CHECK_ERRORS(err, __FILE__, __LINE__);
    for (int i = 0; i < num_of_platforms; i++) {
        for (cl_uint j = 0; j < num_of_devices[i]; ++j) {
            err = clReleaseCommandQueue(queues[i][j]);
            CHECK_ERRORS(err, __FILE__, __LINE__);
        }
        err = clReleaseContext(contexts[i]);
        CHECK_ERRORS(err, __FILE__, __LINE__);
    }
}

const vector<cl_context> &cl_objects::getContexts() const {
    return contexts;
}

const clnet_kernel &cl_objects::getMatvec() const {
    return matvec;
}

const vector<vector<cl_command_queue>> &cl_objects::getQueues() const {
    return queues;
}
