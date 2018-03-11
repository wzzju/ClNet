//
// Created by yuchen on 17-12-25.
//
#include <CL/cl.hpp>
#include <opencl/cl_log.h>
#include <vector>
#include <string>
#include <opencl/cl_objects.h>

using namespace std;
using namespace cl;

cl_objects &cl_objects::getCLObject(cl_device_type required_device_type, const char *path) {
    static cl_objects clObject(required_device_type, path);
    return clObject;
}

// HUAWEI MATE 9 PRO : Mali G71 has a 8-core GPU
// Snapdragon 820 : Adreno 530 has a 4-core GPU
cl_objects::cl_objects(cl_device_type required_device_type, const char *path) {
    try {
        Platform::get(&platforms);
        LOGD("Detect %zu platform(s).\n", platforms.size());

        devices.resize(platforms.size());
        maxComputeUnits.resize(platforms.size());
        contexts.resize(platforms.size());
        queues.resize(platforms.size());

        for (cl_uint i = 0; i < platforms.size(); i++) {
            platforms[i].getDevices(required_device_type, &(devices[i]));
            LOGD("Platform %d has %zu required device(s).", i, devices[i].size());

            maxComputeUnits[i].resize(devices[i].size());
            queues[i].resize(devices[i].size());
            contexts[i] = Context(devices[i]);

            for (cl_uint j = 0; j < devices[i].size(); ++j) {
                maxComputeUnits[i][j] = devices[i][j].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
                LOGD("The max compute units of the device %u::%u(platform_id::device_id) is %u.",
                     i, j, maxComputeUnits[i][j]);
                /******************************Local Memory : Cache******************************/
                cl_ulong local_mem_size = devices[i][j].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
                LOGD("The local memory size of the device %u::%u(platform_id::device_id) is %lu KB.",
                     i, j, local_mem_size / 1024);
                /******************************Local Memory : Cache******************************/
                queues[i][j] = CommandQueue(contexts[i], devices[i][j], CL_QUEUE_PROFILING_ENABLE);
            }
        }

        string program_buffer = load_program(path);
        Program::Sources source(1, std::make_pair(program_buffer.c_str(),
                                                  program_buffer.length() + 1));
        program = Program(contexts[0], source);
        program.build(devices[0], "-O3 -cl-mad-enable -cl-fast-relaxed-math");

        relu.kernel = Kernel(program, "activation_relu_gpu");
        relu.kernel_max_workgroup_size =
                relu.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0][0]);
        LOGD("Relu KERNEL MAX WORK GROUP SIZE : %zu ", relu.kernel_max_workgroup_size);

        inner.kernel = Kernel(program, "inner_gpu");
        inner.kernel_max_workgroup_size =
                inner.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0][0]);
        LOGD("Inner KERNEL MAX WORK GROUP SIZE : %zu ", inner.kernel_max_workgroup_size);

        inner_plus_b.kernel = Kernel(program, "inner_plus_b_gpu");
        inner_plus_b.kernel_max_workgroup_size =
                inner_plus_b.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0][0]);
        LOGD("Inner_plus_b KERNEL MAX WORK GROUP SIZE : %zu ", inner_plus_b.kernel_max_workgroup_size);

        im2col.kernel = Kernel(program, "im2col_gpu");
        im2col.kernel_max_workgroup_size =
                im2col.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0][0]);
        LOGD("Img2Col KERNEL MAX WORK GROUP SIZE : %zu ", im2col.kernel_max_workgroup_size);

        max_pool.kernel = Kernel(program, "max_pool_gpu");
        max_pool.kernel_max_workgroup_size =
                max_pool.kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(devices[0][0]);
        LOGD("MaxPool KERNEL MAX WORK GROUP SIZE : %zu ", max_pool.kernel_max_workgroup_size);
    } catch (cl::Error err) {
        LOGE("ERROR: %s\n", err.what());
        if (err.err() == CL_BUILD_PROGRAM_FAILURE) {
            string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0][0]);

            LOGE("Failed to build the OpenCL program!\nBuild log: %s", log.c_str());
        }
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }
}

clnet_kernel &cl_objects::getRelu() {
    return relu;
}

clnet_kernel &cl_objects::getInner() {
    return inner;
}

clnet_kernel &cl_objects::getInner_plus_b() {
    return inner_plus_b;
}

clnet_kernel &cl_objects::getImg2col() {
    return im2col;
}

clnet_kernel &cl_objects::getMaxPool() {
    return max_pool;
}


const vector<Context> &cl_objects::getContexts() const {
    return contexts;
}

const vector<vector<CommandQueue>> &cl_objects::getQueues() const {
    return queues;
}

const vector<vector<cl_uint>> &cl_objects::getMaxComputeUnits() const {
    return maxComputeUnits;
}
