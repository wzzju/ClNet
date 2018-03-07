//
// Created by yuchen on 17-12-25.
//
// Any of the C++ wrapper objects can return the underlying OpenCL C object using operator().

#ifndef CLNET_CLOBJECTS_H
#define CLNET_CLOBJECTS_H

#include <string>
#include <fstream>
#include <vector>
#include <CL/cl.hpp>

typedef struct {
    cl::Kernel kernel;
    // 每个kernel的workgroup_size，在local_work_size数组中所有元素的乘积不能大于该值。
    size_t kernel_max_workgroup_size;
} clnet_kernel;

class cl_objects {
    /**
     * 构建OpenCL对象
     * @param required_device_type
     * @param path .cl文件的路径
     */
    cl_objects(cl_device_type required_device_type, const char *path);

    cl_objects(const cl_objects &) = delete;

    cl_objects &operator=(const cl_objects &)= delete;

    ~cl_objects() = default;

public:
    static cl_objects &getCLObject(cl_device_type required_device_type, const char *path);

    clnet_kernel &getRelu();
    clnet_kernel &getInner();
    clnet_kernel &getInner_plus_b();
    clnet_kernel &getImg2col();
    clnet_kernel &getMaxPool();

    const std::vector<cl::Context> &getContexts() const;

    const std::vector<std::vector<cl::CommandQueue>> &getQueues() const;

    const std::vector<std::vector<cl_uint>> &getMaxComputeUnits() const;

private:
    // OpenCL平台和设备初始化
    std::vector<cl::Platform> platforms; // 存储OpenCL平台
    // devices[num_of_platforms][num_of_devices]
    std::vector<std::vector<cl::Device>> devices;// 每个OpenCL平台上有一个设备数组
    // maxComputeUnits[num_of_platforms][num_of_devices]
    std::vector<std::vector<cl_uint>> maxComputeUnits; // 每个设备的计算单元数
    // contexts[num_of_platforms]
    std::vector<cl::Context> contexts;
    // 每个OpenCL平台上有一个设备上下文
    // queues[num_of_platforms][num_of_devices]
    std::vector<std::vector<cl::CommandQueue>> queues; // 每个OpenCL平台上每个设备的命令队列
    cl::Program program;// OpenCL程序
    // OpenCL kernels
    clnet_kernel relu;
    clnet_kernel inner;// OpenCL内核函数
    clnet_kernel inner_plus_b;
    clnet_kernel im2col;
    clnet_kernel max_pool;

    /**
     * 根据OpenCL程序路径返回其字符串源码。
     * @param inputPath
     * @return programString
     */
    std::string load_program(const char *inputPath) {
        std::ifstream programFile(inputPath);
        std::string programString{std::istreambuf_iterator<char>(programFile),
                                  std::istreambuf_iterator<char>()};
        return programString;
    }
};


#endif //CLNET_CLOBJECTS_H
