//
// Created by yuchen on 17-12-25.
//

#ifndef CLNET_CLOBJECTS_H
#define CLNET_CLOBJECTS_H

#include <string>
#include <fstream>
#include <vector>
#include <CL/cl.h>

typedef struct {
    cl_kernel kernel;
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

    ~cl_objects();

public:
    static cl_objects &getCLObject(cl_device_type required_device_type, const char *path);

    const std::vector<cl_context> &getContexts() const;

    const clnet_kernel &getMatmul() const;

    const std::vector<std::vector<cl_command_queue>> &getQueues() const;

    const std::vector<std::vector<cl_uint>> &getMaxComputeUnits() const;

private:
    // OpenCL平台和设备
    cl_uint num_of_platforms; // OpenCL平台数量
    // platforms[num_of_platforms]
    std::vector<cl_platform_id> platforms; // 存储OpenCL平台
    // num_of_devices[num_of_platforms]
    std::vector<cl_uint> num_of_devices; // 每个OpenCL平台上的设备数量
    // devices[num_of_platforms][num_of_devices]
    std::vector<std::vector<cl_device_id>> devices;// 每个OpenCL平台上有一个设备数组
    std::vector<std::vector<cl_uint>> maxComputeUnits;// 每个OpenCL平台上有一个设备数组
    // contexts[num_of_platforms]
    std::vector<cl_context> contexts;
    // 每个OpenCL平台上有一个设备上下文
    // queues[num_of_platforms][num_of_devices]
    std::vector<std::vector<cl_command_queue>> queues; // 每个OpenCL平台上每个设备的命令队列
    cl_program program;// OpenCL程序~cl_objects();
    // OpenCL kernels
    clnet_kernel matmul;// OpenCL内核函数

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
