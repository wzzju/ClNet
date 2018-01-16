//
// Created by yuchen on 18-1-15.
//

// num_rows   – number of rows in matrix
// ptr        – the array that stores the offset to the i-th row in ptr[i]
// indices    – the array that stores the column indices for non-zero
// values in the matrix
// vec          - the dense vector
// out          - the output
#include <sstream>
#include <CL/cl.hpp>
#include "helper.h"
#include "opencl/cl_log.h"
#include "spmv.h"

using namespace std;
using namespace cl;

void spmv_csr_cpu(const int num_rows,
                  const int *ptr,
                  const int *indices,
                  const float *data,
                  const float *vec, float *out) {
    for (int row = 0; row < num_rows; row++) {
        float temp = 0;
        int start_row = ptr[row];
        int end_row = ptr[row + 1];
        for (int j = start_row; j < end_row; j++)
            temp += data[j] * vec[indices[j]];
        out[row] = temp;
    }
}

bool compare(const float *gpuOut, const int num_rows,
             const int *ptr,
             const int *indices,
             const float *data,
             const float *vec) {
    float *cpuOut = new float[num_rows]();
    SCOPE_EXIT(delete[] cpuOut);
    {
        CostTimeHelper timeHelper("cpu csr");
        // cpu matmul
        spmv_csr_cpu(num_rows, ptr, indices, data, vec, cpuOut);
    }
    int length = num_rows;
    for (int i = 0; i < length; ++i) {
        LOGD("cpu(%f) vs gpu(%f)\n", cpuOut[i], gpuOut[i]);
        if (cpuOut[i] != gpuOut[i]) {
            LOGD("cpuOut[%d] != gpuMatC[%d], %f != %f", i, i, cpuOut[i], gpuOut[i]);
            return false;
        }
    }
    return true;

}

void csrTest(cl_objects &clObject, stringstream &strs) {
    float values[]{1, 7, 2, 8, 5, 3, 9, 6, 4};
    float vector[]{2, 4, 3, 4};
    float out[4];
    int cols[]{0, 1, 1, 2, 0, 2, 3, 1, 3};
    int ptr[]{0, 2, 4, 7, 9};
    int num_rows = 4;
    try {
        Buffer valMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         9 * sizeof(float),
                         values);
        Buffer vecMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         4 * sizeof(float),
                         vector);
        Buffer colMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         9 * sizeof(float),
                         cols);
        Buffer ptrMemObj(clObject.getContexts()[0],
                         CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         (num_rows + 1) * sizeof(float),
                         ptr);

        Buffer outMemObj(clObject.getContexts()[0],
                         CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                         num_rows * sizeof(float),
                         nullptr);
        clObject.getMatmul().kernel.setArg(0, valMemObj);
        clObject.getMatmul().kernel.setArg(1, vecMemObj);
        clObject.getMatmul().kernel.setArg(2, colMemObj);
        clObject.getMatmul().kernel.setArg(3, ptrMemObj);
        clObject.getMatmul().kernel.setArg(4, num_rows);
        clObject.getMatmul().kernel.setArg(5, outMemObj);
        Event exeEvt;
        cl_ulong executionStart, executionEnd;
        std::size_t maxLocal = clObject.getMatmul().kernel_max_workgroup_size;
        std::size_t localWorkSize = VECTOR_SIZE;
        while (localWorkSize + VECTOR_SIZE <= maxLocal &&
               localWorkSize + VECTOR_SIZE <= BLOCK_SIZE) {
            localWorkSize += VECTOR_SIZE;
        }
        const std::size_t vectorGlobalWSize = num_rows * VECTOR_SIZE; // 1 warp per row
        clObject.getQueues()[0][0].enqueueNDRangeKernel(clObject.getMatmul().kernel,
                                                        NullRange,
                                                        NDRange(vectorGlobalWSize),
                                                        NDRange(localWorkSize),
                                                        nullptr,
                                                        &exeEvt);
        clObject.getQueues()[0][0].flush();
        clObject.getQueues()[0][0].finish();
        // let's understand how long it took?
        executionStart = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        executionEnd = exeEvt.getProfilingInfo<CL_PROFILING_COMMAND_END>();

        LOGD("The spmv-csr on GPU took %f s\n",
             static_cast<double>(executionEnd - executionStart) / 1000000000.0);

        clObject.getQueues()[0][0].enqueueReadBuffer(outMemObj, CL_TRUE, 0,
                                                     num_rows * sizeof(float), out);
    } catch (Error err) {
        LOGE("ERROR: %s\n", err.what());
        CHECK_ERRORS(err.err(), __FILE__, __LINE__);
    }

    if (compare(out, num_rows, ptr, cols, values, vector)) {
        LOGD("Passed!");
        strs << "Passed!" << endl;
    } else {
        LOGD("Failed!");
        strs << "Failed!" << endl;
    }
}