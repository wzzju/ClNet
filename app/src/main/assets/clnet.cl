#define VECTOR_SIZE 32

#ifdef SINGLE_PRECISION
  #define float float
#elif K_DOUBLE_PRECISION
  #pragma OPENCL EXTENSION cl_khr_fp64: enable
  #define float double
#elif AMD_DOUBLE_PRECISION
  #pragma OPENCL EXTENSION cl_amd_fp64: enable
  #define float double
#endif

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)a < (unsigned)b;
}

__kernel void matmul(int widthA,
                     int widthB,
                     __global int *A,
                     __global int *B,
                     __global int *C) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int tmp = 0;
    for (int k = 0; k < widthA; ++k) {
        tmp += A[i * widthA + k] * B[k * widthB + j];
    }
    C[i * widthB + j] = tmp;
}

// ****************************************************************************
// Function: spmv_csr_scalar_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a thread per row of the sparse
//   matrix
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
//
// Modifications:
//
// ****************************************************************************
__kernel void
spmv_csr_scalar_kernel( __global const float * restrict val,
                        __global const float * restrict vec,
                        __global const int * restrict cols,
                        __global const int * restrict rowDelimiters,
                       const int dim, __global float * restrict out)
{
    int myRow = get_global_id(0);

    if (myRow < dim) {
        float t=0;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++) {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[myRow] = t;
    }
}

// ****************************************************************************
// Function: spmv_csr_vector_kernel
//
// Purpose:
//   Computes sparse matrix - vector multiplication on the GPU using
//   the CSR data storage format, using a warp per row of the sparse
//   matrix
//
// Arguments:
//   val: array holding the non-zero values for the matrix
//   vec: dense vector for multiplication
//   cols: array of column indices for each element of the sparse matrix
//   rowDelimiters: array of size dim+1 holding indices to rows of the matrix
//                  last element is the index one past the last
//                  element of the matrix
//   dim: number of rows in the matrix
//   out: output - result from the spmv calculation
//
// Returns:  nothing
//           out indirectly through a pointer
//
//
// Modifications:
//
// ****************************************************************************
__kernel void
spmv_csr_vector_kernel(__global const float * restrict val,
                       __global const float * restrict vec,
                       __global const int * restrict cols,
                       __global const int * restrict rowDelimiters,
                       const int dim,
                       __global float * restrict out) {
    // Thread ID in block
    int t = get_local_id(0);
    // Thread ID within warp/wavefront
    int id = t & (VECTOR_SIZE-1);
    // One warp/wavefront per row
    int threadsPerBlock = get_local_size(0) / VECTOR_SIZE;
    int myRow = (get_group_id(0) * threadsPerBlock) + (t / VECTOR_SIZE);

    __local volatile float partialSums[128];
    partialSums[t] = 0;

    if (myRow < dim) {
        int vecStart = rowDelimiters[myRow];
        int vecEnd = rowDelimiters[myRow+1];
        float mySum = 0;
        for (int j= vecStart + id; j < vecEnd; j += VECTOR_SIZE) {
            int col = cols[j];
            mySum += val[j] * vec[col];
        }

        partialSums[t] = mySum;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Reduce partial sums
        // Needs to be modified if there is a change in vector
        // length
        if (id < 16) partialSums[t] += partialSums[t+16];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  8) partialSums[t] += partialSums[t+ 8];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  4) partialSums[t] += partialSums[t+ 4];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  2) partialSums[t] += partialSums[t+ 2];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id <  1) partialSums[t] += partialSums[t+ 1];
        barrier(CLK_LOCAL_MEM_FENCE);

        // Write result
        if (id == 0) {
            out[myRow] = partialSums[t];
        }
    }
}