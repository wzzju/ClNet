#define VECTOR_SIZE 32
// scalar
__kernel void spmv_csr_scalar( __global const float* val,
                        __global const float* vec,
                        __global const int* cols,
                        __global const int* rowDelimiters,
                        __global const float* bias,
                       const int dim, __global float* out)
{
    int myRow = get_global_id(0);

    if (myRow < dim) {
        float t = 0;
        int start = rowDelimiters[myRow];
        int end = rowDelimiters[myRow+1];
        for (int j = start; j < end; j++) {
            int col = cols[j];
            t += val[j] * vec[col];
        }
        out[myRow] = t + bias[myRow];
    }
}

// vector
__kernel void spmv_csr_vector(__global const float* val,
                       __global const float* vec,
                       __global const int* cols,
                       __global const int* rowDelimiters,
                       __global const float* bias,
                       const int dim,
                       __global float* out) {
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
            out[myRow] = partialSums[t] + bias[myRow];
        }
    }
}

// relu激活函数
__kernel void activation_relu_gpu(__global float* input)
{
    int index = get_global_id(0);
    input[index] = input[index] > 0 ? input[index] : 0;
}

// 内积函数
__kernel void inner_gpu(__global const float* mat_left,
                        __global const float* mat_right,
                        const int col_left,
                        __global float * result) {

    int i = get_global_id(0);
    int j = get_global_id(1);
    int col_right = get_global_size(1);
    for (int k = 0; k < col_left; ++k) {
        result[i * col_right + j] +=
                mat_left[i * col_left + k] * mat_right[k * col_right + j];//+=，所以result必须要初始化为0
        // result[i * col_right + j] = isnan(result[i * col_right + j]) == 1 ? 0 : result[i * col_right + j];
    }
}

// 带偏置的内积函数
__kernel void inner_plus_b_gpu(__global const float* mat_left,
                               __global const float* mat_right,
                               const int col_left,
                               __global const float* bias,
                               __global float * result)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int col_right = get_global_size(1);
    for (int k = 0; k < col_left; ++k) {
        result[i * col_right + j] +=
                mat_left[i * col_left + k] * mat_right[k * col_right + j];//+=，所以result必须要初始化为0
    }
    result[i * col_right + j] += bias[i];
    // result[i * col_right + j] = isnan(result[i * col_right + j]) == 1 ? 0 : result[i * col_right + j];
}

//image to column
__kernel void im2col_gpu(__global const float *im_src,
                     const int channels, const int height_inp, const int width_inp,
                     const int kernel_h, const int kernel_w, const int pad_h, const int pad_w,
                     const int stride_h, const int stride_w,
                     const int height_out, const int width_out,
                     __global float *im_col)
{
    int index = get_global_id(0);
    if (index >= height_out * width_out * channels)
        return;
    int j_out = index % width_out;
    int i_out = (index / width_out) % height_out;
    int c_inp = (index / width_out) / height_out;

    int c_out = c_inp * kernel_h * kernel_w;
    int i_inp = i_out * stride_h - pad_h;
    int j_inp = j_out * stride_w - pad_w;

    im_src += (c_inp * height_inp + i_inp) * width_inp + j_inp;
    im_col += (c_out * height_out + i_out) * width_out + j_out;

    for (int ki = 0; ki < kernel_h; ++ki)
        for (int kj = 0; kj < kernel_w; ++kj) {
            int i = i_inp + ki;
            int j = j_inp + kj;
            *im_col = (i >= 0 && j >= 0 && i < height_inp && j < width_inp) ?
                im_src[ki * width_inp + kj] : 0;
            im_col += height_out * width_out;
      }
}

// max pool
__kernel void max_pool_gpu(
    const int nthreads, __global const float* input,
    const int channels, const int height, const int width,
    const int pad_h, const int pad_w,
    const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w,
    const int pooled_height, const int pooled_width,
    __global float* pooled_res) {
  for (int index = get_global_id(0); index < nthreads;
      index += get_global_size(0)) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, (int)0);
    wstart = max(wstart, (int)0);
    __global const float* input_slice = input
        + (n * channels + c) * height * width;
    float maxval = input_slice[hstart * width + wstart];
    int maxidx = -1;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (input_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = input_slice[maxidx];
        }
      }
    }
    pooled_res[index] = maxval;
  }
}