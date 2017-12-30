inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return (unsigned)a < (unsigned)b;
}
__kernel void matvec_mult(__global float4* matrix,
                          __global float4* vector,
                          __global float* result) {

   int i = get_global_id(0);
   result[i] = dot(matrix[i], vector[0]);
}