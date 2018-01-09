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