
void batchedMultiply(cublasHandle_t handle, float *d_A, float *d_B, float *d_C, int batchSize, int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;

    float *d_A_array[] = { d_A, d_A + m * k };
    float *d_B_array[] = { d_B, d_B + k * n };
    float *d_C_array[] = { d_C, d_C + m * n };

    float **d_A_ptr, **d_B_ptr, **d_C_ptr;
    cudaMalloc(&d_A_ptr, batchSize * sizeof(float *)); // why are the pointers passed like this?
    cudaMalloc(&d_B_ptr, batchSize * sizeof(float *));
    cudaMalloc(&d_C_ptr, batchSize * sizeof(float *));

    cudaMemcpy(d_A_ptr, d_A_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptr, d_B_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ptr, d_C_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);

    checkCublas(cublasSgemmBatched(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_N, // specifying no transpose
                                   m, n, k,
                                   &alpha,
                                   d_A_ptr, m,
                                   d_B_ptr, k,
                                   &beta,
                                   d_C_ptr, m,
                                   batchSize));

    cudaFree(d_A_ptr);
    cudaFree(d_B_ptr);
    cudaFree(d_C_ptr);  // why is d_C_ptr is freed, how can I get the result? shouldn't it be written to host first?
}


void batchedMultiplyTranspose(cublasHandle_t handle, float *d_A, float *d_B, float *d_C, int batchSize, int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;

    float *d_A_array[] = { d_A, d_A + m * k };
    float *d_B_array[] = { d_B, d_B + n * k }; // this will be transposed before the multiplication
    float *d_C_array[] = { d_C, d_C + m * n };

    float **d_A_ptr, **d_B_ptr, **d_C_ptr;
    cudaMalloc(&d_A_ptr, batchSize * sizeof(float *)); // why are the pointers passed like this?
    cudaMalloc(&d_B_ptr, batchSize * sizeof(float *));
    cudaMalloc(&d_C_ptr, batchSize * sizeof(float *));

    cudaMemcpy(d_A_ptr, d_A_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_ptr, d_B_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ptr, d_C_array, batchSize * sizeof(float *), cudaMemcpyHostToDevice);

    checkCublas(cublasSgemmBatched(handle,
                                   CUBLAS_OP_N, CUBLAS_OP_T, // specifying no transpose
                                   m, n, k,
                                   &alpha,
                                   d_A_ptr, m,
                                   d_B_ptr, k,
                                   &beta,
                                   d_C_ptr, m,
                                   batchSize));

    cudaFree(d_A_ptr);
    cudaFree(d_B_ptr);
    cudaFree(d_C_ptr);  // why is d_C_ptr is freed, how can I get the result? shouldn't it be written to host first?
}
