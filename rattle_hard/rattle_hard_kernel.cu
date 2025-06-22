#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <torch/extension.h>

// CUDA kernels for converting between batched and flattened matrices
__global__ void convertFlattenedToBatched(float *d_x_flattened, float **d_x_ptrs, int batchSize, int matrixSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batchSize) {
        d_x_ptrs[idx] = d_x_flattened + idx * matrixSize;
    }
}

__global__ void convertBatchedToFlattened(float **d_x_ptrs, float *d_x_flattened, int batchSize, int matrixSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batchSize) {
        float *matrix = d_x_ptrs[idx];
        for (int i = 0; i < matrixSize; i++) {
            d_x_flattened[idx * matrixSize + i] = matrix[i];
        }
    }
}

// elementwise kernels
__global__ void elementwiseInverse(float* x, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (x[idx] != 0.0f) { // Avoid division by zero
            y[idx] = 1.0f / x[idx];
        } else {
            y[idx] = 0.0f; // Or some other safe value
        }
    }
}

__global__ void elementwiseDiv(float* d_x, float* d_y, float* d_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (d_y[idx] != 0.0f) { // Avoid division by zero
            d_result[idx] = d_x[idx] / d_y[idx];
        } else {
            d_result[idx] = 0.0f; // Or some other safe value
        }
    }
}

__global__ void elementwiseMul(float* d_x, float* d_y, float* d_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
            d_result[idx] = d_x[idx] * d_y[idx];
        }
}

__global__ void G(float **x_ptrs, float *output, int batchSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    float *x = x_ptrs[idx];  // pointer to batch element (3 floats)
    float result = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] - 1.0f;
    output[idx] = result;
}

void checkCublas(cublasStatus_t status, const std::string& functionName) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error in " << functionName << ": " << status << std::endl;
        exit(1);
    }
}

void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        exit(-1);  // Exit if error occurs
    }
}


// Function to copy and print a single matrix/vector from device memory:
void printDeviceMatrix(const float* d_mat, int rows, int cols) {
    std::vector<float> h_mat(rows * cols);
    // Synchronous copy from device to host
    cudaMemcpy(h_mat.data(), d_mat, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << h_mat[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
}


void printWholeBatchR(float** d_R_ptr, int batchSize, int rows, int cols) {
    for (int i = 0; i < batchSize; ++i) {
        // Copy device pointer for ith matrix from device to host
        float* h_single_R_ptr;
        cudaMemcpy(&h_single_R_ptr, d_R_ptr + i, sizeof(float*), cudaMemcpyDeviceToHost);

        std::cout << "R_ptr[" << i << "]:" << std::endl;
        printDeviceMatrix(h_single_R_ptr, rows, cols);
        std::cout << std::endl;
    }
}


struct RattleHardContext {
    float *x_ptr_new, *v_ptr_new, *dL_ptr, *diff_ptr, *v12_ptr;
    float **d_I, **diff_ptrs, **dL_ptrs, **v_ptrs_new;
    float *d_I_flat, *R_ptr_flat, *L_ptrs_flat;
    float **L_ptrs;

    float **x_ptrs, **x_ptrs_new, **R_ptr;
    float **x_ptrs_host, **x_ptrs_new_host, **R_ptr_host;
    float **diff_ptrs_host, **dL_ptrs_host, **d_I_host;
    float **L_ptrs_host, **v_ptrs_new_host;

    int batchSize;
};

void allocateRattleHardContext(RattleHardContext &ctx, int batchSize) {
    ctx.batchSize = batchSize;

    // Allocate device memory similar to your current allocation code...
    checkCudaError(cudaMalloc(&ctx.x_ptr_new, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.v_ptr_new, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.d_I_flat, batchSize * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.R_ptr_flat, batchSize * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.L_ptrs_flat, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.L_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.diff_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.dL_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.d_I, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.v_ptrs_new, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.dL_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.diff_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.v12_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&ctx.x_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.x_ptrs_new, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&ctx.R_ptr, batchSize * sizeof(float*)));

    // Allocate host arrays of pointers
    ctx.x_ptrs_host = new float*[batchSize];
    ctx.x_ptrs_new_host = new float*[batchSize];
    ctx.R_ptr_host = new float*[batchSize];
    ctx.diff_ptrs_host = new float*[batchSize];
    ctx.dL_ptrs_host = new float*[batchSize];
    ctx.d_I_host = new float*[batchSize];
    ctx.L_ptrs_host = new float*[batchSize];
    ctx.v_ptrs_new_host = new float*[batchSize];

    // Allocate per-batch device memory pointers
    for (int i = 0; i < batchSize; i++) {
        checkCudaError(cudaMalloc(&ctx.x_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.x_ptrs_new_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.R_ptr_host[i], sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.diff_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.dL_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.d_I_host[i], sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.L_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&ctx.v_ptrs_new_host[i], 3 * sizeof(float)));

        // Initialize d_I with 1.0f
        float one = 1.0f;
        checkCudaError(cudaMemcpy(ctx.d_I_host[i], &one, sizeof(float), cudaMemcpyHostToDevice));
    }

    // Copy host pointer arrays to device pointer arrays once
    checkCudaError(cudaMemcpy(ctx.x_ptrs, ctx.x_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.x_ptrs_new, ctx.x_ptrs_new_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.R_ptr, ctx.R_ptr_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.diff_ptrs, ctx.diff_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.dL_ptrs, ctx.dL_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.d_I, ctx.d_I_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.L_ptrs, ctx.L_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(ctx.v_ptrs_new, ctx.v_ptrs_new_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
}

void freeRattleHardContext(RattleHardContext &ctx) {
    // Free all device memory and host arrays
    checkCudaError(cudaFree(ctx.x_ptr_new));
    checkCudaError(cudaFree(ctx.v_ptr_new));
    checkCudaError(cudaFree(ctx.d_I_flat));
    checkCudaError(cudaFree(ctx.R_ptr_flat));
    checkCudaError(cudaFree(ctx.L_ptrs_flat));
    checkCudaError(cudaFree(ctx.diff_ptrs));
    checkCudaError(cudaFree(ctx.dL_ptrs));
    checkCudaError(cudaFree(ctx.d_I));
    checkCudaError(cudaFree(ctx.L_ptrs));
    checkCudaError(cudaFree(ctx.v_ptrs_new));
    checkCudaError(cudaFree(ctx.dL_ptr));
    checkCudaError(cudaFree(ctx.diff_ptr));
    checkCudaError(cudaFree(ctx.v12_ptr));
    checkCudaError(cudaFree(ctx.x_ptrs));
    checkCudaError(cudaFree(ctx.x_ptrs_new));
    checkCudaError(cudaFree(ctx.R_ptr));

    for (int i = 0; i < ctx.batchSize; i++) {
        checkCudaError(cudaFree(ctx.x_ptrs_host[i]));
        checkCudaError(cudaFree(ctx.x_ptrs_new_host[i]));
        checkCudaError(cudaFree(ctx.R_ptr_host[i]));
        checkCudaError(cudaFree(ctx.diff_ptrs_host[i]));
        checkCudaError(cudaFree(ctx.dL_ptrs_host[i]));
        checkCudaError(cudaFree(ctx.d_I_host[i]));
        checkCudaError(cudaFree(ctx.L_ptrs_host[i]));
        checkCudaError(cudaFree(ctx.v_ptrs_new_host[i]));
    }

    delete[] ctx.x_ptrs_host;
    delete[] ctx.x_ptrs_new_host;
    delete[] ctx.R_ptr_host;
    delete[] ctx.diff_ptrs_host;
    delete[] ctx.dL_ptrs_host;
    delete[] ctx.d_I_host;
    delete[] ctx.L_ptrs_host;
    delete[] ctx.v_ptrs_new_host;
}


void rattleHard(cublasHandle_t handle, cusolverDnHandle_t cusolver_handle, float *x_ptr, float *v_ptr, int batchSize, float h, RattleHardContext &ctx) {
    float alpha = 4.0f;
    float beta = 0.0f; 
    float alpha2 = 2.0f;
    float diffval = -1.0f;
    float h_t = 4.0f / (h * h);
    
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;

    // 1) Add h*v to x_ptr_new
    checkCudaError(cudaMemcpy(ctx.x_ptr_new, x_ptr, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCublas(cublasSaxpy(handle, batchSize * 3, &h, v_ptr, 1, ctx.x_ptr_new, 1), "cublasSaxpy 1");

    for (int i = 0; i < 3; i++) {
        // 2) Batched multiplication: R = 4 * x * x_new^T
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            1, 1, 3,
            &alpha,
            (const float**)ctx.x_ptrs, 1,
            (const float**)ctx.x_ptrs_new, 1,
            &beta,
            ctx.R_ptr, 1,
            batchSize), "cublasSgemmBatched 1");
        checkCudaError(cudaDeviceSynchronize());

        // 3) Flatten R, compute inverse elementwise, convert back to batched
        convertBatchedToFlattened<<<numBlocks, blockSize>>>(ctx.R_ptr, ctx.R_ptr_flat, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());

        elementwiseInverse<<<numBlocks, blockSize>>>(ctx.R_ptr_flat, ctx.d_I_flat, batchSize);
        checkCudaError(cudaDeviceSynchronize());

        convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.d_I_flat, ctx.d_I, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());

        // 4) Launch kernel G and multiply elementwise with d_I_flat -> dL_ptr, then convert to batched
        G<<<numBlocks, blockSize>>>(ctx.x_ptrs_new, ctx.R_ptr_flat /*reuse*/, batchSize);
        checkCudaError(cudaDeviceSynchronize());

        elementwiseMul<<<numBlocks, blockSize>>>(ctx.d_I_flat, ctx.R_ptr_flat, ctx.dL_ptr, batchSize);
        checkCudaError(cudaDeviceSynchronize());

        convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.dL_ptr, ctx.dL_ptrs, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());

        // 5) Third matrix multiplication: diff = 2 * x * dL_ptr
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            3, 1, 1,
            &alpha2,
            (const float**)ctx.x_ptrs, 3,
            (const float**)ctx.dL_ptrs, 1,
            &beta,
            ctx.diff_ptrs, 3,
            batchSize), "cublasSgemmBatched 3");
        checkCudaError(cudaDeviceSynchronize());

        // 6) Flatten diff, then apply diffval offset to x_ptr_new
        convertBatchedToFlattened<<<numBlocks, blockSize>>>(ctx.diff_ptrs, ctx.diff_ptr, batchSize, 3);
        checkCudaError(cudaDeviceSynchronize());

        checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, ctx.diff_ptr, 1, ctx.x_ptr_new, 1), "cublasSaxpy 2");

        convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.x_ptr_new, ctx.x_ptrs_new, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());
    }

    // 7) Copy x_ptr_new to v_ptr_new, then subtract original x_ptr (v_ptr_new = x_ptr_new - x_ptr)
    checkCudaError(cudaMemcpy(ctx.v_ptr_new, ctx.x_ptr_new, 3 * batchSize * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, x_ptr, 1, ctx.v_ptr_new, 1), "cublasSaxpy 3");

    convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.x_ptr_new, ctx.x_ptrs_new, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());

    // 8) Batched multiplication updating R_ptr (P)
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        1, 1, 3,
        &alpha,
        (const float**)ctx.x_ptrs_new, 1,
        (const float**)ctx.x_ptrs_new, 1,
        &beta,
        ctx.R_ptr, 1,
        batchSize), "cublasSgemmBatched 4");
    checkCudaError(cudaDeviceSynchronize());

    // 9) Convert v_ptr_new to batched pointers
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.v_ptr_new, ctx.v_ptrs_new, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());

    // 10) Batched multiplication: dL_ptrs = h_t * x_ptr_new * v_ptr_new
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 1, 1,
        &h_t,
        (const float**)ctx.x_ptrs_new, 3,
        (const float**)ctx.v_ptrs_new, 1,
        &beta,
        ctx.dL_ptrs, 3,
        batchSize), "cublasSgemmBatched 5");
    checkCudaError(cudaDeviceSynchronize());

    // 11) Flatten R_ptr and dL_ptrs for elementwise division
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(ctx.R_ptr, ctx.R_ptr_flat, batchSize, 1);
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(ctx.dL_ptrs, ctx.dL_ptr, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());

    // 12) L_ptrs_flat = dL_ptr / R_ptr_flat (elementwise division)
    elementwiseDiv<<<numBlocks, blockSize>>>(ctx.R_ptr_flat, ctx.dL_ptr, ctx.L_ptrs_flat, batchSize * 3);
    checkCudaError(cudaDeviceSynchronize());

    // 13) Convert flattened L_ptrs_flat back to batched pointers
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(ctx.L_ptrs_flat, ctx.L_ptrs, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());

    float h2 = h;
    float h3 = 1.0f / h;

    // 14) Final batched multiplication updating v_ptrs_new
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 1, 1,
        &h2,
        (const float**)ctx.x_ptrs_new, 3,
        (const float**)ctx.L_ptrs, 1,
        &h3,
        ctx.v_ptrs_new, 3,
        batchSize), "cublasSgemmBatched 6");
    checkCudaError(cudaDeviceSynchronize());

    // 15) Copy results back to original x and v pointers
    checkCudaError(cudaMemcpy(x_ptr, ctx.x_ptr_new, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
    checkCudaError(cudaMemcpy(v_ptr, ctx.v_ptr_new, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToDevice));
}


void rattle_hard_launcher(torch::Tensor x, torch::Tensor v, float h, int n) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");
    TORCH_CHECK(x.size(0) == v.size(0), "x and v must have the same batch size");

    float* d_x = x.data_ptr<float>();
    float* d_v = v.data_ptr<float>();
    int batchSize = x.size(0);

    // Create cuBLAS and cuSolver handles
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // Allocate context once outside the loop
    RattleHardContext ctx;
    allocateRattleHardContext(ctx, batchSize);

    // Run rattleHard n times
    for (int i = 0; i < n; ++i) {
        rattleHard(cublasHandle, cusolverHandle, d_x, d_v, batchSize, h, ctx);
    }

    // Free context memory
    freeRattleHardContext(ctx);

    // Destroy cuBLAS and cuSolver handles
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
}
