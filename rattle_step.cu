#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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

void rattleHard(cublasHandle_t handle, cusolverDnHandle_t cusolver_handle, float *x, float *v, int batchSize, float h) {
    float alpha = 4.0f;
    float beta = 0.0f; 
    float alpha2 = 2.0f;
    float diffval = -1.0f;
    float h_t = 4.0f/(h*h);
    
    // Device pointers
    float *x_ptr, *v_ptr, *x_ptr_new, *v_ptr_new, *dL_ptr, *diff_ptr, *v12_ptr;
    float **d_I, **diff_ptrs, **dL_ptrs, **v_ptrs_new;
    float *d_I_flat, *R_ptr_flat, *L_ptrs_flat;
    float **L_ptrs;
    
    // Allocate memory for flat arrays
    checkCudaError(cudaMalloc(&x_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&v_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&x_ptr_new, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&v_ptr_new, batchSize * 3 * sizeof(float)));

    checkCudaError(cudaMalloc(&d_I_flat, batchSize * sizeof(float)));
    checkCudaError(cudaMalloc(&R_ptr_flat, batchSize * sizeof(float)));
    
    // Allocate memory for L_ptrs_flat and L_ptrs - FIX: These were missing
    checkCudaError(cudaMalloc(&L_ptrs_flat, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&L_ptrs, batchSize * sizeof(float*)));
    
    // Allocate memory for pointer arrays
    checkCudaError(cudaMalloc(&diff_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&dL_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&d_I, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&v_ptrs_new, batchSize * sizeof(float*)));

    // Allocate memory for other variables
    checkCudaError(cudaMalloc(&dL_ptr, batchSize * 3 * sizeof(float))); // FIX: Size should be 3*batchSize
    checkCudaError(cudaMalloc(&diff_ptr, batchSize * 3 * sizeof(float)));
    checkCudaError(cudaMalloc(&v12_ptr, batchSize * 3 * sizeof(float)));
    
    // Copy data to device
    checkCudaError(cudaMemcpy(x_ptr, x, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(v_ptr, v, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(x_ptr_new, x, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice));
    
    // Add h*v to x_new
    checkCublas(cublasSaxpy(handle, batchSize * 3, &h, v_ptr, 1, x_ptr_new, 1), "cublasSaxpy 1");
    
    // Allocate memory for the batched pointers
    float **x_ptrs, **x_ptrs_new, **R_ptr;
    checkCudaError(cudaMalloc(&x_ptrs, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&x_ptrs_new, batchSize * sizeof(float*)));
    checkCudaError(cudaMalloc(&R_ptr, batchSize * sizeof(float*)));
    
    // Set up host pointers
    float** x_ptrs_host = new float*[batchSize];
    float** x_ptrs_new_host = new float*[batchSize];
    float** R_ptr_host = new float*[batchSize];
    float** diff_ptrs_host = new float*[batchSize];
    float** dL_ptrs_host = new float*[batchSize];
    float** d_I_host = new float*[batchSize];
    float** L_ptrs_host = new float*[batchSize];  // FIX: Added for L_ptrs
    float** v_ptrs_new_host = new float*[batchSize]; // FIX: Added for v_ptrs_new
    
    // Allocate memory for each matrix
    for (int i = 0; i < batchSize; i++) {
        checkCudaError(cudaMalloc(&x_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&x_ptrs_new_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&R_ptr_host[i], sizeof(float)));
        checkCudaError(cudaMalloc(&diff_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&dL_ptrs_host[i], 3 * sizeof(float)));
        checkCudaError(cudaMalloc(&d_I_host[i], sizeof(float)));
        checkCudaError(cudaMalloc(&L_ptrs_host[i], 3 * sizeof(float))); // FIX: Added for L_ptrs
        checkCudaError(cudaMalloc(&v_ptrs_new_host[i], 3 * sizeof(float))); // FIX: Added for v_ptrs_new
        
        // Initialize d_I with value 1.0f
        float one = 1.0f;
        checkCudaError(cudaMemcpy(d_I_host[i], &one, sizeof(float), cudaMemcpyHostToDevice));
        
        // Copy individual matrix data to device
        checkCudaError(cudaMemcpy(x_ptrs_host[i], x + i * 3, 3 * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaError(cudaMemcpy(x_ptrs_new_host[i], x_ptr_new + i * 3, 3 * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Copy pointer arrays to device
    checkCudaError(cudaMemcpy(x_ptrs, x_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(x_ptrs_new, x_ptrs_new_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(R_ptr, R_ptr_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(diff_ptrs, diff_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dL_ptrs, dL_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_I, d_I_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(L_ptrs, L_ptrs_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice)); // FIX: Added for L_ptrs
    checkCudaError(cudaMemcpy(v_ptrs_new, v_ptrs_new_host, batchSize * sizeof(float*), cudaMemcpyHostToDevice)); // FIX: Added for v_ptrs_new
    
    std::cout << "All memory allocated and initialized successfully." << std::endl;
    
    // Set up kernel execution parameters
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;
    
    // Main calculation loop
    for (int i = 0; i < 1; i++) {
        // First matrix multiplication
        // FIX: Correct matrix dimensions - assuming each x is a 1x3 row vector
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T, // FIX: Transpose second matrix
            1, 1, 3,
            &alpha,
            (const float**)x_ptrs, 1,
            (const float**)x_ptrs_new, 1, // FIX: Leading dimension should be 1 for row vector
            &beta,
            R_ptr, 1,
            batchSize), "cublasSgemmBatched 1");
        
        checkCudaError(cudaDeviceSynchronize());
        
        // Convert batched to flattened
        convertBatchedToFlattened<<<numBlocks, blockSize>>>(R_ptr, R_ptr_flat, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());
        
        // Compute elementwise inverse
        elementwiseInverse<<<numBlocks, blockSize>>>(R_ptr_flat, d_I_flat, batchSize);
        checkCudaError(cudaDeviceSynchronize());
        
        // Convert flattened to batched
        convertFlattenedToBatched<<<numBlocks, blockSize>>>(d_I_flat, d_I, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());
        
        // Second matrix multiplication
        // FIX: Correct dimensions - scalar * vector = vector
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            3, 1, 1,
            &alpha2,
            (const float**)x_ptrs_new, 3,
            (const float**)d_I, 1,
            &beta,
            dL_ptrs, 3,
            batchSize), "cublasSgemmBatched 2");
        
        checkCudaError(cudaDeviceSynchronize());
        
        // Third matrix multiplication
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            3, 1, 1,
            &alpha2,
            (const float**)x_ptrs, 3,
            (const float**)dL_ptrs, 1,
            &beta,
            diff_ptrs, 3,
            batchSize), "cublasSgemmBatched 3");
        
        checkCudaError(cudaDeviceSynchronize());
        
        // Convert batched to flattened for diff
        convertBatchedToFlattened<<<numBlocks, blockSize>>>(diff_ptrs, diff_ptr, batchSize, 3);
        checkCudaError(cudaDeviceSynchronize());
        
        // Apply diff to x_ptr_new
        checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, diff_ptr, 1, x_ptr_new, 1), "cublasSaxpy 2");
        
        std::cout << "Finished iteration " << i + 1 << "." << std::endl;
    }
    
    // Copy x_new to v_new
    checkCudaError(cudaMemcpy(v_ptr_new, x_ptr_new, 3 * batchSize * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Subtract original x from v_new
    checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, x_ptr, 1, v_ptr_new, 1), "cublasSaxpy 3");
    
    // Convert flattened to batched for updated x
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(x_ptr_new, x_ptrs_new, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    // Fourth matrix multiplication
    // FIX: Correct dot product dimensions
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, // FIX: Transpose second matrix
        1, 1, 3,
        &alpha,
        (const float**)x_ptrs_new, 1,
        (const float**)x_ptrs_new, 1,
        &beta,
        R_ptr, 1,
        batchSize), "cublasSgemmBatched 4");
    
    checkCudaError(cudaDeviceSynchronize());
    
    // Convert flattened to batched for v_new
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(v_ptr_new, v_ptrs_new, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    // Fifth matrix multiplication
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 1, 1,
        &h_t,
        (const float**)x_ptrs_new, 3,
        (const float**)v_ptrs_new, 1,
        &beta,
        dL_ptrs, 3,
        batchSize), "cublasSgemmBatched 5");
    
    checkCudaError(cudaDeviceSynchronize());
    
    // Convert batched to flattened
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(R_ptr, R_ptr_flat, batchSize, 1);
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(dL_ptrs, dL_ptr, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    // Element-wise division
    elementwiseDiv<<<numBlocks, blockSize>>>(R_ptr_flat, dL_ptr, L_ptrs_flat, batchSize * 3);
    checkCudaError(cudaDeviceSynchronize());
    
    // Convert flattened to batched
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(L_ptrs_flat, L_ptrs, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    float h2 = h / 2.0f;
    float h3 = 1.0f / h;
    
    // Final matrix multiplication
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 1, 1,
        &h2,
        (const float**)x_ptrs_new, 3,
        (const float**)dL_ptrs, 1,
        &h3,
        v_ptrs_new, 3,
        batchSize), "cublasSgemmBatched 6");
    
    checkCudaError(cudaDeviceSynchronize());
    
    // Free all allocated memory
    checkCudaError(cudaFree(x_ptr));
    checkCudaError(cudaFree(v_ptr));
    checkCudaError(cudaFree(x_ptr_new));
    checkCudaError(cudaFree(v_ptr_new));
    checkCudaError(cudaFree(d_I_flat));
    checkCudaError(cudaFree(R_ptr_flat));
    checkCudaError(cudaFree(L_ptrs_flat));  // FIX: Free allocated memory
    checkCudaError(cudaFree(diff_ptrs));
    checkCudaError(cudaFree(dL_ptrs));
    checkCudaError(cudaFree(d_I));
    checkCudaError(cudaFree(L_ptrs));       // FIX: Free allocated memory
    checkCudaError(cudaFree(v_ptrs_new));   // FIX: Free allocated memory
    checkCudaError(cudaFree(dL_ptr));
    checkCudaError(cudaFree(diff_ptr));
    checkCudaError(cudaFree(v12_ptr));
    checkCudaError(cudaFree(x_ptrs));
    checkCudaError(cudaFree(x_ptrs_new));
    checkCudaError(cudaFree(R_ptr));
    
    // Free host temporary arrays
    for (int i = 0; i < batchSize; i++) {
        checkCudaError(cudaFree(x_ptrs_host[i]));
        checkCudaError(cudaFree(x_ptrs_new_host[i]));
        checkCudaError(cudaFree(R_ptr_host[i]));
        checkCudaError(cudaFree(diff_ptrs_host[i]));
        checkCudaError(cudaFree(dL_ptrs_host[i]));
        checkCudaError(cudaFree(d_I_host[i]));
        checkCudaError(cudaFree(L_ptrs_host[i]));     // FIX: Free allocated memory
        checkCudaError(cudaFree(v_ptrs_new_host[i])); // FIX: Free allocated memory
    }
    
    delete[] x_ptrs_host;
    delete[] x_ptrs_new_host;
    delete[] R_ptr_host;
    delete[] diff_ptrs_host;
    delete[] dL_ptrs_host;
    delete[] d_I_host;
    delete[] L_ptrs_host;     // FIX: Delete allocated memory
    delete[] v_ptrs_new_host; // FIX: Delete allocated memory
}

int main() {
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    std::cout << "maxThreadsPerBlock: " << maxThreadsPerBlock << std::endl;
    
    // Initialize cuBLAS and cuSolver handles
    cublasHandle_t cublasHandle;
    cusolverDnHandle_t cusolverHandle;
    
    // Create handles
    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverHandle);
    
    // Define batch size and the value for h
    int batchSize = 2; // Example batch size
    float h = 0.01f;   // Example step size
    
    // Example x and v data (batchSize x 3, 3D matrix per batch)
    float h_x[batchSize * 3] = {1.0f, 2.0f, 3.0f,  // x[0]
                                4.0f, 5.0f, 6.0f}; // x[1]
    float h_v[batchSize * 3] = {0.5f, 0.5f, 0.5f,  // v[0]
                                0.1f, 0.1f, 0.1f}; // v[1]
    
    // Allocate device memory for x and v
    float *d_x, *d_v;
    cudaMalloc(&d_x, batchSize * 3 * sizeof(float));
    cudaMalloc(&d_v, batchSize * 3 * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(d_x, h_x, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, batchSize * 3 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Call the rattleHard function
    rattleHard(cublasHandle, cusolverHandle, d_x, d_v, batchSize, h);
    
    // Copy the result back to host
    cudaMemcpy(h_x, d_x, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Output the result
    std::cout << "Updated x:" << std::endl;
    for (int i = 0; i < batchSize; ++i) {
        std::cout << "x[" << i << "]: ";
        for (int j = 0; j < 3; ++j) {
            std::cout << h_x[i * 3 + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_v);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    
    return 0;
}