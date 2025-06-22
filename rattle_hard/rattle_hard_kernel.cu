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

void rattleHard(cublasHandle_t handle, cusolverDnHandle_t cusolver_handle, float *x, float *v, int batchSize, float h) {
    float alpha = 4.0f;
    float beta = 0.0f; 
    float alpha2 = 2.0f;
    float diffval = -1.0f;
    float h_t = 4.0f/(h*h);
    
    // Device pointers
    float *x_ptr_new, *v_ptr_new, *dL_ptr, *diff_ptr, *v12_ptr;
    float **d_I, **diff_ptrs, **dL_ptrs, **v_ptrs_new;
    float *d_I_flat, *R_ptr_flat, *L_ptrs_flat;
    float **L_ptrs;
    
    // Allocate memory for flat arrays
    float *x_ptr = x;  // Use input device pointer
    float *v_ptr = v;  // Use input device pointer
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
        
    // Set up kernel execution parameters
    int blockSize = 256;
    int numBlocks = (batchSize + blockSize - 1) / blockSize;
    float* diff_host = new float[batchSize * 3];
    float *output_flat;
    checkCudaError(cudaMalloc(&output_flat, batchSize * sizeof(float)));
    // Main calculation loop
    for (int i = 0; i < 3; i++) {
        // First matrix multiplication
        // FIX: Correct matrix dimensions - assuming each x is a 1x3 row vector
        checkCublas(cublasSgemmBatched(handle,
            CUBLAS_OP_N, CUBLAS_OP_T, // FIX: Transpose second matrix
            1, 1, 3,
            &alpha, // multiplying result by 4 for the sphere
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
        // printDeviceMatrix(d_I_flat, 1, 1);  // if your matrices are 1x1 as in your example

        // Convert flattened to batched
        convertFlattenedToBatched<<<numBlocks, blockSize>>>(d_I_flat, d_I, batchSize, 1);
        checkCudaError(cudaDeviceSynchronize());
        
        G<<<numBlocks, blockSize>>>(x_ptrs_new, output_flat, batchSize);
        elementwiseMul<<<numBlocks, blockSize>>>(d_I_flat, output_flat, dL_ptr, batchSize);
        convertFlattenedToBatched<<<numBlocks, blockSize>>>(dL_ptr, dL_ptrs, batchSize, 1);

        
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
            batchSize), "cublasSgemmBatched 3"); // diff = 2 x dL_ptrs


        checkCudaError(cudaDeviceSynchronize());
        
        // Convert batched to flattened for diff
        convertBatchedToFlattened<<<numBlocks, blockSize>>>(diff_ptrs, diff_ptr, batchSize, 3);
        checkCudaError(cudaDeviceSynchronize());

        
        // checkCudaError(cudaMemcpy(diff_host, diff_ptr, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost));

        // // Print diff_host contents
        // std::cout << "diff_ptr after iteration " << i << ":\n";
        // for (int b = 0; b < batchSize; b++) {
        //     std::cout << "Batch " << b << ": ";
        //     for (int j = 0; j < 3; j++) {
        //         std::cout << diff_host[b * 3 + j] << " ";
        //     }
        //     std::cout << "\n";
        // }
        // Apply diff to x_ptr_new
        checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, diff_ptr, 1, x_ptr_new, 1), "cublasSaxpy 2"); //xnew​=xnew​- diff_ptr
        convertFlattenedToBatched<<<numBlocks, blockSize>>>(x_ptr_new, x_ptrs_new, batchSize, 1);

    }
    
    // Copy x_new to v_new
    checkCudaError(cudaMemcpy(v_ptr_new, x_ptr_new, 3 * batchSize * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // Subtract original x from v_new
    checkCublas(cublasSaxpy(handle, batchSize * 3, &diffval, x_ptr, 1, v_ptr_new, 1), "cublasSaxpy 3"); // vptr_new = xptr_new - x_ptr . still needs to be divided by h
    
    // float* v_host = new float[batchSize * 3];
    // cudaMemcpy(v_host, v_ptr_new, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToHost);


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
        batchSize), "cublasSgemmBatched 4"); // updating R_ptr, this is equivalent to P in the original
    
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
        batchSize), "cublasSgemmBatched 5"); // dl_ptrs = 4(x @ 2v/h) - equal to t in the original implementation
        // h_t is 4/h^2 because v still needed to be divided by h, 
    
    checkCudaError(cudaDeviceSynchronize());
    
    // Convert batched to flattened
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(R_ptr, R_ptr_flat, batchSize, 1);
    convertBatchedToFlattened<<<numBlocks, blockSize>>>(dL_ptrs, dL_ptr, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    // Element-wise division
    elementwiseDiv<<<numBlocks, blockSize>>>(R_ptr_flat, dL_ptr, L_ptrs_flat, batchSize * 3); // L = L_ptrs/R = T/P
    checkCudaError(cudaDeviceSynchronize());
    
    // Convert flattened to batched
    convertFlattenedToBatched<<<numBlocks, blockSize>>>(L_ptrs_flat, L_ptrs, batchSize, 3);
    checkCudaError(cudaDeviceSynchronize());
    
    float h2 = h;
    float h3 = 1.0f/h;
    
    // Final matrix multiplication
    checkCublas(cublasSgemmBatched(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        3, 1, 1,
        &h2,
        (const float**)x_ptrs_new, 3,
        (const float**)L_ptrs, 1,
        &h3,
        v_ptrs_new, 3,
        batchSize), "cublasSgemmBatched 6"); // v_ptrs_new = (h) * x_ptrs_new * dL_ptrs + 1/h v_ptrs_new = v + h/2 J @ L 
    
    checkCudaError(cudaDeviceSynchronize());
    cudaMemcpy(x_ptr, x_ptr_new, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(v_ptr, v_ptr_new, batchSize * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free all allocated memory
    // checkCudaError(cudaFree(x_ptr));
    // checkCudaError(cudaFree(v_ptr));
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


void rattle_hard_launcher(torch::Tensor x, torch::Tensor v, float h) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(v.dtype() == torch::kFloat32, "v must be float32");

    float* d_x = x.data_ptr<float>();
    float* d_v = v.data_ptr<float>();
    int batchSize = x.size(0);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    // Pass device pointers directly to rattleHard
    rattleHard(cublasHandle, cusolverHandle, d_x, d_v, batchSize, h);

    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
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
    

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_v);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    
    return 0;
}