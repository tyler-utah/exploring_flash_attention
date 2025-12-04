#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Use common standard.h
#include "../../common/standard.h"

// Kernel selection via compile flag
#ifndef USE_OPT
#define USE_OPT 0
#endif

#if USE_OPT
#include "flash_attention_v1_opt.h"
#define flash_attention_v1 flash_attention_v1_opt
#define flash_attention_kernel flash_attention_kernel_opt
#define KERNEL_NAME "flash_attention_v1_tiled_d_opt"
#else
#include "flash_attention_v1.h"
#define KERNEL_NAME "flash_attention_v1_tiled_d"
#endif

// Configure shared memory for flash attention kernel
bool configure_shared_memory(size_t shared_mem_size) {
    int maxConfigurableSharedMem;
    cudaDeviceGetAttribute(&maxConfigurableSharedMem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           0);
    if (maxConfigurableSharedMem < shared_mem_size) {
        std::cerr << "Warning: Requested shared memory (" << shared_mem_size 
                  << " bytes) exceeds max configurable (" << maxConfigurableSharedMem 
                  << " bytes)" << std::endl;
        return false;
    }

    cudaError_t err = cudaFuncSetAttribute(
        flash_attention_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_size);
    
    if (err != cudaSuccess) {
        std::cerr << "Failed to set shared memory size: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

// Utility: Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Helper to convert DATA_TYPE to float for display/comparison
inline float data_to_float(const DATA_TYPE& val) {
    return DATA_TO_FLOAT(val);
}

inline DATA_TYPE float_to_data(float val) {
    return FLOAT_TO_DATA(val);
}

// Initialize random data
void initialize_random(DATA_TYPE* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = float_to_data(((float)rand() / RAND_MAX) * 2.0f - 1.0f);
    }
}

// Compare two arrays and compute errors
void compare_arrays(const DATA_TYPE* a, const DATA_TYPE* b, int size, 
                    float& max_abs_diff, float& max_rel_diff, float& max_rel_diff_all) {
    max_abs_diff = 0.0f;
    max_rel_diff = 0.0f;
    max_rel_diff_all = 0.0f;
#if USE_FP64
    const float eps = 1e-10f;
#else
    const float eps = 1e-3f;
#endif
    
    for (int i = 0; i < size; i++) {
        float val_a = data_to_float(a[i]);
        float val_b = data_to_float(b[i]);
        float abs_diff = fabsf(val_a - val_b);
        
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        
        // Relative error for all values
        float denom_all = fabsf(val_b) + 1e-8f;
        float rel_diff_all = abs_diff / denom_all;
        if (rel_diff_all > max_rel_diff_all) {
            max_rel_diff_all = rel_diff_all;
        }
        
        // Relative error for significant values only
        if (fabsf(val_b) > eps) {
            float rel_diff = abs_diff / fabsf(val_b);
            if (rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
            }
        }
    }
}

int main() {
    srand(42);
    
    // Problem dimensions
    const int B = 32;      // Batch size
    const int H = 8;       // Number of heads
    const int L = 1024;    // Sequence length
    const int d = D;       // Head dimension (from compile-time constant)
    const int d_tile_qk = D_TILE_QK;  // D-tiling parameter for Q@K^T
    const int d_tile_v = D_TILE_V;    // D-tiling parameter for S@V
    
    const int total_size = B * H * L * d;
    
    std::cout << "Flash Attention V1 with D-Tiling - Test Configuration" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Kernel: " << KERNEL_NAME << std::endl;
    std::cout << "Batch size (B): " << B << std::endl;
    std::cout << "Number of heads (H): " << H << std::endl;
    std::cout << "Sequence length (L): " << L << std::endl;
    std::cout << "Head dimension (d): " << d << std::endl;
    std::cout << "D-tile size (Q@K^T): " << d_tile_qk << std::endl;
    std::cout << "D-tile size (S@V): " << d_tile_v << std::endl;
    std::cout << "Query tile size (BQ): " << BQ << std::endl;
    std::cout << "Key/Value tile size (BK): " << BK << std::endl;
    std::cout << "Threads per block: " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Total heads: " << B * H << std::endl;
    std::cout << "Grid dimensions: (" << (L + BQ - 1) / BQ << ", " << B * H << ")" << std::endl;
    std::cout << "Total blocks: " << ((L + BQ - 1) / BQ) * (B * H) << std::endl;
#if USE_FP64
    std::cout << "Precision: FP64 (double)" << std::endl;
#else
    std::cout << "Precision: FP16 (half)" << std::endl;
#endif
    std::cout << "=====================================================" << std::endl;
    
    // Allocate host memory
    DATA_TYPE* h_Q = new DATA_TYPE[total_size];
    DATA_TYPE* h_K = new DATA_TYPE[total_size];
    DATA_TYPE* h_V = new DATA_TYPE[total_size];
    DATA_TYPE* h_O_gpu = new DATA_TYPE[total_size];
    DATA_TYPE* h_O_cpu = new DATA_TYPE[total_size];
    
    // Initialize input data
    initialize_random(h_Q, total_size);
    initialize_random(h_K, total_size);
    initialize_random(h_V, total_size);
    
    // Allocate device memory
    DATA_TYPE *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_K, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_V, total_size * sizeof(DATA_TYPE)));
    CUDA_CHECK(cudaMalloc(&d_O, total_size * sizeof(DATA_TYPE)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, total_size * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
    
    // Configure shared memory
    size_t shared_mem_size = (
        BQ * d + BK * d + BK * d + BQ * d + BQ * BK
    ) * sizeof(DATA_TYPE) + (BQ + BQ + BQ) * sizeof(float);
    configure_shared_memory(shared_mem_size);
    
    // Warm-up runs
    const int warmup_runs = 10;
    for (int i = 0; i < warmup_runs; i++) {
        flash_attention_v1(d_Q, d_K, d_V, d_O, B, H, L, d, d_tile_qk, d_tile_v);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Timed runs
    const int num_runs = 50;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_runs; i++) {
        flash_attention_v1(d_Q, d_K, d_V, d_O, B, H, L, d, d_tile_qk, d_tile_v);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> gpu_time = end - start;
    double avg_gpu_time = gpu_time.count() / num_runs;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, total_size * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));
    
    // Run CPU reference
    std::cout << "\nRunning CPU reference..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    standard_attention_cpu(h_Q, h_K, h_V, h_O_cpu, B, H, L, d);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;
    
    // Compare results
    float max_abs_diff, max_rel_diff, max_rel_diff_all;
    compare_arrays(h_O_gpu, h_O_cpu, total_size, max_abs_diff, max_rel_diff, max_rel_diff_all);
    
    // Print results
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Average GPU time (" << num_runs << " runs): " << avg_gpu_time << " ms" << std::endl;
    std::cout << "Total GPU time: " << gpu_time.count() << " ms" << std::endl;
    std::cout << "CPU time (1 run): " << cpu_time.count() << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_time.count() / avg_gpu_time << "x" << std::endl;
    
    std::cout << "\n=====================================================" << std::endl;
    std::cout << "Accuracy Check:" << std::endl;
    std::cout << "=====================================================" << std::endl;
    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference (all values): " << max_rel_diff_all << std::endl;
    std::cout << "Max relative difference (filtered |values| > " 
#if USE_FP64
              << "1e-10"
#else
              << "1e-3"
#endif
              << "): " << max_rel_diff << std::endl;
    
    // Print sample outputs
    std::cout << "\nFirst 5 values:" << std::endl;
    std::cout << "GPU: ";
    for (int i = 0; i < 5 && i < total_size; i++) {
        std::cout << data_to_float(h_O_gpu[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "CPU: ";
    for (int i = 0; i < 5 && i < total_size; i++) {
        std::cout << data_to_float(h_O_cpu[i]) << " ";
    }
    std::cout << std::endl;
    
    // Check if results match
    bool passed = (max_abs_diff < 1e-2f);
    std::cout << "\n=====================================================" << std::endl;
    if (passed) {
        std::cout << "✓ Test PASSED - Results match!" << std::endl;
    } else {
        std::cout << "✗ Test FAILED - Results differ significantly!" << std::endl;
    }
    std::cout << "=====================================================" << std::endl;
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_gpu;
    delete[] h_O_cpu;
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    return passed ? 0 : 1;
}
