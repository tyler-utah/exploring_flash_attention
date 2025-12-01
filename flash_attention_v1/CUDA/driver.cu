#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "standard.h"

// Kernel selection via compile flag
#ifndef USE_OPT1
#define USE_OPT1 0
#endif

#if USE_OPT1
#include "flash_attention_v1_opt1.h"
#define flash_attention_v1 flash_attention_v1_opt1
#define flash_attention_kernel flash_attention_kernel_opt1
#define KERNEL_NAME "flash_attention_v1_opt1"
#else
#include "flash_attention_v1.h"
#define KERNEL_NAME "flash_attention_v1"
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
        data[i] = float_to_data(((float)rand() / RAND_MAX) * 2.0f - 1.0f); // Range [-1, 1]
    }
}

// Compare two arrays and compute errors
void compare_arrays(const DATA_TYPE* a, const DATA_TYPE* b, int size, 
                    float& max_abs_diff, float& max_rel_diff, float& max_rel_diff_all) {
    max_abs_diff = 0.0f;
    max_rel_diff = 0.0f;
    max_rel_diff_all = 0.0f;
#if USE_FP64
    const float eps = 1e-10f; // Threshold for FP64
#else
    const float eps = 1e-3f;  // Threshold for FP16
#endif
    int max_rel_idx = -1;
    int max_rel_all_idx = -1;
    
    for (int i = 0; i < size; i++) {
        float val_a = data_to_float(a[i]);
        float val_b = data_to_float(b[i]);
        float abs_diff = fabsf(val_a - val_b);
        
        if (abs_diff > max_abs_diff) {
            max_abs_diff = abs_diff;
        }
        
        // Compute relative error for ALL values (using symmetric denominator)
        float magnitude_all = fmaxf(fmaxf(fabsf(val_a), fabsf(val_b)), 1e-8f);
        float rel_diff_all = abs_diff / magnitude_all;
        if (rel_diff_all > max_rel_diff_all) {
            max_rel_diff_all = rel_diff_all;
            max_rel_all_idx = i;
        }
        
        // Also compute relative error only for values significantly above noise floor
        float magnitude = fmaxf(fabsf(val_a), fabsf(val_b));
        if (magnitude > eps) {
            float rel_diff = abs_diff / magnitude;
            if (rel_diff > max_rel_diff) {
                max_rel_diff = rel_diff;
                max_rel_idx = i;
            }
        }
    }
    
    // Print details about max relative difference location (filtered)
    if (max_rel_idx >= 0) {
        std::cout << "  Max filtered relative diff at index " << max_rel_idx 
                  << ": ref=" << data_to_float(a[max_rel_idx])
                  << ", flash=" << data_to_float(b[max_rel_idx])
                  << ", abs_diff=" << fabsf(data_to_float(a[max_rel_idx]) - data_to_float(b[max_rel_idx])) << std::endl;
    }
    // Print details about max relative difference (all values)
    if (max_rel_all_idx >= 0) {
        std::cout << "  Max unfiltered relative diff at index " << max_rel_all_idx 
                  << ": ref=" << data_to_float(a[max_rel_all_idx])
                  << ", flash=" << data_to_float(b[max_rel_all_idx])
                  << ", abs_diff=" << fabsf(data_to_float(a[max_rel_all_idx]) - data_to_float(b[max_rel_all_idx])) << std::endl;
    }
}

int main() {
    // Set random seed
    srand(42);
    
    // Problem size
    const int B = 32;   // Batch size
    const int H = 8;    // Number of heads
    const int L = 1024; // Sequence length
    const int d = 32;   // Head dimension
    
    std::cout << "Flash Attention CUDA Test" << std::endl;
    std::cout << "Kernel: " << KERNEL_NAME << std::endl;
#if USE_FP64
    std::cout << "Precision: FP64 (double)" << std::endl;
#else
    std::cout << "Precision: FP16 (half)" << std::endl;
#endif
    std::cout << "B=" << B << ", H=" << H << ", L=" << L << ", d=" << d 
              << ", BQ=" << BQ << ", BK=" << BK 
              << ", THREADS_PER_BLOCK=" << THREADS_PER_BLOCK << std::endl;
    std::cout << "Total attention heads: " << B * H << std::endl;
    std::cout << std::endl;
    
    // Allocate host memory for [B, H, L, d] tensors
    size_t total_size = B * H * L * d;
    size_t matrix_size = total_size * sizeof(DATA_TYPE);
    DATA_TYPE* h_Q = new DATA_TYPE[total_size];
    DATA_TYPE* h_K = new DATA_TYPE[total_size];
    DATA_TYPE* h_V = new DATA_TYPE[total_size];
    DATA_TYPE* h_O_ref = new DATA_TYPE[total_size];   // Reference output
    DATA_TYPE* h_O_flash = new DATA_TYPE[total_size]; // Flash attention output
    
    // Initialize input data
    initialize_random(h_Q, total_size);
    initialize_random(h_K, total_size);
    initialize_random(h_V, total_size);
    
    // Compute reference output using CPU standard attention
    std::cout << "Computing reference (standard attention on CPU with OpenMP)..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    standard_attention_cpu(h_Q, h_K, h_V, h_O_ref, B, H, L, d);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "Reference computation complete." << std::endl;
    std::cout << "CPU time: " << cpu_duration.count() << " ms" << std::endl;
    std::cout << std::endl;
    
    // Allocate device memory
    DATA_TYPE *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_K, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_V, matrix_size));
    CUDA_CHECK(cudaMalloc(&d_O, matrix_size));
    
    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, matrix_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, matrix_size, cudaMemcpyHostToDevice));
    
    // Calculate and configure shared memory
    size_t shared_mem_size = (
        BQ * D +      // Q_tile (DATA_TYPE)
        BK * D +      // K_tile (DATA_TYPE)
        BK * D +      // V_tile (DATA_TYPE)
        BQ * D +      // O_acc (DATA_TYPE)
        BQ * BK       // S (DATA_TYPE)
    ) * sizeof(DATA_TYPE) + (
        BQ +          // m (float)
        BQ +          // l (float)
        BQ            // alpha (float)
    ) * sizeof(float);
    
    std::cout << "Configuring shared memory (" << shared_mem_size << " bytes)..." << std::endl;
    if (!configure_shared_memory(shared_mem_size)) {
        std::cerr << "Failed to configure shared memory. Exiting." << std::endl;
        return 1;
    }
    std::cout << "Shared memory configured successfully." << std::endl;
    std::cout << std::endl;
    
    // Run flash attention on GPU
    std::cout << "Running Flash Attention V1 on GPU..." << std::endl;
    
    // Warm-up runs (10 iterations)
    std::cout << "Warming up (10 runs)..." << std::endl;
    for (int i = 0; i < 10; i++) {
        flash_attention_v1(d_Q, d_K, d_V, d_O, B, H, L, d);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Warm-up complete." << std::endl;
    
    // Timed runs (50 iterations)
    std::cout << "Running timed iterations (50 runs)..." << std::endl;
    const int num_timed_runs = 50;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_timed_runs; i++) {
        flash_attention_v1(d_Q, d_K, d_V, d_O, B, H, L, d);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_duration = gpu_end - gpu_start;
    double avg_gpu_time = gpu_duration.count() / num_timed_runs;
    
    std::cout << "Flash Attention complete." << std::endl;
    std::cout << "Average GPU time (50 runs): " << avg_gpu_time << " ms" << std::endl;
    std::cout << "Total GPU time: " << gpu_duration.count() << " ms" << std::endl;
    std::cout << "Speedup: " << cpu_duration.count() / avg_gpu_time << "x" << std::endl;
    std::cout << std::endl;
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_O_flash, d_O, matrix_size, cudaMemcpyDeviceToHost));
    
    // Compare results
    float max_abs_diff, max_rel_diff, max_rel_diff_all;
    compare_arrays(h_O_ref, h_O_flash, total_size, max_abs_diff, max_rel_diff, max_rel_diff_all);
    std::cout << "Results Comparison:" << std::endl;
    std::cout << "Max absolute difference: " << max_abs_diff << std::endl;
    std::cout << "Max relative difference (all values): " << max_rel_diff_all << std::endl;
#if USE_FP64
    std::cout << "Max relative difference (for |values| > 1e-10): " << max_rel_diff << std::endl;
#else
    std::cout << "Max relative difference (for |values| > 1e-3): " << max_rel_diff << std::endl;
#endif
    
    // Print first few values for inspection
    std::cout << std::endl << "First 5 output values:" << std::endl;
    std::cout << "Reference: ";
    for (int i = 0; i < 5; i++) {
        std::cout << data_to_float(h_O_ref[i]) << " ";
    }
    std::cout << std::endl;
    std::cout << "Flash:     ";
    for (int i = 0; i < 5; i++) {
        std::cout << data_to_float(h_O_flash[i]) << " ";
    }
    std::cout << std::endl;
    
    // Check if results match
    if (max_abs_diff < 1e-3f) {
        std::cout << std::endl << "✓ Test PASSED - Results match!" << std::endl;
    } else {
        std::cout << std::endl << "✗ Test FAILED - Results differ significantly!" << std::endl;
    }
    
    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_O_ref;
    delete[] h_O_flash;
    
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
    
    return 0;
}
