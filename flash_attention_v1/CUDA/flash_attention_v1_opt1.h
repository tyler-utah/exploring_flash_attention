#ifndef FLASH_ATTENTION_V1_OPT1_H
#define FLASH_ATTENTION_V1_OPT1_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>
#include <cassert>
#include "load_shared_memory.h"

using namespace nvcuda;

// Configuration constants (must be powers of 2)
// Can be overridden at compile time for autotuning
#ifndef BQ
#define BQ 16           // Query tile size
#endif

#ifndef BK
#define BK 16           // Key/Value tile size
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 32  // Number of threads per block (1 warp for Tensor Cores)
#endif

#ifndef D
#define D 32            // Head dimension (must match runtime value)
#endif

// Precision configuration
#ifndef USE_FP64
#define USE_FP64 0      // Set to 1 for double precision, 0 for half precision
#endif

#if USE_FP64
#define DATA_TYPE double
#define FLOAT_TO_DATA(x) (x)
#define DATA_TO_FLOAT(x) (x)
#else
#define DATA_TYPE __half
#define FLOAT_TO_DATA(x) __float2half(x)
#define DATA_TO_FLOAT(x) __half2float(x)
#endif

// Helper function: 2D to 1D index conversion (row-major)
__device__ __host__ inline int idx2d(int i, int j, int cols) {
    return i * cols + j;
}

// Scaled matrix multiplication with transpose using Tensor Cores (WMMA)
// Computes: C = A @ B^T * scale
// A: [m, k], B: [n, k], C: [m, n]
// Requires: m, n, k must be multiples of 16 for optimal WMMA usage
__device__ void mat_mul_scaled_basic(
    const DATA_TYPE* A, const DATA_TYPE* B, DATA_TYPE* C,
    float scale, int m, int n, int k
) {
#if !USE_FP64  // WMMA only supports FP16, fall back to basic for FP64
    using namespace nvcuda::wmma;
    
    // WMMA fragment dimensions: 16x16x16
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Use WMMA if dimensions are multiples of 16
    if (m == WMMA_M && n == WMMA_N && k % WMMA_K == 0) {
        // Each warp computes one 16x16 output tile
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
        
        if (warp_id < num_warps) {
            // Declare WMMA fragments
            // A is row_major, B needs col_major to compute A @ B^T
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
            
            // Initialize accumulator to zero
            fill_fragment(c_frag, __float2half(0.0f));
            
            // Iterate over k dimension in WMMA_K chunks
            for (int kk = 0; kk < k; kk += WMMA_K) {
                // Load A tile: [WMMA_M, WMMA_K] from A[0:16, kk:kk+16]
                load_matrix_sync(a_frag, A + kk, k);
                
                // Load B tile: [WMMA_N, WMMA_K] from B[0:16, kk:kk+16]
                // B is stored row_major but we load as col_major to get transpose effect
                load_matrix_sync(b_frag, B + kk, k);
                
                // Perform matrix multiply-accumulate: c_frag += a_frag @ b_frag^T
                mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            
            // Scale the result
            for (int i = 0; i < c_frag.num_elements; i++) {
                c_frag.x[i] = __float2half(__half2float(c_frag.x[i]) * scale);
            }
            
            // Store result back to C
            store_matrix_sync(C, c_frag, n, mem_row_major);
        }
        return;
    }
#endif
    
    // Fallback to basic implementation for non-WMMA sizes or FP64
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        float sum = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            sum += DATA_TO_FLOAT(A[i * k + kk]) * DATA_TO_FLOAT(B[j * k + kk]);
        }
        C[idx] = FLOAT_TO_DATA(sum * scale);
    }
}

// Fused: Subtract vector from each row and exponentiate
// out[i,j] = exp(S[i,j] - v[i])
__device__ void mat_sub_vec_exp(
    const DATA_TYPE* S, const float* v, DATA_TYPE* out,
    int m, int n
) {
    // Parallelize over all elements for better thread utilization
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        out[idx] = FLOAT_TO_DATA(expf(DATA_TO_FLOAT(S[idx]) - v[i]));
    }
}

// NOTE: Warp-level reductions were tested but found to be slower than simple sequential loops
// at current tile sizes (BQ=16, BK=16). The overhead of shuffle operations and warp synchronization
// dominates when vectors are small (BK=16 means only 0.5 elements per thread in a 32-thread warp).
// Warp optimizations would likely benefit from larger tiles (BK >= 64) or more threads per block.

// Fused: Row sum combined with multiply-add, updating l in place
// l[i] = l[i] * alpha[i] + sum_j S[i,j]
__device__ void row_sum_mul_add_inplace(
    const DATA_TYPE* S, float* l, const float* alpha,
    int m, int n
) {
    // Parallelize per row (each thread handles complete rows)
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        float sum_val = 0.0f;
        for (int j = 0; j < n; j++) {
            sum_val += DATA_TO_FLOAT(S[idx2d(i, j, n)]);
        }
        l[i] = l[i] * alpha[i] + sum_val;
    }
}

// Fused scale rows and matrix multiply-add with Tensor Cores (WMMA)
// A = A * v[i] + B @ C
// A: [m, n], v: [m], B: [m, k], C: [k, n]
__device__ void mat_scale_rows_mul_add_basic(
    DATA_TYPE* A, const float* v, const DATA_TYPE* B, const DATA_TYPE* C,
    int m, int n, int k
) {
#if !USE_FP64  // WMMA only supports FP16
    using namespace nvcuda::wmma;
    
    // WMMA fragment dimensions: 16x16x16
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Use WMMA if dimensions are compatible (m=16, k=16, n=32)
    if (m == WMMA_M && k == WMMA_K && n == 32) {
        const int warp_id = threadIdx.x / 32;
        const int num_warps = blockDim.x / 32;
        
        if (warp_id < num_warps) {
            // First, scale existing A values: A = A * v[i]
            for (int idx = threadIdx.x; idx < m * n; idx += 32) {
                int i = idx / n;
                A[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(A[idx]) * v[i]);
            }
            __syncwarp();
            
            // Now compute B @ C using WMMA and add to scaled A
            // Process n dimension in 16-column tiles (n=32 means 2 tiles)
            for (int n_tile = 0; n_tile < 2; n_tile++) {
                // Declare WMMA fragments
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> c_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
                
                // Load scaled A as initial accumulator
                load_matrix_sync(acc_frag, A + n_tile * WMMA_N, n, mem_row_major);
                
                // Compute B @ C and accumulate: acc = A_scaled + B @ C
                load_matrix_sync(b_frag, B, k);
                load_matrix_sync(c_frag, C + n_tile * WMMA_N, n);
                mma_sync(acc_frag, b_frag, c_frag, acc_frag);
                
                // Store result back to A
                store_matrix_sync(A + n_tile * WMMA_N, acc_frag, n, mem_row_major);
            }
        }
        return;
    }
#endif
    
    // Fallback to basic implementation
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        int j = idx % n;
        float sum = 0.0f;
        for (int kk = 0; kk < k; kk++) {
            sum += DATA_TO_FLOAT(B[i * k + kk]) * DATA_TO_FLOAT(C[kk * n + j]);
        }
        // Fuse scaling with addition: A[i,j] = A[i,j] * v[i] + sum
        A[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(A[idx]) * v[i] + sum);
    }
}

// Core tile processing function
__device__ void process_kv_tile(
    const DATA_TYPE* Q_tile, const DATA_TYPE* K_tile, const DATA_TYPE* V_tile,
    float* m, float* l, DATA_TYPE* O_acc,
    int bq, int bk, int d,
    DATA_TYPE* S, float* alpha  // Shared memory buffers passed in
) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // 1) Compute scores: S = Q_tile @ K_tile^T / sqrt(d)
    mat_mul_scaled_basic(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d);
    __syncthreads();
    
    // 2) Compute rescale factor and update m in place (parallel per row)
    for (int i = threadIdx.x; i < bq; i += blockDim.x) {
        // Find new max for this row (initialized with running max)
        float new_max = m[i];
        for (int j = 0; j < bk; j++) {
            float s_val = DATA_TO_FLOAT(S[idx2d(i, j, bk)]);
            if (s_val > new_max) {
                new_max = s_val;
            }
        }
        // Compute alpha before overwriting m
        alpha[i] = expf(m[i] - new_max);
        // Update m in place
        m[i] = new_max;
    }
    __syncthreads();
    
    // 3) Fused: Shift scores and exponentiate in place
    mat_sub_vec_exp(S, m, S, bq, bk);
    __syncthreads();
    
    // 4) Fused: Row sum and update running denom in place
    row_sum_mul_add_inplace(S, l, alpha, bq, bk);
    __syncthreads();
    
    // 5) Fused: Rescale old numerator and accumulate contribution
    mat_scale_rows_mul_add_basic(O_acc, alpha, S, V_tile, bq, d, bk);
    __syncthreads();
}

// Main flash attention kernel
__global__ void flash_attention_kernel_opt1(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime
) {
    // Verify runtime d matches compile-time D
    assert(d_runtime == D && "Runtime d must match compile-time D constant");
    
    // blockIdx.y encodes (batch, head) pair
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;
    
    // Each block processes one query tile for one (batch, head) pair
    int q_start = blockIdx.x * BQ;
    if (q_start >= L) return;
    
    int q_end = min(q_start + BQ, L);
    int q_len = q_end - q_start;
    
    // Compute base offset for this (batch, head) pair: [B, H, L, d]
    const int base_offset = (batch_idx * H * L * d_runtime) + (head_idx * L * d_runtime);
    
    // Shared memory allocation
    extern __shared__ char shared_mem_raw[];
    DATA_TYPE* shared_mem_data = reinterpret_cast<DATA_TYPE*>(shared_mem_raw);
    
    DATA_TYPE* Q_tile = shared_mem_data;                          // [BQ * D]
    DATA_TYPE* K_tile = Q_tile + BQ * D;                          // [BK * D]
    DATA_TYPE* V_tile = K_tile + BK * D;                          // [BK * D]
    DATA_TYPE* O_acc = V_tile + BK * D;                           // [BQ * D]
    DATA_TYPE* S = O_acc + BQ * D;                                // [BQ * BK]
    
    // Float buffers after data buffers
    float* float_buf = reinterpret_cast<float*>(S + BQ * BK);
    float* m = float_buf;                                      // [BQ]
    float* l = m + BQ;                                         // [BQ]
    float* alpha = l + BQ;                                     // [BQ]
    
    // Load Q tile in parallel (all threads participate)
    cache_shared_memory<DATA_TYPE, BQ, D, THREADS_PER_BLOCK, false>(
        Q_tile,              // target (shared memory)
        Q + base_offset,     // source (global memory) - offset to this batch/head
        q_start,             // src_row_offset
        0,                   // src_col_offset
        d_runtime,           // src_stride (leading dimension)
        nullptr,             // bar (unused)
        0                    // target_padding
    );
    
    // Initialize streaming state in parallel
    for (int i = threadIdx.x; i < BQ; i += THREADS_PER_BLOCK) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BQ * D; i += THREADS_PER_BLOCK) {
        O_acc[i] = FLOAT_TO_DATA(0.0f);
    }
    __syncthreads();
    
    // Loop over K/V tiles
    for (int k_start = 0; k_start < L; k_start += BK) {
        int k_end = min(k_start + BK, L);
        int k_len = k_end - k_start;
        
        // Load K tile in parallel (all threads participate)
        cache_shared_memory<DATA_TYPE, BK, D, THREADS_PER_BLOCK, false>(
            K_tile, K + base_offset, k_start, 0, d_runtime, nullptr, 0);
        
        // Load V tile in parallel (all threads participate)
        cache_shared_memory<DATA_TYPE, BK, D, THREADS_PER_BLOCK, false>(
            V_tile, V + base_offset, k_start, 0, d_runtime, nullptr, 0);
        __syncthreads();
        
        // Process this KV tile (all threads participate)
        process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc,
                      q_len, k_len, d_runtime, S, alpha);
        // Sync before next iteration to ensure process_kv_tile completes before overwriting K/V tiles
        __syncthreads();
    }
    
    // Finalize and store to output in parallel
    for (int i = threadIdx.x; i < q_len * D; i += THREADS_PER_BLOCK) {
        int row = i / D;
        int col = i % D;
        O[base_offset + idx2d(q_start + row, col, D)] = FLOAT_TO_DATA(DATA_TO_FLOAT(O_acc[idx2d(row, col, D)]) / l[row]);
    }
}

// Host function to launch flash attention opt1
void flash_attention_v1_opt1(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime
) {
    // Safety checks: ensure constants are powers of 2
    static_assert((BQ & (BQ - 1)) == 0, "BQ must be a power of 2");
    static_assert((BK & (BK - 1)) == 0, "BK must be a power of 2");
    static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0, "THREADS_PER_BLOCK must be a power of 2");
    static_assert(BQ > 0 && BK > 0 && THREADS_PER_BLOCK > 0, "Constants must be positive");
    
    // Runtime checks
    assert(B > 0 && H > 0 && L > 0 && d_runtime > 0 && "All dimensions must be positive");
    assert(d_runtime == D && "Runtime d must match compile-time D constant");
    
    // Calculate shared memory size
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
    
    // Check shared memory limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(shared_mem_size <= prop.sharedMemPerBlock && 
           "Shared memory requirement exceeds device limit. Reduce BQ or BK.");
    
    // Launch kernel with 2D grid: x = query tiles, y = batch × heads
    dim3 grid_dim((L + BQ - 1) / BQ, B * H);
    
    flash_attention_kernel_opt1<<<grid_dim, THREADS_PER_BLOCK, shared_mem_size>>>(
        Q, K, V, O, B, H, L, d_runtime
    );
    
    cudaDeviceSynchronize();
}

#endif // FLASH_ATTENTION_V1_OPT1_H
