#ifndef FLASH_ATTENTION_V1_TILED_D_H
#define FLASH_ATTENTION_V1_TILED_D_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cassert>

// Configuration constants
#ifndef BQ
#define BQ 8           // Query tile size
#endif

#ifndef BK
#define BK 8           // Key/Value tile size
#endif

#ifndef D_TILE_QK
#define D_TILE_QK 16   // Head dimension tile size for Q@K^T
#endif

#ifndef D_TILE_V
#define D_TILE_V 16    // Head dimension tile size for S@V
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 64  // Number of threads per block
#endif

#ifndef D
#define D 128          // Head dimension (must match runtime value)
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

// Scaled matrix multiplication with transpose, tiled along d dimension
// Computes: C = A @ B^T * scale
// A: [m, k], B: [n, k], C: [m, n]
// Tiles the inner dimension k into chunks of d_tile
__device__ void mat_mul_scaled_d_tiled(
    const DATA_TYPE* A, const DATA_TYPE* B, DATA_TYPE* C,
    float scale, int m, int n, int k, int d_tile
) {
    // Initialize output (all threads participate)
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        C[idx] = FLOAT_TO_DATA(0.0f);
    }
    __syncthreads();
    
    // Tile over k dimension
    for (int k_start = 0; k_start < k; k_start += d_tile) {
        int k_end = min(k_start + d_tile, k);
        int k_sub = k_end - k_start;
        
        // Each thread computes partial dot products
        for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
            int i = idx / n;
            int j = idx % n;
            
            float sum = 0.0f;
            for (int kk = 0; kk < k_sub; kk++) {
                sum += DATA_TO_FLOAT(A[idx2d(i, k_start + kk, k)]) *
                       DATA_TO_FLOAT(B[idx2d(j, k_start + kk, k)]);
            }
            C[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(C[idx]) + sum);
        }
        __syncthreads();
    }
    
    // Apply scaling
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        C[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(C[idx]) * scale);
    }
    __syncthreads();
}

// Fused: Subtract vector from each row and exponentiate
// out[i,j] = exp(S[i,j] - v[i])
__device__ void mat_sub_vec_exp(
    const DATA_TYPE* S, const float* v, DATA_TYPE* out,
    int m, int n
) {
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        out[idx] = FLOAT_TO_DATA(expf(DATA_TO_FLOAT(S[idx]) - v[i]));
    }
}

// Fused: Row sum combined with multiply-add, updating l in place
// l[i] = l[i] * alpha[i] + sum_j S[i,j]
__device__ void row_sum_mul_add_inplace(
    const DATA_TYPE* S, float* l, const float* alpha,
    int m, int n
) {
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        float sum_val = 0.0f;
        for (int j = 0; j < n; j++) {
            sum_val += DATA_TO_FLOAT(S[idx2d(i, j, n)]);
        }
        l[i] = l[i] * alpha[i] + sum_val;
    }
}

// Fused scale rows and matrix multiply-add with d-tiling for V
// A = A * v[i] + B @ C
// Tiles C (V matrix) along n dimension using d_tile_v
__device__ void mat_scale_rows_mul_add_d_tiled(
    DATA_TYPE* A, const float* v, const DATA_TYPE* B, const DATA_TYPE* C,
    int m, int n, int k, int d_tile_v
) {
    // First scale A by v
    for (int idx = threadIdx.x; idx < m * n; idx += blockDim.x) {
        int i = idx / n;
        A[idx] = FLOAT_TO_DATA(DATA_TO_FLOAT(A[idx]) * v[i]);
    }
    __syncthreads();
    
    // Then compute matmul in tiles along n dimension
    for (int d_v_start = 0; d_v_start < n; d_v_start += d_tile_v) {
        int d_v_end = min(d_v_start + d_tile_v, n);
        int d_v_sub = d_v_end - d_v_start;
        
        // Each thread computes its assigned elements for this d-tile
        for (int idx = threadIdx.x; idx < m * d_v_sub; idx += blockDim.x) {
            int i = idx / d_v_sub;
            int j_local = idx % d_v_sub;
            int j_global = d_v_start + j_local;
            
            float sum = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                sum += DATA_TO_FLOAT(B[idx2d(i, kk, k)]) * 
                       DATA_TO_FLOAT(C[idx2d(kk, j_global, n)]);
            }
            A[idx2d(i, j_global, n)] = 
                FLOAT_TO_DATA(DATA_TO_FLOAT(A[idx2d(i, j_global, n)]) + sum);
        }
        __syncthreads();
    }
}

// Core tile processing function
__device__ void process_kv_tile(
    const DATA_TYPE* Q_tile, const DATA_TYPE* K_tile, const DATA_TYPE* V_tile,
    float* m, float* l, DATA_TYPE* O_acc,
    int bq, int bk, int d, int d_tile_qk, int d_tile_v,
    DATA_TYPE* S, float* alpha
) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    
    // 1) Compute scores: S = Q_tile @ K_tile^T / sqrt(d)
    //    WITH D-TILING along the d dimension (using d_tile_qk)
    mat_mul_scaled_d_tiled(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d, d_tile_qk);
    __syncthreads();
    
    // 2) Compute rescale factor and update m in place
    for (int i = threadIdx.x; i < bq; i += blockDim.x) {
        float new_max = m[i];
        for (int j = 0; j < bk; j++) {
            float s_val = DATA_TO_FLOAT(S[idx2d(i, j, bk)]);
            if (s_val > new_max) {
                new_max = s_val;
            }
        }
        alpha[i] = expf(m[i] - new_max);
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
    //    WITH D-TILING: V is tiled along d dimension (using d_tile_v)
    mat_scale_rows_mul_add_d_tiled(O_acc, alpha, S, V_tile, bq, d, bk, d_tile_v);
    __syncthreads();
}

// Main flash attention kernel
__global__ void flash_attention_kernel(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime, int d_tile_qk_runtime, int d_tile_v_runtime
) {
    // Verify runtime parameters match compile-time constants
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
    
    float* float_buf = reinterpret_cast<float*>(S + BQ * BK);
    float* m = float_buf;                                         // [BQ]
    float* l = m + BQ;                                            // [BQ]
    float* alpha = l + BQ;                                        // [BQ]
    
    // Load Q tile (all threads participate)
    for (int idx = threadIdx.x; idx < q_len * D; idx += blockDim.x) {
        int i = idx / D;
        int j = idx % D;
        Q_tile[idx2d(i, j, D)] = Q[base_offset + idx2d(q_start + i, j, D)];
    }
    
    // Initialize streaming state
    for (int i = threadIdx.x; i < BQ; i += blockDim.x) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }
    for (int i = threadIdx.x; i < BQ * D; i += blockDim.x) {
        O_acc[i] = FLOAT_TO_DATA(0.0f);
    }
    __syncthreads();
    
    // Loop over K/V tiles
    for (int k_start = 0; k_start < L; k_start += BK) {
        int k_end = min(k_start + BK, L);
        int k_len = k_end - k_start;
        
        // Load K tile
        for (int idx = threadIdx.x; idx < k_len * D; idx += blockDim.x) {
            int i = idx / D;
            int j = idx % D;
            K_tile[idx2d(i, j, D)] = K[base_offset + idx2d(k_start + i, j, D)];
        }
        
        // Load V tile
        for (int idx = threadIdx.x; idx < k_len * D; idx += blockDim.x) {
            int i = idx / D;
            int j = idx % D;
            V_tile[idx2d(i, j, D)] = V[base_offset + idx2d(k_start + i, j, D)];
        }
        __syncthreads();
        
        // Process this KV tile
        process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc,
                       q_len, k_len, d_runtime, d_tile_qk_runtime, d_tile_v_runtime, S, alpha);
    }
    
    // Finalize and store to output
    for (int i = threadIdx.x; i < q_len * D; i += blockDim.x) {
        int row = i / D;
        int col = i % D;
        O[base_offset + idx2d(q_start + row, col, D)] = 
            FLOAT_TO_DATA(DATA_TO_FLOAT(O_acc[idx2d(row, col, D)]) / l[row]);
    }
}

// Host function to launch flash attention
void flash_attention_v1(
    const DATA_TYPE* Q, const DATA_TYPE* K, const DATA_TYPE* V,
    DATA_TYPE* O,
    int B, int H, int L, int d_runtime, int d_tile_qk_runtime, int d_tile_v_runtime
) {
    // Safety checks
    static_assert((BQ & (BQ - 1)) == 0, "BQ must be a power of 2");
    static_assert((BK & (BK - 1)) == 0, "BK must be a power of 2");
    static_assert((THREADS_PER_BLOCK & (THREADS_PER_BLOCK - 1)) == 0, "THREADS_PER_BLOCK must be a power of 2");
    static_assert(BQ > 0 && BK > 0 && THREADS_PER_BLOCK > 0, "Constants must be positive");
    
    // Runtime checks
    assert(B > 0 && H > 0 && L > 0 && d_runtime > 0 && "All dimensions must be positive");
    assert(d_runtime == D && "Runtime d must match compile-time D constant");
    assert(d_tile_qk_runtime > 0 && d_tile_qk_runtime <= d_runtime && "d_tile_qk must be valid");
    assert(d_tile_v_runtime > 0 && d_tile_v_runtime <= d_runtime && "d_tile_v must be valid");
    
    // Calculate shared memory size
    size_t shared_mem_size = (
        BQ * D +      // Q_tile
        BK * D +      // K_tile
        BK * D +      // V_tile
        BQ * D +      // O_acc
        BQ * BK       // S
    ) * sizeof(DATA_TYPE) + (
        BQ +          // m
        BQ +          // l
        BQ            // alpha
    ) * sizeof(float);
    
    // Check shared memory limit
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    assert(shared_mem_size <= prop.sharedMemPerBlock && 
           "Shared memory requirement exceeds device limit");
    
    // Launch kernel with 2D grid: x = query tiles, y = batch × heads
    dim3 grid_dim((L + BQ - 1) / BQ, B * H);
    
    flash_attention_kernel<<<grid_dim, THREADS_PER_BLOCK, shared_mem_size>>>(
        Q, K, V, O, B, H, L, d_runtime, d_tile_qk_runtime, d_tile_v_runtime
    );
    
    cudaDeviceSynchronize();
}

#endif // FLASH_ATTENTION_V1_TILED_D_H
