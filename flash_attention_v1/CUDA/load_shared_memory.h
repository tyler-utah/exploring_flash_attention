#ifndef CACHE_SHARED_MEMORY_V2_H
#define CACHE_SHARED_MEMORY_V2_H

#include <cuda_fp16.h>
#include <type_traits>

// Use macros from kernel if defined
#if !defined(index2D)
#define index2D(i, j, stride) ((i) * (stride) + (j))
#endif

#if !defined(get_flattened_id)
#define get_flattened_id() (threadIdx.y * blockDim.x + threadIdx.x)
#endif

// ============================================================================
// CORE DESIGN PRINCIPLES
// ============================================================================
// 1. Think in BYTES, not element types
// 2. Maximize thread utilization (use all threads in block)
// 3. Choose vector width based on: total_bytes and thread_count
// 4. Modular: separate functions for transpose vs non-transpose
// 5. Compile-time path selection (no runtime branches when possible)
// ============================================================================

namespace cache_memory {

// Vector size options (in bytes)
enum class VectorSize {
    VEC16 = 16,  // float4 / int4
    VEC8  = 8,   // float2 / int2 / double
    VEC4  = 4,   // float / int
    SCALAR = 1   // byte-by-byte
};

// Determine optimal vector size based on total bytes and thread count
template <int total_bytes, int thread_count>
__device__ constexpr VectorSize select_vector_size() {
    // Goal: Use largest vector size where (thread_count * vec_size <= total_bytes)
    // This ensures all threads can participate
    
    if constexpr (thread_count * 16 <= total_bytes && total_bytes % 16 == 0) {
        return VectorSize::VEC16;
    } else if constexpr (thread_count * 8 <= total_bytes && total_bytes % 8 == 0) {
        return VectorSize::VEC8;
    } else if constexpr (thread_count * 4 <= total_bytes && total_bytes % 4 == 0) {
        return VectorSize::VEC4;
    } else {
        return VectorSize::SCALAR;
    }
}

// Helper to get vector type from size
template <VectorSize VS>
struct VectorType;

template <> struct VectorType<VectorSize::VEC16> { using type = int4; };
template <> struct VectorType<VectorSize::VEC8>  { using type = int2; };
template <> struct VectorType<VectorSize::VEC4>  { using type = int; };
template <> struct VectorType<VectorSize::SCALAR> { using type = char; };

// ============================================================================
// NON-TRANSPOSE TRANSFER
// ============================================================================

template <typename T, int dim0, int dim1, int flattened_dim, VectorSize VS>
__device__ __forceinline__ void transfer_non_transpose_impl(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    // Use float for 4-byte __half pairs to match original behavior
    using VecT = typename std::conditional<
        std::is_same_v<T, __half> && VS == VectorSize::VEC4,
        float,
        typename VectorType<VS>::type
    >::type;
    
    constexpr int vec_bytes = static_cast<int>(VS);
    constexpr int vec_elements = vec_bytes / sizeof(T);
    constexpr int total_elements = dim0 * dim1;
    
    const uint flattened_id = get_flattened_id();
    
    // Calculate position - match original's order for better optimization
    const uint new_j = (flattened_id * vec_elements) % dim1;
    const uint new_i = (flattened_id * vec_elements) / dim1;
    
    // Perfect fit: each thread loads exactly one vector
    if constexpr (flattened_dim * vec_elements == total_elements) {
        const uint i = new_i;
        VecT* new_target = reinterpret_cast<VecT*>(&target[index2D(i, new_j, dim1 + target_padding)]);
        const VecT* new_source = reinterpret_cast<const VecT*>(&src_ptr[index2D(i, new_j, src_stride)]);
        *new_target = *new_source;
    }
    // More vectors than threads: each thread loads multiple vectors
    else if constexpr (flattened_dim * vec_elements < total_elements) {
        constexpr int stride_i = (flattened_dim * vec_elements) / dim1;
        #pragma unroll
        for (uint outer = 0; outer < dim0 / stride_i; outer++) {
            const uint i = outer * stride_i + new_i;
            VecT* new_target = reinterpret_cast<VecT*>(&target[index2D(i, new_j, dim1 + target_padding)]);
            const VecT* new_source = reinterpret_cast<const VecT*>(&src_ptr[index2D(i, new_j, src_stride)]);
            *new_target = *new_source;
        }
    }
    // More threads than vectors: only first N threads participate
    else {
        constexpr int total_vectors = total_elements / vec_elements;
        if (flattened_id < total_vectors) {
            const uint i = new_i;
            VecT* new_target = reinterpret_cast<VecT*>(&target[index2D(i, new_j, dim1 + target_padding)]);
            const VecT* new_source = reinterpret_cast<const VecT*>(&src_ptr[index2D(i, new_j, src_stride)]);
            *new_target = *new_source;
        }
    }
}

// Scalar fallback for non-transpose (when no vectorization possible)
template <typename T, int dim0, int dim1, int flattened_dim>
__device__ __forceinline__ void transfer_non_transpose_scalar(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    const uint flattened_id = get_flattened_id();
    constexpr int total_elements = dim0 * dim1;
    
    for (uint elem = flattened_id; elem < total_elements; elem += flattened_dim) {
        const uint row = elem / dim1;
        const uint col = elem % dim1;
        target[index2D(row, col, dim1 + target_padding)] = src_ptr[index2D(row, col, src_stride)];
    }
}

// Main non-transpose dispatcher
template <typename T, int dim0, int dim1, int flattened_dim>
__device__ __forceinline__ void transfer_non_transpose(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    constexpr int total_bytes = dim0 * dim1 * sizeof(T);
    constexpr VectorSize vs = select_vector_size<total_bytes, flattened_dim>();
    
    if constexpr (vs == VectorSize::VEC16) {
        transfer_non_transpose_impl<T, dim0, dim1, flattened_dim, VectorSize::VEC16>(target, src_ptr, src_stride, target_padding);
    } else if constexpr (vs == VectorSize::VEC8) {
        transfer_non_transpose_impl<T, dim0, dim1, flattened_dim, VectorSize::VEC8>(target, src_ptr, src_stride, target_padding);
    } else if constexpr (vs == VectorSize::VEC4) {
        transfer_non_transpose_impl<T, dim0, dim1, flattened_dim, VectorSize::VEC4>(target, src_ptr, src_stride, target_padding);
    } else {
        transfer_non_transpose_scalar<T, dim0, dim1, flattened_dim>(target, src_ptr, src_stride, target_padding);
    }
}

// ============================================================================
// TRANSPOSE TRANSFER
// ============================================================================

template <typename T, int dim0, int dim1, int flattened_dim, VectorSize VS>
__device__ __forceinline__ void transfer_transpose_impl(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    using VecT = typename VectorType<VS>::type;
    constexpr int vec_bytes = static_cast<int>(VS);
    constexpr int vec_elements = vec_bytes / sizeof(T);
    constexpr int total_bytes = dim0 * dim1 * sizeof(T);
    constexpr int total_vectors = total_bytes / vec_bytes;
    
    const int flattened_id = get_flattened_id();
    
    // For transpose with vectorization, we need to be more careful
    // We can only vectorize if the transposed access pattern allows it
    // For now, fall back to scalar for transpose with vectors
    // TODO: Implement smart transpose vectorization
    
    constexpr int total_elements = dim0 * dim1;
    for (int elem = flattened_id; elem < total_elements; elem += flattened_dim) {
        const int row = elem / dim1;
        const int col = elem % dim1;
        // Transpose: read (col, row) from source, write (row, col) to target
        target[index2D(row, col, dim1 + target_padding)] = src_ptr[index2D(col, row, src_stride)];
    }
}

// Scalar transpose
template <typename T, int dim0, int dim1, int flattened_dim>
__device__ __forceinline__ void transfer_transpose_scalar(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    const int flattened_id = get_flattened_id();
    constexpr int total_elements = dim0 * dim1;
    
    for (int elem = flattened_id; elem < total_elements; elem += flattened_dim) {
        const int row = elem / dim1;
        const int col = elem % dim1;
        target[index2D(row, col, dim1 + target_padding)] = src_ptr[index2D(col, row, src_stride)];
    }
}

// Main transpose dispatcher
template <typename T, int dim0, int dim1, int flattened_dim>
__device__ __forceinline__ void transfer_transpose(
    T* target,
    const T* src_ptr,
    int src_stride,
    int target_padding
) {
    // For now, use scalar for transpose (can be optimized later)
    transfer_transpose_scalar<T, dim0, dim1, flattened_dim>(target, src_ptr, src_stride, target_padding);
}

} // namespace cache_memory

// ============================================================================
// PUBLIC API
// ============================================================================
// Copies a tile from global memory to shared memory (optionally transposed)
//
// Template parameters:
//   T              - Element type (__half, float, etc.)
//   dim0           - Target tile height (rows)
//   dim1           - Target tile width (columns)
//   flattened_dim  - Total number of threads (blockDim.x * blockDim.y)
//   transpose      - If true, transpose during copy (src[col][row] -> target[row][col])
//
// Function parameters:
//   target         - Destination pointer (shared memory)
//   src            - Source base pointer (global memory)
//   src_row_offset - Row offset into source matrix
//   src_col_offset - Column offset into source matrix
//   src_stride     - Source leading dimension (stride between rows)
//   bar            - Barrier for async operations (currently unused)
//   target_padding - Padding offset for target (for bank conflict avoidance)
//
// Follows memcpy convention: destination first, source second
// Source address computed as: src + (src_row_offset * src_stride + src_col_offset)
//
template <typename T, int dim0, int dim1, int flattened_dim, bool transpose>
__device__ void cache_shared_memory(
    T* target,
    const T* src,
    int src_row_offset,
    int src_col_offset,
    int src_stride,
    void* bar = nullptr,
    int target_padding = 0
) {
    // Calculate the source pointer with proper offset
    const T* src_ptr = src + src_row_offset * src_stride + src_col_offset;
    
    if constexpr (transpose) {
        cache_memory::transfer_transpose<T, dim0, dim1, flattened_dim>(
            target, src_ptr, src_stride, target_padding);
    } else {
        cache_memory::transfer_non_transpose<T, dim0, dim1, flattened_dim>(
            target, src_ptr, src_stride, target_padding);
    }
}

#endif // CACHE_SHARED_MEMORY_V2_H
