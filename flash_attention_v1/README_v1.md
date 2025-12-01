# Flash Attention V1 Implementations

This directory contains multiple implementations of the Flash Attention algorithm:

## Files

- **`pytorch_imp.py`**: PyTorch reference implementation for comparison purposes. Uses `F.scaled_dot_product_attention` to verify correctness of custom implementations.

- **`numpy_basic.py`**: Basic NumPy implementation using high-level NumPy functions and vectorized operations. Straightforward but uses Python/NumPy abstractions.

- **`numpy_gpu_like.py`**: C-style NumPy implementation that resembles GPU kernel code. Uses explicit loops, manual indexing, pre-allocated buffers, and in-place modifications instead of NumPy abstractions. All matrices use 2D arrays. Designed to be easily translatable to GPU kernels.

- **`numpy_gpu_like_1D.py`**: Further optimized C-style implementation where all memory is allocated as 1D arrays (like C/GPU). Uses an `idx2d(i, j, cols)` helper function to convert 2D indices to 1D offsets (row-major layout). Matrix multiplications use NumPy matmul with reshaping for efficiency. Most closely matches actual GPU kernel memory layout.

- **`numpy_gpu_like_opt1.py`**: Memory-optimized version based on `numpy_gpu_like_1D.py`. Reduces shared memory usage in the critical `process_kv_tile` function by reusing buffers:
  - **S buffer reused 3x**: scores → shifted scores → exp scores (saves 2 × [Bq×Bk] arrays)
  - **m_new reused 2x**: stores m_block, then updated to final m_new (saves 1 × [Bq] array)
  - **l_new reused 2x**: stores l_tile, then updated to final l_new (saves 1 × [Bq] array)
  - **Total reduction**: From 9 to 5 buffers (eliminates `m_block`, `S_shifted`, `exp_S`, `l_tile`)
  - This optimization is critical for GPU kernels where shared memory is limited.

- **`numpy_gpu_like_opt2.py`**: Fully optimized version with aggressive operation fusion and in-place updates. Building on opt1, this implementation:
  - **Operation fusion**: Combines sequential operations to reduce memory traffic:
    * `mat_sub_vec + mat_exp → mat_sub_vec_exp`: Subtract and exponentiate in one pass
    * `row_sum + vec_mul_add → row_sum_mul_add_inplace`: Sum and accumulate in place
    * `mat_scale_rows + mat_mul_add → mat_scale_rows_mul_add`: Scale and accumulate matmul
    * `mat_div_vec + store → mat_div_vec_store`: Divide and write directly to output
  - **In-place updates**: Eliminates temporary buffers by directly updating `m` and `l`:
    * Computes rescale factors inline before overwriting old max values
    * Updates running statistics in place (no `vec_copy` calls needed)
  - **Buffer reduction**: Only 2 temporary buffers needed in `process_kv_tile` (S and alpha)
    * From opt1's 5 buffers down to just 2 (eliminates `m_new`, `l_new`, and redundant copies)
  - **Minimal allocations**: Removed 11 unused helper functions; kept only actively-used fused operations
  - **Result**: ~7-8 fewer memory passes through data, 60% reduction in temporary buffers, optimal for GPU translation
  
  This represents the most optimized CPU implementation suitable for direct translation to GPU kernels (CUDA/Triton).

## CUDA Implementation

The **`CUDA/`** directory contains multiple optimization levels of Flash Attention on GPU:

### Optimization Progression

#### Baseline (flash_attention_v1.h)
- **Mixed precision**: FP16 (`__half`) for matrices, FP32 (`float`) for statistics (m, l, alpha)
- **Batched multi-head attention**: Supports [B, H, L, d] tensors with 2D grid launching
- **Parallel data loading**: Vectorized tile loading using `cache_shared_memory` API
- **Element-wise parallelization**: 64 threads per block, strided access pattern
- **Tile configuration**: BQ=16, BK=16, D=32 (4.8KB shared memory per block)
- **Performance** (B=32, H=8, L=1024, d=32): **21.7ms, 85x speedup** over CPU

#### Attempted Optimizations
- **Register-tiled matrix multiplication**: Attempted 2D tiling (BQ=32, BK=32) with per-thread sub-tiles for better reuse
  - Achieved 29% speedup on isolated matmuls
  - Overall slower (30ms vs 21.7ms) due to increased shared memory overhead and reduced occupancy
  - Reverted to BQ=16, BK=16 optimal for this problem scale
- **Warp-level reductions**: Tested warp shuffle operations for row max/sum
  - Slower than simple loops at current tile sizes (BK=16 = 0.5 elements per thread)
  - Overhead of shuffle operations dominates at small scales
  - Would benefit from BK >= 64 to amortize coordination costs

#### Tensor Core Optimization (flash_attention_v1_opt1.h) ✓
After exploring various approaches, achieved breakthrough performance using NVIDIA Tensor Cores:

**Key Insights:**
- Reduced to 32 threads per block (1 warp) to enable clean WMMA usage
- WMMA operations are warp-level - all 32 threads participate
- Multiple warps per block complicated coordination without larger tiles

**Implementation:**
- **First WMMA**: Q @ K^T matrix multiplication
  - Uses 16×16×16 Tensor Core tiles (Ampere architecture)
  - Handles transpose via `col_major` fragment layout
  - Iterates k dimension in 16-element chunks (k=32 = 2×16)
  - Result: **6.95ms, 258x speedup** (3.1× faster than baseline)

- **Second WMMA**: S @ V matrix multiplication with row scaling
  - Two-phase approach: pre-scale A, then WMMA with accumulator initialization
  - Processes n dimension in two 16-column tiles (n=32)
  - Fragment-based scaling applied element-wise across warp
  - Result: **3.39ms, 530x speedup** (6.4× faster than baseline!)

**Final Performance** (B=32, H=8, L=1024, d=32):
- **GPU time**: 3.39ms average (50 runs)
- **Speedup**: 530× vs CPU, 6.4× vs baseline
- **Accuracy**: Max absolute error 3.66e-4, max relative error 13.3%
- **Grid**: 16,384 blocks (256 heads × 64 query tiles)

### Files
- **`flash_attention_v1.h`**: Baseline kernel (21.7ms, 85x speedup)
- **`flash_attention_v1_opt1.h`**: Tensor Core optimized kernel (3.39ms, 530x speedup)
- **`load_shared_memory.h`**: Vectorized parallel memory loading API
- **`standard.h`**: OpenMP-parallelized CPU reference implementation
- **`driver.cu`**: Test harness with conditional compilation for both versions
- **`Makefile`**: Separate build targets for v1 and opt1 variants

### Building and Running

**Baseline version:**
```bash
cd CUDA
make flash_attention_v1
./flash_attention_v1
```

**Tensor Core optimized version:**
```bash
cd CUDA
make flash_attention_v1_opt1
./flash_attention_v1_opt1
```

### Performance Comparison
On B=32, H=8, L=1024, d=32 with Ampere GPU (sm_80):
- **CPU (OpenMP)**: 1797ms
- **GPU Baseline**: 21.7ms (85× speedup)
- **GPU Tensor Cores**: 3.39ms (530× speedup, 6.4× vs baseline)

### Precision Options

The CUDA implementation supports both FP16 (half precision) and FP64 (double precision):

**FP16 (default - optimized for speed):**
- Max absolute error: ~1.7e-4
- Max relative error: ~6.4% (for values > 1e-3)
- Speedup: 26x over CPU
- GPU time: ~1.2ms

**FP64 (for correctness verification):**
```bash
cd CUDA
nvcc -O3 -std=c++17 -arch=sm_80 -Xcompiler -fopenmp -DUSE_FP64=1 driver.cu -o flash_attention -Xcompiler -fopenmp
./flash_attention
```
- Max absolute error: ~1.3e-7 (1000x better than FP16)
- Max relative error: ~2.3% (across all values)
- Speedup: 10.7x over CPU
- GPU time: ~2.9ms

FP64 is recommended for validating correctness, while FP16 provides the best performance for production use.

### Building and Running
```bash
cd CUDA
make
./flash_attention
```

### Optimization Techniques Applied
1. **Batched multi-head attention**: 2D grid launching for massive parallelism (16,384 blocks)
2. **Parallelized tile loading**: All threads cooperatively load Q/K/V tiles using vectorized loads
3. **Tensor Core acceleration**: WMMA API (16×16×16 fragments) for both matrix multiplications
4. **Fused operations**: Same fusion strategy as opt2 (4 major fused kernels)
5. **Single warp design**: 32 threads per block for clean WMMA coordination
6. **Transpose handling**: `col_major` fragment layout for efficient A @ B^T computation
7. **Two-phase scaling**: Pre-scale then WMMA for fused row scaling + matmul

### Lessons Learned
- Simple element-wise parallelization optimal for small tiles (BQ=16, BK=16)
- Warp-level primitives need sufficient work (BK >= 64) to amortize overhead
- Tensor Cores provide dramatic speedups (6.4×) when properly configured
- Single warp per block simplifies WMMA programming model for small tiles
- Register tiling overhead dominates gains without sufficient reuse opportunities
- Batching critical for GPU utilization (256 heads vs 1)

### Future Optimizations
- Larger tile sizes (BQ=32, BK=32) with multi-warp WMMA coordination
- Flash Attention V2 (improved parallelism and work partitioning)
- Dynamic sequence lengths and causal masking
- Further shared memory optimizations with padding for bank conflict avoidance

## Usage

Each Python file can be run standalone:
```bash
python3 pytorch_imp.py
python3 numpy_basic.py
python3 numpy_gpu_like.py
python3 numpy_gpu_like_1D.py
python3 numpy_gpu_like_opt1.py
python3 numpy_gpu_like_opt2.py
```

For CUDA:
```bash
cd CUDA && make flash_attention_v1 && ./flash_attention_v1      # Baseline
cd CUDA && make flash_attention_v1_opt1 && ./flash_attention_v1_opt1  # Tensor Cores
```

All implementations process attention on sequences with L=2048, L=1024, or custom lengths with head dimension d=32 by default. Batched multi-head support with B=32 batches and H=8 heads (256 total attention heads). All optimized versions maintain numerical accuracy within their respective precision limits (FP64: ~3e-16, FP16: ~1e-4).
