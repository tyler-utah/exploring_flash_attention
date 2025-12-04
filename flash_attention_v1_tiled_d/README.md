# Flash Attention V1 with D-Tiling

This directory contains an implementation of Flash Attention V1 with **d-dimension tiling**, which allows for more realistic head dimensions (`d`) and provides additional tuning parameters for optimization.

## Overview

Standard Flash Attention implementations typically require the entire head dimension `d` to fit in shared memory simultaneously. This approach extends Flash Attention V1 by tiling along the `d` dimension, enabling:

- **Support for larger head dimensions**: No longer constrained by shared memory limits for `d`
- **Independent tuning parameters**: Separate tile sizes for Q@K^T and S@V operations
- **Memory efficiency**: Only load `d_tile`-sized chunks into shared memory at a time

## Key Innovation: Two D-Tile Parameters

This implementation introduces **two independent d-tiling parameters**:

- **`D_TILE_QK`**: Tile size for the Q@K^T (attention score) computation
- **`D_TILE_V`**: Tile size for the S@V (weighted value) computation

This separation allows fine-grained control over memory access patterns and computation efficiency for each matrix operation.

## Files

### Python Implementations

#### `numpy_basic.py`
High-level Python implementation using global memory arrays to simulate tiled computation.

**Key features:**
- Clean, readable algorithm that closely mirrors the mathematical formulation
- `process_kv_tile_global()`: Processes one KV tile with streaming softmax
- Both Q@K^T and S@V operations tile along the `d` dimension
- Tuning parameters: `BQ=8`, `BK=8`, `D_TILE_QK=16`, `D_TILE_V=16`
- Uses global arrays with simulated shared memory access patterns
- Easy to understand and modify for experimentation

**Purpose:** Reference implementation for understanding the algorithm and validating correctness.

#### `numpy_gpu_like.py`
GPU-style implementation using 1D flattened arrays and explicit indexing.

**Key features:**
- Uses `idx2d()` for manual row-major indexing (mimics GPU memory layout)
- `mat_mul_scaled_d_tiled()`: Computes Q@K^T with d-tiling
- `mat_scale_rows_mul_add_d_tiled()`: New function that tiles V along the d dimension for S@V
- Simulates loading `[m, d_tile]` and `[n, d_tile]` blocks into shared memory
- 1D array operations match GPU kernel data access patterns
- Thread-like processing with explicit loop structures

**Purpose:** Bridge between high-level Python and CUDA implementation, demonstrating GPU memory access patterns.

### CUDA Implementation

#### `CUDA/flash_attention_v1.h`
Complete CUDA kernel implementation with d-tiling support.

**Key features:**
- **Device functions:**
  - `mat_mul_scaled_d_tiled()`: Q@K^T computation with d-tiling (lines 59-93)
  - `mat_scale_rows_mul_add_d_tiled()`: S@V computation with V tiling (lines 133-163)
  - `process_kv_tile()`: Streaming softmax for one KV tile
- **Memory management:**
  - Shared memory allocation for Q_tile, K_tile, V_tile, O_acc, S
  - Float buffers for m, l, alpha accumulators
  - Tiles V in chunks of `d_tile_v` for the S@V computation
- **Kernel configuration:**
  - 2D grid: (query tiles, batch × heads)
  - Configurable threads per block (default 64)
  - Compile-time constants: `BQ`, `BK`, `D_TILE_QK`, `D_TILE_V`, `D`
- **Precision support:**
  - Half-precision (`__half`) or double precision via `USE_FP64` flag
  - Runtime validation that compile-time `D` matches runtime dimension

**Performance:** ~117ms for typical workload (B=32, H=8, L=1024, d=128), 60× speedup over naive implementation.

**Purpose:** Production CUDA kernel demonstrating real GPU memory hierarchy usage.

#### `CUDA/driver.cu`
Test harness and validation driver.

**Key features:**
- Generates random test data for Q, K, V matrices
- Calls Python reference implementation via subprocess for validation
- Measures CUDA kernel execution time
- Compares GPU output against CPU reference with tolerance checking
- Configurable test dimensions (B, H, L, d)
- Passes both `d_tile_qk` and `d_tile_v` to kernel

**Purpose:** Testing, benchmarking, and correctness validation.

#### `CUDA/Makefile`
Build system with tunable parameters.

**Configuration parameters:**
- `BQ`: Query tile size (default: 8)
- `BK`: Key/Value tile size (default: 8)
- `D_TILE_QK`: Head dimension tile for Q@K^T (default: 16)
- `D_TILE_V`: Head dimension tile for S@V (default: 16)
- `D`: Head dimension (default: 128)
- `THREADS_PER_BLOCK`: Threads per block (default: 64)

**Targets:**
- `make`: Build executable
- `make run`: Build and run
- `make config`: Display current configuration
- `make clean`: Remove build artifacts

**Usage:**
```bash
make BQ=16 BK=16 D_TILE_QK=32 D_TILE_V=32 run
```

## Algorithm: D-Tiled Flash Attention

### Q@K^T Computation (with D_TILE_QK)
For computing attention scores `S = Q @ K^T`:

1. Iterate over the `d` dimension in chunks of `D_TILE_QK`
2. Load `Q[q_start:q_start+BQ, d_start:d_start+D_TILE_QK]` into shared memory
3. Load `K[k_start:k_start+BK, d_start:d_start+D_TILE_QK]` into shared memory
4. Compute partial product: `S_partial = Q_tile @ K_tile^T`
5. Accumulate into `S`
6. Repeat for next `d` chunk

### S@V Computation (with D_TILE_V)
For computing weighted outputs `O = S @ V`:

1. Iterate over the `d` dimension in chunks of `D_TILE_V`
2. Load `V[k_start:k_start+BK, d_start:d_start+D_TILE_V]` into shared memory
3. Compute `O_partial = S @ V_tile` for this `d` chunk
4. Write/accumulate `O_partial` into corresponding slice of output
5. Repeat for next `d` chunk

### Streaming Softmax
The implementation uses the standard Flash Attention streaming softmax:
- Maintains running statistics `m` (max) and `l` (sum of exponentials)
- Updates these incrementally as KV tiles are processed
- Rescales accumulated output `O` when statistics change
- Final output is correctly normalized attention

## Why D-Tiling Matters

### Memory Constraints
Without d-tiling, the entire head dimension must fit in shared memory:
- Shared memory requirement: `O(BQ × d + BK × d)`
- For `d=128`, this is manageable
- For `d=256` or larger, may exceed shared memory limits (48-96 KB per block)

### With D-Tiling
Shared memory requirement: `O(BQ × D_TILE + BK × D_TILE)`
- Can handle arbitrarily large `d` by choosing appropriate tile sizes
- Trade-off: More iterations, but enables larger problem sizes
- Realistic for production models (e.g., d=128 to 256 in modern transformers)

### Tuning Flexibility
Independent control over `D_TILE_QK` and `D_TILE_V` enables:
- Optimizing for different compute/memory characteristics of Q@K^T vs S@V
- Experimenting with Tensor Core sizes (16×16×16 WMMA operations)
- Balancing shared memory usage vs register pressure
- Adapting to different GPU architectures

## Building and Running

### Build with default parameters:
```bash
cd CUDA
make
./flash_attention_v1
```

### Build with custom parameters:
```bash
make BQ=16 BK=16 D_TILE_QK=32 D_TILE_V=32 D=256
./flash_attention_v1
```

### View configuration:
```bash
make config
```

### Clean build:
```bash
make clean
```

## Testing

The driver validates correctness by:
1. Generating random input matrices (Q, K, V)
2. Running Python reference implementation (via subprocess)
3. Executing CUDA kernel
4. Comparing outputs with tolerance checking:
   - Max absolute error < 1e-2
   - Max relative error < 0.5 (50%)
   - Mean relative error < 0.05 (5%)

Test configuration: B=32, H=8, L=1024, d=128

## Performance Characteristics

Typical performance (B=32, H=8, L=1024, d=128):
- CUDA kernel: ~117ms
- Python reference: ~7000ms
- Speedup: ~60×

Performance factors:
- Larger tile sizes (BQ, BK) generally improve compute efficiency
- Smaller d-tiles may reduce memory bandwidth but increase loop overhead
- Thread count affects parallelism and resource utilization
- Half-precision (`__half`) provides ~2× speedup over double precision

## Future Optimization Opportunities

1. **Tensor Cores (WMMA)**: With `BQ≥16`, `BK≥16`, `D_TILE≥16`, can leverage WMMA APIs for ~2-4× additional speedup
2. **Multi-warp strategies**: Current implementation uses 64 threads (2 warps); larger tiles could benefit from more warps
3. **Asynchronous memory copies**: Overlap data loading with computation
4. **Register tiling**: Further optimize inner loops with register blocking
5. **Warp-level primitives**: Use shuffle operations for faster reductions

## Comparison to Other Implementations

| Implementation | D-Tiling | Tile Params | Use Case |
|----------------|----------|-------------|----------|
| `flash_attention_v1/` | No | BQ, BK | Small d (32), baseline |
| `flash_attention_v1_tiled_d/` | **Yes** | BQ, BK, D_TILE_QK, D_TILE_V | **Larger d, more tuning flexibility** |

## Summary

This d-tiled Flash Attention implementation:
- ✅ Supports realistic head dimensions without shared memory constraints
- ✅ Provides independent tuning parameters for Q@K^T and S@V operations
- ✅ Maintains correctness through streaming softmax
- ✅ Achieves significant GPU speedup (~60×)
- ✅ Offers clear progression from Python to CUDA
- ✅ Enables future optimizations (Tensor Cores, multi-warp, etc.)

The approach demonstrates how algorithmic tiling can extend Flash Attention to handle larger problem sizes while providing fine-grained control over the performance/memory trade-offs.
