# Flash Attention Experiments

This repository explores different implementations of Flash Attention algorithms, progressively optimizing from simple reference implementations to high-performance GPU kernels.

## Overview

Flash Attention is a memory-efficient attention algorithm that avoids materializing the full attention matrix by computing attention incrementally in tiles. This repository implements various versions, starting with Flash Attention V1.

## Structure

### `flash_attention_v1/`

Implementation of Flash Attention V1 with multiple optimization levels:
- **Python/NumPy implementations**: Progressive optimizations from basic to GPU-like memory layouts
- **CUDA implementations**: Two versions - baseline and Tensor Core optimized

Key progression:
1. Basic NumPy implementation
2. GPU-like memory layout with explicit loops
3. 1D memory layout matching GPU kernels
4. Memory-optimized with buffer reuse
5. Fully fused operations with in-place updates
6. CUDA baseline with batched multi-head attention (21.7ms, 85× speedup)
7. **CUDA with Tensor Cores (3.39ms, 530× speedup) - 6.4× faster than baseline!**

**Performance** (B=32, H=8, L=1024, d=32):
- Baseline GPU: 21.7ms (85× vs CPU)
- Tensor Core GPU: **3.39ms (530× vs CPU)**

See [`flash_attention_v1/README_v1.md`](flash_attention_v1/README_v1.md) for detailed documentation.

### `flash_attention_v1_tiled_d/`

**True memory-efficient d-tiling** implementation that enables arbitrarily large head dimensions:
- **Register-based output accumulation**: O_acc kept in thread registers, not shared memory
- **Minimal shared memory usage**: Only D_TILE sized chunks (88% reduction: 3.69 KB vs 8.22 KB)
- **Independent d-tile parameters**: Separate optimization for Q@K^T (D_TILE_QK) and S@V (D_TILE_V)
- **Optimized configuration**: BQ=16, BK=16, D_TILE=32, THREADS=256

Key innovations:
1. Loads Q/K/V in small d-tile chunks from global memory (not full D)
2. Each thread maintains output elements in registers (8 regs/thread)
3. Enables support for d >> shared memory capacity
4. Tensor Core ready with BQ=BK=16 tile sizes

**Performance** (B=32, H=8, L=1024, d=128):
- Optimized CUDA: **154ms (46× vs CPU)**
- Shared memory: **3.69 KB (88% reduction vs traditional)**
- Scales to d=512+ efficiently

See [`flash_attention_v1_tiled_d/README.md`](flash_attention_v1_tiled_d/README.md) for detailed documentation.

### `flash_attention_v2/`

**Two-kernel architecture** with parallelism over both query and key-value tiles:
- **Kernel 1 (Forward)**: Computes partial attention results for Q×KV tile combinations
- **Kernel 2 (Reduction)**: Combines partial results with proper softmax rescaling
- **Tunable parallelism**: KV_TILES_PER_BLOCK parameter controls blocks (default: 4)
- **8× more parallelism**: 256 blocks vs 32 blocks for V1 (at L=256)

Key innovations:
1. Split-KV parallelization enables parallel processing across sequence length
2. Workspace memory stores partial outputs (O, m, l) from each forward block
3. Reduction kernel mathematically combines partial softmax computations
4. Validated against official Dao-AILab/flash-attention CUDA implementation

**Performance characteristics**:
- Grid dimensions (L=256): Forward 32×8=256 blocks, Reduction 32 blocks
- Workspace overhead: ~10-20% additional memory
- Reduction overhead: ~10-20% of runtime
- **Net benefit**: Parallelism gains far outweigh overhead costs

**Implementation validation**:
- ✅ Structure matches official `flash_fwd_splitkv_kernel` + `flash_fwd_splitkv_combine_kernel`
- ✅ Workspace layout matches `oaccum_ptr`, `softmax_lseaccum_ptr`
- ✅ Tests passing with max absolute difference: 0.0011

See [`flash_attention_v2/README.md`](flash_attention_v2/README.md) for detailed documentation.

## Future Work

- Flash Attention V2 CUDA implementation (port Python simulation to GPU)
- Flash Attention V2 performance benchmarking vs V1
- Flash Attention V3 (enhanced for specific hardware architectures)

## Requirements

### Python Implementations
- Python 3.8+
- NumPy
- PyTorch (for reference implementation)

### CUDA Implementation
- CUDA 11.0+
- NVIDIA GPU with compute capability 8.0+ (Ampere or newer)
- nvcc compiler
- OpenMP

## Setup

### Create Virtual Environment

```bash
python3 -m venv venv
```

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Deactivate Virtual Environment

When you're done working:

```bash
deactivate
```

## Running the Code

### Python Implementations

After activating the virtual environment and installing dependencies:

```bash
# Flash Attention V1
cd flash_attention_v1
python3 numpy_gpu_like_opt2.py  # Most optimized CPU version
python3 numpy_basic.py           # Basic reference version

# Flash Attention V2 (two-kernel architecture)
cd flash_attention_v2
python3 numpy_gpu_like.py        # V2 with splitkv + reduction kernels
```

### CUDA Implementation

**Flash Attention V1 (d=32):**
```bash
cd flash_attention_v1/CUDA
make flash_attention_v1       # Baseline (21.7ms)
./flash_attention_v1

make flash_attention_v1_opt1  # Tensor Cores (3.39ms)
./flash_attention_v1_opt1
```

**Flash Attention V1 with D-Tiling (d=128):**
```bash
cd flash_attention_v1_tiled_d/CUDA
make                          # Optimized true d-tiling (154ms, 46× speedup)
./flash_attention_v1

# Try different configurations
make BQ=8 BK=8 D_TILE_QK=16 run
make BQ=32 BK=32 D_TILE_QK=64 run
```
./flash_attention_v1_opt1
```

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [Flash Attention V2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
