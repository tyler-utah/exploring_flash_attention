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

## Future Work

- Flash Attention V2 (improved parallelism and reduced non-matmul operations)
- Flash Attention V3 (enhanced for specific hardware architectures)
- Triton implementations for comparison
- Multi-head attention batching
- Causal masking variants

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
cd flash_attention_v1
python3 numpy_gpu_like_opt2.py  # Most optimized CPU version
python3 numpy_basic.py           # Basic reference version
```

### CUDA Implementation

```bash
cd flash_attention_v1/CUDA
make flash_attention_v1       # Baseline (21.7ms)
./flash_attention_v1

make flash_attention_v1_opt1  # Tensor Cores (3.39ms)
./flash_attention_v1_opt1
```

## References

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135) (Dao et al., 2022)
- [Flash Attention V2](https://arxiv.org/abs/2307.08691) (Dao, 2023)
