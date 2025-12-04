import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.reference import naive_attention, check_accuracy, print_comparison

def idx2d(i, j, cols):
    """
    Convert 2D index (i, j) to 1D index for row-major layout.
    Like C/GPU: index = i * cols + j
    """
    return i * cols + j

def mat_mul_scaled(A, B, C, b, m, n, k):
    """
    Scaled matrix multiplication with transpose.
    Computes: C = A @ B.T * b
    More C-like: explicit dimensions passed in.
    All arrays are 1D buffers in row-major layout.
    
    A: [m*k] input matrix (flattened)
    B: [n*k] input matrix (flattened, will be transposed)
    C: [m*n] output matrix (flattened, modified in place)
    b: scalar scale factor
    m: number of rows in A (and C)
    n: number of rows in B (and cols in C)
    k: number of cols in A and B
    """
    # Reshape for matmul, then flatten back
    A_2d = A[:m*k].reshape(m, k)
    B_2d = B[:n*k].reshape(n, k)
    C_result = (A_2d @ B_2d.T) * b
    C[:m*n] = C_result.flatten()

def mat_scale_rows_mul_add(A, v, B, C, m, n, k):
    """
    FUSED: Scale rows of A, then compute and accumulate: A *= v[i]; A += B @ C
    Combines mat_scale_rows + mat_mul_add in one operation.
    More efficient for GPU: fewer passes over A, better cache usage.
    
    A: [m*n] matrix (flattened, modified in place) - both scaled and accumulated
    v: [m] vector of scale factors
    B: [m*k] matrix for matmul (flattened)
    C: [k*n] matrix for matmul (flattened)
    m: number of rows
    n: number of cols in A and C
    k: number of cols in B (rows in C)
    """
    # First scale A by v
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            A[idx] = A[idx] * v[i]
    
    # Then compute matmul and accumulate
    B_2d = B[:m*k].reshape(m, k)
    C_2d = C[:k*n].reshape(k, n)
    result = B_2d @ C_2d
    
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            A[idx] += result[i, j]

def mat_sub_vec_exp(S, v, out, m, n):
    """
    FUSED: Subtract vector from each row and exponentiate: out[i,j] = exp(S[i,j] - v[i])
    More efficient than separate mat_sub_vec + mat_exp.
    All arrays are 1D buffers.
    
    S:   [m*n] input matrix (flattened)
    v:   [m] vector to subtract from each row
    out: [m*n] output matrix (flattened, modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            out[idx] = np.exp(S[idx] - v[i])

def mat_div_vec_store(A, v, out, out_row_offset, m, n):
    """
    FUSED: Divide each row by vector element and store at offset location.
    Combines mat_div_vec with direct storage to output array.
    Eliminates intermediate buffer.
    
    A:   [m*n] input matrix (flattened)
    v:   [m] vector of divisors
    out: [L*n] output matrix (flattened, modified in place)
    out_row_offset: starting row index in output array
    m:   number of rows to process
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            out[idx2d(out_row_offset + i, j, n)] = A[idx2d(i, j, n)] / v[i]

def row_sum_mul_add_inplace(S, l, alpha, m, n):
    """
    FUSED: Row sum combined with multiply-add, updating l IN PLACE.
    Computes: l[i] = l[i] * alpha[i] + sum_j S[i,j]
    More efficient than separate row_sum + vec_mul_add.
    All arrays are 1D buffers.
    
    S:     [m*n] input matrix (flattened)
    l:     [m] running denominator (modified in place)
    alpha: [m] rescale factors
    m:     number of rows
    n:     number of cols
    """
    for i in range(m):
        sum_val = 0.0
        for j in range(n):
            sum_val += S[idx2d(i, j, n)]
        l[i] = l[i] * alpha[i] + sum_val

def load_tile(src, tile, start, end, d):
    """
    Load a tile from source array into tile buffer.
    More GPU-like: explicit copy operation.
    All arrays are 1D buffers.
    
    src:   [L*d] source array (flattened)
    tile:  [tile_len*d] destination buffer (flattened)
    start: starting row index
    end:   ending row index (exclusive)
    d:     number of columns
    """
    tile_len = end - start
    for i in range(tile_len):
        for j in range(d):
            tile[idx2d(i, j, d)] = src[idx2d(start + i, j, d)]

def process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, bq, bk, d):
    """
    One tile step of FlashAttention-style streaming softmax.
    Updates m, l, O_acc in place.
    All matrix arrays are 1D buffers in row-major layout.
    
    OPTIMIZED v3: 
    - Reuses buffers to minimize shared memory usage (from opt1)
    - Fuses operations to reduce memory traffic:
      * row_max + vec_max → row_max_update (in-place)
      * mat_sub_vec + mat_exp → mat_sub_vec_exp
      * row_sum + vec_mul_add → row_sum_mul_add_inplace (in-place)
      * mat_scale_rows + mat_mul_add → mat_scale_rows_mul_add
    - Eliminates temporary buffers by in-place updates (from opt2)
    - Reduces from 4 to 2 buffers (S, alpha) - eliminates m_new, l_new, vec_copy

    Q_tile: [bq*d]    (flattened)
    K_tile: [bk*d]    (flattened)
    V_tile: [bk*d]    (flattened)
    m:      [bq]      (running max per query row) - modified in place
    l:      [bq]      (running denom per query row) - modified in place
    O_acc:  [bq*d]    (flattened, running numerator per query row) - modified in place
    bq:     number of query rows in this tile
    bk:     number of key rows in this tile
    d:      head dimension
    """
    inv_sqrt_d = 1.0 / np.sqrt(d)

    # Allocate shared memory buffers (only 2 arrays needed!)
    S = np.empty(bq * bk, dtype=Q_tile.dtype)      # [bq*bk] - reused 3x
    alpha = np.empty(bq, dtype=Q_tile.dtype)       # [bq] - rescale factors
    # Note: m and l are updated in-place, no temporary buffers needed

    # 1) Compute scores: S = Q_tile @ K_tile^T / sqrt(d)
    mat_mul_scaled(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d)

    # 2) Compute rescale factor BEFORE updating m
    #    We need old m values to compute alpha = exp(m_old - m_new)
    #    So we compute alpha using a temporary calculation:
    for i in range(bq):
        # Find new max for this row (initialized with running max)
        new_max = m[i]
        for j in range(bk):
            if S[idx2d(i, j, bk)] > new_max:
                new_max = S[idx2d(i, j, bk)]
        # Compute alpha before overwriting m
        alpha[i] = np.exp(m[i] - new_max)
        # Update m in place
        m[i] = new_max

    # 3) FUSED: Shift scores and exponentiate IN PLACE
    #    S[i, j] = exp(S[i, j] - m[i])
    mat_sub_vec_exp(S, m, S, bq, bk)

    # 4) FUSED: Row sum and update running denom IN PLACE
    #    l[i] = l[i] * alpha[i] + sum_j S[i, j]
    row_sum_mul_add_inplace(S, l, alpha, bq, bk)

    # 5) FUSED: Rescale old numerator and accumulate contribution
    #    O_acc[i,j] *= alpha[i]; O_acc += S @ V_tile
    mat_scale_rows_mul_add(O_acc, alpha, S, V_tile, bq, d, bk)


def flash_attention_tiled(Q, K, V, O, L, d, Bq=8, Bk=8):
    """
    FlashAttention-style attention using tiles and streaming softmax.
    More C/GPU-like: explicit indexing, output buffer passed in.
    All arrays are 1D buffers in row-major layout.

    Q, K, V: [L*d] NumPy arrays (flattened)
    O:       [L*d] output buffer (flattened, modified in place)
    L:       sequence length
    d:       head dimension
    Bq: query tile size
    Bk: key/value tile size
    """

    # Loop over query tiles
    for q_start in range(0, L, Bq):
        q_end = min(q_start + Bq, L)
        q_len = q_end - q_start

        # Allocate Q tile buffer and load (1D)
        Q_tile = np.empty(Bq * d, dtype=Q.dtype)
        load_tile(Q, Q_tile, q_start, q_end, d)

        # Per-row streaming state for this Q tile
        m = np.full(Bq, -np.inf, dtype=Q.dtype)   # [Bq]
        l = np.zeros(Bq, dtype=Q.dtype)           # [Bq]
        O_acc = np.zeros(Bq * d, dtype=Q.dtype)   # [Bq*d] (flattened)

        # Loop over K/V tiles
        for k_start in range(0, L, Bk):
            k_end = min(k_start + Bk, L)
            k_len = k_end - k_start
            
            # Allocate K, V tile buffers and load (1D)
            K_tile = np.empty(Bk * d, dtype=K.dtype)
            V_tile = np.empty(Bk * d, dtype=V.dtype)
            load_tile(K, K_tile, k_start, k_end, d)
            load_tile(V, V_tile, k_start, k_end, d)

            process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, q_len, k_len, d)

        # FUSED: Finalize and store directly to output
        # O[q_start+i, j] = O_acc[i, j] / l[i]
        mat_div_vec_store(O_acc, l, O, q_start, q_len, d)


if __name__ == "__main__":
    L = 1024
    d = 32
    rng = np.random.default_rng(0)

    # Generate 2D arrays first for naive implementation
    Q_2d = rng.standard_normal((L, d)).astype(np.float64)
    K_2d = rng.standard_normal((L, d)).astype(np.float64)
    V_2d = rng.standard_normal((L, d)).astype(np.float64)
    
    # Flatten to 1D for tiled implementation (like GPU)
    Q = Q_2d.flatten()
    K = K_2d.flatten()
    V = V_2d.flatten()
    O_tiled = np.zeros(L * d, dtype=np.float64)

    # Allocate output buffer outside (like GPU kernel)
    flash_attention_tiled(Q, K, V, O_tiled, L, d, Bq=8, Bk=8)
    
    # Reshape back to 2D for comparison
    O_tiled_2d = O_tiled.reshape(L, d)
    O_reference = naive_attention(Q_2d, K_2d, V_2d)

    print_comparison(O_tiled_2d, O_reference)
    check_accuracy(O_tiled_2d, O_reference)
