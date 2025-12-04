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

def row_max(S, out, m, n):
    """
    Compute row-wise maximum.
    More C-like: explicit dimensions and loops.
    All arrays are 1D buffers.
    
    S:   [m*n] input matrix (flattened)
    out: [m] output vector (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        max_val = S[idx2d(i, 0, n)]
        for j in range(1, n):
            if S[idx2d(i, j, n)] > max_val:
                max_val = S[idx2d(i, j, n)]
        out[i] = max_val

def vec_max(a, b, out, n):
    """
    Element-wise maximum of two vectors.
    More C-like: explicit loop and indexing.
    
    a:   [n] input vector
    b:   [n] input vector
    out: [n] output vector (modified in place)
    n:   vector length
    """
    for i in range(n):
        if a[i] > b[i]:
            out[i] = a[i]
        else:
            out[i] = b[i]

def vec_exp_diff(a, b, out, n):
    """
    Element-wise exp(a - b).
    More C-like: explicit loop and indexing.
    
    a:   [n] input vector
    b:   [n] input vector
    out: [n] output vector (modified in place)
    n:   vector length
    """
    for i in range(n):
        out[i] = np.exp(a[i] - b[i])

def mat_sub_vec(S, v, out, m, n):
    """
    Subtract vector from each row of matrix: out[i,j] = S[i,j] - v[i]
    More C-like: explicit loops and indexing.
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
            out[idx] = S[idx] - v[i]

def mat_exp(S, out, m, n):
    """
    Element-wise exponential of matrix.
    More C-like: explicit loops and indexing.
    All arrays are 1D buffers.
    
    S:   [m*n] input matrix (flattened)
    out: [m*n] output matrix (flattened, modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            out[idx] = np.exp(S[idx])

def row_sum(S, out, m, n):
    """
    Compute row-wise sum.
    More C-like: explicit dimensions and loops.
    All arrays are 1D buffers.
    
    S:   [m*n] input matrix (flattened)
    out: [m] output vector (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        sum_val = 0.0
        for j in range(n):
            sum_val += S[idx2d(i, j, n)]
        out[i] = sum_val

def vec_mul_add(a, b, c, out, n):
    """
    Element-wise multiply and add: out[i] = a[i] * b[i] + c[i]
    More C-like: explicit loop and indexing.
    
    a:   [n] input vector
    b:   [n] input vector
    c:   [n] input vector
    out: [n] output vector (modified in place)
    n:   vector length
    """
    for i in range(n):
        out[i] = a[i] * b[i] + c[i]

def mat_scale_rows(A, v, m, n):
    """
    Scale each row of matrix by corresponding vector element: A[i,j] *= v[i]
    More C-like: explicit loops and indexing.
    Modifies A in place. All arrays are 1D buffers.
    
    A: [m*n] matrix (flattened, modified in place)
    v: [m] vector of scale factors
    m: number of rows
    n: number of cols
    """
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            A[idx] = A[idx] * v[i]

def mat_mul(A, B, C, m, n, k):
    """
    Matrix multiplication: C = A @ B
    More C-like: explicit dimensions passed in.
    All arrays are 1D buffers in row-major layout.
    
    A: [m*k] input matrix (flattened)
    B: [k*n] input matrix (flattened)
    C: [m*n] output matrix (flattened, modified in place)
    m: number of rows in A (and C)
    n: number of cols in B (and C)
    k: number of cols in A and rows in B
    """
    # Reshape for matmul, then flatten back
    A_2d = A[:m*k].reshape(m, k)
    B_2d = B[:k*n].reshape(k, n)
    C_result = A_2d @ B_2d
    C[:m*n] = C_result.flatten()

def mat_add(A, B, m, n):
    """
    Element-wise matrix addition: A[i,j] += B[i,j]
    More C-like: explicit loops and indexing.
    Modifies A in place. All arrays are 1D buffers.
    
    A: [m*n] matrix (flattened, modified in place)
    B: [m*n] matrix (flattened) to add
    m: number of rows
    n: number of cols
    """
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            A[idx] = A[idx] + B[idx]

def vec_copy(src, dst, n):
    """
    Copy vector: dst[i] = src[i]
    More C-like: explicit loop and indexing.
    
    src: [n] source vector
    dst: [n] destination vector (modified in place)
    n:   vector length
    """
    for i in range(n):
        dst[i] = src[i]

def mat_div_vec(A, v, out, m, n):
    """
    Divide each row of matrix by corresponding vector element: out[i,j] = A[i,j] / v[i]
    More C-like: explicit loops and indexing.
    All arrays are 1D buffers.
    
    A:   [m*n] input matrix (flattened)
    v:   [m] vector of divisors
    out: [m*n] output matrix (flattened, modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            out[idx] = A[idx] / v[i]

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
    
    OPTIMIZED: Reuses buffers to minimize shared memory usage.
    - S is reused for shifted and exponentiated scores (3 uses)
    - m_new is reused for m_block computation
    - l_new is reused for l_tile computation

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

    # Allocate shared memory buffers (only 4 arrays needed)
    S = np.empty(bq * bk, dtype=Q_tile.dtype)      # [bq*bk] - reused 3x
    m_new = np.empty(bq, dtype=Q_tile.dtype)       # [bq] - reused 2x
    alpha = np.empty(bq, dtype=Q_tile.dtype)       # [bq]
    l_new = np.empty(bq, dtype=Q_tile.dtype)       # [bq] - reused 2x
    contrib = np.empty(bq * d, dtype=Q_tile.dtype) # [bq*d]

    # 1) Compute scores: S = Q_tile @ K_tile^T / sqrt(d)
    mat_mul_scaled(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d)

    # 2) Block row-max over keys (reuse m_new as temporary for m_block)
    #    m_new[i] = max_j S[i, j]
    row_max(S, m_new, bq, bk)

    # 3) Update running max: m_new[i] = max(m[i], m_new[i])
    #    (m_new now contains the block max, update it to global max)
    vec_max(m, m_new, m_new, bq)

    # 4) Rescale factor from old max -> new max
    #    alpha[i] = exp(m[i] - m_new[i])
    vec_exp_diff(m, m_new, alpha, bq)

    # 5) Shift scores by new max IN PLACE (S becomes S_shifted)
    #    S[i, j] = S[i, j] - m_new[i]
    mat_sub_vec(S, m_new, S, bq, bk)

    # 6) Exponentiate shifted scores IN PLACE (S becomes exp_S)
    #    S[i, j] = exp(S[i, j])
    mat_exp(S, S, bq, bk)

    # 7) Row sum of exp scores (reuse l_new as temporary for l_tile)
    #    l_new[i] = sum_j S[i, j]
    row_sum(S, l_new, bq, bk)

    # 8) Update running denom IN PLACE: l_new[i] = l[i] * alpha[i] + l_new[i]
    #    (l_new contains l_tile, update it to new global l)
    vec_mul_add(l, alpha, l_new, l_new, bq)

    # 9) Rescale old numerator to the new max:
    #    O_acc[i, j] *= alpha[i]
    mat_scale_rows(O_acc, alpha, bq, d)

    # 10) Add this tile's contribution to numerator:
    #     contrib = S @ V_tile  (S is now exp_S)
    mat_mul(S, V_tile, contrib, bq, d, bk)
    
    # O_acc[i, j] += contrib[i, j]
    mat_add(O_acc, contrib, bq, d)

    # Update m and l in place
    vec_copy(m_new, m, bq)
    vec_copy(l_new, l, bq)


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

        # Finalize this Q tile: divide numerator by denom
        # O[i,j] = O_acc[i,j] / l[i]
        O_tile = np.empty(q_len * d, dtype=O.dtype)
        mat_div_vec(O_acc, l, O_tile, q_len, d)
        
        # Copy result back to output (explicit 1D indexing)
        for i in range(q_len):
            for j in range(d):
                O[idx2d(q_start + i, j, d)] = O_tile[idx2d(i, j, d)]


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
