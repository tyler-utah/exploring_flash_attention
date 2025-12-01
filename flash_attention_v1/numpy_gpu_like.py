import numpy as np

def mat_mul_scaled(A, B, C, b, m, n, k):
    """
    Scaled matrix multiplication with transpose.
    Computes: C = A @ B.T * b
    More C-like: explicit dimensions passed in.
    
    A: [m, k] input matrix
    B: [n, k] input matrix (will be transposed)
    C: [m, n] output matrix (modified in place)
    b: scalar scale factor
    m: number of rows in A (and C)
    n: number of rows in B (and cols in C)
    k: number of cols in A and B
    """
    C[:m, :n] = (A[:m, :k] @ B[:n, :k].T) * b

def row_max(S, out, m, n):
    """
    Compute row-wise maximum.
    More C-like: explicit dimensions and loops.
    
    S:   [m, n] input matrix
    out: [m] output vector (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        max_val = S[i, 0]
        for j in range(1, n):
            if S[i, j] > max_val:
                max_val = S[i, j]
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
    
    S:   [m, n] input matrix
    v:   [m] vector to subtract from each row
    out: [m, n] output matrix (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            out[i, j] = S[i, j] - v[i]

def mat_exp(S, out, m, n):
    """
    Element-wise exponential of matrix.
    More C-like: explicit loops and indexing.
    
    S:   [m, n] input matrix
    out: [m, n] output matrix (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            out[i, j] = np.exp(S[i, j])

def row_sum(S, out, m, n):
    """
    Compute row-wise sum.
    More C-like: explicit dimensions and loops.
    
    S:   [m, n] input matrix
    out: [m] output vector (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        sum_val = 0.0
        for j in range(n):
            sum_val += S[i, j]
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
    Modifies A in place.
    
    A: [m, n] matrix (modified in place)
    v: [m] vector of scale factors
    m: number of rows
    n: number of cols
    """
    for i in range(m):
        for j in range(n):
            A[i, j] = A[i, j] * v[i]

def mat_mul(A, B, C, m, n, k):
    """
    Matrix multiplication: C = A @ B
    More C-like: explicit dimensions passed in.
    
    A: [m, k] input matrix
    B: [k, n] input matrix
    C: [m, n] output matrix (modified in place)
    m: number of rows in A (and C)
    n: number of cols in B (and C)
    k: number of cols in A and rows in B
    """
    C[:] = A @ B

def mat_add(A, B, m, n):
    """
    Element-wise matrix addition: A[i,j] += B[i,j]
    More C-like: explicit loops and indexing.
    Modifies A in place.
    
    A: [m, n] matrix (modified in place)
    B: [m, n] matrix to add
    m: number of rows
    n: number of cols
    """
    for i in range(m):
        for j in range(n):
            A[i, j] = A[i, j] + B[i, j]

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
    
    A:   [m, n] input matrix
    v:   [m] vector of divisors
    out: [m, n] output matrix (modified in place)
    m:   number of rows
    n:   number of cols
    """
    for i in range(m):
        for j in range(n):
            out[i, j] = A[i, j] / v[i]

def load_tile(src, tile, start, end):
    """
    Load a tile from source array into tile buffer.
    More GPU-like: explicit copy operation.
    
    src:   [L, d] source array
    tile:  [tile_len, d] destination buffer
    start: starting row index
    end:   ending row index (exclusive)
    """
    tile_len = end - start
    tile[:tile_len, :] = src[start:end, :]

def process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, bq, bk, d):
    """
    One tile step of FlashAttention-style streaming softmax.
    Updates m, l, O_acc in place.

    Q_tile: [bq, d]
    K_tile: [bk, d]
    V_tile: [bk, d]
    m:      [bq]      (running max per query row) - modified in place
    l:      [bq]      (running denom per query row) - modified in place
    O_acc:  [bq, d]   (running numerator per query row) - modified in place
    bq:     number of query rows in this tile
    bk:     number of key rows in this tile
    d:      head dimension
    """
    inv_sqrt_d = 1.0 / np.sqrt(d)

    # 1) Allocate scores matrix and compute: S = Q_tile @ K_tile^T / sqrt(d)
    #    S: [bq, bk]
    S = np.empty((bq, bk), dtype=Q_tile.dtype)
    mat_mul_scaled(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d)

    # 2) Block row-max over keys
    #    m_block[i] = max_j S[i, j]
    m_block = np.empty(bq, dtype=Q_tile.dtype)
    row_max(S, m_block, bq, bk)

    # 3) New running max per row
    #    m_new[i] = max(m[i], m_block[i])
    m_new = np.empty(bq, dtype=Q_tile.dtype)
    vec_max(m, m_block, m_new, bq)

    # 4) Rescale factor from old max -> new max
    #    alpha[i] = exp(m[i] - m_new[i])
    alpha = np.empty(bq, dtype=Q_tile.dtype)
    vec_exp_diff(m, m_new, alpha, bq)

    # 5) Shift scores by new max (for stability)
    #    S_shifted[i, j] = S[i, j] - m_new[i]
    S_shifted = np.empty((bq, bk), dtype=Q_tile.dtype)
    mat_sub_vec(S, m_new, S_shifted, bq, bk)

    # 6) Exponentiate shifted scores
    #    exp_S[i, j] = exp(S_shifted[i, j])
    exp_S = np.empty((bq, bk), dtype=Q_tile.dtype)
    mat_exp(S_shifted, exp_S, bq, bk)

    # 7) This tile's contribution to the denominator
    #    l_tile[i] = sum_j exp_S[i, j]
    l_tile = np.empty(bq, dtype=Q_tile.dtype)
    row_sum(exp_S, l_tile, bq, bk)

    # 8) New denom:
    #    l_new[i] = l[i] * alpha[i] + l_tile[i]
    l_new = np.empty(bq, dtype=Q_tile.dtype)
    vec_mul_add(l, alpha, l_tile, l_new, bq)

    # 9) Rescale old numerator to the new max:
    #    O_acc[i, j] *= alpha[i]
    mat_scale_rows(O_acc, alpha, bq, d)

    # 10) Add this tile's contribution to numerator:
    #     contrib = exp_S @ V_tile
    #     exp_S: [bq, bk]
    #     V_tile: [bk, d]
    contrib = np.empty((bq, d), dtype=Q_tile.dtype)
    mat_mul(exp_S, V_tile, contrib, bq, d, bk)
    
    # O_acc[i, j] += contrib[i, j]
    mat_add(O_acc, contrib, bq, d)

    # Update m and l in place
    vec_copy(m_new, m, bq)
    vec_copy(l_new, l, bq)


def flash_attention_tiled(Q, K, V, O, L, d, Bq=8, Bk=8):
    """
    FlashAttention-style attention using tiles and streaming softmax.
    More C/GPU-like: explicit indexing, output buffer passed in.

    Q, K, V: [L, d] NumPy arrays
    O:       [L, d] output buffer (modified in place)
    L:       sequence length
    d:       head dimension
    Bq: query tile size
    Bk: key/value tile size
    """

    # Loop over query tiles
    for q_start in range(0, L, Bq):
        q_end = min(q_start + Bq, L)
        q_len = q_end - q_start

        # Allocate Q tile buffer and load
        Q_tile = np.empty((Bq, d), dtype=Q.dtype)
        load_tile(Q, Q_tile, q_start, q_end)

        # Per-row streaming state for this Q tile
        m = np.full(Bq, -np.inf, dtype=Q.dtype)   # [Bq]
        l = np.zeros(Bq, dtype=Q.dtype)           # [Bq]
        O_acc = np.zeros((Bq, d), dtype=Q.dtype)  # [Bq, d]

        # Loop over K/V tiles
        for k_start in range(0, L, Bk):
            k_end = min(k_start + Bk, L)
            k_len = k_end - k_start
            
            # Allocate K, V tile buffers and load
            K_tile = np.empty((Bk, d), dtype=K.dtype)
            V_tile = np.empty((Bk, d), dtype=V.dtype)
            load_tile(K, K_tile, k_start, k_end)
            load_tile(V, V_tile, k_start, k_end)

            process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, q_len, k_len, d)

        # Finalize this Q tile: divide numerator by denom
        # O[i,j] = O_acc[i,j] / l[i]
        O_tile = np.empty((q_len, d), dtype=O.dtype)
        mat_div_vec(O_acc, l, O_tile, q_len, d)
        
        # Copy result back to output
        for i in range(q_len):
            for j in range(d):
                O[q_start + i, j] = O_tile[i, j]


# Optional: naive reference implementation to compare
def naive_attention(Q, K, V):
    """
    Naive attention: softmax(Q K^T / sqrt(d)) V
    Q, K, V: [L, d]
    Returns: [L, d]
    """
    L, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = (Q @ K.T) * scale           # [L, L]
    probs = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs @ V                     # [L, d]


if __name__ == "__main__":
    L = 2048
    d = 32
    rng = np.random.default_rng(0)

    # Use float32 for stability; switch to float16 if you like.
    Q = rng.standard_normal((L, d)).astype(np.float64)
    K = rng.standard_normal((L, d)).astype(np.float64)
    V = rng.standard_normal((L, d)).astype(np.float64)
    O_tiled = np.zeros((L, d), dtype=np.float64)

    # Allocate output buffer outside (like GPU kernel)
    flash_attention_tiled(Q, K, V, O_tiled, L, d, Bq=8, Bk=8)
    
    O_naive = naive_attention(Q, K, V)

    print("O_tiled shape:", O_tiled.shape)
    print("First 3 rows (tiled):")
    print(O_tiled[:3, :5])

    print("\nFirst 3 rows (naive):")
    print(O_naive[:3, :5])

    # Check closeness
    diff = np.abs(O_tiled - O_naive).max()
    print("\nMax absolute difference:", diff)
    
    # Relative difference (max can be misleading for near-zero values)
    rel_diff = (np.abs(O_tiled - O_naive) / (np.abs(O_naive) + 1e-8)).max()
    print("Max relative difference:", rel_diff)
    
    # Mean relative error (excluding very small values)
    mask = np.abs(O_naive) > 1e-3
    mean_rel_err = (np.abs(O_tiled[mask] - O_naive[mask]) / np.abs(O_naive[mask])).mean()
    print("Mean relative error (|naive| > 1e-3):", mean_rel_err)