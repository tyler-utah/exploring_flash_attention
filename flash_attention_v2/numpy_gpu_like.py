import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.reference import naive_attention, check_accuracy, print_comparison

# Tuning parameters
BQ = 8           # Query tile size
BK = 8           # Key/Value tile size
D_TILE_QK = 16   # Head dimension tile size for Q@K^T
D_TILE_V = 16    # Head dimension tile size for S@V
KV_TILES_PER_BLOCK = 4  # Number of KV tiles each "block" processes (V2 tuning parameter)
                         # Small values (1-4) = more parallelism, more blocks
                         # Large values = less parallelism, closer to V1

def idx2d(i, j, cols):
    """
    Convert 2D index (i, j) to 1D index for row-major layout.
    Like C/GPU: index = i * cols + j
    """
    return i * cols + j

def mat_mul_scaled_d_tiled(A, B, C, scale, m, n, k, d_tile):
    """
    Scaled matrix multiplication with transpose, tiled along k dimension.
    Computes: C = A @ B.T * scale
    Simulates loading [m, d_tile] and [n, d_tile] blocks into shared memory.
    
    A: [m*k] input matrix (flattened)
    B: [n*k] input matrix (flattened, will be transposed)
    C: [m*n] output matrix (flattened, modified in place)
    scale: scalar scale factor
    m: number of rows in A (and C)
    n: number of rows in B (and cols in C)
    k: number of cols in A and B
    d_tile: tile size along k dimension
    """
    # Initialize output to zero
    for idx in range(m * n):
        C[idx] = 0.0
    
    # Tile over k dimension
    for k_start in range(0, k, d_tile):
        k_end = min(k_start + d_tile, k)
        k_sub = k_end - k_start
        
        # Load A_block: [m, k_sub] (simulates shared memory load)
        A_block = np.empty(m * k_sub, dtype=A.dtype)
        for i in range(m):
            for kk in range(k_sub):
                A_block[idx2d(i, kk, k_sub)] = A[idx2d(i, k_start + kk, k)]
        
        # Load B_block: [n, k_sub] (simulates shared memory load)
        B_block = np.empty(n * k_sub, dtype=B.dtype)
        for i in range(n):
            for kk in range(k_sub):
                B_block[idx2d(i, kk, k_sub)] = B[idx2d(i, k_start + kk, k)]
        
        # Accumulate partial dot products: C += A_block @ B_block.T
        for i in range(m):
            for j in range(n):
                sum_val = 0.0
                for kk in range(k_sub):
                    sum_val += A_block[idx2d(i, kk, k_sub)] * B_block[idx2d(j, kk, k_sub)]
                C[idx2d(i, j, n)] += sum_val
    
    # Apply scaling
    for idx in range(m * n):
        C[idx] *= scale

def mat_scale_rows_mul_add_d_tiled(A, v, B, C, m, n, k, d_tile_v):
    """
    FUSED: Scale rows of A, then compute and accumulate: A *= v[i]; A += B @ C
    Now WITH D-TILING: C (V matrix) is tiled along n dimension with d_tile_v.
    
    A: [m*n] matrix (flattened, modified in place)
    v: [m] vector of scale factors
    B: [m*k] matrix for matmul (flattened)
    C: [k*n] matrix for matmul (flattened)
    m: number of rows
    n: number of cols in A and C
    k: number of cols in B (rows in C)
    d_tile_v: tile size along n dimension for V
    """
    # First scale A by v
    for i in range(m):
        for j in range(n):
            idx = idx2d(i, j, n)
            A[idx] = A[idx] * v[i]
    
    # Then compute matmul in tiles along n dimension
    for d_v_start in range(0, n, d_tile_v):
        d_v_end = min(d_v_start + d_tile_v, n)
        d_v_sub = d_v_end - d_v_start
        
        # Load C_sub: [k, d_v_sub] (simulates loading V tile along d)
        C_sub = np.empty(k * d_v_sub, dtype=C.dtype)
        for i in range(k):
            for j in range(d_v_sub):
                C_sub[idx2d(i, j, d_v_sub)] = C[idx2d(i, d_v_start + j, n)]
        
        # Compute B @ C_sub for this tile
        for i in range(m):
            for j in range(d_v_sub):
                sum_val = 0.0
                for kk in range(k):
                    sum_val += B[idx2d(i, kk, k)] * C_sub[idx2d(kk, j, d_v_sub)]
                A[idx2d(i, d_v_start + j, n)] += sum_val

def mat_sub_vec_exp(S, v, out, m, n):
    """
    FUSED: Subtract vector from each row and exponentiate: out[i,j] = exp(S[i,j] - v[i])
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

def partial_attention_kernel(Q, K, V, workspace_O, workspace_m, workspace_l,
                             q_tile_idx, kv_block_idx, 
                             L, d, Bq, Bk, d_tile_qk, d_tile_v,
                             kv_block_start, kv_block_end):
    """
    KERNEL 1: Partial Attention Computation
    Each block computes attention over a subset of KV tiles.
    Simulates one GPU block in the forward pass.
    
    Grid: (num_q_tiles, num_kv_blocks, batch×heads)
    Block: (256 threads) - not explicitly parallelized in this Python simulation
    
    Inputs (read-only global memory):
      Q, K, V: [L*d] input matrices
    Outputs (write to workspace):
      workspace_O: partial output buffer
      workspace_m: partial max values
      workspace_l: partial denominator values
    Block coordinates:
      q_tile_idx: which query tile this block processes
      kv_block_idx: which KV block this block processes
    """
    q_start = q_tile_idx * Bq
    q_end = min(q_start + Bq, L)
    q_len = q_end - q_start
    
    # Allocate Q tile buffer and load (simulates shared memory)
    Q_tile = np.empty(Bq * d, dtype=Q.dtype)
    load_tile(Q, Q_tile, q_start, q_end, d)

    # Per-row streaming state for this Q tile (simulates registers)
    m = np.full(Bq, -np.inf, dtype=Q.dtype)
    l = np.zeros(Bq, dtype=Q.dtype)
    O_acc = np.zeros(Bq * d, dtype=Q.dtype)

    # Process this block's assigned KV tiles (sequential within block)
    for kv_tile_idx in range(kv_block_start, kv_block_end):
        k_start = kv_tile_idx * Bk
        k_end = min(k_start + Bk, L)
        k_len = k_end - k_start
        
        # Allocate K, V tile buffers and load (simulates shared memory)
        K_tile = np.empty(Bk * d, dtype=K.dtype)
        V_tile = np.empty(Bk * d, dtype=V.dtype)
        load_tile(K, K_tile, k_start, k_end, d)
        load_tile(V, V_tile, k_start, k_end, d)

        process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, q_len, k_len, d, d_tile_qk, d_tile_v)

    # Store partial result to workspace (simulates global memory write)
    workspace_O[(q_tile_idx, kv_block_idx)] = O_acc.copy()
    workspace_m[(q_tile_idx, kv_block_idx)] = m.copy()
    workspace_l[(q_tile_idx, kv_block_idx)] = l.copy()


def reduction_kernel(workspace_O, workspace_m, workspace_l, O_final,
                     q_tile_idx, num_kv_blocks, L, d, Bq):
    """
    KERNEL 2: Reduction Kernel
    Combines partial results from all KV blocks for one query tile.
    Simulates one GPU block in the reduction pass.
    
    Grid: (num_q_tiles, batch×heads)
    Block: (256 threads) - not explicitly parallelized in this Python simulation
    
    Inputs (read from workspace):
      workspace_O: [num_q_tiles][num_kv_blocks][Bq*d] partial outputs
      workspace_m: [num_q_tiles][num_kv_blocks][Bq] partial max values
      workspace_l: [num_q_tiles][num_kv_blocks][Bq] partial denominators
    Output (write to final output):
      O_final: [L*d] final output buffer
    Block coordinate:
      q_tile_idx: which query tile this block reduces
    
    Implements the V2 reduction formula:
      m_global[i] = max(m_1[i], m_2[i], ..., m_K[i])
      scale_k = exp(m_k[i] - m_global[i])
      O_final[i] = sum_k(O_k[i] * scale_k) / sum_k(l_k[i] * scale_k)
    """
    q_start = q_tile_idx * Bq
    q_end = min(q_start + Bq, L)
    q_len = q_end - q_start
    
    # Gather all partial results for this query tile (simulates global memory reads)
    partial_O = []
    partial_m = []
    partial_l = []
    
    for kv_block_idx in range(num_kv_blocks):
        partial_O.append(workspace_O[(q_tile_idx, kv_block_idx)])
        partial_m.append(workspace_m[(q_tile_idx, kv_block_idx)])
        partial_l.append(workspace_l[(q_tile_idx, kv_block_idx)])
    
    # Perform reduction (parallelized over threads in real GPU)
    # For each query row in this tile
    for i in range(q_len):
        # Step 1: Find global max across all partial results
        m_global = -np.inf
        for k in range(num_kv_blocks):
            if partial_m[k][i] > m_global:
                m_global = partial_m[k][i]
        
        # Step 2: Compute scaling factors and global denominator
        scales = np.empty(num_kv_blocks, dtype=np.float32)
        l_global = 0.0
        for k in range(num_kv_blocks):
            scales[k] = np.exp(partial_m[k][i] - m_global)
            l_global += partial_l[k][i] * scales[k]
        
        # Step 3: Combine outputs with proper scaling
        for j in range(d):
            numerator = 0.0
            for k in range(num_kv_blocks):
                numerator += partial_O[k][idx2d(i, j, d)] * scales[k]
            O_final[idx2d(q_start + i, j, d)] = numerator / l_global

def process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc, bq, bk, d, d_tile_qk, d_tile_v):
    """
    One tile step of FlashAttention-style streaming softmax.
    Updates m, l, O_acc in place.
    All matrix arrays are 1D buffers in row-major layout.
    
    WITH D-TILING: Q@K^T and S@V computations are both tiled along d dimension.
    
    Q_tile: [bq*d]    (flattened)
    K_tile: [bk*d]    (flattened)
    V_tile: [bk*d]    (flattened)
    m:      [bq]      (running max per query row) - modified in place
    l:      [bq]      (running denom per query row) - modified in place
    O_acc:  [bq*d]    (flattened, running numerator) - modified in place
    bq:     number of query rows in this tile
    bk:     number of key rows in this tile
    d:      head dimension
    d_tile_qk: tile size along d dimension for Q@K^T
    d_tile_v: tile size along d dimension for S@V
    """
    inv_sqrt_d = 1.0 / np.sqrt(d)

    # Allocate shared memory buffers
    S = np.empty(bq * bk, dtype=Q_tile.dtype)      # [bq*bk]
    alpha = np.empty(bq, dtype=Q_tile.dtype)       # [bq]

    # 1) Compute scores: S = Q_tile @ K_tile^T / sqrt(d)
    #    WITH D-TILING along the d dimension (using d_tile_qk)
    mat_mul_scaled_d_tiled(Q_tile, K_tile, S, inv_sqrt_d, bq, bk, d, d_tile_qk)

    # 2) Compute rescale factor BEFORE updating m
    for i in range(bq):
        # Find new max for this row
        new_max = m[i]
        for j in range(bk):
            if S[idx2d(i, j, bk)] > new_max:
                new_max = S[idx2d(i, j, bk)]
        # Compute alpha before overwriting m
        alpha[i] = np.exp(m[i] - new_max)
        # Update m in place
        m[i] = new_max

    # 3) FUSED: Shift scores and exponentiate IN PLACE
    mat_sub_vec_exp(S, m, S, bq, bk)

    # 4) FUSED: Row sum and update running denom IN PLACE
    row_sum_mul_add_inplace(S, l, alpha, bq, bk)

    # 5) FUSED: Rescale old numerator and accumulate contribution
    #    WITH D-TILING: V is tiled along d dimension (using d_tile_v)
    mat_scale_rows_mul_add_d_tiled(O_acc, alpha, S, V_tile, bq, d, bk, d_tile_v)


def flash_attention_tiled_v2(Q, K, V, O, workspace_O, workspace_m, workspace_l,
                             L, d, Bq=8, Bk=8, d_tile_qk=16, d_tile_v=16, kv_tiles_per_block=1):
    """
    FlashAttention V2 with explicit two-kernel structure.
    Simulates the GPU implementation with separate forward and reduction kernels.
    
    V2 TWO-KERNEL APPROACH:
    1. KERNEL 1 (partial_attention_kernel): Compute partial attention
       - Grid: (num_q_tiles, num_kv_blocks, batch×heads)
       - Each block processes one Q tile × kv_tiles_per_block KV tiles
       - Writes partial results to workspace memory
    
    2. KERNEL 2 (reduction_kernel): Combine partial results
       - Grid: (num_q_tiles, batch×heads)
       - Each block combines all KV blocks for one Q tile
       - Writes final output to O
    
    Memory layout (all 1D buffers, row-major):
      Inputs (read-only global memory):
        Q, K, V: [L*d] input matrices
      Workspace (scratch memory, allocated by caller):
        workspace_O: dict[(q_tile_idx, kv_block_idx)] -> [Bq*d] partial outputs
        workspace_m: dict[(q_tile_idx, kv_block_idx)] -> [Bq] partial max values
        workspace_l: dict[(q_tile_idx, kv_block_idx)] -> [Bq] partial denominators
      Output (write-only global memory):
        O: [L*d] final output buffer
    
    V2 PARALLELISM: kv_tiles_per_block controls the trade-off:
    - Small (1-2): Maximum parallelism, more blocks, more workspace memory
    - Medium (4): Good balance (what we use)
    - Large: Less parallelism, approaches V1 behavior
    """
    num_q_tiles = (L + Bq - 1) // Bq
    num_kv_tiles = (L + Bk - 1) // Bk
    num_kv_blocks = (num_kv_tiles + kv_tiles_per_block - 1) // kv_tiles_per_block
    
    # ===== KERNEL 1: PARTIAL ATTENTION COMPUTATION =====
    # Launch grid: (num_q_tiles, num_kv_blocks, batch×heads)
    # Each iteration simulates launching one GPU block
    for q_tile_idx in range(num_q_tiles):
        for kv_block_idx in range(num_kv_blocks):
            kv_block_start = kv_block_idx * kv_tiles_per_block
            kv_block_end = min(kv_block_start + kv_tiles_per_block, num_kv_tiles)
            
            # Launch partial_attention_kernel
            partial_attention_kernel(
                Q, K, V, workspace_O, workspace_m, workspace_l,
                q_tile_idx, kv_block_idx,
                L, d, Bq, Bk, d_tile_qk, d_tile_v,
                kv_block_start, kv_block_end
            )
    
    # In real GPU: cudaDeviceSynchronize() or __threadfence() to ensure visibility
    
    # ===== KERNEL 2: REDUCTION =====
    # Launch grid: (num_q_tiles, batch×heads)
    # Each iteration simulates launching one GPU block
    for q_tile_idx in range(num_q_tiles):
        # Launch reduction_kernel
        reduction_kernel(
            workspace_O, workspace_m, workspace_l, O,
            q_tile_idx, num_kv_blocks, L, d, Bq
        )


if __name__ == "__main__":
    L = 256  # Reduced from 2048 for faster CPU execution
    d = 128
    rng = np.random.default_rng(0)

    # Generate 2D arrays first for naive implementation
    Q_2d = rng.standard_normal((L, d)).astype(np.float16)
    K_2d = rng.standard_normal((L, d)).astype(np.float16)
    V_2d = rng.standard_normal((L, d)).astype(np.float16)
    
    # Flatten to 1D for tiled implementation (like GPU)
    Q = Q_2d.flatten()
    K = K_2d.flatten()
    V = V_2d.flatten()
    O_tiled = np.zeros(L * d, dtype=np.float16)

    # Calculate grid dimensions
    num_q_tiles = (L + BQ - 1) // BQ
    num_kv_tiles = (L + BK - 1) // BK
    num_kv_blocks = (num_kv_tiles + KV_TILES_PER_BLOCK - 1) // KV_TILES_PER_BLOCK
    total_forward_blocks = num_q_tiles * num_kv_blocks
    
    # Allocate workspace memory (global memory scratch space)
    # In real GPU: cudaMalloc() for these buffers
    # Shape: [num_q_tiles][num_kv_blocks] -> [BQ*d] or [BQ]
    workspace_O = {}  # Partial outputs: [BQ*d] per block
    workspace_m = {}  # Partial max values: [BQ] per block
    workspace_l = {}  # Partial denominators: [BQ] per block
    
    print(f"Configuration:")
    print(f"  L={L}, d={d}, BQ={BQ}, BK={BK}")
    print(f"  D_TILE_QK={D_TILE_QK}, D_TILE_V={D_TILE_V}")
    print(f"  KV_TILES_PER_BLOCK={KV_TILES_PER_BLOCK}")
    print(f"\nGrid dimensions:")
    print(f"  Kernel 1 (forward):  ({num_q_tiles} q_tiles, {num_kv_blocks} kv_blocks) = {total_forward_blocks} blocks")
    print(f"  Kernel 2 (reduction): ({num_q_tiles} q_tiles) = {num_q_tiles} blocks")
    print(f"\nWorkspace memory:")
    print(f"  Total entries: {total_forward_blocks} (one per forward block)")
    print(f"  Size per entry: {BQ}*{d} + {BQ} + {BQ} = {BQ*d + 2*BQ} elements")
    print()
    
    flash_attention_tiled_v2(Q, K, V, O_tiled, workspace_O, workspace_m, workspace_l,
                             L, d, 
                             Bq=BQ, Bk=BK, 
                             d_tile_qk=D_TILE_QK, d_tile_v=D_TILE_V,
                             kv_tiles_per_block=KV_TILES_PER_BLOCK)
    
    # Reshape back to 2D for comparison
    O_tiled_2d = O_tiled.reshape(L, d)
    O_reference = naive_attention(Q_2d, K_2d, V_2d)

    print_comparison(O_tiled_2d, O_reference)
    check_accuracy(O_tiled_2d, O_reference, 
                   f"Bq={BQ}, Bk={BK}, d_tile_qk={D_TILE_QK}, d_tile_v={D_TILE_V}, kv_tiles_per_block={KV_TILES_PER_BLOCK}")