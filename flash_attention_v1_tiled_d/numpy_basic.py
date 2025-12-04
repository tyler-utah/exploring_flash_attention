import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.reference import naive_attention, check_accuracy, print_comparison

# Tuning parameters
BQ = 8        # Query tile size
BK = 8        # Key/Value tile size
D_TILE_QK = 16   # Head dimension tile size for Q@K^T
D_TILE_V = 16    # Head dimension tile size for S@V

def process_kv_tile_global(Q, K, V,
                           q_start, q_len,
                           k_start, k_len,
                           m, l, O_acc,
                           d_tile_qk=16,
                           d_tile_v=16):
    """
    One tile step of FlashAttention-style streaming softmax,
    using *global* Q, K, V and only loading small [Bq, d_tile] / [Bk, d_tile]
    blocks at a time (these blocks correspond to what would live in GPU shared
    memory at any instant).

    Q, K, V: [L, d] global arrays
    q_start: starting row index in Q
    q_len:   number of query rows in this tile
    k_start: starting row index in K/V
    k_len:   number of key/value rows in this tile

    m:      [q_len]    running max per query row
    l:      [q_len]    running denom per query row
    O_acc:  [q_len, d] running numerator per query row

    d_tile_qk: head-dimension tile size for Q@K^T (how much of d we "load" at once)
    d_tile_v:  head-dimension tile size for exp(S)@V
    """
    # Global dims
    _, d = Q.shape

    inv_sqrt_d = 1.0 / np.sqrt(d)

    # 1) Compute scores S for this (q_block, k_block) only,
    #    streaming over d in chunks of size d_tile_qk.
    #    S: [q_len, k_len]
    S = np.zeros((q_len, k_len), dtype=Q.dtype)

    for d_start in range(0, d, d_tile_qk):
        d_end = min(d_start + d_tile_qk, d)
        d_sub = d_end - d_start

        # These correspond to tiles that would live in shared memory:
        # Q_block: [q_len, d_sub], K_block: [k_len, d_sub]
        Q_block = Q[q_start:q_start + q_len, d_start:d_end]  # [q_len, d_sub]
        K_block = K[k_start:k_start + k_len, d_start:d_end]  # [k_len, d_sub]

        # Accumulate partial dot products
        S += Q_block @ K_block.T  # [q_len, k_len]

    S *= inv_sqrt_d

    # 2) Block row-max over keys: m_block[i] = max_j S[i, j]
    m_block = S.max(axis=1)  # [q_len]

    # 3) New running max per row
    m_new = np.maximum(m, m_block)  # [q_len]

    # 4) Rescale factor from old max -> new max: alpha[i] = exp(m[i] - m_new[i])
    alpha = np.exp(m - m_new)       # [q_len]

    # 5) Shift scores by new max (for stability): S_shifted[i, j] = S[i, j] - m_new[i]
    S_shifted = S - m_new[:, None]  # [q_len, k_len]

    # 6) Exponentiate shifted scores
    exp_S = np.exp(S_shifted)       # [q_len, k_len]

    # 7) This tile's contribution to the denominator: l_tile[i] = sum_j exp_S[i, j]
    l_tile = exp_S.sum(axis=1)      # [q_len]

    # 8) New denom: l_new[i] = l[i] * alpha[i] + l_tile[i]
    l_new = l * alpha + l_tile      # [q_len]

    # 9) Rescale old numerator to the new max: O_acc[i, :] *= alpha[i]
    O_acc = O_acc * alpha[:, None]  # [q_len, d]

    # 10) Add this tile's contribution to numerator: exp(S) @ V
    #      Now V is ALSO tiled along d with d_tile_v.
    for d_v_start in range(0, d, d_tile_v):
        d_v_end = min(d_v_start + d_tile_v, d)

        V_sub = V[k_start:k_start + k_len, d_v_start:d_v_end]   # [k_len, d_v_sub]
        contrib_sub = exp_S @ V_sub                              # [q_len, d_v_sub]

        O_acc[:, d_v_start:d_v_end] += contrib_sub

    return m_new, l_new, O_acc


def flash_attention_tiled_global(Q, K, V, Bq=8, Bk=8, d_tile_qk=16, d_tile_v=16):
    """
    FlashAttention-style attention using tiles and streaming softmax,
    where "tiles" correspond to what would be loaded into GPU shared memory:

      - Bq: query tile size (rows of Q handled by a block)
      - Bk: key/value tile size (rows of K/V per tile)
      - d_tile_qk: how much of the head dimension d we process at once for Q@K^T
      - d_tile_v: how much of the head dimension d we process at once for exp(S)@V

    Q, K, V: [L, d] NumPy arrays
    Returns O: [L, d]
    """
    assert Q.shape == K.shape == V.shape, "Q, K, V must have the same shape [L, d]"
    L, d = Q.shape
    assert L > 0 and d > 0

    assert isinstance(Bq, int) and Bq > 0
    assert isinstance(Bk, int) and Bk > 0
    assert isinstance(d_tile_qk, int) and d_tile_qk > 0
    assert isinstance(d_tile_v, int) and d_tile_v > 0
    assert d_tile_qk <= d and d_tile_v <= d

    O = np.zeros((L, d), dtype=Q.dtype)

    # Loop over query *row ranges* (not material tiles)
    for q_start in range(0, L, Bq):
        q_end = min(q_start + Bq, L)
        q_len = q_end - q_start

        # Per-row streaming state for this range of queries
        m = np.full(q_len, -np.inf, dtype=Q.dtype)   # [q_len]
        l = np.zeros(q_len, dtype=Q.dtype)           # [q_len]
        O_acc = np.zeros((q_len, d), dtype=Q.dtype)  # [q_len, d]

        # Loop over key/value *row ranges*
        for k_start in range(0, L, Bk):
            k_end = min(k_start + Bk, L)
            k_len = k_end - k_start

            m, l, O_acc = process_kv_tile_global(
                Q, K, V,
                q_start, q_len,
                k_start, k_len,
                m, l, O_acc,
                d_tile_qk=d_tile_qk,
                d_tile_v=d_tile_v
            )

        # Finalize outputs for these query rows
        O[q_start:q_end, :] = O_acc / l[:, None]

    return O


if __name__ == "__main__":
    L = 2048
    d = 128
    rng = np.random.default_rng(0)

    # Use float16 for stability testing
    Q = rng.standard_normal((L, d)).astype(np.float16)
    K = rng.standard_normal((L, d)).astype(np.float16)
    V = rng.standard_normal((L, d)).astype(np.float16)

    # Use tuning parameters from top of file
    O_tiled = flash_attention_tiled_global(Q, K, V, Bq=BQ, Bk=BK, d_tile_qk=D_TILE_QK, d_tile_v=D_TILE_V)
    O_reference = naive_attention(Q, K, V)

    print_comparison(O_tiled, O_reference)
    check_accuracy(O_tiled, O_reference, f"Bq={BQ}, Bk={BK}, d_tile_qk={D_TILE_QK}, d_tile_v={D_TILE_V}")
