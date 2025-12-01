import numpy as np

def process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc):
    """
    One tile step of FlashAttention-style streaming softmax.

    Q_tile: [q_len, d]
    K_tile: [k_len, d]
    V_tile: [k_len, d]
    m:      [q_len]      (running max per query row)
    l:      [q_len]      (running denom per query row)
    O_acc:  [q_len, d]   (running numerator per query row)

    Returns updated (m, l, O_acc).
    """
    q_len, d = Q_tile.shape
    k_len, _ = K_tile.shape

    inv_sqrt_d = 1.0 / np.sqrt(d)

    # 1) Scores for this tile: S = Q_tile @ K_tile^T / sqrt(d)
    #    S: [q_len, k_len]
    S = (Q_tile @ K_tile.T) * inv_sqrt_d

    # 2) Block row-max over keys
    #    m_block[i] = max_j S[i, j]
    m_block = S.max(axis=1)  # [q_len]

    # 3) New running max per row
    m_new = np.maximum(m, m_block)  # [q_len]

    # 4) Rescale factor from old max -> new max
    #    alpha[i] = exp(m[i] - m_new[i])
    alpha = np.exp(m - m_new)       # [q_len]

    # 5) Shift scores by new max (for stability)
    #    S_shifted[i, j] = S[i, j] - m_new[i]
    S_shifted = S - m_new[:, None]  # [q_len, k_len]

    # 6) Exponentiate shifted scores
    exp_S = np.exp(S_shifted)       # [q_len, k_len]

    # 7) This tile's contribution to the denominator
    #    l_tile[i] = sum_j exp_S[i, j]
    l_tile = exp_S.sum(axis=1)      # [q_len]

    # 8) New denom:
    #    l_new[i] = l[i] * alpha[i] + l_tile[i]
    l_new = l * alpha + l_tile      # [q_len]

    # 9) Rescale old numerator to the new max:
    #    O_acc[i, :] *= alpha[i]
    O_acc = O_acc * alpha[:, None]  # broadcast over d

    # 10) Add this tile's contribution to numerator:
    #     contrib = exp_S @ V_tile
    #     exp_S: [q_len, k_len]
    #     V_tile: [k_len, d]
    contrib = exp_S @ V_tile        # [q_len, d]
    O_acc = O_acc + contrib         # [q_len, d]

    return m_new, l_new, O_acc


def flash_attention_tiled(Q, K, V, Bq=8, Bk=8):
    """
    FlashAttention-style attention using tiles and streaming softmax.

    Q, K, V: [L, d] NumPy arrays
    Bq: query tile size
    Bk: key/value tile size

    Returns O: [L, d]
    """
    L, d = Q.shape
    O = np.zeros((L, d), dtype=Q.dtype)

    # Loop over query tiles
    for q_start in range(0, L, Bq):
        q_end = min(q_start + Bq, L)
        q_len = q_end - q_start

        Q_tile = Q[q_start:q_end, :]       # [q_len, d]

        # Per-row streaming state for this Q tile
        m = np.full(q_len, -np.inf, dtype=Q.dtype)   # [q_len]
        l = np.zeros(q_len, dtype=Q.dtype)           # [q_len]
        O_acc = np.zeros((q_len, d), dtype=Q.dtype)  # [q_len, d]

        # Loop over K/V tiles
        for k_start in range(0, L, Bk):
            k_end = min(k_start + Bk, L)
            K_tile = K[k_start:k_end, :]   # [k_len, d]
            V_tile = V[k_start:k_end, :]   # [k_len, d]

            m, l, O_acc = process_kv_tile(Q_tile, K_tile, V_tile, m, l, O_acc)

        # Finalize this Q tile: divide numerator by denom
        O[q_start:q_end, :] = O_acc / l[:, None]

    return O


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
    Q = rng.standard_normal((L, d)).astype(np.float16)
    K = rng.standard_normal((L, d)).astype(np.float16)
    V = rng.standard_normal((L, d)).astype(np.float16)

    O_tiled = flash_attention_tiled(Q, K, V, Bq=8, Bk=8)
    O_naive = naive_attention(Q, K, V)

    print("O_tiled shape:", O_tiled.shape)
    print("First 3 rows (tiled):")
    print(O_tiled[:3, :5])

    print("\nFirst 3 rows (naive):")
    print(O_naive[:3, :5])

    # Check closeness
    diff = np.abs(O_tiled - O_naive).max()
    print("\nMax absolute difference:", diff)