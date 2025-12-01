import torch
import torch.nn.functional as F

def run_pytorch_attention(L, d):
    # Create random FP16 Q, K, V
    Q = torch.randn(1, 1, L, d, dtype=torch.float16)
    K = torch.randn(1, 1, L, d, dtype=torch.float16)
    V = torch.randn(1, 1, L, d, dtype=torch.float16)

    # Use PyTorch fused attention if available
    if hasattr(F, "scaled_dot_product_attention"):
        out = F.scaled_dot_product_attention(Q, K, V)  # [1, 1, L, d]
    else:
        print("Warning: Using manual attention computation as fused attention is not available.")
        scale = 1.0 / (d ** 0.5)
        scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, V)

    return out[0, 0]   # return shape [L, d]

def main():
    L = 2048
    d = 32
    
    print(f"Running PyTorch attention with L={L}, d={d}")
    out = run_pytorch_attention(L, d)
    
    print(f"Output shape: {out.shape}")
    print(f"First 4 tokens, first 8 dims of output:")
    print(out[:4, :8])

if __name__ == "__main__":
    main()