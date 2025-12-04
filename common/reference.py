"""
Common reference implementation and validation utilities for Flash Attention experiments.
"""
import numpy as np


def naive_attention(Q, K, V):
    """
    Naive attention: softmax(Q K^T / sqrt(d)) V
    Reference implementation for correctness checking.
    
    Q, K, V: [L, d] NumPy arrays
    Returns: [L, d]
    """
    L, d = Q.shape
    scale = 1.0 / np.sqrt(d)
    scores = (Q @ K.T) * scale           # [L, L]
    scores = scores - scores.max(axis=1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs @ V                     # [L, d]


def check_accuracy(output, reference, config_str="", max_abs_tol=1e-2, max_rel_tol=0.5, mean_rel_tol=0.05):
    """
    Check accuracy of output against reference implementation.
    Prints error metrics and raises assertion if tolerances are exceeded.
    
    Args:
        output: [L, d] output array to check
        reference: [L, d] reference array
        config_str: Optional configuration string to print (e.g., "Bq=8, Bk=8")
        max_abs_tol: Maximum allowable absolute difference (default: 1e-2)
        max_rel_tol: Maximum allowable relative difference for |reference| > 1e-3 (default: 0.5 = 50%)
        mean_rel_tol: Maximum allowable mean relative error (default: 0.05 = 5%)
    """
    passed = True
    errors = []
    
    # Check closeness
    diff = np.abs(output - reference).max()
    if config_str:
        print(f"\nMax absolute difference ({config_str}):", diff)
    else:
        print("\nMax absolute difference:", diff)
    
    if diff > max_abs_tol:
        errors.append(f"Max absolute difference {diff:.6f} exceeds tolerance {max_abs_tol}")
        passed = False
    
    # Relative difference (only for significant values to avoid division by near-zero)
    mask = np.abs(reference) > 1e-3
    if mask.any():
        rel_diff_filtered = (np.abs(output[mask] - reference[mask]) / np.abs(reference[mask])).max()
        print(f"Max relative difference (|reference| > 1e-3):", rel_diff_filtered)
        
        if rel_diff_filtered > max_rel_tol:
            errors.append(f"Max relative difference {rel_diff_filtered:.6f} exceeds tolerance {max_rel_tol}")
            passed = False
        
        # Mean relative error (excluding very small values)
        mean_rel_err = (np.abs(output[mask] - reference[mask]) / np.abs(reference[mask])).mean()
        print("Mean relative error (|reference| > 1e-3):", mean_rel_err)
        
        if mean_rel_err > mean_rel_tol:
            errors.append(f"Mean relative error {mean_rel_err:.6f} exceeds tolerance {mean_rel_tol}")
            passed = False
    else:
        print("Warning: No values > 1e-3 for relative error calculation")
    
    # Print result
    if passed:
        print("✓ PASSED - All error metrics within tolerances")
    else:
        print("✗ FAILED - Error tolerance(s) exceeded:")
        for error in errors:
            print(f"  - {error}")
        raise AssertionError(f"Accuracy check failed: {'; '.join(errors)}")


def print_comparison(output, reference, num_rows=3, num_cols=5):
    """
    Print side-by-side comparison of output and reference.
    
    Args:
        output: [L, d] output array
        reference: [L, d] reference array
        num_rows: Number of rows to print
        num_cols: Number of columns to print
    """
    print("Output shape:", output.shape)
    print(f"First {num_rows} rows (output):")
    print(output[:num_rows, :num_cols])
    
    print(f"\nFirst {num_rows} rows (reference):")
    print(reference[:num_rows, :num_cols])
