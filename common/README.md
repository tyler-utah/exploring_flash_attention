# Common Utilities

This directory contains shared utilities used across all Flash Attention implementations.

## Files

### `reference.py`

Contains the reference implementation and validation utilities:

- **`naive_attention(Q, K, V)`**: Reference implementation of attention for correctness checking
  - Computes: `softmax(Q @ K^T / sqrt(d)) @ V`
  - Uses numerically stable softmax with max subtraction
  - Q, K, V: `[L, d]` NumPy arrays
  - Returns: `[L, d]` output

- **`check_accuracy(output, reference, config_str="")`**: Validates output against reference
  - Prints max absolute difference
  - Prints max relative difference (filtered for |values| > 1e-3 to avoid near-zero issues)
  - Prints mean relative error (filtered)
  - Optional config_str for context (e.g., "Bq=8, Bk=8")

- **`print_comparison(output, reference, num_rows=3, num_cols=5)`**: Pretty-prints side-by-side comparison
  - Shows first few rows and columns of both arrays
  - Useful for visual inspection

## Usage

All Python implementations import these utilities:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.reference import naive_attention, check_accuracy, print_comparison

# ... your implementation ...

if __name__ == "__main__":
    # Generate test data
    Q, K, V = ...
    
    # Run your implementation
    output = your_flash_attention(Q, K, V)
    
    # Validate against reference
    reference = naive_attention(Q, K, V)
    print_comparison(output, reference)
    check_accuracy(output, reference, "your_config_here")
```

## Benefits

- **Consistency**: All implementations use the same reference and validation logic
- **Maintainability**: Bug fixes and improvements in one place
- **Readability**: Less duplicate code in each implementation file
- **Reliability**: Uniform error checking across all experiments
