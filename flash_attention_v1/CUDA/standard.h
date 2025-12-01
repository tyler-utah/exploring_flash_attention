#ifndef STANDARD_H
#define STANDARD_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <omp.h>

// Import precision configuration from flash_attention_v1.h
#ifndef USE_FP64
#define USE_FP64 0
#endif

#if USE_FP64
#define DATA_TYPE double
#define FLOAT_TO_DATA(x) (x)
#define DATA_TO_FLOAT(x) (x)
#else
#define DATA_TYPE __half
#define FLOAT_TO_DATA(x) __float2half(x)
#define DATA_TO_FLOAT(x) __half2float(x)
#endif

// Standard attention implementation on CPU for reference
// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
// Input: Q, K, V: [B, H, L, d]
// Output: O: [B, H, L, d]
void standard_attention_cpu(
    const DATA_TYPE* Q,  // [B, H, L, d]
    const DATA_TYPE* K,  // [B, H, L, d]
    const DATA_TYPE* V,  // [B, H, L, d]
    DATA_TYPE* O,        // [B, H, L, d] output
    int B,            // batch size
    int H,            // number of heads
    int L,            // sequence length
    int d             // head dimension
) {
    float scale = 1.0f / sqrtf((float)d);
    
    // Process each (batch, head) pair independently
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // Compute base offset for this (batch, head) pair
            int base_offset = (b * H * L * d) + (h * L * d);
            const DATA_TYPE* Q_bh = Q + base_offset;
            const DATA_TYPE* K_bh = K + base_offset;
            const DATA_TYPE* V_bh = V + base_offset;
            DATA_TYPE* O_bh = O + base_offset;
            
            // Allocate temporary scores matrix [L, L] for this head
            float* scores = new float[L * L];
            
            // Compute scores: Q @ K^T * scale
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < L; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < d; k++) {
                        sum += DATA_TO_FLOAT(Q_bh[i * d + k]) * DATA_TO_FLOAT(K_bh[j * d + k]);
                    }
                    scores[i * L + j] = sum * scale;
                }
            }
            
            // Apply softmax row-wise
            for (int i = 0; i < L; i++) {
                // Find max for numerical stability
                float max_val = scores[i * L];
                for (int j = 1; j < L; j++) {
                    if (scores[i * L + j] > max_val) {
                        max_val = scores[i * L + j];
                    }
                }
                
                // Exp and sum
                float sum = 0.0f;
                for (int j = 0; j < L; j++) {
                    scores[i * L + j] = expf(scores[i * L + j] - max_val);
                    sum += scores[i * L + j];
                }
                
                // Normalize
                for (int j = 0; j < L; j++) {
                    scores[i * L + j] /= sum;
                }
            }
            
            // Compute output: scores @ V
            for (int i = 0; i < L; i++) {
                for (int j = 0; j < d; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < L; k++) {
                        sum += scores[i * L + k] * DATA_TO_FLOAT(V_bh[k * d + j]);
                    }
                    O_bh[i * d + j] = FLOAT_TO_DATA(sum);
                }
            }
            
            delete[] scores;
        }
    }
}

#endif // STANDARD_H
