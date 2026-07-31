/**
 * q4_dequant_common.h
 * Shared Q4_K dequantization device functions for decode benchmarks.
 *
 * Extracted to avoid duplicated Q4_K dequant logic across:
 *   - decode_step_bench.cu
 *   - fused_decode.cu
 *
 * Q4_K block format (from ds4.c):
 *   Each block stores 256 values in 144 bytes:
 *     d (2B F16 scale) + dmin (2B F16 min) + scales (12B) + qs (128B = 256 × 4-bit)
 */

#ifndef Q4_DEQUANT_COMMON_H
#define Q4_DEQUANT_COMMON_H

#include <cuda_runtime.h>
#include <stdint.h>

#define QK_K 256

// Q4_K block format (from ds4.c)
typedef struct {
    uint16_t d;             // scale (f16 stored as uint16_t)
    uint16_t dmin;          // min scale (f16 stored as uint16_t)
    uint8_t  scales[12];    // sub-block scales
    uint8_t  qs[QK_K / 2];  // 4-bit quantized values (128 bytes)
} block_q4_K;               // total: 144 bytes

// F16 → F32 conversion
static __inline__ __device__ float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        f32 = (sign << 31) | 0;
    } else if (exp == 31) {
        f32 = (sign << 31) | 0x7F800000 | (man << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    return __uint_as_float(f32);
}

/**
 * Dequantize a single Q4_K value at position l within a block.
 * Returns the dequantized float value.
 *
 * Parameters:
 *   blk - pointer to Q4_K block
 *   l   - index within the block [0, QK_K)
 *
 * The block has 256 values organized as:
 *   8 sub-blocks of 32 values each
 *   For each sub-block j (0..7), scales[j] and scales[j+4..7+j] provide
 *   the scale and min for that sub-block.
 */
static __inline__ __device__ float q4k_dequantize(const block_q4_K *blk, int l) {
    int j = l / 32;          // sub-block index (0..7)
    int pos = l % 32;        // position within sub-block
    int byte_off = (j >> 1) * 32;
    int shift = (j & 1) * 4;
    int q = (blk->qs[byte_off + pos] >> shift) & 0x0F;

    float d = f16_to_f32(blk->d);
    float dmin = f16_to_f32(blk->dmin);

    uint8_t sc_val, m_val;
    if (j < 4) {
        sc_val = blk->scales[j] & 0x3F;
        m_val  = blk->scales[j + 4] & 0x3F;
    } else {
        sc_val = (blk->scales[j + 4] & 0xF) | ((blk->scales[j - 4] >> 6) << 4);
        m_val  = (blk->scales[j + 4] >> 4) | ((blk->scales[j - 0] >> 6) << 4);
    }

    float scale = d * (float)sc_val;
    float minv = dmin * (float)m_val;
    return scale * (float)q - minv;
}

/**
 * Warp-level reduction helper.
 * Sums a value across all threads in the same warp.
 */
static __inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * Block-level reduction helper.
 * Reduces shared[tid] values across all threads in the block.
 * After call, result is in shared[0].
 */
static __inline__ __device__ void blockReduceSum(volatile float *shared) {
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
}

#endif // Q4_DEQUANT_COMMON_H
