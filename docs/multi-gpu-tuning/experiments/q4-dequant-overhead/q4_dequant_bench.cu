/**
 * q4_dequant_bench.cu
 * Q4_K weight read + dequant micro-benchmark
 *
 * Measures: effective memory bandwidth when reading Q4_K weights with
 * on-the-fly dequantization vs raw F16 read bandwidth.
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 *
 * Build:
 *   nvcc -o q4_dequant_bench q4_dequant_bench.cu -lcuda -lcudart
 *
 * GROUND-RULES: Hypothesis-driven. Each test measures effective BW of
 * Q4_K read+dequant vs raw F16 read. Values tagged per §1.2.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define WARMUP_ITER 20
#define MAX_TRIALS  200
#define STABLE_CV_THRESHOLD 0.03

// Q4_K block format (from ds4.c)
#define QK_K 256
typedef struct {
    uint16_t d;           // scale (f16 stored as uint16_t)
    uint16_t dmin;        // min scale (f16 stored as uint16_t)
    uint8_t  scales[12];  // 12 bytes of sub-block scales
    uint8_t  qs[QK_K / 2]; // 128 bytes of 4-bit quantized values
} block_q4_K;
// sizeof(block_q4_K) == 144 bytes

typedef struct {
    double mean_ms;
    double std_ms;
    double throughput_gbs;   // effective GB/s
    double throughput_toks;  // effective values/s (in billions)
    int    n_trials;
} bench_result;

// F16 → F32 conversion (from ds4.c)
static __inline__ __device__ float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        // Subnormal → zero
        f32 = (sign << 31) | 0;
    } else if (exp == 31) {
        // Inf/NaN
        f32 = (sign << 31) | 0x7F800000 | (man << 13);
    } else {
        f32 = (sign << 31) | ((exp + 112) << 23) | (man << 13);
    }
    return __uint_as_float(f32);
}

// Q4_K dequant + reduce for a single block
__device__ float dequant_q4_K_block(const block_q4_K *block) {
    float d = f16_to_f32(block->d);
    float dmin = f16_to_f32(block->dmin);
    const uint8_t *qs = block->qs;
    const uint8_t *scales = block->scales;
    
    float sum = 0.0f;
    
    // 8 groups of 32 values each (256 total)
    for (int j = 0; j < QK_K / 32; j++) {
        // Extract scale and min for this sub-block
        uint8_t sc_val, m_val;
        if (j < 4) {
            sc_val = scales[j] & 0x3F;
            m_val  = scales[j + 4] & 0x3F;
        } else {
            sc_val = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
            m_val  = (scales[j + 4] >> 4)  | ((scales[j - 0] >> 6) << 4);
        }
        
        const int byte_off = (j >> 1) * 32;
        const int shift = (j & 1) * 4;
        const float scale = d * (float)sc_val;
        const float minv = dmin * (float)m_val;
        
        // Dequantize 32 4-bit values
        for (int l = 0; l < 32; l++) {
            const int q = (qs[byte_off + l] >> shift) & 0x0F;
            sum += scale * (float)q - minv;
        }
    }
    return sum;
}

// ============================================================
// Kernel 1: Q4_K read + dequant + accumulate
// Each thread processes multiple blocks
// ============================================================
__global__ void q4k_read_dequant_kernel(const block_q4_K *in,
                                         float *out,
                                         uint64_t n_blocks,
                                         int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    float reg = 0.0f;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_blocks; i += stride) {
            reg += dequant_q4_K_block(&in[i]);
        }
    }
    
    if (idx == 0) out[0] = reg; // prevent dead code elimination
}

// ============================================================
// Kernel 2: Raw F16 read + accumulate (no dequant)
// Each thread processes multiple F16 values
// ============================================================
__global__ void f16_read_kernel(const uint16_t *in,
                                 float *out,
                                 uint64_t n_values,
                                 int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    float reg = 0.0f;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_values; i += stride) {
            reg += f16_to_f32(in[i]);
        }
    }
    
    if (idx == 0) out[0] = reg;
}

// ============================================================
// Kernel 3: Raw F32 read + accumulate (baseline)
// ============================================================
__global__ void f32_read_kernel(const float *in,
                                 float *out,
                                 uint64_t n_values,
                                 int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    float reg = 0.0f;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_values; i += stride) {
            reg += in[i];
        }
    }
    
    if (idx == 0) out[0] = reg;
}

// GPU elapsed time in ms
static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// Run kernel, measure throughput, return result
static bench_result run_bench(const char *name, int device,
                               cudaStream_t stream,
                               cudaEvent_t start, cudaEvent_t stop,
                               void (*kernel)(const void*, float*, uint64_t, int),
                               const void *d_in, float *d_out,
                               uint64_t n_work_items, int iters,
                               double total_bytes_read, // for bandwidth computation
                               dim3 grid_dim, dim3 block_dim)
{
    bench_result r = {0};
    cudaSetDevice(device);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        // Cast kernel via function pointer
        // SAFETY: name is always non-NULL (callers pass literal strings)
        if (name && name[0] == 'Q') {
            q4k_read_dequant_kernel<<<grid_dim, block_dim, 0, stream>>>(
                (const block_q4_K*)d_in, d_out, n_work_items, iters);
        } else if (name && name[0] == 'F' && strstr(name, "16")) {
            f16_read_kernel<<<grid_dim, block_dim, 0, stream>>>(
                (const uint16_t*)d_in, d_out, n_work_items, iters);
        } else {
            f32_read_kernel<<<grid_dim, block_dim, 0, stream>>>(
                (const float*)d_in, d_out, n_work_items, iters);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = elapsed_ms(start, stop);
    }
    
    // Stats
    double sum = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum += times[i];
    r.n_trials = MAX_TRIALS - WARMUP_ITER;
    r.mean_ms = sum / r.n_trials;
    
    double sum2 = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / r.n_trials);
    
    double total_s = r.mean_ms / 1000.0;
    r.throughput_gbs = total_bytes_read * iters / total_s / 1e9;
    
    // Values processed per second
    uint64_t total_values = n_work_items;
    if (name[0] == 'Q') total_values *= QK_K; // Q4_K: blocks * 256 values
    double values_per_sec = (double)total_values * iters / total_s;
    r.throughput_toks = values_per_sec / 1e9;
    
    printf("  %-25s | GPU %d | mean=%.3fms | std=%.3fms | %.0f GB/s | %.3fB values/s | CV=%.2f%%\n",
           name, device, r.mean_ms, r.std_ms,
           r.throughput_gbs, r.throughput_toks,
           r.std_ms / r.mean_ms * 100);
    
    fprintf(stdout, "DATA_%s\t%d\t%.3f\t%.3f\t%.3f\t%.0f\n",
            name, device, r.mean_ms, r.std_ms, r.throughput_gbs, r.throughput_toks);
    
    free(times);
    return r;
}

int main()
{
    printf("=== Q4_K Dequant Overhead Micro-Benchmark ===\n");
    printf("System: Dual NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q\n");
    printf("CUDA Runtime: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    printf("Devices found: %d\n\n", n_devices);
    if (n_devices < 1) {
        printf("FATAL: Need >=1 GPU\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU0: %s, SMs=%d, %.1f GB HBM, cc=%d.%d\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9,
           prop.major, prop.minor);
    
    // Allocate buffers
    // Q4_K: 144 bytes per block = 256 values
    // F16: 512 bytes per 256 values
    // F32: 1024 bytes per 256 values
    
    // Target ~2 GiB for Q4_K buffer
    uint64_t n_blocks = (uint64_t)(2ULL * 1024 * 1024 * 1024) / sizeof(block_q4_K);
    uint64_t n_values = n_blocks * QK_K; // values represented
    uint64_t f16_bytes = n_values * sizeof(uint16_t); // F16 buffer: 2 bytes * n_values
    
    printf("Buffer sizes:\n");
    printf("  Q4_K blocks: %lu (%.2f GiB, %.1fB logical values)\n",
           (unsigned long)n_blocks,
           (double)n_blocks * sizeof(block_q4_K) / (1024*1024*1024),
           (double)n_values / 1e9);
    printf("  F16 values:  %lu (%.2f GiB)\n",
           (unsigned long)n_values,
           (double)f16_bytes / (1024*1024*1024));
    printf("  F32 values:  %lu (%.2f GiB)\n",
           (unsigned long)n_values,
           (double)n_values * sizeof(float) / (1024*1024*1024));
    printf("\n");
    
    // Allocate on GPU0
    cudaSetDevice(0);
    
    block_q4_K *d_q4k;
    uint16_t *d_f16;
    float *d_f32, *d_out;
    
    cudaMalloc(&d_q4k, n_blocks * sizeof(block_q4_K));
    cudaMalloc(&d_f16, n_values * sizeof(uint16_t));
    cudaMalloc(&d_f32, n_values * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    
    // Initialize Q4_K buffer with known patterns
    block_q4_K *h_q4k = (block_q4_K*)calloc(n_blocks, sizeof(block_q4_K));
    for (uint64_t i = 0; i < n_blocks; i++) {
        h_q4k[i].d = 0x3C00; // F16 1.0
        h_q4k[i].dmin = 0x0000; // F16 0.0
        // Fill scales: each sub-block scale = 1
        for (int j = 0; j < 8; j++) {
            h_q4k[i].scales[j] = (j < 4) ? 0x01 : 0x00;
        }
        h_q4k[i].scales[4] = 0x01;
        h_q4k[i].scales[5] = 0x01;
        // Fill qs with 0x11 → all values = 1
        memset(h_q4k[i].qs, 0x11, QK_K / 2);
    }
    cudaMemcpy(d_q4k, h_q4k, n_blocks * sizeof(block_q4_K), cudaMemcpyHostToDevice);
    
    // Initialize F16 buffer with pattern 0x3C00 (F16 1.0)
    uint16_t *h_f16 = (uint16_t*)calloc(n_values, sizeof(uint16_t));
    for (uint64_t i = 0; i < n_values; i++) h_f16[i] = 0x3C00;
    cudaMemcpy(d_f16, h_f16, n_values * sizeof(uint16_t), cudaMemcpyHostToDevice);
    
    // Initialize F32 buffer with 1.0f
    float *h_f32 = (float*)calloc(n_values, sizeof(float));
    for (uint64_t i = 0; i < n_values; i++) h_f32[i] = 1.0f;
    cudaMemcpy(d_f32, h_f32, n_values * sizeof(float), cudaMemcpyHostToDevice);
    
    free(h_q4k);
    free(h_f16);
    free(h_f32);
    
    // Setup streams and events
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int n_sms = prop.multiProcessorCount;
    int blocks = n_sms * 4; // 4 blocks per SM for occupancy
    dim3 grid(blocks);
    dim3 block(256);
    
    int iters = 5;
    
    printf("============================================================\n");
    printf("TEST 1: Q4_K Read + Dequant + Accumulate\n");
    printf("============================================================\n");
    // Bytes read per iter: n_blocks * sizeof(block_q4_K) = n_blocks * 144
    double q4k_bytes = (double)n_blocks * sizeof(block_q4_K);
    bench_result r_q4k = run_bench("Q4K-read-dequant", 0, stream, start, stop,
                                    NULL, d_q4k, d_out, n_blocks, iters,
                                    q4k_bytes, grid, block);
    
    printf("\n============================================================\n");
    printf("TEST 2: F16 Read + Accumulate (no dequant)\n");
    printf("============================================================\n");
    double f16_bytes_total = (double)n_values * sizeof(uint16_t);
    bench_result r_f16 = run_bench("F16-read", 0, stream, start, stop,
                                    NULL, d_f16, d_out, n_values, iters,
                                    f16_bytes_total, grid, block);
    
    printf("\n============================================================\n");
    printf("TEST 3: F32 Read + Accumulate (baseline)\n");
    printf("============================================================\n");
    double f32_bytes_total = (double)n_values * sizeof(float);
    bench_result r_f32 = run_bench("F32-read", 0, stream, start, stop,
                                    NULL, d_f32, d_out, n_values, iters,
                                    f32_bytes_total, grid, block);
    
    // ===========================
    // Analysis
    // ===========================
    printf("\n============================================================\n");
    printf("ANALYSIS\n");
    printf("============================================================\n");
    printf("\n");
    printf("Q4_K effective BW: %.0f GB/s [measured: q4_dequant_bench]\n", r_q4k.throughput_gbs);
    printf("F16 effective BW:  %.0f GB/s [measured: q4_dequant_bench]\n", r_f16.throughput_gbs);
    printf("F32 effective BW:  %.0f GB/s [measured: q4_dequant_bench]\n", r_f32.throughput_gbs);
    printf("\n");
    
    // Q4_K reads 144 bytes to produce 256 F16 values (512 bytes)
    // Effective compression ratio: physical_read / logical_values_bytes
    double q4k_compression = (double)sizeof(block_q4_K) / (QK_K * sizeof(uint16_t));
    printf("Q4_K compression ratio: %.3f (%.1f bits/value) [derived: 144B / 512B]\n",
           q4k_compression, sizeof(block_q4_K) * 8.0 / QK_K);
    
    // Compare to theoretical HBM peak (from compute-peak experiment)
    double hbm_peak = 1502.0; // GB/s, [measured: compute_peak_bench]
    printf("\nHBM read peak: %.0f GB/s [measured: compute_peak_bench]\n", hbm_peak);
    printf("\n");
    
    double q4k_efficiency = r_q4k.throughput_gbs / hbm_peak * 100;
    double f16_efficiency = r_f16.throughput_gbs / hbm_peak * 100;
    printf("Q4_K BW utilization: %.1f%% of HBM peak [derived: %.0f / %.0f]\n",
           q4k_efficiency, r_q4k.throughput_gbs, hbm_peak);
    printf("F16 BW utilization:  %.1f%% of HBM peak [derived: %.0f / %.0f]\n",
           f16_efficiency, r_f16.throughput_gbs, hbm_peak);
    
    double dequant_overhead = f16_efficiency / q4k_efficiency;
    printf("\nDequant overhead factor: %.2f× [derived: F16_BW_util / Q4K_BW_util]\n",
           dequant_overhead);
    printf("  Interpretation: Q4_K dequant reduces effective BW by %.0f%% vs F16 read\n",
           (1.0 - 1.0/dequant_overhead) * 100);
    
    // Effective values/sec comparison
    printf("\nEffective values processed:\n");
    printf("  Q4_K: %.3f billion values/s [measured: q4_dequant_bench]\n", r_q4k.throughput_toks);
    printf("  F16:  %.3f billion values/s [measured: q4_dequant_bench]\n", r_f16.throughput_toks);
    printf("  F32:  %.3f billion values/s [measured: q4_dequant_bench]\n", r_f32.throughput_toks);
    
    // Convert to model-relevant metric
    // Model has ~40B active params per token (7 experts × 25.2M params each + attention)
    double active_params = 40e9; // approximate per token
    printf("\nModel-relevant extrapolation:\n");
    double q4k_tokens_per_sec = r_q4k.throughput_toks * 1e9 / active_params;
    double f16_tokens_per_sec = r_f16.throughput_toks * 1e9 / active_params;
    printf("  At %s params/token: Q4K ~%.0f t/s, F16 ~%.0f t/s [derived: values_per_sec / params_per_token]\n",
           "40B", q4k_tokens_per_sec, f16_tokens_per_sec);
    printf("  (Extrapolation assumes perfect HBM BW utilization, no pipeline stalls)\n\n");
    
    // Cleanup
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(d_q4k);
    cudaFree(d_f16);
    cudaFree(d_f32);
    cudaFree(d_out);
    
    printf("=== Done ===\n");
    return 0;
}
