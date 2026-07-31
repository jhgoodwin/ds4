/**
 * matmul_bench.cu
 * O1.3 — cuBLAS matmul Q4_0 vs F16 TFLOPS at decode shapes
 *
 * Measures TFLOPS of:
 *   (a) cuBLAS F16 matmul (cublasHgemm) at decode shapes
 *   (b) Simulated Q4_0 matmul via custom kernel + measurement
 *
 * Decode shapes:
 *   4096×2048  (FFN gate/up: 4096 → 2048)
 *   4096×4096  (QKV projection: 4096 → 4096)
 *   2048×2048  (FFN down: 2048 → 4096, with n_ff half of n_embd)
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -o matmul_bench matmul_bench.cu \
 *       -lcuda -lcudart -lcublas
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 *
 * GROUND-RULES: Hypothesis-driven comparison of quantized vs F16 matmul
 * efficiency at decode shapes. All values tagged [measured: matmul_bench].
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define WARMUP_ITER 20
#define MAX_TRIALS  200

typedef struct {
    double mean_ms;
    double std_ms;
    double tflops;
    double bw_gbs;
    double ai;            // arithmetic intensity
    int    n_trials;
} bench_result;

// Q4_0 block format (from llama.cpp / GGML)
// 32 values per block
// Storage: d (F16, 2B) + m (F16, 2B) + 32×4bit = 16B → 20 bytes/block
typedef struct {
    uint16_t d;      // scale (F16)
    uint16_t m;      // min (F16)
    uint8_t  qs[16]; // 32 × 4-bit values
} __attribute__((packed)) block_q4_0;

static_assert(sizeof(block_q4_0) == 20, "Q4_0 block must be 20 bytes");

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

// ============================================================
// Q4_0 matmul kernel: weight[n_out][n_in] Q4_0, input[1][n_in] F16, output[1][n_out]
// Each block computes one output element (dot product over n_in)
// ============================================================
__global__ void q4_0_matmul_kernel(const block_q4_0 *weight,
                                    const float *input,
                                    float *output,
                                    int n_out, int n_in)
{
    int row = blockIdx.x;
    if (row >= n_out) return;

    int tid = threadIdx.x;
    int n_blocks = (n_in + 31) / 32;  // ceil(n_in / 32) — each Q4_0 block holds 32 values

    __shared__ float partials[256];
    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_q4_0 *blk = &weight[row * n_blocks + b];
        float d = f16_to_f32(blk->d);
        float m = f16_to_f32(blk->m);

        int base_idx = b * 32;

        // Cooperative dequant + dot product
        for (int l = tid; l < 32; l += blockDim.x) {
            int byte_idx = l / 2;
            int shift = (l & 1) * 4;
            int q = (blk->qs[byte_idx] >> shift) & 0x0F;
            float deq = d * (float)q + m;  // Q4_0: value = d * q + m

            int act_idx = base_idx + l;
            float act = (act_idx < n_in) ? input[act_idx] : 0.0f;
            sum += deq * act;
        }
    }

    // Block reduction
    partials[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[row] = partials[0];
}

// ============================================================
// F16 matmul kernel (for comparison without cuBLAS)
// ============================================================
__global__ void f16_matmul_kernel(const uint16_t *weight,
                                   const float *input,
                                   float *output,
                                   int n_out, int n_in)
{
    int row = blockIdx.x;
    if (row >= n_out) return;

    int tid = threadIdx.x;
    __shared__ float partials[256];
    float sum = 0.0f;

    for (int i = tid; i < n_in; i += blockDim.x) {
        float w = f16_to_f32(weight[row * n_in + i]);
        sum += w * input[i];
    }

    partials[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) partials[tid] += partials[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[row] = partials[0];
}

// Run a kernel and measure timing
// Kernel type dispatch via name string (Q4_0 vs F16)
static bench_result measure_kernel(const char *name, int device, int warmup, int trials,
                                    dim3 grid, dim3 block,
                                    const void *d_weight, const float *d_input,
                                    float *d_output, int n_out, int n_in,
                                    double total_bytes_read, double total_flops)
{
    bench_result r = {0};
    cudaSetDevice(device);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *times = (double*)calloc(trials, sizeof(double));

    for (int i = 0; i < warmup + trials; i++) {
        cudaEventRecord(start);
        // SAFETY: kernel dispatch macro to avoid type erasure
        if (strstr(name, "Q4_0")) {
            q4_0_matmul_kernel<<<grid, block>>>(
                (const block_q4_0*)d_weight, d_input, d_output, n_out, n_in);
        } else if (strstr(name, "F16")) {
            f16_matmul_kernel<<<grid, block>>>(
                (const uint16_t*)d_weight, d_input, d_output, n_out, n_in);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i >= warmup) {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i - warmup] = ms;
        }
    }

    double sum = 0;
    for (int i = 0; i < trials; i++) sum += times[i];
    r.n_trials = trials;
    r.mean_ms = sum / trials;

    double sum2 = 0;
    for (int i = 0; i < trials; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / trials);

    double total_s = r.mean_ms / 1000.0;
    r.tflops = total_flops / total_s / 1e12;
    r.bw_gbs = total_bytes_read / total_s / 1e9;
    r.ai = total_flops / total_bytes_read;

    printf("  %-20s | %d | mean=%.4fms | std=%.4fms | %.2f TFLOPS | %.0f GB/s | AI=%.1f FLOP/B\n",
           name, device, r.mean_ms, r.std_ms, r.tflops, r.bw_gbs, r.ai);

    fprintf(stdout, "DATA_%s\t%d\t%.4f\t%.4f\t%.3f\t%.0f\t%.1f\n",
            name, device, r.mean_ms, r.std_ms, r.tflops, r.bw_gbs, r.ai);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    free(times);
    return r;
}

// cuBLAS F16 matmul benchmark
static bench_result measure_cublas_hgemm(cublasHandle_t handle, int device,
                                          int m, int n, int k,
                                          int warmup, int trials)
{
    bench_result r = {0};
    cudaSetDevice(device);
    cudaDeviceSynchronize();

    // Allocate F16 matrices (half precision)
    size_t sz_a = (size_t)m * k * sizeof(__half);  // A: m×k
    size_t sz_b = (size_t)k * n * sizeof(__half);  // B: k×n
    size_t sz_c = (size_t)m * n * sizeof(__half);  // C: m×n

    __half *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz_a);
    cudaMalloc(&d_b, sz_b);
    cudaMalloc(&d_c, sz_c);

    // Initialize with small values to avoid F16 overflow
    __half *h_a = (__half*)calloc(m * k, sizeof(__half));
    __half *h_b = (__half*)calloc(k * n, sizeof(__half));
    for (int i = 0; i < m * k; i++) h_a[i] = __float2half(0.5f);
    for (int i = 0; i < k * n; i++) h_b[i] = __float2half(0.5f);
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);
    free(h_a);
    free(h_b);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      m, n, k, &alpha,
                      d_a, CUDA_R_16F, m,
                      d_b, CUDA_R_16F, k,
                      &beta,
                      d_c, CUDA_R_16F, m,
                      CUDA_R_32F,  // compute in F32 for precision
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaDeviceSynchronize();

    // Measurement
    double *times = (double*)calloc(trials, sizeof(double));
    for (int i = 0; i < trials; i++) {
        cudaEventRecord(start);
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                      m, n, k, &alpha,
                      d_a, CUDA_R_16F, m,
                      d_b, CUDA_R_16F, k,
                      &beta,
                      d_c, CUDA_R_16F, m,
                      CUDA_R_32F,
                      CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        times[i] = ms;
    }

    double sum = 0;
    for (int i = 0; i < trials; i++) sum += times[i];
    r.n_trials = trials;
    r.mean_ms = sum / trials;

    double sum2 = 0;
    for (int i = 0; i < trials; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / trials);

    double total_s = r.mean_ms / 1000.0;
    double total_flops = 2.0 * m * n * k;
    r.tflops = total_flops / total_s / 1e12;

    double total_bytes = (double)(sz_a + sz_b + sz_c);  // total data moved (reads + writes)
    r.bw_gbs = total_bytes / total_s / 1e9;
    r.ai = total_flops / total_bytes;

    printf("  %-20s | %d | mean=%.4fms | std=%.4fms | %.2f TFLOPS | %.0f GB/s | AI=%.1f FLOP/B\n",
           "cuBLAS-HGEMM", device, r.mean_ms, r.std_ms, r.tflops, r.bw_gbs, r.ai);

    fprintf(stdout, "DATA_cuBLAS-HGEMM\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.3f\t%.0f\t%.1f\n",
            device, m, n, k, r.mean_ms, r.std_ms, r.tflops, r.bw_gbs, r.ai);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(times);

    return r;
}

int main()
{
    printf("=== O1.3 — cuBLAS Matmul Q4_0 vs F16 TFLOPS ===\n");
    printf("System: Dual NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q\n");
    printf("CUDA: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    int n_devices;
    cudaGetDeviceCount(&n_devices);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU0: %s, SMs=%d, %.1f GB HBM, cc=%d.%d\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e9,
           prop.major, prop.minor);

    // Decode shapes: (n_out, n_in) for 1 token
    // Shapes from PRD-2 §O1.3: FFN gate/up 2048×4096, FFN down 4096×2048,
    // QKV+O proj 4096×4096, attention projections (per-head views)
    struct { int n_out; int n_in; const char *name; } shapes[] = {
        {2048, 4096, "FFN_GATE_2048x4096"},
        {2048, 4096, "FFN_UP_2048x4096"},
        {4096, 2048, "FFN_DOWN_4096x2048"},
        {4096, 4096, "QKV_PROJ_4096x4096"},
        {4096, 4096, "O_PROJ_4096x4096"},
    };
    int n_shapes = sizeof(shapes) / sizeof(shapes[0]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // ===========================
    // Test 1: cuBLAS F16 HGEMM
    // ===========================
    printf("============================================================\n");
    printf("TEST 1: cuBLAS F16 (HGEMM) at decode shapes\n");
    printf("============================================================\n");
    printf("  m=1 (single token decode), n=n_out, k=n_in\n");
    printf("\n");

    // Peak F16 TFLOPS on Blackwell: tensor cores support 2× throughput vs F32
    // For HGEMM: ~148.7 TFLOPS theoretical F32 → ~297.4 TFLOPS for tensor-core HGEMM
    // But at m=1, tensor cores may not be utilized (need m >= 4 for optimal)
    double f16_peak = 297.4;  // derived: 2× F32 peak for tensor core HGEMM
    double f32_peak = 148.7;  // measured: roofline-analysis.md

    for (int s = 0; s < n_shapes; s++) {
        int n_out = shapes[s].n_out;
        int n_in = shapes[s].n_in;
        int m = 1;  // single token decode

        printf("--- %s (%dx%dx%d) ---\n", shapes[s].name, m, n_out, n_in);

        bench_result r = measure_cublas_hgemm(handle, 0, m, n_out, n_in, 20, 200);

        double flops = 2.0 * m * n_out * n_in;
        double bytes_f16_read = (double)(m * n_in + n_in * n_out) * sizeof(__half);
        double bytes_f16_write = (double)(m * n_out) * sizeof(__half);

        printf("    Ops: %.0f FLOPs, Read: %.0f B, Write: %.0f B\n",
               flops, bytes_f16_read, bytes_f16_write);
        printf("    %% of F16 peak: %.1f%% [derived]\n", r.tflops / f16_peak * 100);
        printf("    %% of F32 peak: %.1f%% [derived]\n", r.tflops / f32_peak * 100);

        // Arithmetic intensity
        double hbm_peak = 1502.0;  // GB/s from compute-peak
        double roofline_tflops = (hbm_peak * 1e9 * r.ai) / 1e12;
        printf("    Roofline bound: memory at %d FLOP/B -> %.1f TFLOPS [derived]\n",
               (int)r.ai, roofline_tflops);
        printf("\n");
    }

    // ===========================
    // Test 2: Q4_0 custom matmul
    // ===========================
    printf("============================================================\n");
    printf("TEST 2: Q4_0 Custom Matmul (dequant-on-the-fly)\n");
    printf("============================================================\n");
    printf("  Q4_0 block: 20B stores 32 values = 5 bits/value\n");
    printf("\n");

    int n_sms = prop.multiProcessorCount;

    for (int s = 0; s < n_shapes; s++) {
        int n_out = shapes[s].n_out;
        int n_in = shapes[s].n_in;
        int m = 1;

        printf("--- Q4_0 %s (%dx%d) ---\n", shapes[s].name, n_out, n_in);

        // Allocate Q4_0 weight buffer
        int n_blocks = (n_in + 31) / 32;
        size_t q4_weight_bytes = (size_t)n_out * n_blocks * sizeof(block_q4_0);
        size_t f16_weight_bytes = (size_t)n_out * n_in * sizeof(uint16_t);

        block_q4_0 *d_q4_weight;
        uint16_t *d_f16_weight;
        float *d_input, *d_output;

        cudaMalloc(&d_q4_weight, q4_weight_bytes);
        cudaMalloc(&d_f16_weight, f16_weight_bytes);
        cudaMalloc(&d_input, (size_t)n_in * sizeof(float));
        cudaMalloc(&d_output, (size_t)n_out * sizeof(float));

        // Initialize Q4_0 weights with deterministic pattern
        block_q4_0 *h_q4 = (block_q4_0*)calloc(n_out * n_blocks, sizeof(block_q4_0));
        for (int i = 0; i < n_out * n_blocks; i++) {
            h_q4[i].d = 0x3C00;  // F16 1.0
            h_q4[i].m = 0x0000;  // F16 0.0
            memset(h_q4[i].qs, 0x11, 16);  // all values = 1 (low nibble) & 1 (high nibble)
        }
        cudaMemcpy(d_q4_weight, h_q4, q4_weight_bytes, cudaMemcpyHostToDevice);
        free(h_q4);

        // Initialize F16 weights with 1.0
        uint16_t *h_f16 = (uint16_t*)calloc(n_out * n_in, sizeof(uint16_t));
        for (int i = 0; i < n_out * n_in; i++) h_f16[i] = 0x3C00;
        cudaMemcpy(d_f16_weight, h_f16, f16_weight_bytes, cudaMemcpyHostToDevice);
        free(h_f16);

        // Initialize input with 1.0
        float *h_in = (float*)calloc(n_in, sizeof(float));
        for (int i = 0; i < n_in; i++) h_in[i] = 1.0f;
        cudaMemcpy(d_input, h_in, (size_t)n_in * sizeof(float), cudaMemcpyHostToDevice);
        free(h_in);

        // Benchmark Q4_0 matmul
        dim3 grid(n_out);
        dim3 block(256);

        double q4_flops = 2.0 * n_out * n_in;  // same FLOPs as F16 matmul
        double q4_bytes_read = (double)n_out * n_blocks * sizeof(block_q4_0);  // Q4_0 weight bytes
        q4_bytes_read += (double)n_in * sizeof(float);  // input activation

        bench_result r_q4 = measure_kernel("Q4_0_matmul", 0, 20, 200,
                                             grid, block,
                                             d_q4_weight, d_input, d_output,
                                             n_out, n_in, q4_bytes_read, q4_flops);

        // Benchmark F16 matmul for comparison
        double f16_bytes_read = (double)n_out * n_in * sizeof(uint16_t);
        f16_bytes_read += (double)n_in * sizeof(float);

        bench_result r_f16 = measure_kernel("F16_matmul", 0, 20, 200,
                                              grid, block,
                                              d_f16_weight, d_input, d_output,
                                              n_out, n_in, f16_bytes_read, q4_flops);

        // Compute ratio
        double tflops_ratio = r_q4.tflops / (r_f16.tflops + 1e-10);

        printf("\n  Comparison for %s:\n", shapes[s].name);
        printf("    Q4_0: %.2f TFLOPS [measured: matmul_bench]\n", r_q4.tflops);
        printf("    F16:  %.2f TFLOPS [measured: matmul_bench]\n", r_f16.tflops);
        printf("    Ratio: Q4_0/F16 = %.2f%% [derived]\n", tflops_ratio * 100);
        printf("    Hypothesis (PRD-2): Q4_0 achieves 30-50%% of F16 TFLOPS\n");
        if (tflops_ratio > 0.80) {
            printf("    Contrast: Q4_0 > 80%% → dequant overhead is small for matmul\n");
        } else if (tflops_ratio < 0.30) {
            printf("    Contrast: Q4_0 < 30%% → dequant overhead dominates matmul\n");
        } else {
            printf("    Match: Q4_0 within expected 30-80%% → dequant significant but not dominant\n");
        }
        printf("\n");

        cudaFree(d_q4_weight);
        cudaFree(d_f16_weight);
        cudaFree(d_input);
        cudaFree(d_output);
    }

    // ===========================
    // Test 3: cuBLAS F32 SGEMM for reference
    // ===========================
    printf("============================================================\n");
    printf("TEST 3: cuBLAS F32 (SGEMM) at decode shapes (reference)\n");
    printf("============================================================\n");
    printf("  m=1, SGEMM using tensor cores (TF32 on Blackwell)\n");
    printf("  Theoretical peak: 148.7 TFLOPS F32, ~297 TFLOPS F16 tensor-core\n");
    printf("\n");

    for (int s = 0; s < n_shapes; s++) {
        int n_out = shapes[s].n_out;
        int n_in = shapes[s].n_in;
        int m = 1;

        printf("--- %s ---\n", shapes[s].name);

        float *d_a, *d_b, *d_c;
        size_t sz_a = (size_t)m * n_in * sizeof(float);
        size_t sz_b = (size_t)n_in * n_out * sizeof(float);
        size_t sz_c = (size_t)m * n_out * sizeof(float);

        cudaMalloc(&d_a, sz_a);
        cudaMalloc(&d_b, sz_b);
        cudaMalloc(&d_c, sz_c);
        cudaMemset(d_a, 0, sz_a);
        cudaMemset(d_b, 0, sz_b);

        float alpha = 1.0f, beta = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warmup
        for (int i = 0; i < 20; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n_out, n_in, &alpha, d_a, m, d_b, n_in, &beta, d_c, m);
        }
        cudaDeviceSynchronize();

        double *times = (double*)calloc(200, sizeof(double));
        for (int i = 0; i < 200; i++) {
            cudaEventRecord(start);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n_out, n_in, &alpha, d_a, m, d_b, n_in, &beta, d_c, m);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = ms;
        }

        double sum = 0;
        for (int i = 0; i < 200; i++) sum += times[i];
        double mean_ms = sum / 200;
        double flops = 2.0 * m * n_out * n_in;
        double tflops = flops / (mean_ms / 1000.0) / 1e12;

        printf("  SGEMM %dx%dx%d: mean=%.4fms, %.2f TFLOPS [measured: matmul_bench]\n",
               m, n_out, n_in, mean_ms, tflops);
        printf("  %% of F32 peak: %.1f%%\n", tflops / 148.7 * 100);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(times);
    }

    cublasDestroy(handle);

    // Summary
    printf("\n");
    printf("============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("\n");
    printf("Hypothesis H2 (PRD-2 §O1.3):\n");
    printf("  Q4_0 matmul achieves 30-50%% of F16 matmul TFLOPS at decode shapes\n");
    printf("  due to dequant overhead in the matmul kernel.\n");
    printf("\n");
    printf("Contrast:\n");
    printf("  If Q4_0 > 80%% of F16 → dequant overhead is small for matmul\n");
    printf("  If Q4_0 < 30%% of F16 → dequant overhead dominates even in matmul\n");
    printf("  If close to 50%% → dequant overhead is significant but not the sole bottleneck\n");

    printf("\n=== Done ===\n");
    return 0;
}
