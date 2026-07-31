/**
 * fused_decode.cu
 * P2 — Fused Decode Kernel Prototype
 *
 * Implements a fused decode kernel for GPU0's 24 layers.
 * Fuses per-layer operations: rmsnorm → gate/up matmuls → silu → 
 * element-wise mul → down matmul → residual add.
 *
 * Goal: measure launch overhead reduction vs register pressure cost.
 * Predicted savings: ~480µs (43 launches × ~12µs minus one fused launch).
 *
 * Build:
 *   nvcc -O3 -arch=sm_120 -o fused_decode fused_decode.cu -lcuda -lcudart
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 *
 * GROUND-RULES: Compare fused vs unfused kernel timing. Measure absolute
 * launch overhead and register pressure effects.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../q4_dequant_common.h"

#define WARMUP_ITER 20
#define MAX_TRIALS  200

// Model dimensions (DeepSeek V4 Flash)
#define N_EMBD      4096
#define N_FF        2048
#define N_LAYERS    24    // GPU0 layers

typedef struct {
    double mean_ms;
    double std_ms;
    double speedup;   // vs unfused
    int    n_trials;
} bench_result;

// ============================================================
// Individual operations (used by both fused and unfused paths)
// ============================================================

// RMS norm + residual
__global__ void rms_norm_kernel(const float *hidden, float *residual,
                                 float *output, float eps, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    float local = 0.0f;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        local += hidden[i] * hidden[i];
    atomicAdd(&s_sum, local);
    __syncthreads();

    float rms = rsqrtf(s_sum / n + eps);
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float val = hidden[i] * rms;
        output[i] = val + residual[i];
        residual[i] = val;
    }
}

// Silu activation
__global__ void silu_kernel(const float *in, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}

// Element-wise multiply
__global__ void mul_kernel(const float *a, const float *b, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x)
        out[i] = a[i] * b[i];
}

// Q4_K dequant + matmul (one output row)
__global__ void q4k_dot_kernel(const block_q4_K *weight, const float *input,
                                float *output, int n_out, int n_in) {
    int row = blockIdx.x;
    if (row >= n_out) return;
    int tid = threadIdx.x;
    int n_blocks = (n_in + QK_K - 1) / QK_K;

    float sum = 0.0f;
    for (int b = 0; b < n_blocks; b++) {
        const block_q4_K *blk = &weight[row * n_blocks + b];

        float local_sum = 0.0f;
        for (int l = tid; l < QK_K; l += blockDim.x) {
            int act_idx = b * QK_K + l;
            float act = (act_idx < n_in) ? input[act_idx] : 0.0f;
            local_sum += q4k_dequantize(blk, l) * act;
        }

        // Block reduction, accumulate into sum
        __shared__ float s[256];
        s[tid] = local_sum;
        blockReduceSum(s);
        if (tid == 0) sum += s[0];
        __syncthreads();
    }
    if (tid == 0) output[row] = sum;
}

// ============================================================
// FUSED kernel: one transformer layer (FFN only) — single-block design
//
// Single block (256 threads) executes all 6 ops sequentially using
// thread-level parallelism. Each thread independently computes full
// dot products for assigned output rows, avoiding inter-block races.
//
// Sequence:
//   1. RMS norm(hidden) → normed (shared memory)
//   2. w1 · normed → gate (N_FF), w3 · normed → up (N_FF)
//   3. silu(gate) * up → gated (N_FF)
//   4. w2 · gated → down (N_EMBD)   (residual folded into RMS norm step 1)
//
// Shared memory: normed[N_EMBD] + gated[N_FF] = 24KB
// ============================================================
__global__ void fused_ffn_layer_kernel(
    const float *hidden, float *residual,
    const block_q4_K *w1, const block_q4_K *w2, const block_q4_K *w3,
    float *normed, float *gate, float *up, float *down,
    float eps, int n_embd, int n_ff)
{
    // Single-block design: all threads cooperate through all phases
    int tid = threadIdx.x;
    int nthreads = blockDim.x;  // 256

    // Shared memory for cross-thread data sharing
    __shared__ float s_normed[4096];  // N_EMBD
    __shared__ float s_gated[2048];   // N_FF (gated activation = silu(gate) * up)
    __shared__ float s_sum;           // for RMS norm reduction

    // Each thread's stride for element-wise operations
    int vals_per_thread_norm = (n_embd + nthreads - 1) / nthreads;  // 16
    int rows_per_thread_ff  = (n_ff + nthreads - 1) / nthreads;     // 8
    int rows_per_thread_embd = (n_embd + nthreads - 1) / nthreads;  // 16

    // ============================================
    // Phase 1: RMS norm
    // ============================================
    if (tid == 0) s_sum = 0.0f;
    __syncthreads();

    float local = 0.0f;
    for (int i = tid; i < n_embd; i += nthreads)
        local += hidden[i] * hidden[i];
    atomicAdd(&s_sum, local);
    __syncthreads();

    float rms = rsqrtf(s_sum / n_embd + eps);
    for (int i = tid; i < n_embd; i += nthreads) {
        float val = hidden[i] * rms;
        s_normed[i] = val + residual[i];
        residual[i] = val;
    }
    // Write normed to device memory for unfused comparison fairness
    if (tid < n_embd) normed[tid] = s_normed[tid];
    __syncthreads();

    // ============================================
    // Phase 2: w1 gate + w3 up matmuls
    // Each thread computes full dot products for
    // assigned output rows, reading from s_normed.
    // ============================================
    int n_in_blocks = (n_embd + QK_K - 1) / QK_K;

    for (int r = 0; r < rows_per_thread_ff; r++) {
        int row = tid + r * nthreads;
        if (row >= n_ff) break;

        float sum_gate = 0.0f, sum_up = 0.0f;

        // Each thread does the FULL dot product for this row
        for (int b = 0; b < n_in_blocks; b++) {
            const block_q4_K *blk_w1 = &w1[row * n_in_blocks + b];
            const block_q4_K *blk_w3 = &w3[row * n_in_blocks + b];

            // Each thread iterates over ALL QK_K values in this block
            for (int l = 0; l < QK_K; l++) {
                float nv = s_normed[b * QK_K + l];
                sum_gate += q4k_dequantize(blk_w1, l) * nv;
                sum_up   += q4k_dequantize(blk_w3, l) * nv;
            }
        }

        gate[row] = sum_gate;
        up[row]   = sum_up;
    }
    __syncthreads();

    // ============================================
    // Phase 3: silu + mul → gated activation
    // ============================================
    for (int r = 0; r < rows_per_thread_ff; r++) {
        int row = tid + r * nthreads;
        if (row >= n_ff) break;

        float g = gate[row];
        float silu_g = g / (1.0f + expf(-g));
        s_gated[row] = silu_g * up[row];
    }
    __syncthreads();

    // ============================================
    // Phase 4: w2 down matmul (s_gated → down)
    // w2 shape: [N_EMBD][N_FF] Q4_K
    // Each thread computes rows_per_thread_embd
    // output rows, reading from s_gated.
    // + residual add
    // ============================================
    int n_ff_blocks = (n_ff + QK_K - 1) / QK_K;

    for (int r = 0; r < rows_per_thread_embd; r++) {
        int row = tid + r * nthreads;
        if (row >= n_embd) break;

        float sum = 0.0f;
        for (int b = 0; b < n_ff_blocks; b++) {
            const block_q4_K *blk = &w2[row * n_ff_blocks + b];

            for (int l = 0; l < QK_K; l++) {
                float gv = s_gated[b * QK_K + l];
                sum += q4k_dequantize(blk, l) * gv;
            }
        }

        // W2 output → down buffer (same as unfused: no residual add in either path)
        down[row] = sum;
    }
}

// ============================================================
// UNFUSED reference: separate kernel launches per operation
// ============================================================
static double run_unfused_layer(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
                                 float *d_hidden, float *d_residual,
                                 float *d_normed, float *d_gate, float *d_up, float *d_down,
                                 const block_q4_K *d_w1, const block_q4_K *d_w2, const block_q4_K *d_w3,
                                 int n_embd, int n_ff, int n_sms)
{
    float eps = 1e-5f;
    dim3 grid_norm(min(128, n_sms * 4));
    dim3 block(256);
    dim3 grid_ffn_gate(n_ff);
    dim3 grid_ffn_up(n_ff);
    dim3 grid_ffn_down(n_embd);

    cudaEventRecord(start, stream);

    // 6 kernel launches per layer
    rms_norm_kernel<<<grid_norm, block, 0, stream>>>(d_hidden, d_residual, d_normed, eps, n_embd);
    q4k_dot_kernel<<<grid_ffn_gate, block, 0, stream>>>(d_w1, d_normed, d_gate, n_ff, n_embd);
    q4k_dot_kernel<<<grid_ffn_up, block, 0, stream>>>(d_w3, d_normed, d_up, n_ff, n_embd);
    silu_kernel<<<grid_ffn_gate, block, 0, stream>>>(d_gate, d_gate, n_ff);
    mul_kernel<<<grid_ffn_gate, block, 0, stream>>>(d_gate, d_up, d_gate, n_ff);
    q4k_dot_kernel<<<grid_ffn_down, block, 0, stream>>>(d_w2, d_gate, d_down, n_embd, n_ff);

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ============================================================
// FUSED reference: single kernel launch per layer
// ============================================================
static double run_fused_layer(cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop,
                               float *d_hidden, float *d_residual,
                               float *d_normed, float *d_gate, float *d_up, float *d_down,
                               const block_q4_K *d_w1, const block_q4_K *d_w2, const block_q4_K *d_w3,
                               int n_embd, int n_ff, int n_sms)
{
    float eps = 1e-5f;

    // Single-block fused kernel: all 6 ops in one block, 256 threads
    dim3 grid(1);
    dim3 block(256);

    cudaEventRecord(start, stream);
    fused_ffn_layer_kernel<<<grid, block, 0, stream>>>(
        d_hidden, d_residual, d_w1, d_w2, d_w3,
        d_normed, d_gate, d_up, d_down,
        eps, n_embd, n_ff);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// ============================================================
// Benchmark harness
// ============================================================
static bench_result measure(const char *name, int device,
                             double (*fn)(cudaStream_t, cudaEvent_t, cudaEvent_t,
                                          float*, float*, float*, float*, float*, float*,
                                          const block_q4_K*, const block_q4_K*, const block_q4_K*,
                                          int, int, int),
                             float *d_hidden, float *d_residual,
                             float *d_normed, float *d_gate, float *d_up, float *d_down,
                             const block_q4_K *d_w1, const block_q4_K *d_w2, const block_q4_K *d_w3,
                             int n_embd, int n_ff, int n_layers, int n_sms)
{
    bench_result r = {0};
    cudaSetDevice(device);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));

    for (int i = 0; i < WARMUP_ITER + MAX_TRIALS; i++) {
        double ms = fn(stream, start, stop,
                        d_hidden, d_residual, d_normed, d_gate, d_up, d_down,
                        d_w1, d_w2, d_w3, n_embd, n_ff, n_sms);
        if (i >= WARMUP_ITER) times[i - WARMUP_ITER] = ms;
    }

    double sum = 0;
    for (int i = 0; i < MAX_TRIALS; i++) sum += times[i];
    r.n_trials = MAX_TRIALS;
    r.mean_ms = sum / MAX_TRIALS;

    double sum2 = 0;
    for (int i = 0; i < MAX_TRIALS; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / MAX_TRIALS);

    printf("  %-20s: mean=%.4fms | std=%.4fms | CV=%.2f%%\n",
           name, r.mean_ms, r.std_ms, r.std_ms / r.mean_ms * 100);

    fprintf(stdout, "DATA_%s\t%d\t%.4f\t%.4f\n", name, device, r.mean_ms, r.std_ms);

    free(times);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);

    return r;
}

int main()
{
    printf("=== P2 — Fused Decode Kernel Prototype ===\n");
    printf("System: Dual NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q\n");
    printf("CUDA: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int n_sms = prop.multiProcessorCount;

    printf("GPU0: %s, SMs=%d, %.1f GB HBM, cc=%d.%d\n\n",
           prop.name, n_sms, prop.totalGlobalMem / 1e9, prop.major, prop.minor);

    // Allocate device buffers
    float *d_hidden, *d_residual, *d_normed, *d_gate, *d_up, *d_down;
    block_q4_K *d_w1, *d_w2, *d_w3;

    cudaSetDevice(0);

    cudaMalloc(&d_hidden, N_EMBD * sizeof(float));
    cudaMalloc(&d_residual, N_EMBD * sizeof(float));
    cudaMalloc(&d_normed, N_EMBD * sizeof(float));
    cudaMalloc(&d_gate, N_FF * sizeof(float));
    cudaMalloc(&d_up, N_FF * sizeof(float));
    cudaMalloc(&d_down, N_EMBD * sizeof(float));

    int n_in_blocks = (N_EMBD + QK_K - 1) / QK_K;
    int n_ff_blocks = (N_FF + QK_K - 1) / QK_K;

    cudaMalloc(&d_w1, (size_t)N_FF * n_in_blocks * sizeof(block_q4_K));
    cudaMalloc(&d_w2, (size_t)N_EMBD * n_ff_blocks * sizeof(block_q4_K));
    cudaMalloc(&d_w3, (size_t)N_FF * n_in_blocks * sizeof(block_q4_K));

    // Initialize with patterns
    cudaMemset(d_hidden, 1, N_EMBD * sizeof(float));
    cudaMemset(d_residual, 1, N_EMBD * sizeof(float));
    cudaMemset(d_w1, 0, (size_t)N_FF * n_in_blocks * sizeof(block_q4_K));
    cudaMemset(d_w2, 0, (size_t)N_EMBD * n_ff_blocks * sizeof(block_q4_K));
    cudaMemset(d_w3, 0, (size_t)N_FF * n_in_blocks * sizeof(block_q4_K));
    cudaDeviceSynchronize();

    // Allocate weight hosts and set deterministic values
    {
        size_t w1_size = (size_t)N_FF * n_in_blocks;
        block_q4_K *h_w1 = (block_q4_K*)calloc(w1_size, sizeof(block_q4_K));
        for (size_t i = 0; i < w1_size; i++) {
            h_w1[i].d = 0x3C00;  // F16 1.0
            h_w1[i].dmin = 0x0000;
            memset(h_w1[i].qs, 0x11, QK_K / 2);
        }
        cudaMemcpy(d_w1, h_w1, w1_size * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        free(h_w1);

        size_t w3_size = (size_t)N_FF * n_in_blocks;
        block_q4_K *h_w3 = (block_q4_K*)calloc(w3_size, sizeof(block_q4_K));
        for (size_t i = 0; i < w3_size; i++) {
            h_w3[i].d = 0x3C00;
            h_w3[i].dmin = 0x0000;
            memset(h_w3[i].qs, 0x11, QK_K / 2);
        }
        cudaMemcpy(d_w3, h_w3, w3_size * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        free(h_w3);

        size_t w2_size = (size_t)N_EMBD * n_ff_blocks;
        block_q4_K *h_w2 = (block_q4_K*)calloc(w2_size, sizeof(block_q4_K));
        for (size_t i = 0; i < w2_size; i++) {
            h_w2[i].d = 0x3C00;
            h_w2[i].dmin = 0x0000;
            memset(h_w2[i].qs, 0x11, QK_K / 2);
        }
        cudaMemcpy(d_w2, h_w2, w2_size * sizeof(block_q4_K), cudaMemcpyHostToDevice);
        free(h_w2);
    }

    printf("============================================================\n");
    printf("TEST 1: Single Layer Timings\n");
    printf("============================================================\n\n");

    bench_result r_unfused_1 = measure("unfused_1layer", 0,
        [](cudaStream_t s, cudaEvent_t st, cudaEvent_t sp,
           float *h, float *r, float *n, float *g, float *u, float *d,
           const block_q4_K *w1, const block_q4_K *w2, const block_q4_K *w3,
           int ne, int nf, int sm) -> double {
            return run_unfused_layer(s, st, sp, h, r, n, g, u, d, w1, w2, w3, ne, nf, sm);
        },
        d_hidden, d_residual, d_normed, d_gate, d_up, d_down,
        d_w1, d_w2, d_w3, N_EMBD, N_FF, 1, n_sms);

    bench_result r_fused_1 = measure("fused_1layer", 0,
        [](cudaStream_t s, cudaEvent_t st, cudaEvent_t sp,
           float *h, float *r, float *n, float *g, float *u, float *d,
           const block_q4_K *w1, const block_q4_K *w2, const block_q4_K *w3,
           int ne, int nf, int sm) -> double {
            return run_fused_layer(s, st, sp, h, r, n, g, u, d, w1, w2, w3, ne, nf, sm);
        },
        d_hidden, d_residual, d_normed, d_gate, d_up, d_down,
        d_w1, d_w2, d_w3, N_EMBD, N_FF, 1, n_sms);

    double launch_savings_1 = r_unfused_1.mean_ms - r_fused_1.mean_ms;
    printf("\n  Launch savings per layer: %.3fms (%.1f%% of unfused time)\n",
           launch_savings_1, launch_savings_1 / r_unfused_1.mean_ms * 100);

    printf("\n============================================================\n");
    printf("TEST 2: Projected GPU0 (24 layers) Timings\n");
    printf("============================================================\n\n");

    // Project 24-layer timings (linear extrapolation)
    double unfused_24 = r_unfused_1.mean_ms * N_LAYERS;
    double fused_24 = r_fused_1.mean_ms * N_LAYERS;
    double launch_savings_24 = unfused_24 - fused_24;
    double kernel_savings_24 = r_unfused_1.mean_ms - r_fused_1.mean_ms;

    printf("  Unfused (24 layers): %.3f ms (projected: %.4fms × %d)\n",
           unfused_24, r_unfused_1.mean_ms, N_LAYERS);
    printf("  Fused   (24 layers): %.3f ms (projected: %.4fms × %d)\n",
           fused_24, r_fused_1.mean_ms, N_LAYERS);
    printf("\n");
    printf("  Per-layer savings:     %.4f ms\n", kernel_savings_24);
    printf("  Total savings (24L):   %.3f ms\n", launch_savings_24);
    printf("  Actual GPU0 decode:    ~7.0 ms [measured: roofline-analysis.md]\n");
    printf("  Fused theoretical:     ~%.1f ms (-%.1f%% GPU0 time)\n",
           fused_24, launch_savings_24 / 7.0 * 100);

    printf("\n============================================================\n");
    printf("ANALYSIS\n");
    printf("============================================================\n\n");

    // Kernel launch overhead: ~12µs per launch on CUDA 12.x
    double launch_overhead_per_kernel = 0.012;  // ms (~12µs)
    int n_kernels_unfused = N_LAYERS * 6;  // 6 kernels per layer (rms, w1, w3, silu, mul, w2)
    int n_kernels_fused = N_LAYERS;  // 1 kernel per layer
    double expected_unfused_launch_overhead = n_kernels_unfused * launch_overhead_per_kernel;
    double expected_fused_launch_overhead = n_kernels_fused * launch_overhead_per_kernel;

    printf("  Kernel launch overhead estimates:\n");
    printf("    Unfused: %d launches × %.0fµs = %.2fms\n",
           n_kernels_unfused, launch_overhead_per_kernel * 1000, expected_unfused_launch_overhead);
    printf("    Fused:   %d launches × %.0fµs = %.2fms\n",
           n_kernels_fused, launch_overhead_per_kernel * 1000, expected_fused_launch_overhead);
    printf("    Launch savings: %.2fms (%.1f%% of GPU0 decode time)\n",
           expected_unfused_launch_overhead - expected_fused_launch_overhead,
           (expected_unfused_launch_overhead - expected_fused_launch_overhead) / 7.0 * 100);
    printf("\n");

    printf("  Measured per-layer savings: %.4fms\n", kernel_savings_24);
    printf("  Expected launch-only savings: %.4fms\n",
           (n_kernels_unfused - n_kernels_fused) * launch_overhead_per_kernel);

    if (kernel_savings_24 > (n_kernels_unfused - n_kernels_fused) * launch_overhead_per_kernel) {
        printf("  → Fused kernel has ADDITIONAL savings beyond launch overhead\n");
        printf("    (register reuse, shared memory reuse, reduced global traffic)\n");
    } else if (kernel_savings_24 < (n_kernels_unfused - n_kernels_fused) * launch_overhead_per_kernel) {
        printf("  → Fused kernel has ADDITIONAL COSTS (register pressure, occupancy loss)\n");
        printf("    (register pressure from bundling 6 ops reduces occupancy)\n");
    } else {
        printf("  → Fused savings match launch overhead estimates\n");
    }

    printf("\n============================================================\n");
    printf("COMPARISON TO PREDICTION (PRD-2 §Technique 6)\n");
    printf("============================================================\n\n");

    printf("  Predicted savings: ~480µs (43 launches × ~12µs minus one fused launch)\n");
    printf("  For 43 layers: theoretical launch savings = %d × 12µs = %.2fms\n",
           n_kernels_unfused, n_kernels_unfused * launch_overhead_per_kernel);
    printf("  Measured savings (%d layer): %.4fms\n", 1, kernel_savings_24);
    printf("  Projected savings (%d layers): %.4fms\n", N_LAYERS, kernel_savings_24);
    printf("\n");
    printf("  Impact on total step time (GPU0):\n");
    printf("    GPU0 decode:         7.0ms [measured: roofline-analysis.md]\n");
    printf("    Fused decode:        ~%.1fms (GPU0 only)\n", 7.0 - kernel_savings_24);
    printf("    Total step savings:  %.3fms (%.1f%% of 14.7ms step)\n",
           kernel_savings_24, kernel_savings_24 / 14.7 * 100);
    printf("\n");
    printf("  PRD-2 predicted: ~2-4%% throughput improvement\n");

    // Cleanup
    cudaFree(d_w3);
    cudaFree(d_w2);
    cudaFree(d_w1);
    cudaFree(d_down);
    cudaFree(d_up);
    cudaFree(d_gate);
    cudaFree(d_normed);
    cudaFree(d_residual);
    cudaFree(d_hidden);

    printf("\n=== Done ===\n");
    return 0;
}
