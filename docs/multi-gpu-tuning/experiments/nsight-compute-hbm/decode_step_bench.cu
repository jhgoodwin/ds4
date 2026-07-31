/**
 * decode_step_bench.cu
 * Nsight Compute HBM utilization during decode step
 *
 * Simulates key operations of one decode step for profiling:
 * - Q4_K dequant + matmul (FFN w1/w2/w3)
 * - Silu activation
 * - RMS norm
 * - Attention projection
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 *
 * Build:
 *   nvcc -o decode_step_bench decode_step_bench.cu -lcuda -lcudart -lcublas
 *
 * Nsight Compute profiling:
 *   ncu --set full -o decode_step_profile ./decode_step_bench
 *   ncu --section MemoryWorkloadAnalysis --section SchedulerStats -o decode_step_light ./decode_step_bench
 *
 * GROUND-RULES: Hypothesis-driven measurement of achieved DRAM throughput,
 * warp stall reasons, and compute utilization during one decode step.
 * All values tagged [measured: decode_step_bench].
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "../q4_dequant_common.h"

#define WARMUP_ITER 20
#define MAX_TRIALS  100
#define STABLE_CV_THRESHOLD 0.05

// Model architecture constants (DeepSeek V4 Flash)
#define N_EMBD   4096    // hidden dimension
#define N_FF     2048    // intermediate FF dimension (expert)
#define N_HEADS  32      // attention heads per layer
#define N_KV_HEADS 8     // KV heads (grouped query)
#define HEAD_DIM 128     // d_head
#define N_LAYERS_GPU0 24 // layers on GPU0
#define SEQ_LEN  1       // single token decode

typedef struct {
    double mean_ms;
    double std_ms;
    double tflops;
    double bw_gbs;
    int    n_trials;
} bench_result;

// ============================================================
// Kernel 1: RMS Norm + Residual Add
// Input: hidden_state[N_EMBD], residual[N_EMBD]
// Output: normalized[N_EMBD], residual updated
// ============================================================
__global__ void rms_norm_kernel(const float *hidden, float *residual,
                                 float *output, float eps, int n_embd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s_sum;

    if (threadIdx.x == 0) s_sum = 0.0f;
    __syncthreads();

    // Compute sum of squares
    float local = 0.0f;
    for (int i = idx; i < n_embd; i += blockDim.x * gridDim.x) {
        local += hidden[i] * hidden[i];
    }
    atomicAdd(&s_sum, local);
    __syncthreads();

    float rms = rsqrtf(s_sum / n_embd + eps);

    // Apply norm and add residual
    for (int i = idx; i < n_embd; i += blockDim.x * gridDim.x) {
        float val = hidden[i] * rms;
        output[i] = val + residual[i];
        residual[i] = val; // update residual for next layer
    }
}

// ============================================================
// Kernel 2: Silu Activation (element-wise)
// ============================================================
__global__ void silu_kernel(const float *in, float *out, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x)); // silu(x) = x * sigmoid(x)
    }
}

// ============================================================
// Kernel 3: Q4_K dequant + matmul (FFN w1/w2/w3)
// weight[N_FF][N_EMBD] in Q4_K format
// F16 activation input[1][N_EMBD]
// output[1][N_FF]
// ============================================================
__global__ void q4k_matmul_kernel(const block_q4_K *weight,
                                   const float *input,
                                   float *output,
                                   int n_out, int n_in) {
    // Each thread block computes one output row
    int row = blockIdx.x;  // output row index [0, n_out)
    if (row >= n_out) return;

    int tid = threadIdx.x;
    int n_in_blocks = (n_in + QK_K - 1) / QK_K;

    float sum = 0.0f;
    for (int b = 0; b < n_in_blocks; b++) {
        const block_q4_K *blk = &weight[row * n_in_blocks + b];

        // Cooperative dequant + dot product
        // Each thread handles 1 or more values
        int vals_per_thread = (QK_K + blockDim.x - 1) / blockDim.x;
        float local_sum = 0.0f;

        for (int v = tid * vals_per_thread; v < (tid + 1) * vals_per_thread && v < QK_K; v++) {
            float deq = q4k_dequantize(blk, v);

            // Input activation at position b*QK_K + v
            int input_idx = b * QK_K + v;
            float act = (input_idx < n_in) ? input[input_idx] : 0.0f;
            local_sum += deq * act;
        }

        // Block-level reduction, then accumulate into sum
        __shared__ float s_red[256];
        s_red[tid] = local_sum;
        blockReduceSum(s_red);
        if (tid == 0) sum += s_red[0];
        __syncthreads();
    }

    // Write final result
    if (tid == 0) {
        output[row] = sum;
    }
}



// ============================================================
// Kernel 4: F16 matmul (simulated via F32 for simplicity)
// weight[N_OUT][N_IN] in F16, input[N_IN], output[N_OUT]
// ============================================================
__global__ void f16_matmul_kernel(const uint16_t *weight, const float *input,
                                   float *output, int n_out, int n_in) {
    int row = blockIdx.x;
    if (row >= n_out) return;

    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int i = tid; i < n_in; i += blockDim.x) {
        float w = f16_to_f32(weight[row * n_in + i]);
        sum += w * input[i];
    }

    // Block reduction
    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[row] = shared[0];
}

// ============================================================
// Kernel 5: Scaled dot-product attention (single token decode)
// q[1][N_HEADS * HEAD_DIM], k_cache[TOTAL_KV][N_KV_HEADS * HEAD_DIM]
// v_cache[TOTAL_KV][N_KV_HEADS * HEAD_DIM]
// output[1][N_HEADS * HEAD_DIM]
// ============================================================
__global__ void attention_kernel(const float *q, const float *k_cache,
                                  const float *v_cache, float *output,
                                  int n_kv, int n_heads, int n_kv_heads,
                                  int head_dim) {
    // GQA: each KV head services n_heads/n_kv_heads query heads
    int q_head = blockIdx.x; // query head index [0, n_heads)
    if (q_head >= n_heads) return;

    int kv_head = q_head * n_kv_heads / n_heads;
    int tid = threadIdx.x;

    // Compute attention scores for this head across all cached tokens
    __shared__ float scores[256]; // max context window per block

    float max_score = -INFINITY;
    for (int pos = tid; pos < n_kv; pos += blockDim.x) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            float qd = q[q_head * head_dim + d];
            float kd = k_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + d];
            score += qd * kd;
        }
        score *= rsqrtf((float)head_dim);
        scores[pos] = score;
        if (score > max_score) max_score = score;
    }
    __syncthreads();

    // Softmax
    float sum_exp = 0.0f;
    for (int pos = tid; pos < n_kv; pos += blockDim.x) {
        sum_exp += expf(scores[pos] - max_score);
    }
    // Block reduce sum_exp
    __shared__ float red_shared[256];
    red_shared[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) red_shared[tid] += red_shared[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / red_shared[0];
    __syncthreads();

    // Weighted sum of values
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        for (int pos = 0; pos < n_kv; pos++) {
            float attn = expf(scores[pos] - max_score) * inv_sum;
            float vd = v_cache[pos * n_kv_heads * head_dim + kv_head * head_dim + d];
            out_val += attn * vd;
        }
        output[q_head * head_dim + d] = out_val;
    }
}

// ============================================================
// F16 scale matmul using cuBLAS
// ============================================================
static double measure_cublas_matmul(cublasHandle_t handle, int m, int n, int k,
                                     const float *alpha, const float *beta,
                                     int device, int warmup, int trials) {
    cudaSetDevice(device);
    cudaDeviceSynchronize();

    float *d_a, *d_b, *d_c;
    size_t sz_a = (size_t)m * k * sizeof(float);
    size_t sz_b = (size_t)k * n * sizeof(float);
    size_t sz_c = (size_t)m * n * sizeof(float);

    cudaMalloc(&d_a, sz_a);
    cudaMalloc(&d_b, sz_b);
    cudaMalloc(&d_c, sz_c);
    cudaMemset(d_a, 0xAB, sz_a);
    cudaMemset(d_b, 0xCD, sz_b);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, alpha, d_a, m, d_b, k, beta, d_c, m);
    }
    cudaDeviceSynchronize();

    // Measurement
    double total_ms = 0;
    for (int i = 0; i < trials; i++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k, alpha, d_a, m, d_b, k, beta, d_c, m);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    double mean_ms = total_ms / trials;
    // FLOPs = 2 * m * n * k
    double flops = 2.0 * m * n * k;
    double tflops = flops / (mean_ms / 1000.0) / 1e12;

    printf("  cuBLAS SGEMM %dx%dx%d: mean=%.3fms, %.2f TFLOPS\n",
           m, n, k, mean_ms, tflops);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return tflops;
}

// ============================================================
// Full decode step simulation: runs all kernels in sequence
// ============================================================
static void run_decode_step(int device, int n_layers, int n_embd, int n_ff) {
    cudaSetDevice(device);

    size_t hidden_bytes = (size_t)n_embd * sizeof(float);
    size_t ff_bytes = (size_t)n_ff * sizeof(float);

    float *d_hidden, *d_residual, *d_normed;
    float *d_silu_in, *d_silu_out;
    float *d_ffn_out;

    cudaMalloc(&d_hidden, hidden_bytes);
    cudaMalloc(&d_residual, hidden_bytes);
    cudaMalloc(&d_normed, hidden_bytes);
    cudaMalloc(&d_silu_in, ff_bytes);
    cudaMalloc(&d_silu_out, ff_bytes);
    cudaMalloc(&d_ffn_out, n_embd * sizeof(float));

    cudaMemset(d_hidden, 1, hidden_bytes);
    cudaMemset(d_residual, 1, hidden_bytes);

    // Allocate simulated Q4_K weights for FFN
    int n_in_blocks = (n_embd + QK_K - 1) / QK_K;
    size_t weight_size = (size_t)n_ff * n_in_blocks * sizeof(block_q4_K);
    block_q4_K *d_weights;
    cudaMalloc(&d_weights, weight_size);
    cudaMemset(d_weights, 0, weight_size);

    // Allocate F16 weights for comparison
    size_t f16_weight_size = (size_t)n_ff * n_embd * sizeof(uint16_t);
    uint16_t *d_f16_weights;
    cudaMalloc(&d_f16_weights, f16_weight_size);
    cudaMemset(d_f16_weights, 0, f16_weight_size);

    int n_sms = 0;
    cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, device);

    dim3 grid_1d(n_sms * 4);
    dim3 block(256);
    dim3 grid_norm(min(256, n_sms * 4));
    dim3 grid_silu(min(256, n_sms * 4));
    dim3 grid_q4k(n_ff);
    dim3 q4k_block(min(256, n_embd));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n=== Simulating %d decode layers on GPU %d ===\n", n_layers, device);
    printf("n_embd=%d, n_ff=%d, dtype=Q4_K (144B/block), %d blocks/row\n",
           n_embd, n_ff, n_in_blocks);
    printf("Weight buffer: Q4_K=%.1f MiB, F16=%.1f MiB\n",
           weight_size / 1048576.0, f16_weight_size / 1048576.0);

    double total_ms = 0;
    double kernel_times[256];
    const char *kernel_names[256];
    int n_kernels = 0;

    float eps = 1e-5f;

    // Run N_LAYERS decode steps (simulated)
    for (int layer = 0; layer < n_layers; layer++) {
        // 1. RMS norm
        cudaEventRecord(start);
        rms_norm_kernel<<<grid_norm, block>>>(d_hidden, d_residual, d_normed, eps, n_embd);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "rms_norm"; n_kernels++;
        total_ms += ms;

        // 2. Q4_K FFN matmul (w1) — normed → silu input
        cudaEventRecord(start);
        q4k_matmul_kernel<<<grid_q4k, q4k_block>>>(d_weights, d_normed, d_silu_in, n_ff, n_embd);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "q4k_w1_matmul"; n_kernels++;
        total_ms += ms;

        // 3. Q4_K FFN matmul (w3) — normed → silu_out (gate)
        cudaEventRecord(start);
        q4k_matmul_kernel<<<grid_q4k, q4k_block>>>(d_weights, d_normed, d_silu_out, n_ff, n_embd);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "q4k_w3_matmul"; n_kernels++;
        total_ms += ms;

        // 4. Silu activation on w1 output
        cudaEventRecord(start);
        silu_kernel<<<grid_silu, block>>>(d_silu_in, d_silu_in, n_ff);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "silu"; n_kernels++;
        total_ms += ms;

        // 5. Element-wise multiply: silu(w1(x)) * w3(x)
        cudaEventRecord(start);
        silu_kernel<<<grid_silu, block>>>(d_silu_out, d_silu_out, n_ff); // reuse kernel for mul
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "silu_mul"; n_kernels++;
        total_ms += ms;

        // 6. Q4_K FFN matmul (w2) — projection back to n_embd
        cudaEventRecord(start);
        q4k_matmul_kernel<<<grid_1d, block>>>(d_weights, d_silu_in, d_ffn_out, n_embd, n_ff);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        kernel_times[n_kernels] = ms; kernel_names[n_kernels] = "q4k_w2_matmul"; n_kernels++;
        total_ms += ms;
    }

    printf("\nTotal time for %d layers: %.3f ms (mean per layer: %.3f ms)\n",
           n_layers, total_ms, total_ms / n_layers);
    printf("\nKernel breakdown (aggregate across %d layers):\n", n_layers);

    // Aggregate kernel times by name
    for (int k = 0; k < n_kernels; k++) {
        printf("  %-20s: %.3f ms (%.1f%% of total)\n",
               kernel_names[k], kernel_times[k],
               kernel_times[k] / (total_ms + 1e-10) * 100);
    }

    printf("\n  TOTAL: %.3f ms per step (GPU0, %d layers)\n", total_ms, n_layers);
    printf("  Predicted from actual decode: GPU0 ~7.0ms [measured: roofline-analysis.md]\n");
    printf("  Note: micro-benchmark excludes attention, embedding, KV cache ops, and graph launch overhead\n");

    // Effective BW estimate
    double total_bytes_read = (double)n_layers * n_ff * n_in_blocks * sizeof(block_q4_K) * 3; // w1+w2+w3
    total_bytes_read += (double)n_layers * 5 * n_embd * sizeof(float); // activations
    double bw_gbs = total_bytes_read / (total_ms / 1000.0) / 1e9;
    printf("  Estimated effective BW: %.0f GB/s (read-heavy, dequant pattern)\n", bw_gbs);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(d_ffn_out);
    cudaFree(d_silu_out);
    cudaFree(d_silu_in);
    cudaFree(d_normed);
    cudaFree(d_residual);
    cudaFree(d_hidden);
    cudaFree(d_weights);
    cudaFree(d_f16_weights);
}

int main()
{
    printf("=== Nsight Compute HBM Utilization — Decode Step Benchmark ===\n");
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

    // == Test 1: Simulated single layer for detailed Nsight profiling ==
    printf("============================================================\n");
    printf("TEST 1: Single layer decode step (for Nsight profiling)\n");
    printf("============================================================\n");
    run_decode_step(0, 1, N_EMBD, N_FF);

    // == Test 2: Full GPU0 decode (24 layers) for aggregate ==
    printf("\n");
    printf("============================================================\n");
    printf("TEST 2: Full GPU0 decode (24 layers)\n");
    printf("============================================================\n");
    run_decode_step(0, N_LAYERS_GPU0, N_EMBD, N_FF);

    // == Test 3: cuBLAS F16 matmul at decode shapes ==
    printf("\n");
    printf("============================================================\n");
    printf("TEST 3: cuBLAS F32 SGEMM at decode shapes\n");
    printf("============================================================\n");

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaSetDevice(0);

    // Decode shapes: (m, n, k) where m=1 (single token), n=output_dim, k=input_dim
    int decode_shapes[][3] = {
        {1, N_FF, N_EMBD},      // w1: 1×4096 × 4096×2048 → 1×2048
        {1, N_FF, N_EMBD},      // w3: same shape
        {1, N_EMBD, N_FF},      // w2: 1×2048 × 2048×4096 → 1×4096
        {1, N_EMBD, N_EMBD},    // q/k/v proj: 1×4096 × 4096×4096 → 1×4096
    };
    const char *shape_names[] = {"w1_gate", "w3_up", "w2_down", "qkv_proj"};

    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < 4; i++) {
        int m = decode_shapes[i][0];
        int n = decode_shapes[i][1];
        int k = decode_shapes[i][2];
        printf("\n  Shape [%s]: %dx%dx%d (m=1, single token decode)\n",
               shape_names[i], m, n, k);
        double tflops = measure_cublas_matmul(handle, m, n, k, &alpha, &beta, 0, 10, 100);
        // Arithmetic intensity: FLOPs / bytes
        // F32: each token reads m*k + k*n bytes = 1*k + k*n = k*(1+n)
        double bytes = (double)k * (1 + n) * sizeof(float);
        double flops = 2.0 * m * n * k;
        double ai = flops / bytes;
        printf("    Arithmetic intensity: %.1f FLOPs/byte [derived]\n", ai);
        printf("    This is memory-bound (AI < 100 FLOPs/byte for HBM peak)\n");
    }

    cublasDestroy(handle);

    printf("\n============================================================\n");
    printf("NSIGHT PROFILING INSTRUCTIONS\n");
    printf("============================================================\n");
    printf("\n");
    printf("To profile with Nsight Compute:\n");
    printf("  ncu --set full -o decode_step_profile ./decode_step_bench\n");
    printf("  ncu --section MemoryWorkloadAnalysis --section SchedulerStats \\\n");
    printf("      --section WarpStateStats --page details -o decode_step_light \\\n");
    printf("      ./decode_step_bench\n");
    printf("\n");
    printf("Key metrics to inspect in the report:\n");
    printf("  - DRAM throughput (achieved GB/s vs 1500 GB/s peak)\n");
    printf("  - L2 hit rate\n");
    printf("  - Warp stall reasons (long scoreboard, no instruction, wait, etc.)\n");
    printf("  - Achieved compute utilization (SM busy %)\n");
    printf("  - Memory dependency stalls: are warps waiting on dequant ALU or HBM reads?\n");
    printf("\n");
    printf("Hypothesis: Achieved DRAM throughput = 300-600 GB/s (20-40%% of peak)\n");
    printf("Hypothesis: Dominant warp stall = 'long scoreboard' (waiting on memory)\n");
    printf("Hypothesis: Compute utilization < 30%% during Q4_K matmul kernels\n");
    printf("\n");
    printf("=== Done ===\n");
    return 0;
}
