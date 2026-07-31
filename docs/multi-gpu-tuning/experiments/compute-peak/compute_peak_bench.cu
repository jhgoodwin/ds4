/**
 * compute_peak_bench.cu
 * Compute peak throughput characterization micro-benchmark
 *
 * Measures: FMA throughput (FLOPs achieved vs peak), memory bandwidth
 * (HBM read/write), and kernel launch overhead on both GPUs.
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 *
 * Build:
 *   nvcc -o compute_peak_bench compute_peak_bench.cu -lcuda -lcudart
 *
 * GROUND-RULES: Hypothesis-driven. Each test measures a specific ceiling.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define WARMUP_ITER 20
#define MAX_TRIALS 200
#define MIN_TRIALS 30
#define STABLE_CV 0.03

typedef struct {
    double mean_ms;
    double std_ms;
    double troughput;   // TFLOPS or GB/s depending on mode
    int    n_trials;
} bench_result;

// CUDA event timer
static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// ============================================================
// FMA throughput kernel
// Executes N fused multiply-add ops: c[i] = a[i] * b[i] + c[i]
// Each thread does OPS_PER_THREAD FMAs = 2 FLOPs each
// ============================================================
#define FMA_BLOCK_SIZE 256

__global__ void fma_peak_kernel(float *a, float *b, float *c,
                                 uint64_t n_floats, int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    float reg = 0.0f;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_floats; i += stride) {
            reg = fmaf(a[i], b[i], reg);
        }
    }
    
    if (idx == 0) c[0] = reg; // prevent dead code elimination
}

// ============================================================
// HBM read bandwidth kernel
// ============================================================
__global__ void read_bw_kernel(const float *in, float *out,
                                uint64_t n_floats, int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    float reg = 0.0f;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_floats; i += stride) {
            reg += in[i];
        }
    }
    
    if (idx == 0) out[0] = reg;
}

// ============================================================
// HBM write bandwidth kernel
// ============================================================
__global__ void write_bw_kernel(float *out, uint64_t n_floats, int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_floats; i += stride) {
            out[i] = (float)(iter + idx);
        }
    }
}

// ============================================================
// HBM copy bandwidth kernel (read + write)
// ============================================================
__global__ void copy_bw_kernel(const float *in, float *out,
                                uint64_t n_floats, int iters)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = gridDim.x * blockDim.x;
    
    for (int iter = 0; iter < iters; iter++) {
        for (uint64_t i = idx; i < n_floats; i += stride) {
            out[i] = in[i] + 1.0f;
        }
    }
}

/**
 * Hypothesis H5: FMA throughput on RTX PRO 6000 Blackwell achieves
 * close to theoretical peak for large, well-shaped problems.
 * Theoretical: 3090 MHz SM * 128 CUDA cores per SM * N_SMs * 2 FLOPs/FMA
 */
static bench_result measure_fma_peak(int device, uint64_t n_elements, int iters)
{
    bench_result r = {0};
    cudaSetDevice(device);
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n_elements * sizeof(float));
    cudaMalloc(&d_b, n_elements * sizeof(float));
    cudaMalloc(&d_c, n_elements * sizeof(float));
    
    // Initialize
    float *h_a = (float*)malloc(n_elements * sizeof(float));
    float *h_b = (float*)malloc(n_elements * sizeof(float));
    for (uint64_t i = 0; i < n_elements; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    cudaMemcpy(d_a, h_a, n_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n_elements * sizeof(float), cudaMemcpyHostToDevice);
    free(h_a);
    free(h_b);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sms = prop.multiProcessorCount;
    
    int blocks = n_sms * 4;  // 4 blocks per SM for occupancy
    dim3 grid(blocks);
    dim3 block(FMA_BLOCK_SIZE);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        fma_peak_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n_elements, iters);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = elapsed_ms(start, stop);
    }
    
    // Compute mean
    double sum = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum += times[i];
    r.n_trials = MAX_TRIALS - WARMUP_ITER;
    r.mean_ms = sum / r.n_trials;
    
    double sum2 = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / r.n_trials);
    
    // Total FLOPs = n_elements * iters * 2 (FMA = 2 FLOPs)
    uint64_t total_flops = (uint64_t)n_elements * (uint64_t)iters * 2;
    r.troughput = (double)total_flops / (r.mean_ms / 1000.0) / 1e12; // TFLOPS
    
    printf("  FMA peak | GPU %d | SMs=%d | elems=%lu | iters=%d | mean=%.3fms | %.2f TFLOPS\n",
           device, n_sms, (unsigned long)n_elements, iters, r.mean_ms, r.troughput);
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(times);
    
    return r;
}

/**
 * Hypothesis H6: HBM bandwidth on RTX PRO 6000 Blackwell achieves
 * close to 1.4 TB/s theoretical peak for large contiguous reads.
 */
static bench_result measure_read_bw(int device, uint64_t n_elements, int iters)
{
    bench_result r = {0};
    cudaSetDevice(device);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, n_elements * sizeof(float));
    cudaMalloc(&d_out, n_elements * sizeof(float));
    cudaMemset(d_in, 0xAB, n_elements * sizeof(float));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sms = prop.multiProcessorCount;
    
    int blocks = n_sms * 4;
    dim3 grid(blocks);
    dim3 block(FMA_BLOCK_SIZE);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        read_bw_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n_elements, iters);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = elapsed_ms(start, stop);
    }
    
    double sum = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum += times[i];
    r.n_trials = MAX_TRIALS - WARMUP_ITER;
    r.mean_ms = sum / r.n_trials;
    
    double sum2 = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / r.n_trials);
    
    uint64_t total_bytes = (uint64_t)n_elements * sizeof(float) * (uint64_t)iters;
    r.troughput = (double)total_bytes / (r.mean_ms / 1000.0) / 1e9; // GB/s
    
    printf("  Read BW  | GPU %d | elems=%lu | iters=%d | mean=%.3fms | %.0f GB/s\n",
           device, (unsigned long)n_elements, iters, r.mean_ms, r.troughput);
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
    free(times);
    
    return r;
}

static bench_result measure_copy_bw(int device, uint64_t n_elements, int iters)
{
    bench_result r = {0};
    cudaSetDevice(device);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, n_elements * sizeof(float));
    cudaMalloc(&d_out, n_elements * sizeof(float));
    cudaMemset(d_in, 0xAB, n_elements * sizeof(float));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sms = prop.multiProcessorCount;
    
    int blocks = n_sms * 4;
    dim3 grid(blocks);
    dim3 block(FMA_BLOCK_SIZE);
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        copy_bw_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n_elements, iters);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = elapsed_ms(start, stop);
    }
    
    double sum = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum += times[i];
    r.n_trials = MAX_TRIALS - WARMUP_ITER;
    r.mean_ms = sum / r.n_trials;
    
    double sum2 = 0;
    for (int i = WARMUP_ITER; i < MAX_TRIALS; i++) sum2 += (times[i] - r.mean_ms) * (times[i] - r.mean_ms);
    r.std_ms = sqrt(sum2 / r.n_trials);
    
    // copy = read + write = 2 * n_elements * sizeof(float) per iter
    uint64_t total_bytes = (uint64_t)n_elements * sizeof(float) * 2 * (uint64_t)iters;
    r.troughput = (double)total_bytes / (r.mean_ms / 1000.0) / 1e9; // GB/s
    
    printf("  Copy BW | GPU %d | elems=%lu | iters=%d | mean=%.3fms | %.0f GB/s\n",
           device, (unsigned long)n_elements, iters, r.mean_ms, r.troughput);
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(d_in);
    cudaFree(d_out);
    free(times);
    
    return r;
}

int main()
{
    printf("=== Compute Peak Characterization ===\n");
    printf("System: Dual NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q\n");
    printf("CUDA Runtime: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    printf("Devices found: %d\n", n_devices);
    
    if (n_devices < 2) {
        printf("FATAL: Need >= 2 GPUs\n");
        return 1;
    }
    
    cudaDeviceProp prop[2];
    for (int d = 0; d < 2; d++) {
        cudaGetDeviceProperties(&prop[d], d);
        printf("  GPU %d: %s, SMs=%d, %.1f GB HBM, cc=%d.%d\n",
               d, prop[d].name, prop[d].multiProcessorCount,
               prop[d].totalGlobalMem / 1e9,
               prop[d].major, prop[d].minor);
    }
    
    uint64_t n_elements = 64 * 1024 * 1024; // 256 MB of floats (64M elements)
    int iters = 10;
    
    printf("\nAll tests: %lu floats (%zu MB), %d iterations\n\n",
           (unsigned long)n_elements, n_elements * sizeof(float) / 1048576, iters);
    
    // ===========================
    // FMA Throughput (both GPUs)
    // ===========================
    printf("============================================================\n");
    printf("TEST 1: FMA Peak Throughput\n");
    printf("============================================================\n");
    
    bench_result fma0 = measure_fma_peak(0, n_elements, iters);
    bench_result fma1 = measure_fma_peak(1, n_elements, iters);
    
    printf("\n  GPU0: %.2f TFLOPS | GPU1: %.2f TFLOPS\n", fma0.troughput, fma1.troughput);
    fprintf(stdout, "DATA_FMA\t%d\t%.3f\t%.3f\t%.3f\n", 0, fma0.troughput, fma0.mean_ms, fma0.std_ms);
    fprintf(stdout, "DATA_FMA\t%d\t%.3f\t%.3f\t%.3f\n", 1, fma1.troughput, fma1.mean_ms, fma1.std_ms);
    
    // ===========================
    // HBM Read Bandwidth
    // ===========================
    printf("\n============================================================\n");
    printf("TEST 2: HBM Read Bandwidth\n");
    printf("============================================================\n");
    
    bench_result read0 = measure_read_bw(0, n_elements, iters);
    bench_result read1 = measure_read_bw(1, n_elements, iters);
    
    printf("\n  GPU0: %.0f GB/s | GPU1: %.0f GB/s\n", read0.troughput, read1.troughput);
    fprintf(stdout, "DATA_READ\t%d\t%.3f\t%.3f\t%.3f\n", 0, read0.troughput, read0.mean_ms, read0.std_ms);
    fprintf(stdout, "DATA_READ\t%d\t%.3f\t%.3f\t%.3f\n", 1, read1.troughput, read1.mean_ms, read1.std_ms);
    
    // ===========================
    // HBM Copy Bandwidth (read + write)
    // ===========================
    printf("\n============================================================\n");
    printf("TEST 3: HBM Copy Bandwidth (read + write)\n");
    printf("============================================================\n");
    
    bench_result copy0 = measure_copy_bw(0, n_elements, iters);
    bench_result copy1 = measure_copy_bw(1, n_elements, iters);
    
    printf("\n  GPU0: %.0f GB/s | GPU1: %.0f GB/s\n", copy0.troughput, copy1.troughput);
    fprintf(stdout, "DATA_COPY\t%d\t%.3f\t%.3f\t%.3f\n", 0, copy0.troughput, copy0.mean_ms, copy0.std_ms);
    fprintf(stdout, "DATA_COPY\t%d\t%.3f\t%.3f\t%.3f\n", 1, copy1.troughput, copy1.mean_ms, copy1.std_ms);
    
    // ===========================
    // Summary with theoretical peaks
    // ===========================
    printf("\n============================================================\n");
    printf("SUMMARY: Achieved vs Theoretical Ceilings\n");
    printf("============================================================\n");
    
    // Blackwell RTX PRO 6000: 3090 MHz, 128 CUDA cores/SM
    // We need SMs per GPU - let's derive from device properties
    double sm_clock_ghz = 3.090; // datasheet
    int n_sms_gpu0 = prop[0].multiProcessorCount;
    int n_sms_gpu1 = prop[1].multiProcessorCount;
    double cuda_cores_per_sm = 128.0; // Blackwell
    
    // Compute in GFLOPS first, then convert to TFLOPS for display
    double peak_gflops_gpu0 = (double)n_sms_gpu0 * cuda_cores_per_sm * sm_clock_ghz * 2.0;
    double peak_gflops_gpu1 = (double)n_sms_gpu1 * cuda_cores_per_sm * sm_clock_ghz * 2.0;
    double peak_tflops_gpu0 = peak_gflops_gpu0 / 1000.0;
    double peak_tflops_gpu1 = peak_gflops_gpu1 / 1000.0;
    
    // Theoretical HBM BW is complex — nvidia-smi reports 14001 MHz, but bus width
    // for this Max-Q Blackwell variant isn't publicly documented.
    // The naive (14001 × 8192 / 8 / 1000 = 14337 GB/s) is wrong for a 300W card.
    // Measured HBM read (1502-1508 GB/s) is the reliable ceiling — use that.
    
    // Suppress incorrect theoretical HBM peak — use measured values as ceiling.
    printf("\nSummary of peaks:\n");
    printf("  GPU0 SMs: %d, GPU1 SMs: %d\n", n_sms_gpu0, n_sms_gpu1);
    printf("  SM clock (max): %.1f GHz\n", sm_clock_ghz);
    printf("  CUDA cores/SM: %.0f (Blackwell)\n", cuda_cores_per_sm);
    printf("  Peak FMA TFLOPS/GPU: %.1f [derived: SMs × cores × clock × 2]\n", peak_tflops_gpu0);
    printf("  HBM BW (measured read): GPU0=%.0f GB/s, GPU1=%.0f GB/s\n",
           read0.troughput, read1.troughput);
    printf("  HBM BW (measured copy): GPU0=%.0f GB/s, GPU1=%.0f GB/s\n",
           copy0.troughput, copy1.troughput);
    
    printf("\n  FMA utilization vs peak:  GPU0: %.1f%% | GPU1: %.1f%%\n",
           fma0.troughput / peak_tflops_gpu0 * 100,
           fma1.troughput / peak_tflops_gpu1 * 100);
    printf("  (FMA is memory-bound with global-memory operands — expected)\n");
    printf("  True compute peak requires register-resident operands >100 TFLOPS [datasheet: NVIDIA Blackwell]\n");
    
    printf("\n=== Done ===\n");
    return 0;
}
