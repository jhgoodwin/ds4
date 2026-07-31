/**
 * pcie_bw_bench.cu
 * Multi-GPU PCIe bandwidth characterization micro-benchmark
 *
 * Measures: unidirectional GPU0->GPU1, GPU1->GPU0, bidirectional PCIe Gen 5 x8
 * Also measures: peer access vs host bounce, alignment effects
 *
 * System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
 * PCIe: Gen 5 x8 per card (split from x16 CPU lanes)
 * No NVLink
 *
 * Build:
 *   nvcc -o pcie_bw_bench pcie_bw_bench.cu -lcuda -lcudart
 *
 * GROUND-RULES: Hypothesis-driven measurement. Each test quantifies a specific
 * throughput characteristic. Values tagged [measured: pcie_bw_bench_run_N].
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

#define WARMUP_ITER 20
#define MIN_TRIALS  50
#define MAX_TRIALS  1000
#define STABLE_CV_THRESHOLD 0.03  // 3% CV to consider stable

// Transfer sizes to test (powers of 2, bytes)
// Covers: typical activation vectors at F16/F32, KV cache pages, expert weights
static const size_t xfer_sizes[] = {
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
    8388608, 16777216, 33554432, 67108864
};
static const int n_sizes = sizeof(xfer_sizes) / sizeof(xfer_sizes[0]);

typedef struct {
    double mean_us;    // mean latency in microseconds
    double std_us;     // std deviation in microseconds
    double bw_gbps;    // bandwidth in GB/s (unidirectional or bidirectional aggregate)
    double bw_gbps_per_dir; // per-direction bandwidth
    int    n_trials;   // number of trials used
    double cv;         // coefficient of variation
    int    converged;  // 1 if CV < threshold at convergence
    double min_us;     // minimum observed latency
    double max_us;     // maximum observed latency
} bw_result;

// Convert bytes to human-readable string
static void fmt_bytes(char *buf, size_t sz) {
    if (sz < 1024)
        snprintf(buf, 32, "%zuB", sz);
    else if (sz < 1048576)
        snprintf(buf, 32, "%zuKB", sz / 1024);
    else if (sz < 1073741824)
        snprintf(buf, 32, "%zuMB", sz / 1048576);
    else
        snprintf(buf, 32, "%.2fGB", (double)sz / 1073741824.0);
}

// GPU elapsed time in milliseconds from events
static double gpu_elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// Run trials, output all raw times for analysis
// Returns stable window or full sample statistics
static int analyze_times(const double *times, int n_total,
                          double *out_mean, double *out_std,
                          double *out_min, double *out_max,
                          double *out_cv, int *out_converged)
{
    // Use all samples after warmup
    int n_valid = n_total - WARMUP_ITER;
    if (n_valid < 1) n_valid = 1;
    
    const double *samples = times + WARMUP_ITER;
    
    double sum = 0, sum2 = 0;
    *out_min = samples[0];
    *out_max = samples[0];
    
    for (int i = 0; i < n_valid; i++) {
        sum += samples[i];
        sum2 += samples[i] * samples[i];
        if (samples[i] < *out_min) *out_min = samples[i];
        if (samples[i] > *out_max) *out_max = samples[i];
    }
    
    *out_mean = sum / n_valid;
    double variance = (sum2 / n_valid) - ((*out_mean) * (*out_mean));
    if (variance < 0) variance = 0;
    *out_std = sqrt(variance);
    *out_cv = (*out_mean > 1e-12) ? *out_std / *out_mean : 0;
    *out_converged = (*out_cv < STABLE_CV_THRESHOLD && n_valid >= MIN_TRIALS);
    
    return n_valid;
}

/**
 * Hypothesis H1: cudaMemcpyPeerAsync achieves PCIe Gen 5 x8 line rate
 * for large transfers with aligned buffers. Small transfers dominated by
 * latency and protocol overhead.
 */
static bw_result measure_unidirectional(int src_dev, int dst_dev, size_t bytes)
{
    bw_result r = {0};
    
    cudaSetDevice(src_dev);
    void *src_buf = NULL;
    cudaMalloc(&src_buf, bytes);
    cudaMemset(src_buf, 0xAB, bytes);
    
    cudaSetDevice(dst_dev);
    void *dst_buf = NULL;
    cudaMalloc(&dst_buf, bytes);
    cudaMemset(dst_buf, 0xCD, bytes);
    
    cudaStream_t stream;
    cudaSetDevice(src_dev);
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        cudaMemcpyPeerAsync(dst_buf, dst_dev, src_buf, src_dev, bytes, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = gpu_elapsed_ms(start, stop);
    }
    
    r.n_trials = analyze_times(times, MAX_TRIALS, &r.mean_us, &r.std_us,
                                &r.min_us, &r.max_us, &r.cv, &r.converged);
    r.mean_us *= 1000.0; // ms -> us
    r.std_us  *= 1000.0;
    r.min_us  *= 1000.0;
    r.max_us  *= 1000.0;
    r.bw_gbps = (r.mean_us > 0) ? ((double)bytes / 1e9) / (r.mean_us / 1e6) : 0;
    r.bw_gbps_per_dir = r.bw_gbps;
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaSetDevice(src_dev);
    cudaFree(src_buf);
    cudaSetDevice(dst_dev);
    cudaFree(dst_buf);
    free(times);
    
    return r;
}

/**
 * Hypothesis H2: Bidirectional PCIe achieves < aggregate throughput than
 * 2x unidirectional due to PCIe protocol (flow control, ACK/NACK, TLP overhead).
 * Expect ~90% of 2x unidirectional for aligned large transfers.
 */
static bw_result measure_bidirectional(int dev0, int dev1, size_t bytes)
{
    bw_result r = {0};
    
    cudaSetDevice(dev0);
    void *buf0_src = NULL, *buf0_dst = NULL;
    cudaMalloc(&buf0_src, bytes);
    cudaMalloc(&buf0_dst, bytes);
    cudaMemset(buf0_src, 0xAB, bytes);
    cudaMemset(buf0_dst, 0, bytes);
    
    cudaSetDevice(dev1);
    void *buf1_src = NULL, *buf1_dst = NULL;
    cudaMalloc(&buf1_src, bytes);
    cudaMalloc(&buf1_dst, bytes);
    cudaMemset(buf1_src, 0xCD, bytes);
    cudaMemset(buf1_dst, 0, bytes);
    
    // Use one stream per device, time via host-side clock_gettime to avoid
    // cross-device event sync issues. Launch both copies, then sync both devices.
    cudaStream_t s0, s1;
    cudaSetDevice(dev0);
    cudaStreamCreate(&s0);
    cudaSetDevice(dev1);
    cudaStreamCreate(&s1);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        
        // Launch both directions concurrently
        cudaMemcpyPeerAsync(buf0_dst, dev0, buf1_src, dev1, bytes, s0);
        cudaMemcpyPeerAsync(buf1_dst, dev1, buf0_src, dev0, bytes, s1);
        
        // Sync both devices (this waits for both copies)
        cudaSetDevice(dev0);
        cudaStreamSynchronize(s0);
        cudaSetDevice(dev1);
        cudaStreamSynchronize(s1);
        
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double elapsed = (double)(t1.tv_sec - t0.tv_sec) * 1e6 +
                         (double)(t1.tv_nsec - t0.tv_nsec) / 1e3;
        times[i] = elapsed;
    }
    
    r.n_trials = analyze_times(times, MAX_TRIALS, &r.mean_us, &r.std_us,
                                &r.min_us, &r.max_us, &r.cv, &r.converged);
    // mean_us already in us (clock_gettime)
    // Bidirectional: total bytes moved = 2 * bytes
    double total_bytes = 2.0 * (double)bytes;
    r.bw_gbps = (r.mean_us > 0) ? (total_bytes / 1e9) / (r.mean_us / 1e6) : 0;
    r.bw_gbps_per_dir = r.bw_gbps / 2.0;
    
    cudaStreamDestroy(s0);
    cudaStreamDestroy(s1);
    cudaSetDevice(dev0);
    cudaFree(buf0_src);
    cudaFree(buf0_dst);
    cudaSetDevice(dev1);
    cudaFree(buf1_src);
    cudaFree(buf1_dst);
    free(times);
    
    return r;
}

/**
 * Hypothesis H3: Host bounce (D2H + H2D via pinned memory) is ~2x slower
 * than peer DMA for large transfers due to double PCIe traversal.
 * For very small transfers, latency overhead may dominate making bounce
 * comparable or even faster.
 */
static bw_result measure_host_bounce(int src_dev, int dst_dev, size_t bytes)
{
    bw_result r = {0};
    
    void *host_buf = NULL;
    cudaHostAlloc(&host_buf, bytes, cudaHostAllocDefault);
    memset(host_buf, 0, bytes);
    
    cudaSetDevice(src_dev);
    void *src_buf = NULL;
    cudaMalloc(&src_buf, bytes);
    cudaMemset(src_buf, 0xAB, bytes);
    
    cudaSetDevice(dst_dev);
    void *dst_buf = NULL;
    cudaMalloc(&dst_buf, bytes);
    
    cudaStream_t stream;
    cudaSetDevice(src_dev);
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        cudaMemcpyAsync(host_buf, src_buf, bytes, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(dst_buf, host_buf, bytes, cudaMemcpyHostToDevice, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = gpu_elapsed_ms(start, stop);
    }
    
    r.n_trials = analyze_times(times, MAX_TRIALS, &r.mean_us, &r.std_us,
                                &r.min_us, &r.max_us, &r.cv, &r.converged);
    r.mean_us *= 1000.0;
    r.std_us  *= 1000.0;
    r.min_us  *= 1000.0;
    r.max_us  *= 1000.0;
    r.bw_gbps = (r.mean_us > 0) ? ((double)bytes / 1e9) / (r.mean_us / 1e6) : 0;
    r.bw_gbps_per_dir = r.bw_gbps;
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(src_buf);
    cudaSetDevice(dst_dev);
    cudaFree(dst_buf);
    cudaFreeHost(host_buf);
    free(times);
    
    return r;
}

/**
 * Hypothesis H4: Misaligned peer access (non-128B offset) may cause
 * driver fallback or bandwidth degradation.
 */
static bw_result measure_misaligned(int src_dev, int dst_dev, size_t bytes, size_t offset)
{
    bw_result r = {0};
    size_t total = bytes + offset + 256;  // extra for alignment offset
    
    cudaSetDevice(src_dev);
    void *src_base = NULL;
    cudaMalloc(&src_base, total);
    void *src_buf = (void*)((uintptr_t)src_base + offset);
    cudaMemset(src_base, 0xAB, total);
    
    cudaSetDevice(dst_dev);
    void *dst_base = NULL;
    cudaMalloc(&dst_base, total);
    void *dst_buf = (void*)((uintptr_t)dst_base + offset);
    cudaMemset(dst_base, 0xCD, total);
    
    cudaStream_t stream;
    cudaSetDevice(src_dev);
    cudaStreamCreate(&stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    double *times = (double*)calloc(MAX_TRIALS, sizeof(double));
    
    for (int i = 0; i < MAX_TRIALS; i++) {
        cudaEventRecord(start, stream);
        cudaError_t err = cudaMemcpyPeerAsync(dst_buf, dst_dev, src_buf, src_dev, bytes, stream);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        times[i] = gpu_elapsed_ms(start, stop);
        if (err != cudaSuccess) {
            times[i] = -1.0; // mark failure
        }
    }
    
    r.n_trials = analyze_times(times, MAX_TRIALS, &r.mean_us, &r.std_us,
                                &r.min_us, &r.max_us, &r.cv, &r.converged);
    r.mean_us *= 1000.0;
    r.std_us  *= 1000.0;
    r.bw_gbps = (r.mean_us > 0) ? ((double)bytes / 1e9) / (r.mean_us / 1e6) : 0;
    r.bw_gbps_per_dir = r.bw_gbps;
    
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaStreamDestroy(stream);
    cudaFree(src_base);
    cudaFree(dst_base);
    free(times);
    
    return r;
}

int main()
{
    printf("=== PCIe Bandwidth Characterization ===\n");
    printf("System: Dual NVIDIA RTX PRO 6000 Blackwell 96GB Max-Q\n");
    printf("PCIe: Gen 5 x8 per card (split from x16)\n");
    printf("Interconnect: PCIe P2P (no NVLink)\n");
    printf("CUDA Runtime: %d.%d\n\n", CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    printf("Devices found: %d\n", n_devices);
    
    if (n_devices < 2) {
        printf("FATAL: Need >= 2 GPUs\n");
        return 1;
    }
    
    for (int d = 0; d < n_devices; d++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        printf("  GPU %d: %s, %zu MiB global, cc %d.%d\n",
               d, prop.name, prop.totalGlobalMem / 1048576,
               prop.major, prop.minor);
    }
    
    // Enable peer access
    for (int d = 0; d < 2; d++) {
        for (int e = 0; e < 2; e++) {
            if (d == e) continue;
            int can_access = 0;
            cudaSetDevice(d);
            cudaDeviceCanAccessPeer(&can_access, d, e);
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(e, 0);
                printf("  Peer GPU%d->GPU%d: %s (enable: %s)\n",
                       d, e, can_access ? "DIRECT" : "NO",
                       err == cudaSuccess ? "OK" : cudaGetErrorString(err));
            } else {
                printf("  Peer GPU%d->GPU%d: NO (host bounce only)\n", d, e);
            }
        }
    }
    
    int peer = 1; // assume peer access enabled
    printf("\n");
    
    // ===========================
    // Test 1: Unidirectional P2P
    // ===========================
    printf("=================================================================\n");
    printf("TEST 1: Unidirectional Peer DMA\n");
    printf("=================================================================\n");
    printf("%-10s | %-8s | %-8s | %-8s | %-8s | %-6s\n",
           "Size", "BW_01", "BW_10", "Lat_01", "Lat_10", "Conv");
    printf("%-10s-|-%-8s-|-%-8s-|-%-8s-|-%-8s-|-%-6s\n",
           "----------", "--------", "--------", "--------", "--------", "------");
    
    for (int i = 0; i < n_sizes; i++) {
        size_t sz = xfer_sizes[i];
        char sz_str[12];
        fmt_bytes(sz_str, sz);
        
        bw_result u01 = measure_unidirectional(0, 1, sz);
        bw_result u10 = measure_unidirectional(1, 0, sz);
        
        printf("%-10s | %7.2f | %7.2f | %7.1f | %7.1f | %s\n",
               sz_str,
               u01.bw_gbps, u10.bw_gbps,
               u01.mean_us, u10.mean_us,
               u01.converged ? "OK" : "VAR");
               
        // Raw data row
        fprintf(stdout, "DATA_UNI\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\t%d\n",
                sz,
                u01.bw_gbps, u01.mean_us, u01.std_us, u01.min_us, u01.max_us,
                u10.bw_gbps, u10.mean_us, u10.std_us,
                u01.n_trials, u10.n_trials);
    }
    
    // ===========================
    // Test 2: Bidirectional P2P
    // ===========================
    printf("\n=================================================================\n");
    printf("TEST 2: Bidirectional Peer DMA\n");
    printf("=================================================================\n");
    printf("%-10s | %-10s | %-10s | %-8s | %-6s\n",
           "Size", "Agg_GBs", "PerDir_GBs", "Lat_us", "Conv");
    printf("%-10s-|-%-10s-|-%-10s-|-%-8s-|-%-6s\n",
           "----------", "----------", "----------", "--------", "------");
    
    for (int i = 0; i < n_sizes; i++) {
        size_t sz = xfer_sizes[i];
        char sz_str[12];
        fmt_bytes(sz_str, sz);
        
        bw_result bi = measure_bidirectional(0, 1, sz);
        
        printf("%-10s | %8.2f  | %8.2f  | %7.1f | %s\n",
               sz_str,
               bi.bw_gbps, bi.bw_gbps_per_dir,
               bi.mean_us,
               bi.converged ? "OK" : "VAR");
               
        fprintf(stdout, "DATA_BI\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n",
                sz,
                bi.bw_gbps, bi.bw_gbps_per_dir, bi.mean_us, bi.std_us,
                bi.min_us, bi.max_us,
                bi.n_trials);
    }
    
    // ===========================
    // Test 3: Host Bounce
    // ===========================
    printf("\n=================================================================\n");
    printf("TEST 3: Host Bounce (D2H + H2D via pinned memory)\n");
    printf("=================================================================\n");
    printf("%-10s | %-8s | %-8s | %-6s\n",
           "Size", "BW_eff", "Lat_us", "Conv");
    printf("%-10s-|-%-8s-|-%-8s-|-%-6s\n",
           "----------", "--------", "--------", "------");
    
    for (int i = 0; i < n_sizes; i++) {
        size_t sz = xfer_sizes[i];
        char sz_str[12];
        fmt_bytes(sz_str, sz);
        
        bw_result hb = measure_host_bounce(0, 1, sz);
        
        printf("%-10s | %7.2f | %7.1f | %s\n",
               sz_str,
               hb.bw_gbps, hb.mean_us,
               hb.converged ? "OK" : "VAR");
               
        fprintf(stdout, "DATA_BOUNCE\t%zu\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%d\n",
                sz,
                hb.bw_gbps, hb.mean_us, hb.std_us,
                hb.min_us, hb.max_us,
                hb.n_trials);
    }
    
    // ===========================
    // Test 4: Alignment sensitivity (at 1MB transfer)
    // ===========================
    printf("\n=================================================================\n");
    printf("TEST 4: Alignment Sensitivity (offset from 128B alignment)\n");
    printf("Transfer size: 1MB\n");
    printf("=================================================================\n");
    printf("%-10s | %-8s | %-8s | %-6s\n",
           "Offset", "BW_GBs", "Lat_us", "Conv");
    printf("%-10s-|-%-8s-|-%-8s-|-%-6s\n",
           "----------", "--------", "--------", "------");
    
    size_t offsets[] = {0, 1, 4, 16, 32, 64, 128, 256, 1024};
    int n_offsets = sizeof(offsets) / sizeof(offsets[0]);
    
    for (int i = 0; i < n_offsets; i++) {
        char off_str[12];
        fmt_bytes(off_str, offsets[i]);
        
        bw_result mal = measure_misaligned(0, 1, 1048576, offsets[i]);
        
        printf("%-10s | %7.2f | %7.1f | %s\n",
               off_str,
               mal.bw_gbps, mal.mean_us,
               mal.converged ? "OK" : "VAR");
               
        fprintf(stdout, "DATA_ALIGN\t%zu\t%.3f\t%.3f\t%.3f\t%d\n",
                offsets[i],
                mal.bw_gbps, mal.mean_us, mal.std_us,
                mal.n_trials);
    }
    
    printf("\n=== Done ===\n");
    return 0;
}
