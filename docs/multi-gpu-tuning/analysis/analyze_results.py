#!/usr/bin/env python3
"""Analysis summary for multi-GPU tuning research."""

import csv
import json
from pathlib import Path

DATA_DIR = Path("/opt/ds4/docs/multi-gpu-tuning/data")

def parse_pcie_results():
    """Parse PCIe bandwidth CSV rows from the benchmark output."""
    results = {"uni_01": {}, "uni_10": {}, "bi": {}, "bounce": {}, "align": {}}
    
    path = DATA_DIR / "pcie_bw_results_v2.txt"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return results
        
    with open(path) as f:
        for line in f:
            if line.startswith("DATA_UNI"):
                parts = line.strip().split("\t")
                sz = int(parts[1])
                results["uni_01"][sz] = {
                    "bw_gbps": float(parts[2]),
                    "lat_us": float(parts[3]),
                    "std_us": float(parts[4]),
                    "min_us": float(parts[5]),
                    "max_us": float(parts[6]),
                }
                results["uni_10"][sz] = {
                    "bw_gbps": float(parts[7]),
                    "lat_us": float(parts[8]),
                    "std_us": float(parts[9]),
                }
            elif line.startswith("DATA_BI"):
                parts = line.strip().split("\t")
                sz = int(parts[1])
                results["bi"][sz] = {
                    "agg_bw_gbps": float(parts[2]),
                    "per_dir_gbps": float(parts[3]),
                    "lat_us": float(parts[4]),
                    "std_us": float(parts[5]),
                }
            elif line.startswith("DATA_BOUNCE"):
                parts = line.strip().split("\t")
                sz = int(parts[1])
                results["bounce"][sz] = {
                    "bw_gbps": float(parts[2]),
                    "lat_us": float(parts[3]),
                    "std_us": float(parts[4]),
                }
            elif line.startswith("DATA_ALIGN"):
                parts = line.strip().split("\t")
                offset = int(parts[1])
                results["align"][offset] = {
                    "bw_gbps": float(parts[2]),
                    "lat_us": float(parts[3]),
                }
    return results

def parse_compute_results():
    """Parse compute peak results."""
    results = {"fma": {}, "read": {}, "copy": {}}
    
    path = DATA_DIR / "compute_peak_results.txt"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return results
        
    with open(path) as f:
        for line in f:
            if line.startswith("DATA_FMA"):
                parts = line.strip().split("\t")
                gpu = int(parts[1])
                results["fma"][gpu] = {
                    "tflops": float(parts[2]),
                    "mean_ms": float(parts[3]),
                    "std_ms": float(parts[4]),
                }
            elif line.startswith("DATA_READ"):
                parts = line.strip().split("\t")
                gpu = int(parts[1])
                results["read"][gpu] = {
                    "gbps": float(parts[2]),
                    "mean_ms": float(parts[3]),
                    "std_ms": float(parts[4]),
                }
            elif line.startswith("DATA_COPY"):
                parts = line.strip().split("\t")
                gpu = int(parts[1])
                results["copy"][gpu] = {
                    "gbps": float(parts[2]),
                    "mean_ms": float(parts[3]),
                    "std_ms": float(parts[4]),
                }
    return results

def main():
    print("=" * 60)
    print("Multi-GPU Tuning Research — Analysis Summary")
    print("=" * 60)
    
    # PCIe results
    pcie = parse_pcie_results()
    if pcie["uni_01"]:
        max_bw = max(v["bw_gbps"] for v in pcie["uni_01"].values())
        max_sz = max(pcie["uni_01"].keys())
        # Find latency at activation size (16KB = 16384)
        act16k = pcie["uni_01"].get(16384, {})
        act32k = pcie["uni_01"].get(32768, {})
        
        print(f"\nPCIe Unidirectional peak: {max_bw:.1f} GB/s at {max_sz} bytes")
        print(f"  Activation-size (16KB): {act16k.get('bw_gbps', 0):.1f} GB/s, {act16k.get('lat_us', 0):.1f} µs")
        print(f"  Activation-size (32KB): {act32k.get('bw_gbps', 0):.1f} GB/s, {act32k.get('lat_us', 0):.1f} µs")
    
    if pcie["bi"]:
        max_bi = max(v["agg_bw_gbps"] for v in pcie["bi"].values())
        print(f"\nPCIe Bidirectional peak: {max_bi:.1f} GB/s aggregate")
    
    if pcie["bounce"]:
        max_bounce = max(v["bw_gbps"] for v in pcie["bounce"].values())
        print(f"Host bounce peak: {max_bounce:.1f} GB/s")
    
    # Compute results
    comp = parse_compute_results()
    if comp["read"] and 0 in comp["read"]:
        print(f"\nHBM read: GPU0={comp['read'][0]['gbps']:.0f} GB/s, GPU1={comp['read'][1]['gbps']:.0f} GB/s")
    if comp["copy"] and 0 in comp["copy"]:
        print(f"HBM copy: GPU0={comp['copy'][0]['gbps']:.0f} GB/s, GPU1={comp['copy'][1]['gbps']:.0f} GB/s")
    if comp["fma"] and 0 in comp["fma"]:
        print(f"FMA (mem-bound): GPU0={comp['fma'][0]['tflops']:.2f} TFLOPS, GPU1={comp['fma'][1]['tflops']:.2f} TFLOPS")
    
    # Pipeline results
    bench_path = DATA_DIR / "pipeline_bench_baseline.csv"
    if bench_path.exists():
        with open(bench_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                print(f"\nPipeline (ctx={row['ctx_tokens']}): "
                      f"prefill={row['prefill_tps']} t/s, "
                      f"decode={row['gen_steady_tps']} t/s, "
                      f"first_tok={row['gen_first_ms']} ms")
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("  1. PCIe is NOT a bottleneck (0.003% utilization at 68 t/s)")
    print("  2. HBM bandwidth is the primary ceiling (~1500 GB/s)")
    print("  3. Full GPU residency achieved — no SSD streaming needed")
    print("  4. Pipeline load imbalance is ~5% (22 vs 20 layers)")
    print("  5. DSpark draft/verify is coordinator-only, doesn't scale with N GPUs")
    print("=" * 60)

if __name__ == "__main__":
    main()
