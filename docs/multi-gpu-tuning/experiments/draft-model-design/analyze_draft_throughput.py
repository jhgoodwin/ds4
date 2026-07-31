#!/usr/bin/env python3
"""
O2.1 — Draft Model Design Space Analysis

Estimates achievable draft t/s for 100M-500M param draft models on one GPU.
Uses an analytical model calibrated against measured HBM bandwidth and
small-model throughput characteristics.

Method:
1. Model small transformer architecture at various sizes (100M-500M params)
2. Compute memory-bound throughput ceiling from HBM BW
3. Compute compute-bound ceiling from peak TFLOPS
4. Estimate real-world throughput with bandwidth utilization factor
5. Compare to base model decode t/s to identify optimal draft size

The draft model is memory-bandwidth bound at these sizes (100M-500M params
= 200-1000 MB of weights F16, fitting in <1% of HBM capacity).

Usage:
    python3 analyze_draft_throughput.py

System: Dual RTX PRO 6000 Blackwell 96GB Max-Q
  HBM BW: 1502 GB/s (read peak, [measured: compute_peak_bench])
  F16 peak: ~297 TFLOPS (tensor-core, [derived: 2× 148.7 TFLOPS])

GROUND-RULES: Analytical predictions tagged [hypothesis: draft-model-analysis].
"""

import json
import math
import os

# ============================================================
# System constants
# ============================================================
HBM_READ_BW = 1502.0  # GB/s [measured: compute_peak_bench]
HBM_COPY_BW = 1290.0  # GB/s [measured: compute_peak_bench]
PEAK_F16_TFLOPS = 297.4  # derived: 2× F32 peak for tensor-core HGEMM
N_SMS = 188  # RTX PRO 6000 Blackwell
SM_CLOCK_GHZ = 3.090  # max boost

# ============================================================
# Draft model architecture candidates
# ============================================================
# Small transformer with vocabulary sharing (tied embeddings)
# Vocab = 128256 (same as base model DeepSeek V4 Flash)
VOCAB_SIZE = 128256

draft_configs = [
    {
        "name": "80M_8L_512d",
        "n_layers": 8,
        "n_embd": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "n_ff": 2048,  # 4× n_embd
        "vocab_size": VOCAB_SIZE,
        "tie_embeddings": True,
    },
    {
        "name": "140M_12L_512d",
        "n_layers": 12,
        "n_embd": 512,
        "n_heads": 8,
        "n_kv_heads": 4,
        "n_ff": 2048,
        "vocab_size": VOCAB_SIZE,
        "tie_embeddings": True,
    },
    {
        "name": "200M_12L_768d",
        "n_layers": 12,
        "n_embd": 768,
        "n_heads": 12,
        "n_kv_heads": 4,
        "n_ff": 3072,
        "vocab_size": VOCAB_SIZE,
        "tie_embeddings": True,
    },
    {
        "name": "300M_12L_1024d",
        "n_layers": 12,
        "n_embd": 1024,
        "n_heads": 16,
        "n_kv_heads": 4,
        "n_ff": 4096,
        "vocab_size": VOCAB_SIZE,
        "tie_embeddings": True,
    },
    {
        "name": "500M_16L_1024d",
        "n_layers": 16,
        "n_embd": 1024,
        "n_heads": 16,
        "n_kv_heads": 4,
        "n_ff": 4096,
        "vocab_size": VOCAB_SIZE,
        "tie_embeddings": True,
    },
]

# ============================================================
# Model
# ============================================================
def compute_model_params(cfg):
    """Compute parameter count and memory footprint."""
    n_embd = cfg['n_embd']
    n_layers = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_kv_heads = cfg['n_kv_heads']
    n_ff = cfg['n_ff']
    vocab_size = cfg['vocab_size']
    head_dim = n_embd // n_heads
    tie_emb = cfg['tie_embeddings']

    params = {}

    # Embedding
    params['embedding'] = vocab_size * n_embd  # input embedding

    # Per-layer params
    params_per_layer = {}
    params_per_layer['attn_q'] = n_embd * n_heads * head_dim
    params_per_layer['attn_k'] = n_embd * n_kv_heads * head_dim
    params_per_layer['attn_v'] = n_embd * n_kv_heads * head_dim
    params_per_layer['attn_o'] = n_embd * n_heads * head_dim

    # RMSNorm gains (2 per layer: pre-attn, pre-ffn)
    params_per_layer['rms_norms'] = 2 * n_embd

    # FFN (SwiGLU: gate, up, down)
    params_per_layer['ffn_gate'] = n_embd * n_ff
    params_per_layer['ffn_up'] = n_embd * n_ff
    params_per_layer['ffn_down'] = n_ff * n_embd

    params['per_layer'] = sum(params_per_layer.values())
    params['n_layers'] = n_layers
    params['layers_total'] = params['per_layer'] * n_layers

    # Output head
    if tie_emb:
        params['output_head'] = 0  # shared with embedding
    else:
        params['output_head'] = n_embd * vocab_size

    # Final RMSNorm
    params['final_norm'] = n_embd

    total = params['embedding'] + params['layers_total'] + params['output_head'] + params['final_norm']
    params['total'] = total

    # Memory in F16 (2 bytes per param)
    params['weights_mb_f16'] = total * 2 / (1024 * 1024)

    # KV cache size (1 token for decode, but for context we need more)
    kv_per_token = 2 * n_layers * n_kv_heads * head_dim * 2  # 2 (K+V) × 2 bytes (F16)
    params['kv_cache_bytes_per_token'] = kv_per_token

    return params


def estimate_decode_throughput(params, cfg=None):
    """Estimate achievable draft t/s for one GPU on single token decode.

    At <1 GB model size, the model is entirely memory-bandwidth bound.
    Throughput ceiling = HBM_BW / weights_per_token.

    But real-world BW utilization is less than peak:
    - BW utilization for small models is ~60-80% of HBM peak due to
      launch overhead and small kernel occupancy
    - We factor in the Q4_K dequant overhead too (1.43× factor from O1.2)

    Args:
        params: dict from compute_model_params()
        cfg: original config dict (provides n_embd, n_layers, etc.)

    Returns: dict with optimistic, realistic, conservative estimates
    """
    weights_mb = params['weights_mb_f16']
    weights_gb = weights_mb / 1024

    # Memory-bound ceiling: each token reads all weights from HBM
    # t/s = HBM_BW (GB/s) / weights_per_token (GB)
    tps_memory_peak = HBM_READ_BW / (weights_gb * 1.09)  # 9% overhead for KV + activations

    # Compute-bound ceiling: each token needs 2 FLOPs/param
    total_params = params['total']
    flops_per_token = 2 * total_params  # 2 FLOPs per param per token

    # Use real n_embd from config instead of estimating from params
    if cfg is not None:
        n_embd = cfg['n_embd']
        n_layers = cfg['n_layers']
    else:
        # Fallback: derive n_embd from known small-transformer scaling
        # Standard relation: total ≈ vocab_size * n_embd + n_layers * (12 * n_embd^2) for 4× FFN
        # Solved: n_embd ≈ sqrt((total - vocab_size * n_embd) / (12 * n_layers))
        # Iterative approximation for config-less fallback:
        n_embd = int((total_params / (128256 + 4)) ** 0.5) if total_params < 1e10 else 4096
        n_layers = 12  # default assumption

    # Compute-bound ceiling: FLOPs per token already in flops_per_token (2× total_params)
    tps_compute_peak = PEAK_F16_TFLOPS * 1e12 / flops_per_token

    # BW utilization factors (from O1.2: F16 read achieves 917/1502 = 61% of peak)
    bw_util_optimistic = 0.75  # achievable with tuned kernels
    bw_util_realistic = 0.60   # matching measured F16 util
    bw_util_conservative = 0.45  # with dequant overhead + launch overhead

    # Compute utilization
    compute_util = 0.10  # heavily memory-bound, compute util is low

    tps_optimistic = min(tps_memory_peak * bw_util_optimistic, tps_compute_peak * compute_util * 3)
    tps_realistic = min(tps_memory_peak * bw_util_realistic, tps_compute_peak * compute_util * 2)
    tps_conservative = min(tps_memory_peak * bw_util_conservative, tps_compute_peak * compute_util)

    return {
        'tps_optimistic': tps_optimistic,
        'tps_realistic': tps_realistic,
        'tps_conservative': tps_conservative,
        'tps_memory_bound': tps_memory_peak,
        'tps_compute_bound': tps_compute_peak,
        'bw_util_optimistic': bw_util_optimistic,
        'bw_util_realistic': bw_util_realistic,
        'bw_util_conservative': bw_util_conservative,
        'note': 'Draft model throughput [hypothesis: draft-model-analysis]'
    }


def compute_spec_speedup(draft_tps, draft_size_params, base_step_ms=14.7):
    """Compute speculative decode speedup.

    draft_tps: estimated draft model throughput in t/s
    draft_size_params: parameter dict from compute_model_params
    base_step_ms: base model per-step time in ms (default: decode baseline)
    """
    # Verification step: full model decode for batch of draft tokens
    # t_verify is independent of draft model size — it's the base model
    # For single-token verify (non-batch): ~14.7ms
    # For batch verify (DSpark path): ~8.6ms
    t_verify_ms = 8.6  # DSpark batch-verify [measured: research-log.md]

    # Draft chain time: generate draft_len tokens
    # draft_len chosen to maximize throughput
    draft_len = 5  # DSpark default block size

    # Draft chain time = draft_len / draft_tps * 1000
    t_draft_ms = draft_len / draft_tps * 1000

    # t_spec_step = t_draft + t_verify + t_overhead
    t_overhead_ms = 1.7  # kernel launch + sync [measured: roofline-analysis.md]
    t_step_ms = t_draft_ms + t_verify_ms + t_overhead_ms

    # Expected accepted tokens from draft chain
    # WARNING: Measured acceptance rates in experiments (O2.X) falsified
    # all hypotheses > 0.55. Actual measured rate is ~5%.
    # Defaulting to 0.05 until real draft model measurements exist.
    acceptance_rates = {
        'code_default': 0.05,
        'note': 'All rates > 0.55 falsified by measured 5%. See PRD-2 §O2.X',
    }

    speedups = {}
    for scenario, accept_rate in acceptance_rates.items():
        if not isinstance(accept_rate, (int, float)):
            continue  # skip non-numeric entries (e.g. 'note')
        if accept_rate >= 1.0:
            continue  # skip degenerate: would divide by zero
        expected_accepted = (1 - accept_rate ** draft_len) / (1 - accept_rate)
        spec_tps = expected_accepted / (t_step_ms / 1000.0)
        speedups[scenario] = {
            'accept_rate': accept_rate,
            'expected_accepted': round(expected_accepted, 3),
            't_step_ms': round(t_step_ms, 3),
            'spec_tps': round(spec_tps, 1),
            't_draft_ms': round(t_draft_ms, 3),
            'draft_len': draft_len,
        }

    return speedups


def main():
    print("=" * 72)
    print("O2.1 — Draft Model Design Space Analysis")
    print("=" * 72)
    print()
    print(f"System: Dual RTX PRO 6000 Blackwell 96GB Max-Q")
    print(f"  HBM read peak: {HBM_READ_BW} GB/s [measured: compute_peak_bench]")
    print(f"  F16 peak: {PEAK_F16_TFLOPS} TFLOPS [derived: 2× F32 peak]")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print()

    results = []

    for cfg in draft_configs:
        params = compute_model_params(cfg)
        throughput = estimate_decode_throughput(params, cfg)
        speedups = compute_spec_speedup(throughput['tps_realistic'], params)

        n_params_m = params['total'] / 1e6
        weights_mb = params['weights_mb_f16']

        print(f"{'─'*72}")
        print(f"Config: {cfg['name']}")
        print(f"{'─'*72}")
        print(f"  Architecture: L={cfg['n_layers']}, d={cfg['n_embd']}, "
              f"h={cfg['n_heads']}, ffn={cfg['n_ff']}")
        print(f"  Params: {n_params_m:.0f}M")
        print(f"  Weights (F16): {weights_mb:.1f} MB ({weights_mb/1024:.2f} GB)")
        print(f"  KV cache (1 token): {params['kv_cache_bytes_per_token']:.0f} bytes")
        print()
        print(f"  Decode throughput estimates [hypothesis: draft-model-analysis]:")
        print(f"    Optimistic:    {throughput['tps_optimistic']:>10.0f} t/s "
              f"(BW util={throughput['bw_util_optimistic']:.0%})")
        print(f"    Realistic:     {throughput['tps_realistic']:>10.0f} t/s "
              f"(BW util={throughput['bw_util_realistic']:.0%})")
        print(f"    Conservative:  {throughput['tps_conservative']:>10.0f} t/s "
              f"(BW util={throughput['bw_util_conservative']:.0%})")
        print(f"    Memory-bound:  {throughput['tps_memory_bound']:>10.0f} t/s "
              f"(at 100% BW util)")
        print(f"    Compute-bound: {throughput['tps_compute_bound']:>10.0f} t/s "
              f"(at 100% compute util)")
        print()
        print(f"  Speculative decode speedup (at realistic draft t/s, "
              f"batch-verify {8.6}ms):")
        for scenario, sp in speedups.items():
            print(f"    {scenario:<20}: {sp['spec_tps']:>8.1f} t/s "
                  f"(t_step={sp['t_step_ms']:.1f}ms, "
                  f"accept={sp['expected_accepted']:.2f}/step)")
        print()

        r = {
            'name': cfg['name'],
            'architecture': {
                'n_layers': cfg['n_layers'],
                'n_embd': cfg['n_embd'],
                'n_heads': cfg['n_heads'],
                'n_ff': cfg['n_ff'],
            },
            'params': {
                'total_m': round(n_params_m, 1),
                'weights_mb_f16': round(weights_mb, 1),
                'kv_cache_bytes_per_token': params['kv_cache_bytes_per_token'],
            },
            'decode_throughput': {
                'optimistic': round(throughput['tps_optimistic'], 1),
                'realistic': round(throughput['tps_realistic'], 1),
                'conservative': round(throughput['tps_conservative'], 1),
                'memory_bound': round(throughput['tps_memory_bound'], 1),
                'compute_bound': round(throughput['tps_compute_bound'], 1),
                'tag': '[hypothesis: draft-model-analysis]',
            },
            'speedups': speedups,
        }
        results.append(r)

    # Summary table
    print("=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print()
    print(f"{'Model':<18} {'Params':<10} {'F16 MB':<10} {'Opt t/s':<12} {'Real t/s':<12} {'Cons t/s':<12} {'Spec t/s':<12}")
    print(f"{'─'*18} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")

    for r in results:
        spec = r['speedups'].get('code_default', {}).get('spec_tps', 0)
        print(f"{r['name']:<18} {r['params']['total_m']:<8.0f}M "
              f"{r['params']['weights_mb_f16']:<8.0f} "
              f"{r['decode_throughput']['optimistic']:<10.0f} "
              f"{r['decode_throughput']['realistic']:<10.0f} "
              f"{r['decode_throughput']['conservative']:<10.0f} "
              f"{spec:<10.0f}")

    print()
    print("Notes:")
    print("  t/s values are [hypothesis: draft-model-analysis]")
    print("  Spec t/s uses realistic draft t/s + batch-verify (8.6ms) + code-default acceptance (0.05/tok)")
    print("  WARNING: All acceptance rates > 0.55 falsified by measured 5%. See PRD-2 §O2.X")
    print("  To falsify: measure actual draft model t/s on this system")
    print()

    # Write JSON
    output = {
        'description': 'Draft model design space analysis',
        'method': 'Analytical model calibrated against measured HBM BW (compute_peak_bench) and BW utilization (q4_dequant_bench)',
        'system': '2× RTX PRO 6000 Blackwell 96GB Max-Q',
        'hbm_bw_gbs': HBM_READ_BW,
        'f16_peak_tflops': PEAK_F16_TFLOPS,
        'draft_configs': results,
        'tag': '[hypothesis: draft-model-analysis]',
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'draft_model_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main()
