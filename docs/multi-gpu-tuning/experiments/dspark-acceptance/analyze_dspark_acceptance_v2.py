#!/usr/bin/env python3
"""
DSpark Acceptance Rate — Analysis v2

Parses ds4 log files from n=1024 DSpark runs to extract acceptance rate,
timing breakdown, and speedup projections.

Usage:
    python3 analyze_dspark_acceptance_v2.py --results-dir /path/to/logs
    python3 analyze_dspark_acceptance_v2.py  # auto-detects own directory

GROUD-RULES: All values tagged per §1.2.
"""

import os
import re
import sys
import json
import argparse
import glob


def parse_dspark_stats(line):
    """Parse a DSpark stats line into structured dict."""
    stats = {}
    
    patterns = {
        'cycles': r'cycles=(\d+)',
        'first_tokens': r'first_tokens=(\d+)',
        'proposed': r'proposed=(\d+)',
        'accepted_draft': r'accepted_draft=(\d+)',
        'accept_rate': r'accept_rate=([\d.]+)',
        'avg_accept': r'avg_accept=([\d.]+)',
        'full': r'full=(\d+)',
        'partial': r'partial=(\d+)',
        'miss_first': r'miss_first=(\d+)',
        'no_draft': r'no_draft=(\d+)',
        'no_room': r'no_room=(\d+)',
        'invalid': r'invalid=(\d+)',
        'propose_ms': r'propose=([\d.]+)',
        'prop_cache_ms': r'prop_cache=([\d.]+)',
        'prop_chain_ms': r'prop_chain=([\d.]+)',
        'verify_ms': r'verify=([\d.]+)',
        'target_ms': r'target=([\d.]+)',
        'saved_ms': r'saved=([\d.]+)',
        'net_saved_ms': r'net_saved=([\d.]+)',
        'spec_total_ms': r'spec_total=([\d.]+)',
    }
    
    for key, pat in patterns.items():
        m = re.search(pat, line)
        if m:
            stats[key] = float(m.group(1))
    
    # Determine if capture completed
    stats['capture_completed'] = stats.get('cycles', 0) > 0
    
    return stats


def analyze_log(log_path):
    """Analyze one DSpark log file."""
    with open(log_path) as f:
        content = f.read()
    
    prompt_name = os.path.basename(log_path).replace('_dspark_n1024.log', '').replace('_dspark.log', '')
    
    result = {
        'prompt': prompt_name,
        'log_file': os.path.basename(log_path),
        'tag': '[measured: DSpark v2 run]',
    }
    
    # Parse prefill/generation
    prefill_m = re.search(r'prefill:\s*([\d.]+)\s*t/s', content)
    gen_m = re.search(r'generation:\s*([\d.]+)\s*t/s', content)
    if prefill_m:
        result['prefill_tps'] = float(prefill_m.group(1))
    if gen_m:
        result['gen_tps'] = float(gen_m.group(1))
    
    # Parse DSpark stats
    stats_lines = re.findall(r'ds4: DSpark stats .*', content)
    if stats_lines:
        result['dspark'] = parse_dspark_stats(stats_lines[-1])
        result['dspark_found'] = True
    else:
        result['dspark'] = {'capture_completed': False}
        result['dspark_found'] = False
    
    # Parse draft length histogram
    draft_len = re.search(r'draft_len_hist=([\d,:]+)', content)
    if draft_len:
        pairs = draft_len.group(1).split(',')
        hist = {}
        for p in pairs:
            if ':' in p:
                k, v = p.split(':')
                hist[int(k)] = int(v)
        result['draft_len_histogram'] = hist
    
    accepted_len = re.search(r'accepted_len_hist=([\d,:]+)', content)
    if accepted_len:
        pairs = accepted_len.group(1).split(',')
        hist = {}
        for p in pairs:
            if ':' in p:
                k, v = p.split(':')
                hist[int(k)] = int(v)
        result['accepted_len_histogram'] = hist
    
    return result


def analyze_all(results_dir):
    """Analyze all DSpark log files in a directory."""
    # Check for v2 logs first
    log_files = sorted(glob.glob(os.path.join(results_dir, '*_dspark_n1024.log')))
    
    if not log_files:
        # Fall back to v1 logs
        log_files = sorted(glob.glob(os.path.join(results_dir, '*_dspark.log')))
    
    results = []
    for lf in log_files:
        result = analyze_log(lf)
        results.append(result)
    
    return results


def compute_speedup_projections(results):
    """Compute speculative decode speedup from measured params."""
    projections = []
    
    for r in results:
        dspark = r.get('dspark', {})
        if not dspark.get('capture_completed', False):
            projections.append({
                'prompt': r['prompt'],
                'status': 'no_capture',
                'reason': 'DSpark capture did not complete during generation window'
            })
            continue
        
        t_draft_ms = dspark.get('prop_chain_ms', 0)
        t_verify_ms = dspark.get('verify_ms', 0)
        t_target_ms = dspark.get('target_ms', 0)
        t_propose_ms = dspark.get('propose_ms', 0)
        
        # t_spec_step = t_propose + t_verify + t_prop_cache (amortized)
        cycles = dspark.get('cycles', 1)
        t_cache_amort = dspark.get('prop_cache_ms', 0) / cycles if cycles > 0 else 0
        
        t_step_ms = t_draft_ms + t_verify_ms
        accept_rate = dspark.get('accept_rate', 0) / 100.0  # convert from %
        avg_accept = dspark.get('avg_accept', 0)
        
        # Chain length
        block_size = 5  # DSpark block size from GGUF metadata
        
        # Expected accepted tokens = avg_accept per cycle (from stats) OR geometric series
        if avg_accept > 0:
            expected_accepted = avg_accept
        else:
            # Estimate from acceptance rate
            # E[accepted] = (1 - p^block) / (1-p)
            if accept_rate < 1.0:
                expected_accepted = (1 - accept_rate ** block_size) / (1 - accept_rate)
            else:
                expected_accepted = block_size
        
        spec_tps = expected_accepted / (t_step_ms / 1000.0) if t_step_ms > 0 else 0
        
        projections.append({
            'prompt': r['prompt'],
            'status': 'measured',
            't_draft_ms': round(t_draft_ms, 3),
            't_verify_ms': round(t_verify_ms, 3),
            't_step_ms': round(t_step_ms, 3),
            't_cache_amort_ms': round(t_cache_amort, 3),
            'accept_rate_pct': round(accept_rate * 100, 2),
            'avg_accepted': round(expected_accepted, 3),
            'block_size': block_size,
            'spec_tps': round(spec_tps, 1),
            'base_tps': r.get('gen_tps', 0),
            'speedup': round(spec_tps / r.get('gen_tps', 1), 2) if r.get('gen_tps', 0) > 0 else 0,
            'net_saved_ms': dspark.get('net_saved_ms', 0),
        })
    
    return projections


def main():
    parser = argparse.ArgumentParser(description='DSpark Acceptance Rate Analysis v2')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory containing DSpark log files')
    args = parser.parse_args()
    
    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 72)
    print("DSpark Acceptance Rate — Analysis v2")
    print("=" * 72)
    print()
    
    results = analyze_all(results_dir)
    
    if not results:
        print("No DSpark log files found in", results_dir)
        print("Run run_dspark_acceptance_v2.sh first to generate logs.")
        return 1
    
    # Print per-prompt analysis
    for r in results:
        print(f"\n{'─'*72}")
        print(f"Prompt: {r['prompt']}")
        print(f"{'─'*72}")
        print(f"  Prefill: {r.get('prefill_tps', 'N/A')} t/s")
        print(f"  Generation: {r.get('gen_tps', 'N/A')} t/s")
        
        dspark = r.get('dspark', {})
        if not dspark.get('capture_completed', False):
            print(f"  DSpark: CAPTURE NOT COMPLETED")
            print(f"  [measured: no DSpark stats found in log]")
            if dspark_found := r.get('dspark_found', False):
                print(f"  DSpark line found but stats may be incomplete")
            continue
        
        print(f"  DSpark: ACTIVE")
        print(f"  Cycles: {dspark.get('cycles', 'N/A')}")
        print(f"  Proposed: {dspark.get('proposed', 'N/A')}")
        print(f"  Accepted: {dspark.get('accepted_draft', 'N/A')}")
        print(f"  Accept rate: {dspark.get('accept_rate', 'N/A')}%")
        print(f"  Avg accept/cycle: {dspark.get('avg_accept', 'N/A')}")
        print(f"  Timing breakdown:")
        print(f"    Propose:     {dspark.get('propose_ms', 0):>10.3f} ms")
        print(f"    Prop cache:  {dspark.get('prop_cache_ms', 0):>10.3f} ms")
        print(f"    Prop chain:  {dspark.get('prop_chain_ms', 0):>10.3f} ms")
        print(f"    Verify:      {dspark.get('verify_ms', 0):>10.3f} ms")
        print(f"    Target:      {dspark.get('target_ms', 0):>10.3f} ms")
        print(f"    Net saved:   {dspark.get('net_saved_ms', 0):>10.3f} ms")
        
        # Histograms
        draft_hist = r.get('draft_len_histogram', {})
        if draft_hist:
            print(f"  Draft length histogram: {draft_hist}")
        accepted_hist = r.get('accepted_len_histogram', {})
        if accepted_hist:
            print(f"  Accepted length histogram: {accepted_hist}")
    
    # Speedup projections
    print(f"\n\n{'='*72}")
    print("SPEEDUP PROJECTIONS (from measured DSpark values)")
    print('='*72)
    
    projections = compute_speedup_projections(results)
    
    print(f"\n{'Prompt':<20} {'Accept%':<10} {'AvgAcc':<10} {'t_step(ms)':<12} {'Spec t/s':<12} {'Base t/s':<10} {'Speedup':<10}")
    print(f"{'─'*20} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    
    for p in projections:
        if p['status'] == 'no_capture':
            print(f"{p['prompt']:<20} {'—':<10} {'—':<10} {'—':<12} {'—':<12} {'—':<10} {'—':<10}")
            continue
        
        print(f"{p['prompt']:<20} {p['accept_rate_pct']:<8.1f}% {p['avg_accepted']:<8.2f} "
              f"{p['t_step_ms']:<10.3f} {p['spec_tps']:<10.1f} {p['base_tps']:<8.1f} {p['speedup']:<8.2f}")
    
    # Write JSON results
    output = {
        'date': None,  # will be set from logs
        'description': 'DSpark acceptance rate on speed-bench prompts with n=1024',
        'method': 'ds4 --dspark --mtp with DS4_DSPARK_STATS=1, n=1024 tokens, temp=0, seed=42',
        'system': '2× RTX PRO 6000 Blackwell 96GB Max-Q, CUDA 12.9, PCIe Gen 5 x8',
        'results': [],
        'speedup_projections': projections,
    }
    
    for r in results:
        prompt_result = {
            'prompt': r['prompt'],
            'prefill_tps': r.get('prefill_tps'),
            'gen_tps': r.get('gen_tps'),
            'dspark_stats': r.get('dspark'),
            'draft_len_histogram': r.get('draft_len_histogram'),
            'accepted_len_histogram': r.get('accepted_len_histogram'),
            'tag': r['tag'],
        }
        output['results'].append(prompt_result)
    
    output_path = os.path.join(results_dir, 'dspark_acceptance_results_v2.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFull results saved to: {output_path}")
    print(f"\n{'='*72}")
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
