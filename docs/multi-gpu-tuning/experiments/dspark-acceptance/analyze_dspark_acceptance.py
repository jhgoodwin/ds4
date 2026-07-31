#!/usr/bin/env python3
"""
DSpark Acceptance Rate — Analytical Framework

Measures DSpark acceptance rate by parsing available data from generation runs.
When DSpark stats become available (capture completes), this script will compute:
- Acceptance rate per token
- Average accepted per cycle
- Timing breakdown

Current status: DSpark hidden-state capture does not complete during decode on
this system. Using analytical predictions from PRD-2 §O3.3 instead.

Usage:
    python3 analyze_dspark_acceptance.py [--stats-file LOG_FILE]

GROUD-RULES: All values tagged per §1.2.
"""

import os
import re
import sys
import json

def parse_dspark_stats_line(line):
    """Parse a 'ds4: DSpark stats' line into a dict."""
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
        'no_draft': r'no_draft=(\d+)',
        'propose_ms': r'propose=([\d.]+)',
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
    
    return stats


def analyze_log(log_path):
    """Analyze a ds4 log file for DSpark stats."""
    with open(log_path) as f:
        content = f.read()
    
    # Find DSpark stats lines
    stats_lines = re.findall(r'ds4: DSpark stats .*', content)
    
    if not stats_lines:
        print(f"  No DSpark stats found in {os.path.basename(log_path)}")
        print(f"  Reason: DSpark hidden-state capture did not complete during decode [measured: no capture completion]")
        print(f"  See dspark-profile README for explanation.")
        return None
    
    stats = parse_dspark_stats_line(stats_lines[-1])
    return stats


def analytical_predictions():
    """Analytical predictions from PRD-2 §O3.3."""
    predictions = {
        "django-varbit": {
            "expected_acceptance_rate": 0.75,
            "range": [0.70, 0.85],
            "tag": "[hypothesis: PRD-2 §O3.3, O2.X predictability = 64%]",
            "expected_avg_accepted": 3.3,
            "acceptance_range": [2.8, 5.0],
        },
        "slack-clone": {
            "expected_acceptance_rate": 0.80,
            "range": [0.75, 0.85],
            "tag": "[hypothesis: PRD-2 §O3.3, O2.X predictability = 68%]",
            "expected_avg_accepted": 4.0,
            "acceptance_range": [3.3, 5.7],
        },
        "flappy-bird": {
            "expected_acceptance_rate": 0.55,
            "range": [0.50, 0.65],
            "tag": "[hypothesis: PRD-2 §O3.3, O2.X predictability = 42%]",
            "expected_avg_accepted": 2.0,
            "acceptance_range": [1.8, 2.8],
        },
    }
    return predictions


def main():
    results_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 72)
    print("DSpark Acceptance Rate — Analytical Framework")
    print("=" * 72)
    print()
    
    # Check for log files
    log_files = [f for f in os.listdir(results_dir) if f.endswith('_dspark.log')]
    
    if log_files:
        print("Found DSpark log files:")
        for lf in sorted(log_files):
            log_path = os.path.join(results_dir, lf)
            stats = analyze_log(log_path)
            if stats:
                prompt = lf.replace('_dspark.log', '')
                print(f"\n  {prompt}:")
                for k, v in stats.items():
                    print(f"    {k}: {v}")
    else:
        print("No DSpark log files found. Running with analytical predictions.")
    
    print()
    print()
    print("─" * 72)
    print("ANALYTICAL PREDICTIONS (from PRD-2 §O3.3 + O2.X)")
    print("─" * 72)
    print()
    print(f"{'Prompt':<20} {'Predictability':<18} {'Accept Rate':<15} {'Avg Accepted':<15} {'Source':<40}")
    print(f"{'─'*20} {'─'*18} {'─'*15} {'─'*15} {'─'*40}")
    
    predictions = analytical_predictions()
    for pname, pred in predictions.items():
        o2x_results = {
            "django-varbit": {"predictable": 0.64},
            "slack-clone": {"predictable": 0.68},
            "flappy-bird": {"predictable": 0.42},
        }
        o2x = o2x_results.get(pname, {})
        pred_pct = o2x.get("predictable", 0) * 100
        
        print(f"{pname:<20} {pred_pct:>5.0f}% predictable{'':8} "
              f"{pred['expected_acceptance_rate']:.2f}/tok ({pred['range'][0]:.2f}-{pred['range'][1]:.2f}) "
              f"{pred['expected_avg_accepted']:.1f} ({pred['acceptance_range'][0]:.1f}-{pred['acceptance_range'][1]:.1f}) "
              f"{pred['tag'][:40]}")
    
    print()
    print()
    print("─" * 72)
    print("SPEEDUP PROJECTION (at draft chain=5, t_spec_step=11.7ms)")
    print("─" * 72)
    print()
    
    # Chain=8 is the DSpark block size
    chain_length = 5  # conservative: DSpark block_size=5
    t_spec_step_ms = 11.7  # [derived: PRD-2 §O3.4]
    
    print(f"{'Prompt':<20} {'Accept Rate':<15} {'Spec t/s':<15} {'Free Tokens':<15} {'Total t/s':<15}")
    print(f"{'─'*20} {'─'*15} {'─'*15} {'─'*15} {'─'*15}")
    
    free_tokens = {
        "django-varbit": 30,
        "slack-clone": 50,
        "flappy-bird": 15,
    }
    
    for pname, pred in predictions.items():
        ar = pred['expected_acceptance_rate']
        # Expected accepted = (1 - p^chain) / (1-p)
        expected_acc = (1 - ar**chain_length) / (1 - ar) if ar < 1.0 else chain_length
        spec_tps = expected_acc / (t_spec_step_ms / 1000.0)
        ft = free_tokens.get(pname, 0)
        total_tps = (expected_acc + ft) / (t_spec_step_ms / 1000.0)
        
        print(f"{pname:<20} {ar:.2f}/tok{'':8} {spec_tps:>7.0f}{'':8} {ft:>3d}{'':12} {total_tps:>7.0f}")
    
    print()
    print("All predicted values [hypothesis: based on PRD-2 §O3.3]. Falsifying experiment:")
    print("  Requires DSpark capture to complete during decode. Fix: debug why")
    print("  g->dspark_capture_valid never becomes true on this system.")
    print()
    
    # Write results
    results = {
        "status": "analytical_only",
        "reason": "DSpark hidden-state capture does not complete during decode [measured: no capture complete]",
        "fix_hint": "Check g->dspark_capture_valid initialization in ds4_gpu_graph init. "
                    "Target layers 40-42 may not be reached in the decode graph.",
        "analytical_predictions": predictions,
    }
    
    output_path = os.path.join(results_dir, "dspark_acceptance_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
