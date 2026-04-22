#!/usr/bin/env python3
"""
Plot benchmark results from a JSON file produced by run_bench.py.

Usage:
    python scripts/plot_results.py results/mixed_spike.json --out results/mixed_spike.png
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


COLORS = {"perc": "#2ecc71", "lru": "#e74c3c", "priority": "#3498db"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    with args.results.open() as f:
        data = json.load(f)

    policies = list(data.keys())
    x = np.arange(len(policies))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"CachePilot Benchmark — {args.results.stem}", fontsize=14, fontweight="bold")

    # Throughput
    tps = [data[p]["throughput_tok_s"] for p in policies]
    bars = axes[0].bar(x, tps, color=[COLORS.get(p, "#95a5a6") for p in policies], width=0.5)
    axes[0].set_title("Throughput (tok/s)")
    axes[0].set_ylabel("Tokens / second")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([p.upper() for p in policies])
    for bar, val in zip(bars, tps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # p95 TPOT
    p95 = [data[p]["p95_tpot_ms"] for p in policies]
    bars2 = axes[1].bar(x, p95, color=[COLORS.get(p, "#95a5a6") for p in policies], width=0.5)
    axes[1].set_title("p95 TPOT (ms)")
    axes[1].set_ylabel("Milliseconds")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([p.upper() for p in policies])
    for bar, val in zip(bars2, p95):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # Eviction events
    evict = [data[p]["eviction_events"] for p in policies]
    bars3 = axes[2].bar(x, evict, color=[COLORS.get(p, "#95a5a6") for p in policies], width=0.5)
    axes[2].set_title("Eviction Events")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([p.upper() for p in policies])
    for bar, val in zip(bars3, evict):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=150)
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
