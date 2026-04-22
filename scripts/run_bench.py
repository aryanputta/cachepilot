#!/usr/bin/env python3
"""
Run a benchmark from a YAML config file.

Usage:
    python scripts/run_bench.py benchmarks/mixed_spike.yaml
    python scripts/run_bench.py benchmarks/mixed_spike.yaml --out results/mixed_spike.json
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cachepilot.engine import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="YAML benchmark config")
    parser.add_argument("--out", type=Path, default=None, help="JSON output path")
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", args.config.stem)
    print(f"\n=== {name} ===\n")

    results = {}
    for policy in cfg.get("policies", ["perc"]):
        print(f"  Running {policy}...", end=" ", flush=True)
        r = run(
            policy=policy,
            workload=cfg.get("workload", "mixed"),
            n_requests=cfg.get("requests", 1000),
            arrival_rate=cfg.get("arrival_rate", 10.0),
            vram_gb=cfg.get("vram_gb", 24.0),
            seed=cfg.get("seed", 42),
            spike_at=cfg.get("spike_at"),
            spike_multiplier=cfg.get("spike_multiplier", 4.0),
            batch_mode=cfg.get("batch_mode", "adaptive"),
        )
        results[policy] = r.as_dict()
        print(f"done — {r.throughput_tok_s:.1f} tok/s, p95={r.p95_tpot_ms:.1f}ms, evictions={r.eviction_events}")

    print()
    # Summary table
    lru_tps = results.get("lru", {}).get("throughput_tok_s", 1.0)
    header = f"{'Policy':12s}  {'Throughput':>12s}  {'vs LRU':>8s}  {'p95 TPOT':>10s}  {'Evictions':>10s}  {'Dropped':>8s}"
    print(header)
    print("-" * len(header))
    for policy, r in results.items():
        rel = r["throughput_tok_s"] / max(lru_tps, 1e-6)
        print(
            f"{policy:12s}  {r['throughput_tok_s']:>10.1f}  {rel:>8.3f}x  "
            f"{r['p95_tpot_ms']:>8.2f}ms  {r['eviction_events']:>10d}  {r['requests_dropped']:>8d}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
