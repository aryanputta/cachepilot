from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .engine import POLICY_MAP, WORKLOAD_PRESETS, run

app = typer.Typer(name="cachepilot", add_completion=False, help="CachePilot GPU memory orchestrator.")
console = Console()


@app.command()
def bench(
    policy: str = typer.Option("perc", help="perc | lru | priority"),
    workload: str = typer.Option("mixed", help="chat | code | summarize | longctx | mixed"),
    requests: int = typer.Option(1000, help="number of requests"),
    arrival_rate: float = typer.Option(10.0, help="requests/sec"),
    vram_gb: float = typer.Option(24.0, help="simulated VRAM in GB"),
    seed: int = typer.Option(42),
    spike: Optional[int] = typer.Option(None, help="request index for traffic spike"),
    batch_mode: str = typer.Option("adaptive", help="max_throughput | low_latency | adaptive"),
    out: Optional[Path] = typer.Option(None, help="write JSON results to this path"),
) -> None:
    """Run a single benchmark with one eviction policy."""
    if policy not in POLICY_MAP:
        console.print(f"[red]Unknown policy '{policy}'. Choose from: {list(POLICY_MAP)}[/red]")
        raise typer.Exit(1)
    if workload not in WORKLOAD_PRESETS:
        console.print(f"[red]Unknown workload '{workload}'. Choose from: {list(WORKLOAD_PRESETS)}[/red]")
        raise typer.Exit(1)

    with console.status(f"[bold green]Running {policy.upper()} on {workload} ({requests} requests)..."):
        result = run(
            policy=policy,
            workload=workload,
            n_requests=requests,
            arrival_rate=arrival_rate,
            vram_gb=vram_gb,
            seed=seed,
            spike_at=spike,
            batch_mode=batch_mode,
        )

    table = Table(title=f"CachePilot — {policy.upper()} | {workload} | {requests} requests")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Requests Served", str(result.requests_served))
    table.add_row("Requests Dropped", str(result.requests_dropped))
    table.add_row("Tokens Generated", f"{result.tokens_total:,}")
    table.add_row("Throughput (tok/s)", f"{result.throughput_tok_s:.1f}")
    table.add_row("p50 TPOT (ms)", f"{result.p50_tpot_ms:.2f}")
    table.add_row("p95 TPOT (ms)", f"{result.p95_tpot_ms:.2f}")
    table.add_row("p99 TPOT (ms)", f"{result.p99_tpot_ms:.2f}")
    table.add_row("Eviction Events", str(result.eviction_events))
    table.add_row("Total Eviction Cost (s)", f"{result.total_eviction_cost_s:.3f}")
    table.add_row("Mean Cost/Eviction (s)", f"{result.mean_eviction_cost_s:.4f}")
    table.add_row("VRAM Util (mean)", f"{result.vram_util_mean:.1%}")
    table.add_row("VRAM Util (peak)", f"{result.vram_util_peak:.1%}")
    table.add_row("Wall Time (s)", f"{result.wall_time_s:.2f}")

    console.print(table)

    if out:
        out.write_text(json.dumps(result.as_dict(), indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


@app.command()
def compare(
    workload: str = typer.Option("mixed"),
    requests: int = typer.Option(2000),
    arrival_rate: float = typer.Option(10.0),
    vram_gb: float = typer.Option(24.0),
    seed: int = typer.Option(42),
    spike: Optional[int] = typer.Option(500, help="traffic spike at request N"),
    out: Optional[Path] = typer.Option(None),
) -> None:
    """Compare PERC vs LRU vs Priority eviction side-by-side."""
    results = {}
    for policy in ["perc", "lru", "priority"]:
        with console.status(f"[bold green]Running {policy}..."):
            results[policy] = run(
                policy=policy,
                workload=workload,
                n_requests=requests,
                arrival_rate=arrival_rate,
                vram_gb=vram_gb,
                seed=seed,
                spike_at=spike,
            )

    lru_tps = results["lru"].throughput_tok_s
    lru_lat = results["lru"].p95_tpot_ms

    table = Table(title=f"CachePilot Comparison — {workload} | {requests} requests | spike@{spike}")
    table.add_column("Policy", style="bold")
    table.add_column("Throughput", justify="right")
    table.add_column("vs LRU", justify="right")
    table.add_column("p95 TPOT", justify="right")
    table.add_column("Lat Improv", justify="right")
    table.add_column("Evictions", justify="right")
    table.add_column("Evict Cost (s)", justify="right")
    table.add_column("vs LRU cost", justify="right")
    table.add_column("Dropped", justify="right")
    table.add_column("VRAM Util", justify="right")

    lru_cost = results["lru"].total_eviction_cost_s if "lru" in results else 1.0

    for policy, r in results.items():
        tps_rel = r.throughput_tok_s / max(lru_tps, 1e-6)
        lat_rel = lru_lat / max(r.p95_tpot_ms, 1e-6)
        cost_rel = r.total_eviction_cost_s / max(lru_cost, 1e-9)
        tps_color = "green" if tps_rel > 1.02 else "red" if tps_rel < 0.98 else "white"
        lat_color = "green" if lat_rel > 1.02 else "red" if lat_rel < 0.98 else "white"
        cost_color = "green" if cost_rel < 0.98 else "red" if cost_rel > 1.02 else "white"
        table.add_row(
            policy.upper(),
            f"{r.throughput_tok_s:.1f} tok/s",
            f"[{tps_color}]{tps_rel:.3f}x[/{tps_color}]",
            f"{r.p95_tpot_ms:.2f} ms",
            f"[{lat_color}]{lat_rel:.3f}x[/{lat_color}]",
            str(r.eviction_events),
            f"{r.total_eviction_cost_s:.3f}s",
            f"[{cost_color}]{cost_rel:.3f}x[/{cost_color}]",
            str(r.requests_dropped),
            f"{r.vram_util_mean:.1%}",
        )

    console.print(table)

    if out:
        out.write_text(json.dumps({p: r.as_dict() for p, r in results.items()}, indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


def main() -> None:
    app()
