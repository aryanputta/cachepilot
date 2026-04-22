from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from .cost_model import MODEL_CATALOG
from .dataset_profile import (
    HF_DATASET_PRESETS,
    KAGGLE_DATASET_SUGGESTIONS,
    profile_hf_dataset,
    profile_local_dataset,
)
from .engine import POLICY_MAP, WORKLOAD_PRESETS, run
from .grafana import write_dashboard
from .model_compare import candidate_advantages, compare_vllm_models, load_prompts
from .policy.rl_policy import fine_tune_admission_policy
from .scorecard import compare_hardware_scorecards, resolve_model_spec
from .telemetry_export import LiveTelemetryExporter
from .vllm_benchmark import compare_vllm_backends, load_prompt_set, render_hf_vllm_uv_script

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
    kv_tier: str = typer.Option("fp16", help="fp16 | int8 | fp8"),
    prometheus_out: Optional[Path] = typer.Option(None, help="write Prometheus metrics to this path"),
    snapshots_out: Optional[Path] = typer.Option(None, help="write telemetry snapshots JSON"),
    out: Optional[Path] = typer.Option(None, help="write JSON results to this path"),
) -> None:
    """Run a single benchmark with one eviction policy."""
    if policy not in POLICY_MAP:
        console.print(f"[red]Unknown policy '{policy}'. Choose from: {list(POLICY_MAP)}[/red]")
        raise typer.Exit(1)
    if workload not in WORKLOAD_PRESETS:
        console.print(f"[red]Unknown workload '{workload}'. Choose from: {list(WORKLOAD_PRESETS)}[/red]")
        raise typer.Exit(1)

    exporter = None
    telemetry_listener = None
    if prometheus_out or snapshots_out:
        exporter = LiveTelemetryExporter(labels={"policy": policy, "kv_tier": kv_tier})
        telemetry_listener = exporter.update

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
            kv_tier=kv_tier,
            telemetry_listener=telemetry_listener,
        )

    table = Table(title=f"CachePilot — {policy.upper()} | {workload} | {requests} requests")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")

    table.add_row("Requests Served", str(result.requests_served))
    table.add_row("Requests Dropped", str(result.requests_dropped))
    table.add_row("Requests Deferred", str(result.requests_deferred))
    table.add_row("KV Tier", result.kv_tier.upper())
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
    if exporter and prometheus_out:
        exporter.write_prometheus(prometheus_out)
        console.print(f"[dim]Prometheus metrics written to {prometheus_out}[/dim]")
    if exporter and snapshots_out:
        exporter.write_snapshots_json(snapshots_out)
        console.print(f"[dim]Telemetry snapshots written to {snapshots_out}[/dim]")


@app.command()
def compare(
    workload: str = typer.Option("mixed"),
    requests: int = typer.Option(2000),
    arrival_rate: float = typer.Option(10.0),
    vram_gb: float = typer.Option(24.0),
    seed: int = typer.Option(42),
    spike: Optional[int] = typer.Option(500, help="traffic spike at request N"),
    kv_tier: str = typer.Option("fp16", help="fp16 | int8 | fp8"),
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
                kv_tier=kv_tier,
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


@app.command("rl-admission")
def rl_admission(
    workload: str = typer.Option("mixed"),
    requests: int = typer.Option(400),
    arrival_rate: float = typer.Option(10.0),
    vram_gb: float = typer.Option(16.0),
    kv_tier: str = typer.Option("fp16"),
    episodes: int = typer.Option(12),
    seed: int = typer.Option(42),
    out: Optional[Path] = typer.Option(None),
) -> None:
    result = fine_tune_admission_policy(
        episodes=episodes,
        workload=workload,
        n_requests=requests,
        arrival_rate=arrival_rate,
        vram_gb=vram_gb,
        kv_tier=kv_tier,
        seed=seed,
    )

    table = Table(title=f"Admission RL Fine-Tuning — {workload} | {episodes} episodes")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")
    table.add_row("Baseline Reward", f"{result.baseline_reward:.2f}")
    table.add_row("Tuned Reward", f"{result.tuned_reward:.2f}")
    table.add_row("Improvement", f"{result.improvement_pct:.2f}%")
    table.add_row("Baseline Drop Rate", f"{result.baseline_drop_rate:.2%}")
    table.add_row("Tuned Drop Rate", f"{result.tuned_drop_rate:.2%}")
    table.add_row("Baseline Evict Cost", f"{result.baseline_eviction_cost_s:.3f}s")
    table.add_row("Tuned Evict Cost", f"{result.tuned_eviction_cost_s:.3f}s")
    console.print(table)

    if out:
        out.write_text(json.dumps(result.__dict__, indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


@app.command("grafana-dashboard")
def grafana_dashboard(
    out: Path = typer.Option(..., help="write dashboard JSON to this path"),
) -> None:
    write_dashboard(out)
    console.print(f"[dim]Grafana dashboard written to {out}[/dim]")


@app.command("compare-models")
def compare_models(
    candidate: str = typer.Option(..., help="candidate model ID or local path"),
    baseline: List[str] = typer.Option([], help="baseline model IDs or local paths"),
    prompts: Path | None = typer.Option(None, help="newline-delimited or JSON prompt file"),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Hugging Face dataset ID for prompt sampling",
    ),
    local_dataset: Path | None = typer.Option(
        None,
        "--local-dataset",
        help="Local CSV/JSONL/JSON/Parquet dataset export for prompt sampling",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=f"Named HF prompt preset: {', '.join(sorted(HF_DATASET_PRESETS))}",
    ),
    split: str = typer.Option("train", help="dataset split for --hf-dataset/--preset"),
    config: str | None = typer.Option(None, help="dataset config for --hf-dataset/--preset"),
    limit: int = typer.Option(64, help="number of prompts to sample from dataset sources"),
    max_tokens: int = typer.Option(64),
    gpu_memory_utilization: float = typer.Option(0.8),
    perc: bool = typer.Option(False, "--perc/--no-perc", help="run all models with the PERC patch"),
    max_model_len: int | None = typer.Option(None, help="optional vLLM max model length"),
    tensor_parallel_size: int = typer.Option(1, help="vLLM tensor parallel size"),
    out: Optional[Path] = typer.Option(None),
) -> None:
    models = [candidate, *baseline]
    if prompts is not None:
        prompt_list = load_prompts(prompts)
        prompt_source = str(prompts)
        prompt_schema = "prompt_file"
    else:
        prompt_set = load_prompt_set(
            hf_dataset=hf_dataset,
            local_dataset=local_dataset,
            preset=preset,
            split=split,
            config=config,
            limit=limit,
        )
        prompt_list = prompt_set.prompts
        prompt_source = prompt_set.label
        prompt_schema = prompt_set.schema
    results = compare_vllm_models(
        models=models,
        prompts=prompt_list,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        use_perc_evictor=perc,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        prompt_source=prompt_source,
        prompt_schema=prompt_schema,
    )
    result_map = {result.model: result for result in results}
    candidate_result = result_map[candidate]
    baseline_results = [result_map[name] for name in baseline]

    table = Table(title="vLLM Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("Engine", justify="right")
    table.add_column("Prompt Tok", justify="right")
    table.add_column("Gen Tokens", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Tok/s", justify="right")
    for result in results:
        table.add_row(
            result.model,
            result.engine,
            str(result.prompt_tokens),
            str(result.generated_tokens),
            f"{result.wall_time_s:.2f}s",
            f"{result.tokens_per_second:.2f}",
        )
    console.print(table)

    advantages = candidate_advantages(candidate_result, baseline_results)
    if advantages:
        console.print(
            "[bold green]Candidate wins:[/bold green] " + ", ".join(advantages)
        )
    else:
        console.print("[yellow]No across-the-board advantage detected for the candidate.[/yellow]")

    if out:
        payload = {
            "prompt_source": prompt_source,
            "prompt_schema": prompt_schema,
            "results": [result.as_dict() for result in results],
            "candidate_advantages": advantages,
        }
        out.write_text(json.dumps(payload, indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


@app.command("vllm-benchmark")
def vllm_benchmark(
    model: str = typer.Option(..., help="model ID or local path on the CUDA host"),
    prompts: Path | None = typer.Option(None, help="newline-delimited or JSON prompt file"),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Hugging Face dataset ID for prompt sampling",
    ),
    local_dataset: Path | None = typer.Option(
        None,
        "--local-dataset",
        help="Local CSV/JSONL/JSON/Parquet dataset export for prompt sampling",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=f"Named HF prompt preset: {', '.join(sorted(HF_DATASET_PRESETS))}",
    ),
    split: str = typer.Option("train"),
    config: str | None = typer.Option(None),
    limit: int = typer.Option(64, help="number of prompts to sample from dataset sources"),
    compare_perc: bool = typer.Option(
        True,
        "--compare-perc/--no-compare-perc",
        help="benchmark plain vLLM and vLLM+PERC side-by-side",
    ),
    max_tokens: int = typer.Option(64),
    gpu_memory_utilization: float = typer.Option(0.8),
    max_model_len: int | None = typer.Option(None, help="optional vLLM max model length"),
    tensor_parallel_size: int = typer.Option(1, help="vLLM tensor parallel size"),
    out: Path | None = typer.Option(None),
) -> None:
    if prompts is not None:
        prompt_set = load_prompt_set(prompts_path=prompts)
    else:
        prompt_set = load_prompt_set(
            hf_dataset=hf_dataset,
            local_dataset=local_dataset,
            preset=preset,
            split=split,
            config=config,
            limit=limit,
        )

    results = compare_vllm_backends(
        model=model,
        prompt_set=prompt_set,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        compare_perc=compare_perc,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    table = Table(title=f"vLLM CUDA Benchmark — {model}")
    table.add_column("Engine", style="cyan")
    table.add_column("Prompt Count", justify="right")
    table.add_column("Prompt Tok", justify="right")
    table.add_column("Gen Tokens", justify="right")
    table.add_column("Wall Time", justify="right")
    table.add_column("Tok/s", justify="right")
    for result in results:
        table.add_row(
            result.engine,
            str(result.prompt_count),
            str(result.prompt_tokens),
            str(result.generated_tokens),
            f"{result.wall_time_s:.2f}s",
            f"{result.tokens_per_second:.2f}",
        )
    console.print(table)

    if len(results) == 2:
        baseline, patched = results
        tps_rel = patched.tokens_per_second / max(baseline.tokens_per_second, 1e-6)
        wall_rel = baseline.wall_time_s / max(patched.wall_time_s, 1e-6)
        console.print(
            f"[bold]PERC delta:[/bold] throughput {tps_rel:.3f}x, latency {wall_rel:.3f}x "
            f"on {prompt_set.prompt_count} prompts from {prompt_set.label}"
        )

    if out:
        payload = {
            "prompt_source": prompt_set.as_dict(),
            "results": [result.as_dict() for result in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


@app.command("render-hf-vllm-job")
def render_hf_vllm_job(
    model: str = typer.Option(..., help="public Hugging Face model ID for the remote benchmark"),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Hugging Face dataset ID for prompt sampling",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=f"Named HF prompt preset: {', '.join(sorted(HF_DATASET_PRESETS))}",
    ),
    split: str = typer.Option("train"),
    config: str | None = typer.Option(None),
    limit: int = typer.Option(64),
    compare_perc: bool = typer.Option(True, "--compare-perc/--no-compare-perc"),
    max_tokens: int = typer.Option(64),
    gpu_memory_utilization: float = typer.Option(0.8),
    max_model_len: int | None = typer.Option(None),
    tensor_parallel_size: int = typer.Option(1),
    out: Path = typer.Option(..., help="write the standalone HF Jobs script here"),
) -> None:
    if sum(value is not None for value in (hf_dataset, preset)) != 1:
        console.print("[red]Choose exactly one of --hf-dataset or --preset.[/red]")
        raise typer.Exit(1)

    if preset is not None:
        if preset not in HF_DATASET_PRESETS:
            console.print(f"[red]Unknown preset '{preset}'.[/red]")
            raise typer.Exit(1)
        preset_cfg = HF_DATASET_PRESETS[preset]
        hf_dataset = preset_cfg["dataset"]
        split = preset_cfg.get("split", split)

    script = render_hf_vllm_uv_script(
        model=model,
        hf_dataset=hf_dataset,
        split=split,
        config=config,
        limit=limit,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        compare_perc=compare_perc,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    out.write_text(script)
    console.print(f"[dim]HF Jobs benchmark script written to {out}[/dim]")


@app.command("hardware-scorecard")
def hardware_scorecard(
    model: str | None = typer.Option(
        "llama3_8b",
        help=f"Named model key: {', '.join(sorted(MODEL_CATALOG))}",
    ),
    model_name: str = typer.Option("Custom Model", help="display name for a custom model"),
    params_b: float | None = typer.Option(None, help="custom model parameter count in billions"),
    n_layers: int | None = typer.Option(None, help="custom model layer count"),
    n_heads: int | None = typer.Option(None, help="custom model attention head count"),
    head_dim: int | None = typer.Option(None, help="custom model head dimension"),
    context_tokens: int = typer.Option(2048, help="average active decode context"),
    measured_tok_s: float | None = typer.Option(
        None,
        help="optional measured output tok/s to compare with the physics bound",
    ),
    out: Path | None = typer.Option(None),
) -> None:
    selected_model = resolve_model_spec(
        None if params_b is not None else model,
        name=model_name,
        params_b=params_b,
        n_layers=n_layers,
        n_heads=n_heads,
        head_dim=head_dim,
    )
    scorecards = compare_hardware_scorecards(
        selected_model,
        avg_context_tokens=context_tokens,
        measured_tok_s=measured_tok_s,
    )

    table = Table(title=f"Hardware Scorecard — {selected_model.name} @ {context_tokens} ctx")
    table.add_column("GPU", style="cyan")
    table.add_column("VRAM", justify="right")
    table.add_column("BW", justify="right")
    table.add_column("FP16", justify="right")
    table.add_column("FP8", justify="right")
    table.add_column("FP16 sess", justify="right")
    table.add_column("FP8 sess", justify="right")
    table.add_column("FP16 roof", justify="right")
    table.add_column("FP8 roof", justify="right")
    table.add_column("FP8 tok/$", justify="right")

    payload_rows = []
    for card in scorecards:
        tok_per_dollar = card.tokens_per_dollar_hour("fp8")
        table.add_row(
            card.hardware.name,
            f"{card.hardware.vram_gb:.0f} GB",
            f"{card.hardware.memory_bandwidth_gbps:.0f} GB/s",
            f"{card.hardware.fp16_tflops:.1f}",
            f"{card.hardware.fp8_tflops:.1f}" if card.hardware.fp8_tflops is not None else "n/a",
            str(card.fp16.sessions_at_context),
            str(card.fp8.sessions_at_context),
            f"{card.fp16.roofline_tok_s:.1f}",
            f"{card.fp8.roofline_tok_s:.1f}",
            f"{tok_per_dollar:,.0f}" if tok_per_dollar is not None else "n/a",
        )
        payload_rows.append(
            {
                "hardware": card.hardware.key,
                "name": card.hardware.name,
                "vram_gb": card.hardware.vram_gb,
                "memory_bandwidth_gbps": card.hardware.memory_bandwidth_gbps,
                "fp16_tflops": card.hardware.fp16_tflops,
                "fp8_tflops": card.hardware.fp8_tflops,
                "cost_per_hr_usd": card.hardware.cost_per_hr_usd,
                "provider": card.hardware.provider,
                "availability": card.hardware.availability,
                "usable_vram_gb": card.usable_vram_gb,
                "ridge_point_flops_per_byte": card.ridge_point_flops_per_byte,
                "fp16": card.fp16.__dict__,
                "int8": card.int8.__dict__,
                "fp8": card.fp8.__dict__,
                "measured_tok_s": card.measured_tok_s,
                "roofline_efficiency_fp8": card.roofline_efficiency("fp8"),
            }
        )
    console.print(table)
    console.print(
        "[dim]Roofline bound: min(compute, bandwidth). FP8 doubles KV headroom and usually "
        "raises the bandwidth-bound decode ceiling by about 2x versus FP16.[/dim]"
    )

    if out:
        out.write_text(
            json.dumps(
                {
                    "model": {
                        "name": selected_model.name,
                        "params_b": selected_model.params_b,
                        "n_layers": selected_model.n_layers,
                        "n_heads": selected_model.n_heads,
                        "head_dim": selected_model.head_dim,
                    },
                    "context_tokens": context_tokens,
                    "rows": payload_rows,
                },
                indent=2,
            )
        )
        console.print(f"[dim]Results written to {out}[/dim]")


@app.command("profile-dataset")
def profile_dataset(
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Hugging Face dataset ID, e.g. OpenAssistant/oasst1",
    ),
    path: Path | None = typer.Option(
        None,
        "--path",
        help="Local CSV/JSONL/JSON/Parquet path, including Kaggle exports",
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help=f"Named HF preset: {', '.join(sorted(HF_DATASET_PRESETS))}",
    ),
    split: str = typer.Option("train"),
    config: str | None = typer.Option(None),
    limit: int = typer.Option(1000),
    out: Path | None = typer.Option(None),
) -> None:
    if sum(value is not None for value in (hf_dataset, path, preset)) != 1:
        console.print("[red]Choose exactly one of --hf-dataset, --path, or --preset.[/red]")
        raise typer.Exit(1)

    if preset is not None:
        if preset not in HF_DATASET_PRESETS:
            console.print(f"[red]Unknown preset '{preset}'.[/red]")
            raise typer.Exit(1)
        preset_cfg = HF_DATASET_PRESETS[preset]
        dataset_id = preset_cfg["dataset"]
        split = preset_cfg.get("split", split)
        result = profile_hf_dataset(dataset_id, split=split, config=config, limit=limit)
        source_label = dataset_id
    elif hf_dataset is not None:
        result = profile_hf_dataset(hf_dataset, split=split, config=config, limit=limit)
        source_label = hf_dataset
    else:
        result = profile_local_dataset(path, limit=limit)
        source_label = str(path)

    table = Table(title=f"Dataset Token Profile — {source_label}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold")
    table.add_row("Schema", result.schema)
    table.add_row("Rows Profiled", str(result.rows_profiled))
    table.add_row("Rows Total", str(result.rows_total or "unknown"))
    table.add_row("Prompt Mean", f"{result.prompt_tokens.mean:.2f}")
    table.add_row("Prompt p95", f"{result.prompt_tokens.p95:.2f}")
    table.add_row("Response Mean", f"{result.response_tokens.mean:.2f}")
    table.add_row("Response p95", f"{result.response_tokens.p95:.2f}")
    table.add_row("Total Mean", f"{result.total_tokens.mean:.2f}")
    table.add_row("Total p95", f"{result.total_tokens.p95:.2f}")
    table.add_row("Longest Row", str(result.total_tokens.max))
    table.add_row(
        "Estimated Total Tokens",
        str(result.estimated_total_tokens or "sample-only"),
    )
    console.print(table)

    if result.source == "local":
        console.print(
            "[dim]Kaggle export path support covers CSV/JSONL/Parquet files after download.[/dim]"
        )
    else:
        console.print(
            "[dim]Hugging Face presets:[/dim] "
            + ", ".join(f"{name}={cfg['dataset']}" for name, cfg in sorted(HF_DATASET_PRESETS.items()))
        )
        console.print(
            "[dim]Kaggle suggestions:[/dim] "
            + ", ".join(f"{name}: {url}" for name, url in KAGGLE_DATASET_SUGGESTIONS.items())
        )

    if out:
        out.write_text(json.dumps(result.as_dict(), indent=2))
        console.print(f"[dim]Results written to {out}[/dim]")


def main() -> None:
    app()
