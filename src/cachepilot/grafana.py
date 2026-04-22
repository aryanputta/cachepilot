from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _timeseries_panel(
    panel_id: int,
    title: str,
    expr: str,
    grid_x: int,
    grid_y: int,
    width: int = 12,
    height: int = 8,
    unit: str = "short",
) -> Dict:
    return {
        "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "unit": unit,
            },
            "overrides": [],
        },
        "gridPos": {"h": height, "w": width, "x": grid_x, "y": grid_y},
        "id": panel_id,
        "options": {
            "legend": {"displayMode": "table", "placement": "bottom"},
            "tooltip": {"mode": "single"},
        },
        "targets": [{"expr": expr, "legendFormat": "{{policy}}", "refId": "A"}],
        "title": title,
        "type": "timeseries",
    }


def build_dashboard(title: str = "CachePilot Live Telemetry") -> Dict:
    panels: List[Dict] = [
        _timeseries_panel(
            panel_id=1,
            title="Tokens / Second",
            expr="cachepilot_tokens_per_second",
            grid_x=0,
            grid_y=0,
        ),
        _timeseries_panel(
            panel_id=2,
            title="VRAM Utilization",
            expr="cachepilot_vram_utilization_ratio",
            grid_x=12,
            grid_y=0,
            unit="percentunit",
        ),
        _timeseries_panel(
            panel_id=3,
            title="p95 Latency",
            expr="cachepilot_latency_p95_ms",
            grid_x=0,
            grid_y=8,
            unit="ms",
        ),
        _timeseries_panel(
            panel_id=4,
            title="Eviction Cost",
            expr="cachepilot_total_eviction_cost_seconds",
            grid_x=12,
            grid_y=8,
            unit="s",
        ),
        _timeseries_panel(
            panel_id=5,
            title="Drops vs Defers",
            expr="cachepilot_requests_dropped_total or cachepilot_requests_deferred_total",
            grid_x=0,
            grid_y=16,
        ),
        _timeseries_panel(
            panel_id=6,
            title="Eviction Rate",
            expr="cachepilot_eviction_rate_per_second",
            grid_x=12,
            grid_y=16,
        ),
    ]

    return {
        "annotations": {"list": []},
        "editable": True,
        "graphTooltip": 0,
        "panels": panels,
        "refresh": "5s",
        "schemaVersion": 39,
        "style": "dark",
        "tags": ["cachepilot", "kv-cache", "rl"],
        "templating": {"list": []},
        "time": {"from": "now-15m", "to": "now"},
        "timezone": "browser",
        "title": title,
        "uid": "cachepilot-live",
        "version": 1,
    }


def write_dashboard(path: Path, title: str = "CachePilot Live Telemetry") -> None:
    path.write_text(json.dumps(build_dashboard(title=title), indent=2))
