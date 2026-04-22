from cachepilot.telemetry import TelemetryCollector
from cachepilot.telemetry_export import LiveTelemetryExporter


def test_prometheus_render_contains_expected_metrics():
    telemetry = TelemetryCollector()
    telemetry.record_tokens("req-1", 16, 24.0)
    snapshot = telemetry.snapshot(vram_util=0.5, queue_depth=3, active_sessions=2)

    exporter = LiveTelemetryExporter(labels={"policy": "perc"})
    exporter.update(snapshot, {"cachepilot_requests_served_total": 1.0})
    payload = exporter.render_prometheus()

    assert "cachepilot_vram_utilization_ratio{policy=\"perc\"}" in payload
    assert "cachepilot_requests_served_total{policy=\"perc\"}" in payload
