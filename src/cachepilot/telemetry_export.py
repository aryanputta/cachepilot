from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .telemetry import Snapshot


class LiveTelemetryExporter:
    """
    Thin Prometheus-compatible exporter for CachePilot snapshots.

    The exporter keeps only the latest metrics payload, which is enough for a
    Prometheus scrape target and simple Grafana dashboards.
    """

    def __init__(
        self,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self._labels = dict(labels or {})
        self._latest_extra: dict[str, float] = {}
        self._latest_snapshot: Snapshot | None = None
        self._snapshots: list[dict] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def update(self, snapshot: Snapshot, extra_metrics: Mapping[str, float]) -> None:
        self._latest_snapshot = snapshot
        self._latest_extra = {key: float(value) for key, value in extra_metrics.items()}
        self._snapshots.append(dict(snapshot.__dict__))

    def render_prometheus(self) -> str:
        metrics = {}
        if self._latest_snapshot is not None:
            metrics.update(self._latest_snapshot.as_metrics())
        metrics.update(self._latest_extra)

        label_text = ""
        if self._labels:
            rendered = ",".join(f'{key}="{value}"' for key, value in sorted(self._labels.items()))
            label_text = f"{{{rendered}}}"

        lines = []
        for name, value in sorted(metrics.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name}{label_text} {value:.10g}")
        return "\n".join(lines) + ("\n" if lines else "")

    def write_prometheus(self, path: Path) -> None:
        path.write_text(self.render_prometheus())

    def write_snapshots_json(self, path: Path) -> None:
        path.write_text(json.dumps(self._snapshots, indent=2))

    def serve(self, host: str = "127.0.0.1", port: int = 9464) -> None:
        exporter = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path not in ("/metrics", "/"):
                    self.send_error(404)
                    return
                payload = exporter.render_prometheus().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="cachepilot-prometheus-exporter",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None
