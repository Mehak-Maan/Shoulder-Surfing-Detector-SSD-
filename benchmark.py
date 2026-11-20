"""Built-in FPS / latency / CPU benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass
from os import cpu_count

from capture import FrameSource
from pipeline import Pipeline


@dataclass(frozen=True)
class BenchmarkResult:
    """Measured throughput of the full pipeline (capture excluded from latency)."""

    frames: int
    wall_s: float
    fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    cpu_percent_one_core: float
    cpu_percent_total: float


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, round(fraction * (len(sorted_values) - 1)))
    return sorted_values[index]


def run_benchmark(
    source: FrameSource, pipeline: Pipeline, seconds: float, target_fps: float | None = None
) -> BenchmarkResult:
    """Run the pipeline for ``seconds`` of wall time (or until the source ends).

    ``target_fps`` paces the loop, e.g. ``8`` to measure headless-mode CPU load.
    CPU time is process-wide (all threads), reported as a percentage of one core
    and of the whole machine.
    """
    latencies: list[float] = []
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    next_tick = wall_start
    while time.perf_counter() - wall_start < seconds:
        if target_fps:
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            next_tick += 1.0 / target_fps
        packet = source.read()
        if packet is None:
            break
        latencies.append(pipeline.process(packet).latency_ms)
    wall = max(time.perf_counter() - wall_start, 1e-9)
    cpu = time.process_time() - cpu_start
    latencies.sort()
    one_core = 100.0 * cpu / wall
    return BenchmarkResult(
        frames=len(latencies),
        wall_s=wall,
        fps=len(latencies) / wall,
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        cpu_percent_one_core=one_core,
        cpu_percent_total=one_core / (cpu_count() or 1),
    )


def format_benchmark(result: BenchmarkResult) -> str:
    """Render ``result`` as plain text."""
    return "\n".join(
        [
            f"Frames processed : {result.frames} in {result.wall_s:.1f} s",
            f"Throughput       : {result.fps:.1f} FPS",
            f"Latency          : p50 {result.latency_p50_ms:.1f} ms, p95 {result.latency_p95_ms:.1f} ms",
            f"CPU              : {result.cpu_percent_one_core:.1f}% of one core, "
            f"{result.cpu_percent_total:.1f}% of the machine",
        ]
    )
