"""
Memory benchmark module.

Tools:
  - Intel MLC (Memory Latency Checker) – bandwidth & latency matrix
  - STREAM                              – Triad / Copy / Scale / Add bandwidth

Both are optional; whichever is available runs. If neither is found,
a graceful error is returned.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from hw_bench.utils.shell import run, which

log = logging.getLogger("hw-bench")


# ──────────────────────────── data classes ────────────────────────────────────

@dataclass
class MlcLatencyResult:
    """Idle / loaded latency in nanoseconds per NUMA node pair."""
    from_node: int
    to_node: int
    idle_lat_ns: float
    loaded_lat_ns: float


@dataclass
class MlcBandwidthResult:
    """Bandwidth in MB/s for a given access pattern."""
    pattern: str           # e.g. "all_reads", "3:1_reads-writes"
    bandwidth_mbs: float


@dataclass
class MlcResult:
    latencies: List[MlcLatencyResult]       = field(default_factory=list)
    bandwidths: List[MlcBandwidthResult]    = field(default_factory=list)


@dataclass
class StreamResult:
    """STREAM benchmark results in MB/s."""
    copy_mbs:  float = 0.0
    scale_mbs: float = 0.0
    add_mbs:   float = 0.0
    triad_mbs: float = 0.0


@dataclass
class MemoryResult:
    mlc:    Optional[MlcResult]    = None
    stream: Optional[StreamResult] = None
    errors: List[str]              = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────── MLC ─────────────────────────────────────────────

class MlcBenchmark:
    """
    Wrapper for Intel MLC (mlc binary).

    Common install locations:
      /usr/local/bin/mlc, /opt/mlc/mlc, ./mlc
    """

    SEARCH = ["/usr/local/bin/mlc", "/opt/mlc/mlc", "./mlc"]

    def __init__(self, duration: int = 60):
        self.duration = duration
        self._bin: Optional[str] = which("mlc")
        if not self._bin:
            import os, shutil
            for p in self.SEARCH:
                if shutil.which(p) or __import__("pathlib").Path(p).exists():
                    self._bin = p
                    break

    def available(self) -> bool:
        return self._bin is not None

    def run(self) -> MlcResult:
        result = MlcResult()

        # ── idle latency ───────────────────────────────────────────────────────
        log.info("  MLC – idle latency ...")
        r = run([self._bin, "--latency_matrix"], timeout=120, sudo=True)
        if r.ok:
            result.latencies = self._parse_latency_matrix(r.stdout)
        else:
            log.warning("  MLC latency_matrix failed: %s", r.stderr[:100])

        # ── loaded latency ─────────────────────────────────────────────────────
        log.info("  MLC – loaded latency ...")
        r = run([self._bin, "--loaded_latency"], timeout=self.duration + 60, sudo=True)
        if r.ok:
            self._merge_loaded_latency(result.latencies, r.stdout)
        else:
            log.warning("  MLC loaded_latency failed: %s", r.stderr[:100])

        # ── bandwidth ──────────────────────────────────────────────────────────
        log.info("  MLC – peak bandwidth ...")
        r = run([self._bin, "--peak_injection_bandwidth"], timeout=self.duration + 60, sudo=True)
        if r.ok:
            result.bandwidths = self._parse_bandwidth(r.stdout)
        else:
            log.warning("  MLC peak_injection_bandwidth failed: %s", r.stderr[:100])

        return result

    # ── parsers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_latency_matrix(text: str) -> List[MlcLatencyResult]:
        """
        Parse output of --latency_matrix:
            Numa node
            Numa node    0      1
                 0     X.X    Y.Y
                 1     Z.Z    W.W
        """
        results: List[MlcLatencyResult] = []
        in_matrix = False
        header_nodes: List[int] = []

        for line in text.splitlines():
            line = line.strip()
            if "Numa node" in line and not in_matrix:
                # second occurrence has the column headers
                nums = re.findall(r"\d+", line)
                if nums:
                    header_nodes = [int(n) for n in nums]
                    in_matrix = True
                continue
            if in_matrix:
                nums = re.findall(r"[\d.]+", line)
                if nums and len(nums) >= 2:
                    from_node = int(nums[0])
                    for col_idx, val in enumerate(nums[1:len(header_nodes) + 1]):
                        to_node = header_nodes[col_idx] if col_idx < len(header_nodes) else col_idx
                        results.append(MlcLatencyResult(
                            from_node     = from_node,
                            to_node       = to_node,
                            idle_lat_ns   = float(val),
                            loaded_lat_ns = 0.0,
                        ))
        return results

    @staticmethod
    def _merge_loaded_latency(latencies: List[MlcLatencyResult], text: str):
        """Parse --loaded_latency and update existing entries (best effort)."""
        # Extract the last "X.XX" value after "Delay=" lines – simplified
        vals = re.findall(r"(\d+)\s+([\d.]+)", text)
        if vals and latencies:
            # Use the last row (highest injection rate ≈ loaded)
            try:
                latencies[0].loaded_lat_ns = float(vals[-1][1])
            except (IndexError, ValueError):
                pass

    @staticmethod
    def _parse_bandwidth(text: str) -> List[MlcBandwidthResult]:
        """
        Parse --peak_injection_bandwidth output:
            ALL Reads       :  12345.67
            3:1 Reads-Writes:  9876.54
        """
        results: List[MlcBandwidthResult] = []
        pattern = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")
        for line in text.splitlines():
            m = pattern.match(line.strip())
            if m:
                label = m.group(1).strip().lower().replace(" ", "_")
                bw    = float(m.group(2))
                results.append(MlcBandwidthResult(pattern=label, bandwidth_mbs=bw))
        return results


# ──────────────────────────── STREAM ─────────────────────────────────────────

class StreamBenchmark:
    """
    Wrapper for the STREAM benchmark binary.

    Expects a pre-compiled `stream` or `stream_c` binary in PATH or common paths.
    Optionally also supports the OpenMP variant.
    """

    SEARCH = [
        "stream",
        "stream_c",
        "/usr/local/bin/stream",
        "/opt/stream/stream",
    ]
    ENV_OMP: Dict[str, str] = {"OMP_NUM_THREADS": str(__import__("os").cpu_count() or 1)}

    def __init__(self, duration: int = 60):
        self.duration = duration
        self._bin: Optional[str] = None
        for name in self.SEARCH:
            p = which(name)
            if p:
                self._bin = p
                break
            import pathlib
            if pathlib.Path(name).exists():
                self._bin = name
                break

    def available(self) -> bool:
        return self._bin is not None

    def run(self) -> StreamResult:
        log.info("  STREAM – memory bandwidth (binary: %s) ...", self._bin)
        r = run(
            [self._bin],
            timeout=self.duration + 120,
            env=self.ENV_OMP,
        )
        if not r.ok:
            log.error("  STREAM failed: %s", r.stderr[:200])
            return StreamResult()
        return self._parse(r.stdout)

    @staticmethod
    def _parse(text: str) -> StreamResult:
        """
        Parse STREAM output:
            Copy:      12345.6    ...
            Scale:     11234.5    ...
            Add:       10234.8    ...
            Triad:     10987.6    ...
        """
        result = StreamResult()
        mapping = {
            "copy":  "copy_mbs",
            "scale": "scale_mbs",
            "add":   "add_mbs",
            "triad": "triad_mbs",
        }
        for line in text.splitlines():
            for key, attr in mapping.items():
                if line.strip().lower().startswith(key + ":"):
                    nums = re.findall(r"[\d.]+", line)
                    if nums:
                        setattr(result, attr, float(nums[0]))
        return result


# ──────────────────────────── public facade ───────────────────────────────────

class MemoryBenchmark:
    """
    Runs MLC and STREAM if available.

    Args:
        duration: Seconds hint for MLC loaded/bandwidth tests.
    """

    def __init__(self, duration: int = 60):
        self._mlc    = MlcBenchmark(duration)
        self._stream = StreamBenchmark(duration)

    def run(self) -> MemoryResult:
        log.info("=== Memory Benchmark ===")
        result = MemoryResult()

        if self._mlc.available():
            log.info("  Running MLC ...")
            result.mlc = self._mlc.run()
        else:
            msg = "Intel MLC not found. Install from https://www.intel.com/content/www/us/en/developer/articles/tool/intelr-memory-latency-checker.html"
            log.warning("  %s", msg)
            result.errors.append(msg)

        if self._stream.available():
            log.info("  Running STREAM ...")
            result.stream = self._stream.run()
        else:
            msg = "STREAM binary not found. Build from https://www.cs.virginia.edu/stream/"
            log.warning("  %s", msg)
            result.errors.append(msg)

        if not self._mlc.available() and not self._stream.available():
            result.errors.append(
                "No memory benchmark tool available. "
                "Please install MLC or STREAM."
            )

        log.info("=== Memory Benchmark complete ===")
        return result
