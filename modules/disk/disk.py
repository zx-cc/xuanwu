"""
FIO disk benchmark module.

Tests:
  - Sequential read/write (128K, queue depth 32)
  - Random read/write   (4K,   queue depth 32)
  - 7:3 read/write mix  (4K,   queue depth 32)
  - Latency             (4K,   queue depth 1)
  - Steady-state sweep  (write saturation over time)
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from hw_bench.utils.shell import run, require

log = logging.getLogger("hw-bench")

# ──────────────────────────── data classes ────────────────────────────────────

@dataclass
class FioJobResult:
    name: str
    bs: str
    rw: str
    iodepth: int
    read_bw_mbs: float    = 0.0   # MB/s
    read_iops: float      = 0.0
    read_lat_us: float    = 0.0   # mean latency µs
    read_lat_p99_us: float= 0.0
    write_bw_mbs: float   = 0.0
    write_iops: float     = 0.0
    write_lat_us: float   = 0.0
    write_lat_p99_us: float= 0.0


@dataclass
class SteadyStatePoint:
    elapsed_s: int
    write_bw_mbs: float
    write_iops: float


@dataclass
class DiskResult:
    device: str
    jobs: List[FioJobResult] = field(default_factory=list)
    steady_state: List[SteadyStatePoint] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────── parser ──────────────────────────────────────────

def _parse_fio_json(raw: str) -> List[FioJobResult]:
    """Parse fio --output-format=json output."""
    data = json.loads(raw)
    results: List[FioJobResult] = []

    for job in data.get("jobs", []):
        opt = job.get("job options", {})
        jr = FioJobResult(
            name     = job["jobname"],
            bs       = opt.get("bs", "?"),
            rw       = opt.get("rw", "?"),
            iodepth  = int(opt.get("iodepth", 1)),
        )
        r = job.get("read", {})
        w = job.get("write", {})

        jr.read_bw_mbs     = round(r.get("bw", 0) / 1024, 2)
        jr.read_iops       = round(r.get("iops", 0), 2)
        jr.read_lat_us     = round(r.get("lat_ns", {}).get("mean", 0) / 1000, 2)
        jr.read_lat_p99_us = round(
            r.get("clat_ns", {}).get("percentile", {}).get("99.000000", 0) / 1000, 2
        )
        jr.write_bw_mbs     = round(w.get("bw", 0) / 1024, 2)
        jr.write_iops       = round(w.get("iops", 0), 2)
        jr.write_lat_us     = round(w.get("lat_ns", {}).get("mean", 0) / 1000, 2)
        jr.write_lat_p99_us = round(
            w.get("clat_ns", {}).get("percentile", {}).get("99.000000", 0) / 1000, 2
        )
        results.append(jr)

    return results


# ──────────────────────────── runner ──────────────────────────────────────────

class DiskBenchmark:
    """
    Wraps fio to run a standard battery of disk tests.

    Args:
        device:       Block device path (e.g. /dev/sda) or directory for file-based tests.
        duration:     Seconds per job.
        ss_duration:  Seconds for steady-state sweep (0 = skip).
        numjobs:      Parallel fio workers.
        direct:       Use O_DIRECT (bypass page cache).
    """

    JOBS = [
        # (name,          rw,           bs,    iodepth)
        ("seq_read",      "read",       "128k", 32),
        ("seq_write",     "write",      "128k", 32),
        ("rand_read",     "randread",   "4k",   32),
        ("rand_write",    "randwrite",  "4k",   32),
        ("mix_7r3w",      "randrw",     "4k",   32),   # rwmixread=70
        ("latency_read",  "randread",   "4k",   1),
        ("latency_write", "randwrite",  "4k",   1),
    ]

    def __init__(
        self,
        device: str,
        duration: int = 60,
        ss_duration: int = 300,
        numjobs: int = 1,
        direct: bool = True,
        size: str = "4G",
    ):
        self.device     = device
        self.duration   = duration
        self.ss_duration = ss_duration
        self.numjobs    = numjobs
        self.direct     = direct
        self.size       = size
        self._fio       = require("fio")

    # ── helpers ────────────────────────────────────────────────────────────────

    def _is_block_dev(self) -> bool:
        return self.device.startswith("/dev/")

    def _base_args(self, tmpfile: str) -> List[str]:
        args = [
            self._fio,
            "--output-format=json",
            f"--runtime={self.duration}",
            "--time_based",
            f"--numjobs={self.numjobs}",
            f"--group_reporting",
            "--ioengine=libaio",
            f"--direct={1 if self.direct else 0}",
            "--lat_percentiles=1",
            "--clat_percentiles=1",
            "--percentile_list=50:90:99:99.9",
        ]
        if self._is_block_dev():
            args += [f"--filename={self.device}"]
        else:
            args += [f"--directory={self.device}", f"--size={self.size}"]
        return args

    def _run_job(
        self,
        name: str,
        rw: str,
        bs: str,
        iodepth: int,
        extra: Optional[List[str]] = None,
    ) -> Optional[FioJobResult]:
        args = self._base_args("") + [
            f"--name={name}",
            f"--rw={rw}",
            f"--bs={bs}",
            f"--iodepth={iodepth}",
        ]
        if extra:
            args += extra

        log.info("  fio [%s] bs=%s rw=%s iodepth=%d ...", name, bs, rw, iodepth)
        result = run(args, timeout=self.duration + 60, sudo=self._is_block_dev())
        if not result.ok:
            log.error("  fio job '%s' failed: %s", name, result.stderr[:200])
            return None

        try:
            jobs = _parse_fio_json(result.stdout)
            return jobs[0] if jobs else None
        except Exception as exc:
            log.error("  Failed to parse fio output for job '%s': %s", name, exc)
            return None

    # ── steady-state ───────────────────────────────────────────────────────────

    def _run_steady_state(self) -> List[SteadyStatePoint]:
        """
        Write saturation test: 4K randwrite for ss_duration seconds,
        sample bandwidth every 5 s via fio's log_avg_msec.
        """
        log.info("  fio steady-state write (duration=%ds) ...", self.ss_duration)
        with tempfile.TemporaryDirectory() as tmp:
            bw_log = os.path.join(tmp, "ss_bw")
            args = self._base_args("") + [
                "--name=steady_state",
                "--rw=randwrite",
                "--bs=4k",
                "--iodepth=32",
                f"--runtime={self.ss_duration}",
                f"--write_bw_log={bw_log}",
                "--log_avg_msec=5000",
            ]
            result = run(args, timeout=self.ss_duration + 120, sudo=self._is_block_dev())
            if not result.ok:
                log.error("  Steady-state test failed: %s", result.stderr[:200])
                return []

            # Parse bw log: timestamp_ms, bw_KB/s, direction, ...
            points: List[SteadyStatePoint] = []
            log_file = bw_log + "_bw.1.log"
            if not os.path.exists(log_file):
                log_file = bw_log + "_bw.log"
            if os.path.exists(log_file):
                with open(log_file) as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) >= 2:
                            try:
                                ts_ms  = int(parts[0].strip())
                                bw_kbs = float(parts[1].strip())
                                points.append(SteadyStatePoint(
                                    elapsed_s    = ts_ms // 1000,
                                    write_bw_mbs = round(bw_kbs / 1024, 2),
                                    write_iops   = round(bw_kbs / 4, 2),  # approx for 4K
                                ))
                            except ValueError:
                                pass
            return points

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self) -> DiskResult:
        result = DiskResult(device=self.device)
        log.info("=== Disk Benchmark: %s ===", self.device)

        for name, rw, bs, iodepth in self.JOBS:
            extra = ["--rwmixread=70"] if rw == "randrw" else []
            jr = self._run_job(name, rw, bs, iodepth, extra)
            if jr:
                result.jobs.append(jr)
            else:
                result.errors.append(f"job '{name}' failed or produced no output")

        if self.ss_duration > 0:
            result.steady_state = self._run_steady_state()

        log.info("=== Disk Benchmark complete ===")
        return result
