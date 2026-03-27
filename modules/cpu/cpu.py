"""
CPU benchmark module.

Backends (tried in order):
  1. SPEC CPU 2017 – runcpu wrapper (if installed)
  2. sysbench     – open-source CPU test (fallback)

Metrics:
  - FP (floating-point): SPECfp_rate2017 or sysbench prime-number score
  - INT (integer):       SPECint_rate2017 or sysbench prime-number score
"""
from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

from hw_bench.utils.shell import run, which

log = logging.getLogger("hw-bench")


# ──────────────────────────── data classes ────────────────────────────────────

@dataclass
class SpecScore:
    suite: str          # "intrate" | "fprate" | "intspeed" | "fpspeed"
    score: float
    copies: int
    peak_or_base: str   # "base" | "peak"


@dataclass
class SysbenchScore:
    threads: int
    events_per_sec: float   # events/sec (higher = better)
    latency_avg_ms: float
    latency_p95_ms: float


@dataclass
class CpuResult:
    backend: str                             # "speccpu2017" | "sysbench"
    spec_scores: List[SpecScore]             = field(default_factory=list)
    sysbench_int: Optional[SysbenchScore]   = None
    sysbench_fp: Optional[SysbenchScore]    = None
    errors: List[str]                        = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────── SPEC CPU 2017 ───────────────────────────────────

class SpecCpu2017:
    """
    Wrapper around SPEC CPU 2017's `runcpu` command.

    The tool must be installed and the run environment set up by the user.
    We detect common install locations: /opt/cpu2017, $SPEC env variable.
    """

    SEARCH_PATHS = [
        "/opt/cpu2017",
        "/opt/SPEC/cpu2017",
        "/usr/local/cpu2017",
    ]
    SUITES = {
        "intrate": "intrate",
        "fprate":  "fprate",
    }

    def __init__(self, install_dir: Optional[str] = None, copies: int = 0, duration_hint: int = 0):
        self.install_dir = install_dir or os.environ.get("SPEC", "")
        if not self.install_dir:
            for p in self.SEARCH_PATHS:
                if Path(p).is_dir():
                    self.install_dir = p
                    break
        self.copies = copies or os.cpu_count() or 1
        self.duration_hint = duration_hint  # used only for logging; SPEC manages timing itself

    def available(self) -> bool:
        if not self.install_dir:
            return False
        runcpu = Path(self.install_dir) / "bin" / "runcpu"
        return runcpu.exists()

    def run(self) -> CpuResult:
        result = CpuResult(backend="speccpu2017")
        runcpu = str(Path(self.install_dir) / "bin" / "runcpu")

        for suite_key, suite_arg in self.SUITES.items():
            log.info("  SPEC CPU 2017 – %s (copies=%d) ...", suite_arg, self.copies)
            cmd = [
                runcpu,
                "--config=default.cfg",
                f"--copies={self.copies}",
                "--tune=base",
                "--noreportable",
                "--iterations=1",
                suite_arg,
            ]
            res = run(cmd, timeout=14400, cwd=self.install_dir)  # 4 h hard limit
            if not res.ok:
                msg = f"SPEC {suite_arg} failed: {res.stderr[:200]}"
                log.error("  %s", msg)
                result.errors.append(msg)
                continue

            scores = self._parse_output(res.stdout + res.stderr, suite_key)
            result.spec_scores.extend(scores)

        return result

    @staticmethod
    def _parse_output(text: str, suite: str) -> List[SpecScore]:
        """Extract 'Est. SPECxxx_base2017 = N' lines."""
        scores: List[SpecScore] = []
        # Example: "  Est. SPECint_rate_base2017    =   1234"
        pattern = re.compile(
            r"Est\.\s+(SPEC\w+2017)\s*=\s*([\d.]+)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            label = m.group(1).lower()
            value = float(m.group(2))
            pb = "peak" if "peak" in label else "base"
            scores.append(SpecScore(suite=suite, score=value, copies=0, peak_or_base=pb))
        return scores


# ──────────────────────────── sysbench ───────────────────────────────────────

class SysbenchCpu:
    """
    Uses sysbench cpu test as a fallback when SPEC CPU is unavailable.

    sysbench cpu --cpu-max-prime=20000 --threads=N --time=T run
    """

    def __init__(self, duration: int = 60):
        self.duration = duration
        self._bin = which("sysbench")

    def available(self) -> bool:
        return self._bin is not None

    def run_test(self, label: str, threads: int, max_prime: int) -> Optional[SysbenchScore]:
        log.info("  sysbench cpu [%s] threads=%d prime=%d ...", label, threads, max_prime)
        cmd = [
            self._bin,
            "cpu",
            f"--cpu-max-prime={max_prime}",
            f"--threads={threads}",
            f"--time={self.duration}",
            "run",
        ]
        res = run(cmd, timeout=self.duration + 30)
        if not res.ok:
            log.error("  sysbench [%s] failed: %s", label, res.stderr[:200])
            return None
        return self._parse(res.stdout, threads)

    @staticmethod
    def _parse(text: str, threads: int) -> Optional[SysbenchScore]:
        eps = re.search(r"events per second:\s*([\d.]+)", text)
        avg = re.search(r"avg:\s*([\d.]+)", text)
        p95 = re.search(r"95th percentile:\s*([\d.]+)", text)
        if not eps:
            return None
        return SysbenchScore(
            threads         = threads,
            events_per_sec  = float(eps.group(1)),
            latency_avg_ms  = float(avg.group(1)) if avg else 0.0,
            latency_p95_ms  = float(p95.group(1)) if p95 else 0.0,
        )

    def run(self) -> CpuResult:
        result = CpuResult(backend="sysbench")
        ncpu = os.cpu_count() or 1
        # INT-proxy: small prime, high throughput
        result.sysbench_int = self.run_test("int", threads=ncpu, max_prime=10000)
        # FP-proxy: larger prime, more compute
        result.sysbench_fp  = self.run_test("fp",  threads=ncpu, max_prime=50000)
        if result.sysbench_int is None and result.sysbench_fp is None:
            result.errors.append("All sysbench CPU tests failed.")
        return result


# ──────────────────────────── public facade ───────────────────────────────────

class CpuBenchmark:
    """
    Auto-detects SPEC CPU 2017; falls back to sysbench.

    Args:
        duration:      Seconds per sysbench run (SPEC manages its own timing).
        spec_dir:      Explicit SPEC install directory (overrides auto-detect).
        spec_copies:   Number of parallel SPEC copies (default = nCPU).
        force_sysbench: Always use sysbench, even if SPEC is found.
    """

    def __init__(
        self,
        duration: int = 60,
        spec_dir: Optional[str] = None,
        spec_copies: int = 0,
        force_sysbench: bool = False,
    ):
        self.duration       = duration
        self.force_sysbench = force_sysbench
        self._spec          = SpecCpu2017(spec_dir, spec_copies, duration)
        self._sysbench      = SysbenchCpu(duration)

    def run(self) -> CpuResult:
        log.info("=== CPU Benchmark ===")

        if not self.force_sysbench and self._spec.available():
            log.info("  Using SPEC CPU 2017 (install: %s)", self._spec.install_dir)
            result = self._spec.run()
        elif self._sysbench.available():
            log.info("  SPEC CPU 2017 not found – falling back to sysbench")
            result = self._sysbench.run()
        else:
            result = CpuResult(backend="none")
            result.errors.append(
                "Neither SPEC CPU 2017 nor sysbench is available. "
                "Install sysbench: apt install sysbench / dnf install sysbench"
            )

        log.info("=== CPU Benchmark complete (backend=%s) ===", result.backend)
        return result
