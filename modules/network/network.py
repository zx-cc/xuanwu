"""
Network benchmark module using netperf.

Tests (all TCP unless specified):
  - TCP_STREAM    – bulk throughput (Gbps)
  - TCP_MAERTS    – reverse bulk throughput
  - TCP_RR        – request/response latency (trans/s)
  - UDP_STREAM    – UDP throughput + packet loss
  - TCP_CRR       – connect/request/response (new connection per transaction)

The module can:
  1. Manage a local netserver subprocess automatically.
  2. Connect to a remote netserver (provide --net-server).
"""
from __future__ import annotations

import logging
import os
import re
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from hw_bench.utils.shell import run, which, require

log = logging.getLogger("hw-bench")


# ──────────────────────────── data classes ────────────────────────────────────

@dataclass
class NetperfJobResult:
    test: str               # TCP_STREAM, TCP_RR, ...
    server: str
    throughput_gbps: float  = 0.0   # for STREAM tests
    trans_per_sec: float    = 0.0   # for RR tests
    mean_latency_us: float  = 0.0   # for RR: 1/trans_per_sec * 1e6
    local_cpu_pct: float    = 0.0
    remote_cpu_pct: float   = 0.0


@dataclass
class NetworkResult:
    server: str
    jobs: List[NetperfJobResult] = field(default_factory=list)
    errors: List[str]            = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────── netserver manager ───────────────────────────────

class NetserverManager:
    """Start and stop a local netserver process."""

    def __init__(self, port: int = 12865):
        self.port = port
        self._proc: Optional[subprocess.Popen] = None
        self._bin  = which("netserver")

    def available(self) -> bool:
        return self._bin is not None

    def start(self) -> bool:
        if not self._bin:
            return False
        log.info("  Starting local netserver on port %d ...", self.port)
        try:
            self._proc = subprocess.Popen(
                [self._bin, "-p", str(self.port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.0)  # give it time to bind
            return True
        except Exception as exc:
            log.error("  Failed to start netserver: %s", exc)
            return False

    def stop(self):
        if self._proc and self._proc.poll() is None:
            log.info("  Stopping local netserver ...")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()


# ──────────────────────────── netperf runner ──────────────────────────────────

class NetworkBenchmark:
    """
    Runs a battery of netperf tests against a netserver.

    Args:
        server:       IP / hostname of netserver (default: 127.0.0.1).
        port:         netserver port (default: 12865).
        duration:     Seconds per test.
        manage_server: Auto-start a local netserver subprocess.
        send_size:    Socket send/recv buffer (bytes, 0 = OS default).
    """

    TESTS = [
        # (test_name,    is_rr)
        ("TCP_STREAM",   False),
        ("TCP_MAERTS",   False),
        ("UDP_STREAM",   False),
        ("TCP_RR",       True),
        ("TCP_CRR",      True),
    ]

    def __init__(
        self,
        server: str = "127.0.0.1",
        port: int = 12865,
        duration: int = 30,
        manage_server: bool = True,
        send_size: int = 0,
    ):
        self.server        = server
        self.port          = port
        self.duration      = duration
        self.manage_server = manage_server
        self.send_size     = send_size
        self._netperf      = require("netperf")
        self._srv_mgr      = NetserverManager(port) if manage_server else None

    # ── helpers ────────────────────────────────────────────────────────────────

    def _build_args(self, test: str, is_rr: bool) -> List[str]:
        args = [
            self._netperf,
            "-H", self.server,
            "-p", str(self.port),
            "-l", str(self.duration),
            "-t", test,
            "--",
            "-P", "0",   # no port display overhead
        ]
        if self.send_size > 0:
            args += ["-s", str(self.send_size), "-S", str(self.send_size)]
        if is_rr:
            # Request/Response: 1-byte request, 1-byte response
            args += ["-r", "1,1"]
        return args

    def _run_test(self, test: str, is_rr: bool) -> Optional[NetperfJobResult]:
        log.info("  netperf [%s] server=%s duration=%ds ...", test, self.server, self.duration)
        args = self._build_args(test, is_rr)
        r = run(args, timeout=self.duration + 30)
        if not r.ok:
            log.error("  netperf [%s] failed: %s", test, r.stderr[:200])
            return None
        return self._parse(r.stdout, test)

    @staticmethod
    def _parse(text: str, test: str) -> Optional[NetperfJobResult]:
        """
        netperf output formats vary by test type.

        STREAM last line (CSV-like):
          Recv Socket Size Bytes, Send Socket ... , Throughput 10^6bits/sec
          65536  65536  65536  65536  30.00  9432.56

        RR last line:
          Trans  Rate  per Sec   Trans  per Sec
          23456.78
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return None

        job = NetperfJobResult(test=test, server="")
        last = lines[-1]
        nums = re.findall(r"[\d.]+", last)

        if test in ("TCP_STREAM", "TCP_MAERTS", "UDP_STREAM"):
            # last number is throughput in Mbps
            if nums:
                mbps = float(nums[-1])
                job.throughput_gbps = round(mbps / 1000, 4)
        elif test in ("TCP_RR", "TCP_CRR"):
            # single number = trans/sec
            if nums:
                tps = float(nums[0])
                job.trans_per_sec  = round(tps, 2)
                job.mean_latency_us = round(1e6 / tps, 2) if tps > 0 else 0

        return job

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self) -> NetworkResult:
        log.info("=== Network Benchmark (server=%s) ===", self.server)
        result = NetworkResult(server=self.server)

        ctx = self._srv_mgr if self._srv_mgr else _NullContext()
        with ctx:
            for test, is_rr in self.TESTS:
                jr = self._run_test(test, is_rr)
                if jr:
                    jr.server = self.server
                    result.jobs.append(jr)
                else:
                    result.errors.append(f"netperf test '{test}' failed")

        log.info("=== Network Benchmark complete ===")
        return result


class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *_): pass
