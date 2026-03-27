#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
日志工具
"""

import sys
import logging
from pathlib import Path

class _ColorFormat(logging.Formatter):
    """带颜色的日志格式化器"""
    COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[1;31m", # bold red
    }
    RESET = "\033[0m"

    def format(self,record:logging.LogRecord)->str:
        color = self.COLORS.get(record.levelno,"")
        msg = super().format(record)
        return f"{color}{msg}{self.RESET}" if sys.stderr.isatty() else msg


def setup_logging(name: str = "hw-bench",log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)

    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(level)
    ch.setFormatter(_ColorFormat("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    if log_file:
        log_file.parent.mkdir(parents=True,exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    return logger    