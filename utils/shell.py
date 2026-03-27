#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
shell命令执行工具
"""

import os
import time
import shlex
import shutil
import subprocess
from typing import Dict, List, Any

def run_command(cmd: str | List[str], timeout: int = 3600, shell: bool = True,
                env: dict = None, capture: bool = True) -> Dict[str, Any]:
    """
    执行系统命令

    Returns:
        dict: {
            'returncode': int,
            'stdout': str,
            'stderr': str,
            'duration': float,
            'success': bool
        }
    """
    if isinstance(cmd,str):
        args = shlex.split(cmd)
    else:
        args = list(cmd)

    start = time.time()
    try:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)

        result = subprocess.run(
            args,
            shell=shell,
            capture_output=capture,
            text=True,
            timeout=timeout,
            env=merged_env
        )
        duration = time.time() - start
        return {
            'returncode': result.returncode,
            'stdout': result.stdout if capture else '',
            'stderr': result.stderr if capture else '',
            'duration': duration,
            'success': result.returncode == 0,
            'command': cmd
        }
    except subprocess.TimeoutExpired:
        duration = time.time() - start
        return {
            'returncode': -1,
            'stdout': '',
            'stderr': f'Command timed out after {timeout}s',
            'duration': duration,
            'success': False,
            'command': cmd
        }
    except Exception as e:
        duration = time.time() - start
        return {
            'returncode': -2,
            'stdout': '',
            'stderr': str(e),
            'duration': duration,
            'success': False,
            'command': cmd
        }

def check_root() -> bool:
    """检查是否为root用户"""
    return os.geteuid() == 0

def check_tool_available(tool: str) -> bool:
    """检查工具是否可用"""
    return shutil.which(tool) is not None