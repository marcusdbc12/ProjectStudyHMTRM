from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable


def run_and_stream(cmd: list[str], log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open('w', encoding='utf-8') as fh:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            fh.write(line)
            fh.flush()
            yield line.rstrip('\n')
        rc = process.wait()
    if rc != 0:
        raise RuntimeError(f'Command failed with exit code {rc}: {cmd}')
