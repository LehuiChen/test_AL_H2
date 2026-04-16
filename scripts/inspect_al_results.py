#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and "--legacy-runner" not in sys.argv:
    from inspect_al_results_core import main as _inspect_main

    raise SystemExit(_inspect_main(sys.argv[1:]))
