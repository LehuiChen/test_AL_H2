#!/usr/bin/env bash
set -euo pipefail

python3 scripts/run_h2_al.py \
  --config configs/h2_ani_al.yaml \
  --mode full \
  --submit-mode pbs \
  --no-wait
