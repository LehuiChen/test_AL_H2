#!/usr/bin/env bash
set -euo pipefail

python scripts/active_learning_loop.py \
  --config configs/base_smoke.yaml \
  --mode smoke \
  --submit-mode-labels pbs \
  --submit-mode-train pbs \
  --submit-mode-md pbs \
  --no-wait
