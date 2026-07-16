#!/usr/bin/env bash
set -euo pipefail

cd /data/AlbertLM
mkdir -p reports

export TOKENIZERS_PARALLELISM=false
export RAYON_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

nice -n 19 ionice -c3 \
  .venv-datagen/bin/python \
  tools/tokenizer_audit.py \
  2>&1 | tee reports/tokenizer-audit.log
