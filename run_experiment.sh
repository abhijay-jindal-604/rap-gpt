#!/usr/bin/env bash
# Train every direction, context length and seed. Finished runs are skipped.
# The GPU has 8 GB: 256 fits two at a time and shorter contexts fit three. Block 512 runs
# alone at batch 32, which keeps the 16,384 target characters per step of 256 at batch 64.
set -u
run_grid() {
  local jobs="$1"; shift
  for block in "$@"; do
    for direction in forward reversed; do
      for seed in 0 1 2; do
        echo "$direction $block $seed"
      done
    done
  done | xargs -P "$jobs" -L 1 bash -c '
    run="runs/$0_block$1_seed$2"
    [ -f "$run/result.json" ] && exit 0
    mkdir -p "$run"
    flag=""; [ "$0" = reversed ] && flag="--reversed"
    "${PYTHON:-python}" train.py --block-size "$1" --seed "$2" --batch-size "${BATCH_SIZE:-64}" $flag > "$run/train.log" 2>&1 \
      && echo "finished $run" || echo "FAILED $run"
  '
}
run_grid 2 256
run_grid 3 128 64 32 16
export BATCH_SIZE=32
run_grid 1 512
