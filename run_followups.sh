#!/usr/bin/env bash
# Everything after the main grid: rhyme-swap test, block 512, artist-conditioned models.
set -u
PY="${PYTHON:-python}"
"$PY" counterfactual.py runs/{forward,reversed}_block256_seed{0,1,2} > results/counterfactual.log 2>&1 && echo "finished counterfactual" || echo "FAILED counterfactual"
PYTHON="$PY" ./run_experiment.sh
for seed in 0 1 2; do
  run="runs_artist/reversed_block256_seed$seed"
  [ -f "$run/result.json" ] && continue
  mkdir -p "$run"
  "$PY" train.py --out runs_artist --block-size 256 --seed "$seed" --reversed --artist-embedding --sample-batches 4 --sample-seqs 22 \
    > "$run/train.log" 2>&1 && echo "finished $run" || echo "FAILED $run"
done
"$PY" style_eval.py runs_artist/reversed_block256_seed{0,1,2} > results/style.log 2>&1 && echo "finished style" || echo "FAILED style"
"$PY" analyze.py > results/table.md 2>&1 && echo "finished analyze" || echo "FAILED analyze"
