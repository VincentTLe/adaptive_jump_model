#!/usr/bin/env bash
# Run one market's model fitting with the machine actually saturated.
#
# Measured A/B on this box (32 cores), same input both times: the US frame of
# run 93de627bb755, 13,787 rows giving 10,788 rolling windows, and both runs
# produced the same 10,788 states.
#
#   n_jobs=8,  OMP unset   1165 s   19.4 min
#   n_jobs=30, OMP=1        517 s    8.6 min     2.25x
#
# Note the gap between that and a naive extrapolation. Fitting one window costs
# about 745 ms on one core, so 10,788 windows over 30 workers "should" take
# 268 s; it takes 517. The rest goes to process-pool overhead, the checkpoint
# written every 50 days, and memory bandwidth shared by 30 processes on 32
# cores. Scaling past this point needs a cheaper fit, not more workers.
#
# Three things were wrong, and the third is the one that is easy to miss:
#
#   1. n_jobs sat at 8 on a 32-core box.
#   2. Two markets ran concurrently, so each got half the workers it could have.
#   3. OMP_NUM_THREADS was unset, so every worker's BLAS spawned its own thread
#      pool underneath the process pool. Those threads fight each other for the
#      same cores; on a 3000x1 problem there is no linear algebra worth
#      threading anyway, so the contention is pure loss.
#
# Nothing here changes any result: the work is identical, only its scheduling
# differs. Fitting a window is independent of every other window, so this is a
# scheduling fix, not a numerical one.
#
# Usage:
#   scripts/run_fast.sh <command...>
#   scripts/run_fast.sh uv run python scripts/some_experiment.py us
#
# Set JOBS to override the worker count (default: cores - 2, leaving room for
# the parent process and the machine's own work).

set -euo pipefail

CORES="$(nproc)"
JOBS="${JOBS:-$((CORES > 3 ? CORES - 2 : 1))}"

# One thread per worker. Without this, workers oversubscribe the machine.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# Experiment scripts read this to size their process pool.
export ADAPTIVE_JUMP_N_JOBS="$JOBS"

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <command...>" >&2
  echo "cores=$CORES jobs=$JOBS" >&2
  exit 2
fi

echo "[run_fast] cores=$CORES jobs=$JOBS threads-per-worker=1" >&2
echo "[run_fast] $*" >&2
start=$(date +%s)
"$@"
status=$?
echo "[run_fast] finished in $(( $(date +%s) - start ))s" >&2
exit $status
