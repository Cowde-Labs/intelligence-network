#!/usr/bin/env bash
set -euo pipefail

# Bounded, deterministic in-process discovery scale experiments. This script never
# opens a socket and never starts a node process. Its JSON is emulation
# evidence, not deployment evidence.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
output_dir="$repo_root/.cache/v2-emulator"
mkdir -p -- "$output_dir"

run_case() {
  local nodes=$1 sybil=$2 churn=$3 lookups=$4 workers=$5 steps=$6 seed=$7
  local output="$output_dir/nodes-${nodes}-sybil-${sybil}-churn-${churn}.json"
  cargo run -p intelligence-emulator -- \
    --nodes "$nodes" \
    --sybil-ratio "$sybil" \
    --churn-percent "$churn" \
    --lookups "$lookups" \
    --workers "$workers" \
    --training-steps "$steps" \
    --seed "$seed" \
    --json > "$output"
  printf '%s\n' "$output"
}

run_case 100 0.25 0 100 8 8 100
run_case 1000 0.25 0 200 32 8 1
run_case 10000 0.25 30 200 64 8 7

if [[ "${V2_SCALE_100000:-0}" == 1 ]]; then
  run_case 100000 0.25 30 200 128 8 11
fi

printf 'V2 emulator reports written below %s\n' "$output_dir"
