#!/usr/bin/env bash
set -euo pipefail

# Drive the existing deterministic emulator with distributed training-topology
# measurements. This never opens a socket or starts a node process. Every
# output is EMULATED evidence, not physical deployment evidence.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
output_dir="$repo_root/.cache/v3-emulator"
mkdir -p -- "$output_dir"

run_case() {
  local workers=$1 nodes=$2 sybil=$3 churn=$4 steps=$5 seed=$6
  local output="$output_dir/workers-${workers}-nodes-${nodes}-sybil-${sybil}-churn-${churn}.json"
  cargo run -p intelligence-emulator -- \
    --nodes "$nodes" \
    --sybil-ratio "$sybil" \
    --churn-percent "$churn" \
    --lookups 200 \
    --workers "$workers" \
    --training-steps "$steps" \
    --seed "$seed" \
    --json > "$output"
  printf '%s\n' "$output"
}

run_case 100 100 0.25 10 8 101
run_case 256 1000 0.25 30 8 103
run_case 1024 10000 0.50 30 8 107

if [[ "${V3_SCALE_100000:-0}" == 1 ]]; then
  run_case 4096 100000 0.50 30 8 109
fi

printf 'V3 emulator reports written below %s\n' "$output_dir"
