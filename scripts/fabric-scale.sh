#!/usr/bin/env bash
set -euo pipefail

# Extend the existing deterministic emulator with frontier training-fabric
# metrics. The V5 section models typed heterogeneous capability populations;
# it never claims physical accelerator execution.
# This process never opens a socket and never starts a node. Every output is
# EMULATED evidence and must remain separate from real-process results.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
output_dir="$repo_root/.cache/v4-emulator"
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
  jq -e '.evidence_class == "EMULATED"
    and .v4_training.central_update_fan_in == 0
    and .v5_heterogeneous.central_scheduler_fan_in == 0
    and (.v5_heterogeneous.logical_peer_counts | index(100000)) != null' \
    "$output" >/dev/null
  printf '%s\n' "$output"
}

run_case 100 100 0.10 10 8 401
run_case 512 1000 0.25 30 8 403
run_case 2048 10000 0.50 30 8 407

if [[ "${V4_SCALE_100000:-0}" == 1 ]]; then
  run_case 4096 100000 0.50 30 8 409
fi

printf 'V4 emulator reports written below %s\n' "$output_dir"
