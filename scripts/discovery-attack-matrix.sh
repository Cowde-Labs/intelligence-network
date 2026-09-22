#!/usr/bin/env bash
set -euo pipefail

# Deterministic, in-process discovery attack-ratio experiments. This script never
# opens a socket or starts a node process. Its output is emulation evidence,
# not deployment evidence.
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0

nodes="${V2_ATTACK_NODES:-10000}"
lookups="${V2_ATTACK_LOOKUPS:-200}"
output_dir="$repo_root/.cache/v2-emulator"
mkdir -p -- "$output_dir"

seed=100
for sybil in 0 0.1 0.25 0.5 0.8; do
  for churn in 0 10 30 50; do
    output="$output_dir/attack-nodes-${nodes}-sybil-${sybil}-churn-${churn}.json"
    cargo run -p intelligence-emulator -- \
      --nodes "$nodes" \
      --sybil-ratio "$sybil" \
      --churn-percent "$churn" \
      --lookups "$lookups" \
      --workers 64 \
      --training-steps 8 \
      --seed "$seed" \
      --json > "$output"
    printf '%s\n' "$output"
    seed=$((seed + 1))
  done
done

printf 'V2 attack reports written below %s\n' "$output_dir"
