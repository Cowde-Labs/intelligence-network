#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

# cargo-fuzz's ASAN/libFuzzer mode requires a nightly toolchain. This stable
# fallback executes the same decoder harness with one hundred thousand
# deterministic, hostile inputs and panic isolation. CI runs the short smoke;
# this extended gate is run for release/security verification.
export INTELLIGENCE_FUZZ_ROUNDS=100000
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-.cache/rust-target}"
export CARGO_INCREMENTAL=0
cargo test -p intelligence-protocol --test fuzz_smoke -- --nocapture
INTELLIGENCE_FUZZ_ROUNDS=100000 cargo test -p intelligence-intelligence security::tests::malformed_v6_records_never_panic --lib -- --nocapture
