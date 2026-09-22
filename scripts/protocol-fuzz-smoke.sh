#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

echo "running bounded protocol decoder fuzz smoke"
cargo test -p intelligence-protocol --test fuzz_smoke
