#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

failed=0
check() {
  local pattern=$1
  local description=$2
  if rg -n --glob '*.rs' --glob '!**/tests/**' "$pattern" crates; then
    echo "banned architecture pattern found: $description" >&2
    failed=1
  fi
}

check 'struct [A-Za-z0-9_]*(Manager|Controller|Repository|Service)\b' 'ceremony type'
check 'trait [A-Za-z0-9_]*(Repository|Service|Controller|Manager|Mock)\b' 'mock-only or ceremony trait'
check 'mod (utils|common)\b' 'giant catch-all module'

if (( failed != 0 )); then
  exit 1
fi
echo "banned architecture scan passed"
