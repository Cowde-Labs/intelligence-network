#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

binary=${INTELLIGENCE_BIN:-"$repo_root/.cache/rust-target/debug/intelligence"}
if [[ ! -x "$binary" ]]; then
  cargo build -p intelligence-cli
fi
if [[ ! -x "$binary" ]]; then
  echo "intelligence binary is missing: $binary" >&2
  exit 1
fi

state_root=$(mktemp -d "${TMPDIR:-/tmp}/intelligence-compatibility-smoke.XXXXXX")
bootstrap_pid=""
worker_pid=""
requester_pid=""
cleanup() {
  for pid in "$requester_pid" "$worker_pid" "$bootstrap_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  rm -rf "$state_root"
}
trap cleanup EXIT INT TERM

mkdir -p "$state_root"/bootstrap "$state_root"/worker "$state_root"/requester
cat > "$state_root/bootstrap.toml" <<EOF
data_dir = "$state_root/bootstrap"
identity_path = "$state_root/bootstrap/identity.key"
admin_socket = "$state_root/bootstrap/node.sock"
listen_addr = "127.0.0.1:48101"
advertise_addr = "127.0.0.1:48101"
max_connections = 8
max_frame_size = 1048576
peer_ttl_seconds = 120

[storage]
quota_bytes = 33554432
max_artifact_bytes = 4194304

[runtime]
max_queued_jobs = 8
max_concurrent_jobs = 1
max_input_bytes = 65536
max_output_bytes = 1048576
default_timeout_ms = 5000
process_memory_bytes = 134217728
process_cpu_seconds = 5
work_dir = "$state_root/bootstrap/runtime-work"
EOF

cat > "$state_root/worker.toml" <<EOF
data_dir = "$state_root/worker"
identity_path = "$state_root/worker/identity.key"
admin_socket = "$state_root/worker/node.sock"
listen_addr = "127.0.0.1:48102"
advertise_addr = "127.0.0.1:48102"
bootstrap = ["127.0.0.1:48101"]
max_connections = 8
max_frame_size = 1048576
peer_ttl_seconds = 120

[storage]
quota_bytes = 33554432
max_artifact_bytes = 4194304

[runtime]
max_queued_jobs = 8
max_concurrent_jobs = 1
max_input_bytes = 65536
max_output_bytes = 1048576
default_timeout_ms = 5000
process_memory_bytes = 134217728
process_cpu_seconds = 5
work_dir = "$state_root/worker/runtime-work"

[[capabilities]]
name = "inference.text"
version = 1
public = true
accept_remote_jobs = true
kind = "builtin_text"
model = "builtin.tiny-sentiment.v1"
sandbox = "trusted_local"
max_input_bytes = 65536
max_output_bytes = 16384
memory_bytes = 67108864
cpu_millis = 1000
EOF

cat > "$state_root/requester.toml" <<EOF
data_dir = "$state_root/requester"
identity_path = "$state_root/requester/identity.key"
admin_socket = "$state_root/requester/node.sock"
listen_addr = "127.0.0.1:48103"
advertise_addr = "127.0.0.1:48103"
bootstrap = ["127.0.0.1:48101"]
max_connections = 8
max_frame_size = 1048576
peer_ttl_seconds = 120

[storage]
quota_bytes = 33554432
max_artifact_bytes = 4194304

[runtime]
max_queued_jobs = 8
max_concurrent_jobs = 1
max_input_bytes = 65536
max_output_bytes = 1048576
default_timeout_ms = 5000
process_memory_bytes = 134217728
process_cpu_seconds = 5
work_dir = "$state_root/requester/runtime-work"
EOF

"$binary" --config "$state_root/bootstrap.toml" run >"$state_root/bootstrap.log" 2>&1 &
bootstrap_pid=$!
"$binary" --config "$state_root/worker.toml" run >"$state_root/worker.log" 2>&1 &
worker_pid=$!
"$binary" --config "$state_root/requester.toml" run >"$state_root/requester.log" 2>&1 &
requester_pid=$!

wait_for_socket() {
  local config=$1
  for _ in $(seq 1 100); do
    if "$binary" --config "$config" status >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  echo "node did not become ready: $config" >&2
  return 1
}

wait_for_socket "$state_root/requester.toml"
first_ok=0
for _ in $(seq 1 160); do
  if "$binary" --config "$state_root/requester.toml" infer \
      --capability inference.text \
      --input "first remote testnet inference" \
      --deadline-ms 2000 \
      --max-output-bytes 16384 >/dev/null 2>&1; then
    first_ok=1
    break
  fi
  sleep 0.05
done
if [[ "$first_ok" != 1 ]]; then
  echo "initial remote inference did not complete" >&2
  exit 1
fi

"$binary" --config "$state_root/bootstrap.toml" shutdown >/dev/null
wait "$bootstrap_pid" 2>/dev/null || true
bootstrap_pid=""
sleep 0.5

for _ in $(seq 1 160); do
  if "$binary" --config "$state_root/requester.toml" infer \
      --capability inference.text \
      --input "inference after bootstrap outage" \
      --deadline-ms 2000 \
      --max-output-bytes 16384; then
    echo "compatibility smoke passed: remote inference survived bootstrap shutdown"
    exit 0
  fi
  sleep 0.05
done

echo "testnet failed; logs were in $state_root before cleanup" >&2
exit 1
