#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

binary=${INTELLIGENCE_BIN:-"$repo_root/.cache/rust-target/debug/intelligence"}
if [[ ! -x "$binary" ]]; then
  CARGO_TARGET_DIR=.cache/rust-target CARGO_INCREMENTAL=0 cargo build -p intelligence-cli
fi
[[ -x "$binary" ]] || { echo "intelligence binary is missing: $binary" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq is required by the testnet harness" >&2; exit 1; }
command -v ss >/dev/null 2>&1 || { echo "ss is required by the local testnet harness" >&2; exit 1; }

pick_base_port() {
  for _ in $(seq 1 100); do
    local candidate=$((40000 + RANDOM % 20000))
    local occupied=0
    for offset in 0 1 2 3 4; do
      if ! ss -H -lun 2>/dev/null | awk -v port=":$((candidate + offset))" '$4 ~ port "([^0-9]|$)" { found = 1 } END { exit found }'; then
        occupied=1
        break
      fi
    done
    if [[ "$occupied" == 0 ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  echo "could not find five consecutive UDP ports" >&2
  return 1
}

state_root=$(mktemp -d "${TMPDIR:-/tmp}/intelligence-local-testnet.XXXXXX")
keep_state=${TESTNET_KEEP_STATE:-0}
bootstrap_pid=""
relay_pid=""
worker_a_pid=""
worker_b_pid=""
requester_pid=""

cleanup() {
  for config in "$state_root/requester.toml" "$state_root/worker-b.toml" \
      "$state_root/worker-a.toml" "$state_root/relay.toml" "$state_root/bootstrap.toml"; do
    [[ -f "$config" ]] && "$binary" --config "$config" dev shutdown >/dev/null 2>&1 || true
  done
  for pid in "$requester_pid" "$worker_b_pid" "$worker_a_pid" "$relay_pid" "$bootstrap_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  if [[ "$keep_state" == 1 ]]; then
    echo "testnet state retained at $state_root" >&2
  else
    rm -rf "$state_root"
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$state_root"/{bootstrap,relay,worker-a,worker-b,requester}
artifact_source="$state_root/model-fixture.bin"
head -c 600000 /dev/zero > "$artifact_source"
base_port=$(pick_base_port)
bootstrap_port=$base_port
relay_port=$((base_port + 1))
worker_a_port=$((base_port + 2))
requester_port=$((base_port + 3))
worker_b_port=$((base_port + 4))

write_config() {
  local name=$1 port=$2 bootstrap=$3 relay_addresses=$4 capabilities=$5 prefer_relay=$6 relay_enabled=$7
  cat > "$state_root/$name.toml" <<EOF
data_dir = "$state_root/$name"
identity_path = "$state_root/$name/identity.key"
admin_socket = "$state_root/$name/node.sock"
listen_addr = "127.0.0.1:$port"
advertise_addr = "127.0.0.1:$port"
bootstrap = [$bootstrap]
relay_addresses = [$relay_addresses]
relay_enabled = $relay_enabled
relay_max_sessions = 16
relay_max_bytes = 4194304
prefer_relay = $prefer_relay
allow_private_addresses = true
hole_punch_enabled = true
hole_punch_max_attempts = 4
max_connections = 16
max_frame_size = 1048576
peer_ttl_seconds = 120

[storage]
quota_bytes = 33554432
max_artifact_bytes = 4194304

[runtime]
max_queued_jobs = 8
max_concurrent_jobs = 2
max_input_bytes = 65536
max_output_bytes = 1048576
default_timeout_ms = 5000
process_memory_bytes = 134217728
process_cpu_seconds = 5
work_dir = "$state_root/$name/runtime-work"
$capabilities
EOF
}

worker_capabilities=''
for capability in inference.text evaluation.text training.reference; do
  if [[ "$capability" == training.reference ]]; then
    model_line=''
    executor_kind='builtin_training'
  else
    model_line='model = "builtin.tiny-sentiment.v1"'
    executor_kind='builtin_text'
  fi
  worker_capabilities+="
[[capabilities]]
name = \"$capability\"
version = 1
public = true
accept_remote_jobs = true
kind = \"$executor_kind\"
$model_line
sandbox = \"trusted_local\"
max_input_bytes = 65536
max_output_bytes = 16384
memory_bytes = 67108864
cpu_millis = 1000
"
done

write_config bootstrap "$bootstrap_port" '' '' '' false false
write_config relay "$relay_port" "\"127.0.0.1:$bootstrap_port\"" '' '' false true
write_config worker-a "$worker_a_port" "\"127.0.0.1:$bootstrap_port\"" "\"127.0.0.1:$relay_port\"" "$worker_capabilities" false false
write_config worker-b "$worker_b_port" "\"127.0.0.1:$bootstrap_port\"" "\"127.0.0.1:$relay_port\"" "$worker_capabilities" false false
write_config requester "$requester_port" "\"127.0.0.1:$bootstrap_port\"" "\"127.0.0.1:$relay_port\"" '' true false

start_node() {
  local name=$1
  "$binary" --config "$state_root/$name.toml" run > "$state_root/$name.log" 2>&1 &
  case "$name" in
    bootstrap) bootstrap_pid=$! ;;
    relay) relay_pid=$! ;;
    worker-a) worker_a_pid=$! ;;
    worker-b) worker_b_pid=$! ;;
    requester) requester_pid=$! ;;
  esac
}

wait_ready() {
  local name=$1
  for _ in $(seq 1 160); do
    if "$binary" --config "$state_root/$name.toml" status >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  echo "node did not become ready: $name" >&2
  return 1
}

run_inference() {
  "$binary" --config "$state_root/requester.toml" infer \
    --capability inference.text --input "$1" --deadline-ms 3000 --max-output-bytes 16384 \
    2>/dev/null | jq -e '.state == "Succeeded"' >/dev/null
}

wait_for_peer_address() {
  local node=$1 address=$2
  for _ in $(seq 1 160); do
    if "$binary" --config "$state_root/$node.toml" --json network peers \
        | jq -e --arg address "$address" \
          'any(.[]; any(.addresses[]; . == $address))' >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.05
  done
  echo "peer record did not arrive: $node -> $address" >&2
  return 1
}

start_node bootstrap
start_node relay
start_node worker-a
start_node worker-b
start_node requester
for name in bootstrap relay worker-a worker-b requester; do wait_ready "$name"; done

initial_ok=0
for _ in $(seq 1 120); do
  if run_inference "initial V1 testnet inference"; then
    initial_ok=1
    break
  fi
  sleep 0.05
done
if [[ "$initial_ok" != 1 ]]; then
  echo "initial remote inference did not complete" >&2
  exit 1
fi

artifact_json=$("$binary" --config "$state_root/worker-a.toml" model register \
  --path "$artifact_source" --identity "testnet.fixture.v1" --format opaque --local-only)
artifact=$(jq -r '.artifact' <<<"$artifact_json")
[[ "$artifact" != null && "${#artifact}" == 64 ]] || { echo "artifact registration failed" >&2; exit 1; }
worker_a_id=$("$binary" --config "$state_root/worker-a.toml" --json identity | jq -r '.node_id')
wait_for_peer_address requester "127.0.0.1:$worker_a_port"
"$binary" --config "$state_root/requester.toml" artifact fetch --peer "$worker_a_id" --artifact "$artifact" \
  | jq -e '.verified == true' >/dev/null
"$binary" --config "$state_root/requester.toml" evaluate \
  --text "good and useful" --expected-label positive --deadline-ms 3000 \
  | jq -e '.state == "Succeeded" and .evidence.verified == true' >/dev/null

# Exercise persistence and reconnect before removing bootstrap infrastructure.
"$binary" --config "$state_root/worker-b.toml" dev shutdown >/dev/null
wait "$worker_b_pid" 2>/dev/null || true
worker_b_pid=""
start_node worker-b
wait_ready worker-b
wait_for_peer_address requester "127.0.0.1:$worker_b_port"
sleep 1

"$binary" --config "$state_root/bootstrap.toml" dev shutdown >/dev/null
wait "$bootstrap_pid" 2>/dev/null || true
bootstrap_pid=""
sleep 1
run_inference "inference after bootstrap outage"

# The requester prefers relay transport while it is available; after R goes
# away, send_to falls back to the authenticated direct path among known peers.
"$binary" --config "$state_root/relay.toml" dev shutdown >/dev/null
wait "$relay_pid" 2>/dev/null || true
relay_pid=""
sleep 1
run_inference "inference after relay outage"

training_json=$("$binary" --config "$state_root/requester.toml" train reference --workers 2 --steps 3)
jq -e '.improved == true and (.checkpoints | length) == 3' <<<"$training_json" >/dev/null

echo "local testnet passed: direct/relay inference, evaluation, artifact verification, worker restart, bootstrap outage, relay outage, and distributed reference training"
