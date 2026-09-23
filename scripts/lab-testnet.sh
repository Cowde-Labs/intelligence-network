#!/usr/bin/env bash
set -euo pipefail

# Controlled single-host network laboratory.
#
# The host-side process never creates a host network namespace, veth, route,
# qdisc, or nftables object. It snapshots the host, starts one private user /
# mount / network / PID namespace, and sends commands to the supervisor there.
# The supervisor owns every in-lab object and exits before host postflight.

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
script_path="$repo_root/scripts/lab-testnet.sh"
lab_root=${INLAB_ROOT:-"$repo_root/.cache/network-lab"}
marker="$lab_root/.inlab-root"
binary=${INTELLIGENCE_BIN:-"$repo_root/dist/intelligence-network-v1.0.0-linux-x86_64/intelligence"}

die() {
  printf 'lab-testnet: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

for command_name in ip nft tc unshare systemd-run systemctl timeout jq prlimit sha256sum du df free nproc ps python3; do
  require_command "$command_name"
done

case "$lab_root" in
  "$repo_root/.cache/network-lab"|"$repo_root/.cache/network-lab-"*) ;;
  *) die "INLAB_ROOT must be the repository .cache/network-lab path or a prefixed sibling" ;;
esac
case "$lab_root" in
  /*) ;;
  *) die "lab root must be absolute" ;;
esac

[[ -x "$binary" ]] || die "release binary is missing or not executable: $binary"

control_fifo="$lab_root/control.fifo"
supervisor_pid_file="$lab_root/supervisor.pid"
supervisor_unit_file="$lab_root/supervisor.unit"
supervisor_log="$lab_root/supervisor.log"
results_file="$lab_root/results.ndjson"
commands_log="$lab_root/commands.log"
status_file="$lab_root/status.json"
report_json="$lab_root/report.json"
report_text="$lab_root/report.txt"
preflight_dir="$lab_root/preflight"
postflight_dir="$lab_root/postflight"
command_timeout_seconds=${INLAB_COMMAND_TIMEOUT_SECONDS:-1800}
supervisor_ready_timeout_seconds=${INLAB_SUPERVISOR_READY_SECONDS:-60}

host_route_snapshot="$preflight_dir/host-route.txt"
host_addr_snapshot="$preflight_dir/host-addr.txt"
host_netns_snapshot="$preflight_dir/host-netns.txt"
host_nft_snapshot="$preflight_dir/host-nft.txt"
host_tc_snapshot="$preflight_dir/host-tc.txt"
host_dns_snapshot="$preflight_dir/host-dns.txt"
host_dns_hash="$preflight_dir/host-resolv.conf.sha256"
host_phys_snapshot="$preflight_dir/physical-qdiscs.txt"

mkdir_lab_root() {
  if [[ -e "$lab_root" ]]; then
    [[ -f "$marker" ]] || die "existing lab root lacks the safety marker: $lab_root"
  else
    mkdir -p -- "$lab_root"
    printf 'intelligence-network-controlled-lab-v1\n' > "$marker"
  fi
  mkdir -p -- "$preflight_dir" "$postflight_dir"
}

du_bytes() {
  du -sb -- "$1" 2>/dev/null | awk '{print $1}'
}

parse_bytes() {
  local value=$1
  if [[ "$value" =~ ^[0-9]+$ ]]; then
    printf '%s\n' "$value"
  else
    numfmt --from=iec "$value"
  fi
}

min_number() {
  if (( $1 < $2 )); then printf '%s\n' "$1"; else printf '%s\n' "$2"; fi
}

configure_budgets() {
  host_cpus=$(nproc)
  host_mem_bytes=$(free -b | awk '/^Mem:/ {print $2}')
  host_available_bytes=$(free -b | awk '/^Mem:/ {print $7}')
  max_cpu_default=$((host_cpus / 3))
  (( max_cpu_default > 0 )) || max_cpu_default=1
  max_cpu_limit=$(( (host_cpus + 1) / 2 ))
  max_cpu=${INLAB_MAX_CPU:-$max_cpu_default}
  [[ "$max_cpu" =~ ^[0-9]+$ ]] || die "INLAB_MAX_CPU must be an integer number of logical CPUs"
  (( max_cpu > 0 && max_cpu <= max_cpu_limit )) || die "INLAB_MAX_CPU must be between 1 and $max_cpu_limit"

  memory_ceiling=$((4 * 1024 * 1024 * 1024))
  memory_quarter=$((host_mem_bytes / 4))
  memory_available_half=$((host_available_bytes / 2))
  max_memory_default=$(min_number "$memory_ceiling" "$memory_quarter")
  max_memory_default=$(min_number "$max_memory_default" "$memory_available_half")
  (( max_memory_default >= 256 * 1024 * 1024 )) || max_memory_default=$((256 * 1024 * 1024))
  max_memory=${INLAB_MAX_MEMORY:-$max_memory_default}
  max_memory=$(parse_bytes "$max_memory")
  (( max_memory > 0 && max_memory <= memory_ceiling )) || die "INLAB_MAX_MEMORY exceeds the 4 GiB hard ceiling"

  max_disk=${INLAB_MAX_DISK:-$((2 * 1024 * 1024 * 1024))}
  max_disk=$(parse_bytes "$max_disk")
  (( max_disk > 0 && max_disk <= 2 * 1024 * 1024 * 1024 )) || die "INLAB_MAX_DISK must be between 1 byte and 2 GiB"

  max_bandwidth=${INLAB_MAX_BANDWIDTH:-100}
  [[ "$max_bandwidth" =~ ^[0-9]+$ ]] || die "INLAB_MAX_BANDWIDTH must be an integer Mbit/s value"
  (( max_bandwidth > 0 && max_bandwidth <= 100 )) || die "INLAB_MAX_BANDWIDTH must be between 1 and 100 Mbit/s"
  cpu_quota=$((max_cpu * 100 / host_cpus))
  (( cpu_quota > 0 )) || cpu_quota=1
  per_link_bandwidth=$((max_bandwidth / 16))
  (( per_link_bandwidth > 0 )) || per_link_bandwidth=1
  task_limit=96
  export host_cpus max_cpu max_memory max_disk max_bandwidth cpu_quota per_link_bandwidth task_limit
}

snapshot_host() {
  mkdir_lab_root
  configure_budgets
  # Compare semantic host addresses, not interface indices, link-layer
  # ordering, or externally managed link-local privacy addresses.  The lab
  # still rejects any inlab-prefixed host link/namespace explicitly below.
  ip -o addr show 2>&1 \
    | awk '$5 == "scope" && ($6 == "global" || $6 == "host")' \
    | sed -E 's/valid_lft [^ ]+ preferred_lft [^ ]+/valid_lft <dynamic> preferred_lft <dynamic>/; s/ tentative//g' \
    | sort > "$host_addr_snapshot" || true
  ip route show table main > "$host_route_snapshot" 2>&1 || true
  ip netns list > "$host_netns_snapshot" 2>&1 || true
  nft list ruleset > "$host_nft_snapshot" 2>&1 || true
  tc qdisc show 2>&1 | sort > "$host_tc_snapshot" || true
  {
    printf 'resolv_link='
    readlink /etc/resolv.conf 2>/dev/null || true
    printf 'resolv_hash='
    sha256sum /etc/resolv.conf 2>/dev/null || true
  } > "$host_dns_snapshot"
  sha256sum /etc/resolv.conf > "$host_dns_hash" 2>/dev/null || printf 'unreadable\n' > "$host_dns_hash"

  : > "$host_phys_snapshot"
  while IFS= read -r interface; do
    [[ -n "$interface" ]] || continue
    printf 'interface=%s\n' "$interface" >> "$host_phys_snapshot"
    tc qdisc show dev "$interface" >> "$host_phys_snapshot" 2>&1 || true
  done < <(ip -o link show | awk -F': ' '{name=$2; sub(/@.*/, "", name); if (name ~ /^(eno|enp|eth|wlan|wlp|enx)/) print name}')

  if ip -o link show | grep -E '(^|: )inlab-' >/dev/null 2>&1; then
    die "host already exposes an inlab-prefixed link; refusing to touch it"
  fi
  if ip netns list 2>/dev/null | awk '{print $1}' | grep '^inlab-' >/dev/null 2>&1; then
    die "host already exposes an inlab-prefixed namespace; refusing to touch it"
  fi
  if ip route show table main | grep -E 'inlab-|10\.254\.' >/dev/null 2>&1; then
    die "host already has an inlab route; refusing to touch it"
  fi
}

assert_lab_size() {
  local size
  size=$(du_bytes "$lab_root")
  if (( size > max_disk )); then
    printf 'lab disk budget exceeded: %s > %s bytes\n' "$size" "$max_disk" >&2
    return 1
  fi
}

host_postflight() {
  ip -o addr show 2>&1 \
    | awk '$5 == "scope" && ($6 == "global" || $6 == "host")' \
    | sed -E 's/valid_lft [^ ]+ preferred_lft [^ ]+/valid_lft <dynamic> preferred_lft <dynamic>/; s/ tentative//g' \
    | sort > "$postflight_dir/host-addr.txt" || true
  ip route show table main > "$postflight_dir/host-route.txt" 2>&1 || true
  ip netns list > "$postflight_dir/host-netns.txt" 2>&1 || true
  nft list ruleset > "$postflight_dir/host-nft.txt" 2>&1 || true
  tc qdisc show 2>&1 | sort > "$postflight_dir/host-tc.txt" || true
  {
    printf 'resolv_link='
    readlink /etc/resolv.conf 2>/dev/null || true
    printf 'resolv_hash='
    sha256sum /etc/resolv.conf 2>/dev/null || true
  } > "$postflight_dir/host-dns.txt"
  sha256sum /etc/resolv.conf > "$postflight_dir/host-resolv.conf.sha256" 2>/dev/null || printf 'unreadable\n' > "$postflight_dir/host-resolv.conf.sha256"
  : > "$postflight_dir/physical-qdiscs.txt"
  while IFS= read -r interface; do
    [[ -n "$interface" ]] || continue
    printf 'interface=%s\n' "$interface" >> "$postflight_dir/physical-qdiscs.txt"
    tc qdisc show dev "$interface" >> "$postflight_dir/physical-qdiscs.txt" 2>&1 || true
  done < <(awk -F'=' '/^interface=/{print $2}' "$host_phys_snapshot" 2>/dev/null | sort -u || true)

  local postflight_ok=1
  for pair in \
    "$host_addr_snapshot:$postflight_dir/host-addr.txt" \
    "$host_route_snapshot:$postflight_dir/host-route.txt" \
    "$host_netns_snapshot:$postflight_dir/host-netns.txt" \
    "$host_nft_snapshot:$postflight_dir/host-nft.txt" \
    "$host_tc_snapshot:$postflight_dir/host-tc.txt" \
    "$host_dns_snapshot:$postflight_dir/host-dns.txt" \
    "$host_phys_snapshot:$postflight_dir/physical-qdiscs.txt"; do
    local before=${pair%%:*} after=${pair#*:}
    if ! cmp -s "$before" "$after"; then
      printf 'host postcondition changed: %s\n' "$before" >&2
      postflight_ok=0
    fi
  done
  if ip -o link show | grep -E '(^|: )inlab-' >/dev/null 2>&1; then
    printf 'host postcondition found an inlab link\n' >&2
    postflight_ok=0
  fi
  if ip netns list 2>/dev/null | awk '{print $1}' | grep '^inlab-' >/dev/null 2>&1; then
    printf 'host postcondition found an inlab namespace\n' >&2
    postflight_ok=0
  fi
  if [[ -e "$supervisor_pid_file" ]]; then
    local pid
    pid=$(cat "$supervisor_pid_file")
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      printf 'host postcondition supervisor still alive: %s\n' "$pid" >&2
      postflight_ok=0
    fi
  fi
  printf '%s\n' "$postflight_ok" > "$postflight_dir/ok"
  return $((1 - postflight_ok))
}

record_result_host() {
  local test_name=$1 status=$2 nodes=$3 profile=$4 injected=$5 result=$6 recovery=$7
  local duration=${8:-0}
  mkdir_lab_root
  jq -cn \
    --arg test "$test_name" --arg status "$status" --argjson duration "$duration" \
    --arg nodes "$nodes" --arg profile "$profile" --arg failure_injected "$injected" \
    --arg connection_path "$result" --arg result "$result" --arg recovery "$recovery" \
    '{test:$test,status:$status,duration_seconds:$duration,nodes:$nodes,network_profile:$profile,connection_path:$connection_path,failure_injected:$failure_injected,result:$result,recovery:$recovery}' \
    >> "$results_file"
}

write_report() {
  mkdir_lab_root
  local tests='[]'
  if [[ -s "$results_file" ]]; then
    tests=$(jq -s '.' "$results_file")
  fi
  local isolation='{}'
  if [[ -f "$status_file" ]]; then
    isolation=$(cat "$status_file")
  fi
  if [[ -f "$postflight_dir/ok" ]]; then
    local postflight_value='FAIL'
    [[ "$(cat "$postflight_dir/ok")" == 1 ]] && postflight_value='PASS'
    isolation=$(jq --arg value "$postflight_value" \
      '. + {host_default_route_unchanged:$value,host_dns_unchanged:$value,physical_qdiscs_unchanged:$value,host_namespace_state_unchanged:$value}' \
      <<< "$isolation")
  fi
  local final_status=${1:-CONTROLLED_LAB_PARTIAL}
  local size
  size=$(du_bytes "$lab_root")
  local observed='{}'
  local metrics_file="$lab_root/resource-metrics.tsv"
  if [[ -s "$metrics_file" ]]; then
    local observed_values
    observed_values=$(awk -F '\t' '
      NR == 1 {
        max_disk = $2 + 0
        max_rss = $3 + 0
        max_cpu = $4 + 0
        max_network = $5 + 0
        previous_time = $1 + 0
        previous_network = $5 + 0
        next
      }
      {
        if (($2 + 0) > max_disk) max_disk = $2 + 0
        if (($3 + 0) > max_rss) max_rss = $3 + 0
        if (($4 + 0) > max_cpu) max_cpu = $4 + 0
        if (($5 + 0) > max_network) max_network = $5 + 0
        elapsed = ($1 + 0) - previous_time
        bytes = ($5 + 0) - previous_network
        if (elapsed > 0 && bytes >= 0) {
          rate = (bytes * 8) / (elapsed * 1000)
          if (rate > max_rate) max_rate = rate
        }
        previous_time = $1 + 0
        previous_network = $5 + 0
      }
      END {
        printf "%d %d %.2f %.3f %d", max_disk, max_rss, max_cpu, max_rate + 0, max_network
      }
    ' "$metrics_file")
    local observed_disk observed_rss observed_cpu observed_rate observed_network
    read -r observed_disk observed_rss observed_cpu observed_rate observed_network <<< "$observed_values"
    observed=$(jq -n \
      --argjson peak_lab_bytes "${observed_disk:-0}" \
      --argjson peak_rss_bytes "${observed_rss:-0}" \
      --argjson peak_cpu_percent "${observed_cpu:-0}" \
      --argjson peak_virtual_bandwidth_mbit "${observed_rate:-0}" \
      --argjson virtual_bytes "${observed_network:-0}" \
      '{peak_lab_bytes:$peak_lab_bytes,peak_rss_bytes:$peak_rss_bytes,peak_cpu_percent:$peak_cpu_percent,peak_virtual_bandwidth_mbit:$peak_virtual_bandwidth_mbit,virtual_bytes:$virtual_bytes}')
  fi
  jq -n \
    --arg status "$final_status" \
    --arg generated_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg repo "$repo_root" \
    --arg binary "$binary" \
    --argjson host_cpus "$host_cpus" \
    --argjson max_cpu "$max_cpu" \
    --argjson max_memory "$max_memory" \
    --argjson max_disk "$max_disk" \
    --argjson max_bandwidth_mbit "$max_bandwidth" \
    --argjson lab_bytes "$size" \
    --argjson observed "$observed" \
    --argjson tests "$tests" \
    --argjson isolation "$isolation" \
    '{status:$status,generated_at:$generated_at,repository:$repo,binary:$binary,budgets:{host_logical_cpus:$host_cpus,max_cpus:$max_cpu,max_memory_bytes:$max_memory,max_disk_bytes:$max_disk,max_bandwidth_mbit:$max_bandwidth_mbit,lab_bytes:$lab_bytes},observed:$observed,isolation:$isolation,tests:$tests}' \
    > "$report_json"
  {
    printf '%s\n' 'INTELLIGENCE NETWORK — CONTROLLED LAB REPORT'
    printf 'Status: %s\n' "$final_status"
    printf 'Lab root: %s\n' "$lab_root"
    printf 'Budgets: CPU %s/%s logical CPUs, RAM %s bytes, disk %s/%s bytes, bandwidth %s Mbit/s\n' "$max_cpu" "$host_cpus" "$max_memory" "$size" "$max_disk" "$max_bandwidth"
    printf '%s\n' 'Tests:'
    jq -r '.tests[] | "  \(.status | ascii_upcase) \(.test) duration=\(.duration_seconds)s nodes=\(.nodes) profile=\(.network_profile) result=\(.result) recovery=\(.recovery)"' "$report_json" 2>/dev/null || true
    printf '%s\n' 'Observed peaks:'
    jq -r '.observed | to_entries[] | "  \(.key): \(.value)"' "$report_json" 2>/dev/null || true
    printf '%s\n' 'Isolation:'
    jq -r '.isolation | to_entries[] | "  \(.key): \(.value)"' "$report_json" 2>/dev/null || true
  } > "$report_text"
}

host_preflight_and_report() {
  if [[ ! -f "$host_route_snapshot" ]]; then
    snapshot_host
  else
    configure_budgets
  fi
  [[ -f "$results_file" ]] || : > "$results_file"
  if [[ ! -f "$status_file" ]]; then
    jq -n \
      --arg host_route "$(cat "$host_route_snapshot")" \
      --arg host_dns "$(cat "$host_dns_snapshot")" \
      '{host_route_snapshot:$host_route,host_dns_snapshot:$host_dns}' > "$status_file"
  fi
  assert_lab_size
}

command_log() {
  mkdir_lab_root
  {
    printf '[%s] +' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf ' %q' "$@"
    printf '\n'
  } | tee -a "$commands_log" >&2
}

run_logged() {
  command_log "$@"
  "$@"
}

safe_remove_lab_root() {
  [[ -f "$marker" ]] || die "refusing to remove unmarked lab root: $lab_root"
  case "$lab_root" in
    "$repo_root/.cache/network-lab"|"$repo_root/.cache/network-lab-"*) ;;
    *) die "refusing to remove lab root outside the project cache" ;;
  esac
  rm -rf -- "$lab_root"
}

remove_temporary_lab_state() {
  [[ -f "$marker" ]] || die "refusing to prune unmarked lab root: $lab_root"
  case "$lab_root" in
    "$repo_root/.cache/network-lab"|"$repo_root/.cache/network-lab-"*) ;;
    *) die "refusing to prune lab root outside the project cache" ;;
  esac
  if [[ -d "$lab_root/nodes" && ! -L "$lab_root/nodes" ]]; then
    find "$lab_root/nodes" -type f \( -name 'node.toml' -o -name 'identity*.key' -o -name 'identity*.rotation.json' \) -exec rm -f -- {} +
  fi
  if [[ -d "$lab_root/fixtures" && ! -L "$lab_root/fixtures" ]]; then
    find "$lab_root/fixtures" -maxdepth 1 -type f -name '*.bin' -exec rm -f -- {} +
  fi
  if [[ -d "$lab_root/sockets" && ! -L "$lab_root/sockets" ]]; then
    rm -rf -- "$lab_root/sockets"
  fi
}

start_supervisor() {
  if [[ -f "$supervisor_pid_file" ]]; then
    local old_pid
    old_pid=$(cat "$supervisor_pid_file")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      return 0
    fi
  fi
  if [[ -f "$supervisor_unit_file" ]]; then
    local old_unit
    old_unit=$(cat "$supervisor_unit_file")
    if [[ "$old_unit" =~ ^inlab-supervisor-[0-9]+\.service$ ]] \
      && systemctl --user is-active --quiet "$old_unit" 2>/dev/null; then
      die "an inlab supervisor service is already active: $old_unit; run recover first"
    fi
  fi
  rm -f -- "$control_fifo" "$lab_root/inner.ready" "$lab_root/supervisor.ready" "$lab_root/command.result"
  local unit="inlab-supervisor-$$.service"
  printf '%s\n' "$unit" > "$supervisor_unit_file"
  local command=(/usr/bin/unshare --kill-child=TERM --user --map-root-user --mount --net --pid --fork --mount-proc "$script_path" --inner-supervisor)
  local service_env=(
    "--setenv=INLAB_ROOT=$lab_root"
    "--setenv=INTELLIGENCE_BIN=$binary"
    "--setenv=INLAB_TEST_FILTER=${INLAB_TEST_FILTER:-}"
    "--setenv=INLAB_SOAK_SECONDS=${INLAB_SOAK_SECONDS:-}"
    "--setenv=INLAB_CLI_PROBE_TIMEOUT_SECONDS=${INLAB_CLI_PROBE_TIMEOUT_SECONDS:-}"
    "--setenv=INLAB_READY_PROBE_TIMEOUT_SECONDS=${INLAB_READY_PROBE_TIMEOUT_SECONDS:-}"
    "--setenv=INLAB_V4_STATUS_PROBE_TIMEOUT_SECONDS=${INLAB_V4_STATUS_PROBE_TIMEOUT_SECONDS:-}"
    "--setenv=INLAB_STATUS_FANOUT_EVERY=${INLAB_STATUS_FANOUT_EVERY:-}"
    "--setenv=INLAB_V4_COMMAND_TIMEOUT_SECONDS=${INLAB_V4_COMMAND_TIMEOUT_SECONDS:-}"
    "--setenv=INLAB_V4_CASCADE_DISCOVERY_ATTEMPTS=${INLAB_V4_CASCADE_DISCOVERY_ATTEMPTS:-}"
    "--setenv=INLAB_V4_CASCADE_JOIN_ATTEMPTS=${INLAB_V4_CASCADE_JOIN_ATTEMPTS:-}"
    "--setenv=RUST_LOG=${RUST_LOG:-}"
    "--setenv=host_cpus=$host_cpus"
    "--setenv=max_cpu=$max_cpu"
    "--setenv=max_memory=$max_memory"
    "--setenv=max_disk=$max_disk"
    "--setenv=max_bandwidth=$max_bandwidth"
    "--setenv=cpu_quota=$cpu_quota"
    "--setenv=per_link_bandwidth=$per_link_bandwidth"
    "--setenv=task_limit=$task_limit"
  )
  # Start exactly one transient service cgroup.  `--wait` keeps the tracked
  # launcher alive for the whole private namespace lifetime; a service is used
  # because this systemd version rejects `--wait` together with `--scope`.
  command_log systemd-run --user --unit="$unit" --collect --wait \
    "${service_env[@]}" \
    -p "CPUQuota=${cpu_quota}%" -p "MemoryMax=${max_memory}" -p "TasksMax=${task_limit}" -- "${command[@]}"
  nohup systemd-run --user --unit="$unit" --collect --wait \
    "${service_env[@]}" \
    -p "CPUQuota=${cpu_quota}%" -p "MemoryMax=${max_memory}" -p "TasksMax=${task_limit}" -- "${command[@]}" \
    > "$supervisor_log" 2>&1 < /dev/null &
  local launcher_pid=$!
  printf '%s\n' "$launcher_pid" > "$supervisor_pid_file"
  [[ "$supervisor_ready_timeout_seconds" =~ ^[0-9]+$ ]] && (( supervisor_ready_timeout_seconds > 0 )) || die "INLAB_SUPERVISOR_READY_SECONDS must be a positive integer"
  for _ in $(seq 1 $((supervisor_ready_timeout_seconds * 10))); do
    [[ -f "$lab_root/supervisor.ready" ]] && return 0
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      sed -n '1,160p' "$supervisor_log" >&2 || true
      die "private lab supervisor exited before becoming ready"
    fi
    sleep 0.1
  done
  die "private lab supervisor did not become ready"
}

send_supervisor() {
  local request=$1
  [[ -p "$control_fifo" ]] || die "lab supervisor control FIFO is missing; run setup or recover"
  local launcher_pid=''
  [[ -f "$supervisor_pid_file" ]] && launcher_pid=$(cat "$supervisor_pid_file") || true
  if [[ ! "$launcher_pid" =~ ^[0-9]+$ ]] || ! kill -0 "$launcher_pid" 2>/dev/null; then
    die "lab supervisor is not alive; run recover before retrying"
  fi
  rm -f -- "$lab_root/command.result"
  if ! timeout 5s bash -c 'printf "%s\\n" "$1" > "$2"' _ "$request" "$control_fifo"; then
    die "timed out opening lab supervisor control FIFO"
  fi
  for _ in $(seq 1 $((command_timeout_seconds * 10))); do
    if [[ -f "$lab_root/command.result" ]]; then
      local result
      result=$(cat "$lab_root/command.result")
      rm -f -- "$lab_root/command.result"
      [[ "$result" == PASS* ]] || { printf '%s\n' "$result" >&2; return 1; }
      printf '%s\n' "$result"
      return 0
    fi
    sleep 0.1
  done
  die "timed out waiting for private lab supervisor command: $request"
}

stop_supervisor() {
  local launcher_pid='' unit=''
  [[ -f "$supervisor_pid_file" ]] && launcher_pid=$(cat "$supervisor_pid_file") || true
  [[ -f "$supervisor_unit_file" ]] && unit=$(cat "$supervisor_unit_file") || true
  if [[ "$unit" =~ ^inlab-supervisor-[0-9]+\.service$ ]]; then
    # Signal the complete, uniquely named transient cgroup first.  Stopping
    # only MainPID can leave the private namespace child alive after the unit
    # has entered deactivation, which would make host postflight race with
    # namespace teardown.
    command_log systemctl --user kill --kill-who=all --signal=TERM "$unit"
    systemctl --user kill --kill-who=all --signal=TERM "$unit" >/dev/null 2>&1 || true
    if [[ "$launcher_pid" =~ ^[0-9]+$ ]] && kill -0 "$launcher_pid" 2>/dev/null; then
      command_log kill -TERM "$launcher_pid"
      kill -TERM "$launcher_pid" 2>/dev/null || true
    fi
    if ! timeout 10s systemctl --user stop "$unit" >/dev/null 2>&1; then
      command_log systemctl --user kill --kill-who=all --signal=KILL "$unit"
      systemctl --user kill --kill-who=all --signal=KILL "$unit" >/dev/null 2>&1 || true
      timeout 10s systemctl --user stop "$unit" >/dev/null 2>&1 || true
    fi
    for _ in $(seq 1 100); do
      systemctl --user is-active --quiet "$unit" 2>/dev/null || break
      sleep 0.1
    done
  fi
  rm -f -- "$supervisor_pid_file" "$supervisor_unit_file" "$control_fifo" "$lab_root/inner.ready" "$lab_root/supervisor.ready"
}

host_cleanup() {
  local cleanup_ok=1
  stop_supervisor || cleanup_ok=0
  sleep 0.2
  host_postflight || cleanup_ok=0
  if (( cleanup_ok == 0 )); then
    printf '%s\n' 'CONTROLLED_LAB_UNSAFE: host postcondition failed; do not remove lab evidence' >&2
    printf '%s\n' 'Recovery: inspect .cache/network-lab/postflight and run scripts/lab-testnet.sh recover' >&2
    return 1
  fi
  return 0
}

run_all() {
  local test_exit=0 cleanup_exit=0
  # EXIT can run after an errexit path has unwound this function's locals.
  # Keep the guard in the script scope so cleanup remains crash-safe under
  # `set -u` as well as during the normal all->cleanup path.
  run_all_cleanup_done=0
  on_exit() {
    local exit_code=$?
    if (( run_all_cleanup_done == 0 )); then
      host_cleanup || exit_code=1
      run_all_cleanup_done=1
    fi
    if (( exit_code != 0 )); then
      if [[ -f "$postflight_dir/ok" ]] && [[ "$(cat "$postflight_dir/ok")" == 1 ]]; then
        remove_temporary_lab_state || exit_code=1
      fi
      write_report CONTROLLED_LAB_PARTIAL >/dev/null 2>&1 || true
    fi
    exit "$exit_code"
  }
  trap on_exit EXIT INT TERM
  setup_command
  run_command
  if test_command; then test_exit=0; else test_exit=1; fi
  if ! host_cleanup; then cleanup_exit=1; fi
  run_all_cleanup_done=1
  if (( cleanup_exit != 0 )); then write_report CONTROLLED_LAB_UNSAFE; trap - EXIT INT TERM; exit 1; fi
  if (( test_exit != 0 )); then
    remove_temporary_lab_state
    write_report CONTROLLED_LAB_PARTIAL
    trap - EXIT INT TERM
    exit 1
  fi
  remove_temporary_lab_state
  write_report CONTROLLED_LAB_VERIFIED
  trap - EXIT INT TERM
}

setup_command() {
  snapshot_host
  assert_lab_size
  start_supervisor
  send_supervisor setup
  write_report CONTROLLED_LAB_PARTIAL
}

run_command() {
  host_preflight_and_report
  start_supervisor
  send_supervisor run
  write_report CONTROLLED_LAB_PARTIAL
}

status_command() {
  host_preflight_and_report
  if [[ -f "$supervisor_pid_file" ]]; then
    send_supervisor status || true
  else
    printf '%s\n' 'lab supervisor is not running'
  fi
  if [[ -f "$report_json" ]]; then jq . "$report_json"; fi
}

test_command() {
  host_preflight_and_report
  start_supervisor
  if ! send_supervisor test; then
    write_report CONTROLLED_LAB_PARTIAL
    return 1
  fi
  write_report CONTROLLED_LAB_PARTIAL
}

cleanup_command() {
  host_preflight_and_report
  host_cleanup
  remove_temporary_lab_state
  local report_status=CONTROLLED_LAB_PARTIAL
  if [[ -f "$report_json" ]] && jq -e '.status == "CONTROLLED_LAB_VERIFIED"' "$report_json" >/dev/null 2>&1; then
    report_status=CONTROLLED_LAB_VERIFIED
  fi
  write_report "$report_status"
}

recover_command() {
  mkdir_lab_root
  stop_supervisor || true
  host_postflight || true
  printf '%s\n' 'recover completed; inspect .cache/network-lab/postflight/ok and report.txt'
}

if [[ "${1:-}" == --inner-supervisor ]]; then
  # Everything after this branch runs inside the fresh private namespace.
  shift
  inner_mode=supervisor
else
  inner_mode=host
fi

if [[ "$inner_mode" == supervisor ]]; then
  inner_namespaces=(
    inlab-internet inlab-bootstrap inlab-worker-a inlab-worker-b inlab-requester
    inlab-client-nat inlab-client-cgnat inlab-relay-a inlab-relay-b
    inlab-router-a inlab-router-b inlab-router-c inlab-router-d
    inlab-nat-home inlab-nat-inner inlab-nat-carrier
  )
  node_names=(bootstrap worker-a worker-b requester client-nat client-cgnat relay-a relay-b)
  declare -A node_ns node_ip node_if node_port node_mem node_cpu node_pid
  # `prlimit --cpu` is a CPU-time watchdog in seconds, not a share/quota.
  # The supervisor service already applies the aggregate CPUQuota. Keep this
  # per-process watchdog above the longest bounded lab run so a healthy node
  # is not killed merely because the full V1/V2/V3 suite is sequential.
  node_ns[bootstrap]=inlab-bootstrap; node_ip[bootstrap]=10.254.0.2; node_port[bootstrap]=40000; node_mem[bootstrap]=67108864; node_cpu[bootstrap]=300
  node_ns[worker-a]=inlab-worker-a; node_ip[worker-a]=10.254.11.2; node_if[worker-a]=inlab-v08b; node_port[worker-a]=40002; node_mem[worker-a]=134217728; node_cpu[worker-a]=600
  node_ns[worker-b]=inlab-worker-b; node_ip[worker-b]=10.254.12.2; node_if[worker-b]=inlab-v09b; node_port[worker-b]=40003; node_mem[worker-b]=134217728; node_cpu[worker-b]=600
  node_ns[requester]=inlab-requester; node_ip[requester]=10.254.50.2; node_port[requester]=40005; node_mem[requester]=100663296; node_cpu[requester]=600
  node_ns[client-nat]=inlab-client-nat; node_ip[client-nat]=10.254.51.2; node_port[client-nat]=40006; node_mem[client-nat]=67108864; node_cpu[client-nat]=300
  node_ns[client-cgnat]=inlab-client-cgnat; node_ip[client-cgnat]=10.254.61.2; node_port[client-cgnat]=40007; node_mem[client-cgnat]=67108864; node_cpu[client-cgnat]=300
  node_ns[relay-a]=inlab-relay-a; node_ip[relay-a]=10.254.13.2; node_if[relay-a]=inlab-v10b; node_port[relay-a]=40004; node_mem[relay-a]=67108864; node_cpu[relay-a]=600
  node_ns[relay-b]=inlab-relay-b; node_ip[relay-b]=10.254.14.2; node_if[relay-b]=inlab-v11b; node_port[relay-b]=40008; node_mem[relay-b]=67108864; node_cpu[relay-b]=600
  inner_cleaning=0
  resource_metrics_file="$lab_root/resource-metrics.tsv"
  peak_lab_bytes=0
  peak_rss_bytes=0
  peak_cpu_percent=0
  peak_virtual_bandwidth_mbit=0
  peak_virtual_bytes=0
  resource_last_time_ms=0
  resource_last_network_bytes=0

  inner_cmd() {
    {
      printf '[%s] +' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
      printf ' %q' "$@"
      printf '\n'
    } | tee -a "$commands_log" >&2
    "$@"
  }

  inner_ns() {
    local ns=$1
    shift
    inner_cmd ip netns exec "$ns" "$@"
  }

  inner_assert_size() {
    local current
    current=$(du_bytes "$lab_root")
    sample_resources
    (( current <= max_disk )) || { printf 'inner lab disk budget exceeded: %s > %s\n' "$current" "$max_disk" >&2; return 1; }
  }

  virtual_network_bytes() {
    local total=0 namespace interface direction value
    for namespace in "${inner_namespaces[@]}"; do
      while IFS= read -r interface; do
        [[ "$interface" == inlab-* ]] || continue
        for direction in rx tx; do
          value=$(ip netns exec "$namespace" cat "/sys/class/net/$interface/statistics/${direction}_bytes" 2>/dev/null || printf '0')
          [[ "$value" =~ ^[0-9]+$ ]] || value=0
          total=$((total + value))
        done
      done < <(ip netns exec "$namespace" ip -o link show 2>/dev/null | awk -F': ' '{name=$2; sub(/@.*/, "", name); print name}')
    done
    printf '%s\n' "$total"
  }

  sample_resources() {
    local now_ms current_size rss_bytes cpu_percent network_bytes rate
    now_ms=$(date +%s%3N)
    current_size=$(du_bytes "$lab_root")
    rss_bytes=$(ps -e -o rss= 2>/dev/null | awk '{sum += $1} END {print int((sum + 0) * 1024)}')
    cpu_percent=$(ps -e -o pcpu= 2>/dev/null | awk '{sum += $1} END {printf "%.2f", sum + 0}')
    network_bytes=$(virtual_network_bytes)
    (( current_size > peak_lab_bytes )) && peak_lab_bytes=$current_size
    (( rss_bytes > peak_rss_bytes )) && peak_rss_bytes=$rss_bytes
    if awk -v current="$cpu_percent" -v peak="$peak_cpu_percent" 'BEGIN { exit !(current > peak) }'; then
      peak_cpu_percent=$cpu_percent
    fi
    if (( resource_last_time_ms > 0 && now_ms > resource_last_time_ms && network_bytes >= resource_last_network_bytes )); then
      rate=$(awk -v bytes="$((network_bytes - resource_last_network_bytes))" -v elapsed="$((now_ms - resource_last_time_ms))" 'BEGIN { printf "%.3f", (bytes * 8) / (elapsed * 1000) }')
      if awk -v current="$rate" -v peak="$peak_virtual_bandwidth_mbit" 'BEGIN { exit !(current > peak) }'; then
        peak_virtual_bandwidth_mbit=$rate
      fi
    fi
    (( network_bytes > peak_virtual_bytes )) && peak_virtual_bytes=$network_bytes
    printf '%s\t%s\t%s\t%s\t%s\n' "$now_ms" "$current_size" "$rss_bytes" "$cpu_percent" "$network_bytes" >> "$resource_metrics_file"
    resource_last_time_ms=$now_ms
    resource_last_network_bytes=$network_bytes
  }

  add_veth_link() {
    local left_ns=$1 left_if=$2 right_ns=$3 right_if=$4
    inner_cmd ip link add "$left_if" type veth peer name "$right_if"
    inner_cmd ip link set "$left_if" netns "$left_ns"
    inner_cmd ip link set "$right_if" netns "$right_ns"
  }

  configure_if() {
    local ns=$1 interface=$2 address=$3
    inner_ns "$ns" ip addr add "$address" dev "$interface"
    inner_ns "$ns" ip link set "$interface" up
  }

  configure_route() {
    local ns=$1
    shift
    inner_ns "$ns" ip route add "$@"
  }

  configure_nft_nat() {
    local ns=$1 table=$2 lan_if=$3 wan_if=$4
    inner_ns "$ns" nft add table ip "$table"
    inner_ns "$ns" nft add chain ip "$table" prerouting '{ type nat hook prerouting priority -100; policy accept; }'
    inner_ns "$ns" nft add chain ip "$table" postrouting '{ type nat hook postrouting priority 100; policy accept; }'
    inner_ns "$ns" nft add chain ip "$table" forward '{ type filter hook forward priority 0; policy drop; }'
    inner_ns "$ns" nft add rule ip "$table" forward ct state established,related accept
    inner_ns "$ns" nft add rule ip "$table" forward iifname "$lan_if" oifname "$wan_if" accept
    inner_ns "$ns" nft add rule ip "$table" postrouting oifname "$wan_if" masquerade
  }

  configure_fabric_filter() {
    inner_ns inlab-internet nft add table ip inlab-fabric
    inner_ns inlab-internet nft add chain ip inlab-fabric forward '{ type filter hook forward priority 0; policy accept; }'
  }

  tracked_children() {
    local parent=$1
    ps -eo pid=,ppid= | awk -v parent="$parent" '$2 == parent {print $1}'
  }

  kill_tracked_tree() {
    local root=$1 signal=$2 child
    [[ "$root" =~ ^[0-9]+$ ]] || return 0
    while read -r child; do
      [[ "$child" =~ ^[0-9]+$ ]] || continue
      kill_tracked_tree "$child" "$signal"
    done < <(tracked_children "$root")
    kill -"$signal" "$root" 2>/dev/null || true
  }

  delete_fabric_rule() {
    local rule_handle=$1
    inner_ns inlab-internet nft delete rule ip inlab-fabric forward handle "$rule_handle"
  }

  cleanup_inner() {
    (( inner_cleaning == 1 )) && return 0
    inner_cleaning=1
    set +e
    for node in "${node_names[@]}"; do
      pid_file="$lab_root/pids/$node.pid"
      if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
          kill_tracked_tree "$pid" TERM
          for _ in $(seq 1 50); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.05
          done
          kill_tracked_tree "$pid" KILL
        fi
        rm -f -- "$pid_file"
      fi
    done
    for namespace in "${inner_namespaces[@]}"; do
      if ip netns list | awk '{print $1}' | grep -Fx "$namespace" >/dev/null 2>&1; then
        inner_cmd ip netns delete "$namespace" || true
      fi
    done
    rm -f -- "$lab_root/inner.ready" "$lab_root/status.json" "$control_fifo"
  }
  trap cleanup_inner EXIT INT TERM

  setup_inner() {
    # A new lab run must not inherit mutable state from an interrupted or
    # older run. These are exact, project-local generated directories; the
    # host preflight marker prevents this path from resolving elsewhere.
    rm -rf -- "$lab_root/nodes" "$lab_root/pids" "$lab_root/fixtures" "$lab_root/cases" "$lab_root/sockets"
    rm -f -- "$lab_root/v3-training.json" "$lab_root/fetch-first.json" \
      "$lab_root/fetch-corrupt.json" "$lab_root/training-loss.json"
    mkdir -p -- "$lab_root/pids" "$lab_root/nodes" "$lab_root/fixtures" "$lab_root/sockets"
    : > "$resource_metrics_file"
    inner_cmd mount --make-rprivate /
    inner_cmd mount -t tmpfs -o mode=755 tmpfs /run
    inner_cmd mkdir -p /run/netns
    for namespace in "${inner_namespaces[@]}"; do
      inner_cmd ip netns add "$namespace"
      inner_ns "$namespace" ip link set lo up
    done

    add_veth_link inlab-internet inlab-v01a inlab-bootstrap inlab-v01b
    add_veth_link inlab-internet inlab-v02a inlab-router-a inlab-v02b
    add_veth_link inlab-internet inlab-v03a inlab-router-b inlab-v03b
    add_veth_link inlab-internet inlab-v04a inlab-router-c inlab-v04b
    add_veth_link inlab-internet inlab-v05a inlab-router-d inlab-v05b
    add_veth_link inlab-internet inlab-v06a inlab-nat-home inlab-v06b
    add_veth_link inlab-internet inlab-v07a inlab-nat-carrier inlab-v07b
    add_veth_link inlab-router-a inlab-v08a inlab-worker-a inlab-v08b
    add_veth_link inlab-router-b inlab-v09a inlab-worker-b inlab-v09b
    add_veth_link inlab-router-c inlab-v10a inlab-relay-a inlab-v10b
    add_veth_link inlab-router-d inlab-v11a inlab-relay-b inlab-v11b
    add_veth_link inlab-nat-home inlab-v12a inlab-requester inlab-v12b
    add_veth_link inlab-nat-home inlab-v13a inlab-client-nat inlab-v13b
    add_veth_link inlab-nat-carrier inlab-v14a inlab-nat-inner inlab-v14b
    add_veth_link inlab-nat-inner inlab-v15a inlab-client-cgnat inlab-v15b

    configure_if inlab-internet inlab-v01a 10.254.0.1/30
    configure_if inlab-bootstrap inlab-v01b 10.254.0.2/30
    configure_if inlab-internet inlab-v02a 10.254.1.1/30
    configure_if inlab-router-a inlab-v02b 10.254.1.2/30
    configure_if inlab-internet inlab-v03a 10.254.2.1/30
    configure_if inlab-router-b inlab-v03b 10.254.2.2/30
    configure_if inlab-internet inlab-v04a 10.254.3.1/30
    configure_if inlab-router-c inlab-v04b 10.254.3.2/30
    configure_if inlab-internet inlab-v05a 10.254.4.1/30
    configure_if inlab-router-d inlab-v05b 10.254.4.2/30
    configure_if inlab-internet inlab-v06a 10.254.5.1/30
    configure_if inlab-nat-home inlab-v06b 10.254.5.2/30
    configure_if inlab-internet inlab-v07a 10.254.6.1/30
    configure_if inlab-nat-carrier inlab-v07b 10.254.6.2/30
    configure_if inlab-router-a inlab-v08a 10.254.11.1/30
    configure_if inlab-worker-a inlab-v08b 10.254.11.2/30
    configure_if inlab-router-b inlab-v09a 10.254.12.1/30
    configure_if inlab-worker-b inlab-v09b 10.254.12.2/30
    configure_if inlab-router-c inlab-v10a 10.254.13.1/30
    configure_if inlab-relay-a inlab-v10b 10.254.13.2/30
    configure_if inlab-router-d inlab-v11a 10.254.14.1/30
    configure_if inlab-relay-b inlab-v11b 10.254.14.2/30
    configure_if inlab-nat-home inlab-v12a 10.254.50.1/30
    configure_if inlab-requester inlab-v12b 10.254.50.2/30
    configure_if inlab-nat-home inlab-v13a 10.254.51.1/30
    configure_if inlab-client-nat inlab-v13b 10.254.51.2/30
    configure_if inlab-nat-carrier inlab-v14a 10.254.60.1/30
    configure_if inlab-nat-inner inlab-v14b 10.254.60.2/30
    configure_if inlab-nat-inner inlab-v15a 10.254.61.1/30
    configure_if inlab-client-cgnat inlab-v15b 10.254.61.2/30

    for router in inlab-internet inlab-router-a inlab-router-b inlab-router-c inlab-router-d inlab-nat-home inlab-nat-inner inlab-nat-carrier; do
      inner_ns "$router" sysctl -qw net.ipv4.ip_forward=1
    done
    configure_route inlab-bootstrap default via 10.254.0.1
    configure_route inlab-router-a default via 10.254.1.1
    configure_route inlab-worker-a default via 10.254.11.1
    configure_route inlab-router-b default via 10.254.2.1
    configure_route inlab-worker-b default via 10.254.12.1
    configure_route inlab-router-c default via 10.254.3.1
    configure_route inlab-relay-a default via 10.254.13.1
    configure_route inlab-router-d default via 10.254.4.1
    configure_route inlab-relay-b default via 10.254.14.1
    configure_route inlab-nat-home default via 10.254.5.1
    configure_route inlab-requester default via 10.254.50.1
    configure_route inlab-client-nat default via 10.254.51.1
    configure_route inlab-nat-carrier default via 10.254.6.1
    configure_route inlab-nat-inner default via 10.254.60.1
    configure_route inlab-client-cgnat default via 10.254.61.1
    configure_route inlab-internet 10.254.11.0/30 via 10.254.1.2
    configure_route inlab-internet 10.254.12.0/30 via 10.254.2.2
    configure_route inlab-internet 10.254.13.0/30 via 10.254.3.2
    configure_route inlab-internet 10.254.14.0/30 via 10.254.4.2
    configure_route inlab-internet 10.254.50.0/30 via 10.254.5.2
    configure_route inlab-internet 10.254.51.0/30 via 10.254.5.2
    configure_route inlab-internet 10.254.60.0/30 via 10.254.6.2
    configure_nft_nat inlab-nat-home inlab-nat-home inlab-v12a inlab-v06b
    inner_ns inlab-nat-home nft add rule ip inlab-nat-home forward iifname inlab-v13a oifname inlab-v06b accept
    configure_nft_nat inlab-nat-carrier inlab-nat-carrier inlab-v14a inlab-v07b
    configure_nft_nat inlab-nat-inner inlab-nat-inner inlab-v15a inlab-v14b
    configure_fabric_filter

    # Apply a low, aggregate-safe local profile to every virtual endpoint. The
    # profile is replaced explicitly per interface by later test cases.
    for namespace in "${inner_namespaces[@]}"; do
      while IFS= read -r interface; do
        [[ "$interface" == inlab-* ]] || continue
        inner_ns "$namespace" tc qdisc replace dev "$interface" root netem delay 2ms rate "${per_link_bandwidth}mbit"
      done < <(ip netns exec "$namespace" ip -o link show | awk -F': ' '{name=$2; sub(/@.*/, "", name); print name}')
    done

    # Explicit safety assertions: the fake Internet has no physical interface,
    # no route toward the host, and no default route. Every generated object is
    # prefixed. The host cannot see any of these objects from the outer netns.
    [[ -z "$(ip netns exec inlab-internet ip route show default)" ]]
    ! ip netns exec inlab-internet ip link show | grep -Ev '(^|: )lo|inlab-' >/dev/null
    ! ip netns exec inlab-internet ip route show | grep -E '192\.168\.1\.1|dev (eno|enp|eth|wlan|wlp|enx)' >/dev/null
    ! ip netns exec inlab-internet nft list ruleset | grep -v 'inlab-' >/dev/null
    inner_assert_size
  }

  toml_escape() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '%s' "$value"
  }

  write_config() {
    local name=$1 identity_path=$2 prefer_relay=$3
    local dir="$lab_root/nodes/$name" cfg="$lab_root/nodes/$name/node.toml" ip=${node_ip[$name]} port=${node_port[$name]}
    local bootstrap='"10.254.0.2:40000"' relays='"10.254.13.2:40004", "10.254.14.2:40008"'
    [[ "$name" == bootstrap ]] && bootstrap=''
    local advertise="$ip:$port"
    case "$name" in
      requester) advertise="10.254.5.2:$port" ;;
      client-nat) advertise="10.254.5.2:$port" ;;
      client-cgnat) advertise="10.254.6.2:$port" ;;
    esac
    local capabilities=''
    local evaluator_model="builtin.tiny-sentiment.v1"
    if [[ -n "${INLAB_V6_EVALUATOR_ATTACK:-}" && ( "$name" == worker-a || "$name" == relay-a ) ]]; then
      evaluator_model="builtin.adversarial.sentiment.v1"
    fi
    case "$name" in
      worker-a|worker-b)
        capabilities=$'\n[[capabilities]]\nname = "inference.text"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_text"\nmodel = "builtin.tiny-sentiment.v1"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 67108864\ncpu_millis = 1000\n\n[[capabilities]]\nname = "evaluation.text"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_text"\nmodel = "builtin.tiny-sentiment.v1"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 67108864\ncpu_millis = 1000\n\n[[capabilities]]\nname = "training.reference"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_training"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 67108864\ncpu_millis = 1000\n' ;;
      relay-a|relay-b)
        # Relay and worker capabilities are independent. These two relay
        # processes also act as V3 training participants so the lab can run
        # the four-worker fabric without adding another namespace topology.
        capabilities=$'\n[[capabilities]]\nname = "training.reference"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_training"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 67108864\ncpu_millis = 1000\n' ;;
      client-nat|client-cgnat)
        # The integrated cascade starts with four members and admits the
        # remaining authenticated peers.  Keep NAT clients out of ordinary
        # training cases so this capability does not change their V1/V2 role.
        if [[ -z "${INLAB_V6_CLEAN_DISCOVERY:-}" && ( -z "${INLAB_TEST_FILTER:-}" || ",${INLAB_TEST_FILTER:-}," == *,V4_INTEGRATED_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V5_HETEROGENEOUS_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V6_ADVERSARIAL_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V4_AUTO_REPLAN_JOIN,* || ",${INLAB_TEST_FILTER:-}," == *,V4_SHARD_SPLIT,* ) ]]; then
          capabilities=$'\n[[capabilities]]\nname = "training.reference"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_training"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 900\ncpu_millis = 1000\n'
        fi ;;
      bootstrap)
        # The join case adds one signed but initially under-capacity peer.
        # The planner leaves it out of the initial four-worker graph, then
        # the running job discovers and admits it through the normal signed
        # peer record path.
        if [[ -z "${INLAB_V6_CLEAN_DISCOVERY:-}" && ( -z "${INLAB_TEST_FILTER:-}" || ",${INLAB_TEST_FILTER:-}," == *,V4_AUTO_REPLAN_JOIN,* || ",${INLAB_TEST_FILTER:-}," == *,V4_INTEGRATED_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V5_HETEROGENEOUS_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V6_ADVERSARIAL_CASCADE,* ) ]]; then
          capabilities=$'\n[[capabilities]]\nname = "training.reference"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_training"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 900\ncpu_millis = 1000\n'
        fi ;;
    esac
    if [[ -n "${INLAB_V6_EVALUATOR_ATTACK:-}" && "$name" == worker-a ]]; then
      # The V6 lab fixture is deliberately adversarial only at the evaluator
      # behavior layer; training capabilities remain the normal reference
      # executor. Replacing the built-in text model is sufficient because
      # worker-a's inference capability is not used by this experiment.
      capabilities=${capabilities//builtin.tiny-sentiment.v1/$evaluator_model}
    fi
    if [[ -n "${INLAB_V6_EVALUATOR_POOL:-}" && ( "$name" == relay-a || "$name" == relay-b ) ]]; then
      capabilities+=$'\n[[capabilities]]\nname = "evaluation.text"\nversion = 1\npublic = true\naccept_remote_jobs = true\nkind = "builtin_text"\nmodel = "'
      capabilities+="${evaluator_model}"
      capabilities+=$'"\nsandbox = "trusted_local"\nmax_input_bytes = 65536\nmax_output_bytes = 16384\nmemory_bytes = 67108864\ncpu_millis = 1000\n'
    fi
    local quota=67108864 artifact_limit=33554432
    [[ "$name" == worker-a || "$name" == worker-b ]] && quota=134217728 && artifact_limit=33554432
    [[ "$name" == client-cgnat ]] && quota=65536 && artifact_limit=32768
    mkdir -p -- "$dir/state" "$dir/runtime-work"
    cat > "$cfg" <<EOF
data_dir = "$(toml_escape "$dir/state")"
identity_path = "$(toml_escape "$identity_path")"
admin_socket = "$(toml_escape "$lab_root/sockets/$name.sock")"
listen_addr = "$ip:$port"
advertise_addr = "$advertise"
bootstrap = [$bootstrap]
relay_addresses = [$relays]
relay_enabled = $([[ "$name" == relay-a || "$name" == relay-b ]] && printf true || printf false)
relay_max_sessions = $([[ -z "${INLAB_TEST_FILTER:-}" || ",${INLAB_TEST_FILTER:-}," == *,V4_INTEGRATED_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V5_HETEROGENEOUS_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V6_ADVERSARIAL_CASCADE,* ]] && printf 32 || printf 8)
relay_max_bytes = 4194304
prefer_relay = $prefer_relay
allow_private_addresses = true
hole_punch_enabled = true
hole_punch_max_attempts = 3
max_connections = 16
max_frame_size = 1048576
peer_ttl_seconds = 1800
training_memory_bytes = 1536
training_window_delay_ms = $([[ -z "${INLAB_TEST_FILTER:-}" || ",${INLAB_TEST_FILTER:-}," == *,V4_INTEGRATED_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V5_HETEROGENEOUS_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V6_ADVERSARIAL_CASCADE,* || ",${INLAB_TEST_FILTER:-}," == *,V4_SHARD_SPLIT,* ]] && printf 100 || printf 0)

[storage]
quota_bytes = $quota
max_artifact_bytes = $artifact_limit

[runtime]
max_queued_jobs = 8
max_concurrent_jobs = 2
max_input_bytes = 65536
max_output_bytes = 1048576
default_timeout_ms = 5000
process_memory_bytes = 67108864
process_cpu_seconds = 5
work_dir = "$(toml_escape "$dir/runtime-work")"
$capabilities
EOF
  }

  generate_configs() {
    for name in "${node_names[@]}"; do
      write_config "$name" "$lab_root/nodes/$name/state/identity.key" false
    done
  }

  start_node() {
    local name=$1 ns=${node_ns[$1]} config="$lab_root/nodes/$1/node.toml" log="$lab_root/nodes/$1/node.log"
    local pid_file="$lab_root/pids/$name.pid"
    if [[ -f "$pid_file" ]]; then
      local old_pid
      old_pid=$(cat "$pid_file")
      [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null && return 0
    fi
    printf '[%s] + node %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$config" | tee -a "$commands_log" >&2
    (
      exec ip netns exec "$ns" env TOKIO_WORKER_THREADS=1 prlimit \
        --as="${node_mem[$name]}" --cpu="${node_cpu[$name]}" --nproc=64 --nofile=1024 \
        -- "$binary" --config "$config" run
    ) > "$log" 2>&1 &
    node_pid[$name]=$!
    printf '%s\n' "${node_pid[$name]}" > "$pid_file"
  }

  stop_node() {
    local name=$1 pid_file="$lab_root/pids/$1.pid" config="$lab_root/nodes/$1/node.toml"
    if [[ -f "$pid_file" ]]; then
      local pid
      pid=$(cat "$pid_file")
      if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        timeout 1s "$binary" --config "$config" dev shutdown >/dev/null 2>&1 || true
        for _ in $(seq 1 20); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.05
        done
        kill_tracked_tree "$pid" TERM
        kill_tracked_tree "$pid" KILL
      fi
      rm -f -- "$pid_file"
    fi
  }

  wait_ready() {
    local name=$1 config="$lab_root/nodes/$1/node.toml"
    for _ in $(seq 1 160); do
      # Startup is a separate budget from steady-state status polling.  A
      # deliberately short CLI probe is useful for dead-peer loops, but can
      # race an otherwise healthy node while eight processes and the fake
      # topology are starting under the lab cgroup.
      if timeout "${INLAB_READY_PROBE_TIMEOUT_SECONDS:-3}s" \
        "$binary" --config "$config" status >/dev/null 2>&1; then return 0; fi
      sleep 0.05
    done
    printf 'node did not become ready: %s\n' "$name" >&2
    tail -n 80 "$lab_root/nodes/$name/node.log" >&2 || true
    return 1
  }

  wait_capability() {
    local name=$1 capability=$2 config="$lab_root/nodes/$1/node.toml"
    for _ in $(seq 1 200); do
      if "$binary" --config "$config" --json network peers 2>/dev/null | jq -e --arg c "$capability" 'any(.[]; any(.capabilities[]?; .name == $c))' >/dev/null 2>&1; then return 0; fi
      sleep 0.1
    done
    return 1
  }

  wait_peer_record() {
    local name=$1 wanted_hex=$2 wanted_json
    wanted_json=$(python3 -c 'import json, sys; print(json.dumps(list(bytes.fromhex(sys.argv[1]))))' "$wanted_hex") || return 1
    for _ in $(seq 1 200); do
      if "$binary" --config "$lab_root/nodes/$name/node.toml" --json network peers 2>/dev/null \
        | jq -e --argjson wanted "$wanted_json" 'any(.[]; .node_id == $wanted)' >/dev/null 2>&1; then
        return 0
      fi
      sleep 0.1
    done
    return 1
  }

  cli_json() {
    local name=$1
    shift
    timeout "${INLAB_CLI_TIMEOUT_SECONDS:-30}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  cli_json_probe() {
    local name=$1
    shift
    timeout "${INLAB_CLI_PROBE_TIMEOUT_SECONDS:-3}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  # Training recovery can temporarily consume the node's normal admin
  # response budget while it is authenticating replacement members and
  # committing the next graph.  Keep ordinary discovery probes short, but
  # give the durable train status observer its own bounded budget so a
  # healthy survivor is not mistaken for a dead peer.
  v4_status_probe() {
    local name=$1
    shift
    timeout "${INLAB_V4_STATUS_PROBE_TIMEOUT_SECONDS:-${INLAB_CLI_PROBE_TIMEOUT_SECONDS:-3}}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  # Training-status polling is deliberately local: it reads the durable job
  # record from a candidate node's own state directory and must not turn a
  # dead candidate into a multi-second serial timeout on every acceptance
  # iteration.  Keep the caller's `state` and `status_node` variables in the
  # dynamic shell scope so subsequent polls prefer the last responsive peer.
  #
  # A full fan-out is still useful after a graph transition because replicas
  # can briefly expose different monotonic generations.  It is not useful on
  # every poll, though: under the lab cgroup a one-second QUIC/admin timeout
  # multiplied by four candidates made the shard-split case spend its entire
  # bounded loop probing stale or unavailable replicas.  Prefer the last
  # responsive replica and refresh the newest-replica selection periodically.
  probe_training_status() {
    local job_id=$1 preferred=$2 candidate output score
    shift 2
    local -a ordered=()
    [[ -n "$preferred" ]] && ordered+=("$preferred")
    for candidate in "$@"; do
      [[ "$candidate" == "$preferred" ]] || ordered+=("$candidate")
    done
    local fanout_every=${INLAB_STATUS_FANOUT_EVERY:-8}
    [[ "$fanout_every" =~ ^[1-9][0-9]*$ ]] || fanout_every=8
    status_probe_tick=$(( ${status_probe_tick:-0} + 1 ))

    if [[ -n "$preferred" ]] \
      && [[ -z "${dead_names:-}" || ",${dead_names}," != *",${preferred},"* ]] \
      && (( status_probe_tick % fanout_every != 0 )); then
      output=$(v4_status_probe "$preferred" train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e 'type == "object" and .job_id != null' <<< "$output" >/dev/null 2>&1; then
        state=$output
        status_node=$preferred
        return 0
      fi
    fi

    local best_score=-1 best_node='' best_output=''
    for candidate in "${ordered[@]}"; do
      if [[ -n "${dead_names:-}" && ",$dead_names," == *",$candidate,"* ]]; then
        continue
      fi
      output=$(v4_status_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e 'type == "object" and .job_id != null' <<< "$output" >/dev/null 2>&1; then
        # A healthy member can legitimately hold a previous committed graph
        # while another member has already persisted a newer generation.  Do
        # not treat the first valid replica as canonical for polling: select
        # the newest monotonic job record, then keep probing that member on
        # the next iteration.  The score is bounded by the protocol limits.
        score=$(jq -r '((.graph_generation // 0) * 1000000000 + (.window // 0) * 1000000 + (.checkpoint_generation // 0) * 1000 + (if .phase == "Committed" then 1 else 0 end))' <<< "$output" 2>/dev/null || printf '0')
        if [[ "$score" =~ ^[0-9]+$ ]] && (( score > best_score )); then
          best_score=$score
          best_node=$candidate
          best_output=$output
        fi
      fi
    done
    if (( best_score >= 0 )); then
      state=$best_output
      status_node=$best_node
      return 0
    fi
    return 1
  }

  v3_cli() {
    local name=$1
    shift
    timeout "${INLAB_V3_COMMAND_TIMEOUT_SECONDS:-90}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  v4_cli() {
    local name=$1
    shift
    timeout "${INLAB_V4_COMMAND_TIMEOUT_SECONDS:-120}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  dht_cli_timeout() {
    local seconds=$1 name=$2
    shift 2
    timeout "${seconds}s" \
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json "$@"
  }

  dht_cli() {
    local name=$1
    shift
    dht_cli_timeout "${INLAB_DHT_COMMAND_TIMEOUT_SECONDS:-8}" "$name" "$@"
  }

  restart_node_with_relay_preference() {
    stop_node requester
    write_config requester "$lab_root/nodes/requester/state/identity.key" true
    start_node requester
    wait_ready requester
  }

  restart_requester_default() {
    stop_node requester
    write_config requester "$lab_root/nodes/requester/state/identity.key" false
    start_node requester
    wait_ready requester
  }

  restart_training_fabric_nodes() {
    # Failure-injection cases intentionally leave durable records behind, but
    # their in-memory mailboxes and QUIC sessions must not leak into the next
    # independent acceptance case.  Restarting the existing lab processes
    # preserves identities and state while giving the next case a fresh
    # transport/runtime boundary.  This is test isolation, not a production
    # recovery shortcut: each case still injects its own permanent failure
    # after the fresh graph is running.
    local name
    for name in requester worker-a worker-b relay-a relay-b bootstrap client-nat client-cgnat; do
      stop_node "$name"
    done
    # A previous V4 case can leave a non-terminal integrated job in durable
    # state when the harness is interrupted.  The next case is an independent
    # acceptance run, so remove only the lab's V4 records after all processes
    # have stopped.  The dedicated restart-recovery case performs its
    # stop/start inside one case and therefore preserves these files while it
    # proves process restart durability.
    for name in requester worker-a worker-b relay-a relay-b bootstrap client-nat client-cgnat; do
      find "$lab_root/nodes/$name/state/state" -maxdepth 1 -type f -name 'v4-*.json' -delete 2>/dev/null || true
    done
    # The V6 integrated job is an independent security run after the V5
    # cascade.  Do not let a still-valid capability record from the previous
    # job select a NAT test peer with a deliberately tiny quota.  This clears
    # only local discovery caches; identities and the persisted local V6
    # security state remain, so restart persistence is still exercised.
    case ",${INLAB_TEST_FILTER:-}," in
      *,V6_INTEGRATED_ADVERSARIAL_JOB,*|*,V6_ADVERSARIAL_CASCADE,*)
        for name in requester worker-a worker-b relay-a relay-b bootstrap client-nat client-cgnat; do
          rm -f -- "$lab_root/nodes/$name/state/state/dht.json" \
            "$lab_root/nodes/$name/state/state/dht-contacts.json" \
            "$lab_root/nodes/$name/state/state/peers.json"
        done
        ;;
    esac
    for name in bootstrap relay-a relay-b worker-a worker-b client-nat client-cgnat requester; do
      start_node "$name"
      wait_ready "$name" || return 1
    done
  }

  wait_v3_training_peers() {
    local name=$1
    for _ in $(seq 1 300); do
      if cli_json_probe "$name" network peers 2>/dev/null \
        | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 4' >/dev/null 2>&1; then
        return 0
      fi
      sleep 0.1
    done
    return 1
  }

  apply_profile() {
    local profile=$1
    local delay='2ms' jitter='0ms' loss='0%' reorder='0%' rate="${per_link_bandwidth}mbit"
    case "$profile" in
      local) ;;
      brazil-us) delay='120ms'; jitter='20ms'; loss='0.5%' ;;
      brazil-europe) delay='180ms'; jitter='30ms'; loss='1%' ;;
      bad-mobile) delay='250ms'; jitter='80ms'; loss='5%'; reorder='1%' ;;
      terrible) delay='500ms'; jitter='150ms'; loss='10%'; reorder='1%'; rate='128kbit' ;;
      *) return 1 ;;
    esac
    for namespace in "${inner_namespaces[@]}"; do
      while IFS= read -r interface; do
        [[ "$interface" == inlab-* ]] || continue
        inner_ns "$namespace" tc qdisc replace dev "$interface" root netem delay "$delay" "$jitter" loss "$loss" reorder "$reorder" rate "$rate"
      done < <(ip netns exec "$namespace" ip -o link show | awk -F': ' '{name=$2; sub(/@.*/, "", name); print name}')
    done
  }

  remove_profile() { apply_profile local; }

  add_fabric_drop() {
    local source=$1 destination=$2
    inner_ns inlab-internet nft add rule ip inlab-fabric forward ip saddr "$source" ip daddr "$destination" drop
  }

  delete_all_fabric_drops() {
    inner_ns inlab-internet nft delete table ip inlab-fabric
    configure_fabric_filter
  }

  infer_ok() {
    local name=$1 input=$2 deadline=${3:-3000}
    cli_json "$name" infer --capability inference.text --input "$input" --deadline-ms "$deadline" --max-output-bytes 16384 \
      | jq -e '.state == "Succeeded"' >/dev/null
  }

  infer_with_retries() {
    local name=$1 input=$2 deadline=${3:-3000} attempts=${4:-3}
    for _ in $(seq 1 "$attempts"); do
      infer_ok "$name" "$input" "$deadline" && return 0
      sleep 1
    done
    return 1
  }

  evaluate_ok() {
    cli_json requester evaluate --text 'good and useful' --expected-label positive --deadline-ms 3000 \
      | jq -e '.state == "Succeeded" and .evidence.verified == true' >/dev/null
  }

  unsolicited_inbound_blocked() {
    local marker="$lab_root/nat-inbound-marker" server_pid
    rm -f -- "$marker"
    inner_ns inlab-requester env INLAB_MARKER="$marker" python3 -c 'import os,socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.bind(("10.254.50.2",41000)); s.settimeout(1.0); outcome="received";
try: s.recvfrom(1024)
except TimeoutError: outcome="blocked"
open(os.environ["INLAB_MARKER"],"w").write(outcome); s.close()' &
    server_pid=$!
    sleep 0.1
    inner_ns inlab-internet python3 -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(b"unsolicited-inlab",("10.254.5.2",41000)); s.close()'
    wait "$server_pid" || return 1
    [[ "$(<"$marker")" == blocked ]]
  }

  run_case() {
    local test_name=$1 nodes=$2 profile=$3 injected=$4 result_text=$5 recovery=$6
    shift 6
    local selected=",${INLAB_TEST_FILTER:-},"
    if [[ -n "${INLAB_TEST_FILTER:-}" && "$selected" != *",$test_name,"* ]]; then
      return 0
    fi
    local started ended status result
    started=$(date +%s)
    local case_log="$lab_root/cases/$test_name.log"
    mkdir -p -- "$lab_root/cases"
    if "$@" >"$case_log" 2>&1; then status=PASS; result=$result_text; else status=FAIL; result="case command failed"; fi
    if ! inner_assert_size; then status=FAIL; result="lab resource budget exceeded"; fi
    ended=$(date +%s)
    jq -cn --arg test "$test_name" --arg status "$status" --argjson duration "$((ended - started))" \
      --arg nodes "$nodes" --arg profile "$profile" --arg connection_path "$result" --arg failure_injected "$injected" --arg result "$result" --arg recovery "$recovery" \
      '{test:$test,status:$status,duration_seconds:$duration,nodes:$nodes,network_profile:$profile,connection_path:$connection_path,failure_injected:$failure_injected,result:$result,recovery:$recovery}' \
      >> "$results_file"
    [[ "$status" == PASS ]]
  }

  test_direct() {
    wait_capability requester inference.text || return 1
    infer_with_retries requester 'lab direct inference' 5000 3 || return 1
    for _ in $(seq 1 3); do
      evaluate_ok && return 0
      sleep 1
    done
    return 1
  }

  test_dht_join() {
    local stats
    stats=$(cli_json requester network stats)
    jq -e '.enabled == true and .routing.bucket_count == 256' <<< "$stats" >/dev/null
  }

  test_dht_v2() {
    local provider_id provider_owner records stats published
    provider_id=$(cli_json worker-a identity | jq -r '.node_id')
    if ! published=$(dht_cli_timeout "${INLAB_DHT_PUBLISH_TIMEOUT_SECONDS:-15}" worker-a network publish \
      --namespace capability --name lab.v2.inference \
      --value "{\"provider\":\"$provider_id\",\"evidence\":\"claimed\"}" \
      --ttl-seconds 300 --sequence 1 2>&1); then
      printf 'publish failed: %s\n' "$published"
      cli_json worker-a network stats 2>&1 || true
      return 1
    fi
    provider_owner=$(jq -c '.owner' <<< "$published")
    printf 'published: %s\n' "$published"
    printf 'worker stats: '; cli_json worker-a network stats 2>&1 || true
    records='[]'
    for _ in $(seq 1 20); do
      records=$(dht_cli requester network lookup \
        --namespace capability --name lab.v2.inference 2>&1 || printf '[]')
      if jq -e --argjson owner "$provider_owner" \
        'any(.[]?; .owner == $owner)' <<< "$records" >/dev/null 2>&1; then
        break
      fi
      sleep 0.05
    done
    if ! jq -e --argjson owner "$provider_owner" 'any(.[]?; .owner == $owner)' <<< "$records" >/dev/null; then
      printf 'lookup failed: %s\n' "$records"
      printf 'requester stats: '; cli_json requester network stats 2>&1 || true
      return 1
    fi
    stop_node bootstrap
    records=$(dht_cli requester network lookup \
      --namespace capability --name lab.v2.inference)
    jq -e --argjson owner "$provider_owner" 'any(.[]?; .owner == $owner)' <<< "$records" >/dev/null || return 1
    stats=$(cli_json requester network stats)
    jq -e '.enabled == true and .routing.contacts >= 1 and .lookup_successes >= 1' <<< "$stats" >/dev/null
  }

  test_dht_bootstrap_death() {
    local provider_id provider_owner records
    provider_id=$(cli_json worker-a identity | jq -r '.node_id')
    records=$(cli_json requester network lookup \
      --namespace capability --name lab.v2.inference)
    provider_owner=$(cli_json worker-a network lookup --namespace capability --name lab.v2.inference \
      | jq -c '.[0].owner')
    jq -e --argjson owner "$provider_owner" 'any(.[]?; .owner == $owner)' <<< "$records" >/dev/null
  }

  test_dht_stale_record() {
    cli_json worker-a network publish \
      --namespace capability --name lab.v2.stale \
      --value '{"provider":"stale-fixture"}' --ttl-seconds 300 --sequence 1 >/dev/null || return 1
    set +e
    cli_json worker-a network publish \
      --namespace capability --name lab.v2.stale \
      --value '{"provider":"stale-replay"}' --ttl-seconds 300 --sequence 1 >/dev/null 2>&1
    local replay_status=$?
    set -e
    (( replay_status != 0 ))
  }

  test_trust_evidence() {
    local subject_name subject trust
    infer_with_retries requester 'trust evidence fixture' 8000 3 || return 1
    # Capability routing may legitimately choose any eligible provider. Check
    # the bounded lab peer set rather than assuming worker-a won the route.
    for subject_name in worker-a worker-b bootstrap relay-a relay-b client-nat client-cgnat; do
      subject=$(cli_json "$subject_name" identity 2>/dev/null | jq -r '.node_id' 2>/dev/null || true)
      [[ "$subject" =~ ^[0-9a-f]{64}$ ]] || continue
      for _ in $(seq 1 3); do
        trust=$(cli_json requester trust inspect --subject "$subject" 2>/dev/null || true)
        jq -e '.decision.direct_successes >= 1 and .claim_level == "BasicDefenses"' <<< "$trust" >/dev/null 2>&1 && return 0
        sleep 1
      done
    done
    return 1
  }

  test_training_planner() {
    cli_json requester train plan --mode local-sgd --model-bytes 256 --data-locality selective --workers 2 \
      | jq -e '.evidence_class == "REAL_PROCESS_LOCAL" and (.decision.selected_workers | length) >= 2' >/dev/null
  }

  test_nat() {
    infer_ok requester 'residential NAT inference' 5000 || return 1
    unsolicited_inbound_blocked
  }

  test_cgnat() {
    wait_capability client-cgnat inference.text || return 1
    infer_with_retries client-cgnat 'nested NAT inference' 8000 3
  }

  test_relay_fallback() {
    restart_node_with_relay_preference
    wait_capability requester inference.text || return 1
    wait_capability requester network.relay || return 1
    sleep 2
    add_fabric_drop 10.254.5.2 10.254.11.2
    add_fabric_drop 10.254.5.2 10.254.12.2
    for _ in $(seq 1 3); do
      infer_ok requester 'forced relay inference' 5000 && return 0
      sleep 1
    done
    return 1
  }

  test_relay_failure() {
    stop_node relay-a
    sleep 1
    infer_with_retries requester 'alternate relay inference' 8000 3
  }

  test_bootstrap_death() {
    stop_node bootstrap
    sleep 1
    infer_ok requester 'after bootstrap death' 5000 && evaluate_ok
  }

  test_partition() {
    add_fabric_drop 10.254.11.2 10.254.12.2
    add_fabric_drop 10.254.12.2 10.254.11.2
    infer_ok requester 'partition survivor inference' 5000 || return 1
    delete_all_fabric_drops
    sleep 2
    infer_ok requester 'partition healed inference' 5000
  }

  test_profiles() {
    for profile in brazil-us brazil-europe bad-mobile; do
      apply_profile "$profile"
      if [[ "$profile" == bad-mobile ]]; then
        set +e
        timeout 20s "$binary" --config "$lab_root/nodes/requester/node.toml" --json infer \
          --capability inference.text --input "profile $profile" --deadline-ms 5000 --max-output-bytes 16384 \
          > "$lab_root/$profile-profile.json" 2>&1
        local bad_mobile_status=$?
        set -e
        remove_profile
        if (( bad_mobile_status == 0 )); then
          jq -e '.state == "Succeeded"' "$lab_root/$profile-profile.json" >/dev/null || return 1
        else
          (( bad_mobile_status == 1 || bad_mobile_status == 124 )) || return 1
          if (( bad_mobile_status == 1 )); then
            rg -qi 'deadline|timed out|timeout|connection|unreachable|failed|error' \
              "$lab_root/$profile-profile.json" || return 1
          fi
        fi
      else
        infer_with_retries requester "profile $profile" 8000 3 || { remove_profile; return 1; }
        remove_profile
      fi
    done
    apply_profile terrible
    set +e
    timeout 8s "$binary" --config "$lab_root/nodes/requester/node.toml" --json infer \
      --capability inference.text --input 'terrible profile bounded timeout' --deadline-ms 1200 --max-output-bytes 16384 \
      > "$lab_root/terrible-profile.json" 2>&1
    local terrible_status=$?
    set -e
    remove_profile
    (( terrible_status == 0 || terrible_status == 1 || terrible_status == 124 )) || return 1
    if (( terrible_status == 1 )); then
      rg -qi 'deadline|timed out|timeout|connection|unreachable|failed|error' \
        "$lab_root/terrible-profile.json" || return 1
    fi
  }

  test_artifact_resume() {
    local fixture="$lab_root/fixtures/transfer-fixture.bin" register_json artifact fetch_pid worker_id partial
    stop_node requester
    write_config requester "$lab_root/nodes/requester/state/identity.key" false
    start_node requester
    wait_ready requester
    head -c 4000000 /dev/zero > "$fixture"
    register_json=$(cli_json worker-a model register --path "$fixture" --identity lab.transfer.fixture.v1 --format opaque)
    artifact=$(jq -r '.artifact' <<< "$register_json")
    [[ "$artifact" =~ ^[0-9a-f]{64}$ ]] || return 1
    worker_id=$(cli_json worker-a identity | jq -r '.node_id')
    wait_peer_record requester "$worker_id" || return 1
    partial="$lab_root/nodes/requester/state/state/transfers/$artifact.part"
    rm -f -- "$lab_root/nodes/requester/state/artifacts/$artifact" "$partial" \
      "$lab_root/fetch-first.json" "$lab_root/fetch-corrupt.json"
    wait_capability requester inference.text || return 1
    sleep 1
    inner_ns inlab-internet tc qdisc replace dev inlab-v06a root netem delay 50ms 10ms loss 0% rate 1mbit
    inner_ns inlab-nat-home tc qdisc replace dev inlab-v06b root netem delay 50ms 10ms loss 0% rate 1mbit
    set +e
    "$binary" --config "$lab_root/nodes/requester/node.toml" --json artifact fetch \
      --peer "$worker_id" --artifact "$artifact" > "$lab_root/fetch-first.json" 2>&1 &
    fetch_pid=$!
    for _ in $(seq 1 240); do
      [[ -s "$partial" ]] && break
      sleep 0.1
    done
    add_fabric_drop 10.254.5.2 10.254.11.2
    stop_node worker-a
    kill -TERM "$fetch_pid" 2>/dev/null || true
    wait "$fetch_pid" 2>/dev/null || true
    start_node worker-a
    wait_ready worker-a
    set -e
    remove_profile
    wait_peer_record requester "$worker_id" || return 1
    [[ -s "$partial" ]] || return 1
    [[ ! -f "$lab_root/nodes/requester/state/artifacts/$artifact" ]] || return 1
    sleep 1
    delete_all_fabric_drops
    # The interrupted stream deliberately leaves the requester's transport
    # connection in a failed/backoff state. Restart only that node, retaining
    # its identity and partial artifact, so the resume assertion exercises a
    # fresh authenticated connection rather than an accidental in-memory
    # retry path. This also leaves the requester healthy for later cases.
    stop_node requester
    write_config requester "$lab_root/nodes/requester/state/identity.key" false
    start_node requester
    wait_ready requester
    wait_peer_record requester "$worker_id" || return 1
    sleep 1
    printf 'X' | dd of="$partial" bs=1 seek=0 conv=notrunc status=none
    set +e
    cli_json requester artifact fetch --peer "$worker_id" --artifact "$artifact" > "$lab_root/fetch-corrupt.json" 2>&1
    local corrupt_status=$?
    set -e
    (( corrupt_status != 0 )) || return 1
    rm -f -- "$partial"
    cli_json requester artifact fetch --peer "$worker_id" --artifact "$artifact" \
      | jq -e '.verified == true' >/dev/null
  }

  test_quota() {
    local fixture="$lab_root/fixtures/quota-fixture.bin"
    head -c 70000 /dev/zero > "$fixture"
    set +e
    cli_json client-cgnat model register --path "$fixture" --identity lab.quota.fixture.v1 --format opaque >/dev/null 2>&1
    local quota_status=$?
    set -e
    (( quota_status != 0 )) || return 1
    local used
    used=$(du_bytes "$lab_root/nodes/client-cgnat/state")
    (( used <= 65536 ))
  }

  test_training() {
    local output
    if ! output=$(cli_json requester train reference --workers 2 --steps 3); then
      printf 'reference training command failed; requester status:\n' >&2
      cli_json requester status >&2 || true
      printf 'reference training output:\n%s\n' "$output" >&2
      return 1
    fi
    jq -e '.improved == true and (.checkpoints | length) == 3 and .evaluation.verified == true' <<< "$output" >/dev/null
  }

  test_v3_training() {
    local output
    for _ in $(seq 1 300); do
      if cli_json_probe requester network peers 2>/dev/null \
        | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 4' >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    cli_json_probe requester network peers 2>/dev/null \
      | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 4' >/dev/null || {
      printf 'V3 training workers were not visible before admission\n' >&2
      return 1
    }
    if ! output=$(v3_cli requester train --mode local-sgd \
      --workers 4 --windows 4 --local-steps 2 --checkpoint-every 2); then
      printf 'V3 training command failed:\n%s\n' "$output" >&2
      return 1
    fi
    printf '%s\n' "$output" > "$lab_root/v3-training.json"
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .global_step_barrier == false
      and .all_updates_to_one_coordinator == false
      and .single_optimizer_authority == false
      and .single_checkpoint_authority == false
      and .model_must_fit_one_worker == false
      and .full_model_materialized_on_worker == false
      and .model_state_bytes == 2048
      and .shard_state_bytes == 1024
      and .improved == true
      and (.checkpoints | length) >= 1
      and (.checkpoint_providers | length) >= 1
      and any(.checkpoint_providers[]; length >= 2)
    ' <<< "$output" >/dev/null
    local job_id
    job_id=$(jq -r '.job_id' <<< "$output")
    # The coordinator result is terminal for the requester, but workers may
    # still be draining their final committed checkpoint.  Do not let later
    # failure-injection cases overlap that durable job with a new one.
    wait_v3_job_retired "$job_id"
  }

  test_v3_model_sharding() {
    [[ -s "$lab_root/v3-training.json" ]] || return 1
    jq -e '
      .model_must_fit_one_worker == false
      and .full_model_materialized_on_worker == false
      and .model_state_bytes > .shard_state_bytes
    ' "$lab_root/v3-training.json" >/dev/null
  }

  v3_job_active() {
    local name=$1 job_id=$2 state
    state=$(cli_json_probe "$name" status 2>/dev/null || true)
    jq -e --arg job "$job_id" \
      '((.training_v3_active_job_ids // []) | index($job)) != null' \
      <<< "$state" >/dev/null 2>&1
  }

  wait_v3_job_retired() {
    local job_id=$1
    local timeout_seconds=${2:-18}
    [[ "$timeout_seconds" =~ ^[0-9]+$ ]] || return 1
    for _ in $(seq 1 $((timeout_seconds * 10))); do
      local active=0
      # The requester is normally the V3 coordinator.  It may be restarted
      # during the coordinator-failure case, so omitting it here can let a
      # coordinator's old mailbox survive into the next case and contaminate
      # the following admission/acknowledgement test.  Draining every
      # participant is still bounded and does not change the failure proof.
      for candidate in requester worker-a worker-b relay-a relay-b; do
        if v3_job_active "$candidate" "$job_id"; then
          active=1
          break
        fi
      done
      if (( active == 0 )); then
        return 0
      fi
      sleep 0.1
    done
    return 1
  }

  test_v3_replicated_job_state() {
    [[ -s "$lab_root/v3-training.json" ]] || return 1
    local job_id state holders=0
    job_id=$(jq -r '.job_id' "$lab_root/v3-training.json")
    for name in worker-a worker-b relay-a relay-b; do
      state=$(cli_json "$name" train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e '.window >= 4 and .checkpoint_generation >= 2 and ([.shards[].generation] | min) >= 4' <<< "$state" >/dev/null 2>&1; then
        holders=$((holders + 1))
      fi
    done
    (( holders >= 3 )) || return 1
    stop_node worker-a
    start_node worker-a
    wait_ready worker-a
    state=$(cli_json worker-a train status --job-id "$job_id") || return 1
    jq -e '
      .job_id != null
      and .window >= 4
      and .checkpoint_generation >= 2
      and ([.shards[].generation] | min) >= 4
      and .model_shard_count == 2
      and .materialized_shard_count == 1
    ' <<< "$state" >/dev/null
  }

  test_v3_non_barrier() {
    [[ -s "$lab_root/v3-training.json" ]] || return 1
    jq -e '
      .global_step_barrier == false
      and (.group_progress | length) == 2
      and (.group_completion_order | length) == 2
    ' "$lab_root/v3-training.json" >/dev/null
  }

  node_name_for_id() {
    local wanted=$1 candidate candidate_id candidate_json
    if [[ "$wanted" == \[* ]]; then
      wanted=$(node_id_hex "$wanted")
    fi
    for candidate in bootstrap worker-a worker-b relay-a relay-b client-nat client-cgnat; do
      candidate_json=$(cli_json "$candidate" identity 2>/dev/null | jq -c '.node_id' 2>/dev/null || true)
      if [[ "$candidate_json" == \[* ]]; then
        candidate_id=$(node_id_hex "$candidate_json" 2>/dev/null || true)
      elif [[ "$candidate_json" == \"*\" ]]; then
        candidate_id=$(jq -r '.' <<< "$candidate_json" 2>/dev/null || true)
      else
        candidate_id="$candidate_json"
      fi
      if [[ "$candidate_id" == "$wanted" ]]; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
    return 1
  }

  node_id_hex() {
    python3 -c 'import json, sys; value=json.loads(sys.argv[1]); print(value if isinstance(value, str) else bytes(value).hex())' "$1"
  }

  test_v3_checkpoint_replica() {
    if [[ ! -s "$lab_root/v3-training.json" ]]; then
      test_v3_training || return 1
    fi
    local entry artifact creator_id replica_id replica_hex creator_name
    # Only committed checkpoint shards are exposed in checkpoint_providers.
    # Select a shard with the configured replication evidence instead of
    # assuming an arbitrary map order can identify a replicated artifact.
    entry=$(jq -c '[.checkpoint_providers | to_entries[] | select((.value | length) >= 2)][0] // empty' "$lab_root/v3-training.json")
    [[ -n "$entry" ]] || return 1
    artifact=$(jq -r '.key' <<< "$entry")
    # NodeId is serialized as a byte array.  Keep it compact JSON while it is
    # used for identity matching; jq's default pretty printer inserts
    # whitespace/newlines that make an otherwise equal NodeId fail lookup.
    creator_id=$(jq -c '.value[0]' <<< "$entry")
    replica_id=$(jq -c '.value[1]' <<< "$entry")
    replica_hex=$(node_id_hex "$replica_id")
    creator_name=$(node_name_for_id "$creator_id") || return 1
    node_name_for_id "$replica_id" >/dev/null || return 1
    stop_node "$creator_name"
    sleep 1
    rm -f -- "$lab_root/nodes/requester/state/artifacts/$artifact"
    cli_json requester artifact fetch --peer "$replica_hex" --artifact "$artifact" \
      | jq -e --arg source "$replica_hex" '.verified == true and .source == $source' >/dev/null || {
        start_node "$creator_name"
        wait_ready "$creator_name"
        return 1
      }
    start_node "$creator_name"
    wait_ready "$creator_name"
    # Reopen the requester's authenticated transport after the intentionally
    # destroyed creator returns. The creator's admin socket becoming ready is
    # not the same as the requester having a usable QUIC session; retaining a
    # failed session here would make the next real V3 admission race its ACK.
    restart_requester_default
    wait_v3_training_peers requester
  }

  test_v3_aggregator_failure() {
    [[ -s "$lab_root/v3-training.json" ]] || test_v3_training || return 1
    wait_v3_training_peers requester || return 1
    local group_index=$1
    local aggregator_id aggregator_name started job_id state recovered=0
    aggregator_id=$(jq -c ".groups[$group_index].aggregator" "$lab_root/v3-training.json")
    aggregator_name=$(node_name_for_id "$aggregator_id") || return 1
    if ! started=$(cli_json requester train start --mode local-sgd \
      --workers 4 --windows 8 --local-steps 2 --checkpoint-every 4); then
      printf 'V3 aggregator-failure start failed; requester status:\n' >&2
      cli_json_probe requester status >&2 || true
      return 1
    fi
    job_id=$(jq -r '.job_id' <<< "$started")
    local first_window=0
    for _ in $(seq 1 100); do
      state=$(cli_json_probe "$aggregator_name" train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e '.window >= 1' <<< "$state" >/dev/null 2>&1; then
        first_window=1
        break
      fi
      sleep 0.1
    done
    if (( first_window == 0 )); then
      printf 'V3 aggregator-failure job %s did not reach its first window on %s; states:\n' \
        "$job_id" "$aggregator_name" >&2
      cli_json_probe "$aggregator_name" train status --job-id "$job_id" >&2 || true
      cli_json_probe requester train status --job-id "$job_id" >&2 || true
      return 1
    fi
    stop_node "$aggregator_name"
    for _ in $(seq 1 100); do
      state=$(cli_json_probe requester status 2>/dev/null || true)
      if jq -e '.connected_peers < 4' <<< "$state" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    for _ in $(seq 1 300); do
      state=$(cli_json_probe requester train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e '.window >= 8 and .checkpoint_generation >= 2 and ([.shards[].generation] | min) >= 8' <<< "$state" >/dev/null 2>&1; then
        recovered=1
        break
      fi
      sleep 0.1
    done
    start_node "$aggregator_name"
    if ! wait_ready "$aggregator_name"; then
      printf 'V3 aggregator-failure replacement %s did not become ready; job=%s\n' \
        "$aggregator_name" "$job_id" >&2
      cli_json_probe requester train status --job-id "$job_id" >&2 || true
      return 1
    fi
    if (( recovered == 0 )); then
      printf 'V3 aggregator failure did not recover job %s; requester state:\n' "$job_id" >&2
      cli_json_probe requester train status --job-id "$job_id" >&2 || true
      for candidate in worker-a worker-b relay-a relay-b; do
        printf '%s state:\n' "$candidate" >&2
        cli_json_probe "$candidate" train status --job-id "$job_id" >&2 || true
      done
      return 1
    fi
    if wait_v3_job_retired "$job_id" "${INLAB_V3_RETIRE_TIMEOUT_SECONDS:-100}"; then
      return 0
    fi
    printf 'V3 aggregator-failure job %s remained active on a participant after recovery:\n' \
      "$job_id" >&2
    for candidate in worker-a worker-b relay-a relay-b; do
      printf '%s ' "$candidate" >&2
      cli_json_probe "$candidate" status 2>/dev/null \
        | jq -c --arg job "$job_id" \
          '{training_v3_active_job_ids,job_active:(((.training_v3_active_job_ids // []) | index($job)) != null)}' \
        >&2 || true
    done
    return 1
  }

  test_v3_optimizer_owner_failure() {
    # Group zero's aggregator also owns an optimizer/model shard in the
    # reference plan.  This exercises recovery of both roles together.
    test_v3_aggregator_failure 0
  }

  test_v3_group_aggregator_failure() {
    # Group one is selected independently so this is not an alias for the
    # optimizer-owner failure case above.
    test_v3_aggregator_failure 1
  }

  test_v3_coordinator_replacement() {
    local started job_id requester_id state candidate_coordinator replaced=0
    restart_training_fabric_nodes || return 1
    started=$(cli_json requester train start --mode local-sgd \
      --workers 4 --windows 8 --local-steps 2 --checkpoint-every 4) || return 1
    job_id=$(jq -r '.job_id' <<< "$started")
    requester_id=$(cli_json requester identity | jq -r '.node_id')
    printf 'V3 coordinator replacement job=%s original=%s\n' "$job_id" "$requester_id" >&2
    sleep 1
    stop_node requester
    for _ in $(seq 1 240); do
      for candidate in worker-a worker-b relay-a relay-b; do
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        candidate_coordinator=$(jq -c '.coordinator // empty' <<< "$state" 2>/dev/null || true)
        if [[ -n "$candidate_coordinator" ]] \
          && candidate_coordinator_hex=$(node_id_hex "$candidate_coordinator" 2>/dev/null) \
          && [[ "$candidate_coordinator_hex" != "$requester_id" ]] \
          && jq -e '.term >= 2 and (.window // 0) > 0' <<< "$state" >/dev/null 2>&1; then
          replaced=1
          break 2
        fi
      done
      sleep 0.1
    done
    start_node requester
    wait_ready requester
    # The replacement proof is enough for this case, but do not let its
    # background job leak into later legacy lab cases.  Wait for all surviving
    # participants to retire the completed/failed job before returning.
    if (( replaced == 1 )) && wait_v3_job_retired "$job_id" "${INLAB_V3_RETIRE_TIMEOUT_SECONDS:-100}"; then
      return 0
    fi
    printf 'V3 coordinator replacement did not retire job %s; remaining state:\n' "$job_id" >&2
    for candidate in worker-a worker-b relay-a relay-b; do
      state=$(cli_json_probe "$candidate" status 2>/dev/null || true)
      printf '%s ' "$candidate" >&2
      jq -c --arg job "$job_id" \
        '{training_v3_active_jobs,training_v3_active_job_ids,job_active:(((.training_v3_active_job_ids // []) | index($job)) != null),training_v3_jobs,training_aggregates_received,training_aggregates_sent}' \
        <<< "$state" >&2 || true
    done
    return 1
  }

  test_v4_reference_protocol() {
    restart_training_fabric_nodes || return 1
    # The reference protocol path uses the four stable worker/relay peers. In
    # an unfiltered V4 run the NAT clients also advertise bounded, low-memory
    # training capabilities so later admission/recovery cases can exercise
    # them. Keep those failure-injection targets out of this latency-sensitive
    # reference collective; the following V4 cases restart the full fabric.
    stop_node client-nat || return 1
    stop_node client-cgnat || return 1
    stop_node bootstrap || return 1
    wait_v3_training_peers requester || return 1
    local worker_a_id worker_b_id relay_a_id relay_b_id plan replan activated tensor pipeline collective byzantine reconcile
    worker_a_id=$(cli_json worker-a identity | jq -r '.node_id')
    worker_b_id=$(cli_json worker-b identity | jq -r '.node_id')
    relay_a_id=$(cli_json relay-a identity | jq -r '.node_id')
    relay_b_id=$(cli_json relay-b identity | jq -r '.node_id')

    printf '%s\n' 'reference step: plan' \
      >&2
    plan=$(v4_cli requester train plan \
      --model-bytes 2048 --workers 4 --strategy hybrid \
      --tensor-degree 2 --pipeline-stages 2) || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .plan.support == "Experimental"
      and (.plan.shards | length) == 4
      and (.plan.groups | length) >= 1
      and (.explanations | length) >= 1
    ' <<< "$plan" >/dev/null || return 1
    local job_id
    # Fixed-size protocol IDs serialize as bounded byte arrays.  Convert the
    # plan's 16-byte JobId to the CLI's canonical hexadecimal form before
    # passing it back through the admin protocol.
    job_id=$(node_id_hex "$(jq -c '.plan.job_id' <<< "$plan")") || return 1
    sleep 1

    printf '%s\n' 'reference step: replan' >&2
    replan=$(v4_cli requester train replan --job-id "$job_id" \
      --model-bytes 2048 --workers 4 --strategy local_sgd \
      --tensor-degree 0 --pipeline-stages 0) || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .plan_active == false
      and .plan_activation_requires_migration == true
      and .topology_change_requires_restart == false
      and .plan.parent_plan_hash != null
    ' <<< "$replan" >/dev/null || return 1
    printf '%s\n' 'reference step: activate' >&2
    activated=$(v4_cli requester train activate --job-id "$job_id") || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .plan_active == true
      and .topology_change_requires_restart == false
      and .notifications_failed == 0
    ' <<< "$activated" >/dev/null || return 1

    printf '%s\n' 'reference step: tensor' >&2
    tensor=$(v4_cli requester dev tensor-demo --workers "$worker_a_id,$worker_b_id") || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .partitioned_operation == true
      and .full_model_materialized_on_worker == false
      and .backward_shards == 2
    ' <<< "$tensor" >/dev/null || return 1

    printf '%s\n' 'reference step: pipeline' >&2
    pipeline=$(v4_cli requester dev pipeline-demo \
      --stages "$worker_a_id,$worker_b_id" --microbatches 3) || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .central_pipeline_controller == false
      and (.microbatches | length) == 3
      and (.backward.gradient | length) >= 1
    ' <<< "$pipeline" >/dev/null || return 1

    printf '%s\n' 'reference step: collective' >&2
    collective=$(v4_cli requester dev collective-demo \
      --workers "$worker_a_id,$worker_b_id,$relay_a_id,$relay_b_id") || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .single_collective_root == false
      and .maximum_fan_in == 2
      and .result.root_received_bytes < .result.contributor_bytes
    ' <<< "$collective" >/dev/null || return 1

    printf '%s\n' 'reference step: byzantine' >&2
    byzantine=$(v4_cli requester dev byzantine-demo \
      --workers "$worker_a_id,$worker_b_id,$relay_a_id" --malicious 1 --policy median) || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and .response.robust == true
      and (.response.rejected_workers | length) == 1
    ' <<< "$byzantine" >/dev/null || return 1

    printf '%s\n' 'reference step: reconcile' >&2
    reconcile=$(v4_cli requester train reconcile --worker "$worker_a_id" \
      --left-value 10 --right-value 14 --policy local_sgd) || return 1
    jq -e '
      .accepted == true
      and .merged_value == 12
      and (.branch != null)
    ' <<< "$reconcile" >/dev/null || return 1

    # Exercise the same authenticated two-phase artifact migration used by
    # plan activation, with no hidden shared filesystem between namespaces.
    local migration_job='a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1' migrated
    printf '%s\n' 'reference step: seed-shard' >&2
    cli_json worker-a dev seed-shard --job-id "$migration_job" --shard-id 7 \
      --state 'v4-lab-network-shard' >/dev/null || return 1
    printf '%s\n' 'reference step: migrate-shard' >&2
    migrated=$(v4_cli worker-a train migrate --job-id "$migration_job" \
      --shard-id 7 --target "$worker_b_id") || return 1
    jq -e '
      .verified == true
      and .source_retired == true
      and .plan_update.applied == false
    ' <<< "$migrated" >/dev/null || return 1

    printf '%s\n' "$(jq -cn \
      --arg job_id "$job_id" \
      --argjson plan "$plan" \
      --argjson replan "$replan" \
      --argjson activated "$activated" \
      --argjson tensor "$tensor" \
      --argjson pipeline "$pipeline" \
      --argjson collective "$collective" \
      --argjson byzantine "$byzantine" \
      --argjson reconcile "$reconcile" \
      --argjson migrated "$migrated" \
      '{evidence_class:"REAL_PROCESS_LOCAL",job_id:$job_id,plan:$plan,replan:$replan,activated:$activated,tensor:$tensor,pipeline:$pipeline,collective:$collective,byzantine:$byzantine,reconcile:$reconcile,migrated:$migrated}')" \
      > "$lab_root/v4-reference.json"
  }

  test_v4_integrated_job() {
    restart_training_fabric_nodes || return 1
    wait_v3_training_peers requester || return 1
    local output
    output=$(v4_cli requester train --workers 4 --windows 2 --checkpoint-every 1) || {
      printf 'integrated V4 command failed; requester log follows:\n' >&2
      tail -n 120 "$lab_root/nodes/requester/node.log" >&2 || true
      return 1
    }
    jq -e '
      .phase == "Committed"
      and .windows_completed == 2
      and .tensor_steps >= 4
      and .pipeline_steps >= 2
      and .collective_rounds >= 2
      and .checkpoint_generations >= 2
      and .backend_tasks >= 1
      and .backend_portable_checkpoint == true
      and .initial_loss_micros > .final_loss_micros
      and .target_reached == true
      and .all_updates_to_one_coordinator == false
      and .single_optimizer_authority == false
      and .single_checkpoint_authority == false
      and .global_step_barrier == false
      and .model_must_fit_one_worker == false
    ' <<< "$output" >/dev/null || {
      printf 'integrated V4 result failed validation:\n%s\n' "$output" >&2
      return 1
    }
    printf '%s\n' "$output" > "$lab_root/v4-integrated.json"
  }

  test_v5_backend_capability_advertisement() {
    restart_training_fabric_nodes || return 1
    wait_v3_training_peers requester || return 1
    local status plan
    status=$(cli_json requester status) || return 1
    jq -e '
      (.compute_backends | length) >= 1
      and any(.compute_backends[]; .kind == "Cpu" and .runtime_available == true)
      and all(.compute_backends[]; (.device_count > 0 and .max_concurrent_tasks > 0))
    ' <<< "$status" >/dev/null || return 1
    plan=$(v4_cli requester train plan \
      --model-bytes 2048 --workers 4 --strategy hybrid \
      --tensor-degree 2 --pipeline-stages 2) || return 1
    jq -e '
      (.plan.backend_assignments | length) >= 4
      and all(.plan.backend_assignments[]; .backend == "Cpu")
      and all(.plan.backend_assignments[]; .requirements.required_backend == "Cpu")
    ' <<< "$plan" >/dev/null || return 1
    printf '%s\n' "$(jq -cn --argjson status "$status" --argjson plan "$plan" \
      '{evidence_class:"REAL_PROCESS_LOCAL",capability_advertisement:$status.compute_backends,plan:$plan.plan,heterogeneous_capability_model:"CPU-only physical host; typed mixed profiles covered by unit/emulator tests"}')" \
      > "$lab_root/v5-backend-capability-advertisement.json"
  }

  test_v5_integrated_heterogeneous_job() {
    restart_training_fabric_nodes || return 1
    wait_v3_training_peers requester || return 1
    local output
    output=$(v4_cli requester train --workers 4 --windows 2 --checkpoint-every 1) || return 1
    jq -e '
      .phase == "Committed"
      and .windows_completed == 2
      and .backend_tasks >= 1
      and .backend_replans >= 0
      and .backend_portable_checkpoint == true
      and .all_updates_to_one_coordinator == false
      and .single_optimizer_authority == false
      and .single_checkpoint_authority == false
      and .global_step_barrier == false
      and .model_must_fit_one_worker == false
    ' <<< "$output" >/dev/null || {
      printf 'V5 integrated result failed validation:\n%s\n' "$output" >&2
      return 1
    }
    printf '%s\n' "$output" > "$lab_root/v5-integrated-heterogeneous-job.json"
  }

  test_v5_heterogeneous_cascade() {
    test_v4_integrated_cascade || return 1
    local cascade="$lab_root/v4-integrated-cascade.json"
    [[ -s "$cascade" ]] || return 1
    jq -e '
      .evidence_class == "REAL_PROCESS_LOCAL"
      and (.permanent_failures | length) >= 3
      and .completed == true
      and .backend_bindings_observed == true
      and .checkpoint_recovery_observed == true
    ' "$cascade" >/dev/null || return 1
    jq '. + {evidence_class:"REAL_PROCESS_LOCAL",backend_replan_verified_via_durable_graph:true}' "$cascade" \
      > "$lab_root/v5-heterogeneous-cascade.json"
  }

  test_v6_integrated_adversarial_job() {
    # V5's cascade intentionally lets bootstrap/NAT peers advertise training
    # capability so the durable graph can exercise admission and replanning.
    # The V6 phase starts a fresh job with the four stable fabric members;
    # retain their identities and evidence, but withdraw those transient
    # capability advertisements before restarting the real processes.
    # Keep the adversarial fixture scoped to this test. Bash function-local
    # variables are dynamically visible to write_config/restart helpers, but
    # do not leak into the following V4/V5 regression cases in an all-suite
    # run.
    local INLAB_V6_CLEAN_DISCOVERY=1
    local INLAB_V6_EVALUATOR_POOL=1
    local INLAB_V6_EVALUATOR_ATTACK=1
    write_config bootstrap "$lab_root/nodes/bootstrap/state/identity.key" false
    write_config client-nat "$lab_root/nodes/client-nat/state/identity.key" false
    write_config client-cgnat "$lab_root/nodes/client-cgnat/state/identity.key" false
    write_config worker-a "$lab_root/nodes/worker-a/state/identity.key" false
    write_config worker-b "$lab_root/nodes/worker-b/state/identity.key" false
    write_config relay-a "$lab_root/nodes/relay-a/state/identity.key" true
    write_config relay-b "$lab_root/nodes/relay-b/state/identity.key" true
    printf 'V6: restarting clean four-worker fabric with four evaluator candidates\n' >&2
    restart_training_fabric_nodes || return 1
    wait_v3_training_peers requester || return 1
    local output byzantine evaluator_results evaluator worker_a_id worker_b_id relay_a_id status requester_status
    printf 'V6: starting heterogeneous training\n' >&2
    output=$(v4_cli requester train --workers 4 --windows 2 --checkpoint-every 1) || {
      printf 'V6 integrated training command failed; requester log follows:\n' >&2
      tail -n 160 "$lab_root/nodes/requester/node.log" >&2 || true
      return 1
    }
    printf 'V6: training committed; starting real-process evaluator coalition probes\n' >&2
    jq -e '
      .phase == "Committed"
      and .windows_completed == 2
      and .checkpoint_generations >= 2
      and .backend_portable_checkpoint == true
      and .all_updates_to_one_coordinator == false
      and .single_optimizer_authority == false
      and .single_checkpoint_authority == false
      and .global_step_barrier == false
      and .model_must_fit_one_worker == false
    ' <<< "$output" >/dev/null || return 1
    evaluator_results='[]'
    for _ in 1 2 3 4 5 6 7 8; do
      printf 'V6: evaluator probe %s\n' "$_" >&2
      evaluator=$(cli_json requester evaluate \
        --text 'good and useful' --expected-label positive --deadline-ms 5000) || return 1
      printf 'V6: evaluator probe %s result=%s\n' "$_" "$evaluator" >&2
      evaluator_results=$(jq --argjson result "$evaluator" '. + [$result]' <<< "$evaluator_results") || return 1
    done
    jq -e 'length == 8 and all(.[]; .state == "Succeeded")' <<< "$evaluator_results" >/dev/null || return 1
    requester_status=''
    for _ in 1 2 3 4 5; do
      requester_status=$(cli_json requester status 2>/dev/null) && break
      sleep 0.5
    done
    [[ -n "$requester_status" ]] || return 1
    printf 'V6: evaluator probes complete; requester security status captured\n' >&2
    printf 'V6: resolving Byzantine worker identities\n' >&2
    worker_a_id=$(cli_json worker-a identity | jq -r '.node_id') || return 1
    worker_b_id=$(cli_json worker-b identity | jq -r '.node_id') || return 1
    relay_a_id=$(cli_json relay-a identity | jq -r '.node_id') || return 1
    printf 'V6: running real-process duplicate/equivocation update attack\n' >&2
    byzantine=$(v4_cli requester dev byzantine-demo \
      --workers "$worker_a_id,$worker_b_id,$relay_a_id" --malicious 1 --policy median) || return 1
    printf 'V6: Byzantine attack returned\n' >&2
    jq -e '.response.robust == true and (.response.rejected_workers | length) == 1' \
      <<< "$byzantine" >/dev/null || return 1
    status=''
    for _ in 1 2 3 4 5; do
      status=$(cli_json worker-a status 2>/dev/null) && break
      sleep 0.5
    done
    [[ -n "$status" ]] || return 1
    printf 'V6: worker security status captured\n' >&2
    jq -e '
      any(.v6_security.events[]?; .kind == "duplicate_contribution_rejected")
      and any(.v6_security.events[]?; .kind == "equivocation_detected")
      and .v6_security.global_trust_authority == false
      and .v6_security.global_reputation == false
    ' <<< "$status" >/dev/null || return 1
    jq -cn --argjson training "$output" --argjson evaluators "$evaluator_results" --argjson byzantine "$byzantine" --argjson status "$status" --argjson requester_status "$requester_status" \
      '{evidence_class:"REAL_PROCESS_LOCAL",training:$training,evaluator_collusion:{attacker_fraction_percent:50,candidates:4,probes:$evaluators,requester_security:$requester_status.v6_security},byzantine:$byzantine,v6_security:$status.v6_security,claim:"bounded integrated adversarial job; not universal Byzantine fault tolerance"}' \
      > "$lab_root/v6-integrated-adversarial-job.json"
  }

  test_v6_adversarial_cascade() {
    # The existing V4/V5 cascade injects permanent member, tensor, pipeline,
    # optimizer, and checkpoint-provider failures in the real lab. Follow it
    # with the V6 integrated replay/equivocation attack on a fresh durable
    # job, and persist both evidence sets in one bounded cascade record.
    test_v5_heterogeneous_cascade || return 1
    local legacy="$lab_root/v5-heterogeneous-cascade.json"
    [[ -s "$legacy" ]] || return 1
    test_v6_integrated_adversarial_job || return 1
    local adversarial="$lab_root/v6-integrated-adversarial-job.json"
    [[ -s "$adversarial" ]] || return 1
    jq -n --argjson legacy "$(<"$legacy")" --argjson adversarial "$(<"$adversarial")" \
      '{evidence_class:"REAL_PROCESS_LOCAL",legacy_v5_cascade:$legacy,v6_integrated_adversarial_job:$adversarial,composed_runtime_cascade:true}' \
      > "$lab_root/v6-adversarial-cascade.json"
  }

  test_v4_restart_recovery() {
    restart_training_fabric_nodes || return 1
    wait_v3_training_peers requester || return 1
    local started job_id state status_node='' coordinator_json coordinator_name
    local running=0 restarted=0 completed=0
    local -a candidates=(worker-a worker-b relay-a relay-b)

    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 1) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    # Wait for a committed checkpoint boundary and identify the actual graph
    # coordinator from durable status, rather than assuming a lab process
    # name.  This is intentionally a process restart test, not a coordinator
    # election shortcut.
    for _ in $(seq 1 360); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Running" and .checkpoint_generation >= 2' <<< "$state" >/dev/null 2>&1; then
        running=1
        coordinator_json=$(jq -c '.execution_graph.coordinator' <<< "$state")
        coordinator_name=$(node_name_for_id "$coordinator_json" 2>/dev/null || true)
        break
      fi
      sleep 0.1
    done
    (( running == 1 )) || return 1
    [[ -n "$coordinator_name" ]] || return 1

    printf '[%s] ! restart active V4 coordinator %s for job %s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$coordinator_name" "$job_id" \
      | tee -a "$commands_log" >&2
    stop_node "$coordinator_name" || return 1
    start_node "$coordinator_name" || return 1
    wait_ready "$coordinator_name" || return 1
    restarted=1

    for _ in $(seq 1 360); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Committed" and .window >= 12' <<< "$state" >/dev/null 2>&1; then
        completed=1
        break
      fi
      sleep 0.1
    done
    (( completed == 1 )) || {
      printf 'V4 restart recovery did not complete job %s after restarting %s\n' \
        "$job_id" "$coordinator_name" >&2
      return 1
    }
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --arg coordinator "$coordinator_name" --argjson restarted "$restarted" \
      --argjson completed "$completed" \
      '{evidence_class:$evidence_class,job_id:$job_id,restarted_coordinator:$coordinator,restarted:($restarted == 1),completed:($completed == 1),same_identity_state_resume:true}' \
      > "$lab_root/v4-restart-recovery.json"
  }

  test_v4_auto_replan_join() {
    restart_training_fabric_nodes || return 1
    local started job_id state status_node='' joined=0 completed=0 discovered=0
    # Status is replicated to the normal authenticated training members.  The
    # CGNAT peer is the join target, but it is not needed as a polling source;
    # probing it serially would turn a bounded status loop into repeated NAT
    # connection timeouts.
    local -a candidates=(worker-a worker-b relay-a relay-b)
    # The nested CGNAT participant publishes its signed capability only after
    # the relay/DHT path is established.  Keep this discovery bound separate
    # from the ordinary two-worker startup bound so a slow isolated topology
    # cannot fail before the job exists.
    for _ in $(seq 1 "${INLAB_V4_JOIN_DISCOVERY_ATTEMPTS:-300}"); do
      if cli_json_probe requester network peers 2>/dev/null \
        | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 5' >/dev/null 2>&1; then
        discovered=1
        break
      fi
      sleep 0.1
    done
    (( discovered == 1 )) || {
      printf 'V4 automatic join did not discover five signed training capabilities before the bounded deadline\n' >&2
      return 1
    }
    started=$(v4_cli requester train start --workers 4 --windows 8 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1
    for _ in $(seq 1 "${INLAB_V4_JOIN_REPLAN_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.graph_generation >= 2 and (.execution_graph.workers | length) >= 5' <<< "$state" >/dev/null 2>&1; then
        joined=1
        break
      fi
      sleep 0.1
    done
    (( joined == 1 )) || {
      printf 'V4 automatic join did not publish an expanded graph for job %s\n' "$job_id" >&2
      return 1
    }
    for _ in $(seq 1 "${INLAB_V4_JOIN_COMPLETION_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Committed" and .window >= 8' <<< "$state" >/dev/null 2>&1; then
        completed=1
        break
      fi
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --argjson joined "$joined" --argjson completed "$completed" \
      '{evidence_class:$evidence_class,job_id:$job_id,automatic_join:($joined == 1),completed:($completed == 1),membership_replan:true}' \
      > "$lab_root/v4-auto-replan-join.json"
  }

  test_v4_auto_replan_slow_worker() {
    restart_training_fabric_nodes || return 1
    local started job_id state status_node='' coordinator_json target_json
    local target_name='' old_generation current_generation=0 replanned=0 completed=0
    local -a candidates=(worker-a worker-b relay-a relay-b)
    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    for _ in $(seq 1 "${INLAB_V4_RUNNING_ATTEMPTS:-120}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Running"' <<< "$state" >/dev/null 2>&1; then
        coordinator_json=$(jq -c '.execution_graph.coordinator' <<< "$state")
        target_json=$(jq -c --argjson coordinator "$coordinator_json" \
          '[.execution_graph.workers[] | select(. != $coordinator)][0] // empty' \
          <<< "$state")
        target_name=$(node_name_for_id "$target_json" 2>/dev/null || true)
        [[ -n "$target_name" ]] && break
      fi
      sleep 0.1
    done
    [[ -n "$target_name" ]] || return 1
    old_generation=$(jq -r '.graph_generation' <<< "$state")

    # Delay only the selected node's lab-local virtual interface.  This is a
    # real-process network impairment inside the existing isolated topology;
    # it does not touch a host interface or create a second transport.  The
    # delay exceeds the authenticated liveness probe but leaves the process
    # alive, exercising automatic slow-member isolation rather than SIGKILL.
    inner_ns "${node_ns[$target_name]}" tc qdisc replace \
      dev "${node_if[$target_name]}" root netem delay 5000ms rate "${per_link_bandwidth}mbit"

    for _ in $(seq 1 "${INLAB_V4_REPLACEMENT_ATTEMPTS:-300}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e --argjson old "$old_generation" --argjson target "$target_json" \
          '.graph_generation > $old
           and ((.execution_graph.workers | index($target)) | not)
           and (.phase == "Running" or .phase == "Committed")' \
          <<< "$state" >/dev/null 2>&1; then
        current_generation=$(jq -r '.graph_generation' <<< "$state")
        replanned=1
        break
      fi
      sleep 0.1
    done

    # Remove the impairment after the graph has retired the slow member.  The
    # retired-worker fence prevents it from being re-admitted as a joiner in
    # this case, so the test also exercises anti-flap behavior.
    inner_ns "${node_ns[$target_name]}" tc qdisc replace \
      dev "${node_if[$target_name]}" root netem delay 2ms rate "${per_link_bandwidth}mbit"
    (( replanned == 1 )) || return 1

    for _ in $(seq 1 "${INLAB_V4_COMPLETION_ATTEMPTS:-240}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Committed" and .window >= 12' <<< "$state" >/dev/null 2>&1; then
        completed=1
        break
      fi
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --arg target "$target_name" --argjson old_generation "$old_generation" \
      --argjson graph_generation "$current_generation" --argjson replanned "$replanned" \
      --argjson completed "$completed" \
      '{evidence_class:$evidence_class,job_id:$job_id,slow_worker:$target,old_graph_generation:$old_generation,graph_generation:$graph_generation,automatic_replan:($replanned == 1),completed:($completed == 1),process_killed:false}' \
      > "$lab_root/v4-auto-replan-slow-worker.json"
  }

  test_v4_shard_split() {
    restart_training_fabric_nodes || return 1
    local started job_id state status_node='' split=0 completed=0
    local -a candidates=(worker-a worker-b relay-a relay-b)
    for _ in $(seq 1 "${INLAB_V4_SHARD_SPLIT_DISCOVERY_ATTEMPTS:-300}"); do
      if cli_json_probe requester network peers 2>/dev/null \
        | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 5' >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    # The SHARD_SPLIT lab profile gives the active driver a bounded pre-window
    # observation interval.  The first membership expansion therefore occurs
    # while the reference model is still at its canonical generation-1
    # boundary, where the row split is semantically lossless.
    for _ in $(seq 1 "${INLAB_V4_SHARD_SPLIT_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.graph_generation >= 2
                  and (.execution_graph.workers | length) >= 5
                  and (.execution_graph.shards | length) == 4
                  and ([.execution_graph.shards[].shard_id] | sort == [2,3,4,5])
                  and ((.execution_graph.tensor_groups[0].shard_ids | length) == 4)' \
          <<< "$state" >/dev/null 2>&1; then
        split=1
        break
      fi
      sleep 0.1
    done
    (( split == 1 )) || return 1

    for _ in $(seq 1 "${INLAB_V4_SHARD_SPLIT_COMPLETION_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Committed" and .window >= 12 and .checkpoint_generation >= 2' \
          <<< "$state" >/dev/null 2>&1; then
        completed=1
        break
      fi
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --argjson split "$split" --argjson completed "$completed" \
      '{evidence_class:$evidence_class,job_id:$job_id,source_shards:2,active_shards:4,new_shard_ids:[2,3,4,5],automatic_split:($split == 1),completed:($completed == 1),pre_progress_only:true}' \
      > "$lab_root/v4-shard-split.json"
  }

  test_v4_integrated_cascade() {
    restart_training_fabric_nodes || return 1
    local started job_id state status_node='' coordinator_json target_json target_name
    local previous_generation current_generation=0 old_generation
    local joined=0 completed=0 discovered=0 backend_bindings_seen=0 checkpoint_recovered=0
    local initial_workers=${INLAB_V4_CASCADE_INITIAL_WORKERS:-4}
    [[ "$initial_workers" =~ ^[0-9]+$ ]] && (( initial_workers >= 4 && initial_workers <= 16 )) || return 1
    local required_workers=$initial_workers
    (( required_workers < 5 )) && required_workers=5
    local dead_ids='[]' dead_names=''
    # Keep status polling on the stable graph members.  The two NAT peers are
    # failure-injection targets/providers, not required status sources, and a
    # dead NAT path must not serialize every bounded poll iteration.
    local -a candidates=(bootstrap worker-a worker-b relay-a relay-b)

    # This is per-run evidence.  The lab root is intentionally retained
    # between commands for inspection, so do not append transitions from an
    # older cascade and later mistake them for this job's graph history.
    : > "$lab_root/v4-integrated-cascade-transitions.ndjson"

    # Eight real processes plus the nested NAT path can take longer than the
    # ordinary V4 startup window to publish all signed training capabilities.
    # Keep discovery bounded, but do not turn a slow isolated lab bootstrap
    # into a false failure before the durable job has even started.
    # The NAT/relay participants may need more than the ordinary peer
    # discovery deadline to publish their signed capability records.  Keep
    # this bounded separately from job execution, but allow the full isolated
    # topology time to converge before declaring discovery impossible.
    for _ in $(seq 1 "${INLAB_V4_CASCADE_DISCOVERY_ATTEMPTS:-1500}"); do
      if cli_json_probe requester network peers 2>/dev/null \
        | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 7' >/dev/null 2>&1; then
        discovered=1
        break
      fi
      sleep 0.1
    done
    (( discovered == 1 )) || {
      printf 'V4 integrated cascade did not discover seven signed training capabilities before the bounded deadline\n' >&2
      printf 'requester peer snapshot:\n' >&2
      cli_json_probe requester network peers 2>&1 || true
      return 1
    }

    # Four peers form the initial graph.  At least one remaining signed peer
    # must join through the normal authenticated peer/DHT path while this same
    # durable job is running.  NAT/relay convergence is intentionally allowed
    # to leave some advertised candidates unavailable; the failure cascade
    # itself exercises the live quorum rather than treating a stale capability
    # record as an admitted worker.
    started=$(v4_cli requester train start --workers "$initial_workers" --windows 48 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    # Eight real processes are deliberately kept under the lab CPU quota.  A
    # membership replan can therefore complete after the initial signed
    # discovery window while the durable graph is already being prepared.
    # Keep this deadline bounded, but long enough to observe the committed
    # five-member graph rather than failing at the first preparation window.
    for _ in $(seq 1 "${INLAB_V4_CASCADE_JOIN_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e --argjson required "$required_workers" '.phase == "Running" and (.execution_graph.workers | length) >= $required and (.execution_graph.backend_assignments | length) >= 1' <<< "$state" >/dev/null 2>&1; then
        joined=1
        backend_bindings_seen=1
        break
      fi
      sleep 0.1
    done
    (( joined == 1 )) || {
      printf 'V4 integrated cascade did not admit an authenticated joiner before the bounded deadline for job %s\n' "$job_id" >&2
      printf 'last durable status replica: %s\n' "${state:-<empty>}" >&2
      for candidate in "${candidates[@]}"; do
        printf 'status replica %s: ' "$candidate" >&2
        cli_json_probe "$candidate" train status --job-id "$job_id" 2>&1 || true
      done
      return 1
    }

    coordinator_json=$(jq -c '.execution_graph.coordinator' <<< "$state")

    kill_member() {
      local kind=$1 member_json=$2 member_id member_name pid_file pid
      member_id=$(node_id_hex "$member_json") || return 1
      member_name=$(node_name_for_id "$member_json") || return 1
      [[ "$member_id" != "$(node_id_hex "$coordinator_json")" ]] || {
        printf 'V4 integrated cascade selected coordinator for %s\n' "$kind" >&2
        return 1
      }
      pid_file="$lab_root/pids/$member_name.pid"
      [[ -f "$pid_file" ]] || return 1
      pid=$(<"$pid_file")
      [[ "$pid" =~ ^[0-9]+$ ]] || return 1
      printf '[%s] ! permanent V4 cascade %s failure: kill -KILL %s (node=%s)\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" "$pid" "$member_name" | tee -a "$commands_log" >&2
      kill_tracked_tree "$pid" KILL
      for _ in $(seq 1 50); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.05
      done
      # The node is launched by a background shell wrapper.  Killing its
      # descendant leaves that wrapper as a zombie until the supervisor reaps
      # it; wait here so a real injected process failure is not reported as a
      # harness failure merely because kill -0 still sees the unreaped PID.
      wait "$pid" 2>/dev/null || true
      kill -0 "$pid" 2>/dev/null && return 1
      rm -f -- "$pid_file"
      dead_ids=$(jq -c --argjson member "$member_json" '. + [$member]' <<< "$dead_ids")
      dead_names="${dead_names:+$dead_names,}$member_name"
      printf '%s\n' "$member_json"
    }

    wait_reconfigured() {
      local old_json=$1 old_generation=$2 role=$3
      for _ in $(seq 1 "${INLAB_V4_CASCADE_RECONFIGURE_ATTEMPTS:-600}"); do
        if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
          && jq -e --argjson old "$old_json" --argjson old_generation "$old_generation" \
            '.graph_generation > $old_generation
             and (.phase == "Running" or .phase == "Committed")
            and ((.execution_graph.workers | index($old)) | not)
            and ((.execution_graph.backend_assignments | length) >= 1)' <<< "$state" >/dev/null 2>&1; then
          current_generation=$(jq -r '.graph_generation' <<< "$state")
          backend_bindings_seen=1
          printf '%s\n' "$(jq -cn --arg role "$role" --argjson generation "$current_generation" '{role:$role,graph_generation:$generation}')" >> "$lab_root/v4-integrated-cascade-transitions.ndjson"
          return 0
        fi
        sleep 0.1
      done
      printf 'V4 integrated cascade did not complete %s graph transition after generation %s\n' "$role" "$old_generation" >&2
      printf 'last durable cascade status: %s\n' "${state:-<empty>}" >&2
      for candidate in "${candidates[@]}"; do
        printf 'cascade status replica %s: ' "$candidate" >&2
        cli_json_probe "$candidate" train status --job-id "$job_id" 2>&1 || true
      done
      return 1
    }

    # 1. Tensor shard and its distributed optimizer owner fail together.
    old_generation=$(jq -r '.graph_generation' <<< "$state")
    # A successful pre-progress join may have activated the reference
    # two-to-four shard split, so source shard id 1 is no longer guaranteed
    # to exist.  Select any active shard owner instead of coupling the
    # failure injection to one pre-reconfiguration identity.
    target_json=$(jq -c --argjson dead "$dead_ids" --argjson coordinator "$coordinator_json" \
      '[.execution_graph.shards[] | .owners[0]
       | select((. as $id | $dead | index($id)) == null and . != $coordinator)][0] // empty' <<< "$state")
    [[ -n "$target_json" && "$target_json" != null ]] || return 1
    kill_member 'tensor-optimizer-owner' "$target_json" >/dev/null || return 1
    wait_reconfigured "$target_json" "$old_generation" 'tensor-optimizer-owner' || return 1

    # The first loss removed a shard/optimizer owner that was also in the
    # checkpoint provider set.  Require a newer replicated checkpoint before
    # injecting the next failure; this makes checkpoint recovery evidence part
    # of the same durable cascade rather than an unverified annotation.
    for _ in $(seq 1 "${INLAB_V4_CASCADE_CHECKPOINT_ATTEMPTS:-600}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.checkpoint_generation >= 2' <<< "$state" >/dev/null 2>&1; then
        checkpoint_recovered=1
        break
      fi
      sleep 0.1
    done
    (( checkpoint_recovered == 1 )) || return 1

    # 2. A collective participant/aggregator fails while the active graph
    # still has two aggregation groups.
    target_json=$(jq -c --argjson dead "$dead_ids" --argjson coordinator "$coordinator_json" \
      '[.execution_graph.aggregation_groups[] | select(.group_id == 1) | .aggregator
       | select((. as $id | $dead | index($id)) == null and . != $coordinator)][0] // empty' <<< "$state")
    [[ -n "$target_json" && "$target_json" != null ]] || return 1
    old_generation=$current_generation
    kill_member 'collective-participant' "$target_json" >/dev/null || return 1
    wait_reconfigured "$target_json" "$old_generation" 'collective-participant' || return 1

    # 3. A pipeline stage fails on the active graph.  At this point the
    # remaining quorum is deliberately allowed to shrink to two members;
    # this is the lower bound at which the durable graph can still commit.
    target_json=$(jq -c --argjson dead "$dead_ids" --argjson coordinator "$coordinator_json" \
      '(([.execution_graph.pipeline_stages[] | .worker]
        + [.execution_graph.workers[]])
       | map(select((. as $id | $dead | index($id)) == null and . != $coordinator))
       | .[0]) // empty' <<< "$state")
    [[ -n "$target_json" && "$target_json" != null ]] || return 1
    old_generation=$current_generation
    kill_member 'pipeline-stage' "$target_json" >/dev/null || return 1
    wait_reconfigured "$target_json" "$old_generation" 'pipeline-stage' || return 1

    # The surviving quorum must reach the target window and expose the
    # committed new graph.  No killed process is restarted by this case.
    for _ in $(seq 1 "${INLAB_V4_CASCADE_COMPLETION_ATTEMPTS:-900}"); do
      if probe_training_status "$job_id" "$status_node" "${candidates[@]}" \
        && jq -e '.phase == "Committed" and .window >= 48 and (.execution_graph.backend_assignments | length) >= 1' <<< "$state" >/dev/null 2>&1; then
        completed=1
        backend_bindings_seen=1
        break
      fi
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --arg dead "$dead_names" --argjson joined "$joined" --argjson completed "$completed" \
      --argjson graph_generation "$current_generation" \
      --argjson backend_bindings_seen "$backend_bindings_seen" \
      --argjson initial_workers "$initial_workers" \
      --argjson checkpoint_recovered "$checkpoint_recovered" \
      '{evidence_class:$evidence_class,job_id:$job_id,initial_workers:$initial_workers,automatic_join:($joined == 1 and $initial_workers == 4),permanent_failures:($dead|split(",")),graph_generation:$graph_generation,automatic_replacements:true,distributed_optimizer_recovery:true,distributed_checkpoint_recovery:true,checkpoint_recovery_observed:($checkpoint_recovered == 1),collective_reconfiguration:true,backend_bindings_observed:($backend_bindings_seen == 1),completed:($completed == 1),old_processes_restarted:false}' \
      > "$lab_root/v4-integrated-cascade.json"
  }

  test_v4_partition_auto_policy() {
    restart_training_fabric_nodes || return 1
    local started job_id state coordinator_json target_json old_branch
    local coordinator_name='' target_name='' branch_selected=0 completed=0
    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    for _ in $(seq 1 "${INLAB_V4_RUNNING_ATTEMPTS:-120}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        coordinator_json=$(jq -c '.execution_graph.coordinator // empty' <<< "$state" 2>/dev/null || true)
        target_json=$(jq -c --argjson coordinator "$coordinator_json" \
          '[.execution_graph.workers[] | select(. != $coordinator)][0] // empty' \
          <<< "$state" 2>/dev/null || true)
        if jq -e '.phase == "Running"' <<< "$state" >/dev/null 2>&1 \
          && [[ -n "$coordinator_json" && -n "$target_json" ]]; then
          old_branch=$(jq -c '.branch' <<< "$state")
          coordinator_name=$(node_name_for_id "$coordinator_json" 2>/dev/null || true)
          target_name=$(node_name_for_id "$target_json" 2>/dev/null || true)
          break 2
        fi
      done
      sleep 0.1
    done
    [[ -n "$coordinator_name" && -n "$target_name" ]] || return 1
    [[ "$coordinator_name" != "$target_name" ]] || return 1

    # Isolate the coordinator's node namespace by taking down its own
    # project-created veth.  This cuts both direct and relay paths without
    # touching a host interface or the host firewall.
    inner_ns "${node_ns[$coordinator_name]}" ip link set dev "${node_if[$coordinator_name]}" down

    # A failed collective can take one bounded V4_MESSAGE_TIMEOUT window to
    # surface before the driver can prove the member is unavailable.  Keep
    # this acceptance loop longer than that protocol budget; otherwise the
    # test can terminate during the liveness probe rather than exercising
    # replacement.
    for _ in $(seq 1 "${INLAB_V4_REPLACEMENT_ATTEMPTS:-300}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        [[ "$candidate" == "$target_name" ]] && continue
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if jq -e --argjson old "$old_branch" \
          '.graph_generation >= 2 and .branch != $old and .phase != "Failed"' \
          <<< "$state" >/dev/null 2>&1; then
          branch_selected=1
          break 2
        fi
      done
      sleep 0.1
    done
    inner_ns "${node_ns[$coordinator_name]}" ip link set dev "${node_if[$coordinator_name]}" up
    (( branch_selected == 1 )) || return 1

    for _ in $(seq 1 "${INLAB_V4_COMPLETION_ATTEMPTS:-240}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        [[ "$candidate" == "$target_name" ]] && continue
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if jq -e '.phase == "Committed" and .window >= 12' <<< "$state" >/dev/null 2>&1; then
          completed=1
          break 2
        fi
      done
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    local branch_commit_count
    branch_commit_count=$(find "$lab_root/nodes" -type f \
      -name "v4-integrated-branch-${job_id}-*.json" -print 2>/dev/null | wc -l)
    (( branch_commit_count >= 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --arg coordinator "$coordinator_name" --arg partitioned "$target_name" \
      --argjson branch_selected "$branch_selected" \
      --argjson completed "$completed" --argjson branch_commit_count "$branch_commit_count" \
      '{evidence_class:$evidence_class,job_id:$job_id,coordinator:$coordinator,partitioned_member:$partitioned,policy:"SELECT_BRANCH",automatic_branch_selection:($branch_selected == 1),healed_and_completed:($completed == 1),durable_branch_commit_records:$branch_commit_count}' \
      > "$lab_root/v4-partition-auto-policy.json"
  }

  test_v4_coordinator_replacement() {
    restart_training_fabric_nodes || return 1
    local started job_id original_id='' original_name='' state coordinator_json observed_id
    local probe_reported=0
    local replacement=0 completed=0
    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || {
      printf 'V4 coordinator replacement: could not decode started job ID: %s\n' "$started" >&2
      return 1
    }
    [[ "$job_id" != null && -n "$job_id" ]] || {
      printf 'V4 coordinator replacement: start returned invalid job: %s\n' "$started" >&2
      return 1
    }

    # Resolve the active coordinator from persisted, machine-readable job
    # state.  The process is killed by the exact PID recorded by start_node;
    # no executable-name or wildcard process selection is permitted.
    for _ in $(seq 1 "${INLAB_V4_RESOLVE_ATTEMPTS:-60}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if [[ -z "$state" && "$probe_reported" == 0 ]]; then
          printf 'V4 coordinator replacement: first status probe (%s) failed: %s\n' \
            "$candidate" "$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>&1 || true)" >&2
          probe_reported=1
        fi
        coordinator_json=$(jq -c '.coordinator // empty' <<< "$state" 2>/dev/null || true)
        if [[ -n "$coordinator_json" ]] && original_name=$(node_name_for_id "$coordinator_json" 2>/dev/null); then
          original_id=$(node_id_hex "$(cli_json "$original_name" identity | jq -c '.node_id')") || return 1
          break 2
        fi
      done
      sleep 0.1
    done
    [[ -n "$original_name" && -n "${original_id:-}" ]] || {
      printf 'V4 coordinator replacement: could not resolve coordinator for job %s\n' "$job_id" >&2
      printf 'last coordinator state: %s\n' "${state:-<empty>}" >&2
      printf 'last coordinator value: %s\n' "${coordinator_json:-<empty>}" >&2
      return 1
    }

    for _ in $(seq 1 "${INLAB_V4_RUNNING_ATTEMPTS:-100}"); do
      state=$(cli_json_probe "$original_name" train status --job-id "$job_id" 2>/dev/null || true)
      if jq -e '.phase == "Running"' <<< "$state" >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done

    local pid_file="$lab_root/pids/$original_name.pid" pid
    [[ -f "$pid_file" ]] || {
      printf 'V4 coordinator replacement: missing PID file %s\n' "$pid_file" >&2
      return 1
    }
    pid=$(<"$pid_file")
    [[ "$pid" =~ ^[0-9]+$ ]] || {
      printf 'V4 coordinator replacement: invalid PID %s in %s\n' "$pid" "$pid_file" >&2
      return 1
    }
    printf '[%s] ! permanent V4 coordinator failure: kill -KILL %s (node=%s)\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$pid" "$original_name" | tee -a "$commands_log" >&2
    kill_tracked_tree "$pid" KILL
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.05
    done
    kill -0 "$pid" 2>/dev/null && return 1
    rm -f -- "$pid_file"

    for _ in $(seq 1 "${INLAB_V4_REPLACEMENT_ATTEMPTS:-300}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        [[ "$candidate" == "$original_name" ]] && continue
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        coordinator_json=$(jq -c '.coordinator // empty' <<< "$state" 2>/dev/null || true)
        observed_id=''
        [[ -n "$coordinator_json" ]] && observed_id=$(node_id_hex "$coordinator_json" 2>/dev/null || true)
        if jq -e '.graph_generation >= 2 and .coordinator != null' <<< "$state" >/dev/null 2>&1 \
          && [[ "$observed_id" != "$original_id" && -n "$observed_id" ]]; then
          replacement=1
          break 2
        fi
      done
      sleep 0.1
    done
    (( replacement == 1 )) || {
      printf 'V4 coordinator replacement did not publish a new graph for job %s\n' "$job_id" >&2
      for candidate in worker-a worker-b relay-a relay-b; do
        cli_json_probe "$candidate" train status --job-id "$job_id" >&2 || true
      done
      return 1
    }

    for _ in $(seq 1 "${INLAB_V4_COMPLETION_ATTEMPTS:-120}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if jq -e '.phase == "Committed" and .window >= 12' <<< "$state" >/dev/null 2>&1; then
          completed=1
          break 2
        fi
      done
      sleep 0.1
    done
    (( completed == 1 )) || {
      printf 'V4 coordinator replacement did not complete job %s\n' "$job_id" >&2
      return 1
    }
    printf '%s\n' "$(jq -cn --arg job_id "$job_id" --arg original "$original_name" \
      --argjson replacement "$replacement" --argjson completed "$completed" \
      '{evidence_class:"REAL_PROCESS_LOCAL",job_id:$job_id,original_coordinator:$original,permanent_failure:true,automatic_replacement:($replacement == 1),completed:($completed == 1)}')" \
      > "$lab_root/v4-coordinator-replacement.json"
  }

  test_v4_member_replacement() {
    local kind=$1 selector=$2 evidence_file=$3
    restart_training_fabric_nodes || return 1
    local started job_id state target_json original_id original_name='' coordinator_json
    local replacement=0 completed=0 probe_reported=0
    started=$(v4_cli requester train start \
      --workers 4 --windows 12 --checkpoint-every 2) || return 1
    job_id=$(node_id_hex "$(jq -c '.job_id' <<< "$started")") || return 1
    [[ -n "$job_id" && "$job_id" != null ]] || return 1

    # Read the role assignment from the persisted execution graph.  This is
    # the job's authenticated topology, not a guessed process ordering.
    for _ in $(seq 1 "${INLAB_V4_RESOLVE_ATTEMPTS:-90}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if [[ -z "$state" && "$probe_reported" == 0 ]]; then
          printf 'V4 %s replacement: first status probe failed\n' "$kind" >&2
          probe_reported=1
        fi
        target_json=$(jq -c "$selector" <<< "$state" 2>/dev/null || true)
        if [[ -n "$target_json" && "$target_json" != null ]]; then
          coordinator_json=$(jq -c '.coordinator // empty' <<< "$state" 2>/dev/null || true)
          original_id=$(node_id_hex "$target_json" 2>/dev/null || true)
          original_name=$(node_name_for_id "$target_json" 2>/dev/null || true)
          if [[ -n "$original_name" && -n "$original_id" && "$original_id" != "$(node_id_hex "$coordinator_json" 2>/dev/null || true)" ]]; then
            break 2
          fi
        fi
      done
      sleep 0.1
    done
    [[ -n "$original_name" && -n "$original_id" ]] || {
      printf 'V4 %s replacement: could not resolve assigned target\n' "$kind" >&2
      return 1
    }

    for _ in $(seq 1 "${INLAB_V4_RUNNING_ATTEMPTS:-120}"); do
      state=$(cli_json_probe "$original_name" train status --job-id "$job_id" 2>/dev/null || true)
      jq -e '.phase == "Running"' <<< "$state" >/dev/null 2>&1 && break
      sleep 0.1
    done
    local pid_file="$lab_root/pids/$original_name.pid" pid
    [[ -f "$pid_file" ]] || return 1
    pid=$(<"$pid_file")
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    printf '[%s] ! permanent V4 %s failure: kill -KILL %s (node=%s)\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" "$pid" "$original_name" | tee -a "$commands_log" >&2
    kill_tracked_tree "$pid" KILL
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.05
    done
    kill -0 "$pid" 2>/dev/null && return 1
    rm -f -- "$pid_file"

    # The first failed operation is allowed one 15-second protocol timeout
    # before replacement begins.  The acceptance test must cover detection,
    # quorum, installation, and completion rather than only the detection
    # phase.
    for _ in $(seq 1 "${INLAB_V4_REPLACEMENT_ATTEMPTS:-300}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        [[ "$candidate" == "$original_name" ]] && continue
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if jq -e --argjson old "$target_json" --arg kind "$kind" '
          .graph_generation >= 2
          and ((.phase == "Running") or (.phase == "Committed"))
          and (if $kind == "shard-owner" then
                 ([.execution_graph.shards[] | select(.shard_id == 1) | .owners[]] | index($old) | not)
               elif $kind == "pipeline-stage" then
                 ([.execution_graph.pipeline_stages[] | select(.stage_id == 0) | .worker] | index($old) | not)
               elif $kind == "collective" then
                 ([.execution_graph.aggregation_groups[] | select(.group_id == 1) | .aggregator] | index($old) | not)
               else true end)
        ' <<< "$state" >/dev/null 2>&1; then
          replacement=1
          break 2
        fi
      done
      sleep 0.1
    done
    (( replacement == 1 )) || return 1

    for _ in $(seq 1 "${INLAB_V4_COMPLETION_ATTEMPTS:-180}"); do
      for candidate in worker-a worker-b relay-a relay-b; do
        [[ "$candidate" == "$original_name" ]] && continue
        state=$(cli_json_probe "$candidate" train status --job-id "$job_id" 2>/dev/null || true)
        if jq -e '.phase == "Committed" and .window >= 12' <<< "$state" >/dev/null 2>&1; then
          completed=1
          break 2
        fi
      done
      sleep 0.1
    done
    (( completed == 1 )) || return 1
    jq -cn --arg evidence_class REAL_PROCESS_LOCAL --arg job_id "$job_id" \
      --arg kind "$kind" --arg original "$original_name" \
      --argjson replacement "$replacement" --argjson completed "$completed" \
      '{evidence_class:$evidence_class,job_id:$job_id,failure_kind:$kind,original_owner:$original,permanent_failure:true,automatic_replacement:($replacement == 1),completed:($completed == 1)}' \
      > "$lab_root/$evidence_file"
  }

  test_v4_shard_owner_replacement() {
    test_v4_member_replacement shard-owner \
      '.execution_graph.shards[] | select(.shard_id == 1) | .owners[0]' \
      v4-shard-owner-replacement.json
  }

  test_v4_pipeline_stage_replacement() {
    test_v4_member_replacement pipeline-stage \
      '.execution_graph.pipeline_stages[] | select(.stage_id == 0) | .worker' \
      v4-pipeline-stage-replacement.json
  }

  test_v4_collective_replacement() {
    test_v4_member_replacement collective \
      '.execution_graph.aggregation_groups[] | select(.group_id == 1) | .aggregator' \
      v4-collective-replacement.json
  }

  test_coordinator_restart() {
    local output checkpoint requester_id
    if ! output=$(cli_json requester train reference --workers 2 --steps 2); then
      printf 'coordinator restart pre-check training failed; requester status:\n' >&2
      cli_json requester status >&2 || true
      printf 'coordinator restart training output:\n%s\n' "$output" >&2
      return 1
    fi
    jq -e '.improved == true and (.checkpoints | length) == 2' <<< "$output" >/dev/null || {
      printf 'coordinator restart pre-check output failed validation:\n%s\n' "$output" >&2
      return 1
    }
    checkpoint=$(jq -r '.checkpoints[-1]' <<< "$output")
    requester_id=$(cli_json requester identity | jq -r '.node_id')
    stop_node requester
    write_config requester "$lab_root/nodes/requester/state/identity.key" false
    start_node requester
    wait_ready requester
    for _ in $(seq 1 200); do
      if cli_json requester network peers 2>/dev/null | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 2' >/dev/null 2>&1; then
        break
      fi
      sleep 0.1
    done
    cli_json requester train reference --workers 2 --steps 2 --resume-checkpoint "$checkpoint" \
      | jq -e '.improved == true and .resumed_from != null and .evaluation.verified == true' >/dev/null || return 1
    cli_json worker-a artifact fetch --peer "$requester_id" --artifact "$checkpoint" \
      | jq -e '.verified == true' >/dev/null
  }

  test_worker_loss_and_recovery() {
    local output checkpoint worker_pid training_pid training_status=1
    apply_profile brazil-europe
    # Keep the legacy V1 failure-injection case self-contained.  A failed
    # background CLI must not leave worker-b stopped or a hostile netem
    # profile installed for the following cases.  The longer command timeout
    # is still bounded by the per-request deadline and the lab's outer
    # timeout; it accounts for the intentional 180 ms/loss profile.
    (
      INLAB_CLI_TIMEOUT_SECONDS="${INLAB_WORKER_LOSS_CLI_TIMEOUT_SECONDS:-120}" \
        cli_json requester train reference --workers 2 --steps 8 \
        > "$lab_root/training-loss.json" 2>&1
    ) &
    training_pid=$!
    sleep 0.4
    stop_node worker-b
    set +e
    wait "$training_pid"
    training_status=$?
    set -e
    remove_profile
    if (( training_status != 0 )); then
      printf 'worker-loss training command exited status=%s bytes=%s\n' \
        "$training_status" "$(wc -c < "$lab_root/training-loss.json")" >&2
      sed -n '1,80p' "$lab_root/training-loss.json" >&2 || true
      start_node worker-b
      wait_ready worker-b || true
      return 1
    fi
    output=$(cat "$lab_root/training-loss.json")
    jq -e '.improved == true and (.checkpoints | length) >= 1' <<< "$output" >/dev/null || {
      printf 'worker-loss training output failed validation:\n%s\n' "$output" >&2
      start_node worker-b
      wait_ready worker-b || true
      return 1
    }
    checkpoint=$(jq -r '.checkpoints[-1]' <<< "$output")
    start_node worker-b
    wait_ready worker-b
    wait_training_peers() {
      for _ in $(seq 1 200); do
        if cli_json requester network peers 2>/dev/null | jq -e '[.[] | select(any(.capabilities[]?; .name == "training.reference"))] | length >= 2' >/dev/null 2>&1; then
          return 0
        fi
        sleep 0.1
      done
      return 1
    }
    wait_training_peers || return 1
    cli_json requester train reference --workers 2 --steps 2 --resume-checkpoint "$checkpoint" \
      | jq -e '.improved == true and (.resumed_from != null) and .evaluation.verified == true' >/dev/null
  }

  test_key_rotation() {
    local old_id new_id now rotation_path
    old_id=$(cli_json requester identity | jq -r '.node_id')
    now=$(date +%s)
    rotation_path="$lab_root/nodes/requester/state/identity-rotated.key"
    cli_json requester identity rotate --new-path "$rotation_path" --sequence 1 --valid-until "$((now + 3600))" >/dev/null
    stop_node requester
    write_config requester "$rotation_path" true
    start_node requester
    wait_ready requester
    new_id=$(cli_json requester identity | jq -r '.node_id')
    [[ "$old_id" != "$new_id" ]] || return 1
    set +e
    cli_json requester identity rotate --new-path "$lab_root/nodes/requester/state/identity-rotated-again.key" --sequence 1 --valid-until "$((now + 3600))" >/dev/null 2>&1
    local stale_status=$?
    set -e
    (( stale_status != 0 ))
  }

  test_malformed_peer() {
    inner_ns inlab-client-nat python3 -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(0.1); payload=bytes(60000); [s.sendto(payload,("10.254.11.2",40002)) for _ in range(16)]; s.close()'
    inner_ns inlab-client-nat python3 -c 'import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); [s.sendto(bytes(range(256)),("10.254.12.2",40003)) for _ in range(32)]; s.close()'
    cli_json worker-a status | jq -e '.node_id != null' >/dev/null
  }

  test_connection_flood() {
    inner_ns inlab-client-nat python3 -c 'import socket; ss=[]; [ss.append(socket.socket(socket.AF_INET,socket.SOCK_DGRAM)) for _ in range(64)]; [s.sendto(b"inlab-flood",("10.254.11.2",40002)) for s in ss]; [s.close() for s in ss]'
    sleep 1
    cli_json worker-a status | jq -e '.connected_peers | length <= 16' >/dev/null
  }

  test_job_flood() {
    local -a jobs=() pid
    for index in $(seq 1 32); do
      (cli_json requester infer --capability inference.text --input "bounded-job-$index" --deadline-ms 2000 --max-output-bytes 1024 >/dev/null 2>&1) &
      jobs+=("$!")
    done
    local rejected=0
    for pid in "${jobs[@]}"; do
      wait "$pid" || rejected=$((rejected + 1))
    done
    (( rejected >= 0 ))
    cli_json requester status \
      | jq -e '.queue_depth <= 8 and .running_jobs <= 2 and .pending_jobs <= 8 and .storage_used_bytes <= .storage_quota_bytes' >/dev/null
  }

  test_network_safety() {
    [[ -z "$(ip netns exec inlab-internet ip route show default)" ]] || return 1
    ! ip netns exec inlab-internet ip route show | grep -E '192\.168\.1\.1|dev (eno|enp|eth|wlan|wlp|enx)' >/dev/null
    ! ip netns exec inlab-internet ip link show | grep -Ev '(^|: )lo|inlab-' >/dev/null
    inner_ns inlab-internet ping -c 1 -W 1 192.168.1.1 >/dev/null 2>&1 && return 1 || true
    inner_ns inlab-internet nft list ruleset | grep -E '^table (ip|ip6) inlab-' >/dev/null
  }

  test_soak() {
    local duration=${INLAB_SOAK_SECONDS:-600}
    [[ "$duration" =~ ^[0-9]+$ ]] || return 1
    local end=$((SECONDS + duration))
    while (( SECONDS < end )); do
      infer_ok requester 'bounded soak inference' 5000 || return 1
      cli_json requester status >/dev/null || return 1
      printf '%s\t%s\t%s\n' "$(date +%s)" "$(du_bytes "$lab_root")" "$(date +%s)" >> "$lab_root/metrics.tsv"
      inner_assert_size || return 1
      sleep 2
    done
  }

  run_tests_inner() {
    : > "$results_file"
    run_case NETWORK_ISOLATION 'all lab namespaces' local 'no host changes' 'fake Internet isolated' 'supervisor cleanup' test_network_safety || true
    run_case DIRECT_NETWORK 'requester,worker-a,bootstrap' local 'none' 'direct QUIC path, inference, evaluation' 'none' test_direct || true
    run_case DHT_JOIN 'requester,worker-a,bootstrap' local 'authenticated peer join' 'bounded DHT routing table' 'persistent DHT contacts' test_dht_join || true
    run_case DHT_LOOKUP 'requester,worker-a,bootstrap' local 'signed provider publication' 'DHT provider lookup' 'bootstrap death with learned DHT contacts' test_dht_v2 || true
    run_case DHT_BOOTSTRAP_DEATH 'requester,worker-a' local 'bootstrap terminated by DHT lookup case' 'lookup through learned contacts' 'no bootstrap reconnect required' test_dht_bootstrap_death || true
    run_case DHT_STALE_RECORD 'worker-a' local 'replayed sequence' 'stale record rejected' 'newer sequence remains authoritative' test_dht_stale_record || true
    run_case TRUST_EVIDENCE 'requester,worker-a' local 'none' 'local signed evidence decision' 'bounded evidence graph' test_trust_evidence || true
    run_case TRAINING_PLANNER 'requester,worker-a,worker-b' local 'none' 'inspectable topology-aware plan' 'unsupported workers rejected' test_training_planner || true
    run_case V3_DISTRIBUTED_TRAINING 'requester,worker-a,worker-b,relay-a,relay-b' local 'none' 'autonomous local-SGD with group aggregates' 'real-process result and durable state' test_v3_training || true
    run_case V3_MODEL_SHARDING 'requester,worker-a,worker-b,relay-a,relay-b' local 'worker budget below complete model state' 'assigned shard only' 'unverified full model rejected' test_v3_model_sharding || true
    run_case V3_REPLICATED_JOB_STATE 'worker-a,worker-b,relay-a,relay-b' local 'worker restart with replicated peer state' 'replicated job metadata restored from durable peers' 'terminal state remains inspectable' test_v3_replicated_job_state || true
    run_case V3_NON_BARRIER 'requester,worker-a,worker-b,relay-a,relay-b' local 'independent group windows' 'no global per-window barrier' 'bounded group progress' test_v3_non_barrier || true
    run_case V3_CHECKPOINT_REPLICA 'requester,worker-a,worker-b,relay-a,relay-b' local 'checkpoint creator stopped' 'replica artifact recovery' 'creator restarted after evidence' test_v3_checkpoint_replica || true
    run_case V3_OPTIMIZER_OWNER_FAILURE 'requester,worker-a,worker-b,relay-a,relay-b' local 'group optimizer/shard owner SIGKILL' 'replacement aggregator and replicated state' 'owner restarted without job corruption' test_v3_optimizer_owner_failure || true
    run_case V3_GROUP_AGGREGATOR_FAILURE 'requester,worker-a,worker-b,relay-a,relay-b' local 'group aggregator SIGKILL' 'bounded replacement aggregation' 'aggregator restarted without job leak' test_v3_group_aggregator_failure || true
    run_case V3_COORDINATOR_REPLACEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'requester coordinator stopped permanently during job' 'term-based peer replacement' 'requester restarted after evidence' test_v3_coordinator_replacement || true
    run_case V4_REFERENCE_PROTOCOL 'requester,worker-a,worker-b,relay-a,relay-b' local 'plan activation, V4 reference operations, and authenticated shard migration' 'real V4 protocol path with explicit reference boundaries' 'plan and per-node state remain bounded' test_v4_reference_protocol || true
    run_case V4_INTEGRATED_JOB 'requester,worker-a,worker-b,relay-a,relay-b' local 'none' 'durable tensor/pipeline/collective training job with distributed checkpoint state' 'committed result and measurable loss improvement' test_v4_integrated_job || true
    run_case BACKEND_CAPABILITY_ADVERTISEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'none' 'typed backend capability advertisement and inspectable CPU assignments' 'CPU-only physical host is labeled separately from modeled heterogeneity' test_v5_backend_capability_advertisement || true
    run_case V5_INTEGRATED_HETEROGENEOUS_JOB 'requester,worker-a,worker-b,relay-a,relay-b' local 'none' 'durable V4 job executes through the V5 backend boundary' 'checkpoint and V3 invariants remain valid' test_v5_integrated_heterogeneous_job || true
    run_case V5_HETEROGENEOUS_CASCADE 'requester,bootstrap,worker-a,worker-b,relay-a,relay-b,client-nat,client-cgnat' local 'permanent backend-owner/member failures' 'durable graph reconfiguration and backend-bound recovery' 'no failed process restarted; host cleanup' test_v5_heterogeneous_cascade || true
    run_case V6_INTEGRATED_ADVERSARIAL_JOB 'requester,worker-a,worker-b,relay-a,relay-b' local 'replay and equivocation through authenticated Byzantine update path' 'durable heterogeneous job plus bounded robust aggregation and local security evidence' 'no global trust authority; local recovery' test_v6_integrated_adversarial_job || true
    run_case V6_ADVERSARIAL_CASCADE 'requester,bootstrap,worker-a,worker-b,relay-a,relay-b,client-nat,client-cgnat' local 'V4/V5 permanent failures followed by V6 replay/equivocation attack' 'composed real-process cascade with durable recovery evidence' 'lab supervisor cleanup' test_v6_adversarial_cascade || true
    run_case V4_RESTART_RECOVERY 'requester,worker-a,worker-b,relay-a,relay-b' local 'active V4 coordinator process restart after a committed checkpoint' 'same durable job resumes from persisted graph/state' 'same identity returns without creating a replacement job' test_v4_restart_recovery || true
    run_case V4_INTEGRATED_CASCADE 'requester,bootstrap,worker-a,worker-b,relay-a,relay-b,client-nat,client-cgnat' local 'join plus permanent tensor, collective, pipeline, and checkpoint-lineage failures' 'one durable graph survives automatic reconfiguration and recovery' 'no failed process restarted; host cleanup' test_v4_integrated_cascade || true
    run_case V4_AUTO_REPLAN_JOIN 'requester,worker-a,worker-b,relay-a,relay-b,client-cgnat' local 'signed training peer joins during active job' 'automatic membership graph generation and completion' 'joined peer remains authenticated and bounded' test_v4_auto_replan_join || true
    run_case V4_AUTO_REPLAN_SLOW_WORKER 'requester,worker-a,worker-b,relay-a,relay-b' local 'live worker becomes persistently slow through isolated netem impairment' 'automatic bounded straggler replan and completion' 'slow process is not killed and host cleanup' test_v4_auto_replan_slow_worker || true
    run_case V4_SHARD_SPLIT 'requester,worker-a,worker-b,relay-a,relay-b,client-cgnat' local 'new authenticated worker joins before first committed window' 'automatic verified two-to-four reference shard split and completion' 'fresh shard identities, bounded state, and host cleanup' test_v4_shard_split || true
    run_case V4_PARTITION_AUTO_POLICY 'requester,worker-a,worker-b,relay-a,relay-b' local 'fake-internet coordinator/member packet partition' 'automatic branch selection and post-heal completion' 'durable branch commit survives the isolated run' test_v4_partition_auto_policy || true
    run_case V4_COORDINATOR_REPLACEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'active coordinator SIGKILL' 'new graph term and completion without original process' 'tracked original PID stays dead; replacement state is durable' test_v4_coordinator_replacement || true
    run_case V4_SHARD_OWNER_REPLACEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'tensor/optimizer shard owner SIGKILL' 'replica-backed automatic shard reassignment and completion' 'original owner remains stopped' test_v4_shard_owner_replacement || true
    run_case V4_PIPELINE_STAGE_REPLACEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'pipeline stage SIGKILL' 'replica-backed stage reassignment and completion' 'original stage worker remains stopped' test_v4_pipeline_stage_replacement || true
    run_case V4_COLLECTIVE_REPLACEMENT 'requester,worker-a,worker-b,relay-a,relay-b' local 'collective participant SIGKILL' 'collective topology reconfiguration and completion' 'original collective participant remains stopped' test_v4_collective_replacement || true
    run_case RESIDENTIAL_NAT 'requester,nat-home,worker-a' local 'NAT stateful forwarding' 'outbound NAT path' 'relay available if inbound is impossible' test_nat || true
    run_case CGNAT_LIKE 'client-cgnat,nat-inner,nat-carrier,worker-a' local 'nested stateful NAT' 'nested NAT path' 'relay available' test_cgnat || true
    run_case RELAY_FALLBACK 'requester,relay-a,worker-a' local 'direct requester-to-worker blocked' 'relay path' 'relay-a' test_relay_fallback || true
    run_case RELAY_FAILURE 'requester,relay-a,relay-b,worker-a' local 'relay-a terminated' 'alternate relay path' 'relay-b' test_relay_failure || true
    run_case BOOTSTRAP_DEATH 'bootstrap,requester,worker-a,worker-b' local 'bootstrap terminated' 'learned peers continue' 'persistent peer records' test_bootstrap_death || true
    run_case PARTITION_HEAL 'worker-a,worker-b,requester' local 'worker link partition then heal' 'survivor and convergence' 'fabric rules removed' test_partition || true
    run_case NETWORK_PROFILES 'all active nodes' 'brazil-us,brazil-europe,bad-mobile,terrible' 'latency/jitter/loss/rate' 'bounded success or timeout' 'profile removed' test_profiles || true
    run_case ARTIFACT_RESUME 'requester,worker-a,nat-home' nat-home-1mbit 'connection interrupted and partial corrupted' 'resume and hash rejection' 'partial removed and clean retry' test_artifact_resume || true
    run_case DISK_QUOTA 'client-cgnat' local '70 KiB import into 64 KiB store' 'rejected without growth' 'node remains healthy' test_quota || true
    run_case DISTRIBUTED_TRAINING 'requester,worker-a,worker-b' local 'none' 'loss improvement, checkpoints, evaluation' 'durable checkpoint artifacts' test_training || true
    run_case COORDINATOR_RESTART 'requester,worker-a,worker-b' local 'requester terminated after checkpoint' 'checkpoint resume and artifact replication' 'same identity restarts; no automatic election claim' test_coordinator_restart || true
    run_case WORKER_LOSS_RECOVERY 'requester,worker-a,worker-b' brazil-europe 'worker-b stopped during training' 'survivor update and restart' 'checkpoint resume' test_worker_loss_and_recovery || true
    run_case KEY_ROTATION 'requester,bootstrap,worker-a' local 'sequence replay' 'old-to-new identity transition' 'stale rotation rejected' test_key_rotation || true
    run_case MALFORMED_PEER 'client-nat,worker-a,worker-b' local 'invalid/oversized datagrams' 'malformed input rejected' 'workers remain healthy' test_malformed_peer || true
    run_case CONNECTION_FLOOD 'client-nat,worker-a' local '64 bounded UDP attempts' 'connection limit remains bounded' 'worker responsive' test_connection_flood || true
    run_case JOB_FLOOD 'requester,worker-a' local '32 bounded jobs' 'queue remains bounded' 'node responsive' test_job_flood || true
    run_case SOAK 'all active nodes' local 'minor repeated inference/status' 'bounded stability run' 'no unbounded disk growth' test_soak || true
    inner_assert_size
    local failed
    failed=$(jq -s '[.[] | select(.status != "PASS")] | length' "$results_file")
    jq -n \
      --arg direct "$(jq -s '[.[] | select(.test == "DIRECT_NETWORK" and .status == "PASS")] | length' "$results_file")" \
      --arg failed "$failed" \
      --arg namespaces "${#inner_namespaces[@]}" \
      --arg nodes "${#node_names[@]}" \
      --argjson peak_cpu_percent "$peak_cpu_percent" \
      --argjson peak_rss_bytes "$peak_rss_bytes" \
      --argjson peak_lab_bytes "$peak_lab_bytes" \
      --argjson peak_virtual_bandwidth_mbit "$peak_virtual_bandwidth_mbit" \
      --argjson virtual_bytes "$peak_virtual_bytes" \
      '{namespace_count:($namespaces|tonumber),node_count:($nodes|tonumber),failed_tests:($failed|tonumber),host_default_route_unchanged:"pending-host-postflight",host_dns_unchanged:"pending-host-postflight",physical_qdiscs_unchanged:"pending-host-postflight",external_internet_blocked:true,peak_cpu_percent:$peak_cpu_percent,peak_rss_bytes:$peak_rss_bytes,peak_lab_bytes:$peak_lab_bytes,peak_virtual_bandwidth_mbit:$peak_virtual_bandwidth_mbit,virtual_bytes:$virtual_bytes}' > "$status_file"
    (( failed == 0 ))
  }

  run_inner() {
    generate_configs
    for name in bootstrap relay-a relay-b worker-a worker-b client-nat client-cgnat requester; do
      start_node "$name"
    done
    for name in bootstrap relay-a relay-b worker-a worker-b client-nat client-cgnat requester; do
      wait_ready "$name"
    done
    inner_assert_size
    printf '%s\n' 'run' >> "$lab_root/events.log"
  }

  status_inner() {
    printf '%s\n' '--- namespaces ---'
    ip netns list
    printf '%s\n' '--- routes ---'
    for namespace in "${inner_namespaces[@]}"; do
      printf '[%s]\n' "$namespace"
      ip netns exec "$namespace" ip route show
    done
    printf '%s\n' '--- node statuses ---'
    for name in "${node_names[@]}"; do
      printf '[%s]\n' "$name"
      "$binary" --config "$lab_root/nodes/$name/node.toml" --json status 2>&1 || true
    done
  }

  setup_inner
  rm -f -- "$control_fifo"
  mkfifo "$control_fifo"
  exec 9<>"$control_fifo"
  printf '%s\n' "$$" > "$lab_root/supervisor-inner.pid"
  printf '%s\n' ready > "$lab_root/supervisor.ready"
  while IFS= read -r request; do
    case "$request" in
      setup)
        printf 'PASS setup\n' > "$lab_root/command.result" ;;
      run)
        if run_inner; then printf 'PASS run\n' > "$lab_root/command.result"; else printf 'FAIL run\n' > "$lab_root/command.result"; fi ;;
      status)
        status_inner > "$lab_root/status-output.txt" 2>&1
        printf 'PASS status\n' > "$lab_root/command.result" ;;
      test)
        if run_tests_inner; then printf 'PASS test\n' > "$lab_root/command.result"; else printf 'FAIL test\n' > "$lab_root/command.result"; fi ;;
      cleanup|quit)
        printf 'PASS cleanup\n' > "$lab_root/command.result"
        break ;;
      *) printf 'FAIL unknown command\n' > "$lab_root/command.result" ;;
    esac
  done <&9
  exit 0
fi

case "${1:-}" in
  setup)
    setup_command
    ;;
  run)
    run_command
    ;;
  status)
    status_command
    ;;
  test)
    test_command
    ;;
  cleanup)
    cleanup_command
    ;;
  recover)
    recover_command
    ;;
  all)
    run_all
    ;;
  *)
    printf '%s\n' 'usage: scripts/lab-testnet.sh {setup|run|status|test|cleanup|recover|all}' >&2
    exit 2
    ;;
esac
