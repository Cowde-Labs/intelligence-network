# Intelligence Network

Run AI across independent machines without one central server controlling the network.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
intelligence up
```

Intelligence Network is a single Rust binary that turns a machine into a node.
Nodes find each other over authenticated QUIC, advertise what they can run,
route inference and training jobs to peers that can do the work, shard state
across machines, and keep going when peers disappear. There is no account to
create, no cloud to log into, and nothing to download except the binary.
Models never move unless you put them there.

It is not a crypto project. There is no token, no blockchain, no staking and no
marketplace. Nodes cooperate because their operators point them at each other.

## Quickstart

Linux x86_64 or aarch64. The installer fetches the release binary and its
checksum from GitHub Releases, verifies the checksum, and puts one file in
`~/.local/bin`. Nothing else is installed.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

intelligence up              # create an identity, write a config, start the node in the background
intelligence status          # health, peers, jobs, resource counters
intelligence infer "hello"   # route a job through the node
intelligence down            # stop it
```

`up` is idempotent. Run it again and it reports the node that is already
running. Config lands in `~/.config/intelligence/config.toml`, state and logs
in `~/.local/state/intelligence`, and models are looked for in
`~/.local/share/intelligence/models` (`XDG_*_HOME` variables are honoured if
set).

Out of the box the node exposes three small built-in capabilities that run on
any CPU: `inference.text`, `evaluation.text` and `training.reference`. The
first `infer` you run hits a tiny built-in sentiment classifier, so you get a
label back, not a chat reply. That is deliberate: the default install has to
work on a Raspberry Pi with no model files, and it proves the routing, identity
and job machinery end to end before you plug in a real model. See
[Models](#models) for the llama.cpp path.

To try the peer-to-peer part on one laptop, start a second node with its own
config and point it at the first:

```bash
intelligence up
intelligence --config /tmp/node-b.toml up --peer 127.0.0.1:4000 --no-default-seeds
intelligence --config /tmp/node-b.toml network peers
intelligence --config /tmp/node-b.toml infer "this is great"
```

Node B discovers node A, learns its capabilities, and routes the job to
whichever peer has the best local evidence for it. Stop node A and node B keeps
serving what it can on its own.

Something off? `intelligence doctor` checks config, identity, storage
permissions, the QUIC endpoint, NAT reachability, detected hardware, local
model files and node health, and tells you what to fix.

## What you can do

- **Discover peers without a directory service.** Bootstrap addresses are
  hints, not authorities. Once a node has met one peer it learns about others
  through signed peer exchange and a bounded Kademlia-style DHT, and remembers
  them across restarts.
- **Run inference across machines.** Submit a job locally; the node routes it
  to itself or to a peer that advertises the capability and has earned local
  trust for it. Results stream back with size and deadline bounds.
- **Serve a real model to the network.** Point a `llama_cpp` capability at a
  `llama-cli` binary and a `.gguf` file you already have. Remote jobs run it
  inside bubblewrap.
- **Train across machines.** Reference training jobs run on real worker
  processes with genuine parameter sharding, replicated optimizer state,
  content-addressed checkpoints, hierarchical aggregation, and tensor- and
  pipeline-parallel reference operations. No single coordinator sees every
  update, and no worker needs the whole model.
- **Survive failure.** Kill the bootstrap peer, the coordinator, a shard owner,
  a relay, or a pipeline stage mid-job. Reconnect, backoff, term-based
  coordinator replacement, replica promotion and checkpoint resume are all
  exercised by the test suite.
- **Mix hardware.** Nodes advertise their compute backends (CPU today; CUDA,
  ROCm and Metal when built with those features), formats and memory, and the
  planner uses that to place shards and pick strategies.
- **Move artifacts safely.** Models, checkpoints and datasets are
  content-addressed with BLAKE3, transferred in resumable chunks, verified on
  arrival, and quarantined if they fail integrity or quota checks.
- **Stay in control.** Every capability is opt-in, bounded, and either local
  only or public. Nothing is fetched implicitly. A node is fully functional
  with zero peers.

## How it works, briefly

Each node has a persistent Ed25519 identity. Peers talk over QUIC and bind the
transport to that identity, so every record a node receives is signed by the
peer that produced it: addresses, capability advertisements, DHT records,
training updates, checkpoints.

Capabilities are claims. Trust is local and evidence-based: a node tracks what
it has directly observed from each peer, separately from what others have told
it, and only routes work to peers whose evidence clears its own policy. There
is no global reputation score to game.

Work is a job with explicit bounds on input size, output size, memory, CPU and
deadline. The runtime admits jobs into a bounded queue, runs them with
cancellation, and persists their state so a restart does not lose them.

Distributed training splits model state into shards owned by workers, groups
workers under aggregators so the coordinator only sees aggregates, replicates
optimizer state and checkpoints across peers, and rotates coordination roles
using monotonically increasing terms when a role holder disappears.

```text
all updates -> one coordinator       NO
one optimizer-state authority        NO
one checkpoint authority             NO
global synchronous barrier           NO
model must fit on one worker         NO
```

## Install

### Release binary

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
```

Pin a version with `INTELLIGENCE_VERSION=v1.0.0`, or mirror the artifacts and
set `INTELLIGENCE_RELEASE_BASE_URL`. The script refuses to install on a
checksum mismatch or an archive with unsafe paths.

### From source

Rust 1.85 or newer.

```bash
git clone https://github.com/Cowde-Labs/intelligence-network
cd intelligence-network
cargo build --release -p intelligence-cli
./target/release/intelligence up
```

Enable accelerator detection with `--features cuda`, `--features rocm` or
`--features metal`. This makes the node probe for the driver and advertise the
backend; it does not pull in any vendor SDK.

### As a service

```bash
intelligence service install    # per-user systemd unit, no root
```

For a system-wide deployment under a restricted service account see
[`infra/systemd`](infra/systemd).

## Run

`intelligence up` starts the node in the background and returns when it is
healthy. `intelligence run` is the same node in the foreground, useful under a
supervisor or while debugging.

```bash
intelligence up --peer 203.0.113.10:4000 --peer 198.51.100.20:4000
intelligence up --no-default-seeds                 # only the peers you name
intelligence up --listen-addr 0.0.0.0:4000         # accept connections from other machines
intelligence up --capability inference.text        # enable a subset of the built-ins
intelligence --config ./node.toml up               # a second node, or a non-standard location
```

The default listen address is `127.0.0.1:4000`. To be reachable from another
machine you have to bind a routable address and, if you want the node to tell
peers about it, set `advertise_addr` in the config. A node bound to a
loopback address accepts private peer addresses automatically; a node bound to
anything else filters them out of peer records unless you set
`allow_private_addresses = true`, which is what you want on a LAN.

The compiled default seeds are `bootstrap-1.intelligence.network:4000` and
`bootstrap-2.intelligence.network:4000`. They are replaceable hints: if they
are down the node says so and carries on, and `--no-default-seeds` or
`INTELLIGENCE_DEFAULT_SEEDS=host:port,host:port` swaps them out entirely.

## Models

The node never downloads weights. You bring the file, the node hashes it and
tracks it as a content-addressed artifact.

```bash
intelligence model list                                     # list files in the standard model dirs
intelligence model add ~/models/llama-3.2-1b-q4.gguf        # import and hash (needs a running node)
intelligence model add ./model.bin --identity local.mymodel.v1 --format opaque
intelligence artifact inspect --artifact <hash>
intelligence artifact fetch --peer <node-id> --artifact <hash>   # pull and verify from a peer
```

To actually run a model, add a `llama_cpp` capability to your config. You
supply the `llama-cli` binary and the model path; the node supplies the
sandbox, bounds and routing:

```toml
[[capabilities]]
name = "inference.llama-cpp"
version = 1
public = true
accept_remote_jobs = true
kind = "llama_cpp"
program = "/usr/local/bin/llama-cli"
model_path = "/srv/models/llama-3.2-1b-q4.gguf"
args = ["--ctx-size", "2048"]
sandbox = "bubblewrap"
max_input_bytes = 65536
max_output_bytes = 65536
memory_bytes = 4294967296
cpu_millis = 30000
```

Then:

```bash
intelligence infer --capability inference.llama-cpp "Explain QUIC in one paragraph"
intelligence infer --capability inference.llama-cpp --local-only "..."   # never leave this machine
```

A `llama_cpp` or `process` capability that is `public` and accepts remote jobs
must use `sandbox = "bubblewrap"`; the node rejects the config otherwise. Set
`public = false` for a model you only want to use yourself. The llama.cpp
adapter is marked experimental: the process boundary, limits and config
validation are tested, but it has not been exercised against a wide range of
models or builds.

## CLI reference

Global flags: `--config <path>` (or `INTELLIGENCE_CONFIG`) and `--json` for
machine-readable output on every command that has any.

### Lifecycle

| Command | What it does |
|---|---|
| `up [--peer …] [--no-default-seeds] [--capability …] [--listen-addr …]` | Create config and identity if missing, start the node in the background |
| `down` | Gracefully stop the local node |
| `run` | Start the node in the foreground |
| `init` | Write a config and identity without starting anything |
| `service install` | Install a per-user systemd unit |
| `version` | Software and protocol versions |

### Inspect

| Command | What it does |
|---|---|
| `status` | Health, peers, jobs, resource counters |
| `doctor` | Check networking, storage, hardware, models and node health |
| `config` | Effective configuration after env and defaults |
| `jobs` | Persisted and active jobs |
| `identity` | This node's ID and public key |
| `network peers` | Currently known peers |
| `network capabilities` | Capabilities this node advertises |
| `network stats` | Routing table and record statistics |
| `trust inspect --subject <node-id>` | The local evidence-based trust decision for a peer |

### Work

| Command | What it does |
|---|---|
| `infer [TEXT] [--capability …] [--input …] [--deadline-ms …] [--max-output-bytes …] [--local-only] [--job-id …]` | Submit an inference job |
| `evaluate --text … --expected-label … [--deadline-ms …]` | Evaluate a sample through the network and record evidence |
| `jobs cancel --job-id <id>` | Cancel a running job |

### Models and artifacts

| Command | What it does |
|---|---|
| `model list` | List model files present in the standard local directories |
| `model add <path> [--identity …] [--format …]` | Import and hash a local model |
| `model register --path … --identity … [--format …] [--local-only]` | Lower-level model registration |
| `artifact inspect --artifact <hash>` | Inspect a locally stored artifact |
| `artifact fetch --peer <node-id> --artifact <hash>` | Fetch and verify an artifact from a peer |

### Identity and DHT

| Command | What it does |
|---|---|
| `identity rotate --new-path … [--sequence …] [--valid-until …]` | Rotate the node identity with a signed transition |
| `network publish --namespace … --name … --value … [--ttl-seconds …] [--sequence …]` | Publish a signed record |
| `network lookup --namespace … --name …` | Look up a record |
| `network find --key …` | Find peers near a routing key |

### Training

| Command | What it does |
|---|---|
| `train [--mode sharded\|local-sgd] [--workers …] [--windows …] [--checkpoint-every …] [--local-steps …]` | Run a distributed training job and wait (`local-sgd` = synchronous rounds, `sharded` = sharded graph) |
| `train start …` | Same, but return immediately |
| `train plan --model-bytes … [--mode …] [--workers …] [--strategy …] [--tensor-degree …] [--pipeline-stages …] [--data-locality …]` | Plan a distributed training job over known peers |
| `train replan --job-id … --model-bytes … …` | Propose a new graph for a running job |
| `train activate --job-id …` | Activate an approved graph |
| `train status --job-id …` | State of a distributed training job |
| `train cancel --job-id …` | Cancel a running job |
| `train migrate --job-id … --shard-id … --target …` | Migrate a training shard to another peer |
| `train replicate --workers a,b` | Replicate a training state record to selected peers |
| `train reconcile --worker … --left-value … --right-value … [--policy …]` | Reconcile two compatible training branches |
| `train reference [--workers …] [--steps …] [--resume-checkpoint …]` | Run the small synchronous reference training job |

`intelligence <command> --help` has the full flag list. Pre-1.0 command names
still work as hidden aliases.

## Advanced configuration

`intelligence config` prints the effective configuration. Every field can be
set in TOML or overridden with an `INTELLIGENCE_*` environment variable of the
same name (`INTELLIGENCE_LISTEN_ADDR`, `INTELLIGENCE_BOOTSTRAP`,
`INTELLIGENCE_RELAY_ADDRESSES`, `INTELLIGENCE_STORAGE_QUOTA_BYTES`,
`INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS`, and so on). A fully commented
example is in [`config/node.toml.example`](config/node.toml.example).

Things you are likely to touch:

**Networking.** `listen_addr`, `advertise_addr`, `bootstrap`,
`allow_private_addresses`, `max_connections`, `peer_ttl_seconds`.
Hole punching is on by default (`hole_punch_enabled`,
`hole_punch_max_attempts`) and is a finite, authenticated candidate-dialing
path; it is not proven against hostile NATs.

**Relays.** Any node can be a relay. Set `relay_enabled = true` and cap it with
`relay_max_sessions` and `relay_max_bytes`. Clients list `relay_addresses` and
can `prefer_relay`. Relays forward opaque authenticated envelopes; they cannot
read jobs or become authorities for anything.

**DHT.** `dht_enabled`, `dht_k`, `dht_alpha`, `dht_max_records`. The DHT runs
over direct authenticated peers only; it does not use relays.

**Storage.** `[storage] quota_bytes` and `max_artifact_bytes`. Artifacts that
fail integrity checks are quarantined, not deleted.

**Runtime.** `[runtime] max_queued_jobs`, `max_concurrent_jobs`,
`max_input_bytes`, `max_output_bytes`, `default_timeout_ms`,
`process_memory_bytes`, `process_cpu_seconds`, `work_dir`.

**Capabilities.** Each `[[capabilities]]` entry has a `kind` of
`builtin_text`, `builtin_training`, `process` or `llama_cpp`, a `sandbox` of
`trusted_local` or `bubblewrap`, per-capability limits, and `public` /
`accept_remote_jobs` flags. `metadata` entries are copied into signed
advertisements and used for planning; they are claims, and peers weigh them
against observed evidence.

**Training.** `training_memory_bytes` is the local state budget a worker must
fit its assigned shard into. `training_window_delay_ms` injects straggler
delay for experiments; leave it at zero.

## Architecture

Seven Rust crates with one-directional dependencies:

```text
protocol      versioned wire types, bounded validation, manifests, training messages
storage       local state, content-addressed blobs, quarantine, quotas
runtime       admission, deadlines, cancellation, process limits, bubblewrap
network       Ed25519 identity, QUIC transport, signed discovery, gossip, DHT, relays
intelligence  capability evidence, local trust, compute planning, training algorithms
node          composes the above, persists jobs, exposes the local admin socket
cli           the intelligence binary
```

The wire protocol is versioned (currently `1.6`) and every peer-facing parser
is bounded and fuzzed. The CLI talks to a running node over a Unix admin
socket in the state directory; nothing listens on TCP.

Design rules the codebase holds itself to: no central control plane, no
mandatory first-party infrastructure, no economic layer, and no protocol
dependency whose disappearance takes the network down with it. A separate
deterministic emulator (`crates/emulator`) models discovery and training
topologies at 100 to 100,000 logical peers so scale behaviour can be measured
without launching that many processes. Its output is labelled `EMULATED`.

Longer treatment in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Security and limitations

Every remote peer is treated as untrusted input. Identities and records are
signed, artifacts are hash-verified before use, replays and equivocation are
rejected, and remote work is bounded by size, queue, concurrency, memory,
rate and timeout. Remote execution of an external process is only allowed
inside bubblewrap. Full posture in [docs/SECURITY.md](docs/SECURITY.md).

Be clear-eyed about where this is:

- **Validated on one machine and in a controlled lab, not on the public
  Internet.** The integration tests, the three-process smoke test and the
  network-namespace lab with NAT, relays, impairment and failure injection all
  pass. Multi-host deployments across independent operators, real hostile
  NATs, and public-scale behaviour have not been measured. Treat scale numbers
  from the emulator as emulated.
- **Training is real but reference-scale.** Sharding, replication, coordinator
  replacement, tensor/pipeline parallelism and branch reconciliation run on
  real worker processes, on bounded deterministic CPU workloads. This is not
  frontier model training and does not claim to be.
- **Trust is local, not global.** The evidence model cannot prove one human per
  identity and does not claim Sybil resistance beyond the tested attacker
  fractions and topologies. Nodes you point at each other should be operated
  by people you have some reason to trust.
- **Byzantine tolerance is bounded.** Median aggregation and update validation
  raise the cost of poisoning; they are not BFT.
- **Accelerators are advertised, not executed.** With the `cuda`/`rocm`/`metal`
  features the node detects and advertises the backend for planning. Actual
  GPU inference happens inside your llama.cpp process, not in the node.
- **Linux first.** Release binaries are Linux x86_64 and aarch64. Process
  isolation and `service install` depend on bubblewrap and systemd. Other
  platforms build from source without those guarantees.

Report vulnerabilities privately to the maintainers before publishing.

## Development

```bash
export CARGO_TARGET_DIR=.cache/rust-target
cargo build -p intelligence-cli
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

Integration scenarios that launch real node processes:

```bash
./scripts/compatibility-smoke.sh     # three nodes; inference survives bootstrap shutdown
./scripts/local-testnet.sh           # direct/relay inference, artifact transfer, restart, training
```

The controlled network lab, scale emulators, fuzzing and the release pipeline
are documented in [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Contributions should make the protocol clearer, more interoperable or more
survivable under failure. Peer-facing parsers need bounds and malformed-input
tests; long-running job paths need cancellation and recovery; capability
claims need a path to evidence. Do not add tokens, blockchains, marketplaces,
or a central service without an ADR that changes the project's thesis. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

GNU Affero General Public License v3.0 only. See [LICENSE](LICENSE).
