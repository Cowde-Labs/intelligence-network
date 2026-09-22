# Intelligence Network

Intelligence Network is a Rust peer-to-peer fabric for locally sovereign AI
capabilities. Each node owns its identity, hardware, models, data, storage,
and execution policy. Nodes exchange authenticated records and route bounded
work directly or through optional operator-run relays.

The 1.0 release is a compact, executable reference implementation. It proves
the protocol and recovery behavior exercised by the repository's local tests;
it does not claim public-Internet scale, universal Sybil resistance, or
frontier-model performance.

## What is included

- Authenticated QUIC transport with persistent Ed25519 node identities.
- Signed peer discovery, bounded DHT routing, optional replaceable relays,
  and bootstrap peers that are not permanent authorities.
- Content-addressed model, dataset, artifact, optimizer-state, and checkpoint
  storage with size, hash, generation, and lineage validation.
- Local capability evidence, bounded trust decisions, capability challenges,
  and diverse routing/provider selection.
- CPU reference inference, evaluation, distributed training, checkpoint
  recovery, and heterogeneous backend planning.
- Hierarchical training groups with local aggregation and replaceable
  coordination roles; no global synchronous barrier is required.
- Replay, stale-state, duplicate-contribution, malformed-input, and signed
  equivocation defenses.
- Deterministic controlled-lab, adversarial, protocol-fuzz, and logical-peer
  emulator harnesses.

The strongest security statements are local and measured: see
[Security posture](docs/SECURITY.md). The architecture and compatibility
boundaries are described in [Architecture](docs/ARCHITECTURE.md).

## Requirements

- Rust 1.85 or newer
- Linux for the complete process-isolation and network-laboratory checks
- `jq`, `ip`, `nft`, `tc`, `unshare`, `systemd-run`, `systemctl`, and related
  tools only for the controlled-lab harness

No cloud service, token, blockchain, staking system, mandatory relay, or
central scheduler is required.

## Build and run a node

The repository stores build output under `.cache/`, which is ignored by Git.

```bash
export CARGO_TARGET_DIR=.cache/rust-target
export CARGO_INCREMENTAL=0
cargo build -p intelligence-cli

./.cache/rust-target/debug/intelligence --help
./.cache/rust-target/debug/intelligence --config /tmp/intelligence-node.toml init
./.cache/rust-target/debug/intelligence --config /tmp/intelligence-node.toml run
```

Copy [config/node.toml.example](config/node.toml.example), replace the example
addresses, and choose the capabilities an operator is willing to expose.
Model weights are operator-supplied; the node does not download them.

Useful commands against a running node include:

```bash
intelligence --config /tmp/intelligence-node.toml identity
intelligence --config /tmp/intelligence-node.toml status
intelligence --config /tmp/intelligence-node.toml peers
intelligence --config /tmp/intelligence-node.toml capabilities
intelligence --config /tmp/intelligence-node.toml infer \
  --capability inference.text --input "good and useful"
```

The CLI emits JSON where a command returns structured state. `version` reports
the software version, protocol version, commit, target architecture, and
operating system.

## Local validation

Run the small local process checks first:

```bash
./scripts/compatibility-smoke.sh
./scripts/local-testnet.sh
```

The local harnesses create all identities and state in temporary directories.
They cover direct and relay inference, evaluation, artifact transfer, restart,
bootstrap loss, relay loss with direct fallback, and reference training.

For the controlled Linux laboratory, build a release binary and run the
isolated namespace test:

```bash
./scripts/release.sh x86_64-unknown-linux-gnu .cache/release
INTELLIGENCE_BIN=.cache/release/intelligence-network-v1.0.0-linux-x86_64/intelligence \
  ./scripts/lab-testnet.sh all
```

The lab uses private namespaces, bounded resources, real node processes,
failure injection, and host preflight/postflight checks. It is local evidence,
not evidence of public Internet reachability or independent operators.

The deterministic logical-peer and training-topology experiments are separate
from real-process evidence:

```bash
./scripts/discovery-scale.sh
./scripts/discovery-attack-matrix.sh
./scripts/training-topology-scale.sh
./scripts/fabric-scale.sh
```

Their output is explicitly `EMULATED`; no script starts one process per logical
peer. Set the corresponding scale environment variable documented in each
script only when the host has enough memory and disk.

## Architecture

```text
protocol  -> storage, runtime
network   -> protocol, storage
intelligence -> protocol, runtime, storage
node      -> protocol, network, runtime, storage, intelligence
cli       -> node
```

The protocol crate owns wire types and bounded validation. The network crate
owns authenticated transport and discovery. Storage owns local state and
content-addressed blobs. Runtime owns admission, deadlines, cancellation,
and execution limits. Intelligence owns local evidence, capability planning,
evaluation, and training algorithms. Node composes those boundaries, and the
CLI is the operator surface.

Training coordination is distributed across groups, shard owners, replicas,
and replaceable coordination roles. A role is not a permanent authority:

```text
all updates -> one coordinator       NO
one optimizer-state authority        NO
one checkpoint authority             NO
global synchronous barrier           NO
model must fit on one worker         NO
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the compatibility and
decentralization boundaries.

## Protocol and state compatibility

The current wire protocol is `1.6`. Persistent state format compatibility is
managed by the storage crate. Existing additive V2–V6 records remain
versioned where their names are part of the wire or state contract; source
modules use responsibility-based filenames where the filename itself is not a
compatibility surface.

The public release keeps the CPU reference path available by default. CUDA,
ROCm, and Metal backend feature flags preserve the heterogeneous compute
contracts; physical accelerator availability remains environment-dependent.

## Security boundaries

Remote input is untrusted. Signatures authenticate issuers, while hashes,
ownership checks, sequence/generation checks, expiry, and lineage checks decide
whether state may be used. Trust is local and bounded; there is no global
reputation oracle. New honest peers remain eligible for bounded exploration,
while critical roles require stronger local evidence.

This release does not claim one human per identity, universal Byzantine fault
tolerance, public botnet resistance, or perfect evaluator independence. Read
[docs/SECURITY.md](docs/SECURITY.md) before exposing a node to untrusted
operators.

## Release checks

The release checks are intentionally reproducible:

```bash
cargo fmt --all -- --check
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
cargo audit
cargo deny check
./scripts/banned-architecture-check.sh
./scripts/protocol-fuzz-extended.sh
./scripts/release.sh x86_64-unknown-linux-gnu .cache/release
```

The CI workflow runs the same core checks on every push and pull request. The
security workflow runs dependency policy, RustSec, and decoder checks. The
release workflow packages Linux x86_64 and aarch64 artifacts when the
corresponding runner is available.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes to peer-facing formats require
bounds, malformed-input tests, and a compatibility note. Long-running jobs
must retain cancellation and recovery behavior. Do not add central authorities
or economic requirements to solve a protocol problem.

## License

Apache-2.0. See [LICENSE](LICENSE).
