# Development and release checks

This document is for maintainers and contributors. It is separate from the
first-run installation path in the root README.

## Requirements

- Rust 1.85 or newer
- Linux for the complete process-isolation and network-laboratory checks
- `jq`, `ip`, `nft`, `tc`, `unshare`, `systemd-run`, `systemctl`, and related
  tools for the controlled lab

Build output belongs under `.cache/`:

```bash
export CARGO_TARGET_DIR=.cache/rust-target
export CARGO_INCREMENTAL=0
cargo build -p intelligence-cli
```

## Compatibility and process checks

```bash
./scripts/compatibility-smoke.sh
./scripts/local-testnet.sh
```

These use temporary local state and real node processes. They cover direct and
relay inference, evaluation, artifact transfer, restart, bootstrap loss,
relay loss with direct fallback, and distributed reference training.

## Controlled lab and emulators

```bash
./scripts/release.sh x86_64-unknown-linux-gnu .cache/release
INTELLIGENCE_BIN=.cache/release/intelligence-network-v1.0.0-linux-x86_64/intelligence \
  ./scripts/lab-testnet.sh all

./scripts/discovery-scale.sh
./scripts/discovery-attack-matrix.sh
./scripts/training-topology-scale.sh
./scripts/fabric-scale.sh
```

The lab uses private namespaces, bounded resources, real processes, and
failure injection. The scale scripts use deterministic logical peers and label
their output `EMULATED`; they do not launch one process per peer.

## Validation and release

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

The release workflow packages Linux x86_64 and aarch64 binaries and publishes
an archive checksum beside each artifact. The installer verifies that checksum
before writing to `~/.local/bin`.

Changes to peer-facing formats require bounds, malformed-input tests, and a
compatibility note. Long-running jobs must retain cancellation and recovery
behavior. Do not add central authorities, economic requirements, or hidden
cloud dependencies to solve a protocol problem.

See [CONTRIBUTING.md](../CONTRIBUTING.md), [Architecture](ARCHITECTURE.md),
and [Security posture](SECURITY.md).
