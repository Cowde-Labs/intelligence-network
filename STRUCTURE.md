# Repository Structure

[English](STRUCTURE.md) | [Português (Brasil)](STRUCTURE.pt-BR.md) | [简体中文](STRUCTURE.zh-CN.md) | [Español](STRUCTURE.es.md)

This repository contains the executable Rust release and the public architecture/security
documentation. Development and research notes are retained locally under `internal/` but are
excluded from the public Git surface.

## Rust crates

- `crates/node`: process startup, configuration, lifecycle, local admin socket, job persistence,
  distributed training, and frontier training runtime.
- `crates/protocol`: versioned wire types, capability descriptions, jobs, artifacts, validation, and codecs.
- `crates/network`: identity, QUIC transport, signed discovery, bounded gossip, peer tables, and reconnect.
- `crates/runtime`: bounded execution, admission, cancellation, process limits, and optional bubblewrap.
- `crates/intelligence`: inference/evaluation contracts, evidence-aware routing, manifests,
  capability graph, compute planning, and training-fabric algorithms.
- `crates/storage`: local state, content-addressed blobs, integrity quarantine, and quota enforcement.
- `crates/cli`: operator-facing node commands and reproducible local operation.
- `crates/emulator`: deterministic DHT, trust, and training-topology experiments; simulated results do not represent real network execution.

## Public documentation

- `README.md`: installation, operation, architecture boundaries, and release checks.
- `docs/ARCHITECTURE.md`: crate responsibilities and decentralization boundaries.
- `docs/SECURITY.md`: threat assumptions, defenses, and operator limits.

## Specifications

The internal Markdown tree remains the normative long-term design source. `internal/IMPLEMENTATION_MATRIX.md`
records which requirements are implemented, experimental, deferred, or research for this release.

## Open architectural work

- Unify local hardware detection and contribution budgets without exposing sensitive details.
- Integrate adaptive budgets with runtime limits and job scheduling.
- Extend planners to account for communication cost and observed reliability while preserving versioned wire compatibility.
- Expand real validation across backends, heterogeneous hardware, and mobile platforms.
