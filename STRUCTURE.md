# Repository Structure

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

## Public documentation

- `README.md`: installation, operation, architecture boundaries, and release checks.
- `docs/ARCHITECTURE.md`: crate responsibilities and decentralization boundaries.
- `docs/SECURITY.md`: threat assumptions, defenses, and operator limits.

## Specifications

The internal Markdown tree remains the normative long-term design source. `internal/IMPLEMENTATION_MATRIX.md`
records which requirements are implemented, experimental, deferred, or research for this release.
