# Architecture

Intelligence Network is a peer-to-peer Rust system for locally sovereign AI
capabilities. A node owns its identity, hardware, models, data, storage, and
execution policy. Peers exchange authenticated records and route bounded work
directly or through optional operator-run relays.

## Runtime boundaries

```text
protocol  -> storage, runtime
network   -> protocol, storage
intelligence -> protocol, runtime, storage
node      -> protocol, network, runtime, storage, intelligence
cli       -> node
```

- `protocol` defines the versioned wire types, bounded validation, artifact and
  checkpoint manifests, capability records, and training messages.
- `network` provides authenticated QUIC transport, Ed25519 identity binding,
  signed discovery, bounded gossip, routing, and optional relays.
- `storage` owns local state and content-addressed blobs with atomic writes,
  integrity checks, quarantine, and quotas.
- `runtime` provides bounded admission, deadlines, cancellation, deterministic
  CPU reference execution, and operator-configured local processes.
- `intelligence` contains capability evidence, local trust decisions, compute
  planning, evaluation, and distributed-training reference algorithms.
- `node` composes the services, persists jobs, and exposes the local admin
  socket.
- `cli` is the operator-facing interface.

## Decentralization boundaries

Bootstrap peers only help a node make initial contact. They do not own
membership or job state. Relays forward authenticated envelopes and are
replaceable; they do not become identity, capability, artifact, or training
authorities.

Training state is distributed across groups, shard owners, replicas, and
replaceable coordination roles. The runtime does not require all updates to
flow through one coordinator, a global synchronous barrier, or a model that
fits on one worker.

Trust is local and evidence-based. Nodes do not publish or consume a global
reputation score. New peers remain discoverable through bounded exploration,
while critical roles require stronger local observations and valid state
lineage.

## Compatibility

The current wire protocol is `1.6`. Persistent state format compatibility is
managed by the storage crate. Additive V2–V6 records remain versioned where
their names are part of the wire or state contract; source modules use
responsibility-based names where the filename itself is not a compatibility
surface.
