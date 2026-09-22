# Security posture

The 1.0 release treats every remote peer as untrusted input. This is a
bounded, local defense profile rather than a claim of universal Byzantine or
global Sybil resistance.

## Protected paths

- Ed25519 signatures bind peer identities and signed records.
- Protocol and persistent records validate ownership, sequence, generation,
  branch, expiry, size, and lineage before use.
- Content hashes, manifests, and generation checks prevent corrupt or stale
  artifacts, checkpoints, optimizer state, and shards from becoming active.
- Local evidence distinguishes direct observation from third-party reports;
  endorsement influence, evidence storage, and quarantine are bounded.
- DHT routing uses source diversity, protected contacts, replacement entries,
  bounded provider sets, multi-path lookup, and exploration of new peers.
- Capability claims are separate from observed and challenge-verified
  capability evidence.
- Training jobs select a local aggregation policy and can use bounded robust
  aggregation through the existing hierarchical topology.
- Duplicate contributions, stale state, replayed messages, and provable
  equivocation are rejected or recorded as local security events.
- Remote work is bounded by message size, queue, concurrency, memory, rate,
  and timeout limits.

## Operator responsibilities

Keep identity files private, use an operator-owned data directory, review
capabilities before exposing them, and isolate high-risk workloads. A relay,
bootstrap peer, cloud service, token, blockchain, staking system, or global
trust service is not required by the protocol.

## Limits

The local evidence model cannot prove one human per identity. Results depend on
the tested attacker fraction, topology, churn, resource limits, and local
policy. Public-Internet botnets, geographic diversity, unknown implementation
flaws, and independent-operator behavior require external validation.

Report vulnerabilities privately to the repository maintainers before
publishing an exploit or sensitive material.
