# Infrastructure

Optional packaging examples. First-party infrastructure is never required by the protocol.

- `systemd/intelligence-node.service` runs the native binary as a restricted service account.
- `../scripts/compatibility-smoke.sh` launches three independent local processes, then verifies inference after
  the bootstrap process is shut down.

The native binary remains the primary deployment artifact. No Kubernetes, Redis, PostgreSQL, cloud
service, or first-party API gateway is required.
