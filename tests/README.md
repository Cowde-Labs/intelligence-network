# Tests

Executable tests live beside their crates so each boundary can be checked narrowly. The important
scenarios are in `crates/network/tests/transport.rs`, `crates/node/tests/end_to_end.rs`, and
`crates/protocol/tests/fuzz_smoke.rs`. Run the three-process operator scenario with
`../scripts/compatibility-smoke.sh` from the repository root.
