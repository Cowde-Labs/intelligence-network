# Fuzz

`fuzz_targets/decode_frame.rs` is the libFuzzer target for peer-controlled protocol frames.

When `cargo-fuzz` is installed, run it from the repository root with:

```bash
(cd fuzz && cargo fuzz run decode_frame)
```

The V2 record harness covers DHT RPC/records, signed evidence, training plans,
and checkpoint manifests:

```bash
(cd fuzz && cargo fuzz run decode_v2_records)
```

The V3 training harness covers training messages, generation/term records,
optimizer and model shard state, checkpoint commitments, and replicated-state
metadata:

```bash
(cd fuzz && cargo fuzz run decode_v3_training)
```

The V5 compute harness covers typed backend capabilities, task requirements,
assignments, challenges, evidence, and the V5 training frame:

```bash
(cd fuzz && cargo fuzz run decode_v5_compute)
```

The repository gate that does not require installing a large toolchain is
`cargo test -p intelligence-protocol --test fuzz_smoke`, which executes 20,000 deterministic
bounded cases for the frame, V2, and V3 decoders. The extended release script
sets `INTELLIGENCE_FUZZ_ROUNDS=100000`. These runs are not represented as a
libFuzzer coverage claim.
