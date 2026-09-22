use intelligence_protocol::{
    BackendAssignment, BackendCapabilities, CapabilityChallenge, CapabilityChallengeResult,
    CapabilityEvidenceRecord, CheckpointManifest, ComputeRequirements, DhtRecord, DhtRequest,
    DhtResponse, MAX_FRAME_SIZE, SignedEvidence, TrainingAck, TrainingAggregate,
    TrainingCheckpointCommit, TrainingCheckpointManifest, TrainingCheckpointOffer,
    TrainingElection, TrainingMessage, TrainingPlan, TrainingShardAssignment, TrainingShardState,
    TrainingShardSummary, TrainingStart, TrainingState, TrainingUpdate, TrainingV4Message,
    TrainingV5Message, TrainingWindow, decode_frame,
};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn arbitrary_bounded_frames_never_panic_the_decoder() {
    let mut state = 0x8f31_7a2du64;
    let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20_000)
        .min(1_000_000);
    for round in 0..rounds {
        let length = (next(&mut state) as usize % 2048).saturating_add((round % 3) as usize);
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut state) as u8;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = decode_frame(&bytes, MAX_FRAME_SIZE);
        }));
        assert!(result.is_ok(), "decoder panicked for {} bytes", bytes.len());
    }
}

#[test]
fn a_hostile_header_is_rejected_before_payload_allocation() {
    let mut header = [0u8; 16];
    header[..4].copy_from_slice(b"INP1");
    header[4..6].copy_from_slice(&1u16.to_le_bytes());
    header[6..8].copy_from_slice(&0u16.to_le_bytes());
    header[8..10].copy_from_slice(&9u16.to_le_bytes());
    header[12..16].copy_from_slice(&(u32::MAX).to_le_bytes());
    let result = catch_unwind(AssertUnwindSafe(|| {
        let _ = decode_frame(&header, MAX_FRAME_SIZE);
    }));
    assert!(result.is_ok());
}

#[test]
fn v2_record_decoders_never_panic_on_bounded_inputs() {
    let mut state = 0x4d5f_22a1u64;
    let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20_000)
        .min(1_000_000);
    for round in 0..rounds {
        let length = (next(&mut state) as usize % 4096).saturating_add((round % 5) as usize);
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut state) as u8;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = postcard::from_bytes::<DhtRecord>(&bytes);
            let _ = postcard::from_bytes::<DhtRequest>(&bytes);
            let _ = postcard::from_bytes::<DhtResponse>(&bytes);
            let _ = postcard::from_bytes::<SignedEvidence>(&bytes);
            let _ = postcard::from_bytes::<TrainingPlan>(&bytes);
            let _ = postcard::from_bytes::<CheckpointManifest>(&bytes);
        }));
        assert!(result.is_ok(), "V2 decoder panicked for {length} bytes");
    }
}

#[test]
fn v3_training_decoders_never_panic_on_bounded_inputs() {
    let mut state = 0x2a91_77e4u64;
    let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20_000)
        .min(1_000_000);
    for round in 0..rounds {
        let length = (next(&mut state) as usize % 8192).saturating_add((round % 7) as usize);
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut state) as u8;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = postcard::from_bytes::<TrainingMessage>(&bytes);
            let _ = postcard::from_bytes::<TrainingStart>(&bytes);
            let _ = postcard::from_bytes::<TrainingWindow>(&bytes);
            let _ = postcard::from_bytes::<TrainingUpdate>(&bytes);
            let _ = postcard::from_bytes::<TrainingAggregate>(&bytes);
            let _ = postcard::from_bytes::<TrainingShardAssignment>(&bytes);
            let _ = postcard::from_bytes::<TrainingShardState>(&bytes);
            let _ = postcard::from_bytes::<TrainingShardSummary>(&bytes);
            let _ = postcard::from_bytes::<TrainingState>(&bytes);
            let _ = postcard::from_bytes::<TrainingAck>(&bytes);
            let _ = postcard::from_bytes::<TrainingElection>(&bytes);
            let _ = postcard::from_bytes::<TrainingCheckpointOffer>(&bytes);
            let _ = postcard::from_bytes::<TrainingCheckpointManifest>(&bytes);
            let _ = postcard::from_bytes::<TrainingCheckpointCommit>(&bytes);
        }));
        assert!(result.is_ok(), "V3 decoder panicked for {length} bytes");
    }
}

#[test]
fn v4_training_decoders_never_panic_on_bounded_inputs() {
    let mut state = 0x71c4_22e9u64;
    let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20_000)
        .min(1_000_000);
    for round in 0..rounds {
        let length = (next(&mut state) as usize % 16_384).saturating_add((round % 11) as usize);
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut state) as u8;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = postcard::from_bytes::<TrainingV4Message>(&bytes);
            let _ = decode_frame(&bytes, MAX_FRAME_SIZE);
        }));
        assert!(result.is_ok(), "V4 decoder panicked for {length} bytes");
    }
}

#[test]
fn v5_compute_decoders_never_panic_on_bounded_inputs() {
    let mut state = 0x95b2_4ac1u64;
    let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(20_000)
        .min(1_000_000);
    for round in 0..rounds {
        let length = (next(&mut state) as usize % 16_384).saturating_add((round % 13) as usize);
        let mut bytes = vec![0u8; length];
        for byte in &mut bytes {
            *byte = next(&mut state) as u8;
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = postcard::from_bytes::<BackendCapabilities>(&bytes);
            let _ = postcard::from_bytes::<ComputeRequirements>(&bytes);
            let _ = postcard::from_bytes::<BackendAssignment>(&bytes);
            let _ = postcard::from_bytes::<CapabilityChallenge>(&bytes);
            let _ = postcard::from_bytes::<CapabilityChallengeResult>(&bytes);
            let _ = postcard::from_bytes::<CapabilityEvidenceRecord>(&bytes);
            let _ = postcard::from_bytes::<TrainingV5Message>(&bytes);
        }));
        assert!(result.is_ok(), "V5 decoder panicked for {length} bytes");
    }
}

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
