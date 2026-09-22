#![no_main]

use intelligence_protocol::{
    BackendCapabilities, BackendAssignment, CapabilityChallenge, CapabilityChallengeResult,
    CapabilityEvidenceRecord, ComputeRequirements, TrainingV5Message,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // V5 records are bounded capability/task metadata.  Decode each record
    // independently; native runtimes are never entered by this harness.
    let _ = postcard::from_bytes::<BackendCapabilities>(data);
    let _ = postcard::from_bytes::<ComputeRequirements>(data);
    let _ = postcard::from_bytes::<BackendAssignment>(data);
    let _ = postcard::from_bytes::<CapabilityChallenge>(data);
    let _ = postcard::from_bytes::<CapabilityChallengeResult>(data);
    let _ = postcard::from_bytes::<CapabilityEvidenceRecord>(data);
    let _ = postcard::from_bytes::<TrainingV5Message>(data);
});
