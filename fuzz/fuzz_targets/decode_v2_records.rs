#![no_main]

use intelligence_protocol::{
    CheckpointManifest, DhtRecord, DhtRequest, DhtResponse, SignedEvidence, TrainingPlan,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // These are independent bounded decoders. The target intentionally
    // exercises the V2 record families without opening a socket or executing
    // a job.
    let _ = postcard::from_bytes::<DhtRecord>(data);
    let _ = postcard::from_bytes::<DhtRequest>(data);
    let _ = postcard::from_bytes::<DhtResponse>(data);
    let _ = postcard::from_bytes::<SignedEvidence>(data);
    let _ = postcard::from_bytes::<TrainingPlan>(data);
    let _ = postcard::from_bytes::<CheckpointManifest>(data);
});
