#![no_main]

use intelligence_protocol::{
    TrainingAck, TrainingAggregate, TrainingCheckpointCommit, TrainingCheckpointManifest,
    TrainingCheckpointOffer, TrainingElection, TrainingMessage, TrainingShardAssignment,
    TrainingShardState, TrainingShardSummary, TrainingStart, TrainingState, TrainingUpdate,
    TrainingWindow,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // V3 training records are all peer-controlled. Decode each bounded family
    // independently so malformed nested vectors and generation metadata are
    // exercised without opening a socket or executing a job.
    let _ = postcard::from_bytes::<TrainingMessage>(data);
    let _ = postcard::from_bytes::<TrainingStart>(data);
    let _ = postcard::from_bytes::<TrainingWindow>(data);
    let _ = postcard::from_bytes::<TrainingUpdate>(data);
    let _ = postcard::from_bytes::<TrainingAggregate>(data);
    let _ = postcard::from_bytes::<TrainingShardAssignment>(data);
    let _ = postcard::from_bytes::<TrainingShardState>(data);
    let _ = postcard::from_bytes::<TrainingShardSummary>(data);
    let _ = postcard::from_bytes::<TrainingState>(data);
    let _ = postcard::from_bytes::<TrainingAck>(data);
    let _ = postcard::from_bytes::<TrainingElection>(data);
    let _ = postcard::from_bytes::<TrainingCheckpointOffer>(data);
    let _ = postcard::from_bytes::<TrainingCheckpointManifest>(data);
    let _ = postcard::from_bytes::<TrainingCheckpointCommit>(data);
});
