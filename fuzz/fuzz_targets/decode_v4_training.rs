#![no_main]

use intelligence_protocol::{
    TrainingV4Message, V4ByzantineResult, V4ByzantineUpdate, V4CollectiveAggregate,
    V4CollectiveContribute, V4CollectiveResult, V4PipelineBackward, V4PipelineBackwardResult,
    V4PipelineForward, V4PipelineForwardResult, V4PipelineInstall, V4PipelineInstallAck,
    V4ReconcileRequest, V4ReconcileResult, V4ShardMigration, V4ShardMigrationAck,
    V4TensorBackward, V4TensorBackwardResult, V4TensorForward, V4TensorForwardResult,
    V4TensorInstall, V4TensorInstallAck, V4TrainingBranch, V4TrainingPlan,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // V4 records contain peer-controlled generations, dimensions, ownership
    // lists and bounded vectors.  Decode each family independently; no
    // runtime, socket, filesystem or process execution is involved here.
    let _ = postcard::from_bytes::<TrainingV4Message>(data);
    let _ = postcard::from_bytes::<V4TrainingPlan>(data);
    let _ = postcard::from_bytes::<V4ShardMigration>(data);
    let _ = postcard::from_bytes::<V4ShardMigrationAck>(data);
    let _ = postcard::from_bytes::<V4TensorInstall>(data);
    let _ = postcard::from_bytes::<V4TensorInstallAck>(data);
    let _ = postcard::from_bytes::<V4TensorForward>(data);
    let _ = postcard::from_bytes::<V4TensorForwardResult>(data);
    let _ = postcard::from_bytes::<V4TensorBackward>(data);
    let _ = postcard::from_bytes::<V4TensorBackwardResult>(data);
    let _ = postcard::from_bytes::<V4PipelineInstall>(data);
    let _ = postcard::from_bytes::<V4PipelineInstallAck>(data);
    let _ = postcard::from_bytes::<V4PipelineForward>(data);
    let _ = postcard::from_bytes::<V4PipelineForwardResult>(data);
    let _ = postcard::from_bytes::<V4PipelineBackward>(data);
    let _ = postcard::from_bytes::<V4PipelineBackwardResult>(data);
    let _ = postcard::from_bytes::<V4CollectiveContribute>(data);
    let _ = postcard::from_bytes::<V4CollectiveAggregate>(data);
    let _ = postcard::from_bytes::<V4CollectiveResult>(data);
    let _ = postcard::from_bytes::<V4TrainingBranch>(data);
    let _ = postcard::from_bytes::<V4ReconcileRequest>(data);
    let _ = postcard::from_bytes::<V4ReconcileResult>(data);
    let _ = postcard::from_bytes::<V4ByzantineUpdate>(data);
    let _ = postcard::from_bytes::<V4ByzantineResult>(data);
});
