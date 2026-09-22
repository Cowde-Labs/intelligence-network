use crate::{
    AddressObservation, AddressRecord, AddressUpdate, ArtifactChunk, ArtifactManifest,
    ArtifactRequest, ArtifactTransferChunk, ArtifactTransferRequest, Capability,
    CheckpointManifest, DatasetManifest, DhtRecord, DhtRequest, DhtResponse, Hello, JobId,
    JobRequest, JobUpdate, KeyRotation, Message, ModelManifest, PeerExchange, PeerRecord,
    ProtocolError, RelayEnvelope, SignedAnnouncement, SignedEvidence, TrainingAck,
    TrainingCheckpointCommit, TrainingCheckpointOffer, TrainingElection, TrainingMessage,
    TrainingPlan, TrainingStart, TrainingState, TrainingUpdate, TrainingV4Message, TrainingWindow,
};
use thiserror::Error;

pub const MAX_FRAME_SIZE: usize = 1024 * 1024;
pub const MAX_STRING_LEN: usize = 256;
pub const MAX_ADDRESS_LEN: usize = 256;
pub const MAX_CAPABILITIES: usize = 64;
pub const MAX_PEERS: usize = 128;
pub const MAX_METADATA: usize = 32;
pub const MAX_JOB_INPUT: usize = 64 * 1024;
pub const MAX_JOB_OUTPUT: usize = 1024 * 1024;
pub const MAX_ARTIFACT_CHUNK: usize = 256 * 1024;
pub const MAX_ARTIFACT_SHARDS: usize = 4096;
pub const MAX_ADDRESSES: usize = 16;
pub const MAX_RELAY_PAYLOAD: usize = 512 * 1024;
pub const MAX_DHT_RECORD_VALUE: usize = 8 * 1024;
pub const MAX_DHT_RECORDS: usize = 64;
pub const MAX_DHT_CONTACTS: usize = 64;
pub const MAX_DHT_QUERY_MESSAGE: usize = 128;
pub const MAX_EVIDENCE_PAYLOAD: usize = 8 * 1024;
pub const MAX_TRAINING_WORKERS: usize = 64;
pub const MAX_TRAINING_GROUPS: usize = 32;
pub const MAX_TRAINING_SAMPLES: usize = 256;
pub const MAX_TRAINING_CONTRIBUTORS: usize = 64;
pub const MAX_TRAINING_SHARDS: usize = 64;
pub const MAX_TRAINING_REPLICAS: usize = 16;
pub const MAX_TRAINING_STATE_BYTES: u64 = 1 << 40;
pub const MAX_V4_VECTOR: usize = 128;
pub const MAX_V4_BYTES: usize = 64 * 1024;
pub const MAX_V4_STAGES: usize = 16;
pub const MAX_V4_GROUPS: usize = 32;
pub const MAX_V4_PLAN_WORKERS: usize = 64;
pub const MAX_V5_BACKENDS: usize = 8;
pub const MAX_V5_FEATURES: usize = 16;
pub const MAX_V5_FORMATS: usize = 8;
pub const MAX_V5_ASSIGNMENTS: usize = 128;
pub const MAX_V5_REQUIREMENTS: usize = 64;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ValidationError {
    #[error("{field} exceeds maximum length {max}")]
    TooLong { field: &'static str, max: usize },
    #[error("{field} exceeds maximum count {max}")]
    TooMany { field: &'static str, max: usize },
    #[error("{field} must not be empty")]
    Empty { field: &'static str },
    #[error("{field} has invalid value")]
    Invalid { field: &'static str },
    #[error("{field} exceeds the protocol frame limit")]
    Oversized { field: &'static str },
}

fn string(value: &str, field: &'static str) -> Result<(), ValidationError> {
    if value.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    if value.len() > MAX_STRING_LEN {
        return Err(ValidationError::TooLong {
            field,
            max: MAX_STRING_LEN,
        });
    }
    Ok(())
}

fn bytes(value: &[u8], max: usize, field: &'static str) -> Result<(), ValidationError> {
    if value.len() > max {
        return Err(ValidationError::Oversized { field });
    }
    Ok(())
}

fn capability(capability: &Capability) -> Result<(), ValidationError> {
    string(&capability.name, "capability.name")?;
    if capability.version == 0 {
        return Err(ValidationError::Invalid {
            field: "capability.version",
        });
    }
    if capability.name.len() > 128 {
        return Err(ValidationError::TooLong {
            field: "capability.name",
            max: 128,
        });
    }
    if let Some(model) = &capability.model {
        string(model, "capability.model")?;
    }
    if capability.metadata.len() > MAX_METADATA {
        return Err(ValidationError::TooMany {
            field: "capability.metadata",
            max: MAX_METADATA,
        });
    }
    for entry in &capability.metadata {
        string(&entry.key, "capability.metadata.key")?;
        string(&entry.value, "capability.metadata.value")?;
    }
    if capability.compute_backends.len() > MAX_V5_BACKENDS {
        return Err(ValidationError::TooMany {
            field: "capability.compute_backends",
            max: MAX_V5_BACKENDS,
        });
    }
    let mut backend_kinds = std::collections::HashSet::new();
    for backend in &capability.compute_backends {
        if !backend_kinds.insert(backend.kind) {
            return Err(ValidationError::Invalid {
                field: "capability.compute_backends",
            });
        }
        backend_capabilities(backend, "capability.compute_backend")?;
    }
    if capability.resources.max_input_bytes == 0
        || capability.resources.max_input_bytes as usize > MAX_JOB_INPUT
        || capability.resources.max_output_bytes == 0
        || capability.resources.max_output_bytes as usize > MAX_JOB_OUTPUT
        || capability.resources.memory_bytes == 0
        || capability.resources.cpu_millis == 0
    {
        return Err(ValidationError::Invalid {
            field: "capability.resources",
        });
    }
    match &capability.evidence {
        crate::CapabilityEvidence::Claimed | crate::CapabilityEvidence::Observed { .. } => {}
        crate::CapabilityEvidence::Verified { score_basis, .. } => {
            string(score_basis, "capability.evidence.score_basis")?;
        }
    }
    Ok(())
}

fn peer(peer: &PeerRecord) -> Result<(), ValidationError> {
    if peer.addresses.len() > 8 {
        return Err(ValidationError::TooMany {
            field: "peer.addresses",
            max: 8,
        });
    }
    for address in &peer.addresses {
        if address.is_empty() || address.len() > MAX_ADDRESS_LEN {
            return Err(ValidationError::TooLong {
                field: "peer.address",
                max: MAX_ADDRESS_LEN,
            });
        }
    }
    if peer.capabilities.len() > MAX_CAPABILITIES {
        return Err(ValidationError::TooMany {
            field: "peer.capabilities",
            max: MAX_CAPABILITIES,
        });
    }
    for capability_value in &peer.capabilities {
        capability(capability_value)?;
    }
    if peer.expires_at < peer.announced_at {
        return Err(ValidationError::Invalid {
            field: "peer.expiry",
        });
    }
    Ok(())
}

fn address_record(address: &AddressRecord) -> Result<(), ValidationError> {
    if address.address.is_empty() || address.address.len() > MAX_ADDRESS_LEN {
        return Err(ValidationError::TooLong {
            field: "address.address",
            max: MAX_ADDRESS_LEN,
        });
    }
    if address.expires_at == 0 || address.confidence > 100 {
        return Err(ValidationError::Invalid {
            field: "address.metadata",
        });
    }
    Ok(())
}

fn signature(value: &[u8], field: &'static str) -> Result<(), ValidationError> {
    if value.len() != 64 {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn dht_record(record: &DhtRecord) -> Result<(), ValidationError> {
    if record.owner != crate::NodeId::from_public_key(&record.owner_public_key)
        || record.expires_at == 0
        || record.value.len() > MAX_DHT_RECORD_VALUE
    {
        return Err(ValidationError::Invalid {
            field: "dht.record.identity_or_value",
        });
    }
    signature(&record.signature, "dht.record.signature")
}

fn signed_evidence(value: &SignedEvidence) -> Result<(), ValidationError> {
    if value.issuer != crate::NodeId::from_public_key(&value.issuer_public_key)
        || value.expires_at < value.observed_at
    {
        return Err(ValidationError::Invalid {
            field: "evidence.identity_or_validity",
        });
    }
    if value.payload.len() > MAX_EVIDENCE_PAYLOAD {
        return Err(ValidationError::Oversized {
            field: "evidence.payload",
        });
    }
    signature(&value.signature, "evidence.signature")
}

fn manifest_shards(
    shards: &[crate::ArtifactShard],
    field: &'static str,
) -> Result<(), ValidationError> {
    if shards.len() > MAX_ARTIFACT_SHARDS {
        return Err(ValidationError::TooMany {
            field,
            max: MAX_ARTIFACT_SHARDS,
        });
    }
    Ok(())
}

fn node_id(value: &crate::NodeId, field: &'static str) -> Result<(), ValidationError> {
    if *value == crate::NodeId::default() {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn artifact_id(value: &crate::ArtifactId, field: &'static str) -> Result<(), ValidationError> {
    if *value == crate::ArtifactId::default() {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn training_job_id(value: &JobId, field: &'static str) -> Result<(), ValidationError> {
    if *value == JobId::default() {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn training_nodes(
    nodes: &[crate::NodeId],
    field: &'static str,
    max: usize,
) -> Result<(), ValidationError> {
    if nodes.is_empty() || nodes.len() > max {
        return Err(ValidationError::TooMany { field, max });
    }
    let mut seen = std::collections::HashSet::new();
    for node in nodes {
        node_id(node, field)?;
        if !seen.insert(node) {
            return Err(ValidationError::Invalid { field });
        }
    }
    Ok(())
}

fn training_assignment(assignment: &crate::TrainingShardAssignment) -> Result<(), ValidationError> {
    training_nodes(
        &assignment.owners,
        "training.shard.owners",
        MAX_TRAINING_WORKERS,
    )?;
    training_nodes(
        &assignment.replicas,
        "training.shard.replicas",
        MAX_TRAINING_REPLICAS,
    )?;
    artifact_id(&assignment.state_hash, "training.shard.state_hash")
}

fn training_start(value: &TrainingStart) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.start.job_id")?;
    node_id(&value.coordinator, "training.start.coordinator")?;
    artifact_id(&value.branch, "training.start.branch")?;
    artifact_id(&value.plan_hash, "training.start.plan_hash")?;
    if value.term == 0
        || value.membership_epoch == 0
        || value.max_windows == 0
        || value.local_steps == 0
        || value.local_steps > 1024
        || value.max_staleness > 1024
        || value.checkpoint_every == 0
    {
        return Err(ValidationError::Invalid {
            field: "training.start.generation",
        });
    }
    if value.model_state_bytes > MAX_TRAINING_STATE_BYTES
        || value.shard_state_bytes > MAX_TRAINING_STATE_BYTES
        || (value.model_state_bytes > 0
            && (value.shard_state_bytes == 0 || value.shard_state_bytes > value.model_state_bytes))
    {
        return Err(ValidationError::Invalid {
            field: "training.start.state_bytes",
        });
    }
    training_nodes(
        &value.participants,
        "training.start.participants",
        MAX_TRAINING_WORKERS,
    )?;
    if value.groups.is_empty() || value.groups.len() > MAX_TRAINING_GROUPS {
        return Err(ValidationError::TooMany {
            field: "training.start.groups",
            max: MAX_TRAINING_GROUPS,
        });
    }
    if value.shards.is_empty() || value.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "training.start.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    if value.samples.is_empty() || value.samples.len() > MAX_TRAINING_SAMPLES {
        return Err(ValidationError::TooMany {
            field: "training.start.samples",
            max: MAX_TRAINING_SAMPLES,
        });
    }
    training_assignment(&value.assignment)
}

fn training_window(value: &TrainingWindow) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.window.job_id")?;
    node_id(&value.coordinator, "training.window.coordinator")?;
    node_id(&value.aggregator, "training.window.aggregator")?;
    artifact_id(&value.branch, "training.window.branch")?;
    if value.term == 0 || value.membership_epoch == 0 || value.local_steps == 0 {
        return Err(ValidationError::Invalid {
            field: "training.window.generation",
        });
    }
    training_nodes(
        &value.members,
        "training.window.members",
        MAX_TRAINING_WORKERS,
    )
}

fn training_update(value: &TrainingUpdate) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.update.job_id")?;
    node_id(&value.worker, "training.update.worker")?;
    artifact_id(&value.branch, "training.update.branch")?;
    if value.term == 0
        || value.membership_epoch == 0
        || value.update_sequence == 0
        || value.samples == 0
        || value.value.unsigned_abs() > 1_000_000_000
        || value.optimizer.unsigned_abs() > 1_000_000_000
        || value.loss.unsigned_abs() > 1_000_000_000
    {
        return Err(ValidationError::Invalid {
            field: "training.update.bounds",
        });
    }
    Ok(())
}

fn training_aggregate(value: &crate::TrainingAggregate) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.aggregate.job_id")?;
    node_id(&value.aggregator, "training.aggregate.aggregator")?;
    artifact_id(&value.branch, "training.aggregate.branch")?;
    artifact_id(&value.state_artifact, "training.aggregate.state_artifact")?;
    if value.term == 0 || value.membership_epoch == 0 || value.generation == 0 {
        return Err(ValidationError::Invalid {
            field: "training.aggregate.generation",
        });
    }
    if value.value.unsigned_abs() > 1_000_000_000
        || value.optimizer.unsigned_abs() > 1_000_000_000
        || value.loss.unsigned_abs() > 1_000_000_000
    {
        return Err(ValidationError::Invalid {
            field: "training.aggregate.bounds",
        });
    }
    training_nodes(
        &value.contributors,
        "training.aggregate.contributors",
        MAX_TRAINING_CONTRIBUTORS,
    )?;
    training_nodes(
        &value.replicas,
        "training.aggregate.replicas",
        MAX_TRAINING_REPLICAS,
    )?;
    Ok(())
}

fn training_state(value: &TrainingState) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.state.job_id")?;
    node_id(&value.coordinator, "training.state.coordinator")?;
    artifact_id(&value.branch, "training.state.branch")?;
    if value.term == 0 || value.membership_epoch == 0 {
        return Err(ValidationError::Invalid {
            field: "training.state.generation",
        });
    }
    if value.shards.is_empty() || value.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "training.state.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    let mut shard_ids = std::collections::HashSet::new();
    for shard in &value.shards {
        if !shard_ids.insert(shard.shard_id) {
            return Err(ValidationError::Invalid {
                field: "training.state.shard_id",
            });
        }
        artifact_id(&shard.state_hash, "training.state.shard_hash")?;
        training_nodes(
            &shard.providers,
            "training.state.providers",
            MAX_TRAINING_REPLICAS,
        )?;
    }
    if let Some(checkpoint) = value.checkpoint {
        artifact_id(&checkpoint, "training.state.checkpoint")?;
    }
    Ok(())
}

fn training_election(value: &TrainingElection) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.election.job_id")?;
    node_id(&value.candidate, "training.election.candidate")?;
    artifact_id(&value.branch, "training.election.branch")?;
    if value.term == 0 || value.membership_epoch == 0 {
        return Err(ValidationError::Invalid {
            field: "training.election.generation",
        });
    }
    Ok(())
}

fn training_checkpoint_offer(value: &TrainingCheckpointOffer) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.checkpoint.job_id")?;
    node_id(&value.creator, "training.checkpoint.creator")?;
    artifact_id(&value.artifact, "training.checkpoint.artifact")?;
    if value.term == 0 || value.checkpoint_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "training.checkpoint.generation",
        });
    }
    Ok(())
}

fn training_checkpoint_commit(value: &TrainingCheckpointCommit) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.commit.job_id")?;
    node_id(&value.coordinator, "training.commit.coordinator")?;
    artifact_id(&value.branch, "training.commit.branch")?;
    if value.term == 0 || value.membership_epoch == 0 || value.quorum.is_empty() {
        return Err(ValidationError::Invalid {
            field: "training.commit.generation",
        });
    }
    if value.manifest.job_id != value.job_id
        || value.manifest.membership_epoch != value.membership_epoch
        || value.manifest.term > value.term
        || value.manifest.checkpoint_generation == 0
    {
        return Err(ValidationError::Invalid {
            field: "training.commit.manifest_lineage",
        });
    }
    training_nodes(
        &value.quorum,
        "training.commit.quorum",
        MAX_TRAINING_WORKERS,
    )?;
    if value.manifest.shards.is_empty() || value.manifest.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "training.commit.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    for shard in &value.manifest.shards {
        artifact_id(&shard.artifact, "training.commit.shard_artifact")?;
        if shard.generation == 0 {
            return Err(ValidationError::Invalid {
                field: "training.commit.shard_generation",
            });
        }
        training_nodes(
            &shard.providers,
            "training.commit.shard_providers",
            MAX_TRAINING_REPLICAS,
        )?;
    }
    artifact_id(&value.manifest.hash, "training.commit.manifest_hash")
}

fn training_ack(value: &TrainingAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "training.ack.job_id")?;
    node_id(&value.peer, "training.ack.peer")?;
    if value.term == 0 {
        return Err(ValidationError::Invalid {
            field: "training.ack.term",
        });
    }
    Ok(())
}

fn training_message(value: &TrainingMessage) -> Result<(), ValidationError> {
    match value {
        TrainingMessage::Start(value) => training_start(value),
        TrainingMessage::Window(value) => training_window(value),
        TrainingMessage::Update(value) => training_update(value),
        TrainingMessage::Aggregate(value) => training_aggregate(value),
        TrainingMessage::ShardState(value) => {
            training_job_id(&value.job_id, "training.shard_state.job_id")?;
            node_id(&value.owner, "training.shard_state.owner")?;
            artifact_id(&value.branch, "training.shard_state.branch")?;
            if value.term == 0
                || value.membership_epoch == 0
                || value.generation == 0
                || value.value.unsigned_abs() > 1_000_000_000
                || value.optimizer.unsigned_abs() > 1_000_000_000
            {
                return Err(ValidationError::Invalid {
                    field: "training.shard_state.bounds",
                });
            }
            artifact_id(&value.state_hash, "training.shard_state.hash")
        }
        TrainingMessage::State(value) => training_state(value),
        TrainingMessage::Ack(value) => training_ack(value),
        TrainingMessage::Election(value) => training_election(value),
        TrainingMessage::CheckpointOffer(value) => training_checkpoint_offer(value),
        TrainingMessage::CheckpointCommit(value) => training_checkpoint_commit(value),
    }
}

fn v4_node_list(
    nodes: &[crate::NodeId],
    field: &'static str,
    max: usize,
) -> Result<(), ValidationError> {
    if nodes.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    if nodes.len() > max {
        return Err(ValidationError::TooMany { field, max });
    }
    let mut seen = std::collections::HashSet::new();
    for node in nodes {
        node_id(node, field)?;
        if !seen.insert(node) {
            return Err(ValidationError::Invalid { field });
        }
    }
    Ok(())
}

fn v4_vector(values: &[i64], field: &'static str) -> Result<(), ValidationError> {
    if values.is_empty() {
        return Err(ValidationError::Empty { field });
    }
    if values.len() > MAX_V4_VECTOR {
        return Err(ValidationError::TooMany {
            field,
            max: MAX_V4_VECTOR,
        });
    }
    if values
        .iter()
        .any(|value| value.unsigned_abs() > 1_000_000_000)
    {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn v4_hash(value: &crate::ArtifactId, field: &'static str) -> Result<(), ValidationError> {
    artifact_id(value, field)
}

fn backend_capabilities(
    value: &crate::BackendCapabilities,
    field: &'static str,
) -> Result<(), ValidationError> {
    string(&value.runtime_version, field)?;
    if value.device_count == 0
        || value.device_memory_bytes == 0
        || value.available_memory_bytes
            > value
                .device_memory_bytes
                .saturating_mul(u64::from(value.device_count))
        || value.available_memory_bytes == 0
        || value.formats.is_empty()
        || value.formats.len() > MAX_V5_FORMATS
        || value.max_tensor_elements == 0
        || value.features.len() > MAX_V5_FEATURES
        || value.max_concurrent_tasks == 0
        || value.safety_margin_permille > 500
        || value.observed_successes > 1_000_000
        || value.observed_failures > 1_000_000
    {
        return Err(ValidationError::Invalid { field });
    }
    if let Some(architecture) = &value.device_architecture {
        string(architecture, field)?;
    }
    if value
        .features
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != value.features.len()
        || value
            .formats
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.formats.len()
    {
        return Err(ValidationError::Invalid { field });
    }
    if value.physical_verified && (!value.driver_available || !value.runtime_available) {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn compute_requirements(
    value: &crate::ComputeRequirements,
    field: &'static str,
) -> Result<(), ValidationError> {
    if value.required_memory_bytes == 0
        || value.max_tensor_elements == 0
        || value.kernel_version == 0
        || value.allowed_backends.len() > MAX_V5_BACKENDS
        || value.fallback_backends.len() > MAX_V5_BACKENDS
        || value.required_formats.is_empty()
        || value.required_formats.len() > MAX_V5_FORMATS
        || value.required_features.len() > MAX_V5_FEATURES
    {
        return Err(ValidationError::Invalid { field });
    }
    string(&value.kernel_id, field)?;
    for backends in [&value.allowed_backends, &value.fallback_backends] {
        if backends
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != backends.len()
        {
            return Err(ValidationError::Invalid { field });
        }
    }
    if value.required_backend.is_none() && value.allowed_backends.is_empty() {
        return Err(ValidationError::Invalid { field });
    }
    if let Some(required) = value.required_backend
        && !value.allowed_backends.is_empty()
        && !value.allowed_backends.contains(&required)
    {
        return Err(ValidationError::Invalid { field });
    }
    if value
        .required_formats
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len()
        != value.required_formats.len()
        || value
            .required_features
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.required_features.len()
    {
        return Err(ValidationError::Invalid { field });
    }
    if !value.fallback_allowed && !value.fallback_backends.is_empty() {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn backend_assignment(
    value: &crate::BackendAssignment,
    worker_set: &std::collections::HashSet<crate::NodeId>,
    capabilities: &[crate::V4WorkerCapability],
    field: &'static str,
) -> Result<(), ValidationError> {
    node_id(&value.worker, field)?;
    if !worker_set.contains(&value.worker) || value.device_index > 1024 {
        return Err(ValidationError::Invalid { field });
    }
    compute_requirements(&value.requirements, field)?;
    if value.requirements.required_backend != Some(value.backend)
        && !value.requirements.allowed_backends.contains(&value.backend)
    {
        return Err(ValidationError::Invalid { field });
    }
    let Some(worker) = capabilities
        .iter()
        .find(|worker| worker.node == value.worker)
    else {
        return Err(ValidationError::Invalid { field });
    };
    let Some(backend) = worker
        .backends
        .iter()
        .find(|backend| backend.kind == value.backend)
    else {
        return Err(ValidationError::Invalid { field });
    };
    if !backend.runtime_available
        || backend.health == crate::BackendHealth::Unavailable
        || (backend.observed_failures > 0 && backend.observed_successes == 0)
        || value.device_index >= backend.device_count
        || value
            .requirements
            .required_formats
            .iter()
            .any(|format| !backend.formats.contains(format))
        || value.requirements.required_memory_bytes
            > backend.available_memory_bytes.saturating_mul(u64::from(
                1000_u16.saturating_sub(backend.safety_margin_permille),
            )) / 1000
        || value.requirements.max_tensor_elements > backend.max_tensor_elements
        || value
            .requirements
            .required_features
            .iter()
            .any(|feature| !backend.features.contains(feature))
    {
        return Err(ValidationError::Invalid { field });
    }
    Ok(())
}

fn v5_challenge(value: &crate::CapabilityChallenge) -> Result<(), ValidationError> {
    if value.challenge_id == crate::RequestId::default()
        || value.job_id == crate::JobId::default()
        || value.worker == crate::NodeId::default()
        || value.task_kind != crate::ComputeTaskKind::CapabilityChallenge
        || value.input_elements == 0
        || value.input_elements > MAX_V4_VECTOR as u32
        || value.deadline_ms == 0
        || value.deadline_ms > 120_000
    {
        return Err(ValidationError::Invalid {
            field: "v5.challenge.bounds",
        });
    }
    Ok(())
}

fn v5_challenge_result(value: &crate::CapabilityChallengeResult) -> Result<(), ValidationError> {
    if value.challenge_id == crate::RequestId::default()
        || value.worker == crate::NodeId::default()
        || value.elapsed_micros == 0
        || value.allocated_bytes > MAX_TRAINING_STATE_BYTES
    {
        return Err(ValidationError::Invalid {
            field: "v5.challenge_result.bounds",
        });
    }
    if let Some(error) = &value.error {
        string(error, "v5.challenge_result.error")?;
    }
    Ok(())
}

fn v5_evidence(value: &crate::CapabilityEvidenceRecord) -> Result<(), ValidationError> {
    if value.worker == crate::NodeId::default()
        || value.observed_at == 0
        || value.last_throughput_micros == 0
        || value.successful_challenges > 1_000_000
        || value.failed_challenges > 1_000_000
        || value.successful_tasks > 1_000_000
        || value.failed_tasks > 1_000_000
    {
        return Err(ValidationError::Invalid {
            field: "v5.evidence.bounds",
        });
    }
    if let Some(error) = &value.last_error {
        string(error, "v5.evidence.error")?;
    }
    Ok(())
}

fn v4_plan(value: &crate::V4TrainingPlan) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.plan.job_id")?;
    node_id(&value.proposer, "v4.plan.proposer")?;
    artifact_id(&value.branch, "v4.plan.branch")?;
    artifact_id(&value.plan_hash, "v4.plan.hash")?;
    let strategy_shape_valid = match value.strategy {
        crate::V4ParallelismStrategy::LocalSgd => {
            value.tensor_degree == 0 && value.pipeline_stages == 0
        }
        crate::V4ParallelismStrategy::TensorParallel => {
            value.tensor_degree >= 2 && value.pipeline_stages == 0
        }
        crate::V4ParallelismStrategy::PipelineParallel => {
            value.tensor_degree == 0 && value.pipeline_stages == 2
        }
        crate::V4ParallelismStrategy::Hybrid => {
            value.tensor_degree == 2 && value.pipeline_stages == 2
        }
    };
    let support_matches_strategy = match value.strategy {
        crate::V4ParallelismStrategy::LocalSgd => value.support == crate::V4SupportLevel::Supported,
        crate::V4ParallelismStrategy::TensorParallel => {
            value.support == crate::V4SupportLevel::Supported && value.tensor_degree == 2
        }
        crate::V4ParallelismStrategy::PipelineParallel => {
            value.support == crate::V4SupportLevel::Supported && value.pipeline_stages == 2
        }
        crate::V4ParallelismStrategy::Hybrid => {
            value.support == crate::V4SupportLevel::Experimental
                && value.tensor_degree == 2
                && value.pipeline_stages == 2
        }
    };
    if !strategy_shape_valid || !support_matches_strategy {
        return Err(ValidationError::Invalid {
            field: "v4.plan.strategy",
        });
    }
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.training_epoch == 0
        || value.membership_epoch == 0
        || value.coordination_term == 0
        || value.local_steps == 0
        || value.local_steps > 1024
        || value.max_staleness > 1024
        || value.checkpoint_replication == 0
        || value.checkpoint_replication > 16
        || value.tensor_degree > MAX_V4_STAGES as u16
        || value.pipeline_stages > MAX_V4_STAGES as u16
    {
        return Err(ValidationError::Invalid {
            field: "v4.plan.generation",
        });
    }
    if !(2..=MAX_V4_PLAN_WORKERS).contains(&value.workers.len()) {
        return Err(ValidationError::Invalid {
            field: "v4.plan.workers",
        });
    }
    v4_node_list(&value.workers, "v4.plan.workers", MAX_V4_PLAN_WORKERS)?;
    if value.worker_capabilities.len() != value.workers.len() {
        return Err(ValidationError::Invalid {
            field: "v4.plan.worker_capabilities",
        });
    }
    let mut capability_nodes = std::collections::HashSet::new();
    for capability in &value.worker_capabilities {
        node_id(&capability.node, "v4.plan.worker_capability.node")?;
        if !value.workers.contains(&capability.node)
            || !capability_nodes.insert(capability.node)
            || capability.memory_bytes == 0
            || capability.compute_units == 0
            || capability.reliability_permille > 1000
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.worker_capability",
            });
        }
        if capability.accelerator.device_count == 0
            || capability.accelerator.memory_bytes == 0
            || capability.accelerator.formats.is_empty()
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.worker_accelerator",
            });
        }
        if capability.backends.len() > MAX_V5_BACKENDS {
            return Err(ValidationError::TooMany {
                field: "v5.plan.worker_backends",
                max: MAX_V5_BACKENDS,
            });
        }
        for backend in &capability.backends {
            backend_capabilities(backend, "v5.plan.backend_capabilities")?;
        }
    }
    if value.groups.is_empty() || value.groups.len() > MAX_V4_GROUPS {
        return Err(ValidationError::TooMany {
            field: "v4.plan.groups",
            max: MAX_V4_GROUPS,
        });
    }
    let worker_set = value
        .workers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut group_ids = std::collections::HashSet::new();
    let mut grouped_workers = std::collections::HashSet::new();
    for group in &value.groups {
        if group.members.is_empty() || group.members.len() > MAX_V4_PLAN_WORKERS {
            return Err(ValidationError::TooMany {
                field: "v4.plan.group.members",
                max: MAX_V4_PLAN_WORKERS,
            });
        }
        v4_node_list(&group.members, "v4.plan.group.members", MAX_V4_PLAN_WORKERS)?;
        if !group_ids.insert(group.group_id)
            || group
                .members
                .iter()
                .any(|member| !worker_set.contains(member))
            || group
                .members
                .iter()
                .any(|member| !grouped_workers.insert(*member))
            || !group.members.contains(&group.aggregator)
            || group
                .parent
                .is_some_and(|parent| !value.workers.contains(&parent))
            || group
                .parent
                .is_some_and(|parent| group.members.contains(&parent))
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.group.topology",
            });
        }
    }
    if grouped_workers != worker_set {
        return Err(ValidationError::Invalid {
            field: "v4.plan.group.coverage",
        });
    }
    if value.shards.is_empty() || value.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "v4.plan.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    let mut shard_ids = std::collections::HashSet::new();
    for shard in &value.shards {
        if !shard_ids.insert(shard.shard_id) {
            return Err(ValidationError::Invalid {
                field: "v4.plan.shard.id",
            });
        }
        if shard.model_generation == 0
            || shard.ownership_generation == 0
            || shard.state_bytes == 0
            || shard.memory_bytes == 0
            || shard.state_bytes > MAX_TRAINING_STATE_BYTES
            || shard.memory_bytes > MAX_TRAINING_STATE_BYTES
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.shard.generation",
            });
        }
        if shard.owners.len() != 1 {
            return Err(ValidationError::Invalid {
                field: "v4.plan.shard.owners",
            });
        }
        v4_node_list(&shard.owners, "v4.plan.shard.owners", MAX_V4_PLAN_WORKERS)?;
        if shard
            .owners
            .iter()
            .chain(shard.replicas.iter())
            .any(|node| !worker_set.contains(node))
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.shard.worker",
            });
        }
        if shard.replicas.len() > MAX_TRAINING_REPLICAS {
            return Err(ValidationError::TooMany {
                field: "v4.plan.shard.replicas",
                max: MAX_TRAINING_REPLICAS,
            });
        }
        if shard
            .replicas
            .iter()
            .any(|replica| shard.owners.contains(replica))
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.shard.replica_owner_overlap",
            });
        }
        if !shard.replicas.is_empty() {
            v4_node_list(
                &shard.replicas,
                "v4.plan.shard.replicas",
                MAX_TRAINING_REPLICAS,
            )?;
        }
        v4_hash(&shard.content_hash, "v4.plan.shard.hash")?;
        string(&shard.runtime_requirement, "v4.plan.shard.runtime")?;
    }
    if let Some(accelerator) = &value.accelerator {
        string(&accelerator.device_model, "v4.plan.accelerator.model")?;
        string(&accelerator.runtime, "v4.plan.accelerator.runtime")?;
        string(
            &accelerator.runtime_version,
            "v4.plan.accelerator.runtime_version",
        )?;
        if accelerator.device_count == 0
            || accelerator.memory_bytes == 0
            || accelerator.formats.is_empty()
            || accelerator.formats.len() > 8
        {
            return Err(ValidationError::Invalid {
                field: "v4.plan.accelerator",
            });
        }
    }
    if let Some(parent) = &value.parent_plan_hash {
        v4_hash(parent, "v4.plan.parent_hash")?;
        if parent == &value.plan_hash {
            return Err(ValidationError::Invalid {
                field: "v4.plan.parent_hash",
            });
        }
    }
    if value.compute_requirements.len() > MAX_V5_REQUIREMENTS {
        return Err(ValidationError::TooMany {
            field: "v5.plan.compute_requirements",
            max: MAX_V5_REQUIREMENTS,
        });
    }
    for requirement in &value.compute_requirements {
        compute_requirements(requirement, "v5.plan.compute_requirement")?;
    }
    if value.backend_assignments.len() > MAX_V5_ASSIGNMENTS {
        return Err(ValidationError::TooMany {
            field: "v5.plan.backend_assignments",
            max: MAX_V5_ASSIGNMENTS,
        });
    }
    for assignment in &value.backend_assignments {
        backend_assignment(
            assignment,
            &worker_set,
            &value.worker_capabilities,
            "v5.plan.backend_assignment",
        )?;
    }
    string(&value.rationale, "v4.plan.rationale")
}

fn v4_execution_graph(value: &crate::V4ExecutionGraph) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.graph.job_id")?;
    node_id(&value.proposer, "v4.graph.proposer")?;
    node_id(&value.coordinator, "v4.graph.coordinator")?;
    node_id(&value.reply_to, "v4.graph.reply_to")?;
    v4_hash(&value.plan_hash, "v4.graph.plan_hash")?;
    v4_hash(&value.branch, "v4.graph.branch")?;
    v4_hash(&value.graph_hash, "v4.graph.hash")?;
    if value.graph_generation == 0
        || value
            .parent_graph_generation
            .is_some_and(|parent| parent == 0 || parent >= value.graph_generation)
        || value.training_epoch == 0
        || value.membership_epoch == 0
        || value.coordination_term == 0
        || value.optimizer_generation == 0
        || value.checkpoint_generation == 0
        || value.collective_generation == 0
        || value.local_steps == 0
        || value.local_steps > 1024
        || value.max_staleness > 1024
        || value.checkpoint_replication == 0
        || value.checkpoint_replication > 16
    {
        return Err(ValidationError::Invalid {
            field: "v4.graph.generation",
        });
    }
    if !(2..=MAX_V4_PLAN_WORKERS).contains(&value.workers.len()) {
        return Err(ValidationError::Invalid {
            field: "v4.graph.workers",
        });
    }
    v4_node_list(&value.workers, "v4.graph.workers", MAX_V4_PLAN_WORKERS)?;
    if !value.workers.contains(&value.proposer) || !value.workers.contains(&value.coordinator) {
        return Err(ValidationError::Invalid {
            field: "v4.graph.role_worker",
        });
    }
    if value.retired_workers.len() > MAX_V4_PLAN_WORKERS
        || value
            .retired_workers
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.retired_workers.len()
        || value
            .retired_workers
            .iter()
            .any(|worker| value.workers.contains(worker))
    {
        return Err(ValidationError::Invalid {
            field: "v4.graph.retired_workers",
        });
    }
    if value.election_certificate.len() > MAX_V4_PLAN_WORKERS
        || value
            .election_certificate
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            != value.election_certificate.len()
    {
        return Err(ValidationError::Invalid {
            field: "v4.graph.election_certificate",
        });
    }

    if value.aggregation_groups.is_empty() || value.aggregation_groups.len() > MAX_V4_GROUPS {
        return Err(ValidationError::TooMany {
            field: "v4.graph.aggregation_groups",
            max: MAX_V4_GROUPS,
        });
    }
    let worker_set = value
        .workers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let mut grouped = std::collections::HashSet::new();
    let mut group_ids = std::collections::HashSet::new();
    for group in &value.aggregation_groups {
        if !group_ids.insert(group.group_id)
            || group.members.is_empty()
            || !group
                .members
                .iter()
                .all(|member| worker_set.contains(member))
            || !group.members.iter().all(|member| grouped.insert(*member))
            || !group.members.contains(&group.aggregator)
            || group
                .parent
                .is_some_and(|parent| !worker_set.contains(&parent))
        {
            return Err(ValidationError::Invalid {
                field: "v4.graph.aggregation_group",
            });
        }
        v4_node_list(
            &group.members,
            "v4.graph.aggregation_group.members",
            MAX_V4_PLAN_WORKERS,
        )?;
    }
    if grouped != worker_set {
        return Err(ValidationError::Invalid {
            field: "v4.graph.aggregation_group.coverage",
        });
    }

    if value.tensor_groups.len() > MAX_V4_GROUPS {
        return Err(ValidationError::TooMany {
            field: "v4.graph.tensor_groups",
            max: MAX_V4_GROUPS,
        });
    }
    let mut tensor_group_ids = std::collections::HashSet::new();
    let mut tensor_shards = std::collections::HashSet::new();
    for group in &value.tensor_groups {
        if !tensor_group_ids.insert(group.group_id)
            || group.members.len() < 2
            || group.members.len() > MAX_V4_PLAN_WORKERS
            || group.shard_ids.is_empty()
            || group.generation == 0
            || !group
                .members
                .iter()
                .all(|member| worker_set.contains(member))
            || group
                .shard_ids
                .iter()
                .any(|shard_id| !tensor_shards.insert(*shard_id))
        {
            return Err(ValidationError::Invalid {
                field: "v4.graph.tensor_group",
            });
        }
        v4_node_list(
            &group.members,
            "v4.graph.tensor_group.members",
            MAX_V4_PLAN_WORKERS,
        )?;
    }

    if value.pipeline_stages.len() > MAX_V4_STAGES {
        return Err(ValidationError::TooMany {
            field: "v4.graph.pipeline_stages",
            max: MAX_V4_STAGES,
        });
    }
    let mut stages = std::collections::HashSet::new();
    for stage in &value.pipeline_stages {
        if !stages.insert(stage.stage_id)
            || stage.generation == 0
            || !worker_set.contains(&stage.worker)
            || stage.replicas.len() > MAX_TRAINING_REPLICAS
            || stage.replicas.contains(&stage.worker)
            || stage
                .replicas
                .iter()
                .any(|replica| !worker_set.contains(replica))
        {
            return Err(ValidationError::Invalid {
                field: "v4.graph.pipeline_stage",
            });
        }
        if !stage.replicas.is_empty() {
            v4_node_list(
                &stage.replicas,
                "v4.graph.pipeline_stage.replicas",
                MAX_TRAINING_REPLICAS,
            )?;
        }
    }

    if value.shards.is_empty() || value.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "v4.graph.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    let mut shard_ids = std::collections::HashSet::new();
    for shard in &value.shards {
        if !shard_ids.insert(shard.shard_id)
            || shard.owners.len() != 1
            || shard.model_generation == 0
            || shard.ownership_generation == 0
            || shard.state_bytes == 0
            || shard.memory_bytes == 0
            || !shard.owners.iter().all(|owner| worker_set.contains(owner))
            || shard.replicas.len() > MAX_TRAINING_REPLICAS
            || shard
                .replicas
                .iter()
                .any(|replica| !worker_set.contains(replica) || shard.owners.contains(replica))
        {
            return Err(ValidationError::Invalid {
                field: "v4.graph.shard",
            });
        }
        v4_hash(&shard.content_hash, "v4.graph.shard.hash")?;
    }
    if value.optimizer_shards.is_empty() || value.optimizer_shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "v4.graph.optimizer_shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    let mut optimizer_ids = std::collections::HashSet::new();
    for placement in &value.optimizer_shards {
        if !optimizer_ids.insert(placement.shard_id)
            || placement.generation == 0
            || !shard_ids.contains(&placement.shard_id)
            || !worker_set.contains(&placement.owner)
            || placement.replicas.len() > MAX_TRAINING_REPLICAS
            || placement.replicas.contains(&placement.owner)
            || placement
                .replicas
                .iter()
                .any(|replica| !worker_set.contains(replica))
        {
            return Err(ValidationError::Invalid {
                field: "v4.graph.optimizer_shard",
            });
        }
    }
    if value.backend_assignments.len() > MAX_V5_ASSIGNMENTS {
        return Err(ValidationError::TooMany {
            field: "v5.graph.backend_assignments",
            max: MAX_V5_ASSIGNMENTS,
        });
    }
    // Graph records do not repeat the full capability advertisement.  The
    // actual V5 plan performs capability validation; graph validation still
    // checks bounded task bindings and generation-independent worker scope.
    for assignment in &value.backend_assignments {
        compute_requirements(&assignment.requirements, "v5.graph.backend_requirement")?;
        node_id(&assignment.worker, "v5.graph.backend_worker")?;
        if !worker_set.contains(&assignment.worker) || assignment.device_index > 1024 {
            return Err(ValidationError::Invalid {
                field: "v5.graph.backend_assignment",
            });
        }
    }
    Ok(())
}

fn v4_integrated_state(value: &crate::V4IntegratedStateRecord) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.integrated_state.job_id")?;
    v4_hash(&value.plan_hash, "v4.integrated_state.plan_hash")?;
    v4_hash(&value.branch, "v4.integrated_state.branch")?;
    v4_hash(&value.state_hash, "v4.integrated_state.hash")?;
    node_id(&value.coordinator, "v4.integrated_state.coordinator")?;
    if value.graph_generation == 0
        || value.target_windows == 0
        || value.target_windows > 4096
        || value.window > value.target_windows
        || value.checkpoint_generation == 0
        || value.model_generation == 0
        || value.optimizer_generation == 0
        || value.membership_epoch == 0
        || value.collective_generation == 0
    {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_state.generation",
        });
    }
    Ok(())
}

fn v4_integrated_start(value: &crate::V4IntegratedStart) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.integrated_start.request_id")?;
    node_id(&value.reply_to, "v4.integrated_start.reply_to")?;
    if value.windows == 0 || value.windows > 4096 || value.checkpoint_every == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_start.bounds",
        });
    }
    if value.graph.job_id != value.plan.job_id
        || value.graph.plan_hash != value.plan.plan_hash
        || value.graph.strategy != value.plan.strategy
        || value.graph.graph_generation == 0
        || value.plan.validate().is_err()
        || v4_execution_graph(&value.graph).is_err()
    {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_start.graph",
        });
    }
    Ok(())
}

fn v4_integrated_ack(value: &crate::V4IntegratedAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.integrated_ack.job_id")?;
    node_id(&value.worker, "v4.integrated_ack.worker")?;
    if value.graph_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_ack.generation",
        });
    }
    string(&value.reason, "v4.integrated_ack.reason")
}

fn v4_integrated_state_ack(value: &crate::V4IntegratedStateAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.integrated_state_ack.job_id")?;
    node_id(&value.worker, "v4.integrated_state_ack.worker")?;
    v4_hash(&value.state_hash, "v4.integrated_state_ack.hash")?;
    if value.graph_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_state_ack.generation",
        });
    }
    Ok(())
}

fn v4_integrated_probe(value: &crate::V4IntegratedProbe) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.integrated_probe.request_id")?;
    training_job_id(&value.job_id, "v4.integrated_probe.job_id")?;
    node_id(&value.requester, "v4.integrated_probe.requester")?;
    v4_hash(&value.graph_hash, "v4.integrated_probe.graph_hash")?;
    v4_hash(&value.state_hash, "v4.integrated_probe.state_hash")?;
    if value.graph_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_probe.generation",
        });
    }
    Ok(())
}

fn v4_integrated_probe_ack(value: &crate::V4IntegratedProbeAck) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.integrated_probe_ack.request_id")?;
    training_job_id(&value.job_id, "v4.integrated_probe_ack.job_id")?;
    node_id(&value.responder, "v4.integrated_probe_ack.responder")?;
    if value.graph_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_probe_ack.generation",
        });
    }
    Ok(())
}

fn v4_integrated_election_request(
    value: &crate::V4IntegratedElectionRequest,
) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.integrated_election.request_id")?;
    training_job_id(&value.job_id, "v4.integrated_election.job_id")?;
    node_id(&value.candidate, "v4.integrated_election.candidate")?;
    v4_hash(&value.graph_hash, "v4.integrated_election.graph_hash")?;
    v4_hash(&value.branch, "v4.integrated_election.branch")?;
    v4_hash(&value.state_hash, "v4.integrated_election.state_hash")?;
    if value.graph_generation == 0 || value.term == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_election.generation",
        });
    }
    Ok(())
}

fn v4_integrated_election_vote(
    value: &crate::V4IntegratedElectionVote,
) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.integrated_election_vote.job_id")?;
    node_id(&value.candidate, "v4.integrated_election_vote.candidate")?;
    node_id(&value.voter, "v4.integrated_election_vote.voter")?;
    v4_hash(&value.graph_hash, "v4.integrated_election_vote.graph_hash")?;
    if value.graph_generation == 0 || value.term == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_election_vote.generation",
        });
    }
    string(&value.reason, "v4.integrated_election_vote.reason")
}

fn v4_integrated_result(value: &crate::V4IntegratedResult) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.integrated_result.request_id")?;
    training_job_id(&value.job_id, "v4.integrated_result.job_id")?;
    if value.graph_generation == 0 || value.windows_completed > 4096 {
        return Err(ValidationError::Invalid {
            field: "v4.integrated_result.generation",
        });
    }
    Ok(())
}

fn v4_tensor_replica(value: &crate::V4TensorReplica) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_replica.job_id")?;
    node_id(&value.source, "v4.tensor_replica.source")?;
    v4_hash(&value.state_hash, "v4.tensor_replica.hash")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.state_generation == 0
        || value.ownership_generation == 0
        || value.rows == 0
        || value.cols == 0
        || value.rows as usize * value.cols as usize != value.weights.len()
        || value.weights.len() > MAX_V4_VECTOR * MAX_V4_VECTOR
        || value.row_offset as usize + value.rows as usize > MAX_V4_VECTOR
        || value
            .weights
            .iter()
            .any(|weight| weight.unsigned_abs() > 1_000_000_000)
    {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_replica.shape",
        });
    }
    Ok(())
}

fn v4_tensor_replica_ack(value: &crate::V4TensorReplicaAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_replica_ack.job_id")?;
    v4_hash(&value.state_hash, "v4.tensor_replica_ack.hash")?;
    if value.plan_generation == 0 || value.state_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_replica_ack.generation",
        });
    }
    Ok(())
}

fn v4_optimizer_state_install(
    value: &crate::V4OptimizerStateInstall,
) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.optimizer_state.job_id")?;
    node_id(&value.source, "v4.optimizer_state.source")?;
    v4_hash(&value.state_hash, "v4.optimizer_state.hash")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.optimizer_generation == 0
        || value.state_generation == 0
        || value.sequence == 0
        || value.values.is_empty()
    {
        return Err(ValidationError::Invalid {
            field: "v4.optimizer_state.generation",
        });
    }
    v4_vector(&value.values, "v4.optimizer_state.values")
}

fn v4_optimizer_state_ack(value: &crate::V4OptimizerStateAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.optimizer_state_ack.job_id")?;
    v4_hash(&value.state_hash, "v4.optimizer_state_ack.hash")?;
    if value.plan_generation == 0 || value.optimizer_generation == 0 || value.state_generation == 0
    {
        return Err(ValidationError::Invalid {
            field: "v4.optimizer_state_ack.generation",
        });
    }
    Ok(())
}

fn v4_training_state(value: &crate::V4TrainingStateRecord) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.state.job_id")?;
    node_id(&value.coordinator, "v4.state.coordinator")?;
    v4_hash(&value.plan_hash, "v4.state.plan_hash")?;
    v4_hash(&value.branch, "v4.state.branch")?;
    v4_hash(&value.state_hash, "v4.state.hash")?;
    if value.plan_generation == 0
        || value.training_epoch == 0
        || value.membership_epoch == 0
        || value.coordination_term == 0
        || value.optimizer_generation == 0
        || value.checkpoint_generation == 0
    {
        return Err(ValidationError::Invalid {
            field: "v4.state.generation",
        });
    }
    if !(2..=MAX_V4_PLAN_WORKERS).contains(&value.workers.len()) {
        return Err(ValidationError::Invalid {
            field: "v4.state.workers",
        });
    }
    v4_node_list(&value.workers, "v4.state.workers", MAX_V4_PLAN_WORKERS)?;
    let worker_set = value
        .workers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    if value.shards.is_empty() || value.shards.len() > MAX_TRAINING_SHARDS {
        return Err(ValidationError::TooMany {
            field: "v4.state.shards",
            max: MAX_TRAINING_SHARDS,
        });
    }
    if value.data_progress.len() > MAX_V4_PLAN_WORKERS
        || (!value.data_progress.is_empty() && value.data_progress.len() != value.workers.len())
    {
        return Err(ValidationError::TooMany {
            field: "v4.state.data_progress",
            max: MAX_V4_PLAN_WORKERS,
        });
    }
    let mut shard_ids = std::collections::HashSet::new();
    for shard in &value.shards {
        if !shard_ids.insert(shard.shard_id)
            || shard
                .owners
                .iter()
                .chain(shard.replicas.iter())
                .any(|node| !worker_set.contains(node))
        {
            return Err(ValidationError::Invalid {
                field: "v4.state.shard.worker",
            });
        }
        if shard.owners.len() != 1 {
            return Err(ValidationError::Invalid {
                field: "v4.state.shard.owners",
            });
        }
        v4_node_list(&shard.owners, "v4.state.shard.owners", MAX_V4_PLAN_WORKERS)?;
        if shard.replicas.len() > MAX_TRAINING_REPLICAS {
            return Err(ValidationError::TooMany {
                field: "v4.state.shard.replicas",
                max: MAX_TRAINING_REPLICAS,
            });
        }
        if shard
            .replicas
            .iter()
            .any(|replica| shard.owners.contains(replica))
        {
            return Err(ValidationError::Invalid {
                field: "v4.state.shard.replica_owner_overlap",
            });
        }
        v4_hash(&shard.content_hash, "v4.state.shard.hash")?;
    }
    if let Some(checkpoint) = value.checkpoint {
        v4_hash(&checkpoint, "v4.state.checkpoint")?;
    }
    Ok(())
}

fn v4_optimizer_shard(value: &crate::V4OptimizerShardRecord) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.optimizer.job_id")?;
    node_id(&value.owner, "v4.optimizer.owner")?;
    v4_hash(&value.content_hash, "v4.optimizer.content_hash")?;
    v4_hash(&value.state_hash, "v4.optimizer.state_hash")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.optimizer_generation == 0
        || value.state_bytes == 0
        || value.state_bytes > MAX_TRAINING_STATE_BYTES
        || value.replicas.len() > MAX_TRAINING_REPLICAS
        || value.replicas.contains(&value.owner)
    {
        return Err(ValidationError::Invalid {
            field: "v4.optimizer.bounds",
        });
    }
    if !value.replicas.is_empty() {
        v4_node_list(
            &value.replicas,
            "v4.optimizer.replicas",
            MAX_TRAINING_REPLICAS,
        )?;
    }
    Ok(())
}

fn v4_checkpoint(value: &crate::V4CheckpointRecord) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.checkpoint.job_id")?;
    v4_hash(&value.branch, "v4.checkpoint.branch")?;
    v4_hash(&value.manifest_hash, "v4.checkpoint.manifest_hash")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.optimizer_generation == 0
        || value.membership_epoch == 0
        || value.checkpoint_generation == 0
        || value.shard_hashes.is_empty()
        || value.shard_hashes.len() > MAX_TRAINING_SHARDS
    {
        return Err(ValidationError::Invalid {
            field: "v4.checkpoint.generation",
        });
    }
    for hash in &value.shard_hashes {
        v4_hash(hash, "v4.checkpoint.shard_hash")?;
    }
    v4_node_list(
        &value.providers,
        "v4.checkpoint.providers",
        MAX_V4_PLAN_WORKERS,
    )?;
    if value.complete && value.providers.is_empty() {
        return Err(ValidationError::Invalid {
            field: "v4.checkpoint.providers",
        });
    }
    if let Some(parent) = value.parent {
        v4_hash(&parent, "v4.checkpoint.parent")?;
    }
    Ok(())
}

fn v4_state_ack(value: &crate::V4StateAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.state_ack.job_id")?;
    v4_hash(&value.state_hash, "v4.state_ack.hash")?;
    if value.plan_generation == 0 || value.generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.state_ack.generation",
        });
    }
    Ok(())
}

fn v4_shard_migration(value: &crate::V4ShardMigration) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.migration.job_id")?;
    node_id(&value.from, "v4.migration.from")?;
    node_id(&value.to, "v4.migration.to")?;
    v4_hash(&value.content_hash, "v4.migration.hash")?;
    if value.plan_generation == 0
        || value.ownership_generation == 0
        || value.from == value.to
        || value.state.len() > MAX_V4_BYTES
    {
        return Err(ValidationError::Invalid {
            field: "v4.migration.bounds",
        });
    }
    if matches!(value.phase, crate::V4ShardMigrationPhase::Prepare) && value.state.is_empty() {
        return Err(ValidationError::Empty {
            field: "v4.migration.prepare_state",
        });
    }
    if !matches!(value.phase, crate::V4ShardMigrationPhase::Prepare) && !value.state.is_empty() {
        return Err(ValidationError::Invalid {
            field: "v4.migration.commit_state",
        });
    }
    Ok(())
}

fn v4_shard_migration_ack(value: &crate::V4ShardMigrationAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.migration_ack.job_id")?;
    node_id(&value.owner, "v4.migration_ack.owner")?;
    v4_hash(&value.content_hash, "v4.migration_ack.hash")?;
    if value.plan_generation == 0 || value.ownership_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.migration_ack.generation",
        });
    }
    Ok(())
}

fn v4_shard_migration_request(
    value: &crate::V4ShardMigrationRequest,
) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.migration_request.request_id")?;
    training_job_id(&value.job_id, "v4.migration_request.job_id")?;
    node_id(&value.proposer, "v4.migration_request.proposer")?;
    node_id(&value.from, "v4.migration_request.from")?;
    node_id(&value.to, "v4.migration_request.to")?;
    node_id(&value.reply_to, "v4.migration_request.reply_to")?;
    v4_hash(&value.proposal_hash, "v4.migration_request.proposal_hash")?;
    v4_hash(&value.content_hash, "v4.migration_request.content_hash")?;
    if value.plan_generation == 0
        || value.ownership_generation == 0
        || value.from == value.to
        || value.reply_to != value.proposer
    {
        return Err(ValidationError::Invalid {
            field: "v4.migration_request.authorization",
        });
    }
    Ok(())
}

fn v4_shard_migration_result(value: &crate::V4ShardMigrationResult) -> Result<(), ValidationError> {
    training_job_id(&value.request_id, "v4.migration_result.request_id")?;
    training_job_id(&value.job_id, "v4.migration_result.job_id")?;
    node_id(&value.from, "v4.migration_result.from")?;
    node_id(&value.to, "v4.migration_result.to")?;
    v4_hash(&value.content_hash, "v4.migration_result.hash")?;
    if value.plan_generation == 0 || value.ownership_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.migration_result.generation",
        });
    }
    string(&value.reason, "v4.migration_result.reason")
}

fn v4_plan_proposal_ack(value: &crate::V4PlanProposalAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.plan_proposal_ack.job_id")?;
    v4_hash(&value.proposal_hash, "v4.plan_proposal_ack.hash")?;
    if value.plan_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.plan_proposal_ack.generation",
        });
    }
    string(&value.reason, "v4.plan_proposal_ack.reason")
}

fn v4_plan_ack(value: &crate::V4PlanAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.plan_ack.job_id")?;
    v4_hash(&value.plan_hash, "v4.plan_ack.hash")?;
    if value.plan_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.plan_ack.generation",
        });
    }
    string(&value.reason, "v4.plan_ack.reason")
}

fn v4_tensor_install(value: &crate::V4TensorInstall) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_install.job_id")?;
    v4_hash(&value.state_hash, "v4.tensor_install.hash")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.rows == 0
        || value.cols == 0
        || value.rows as usize * value.cols as usize != value.weights.len()
        || value.weights.len() > MAX_V4_VECTOR * MAX_V4_VECTOR
        || value.row_offset as usize + value.rows as usize > MAX_V4_VECTOR
    {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_install.shape",
        });
    }
    if value
        .weights
        .iter()
        .any(|weight| weight.unsigned_abs() > 1_000_000_000)
    {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_install.weights",
        });
    }
    Ok(())
}

fn v4_tensor_install_ack(value: &crate::V4TensorInstallAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_install_ack.job_id")?;
    v4_hash(&value.state_hash, "v4.tensor_install_ack.hash")?;
    if value.plan_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_install_ack.generation",
        });
    }
    Ok(())
}

fn v4_tensor_forward(value: &crate::V4TensorForward) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_forward.job_id")?;
    training_job_id(&value.request_id, "v4.tensor_forward.request_id")?;
    if value.plan_generation == 0 || value.model_generation == 0 || value.sequence == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_forward.generation",
        });
    }
    v4_vector(&value.input, "v4.tensor_forward.input")
}

fn v4_tensor_forward_result(value: &crate::V4TensorForwardResult) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_forward_result.job_id")?;
    training_job_id(&value.request_id, "v4.tensor_forward_result.request_id")?;
    v4_hash(&value.state_hash, "v4.tensor_forward_result.hash")?;
    if value.plan_generation == 0 || value.sequence == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_forward_result.generation",
        });
    }
    v4_vector(&value.output, "v4.tensor_forward_result.output")
}

fn v4_tensor_backward(value: &crate::V4TensorBackward) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_backward.job_id")?;
    training_job_id(&value.request_id, "v4.tensor_backward.request_id")?;
    if value.plan_generation == 0
        || value.model_generation == 0
        || value.sequence == 0
        || value.learning_rate_micros <= 0
        || value.learning_rate_micros > 1_000_000
    {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_backward.generation",
        });
    }
    v4_vector(&value.input, "v4.tensor_backward.input")?;
    v4_vector(&value.upstream, "v4.tensor_backward.upstream")
}

fn v4_tensor_backward_result(value: &crate::V4TensorBackwardResult) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.tensor_backward_result.job_id")?;
    training_job_id(&value.request_id, "v4.tensor_backward_result.request_id")?;
    v4_hash(&value.state_hash, "v4.tensor_backward_result.hash")?;
    if value.plan_generation == 0 || value.sequence == 0 || value.state_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_backward_result.generation",
        });
    }
    v4_vector(
        &value.weight_gradient,
        "v4.tensor_backward_result.weight_gradient",
    )?;
    v4_vector(
        &value.input_gradient,
        "v4.tensor_backward_result.input_gradient",
    )?;
    if let Some(hash) = value.optimizer_state_hash {
        v4_hash(&hash, "v4.tensor_backward_result.optimizer_hash")?;
        if value.optimizer_state_generation == 0 {
            return Err(ValidationError::Invalid {
                field: "v4.tensor_backward_result.optimizer_generation",
            });
        }
    } else if value.optimizer_state_generation != 0 {
        return Err(ValidationError::Invalid {
            field: "v4.tensor_backward_result.optimizer_hash",
        });
    }
    Ok(())
}

fn v4_pipeline_install(value: &crate::V4PipelineInstall) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_install.job_id")?;
    v4_hash(&value.state_hash, "v4.pipeline_install.hash")?;
    if value.plan_generation == 0
        || value.stage_count < 2
        || value.stage_count > MAX_V4_STAGES as u16
        || value.stage_id >= value.stage_count
        || value.coefficient.unsigned_abs() > 1_000_000_000
        || value.bias.unsigned_abs() > 1_000_000_000
    {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_install.stage",
        });
    }
    Ok(())
}

fn v4_pipeline_install_ack(value: &crate::V4PipelineInstallAck) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_install_ack.job_id")?;
    v4_hash(&value.state_hash, "v4.pipeline_install_ack.hash")?;
    if value.plan_generation == 0 || value.stage_id >= MAX_V4_STAGES as u16 {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_install_ack.stage",
        });
    }
    Ok(())
}

fn v4_pipeline_forward(value: &crate::V4PipelineForward) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_forward.job_id")?;
    training_job_id(&value.request_id, "v4.pipeline_forward.request_id")?;
    if value.plan_generation == 0
        || value.stage_count < 2
        || value.stage_count > MAX_V4_STAGES as u16
        || value.stage_id >= value.stage_count
        || value.deadline_ms == 0
        || value.deadline_ms > 300_000
    {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_forward.stage",
        });
    }
    v4_vector(&value.activation, "v4.pipeline_forward.activation")
}

fn v4_pipeline_forward_result(
    value: &crate::V4PipelineForwardResult,
) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_forward_result.job_id")?;
    training_job_id(&value.request_id, "v4.pipeline_forward_result.request_id")?;
    if value.plan_generation == 0 || value.stage_id >= MAX_V4_STAGES as u16 {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_forward_result.stage",
        });
    }
    v4_vector(&value.activation, "v4.pipeline_forward_result.activation")?;
    v4_hash(&value.state_hash, "v4.pipeline_forward_result.hash")
}

fn v4_pipeline_backward(value: &crate::V4PipelineBackward) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_backward.job_id")?;
    training_job_id(&value.request_id, "v4.pipeline_backward.request_id")?;
    if value.plan_generation == 0
        || value.stage_count < 2
        || value.stage_count > MAX_V4_STAGES as u16
        || value.stage_id >= value.stage_count
        || value.deadline_ms == 0
        || value.deadline_ms > 300_000
    {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_backward.stage",
        });
    }
    v4_vector(&value.gradient, "v4.pipeline_backward.gradient")
}

fn v4_pipeline_backward_result(
    value: &crate::V4PipelineBackwardResult,
) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.pipeline_backward_result.job_id")?;
    training_job_id(&value.request_id, "v4.pipeline_backward_result.request_id")?;
    if value.plan_generation == 0 || value.stage_id >= MAX_V4_STAGES as u16 {
        return Err(ValidationError::Invalid {
            field: "v4.pipeline_backward_result.stage",
        });
    }
    v4_vector(&value.gradient, "v4.pipeline_backward_result.gradient")?;
    v4_hash(&value.state_hash, "v4.pipeline_backward_result.hash")
}

fn v4_collective_contribute(value: &crate::V4CollectiveContribute) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.collective.job_id")?;
    training_job_id(&value.request_id, "v4.collective.request_id")?;
    node_id(&value.contributor, "v4.collective.contributor")?;
    node_id(&value.aggregator, "v4.collective.aggregator")?;
    if let Some(parent) = value.parent {
        node_id(&parent, "v4.collective.parent")?;
    }
    node_id(&value.reply_to, "v4.collective.reply_to")?;
    if value.plan_generation == 0
        || value.generation == 0
        || value.group_id >= value.expected_groups
        || value.expected_contributors == 0
        || value.expected_contributors > MAX_V4_PLAN_WORKERS as u16
        || value.expected_groups == 0
        || value.expected_groups > MAX_V4_GROUPS as u16
    {
        return Err(ValidationError::Invalid {
            field: "v4.collective.generation",
        });
    }
    v4_vector(&value.values, "v4.collective.values")
}

fn v4_collective_aggregate(value: &crate::V4CollectiveAggregate) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.collective_aggregate.job_id")?;
    training_job_id(&value.request_id, "v4.collective_aggregate.request_id")?;
    node_id(&value.aggregator, "v4.collective_aggregate.aggregator")?;
    node_id(&value.reply_to, "v4.collective_aggregate.reply_to")?;
    if value.plan_generation == 0
        || value.generation == 0
        || value.group_id >= value.expected_groups
        || value.contributors == 0
        || value.expected_groups == 0
        || value.expected_groups > MAX_V4_GROUPS as u16
    {
        return Err(ValidationError::Invalid {
            field: "v4.collective_aggregate.generation",
        });
    }
    v4_vector(&value.values, "v4.collective_aggregate.values")
}

fn v4_collective_result(value: &crate::V4CollectiveResult) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.collective_result.job_id")?;
    training_job_id(&value.request_id, "v4.collective_result.request_id")?;
    if value.plan_generation == 0
        || value.generation == 0
        || value.group_count == 0
        || value.max_fan_in == 0
        || value.root_received_bytes > value.contributor_bytes
    {
        return Err(ValidationError::Invalid {
            field: "v4.collective_result.generation",
        });
    }
    v4_vector(&value.values, "v4.collective_result.values")
}

fn v4_branch(value: &crate::V4TrainingBranch) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.branch.job_id")?;
    node_id(&value.created_by, "v4.branch.created_by")?;
    artifact_id(&value.branch, "v4.branch.id")?;
    if value.parent_generation == 0
        || value.model_generation == 0
        || value.optimizer_generation == 0
        || value.plan_generation == 0
    {
        return Err(ValidationError::Invalid {
            field: "v4.branch.generation",
        });
    }
    Ok(())
}

fn v4_reconcile(value: &crate::V4ReconcileRequest) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.reconcile.job_id")?;
    training_job_id(&value.request_id, "v4.reconcile.request_id")?;
    if value.plan_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.reconcile.plan_generation",
        });
    }
    v4_branch(&value.left)?;
    v4_branch(&value.right)?;
    if value.left.job_id != value.right.job_id
        || value.left.parent_generation != value.right.parent_generation
        || value.left.model_generation != value.right.model_generation
        || value.left.plan_generation != value.right.plan_generation
        || value.left.branch == value.right.branch
    {
        return Err(ValidationError::Invalid {
            field: "v4.reconcile.branch_lineage",
        });
    }
    if value.left_value.unsigned_abs() > 1_000_000_000
        || value.right_value.unsigned_abs() > 1_000_000_000
    {
        return Err(ValidationError::Invalid {
            field: "v4.reconcile.values",
        });
    }
    Ok(())
}

fn v4_reconcile_result(value: &crate::V4ReconcileResult) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.reconcile_result.job_id")?;
    training_job_id(&value.request_id, "v4.reconcile_result.request_id")?;
    if value.plan_generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.reconcile_result.plan_generation",
        });
    }
    if let Some(branch) = value.branch {
        artifact_id(&branch, "v4.reconcile_result.branch")?;
    }
    string(&value.reason, "v4.reconcile_result.reason")
}

fn v4_byzantine_update(value: &crate::V4ByzantineUpdate) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.byzantine.job_id")?;
    training_job_id(&value.request_id, "v4.byzantine.request_id")?;
    node_id(&value.worker, "v4.byzantine.worker")?;
    node_id(&value.aggregator, "v4.byzantine.aggregator")?;
    node_id(&value.reply_to, "v4.byzantine.reply_to")?;
    if value.plan_generation == 0
        || value.generation == 0
        || value.sequence == 0
        || value.expected_updates == 0
        || value.expected_updates > MAX_V4_PLAN_WORKERS as u16
    {
        return Err(ValidationError::Invalid {
            field: "v4.byzantine.generation",
        });
    }
    v4_vector(&value.values, "v4.byzantine.values")
}

fn v4_byzantine_result(value: &crate::V4ByzantineResult) -> Result<(), ValidationError> {
    training_job_id(&value.job_id, "v4.byzantine_result.job_id")?;
    training_job_id(&value.request_id, "v4.byzantine_result.request_id")?;
    if value.plan_generation == 0 || value.generation == 0 {
        return Err(ValidationError::Invalid {
            field: "v4.byzantine_result.generation",
        });
    }
    v4_vector(&value.aggregate, "v4.byzantine_result.aggregate")?;
    v4_node_list(
        &value.accepted_workers,
        "v4.byzantine_result.accepted_workers",
        MAX_V4_PLAN_WORKERS,
    )?;
    v4_node_list(
        &value.rejected_workers,
        "v4.byzantine_result.rejected_workers",
        MAX_V4_PLAN_WORKERS,
    )?;
    if value
        .accepted_workers
        .iter()
        .any(|worker| value.rejected_workers.contains(worker))
        || value.accepted_workers.len() + value.rejected_workers.len() > MAX_V4_PLAN_WORKERS
        || value.robust == matches!(value.policy, crate::V4ByzantinePolicy::Mean)
    {
        return Err(ValidationError::Invalid {
            field: "v4.byzantine_result.workers",
        });
    }
    Ok(())
}

fn training_v4_message(value: &TrainingV4Message) -> Result<(), ValidationError> {
    match value {
        TrainingV4Message::Plan(value) => v4_plan(value),
        TrainingV4Message::PlanAck(value) => v4_plan_ack(value),
        TrainingV4Message::PlanProposal(value) => v4_plan(value),
        TrainingV4Message::PlanProposalAck(value) => v4_plan_proposal_ack(value),
        TrainingV4Message::StateRecord(value) => v4_training_state(value),
        TrainingV4Message::OptimizerShard(value) => v4_optimizer_shard(value),
        TrainingV4Message::CheckpointRecord(value) => v4_checkpoint(value),
        TrainingV4Message::StateAck(value) => v4_state_ack(value),
        TrainingV4Message::ShardMigration(value) => v4_shard_migration(value),
        TrainingV4Message::ShardMigrationAck(value) => v4_shard_migration_ack(value),
        TrainingV4Message::ShardMigrationRequest(value) => v4_shard_migration_request(value),
        TrainingV4Message::ShardMigrationResult(value) => v4_shard_migration_result(value),
        TrainingV4Message::TensorInstall(value) => v4_tensor_install(value),
        TrainingV4Message::TensorInstallAck(value) => v4_tensor_install_ack(value),
        TrainingV4Message::TensorForward(value) => v4_tensor_forward(value),
        TrainingV4Message::TensorForwardResult(value) => v4_tensor_forward_result(value),
        TrainingV4Message::TensorBackward(value) => v4_tensor_backward(value),
        TrainingV4Message::TensorBackwardResult(value) => v4_tensor_backward_result(value),
        TrainingV4Message::PipelineInstall(value) => v4_pipeline_install(value),
        TrainingV4Message::PipelineInstallAck(value) => v4_pipeline_install_ack(value),
        TrainingV4Message::PipelineForward(value) => v4_pipeline_forward(value),
        TrainingV4Message::PipelineForwardResult(value) => v4_pipeline_forward_result(value),
        TrainingV4Message::PipelineBackward(value) => v4_pipeline_backward(value),
        TrainingV4Message::PipelineBackwardResult(value) => v4_pipeline_backward_result(value),
        TrainingV4Message::CollectiveContribute(value) => v4_collective_contribute(value),
        TrainingV4Message::CollectiveAggregate(value) => v4_collective_aggregate(value),
        TrainingV4Message::CollectiveResult(value) => v4_collective_result(value),
        TrainingV4Message::Branch(value) => v4_branch(value),
        TrainingV4Message::Reconcile(value) => v4_reconcile(value),
        TrainingV4Message::ReconcileResult(value) => v4_reconcile_result(value),
        TrainingV4Message::ByzantineUpdate(value) => v4_byzantine_update(value),
        TrainingV4Message::ByzantineResult(value) => v4_byzantine_result(value),
        TrainingV4Message::IntegratedStart(value) => v4_integrated_start(value),
        TrainingV4Message::IntegratedAck(value) => v4_integrated_ack(value),
        TrainingV4Message::IntegratedState(value) => v4_integrated_state(value),
        TrainingV4Message::IntegratedStateAck(value) => v4_integrated_state_ack(value),
        TrainingV4Message::IntegratedProbe(value) => v4_integrated_probe(value),
        TrainingV4Message::IntegratedProbeAck(value) => v4_integrated_probe_ack(value),
        TrainingV4Message::IntegratedElectionRequest(value) => {
            v4_integrated_election_request(value)
        }
        TrainingV4Message::IntegratedElectionVote(value) => v4_integrated_election_vote(value),
        TrainingV4Message::IntegratedResult(value) => v4_integrated_result(value),
        TrainingV4Message::TensorReplica(value) => v4_tensor_replica(value),
        TrainingV4Message::TensorReplicaAck(value) => v4_tensor_replica_ack(value),
        TrainingV4Message::OptimizerStateInstall(value) => v4_optimizer_state_install(value),
        TrainingV4Message::OptimizerStateAck(value) => v4_optimizer_state_ack(value),
    }
}

fn training_v5_message(value: &crate::TrainingV5Message) -> Result<(), ValidationError> {
    match value {
        crate::TrainingV5Message::CapabilityChallenge(value) => v5_challenge(value),
        crate::TrainingV5Message::CapabilityChallengeResult(value) => v5_challenge_result(value),
        crate::TrainingV5Message::CapabilityEvidence(value) => v5_evidence(value),
    }
}

impl Capability {
    pub fn validate(&self) -> Result<(), ValidationError> {
        capability(self)
    }
}

impl PeerRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        peer(self)
    }
}

impl SignedAnnouncement {
    pub fn validate(&self) -> Result<(), ValidationError> {
        peer(&self.record)?;
        signature(&self.signature, "announcement.signature")
    }
}

impl Hello {
    pub fn validate(&self) -> Result<(), ValidationError> {
        peer(&self.record)?;
        signature(&self.signature, "hello.signature")
    }
}

impl PeerExchange {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.peers.len() > MAX_PEERS {
            return Err(ValidationError::TooMany {
                field: "peer_exchange.peers",
                max: MAX_PEERS,
            });
        }
        for item in &self.peers {
            item.validate()?;
        }
        Ok(())
    }
}

impl AddressUpdate {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.sequence == 0 {
            return Err(ValidationError::Invalid {
                field: "address_update.sequence",
            });
        }
        if self.addresses.len() > MAX_ADDRESSES {
            return Err(ValidationError::TooMany {
                field: "address_update.addresses",
                max: MAX_ADDRESSES,
            });
        }
        for address in &self.addresses {
            address_record(address)?;
        }
        signature(&self.signature, "address_update.signature")
    }
}

impl AddressObservation {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.address.is_empty() || self.address.len() > MAX_ADDRESS_LEN {
            return Err(ValidationError::TooLong {
                field: "address_observation.address",
                max: MAX_ADDRESS_LEN,
            });
        }
        if self.expires_at == 0 {
            return Err(ValidationError::Invalid {
                field: "address_observation.expires_at",
            });
        }
        signature(&self.signature, "address_observation.signature")
    }
}

impl JobRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        string(&self.capability, "job.capability")?;
        if self.capability.len() > 128 {
            return Err(ValidationError::TooLong {
                field: "job.capability",
                max: 128,
            });
        }
        bytes(&self.input, MAX_JOB_INPUT, "job.input")?;
        if self.deadline_ms == 0 || self.deadline_ms > 24 * 60 * 60 * 1000 {
            return Err(ValidationError::Invalid {
                field: "job.deadline_ms",
            });
        }
        if self.max_output_bytes == 0 || self.max_output_bytes as usize > MAX_JOB_OUTPUT {
            return Err(ValidationError::Invalid {
                field: "job.max_output_bytes",
            });
        }
        Ok(())
    }
}

impl JobUpdate {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.sequence == 0 {
            return Err(ValidationError::Invalid {
                field: "job.sequence",
            });
        }
        let state_matches = match &self.update {
            crate::JobUpdateKind::Accepted { state } => *state == self.state,
            crate::JobUpdateKind::Started | crate::JobUpdateKind::Chunk { .. } => {
                self.state == crate::JobState::Running
            }
            crate::JobUpdateKind::Succeeded { .. } => self.state == crate::JobState::Succeeded,
            crate::JobUpdateKind::Rejected { .. } => self.state == crate::JobState::Rejected,
            crate::JobUpdateKind::Cancelled => self.state == crate::JobState::Cancelled,
            crate::JobUpdateKind::TimedOut => self.state == crate::JobState::TimedOut,
            crate::JobUpdateKind::Failed { .. } => self.state == crate::JobState::Failed,
        };
        if !state_matches {
            return Err(ValidationError::Invalid {
                field: "job.state_update",
            });
        }
        match &self.update {
            crate::JobUpdateKind::Chunk { data, .. } => {
                bytes(data, MAX_ARTIFACT_CHUNK, "job.chunk")?
            }
            crate::JobUpdateKind::Succeeded {
                output, evidence, ..
            } => {
                bytes(output, MAX_JOB_OUTPUT, "job.output")?;
                if let Some(value) = evidence {
                    string(&value.score_scale, "evidence.score_scale")?;
                }
            }
            crate::JobUpdateKind::Rejected { code, message }
            | crate::JobUpdateKind::Failed { code, message } => {
                string(code, "job.error.code")?;
                string(message, "job.error.message")?;
            }
            crate::JobUpdateKind::Accepted { .. }
            | crate::JobUpdateKind::Started
            | crate::JobUpdateKind::Cancelled
            | crate::JobUpdateKind::TimedOut => {}
        }
        Ok(())
    }
}

impl ArtifactRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.max_bytes == 0 || self.max_bytes as usize > MAX_ARTIFACT_CHUNK {
            return Err(ValidationError::Invalid {
                field: "artifact_request.max_bytes",
            });
        }
        Ok(())
    }
}

impl ArtifactChunk {
    pub fn validate(&self) -> Result<(), ValidationError> {
        bytes(&self.data, MAX_ARTIFACT_CHUNK, "artifact_chunk.data")
    }
}

impl ArtifactTransferRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.max_bytes == 0 || self.max_bytes as usize > MAX_ARTIFACT_CHUNK {
            return Err(ValidationError::Invalid {
                field: "artifact_transfer_request.max_bytes",
            });
        }
        // Zero means that the requester is resuming without a manifest and asks
        // the provider to return the authoritative size in the first chunk.
        Ok(())
    }
}

impl ArtifactTransferChunk {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.total_size == 0 || self.offset > self.total_size {
            return Err(ValidationError::Invalid {
                field: "artifact_transfer_chunk.range",
            });
        }
        bytes(
            &self.data,
            MAX_ARTIFACT_CHUNK,
            "artifact_transfer_chunk.data",
        )?;
        if self.data.is_empty() && !self.final_chunk {
            return Err(ValidationError::Invalid {
                field: "artifact_transfer_chunk.data",
            });
        }
        if self.offset.saturating_add(self.data.len() as u64) > self.total_size {
            return Err(ValidationError::Invalid {
                field: "artifact_transfer_chunk.range",
            });
        }
        Ok(())
    }
}

impl KeyRotation {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.old_node_id == self.new_node_id
            || self.old_node_id != crate::NodeId::from_public_key(&self.old_public_key)
            || self.new_node_id != crate::NodeId::from_public_key(&self.new_public_key)
            || self.sequence == 0
            || self.valid_from == 0
            || self.valid_until < self.valid_from
        {
            return Err(ValidationError::Invalid {
                field: "key_rotation.identity_or_validity",
            });
        }
        signature(&self.old_signature, "key_rotation.old_signature")?;
        signature(&self.new_signature, "key_rotation.new_signature")
    }
}

impl RelayEnvelope {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.origin != crate::NodeId::from_public_key(&self.origin_public_key)
            || self.origin == self.target
            || self.payload.is_empty()
            || self.payload.len() > MAX_RELAY_PAYLOAD
        {
            return Err(ValidationError::Invalid {
                field: "relay_envelope.identity_or_payload",
            });
        }
        signature(&self.signature, "relay_envelope.signature")
    }
}

impl ArtifactManifest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        string(&self.format, "artifact.format")?;
        manifest_shards(&self.shards, "artifact.shards")
    }
}

impl ModelManifest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        string(&self.identity, "model.identity")?;
        string(&self.format, "model.format")?;
        manifest_shards(&self.weights, "model.weights")?;
        if self.adapters.len() > MAX_CAPABILITIES {
            return Err(ValidationError::TooMany {
                field: "model.adapters",
                max: MAX_CAPABILITIES,
            });
        }
        if self.capabilities.len() > MAX_CAPABILITIES {
            return Err(ValidationError::TooMany {
                field: "model.capabilities",
                max: MAX_CAPABILITIES,
            });
        }
        for item in &self.capabilities {
            string(item, "model.capability")?;
        }
        if self.runtime_requirements.len() > MAX_METADATA {
            return Err(ValidationError::TooMany {
                field: "model.runtime_requirements",
                max: MAX_METADATA,
            });
        }
        for item in &self.runtime_requirements {
            string(&item.key, "model.runtime_requirement.key")?;
            string(&item.value, "model.runtime_requirement.value")?;
        }
        if let Some(path) = &self.local_path {
            string(path, "model.local_path")?;
        }
        Ok(())
    }
}

impl DatasetManifest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        string(&self.identity, "dataset.identity")?;
        string(&self.format, "dataset.format")?;
        string(&self.sample_policy, "dataset.sample_policy")?;
        manifest_shards(&self.shards, "dataset.shards")
    }
}

impl CheckpointManifest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if !self.complete_weights {
            return Err(ValidationError::Invalid {
                field: "checkpoint.complete_weights",
            });
        }
        Ok(())
    }
}

impl TrainingPlan {
    pub fn validate(&self) -> Result<(), ValidationError> {
        if self.workers == 0
            || self.workers > 1024
            || self.max_steps == 0
            || self.checkpoint_every == 0
        {
            return Err(ValidationError::Invalid {
                field: "training.plan.bounds",
            });
        }
        if let Some(capability) = &self.evaluation_capability {
            string(capability, "training.evaluation_capability")?;
        }
        Ok(())
    }
}

impl DhtRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        dht_record(self)
    }
}

impl DhtRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match &self.request {
            crate::DhtRequestKind::Ping | crate::DhtRequestKind::FindNode { .. } => Ok(()),
            crate::DhtRequestKind::Get { .. } => Ok(()),
            crate::DhtRequestKind::Put { record } => record.validate(),
        }
    }
}

impl crate::V4TrainingStateRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_training_state(self)
    }
}

impl crate::V4TrainingPlan {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_plan(self)
    }
}

impl crate::V4OptimizerShardRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_optimizer_shard(self)
    }
}

impl crate::V4CheckpointRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_checkpoint(self)
    }
}

impl crate::V4ExecutionGraph {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_execution_graph(self)
    }
}

impl crate::V4IntegratedStateRecord {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_state(self)
    }
}

impl crate::V4IntegratedStart {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_start(self)
    }
}

impl crate::V4IntegratedAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_ack(self)
    }
}

impl crate::V4IntegratedStateAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_state_ack(self)
    }
}

impl crate::V4IntegratedProbe {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_probe(self)
    }
}

impl crate::V4IntegratedProbeAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_probe_ack(self)
    }
}

impl crate::V4IntegratedElectionRequest {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_election_request(self)
    }
}

impl crate::V4IntegratedElectionVote {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_election_vote(self)
    }
}

impl crate::V4IntegratedResult {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_integrated_result(self)
    }
}

impl crate::V4TensorReplica {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_tensor_replica(self)
    }
}

impl crate::V4TensorReplicaAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_tensor_replica_ack(self)
    }
}

impl crate::V4OptimizerStateInstall {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_optimizer_state_install(self)
    }
}

impl crate::V4OptimizerStateAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_optimizer_state_ack(self)
    }
}

impl crate::V4StateAck {
    pub fn validate(&self) -> Result<(), ValidationError> {
        v4_state_ack(self)
    }
}

impl DhtResponse {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match &self.response {
            crate::DhtResponseKind::Nodes { contacts } => {
                if contacts.len() > MAX_DHT_CONTACTS {
                    return Err(ValidationError::TooMany {
                        field: "dht.response.contacts",
                        max: MAX_DHT_CONTACTS,
                    });
                }
                contacts.iter().try_for_each(SignedAnnouncement::validate)
            }
            crate::DhtResponseKind::Records { records } => {
                if records.len() > MAX_DHT_RECORDS {
                    return Err(ValidationError::TooMany {
                        field: "dht.response.records",
                        max: MAX_DHT_RECORDS,
                    });
                }
                records.iter().try_for_each(DhtRecord::validate)
            }
            crate::DhtResponseKind::Error { message, .. } => string(message, "dht.error.message"),
            crate::DhtResponseKind::Pong
            | crate::DhtResponseKind::Stored
            | crate::DhtResponseKind::NotFound => Ok(()),
        }
    }
}

impl SignedEvidence {
    pub fn validate(&self) -> Result<(), ValidationError> {
        signed_evidence(self)
    }
}

impl ProtocolError {
    pub fn validate(&self) -> Result<(), ValidationError> {
        string(&self.message, "protocol_error.message")
    }
}

impl Message {
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Hello(value) => value.validate(),
            Self::PeerExchange(value) => value.validate(),
            Self::Announcement(value) => value.validate(),
            Self::JobRequest(value) => value.validate(),
            Self::JobUpdate(value) => value.validate(),
            Self::CancelJob(value) => string(&value.reason, "cancel.reason"),
            Self::ArtifactRequest(value) => value.validate(),
            Self::ArtifactChunk(value) => value.validate(),
            Self::Ping(_) | Self::Pong(_) => Ok(()),
            Self::Goodbye { reason } => reason
                .as_deref()
                .map_or(Ok(()), |value| string(value, "goodbye.reason")),
            Self::Error(value) => value.validate(),
            Self::AddressUpdate(value) => value.validate(),
            Self::AddressObservation(value) => value.validate(),
            Self::ArtifactTransferRequest(value) => value.validate(),
            Self::ArtifactTransferChunk(value) => value.validate(),
            Self::KeyRotation(value) => value.validate(),
            Self::RelayEnvelope(value) => value.validate(),
            Self::DhtRequest(value) => value.validate(),
            Self::DhtResponse(value) => value.validate(),
            Self::Training(value) => training_message(value),
            Self::TrainingV4(value) => training_v4_message(value),
            Self::TrainingV5(value) => training_v5_message(value),
        }
    }
}
