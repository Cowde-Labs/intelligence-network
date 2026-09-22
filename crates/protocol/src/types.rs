use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const PROTOCOL_MAJOR: u16 = 1;
pub const PROTOCOL_MINOR: u16 = 6;

macro_rules! fixed_id {
    ($name:ident, $len:expr) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            Serialize,
            Deserialize,
        )]
        pub struct $name(pub [u8; $len]);

        impl $name {
            pub const fn from_bytes(bytes: [u8; $len]) -> Self {
                Self(bytes)
            }

            pub const fn as_bytes(&self) -> &[u8; $len] {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", hex::encode(self.0))
            }
        }
    };
}

fixed_id!(NodeId, 32);
fixed_id!(JobId, 16);
fixed_id!(ArtifactId, 32);
fixed_id!(RequestId, 16);
fixed_id!(DhtKey, 32);

impl NodeId {
    pub fn from_public_key(public_key: &[u8; 32]) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"intelligence-network/node-id/v1\0");
        hasher.update(public_key);
        Self(*hasher.finalize().as_bytes())
    }
}

impl ArtifactId {
    pub fn from_bytes_hashed(bytes: &[u8]) -> Self {
        Self(*blake3::hash(bytes).as_bytes())
    }
}

impl DhtKey {
    pub fn for_name(namespace: DhtNamespace, name: &str) -> Self {
        let mut hasher = Hasher::new();
        hasher.update(b"intelligence-network/dht-key/v1\0");
        hasher.update(&[namespace.tag()]);
        hasher.update(name.as_bytes());
        Self(*hasher.finalize().as_bytes())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VersionRange {
    pub major: u16,
    pub min_minor: u16,
    pub max_minor: u16,
}

impl VersionRange {
    pub const fn current() -> Self {
        Self {
            major: PROTOCOL_MAJOR,
            min_minor: 0,
            max_minor: PROTOCOL_MINOR,
        }
    }

    pub const fn supports(&self, major: u16, minor: u16) -> bool {
        self.major == major && minor >= self.min_minor && minor <= self.max_minor
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetadataEntry {
    pub key: String,
    pub value: String,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_input_bytes: u32,
    pub max_output_bytes: u32,
    pub memory_bytes: u64,
    pub cpu_millis: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CapabilityEvidence {
    Claimed,
    Observed {
        successful_jobs: u32,
        failed_jobs: u32,
        last_observed_at: u64,
    },
    Verified {
        evaluation_id: ArtifactId,
        score_basis: String,
        observed_at: u64,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Capability {
    pub name: String,
    pub version: u16,
    pub model: Option<String>,
    pub resources: ResourceLimits,
    pub evidence: CapabilityEvidence,
    pub expires_at: u64,
    pub metadata: Vec<MetadataEntry>,
    /// Optional typed V5 backend advertisement.  Empty preserves V1-V4
    /// capability records and does not make accelerator support mandatory.
    #[serde(default)]
    pub compute_backends: Vec<BackendCapabilities>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeerRecord {
    pub node_id: NodeId,
    pub public_key: [u8; 32],
    pub addresses: Vec<String>,
    pub capabilities: Vec<Capability>,
    pub announced_at: u64,
    pub expires_at: u64,
    pub observed_latency_ms: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignedAnnouncement {
    pub record: PeerRecord,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub nonce: [u8; 16],
    pub supported: VersionRange,
    pub record: PeerRecord,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeerExchange {
    pub peers: Vec<SignedAnnouncement>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum DhtNamespace {
    Peer,
    Capability,
    Artifact,
    Model,
    Dataset,
    Relay,
    Evidence,
}

impl DhtNamespace {
    pub const fn tag(self) -> u8 {
        match self {
            Self::Peer => 1,
            Self::Capability => 2,
            Self::Artifact => 3,
            Self::Model => 4,
            Self::Dataset => 5,
            Self::Relay => 6,
            Self::Evidence => 7,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DhtRecord {
    pub namespace: DhtNamespace,
    pub key: DhtKey,
    pub owner: NodeId,
    pub owner_public_key: [u8; 32],
    pub sequence: u64,
    pub expires_at: u64,
    pub value: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DhtRequestKind {
    Ping,
    FindNode {
        target: DhtKey,
    },
    Get {
        namespace: DhtNamespace,
        key: DhtKey,
    },
    Put {
        record: DhtRecord,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DhtRequest {
    pub request_id: RequestId,
    pub origin: NodeId,
    pub request: DhtRequestKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DhtResponseKind {
    Pong,
    Nodes { contacts: Vec<SignedAnnouncement> },
    Records { records: Vec<DhtRecord> },
    Stored,
    NotFound,
    Error { code: ErrorCode, message: String },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DhtResponse {
    pub request_id: RequestId,
    pub responder: NodeId,
    pub response: DhtResponseKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AddressTransport {
    QuicUdp,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum AddressSource {
    Configured,
    Observed,
    Learned,
    Local,
    Relay,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ReachabilityState {
    PubliclyReachable,
    DirectlyReachable,
    NatUnknown,
    NatRestricted,
    RelayRequired,
    Unreachable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AddressRecord {
    pub address: String,
    pub transport: AddressTransport,
    pub source: AddressSource,
    pub expires_at: u64,
    pub confidence: u8,
    pub relay: Option<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AddressUpdate {
    pub node_id: NodeId,
    pub sequence: u64,
    pub addresses: Vec<AddressRecord>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AddressObservation {
    pub subject: NodeId,
    pub address: String,
    pub observed_by: NodeId,
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum JobKind {
    Inference,
    Evaluation,
    Training,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    pub allow_input_transfer: bool,
    pub allow_output_persistence: bool,
    pub require_trusted_peer: bool,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self {
            allow_input_transfer: true,
            allow_output_persistence: false,
            require_trusted_peer: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct JobRequest {
    pub job_id: JobId,
    pub origin: NodeId,
    pub kind: JobKind,
    pub capability: String,
    pub model: Option<ArtifactId>,
    pub input: Vec<u8>,
    pub deadline_ms: u64,
    pub max_output_bytes: u32,
    pub privacy: PrivacyPolicy,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum JobState {
    Received,
    Validated,
    Admitted,
    Queued,
    Running,
    Succeeded,
    Rejected,
    Cancelled,
    TimedOut,
    Failed,
}

impl JobState {
    pub const fn terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Rejected | Self::Cancelled | Self::TimedOut | Self::Failed
        )
    }

    pub const fn can_transition(self, next: Self) -> bool {
        matches!(
            (self, next),
            (Self::Received, Self::Validated | Self::Rejected)
                | (Self::Validated, Self::Admitted | Self::Rejected)
                | (Self::Admitted, Self::Queued | Self::Rejected)
                | (Self::Admitted, Self::Running)
                | (
                    Self::Queued,
                    Self::Running | Self::Cancelled | Self::TimedOut | Self::Failed
                )
                | (
                    Self::Running,
                    Self::Succeeded | Self::Cancelled | Self::TimedOut | Self::Failed
                )
        )
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub evaluator: NodeId,
    pub evaluation_id: ArtifactId,
    pub score: i64,
    pub score_scale: String,
    pub result_hash: ArtifactId,
    pub observed_at: u64,
    pub verified: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum EvidenceKind {
    JobCompleted,
    Evaluation,
    ObservedOnline,
    ArtifactVerified,
    Endorsement,
    ProtocolViolation,
    InvalidSignature,
    ReplayAttempt,
    Equivocation,
    CorruptArtifact,
    CorruptCheckpoint,
    CapabilityFailure,
    Timeout,
    DhtPoisoning,
    TrainingUpdateRejected,
    ViolationReport,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignedEvidence {
    pub issuer: NodeId,
    pub issuer_public_key: [u8; 32],
    pub subject: NodeId,
    pub kind: EvidenceKind,
    pub sequence: u64,
    pub observed_at: u64,
    pub expires_at: u64,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum JobUpdateKind {
    Accepted {
        state: JobState,
    },
    Started,
    Chunk {
        sequence: u32,
        data: Vec<u8>,
    },
    Succeeded {
        output: Vec<u8>,
        output_hash: ArtifactId,
        evidence: Option<Evidence>,
    },
    Rejected {
        code: String,
        message: String,
    },
    Cancelled,
    TimedOut,
    Failed {
        code: String,
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct JobUpdate {
    pub job_id: JobId,
    pub state: JobState,
    pub sequence: u32,
    pub update: JobUpdateKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CancelJob {
    pub job_id: JobId,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ErrorCode {
    Malformed,
    UnsupportedVersion,
    LimitExceeded,
    Unauthorized,
    NotFound,
    Busy,
    Duplicate,
    PolicyDenied,
    InvalidState,
    Internal,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ProtocolError {
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Ping {
    pub nonce: [u8; 16],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactRequest {
    pub artifact: ArtifactId,
    pub offset: u64,
    pub max_bytes: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactChunk {
    pub artifact: ArtifactId,
    pub offset: u64,
    pub data: Vec<u8>,
    pub final_chunk: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactTransferRequest {
    pub request_id: RequestId,
    pub artifact: ArtifactId,
    pub offset: u64,
    pub max_bytes: u32,
    pub expected_size: u64,
    pub expected_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactTransferChunk {
    pub request_id: RequestId,
    pub artifact: ArtifactId,
    pub offset: u64,
    pub total_size: u64,
    pub data: Vec<u8>,
    pub final_chunk: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct KeyRotation {
    pub old_node_id: NodeId,
    pub old_public_key: [u8; 32],
    pub new_node_id: NodeId,
    pub new_public_key: [u8; 32],
    pub sequence: u64,
    pub valid_from: u64,
    pub valid_until: u64,
    pub old_signature: Vec<u8>,
    pub new_signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RelayEnvelope {
    pub session_id: RequestId,
    pub origin: NodeId,
    pub origin_public_key: [u8; 32],
    pub target: NodeId,
    pub payload: Vec<u8>,
    pub signature: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ArtifactKind {
    Model,
    Dataset,
    Checkpoint,
    Evaluation,
    Adapter,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactShard {
    pub artifact: ArtifactId,
    pub index: u32,
    pub size: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ArtifactManifest {
    pub artifact: ArtifactId,
    pub kind: ArtifactKind,
    pub format: String,
    pub size: u64,
    pub hash: ArtifactId,
    pub shards: Vec<ArtifactShard>,
    pub owner: Option<NodeId>,
    pub local_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub artifact: ArtifactId,
    pub identity: String,
    pub format: String,
    pub size: u64,
    pub weights: Vec<ArtifactShard>,
    pub capabilities: Vec<String>,
    pub runtime_requirements: Vec<MetadataEntry>,
    pub adapters: Vec<ArtifactId>,
    pub local_path: Option<String>,
    pub local_only: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum DataLocality {
    LocalOnly,
    Selective,
    Streamable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DatasetManifest {
    pub artifact: ArtifactId,
    pub identity: String,
    pub format: String,
    pub size: u64,
    pub shards: Vec<ArtifactShard>,
    pub locality: DataLocality,
    pub sample_policy: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub artifact: ArtifactId,
    pub model: ArtifactId,
    pub parent: Option<ArtifactId>,
    pub step: u64,
    pub complete_weights: bool,
    pub complete_optimizer: bool,
    pub complete_scheduler: bool,
    pub complete_rng: bool,
    pub workers: u16,
    pub hash: ArtifactId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SyncMode {
    Synchronous,
    BoundedStaleness,
    Asynchronous,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingPlan {
    pub model: ArtifactId,
    pub dataset: ArtifactId,
    pub checkpoint: Option<ArtifactId>,
    pub workers: u16,
    pub sync: SyncMode,
    pub max_steps: u64,
    pub checkpoint_every: u64,
    pub evaluation_capability: Option<String>,
}

/// V3 training messages carry bounded coordination metadata and small
/// reference-fixture updates.  Large model, optimizer, and checkpoint state
/// remains content-addressed artifact data; it is never placed in DHT values
/// or a generic metadata consensus record.
pub const TRAINING_SCALE: i64 = 1_000_000;

/// V5 compute identities are a closed, versioned set.  They are deliberately
/// not arbitrary strings: a planner must be able to reject an assignment
/// before a native runtime is entered.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Rocm,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum NumericFormat {
    F32,
    F16,
    Bf16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum BackendHealth {
    Ready,
    Busy,
    Degraded,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub enum ComputeFeature {
    MatrixMultiply,
    TensorParallel,
    PipelineStage,
    PeerToPeer,
    UnifiedMemory,
    DeterministicKernel,
}

/// Bounded, signed-advertisement data describing one locally discovered
/// backend.  Values are observations or claims, never cryptographic proof of
/// a device's physical identity.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub kind: BackendKind,
    pub runtime_version: String,
    pub device_count: u16,
    pub device_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub formats: Vec<NumericFormat>,
    pub max_tensor_elements: u64,
    pub features: Vec<ComputeFeature>,
    pub device_architecture: Option<String>,
    pub driver_available: bool,
    pub runtime_available: bool,
    pub peer_to_peer: bool,
    pub unified_memory: bool,
    pub max_concurrent_tasks: u16,
    pub safety_margin_permille: u16,
    pub health: BackendHealth,
    pub physical_verified: bool,
    /// Bounded local observations.  These are evidence, not cryptographic
    /// proof of vendor identity, and are intentionally separate from the
    /// self-reported device fields above.
    #[serde(default)]
    pub observed_successes: u32,
    #[serde(default)]
    pub observed_failures: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ComputeTaskKind {
    TensorForward,
    TensorBackward,
    PipelineForward,
    PipelineBackward,
    TrainingStep,
    CapabilityChallenge,
}

/// Backend requirements are semantic constraints, not executable code.  A
/// task may name an explicit backend or a bounded allow-list.  Fallbacks are
/// opt-in and precision is never implicitly downgraded.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComputeRequirements {
    pub task_kind: ComputeTaskKind,
    pub required_backend: Option<BackendKind>,
    pub allowed_backends: Vec<BackendKind>,
    pub required_formats: Vec<NumericFormat>,
    pub required_memory_bytes: u64,
    pub max_tensor_elements: u64,
    pub required_features: Vec<ComputeFeature>,
    pub kernel_id: String,
    pub kernel_version: u16,
    pub fallback_backends: Vec<BackendKind>,
    pub fallback_allowed: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct BackendAssignment {
    pub task_kind: ComputeTaskKind,
    pub worker: NodeId,
    pub backend: BackendKind,
    pub device_index: u16,
    pub requirements: ComputeRequirements,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityChallenge {
    pub challenge_id: RequestId,
    pub job_id: JobId,
    pub worker: NodeId,
    pub backend: BackendKind,
    pub task_kind: ComputeTaskKind,
    pub format: NumericFormat,
    pub input_elements: u32,
    pub seed: u64,
    pub deadline_ms: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityChallengeResult {
    pub challenge_id: RequestId,
    pub worker: NodeId,
    pub backend: BackendKind,
    pub success: bool,
    pub output_hash: ArtifactId,
    pub elapsed_micros: u64,
    pub allocated_bytes: u64,
    pub health: BackendHealth,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityEvidenceRecord {
    pub worker: NodeId,
    pub backend: BackendKind,
    pub observed_at: u64,
    pub successful_challenges: u32,
    pub failed_challenges: u32,
    pub successful_tasks: u32,
    pub failed_tasks: u32,
    pub last_throughput_micros: u64,
    pub last_error: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingExecution {
    /// Coordinator-driven windows retained for deterministic compatibility
    /// comparisons and regression tests.
    #[default]
    Windowed,
    /// Groups advance their own local-SGD windows and publish aggregates as
    /// they become available. No global per-window barrier is required.
    AutonomousLocalSgd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingSample {
    pub feature: i64,
    pub target: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingShardAssignment {
    pub shard_id: u16,
    pub group_id: u16,
    pub owners: Vec<NodeId>,
    pub replicas: Vec<NodeId>,
    pub generation: u64,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingGroup {
    pub group_id: u16,
    pub shard_id: u16,
    pub members: Vec<NodeId>,
    pub aggregator: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingStart {
    pub job_id: JobId,
    pub coordinator: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub plan_hash: ArtifactId,
    pub max_windows: u64,
    pub local_steps: u16,
    pub max_staleness: u16,
    pub checkpoint_every: u64,
    #[serde(default)]
    pub execution: TrainingExecution,
    /// Logical bytes occupied by the complete reference model and optimizer
    /// state.  This is deliberately metadata: large state travels as
    /// content-addressed artifacts, never inside the start message.
    #[serde(default)]
    pub model_state_bytes: u64,
    /// Logical bytes occupied by the recipient's assigned model/optimizer
    /// shard.  A worker is only required to fit this value in its local
    /// training-state budget.
    #[serde(default)]
    pub shard_state_bytes: u64,
    pub participants: Vec<NodeId>,
    pub groups: Vec<TrainingGroup>,
    pub shards: Vec<TrainingShardAssignment>,
    /// Only the recipient's assigned shard is included in this message.
    pub assignment: TrainingShardAssignment,
    pub initial_value: i64,
    pub initial_optimizer: i64,
    pub samples: Vec<TrainingSample>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingWindow {
    pub job_id: JobId,
    pub coordinator: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub window: u64,
    pub group_id: u16,
    pub shard_id: u16,
    pub aggregator: NodeId,
    pub members: Vec<NodeId>,
    pub base_generation: u64,
    pub local_steps: u16,
    pub max_staleness: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingUpdate {
    pub job_id: JobId,
    pub worker: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub window: u64,
    pub shard_id: u16,
    pub base_generation: u64,
    pub update_sequence: u64,
    pub value: i64,
    pub optimizer: i64,
    pub loss: i64,
    pub samples: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingAggregate {
    pub job_id: JobId,
    pub aggregator: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub window: u64,
    pub group_id: u16,
    pub shard_id: u16,
    pub generation: u64,
    pub value: i64,
    pub optimizer: i64,
    pub loss: i64,
    pub contributors: Vec<NodeId>,
    pub state_artifact: ArtifactId,
    pub replicas: Vec<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingShardState {
    pub job_id: JobId,
    pub owner: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub shard_id: u16,
    pub generation: u64,
    pub value: i64,
    pub optimizer: i64,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingShardSummary {
    pub shard_id: u16,
    pub generation: u64,
    pub state_hash: ArtifactId,
    pub providers: Vec<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingState {
    pub job_id: JobId,
    pub coordinator: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub window: u64,
    pub checkpoint_generation: u64,
    pub checkpoint: Option<ArtifactId>,
    pub shards: Vec<TrainingShardSummary>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingAckKind {
    Start,
    State {
        window: u64,
        state_hash: ArtifactId,
    },
    Checkpoint {
        artifact: ArtifactId,
    },
    Vote {
        candidate: NodeId,
        term: u64,
        last_window: u64,
        granted: bool,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingAck {
    pub job_id: JobId,
    pub peer: NodeId,
    pub term: u64,
    pub kind: TrainingAckKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingElection {
    pub job_id: JobId,
    pub candidate: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub last_window: u64,
    pub last_state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpointShard {
    pub shard_id: u16,
    pub artifact: ArtifactId,
    pub generation: u64,
    pub providers: Vec<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpointManifest {
    pub job_id: JobId,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub membership_epoch: u64,
    pub term: u64,
    pub checkpoint_generation: u64,
    pub parent: Option<ArtifactId>,
    pub shards: Vec<TrainingCheckpointShard>,
    pub dataset_progress: Vec<(u16, u64)>,
    pub hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpointOffer {
    pub job_id: JobId,
    pub creator: NodeId,
    pub term: u64,
    pub checkpoint_generation: u64,
    pub shard_id: u16,
    pub artifact: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingCheckpointCommit {
    pub job_id: JobId,
    pub coordinator: NodeId,
    pub term: u64,
    pub membership_epoch: u64,
    pub branch: ArtifactId,
    pub manifest: TrainingCheckpointManifest,
    pub quorum: Vec<NodeId>,
}

/// V4 training records describe execution topology and bounded state
/// transitions.  Large tensors and checkpoints remain content-addressed
/// artifacts; these messages carry only small reference-fixture payloads and
/// control metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4SupportLevel {
    Supported,
    Experimental,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4AcceleratorFamily {
    Cpu,
    Cuda,
    Rocm,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4NumericalFormat {
    F32,
    F64,
    F16,
    Bf16,
    I8,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4AcceleratorCapability {
    pub family: V4AcceleratorFamily,
    pub device_model: String,
    pub device_count: u16,
    pub memory_bytes: u64,
    pub formats: Vec<V4NumericalFormat>,
    pub runtime: String,
    pub runtime_version: String,
    pub physical_verified: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4WorkerCapability {
    pub node: NodeId,
    pub accelerator: V4AcceleratorCapability,
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub rtt_ms: u32,
    pub bandwidth_mbps: u32,
    pub reliability_permille: u16,
    /// V5 additive capability records.  An empty list is the legacy V4
    /// representation and remains valid for older workers.
    #[serde(default)]
    pub backends: Vec<BackendCapabilities>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4ParallelismStrategy {
    LocalSgd,
    TensorParallel,
    PipelineParallel,
    Hybrid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4ShardLifecycle {
    Assigned,
    Transferring,
    Verified,
    Active,
    Replicating,
    Draining,
    Retired,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4AggregationGroup {
    pub group_id: u16,
    pub members: Vec<NodeId>,
    pub aggregator: NodeId,
    pub parent: Option<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ShardOwnership {
    pub shard_id: u16,
    pub model_generation: u64,
    pub owners: Vec<NodeId>,
    pub replicas: Vec<NodeId>,
    pub ownership_generation: u64,
    pub content_hash: ArtifactId,
    pub state_bytes: u64,
    pub memory_bytes: u64,
    pub runtime_requirement: String,
    pub lifecycle: V4ShardLifecycle,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TrainingPlan {
    pub job_id: JobId,
    /// Authenticated peer that proposed this plan. A plan is a proposal, not
    /// a global authority; binding it to its sender prevents a valid plan
    /// hash from being replayed as another peer's state transition.
    pub proposer: NodeId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub training_epoch: u64,
    pub membership_epoch: u64,
    pub coordination_term: u64,
    pub branch: ArtifactId,
    pub strategy: V4ParallelismStrategy,
    pub support: V4SupportLevel,
    pub workers: Vec<NodeId>,
    pub groups: Vec<V4AggregationGroup>,
    pub shards: Vec<V4ShardOwnership>,
    pub tensor_degree: u16,
    pub pipeline_stages: u16,
    pub local_steps: u16,
    pub max_staleness: u16,
    pub checkpoint_replication: u16,
    pub data_locality: DataLocality,
    pub accelerator: Option<V4AcceleratorCapability>,
    pub worker_capabilities: Vec<V4WorkerCapability>,
    pub rationale: String,
    pub plan_hash: ArtifactId,
    /// The active plan this proposal supersedes.  Initial plans have no
    /// parent; live reconfiguration must name the exact active plan hash so a
    /// delayed proposal cannot be applied on top of a different generation.
    // This is an append-only wire field.  It must remain encoded even when
    // absent so postcard peers do not interpret an initial plan as a
    // truncated record.  `default` keeps plans written before lineage was
    // added readable.
    #[serde(default)]
    pub parent_plan_hash: Option<ArtifactId>,
    /// Explicit V5 task requirements and worker bindings.  These are part of
    /// the plan hash whenever present; absent values preserve V4 readability.
    #[serde(default)]
    pub compute_requirements: Vec<ComputeRequirements>,
    #[serde(default)]
    pub backend_assignments: Vec<BackendAssignment>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4TrainingPhase {
    Planned,
    Running,
    Reconfiguring,
    Paused,
    Committed,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TrainingStateRecord {
    pub job_id: JobId,
    pub coordinator: NodeId,
    pub plan_hash: ArtifactId,
    pub plan_generation: u64,
    pub training_epoch: u64,
    pub membership_epoch: u64,
    pub coordination_term: u64,
    pub optimizer_generation: u64,
    pub checkpoint_generation: u64,
    pub branch: ArtifactId,
    pub phase: V4TrainingPhase,
    pub workers: Vec<NodeId>,
    pub shards: Vec<V4ShardOwnership>,
    pub data_progress: Vec<u64>,
    pub checkpoint: Option<ArtifactId>,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4StateAckKind {
    TrainingState,
    OptimizerShard,
    Checkpoint,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4OptimizerShardRecord {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub shard_id: u16,
    pub owner: NodeId,
    pub replicas: Vec<NodeId>,
    pub content_hash: ArtifactId,
    pub state_bytes: u64,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4CheckpointRecord {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub membership_epoch: u64,
    pub checkpoint_generation: u64,
    pub branch: ArtifactId,
    pub parent: Option<ArtifactId>,
    pub shard_hashes: Vec<ArtifactId>,
    pub providers: Vec<NodeId>,
    pub complete: bool,
    pub manifest_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4StateAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub kind: V4StateAckKind,
    pub state_hash: ArtifactId,
    pub accepted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4ShardMigrationPhase {
    Prepare,
    Commit,
    Abort,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ShardMigration {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub from: NodeId,
    pub to: NodeId,
    pub ownership_generation: u64,
    pub content_hash: ArtifactId,
    pub state: Vec<u8>,
    pub phase: V4ShardMigrationPhase,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ShardMigrationAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub owner: NodeId,
    pub ownership_generation: u64,
    pub content_hash: ArtifactId,
    pub verified: bool,
    pub phase: V4ShardMigrationPhase,
}

/// A proposer asks the current owner to execute the already authenticated
/// two-phase transfer for a pending plan.  The request carries no shard bytes;
/// the owner streams them to the target through the existing migration path.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ShardMigrationRequest {
    pub request_id: JobId,
    pub job_id: JobId,
    pub proposer: NodeId,
    pub proposal_hash: ArtifactId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub from: NodeId,
    pub to: NodeId,
    pub ownership_generation: u64,
    pub content_hash: ArtifactId,
    pub reply_to: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ShardMigrationResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub from: NodeId,
    pub to: NodeId,
    pub ownership_generation: u64,
    pub content_hash: ArtifactId,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PlanProposalAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub proposal_hash: ArtifactId,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PlanAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub plan_hash: ArtifactId,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorInstall {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub shard_id: u16,
    pub rows: u16,
    pub cols: u16,
    pub row_offset: u16,
    pub weights: Vec<i64>,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorInstallAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub state_hash: ArtifactId,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorForward {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub shard_id: u16,
    pub sequence: u64,
    pub input: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorForwardResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub sequence: u64,
    pub output: Vec<i64>,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorBackward {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub shard_id: u16,
    pub sequence: u64,
    pub input: Vec<i64>,
    pub upstream: Vec<i64>,
    pub learning_rate_micros: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorBackwardResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub sequence: u64,
    pub weight_gradient: Vec<i64>,
    pub input_gradient: Vec<i64>,
    pub state_hash: ArtifactId,
    pub state_generation: u64,
    #[serde(default)]
    pub optimizer_state_hash: Option<ArtifactId>,
    #[serde(default)]
    pub optimizer_state_generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineInstall {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub stage_id: u16,
    pub stage_count: u16,
    pub coefficient: i64,
    pub bias: i64,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineInstallAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub stage_id: u16,
    pub state_hash: ArtifactId,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineForward {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub stage_id: u16,
    pub stage_count: u16,
    pub microbatch: u32,
    pub activation: Vec<i64>,
    pub next_stage: Option<NodeId>,
    pub reply_to: NodeId,
    pub deadline_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineForwardResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub microbatch: u32,
    pub stage_id: u16,
    pub activation: Vec<i64>,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineBackward {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub stage_id: u16,
    pub stage_count: u16,
    pub microbatch: u32,
    pub gradient: Vec<i64>,
    pub previous_stage: Option<NodeId>,
    pub reply_to: NodeId,
    pub deadline_ms: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineBackwardResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub microbatch: u32,
    pub stage_id: u16,
    pub gradient: Vec<i64>,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4CollectiveContribute {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub group_id: u16,
    pub contributor: NodeId,
    pub aggregator: NodeId,
    pub parent: Option<NodeId>,
    pub expected_contributors: u16,
    pub expected_groups: u16,
    pub values: Vec<i64>,
    pub reply_to: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4CollectiveAggregate {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub group_id: u16,
    pub aggregator: NodeId,
    pub values: Vec<i64>,
    pub contributors: u16,
    pub expected_groups: u16,
    pub reply_to: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4CollectiveResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub values: Vec<i64>,
    pub group_count: u16,
    pub max_fan_in: u16,
    pub contributor_bytes: u64,
    pub root_received_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TrainingBranch {
    pub job_id: JobId,
    pub branch: ArtifactId,
    pub parent_generation: u64,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub dataset_progress: u64,
    pub plan_generation: u64,
    pub created_by: NodeId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4ReconciliationPolicy {
    AbortBranch,
    SelectBranch,
    AverageCompatibleState,
    MergeLocalSgdState,
    ManualOperatorRequired,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ReconcileRequest {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub left: V4TrainingBranch,
    pub right: V4TrainingBranch,
    pub policy: V4ReconciliationPolicy,
    pub left_value: i64,
    pub right_value: i64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ReconcileResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub accepted: bool,
    pub branch: Option<ArtifactId>,
    pub merged_value: Option<i64>,
    pub reason: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4ByzantinePolicy {
    Mean,
    ClippedMean,
    TrimmedMean,
    CoordinateMedian,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ByzantineUpdate {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub worker: NodeId,
    pub aggregator: NodeId,
    pub sequence: u64,
    pub values: Vec<i64>,
    pub policy: V4ByzantinePolicy,
    pub expected_updates: u16,
    pub reply_to: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ByzantineResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub plan_generation: u64,
    pub generation: u64,
    pub policy: V4ByzantinePolicy,
    pub aggregate: Vec<i64>,
    pub accepted_workers: Vec<NodeId>,
    pub rejected_workers: Vec<NodeId>,
    pub robust: bool,
}

/// The durable execution graph binds all V4 data-plane roles to one
/// versioned job state.  It contains references and bounded topology only;
/// tensors, activations and checkpoints remain content-addressed artifacts or
/// streamed protocol payloads.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorGroup {
    pub group_id: u16,
    pub members: Vec<NodeId>,
    pub shard_ids: Vec<u16>,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4PipelineStageAssignment {
    pub stage_id: u16,
    pub worker: NodeId,
    pub replicas: Vec<NodeId>,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4OptimizerPlacement {
    pub shard_id: u16,
    pub owner: NodeId,
    pub replicas: Vec<NodeId>,
    pub generation: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4ExecutionGraph {
    pub job_id: JobId,
    pub graph_generation: u64,
    pub parent_graph_generation: Option<u64>,
    pub proposer: NodeId,
    pub coordinator: NodeId,
    pub reply_to: NodeId,
    pub plan_hash: ArtifactId,
    pub branch: ArtifactId,
    pub training_epoch: u64,
    pub membership_epoch: u64,
    pub coordination_term: u64,
    pub optimizer_generation: u64,
    pub checkpoint_generation: u64,
    pub workers: Vec<NodeId>,
    /// Members removed by a committed failure/leave transition are not
    /// silently re-admitted merely because their old address comes back.
    /// Re-entry must use the authenticated join path so stale local graph
    /// state cannot be mistaken for a current participant.
    #[serde(default)]
    pub retired_workers: Vec<NodeId>,
    pub tensor_groups: Vec<V4TensorGroup>,
    pub pipeline_stages: Vec<V4PipelineStageAssignment>,
    pub aggregation_groups: Vec<V4AggregationGroup>,
    pub shards: Vec<V4ShardOwnership>,
    pub optimizer_shards: Vec<V4OptimizerPlacement>,
    pub collective_generation: u64,
    pub local_steps: u16,
    pub max_staleness: u16,
    pub checkpoint_replication: u16,
    pub strategy: V4ParallelismStrategy,
    /// Majority voter identities for a coordinator replacement.  An empty
    /// certificate is valid for the initial graph and for graph changes made
    /// by the currently fenced coordinator (join/rebalance/failure of a
    /// non-coordinator).  A changed coordinator must carry a certificate
    /// proving a majority of the parent graph authorized the new term.
    #[serde(default)]
    pub election_certificate: Vec<NodeId>,
    /// Every backend-bound task in the active graph is inspectable and
    /// generation-scoped.  Empty means a legacy V4 graph.
    #[serde(default)]
    pub backend_assignments: Vec<BackendAssignment>,
    pub graph_hash: ArtifactId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum V4IntegratedPhase {
    Preparing,
    Running,
    Reconfiguring,
    Partitioned,
    Committed,
    Failed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedStateRecord {
    pub job_id: JobId,
    pub graph_generation: u64,
    pub plan_hash: ArtifactId,
    pub branch: ArtifactId,
    pub coordinator: NodeId,
    pub phase: V4IntegratedPhase,
    pub window: u64,
    pub target_windows: u64,
    pub initial_loss_micros: i64,
    pub current_loss_micros: i64,
    pub checkpoint_generation: u64,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub membership_epoch: u64,
    pub collective_generation: u64,
    pub recovery_count: u32,
    pub state_hash: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedStart {
    pub request_id: JobId,
    pub graph: V4ExecutionGraph,
    pub plan: V4TrainingPlan,
    pub windows: u64,
    pub checkpoint_every: u64,
    pub reply_to: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedAck {
    pub job_id: JobId,
    pub graph_generation: u64,
    pub worker: NodeId,
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedStateAck {
    pub job_id: JobId,
    pub graph_generation: u64,
    pub state_hash: ArtifactId,
    pub worker: NodeId,
    pub accepted: bool,
}

/// Request-correlated liveness probe for a durable integrated training job.
/// It is deliberately separate from state publication: an old state
/// acknowledgement must never satisfy a new failure detector probe.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedProbe {
    pub request_id: JobId,
    pub job_id: JobId,
    pub graph_generation: u64,
    pub graph_hash: ArtifactId,
    pub state_hash: ArtifactId,
    pub requester: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedProbeAck {
    pub request_id: JobId,
    pub job_id: JobId,
    pub graph_generation: u64,
    pub responder: NodeId,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedElectionRequest {
    pub request_id: JobId,
    pub job_id: JobId,
    pub graph_generation: u64,
    pub graph_hash: ArtifactId,
    pub branch: ArtifactId,
    pub state_hash: ArtifactId,
    pub term: u64,
    pub candidate: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedElectionVote {
    pub job_id: JobId,
    pub graph_generation: u64,
    pub graph_hash: ArtifactId,
    pub term: u64,
    pub candidate: NodeId,
    pub voter: NodeId,
    pub granted: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4IntegratedResult {
    pub request_id: JobId,
    pub job_id: JobId,
    pub graph_generation: u64,
    pub windows_completed: u64,
    pub initial_loss_micros: i64,
    pub final_loss_micros: i64,
    pub tensor_steps: u64,
    pub pipeline_steps: u64,
    pub collective_rounds: u64,
    pub checkpoint_generations: u64,
    pub graph_reconfigurations: u64,
    pub coordinator_replacements: u64,
    pub shard_recoveries: u64,
    pub tensor_recoveries: u64,
    pub pipeline_recoveries: u64,
    pub optimizer_recoveries: u64,
    pub checkpoint_recoveries: u64,
    pub max_update_fanin: u16,
    pub all_updates_to_one_coordinator: bool,
    pub single_optimizer_authority: bool,
    pub single_checkpoint_authority: bool,
    pub global_step_barrier: bool,
    pub model_must_fit_one_worker: bool,
    pub target_reached: bool,
    pub branch_reconciled: bool,
    pub phase: V4IntegratedPhase,
    #[serde(default)]
    pub backend_tasks: u64,
    #[serde(default)]
    pub backend_replans: u64,
    #[serde(default)]
    pub backend_fallbacks: u64,
    #[serde(default)]
    pub backend_portable_checkpoint: bool,
}

/// A replica carries only one bounded reference tensor shard.  It is sent
/// over authenticated peer transport and is never recovered from the
/// original owner's filesystem.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorReplica {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub shard_id: u16,
    pub rows: u16,
    pub cols: u16,
    pub row_offset: u16,
    pub weights: Vec<i64>,
    pub state_hash: ArtifactId,
    pub state_generation: u64,
    pub last_sequence: u64,
    pub ownership_generation: u64,
    pub source: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4TensorReplicaAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub shard_id: u16,
    pub state_hash: ArtifactId,
    pub state_generation: u64,
    pub accepted: bool,
}

/// Bounded optimizer state for one model/tensor shard.  The state is carried
/// separately from the small optimizer-placement record so a graph transition
/// can verify and recover the actual optimizer contents without putting large
/// tensors into replicated metadata.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4OptimizerStateInstall {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub model_generation: u64,
    pub optimizer_generation: u64,
    pub shard_id: u16,
    pub values: Vec<i64>,
    pub state_hash: ArtifactId,
    pub state_generation: u64,
    pub sequence: u64,
    pub last_sequence: u64,
    pub source: NodeId,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct V4OptimizerStateAck {
    pub job_id: JobId,
    pub plan_generation: u64,
    pub optimizer_generation: u64,
    pub shard_id: u16,
    pub state_hash: ArtifactId,
    pub state_generation: u64,
    pub accepted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum TrainingV4Message {
    Plan(V4TrainingPlan),
    StateRecord(V4TrainingStateRecord),
    OptimizerShard(V4OptimizerShardRecord),
    CheckpointRecord(V4CheckpointRecord),
    StateAck(V4StateAck),
    ShardMigration(V4ShardMigration),
    ShardMigrationAck(V4ShardMigrationAck),
    TensorInstall(V4TensorInstall),
    TensorInstallAck(V4TensorInstallAck),
    TensorForward(V4TensorForward),
    TensorForwardResult(V4TensorForwardResult),
    TensorBackward(V4TensorBackward),
    TensorBackwardResult(V4TensorBackwardResult),
    PipelineInstall(V4PipelineInstall),
    PipelineInstallAck(V4PipelineInstallAck),
    PipelineForward(V4PipelineForward),
    PipelineForwardResult(V4PipelineForwardResult),
    PipelineBackward(V4PipelineBackward),
    PipelineBackwardResult(V4PipelineBackwardResult),
    CollectiveContribute(V4CollectiveContribute),
    CollectiveAggregate(V4CollectiveAggregate),
    CollectiveResult(V4CollectiveResult),
    Branch(V4TrainingBranch),
    Reconcile(V4ReconcileRequest),
    ReconcileResult(V4ReconcileResult),
    ByzantineUpdate(V4ByzantineUpdate),
    ByzantineResult(V4ByzantineResult),
    // Append-only V4 variants preserve postcard discriminants for the
    // protocol 1.4 messages shipped before live plan activation.
    PlanAck(V4PlanAck),
    PlanProposal(V4TrainingPlan),
    PlanProposalAck(V4PlanProposalAck),
    ShardMigrationRequest(V4ShardMigrationRequest),
    ShardMigrationResult(V4ShardMigrationResult),
    // Protocol 1.5 append-only durable V4 job and recovery messages.
    IntegratedStart(V4IntegratedStart),
    IntegratedAck(V4IntegratedAck),
    IntegratedState(V4IntegratedStateRecord),
    IntegratedStateAck(V4IntegratedStateAck),
    IntegratedElectionRequest(V4IntegratedElectionRequest),
    IntegratedElectionVote(V4IntegratedElectionVote),
    IntegratedResult(V4IntegratedResult),
    TensorReplica(V4TensorReplica),
    TensorReplicaAck(V4TensorReplicaAck),
    OptimizerStateInstall(V4OptimizerStateInstall),
    OptimizerStateAck(V4OptimizerStateAck),
    // Protocol 1.5 append-only request-correlated liveness probes.  The
    // request ID prevents stale state acknowledgements from satisfying a
    // later coordinator-failure decision.
    IntegratedProbe(V4IntegratedProbe),
    IntegratedProbeAck(V4IntegratedProbeAck),
}

impl TrainingV4Message {
    /// New durable-job messages require protocol 1.5.  Keeping this check at
    /// the message boundary prevents a 1.4 peer from receiving an appended
    /// postcard discriminant it cannot decode.
    pub fn minimum_minor(&self) -> u16 {
        match self {
            Self::IntegratedStart(start)
                if !start.graph.backend_assignments.is_empty()
                    || !start.plan.backend_assignments.is_empty()
                    || !start.plan.compute_requirements.is_empty() =>
            {
                6
            }
            Self::IntegratedStart(_)
            | Self::IntegratedAck(_)
            | Self::IntegratedState(_)
            | Self::IntegratedStateAck(_)
            | Self::IntegratedElectionRequest(_)
            | Self::IntegratedElectionVote(_)
            | Self::IntegratedResult(_)
            | Self::IntegratedProbe(_)
            | Self::IntegratedProbeAck(_)
            | Self::TensorBackwardResult(_)
            | Self::TensorReplica(_)
            | Self::TensorReplicaAck(_)
            | Self::OptimizerStateInstall(_)
            | Self::OptimizerStateAck(_) => 5,
            Self::Plan(plan) | Self::PlanProposal(plan)
                if !plan.backend_assignments.is_empty()
                    || !plan.compute_requirements.is_empty()
                    || plan
                        .worker_capabilities
                        .iter()
                        .any(|worker| !worker.backends.is_empty()) =>
            {
                6
            }
            _ => 4,
        }
    }
}

/// V5 capability evidence uses its own append-only training frame.  The
/// existing V4 frame remains readable by 1.5 peers and is used for the
/// durable graph/control path.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingV5Message {
    CapabilityChallenge(CapabilityChallenge),
    CapabilityChallengeResult(CapabilityChallengeResult),
    CapabilityEvidence(CapabilityEvidenceRecord),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingMessage {
    Start(TrainingStart),
    Window(TrainingWindow),
    Update(TrainingUpdate),
    Aggregate(TrainingAggregate),
    ShardState(TrainingShardState),
    State(TrainingState),
    Ack(TrainingAck),
    Election(TrainingElection),
    CheckpointOffer(TrainingCheckpointOffer),
    CheckpointCommit(TrainingCheckpointCommit),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
pub enum Message {
    Hello(Hello),
    PeerExchange(PeerExchange),
    Announcement(SignedAnnouncement),
    JobRequest(JobRequest),
    JobUpdate(JobUpdate),
    CancelJob(CancelJob),
    ArtifactRequest(ArtifactRequest),
    ArtifactChunk(ArtifactChunk),
    Ping(Ping),
    Pong(Ping),
    Goodbye { reason: Option<String> },
    Error(ProtocolError),
    AddressUpdate(AddressUpdate),
    AddressObservation(AddressObservation),
    ArtifactTransferRequest(ArtifactTransferRequest),
    ArtifactTransferChunk(ArtifactTransferChunk),
    KeyRotation(KeyRotation),
    RelayEnvelope(RelayEnvelope),
    DhtRequest(DhtRequest),
    DhtResponse(DhtResponse),
    Training(TrainingMessage),
    TrainingV4(TrainingV4Message),
    TrainingV5(TrainingV5Message),
}

impl Message {
    pub const fn kind(&self) -> u16 {
        match self {
            Self::Hello(_) => 1,
            Self::PeerExchange(_) => 2,
            Self::Announcement(_) => 3,
            Self::JobRequest(_) => 4,
            Self::JobUpdate(_) => 5,
            Self::CancelJob(_) => 6,
            Self::ArtifactRequest(_) => 7,
            Self::ArtifactChunk(_) => 8,
            Self::Ping(_) => 9,
            Self::Pong(_) => 10,
            Self::Goodbye { .. } => 11,
            Self::Error(_) => 12,
            Self::AddressUpdate(_) => 13,
            Self::AddressObservation(_) => 14,
            Self::ArtifactTransferRequest(_) => 15,
            Self::ArtifactTransferChunk(_) => 16,
            Self::KeyRotation(_) => 17,
            Self::RelayEnvelope(_) => 18,
            Self::DhtRequest(_) => 19,
            Self::DhtResponse(_) => 20,
            Self::Training(_) => 21,
            Self::TrainingV4(_) => 22,
            Self::TrainingV5(_) => 23,
        }
    }
}
