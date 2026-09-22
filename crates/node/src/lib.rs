//! Concrete node composition and operator configuration.

#![recursion_limit = "256"]

use intelligence_intelligence::{
    BackendRegistry, CapabilityGraph, CapabilityObservation, EvaluatorCandidate, EvidenceGraph,
    HardwareKind, LocalSecurityState, SecurityEventKind, SecurityPolicy, TrustPolicy,
    WorkerProfile, choose_peer, evidence_signing_bytes, make_evaluation_job, make_inference_job,
    now_millis, plan_training, score_builtin_output, select_evaluators,
};
pub use intelligence_network::Identity;
use intelligence_network::{
    NetworkConfig, NetworkError, NetworkEvent, NetworkHandle, start as start_network,
};
use intelligence_protocol::{
    ArtifactChunk, ArtifactId, ArtifactRequest, ArtifactTransferChunk, ArtifactTransferRequest,
    Capability, CapabilityEvidence, CheckpointManifest, DataLocality, DatasetManifest, DhtKey,
    DhtNamespace, EvidenceKind, JobId, JobKind, JobRequest, JobState, JobUpdate, JobUpdateKind,
    MAX_ARTIFACT_CHUNK, Message, ModelManifest, NodeId, PrivacyPolicy, RequestId, ResourceLimits,
    SignedEvidence, SyncMode, TrainingMessage, TrainingPlan, TrainingStart, TrainingState,
    TrainingV4Message, TrainingV5Message, V4CheckpointRecord, V4ExecutionGraph,
    V4IntegratedStateRecord, V4OptimizerShardRecord, V4TrainingPlan, V4TrainingStateRecord,
};
use intelligence_runtime::{
    ExecutionOutcome, ExecutorSpec, ProcessSandbox, Runtime, RuntimeConfig, RuntimeError,
};
use intelligence_storage::{LocalStore, PersistedJob, StorageError};
use rand::{RngCore, rngs::OsRng};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Duration,
};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{UnixListener, UnixStream},
    sync::{Mutex, Notify, mpsc, oneshot},
    time::timeout,
};

mod distributed_training;
mod frontier_training;

const ADMIN_MAX_LINE: usize = 64 * 1024;
const MAX_JOB_HISTORY: usize = 4096;
const DEFAULT_STORAGE_QUOTA: u64 = 1024 * 1024 * 1024;
const DEFAULT_ARTIFACT_LIMIT: u64 = 256 * 1024 * 1024;
const DEFAULT_TRAINING_MEMORY_BYTES: u64 = 64 * 1024 * 1024;

#[derive(Default)]
struct NodeMetrics {
    rejected_requests: AtomicU64,
    execution_failures: AtomicU64,
    training_updates_sent: AtomicU64,
    training_updates_received: AtomicU64,
    training_aggregates_sent: AtomicU64,
    training_aggregates_received: AtomicU64,
    training_max_update_fanin: AtomicU64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StorageConfig {
    pub quota_bytes: u64,
    pub max_artifact_bytes: u64,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            quota_bytes: DEFAULT_STORAGE_QUOTA,
            max_artifact_bytes: DEFAULT_ARTIFACT_LIMIT,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CapabilityConfig {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: u16,
    #[serde(default = "default_true")]
    pub public: bool,
    #[serde(default = "default_true")]
    pub accept_remote_jobs: bool,
    #[serde(default = "default_executor_kind")]
    pub kind: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default)]
    pub program: Option<PathBuf>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Small operator-declared capability metadata used for local planning.
    /// It is a claim, not verification; runtime observations/evidence remain
    /// authoritative for trust decisions.  Keep this bounded because it is
    /// copied into signed peer advertisements.
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
    #[serde(default = "default_sandbox")]
    pub sandbox: String,
    #[serde(default = "default_input_limit")]
    pub max_input_bytes: u32,
    #[serde(default = "default_output_limit")]
    pub max_output_bytes: u32,
    #[serde(default = "default_memory_limit")]
    pub memory_bytes: u64,
    #[serde(default = "default_cpu_limit")]
    pub cpu_millis: u64,
}

impl CapabilityConfig {
    fn validate(&self) -> Result<(), NodeError> {
        if self.name.is_empty() || self.name.len() > 128 {
            return Err(NodeError::InvalidConfig(format!(
                "capability name is invalid: {}",
                self.name
            )));
        }
        if self.version == 0
            || self.max_input_bytes == 0
            || self.max_input_bytes as usize > intelligence_protocol::MAX_JOB_INPUT
            || self.max_output_bytes == 0
            || self.max_output_bytes as usize > intelligence_protocol::MAX_JOB_OUTPUT
            || self.memory_bytes == 0
            || self.cpu_millis == 0
        {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} has invalid version or limits",
                self.name
            )));
        }
        if self.kind != "builtin_text"
            && self.kind != "builtin_training"
            && self.kind != "process"
            && self.kind != "llama_cpp"
        {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} kind must be builtin_text, builtin_training, process, or llama_cpp",
                self.name
            )));
        }
        if self.kind == "builtin_text" && self.model.as_deref().unwrap_or_default().is_empty() {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} needs a model for builtin_text",
                self.name
            )));
        }
        if (self.kind == "process" || self.kind == "llama_cpp") && self.program.is_none() {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} needs a program for process execution",
                self.name
            )));
        }
        if self.kind == "llama_cpp" {
            let Some(model_path) = self.model_path.as_ref() else {
                return Err(NodeError::InvalidConfig(format!(
                    "capability {} needs an operator-supplied model_path",
                    self.name
                )));
            };
            let metadata = fs::metadata(model_path).map_err(|error| {
                NodeError::InvalidConfig(format!(
                    "llama_cpp model_path {} is not readable: {error}",
                    model_path.display()
                ))
            })?;
            if !metadata.is_file() || metadata.len() == 0 {
                return Err(NodeError::InvalidConfig(format!(
                    "llama_cpp model_path {} must be a non-empty file",
                    model_path.display()
                )));
            }
        }
        if self.metadata.len() > intelligence_protocol::MAX_METADATA.saturating_sub(1) {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} has too many metadata entries",
                self.name
            )));
        }
        for (key, value) in &self.metadata {
            if key.is_empty()
                || key.len() > 64
                || value.is_empty()
                || value.len() > 256
                || key == "executor"
            {
                return Err(NodeError::InvalidConfig(format!(
                    "capability {} has invalid or reserved metadata",
                    self.name
                )));
            }
        }
        if self.sandbox != "trusted_local" && self.sandbox != "bubblewrap" {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} sandbox must be trusted_local or bubblewrap",
                self.name
            )));
        }
        if (self.kind == "process" || self.kind == "llama_cpp")
            && self.sandbox == "trusted_local"
            && self.public
            && self.accept_remote_jobs
        {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} cannot expose a trusted_local process to peers",
                self.name
            )));
        }
        Ok(())
    }

    fn protocol_capability(&self, expires_at: u64) -> Capability {
        Capability {
            name: self.name.clone(),
            version: self.version,
            model: self.model.clone(),
            resources: ResourceLimits {
                max_input_bytes: self.max_input_bytes,
                max_output_bytes: self.max_output_bytes,
                memory_bytes: self.memory_bytes,
                cpu_millis: self.cpu_millis,
            },
            // Advertisement is a local claim. Observations and verified evaluations are
            // recorded by the requester/capability graph; a node must not self-upgrade it.
            evidence: CapabilityEvidence::Claimed,
            expires_at,
            metadata: std::iter::once(intelligence_protocol::MetadataEntry {
                key: "executor".to_string(),
                value: self.kind.clone(),
            })
            .chain(
                self.metadata
                    .iter()
                    .map(|(key, value)| intelligence_protocol::MetadataEntry {
                        key: key.clone(),
                        value: value.clone(),
                    }),
            )
            .collect(),
            compute_backends: Vec::new(),
        }
    }

    fn executor(&self) -> Result<ExecutorSpec, NodeError> {
        match self.kind.as_str() {
            "builtin_text" => Ok(ExecutorSpec::BuiltinText {
                model: self
                    .model
                    .clone()
                    .unwrap_or_else(|| "builtin.tiny-sentiment.v1".to_string()),
            }),
            "builtin_training" => Ok(ExecutorSpec::BuiltinTraining),
            "process" => Ok(ExecutorSpec::ExternalProcess {
                program: self.program.clone().ok_or_else(|| {
                    NodeError::InvalidConfig("process capability has no program".to_string())
                })?,
                args: self.args.clone(),
                env: self.env.clone(),
                sandbox: if self.sandbox == "bubblewrap" {
                    ProcessSandbox::Bubblewrap {
                        executable: PathBuf::from("bwrap"),
                    }
                } else {
                    ProcessSandbox::TrustedLocal
                },
            }),
            "llama_cpp" => Ok(ExecutorSpec::LlamaCpp {
                program: self.program.clone().ok_or_else(|| {
                    NodeError::InvalidConfig("llama_cpp capability has no program".to_string())
                })?,
                model_path: self.model_path.clone().ok_or_else(|| {
                    NodeError::InvalidConfig("llama_cpp capability has no model_path".to_string())
                })?,
                args: self.args.clone(),
                env: self.env.clone(),
                sandbox: if self.sandbox == "bubblewrap" {
                    ProcessSandbox::Bubblewrap {
                        executable: PathBuf::from("bwrap"),
                    }
                } else {
                    ProcessSandbox::TrustedLocal
                },
            }),
            _ => Err(NodeError::InvalidConfig(
                "unsupported executor kind".to_string(),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NodeConfig {
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default)]
    pub identity_path: Option<PathBuf>,
    #[serde(default)]
    pub admin_socket: Option<PathBuf>,
    #[serde(default = "default_listen_addr")]
    pub listen_addr: SocketAddr,
    #[serde(default)]
    pub advertise_addr: Option<String>,
    #[serde(default)]
    pub bootstrap: Vec<String>,
    #[serde(default)]
    pub relay_addresses: Vec<String>,
    #[serde(default)]
    pub relay_enabled: bool,
    #[serde(default = "default_relay_sessions")]
    pub relay_max_sessions: usize,
    #[serde(default = "default_relay_bytes")]
    pub relay_max_bytes: u64,
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
    #[serde(default = "default_max_frame_size")]
    pub max_frame_size: usize,
    #[serde(default = "default_peer_ttl")]
    pub peer_ttl_seconds: u64,
    #[serde(default)]
    pub allow_private_addresses: bool,
    #[serde(default)]
    pub prefer_relay: bool,
    #[serde(default = "default_hole_punch_enabled")]
    pub hole_punch_enabled: bool,
    #[serde(default = "default_hole_punch_attempts")]
    pub hole_punch_max_attempts: usize,
    #[serde(default = "default_dht_enabled")]
    pub dht_enabled: bool,
    #[serde(default = "default_dht_k")]
    pub dht_k: usize,
    #[serde(default = "default_dht_alpha")]
    pub dht_alpha: usize,
    #[serde(default = "default_dht_max_records")]
    pub dht_max_records: usize,
    /// Local budget for materialized training state.  V3 workers must fit
    /// their assigned shard, not the complete model state, in this budget.
    #[serde(default = "default_training_memory_bytes")]
    pub training_memory_bytes: u64,
    /// Test/operator knob for making one real worker a bounded straggler.
    /// It is intentionally capped and disabled by default.
    #[serde(default)]
    pub training_window_delay_ms: u64,
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
    #[serde(default)]
    pub capabilities: Vec<CapabilityConfig>,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            data_dir: default_data_dir(),
            identity_path: None,
            admin_socket: None,
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 4000)),
            advertise_addr: None,
            bootstrap: Vec::new(),
            relay_addresses: Vec::new(),
            relay_enabled: false,
            relay_max_sessions: 64,
            relay_max_bytes: 64 * 1024 * 1024,
            max_connections: 64,
            max_frame_size: intelligence_protocol::MAX_FRAME_SIZE,
            peer_ttl_seconds: 300,
            allow_private_addresses: false,
            prefer_relay: false,
            hole_punch_enabled: true,
            hole_punch_max_attempts: 4,
            dht_enabled: true,
            dht_k: 20,
            dht_alpha: 3,
            dht_max_records: 2_048,
            training_memory_bytes: DEFAULT_TRAINING_MEMORY_BYTES,
            training_window_delay_ms: 0,
            storage: StorageConfig::default(),
            runtime: RuntimeConfig::default(),
            capabilities: Vec::new(),
        }
    }
}

impl NodeConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, NodeError> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)?;
        let mut config: Self =
            toml::from_str(&contents).map_err(|error| NodeError::ConfigParse {
                path: path.to_path_buf(),
                message: error.to_string(),
            })?;
        config.apply_environment()?;
        config.normalize();
        config.validate()?;
        Ok(config)
    }

    pub fn from_environment() -> Result<Self, NodeError> {
        let mut config = Self::default();
        config.apply_environment()?;
        config.normalize();
        config.validate()?;
        Ok(config)
    }

    pub fn normalize(&mut self) {
        dedupe_config_addresses(&mut self.bootstrap);
        dedupe_config_addresses(&mut self.relay_addresses);
        if self.listen_addr.ip().is_loopback() {
            self.allow_private_addresses = true;
        }
        if self.identity_path.is_none() {
            self.identity_path = Some(self.data_dir.join("identity.key"));
        }
        if self.admin_socket.is_none() {
            self.admin_socket = Some(self.data_dir.join("node.sock"));
        }
        if self.runtime.work_dir.as_path() == Path::new("runtime-work") {
            self.runtime.work_dir = self.data_dir.join("runtime-work");
        }
        if self.advertise_addr.is_none() {
            self.advertise_addr = Some(self.listen_addr.to_string());
        }
    }

    pub fn validate(&self) -> Result<(), NodeError> {
        if self.max_connections == 0
            || self.max_frame_size == 0
            || self.max_frame_size > intelligence_protocol::MAX_FRAME_SIZE
        {
            return Err(NodeError::InvalidConfig(
                "network limits are invalid".to_string(),
            ));
        }
        if self.bootstrap.len() > intelligence_protocol::MAX_PEERS {
            return Err(NodeError::InvalidConfig(
                "too many bootstrap peers".to_string(),
            ));
        }
        if self.dht_k == 0
            || self.dht_k > intelligence_protocol::MAX_DHT_CONTACTS
            || self.dht_alpha == 0
            || self.dht_alpha > self.dht_k
            || self.dht_max_records == 0
            || self.dht_max_records > 32_768
        {
            return Err(NodeError::InvalidConfig(
                "DHT limits are invalid".to_string(),
            ));
        }
        if self.training_memory_bytes == 0
            || self.training_memory_bytes > (1 << 40)
            || self.training_window_delay_ms > 30_000
        {
            return Err(NodeError::InvalidConfig(
                "training resource limits are invalid".to_string(),
            ));
        }
        if self.relay_addresses.len() > intelligence_protocol::MAX_PEERS
            || self.relay_max_sessions == 0
            || self.relay_max_bytes == 0
            || self.relay_max_bytes > intelligence_protocol::MAX_RELAY_PAYLOAD as u64 * 4096
        {
            return Err(NodeError::InvalidConfig(
                "relay limits or relay address count are invalid".to_string(),
            ));
        }
        if self
            .advertise_addr
            .as_ref()
            .is_some_and(|address| address.is_empty() || address.len() > 256)
        {
            return Err(NodeError::InvalidConfig(
                "advertise address is invalid".to_string(),
            ));
        }
        if let Some(address) = self.advertise_addr.as_deref() {
            address.parse::<SocketAddr>().map_err(|error| {
                NodeError::InvalidConfig(format!("advertise address must be SocketAddr: {error}"))
            })?;
        }
        for address in self.bootstrap.iter().chain(self.relay_addresses.iter()) {
            address.parse::<SocketAddr>().map_err(|error| {
                NodeError::InvalidConfig(format!("peer address is invalid: {error}"))
            })?;
        }
        if self.storage.quota_bytes == 0
            || self.storage.max_artifact_bytes == 0
            || self.storage.max_artifact_bytes > self.storage.quota_bytes
        {
            return Err(NodeError::InvalidConfig(
                "storage limits are invalid".to_string(),
            ));
        }
        self.runtime.validate()?;
        if self.capabilities.len() > intelligence_protocol::MAX_CAPABILITIES {
            return Err(NodeError::InvalidConfig(
                "too many local capabilities".to_string(),
            ));
        }
        let mut names = HashSet::new();
        for capability in &self.capabilities {
            capability.validate()?;
            if !names.insert(&capability.name) {
                return Err(NodeError::InvalidConfig(format!(
                    "duplicate capability {}",
                    capability.name
                )));
            }
        }
        Ok(())
    }

    pub fn public_capabilities(&self) -> Vec<Capability> {
        let expires_at = now_secs().saturating_add(self.peer_ttl_seconds);
        self.capabilities
            .iter()
            .filter(|capability| capability.public)
            .map(|capability| capability.protocol_capability(expires_at))
            .collect()
    }

    pub fn to_toml(&self) -> Result<String, NodeError> {
        toml::to_string_pretty(self).map_err(|error| NodeError::ConfigParse {
            path: PathBuf::from("<memory>"),
            message: error.to_string(),
        })
    }

    fn apply_environment(&mut self) -> Result<(), NodeError> {
        if let Ok(value) = std::env::var("INTELLIGENCE_DATA_DIR") {
            self.data_dir = PathBuf::from(value);
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_IDENTITY_PATH") {
            self.identity_path = Some(PathBuf::from(value));
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_ADMIN_SOCKET") {
            self.admin_socket = Some(PathBuf::from(value));
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_LISTEN_ADDR") {
            self.listen_addr = value.parse().map_err(|error| {
                NodeError::InvalidConfig(format!("invalid listen address: {error}"))
            })?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_ADVERTISE_ADDR") {
            self.advertise_addr = Some(value);
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_BOOTSTRAP") {
            self.bootstrap = value
                .split(',')
                .filter(|item| !item.trim().is_empty())
                .map(|item| item.trim().to_string())
                .collect();
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RELAY_ADDRESSES") {
            self.relay_addresses = value
                .split(',')
                .filter(|item| !item.trim().is_empty())
                .map(|item| item.trim().to_string())
                .collect();
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RELAY_ENABLED") {
            self.relay_enabled = parse_env("INTELLIGENCE_RELAY_ENABLED", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RELAY_MAX_SESSIONS") {
            self.relay_max_sessions = parse_env("INTELLIGENCE_RELAY_MAX_SESSIONS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RELAY_MAX_BYTES") {
            self.relay_max_bytes = parse_env("INTELLIGENCE_RELAY_MAX_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_MAX_CONNECTIONS") {
            self.max_connections = parse_env("INTELLIGENCE_MAX_CONNECTIONS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_MAX_FRAME_SIZE") {
            self.max_frame_size = parse_env("INTELLIGENCE_MAX_FRAME_SIZE", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_PEER_TTL_SECONDS") {
            self.peer_ttl_seconds = parse_env("INTELLIGENCE_PEER_TTL_SECONDS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_ALLOW_PRIVATE_ADDRESSES") {
            self.allow_private_addresses =
                parse_env("INTELLIGENCE_ALLOW_PRIVATE_ADDRESSES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_PREFER_RELAY") {
            self.prefer_relay = parse_env("INTELLIGENCE_PREFER_RELAY", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_HOLE_PUNCH_ENABLED") {
            self.hole_punch_enabled = parse_env("INTELLIGENCE_HOLE_PUNCH_ENABLED", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_HOLE_PUNCH_MAX_ATTEMPTS") {
            self.hole_punch_max_attempts =
                parse_env("INTELLIGENCE_HOLE_PUNCH_MAX_ATTEMPTS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_DHT_ENABLED") {
            self.dht_enabled = parse_env("INTELLIGENCE_DHT_ENABLED", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_DHT_K") {
            self.dht_k = parse_env("INTELLIGENCE_DHT_K", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_DHT_ALPHA") {
            self.dht_alpha = parse_env("INTELLIGENCE_DHT_ALPHA", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_DHT_MAX_RECORDS") {
            self.dht_max_records = parse_env("INTELLIGENCE_DHT_MAX_RECORDS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_TRAINING_MEMORY_BYTES") {
            self.training_memory_bytes = parse_env("INTELLIGENCE_TRAINING_MEMORY_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_TRAINING_WINDOW_DELAY_MS") {
            self.training_window_delay_ms =
                parse_env("INTELLIGENCE_TRAINING_WINDOW_DELAY_MS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_STORAGE_QUOTA_BYTES") {
            self.storage.quota_bytes = parse_env("INTELLIGENCE_STORAGE_QUOTA_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_MAX_ARTIFACT_BYTES") {
            self.storage.max_artifact_bytes = parse_env("INTELLIGENCE_MAX_ARTIFACT_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_MAX_QUEUED_JOBS") {
            self.runtime.max_queued_jobs =
                parse_env("INTELLIGENCE_RUNTIME_MAX_QUEUED_JOBS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS") {
            self.runtime.max_concurrent_jobs =
                parse_env("INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_MAX_INPUT_BYTES") {
            self.runtime.max_input_bytes =
                parse_env("INTELLIGENCE_RUNTIME_MAX_INPUT_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_MAX_OUTPUT_BYTES") {
            self.runtime.max_output_bytes =
                parse_env("INTELLIGENCE_RUNTIME_MAX_OUTPUT_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_DEFAULT_TIMEOUT_MS") {
            self.runtime.default_timeout_ms =
                parse_env("INTELLIGENCE_RUNTIME_DEFAULT_TIMEOUT_MS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_PROCESS_MEMORY_BYTES") {
            self.runtime.process_memory_bytes =
                parse_env("INTELLIGENCE_RUNTIME_PROCESS_MEMORY_BYTES", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_PROCESS_CPU_SECONDS") {
            self.runtime.process_cpu_seconds =
                parse_env("INTELLIGENCE_RUNTIME_PROCESS_CPU_SECONDS", &value)?;
        }
        if let Ok(value) = std::env::var("INTELLIGENCE_RUNTIME_WORK_DIR") {
            self.runtime.work_dir = PathBuf::from(value);
        }
        Ok(())
    }
}

fn dedupe_config_addresses(addresses: &mut Vec<String>) {
    let mut seen = HashSet::new();
    addresses.retain(|address| seen.insert(address.clone()));
}

fn parse_env<T>(name: &str, value: &str) -> Result<T, NodeError>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| NodeError::InvalidConfig(format!("invalid {name}: {error}")))
}

#[derive(Debug, Error)]
pub enum NodeError {
    #[error("node I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("configuration parse failed for {path}: {message}")]
    ConfigParse { path: PathBuf, message: String },
    #[error("invalid node configuration: {0}")]
    InvalidConfig(String),
    #[error("network failed: {0}")]
    Network(#[from] NetworkError),
    #[error("identity failed: {0}")]
    Identity(#[from] intelligence_network::IdentityError),
    #[error("runtime failed: {0}")]
    Runtime(#[from] RuntimeError),
    #[error("storage failed: {0}")]
    Storage(#[from] StorageError),
    #[error("intelligence operation failed: {0}")]
    Intelligence(#[from] intelligence_intelligence::IntelligenceError),
    #[error("trust evidence failed: {0}")]
    Trust(#[from] intelligence_intelligence::TrustError),
    #[error("V6 security state failed: {0}")]
    Security(#[from] intelligence_intelligence::SecurityError),
    #[error("training planner failed: {0}")]
    Planner(#[from] intelligence_intelligence::PlannerError),
    #[error("JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug, Serialize)]
pub struct JobResult {
    pub job_id: JobId,
    pub state: JobState,
    pub output: Option<Vec<u8>>,
    pub output_hash: Option<ArtifactId>,
    pub evidence: Option<intelligence_protocol::Evidence>,
    pub error: Option<String>,
}

struct PendingJob {
    job_id: JobId,
    peer: NodeId,
    capability: String,
    started_at: u64,
    last_sequence: u32,
    sender: oneshot::Sender<JobResult>,
}

struct PendingArtifact {
    peer: NodeId,
    artifact: ArtifactId,
    offset: u64,
    expected_size: Option<u64>,
    sender: oneshot::Sender<Result<serde_json::Value, NodeError>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
struct ReferenceModel {
    weight: f64,
    bias: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct ReferenceSample {
    x: f64,
    y: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ReferenceGradient {
    job_id: String,
    model_artifact: String,
    dataset_artifact: String,
    step: u64,
    samples: usize,
    weight_gradient: f64,
    bias_gradient: f64,
    loss: f64,
}

#[derive(Clone, Debug, Deserialize)]
struct ReferenceCheckpoint {
    kind: String,
    step: u64,
    model: ReferenceModel,
    optimizer: ReferenceOptimizer,
    parent: Option<ArtifactId>,
    workers: Vec<NodeId>,
    model_artifact: ArtifactId,
    dataset_artifact: ArtifactId,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct ReferenceOptimizer {
    learning_rate: f64,
}

enum RemoteJobReservation {
    Reserved,
    ExistingTerminal,
    ExistingActive,
    OriginCollision,
}

#[derive(Serialize)]
struct AdminResponse {
    ok: bool,
    data: serde_json::Value,
    error: Option<String>,
}

#[derive(Deserialize)]
struct AdminWireResponse {
    ok: bool,
    data: serde_json::Value,
    error: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum AdminRequest {
    Status,
    Identity,
    Peers,
    Capabilities,
    Jobs,
    Config,
    DhtStats,
    DhtPublish {
        namespace: String,
        name: String,
        value: String,
        ttl_seconds: Option<u64>,
        sequence: Option<u64>,
    },
    DhtLookup {
        namespace: String,
        name: String,
    },
    DhtFindNode {
        key: String,
    },
    Trust {
        subject: String,
    },
    PlanTraining {
        model_bytes: u64,
        data_locality: String,
        workers: Option<u16>,
    },
    Infer {
        capability: String,
        input: String,
        deadline_ms: Option<u64>,
        max_output_bytes: Option<u32>,
        allow_input_transfer: Option<bool>,
        #[serde(default)]
        job_id: Option<String>,
    },
    RegisterModel {
        path: String,
        identity: String,
        format: String,
        local_only: bool,
    },
    Inspect {
        artifact: String,
    },
    FetchArtifact {
        peer: String,
        artifact: String,
    },
    Evaluate {
        text: String,
        expected_label: String,
        deadline_ms: Option<u64>,
    },
    TrainReference {
        workers: Option<u16>,
        steps: Option<u64>,
        #[serde(default)]
        resume_checkpoint: Option<String>,
    },
    TrainV3 {
        workers: Option<u16>,
        windows: Option<u64>,
        local_steps: Option<u16>,
        checkpoint_every: Option<u64>,
    },
    TrainV3Start {
        workers: Option<u16>,
        windows: Option<u64>,
        local_steps: Option<u16>,
        checkpoint_every: Option<u64>,
    },
    TrainV4 {
        workers: Option<u16>,
        windows: Option<u64>,
        checkpoint_every: Option<u64>,
    },
    TrainV4Start {
        workers: Option<u16>,
        windows: Option<u64>,
        checkpoint_every: Option<u64>,
    },
    PlanTrainingV4 {
        model_bytes: u64,
        workers: Option<u16>,
        strategy: String,
        tensor_degree: Option<u16>,
        pipeline_stages: Option<u16>,
    },
    ReplanTrainingV4 {
        job_id: String,
        model_bytes: u64,
        workers: Option<u16>,
        strategy: String,
        tensor_degree: Option<u16>,
        pipeline_stages: Option<u16>,
    },
    ActivateTrainingV4 {
        job_id: String,
    },
    V4SeedShard {
        job_id: String,
        shard_id: u16,
        state: String,
    },
    V4MigrateShard {
        job_id: String,
        shard_id: u16,
        target: String,
    },
    V4TensorDemo {
        workers: Vec<String>,
    },
    V4PipelineDemo {
        stages: Vec<String>,
        microbatches: Option<u16>,
    },
    V4CollectiveDemo {
        workers: Vec<String>,
    },
    V4ReconcileDemo {
        worker: String,
        left_value: i64,
        right_value: i64,
        policy: String,
    },
    V4ByzantineDemo {
        workers: Vec<String>,
        malicious: Option<u16>,
        policy: String,
    },
    V4ReplicateState {
        workers: Vec<String>,
    },
    TrainingStatus {
        job_id: String,
    },
    Cancel {
        job_id: String,
    },
    Shutdown,
}

pub async fn admin_call(
    path: impl AsRef<Path>,
    request: &AdminRequest,
) -> Result<serde_json::Value, NodeError> {
    let mut stream = UnixStream::connect(path).await?;
    let mut bytes = serde_json::to_vec(request)?;
    bytes.push(b'\n');
    stream.write_all(&bytes).await?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).await?;
    let response: AdminWireResponse = serde_json::from_str(&line)?;
    if response.ok {
        Ok(response.data)
    } else {
        Err(NodeError::InvalidConfig(response.error.unwrap_or_else(
            || "node rejected admin request".to_string(),
        )))
    }
}

pub struct Node {
    config: NodeConfig,
    identity: Identity,
    store: LocalStore,
    network: NetworkHandle,
    runtime: Arc<Runtime>,
    capabilities: HashMap<String, CapabilityConfig>,
    jobs: Arc<Mutex<HashMap<JobId, PersistedJob>>>,
    pending: Arc<Mutex<HashMap<JobId, PendingJob>>>,
    pending_artifacts: Arc<Mutex<HashMap<RequestId, PendingArtifact>>>,
    graph: Arc<Mutex<CapabilityGraph>>,
    metrics: Arc<NodeMetrics>,
    dht_sequence: AtomicU64,
    evidence_sequence: AtomicU64,
    trust: Arc<Mutex<EvidenceGraph>>,
    shutdown: Arc<Notify>,
    shutdown_requested: Arc<AtomicBool>,
    pub(crate) v3_jobs: Arc<Mutex<HashMap<JobId, distributed_training::V3JobHandle>>>,
    pub(crate) v3_states: Arc<Mutex<HashMap<JobId, TrainingState>>>,
    pub(crate) v4_jobs: Arc<Mutex<HashMap<JobId, frontier_training::V4JobHandle>>>,
    pub(crate) v4_data_shards:
        Arc<Mutex<HashMap<(JobId, u16), frontier_training::V4LocalDataShard>>>,
    pub(crate) v4_tensor_shards:
        Arc<Mutex<HashMap<(JobId, u16), frontier_training::V4LocalTensorShard>>>,
    pub(crate) v4_pipeline_stages:
        Arc<Mutex<HashMap<(JobId, u16), frontier_training::V4LocalPipelineStage>>>,
    pub(crate) v4_collectives: Arc<Mutex<HashMap<JobId, frontier_training::V4CollectiveState>>>,
    pub(crate) v4_byzantine: Arc<Mutex<HashMap<JobId, frontier_training::V4ByzantineState>>>,
    pub(crate) v4_training_states: Arc<Mutex<HashMap<JobId, V4TrainingStateRecord>>>,
    pub(crate) v4_integrated_states: Arc<Mutex<HashMap<JobId, V4IntegratedStateRecord>>>,
    pub(crate) v4_integrated_graphs: Arc<Mutex<HashMap<JobId, V4ExecutionGraph>>>,
    pub(crate) v4_integrated_elections:
        Arc<Mutex<HashMap<JobId, frontier_training::V4IntegratedElectionRecord>>>,
    pub(crate) v4_optimizer_states:
        Arc<Mutex<HashMap<(JobId, u16), frontier_training::V4LocalOptimizerState>>>,
    pub(crate) v4_optimizer_shards: Arc<Mutex<HashMap<(JobId, u16), V4OptimizerShardRecord>>>,
    pub(crate) v4_checkpoints: Arc<Mutex<HashMap<(JobId, u64), V4CheckpointRecord>>>,
    pub(crate) v4_plans: Arc<Mutex<HashMap<JobId, V4TrainingPlan>>>,
    pub(crate) v4_plan_proposals: Arc<Mutex<HashMap<JobId, V4TrainingPlan>>>,
    /// Local backend registry.  It is intentionally owned by the worker
    /// process and is never a network/global scheduler.
    pub(crate) compute: Arc<Mutex<BackendRegistry>>,
    /// Per-peer challenge budget.  Evidence challenges are deliberately
    /// local observations and must not become an unbounded resource endpoint.
    v5_challenge_windows: Arc<Mutex<HashMap<NodeId, (u64, u16)>>>,
    /// Bounded local V6 security state. It is persisted locally and is never
    /// published as a global score or authority.
    pub(crate) security: Arc<Mutex<LocalSecurityState>>,
}

impl Node {
    pub async fn start(mut config: NodeConfig) -> Result<Arc<Self>, NodeError> {
        config.normalize();
        for capability in &mut config.capabilities {
            if let Some(model_path) = capability.model_path.as_mut()
                && model_path.is_relative()
            {
                *model_path = config.data_dir.join(&*model_path);
            }
        }
        config.validate()?;
        fs::create_dir_all(&config.data_dir)?;
        if let Some(parent) = config.identity_path.as_ref().and_then(|path| path.parent()) {
            fs::create_dir_all(parent)?;
        }
        if let Some(parent) = config.admin_socket.as_ref().and_then(|path| path.parent()) {
            fs::create_dir_all(parent)?;
        }
        if config.runtime.work_dir.is_relative() {
            config.runtime.work_dir = config.data_dir.join(&config.runtime.work_dir);
        }
        let identity =
            Identity::load_or_generate(config.identity_path.as_ref().ok_or_else(|| {
                NodeError::InvalidConfig("identity path is missing".to_string())
            })?)?;
        let store = LocalStore::open(
            &config.data_dir,
            config.storage.quota_bytes,
            config.storage.max_artifact_bytes,
        )?;
        let mut training_states = HashMap::new();
        for entry in fs::read_dir(store.root().join("state"))? {
            let entry = entry?;
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            let Some(job_text) = file_name
                .strip_prefix("training-v3-")
                .and_then(|value| value.strip_suffix(".json"))
            else {
                continue;
            };
            if job_text.len() != 32 || !job_text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                continue;
            }
            let Some(state) = store.read_json::<TrainingState>(file_name)? else {
                continue;
            };
            if Message::Training(TrainingMessage::State(state.clone()))
                .validate()
                .is_ok()
            {
                training_states.insert(state.job_id, state);
            } else {
                tracing::warn!(
                    file = file_name,
                    "ignoring invalid persisted V3 training state"
                );
            }
        }
        let mut training_starts = Vec::new();
        for entry in fs::read_dir(store.root().join("state"))? {
            let entry = entry?;
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };
            let Some(job_text) = file_name
                .strip_prefix("training-v3-start-")
                .and_then(|value| value.strip_suffix(".json"))
            else {
                continue;
            };
            if job_text.len() != 32 || !job_text.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                continue;
            }
            if let Some(start) = store.read_json(file_name)? {
                training_starts.push(start);
            }
        }
        let runtime = Arc::new(Runtime::new(config.runtime.clone())?);
        let capabilities = config
            .capabilities
            .iter()
            .cloned()
            .map(|capability| (capability.name.clone(), capability))
            .collect::<HashMap<_, _>>();
        let mut jobs = HashMap::new();
        let mut recovered_jobs = false;
        for mut job in store.load_jobs()? {
            if !job.state.terminal() {
                job.state = JobState::Failed;
                job.updated_at = now_secs();
                job.error =
                    Some("node restarted before the job reached a terminal state".to_string());
                recovered_jobs = true;
            }
            jobs.insert(job.job_id, job);
        }
        if recovered_jobs {
            store.save_jobs(&jobs.values().cloned().collect::<Vec<_>>())?;
        }
        let mut evidence_graph =
            EvidenceGraph::new(TrustPolicy::default())?.with_local_issuer(identity.node_id());
        for evidence in store.load_evidence()? {
            let _ = evidence_graph.insert(evidence, now_secs());
        }
        let security = match store.read_json::<Vec<u8>>("v6-security.json") {
            Ok(Some(bytes)) => match postcard::from_bytes::<LocalSecurityState>(&bytes) {
                Ok(state) if state.local_node == identity.node_id() && state.validate().is_ok() => {
                    state
                }
                Ok(state) if state.local_node == identity.node_id() => {
                    tracing::warn!("ignoring malformed V6 security cache");
                    LocalSecurityState::new(identity.node_id(), SecurityPolicy::default())?
                }
                Ok(_) => {
                    tracing::warn!("ignoring V6 security cache for a different local identity");
                    LocalSecurityState::new(identity.node_id(), SecurityPolicy::default())?
                }
                Err(error) => {
                    tracing::warn!(error = %error, "ignoring malformed V6 security cache");
                    LocalSecurityState::new(identity.node_id(), SecurityPolicy::default())?
                }
            },
            Err(error) => {
                // A corrupt or legacy cache is not evidence of trust. Rebuild
                // the bounded runtime ledger from fresh observations instead
                // of failing node startup or treating it as high trust.
                tracing::debug!(error = %error, "V6 security cache was unreadable");
                LocalSecurityState::new(identity.node_id(), SecurityPolicy::default())?
            }
            Ok(None) => LocalSecurityState::new(identity.node_id(), SecurityPolicy::default())?,
        };
        let (event_tx, event_rx) = mpsc::channel(256);
        let compute = Arc::new(Mutex::new(BackendRegistry::discover()));
        let local_backend_capabilities = compute.lock().await.advertised_capabilities();
        let mut public_capabilities = config.public_capabilities();
        for capability in &mut public_capabilities {
            if capability.name == "training.reference" {
                capability.compute_backends = local_backend_capabilities.clone();
            }
        }
        if config.relay_enabled {
            public_capabilities.push(relay_capability(
                now_secs().saturating_add(config.peer_ttl_seconds),
            ));
        }
        let network = start_network(
            NetworkConfig {
                listen_addr: config.listen_addr,
                advertise_addr: config
                    .advertise_addr
                    .clone()
                    .unwrap_or_else(|| config.listen_addr.to_string()),
                bootstrap: config.bootstrap.clone(),
                max_frame_size: config.max_frame_size,
                max_connections: config.max_connections,
                peer_ttl_seconds: config.peer_ttl_seconds,
                allow_private_addresses: config.allow_private_addresses,
                prefer_relay: config.prefer_relay,
                hole_punch_enabled: config.hole_punch_enabled,
                hole_punch_max_attempts: config.hole_punch_max_attempts,
                relay_addresses: config.relay_addresses.clone(),
                relay_enabled: config.relay_enabled,
                relay_max_sessions: config.relay_max_sessions,
                relay_max_bytes: config.relay_max_bytes,
                capabilities: public_capabilities,
                dht_enabled: config.dht_enabled,
                dht_k: config.dht_k,
                dht_alpha: config.dht_alpha,
                dht_max_records: config.dht_max_records,
            },
            identity.clone(),
            store.clone(),
            event_tx,
        )
        .await?;
        let node = Arc::new(Self {
            config,
            identity,
            store,
            network,
            runtime,
            capabilities,
            jobs: Arc::new(Mutex::new(jobs)),
            pending: Arc::new(Mutex::new(HashMap::new())),
            pending_artifacts: Arc::new(Mutex::new(HashMap::new())),
            graph: Arc::new(Mutex::new(CapabilityGraph::new(4096)?)),
            metrics: Arc::new(NodeMetrics::default()),
            dht_sequence: AtomicU64::new(now_secs().max(1)),
            evidence_sequence: AtomicU64::new(now_secs().max(1)),
            trust: Arc::new(Mutex::new(evidence_graph)),
            shutdown: Arc::new(Notify::new()),
            shutdown_requested: Arc::new(AtomicBool::new(false)),
            v3_jobs: Arc::new(Mutex::new(HashMap::new())),
            v3_states: Arc::new(Mutex::new(training_states)),
            v4_jobs: Arc::new(Mutex::new(HashMap::new())),
            v4_data_shards: Arc::new(Mutex::new(HashMap::new())),
            v4_tensor_shards: Arc::new(Mutex::new(HashMap::new())),
            v4_pipeline_stages: Arc::new(Mutex::new(HashMap::new())),
            v4_collectives: Arc::new(Mutex::new(HashMap::new())),
            v4_byzantine: Arc::new(Mutex::new(HashMap::new())),
            v4_training_states: Arc::new(Mutex::new(HashMap::new())),
            v4_integrated_states: Arc::new(Mutex::new(HashMap::new())),
            v4_integrated_graphs: Arc::new(Mutex::new(HashMap::new())),
            v4_integrated_elections: Arc::new(Mutex::new(HashMap::new())),
            v4_optimizer_states: Arc::new(Mutex::new(HashMap::new())),
            v4_optimizer_shards: Arc::new(Mutex::new(HashMap::new())),
            v4_checkpoints: Arc::new(Mutex::new(HashMap::new())),
            v4_plans: Arc::new(Mutex::new(HashMap::new())),
            v4_plan_proposals: Arc::new(Mutex::new(HashMap::new())),
            compute,
            v5_challenge_windows: Arc::new(Mutex::new(HashMap::new())),
            security: Arc::new(Mutex::new(security)),
        });
        let event_node = node.clone();
        tokio::spawn(async move { event_node.event_loop(event_rx).await });
        node.start_admin().await?;
        node.start_dht_advertisements();
        distributed_training::restore_workers(node.clone(), training_starts).await?;
        frontier_training::restore(&node).await?;
        frontier_training::resume_integrated_jobs(&node).await?;
        tracing::info!(node_id = %node.identity.node_id(), listen = %node.network.local_address(), "node started");
        Ok(node)
    }

    pub fn node_id(&self) -> NodeId {
        self.identity.node_id()
    }

    pub fn admin_socket(&self) -> &Path {
        self.config.admin_socket.as_deref().unwrap_or_else(|| {
            // `normalize` is called before the node is constructed. Returning a
            // stable fallback keeps this accessor total for diagnostics while
            // preserving the invariant that the running node has an explicit path.
            Path::new("node.sock")
        })
    }

    pub async fn wait_for_shutdown(&self) {
        loop {
            if self.shutdown_requested.load(Ordering::Acquire) {
                return;
            }
            let notified = self.shutdown.notified();
            if self.shutdown_requested.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    pub async fn shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
        self.runtime.cancel_all().await;
        // Closing the per-job mailboxes lets background V3 training contexts
        // observe termination and exit.  Keeping their senders alive after
        // the transport shuts down would leave autonomous coordinators and
        // workers running against dead peers and would make graceful node
        // restarts indistinguishable from a leaked training process.
        self.v3_jobs.lock().await.clear();
        self.v4_jobs.lock().await.clear();
        self.v4_collectives.lock().await.clear();
        self.v4_byzantine.lock().await.clear();
        self.v4_training_states.lock().await.clear();
        self.v4_integrated_states.lock().await.clear();
        self.v4_integrated_graphs.lock().await.clear();
        self.v4_integrated_elections.lock().await.clear();
        self.v4_optimizer_states.lock().await.clear();
        self.v4_optimizer_shards.lock().await.clear();
        self.v4_checkpoints.lock().await.clear();
        self.v4_plans.lock().await.clear();
        self.v4_plan_proposals.lock().await.clear();
        self.network.shutdown().await;
        self.shutdown.notify_waiters();
        self.shutdown.notify_one();
        if let Some(path) = &self.config.admin_socket {
            let _ = fs::remove_file(path);
        }
    }

    pub async fn status(&self) -> Result<serde_json::Value, NodeError> {
        let jobs = self.jobs.lock().await;
        let state_counts =
            jobs.values()
                .fold(BTreeMap::<String, usize>::new(), |mut counts, job| {
                    *counts.entry(format!("{:?}", job.state)).or_default() += 1;
                    counts
                });
        let job_count = jobs.len();
        drop(jobs);
        let connected_peers = self.network.connected_peers().await;
        let known_peers = self.network.peer_records().await.len();
        let pending_jobs = self.pending.lock().await.len();
        let network = self.network.metrics();
        let observed_addresses = self.network.observed_addresses().await;
        let relay_peers = self.network.relay_peers().await;
        let reachability = self.network.reachability_state().await;
        let dht = self.network.dht_stats().await;
        let training_v3_jobs = self.v3_states.lock().await.len();
        let training_v3_active_jobs = self.v3_jobs.lock().await.len();
        let mut training_v3_active_job_ids = self
            .v3_jobs
            .lock()
            .await
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        training_v3_active_job_ids.sort_unstable();
        let training_v4_active_jobs = self.v4_jobs.lock().await.len();
        let training_v4_data_shards = self.v4_data_shards.lock().await.len();
        let training_v4_tensor_shards = self.v4_tensor_shards.lock().await.len();
        let training_v4_pipeline_stages = self.v4_pipeline_stages.lock().await.len();
        let training_v4_state_records = self.v4_training_states.lock().await.len();
        let training_v4_integrated_states = self.v4_integrated_states.lock().await.len();
        let training_v4_integrated_graphs = self.v4_integrated_graphs.lock().await.len();
        let training_v4_optimizer_states = self.v4_optimizer_states.lock().await.len();
        let training_v4_optimizer_shards = self.v4_optimizer_shards.lock().await.len();
        let training_v4_checkpoints = self.v4_checkpoints.lock().await.len();
        let training_v4_plans = self.v4_plans.lock().await.len();
        let training_v4_plan_proposals = self.v4_plan_proposals.lock().await.len();
        let training_v4_plan_generations = self
            .v4_plans
            .lock()
            .await
            .iter()
            .map(|(job_id, plan)| (job_id.to_string(), plan.plan_generation))
            .collect::<BTreeMap<_, _>>();
        let backend_capabilities = self.compute.lock().await.capabilities();
        let security = self.security.lock().await;
        let security_events = security.events().cloned().collect::<Vec<_>>();
        let security_event_count = security.event_count();
        Ok(serde_json::json!({
            "node_id": self.identity.node_id().to_string(),
            "public_key": hex::encode(self.identity.public_key()),
            "protocol": format!("{}.{}", intelligence_protocol::PROTOCOL_MAJOR, intelligence_protocol::PROTOCOL_MINOR),
            "listen_addr": self.network.local_address().to_string(),
            "reachability": format!("{:?}", reachability),
            "observed_addresses": observed_addresses,
            "relay_peers": relay_peers,
            "dht": dht,
            "connected_peers": connected_peers,
            "known_peers": known_peers,
            "capability_count": self.capabilities.len(),
            "job_count": job_count,
            "pending_jobs": pending_jobs,
            "queue_depth": self.runtime.queue_depth(),
            "running_jobs": self.runtime.running_jobs(),
            "job_states": state_counts,
            "storage_used_bytes": self.store.used_bytes()?,
            "storage_quota_bytes": self.store.quota_bytes(),
            "process_memory_bytes": process_memory_bytes(),
            "process_cpu_time_ms": process_cpu_time_ms(),
            "network": network,
            "rejected_requests": self.metrics.rejected_requests.load(Ordering::Relaxed),
            "execution_failures": self.metrics.execution_failures.load(Ordering::Relaxed),
            "training_v3_jobs": training_v3_jobs,
            "training_v3_active_jobs": training_v3_active_jobs,
            "training_v3_active_job_ids": training_v3_active_job_ids,
            "training_v4_active_jobs": training_v4_active_jobs,
            "training_v4_data_shards": training_v4_data_shards,
            "training_v4_tensor_shards": training_v4_tensor_shards,
            "training_v4_pipeline_stages": training_v4_pipeline_stages,
            "training_v4_state_records": training_v4_state_records,
            "training_v4_integrated_states": training_v4_integrated_states,
            "training_v4_integrated_graphs": training_v4_integrated_graphs,
            "training_v4_optimizer_states": training_v4_optimizer_states,
            "training_v4_optimizer_shards": training_v4_optimizer_shards,
            "training_v4_checkpoints": training_v4_checkpoints,
            "training_v4_plans": training_v4_plans,
            "training_v4_plan_proposals": training_v4_plan_proposals,
            "training_v4_plan_generations": training_v4_plan_generations,
            "compute_backends": backend_capabilities,
            "training_updates_sent": self.metrics.training_updates_sent.load(Ordering::Relaxed),
            "training_updates_received": self.metrics.training_updates_received.load(Ordering::Relaxed),
            "training_aggregates_sent": self.metrics.training_aggregates_sent.load(Ordering::Relaxed),
            "training_aggregates_received": self.metrics.training_aggregates_received.load(Ordering::Relaxed),
            "training_max_update_fanin": self.metrics.training_max_update_fanin.load(Ordering::Relaxed),
            "v6_security": {
                "profile": format!("{:?}", security.policy().profile),
                "event_count": security_event_count,
                "events": security_events,
                "global_trust_authority": false,
                "global_reputation": false,
            },
        }))
    }

    async fn persist_security_state(&self) {
        let state = self.security.lock().await.clone();
        let encoded = match postcard::to_allocvec(&state) {
            Ok(encoded) => encoded,
            Err(error) => {
                tracing::warn!(error = %error, "failed to encode V6 local security state");
                return;
            }
        };
        if let Err(error) = self.store.write_json("v6-security.json", &encoded) {
            tracing::warn!(error = %error, "failed to persist V6 local security state");
        }
    }

    async fn note_security_event(
        &self,
        subject: Option<NodeId>,
        kind: SecurityEventKind,
        scope: &str,
        detail: &str,
    ) {
        {
            let mut security = self.security.lock().await;
            security.record_event(now_secs(), subject, kind, scope, detail);
        }
        self.persist_security_state().await;
    }

    async fn training_status(&self, job_id: JobId) -> Result<serde_json::Value, NodeError> {
        if let Some(state) = self.v4_integrated_states.lock().await.get(&job_id).cloned() {
            let mut status = serde_json::to_value(&state).unwrap_or_else(|_| serde_json::json!({}));
            if let Some(object) = status.as_object_mut() {
                object.insert(
                    "training_version".to_string(),
                    serde_json::json!("v4-integrated"),
                );
                object.insert(
                    "durable_execution_graph".to_string(),
                    serde_json::json!(true),
                );
                object.insert(
                    "active".to_string(),
                    serde_json::json!(self.v4_jobs.lock().await.contains_key(&job_id)),
                );
                if let Some(graph) = self.v4_integrated_graphs.lock().await.get(&job_id).cloned() {
                    object.insert(
                        "execution_graph".to_string(),
                        serde_json::to_value(graph).unwrap_or_else(|_| serde_json::json!({})),
                    );
                }
                let optimizer_states = self
                    .v4_optimizer_states
                    .lock()
                    .await
                    .values()
                    .filter(|optimizer| optimizer.job_id == job_id)
                    .map(|optimizer| {
                        serde_json::json!({
                            "shard_id": optimizer.shard_id,
                            "optimizer_generation": optimizer.optimizer_generation,
                            "state_generation": optimizer.state_generation,
                            "last_sequence": optimizer.last_sequence,
                            "state_hash": optimizer.state_hash,
                        })
                    })
                    .collect::<Vec<_>>();
                object.insert(
                    "optimizer_states".to_string(),
                    serde_json::Value::Array(optimizer_states),
                );
            }
            return Ok(status);
        }
        let state = self
            .v3_states
            .lock()
            .await
            .get(&job_id)
            .cloned()
            .ok_or_else(|| {
                NodeError::InvalidConfig("V3 training job is not known locally".to_string())
            })?;
        let mut status = serde_json::to_value(&state).unwrap_or_else(|_| serde_json::json!({}));
        // The persisted start record is the inspectable assignment boundary:
        // it describes the complete shard topology, while the worker's local
        // shard state file contains only its assigned parameter/optimizer
        // shard.  Expose that distinction to operators and tests without
        // copying tensor state into status responses.
        let start_name = format!("training-v3-start-{job_id}.json");
        if let Some(start) = self.store.read_json::<TrainingStart>(&start_name)? {
            if let Some(object) = status.as_object_mut() {
                let coordinator = state.coordinator == self.node_id();
                object.insert(
                    "owned_shard_id".to_string(),
                    serde_json::json!(start.assignment.shard_id),
                );
                object.insert(
                    "model_shard_count".to_string(),
                    serde_json::json!(start.shards.len()),
                );
                object.insert(
                    "role".to_string(),
                    serde_json::json!(if coordinator { "coordinator" } else { "worker" }),
                );
                object.insert(
                    "materialized_shard_count".to_string(),
                    serde_json::json!(if coordinator { start.shards.len() } else { 1 }),
                );
                object.insert(
                    "full_model_materialized_on_worker".to_string(),
                    serde_json::json!(false),
                );
                object.insert(
                    "model_state_bytes".to_string(),
                    serde_json::json!(start.model_state_bytes),
                );
                object.insert(
                    "assigned_shard_state_bytes".to_string(),
                    serde_json::json!(start.shard_state_bytes),
                );
                object.insert(
                    "training_memory_bytes".to_string(),
                    serde_json::json!(self.config.training_memory_bytes),
                );
                object.insert(
                    "model_requires_sharding".to_string(),
                    serde_json::json!(start.model_state_bytes > self.config.training_memory_bytes),
                );
            }
        }
        Ok(status)
    }

    fn start_dht_advertisements(self: &Arc<Self>) {
        if !self.config.dht_enabled {
            return;
        }
        let node = self.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs((node.config.peer_ttl_seconds / 2).clamp(5, 30));
            // Let the authenticated peer graph form before the first publish.
            // The record is still stored locally immediately by the network;
            // this delay makes initial replication useful without a central
            // registry or an unbounded retry loop.
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                _ = node.shutdown.notified() => return,
            }
            loop {
                let mut capabilities = node.config.public_capabilities();
                let local_backends = node.compute.lock().await.advertised_capabilities();
                for capability in &mut capabilities {
                    if capability.name == "training.reference" {
                        capability.compute_backends = local_backends.clone();
                    }
                }
                for capability in capabilities {
                    let Ok(value) = serde_json::to_vec(&capability) else {
                        continue;
                    };
                    let sequence = node.dht_sequence.fetch_add(1, Ordering::Relaxed);
                    let _ = node
                        .network
                        .dht_publish(
                            DhtNamespace::Capability,
                            &capability.name,
                            value,
                            node.config.peer_ttl_seconds,
                            sequence,
                        )
                        .await;
                }
                tokio::select! {
                    _ = tokio::time::sleep(interval) => {},
                    _ = node.shutdown.notified() => break,
                }
            }
        });
    }

    pub async fn identity_json(&self) -> serde_json::Value {
        serde_json::json!({
            "node_id": self.identity.node_id().to_string(),
            "public_key": hex::encode(self.identity.public_key()),
            "rotation": self.identity.rotation(),
        })
    }

    async fn trust_json(&self, subject: NodeId) -> serde_json::Value {
        // Evidence is locally authoritative only after its own signature and
        // expiry checks pass. DHT records are merely a bounded transport for
        // the latest issuer claim for this subject; they are never treated as
        // a global reputation answer.
        let evidence_name = format!("subject:{subject}");
        if let Ok(records) = self
            .network
            .dht_lookup(DhtNamespace::Evidence, &evidence_name)
            .await
        {
            let mut graph = self.trust.lock().await;
            for record in records {
                let Ok(evidence) = serde_json::from_slice::<SignedEvidence>(&record.value) else {
                    continue;
                };
                if evidence.subject == subject {
                    let _ = graph.insert(evidence, now_secs());
                }
            }
            let _ = self.store.save_evidence(graph.entries());
        }
        let graph = self.trust.lock().await;
        serde_json::json!({
            "subject": subject,
            "decision": graph.decision(subject, now_secs()),
            "claim_level": format!("{:?}", graph.sybil_claim_level()),
            "evidence_count": graph.entries().len(),
            "rejected_evidence": graph.rejected_count(),
        })
    }

    async fn plan_training(
        &self,
        model_bytes: u64,
        locality_name: &str,
        requested_workers: usize,
    ) -> Result<serde_json::Value, NodeError> {
        let locality = parse_data_locality(locality_name)?;
        let records = self.network.peer_records().await;
        let graph = self.trust.lock().await;
        let now = now_secs();
        let mut profiles = Vec::new();
        for record in records {
            let Some(capability) = record
                .capabilities
                .iter()
                .find(|capability| capability.name == "training.reference")
            else {
                continue;
            };
            if capability.expires_at < now {
                continue;
            }
            let decision = graph.decision(record.node_id, now);
            let reliability = (0.5 + f32::from(decision.score) / 200.0).min(0.99);
            let dataset_available = !matches!(&locality, DataLocality::LocalOnly)
                || capability
                    .metadata
                    .iter()
                    .any(|metadata| metadata.key == "dataset_local" && metadata.value == "true");
            let hardware = capability
                .metadata
                .iter()
                .find(|metadata| metadata.key == "hardware")
                .map(|metadata| parse_hardware_kind(&metadata.value))
                .unwrap_or(HardwareKind::Cpu);
            let throughput_mbps = capability
                .metadata
                .iter()
                .find(|metadata| metadata.key == "throughput_mbps")
                .and_then(|metadata| metadata.value.parse().ok())
                .unwrap_or(0);
            profiles.push(WorkerProfile {
                node: record.node_id,
                hardware,
                memory_bytes: capability.resources.memory_bytes,
                compute_units: (capability.resources.cpu_millis / 1000).max(1) as u32,
                rtt_ms: record.observed_latency_ms.unwrap_or(5000),
                throughput_mbps,
                reliability,
                dataset_available,
                backends: capability.compute_backends.clone(),
            });
        }
        let decision = plan_training(model_bytes, locality, requested_workers, &profiles)?;
        Ok(serde_json::json!({
            "evidence_class": "REAL_PROCESS_LOCAL",
            "worker_profiles": profiles,
            "decision": decision,
        }))
    }

    async fn record_signed_evidence(&self, subject: NodeId, kind: EvidenceKind, payload: Vec<u8>) {
        if subject == self.node_id() {
            return;
        }
        let observed_at = now_secs();
        let mut evidence = SignedEvidence {
            issuer: self.node_id(),
            issuer_public_key: self.identity.public_key(),
            subject,
            kind,
            sequence: self.evidence_sequence.fetch_add(1, Ordering::Relaxed),
            observed_at,
            expires_at: observed_at.saturating_add(self.config.peer_ttl_seconds),
            payload,
            signature: Vec::new(),
        };
        let Ok(bytes) = evidence_signing_bytes(&evidence) else {
            return;
        };
        evidence.signature = self.identity.sign(&bytes);
        if let Ok(value) = serde_json::to_vec(&evidence) {
            let _ = self
                .network
                .dht_publish(
                    DhtNamespace::Evidence,
                    &format!("subject:{subject}"),
                    value,
                    self.config.peer_ttl_seconds,
                    evidence.sequence,
                )
                .await;
        }
        let mut graph = self.trust.lock().await;
        match graph.insert(evidence, observed_at) {
            Ok(true) => {
                if let Err(error) = self.store.save_evidence(graph.entries()) {
                    tracing::warn!(error = %error, "failed to persist trust evidence");
                }
            }
            Ok(false) => {}
            Err(error) => tracing::debug!(error = %error, "local trust evidence rejected"),
        }
    }

    pub async fn peers_json(&self) -> serde_json::Value {
        serde_json::to_value(self.network.peer_records().await)
            .unwrap_or_else(|_| serde_json::json!([]))
    }

    pub fn capabilities_json(&self) -> serde_json::Value {
        let mut values = self
            .capabilities
            .values()
            .map(|capability| {
                capability
                    .protocol_capability(now_secs().saturating_add(self.config.peer_ttl_seconds))
            })
            .collect::<Vec<_>>();
        if self.config.relay_enabled {
            values.push(relay_capability(
                now_secs().saturating_add(self.config.peer_ttl_seconds),
            ));
        }
        serde_json::to_value(values).unwrap_or_else(|_| serde_json::json!([]))
    }

    pub async fn jobs_json(&self) -> serde_json::Value {
        let jobs = self.jobs.lock().await.values().cloned().collect::<Vec<_>>();
        serde_json::to_value(jobs).unwrap_or_else(|_| serde_json::json!([]))
    }

    async fn start_admin(self: &Arc<Self>) -> Result<(), NodeError> {
        let path = self.admin_socket();
        if path.exists() {
            fs::remove_file(path)?;
        }
        let listener = UnixListener::bind(path)?;
        let node = self.clone();
        tokio::spawn(async move { node.admin_loop(listener).await });
        Ok(())
    }

    async fn admin_loop(self: Arc<Self>, listener: UnixListener) {
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let Ok((stream, _)) = accepted else { continue };
                    let node = self.clone();
                    tokio::spawn(async move { node.admin_connection(stream).await });
                }
                _ = self.shutdown.notified() => break,
            }
        }
    }

    async fn admin_connection(self: Arc<Self>, stream: UnixStream) {
        let (read_half, mut write_half) = stream.into_split();
        let reader = BufReader::new(read_half);
        let mut line = Vec::with_capacity(4096);
        let response = match reader
            .take((ADMIN_MAX_LINE + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
        {
            Ok(size) if size > 0 && size <= ADMIN_MAX_LINE => {
                match serde_json::from_slice::<AdminRequest>(&line) {
                    Ok(request) => self.handle_admin(request).await,
                    Err(error) => AdminResponse::error(format!("invalid admin request: {error}")),
                }
            }
            Ok(_) => AdminResponse::error("admin request is empty or oversized".to_string()),
            Err(error) => AdminResponse::error(error.to_string()),
        };
        if let Ok(bytes) = serde_json::to_vec(&response) {
            let _ = write_half.write_all(&bytes).await;
            let _ = write_half.write_all(b"\n").await;
        }
    }

    async fn handle_admin(self: &Arc<Self>, request: AdminRequest) -> AdminResponse {
        match request {
            AdminRequest::Status => response_result(self.status().await),
            AdminRequest::Identity => AdminResponse::ok(self.identity_json().await),
            AdminRequest::Peers => AdminResponse::ok(self.peers_json().await),
            AdminRequest::Capabilities => AdminResponse::ok(self.capabilities_json()),
            AdminRequest::Jobs => AdminResponse::ok(self.jobs_json().await),
            AdminRequest::Config => AdminResponse::ok(
                serde_json::to_value(&self.config).unwrap_or_else(|_| serde_json::json!({})),
            ),
            AdminRequest::DhtStats => AdminResponse::ok(
                serde_json::to_value(self.network.dht_stats().await)
                    .unwrap_or_else(|_| serde_json::json!({})),
            ),
            AdminRequest::DhtPublish {
                namespace,
                name,
                value,
                ttl_seconds,
                sequence,
            } => {
                let result = parse_dht_namespace(&namespace).map(|namespace| {
                    let ttl_seconds = ttl_seconds.unwrap_or(self.config.peer_ttl_seconds);
                    let sequence = sequence
                        .unwrap_or_else(|| self.dht_sequence.fetch_add(1, Ordering::Relaxed));
                    (namespace, ttl_seconds, sequence)
                });
                match result {
                    Ok((namespace, ttl_seconds, sequence)) => response_result(
                        self.network
                            .dht_publish(
                                namespace,
                                &name,
                                value.into_bytes(),
                                ttl_seconds,
                                sequence,
                            )
                            .await
                            .map(|record| serde_json::to_value(record).unwrap_or_default())
                            .map_err(NodeError::from),
                    ),
                    Err(error) => AdminResponse::error(error),
                }
            }
            AdminRequest::DhtLookup { namespace, name } => match parse_dht_namespace(&namespace) {
                Ok(namespace) => response_result(
                    self.network
                        .dht_lookup(namespace, &name)
                        .await
                        .map(|records| serde_json::to_value(records).unwrap_or_default())
                        .map_err(NodeError::from),
                ),
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::DhtFindNode { key } => match parse_dht_key(&key) {
                Ok(key) => response_result(
                    self.network
                        .dht_find_node(key)
                        .await
                        .map(|contacts| serde_json::to_value(contacts).unwrap_or_default())
                        .map_err(NodeError::from),
                ),
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::Trust { subject } => match parse_node_id(&subject) {
                Ok(subject) => AdminResponse::ok(self.trust_json(subject).await),
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::PlanTraining {
                model_bytes,
                data_locality,
                workers,
            } => response_result(
                self.plan_training(model_bytes, &data_locality, workers.unwrap_or(2) as usize)
                    .await,
            ),
            AdminRequest::Infer {
                capability,
                input,
                deadline_ms,
                max_output_bytes,
                allow_input_transfer,
                job_id,
            } => {
                let result = self
                    .submit_inference(
                        job_id.as_deref(),
                        capability,
                        input.into_bytes(),
                        deadline_ms.unwrap_or(30_000),
                        max_output_bytes.unwrap_or(64 * 1024),
                        allow_input_transfer.unwrap_or(true),
                    )
                    .await;
                response_result(result.map(|value| {
                    serde_json::to_value(value).unwrap_or_else(|_| serde_json::json!({}))
                }))
            }
            AdminRequest::RegisterModel {
                path,
                identity,
                format,
                local_only,
            } => match self.register_model(&path, &identity, &format, local_only) {
                Ok(manifest) => {
                    let mut model_published = false;
                    let mut artifact_published = false;
                    if !manifest.local_only {
                        if let Ok(value) = serde_json::to_vec(&manifest) {
                            model_published = self
                                .network
                                .dht_publish(
                                    DhtNamespace::Model,
                                    &manifest.identity,
                                    value.clone(),
                                    self.config.peer_ttl_seconds,
                                    self.dht_sequence.fetch_add(1, Ordering::Relaxed),
                                )
                                .await
                                .is_ok();
                            artifact_published = self
                                .network
                                .dht_publish(
                                    DhtNamespace::Artifact,
                                    &manifest.artifact.to_string(),
                                    value,
                                    self.config.peer_ttl_seconds,
                                    self.dht_sequence.fetch_add(1, Ordering::Relaxed),
                                )
                                .await
                                .is_ok();
                        }
                    }
                    match serde_json::to_value(&manifest) {
                        Ok(manifest_json) => AdminResponse::ok(serde_json::json!({
                            "artifact": manifest.artifact.to_string(),
                            "manifest": manifest_json,
                            "dht": {
                                "model_published": model_published,
                                "artifact_published": artifact_published,
                            },
                        })),
                        Err(error) => AdminResponse::error(error.to_string()),
                    }
                }
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::Inspect { artifact } => response_result(self.inspect_artifact(&artifact)),
            AdminRequest::FetchArtifact { peer, artifact } => {
                response_result(self.fetch_artifact(&peer, &artifact).await)
            }
            AdminRequest::Evaluate {
                text,
                expected_label,
                deadline_ms,
            } => {
                let result = self
                    .submit_evaluation(&text, &expected_label, deadline_ms.unwrap_or(30_000))
                    .await;
                response_result(result.map(|value| {
                    serde_json::to_value(value).unwrap_or_else(|_| serde_json::json!({}))
                }))
            }
            AdminRequest::TrainReference {
                workers,
                steps,
                resume_checkpoint,
            } => response_result(
                self.train_reference(
                    workers.unwrap_or(2),
                    steps.unwrap_or(8),
                    resume_checkpoint.as_deref(),
                )
                .await,
            ),
            AdminRequest::TrainV3 {
                workers,
                windows,
                local_steps,
                checkpoint_every,
            } => response_result(
                distributed_training::start_training(
                    self,
                    workers.unwrap_or(4),
                    windows.unwrap_or(8),
                    local_steps.unwrap_or(3),
                    checkpoint_every.unwrap_or(2),
                )
                .await,
            ),
            AdminRequest::TrainV3Start {
                workers,
                windows,
                local_steps,
                checkpoint_every,
            } => response_result(
                distributed_training::start_training_background(
                    self,
                    workers.unwrap_or(4),
                    windows.unwrap_or(8),
                    local_steps.unwrap_or(3),
                    checkpoint_every.unwrap_or(2),
                )
                .await,
            ),
            AdminRequest::TrainV4 {
                workers,
                windows,
                checkpoint_every,
            } => response_result(
                frontier_training::start_integrated(
                    self,
                    workers.unwrap_or(6),
                    windows.unwrap_or(8),
                    checkpoint_every.unwrap_or(2),
                )
                .await,
            ),
            AdminRequest::TrainV4Start {
                workers,
                windows,
                checkpoint_every,
            } => response_result(
                frontier_training::start_integrated_background(
                    self,
                    workers.unwrap_or(6),
                    windows.unwrap_or(8),
                    checkpoint_every.unwrap_or(2),
                )
                .await,
            ),
            AdminRequest::PlanTrainingV4 {
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            } => response_result(
                frontier_training::plan(
                    self,
                    model_bytes,
                    workers.unwrap_or(4),
                    &strategy,
                    tensor_degree.unwrap_or(0),
                    pipeline_stages.unwrap_or(0),
                )
                .await,
            ),
            AdminRequest::ReplanTrainingV4 {
                job_id,
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            } => match parse_job_id(&job_id) {
                Ok(job_id) => response_result(
                    frontier_training::replan(
                        self,
                        job_id,
                        model_bytes,
                        workers.unwrap_or(4),
                        &strategy,
                        tensor_degree.unwrap_or(0),
                        pipeline_stages.unwrap_or(0),
                    )
                    .await,
                ),
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::ActivateTrainingV4 { job_id } => match parse_job_id(&job_id) {
                Ok(job_id) => response_result(frontier_training::activate_plan(self, job_id).await),
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::V4SeedShard {
                job_id,
                shard_id,
                state,
            } => match parse_job_id(&job_id) {
                Ok(job_id) => response_result(
                    frontier_training::seed_shard(self, job_id, shard_id, state.into_bytes()).await,
                ),
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::V4MigrateShard {
                job_id,
                shard_id,
                target,
            } => match (parse_job_id(&job_id), parse_node_id(&target)) {
                (Ok(job_id), Ok(target)) => response_result(
                    frontier_training::migrate_shard(self, job_id, shard_id, target).await,
                ),
                (Err(error), _) | (_, Err(error)) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::V4TensorDemo { workers } => match parse_node_ids(&workers) {
                Ok(workers) => response_result(frontier_training::tensor_demo(self, workers).await),
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::V4PipelineDemo {
                stages,
                microbatches,
            } => match parse_node_ids(&stages) {
                Ok(stages) => response_result(
                    frontier_training::pipeline_demo(self, stages, microbatches.unwrap_or(2)).await,
                ),
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::V4CollectiveDemo { workers } => match parse_node_ids(&workers) {
                Ok(workers) => {
                    response_result(frontier_training::collective_demo(self, workers).await)
                }
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::V4ReconcileDemo {
                worker,
                left_value,
                right_value,
                policy,
            } => match (
                parse_node_id(&worker),
                frontier_training::parse_policy(&policy),
            ) {
                (Ok(worker), Ok(policy)) => response_result(
                    frontier_training::reconcile_demo(
                        self,
                        worker,
                        left_value,
                        right_value,
                        policy,
                    )
                    .await,
                ),
                (Err(error), _) => AdminResponse::error(error.to_string()),
                (_, Err(error)) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::V4ByzantineDemo {
                workers,
                malicious,
                policy,
            } => match (
                parse_node_ids(&workers),
                frontier_training::parse_byzantine_policy(&policy),
            ) {
                (Ok(workers), Ok(policy)) => response_result(
                    frontier_training::byzantine_demo(
                        self,
                        workers,
                        malicious.unwrap_or(1) as usize,
                        policy,
                    )
                    .await,
                ),
                (Err(error), _) => AdminResponse::error(error),
                (_, Err(error)) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::V4ReplicateState { workers } => match parse_node_ids(&workers) {
                Ok(workers) => {
                    response_result(frontier_training::replicate_state(self, workers).await)
                }
                Err(error) => AdminResponse::error(error),
            },
            AdminRequest::TrainingStatus { job_id } => match parse_job_id(&job_id) {
                Ok(job_id) => response_result(self.training_status(job_id).await),
                Err(error) => AdminResponse::error(error.to_string()),
            },
            AdminRequest::Cancel { job_id } => {
                let result = self.cancel_job(&job_id).await;
                response_result(
                    result.map(|cancelled| serde_json::json!({ "cancelled": cancelled })),
                )
            }
            AdminRequest::Shutdown => {
                self.shutdown().await;
                AdminResponse::ok(serde_json::json!({ "shutting_down": true }))
            }
        }
    }

    async fn submit_inference(
        &self,
        requested_job_id: Option<&str>,
        capability: String,
        input: Vec<u8>,
        deadline_ms: u64,
        max_output_bytes: u32,
        allow_input_transfer: bool,
    ) -> Result<JobResult, NodeError> {
        let job_id = requested_job_id
            .map(parse_job_id)
            .transpose()?
            .unwrap_or_else(random_job_id);
        let request = make_inference_job(
            job_id,
            self.node_id(),
            capability.clone(),
            input,
            deadline_ms,
            max_output_bytes,
            PrivacyPolicy {
                allow_input_transfer,
                ..PrivacyPolicy::default()
            },
        )?;
        self.submit_request(request).await
    }

    async fn submit_evaluation(
        &self,
        text: &str,
        expected_label: &str,
        deadline_ms: u64,
    ) -> Result<JobResult, NodeError> {
        let base_job_id = random_job_id();
        if self.capabilities.contains_key("evaluation.text") {
            let request = make_evaluation_job(
                base_job_id,
                self.node_id(),
                text,
                expected_label,
                deadline_ms,
            )?;
            let result = self.submit_request(request).await?;
            return self.attach_local_evaluation(result, expected_label);
        }
        let records = self.network.peer_records().await;
        let now = now_secs();
        let mut security = self.security.lock().await;
        let candidates = records
            .iter()
            .filter(|record| {
                record.expires_at >= now
                    && record.node_id != self.node_id()
                    && record.capabilities.iter().any(|capability| {
                        capability.name == "evaluation.text" && capability.expires_at >= now
                    })
            })
            .map(|record| {
                let assessment = security.assessment(record.node_id, now);
                EvaluatorCandidate {
                    node: record.node_id,
                    maturity: assessment.maturity,
                    direct_successes: assessment.direct_successes,
                    // This is a local, bounded diversity heuristic. The
                    // address is not published as trust evidence and does
                    // not assert independent human control.
                    source_group: record
                        .addresses
                        .first()
                        .map(|address| local_source_group(address))
                        .unwrap_or_default(),
                }
            })
            .collect::<Vec<_>>();
        drop(security);
        if candidates.is_empty() {
            // Preserve the normal local-capability and route behavior for
            // callers that have no remote evaluator candidates.
            let request = make_evaluation_job(
                base_job_id,
                self.node_id(),
                text,
                expected_label,
                deadline_ms,
            )?;
            let result = self.submit_request(request).await?;
            return self.attach_local_evaluation(result, expected_label);
        }
        let selection = select_evaluators(
            &candidates,
            candidates.len().min(3),
            u64::from_le_bytes(base_job_id.0[..8].try_into().unwrap_or([0; 8])),
            1,
        );
        let selected_evaluators = selection.selected.clone();
        self.note_security_event(
            None,
            SecurityEventKind::EvaluatorSelection,
            "evaluator_selection",
            &format!(
                "candidates={} selected={} source_groups={}",
                candidates.len(),
                selected_evaluators.len(),
                selection.source_groups
            ),
        )
        .await;
        let mut successful = Vec::new();
        let mut last_error = None;
        for peer in selected_evaluators {
            let request = make_evaluation_job(
                random_job_id(),
                self.node_id(),
                text,
                expected_label,
                deadline_ms,
            )?;
            match self.submit_remote_to_peer(peer, request).await {
                Ok(result) if result.state == JobState::Succeeded => {
                    let Some(output) = result.output.as_ref() else {
                        last_error = Some("evaluator returned no output".to_string());
                        continue;
                    };
                    let Ok(value) = serde_json::from_slice::<serde_json::Value>(output) else {
                        self.note_security_event(
                            Some(peer),
                            SecurityEventKind::EvaluatorDisagreement,
                            "evaluator_output",
                            "evaluator output was not valid JSON",
                        )
                        .await;
                        continue;
                    };
                    let Some(label) = value.get("label").and_then(serde_json::Value::as_str) else {
                        continue;
                    };
                    match score_builtin_output(peer, output, expected_label, now_secs()) {
                        Ok(evidence) => successful.push((label.to_string(), result, evidence)),
                        Err(error) => {
                            last_error = Some(error.to_string());
                            self.note_security_event(
                                Some(peer),
                                SecurityEventKind::EvaluatorDisagreement,
                                "evaluator_output",
                                "evaluator output failed local semantic validation",
                            )
                            .await;
                        }
                    }
                }
                Ok(result) => last_error = result.error,
                Err(error) => last_error = Some(error.to_string()),
            }
        }
        if successful.is_empty() {
            return Err(NodeError::InvalidConfig(
                last_error.unwrap_or_else(|| "all selected evaluators failed".to_string()),
            ));
        }
        let mut labels = HashMap::<String, usize>::new();
        for (label, _, _) in &successful {
            *labels.entry(label.clone()).or_default() += 1;
        }
        let disagreement = labels.len() > 1;
        let majority = labels
            .into_iter()
            .max_by_key(|(label, count)| (*count, std::cmp::Reverse(label.clone())))
            .map(|(label, _)| label)
            .unwrap_or_default();
        if disagreement {
            self.note_security_event(
                None,
                SecurityEventKind::EvaluatorDisagreement,
                "evaluator_selection",
                "selected evaluator outputs disagreed; deterministic majority used",
            )
            .await;
        }
        let (_, result, evidence) = successful
            .into_iter()
            .find(|(label, _, _)| *label == majority)
            .unwrap_or_else(|| unreachable!("majority label must have a result"));
        Ok(JobResult {
            evidence: Some(evidence),
            ..result
        })
    }

    fn attach_local_evaluation(
        &self,
        result: JobResult,
        expected_label: &str,
    ) -> Result<JobResult, NodeError> {
        if let Some(output) = result.output.as_ref() {
            let evidence =
                score_builtin_output(self.node_id(), output, expected_label, now_secs())?;
            return Ok(JobResult {
                evidence: Some(evidence),
                ..result
            });
        }
        Ok(result)
    }

    async fn train_reference(
        &self,
        requested_workers: u16,
        max_steps: u64,
        resume_checkpoint: Option<&str>,
    ) -> Result<serde_json::Value, NodeError> {
        if !(2..=16).contains(&requested_workers) || !(1..=64).contains(&max_steps) {
            return Err(NodeError::InvalidConfig(
                "reference training requires 2-16 workers and 1-64 steps".to_string(),
            ));
        }
        let records = self.network.peer_records().await;
        let mut workers = records
            .iter()
            .filter(|record| {
                record.node_id != self.node_id()
                    && record.capabilities.iter().any(|capability| {
                        capability.name == "training.reference"
                            && capability.expires_at >= now_secs()
                            && capability.resources.memory_bytes >= 64 * 1024 * 1024
                    })
            })
            .map(|record| record.node_id)
            .collect::<Vec<_>>();
        workers.sort_unstable();
        workers.truncate(requested_workers as usize);
        if workers.len() < 2 {
            return Err(NodeError::InvalidConfig(
                "fewer than two live training.reference peers are known".to_string(),
            ));
        }
        let (
            samples,
            initial_model,
            initial_model_artifact,
            dataset_artifact,
            start_step,
            mut previous_checkpoint,
            resumed_from,
        ) = if let Some(checkpoint_value) = resume_checkpoint {
            let checkpoint_artifact = parse_artifact_id(checkpoint_value)?;
            self.store.verify_artifact(checkpoint_artifact)?;
            let checkpoint_bytes = self.store.get_artifact(checkpoint_artifact)?;
            let checkpoint: ReferenceCheckpoint = serde_json::from_slice(&checkpoint_bytes)
                .map_err(|error| {
                    NodeError::InvalidConfig(format!(
                        "reference checkpoint {checkpoint_artifact} is invalid: {error}"
                    ))
                })?;
            if checkpoint.kind != "reference_checkpoint"
                || checkpoint.step == 0
                || checkpoint.workers.is_empty()
                || checkpoint.workers.len() > 16
                || checkpoint.parent == Some(checkpoint_artifact)
                || !checkpoint.model.weight.is_finite()
                || !checkpoint.model.bias.is_finite()
                || !checkpoint.optimizer.learning_rate.is_finite()
                || checkpoint.optimizer.learning_rate <= 0.0
            {
                return Err(NodeError::InvalidConfig(
                    "reference checkpoint failed structural validation".to_string(),
                ));
            }
            self.store.verify_artifact(checkpoint.model_artifact)?;
            self.store.verify_artifact(checkpoint.dataset_artifact)?;
            let samples: Vec<ReferenceSample> =
                serde_json::from_slice(&self.store.get_artifact(checkpoint.dataset_artifact)?)
                    .map_err(|error| {
                        NodeError::InvalidConfig(format!(
                            "reference checkpoint dataset is invalid: {error}"
                        ))
                    })?;
            if samples.is_empty() || samples.len() > 4096 {
                return Err(NodeError::InvalidConfig(
                    "reference checkpoint dataset has an invalid size".to_string(),
                ));
            }
            let model: ReferenceModel =
                serde_json::from_slice(&self.store.get_artifact(checkpoint.model_artifact)?)
                    .map_err(|error| {
                        NodeError::InvalidConfig(format!(
                            "reference checkpoint model is invalid: {error}"
                        ))
                    })?;
            if model != checkpoint.model {
                return Err(NodeError::InvalidConfig(
                    "reference checkpoint model does not match its model artifact".to_string(),
                ));
            }
            (
                samples,
                model,
                checkpoint.model_artifact,
                checkpoint.dataset_artifact,
                checkpoint.step.checked_add(1).ok_or_else(|| {
                    NodeError::InvalidConfig("reference checkpoint step overflow".to_string())
                })?,
                Some(checkpoint_artifact),
                Some(checkpoint_artifact),
            )
        } else {
            let samples = reference_samples();
            let initial_model = ReferenceModel {
                weight: 0.0,
                bias: 0.0,
            };
            let initial_model_bytes = serde_json::to_vec(&initial_model)?;
            let initial_model_artifact = self.store.put_artifact(&initial_model_bytes)?;
            let dataset_bytes = serde_json::to_vec(&samples)?;
            let dataset_artifact = self.store.put_artifact(&dataset_bytes)?;
            let dataset_manifest = DatasetManifest {
                artifact: dataset_artifact,
                identity: "reference.linear.synthetic.v1".to_string(),
                format: "json.samples".to_string(),
                size: dataset_bytes.len() as u64,
                shards: Vec::new(),
                locality: DataLocality::Selective,
                sample_policy: "repository-contained deterministic samples".to_string(),
            };
            dataset_manifest
                .validate()
                .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
            self.store.write_json(
                &format!("dataset-{dataset_artifact}.json"),
                &dataset_manifest,
            )?;
            let initial_manifest = ModelManifest {
                artifact: initial_model_artifact,
                identity: "reference.linear.v1".to_string(),
                format: "json.linear".to_string(),
                size: initial_model_bytes.len() as u64,
                weights: Vec::new(),
                capabilities: vec!["training.reference".to_string()],
                runtime_requirements: Vec::new(),
                adapters: Vec::new(),
                local_path: None,
                local_only: false,
            };
            initial_manifest
                .validate()
                .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
            self.store.write_json(
                &format!("model-{initial_model_artifact}.json"),
                &initial_manifest,
            )?;
            (
                samples,
                initial_model,
                initial_model_artifact,
                dataset_artifact,
                1,
                None,
                None,
            )
        };
        let mut plan = TrainingPlan {
            model: initial_model_artifact,
            dataset: dataset_artifact,
            checkpoint: previous_checkpoint,
            workers: workers.len() as u16,
            sync: SyncMode::Synchronous,
            max_steps,
            checkpoint_every: 1,
            evaluation_capability: Some("evaluation.text".to_string()),
        };
        plan.validate()
            .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
        let learning_rate = 0.1;
        let initial_loss = reference_loss(initial_model, &samples);
        let partitions = partition_samples(&samples, requested_workers as usize);
        let mut worker_partitions = workers
            .iter()
            .enumerate()
            .map(|(index, worker)| (*worker, partitions[index % partitions.len()].clone()))
            .collect::<HashMap<_, _>>();
        let initial_worker_count = workers.len();
        let mut worker_joins = 0usize;
        let mut active_workers = workers.clone();
        let mut model = initial_model;
        let mut current_model_artifact = initial_model_artifact;
        let mut checkpoints = Vec::new();
        let mut failed_workers = Vec::new();
        let mut rejected_updates = 0usize;
        for step_offset in 0..max_steps {
            let step = start_step.checked_add(step_offset).ok_or_else(|| {
                NodeError::InvalidConfig("reference training step overflow".to_string())
            })?;
            if workers.len() < requested_workers as usize {
                let known = self.network.peer_records().await;
                for record in known {
                    if workers.len() >= requested_workers as usize
                        || record.node_id == self.node_id()
                        || workers.contains(&record.node_id)
                        || !record.capabilities.iter().any(|capability| {
                            capability.name == "training.reference"
                                && capability.expires_at >= now_secs()
                        })
                    {
                        continue;
                    }
                    let index = workers.len();
                    workers.push(record.node_id);
                    active_workers.push(record.node_id);
                    worker_partitions
                        .insert(record.node_id, partitions[index % partitions.len()].clone());
                    worker_joins = worker_joins.saturating_add(1);
                }
                plan.workers = workers.len() as u16;
            }
            let mut gradients = Vec::new();
            let mut survivors = Vec::new();
            for peer in workers.clone() {
                if !active_workers.contains(&peer) {
                    continue;
                }
                let Some(worker_samples) = worker_partitions.get(&peer) else {
                    continue;
                };
                let training_job_id = random_job_id();
                let input = serde_json::to_vec(&serde_json::json!({
                    "model": model,
                    "samples": worker_samples,
                    "learning_rate": learning_rate,
                    "step": step,
                    "job_id": training_job_id.to_string(),
                    "model_artifact": current_model_artifact.to_string(),
                    "dataset_artifact": dataset_artifact.to_string(),
                }))?;
                let request = JobRequest {
                    job_id: training_job_id,
                    origin: self.node_id(),
                    kind: JobKind::Training,
                    capability: "training.reference".to_string(),
                    model: Some(current_model_artifact),
                    input,
                    deadline_ms: 10_000,
                    max_output_bytes: 16 * 1024,
                    privacy: PrivacyPolicy::default(),
                };
                let request_job_id = request.job_id;
                match self.submit_remote_to_peer(peer, request).await {
                    Ok(result) if result.state == JobState::Succeeded => {
                        let output = result.output.ok_or_else(|| {
                            NodeError::InvalidConfig(
                                "training worker returned no gradient".to_string(),
                            )
                        })?;
                        let Ok(gradient) = serde_json::from_slice::<ReferenceGradient>(&output)
                        else {
                            rejected_updates = rejected_updates.saturating_add(1);
                            failed_workers.push(peer);
                            continue;
                        };
                        if gradient.job_id != request_job_id.to_string()
                            || gradient.model_artifact != current_model_artifact.to_string()
                            || gradient.dataset_artifact != dataset_artifact.to_string()
                            || result.output_hash != Some(ArtifactId::from_bytes_hashed(&output))
                            || gradient.step != step
                            || gradient.samples == 0
                            || !gradient.loss.is_finite()
                            || !gradient.weight_gradient.is_finite()
                            || !gradient.bias_gradient.is_finite()
                            || gradient.weight_gradient.abs() > 100.0
                            || gradient.bias_gradient.abs() > 100.0
                        {
                            rejected_updates = rejected_updates.saturating_add(1);
                            failed_workers.push(peer);
                            continue;
                        }
                        gradients.push(gradient);
                        survivors.push(peer);
                    }
                    Ok(result) => {
                        tracing::warn!(
                            peer = %peer,
                            state = ?result.state,
                            error = ?result.error,
                            "reference training worker failed"
                        );
                        failed_workers.push(peer);
                    }
                    Err(error) => {
                        tracing::warn!(peer = %peer, error = %error, "reference training worker unavailable");
                        failed_workers.push(peer);
                    }
                }
            }
            if gradients.is_empty() {
                return Err(NodeError::InvalidConfig(
                    "all reference training workers failed before a synchronized update"
                        .to_string(),
                ));
            }
            let total_samples = gradients
                .iter()
                .map(|gradient| gradient.samples)
                .sum::<usize>() as f64;
            let weight_gradient = gradients
                .iter()
                .map(|gradient| gradient.weight_gradient * gradient.samples as f64)
                .sum::<f64>()
                / total_samples;
            let bias_gradient = gradients
                .iter()
                .map(|gradient| gradient.bias_gradient * gradient.samples as f64)
                .sum::<f64>()
                / total_samples;
            model.weight -= learning_rate * weight_gradient;
            model.bias -= learning_rate * bias_gradient;
            active_workers = survivors;
            let model_bytes = serde_json::to_vec(&model)?;
            let model_artifact = self.store.put_artifact(&model_bytes)?;
            let checkpoint_bytes = serde_json::to_vec(&serde_json::json!({
                "kind": "reference_checkpoint",
                "step": step,
                "model": model,
                "optimizer": {"learning_rate": learning_rate},
                "parent": previous_checkpoint,
                "workers": active_workers,
                "model_artifact": model_artifact,
                "dataset_artifact": dataset_artifact,
            }))?;
            let checkpoint_artifact = self.store.put_artifact(&checkpoint_bytes)?;
            let checkpoint = CheckpointManifest {
                artifact: checkpoint_artifact,
                model: model_artifact,
                parent: previous_checkpoint,
                step,
                complete_weights: true,
                complete_optimizer: true,
                complete_scheduler: true,
                complete_rng: true,
                workers: active_workers.len() as u16,
                hash: checkpoint_artifact,
            };
            checkpoint
                .validate()
                .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
            self.store.write_json(
                &format!("checkpoint-{checkpoint_artifact}.json"),
                &checkpoint,
            )?;
            previous_checkpoint = Some(checkpoint_artifact);
            checkpoints.push(checkpoint_artifact);
            current_model_artifact = model_artifact;
            plan.checkpoint = previous_checkpoint;
        }
        let final_loss = reference_loss(model, &samples);
        let final_model_bytes = serde_json::to_vec(&model)?;
        let final_model_artifact = self.store.put_artifact(&final_model_bytes)?;
        let final_manifest = ModelManifest {
            artifact: final_model_artifact,
            identity: "reference.linear.v1".to_string(),
            format: "json.linear".to_string(),
            size: final_model_bytes.len() as u64,
            weights: Vec::new(),
            capabilities: vec![
                "inference.reference".to_string(),
                "training.reference".to_string(),
            ],
            runtime_requirements: Vec::new(),
            adapters: Vec::new(),
            local_path: None,
            local_only: false,
        };
        final_manifest
            .validate()
            .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
        self.store.write_json(
            &format!("model-{final_model_artifact}.json"),
            &final_manifest,
        )?;
        let evaluation_bytes = serde_json::to_vec(&serde_json::json!({
            "model": final_model_artifact,
            "dataset": dataset_artifact,
            "mse": final_loss,
            "initial_mse": initial_loss,
        }))?;
        let evaluation_artifact = self.store.put_artifact(&evaluation_bytes)?;
        let evidence = intelligence_protocol::Evidence {
            evaluator: self.node_id(),
            evaluation_id: evaluation_artifact,
            score: ((initial_loss - final_loss) * 1_000_000.0) as i64,
            score_scale: "mse_improvement_micro".to_string(),
            result_hash: evaluation_artifact,
            observed_at: now_secs(),
            verified: final_loss < initial_loss,
        };
        let result = serde_json::json!({
            "kind": "v1_distributed_training_reference",
            "coordinator": self.node_id().to_string(),
            "plan": plan,
            "workers": workers,
            "initial_worker_count": initial_worker_count,
            "worker_joins": worker_joins,
            "active_workers": active_workers,
            "failed_workers": failed_workers,
            "rejected_updates": rejected_updates,
            "initial_model": initial_model_artifact.to_string(),
            "dataset": dataset_artifact.to_string(),
            "checkpoints": checkpoints.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "resumed_from": resumed_from.map(|artifact| artifact.to_string()),
            "start_step": start_step,
            "final_model": final_model_artifact.to_string(),
            "evaluation": evidence,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improved": final_loss < initial_loss,
        });
        if !final_loss.is_finite() || final_loss >= initial_loss {
            return Err(NodeError::InvalidConfig(
                "reference training did not improve its deterministic evaluation target"
                    .to_string(),
            ));
        }
        let output = serde_json::to_vec(&result)?;
        self.record_job(PersistedJob {
            job_id: random_job_id(),
            origin: self.node_id(),
            state: JobState::Succeeded,
            updated_at: now_secs(),
            output_hash: Some(final_model_artifact),
            output: Some(output),
            error: None,
        })
        .await?;
        Ok(result)
    }

    async fn submit_remote_to_peer(
        &self,
        peer: NodeId,
        request: JobRequest,
    ) -> Result<JobResult, NodeError> {
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(
            request.job_id,
            PendingJob {
                job_id: request.job_id,
                peer,
                capability: request.capability.clone(),
                started_at: now_millis(),
                last_sequence: 0,
                sender,
            },
        );
        if let Err(error) = self
            .network
            .send_to(peer, Message::JobRequest(request.clone()))
            .await
        {
            self.pending.lock().await.remove(&request.job_id);
            return Err(error.into());
        }
        match timeout(
            Duration::from_millis(request.deadline_ms.saturating_add(5_000)),
            receiver,
        )
        .await
        {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(NodeError::Network(NetworkError::NotConnected)),
            Err(_) => {
                self.pending.lock().await.remove(&request.job_id);
                // A relay can accept an envelope into its local queue and then disappear
                // before the worker receives it. Retry the same job ID once over the direct
                // peer path; the worker's persistent job reservation makes this at-most-once.
                let (retry_sender, retry_receiver) = oneshot::channel();
                self.pending.lock().await.insert(
                    request.job_id,
                    PendingJob {
                        job_id: request.job_id,
                        peer,
                        capability: request.capability.clone(),
                        started_at: now_millis(),
                        last_sequence: 0,
                        sender: retry_sender,
                    },
                );
                let _ = self.network.reconnect_peer(peer).await;
                if self
                    .network
                    .send_direct(peer, Message::JobRequest(request.clone()))
                    .await
                    .is_ok()
                {
                    if let Ok(Ok(result)) = timeout(
                        Duration::from_millis(
                            request.deadline_ms.min(30_000).saturating_add(1_000),
                        ),
                        retry_receiver,
                    )
                    .await
                    {
                        return Ok(result);
                    }
                }
                self.pending.lock().await.remove(&request.job_id);
                let _ = self
                    .network
                    .send_direct(
                        peer,
                        Message::CancelJob(intelligence_protocol::CancelJob {
                            job_id: request.job_id,
                            reason: "training coordinator deadline exceeded".to_string(),
                        }),
                    )
                    .await;
                Err(NodeError::InvalidConfig(
                    "training worker deadline exceeded".to_string(),
                ))
            }
        }
    }

    fn register_model(
        &self,
        path: &str,
        identity: &str,
        format: &str,
        local_only: bool,
    ) -> Result<ModelManifest, NodeError> {
        if identity.is_empty() || identity.len() > intelligence_protocol::MAX_STRING_LEN {
            return Err(NodeError::InvalidConfig(
                "model identity is empty or too long".to_string(),
            ));
        }
        if format.is_empty() || format.len() > intelligence_protocol::MAX_STRING_LEN {
            return Err(NodeError::InvalidConfig(
                "model format is empty or too long".to_string(),
            ));
        }
        let source = Path::new(path);
        let (artifact, size) = self.store.import_artifact(source)?;
        let manifest = ModelManifest {
            artifact,
            identity: identity.to_string(),
            format: format.to_string(),
            size,
            weights: Vec::new(),
            capabilities: Vec::new(),
            runtime_requirements: Vec::new(),
            adapters: Vec::new(),
            local_path: Some(source.to_string_lossy().into_owned()),
            local_only,
        };
        manifest
            .validate()
            .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
        self.store
            .write_json(&format!("model-{artifact}.json"), &manifest)?;
        Ok(manifest)
    }

    fn inspect_artifact(&self, value: &str) -> Result<serde_json::Value, NodeError> {
        let artifact = parse_artifact_id(value)?;
        self.store.verify_artifact(artifact)?;
        let size = self.store.artifact_size(artifact)?;
        Ok(serde_json::json!({
            "artifact": artifact.to_string(),
            "size": size,
            "hash": artifact.to_string(),
            "verified": true,
        }))
    }

    async fn fetch_artifact(
        &self,
        peer: &str,
        artifact: &str,
    ) -> Result<serde_json::Value, NodeError> {
        let peer = parse_node_id(peer)?;
        let artifact = parse_artifact_id(artifact)?;
        if self.store.has_artifact(artifact) {
            self.store.verify_artifact(artifact)?;
            return Ok(serde_json::json!({
                "artifact": artifact.to_string(),
                "size": self.store.artifact_size(artifact)?,
                "verified": true,
                "source": "local",
            }));
        }
        if self.pending_artifacts.lock().await.len() >= 8 {
            return Err(NodeError::InvalidConfig(
                "artifact transfer concurrency limit reached".to_string(),
            ));
        }
        for attempt in 0..2 {
            let request_id = random_request_id();
            let offset = self.store.partial_artifact_size(artifact)?;
            let (sender, receiver) = oneshot::channel();
            self.pending_artifacts.lock().await.insert(
                request_id,
                PendingArtifact {
                    peer,
                    artifact,
                    offset,
                    expected_size: None,
                    sender,
                },
            );
            let request = ArtifactTransferRequest {
                request_id,
                artifact,
                offset,
                max_bytes: MAX_ARTIFACT_CHUNK as u32,
                expected_size: 0,
                expected_hash: artifact,
            };
            let send_result = if attempt == 0 {
                self.network
                    .send_to(peer, Message::ArtifactTransferRequest(request))
                    .await
            } else {
                let _ = self.network.reconnect_peer(peer).await;
                self.network
                    .send_direct(peer, Message::ArtifactTransferRequest(request))
                    .await
            };
            if let Err(error) = send_result {
                self.pending_artifacts.lock().await.remove(&request_id);
                if attempt == 0 {
                    continue;
                }
                return Err(error.into());
            }
            let wait = if attempt == 0 {
                Duration::from_secs(5)
            } else {
                Duration::from_secs(300)
            };
            match timeout(wait, receiver).await {
                Ok(Ok(result)) => return result,
                Ok(Err(_)) | Err(_) => {
                    self.pending_artifacts.lock().await.remove(&request_id);
                    if attempt == 0 {
                        continue;
                    }
                    return Err(NodeError::InvalidConfig(
                        "artifact transfer timed out; the partial artifact was retained for resume"
                            .to_string(),
                    ));
                }
            }
        }
        Err(NodeError::Network(NetworkError::NotConnected))
    }

    async fn submit_request(&self, request: JobRequest) -> Result<JobResult, NodeError> {
        if let Some(capability) = self.capabilities.get(&request.capability) {
            return self.execute_local(request, capability).await;
        }
        if !request.privacy.allow_input_transfer {
            return Err(NodeError::InvalidConfig(format!(
                "no local capability {} satisfies --local-only",
                request.capability
            )));
        }
        let records = self.network.peer_records().await;
        let choice = choose_peer(
            &records,
            &request.capability,
            now_secs(),
            Some(self.node_id()),
        )
        .ok_or_else(|| {
            NodeError::InvalidConfig(format!("no route for capability {}", request.capability))
        })?;
        let (sender, receiver) = oneshot::channel();
        self.pending.lock().await.insert(
            request.job_id,
            PendingJob {
                job_id: request.job_id,
                peer: choice.peer,
                capability: request.capability.clone(),
                started_at: now_millis(),
                last_sequence: 0,
                sender,
            },
        );
        if let Err(error) = self
            .network
            .send_to(choice.peer, Message::JobRequest(request.clone()))
            .await
        {
            self.pending.lock().await.remove(&request.job_id);
            return Err(error.into());
        }
        match timeout(
            Duration::from_millis(request.deadline_ms.saturating_add(5000)),
            receiver,
        )
        .await
        {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err(NodeError::Network(NetworkError::NotConnected)),
            Err(_) => {
                self.pending.lock().await.remove(&request.job_id);

                // A relay accepts an envelope before it can know whether the target's
                // connection is currently usable. Retry the same job ID over the direct
                // path once. The worker persists job reservations, so this is at-most-once
                // execution while still recovering a request that was dropped at a relay
                // during peer reconnect.
                let (retry_sender, retry_receiver) = oneshot::channel();
                self.pending.lock().await.insert(
                    request.job_id,
                    PendingJob {
                        job_id: request.job_id,
                        peer: choice.peer,
                        capability: request.capability.clone(),
                        started_at: now_millis(),
                        last_sequence: 0,
                        sender: retry_sender,
                    },
                );
                let _ = self.network.reconnect_peer(choice.peer).await;
                if self
                    .network
                    .send_direct(choice.peer, Message::JobRequest(request.clone()))
                    .await
                    .is_ok()
                {
                    if let Ok(Ok(result)) = timeout(
                        Duration::from_millis(
                            request.deadline_ms.min(30_000).saturating_add(1_000),
                        ),
                        retry_receiver,
                    )
                    .await
                    {
                        return Ok(result);
                    }
                }
                self.pending.lock().await.remove(&request.job_id);
                let _ = self
                    .network
                    .send_direct(
                        choice.peer,
                        Message::CancelJob(intelligence_protocol::CancelJob {
                            job_id: request.job_id,
                            reason: "requester deadline exceeded".to_string(),
                        }),
                    )
                    .await;
                Err(NodeError::InvalidConfig(
                    "remote job deadline exceeded".to_string(),
                ))
            }
        }
    }

    async fn execute_local(
        &self,
        request: JobRequest,
        capability: &CapabilityConfig,
    ) -> Result<JobResult, NodeError> {
        if request.input.len() > capability.max_input_bytes as usize
            || request.max_output_bytes > capability.max_output_bytes
        {
            return Err(NodeError::InvalidConfig(format!(
                "capability {} resource limits reject this job",
                capability.name
            )));
        }
        let executor = capability.executor()?;
        if !matches!(
            self.reserve_remote_job(self.node_id(), &request).await?,
            RemoteJobReservation::Reserved
        ) {
            return Err(NodeError::InvalidConfig(format!(
                "job {} already exists",
                request.job_id
            )));
        }
        let admission = match self.runtime.admit(&request).await {
            Ok(admission) => admission,
            Err(error) => {
                let _ = self
                    .record_job(PersistedJob {
                        job_id: request.job_id,
                        origin: request.origin,
                        state: JobState::Rejected,
                        updated_at: now_secs(),
                        output_hash: None,
                        output: None,
                        error: Some(error.to_string()),
                    })
                    .await;
                return Err(error.into());
            }
        };
        if let Err(error) = self
            .record_job(PersistedJob {
                job_id: request.job_id,
                origin: request.origin,
                state: JobState::Admitted,
                updated_at: now_secs(),
                output_hash: None,
                output: None,
                error: None,
            })
            .await
        {
            self.runtime.abandon(admission).await;
            return Err(error);
        }
        let outcome = match self.runtime.execute(admission, &request, &executor).await {
            Ok(outcome) => outcome,
            Err(error) => {
                self.metrics
                    .execution_failures
                    .fetch_add(1, Ordering::Relaxed);
                let _ = self
                    .record_job(PersistedJob {
                        job_id: request.job_id,
                        origin: request.origin,
                        state: JobState::Failed,
                        updated_at: now_secs(),
                        output_hash: None,
                        output: None,
                        error: Some(error.to_string()),
                    })
                    .await;
                return Err(error.into());
            }
        };
        self.outcome_to_result(request, outcome, None).await
    }

    async fn event_loop(self: Arc<Self>, mut events: mpsc::Receiver<NetworkEvent>) {
        loop {
            let Some(event) = (tokio::select! {
                event = events.recv() => event,
                _ = self.shutdown.notified() => return,
            }) else {
                return;
            };
            match event {
                NetworkEvent::PeerConnected(peer) => {
                    tracing::info!(peer = %peer.node_id, "peer connected")
                }
                NetworkEvent::PeerDisconnected(peer) => {
                    self.fail_pending_for_peer(peer).await;
                    self.v3_peer_disconnected(peer).await;
                    frontier_training::peer_disconnected(&self, peer).await;
                }
                NetworkEvent::ProtocolError { peer, message } => {
                    tracing::warn!(peer = ?peer, error = %message, "peer protocol error")
                }
                NetworkEvent::Message { peer, message } => {
                    let node = self.clone();
                    tokio::spawn(async move { node.handle_network_message(peer, message).await });
                }
            }
        }
    }

    async fn handle_network_message(self: Arc<Self>, peer: NodeId, message: Message) {
        match message {
            Message::Training(training) => self.handle_v3_message(peer, training).await,
            Message::TrainingV4(training) => self.handle_v4_message(peer, training).await,
            Message::TrainingV5(training) => self.handle_v5_message(peer, training).await,
            Message::JobRequest(request) => self.handle_remote_job(peer, request).await,
            Message::JobUpdate(update) => self.handle_job_update(peer, update).await,
            Message::CancelJob(cancel) => {
                let allowed = self
                    .jobs
                    .lock()
                    .await
                    .get(&cancel.job_id)
                    .is_some_and(|job| job.origin == peer);
                if allowed {
                    let _ = self.runtime.cancel(cancel.job_id).await;
                }
            }
            Message::ArtifactRequest(request) => self.handle_artifact_request(peer, request).await,
            Message::ArtifactChunk(_chunk) => {}
            Message::ArtifactTransferRequest(request) => {
                self.handle_artifact_transfer_request(peer, request).await
            }
            Message::ArtifactTransferChunk(chunk) => {
                self.handle_artifact_transfer_chunk(peer, chunk).await
            }
            Message::AddressUpdate(_)
            | Message::AddressObservation(_)
            | Message::KeyRotation(_)
            | Message::RelayEnvelope(_)
            | Message::DhtRequest(_)
            | Message::DhtResponse(_) => {}
            Message::PeerExchange(_)
            | Message::Announcement(_)
            | Message::Pong(_)
            | Message::Ping(_)
            | Message::Hello(_)
            | Message::Goodbye { .. }
            | Message::Error(_) => {}
        }
    }

    async fn handle_v5_message(&self, peer: NodeId, message: TrainingV5Message) {
        match message {
            TrainingV5Message::CapabilityChallenge(challenge) => {
                if challenge.worker != self.node_id()
                    || challenge.task_kind
                        != intelligence_protocol::ComputeTaskKind::CapabilityChallenge
                {
                    return;
                }
                let authorized = self
                    .v4_plans
                    .lock()
                    .await
                    .get(&challenge.job_id)
                    .is_some_and(|plan| plan.workers.contains(&peer) || plan.proposer == peer);
                if !authorized {
                    tracing::debug!(peer = %peer, job = %challenge.job_id, "rejecting unauthorized V5 capability challenge");
                    return;
                }
                let now = now_secs().max(1);
                let allowed = {
                    let mut windows = self.v5_challenge_windows.lock().await;
                    let entry = windows.entry(peer).or_insert((now, 0));
                    if now.saturating_sub(entry.0) >= 60 {
                        *entry = (now, 0);
                    }
                    if entry.1 >= 8 {
                        false
                    } else {
                        entry.1 = entry.1.saturating_add(1);
                        true
                    }
                };
                if !allowed {
                    tracing::debug!(peer = %peer, "V5 capability challenge rate limit reached");
                    return;
                }
                let result = self.compute.lock().await.challenge(&challenge, now);
                let _ = timeout(
                    Duration::from_secs(6),
                    self.network.send_to(
                        peer,
                        Message::TrainingV5(TrainingV5Message::CapabilityChallengeResult(result)),
                    ),
                )
                .await;
            }
            TrainingV5Message::CapabilityChallengeResult(result) => {
                if result.worker != peer {
                    return;
                }
                if let Err(error) = Message::TrainingV5(
                    TrainingV5Message::CapabilityChallengeResult(result.clone()),
                )
                .validate()
                {
                    tracing::debug!(peer = %peer, error = %error, "discarding invalid V5 challenge result");
                    return;
                }
                {
                    let mut security = self.security.lock().await;
                    security.record_capability_challenge(
                        peer,
                        result.backend,
                        result.success,
                        now_secs(),
                    );
                }
                self.persist_security_state().await;
                tracing::info!(
                    peer = %peer,
                    backend = ?result.backend,
                    success = result.success,
                    elapsed_micros = result.elapsed_micros,
                    "received bounded V5 capability evidence"
                );
            }
            TrainingV5Message::CapabilityEvidence(evidence) => {
                if evidence.worker != peer {
                    return;
                }
                tracing::debug!(
                    peer = %peer,
                    backend = ?evidence.backend,
                    successful_challenges = evidence.successful_challenges,
                    failed_challenges = evidence.failed_challenges,
                    "received V5 local capability evidence"
                );
            }
        }
    }

    async fn handle_v3_message(self: Arc<Self>, peer: NodeId, message: TrainingMessage) {
        if let TrainingMessage::Start(start) = message {
            if let Err(error) = distributed_training::accept_start(self, peer, start).await {
                tracing::warn!(peer = %peer, error = %error, "V3 training start rejected");
            }
            return;
        }
        let job_id = match &message {
            TrainingMessage::Window(value) => value.job_id,
            TrainingMessage::Update(value) => value.job_id,
            TrainingMessage::Aggregate(value) => value.job_id,
            TrainingMessage::ShardState(value) => value.job_id,
            TrainingMessage::State(value) => value.job_id,
            TrainingMessage::Ack(value) => value.job_id,
            TrainingMessage::Election(value) => value.job_id,
            TrainingMessage::CheckpointOffer(value) => value.job_id,
            TrainingMessage::CheckpointCommit(value) => value.job_id,
            TrainingMessage::Start(_) => return,
        };
        match &message {
            TrainingMessage::Update(_) => {
                self.metrics
                    .training_updates_received
                    .fetch_add(1, Ordering::Relaxed);
            }
            TrainingMessage::Aggregate(_) => {
                self.metrics
                    .training_aggregates_received
                    .fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let sender = self
            .v3_jobs
            .lock()
            .await
            .get(&job_id)
            .map(|job| job.sender.clone());
        if let Some(sender) = sender {
            let _ = sender
                .send(distributed_training::V3Inbound::Message { peer, message })
                .await;
        }
    }

    async fn handle_v4_message(self: Arc<Self>, peer: NodeId, message: TrainingV4Message) {
        frontier_training::handle_message(self, peer, message).await;
    }

    async fn v3_peer_disconnected(&self, peer: NodeId) {
        let senders = self
            .v3_jobs
            .lock()
            .await
            .values()
            .map(|job| job.sender.clone())
            .collect::<Vec<_>>();
        for sender in senders {
            let _ = sender
                .send(distributed_training::V3Inbound::PeerDisconnected(peer))
                .await;
        }
    }

    async fn handle_job_update(&self, peer: NodeId, update: JobUpdate) {
        if !update.state.terminal() {
            let mut pending = self.pending.lock().await;
            if let Some(job) = pending.get_mut(&update.job_id)
                && job.peer == peer
                && update.sequence > job.last_sequence
            {
                let valid_chunk = match &update.update {
                    JobUpdateKind::Chunk { sequence, .. } => *sequence == update.sequence,
                    _ => true,
                };
                if valid_chunk {
                    job.last_sequence = update.sequence;
                }
            }
            return;
        }

        let mut pending = self.pending.lock().await;
        let Some(job) = pending.get_mut(&update.job_id) else {
            return;
        };
        if job.peer != peer || update.sequence <= job.last_sequence {
            return;
        }
        job.last_sequence = update.sequence;
        let Some(job) = pending.remove(&update.job_id) else {
            return;
        };
        drop(pending);

        let (success, result_hash, result) = match update.update {
            JobUpdateKind::Succeeded {
                output,
                output_hash,
                evidence,
            } if ArtifactId::from_bytes_hashed(&output) == output_hash => (
                true,
                Some(output_hash),
                JobResult {
                    job_id: update.job_id,
                    state: JobState::Succeeded,
                    output: Some(output),
                    output_hash: Some(output_hash),
                    evidence,
                    error: None,
                },
            ),
            JobUpdateKind::Succeeded { .. } => (
                false,
                None,
                JobResult {
                    job_id: update.job_id,
                    state: JobState::Failed,
                    output: None,
                    output_hash: None,
                    evidence: None,
                    error: Some("remote result hash does not match output".to_string()),
                },
            ),
            JobUpdateKind::Cancelled => (
                false,
                None,
                JobResult {
                    job_id: update.job_id,
                    state: JobState::Cancelled,
                    output: None,
                    output_hash: None,
                    evidence: None,
                    error: Some("cancelled".to_string()),
                },
            ),
            JobUpdateKind::TimedOut => (
                false,
                None,
                JobResult {
                    job_id: update.job_id,
                    state: JobState::TimedOut,
                    output: None,
                    output_hash: None,
                    evidence: None,
                    error: Some("timed out".to_string()),
                },
            ),
            JobUpdateKind::Rejected { code, message } | JobUpdateKind::Failed { code, message } => {
                (
                    false,
                    None,
                    JobResult {
                        job_id: update.job_id,
                        state: update.state,
                        output: None,
                        output_hash: None,
                        evidence: None,
                        error: Some(format!("{code}: {message}")),
                    },
                )
            }
            _ => (
                false,
                None,
                JobResult {
                    job_id: update.job_id,
                    state: JobState::Failed,
                    output: None,
                    output_hash: None,
                    evidence: None,
                    error: Some("invalid terminal update".to_string()),
                },
            ),
        };
        let capability = job.capability.clone();
        self.graph.lock().await.record(CapabilityObservation {
            node: peer,
            capability: capability.clone(),
            success,
            latency_ms: now_millis()
                .saturating_sub(job.started_at)
                .min(u32::MAX as u64) as u32,
            observed_at: now_secs(),
            result_hash,
        });
        {
            let mut security = self.security.lock().await;
            security.record_direct_observation(peer, success, now_secs());
        }
        self.persist_security_state().await;
        let evidence_kind = if capability == "evaluation.text" && success {
            EvidenceKind::Evaluation
        } else if !success {
            EvidenceKind::Timeout
        } else {
            EvidenceKind::JobCompleted
        };
        let payload = serde_json::to_vec(&serde_json::json!({
            "capability": capability,
            "success": success,
            "result_hash": result_hash,
        }))
        .unwrap_or_default();
        self.record_signed_evidence(peer, evidence_kind, payload)
            .await;
        let _ = job.sender.send(result);
    }

    async fn fail_pending_for_peer(&self, peer: NodeId) {
        let mut pending = self.pending.lock().await;
        let ids = pending
            .iter()
            .filter_map(|(id, job)| (job.peer == peer).then_some(*id))
            .collect::<Vec<_>>();
        let jobs = ids
            .into_iter()
            .filter_map(|id| pending.remove(&id))
            .collect::<Vec<_>>();
        for job in jobs {
            let _ = job.sender.send(JobResult {
                job_id: job.job_id,
                state: JobState::Failed,
                output: None,
                output_hash: None,
                evidence: None,
                error: Some("peer disconnected during job".to_string()),
            });
        }
        let mut artifacts = self.pending_artifacts.lock().await;
        let ids = artifacts
            .iter()
            .filter_map(|(id, transfer)| (transfer.peer == peer).then_some(*id))
            .collect::<Vec<_>>();
        for id in ids {
            if let Some(transfer) = artifacts.remove(&id) {
                let _ = transfer
                    .sender
                    .send(Err(NodeError::Network(NetworkError::NotConnected)));
            }
        }
    }

    async fn handle_artifact_request(&self, peer: NodeId, request: ArtifactRequest) {
        if request.validate().is_err() {
            return;
        }
        let Ok((bytes, _total_size)) = self.store.read_artifact_range(
            request.artifact,
            request.offset,
            request.max_bytes as usize,
        ) else {
            return;
        };
        let chunk = ArtifactChunk {
            artifact: request.artifact,
            offset: request.offset,
            final_chunk: bytes.len() < request.max_bytes as usize,
            data: bytes,
        };
        let _ = self
            .network
            .send_to(peer, Message::ArtifactChunk(chunk))
            .await;
    }

    async fn handle_artifact_transfer_request(
        &self,
        peer: NodeId,
        request: ArtifactTransferRequest,
    ) {
        if request.validate().is_err() {
            return;
        }
        let Ok(total_size) = self.store.artifact_size(request.artifact) else {
            return;
        };
        if request.expected_hash != request.artifact
            || (request.expected_size != 0 && request.expected_size != total_size)
            || request.offset > total_size
        {
            return;
        }
        let Ok((data, _)) = self.store.read_artifact_range(
            request.artifact,
            request.offset,
            request.max_bytes as usize,
        ) else {
            return;
        };
        let chunk = ArtifactTransferChunk {
            request_id: request.request_id,
            artifact: request.artifact,
            offset: request.offset,
            total_size,
            final_chunk: request.offset.saturating_add(data.len() as u64) >= total_size,
            data,
        };
        let _ = self
            .network
            .send_to(peer, Message::ArtifactTransferChunk(chunk))
            .await;
    }

    async fn handle_artifact_transfer_chunk(&self, peer: NodeId, chunk: ArtifactTransferChunk) {
        if chunk.validate().is_err() {
            self.note_security_event(
                Some(peer),
                SecurityEventKind::ArtifactCorrupt,
                "artifact_transfer",
                "malformed or oversized artifact chunk rejected",
            )
            .await;
            return;
        }
        let (artifact, expected_size, expected_offset) = {
            let transfers = self.pending_artifacts.lock().await;
            let Some(transfer) = transfers.get(&chunk.request_id) else {
                return;
            };
            if transfer.peer != peer || transfer.artifact != chunk.artifact {
                return;
            }
            (transfer.artifact, transfer.expected_size, transfer.offset)
        };
        if chunk.offset != expected_offset
            || expected_size.is_some_and(|size| size != chunk.total_size)
        {
            self.note_security_event(
                Some(peer),
                SecurityEventKind::ArtifactCorrupt,
                "artifact_transfer",
                "provider changed artifact range or total size during resume",
            )
            .await;
            if let Some(transfer) = self
                .pending_artifacts
                .lock()
                .await
                .remove(&chunk.request_id)
            {
                let _ = self.store.discard_partial_artifact(artifact);
                let _ = transfer.sender.send(Err(NodeError::InvalidConfig(
                    "artifact transfer range or size changed during resume".to_string(),
                )));
            }
            return;
        }
        let next_offset = match self.store.write_partial_artifact(
            artifact,
            chunk.total_size,
            chunk.offset,
            &chunk.data,
        ) {
            Ok(value) => value,
            Err(error) => {
                self.note_security_event(
                    Some(peer),
                    SecurityEventKind::ArtifactCorrupt,
                    "artifact_transfer",
                    "artifact chunk failed local integrity or quota checks",
                )
                .await;
                if let Some(transfer) = self
                    .pending_artifacts
                    .lock()
                    .await
                    .remove(&chunk.request_id)
                {
                    let _ = transfer.sender.send(Err(error.into()));
                }
                return;
            }
        };
        if chunk.final_chunk {
            let result = self
                .store
                .finalize_partial_artifact(artifact, chunk.total_size, artifact)
                .map(|()| {
                    serde_json::json!({
                        "artifact": artifact.to_string(),
                        "size": chunk.total_size,
                        "verified": true,
                        "source": peer.to_string(),
                    })
                })
                .map_err(NodeError::from);
            if result.is_err() {
                self.note_security_event(
                    Some(peer),
                    SecurityEventKind::ArtifactCorrupt,
                    "artifact_transfer",
                    "final artifact hash or size verification failed",
                )
                .await;
            }
            if let Some(transfer) = self
                .pending_artifacts
                .lock()
                .await
                .remove(&chunk.request_id)
            {
                let _ = transfer.sender.send(result);
            }
            return;
        }
        let next_request = ArtifactTransferRequest {
            request_id: chunk.request_id,
            artifact,
            offset: next_offset,
            max_bytes: MAX_ARTIFACT_CHUNK as u32,
            expected_size: chunk.total_size,
            expected_hash: artifact,
        };
        if let Err(error) = self
            .network
            .send_to(peer, Message::ArtifactTransferRequest(next_request))
            .await
        {
            if let Some(transfer) = self
                .pending_artifacts
                .lock()
                .await
                .remove(&chunk.request_id)
            {
                let _ = transfer.sender.send(Err(error.into()));
            }
        } else if let Some(transfer) = self
            .pending_artifacts
            .lock()
            .await
            .get_mut(&chunk.request_id)
        {
            transfer.offset = next_offset;
            transfer.expected_size = Some(chunk.total_size);
        }
    }

    async fn handle_remote_job(self: Arc<Self>, peer: NodeId, request: JobRequest) {
        let Some(capability) = self.capabilities.get(&request.capability).cloned() else {
            let _ = self
                .send_rejection(
                    peer,
                    request.job_id,
                    "capability_not_found",
                    "this node does not expose that capability",
                )
                .await;
            return;
        };
        if (capability.kind == "process" || capability.kind == "llama_cpp")
            && capability.sandbox != "bubblewrap"
        {
            let _ = self
                .send_rejection(
                    peer,
                    request.job_id,
                    "sandbox_required",
                    "remote process execution requires sandbox = bubblewrap",
                )
                .await;
            return;
        }
        if !capability.public
            || !capability.accept_remote_jobs
            || !request.privacy.allow_input_transfer
        {
            let _ = self
                .send_rejection(
                    peer,
                    request.job_id,
                    "policy_denied",
                    "local policy denied remote execution",
                )
                .await;
            return;
        }
        if let Err(error) = request.validate() {
            let _ = self
                .send_rejection(peer, request.job_id, "invalid_request", &error.to_string())
                .await;
            return;
        }
        if request.input.len() > capability.max_input_bytes as usize
            || request.max_output_bytes > capability.max_output_bytes
        {
            let _ = self
                .send_rejection(
                    peer,
                    request.job_id,
                    "capability_limits",
                    "job exceeds the advertised capability resource limits",
                )
                .await;
            return;
        }
        let executor = match capability.executor() {
            Ok(executor) => executor,
            Err(error) => {
                let _ = self
                    .record_job(PersistedJob {
                        job_id: request.job_id,
                        origin: request.origin,
                        state: JobState::Rejected,
                        updated_at: now_secs(),
                        output_hash: None,
                        output: None,
                        error: Some(error.to_string()),
                    })
                    .await;
                let _ = self
                    .send_rejection(peer, request.job_id, "invalid_executor", &error.to_string())
                    .await;
                return;
            }
        };
        match self.reserve_remote_job(peer, &request).await {
            Ok(RemoteJobReservation::ExistingTerminal) => {
                let _ = self.resend_terminal(peer, request.job_id).await;
                return;
            }
            Ok(RemoteJobReservation::ExistingActive) => {
                tracing::debug!(
                    peer = %peer,
                    job_id = %request.job_id,
                    "ignoring duplicate in-flight job request"
                );
                return;
            }
            Ok(RemoteJobReservation::OriginCollision) => {
                let _ = self
                    .send_rejection(
                        peer,
                        request.job_id,
                        "duplicate_job_id",
                        "job ID is already owned by another peer",
                    )
                    .await;
                return;
            }
            Ok(RemoteJobReservation::Reserved) => {}
            Err(error) => {
                let _ = self
                    .send_rejection(peer, request.job_id, "admission_denied", &error.to_string())
                    .await;
                return;
            }
        }
        let admission = match self.runtime.admit(&request).await {
            Ok(admission) => admission,
            Err(error) => {
                let _ = self
                    .record_job(PersistedJob {
                        job_id: request.job_id,
                        origin: request.origin,
                        state: JobState::Rejected,
                        updated_at: now_secs(),
                        output_hash: None,
                        output: None,
                        error: Some(error.to_string()),
                    })
                    .await;
                let _ = self
                    .send_rejection(peer, request.job_id, "admission_denied", &error.to_string())
                    .await;
                return;
            }
        };
        if let Err(error) = self
            .record_job(PersistedJob {
                job_id: request.job_id,
                origin: request.origin,
                state: JobState::Admitted,
                updated_at: now_secs(),
                output_hash: None,
                output: None,
                error: None,
            })
            .await
        {
            self.runtime.abandon(admission).await;
            let _ = self
                .send_rejection(peer, request.job_id, "storage_error", &error.to_string())
                .await;
            return;
        }
        if self
            .send_update(
                peer,
                JobUpdate {
                    job_id: request.job_id,
                    state: JobState::Admitted,
                    sequence: 1,
                    update: JobUpdateKind::Accepted {
                        state: JobState::Admitted,
                    },
                },
            )
            .await
            .is_err()
        {
            let _ = self.runtime.cancel(request.job_id).await;
            let _ = self
                .record_job(PersistedJob {
                    job_id: request.job_id,
                    origin: request.origin,
                    state: JobState::Failed,
                    updated_at: now_secs(),
                    output_hash: None,
                    output: None,
                    error: Some("peer disconnected before job admission".to_string()),
                })
                .await;
            return;
        }
        let node = self.clone();
        tokio::spawn(async move {
            if node
                .record_job(PersistedJob {
                    job_id: request.job_id,
                    origin: request.origin,
                    state: JobState::Running,
                    updated_at: now_secs(),
                    output_hash: None,
                    output: None,
                    error: None,
                })
                .await
                .is_err()
            {
                let _ = node.runtime.cancel(request.job_id).await;
                return;
            }
            if node
                .send_update(
                    peer,
                    JobUpdate {
                        job_id: request.job_id,
                        state: JobState::Running,
                        sequence: 2,
                        update: JobUpdateKind::Started,
                    },
                )
                .await
                .is_err()
            {
                let _ = node.runtime.cancel(request.job_id).await;
                let _ = node
                    .record_job(PersistedJob {
                        job_id: request.job_id,
                        origin: request.origin,
                        state: JobState::Failed,
                        updated_at: now_secs(),
                        output_hash: None,
                        output: None,
                        error: Some("peer disconnected before job execution".to_string()),
                    })
                    .await;
                return;
            }
            let (chunk_sender, mut chunk_receiver) = mpsc::channel(8);
            let mut execution = Box::pin(node.runtime.execute_streaming(
                admission,
                &request,
                &executor,
                chunk_sender,
            ));
            let mut sequence = 2_u32;
            let outcome = loop {
                tokio::select! {
                    chunk = chunk_receiver.recv() => {
                        let Some(data) = chunk else {
                            break execution.await;
                        };
                        sequence = sequence.saturating_add(1);
                        let _ = node.send_update(
                            peer,
                            JobUpdate {
                                job_id: request.job_id,
                                state: JobState::Running,
                                sequence,
                                update: JobUpdateKind::Chunk { sequence, data },
                            },
                        ).await;
                    }
                    result = &mut execution => {
                        while let Some(data) = chunk_receiver.recv().await {
                            sequence = sequence.saturating_add(1);
                            let _ = node.send_update(
                                peer,
                                JobUpdate {
                                    job_id: request.job_id,
                                    state: JobState::Running,
                                    sequence,
                                    update: JobUpdateKind::Chunk { sequence, data },
                                },
                            ).await;
                        }
                        break result;
                    }
                }
            };
            match outcome {
                Ok(outcome) => {
                    if let Err(error) = node
                        .send_outcome(peer, request.clone(), outcome, sequence)
                        .await
                    {
                        tracing::warn!(job_id = %request.job_id, error = %error, "failed to persist or send job outcome");
                        let _ = node
                            .record_job(PersistedJob {
                                job_id: request.job_id,
                                origin: request.origin,
                                state: JobState::Failed,
                                updated_at: now_secs(),
                                output_hash: None,
                                output: None,
                                error: Some(error.to_string()),
                            })
                            .await;
                        let _ = node
                            .send_update(
                                peer,
                                JobUpdate {
                                    job_id: request.job_id,
                                    state: JobState::Failed,
                                    sequence: sequence.saturating_add(1),
                                    update: JobUpdateKind::Failed {
                                        code: "outcome_error".to_string(),
                                        message: error.to_string(),
                                    },
                                },
                            )
                            .await;
                    }
                }
                Err(error) => {
                    node.metrics
                        .execution_failures
                        .fetch_add(1, Ordering::Relaxed);
                    let _ = node
                        .record_job(PersistedJob {
                            job_id: request.job_id,
                            origin: request.origin,
                            state: JobState::Failed,
                            updated_at: now_secs(),
                            output_hash: None,
                            output: None,
                            error: Some(error.to_string()),
                        })
                        .await;
                    let _ = node
                        .send_update(
                            peer,
                            JobUpdate {
                                job_id: request.job_id,
                                state: JobState::Failed,
                                sequence: sequence.saturating_add(1),
                                update: JobUpdateKind::Failed {
                                    code: "execution_error".to_string(),
                                    message: error.to_string(),
                                },
                            },
                        )
                        .await;
                }
            }
        });
    }

    async fn send_outcome(
        &self,
        peer: NodeId,
        request: JobRequest,
        outcome: ExecutionOutcome,
        starting_sequence: u32,
    ) -> Result<(), NodeError> {
        let (state, update) = match outcome {
            ExecutionOutcome::Succeeded(result) => {
                self.record_job(PersistedJob {
                    job_id: request.job_id,
                    origin: request.origin,
                    state: JobState::Succeeded,
                    updated_at: now_secs(),
                    output_hash: Some(result.output_hash),
                    output: request
                        .privacy
                        .allow_output_persistence
                        .then_some(result.output.clone()),
                    error: None,
                })
                .await?;
                (
                    JobState::Succeeded,
                    JobUpdateKind::Succeeded {
                        output: result.output,
                        output_hash: result.output_hash,
                        evidence: None,
                    },
                )
            }
            ExecutionOutcome::Cancelled => {
                self.record_job(PersistedJob {
                    job_id: request.job_id,
                    origin: request.origin,
                    state: JobState::Cancelled,
                    updated_at: now_secs(),
                    output_hash: None,
                    output: None,
                    error: Some("cancelled".to_string()),
                })
                .await?;
                (JobState::Cancelled, JobUpdateKind::Cancelled)
            }
            ExecutionOutcome::TimedOut => {
                self.record_job(PersistedJob {
                    job_id: request.job_id,
                    origin: request.origin,
                    state: JobState::TimedOut,
                    updated_at: now_secs(),
                    output_hash: None,
                    output: None,
                    error: Some("timed out".to_string()),
                })
                .await?;
                (JobState::TimedOut, JobUpdateKind::TimedOut)
            }
        };
        self.send_update(
            peer,
            JobUpdate {
                job_id: request.job_id,
                state,
                sequence: starting_sequence.saturating_add(1),
                update,
            },
        )
        .await
    }

    async fn send_rejection(
        &self,
        peer: NodeId,
        job_id: JobId,
        code: &str,
        message: &str,
    ) -> Result<(), NodeError> {
        self.metrics
            .rejected_requests
            .fetch_add(1, Ordering::Relaxed);
        self.send_update(
            peer,
            JobUpdate {
                job_id,
                state: JobState::Rejected,
                sequence: 1,
                update: JobUpdateKind::Rejected {
                    code: code.to_string(),
                    message: message.to_string(),
                },
            },
        )
        .await
    }

    async fn send_update(&self, peer: NodeId, update: JobUpdate) -> Result<(), NodeError> {
        self.network
            .send_to(peer, Message::JobUpdate(update))
            .await?;
        Ok(())
    }

    async fn resend_terminal(&self, peer: NodeId, job_id: JobId) -> Result<(), NodeError> {
        let job = self
            .jobs
            .lock()
            .await
            .get(&job_id)
            .filter(|job| job.origin == peer)
            .cloned();
        let Some(job) = job else { return Ok(()) };
        let (state, update) = match job.state {
            JobState::Succeeded => match (job.output, job.output_hash) {
                (Some(output), Some(output_hash))
                    if ArtifactId::from_bytes_hashed(&output) == output_hash =>
                {
                    (
                        JobState::Succeeded,
                        JobUpdateKind::Succeeded {
                            output,
                            output_hash,
                            evidence: None,
                        },
                    )
                }
                _ => (
                    JobState::Failed,
                    JobUpdateKind::Failed {
                        code: "result_not_retained".to_string(),
                        message: "the completed result was not retained by local policy"
                            .to_string(),
                    },
                ),
            },
            JobState::Cancelled => (JobState::Cancelled, JobUpdateKind::Cancelled),
            JobState::TimedOut => (JobState::TimedOut, JobUpdateKind::TimedOut),
            JobState::Rejected => (
                JobState::Rejected,
                JobUpdateKind::Rejected {
                    code: "rejected".to_string(),
                    message: job.error.unwrap_or_default(),
                },
            ),
            _ => (
                JobState::Failed,
                JobUpdateKind::Failed {
                    code: "not_terminal".to_string(),
                    message: "job state is not terminal".to_string(),
                },
            ),
        };
        self.send_update(
            peer,
            JobUpdate {
                job_id,
                state,
                sequence: 4,
                update,
            },
        )
        .await
    }

    async fn outcome_to_result(
        &self,
        request: JobRequest,
        outcome: ExecutionOutcome,
        evidence: Option<intelligence_protocol::Evidence>,
    ) -> Result<JobResult, NodeError> {
        let result = match outcome {
            ExecutionOutcome::Succeeded(value) => JobResult {
                job_id: request.job_id,
                state: JobState::Succeeded,
                output: Some(value.output),
                output_hash: Some(value.output_hash),
                evidence,
                error: None,
            },
            ExecutionOutcome::Cancelled => JobResult {
                job_id: request.job_id,
                state: JobState::Cancelled,
                output: None,
                output_hash: None,
                evidence,
                error: Some("cancelled".to_string()),
            },
            ExecutionOutcome::TimedOut => JobResult {
                job_id: request.job_id,
                state: JobState::TimedOut,
                output: None,
                output_hash: None,
                evidence,
                error: Some("timed out".to_string()),
            },
        };
        self.record_job(PersistedJob {
            job_id: result.job_id,
            origin: request.origin,
            state: result.state,
            updated_at: now_secs(),
            output_hash: result.output_hash,
            output: request
                .privacy
                .allow_output_persistence
                .then(|| result.output.clone())
                .flatten(),
            error: result.error.clone(),
        })
        .await?;
        Ok(result)
    }

    async fn cancel_job(&self, job_id: &str) -> Result<bool, NodeError> {
        let bytes =
            hex::decode(job_id).map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
        if bytes.len() != 16 {
            return Err(NodeError::InvalidConfig(
                "job ID must be 16 bytes of hex".to_string(),
            ));
        }
        let mut id = [0u8; 16];
        id.copy_from_slice(&bytes);
        if self.runtime.cancel(JobId::from_bytes(id)).await {
            return Ok(true);
        }
        if let Some(pending) = self.pending.lock().await.remove(&JobId::from_bytes(id)) {
            let peer = pending.peer;
            let _ = pending.sender.send(JobResult {
                job_id: JobId::from_bytes(id),
                state: JobState::Cancelled,
                output: None,
                output_hash: None,
                evidence: None,
                error: Some("cancelled".to_string()),
            });
            let _ = self
                .network
                .send_to(
                    peer,
                    Message::CancelJob(intelligence_protocol::CancelJob {
                        job_id: JobId::from_bytes(id),
                        reason: "operator requested cancellation".to_string(),
                    }),
                )
                .await;
            return Ok(true);
        }
        Ok(false)
    }

    async fn record_job(&self, job: PersistedJob) -> Result<(), NodeError> {
        let mut jobs = self.jobs.lock().await;
        if !jobs.contains_key(&job.job_id) && jobs.len() >= MAX_JOB_HISTORY {
            let remove = jobs
                .values()
                .filter(|item| item.state.terminal())
                .min_by_key(|item| item.updated_at)
                .map(|item| item.job_id)
                .ok_or_else(|| {
                    NodeError::Storage(StorageError::InvalidState(
                        "job history is full with active jobs".to_string(),
                    ))
                })?;
            jobs.remove(&remove);
        }
        jobs.insert(job.job_id, job);
        let values = jobs.values().cloned().collect::<Vec<_>>();
        drop(jobs);
        self.store.save_jobs(&values)?;
        Ok(())
    }

    async fn reserve_remote_job(
        &self,
        peer: NodeId,
        request: &JobRequest,
    ) -> Result<RemoteJobReservation, NodeError> {
        let mut jobs = self.jobs.lock().await;
        if let Some(existing) = jobs.get(&request.job_id) {
            if existing.origin != peer {
                return Ok(RemoteJobReservation::OriginCollision);
            }
            return Ok(if existing.state.terminal() {
                RemoteJobReservation::ExistingTerminal
            } else {
                RemoteJobReservation::ExistingActive
            });
        }
        let evicted = if jobs.len() >= MAX_JOB_HISTORY {
            let remove = jobs
                .values()
                .filter(|item| item.state.terminal())
                .min_by_key(|item| item.updated_at)
                .map(|item| item.job_id)
                .ok_or_else(|| {
                    NodeError::Storage(StorageError::InvalidState(
                        "job history is full with active jobs".to_string(),
                    ))
                })?;
            jobs.remove(&remove)
        } else {
            None
        };
        jobs.insert(
            request.job_id,
            PersistedJob {
                job_id: request.job_id,
                origin: request.origin,
                state: JobState::Validated,
                updated_at: now_secs(),
                output_hash: None,
                output: None,
                error: None,
            },
        );
        let values = jobs.values().cloned().collect::<Vec<_>>();
        if let Err(error) = self.store.save_jobs(&values) {
            jobs.remove(&request.job_id);
            if let Some(evicted) = evicted {
                jobs.insert(evicted.job_id, evicted);
            }
            return Err(error.into());
        }
        Ok(RemoteJobReservation::Reserved)
    }
}

impl AdminResponse {
    fn ok(data: serde_json::Value) -> Self {
        Self {
            ok: true,
            data,
            error: None,
        }
    }

    fn error(error: String) -> Self {
        Self {
            ok: false,
            data: serde_json::json!({}),
            error: Some(error),
        }
    }
}

fn response_result<T: Serialize>(result: Result<T, NodeError>) -> AdminResponse {
    match result {
        Ok(data) => {
            AdminResponse::ok(serde_json::to_value(data).unwrap_or_else(|_| serde_json::json!({})))
        }
        Err(error) => AdminResponse::error(error.to_string()),
    }
}

fn random_job_id() -> JobId {
    let mut bytes = [0u8; 16];
    OsRng.fill_bytes(&mut bytes);
    JobId::from_bytes(bytes)
}

fn random_request_id() -> RequestId {
    let mut bytes = [0u8; 16];
    OsRng.fill_bytes(&mut bytes);
    RequestId::from_bytes(bytes)
}

fn parse_node_id(value: &str) -> Result<NodeId, NodeError> {
    let bytes = hex::decode(value)
        .map_err(|error| NodeError::InvalidConfig(format!("invalid node ID: {error}")))?;
    if bytes.len() != 32 {
        return Err(NodeError::InvalidConfig(
            "node ID must contain 32 bytes of hex".to_string(),
        ));
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&bytes);
    Ok(NodeId::from_bytes(id))
}

fn parse_node_ids(values: &[String]) -> Result<Vec<NodeId>, String> {
    if values.is_empty() {
        return Err("at least one peer node ID is required".to_string());
    }
    let mut parsed = Vec::with_capacity(values.len());
    let mut seen = std::collections::HashSet::with_capacity(values.len());
    for value in values {
        let node = parse_node_id(value).map_err(|error| error.to_string())?;
        if !seen.insert(node) {
            return Err("peer node IDs must be unique".to_string());
        }
        parsed.push(node);
    }
    Ok(parsed)
}

fn parse_artifact_id(value: &str) -> Result<ArtifactId, NodeError> {
    let bytes = hex::decode(value)
        .map_err(|error| NodeError::InvalidConfig(format!("invalid artifact ID: {error}")))?;
    if bytes.len() != 32 {
        return Err(NodeError::InvalidConfig(
            "artifact ID must contain 32 bytes of hex".to_string(),
        ));
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&bytes);
    Ok(ArtifactId::from_bytes(id))
}

fn reference_samples() -> Vec<ReferenceSample> {
    vec![
        ReferenceSample { x: -2.0, y: -3.0 },
        ReferenceSample { x: -1.0, y: -1.0 },
        ReferenceSample { x: 1.0, y: 3.0 },
        ReferenceSample { x: 2.0, y: 5.0 },
    ]
}

fn partition_samples(samples: &[ReferenceSample], workers: usize) -> Vec<Vec<ReferenceSample>> {
    let mut partitions = vec![Vec::new(); workers];
    for (index, sample) in samples.iter().copied().enumerate() {
        partitions[index % workers].push(sample);
    }
    partitions
}

fn reference_loss(model: ReferenceModel, samples: &[ReferenceSample]) -> f64 {
    samples
        .iter()
        .map(|sample| {
            let error = model.weight * sample.x + model.bias - sample.y;
            error * error
        })
        .sum::<f64>()
        / samples.len().max(1) as f64
}

fn parse_job_id(value: &str) -> Result<JobId, NodeError> {
    let bytes = hex::decode(value)
        .map_err(|error| NodeError::InvalidConfig(format!("invalid job ID: {error}")))?;
    if bytes.len() != 16 {
        return Err(NodeError::InvalidConfig(
            "job ID must contain 16 bytes of hex".to_string(),
        ));
    }
    let mut id = [0u8; 16];
    id.copy_from_slice(&bytes);
    Ok(JobId::from_bytes(id))
}

fn parse_dht_namespace(value: &str) -> Result<DhtNamespace, String> {
    match value {
        "peer" => Ok(DhtNamespace::Peer),
        "capability" => Ok(DhtNamespace::Capability),
        "artifact" => Ok(DhtNamespace::Artifact),
        "model" => Ok(DhtNamespace::Model),
        "dataset" => Ok(DhtNamespace::Dataset),
        "relay" => Ok(DhtNamespace::Relay),
        "evidence" => Ok(DhtNamespace::Evidence),
        _ => Err(format!(
            "unknown DHT namespace {value}; expected peer, capability, artifact, model, dataset, relay, or evidence"
        )),
    }
}

fn parse_dht_key(value: &str) -> Result<DhtKey, String> {
    let bytes = hex::decode(value).map_err(|error| format!("invalid DHT key: {error}"))?;
    if bytes.len() != 32 {
        return Err("DHT key must contain 32 bytes of hex".to_string());
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes);
    Ok(DhtKey::from_bytes(key))
}

fn parse_data_locality(value: &str) -> Result<DataLocality, NodeError> {
    match value.to_ascii_lowercase().as_str() {
        "local_only" | "local-only" => Ok(DataLocality::LocalOnly),
        "selective" => Ok(DataLocality::Selective),
        "streamable" => Ok(DataLocality::Streamable),
        _ => Err(NodeError::InvalidConfig(
            "data locality must be local_only, selective, or streamable".to_string(),
        )),
    }
}

fn parse_hardware_kind(value: &str) -> HardwareKind {
    match value.to_ascii_lowercase().as_str() {
        "cpu" => HardwareKind::Cpu,
        "apple_silicon" | "apple-silicon" => HardwareKind::AppleSilicon,
        "amd_gpu" | "amd-gpu" => HardwareKind::AmdGpu,
        "nvidia_consumer" | "nvidia-consumer" => HardwareKind::NvidiaConsumer,
        "a100" => HardwareKind::A100,
        "h100" => HardwareKind::H100,
        "b200" => HardwareKind::B200,
        _ => HardwareKind::Unknown,
    }
}

fn process_memory_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
        let kilobytes = line.split_whitespace().nth(1)?.parse::<u64>().ok()?;
        Some(kilobytes.saturating_mul(1024))
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn process_cpu_time_ms() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = fs::read_to_string("/proc/self/stat").ok()?;
        let fields = stat
            .get(stat.rfind(')')?.saturating_add(2)..)?
            .split_whitespace();
        let mut fields = fields.skip(11);
        let user_ticks = fields.next()?.parse::<u64>().ok()?;
        let system_ticks = fields.next()?.parse::<u64>().ok()?;
        let ticks_per_second = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if ticks_per_second <= 0 {
            return None;
        }
        Some(user_ticks.saturating_add(system_ticks).saturating_mul(1000) / ticks_per_second as u64)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

/// Local-only source diversity bucket for evaluator selection. It is a
/// heuristic over an already advertised address string, never published as
/// identity evidence and never treated as proof of independent operators.
fn local_source_group(address: &str) -> u16 {
    address.bytes().fold(0x4D3A_u16, |value, byte| {
        value.rotate_left(5) ^ u16::from(byte)
    })
}

fn default_data_dir() -> PathBuf {
    PathBuf::from("state")
}

fn default_listen_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 4000))
}

fn default_max_connections() -> usize {
    64
}

fn default_max_frame_size() -> usize {
    intelligence_protocol::MAX_FRAME_SIZE
}

fn default_peer_ttl() -> u64 {
    300
}

fn default_relay_sessions() -> usize {
    64
}

fn default_relay_bytes() -> u64 {
    64 * 1024 * 1024
}

fn default_hole_punch_enabled() -> bool {
    true
}

fn default_hole_punch_attempts() -> usize {
    4
}

fn default_dht_enabled() -> bool {
    true
}

fn default_dht_k() -> usize {
    20
}

fn default_dht_alpha() -> usize {
    3
}

fn default_dht_max_records() -> usize {
    2_048
}

fn default_training_memory_bytes() -> u64 {
    DEFAULT_TRAINING_MEMORY_BYTES
}

fn relay_capability(expires_at: u64) -> Capability {
    Capability {
        name: "network.relay".to_string(),
        version: 1,
        model: None,
        resources: ResourceLimits {
            max_input_bytes: intelligence_protocol::MAX_JOB_INPUT as u32,
            max_output_bytes: intelligence_protocol::MAX_JOB_OUTPUT as u32,
            memory_bytes: 64 * 1024 * 1024,
            cpu_millis: 1000,
        },
        evidence: CapabilityEvidence::Claimed,
        expires_at,
        metadata: vec![intelligence_protocol::MetadataEntry {
            key: "role".to_string(),
            value: "opaque-forwarder".to_string(),
        }],
        compute_backends: Vec::new(),
    }
}

fn default_version() -> u16 {
    1
}

fn default_true() -> bool {
    true
}

fn default_executor_kind() -> String {
    "builtin_text".to_string()
}

fn default_sandbox() -> String {
    "trusted_local".to_string()
}

fn default_input_limit() -> u32 {
    64 * 1024
}

fn default_output_limit() -> u32 {
    16 * 1024
}

fn default_memory_limit() -> u64 {
    64 * 1024 * 1024
}

fn default_cpu_limit() -> u64 {
    1000
}
