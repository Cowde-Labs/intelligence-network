//! Encrypted peer transport, identity binding, discovery, and local routing.

mod dht;

use dht::{
    DEFAULT_K, DEFAULT_MAX_RECORDS, DEFAULT_MAX_RECORDS_PER_KEY, DEFAULT_REPLACEMENT_CACHE,
    DhtRecordStats, DhtStore, DhtTable, DhtTableStats,
};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use intelligence_protocol::{
    AddressObservation, AddressRecord, AddressSource, AddressTransport, AddressUpdate, DhtKey,
    DhtNamespace, DhtRecord, DhtRequest, DhtRequestKind, DhtResponse, DhtResponseKind,
    FRAME_HEADER_SIZE, FrameHeader, Hello, KeyRotation, MAX_DHT_CONTACTS, MAX_DHT_RECORDS,
    MAX_FRAME_SIZE, MAX_PEERS, MAX_RELAY_PAYLOAD, Message, NodeId, PROTOCOL_MAJOR, PROTOCOL_MINOR,
    PeerExchange, PeerRecord, ProtocolError, ReachabilityState, RelayEnvelope, RequestId,
    SignedAnnouncement, VersionRange, decode_frame, encode_message_at_version,
};
use intelligence_storage::LocalStore;
use quinn::{
    ClientConfig, Connection, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig,
    VarInt,
    crypto::rustls::{QuicClientConfig, QuicServerConfig},
};
use rand::{RngCore, rngs::OsRng};
use rcgen::generate_simple_self_signed;
use rustls::{
    ClientConfig as RustlsClientConfig, DigitallySignedStruct, Error as RustlsError,
    ServerConfig as RustlsServerConfig, SignatureScheme,
    client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier},
    pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fs::{self, OpenOptions},
    io::{self, Write},
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use tokio::{
    sync::{Mutex, Notify, mpsc, oneshot},
    time::{Instant, sleep, timeout},
};

const CONNECTION_QUEUE: usize = 128;
const PEER_TTL_SECONDS: u64 = 300;
const HELLO_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_LEARNED_PEERS: usize = 2048;
const MAX_SEEN_DHT_REQUESTS: usize = 4096;
const DHT_LOOKUP_DEADLINE: Duration = Duration::from_secs(5);
const DHT_RPC_TIMEOUT: Duration = Duration::from_secs(3);
const DHT_REQUESTS_PER_WINDOW: u32 = 128;
const DHT_REQUEST_WINDOW_SECONDS: u64 = 60;

type SeenHello = (NodeId, [u8; 16]);
type SeenHellos = Arc<Mutex<HashSet<SeenHello>>>;

#[derive(Debug, Error)]
pub enum IdentityError {
    #[error("identity I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("identity file must contain exactly 32 private-key bytes")]
    InvalidLength,
    #[error("identity rotation record is invalid: {0}")]
    InvalidRotation(String),
    #[error("identity rotation encoding failed: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone)]
pub struct Identity {
    signing_key: SigningKey,
    public_key: [u8; 32],
    node_id: NodeId,
    rotation: Option<KeyRotation>,
}

impl Identity {
    pub fn load_or_generate(path: impl AsRef<Path>) -> Result<Self, IdentityError> {
        let path = path.as_ref();
        let secret = match fs::read(path) {
            Ok(bytes) => {
                if bytes.len() != 32 {
                    return Err(IdentityError::InvalidLength);
                }
                let mut value = [0u8; 32];
                value.copy_from_slice(&bytes);
                value
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut value = [0u8; 32];
                OsRng.fill_bytes(&mut value);
                let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
                file.write_all(&value)?;
                file.sync_all()?;
                set_private_permissions(path)?;
                value
            }
            Err(error) => return Err(IdentityError::Io(error)),
        };
        set_private_permissions(path)?;
        let signing_key = SigningKey::from_bytes(&secret);
        let public_key = signing_key.verifying_key().to_bytes();
        let node_id = NodeId::from_public_key(&public_key);
        let rotation = load_rotation(path, node_id, &public_key)?;
        Ok(Self {
            signing_key,
            public_key,
            node_id,
            rotation,
        })
    }

    pub fn rotate(
        old_path: impl AsRef<Path>,
        new_path: impl AsRef<Path>,
        sequence: u64,
        valid_until: u64,
    ) -> Result<KeyRotation, IdentityError> {
        let old_path = old_path.as_ref();
        let new_path = new_path.as_ref();
        if sequence == 0 || valid_until <= now_secs() {
            return Err(IdentityError::InvalidRotation(
                "rotation sequence or validity is invalid".to_string(),
            ));
        }
        let old = Self::load_or_generate(old_path)?;
        if old
            .rotation()
            .is_some_and(|previous| sequence <= previous.sequence)
        {
            return Err(IdentityError::InvalidRotation(
                "rotation sequence must increase beyond the previous rotation".to_string(),
            ));
        }
        if new_path.exists() {
            return Err(IdentityError::InvalidRotation(
                "new identity path already exists".to_string(),
            ));
        }
        if let Some(parent) = new_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut secret = [0u8; 32];
        OsRng.fill_bytes(&mut secret);
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(new_path)?;
        file.write_all(&secret)?;
        file.sync_all()?;
        set_private_permissions(new_path)?;
        let new_key = SigningKey::from_bytes(&secret);
        let new_public_key = new_key.verifying_key().to_bytes();
        let new_node_id = NodeId::from_public_key(&new_public_key);
        let valid_from = now_secs();
        let bytes = rotation_signing_bytes(
            old.node_id,
            old.public_key,
            new_node_id,
            new_public_key,
            sequence,
            valid_from,
            valid_until,
        )?;
        let rotation = KeyRotation {
            old_node_id: old.node_id,
            old_public_key: old.public_key,
            new_node_id,
            new_public_key,
            sequence,
            valid_from,
            valid_until,
            old_signature: old.sign(&bytes),
            new_signature: new_key.sign(&bytes).to_bytes().to_vec(),
        };
        rotation
            .validate()
            .map_err(|error| IdentityError::InvalidRotation(error.to_string()))?;
        let rotation_path = rotation_path(new_path);
        let encoded = serde_json::to_vec_pretty(&rotation)?;
        let mut rotation_file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&rotation_path)?;
        rotation_file.write_all(&encoded)?;
        rotation_file.sync_all()?;
        set_private_permissions(&rotation_path)?;
        Ok(rotation)
    }

    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    pub fn public_key(&self) -> [u8; 32] {
        self.public_key
    }

    pub fn rotation(&self) -> Option<&KeyRotation> {
        self.rotation.as_ref()
    }

    pub fn sign(&self, bytes: &[u8]) -> Vec<u8> {
        self.signing_key.sign(bytes).to_bytes().to_vec()
    }

    pub fn verify(public_key: &[u8; 32], bytes: &[u8], signature: &[u8]) -> bool {
        let Ok(key) = VerifyingKey::from_bytes(public_key) else {
            return false;
        };
        let Ok(signature) = Signature::from_slice(signature) else {
            return false;
        };
        key.verify(bytes, &signature).is_ok()
    }
}

fn set_private_permissions(path: &Path) -> Result<(), IdentityError> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    }
    Ok(())
}

fn rotation_path(identity_path: &Path) -> PathBuf {
    identity_path.with_extension("rotation.json")
}

fn rotation_signing_bytes(
    old_node_id: NodeId,
    old_public_key: [u8; 32],
    new_node_id: NodeId,
    new_public_key: [u8; 32],
    sequence: u64,
    valid_from: u64,
    valid_until: u64,
) -> Result<Vec<u8>, IdentityError> {
    postcard::to_allocvec(&(
        old_node_id,
        old_public_key,
        new_node_id,
        new_public_key,
        sequence,
        valid_from,
        valid_until,
    ))
    .map_err(|error| IdentityError::InvalidRotation(error.to_string()))
}

fn load_rotation(
    identity_path: &Path,
    node_id: NodeId,
    public_key: &[u8; 32],
) -> Result<Option<KeyRotation>, IdentityError> {
    let path = rotation_path(identity_path);
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(IdentityError::Io(error)),
    };
    let rotation: KeyRotation = serde_json::from_slice(&bytes)?;
    rotation
        .validate()
        .map_err(|error| IdentityError::InvalidRotation(error.to_string()))?;
    if rotation.new_node_id != node_id || rotation.new_public_key != *public_key {
        return Err(IdentityError::InvalidRotation(
            "rotation does not describe the loaded identity".to_string(),
        ));
    }
    let signing_bytes = rotation_signing_bytes(
        rotation.old_node_id,
        rotation.old_public_key,
        rotation.new_node_id,
        rotation.new_public_key,
        rotation.sequence,
        rotation.valid_from,
        rotation.valid_until,
    )?;
    if !Identity::verify(
        &rotation.old_public_key,
        &signing_bytes,
        &rotation.old_signature,
    ) || !Identity::verify(
        &rotation.new_public_key,
        &signing_bytes,
        &rotation.new_signature,
    ) {
        return Err(IdentityError::InvalidRotation(
            "rotation signatures do not verify".to_string(),
        ));
    }
    Ok(Some(rotation))
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NetworkConfig {
    pub listen_addr: SocketAddr,
    pub advertise_addr: String,
    pub bootstrap: Vec<String>,
    pub max_frame_size: usize,
    pub max_connections: usize,
    pub peer_ttl_seconds: u64,
    #[serde(default)]
    pub allow_private_addresses: bool,
    #[serde(default)]
    pub prefer_relay: bool,
    #[serde(default = "default_hole_punch_enabled")]
    pub hole_punch_enabled: bool,
    #[serde(default = "default_hole_punch_attempts")]
    pub hole_punch_max_attempts: usize,
    #[serde(default)]
    pub relay_addresses: Vec<String>,
    #[serde(default)]
    pub relay_enabled: bool,
    #[serde(default = "default_relay_sessions")]
    pub relay_max_sessions: usize,
    #[serde(default = "default_relay_bytes")]
    pub relay_max_bytes: u64,
    pub capabilities: Vec<intelligence_protocol::Capability>,
    #[serde(default = "default_dht_enabled")]
    pub dht_enabled: bool,
    #[serde(default = "default_dht_k")]
    pub dht_k: usize,
    #[serde(default = "default_dht_alpha")]
    pub dht_alpha: usize,
    #[serde(default = "default_dht_max_records")]
    pub dht_max_records: usize,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: SocketAddr::from(([127, 0, 0, 1], 0)),
            advertise_addr: "127.0.0.1:0".to_string(),
            bootstrap: Vec::new(),
            max_frame_size: MAX_FRAME_SIZE,
            max_connections: 64,
            peer_ttl_seconds: PEER_TTL_SECONDS,
            allow_private_addresses: false,
            prefer_relay: false,
            hole_punch_enabled: true,
            hole_punch_max_attempts: 4,
            relay_addresses: Vec::new(),
            relay_enabled: false,
            relay_max_sessions: 64,
            relay_max_bytes: 64 * 1024 * 1024,
            capabilities: Vec::new(),
            dht_enabled: true,
            dht_k: DEFAULT_K,
            dht_alpha: 3,
            dht_max_records: DEFAULT_MAX_RECORDS,
        }
    }
}

impl NetworkConfig {
    pub fn validate(&self) -> Result<(), NetworkError> {
        if self.max_frame_size == 0 || self.max_frame_size > MAX_FRAME_SIZE {
            return Err(NetworkError::InvalidConfig(
                "max_frame_size must be positive and within the protocol maximum".to_string(),
            ));
        }
        if self.max_connections == 0
            || self.peer_ttl_seconds == 0
            || self.peer_ttl_seconds > 24 * 60 * 60
        {
            return Err(NetworkError::InvalidConfig(
                "network limits are invalid".to_string(),
            ));
        }
        if self.advertise_addr.is_empty() || self.advertise_addr.len() > 256 {
            return Err(NetworkError::InvalidConfig(
                "advertise_addr is invalid".to_string(),
            ));
        }
        self.advertise_addr.parse::<SocketAddr>().map_err(|error| {
            NetworkError::InvalidConfig(format!("advertise_addr must be SocketAddr: {error}"))
        })?;
        if self.capabilities.len() > intelligence_protocol::MAX_CAPABILITIES {
            return Err(NetworkError::InvalidConfig(
                "too many capabilities".to_string(),
            ));
        }
        if self.bootstrap.len() > MAX_PEERS {
            return Err(NetworkError::InvalidConfig(
                "too many bootstrap peers".to_string(),
            ));
        }
        if self.relay_addresses.len() > MAX_PEERS
            || self.relay_max_sessions == 0
            || self.relay_max_bytes == 0
            || self.relay_max_bytes > MAX_RELAY_PAYLOAD as u64 * 4096
        {
            return Err(NetworkError::InvalidConfig(
                "relay limits or relay address count are invalid".to_string(),
            ));
        }
        if self.hole_punch_max_attempts == 0 || self.hole_punch_max_attempts > 16 {
            return Err(NetworkError::InvalidConfig(
                "hole-punch attempt limit is invalid".to_string(),
            ));
        }
        if self.dht_k == 0
            || self.dht_k > MAX_DHT_CONTACTS
            || self.dht_alpha == 0
            || self.dht_alpha > self.dht_k
            || self.dht_max_records == 0
            || self.dht_max_records > 16 * DEFAULT_MAX_RECORDS
        {
            return Err(NetworkError::InvalidConfig(
                "DHT limits are invalid".to_string(),
            ));
        }
        for address in self.bootstrap.iter().chain(self.relay_addresses.iter()) {
            address.parse::<SocketAddr>().map_err(|error| {
                NetworkError::InvalidConfig(format!("peer address is invalid: {error}"))
            })?;
        }
        for capability in &self.capabilities {
            capability
                .validate()
                .map_err(|error| NetworkError::InvalidConfig(error.to_string()))?;
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum NetworkError {
    #[error("network configuration is invalid: {0}")]
    InvalidConfig(String),
    #[error("network I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("QUIC transport failed: {0}")]
    Quic(String),
    #[error("wire codec failed: {0}")]
    Codec(#[from] intelligence_protocol::CodecError),
    #[error("peer is not connected")]
    NotConnected,
    #[error("peer table is full")]
    PeerTableFull,
    #[error("peer identity verification failed")]
    IdentityVerification,
    #[error("peer protocol error: {0}")]
    PeerProtocol(String),
    #[error("storage failed: {0}")]
    Storage(#[from] intelligence_storage::StorageError),
}

// NetworkEvent is an internal bounded mailbox item.  Keeping the decoded
// message inline avoids an allocation on every transport event; the protocol
// decoder already enforces the maximum frame size before this enum is built.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum NetworkEvent {
    PeerConnected(PeerRecord),
    PeerDisconnected(NodeId),
    Message {
        peer: NodeId,
        message: Message,
    },
    ProtocolError {
        peer: Option<NodeId>,
        message: String,
    },
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct NetworkMetricsSnapshot {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub protocol_errors: u64,
    pub rejected_connections: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct DhtNetworkStats {
    pub enabled: bool,
    pub routing: DhtTableStats,
    pub records: DhtRecordStats,
    pub rpc_sent: u64,
    pub rpc_received: u64,
    pub lookups: u64,
    pub lookup_successes: u64,
    pub invalid_records: u64,
    pub rate_limited_requests: u64,
}

#[derive(Default)]
struct NetworkMetrics {
    bytes_sent: AtomicU64,
    bytes_received: AtomicU64,
    protocol_errors: AtomicU64,
    rejected_connections: AtomicU64,
}

impl NetworkMetrics {
    fn snapshot(&self) -> NetworkMetricsSnapshot {
        NetworkMetricsSnapshot {
            bytes_sent: self.bytes_sent.load(Ordering::Relaxed),
            bytes_received: self.bytes_received.load(Ordering::Relaxed),
            protocol_errors: self.protocol_errors.load(Ordering::Relaxed),
            rejected_connections: self.rejected_connections.load(Ordering::Relaxed),
        }
    }
}

#[derive(Clone)]
pub struct NetworkHandle {
    identity: Arc<Identity>,
    config: Arc<NetworkConfig>,
    endpoint: Arc<StdMutex<Option<Endpoint>>>,
    peers: Arc<Mutex<PeerState>>,
    store: Arc<LocalStore>,
    events: mpsc::Sender<NetworkEvent>,
    connecting: Arc<Mutex<HashSet<String>>>,
    seen_hellos: SeenHellos,
    address_backoff: Arc<Mutex<HashMap<String, AddressBackoff>>>,
    observed_addresses: Arc<Mutex<Vec<AddressObservation>>>,
    address_sequence: Arc<AtomicU64>,
    connection_sequence: Arc<AtomicU64>,
    relay_routes: Arc<Mutex<HashMap<NodeId, NodeId>>>,
    relay_session_ids: Arc<Mutex<HashMap<(NodeId, NodeId), RequestId>>>,
    relay_sessions: Arc<Mutex<HashMap<RequestId, RelaySession>>>,
    dht: Arc<Mutex<DhtRuntime>>,
    shutdown: Arc<Notify>,
    metrics: Arc<NetworkMetrics>,
}

struct PeerState {
    records: HashMap<NodeId, PeerRecord>,
    announcements: HashMap<NodeId, SignedAnnouncement>,
    address_updates: HashMap<NodeId, AddressUpdate>,
    rotations: HashMap<NodeId, KeyRotation>,
    senders: HashMap<NodeId, mpsc::Sender<Message>>,
    connections: HashMap<NodeId, Connection>,
    connection_ids: HashMap<NodeId, u64>,
    connection_directions: HashMap<NodeId, bool>,
    versions: HashMap<NodeId, u16>,
}

#[derive(Clone, Copy)]
struct AddressBackoff {
    failures: u32,
    next_attempt: u64,
}

struct RelaySession {
    origin: NodeId,
    target: NodeId,
    bytes: u64,
    last_activity: u64,
}

struct DhtRuntime {
    table: DhtTable,
    store: DhtStore,
    pending: HashMap<RequestId, oneshot::Sender<DhtResponse>>,
    seen_requests: HashSet<RequestId>,
    request_order: VecDeque<RequestId>,
    rpc_sent: u64,
    rpc_received: u64,
    lookups: u64,
    lookup_successes: u64,
    invalid_records: u64,
    request_windows: HashMap<NodeId, (u64, u32)>,
    rate_limited_requests: u64,
}

impl NetworkHandle {
    pub fn node_id(&self) -> NodeId {
        self.identity.node_id()
    }

    pub fn local_address(&self) -> SocketAddr {
        self.endpoint
            .lock()
            .ok()
            .and_then(|endpoint| endpoint.as_ref().and_then(|value| value.local_addr().ok()))
            .unwrap_or(self.config.listen_addr)
    }

    pub async fn peer_records(&self) -> Vec<PeerRecord> {
        self.peers.lock().await.records.values().cloned().collect()
    }

    pub async fn connected_peers(&self) -> usize {
        self.peers.lock().await.senders.len()
    }

    /// Return the identities for currently authenticated connections.
    ///
    /// This is intentionally a snapshot: callers must still tolerate a peer
    /// disappearing immediately after the snapshot is taken.
    pub async fn connected_peer_ids(&self) -> Vec<NodeId> {
        self.peers.lock().await.senders.keys().copied().collect()
    }

    pub async fn is_connected(&self, peer: NodeId) -> bool {
        self.peers.lock().await.senders.contains_key(&peer)
    }

    pub fn metrics(&self) -> NetworkMetricsSnapshot {
        self.metrics.snapshot()
    }

    pub async fn observed_addresses(&self) -> Vec<AddressObservation> {
        self.observed_addresses.lock().await.clone()
    }

    pub async fn relay_peers(&self) -> Vec<NodeId> {
        self.relay_candidates().await
    }

    pub async fn dht_stats(&self) -> DhtNetworkStats {
        let dht = self.dht.lock().await;
        let now = now_secs();
        DhtNetworkStats {
            enabled: self.config.dht_enabled,
            routing: dht.table.stats(),
            records: dht.store.stats(now),
            rpc_sent: dht.rpc_sent,
            rpc_received: dht.rpc_received,
            lookups: dht.lookups,
            lookup_successes: dht.lookup_successes,
            invalid_records: dht.invalid_records,
            rate_limited_requests: dht.rate_limited_requests,
        }
    }

    /// Perform a bounded authenticated request over the existing peer
    /// connection.  This is deliberately separate from `send_to`: a queued
    /// QUIC write is not proof that the remote process or path is still
    /// responsive, while the request/response correlation is.  V4 uses this
    /// as a liveness probe during coordinator replacement without closing a
    /// healthy connection or opening an unrestricted address scan.
    pub async fn authenticated_ping(&self, peer: NodeId) -> Result<(), NetworkError> {
        let response = timeout(
            Duration::from_millis(750),
            self.dht_rpc(peer, DhtRequestKind::Ping),
        )
        .await
        .map_err(|_| NetworkError::PeerProtocol("authenticated ping timed out".to_string()))??;
        match response {
            DhtResponseKind::Pong => Ok(()),
            DhtResponseKind::Error { message, .. } => Err(NetworkError::PeerProtocol(message)),
            _ => Err(NetworkError::PeerProtocol(
                "authenticated ping returned an unexpected response".to_string(),
            )),
        }
    }

    pub async fn dht_publish(
        &self,
        namespace: DhtNamespace,
        name: &str,
        value: Vec<u8>,
        ttl_seconds: u64,
        sequence: u64,
    ) -> Result<DhtRecord, NetworkError> {
        if !self.config.dht_enabled || name.is_empty() || ttl_seconds == 0 || sequence == 0 {
            return Err(NetworkError::InvalidConfig(
                "DHT is disabled or publish parameters are invalid".to_string(),
            ));
        }
        let record = DhtRecord {
            namespace,
            key: DhtKey::for_name(namespace, name),
            owner: self.node_id(),
            owner_public_key: self.identity.public_key(),
            sequence,
            expires_at: now_secs().saturating_add(ttl_seconds),
            value,
            signature: Vec::new(),
        };
        let bytes = dht_record_signing_bytes(&record)?;
        let mut record = record;
        record.signature = self.identity.sign(&bytes);
        if !self.store_dht_record(record.clone()).await? {
            return Err(NetworkError::PeerProtocol(
                "DHT publish sequence is stale or already stored".to_string(),
            ));
        }
        let contacts = self.dht_find_node(record.key).await.unwrap_or_default();
        let deadline = Instant::now() + DHT_LOOKUP_DEADLINE;
        let mut writes = Vec::new();
        for contact in contacts.into_iter().take(self.config.dht_k) {
            let handle = self.clone();
            let record = record.clone();
            writes.push(tokio::spawn(async move {
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    return;
                }
                let _ = timeout(
                    remaining,
                    handle.dht_rpc(contact.record.node_id, DhtRequestKind::Put { record }),
                )
                .await;
            }));
        }
        for write in writes {
            let _ = write.await;
        }
        Ok(record)
    }

    pub async fn dht_lookup(
        &self,
        namespace: DhtNamespace,
        name: &str,
    ) -> Result<Vec<DhtRecord>, NetworkError> {
        if !self.config.dht_enabled || name.is_empty() {
            return Err(NetworkError::InvalidConfig(
                "DHT is disabled or lookup name is empty".to_string(),
            ));
        }
        let key = DhtKey::for_name(namespace, name);
        let mut records = {
            let mut dht = self.dht.lock().await;
            dht.lookups = dht.lookups.saturating_add(1);
            dht.store
                .get_diverse(namespace, key, now_secs(), MAX_DHT_RECORDS)
        };
        let contacts = self.dht_find_node(key).await?;
        let deadline = Instant::now() + DHT_LOOKUP_DEADLINE;
        let mut reads = Vec::new();
        for contact in contacts.into_iter().take(self.config.dht_k) {
            let handle = self.clone();
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            reads.push(tokio::spawn(async move {
                timeout(
                    remaining,
                    handle.dht_rpc(
                        contact.record.node_id,
                        DhtRequestKind::Get { namespace, key },
                    ),
                )
                .await
            }));
        }
        for read in reads {
            let Ok(Ok(Ok(DhtResponseKind::Records { records: remote }))) = read.await else {
                continue;
            };
            for record in remote {
                if verify_dht_record(&record) && record.namespace == namespace && record.key == key
                {
                    self.store_dht_record(record.clone()).await?;
                    if !records.iter().any(|existing| {
                        existing.owner == record.owner && existing.sequence == record.sequence
                    }) {
                        records.push(record);
                    }
                }
            }
        }
        records.retain(|record| record.expires_at >= now_secs());
        records.sort_by_key(|record| (record.owner, std::cmp::Reverse(record.sequence)));
        records.dedup_by_key(|record| (record.owner, record.sequence));
        if !records.is_empty() {
            let mut dht = self.dht.lock().await;
            dht.lookup_successes = dht.lookup_successes.saturating_add(1);
        }
        Ok(records)
    }

    pub async fn dht_find_node(
        &self,
        target: DhtKey,
    ) -> Result<Vec<SignedAnnouncement>, NetworkError> {
        if !self.config.dht_enabled {
            return Err(NetworkError::InvalidConfig("DHT is disabled".to_string()));
        }
        let mut candidates = {
            let dht = self.dht.lock().await;
            dht.table
                .closest_diverse(target, self.config.dht_k, now_secs())
        };
        let mut queried = HashSet::new();
        let deadline = Instant::now() + DHT_LOOKUP_DEADLINE;
        for _ in 0..8 {
            if Instant::now() >= deadline {
                break;
            }
            candidates
                .sort_by_key(|contact| dht::xor_distance(contact.record.node_id, NodeId(target.0)));
            candidates.truncate(
                self.config
                    .dht_k
                    .saturating_mul(2)
                    .max(self.config.dht_alpha),
            );
            let batch = candidates
                .iter()
                .filter(|contact| queried.insert(contact.record.node_id))
                .take(self.config.dht_alpha)
                .cloned()
                .collect::<Vec<_>>();
            if batch.is_empty() {
                break;
            }
            let mut queries = Vec::new();
            for contact in batch {
                let handle = self.clone();
                let remaining = deadline.saturating_duration_since(Instant::now());
                if remaining.is_zero() {
                    break;
                }
                queries.push(tokio::spawn(async move {
                    timeout(
                        remaining,
                        handle.dht_rpc(contact.record.node_id, DhtRequestKind::FindNode { target }),
                    )
                    .await
                }));
            }
            let mut changed = false;
            for query in queries {
                let Ok(Ok(Ok(DhtResponseKind::Nodes { contacts: remote }))) = query.await else {
                    continue;
                };
                for remote_contact in remote {
                    if !verify_announcement(&remote_contact)
                        || remote_contact.record.node_id == self.node_id()
                        || remote_contact.record.expires_at < now_secs()
                    {
                        continue;
                    }
                    upsert_peer(self, remote_contact.clone()).await?;
                    let id = remote_contact.record.node_id;
                    if !candidates.iter().any(|item| item.record.node_id == id) {
                        candidates.push(remote_contact);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        candidates
            .sort_by_key(|contact| dht::xor_distance(contact.record.node_id, NodeId(target.0)));
        candidates.truncate(self.config.dht_k);
        Ok(candidates)
    }

    pub async fn reachability_state(&self) -> ReachabilityState {
        if self
            .config
            .advertise_addr
            .parse::<SocketAddr>()
            .is_ok_and(|address| {
                address_is_public(address) && self.config.listen_addr.ip().is_unspecified()
            })
        {
            ReachabilityState::PubliclyReachable
        } else {
            ReachabilityState::NatUnknown
        }
    }

    pub async fn send_to(&self, node: NodeId, message: Message) -> Result<(), NetworkError> {
        let relay_allowed = !matches!(message, Message::RelayEnvelope(_));
        if relay_allowed
            && self.direct_connection_available(node).await
            && self.send_direct(node, message.clone()).await.is_ok()
        {
            return Ok(());
        }
        if self.config.prefer_relay
            && relay_allowed
            && self
                .send_via_relay_for_target(node, message.clone())
                .await
                .is_ok()
        {
            return Ok(());
        }
        if let Some(relay) = self.relay_routes.lock().await.get(&node).copied()
            && relay_allowed
            && self
                .send_via_relay(relay, node, message.clone())
                .await
                .is_ok()
        {
            return Ok(());
        }
        if self.send_direct(node, message.clone()).await.is_ok() {
            return Ok(());
        }
        if relay_allowed {
            self.send_via_relay_for_target(node, message).await
        } else {
            Err(NetworkError::NotConnected)
        }
    }

    async fn direct_connection_available(&self, node: NodeId) -> bool {
        let peers = self.peers.lock().await;
        peers
            .senders
            .get(&node)
            .is_some_and(|sender| !sender.is_closed())
            && peers
                .connections
                .get(&node)
                .is_some_and(|connection| connection.close_reason().is_none())
    }

    // DHT RPCs are only sent over an already authenticated connection.  They
    // must not recursively invoke address dialing from inside the connection
    // reader task: a malformed or adversarial DHT request must never create a
    // second connection-task future cycle.
    async fn send_connected(&self, node: NodeId, message: Message) -> Result<(), NetworkError> {
        let sender = {
            let peers = self.peers.lock().await;
            if peers.versions.get(&node).copied().unwrap_or(0) < message_required_minor(&message) {
                return Err(NetworkError::PeerProtocol(
                    "message requires a newer protocol minor".to_string(),
                ));
            }
            peers
                .senders
                .get(&node)
                .cloned()
                .ok_or(NetworkError::NotConnected)?
        };
        sender
            .send(message)
            .await
            .map_err(|_| NetworkError::NotConnected)
    }

    pub async fn send_direct(&self, node: NodeId, message: Message) -> Result<(), NetworkError> {
        let mut sender = {
            let peers = self.peers.lock().await;
            if peers.versions.get(&node).copied().unwrap_or(0) < message_required_minor(&message) {
                return Err(NetworkError::PeerProtocol(
                    "message requires protocol minor 1".to_string(),
                ));
            }
            peers.senders.get(&node).cloned()
        };
        if sender.is_none() {
            let addresses = self
                .peers
                .lock()
                .await
                .records
                .get(&node)
                .map(|record| preferred_addresses(self, &record.addresses))
                .ok_or(NetworkError::NotConnected)?;
            for address in addresses {
                let _ = self.connect_address(address).await;
            }
            for _ in 0..50 {
                if let Some(found) = self.peers.lock().await.senders.get(&node).cloned() {
                    sender = Some(found);
                    break;
                }
                sleep(Duration::from_millis(100)).await;
            }
        }
        let Some(sender) = sender else {
            return Err(NetworkError::NotConnected);
        };
        if self
            .peers
            .lock()
            .await
            .versions
            .get(&node)
            .copied()
            .unwrap_or(0)
            < message_required_minor(&message)
        {
            return Err(NetworkError::PeerProtocol(
                "message requires protocol minor 1".to_string(),
            ));
        }
        sender
            .send(message)
            .await
            .map_err(|_| NetworkError::NotConnected)
    }

    pub async fn reconnect_peer(&self, node: NodeId) -> Result<(), NetworkError> {
        let (addresses, connection) = {
            let mut peers = self.peers.lock().await;
            let connection = peers.connections.remove(&node);
            peers.senders.remove(&node);
            peers.connection_ids.remove(&node);
            peers.connection_directions.remove(&node);
            peers.versions.remove(&node);
            let addresses = peers
                .records
                .get(&node)
                .map(|record| preferred_addresses(self, &record.addresses))
                .ok_or(NetworkError::NotConnected)?;
            (addresses, connection)
        };
        if let Some(connection) = connection {
            connection.close(VarInt::from_u32(0), b"reconnect");
        }
        for attempt in 0..100 {
            if self.peers.lock().await.senders.contains_key(&node) {
                return Ok(());
            }
            if attempt % 5 == 0 {
                for address in &addresses {
                    self.address_backoff.lock().await.remove(address);
                    let _ = self.connect_address(address.clone()).await;
                }
            }
            sleep(Duration::from_millis(100)).await;
        }
        Err(NetworkError::NotConnected)
    }

    async fn send_via_relay_for_target(
        &self,
        target: NodeId,
        message: Message,
    ) -> Result<(), NetworkError> {
        let preferred = self.relay_routes.lock().await.get(&target).copied();
        let mut candidates = Vec::new();
        if let Some(relay) = preferred {
            candidates.push(relay);
        }
        for relay in self.relay_candidates().await {
            if !candidates.contains(&relay) {
                candidates.push(relay);
            }
        }
        if candidates.is_empty() {
            for address in &self.config.relay_addresses {
                let _ = self.connect_address(address.clone()).await;
            }
            for _ in 0..20 {
                candidates = self.relay_candidates().await;
                if let Some(relay) = candidates.first().copied() {
                    if self
                        .send_via_relay(relay, target, message.clone())
                        .await
                        .is_ok()
                    {
                        self.relay_routes.lock().await.insert(target, relay);
                        return Ok(());
                    }
                }
                sleep(Duration::from_millis(100)).await;
            }
            return Err(NetworkError::NotConnected);
        }
        for relay in candidates {
            if self
                .send_via_relay(relay, target, message.clone())
                .await
                .is_ok()
            {
                self.relay_routes.lock().await.insert(target, relay);
                return Ok(());
            }
        }
        self.relay_routes.lock().await.remove(&target);
        Err(NetworkError::NotConnected)
    }

    async fn relay_candidates(&self) -> Vec<NodeId> {
        let peers = self.peers.lock().await;
        peers
            .records
            .iter()
            .filter_map(|(node_id, record)| {
                (record
                    .capabilities
                    .iter()
                    .any(|capability| capability.name == "network.relay")
                    && peers
                        .connections
                        .get(node_id)
                        .is_some_and(|connection| connection.close_reason().is_none())
                    && peers
                        .senders
                        .get(node_id)
                        .is_some_and(|sender| !sender.is_closed())
                    && *node_id != self.node_id())
                .then_some(*node_id)
            })
            .collect()
    }

    async fn send_via_relay(
        &self,
        relay: NodeId,
        target: NodeId,
        message: Message,
    ) -> Result<(), NetworkError> {
        if relay == target {
            return Err(NetworkError::NotConnected);
        }
        message
            .validate()
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        let payload = postcard::to_allocvec(&message)
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        if payload.len() > MAX_RELAY_PAYLOAD {
            return Err(NetworkError::PeerProtocol(
                "message is too large for relay transport".to_string(),
            ));
        }
        let session_id = {
            let mut sessions = self.relay_session_ids.lock().await;
            *sessions
                .entry((relay, target))
                .or_insert_with(random_request_id)
        };
        let origin = self.node_id();
        let public_key = self.identity.public_key();
        let bytes = relay_signing_bytes(session_id, origin, target, &payload)?;
        let envelope = RelayEnvelope {
            session_id,
            origin,
            origin_public_key: public_key,
            target,
            payload,
            signature: self.identity.sign(&bytes),
        };
        tracing::debug!(
            origin = %origin,
            relay = %relay,
            target = %target,
            payload_bytes = envelope.payload.len(),
            "sending relay envelope"
        );
        let sender = {
            let peers = self.peers.lock().await;
            if peers.versions.get(&relay).copied().unwrap_or(0) < message_required_minor(&message) {
                return Err(NetworkError::PeerProtocol(
                    "message requires protocol minor 1".to_string(),
                ));
            }
            let sender = peers
                .senders
                .get(&relay)
                .cloned()
                .ok_or(NetworkError::NotConnected)?;
            if sender.is_closed()
                || peers
                    .connections
                    .get(&relay)
                    .is_some_and(|connection| connection.close_reason().is_some())
            {
                return Err(NetworkError::NotConnected);
            }
            sender
        };
        sender
            .send(Message::RelayEnvelope(envelope))
            .await
            .map_err(|_| NetworkError::NotConnected)
    }

    pub async fn connect_address(&self, address: String) -> Result<(), NetworkError> {
        let parsed: SocketAddr = address.parse().map_err(|error| {
            NetworkError::InvalidConfig(format!("invalid peer address: {error}"))
        })?;
        let configured = self.config.bootstrap.iter().any(|item| item == &address)
            || self
                .config
                .relay_addresses
                .iter()
                .any(|item| item == &address);
        if !configured && !address_allowed(&self.config, parsed, AddressSource::Learned) {
            return Err(NetworkError::InvalidConfig(
                "peer address is not allowed by local address policy".to_string(),
            ));
        }
        let already_connected = {
            let peers = self.peers.lock().await;
            peers.records.iter().any(|(node, record)| {
                record.addresses.iter().any(|item| item == &address)
                    && peers
                        .connections
                        .get(node)
                        .is_some_and(|connection| connection.close_reason().is_none())
                    && peers
                        .senders
                        .get(node)
                        .is_some_and(|sender| !sender.is_closed())
            })
        };
        if already_connected {
            return Ok(());
        }
        let key = parsed.to_string();
        {
            let mut connecting = self.connecting.lock().await;
            if !connecting.insert(key.clone()) {
                return Ok(());
            }
        }
        {
            let backoff = self.address_backoff.lock().await;
            if let Some(state) = backoff.get(&key)
                && state.next_attempt > now_secs()
            {
                self.connecting.lock().await.remove(&key);
                return Ok(());
            }
        }
        let handle = self.clone();
        tokio::spawn(async move {
            let result = outbound_connection(handle.clone(), parsed).await;
            handle.connecting.lock().await.remove(&key);
            match result {
                Ok(()) => {
                    handle.address_backoff.lock().await.remove(&key);
                }
                Err(error) => {
                    let mut backoff = handle.address_backoff.lock().await;
                    let state = backoff.entry(key.clone()).or_insert(AddressBackoff {
                        failures: 0,
                        next_attempt: 0,
                    });
                    state.failures = state.failures.saturating_add(1).min(8);
                    state.next_attempt =
                        now_secs().saturating_add(2u64.saturating_pow(state.failures).min(300));
                    tracing::debug!(address = %parsed, error = %error, retry_after = state.next_attempt, "outbound peer connection failed");
                }
            }
        });
        Ok(())
    }

    async fn store_dht_record(&self, record: DhtRecord) -> Result<bool, NetworkError> {
        if !verify_dht_record(&record)
            || !dht_expiry_valid(&record, now_secs(), self.config.peer_ttl_seconds)
        {
            let mut dht = self.dht.lock().await;
            dht.invalid_records = dht.invalid_records.saturating_add(1);
            return Err(NetworkError::PeerProtocol(
                "DHT record signature, owner binding, or expiry is invalid".to_string(),
            ));
        }
        let mut dht = self.dht.lock().await;
        let accepted = dht.store.put(record, now_secs());
        let records = dht.store.records(now_secs());
        drop(dht);
        if accepted {
            self.store.save_dht_records(&records)?;
        }
        Ok(accepted)
    }

    async fn dht_rpc(
        &self,
        peer: NodeId,
        request: DhtRequestKind,
    ) -> Result<DhtResponseKind, NetworkError> {
        let request_id = random_request_id();
        let (sender, receiver) = oneshot::channel();
        {
            let mut dht = self.dht.lock().await;
            if dht.pending.len() >= self.config.dht_k.saturating_mul(4).max(16) {
                return Err(NetworkError::PeerProtocol(
                    "DHT request concurrency limit reached".to_string(),
                ));
            }
            dht.pending.insert(request_id, sender);
            dht.rpc_sent = dht.rpc_sent.saturating_add(1);
        }
        let message = Message::DhtRequest(DhtRequest {
            request_id,
            origin: self.node_id(),
            request,
        });
        if let Err(error) = self.send_connected(peer, message).await {
            self.dht.lock().await.pending.remove(&request_id);
            return Err(error);
        }
        let response = match timeout(DHT_RPC_TIMEOUT, receiver).await {
            Ok(result) => result.map_err(|_| NetworkError::NotConnected)?,
            Err(_) => {
                self.dht.lock().await.pending.remove(&request_id);
                return Err(NetworkError::PeerProtocol("DHT RPC timed out".to_string()));
            }
        };
        Ok(response.response)
    }

    async fn handle_dht_request(
        &self,
        authenticated_peer: NodeId,
        request: DhtRequest,
    ) -> Result<(), NetworkError> {
        request
            .validate()
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        if request.origin != authenticated_peer {
            return Err(NetworkError::IdentityVerification);
        }
        let (duplicate, rate_limited) = {
            let mut dht = self.dht.lock().await;
            let window = now_secs() / DHT_REQUEST_WINDOW_SECONDS;
            let over_budget = {
                let budget = dht
                    .request_windows
                    .entry(authenticated_peer)
                    .or_insert((window, 0));
                if budget.0 != window {
                    *budget = (window, 0);
                }
                if budget.1 >= DHT_REQUESTS_PER_WINDOW {
                    true
                } else {
                    budget.1 = budget.1.saturating_add(1);
                    false
                }
            };
            if over_budget {
                dht.rate_limited_requests = dht.rate_limited_requests.saturating_add(1);
                (false, true)
            } else if !dht.seen_requests.insert(request.request_id) {
                (true, false)
            } else {
                dht.request_order.push_back(request.request_id);
                if dht.request_order.len() > MAX_SEEN_DHT_REQUESTS
                    && let Some(expired) = dht.request_order.pop_front()
                {
                    dht.seen_requests.remove(&expired);
                }
                (false, false)
            }
        };
        if rate_limited {
            let response = DhtResponse {
                request_id: request.request_id,
                responder: self.node_id(),
                response: DhtResponseKind::Error {
                    code: intelligence_protocol::ErrorCode::Busy,
                    message: "DHT request rate limit reached".to_string(),
                },
            };
            return self
                .send_connected(authenticated_peer, Message::DhtResponse(response))
                .await;
        }
        if duplicate {
            let response = DhtResponse {
                request_id: request.request_id,
                responder: self.node_id(),
                response: DhtResponseKind::Error {
                    code: intelligence_protocol::ErrorCode::Duplicate,
                    message: "DHT request ID was already processed".to_string(),
                },
            };
            return self
                .send_connected(authenticated_peer, Message::DhtResponse(response))
                .await;
        }
        let response = match request.request {
            DhtRequestKind::Ping => DhtResponseKind::Pong,
            DhtRequestKind::FindNode { target } => {
                let contacts = self.dht.lock().await.table.closest_diverse(
                    target,
                    self.config.dht_k.min(MAX_DHT_CONTACTS),
                    now_secs(),
                );
                DhtResponseKind::Nodes { contacts }
            }
            DhtRequestKind::Get { namespace, key } => {
                let records = self.dht.lock().await.store.get_diverse(
                    namespace,
                    key,
                    now_secs(),
                    MAX_DHT_RECORDS,
                );
                if records.is_empty() {
                    DhtResponseKind::NotFound
                } else {
                    DhtResponseKind::Records { records }
                }
            }
            DhtRequestKind::Put { record } => {
                if record.owner != authenticated_peer || !verify_dht_record(&record) {
                    let mut dht = self.dht.lock().await;
                    dht.invalid_records = dht.invalid_records.saturating_add(1);
                    DhtResponseKind::Error {
                        code: intelligence_protocol::ErrorCode::Unauthorized,
                        message: "DHT record owner signature is invalid".to_string(),
                    }
                } else if self.store_dht_record(record).await? {
                    DhtResponseKind::Stored
                } else {
                    DhtResponseKind::Error {
                        code: intelligence_protocol::ErrorCode::Duplicate,
                        message: "DHT record is stale or already stored".to_string(),
                    }
                }
            }
        };
        let response = DhtResponse {
            request_id: request.request_id,
            responder: self.node_id(),
            response,
        };
        self.send_connected(authenticated_peer, Message::DhtResponse(response))
            .await
    }

    async fn handle_dht_response(
        &self,
        authenticated_peer: NodeId,
        response: DhtResponse,
    ) -> Result<(), NetworkError> {
        response
            .validate()
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        if response.responder != authenticated_peer {
            return Err(NetworkError::IdentityVerification);
        }
        let sender = self.dht.lock().await.pending.remove(&response.request_id);
        if let Some(sender) = sender {
            let mut dht = self.dht.lock().await;
            dht.rpc_received = dht.rpc_received.saturating_add(1);
            drop(dht);
            let _ = sender.send(response);
        }
        Ok(())
    }

    pub async fn shutdown(&self) {
        self.shutdown.notify_waiters();
        let endpoint = self.endpoint.lock().ok().and_then(|mut value| value.take());
        let Some(endpoint) = endpoint else {
            return;
        };
        let listen_addr = endpoint.local_addr().ok();
        endpoint.close(VarInt::from_u32(0), b"shutdown");
        // Closing the endpoint is asynchronous. Wait for connections to
        // drain, then for the endpoint driver to release the UDP socket, so
        // that a same-address restart is possible as soon as this returns.
        // Both waits stay bounded if a peer or the runtime is uncooperative.
        let _ = timeout(Duration::from_secs(2), endpoint.wait_idle()).await;
        drop(endpoint);
        let Some(listen_addr) = listen_addr else {
            return;
        };
        let released = Instant::now() + Duration::from_secs(2);
        while Instant::now() < released {
            match std::net::UdpSocket::bind(listen_addr) {
                Ok(_) => return,
                Err(error) if error.kind() == io::ErrorKind::AddrInUse => {
                    tokio::task::yield_now().await;
                    sleep(Duration::from_millis(5)).await;
                }
                Err(_) => return,
            }
        }
        tracing::warn!(%listen_addr, "UDP socket was not released within the shutdown bound");
    }
}

pub async fn start(
    config: NetworkConfig,
    identity: Identity,
    store: LocalStore,
    events: mpsc::Sender<NetworkEvent>,
) -> Result<NetworkHandle, NetworkError> {
    config.validate()?;
    let (server_config, client_config) = transport_configs().map_err(NetworkError::Quic)?;
    let mut endpoint = Endpoint::server(server_config, config.listen_addr)
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    endpoint.set_default_client_config(client_config);
    let identity = Arc::new(identity);
    let store = Arc::new(store);
    let records = store
        .load_peer_records()?
        .into_iter()
        .filter(|record| record.node_id != identity.node_id())
        .map(|record| (record.node_id, record))
        .collect();
    let rotations = store
        .load_key_rotations()?
        .into_iter()
        .map(|rotation| (rotation.new_node_id, rotation))
        .collect();
    let mut dht_store = DhtStore::new(config.dht_max_records, DEFAULT_MAX_RECORDS_PER_KEY);
    for record in store.load_dht_records()? {
        if verify_dht_record(&record)
            && dht_expiry_valid(&record, now_secs(), config.peer_ttl_seconds)
        {
            let _ = dht_store.put(record, now_secs());
        }
    }
    let persisted_dht_contacts = store.load_dht_contacts()?;
    let mut dht_table = DhtTable::new(identity.node_id(), config.dht_k, DEFAULT_REPLACEMENT_CACHE);
    for contact in persisted_dht_contacts {
        if verify_announcement(&contact) {
            let _ = dht_table.insert(contact, now_secs());
        }
    }
    let dht = Arc::new(Mutex::new(DhtRuntime {
        table: dht_table,
        store: dht_store,
        pending: HashMap::new(),
        seen_requests: HashSet::new(),
        request_order: VecDeque::new(),
        rpc_sent: 0,
        rpc_received: 0,
        lookups: 0,
        lookup_successes: 0,
        invalid_records: 0,
        request_windows: HashMap::new(),
        rate_limited_requests: 0,
    }));
    let peers = Arc::new(Mutex::new(PeerState {
        records,
        announcements: HashMap::new(),
        address_updates: HashMap::new(),
        rotations,
        senders: HashMap::new(),
        connections: HashMap::new(),
        connection_ids: HashMap::new(),
        connection_directions: HashMap::new(),
        versions: HashMap::new(),
    }));
    let endpoint_for_handle = Arc::new(StdMutex::new(Some(endpoint.clone())));
    let handle = NetworkHandle {
        identity,
        config: Arc::new(config),
        endpoint: endpoint_for_handle,
        peers,
        store,
        events,
        connecting: Arc::new(Mutex::new(HashSet::new())),
        seen_hellos: Arc::new(Mutex::new(HashSet::new())),
        address_backoff: Arc::new(Mutex::new(HashMap::new())),
        observed_addresses: Arc::new(Mutex::new(Vec::new())),
        address_sequence: Arc::new(AtomicU64::new(1)),
        connection_sequence: Arc::new(AtomicU64::new(1)),
        relay_routes: Arc::new(Mutex::new(HashMap::new())),
        relay_session_ids: Arc::new(Mutex::new(HashMap::new())),
        relay_sessions: Arc::new(Mutex::new(HashMap::new())),
        dht,
        shutdown: Arc::new(Notify::new()),
        metrics: Arc::new(NetworkMetrics::default()),
    };
    let accept_handle = handle.clone();
    tokio::spawn(async move { accept_loop(endpoint, accept_handle).await });
    let maintenance_handle = handle.clone();
    tokio::spawn(async move { maintenance_loop(maintenance_handle).await });
    for address in handle.config.bootstrap.clone() {
        handle.connect_address(address).await?;
    }
    for address in handle.config.relay_addresses.clone() {
        handle.connect_address(address).await?;
    }
    Ok(handle)
}

async fn accept_loop(endpoint: Endpoint, handle: NetworkHandle) {
    loop {
        tokio::select! {
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else { break };
                let connection = match incoming.await {
                    Ok(connection) => connection,
                    Err(error) => {
                        tracing::debug!(error = %error, "inbound QUIC handshake failed");
                        continue;
                    }
                };
                if handle.peers.lock().await.senders.len() >= handle.config.max_connections {
                    handle
                        .metrics
                        .rejected_connections
                        .fetch_add(1, Ordering::Relaxed);
                    connection.close(VarInt::from_u32(0), b"connection limit");
                    continue;
                }
                let task_handle = handle.clone();
                tokio::spawn(async move {
                    if let Err(error) = connection_task(task_handle.clone(), connection, false).await {
                        tracing::debug!(error = %error, "inbound connection ended");
                    }
                });
            }
            _ = handle.shutdown.notified() => break,
        }
    }
}

async fn maintenance_loop(handle: NetworkHandle) {
    loop {
        tokio::select! {
            _ = sleep(Duration::from_secs(3)) => {
                let mut addresses = handle.config.bootstrap.clone();
                let bootstrap_count = addresses.len();
                addresses.extend(
                    handle
                        .peers
                        .lock()
                        .await
                        .records
                        .values()
                        .filter(|record| record.node_id != handle.node_id())
                        .flat_map(|record| preferred_addresses(&handle, &record.addresses)),
                );
                for (index, address) in addresses.into_iter().enumerate() {
                    if index >= bootstrap_count
                        && (!handle.config.hole_punch_enabled
                            || index.saturating_sub(bootstrap_count)
                                >= handle.config.hole_punch_max_attempts)
                    {
                        continue;
                    }
                    let _ = handle.connect_address(address).await;
                }
                let address_update = if handle.config.hole_punch_enabled {
                    sign_address_update(
                        &handle,
                        address_records(&handle).await,
                        handle.address_sequence.fetch_add(1, Ordering::Relaxed),
                    )
                    .ok()
                } else {
                    None
                };
                let now = now_secs();
                let (dht_records, dht_contacts) = {
                    let mut dht = handle.dht.lock().await;
                    dht.table.expire(now);
                    dht.store.expire(now);
                    (dht.store.records(now), dht.table.all_contacts(now))
                };
                if let Err(error) = handle.store.save_dht_records(&dht_records) {
                    tracing::warn!(error = %error, "failed to persist DHT records");
                }
                if let Err(error) = handle.store.save_dht_contacts(&dht_contacts) {
                    tracing::warn!(error = %error, "failed to persist DHT contacts");
                }
                // A provider may publish before its first authenticated peer
                // is connected. Re-advertise only records owned by this node
                // on a bounded maintenance cadence; this is local replication
                // policy, not a central registry or an unbounded gossip loop.
                for record in dht_records
                    .iter()
                    .filter(|record| record.owner == handle.node_id())
                    .take(16)
                    .cloned()
                {
                    let contacts = handle
                        .dht_find_node(record.key)
                        .await
                        .unwrap_or_default();
                    for contact in contacts
                        .into_iter()
                        .filter(|contact| contact.record.node_id != handle.node_id())
                        .take(handle.config.dht_k)
                    {
                        let _ = handle
                            .dht_rpc(
                                contact.record.node_id,
                                DhtRequestKind::Put {
                                    record: record.clone(),
                                },
                            )
                            .await;
                    }
                }
                let mut peers = handle.peers.lock().await;
                peers.records.retain(|_, record| record.expires_at >= now);
                let live_ids = peers.records.keys().copied().collect::<HashSet<_>>();
                peers.announcements.retain(|node_id, announcement| {
                    announcement.record.expires_at >= now && live_ids.contains(node_id)
                });
                let records = peers.records.values().cloned().collect::<Vec<_>>();
                let update_senders = address_update.as_ref().map(|_| {
                    peers
                        .senders
                        .iter()
                        .filter(|(node_id, _)| peers.versions.get(node_id).copied().unwrap_or(0) >= 1)
                        .map(|(_, sender)| sender.clone())
                        .collect::<Vec<_>>()
                });
                drop(peers);
                if let Some(update) = address_update {
                    if let Some(senders) = update_senders {
                        for sender in senders {
                            let _ = sender.try_send(Message::AddressUpdate(update.clone()));
                        }
                    }
                }
                if let Err(error) = handle.store.save_peer_records(&records) {
                    tracing::warn!(error = %error, "failed to persist peer table");
                }
            }
            _ = handle.shutdown.notified() => break,
        }
    }
}

async fn outbound_connection(
    handle: NetworkHandle,
    address: SocketAddr,
) -> Result<(), NetworkError> {
    let endpoint = handle
        .endpoint
        .lock()
        .map_err(|_| NetworkError::Quic("network endpoint lock poisoned".to_string()))?
        .as_ref()
        .cloned()
        .ok_or_else(|| NetworkError::Quic("network endpoint is shut down".to_string()))?;
    let connecting = endpoint
        .connect(address, "intelligence-network")
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    // The endpoint handle is only needed to dial.  Holding it for the life of
    // the connection would keep the UDP socket open until every outbound
    // session task has exited, which delays a same-address restart.
    drop(endpoint);
    let connection = timeout(Duration::from_secs(5), connecting)
        .await
        .map_err(|_| NetworkError::Quic("peer connection timed out".to_string()))?
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    connection_task(handle, connection, true).await
}

async fn connection_task(
    handle: NetworkHandle,
    connection: Connection,
    outbound: bool,
) -> Result<(), NetworkError> {
    let (mut send, mut recv) = if outbound {
        connection
            .open_bi()
            .await
            .map_err(|error| NetworkError::Quic(error.to_string()))?
    } else {
        connection
            .accept_bi()
            .await
            .map_err(|error| NetworkError::Quic(error.to_string()))?
    };
    let metrics = handle.metrics.clone();
    let local_hello = make_hello(&handle)?;
    write_message(
        &mut send,
        &Message::Hello(local_hello),
        handle.config.max_frame_size,
        0,
        &metrics,
    )
    .await?;
    let first = timeout(
        HELLO_TIMEOUT,
        read_message(&mut recv, handle.config.max_frame_size, &metrics),
    )
    .await
    .map_err(|_| NetworkError::PeerProtocol("hello timeout".to_string()))??;
    let Message::Hello(remote_hello) = first else {
        return Err(NetworkError::PeerProtocol(
            "first message was not hello".to_string(),
        ));
    };
    let protocol_minor = verify_hello(&handle, &remote_hello).await?;
    let peer = remote_hello.record.clone();
    let mut announcements = handle
        .peers
        .lock()
        .await
        .announcements
        .values()
        .filter(|item| item.record.expires_at >= now_secs())
        .cloned()
        .collect::<Vec<_>>();
    announcements.push(sign_announcement(&handle, local_record(&handle))?);
    announcements.sort_by_key(|item| item.record.node_id);
    announcements.dedup_by_key(|item| item.record.node_id);
    announcements.truncate(MAX_PEERS);
    let exchange = PeerExchange {
        peers: announcements,
    };
    write_message(
        &mut send,
        &Message::PeerExchange(exchange),
        handle.config.max_frame_size,
        protocol_minor,
        &metrics,
    )
    .await?;
    let (tx, mut rx) = mpsc::channel::<Message>(CONNECTION_QUEUE);
    let connection_id = handle.connection_sequence.fetch_add(1, Ordering::Relaxed);
    // Both peers may dial at the same time during recovery.  Without a
    // deterministic winner, each side can classify the other's connection as
    // the duplicate and close the connection that the other side currently
    // considers canonical.  That race is especially visible during a V4 graph
    // activation, when several control messages can cause reconnect attempts
    // concurrently.  The lower NodeId is the canonical dialer; the higher
    // NodeId is the canonical acceptor.
    let preferred_outbound = handle.node_id() < peer.node_id;
    let existing_canonical = {
        let peers = handle.peers.lock().await;
        let sender = peers.senders.get(&peer.node_id);
        let connection = peers.connections.get(&peer.node_id);
        let direction = peers.connection_directions.get(&peer.node_id).copied();
        let connection_id = peers.connection_ids.get(&peer.node_id).copied();
        (|| {
            if sender.is_none_or(|sender| sender.is_closed())
                || connection.is_none_or(|connection| connection.close_reason().is_some())
                || direction != Some(preferred_outbound)
                || outbound != preferred_outbound
            {
                return None;
            }
            let protocol_minor = peers.versions.get(&peer.node_id).copied().unwrap_or(0);
            Some((connection_id?, protocol_minor))
        })()
    };
    // A process or virtual link can disappear without QUIC immediately
    // exposing a close reason to the surviving endpoint.  In that state the
    // deterministic duplicate rule would keep the dead canonical connection
    // forever and reject the only usable replacement.  Probe only the exact
    // same-direction canonical connection, and only when the negotiated
    // protocol supports authenticated DHT ping.  A live connection wins;
    // an unanswered one may be replaced by this already-authenticated
    // handshake.  This is intentionally bounded and never tears down a
    // healthy session speculatively.
    let replace_stale_canonical = if let Some((_, protocol_minor)) = existing_canonical {
        protocol_minor >= 2
            && timeout(
                Duration::from_millis(750),
                handle.authenticated_ping(peer.node_id),
            )
            .await
            .is_err()
    } else {
        false
    };
    let mut replaced_connection = None;
    let mut reject_connection = false;
    let mut existing_direction = None;
    let mut existing_connection_id = None;
    let mut replacing_closed = false;
    {
        let mut peers = handle.peers.lock().await;
        if peers
            .senders
            .get(&peer.node_id)
            .is_some_and(|sender| !sender.is_closed())
        {
            let replace_closed = peers
                .connections
                .get(&peer.node_id)
                .is_none_or(|existing| existing.close_reason().is_some())
                || peers
                    .senders
                    .get(&peer.node_id)
                    .is_none_or(|sender| sender.is_closed());
            existing_direction = peers.connection_directions.get(&peer.node_id).copied();
            existing_connection_id = peers.connection_ids.get(&peer.node_id).copied();
            replacing_closed = replace_closed;
            let replacing_stale_canonical = replace_stale_canonical
                && existing_canonical.is_some_and(|(connection_id, _)| {
                    peers.connection_ids.get(&peer.node_id).copied() == Some(connection_id)
                });
            if replace_closed || replacing_stale_canonical {
                replaced_connection = peers.connections.remove(&peer.node_id);
                peers.senders.remove(&peer.node_id);
                peers.connection_ids.remove(&peer.node_id);
                peers.connection_directions.remove(&peer.node_id);
                peers.versions.remove(&peer.node_id);
            } else {
                let existing_outbound = peers.connection_directions.get(&peer.node_id).copied();
                let keep_new = existing_outbound.is_some_and(|existing| {
                    existing != preferred_outbound && outbound == preferred_outbound
                });
                if !keep_new {
                    reject_connection = true;
                } else {
                    replaced_connection = peers.connections.remove(&peer.node_id);
                    peers.senders.remove(&peer.node_id);
                    peers.connection_ids.remove(&peer.node_id);
                    peers.connection_directions.remove(&peer.node_id);
                    peers.versions.remove(&peer.node_id);
                }
            }
        }
        tracing::debug!(
            peer = %peer.node_id,
            outbound,
            preferred_outbound,
            existing_direction = ?existing_direction,
            existing_connection_id = ?existing_connection_id,
            replacing_closed,
            replace_stale_canonical,
            reject_connection,
            connection_id,
            "evaluated canonical peer connection"
        );
        if !reject_connection && peers.senders.len() >= handle.config.max_connections {
            connection.close(VarInt::from_u32(0), b"connection limit");
            return Err(NetworkError::PeerTableFull);
        }
        if !reject_connection
            && !peers.records.contains_key(&peer.node_id)
            && peers.records.len() >= MAX_LEARNED_PEERS
        {
            let now = now_secs();
            peers.records.retain(|_, value| value.expires_at >= now);
            if peers.records.len() >= MAX_LEARNED_PEERS {
                connection.close(VarInt::from_u32(0), b"peer table full");
                return Err(NetworkError::PeerTableFull);
            }
        }
        if peers
            .records
            .get(&peer.node_id)
            .is_none_or(|previous| record_is_fresher(previous, &peer))
        {
            peers.records.insert(peer.node_id, peer.clone());
        }
        if !reject_connection {
            peers.senders.insert(peer.node_id, tx.clone());
            peers.connections.insert(peer.node_id, connection.clone());
            peers.connection_ids.insert(peer.node_id, connection_id);
            peers.connection_directions.insert(peer.node_id, outbound);
            peers.versions.insert(peer.node_id, protocol_minor);
        }
        let records = peers.records.values().cloned().collect::<Vec<_>>();
        drop(peers);
        if !reject_connection {
            handle.store.save_peer_records(&records)?;
        }
    }
    if reject_connection {
        tracing::debug!(
            peer = %peer.node_id,
            outbound,
            connection_id,
            "closing non-canonical duplicate peer connection"
        );
        connection.close(VarInt::from_u32(0), b"duplicate connection");
        return Ok(());
    }
    if let Some(connection) = replaced_connection {
        tracing::debug!(
            peer = %peer.node_id,
            connection_id,
            "closing replaced peer connection"
        );
        connection.close(VarInt::from_u32(0), b"replaced");
    }
    handle
        .events
        .send(NetworkEvent::PeerConnected(peer.clone()))
        .await
        .ok();
    let announcement = sign_announcement(&handle, local_record(&handle))?;
    tx.send(Message::Announcement(announcement))
        .await
        .map_err(|_| NetworkError::NotConnected)?;
    if protocol_minor >= 1 {
        tx.send(Message::AddressUpdate(sign_address_update(
            &handle,
            address_records(&handle).await,
            handle.address_sequence.fetch_add(1, Ordering::Relaxed),
        )?))
        .await
        .map_err(|_| NetworkError::NotConnected)?;
        if let Ok(observation) =
            sign_address_observation(&handle, peer.node_id, connection.remote_address())
        {
            tx.send(Message::AddressObservation(observation))
                .await
                .map_err(|_| NetworkError::NotConnected)?;
        }
        if let Some(rotation) = handle.identity.rotation().cloned() {
            tx.send(Message::KeyRotation(rotation))
                .await
                .map_err(|_| NetworkError::NotConnected)?;
        }
    }
    let writer_handle = handle.clone();
    let peer_id = peer.node_id;
    let writer_metrics = metrics.clone();
    let writer = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if let Err(error) = write_message(
                &mut send,
                &message,
                writer_handle.config.max_frame_size,
                protocol_minor,
                &writer_metrics,
            )
            .await
            {
                tracing::debug!(peer = %peer_id, error = %error, "peer writer stopped");
                break;
            }
        }
        let _ = send.finish();
    });
    loop {
        let mut message =
            match read_message(&mut recv, handle.config.max_frame_size, &metrics).await {
                Ok(message) => message,
                Err(error) => {
                    metrics.protocol_errors.fetch_add(1, Ordering::Relaxed);
                    handle
                        .events
                        .send(NetworkEvent::ProtocolError {
                            peer: Some(peer.node_id),
                            message: error.to_string(),
                        })
                        .await
                        .ok();
                    break;
                }
            };
        let mut message_peer = peer.node_id;
        if let Message::RelayEnvelope(envelope) = message.clone() {
            match handle_relay_envelope(&handle, peer.node_id, envelope.clone()).await {
                Ok(Some(inner)) => {
                    message = inner;
                    message_peer = envelope.origin;
                }
                Ok(None) => continue,
                Err(error) => {
                    metrics.protocol_errors.fetch_add(1, Ordering::Relaxed);
                    handle
                        .events
                        .send(NetworkEvent::ProtocolError {
                            peer: Some(peer.node_id),
                            message: error.to_string(),
                        })
                        .await
                        .ok();
                    break;
                }
            }
        }
        match &message {
            Message::DhtRequest(request) if protocol_minor >= 2 => {
                if let Err(error) = handle
                    .handle_dht_request(message_peer, request.clone())
                    .await
                {
                    metrics.protocol_errors.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(peer = %message_peer, error = %error, "DHT request rejected");
                }
                continue;
            }
            Message::DhtResponse(response) if protocol_minor >= 2 => {
                if let Err(error) = handle
                    .handle_dht_response(message_peer, response.clone())
                    .await
                {
                    metrics.protocol_errors.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(peer = %message_peer, error = %error, "DHT response rejected");
                }
                continue;
            }
            Message::PeerExchange(exchange) => {
                for item in &exchange.peers {
                    if verify_announcement(item) && item.record.node_id != handle.node_id() {
                        upsert_peer(&handle, item.clone()).await?;
                    }
                }
            }
            Message::Announcement(item) => {
                if verify_announcement(item) && item.record.node_id != handle.node_id() {
                    upsert_peer(&handle, item.clone()).await?;
                }
            }
            Message::AddressUpdate(update) => {
                if verify_address_update(&handle, update).await {
                    upsert_address_update(&handle, update.clone()).await?;
                }
            }
            Message::AddressObservation(observation) => {
                if verify_address_observation(&handle, peer.node_id, observation).await {
                    record_observation(&handle, observation.clone()).await;
                }
            }
            Message::KeyRotation(rotation) => {
                if verify_key_rotation(rotation) && rotation.new_node_id == message_peer {
                    upsert_key_rotation(&handle, rotation.clone()).await?;
                }
            }
            Message::Ping(ping) => {
                tx.send(Message::Pong(ping.clone()))
                    .await
                    .map_err(|_| NetworkError::NotConnected)?;
            }
            Message::JobRequest(request) if request.origin != message_peer => {
                tx.send(Message::Error(ProtocolError {
                    code: intelligence_protocol::ErrorCode::Unauthorized,
                    message: "job origin does not match authenticated peer".to_string(),
                }))
                .await
                .map_err(|_| NetworkError::NotConnected)?;
                continue;
            }
            Message::Goodbye { .. } => break,
            _ => {}
        }
        handle
            .events
            .send(NetworkEvent::Message {
                peer: message_peer,
                message,
            })
            .await
            .ok();
    }
    writer.abort();
    let mut peers = handle.peers.lock().await;
    let current_connection = peers.connection_ids.get(&peer.node_id).copied();
    let is_current = current_connection == Some(connection_id);
    if is_current {
        peers.senders.remove(&peer.node_id);
        peers.connections.remove(&peer.node_id);
        peers.connection_ids.remove(&peer.node_id);
        peers.connection_directions.remove(&peer.node_id);
        peers.versions.remove(&peer.node_id);
    }
    tracing::debug!(
        peer = %peer.node_id,
        connection_id,
        is_current,
        current_connection_id = ?current_connection,
        "cleaning up peer connection"
    );
    drop(peers);
    if is_current {
        handle
            .relay_routes
            .lock()
            .await
            .retain(|target, relay| *target != peer.node_id && *relay != peer.node_id);
        handle
            .relay_session_ids
            .lock()
            .await
            .retain(|(relay, target), _| *relay != peer.node_id && *target != peer.node_id);
        handle
            .events
            .send(NetworkEvent::PeerDisconnected(peer.node_id))
            .await
            .ok();
    }
    Ok(())
}

async fn verify_hello(handle: &NetworkHandle, hello: &Hello) -> Result<u16, NetworkError> {
    hello
        .validate()
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    if hello.record.node_id == handle.node_id()
        || hello.supported.major != PROTOCOL_MAJOR
        || hello.supported.min_minor > PROTOCOL_MINOR
        || hello.supported.min_minor > hello.supported.max_minor
        || hello.supported.max_minor < VersionRange::current().min_minor
        || hello.record.node_id != NodeId::from_public_key(&hello.record.public_key)
        || !record_time_valid(handle, &hello.record)
    {
        return Err(NetworkError::IdentityVerification);
    }
    let bytes = postcard::to_allocvec(&(hello.nonce, hello.supported, &hello.record))
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    if !Identity::verify(&hello.record.public_key, &bytes, &hello.signature) {
        return Err(NetworkError::IdentityVerification);
    }
    let mut seen = handle.seen_hellos.lock().await;
    if !seen.insert((hello.record.node_id, hello.nonce)) {
        return Err(NetworkError::PeerProtocol(
            "replayed hello nonce".to_string(),
        ));
    }
    if seen.len() > 4096 {
        seen.clear();
    }
    Ok(PROTOCOL_MINOR.min(hello.supported.max_minor))
}

async fn handle_relay_envelope(
    handle: &NetworkHandle,
    authenticated_peer: NodeId,
    envelope: RelayEnvelope,
) -> Result<Option<Message>, NetworkError> {
    envelope
        .validate()
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    let bytes = relay_signing_bytes(
        envelope.session_id,
        envelope.origin,
        envelope.target,
        &envelope.payload,
    )?;
    if !Identity::verify(&envelope.origin_public_key, &bytes, &envelope.signature) {
        return Err(NetworkError::IdentityVerification);
    }
    if envelope.target == handle.node_id() {
        tracing::debug!(
            local = %handle.node_id(),
            origin = %envelope.origin,
            relay = %authenticated_peer,
            "relay envelope delivered"
        );
        if authenticated_peer != envelope.origin {
            handle
                .relay_routes
                .lock()
                .await
                .insert(envelope.origin, authenticated_peer);
        }
        let inner: Message = postcard::from_bytes(&envelope.payload)
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        inner
            .validate()
            .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
        if matches!(inner, Message::RelayEnvelope(_)) {
            return Err(NetworkError::PeerProtocol(
                "nested relay envelopes are not accepted".to_string(),
            ));
        }
        return Ok(Some(inner));
    }
    if !handle.config.relay_enabled {
        return Err(NetworkError::PeerProtocol(
            "peer is not configured as a relay".to_string(),
        ));
    }
    tracing::debug!(
        relay = %handle.node_id(),
        origin = %envelope.origin,
        target = %envelope.target,
        "relay envelope forwarding"
    );
    {
        let mut sessions = handle.relay_sessions.lock().await;
        let now = now_secs();
        sessions.retain(|_, session| now.saturating_sub(session.last_activity) <= 60);
        if let Some(session) = sessions.get_mut(&envelope.session_id) {
            if session.origin != envelope.origin || session.target != envelope.target {
                return Err(NetworkError::PeerProtocol(
                    "relay session identity changed".to_string(),
                ));
            }
            session.bytes = session.bytes.saturating_add(envelope.payload.len() as u64);
            session.last_activity = now;
            if session.bytes > handle.config.relay_max_bytes {
                return Err(NetworkError::PeerProtocol(
                    "relay session byte limit exceeded".to_string(),
                ));
            }
        } else {
            if sessions.len() >= handle.config.relay_max_sessions {
                return Err(NetworkError::PeerProtocol(
                    "relay session limit exceeded".to_string(),
                ));
            }
            sessions.insert(
                envelope.session_id,
                RelaySession {
                    origin: envelope.origin,
                    target: envelope.target,
                    bytes: envelope.payload.len() as u64,
                    last_activity: now,
                },
            );
        }
    }
    let forward = handle.clone();
    tokio::spawn(async move {
        let sender = forward
            .peers
            .lock()
            .await
            .senders
            .get(&envelope.target)
            .cloned();
        if let Some(sender) = sender {
            if let Err(error) = sender.send(Message::RelayEnvelope(envelope)).await {
                tracing::debug!(error = %error, "relay could not write to target");
            }
        } else {
            tracing::debug!(target = %envelope.target, "relay has no direct target connection");
        }
    });
    Ok(None)
}

fn verify_announcement(announcement: &SignedAnnouncement) -> bool {
    if announcement.record.node_id != NodeId::from_public_key(&announcement.record.public_key) {
        return false;
    }
    let Ok(bytes) = postcard::to_allocvec(&announcement.record) else {
        return false;
    };
    Identity::verify(
        &announcement.record.public_key,
        &bytes,
        &announcement.signature,
    )
}

async fn upsert_peer(
    handle: &NetworkHandle,
    announcement: SignedAnnouncement,
) -> Result<(), NetworkError> {
    let mut record = announcement.record.clone();
    if record.node_id == handle.node_id() {
        return Ok(());
    }
    record
        .validate()
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    if record.expires_at < now_secs() {
        return Ok(());
    }
    if !record_time_valid(handle, &record) {
        return Err(NetworkError::PeerProtocol(
            "peer announcement has an invalid lifetime".to_string(),
        ));
    }
    if record
        .addresses
        .iter()
        .any(|address| match address.parse::<SocketAddr>() {
            Ok(parsed) => !address_allowed(&handle.config, parsed, AddressSource::Learned),
            Err(_) => true,
        })
    {
        return Ok(());
    }
    record.addresses = dedupe_addresses(record.addresses);
    let mut peers = handle.peers.lock().await;
    if peers
        .records
        .get(&record.node_id)
        .is_some_and(|previous| !record_is_fresher(previous, &record))
    {
        return Ok(());
    }
    if !peers.records.contains_key(&record.node_id) && peers.records.len() >= MAX_LEARNED_PEERS {
        let now = now_secs();
        peers.records.retain(|_, value| value.expires_at >= now);
        let live_ids = peers.records.keys().copied().collect::<HashSet<_>>();
        peers
            .announcements
            .retain(|node_id, _| live_ids.contains(node_id));
        if peers.records.len() >= MAX_LEARNED_PEERS {
            return Err(NetworkError::PeerTableFull);
        }
    }
    let changed = peers
        .announcements
        .get(&record.node_id)
        .is_none_or(|previous| previous != &announcement);
    peers.records.insert(record.node_id, record);
    peers
        .announcements
        .insert(announcement.record.node_id, announcement.clone());
    let records = peers.records.values().cloned().collect::<Vec<_>>();
    let senders = if changed {
        peers.senders.values().cloned().collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    drop(peers);
    handle.store.save_peer_records(&records)?;
    if handle.config.dht_enabled {
        let authenticated = handle
            .peers
            .lock()
            .await
            .senders
            .contains_key(&announcement.record.node_id);
        let mut dht = handle.dht.lock().await;
        dht.table.insert(announcement.clone(), now_secs());
        // Bootstrap is a connectivity hint, not a permanent trust root.
        // Protect only an authenticated, currently connected contact and let
        // its signed expiry remove that protection later.
        if authenticated {
            dht.table.protect(announcement.clone(), now_secs());
        }
    }
    if changed {
        for sender in senders {
            let _ = sender.try_send(Message::Announcement(announcement.clone()));
        }
    }
    Ok(())
}

fn local_record(handle: &NetworkHandle) -> PeerRecord {
    let now = now_secs();
    PeerRecord {
        node_id: handle.identity.node_id(),
        public_key: handle.identity.public_key(),
        addresses: vec![handle.config.advertise_addr.clone()],
        capabilities: handle.config.capabilities.clone(),
        announced_at: now,
        expires_at: now.saturating_add(handle.config.peer_ttl_seconds),
        observed_latency_ms: None,
    }
}

async fn address_records(handle: &NetworkHandle) -> Vec<AddressRecord> {
    let now = now_secs();
    let mut records = Vec::new();
    if let Ok(address) = handle.config.advertise_addr.parse::<SocketAddr>()
        && address_allowed(&handle.config, address, AddressSource::Configured)
    {
        records.push(AddressRecord {
            address: address.to_string(),
            transport: AddressTransport::QuicUdp,
            source: AddressSource::Configured,
            expires_at: now.saturating_add(handle.config.peer_ttl_seconds),
            confidence: 100,
            relay: None,
        });
    }
    for observation in handle.observed_addresses.lock().await.iter() {
        if observation.expires_at >= now {
            records.push(AddressRecord {
                address: observation.address.clone(),
                transport: AddressTransport::QuicUdp,
                source: AddressSource::Observed,
                expires_at: observation.expires_at,
                confidence: 60,
                relay: None,
            });
        }
    }
    records.sort_by_key(|record| (record.source as u8, std::cmp::Reverse(record.confidence)));
    records.truncate(intelligence_protocol::MAX_ADDRESSES);
    records
}

fn dedupe_addresses(mut addresses: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    addresses.retain(|address| seen.insert(address.clone()));
    addresses.truncate(8);
    addresses
}

fn preferred_addresses(handle: &NetworkHandle, addresses: &[String]) -> Vec<String> {
    let mut values = addresses
        .iter()
        .filter_map(|address| {
            let parsed = address.parse::<SocketAddr>().ok()?;
            address_allowed(&handle.config, parsed, AddressSource::Learned)
                .then_some((address.clone(), address_is_public(parsed)))
        })
        .collect::<Vec<_>>();
    values.sort_by_key(|(_, global)| !*global);
    values.into_iter().map(|(address, _)| address).collect()
}

fn address_allowed(config: &NetworkConfig, address: SocketAddr, source: AddressSource) -> bool {
    if address.port() == 0 {
        return false;
    }
    if matches!(source, AddressSource::Configured | AddressSource::Local)
        || config.allow_private_addresses
    {
        return true;
    }
    let ip = address.ip();
    if ip.is_loopback() || ip.is_unspecified() || ip.is_multicast() {
        return false;
    }
    !match ip {
        std::net::IpAddr::V4(value) => value.is_private() || value.is_link_local(),
        std::net::IpAddr::V6(value) => value.is_unique_local() || value.is_unicast_link_local(),
    }
}

fn address_is_public(address: SocketAddr) -> bool {
    let ip = address.ip();
    !ip.is_loopback()
        && !ip.is_unspecified()
        && !ip.is_multicast()
        && !match ip {
            std::net::IpAddr::V4(value) => value.is_private() || value.is_link_local(),
            std::net::IpAddr::V6(value) => value.is_unique_local() || value.is_unicast_link_local(),
        }
}

fn address_signing_bytes(
    node_id: NodeId,
    sequence: u64,
    addresses: &[AddressRecord],
) -> Result<Vec<u8>, NetworkError> {
    postcard::to_allocvec(&(node_id, sequence, addresses))
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))
}

fn relay_signing_bytes(
    session_id: RequestId,
    origin: NodeId,
    target: NodeId,
    payload: &[u8],
) -> Result<Vec<u8>, NetworkError> {
    postcard::to_allocvec(&(session_id, origin, target, payload))
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))
}

fn dht_record_signing_bytes(record: &DhtRecord) -> Result<Vec<u8>, NetworkError> {
    postcard::to_allocvec(&(
        record.namespace,
        record.key,
        record.owner,
        record.owner_public_key,
        record.sequence,
        record.expires_at,
        &record.value,
    ))
    .map_err(|error| NetworkError::PeerProtocol(error.to_string()))
}

fn verify_dht_record(record: &DhtRecord) -> bool {
    if record.validate().is_err() {
        return false;
    }
    let Ok(bytes) = dht_record_signing_bytes(record) else {
        return false;
    };
    Identity::verify(&record.owner_public_key, &bytes, &record.signature)
}

fn dht_expiry_valid(record: &DhtRecord, now: u64, peer_ttl_seconds: u64) -> bool {
    record.expires_at >= now
        && record.expires_at.saturating_sub(now) <= peer_ttl_seconds.saturating_mul(2).max(60)
}

fn message_required_minor(message: &Message) -> u16 {
    if message.kind() >= 22 {
        4
    } else if message.kind() >= 21 {
        3
    } else if message.kind() >= 19 {
        2
    } else if message.kind() >= 13 {
        1
    } else {
        0
    }
}

fn sign_address_update(
    handle: &NetworkHandle,
    addresses: Vec<AddressRecord>,
    sequence: u64,
) -> Result<AddressUpdate, NetworkError> {
    let node_id = handle.node_id();
    let bytes = address_signing_bytes(node_id, sequence, &addresses)?;
    Ok(AddressUpdate {
        node_id,
        sequence,
        addresses,
        signature: handle.identity.sign(&bytes),
    })
}

fn address_observation_signing_bytes(
    subject: NodeId,
    address: &str,
    observed_by: NodeId,
    expires_at: u64,
    nonce: [u8; 16],
) -> Result<Vec<u8>, NetworkError> {
    postcard::to_allocvec(&(subject, address, observed_by, expires_at, nonce))
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))
}

fn sign_address_observation(
    handle: &NetworkHandle,
    subject: NodeId,
    address: SocketAddr,
) -> Result<AddressObservation, NetworkError> {
    let expires_at = now_secs().saturating_add(handle.config.peer_ttl_seconds);
    let mut nonce = [0u8; 16];
    OsRng.fill_bytes(&mut nonce);
    let address = address.to_string();
    let bytes =
        address_observation_signing_bytes(subject, &address, handle.node_id(), expires_at, nonce)?;
    Ok(AddressObservation {
        subject,
        address,
        observed_by: handle.node_id(),
        expires_at,
        nonce,
        signature: handle.identity.sign(&bytes),
    })
}

async fn verify_address_update(handle: &NetworkHandle, update: &AddressUpdate) -> bool {
    if update.validate().is_err() || update.node_id == handle.node_id() {
        return false;
    }
    let public_key = handle
        .peers
        .lock()
        .await
        .records
        .get(&update.node_id)
        .map(|record| record.public_key);
    let Some(public_key) = public_key else {
        return false;
    };
    let Ok(bytes) = address_signing_bytes(update.node_id, update.sequence, &update.addresses)
    else {
        return false;
    };
    Identity::verify(&public_key, &bytes, &update.signature)
}

async fn upsert_address_update(
    handle: &NetworkHandle,
    update: AddressUpdate,
) -> Result<(), NetworkError> {
    let mut peers = handle.peers.lock().await;
    if peers
        .address_updates
        .get(&update.node_id)
        .is_some_and(|previous| previous.sequence >= update.sequence)
    {
        return Ok(());
    }
    let addresses = dedupe_addresses(
        update
            .addresses
            .iter()
            .filter(|address| address.expires_at >= now_secs())
            .filter_map(|address| {
                let parsed = address.address.parse::<SocketAddr>().ok()?;
                address_allowed(&handle.config, parsed, AddressSource::Learned)
                    .then_some(address.address.clone())
            })
            .collect(),
    );
    if let Some(record) = peers.records.get_mut(&update.node_id) {
        record.addresses = addresses;
    }
    peers.address_updates.insert(update.node_id, update);
    let records = peers.records.values().cloned().collect::<Vec<_>>();
    drop(peers);
    handle.store.save_peer_records(&records)?;
    Ok(())
}

async fn verify_address_observation(
    handle: &NetworkHandle,
    authenticated_peer: NodeId,
    observation: &AddressObservation,
) -> bool {
    if observation.validate().is_err()
        || observation.observed_by != authenticated_peer
        || observation.subject != handle.node_id()
    {
        return false;
    }
    let public_key = handle
        .peers
        .lock()
        .await
        .records
        .get(&authenticated_peer)
        .map(|record| record.public_key);
    let Some(public_key) = public_key else {
        return false;
    };
    let Ok(bytes) = address_observation_signing_bytes(
        observation.subject,
        &observation.address,
        observation.observed_by,
        observation.expires_at,
        observation.nonce,
    ) else {
        return false;
    };
    Identity::verify(&public_key, &bytes, &observation.signature)
}

async fn record_observation(handle: &NetworkHandle, observation: AddressObservation) {
    let mut observations = handle.observed_addresses.lock().await;
    observations.retain(|item| item.expires_at >= now_secs() && item.nonce != observation.nonce);
    observations.push(observation);
    observations.sort_by_key(|item| item.expires_at);
    if observations.len() > 16 {
        let excess = observations.len().saturating_sub(16);
        observations.drain(..excess);
    }
}

fn verify_key_rotation(rotation: &KeyRotation) -> bool {
    if rotation.validate().is_err()
        || rotation.valid_from > now_secs().saturating_add(60)
        || rotation.valid_until < now_secs()
    {
        return false;
    }
    let Ok(bytes) = rotation_signing_bytes(
        rotation.old_node_id,
        rotation.old_public_key,
        rotation.new_node_id,
        rotation.new_public_key,
        rotation.sequence,
        rotation.valid_from,
        rotation.valid_until,
    ) else {
        return false;
    };
    Identity::verify(&rotation.old_public_key, &bytes, &rotation.old_signature)
        && Identity::verify(&rotation.new_public_key, &bytes, &rotation.new_signature)
}

async fn upsert_key_rotation(
    handle: &NetworkHandle,
    rotation: KeyRotation,
) -> Result<(), NetworkError> {
    let mut peers = handle.peers.lock().await;
    if let Some(old_record) = peers.records.get(&rotation.old_node_id)
        && old_record.public_key != rotation.old_public_key
    {
        return Err(NetworkError::IdentityVerification);
    }
    if peers.rotations.values().any(|previous| {
        previous.old_node_id == rotation.old_node_id
            && previous.sequence >= rotation.sequence
            && previous.new_node_id != rotation.new_node_id
    }) {
        return Ok(());
    }
    if peers
        .rotations
        .get(&rotation.new_node_id)
        .is_some_and(|previous| previous.sequence >= rotation.sequence)
    {
        return Ok(());
    }
    peers
        .rotations
        .insert(rotation.new_node_id, rotation.clone());
    let rotations = peers.rotations.values().cloned().collect::<Vec<_>>();
    let senders = peers
        .senders
        .iter()
        .filter(|(node_id, _)| peers.versions.get(node_id).copied().unwrap_or(0) >= 1)
        .map(|(_, sender)| sender.clone())
        .collect::<Vec<_>>();
    drop(peers);
    handle.store.save_key_rotations(&rotations)?;
    for sender in senders {
        let _ = sender.try_send(Message::KeyRotation(rotation.clone()));
    }
    Ok(())
}

fn record_time_valid(handle: &NetworkHandle, record: &PeerRecord) -> bool {
    let now = now_secs();
    record.announced_at <= now.saturating_add(60)
        && record.expires_at >= now
        && record.expires_at.saturating_sub(record.announced_at)
            <= handle.config.peer_ttl_seconds.saturating_mul(2)
}

fn record_is_fresher(previous: &PeerRecord, next: &PeerRecord) -> bool {
    next.announced_at > previous.announced_at
        || (next.announced_at == previous.announced_at && next.expires_at >= previous.expires_at)
}

fn make_hello(handle: &NetworkHandle) -> Result<Hello, NetworkError> {
    let mut nonce = [0u8; 16];
    OsRng.fill_bytes(&mut nonce);
    let supported = VersionRange::current();
    let record = local_record(handle);
    let bytes = postcard::to_allocvec(&(nonce, supported, &record))
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    Ok(Hello {
        nonce,
        supported,
        record,
        signature: handle.identity.sign(&bytes),
    })
}

fn sign_announcement(
    handle: &NetworkHandle,
    record: PeerRecord,
) -> Result<SignedAnnouncement, NetworkError> {
    let bytes = postcard::to_allocvec(&record)
        .map_err(|error| NetworkError::PeerProtocol(error.to_string()))?;
    Ok(SignedAnnouncement {
        record,
        signature: handle.identity.sign(&bytes),
    })
}

async fn write_message(
    stream: &mut SendStream,
    message: &Message,
    max_frame_size: usize,
    protocol_minor: u16,
    metrics: &NetworkMetrics,
) -> Result<(), NetworkError> {
    let frame = encode_message_at_version(message, max_frame_size, PROTOCOL_MAJOR, protocol_minor)?;
    stream
        .write_all(&frame)
        .await
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    metrics
        .bytes_sent
        .fetch_add(frame.len() as u64, Ordering::Relaxed);
    Ok(())
}

async fn read_message(
    stream: &mut RecvStream,
    max_frame_size: usize,
    metrics: &NetworkMetrics,
) -> Result<Message, NetworkError> {
    let mut header_bytes = [0u8; FRAME_HEADER_SIZE];
    stream
        .read_exact(&mut header_bytes)
        .await
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    let header = FrameHeader::decode(&header_bytes, max_frame_size)?;
    let mut frame = Vec::with_capacity(FRAME_HEADER_SIZE + header.payload_len as usize);
    frame.extend_from_slice(&header_bytes);
    let mut payload = vec![0u8; header.payload_len as usize];
    stream
        .read_exact(&mut payload)
        .await
        .map_err(|error| NetworkError::Quic(error.to_string()))?;
    frame.extend_from_slice(&payload);
    metrics
        .bytes_received
        .fetch_add(frame.len() as u64, Ordering::Relaxed);
    decode_frame(&frame, max_frame_size).map_err(NetworkError::Codec)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn random_request_id() -> RequestId {
    let mut bytes = [0u8; 16];
    OsRng.fill_bytes(&mut bytes);
    RequestId::from_bytes(bytes)
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
    DEFAULT_K
}

fn default_dht_alpha() -> usize {
    3
}

fn default_dht_max_records() -> usize {
    DEFAULT_MAX_RECORDS
}

fn transport_configs() -> Result<(ServerConfig, ClientConfig), String> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let certificate = generate_simple_self_signed(vec!["intelligence-network".to_string()])
        .map_err(|error| error.to_string())?;
    let certificate_der = CertificateDer::from(certificate.cert.der().to_vec());
    let private_key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        certificate.key_pair.serialize_der(),
    ));
    let mut server_crypto = RustlsServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![certificate_der], private_key)
        .map_err(|error| error.to_string())?;
    server_crypto.alpn_protocols = vec![b"intelligence/1".to_vec()];
    let mut transport = TransportConfig::default();
    transport.max_concurrent_bidi_streams(VarInt::from_u32(1));
    transport.max_concurrent_uni_streams(VarInt::from_u32(0));
    transport.max_idle_timeout(Some(
        Duration::from_secs(30)
            .try_into()
            .map_err(|_| "invalid idle timeout".to_string())?,
    ));
    transport.keep_alive_interval(Some(Duration::from_secs(10)));
    let quic_server =
        QuicServerConfig::try_from(server_crypto).map_err(|error| error.to_string())?;
    let mut server = ServerConfig::with_crypto(Arc::new(quic_server));
    server.transport_config(Arc::new(transport));
    let mut client_crypto = RustlsClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification))
        .with_no_client_auth();
    client_crypto.alpn_protocols = vec![b"intelligence/1".to_vec()];
    let quic_client =
        QuicClientConfig::try_from(client_crypto).map_err(|error| error.to_string())?;
    Ok((server, ClientConfig::new(Arc::new(quic_client))))
}

#[derive(Debug)]
struct SkipServerVerification;

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, RustlsError> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, RustlsError> {
        Ok(HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ED25519,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::RSA_PSS_SHA256,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_persistent_and_derives_node_id() {
        let path =
            std::env::temp_dir().join(format!("intelligence-identity-{}", std::process::id()));
        let first = Identity::load_or_generate(&path).unwrap();
        let second = Identity::load_or_generate(&path).unwrap();
        assert_eq!(first.node_id(), second.node_id());
        assert_eq!(first.public_key(), second.public_key());
        let message = b"signed record";
        let signature = first.sign(message);
        assert!(Identity::verify(&first.public_key(), message, &signature));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        let _ = fs::remove_file(path);
    }

    #[test]
    fn protocol_key_rotation_is_persisted_and_verifiable() {
        let root = std::env::temp_dir().join(format!(
            "intelligence-rotation-{}-{}",
            std::process::id(),
            now_secs()
        ));
        fs::create_dir_all(&root).unwrap();
        let old_path = root.join("old.key");
        let new_path = root.join("new.key");
        let old = Identity::load_or_generate(&old_path).unwrap();
        let rotation =
            Identity::rotate(&old_path, &new_path, 1, now_secs().saturating_add(3600)).unwrap();
        assert_eq!(rotation.old_node_id, old.node_id());
        let new = Identity::load_or_generate(&new_path).unwrap();
        assert_eq!(new.rotation(), Some(&rotation));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn expired_key_rotation_is_not_accepted() {
        let root = std::env::temp_dir().join(format!(
            "intelligence-expired-rotation-{}-{}",
            std::process::id(),
            now_secs()
        ));
        fs::create_dir_all(&root).unwrap();
        let old_path = root.join("old.key");
        let new_path = root.join("new.key");
        let old = Identity::load_or_generate(&old_path).unwrap();
        let new = Identity::load_or_generate(&new_path).unwrap();
        let valid_from = now_secs().saturating_sub(7200);
        let valid_until = now_secs().saturating_sub(3600);
        let signing_bytes = rotation_signing_bytes(
            old.node_id(),
            old.public_key(),
            new.node_id(),
            new.public_key(),
            1,
            valid_from,
            valid_until,
        )
        .unwrap();
        let expired = KeyRotation {
            old_node_id: old.node_id(),
            old_public_key: old.public_key(),
            new_node_id: new.node_id(),
            new_public_key: new.public_key(),
            sequence: 1,
            valid_from,
            valid_until,
            old_signature: old.sign(&signing_bytes),
            new_signature: new.sign(&signing_bytes),
        };
        assert!(!verify_key_rotation(&expired));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn learned_addresses_reject_private_and_invalid_destinations_by_default() {
        let config = NetworkConfig::default();
        assert!(!address_allowed(
            &config,
            "127.0.0.1:4000".parse().unwrap(),
            AddressSource::Learned
        ));
        assert!(!address_allowed(
            &config,
            "10.0.0.10:4000".parse().unwrap(),
            AddressSource::Learned
        ));
        assert!(!address_allowed(
            &config,
            "198.51.100.10:0".parse().unwrap(),
            AddressSource::Learned
        ));
        assert!(address_allowed(
            &config,
            "198.51.100.10:4000".parse().unwrap(),
            AddressSource::Configured
        ));
    }

    #[test]
    fn address_candidates_are_deduplicated_and_bounded() {
        let values = (0..32)
            .map(|index| format!("198.51.100.{}:4000", index % 8 + 1))
            .collect::<Vec<_>>();
        let deduplicated = dedupe_addresses(values);
        assert_eq!(deduplicated.len(), 8);
        assert_eq!(deduplicated[0], "198.51.100.1:4000");
    }

    #[test]
    fn peer_records_do_not_roll_back_in_freshness() {
        let previous = PeerRecord {
            node_id: NodeId::from_bytes([1; 32]),
            public_key: [2; 32],
            addresses: vec!["198.51.100.10:4000".to_string()],
            capabilities: Vec::new(),
            announced_at: 20,
            expires_at: 320,
            observed_latency_ms: None,
        };
        let mut older = previous.clone();
        older.announced_at = 19;
        assert!(!record_is_fresher(&previous, &older));
        let mut shorter = previous.clone();
        shorter.expires_at = 319;
        assert!(!record_is_fresher(&previous, &shorter));
        let mut newer = previous;
        newer.announced_at = 21;
        assert!(record_is_fresher(&shorter, &newer));
    }

    #[test]
    fn dht_records_require_owner_signature_and_reject_replay_material() {
        let path = std::env::temp_dir().join(format!(
            "intelligence-dht-signature-{}-{}",
            std::process::id(),
            now_secs()
        ));
        let identity = Identity::load_or_generate(&path).unwrap();
        let mut record = DhtRecord {
            namespace: DhtNamespace::Capability,
            key: DhtKey::for_name(DhtNamespace::Capability, "inference.text"),
            owner: identity.node_id(),
            owner_public_key: identity.public_key(),
            sequence: 1,
            expires_at: now_secs().saturating_add(60),
            value: b"provider".to_vec(),
            signature: Vec::new(),
        };
        record.signature = identity.sign(&dht_record_signing_bytes(&record).unwrap());
        assert!(verify_dht_record(&record));
        record.value = b"forged".to_vec();
        assert!(!verify_dht_record(&record));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn dht_expiry_is_bounded_to_peer_ttl_and_rejects_far_future_records() {
        let record = DhtRecord {
            namespace: DhtNamespace::Capability,
            key: DhtKey::from_bytes([3; 32]),
            owner: NodeId::from_bytes([4; 32]),
            owner_public_key: [5; 32],
            sequence: 1,
            expires_at: 1_600,
            value: vec![1],
            signature: vec![0; 64],
        };
        assert!(dht_expiry_valid(&record, 1_000, 300));
        let mut too_far = record.clone();
        too_far.expires_at = 2_000;
        assert!(!dht_expiry_valid(&too_far, 1_000, 300));
        let mut expired = record;
        expired.expires_at = 999;
        assert!(!dht_expiry_valid(&expired, 1_000, 300));
    }
}
