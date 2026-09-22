//! Local-first state and content-addressed artifact storage.

use intelligence_protocol::{
    ArtifactId, DhtRecord, JobId, JobState, KeyRotation, MAX_JOB_OUTPUT, NodeId, PeerRecord,
    SignedAnnouncement, SignedEvidence,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;

const MAX_STATE_BYTES: u64 = 8 * 1024 * 1024;
const MAX_PERSISTED_JOBS: usize = 4096;
const MAX_PERSISTED_PEERS: usize = 2048;
const MAX_PERSISTED_ROTATIONS: usize = 2048;
const MAX_PERSISTED_DHT_RECORDS: usize = 4096;
const MAX_PERSISTED_DHT_CONTACTS: usize = 2048;
const MAX_PERSISTED_EVIDENCE: usize = 4096;
const MAX_QUARANTINE_FRACTION: u64 = 8;
pub const STATE_FORMAT_VERSION: u16 = 1;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Error)]
pub enum StorageError {
    #[error("storage I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("storage encoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("artifact {artifact} is {size} bytes, maximum is {max}")]
    ArtifactTooLarge {
        artifact: ArtifactId,
        size: u64,
        max: u64,
    },
    #[error("artifact {artifact} failed integrity verification")]
    Integrity { artifact: ArtifactId },
    #[error("storage quota exceeded: used {used}, requested {requested}, quota {quota}")]
    QuotaExceeded {
        used: u64,
        requested: u64,
        quota: u64,
    },
    #[error("invalid stored state: {0}")]
    InvalidState(String),
}

#[derive(Clone, Debug)]
pub struct LocalStore {
    root: PathBuf,
    quota_bytes: u64,
    max_artifact_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PersistedJob {
    pub job_id: JobId,
    pub origin: NodeId,
    pub state: JobState,
    pub updated_at: u64,
    pub output_hash: Option<ArtifactId>,
    pub output: Option<Vec<u8>>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PeerState {
    #[serde(default)]
    version: u16,
    peers: Vec<PeerRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RotationState {
    #[serde(default)]
    version: u16,
    rotations: Vec<KeyRotation>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct JobStateFile {
    #[serde(default)]
    version: u16,
    jobs: Vec<PersistedJob>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DhtState {
    #[serde(default)]
    version: u16,
    records: Vec<DhtRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DhtContactState {
    #[serde(default)]
    version: u16,
    contacts: Vec<SignedAnnouncement>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvidenceState {
    #[serde(default)]
    version: u16,
    evidence: Vec<SignedEvidence>,
}

impl LocalStore {
    pub fn open(
        root: impl AsRef<Path>,
        quota_bytes: u64,
        max_artifact_bytes: u64,
    ) -> Result<Self, StorageError> {
        if quota_bytes == 0 || max_artifact_bytes == 0 || max_artifact_bytes > quota_bytes {
            return Err(StorageError::InvalidState(
                "storage quota and artifact limit must be positive and ordered".to_string(),
            ));
        }
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(root.join("artifacts"))?;
        fs::create_dir_all(root.join("quarantine"))?;
        fs::create_dir_all(root.join("state"))?;
        fs::create_dir_all(root.join("state").join("transfers"))?;
        let store = Self {
            root,
            quota_bytes,
            max_artifact_bytes,
        };
        store.garbage_collect()?;
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn quota_bytes(&self) -> u64 {
        self.quota_bytes
    }

    pub fn used_bytes(&self) -> Result<u64, StorageError> {
        fn walk(path: &Path) -> io::Result<u64> {
            let mut total: u64 = 0;
            for entry in fs::read_dir(path)? {
                let entry = entry?;
                let metadata = entry.metadata()?;
                if metadata.is_dir() {
                    total = total.saturating_add(walk(&entry.path())?);
                } else if metadata.is_file() {
                    total = total.saturating_add(metadata.len());
                }
            }
            Ok(total)
        }
        Ok(walk(&self.root)?)
    }

    pub fn put_artifact(&self, bytes: &[u8]) -> Result<ArtifactId, StorageError> {
        let artifact = ArtifactId::from_bytes_hashed(bytes);
        self.put_artifact_with_id(artifact, bytes)?;
        Ok(artifact)
    }

    pub fn import_artifact(
        &self,
        source: impl AsRef<Path>,
    ) -> Result<(ArtifactId, u64), StorageError> {
        let source = source.as_ref();
        let size = fs::metadata(source)?.len();
        if size > self.max_artifact_bytes {
            return Err(StorageError::ArtifactTooLarge {
                artifact: ArtifactId::default(),
                size,
                max: self.max_artifact_bytes,
            });
        }
        let used = self.used_bytes()?;
        if used.saturating_add(size) > self.quota_bytes {
            return Err(StorageError::QuotaExceeded {
                used,
                requested: size,
                quota: self.quota_bytes,
            });
        }
        let temp = self.temp_path("artifact-import");
        let mut input = File::open(source)?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp)?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = input.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            output.write_all(&buffer[..read])?;
        }
        output.sync_all()?;
        let artifact = ArtifactId::from_bytes(*hasher.finalize().as_bytes());
        let destination = self.artifact_path(artifact);
        if destination.exists() {
            fs::remove_file(temp)?;
            self.verify_artifact(artifact)?;
        } else if let Err(error) = fs::rename(&temp, destination) {
            let _ = fs::remove_file(&temp);
            return Err(StorageError::Io(error));
        }
        Ok((artifact, size))
    }

    pub fn put_artifact_with_id(
        &self,
        artifact: ArtifactId,
        bytes: &[u8],
    ) -> Result<(), StorageError> {
        if bytes.len() as u64 > self.max_artifact_bytes {
            return Err(StorageError::ArtifactTooLarge {
                artifact,
                size: bytes.len() as u64,
                max: self.max_artifact_bytes,
            });
        }
        if ArtifactId::from_bytes_hashed(bytes) != artifact {
            return Err(StorageError::Integrity { artifact });
        }
        let path = self.artifact_path(artifact);
        if path.exists() {
            return self.verify_artifact(artifact);
        }
        let used = self.used_bytes()?;
        if used.saturating_add(bytes.len() as u64) > self.quota_bytes {
            return Err(StorageError::QuotaExceeded {
                used,
                requested: bytes.len() as u64,
                quota: self.quota_bytes,
            });
        }
        let temp = self.temp_path("artifact");
        {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
        }
        if let Err(error) = fs::rename(&temp, &path) {
            let _ = fs::remove_file(&temp);
            if path.exists() {
                return self.verify_artifact(artifact);
            }
            return Err(StorageError::Io(error));
        }
        Ok(())
    }

    pub fn get_artifact(&self, artifact: ArtifactId) -> Result<Vec<u8>, StorageError> {
        let path = self.artifact_path(artifact);
        let metadata = fs::metadata(&path)?;
        if metadata.len() > self.max_artifact_bytes {
            self.quarantine(artifact, &path)?;
            return Err(StorageError::ArtifactTooLarge {
                artifact,
                size: metadata.len(),
                max: self.max_artifact_bytes,
            });
        }
        let mut bytes = Vec::with_capacity(metadata.len() as usize);
        File::open(&path)?.read_to_end(&mut bytes)?;
        if ArtifactId::from_bytes_hashed(&bytes) != artifact {
            self.quarantine(artifact, &path)?;
            return Err(StorageError::Integrity { artifact });
        }
        Ok(bytes)
    }

    pub fn has_artifact(&self, artifact: ArtifactId) -> bool {
        self.artifact_path(artifact).is_file()
    }

    pub fn artifact_size(&self, artifact: ArtifactId) -> Result<u64, StorageError> {
        let metadata = fs::metadata(self.artifact_path(artifact))?;
        if metadata.len() > self.max_artifact_bytes {
            return Err(StorageError::ArtifactTooLarge {
                artifact,
                size: metadata.len(),
                max: self.max_artifact_bytes,
            });
        }
        Ok(metadata.len())
    }

    pub fn read_artifact_range(
        &self,
        artifact: ArtifactId,
        offset: u64,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, u64), StorageError> {
        if max_bytes == 0 {
            return Err(StorageError::InvalidState(
                "artifact range must request at least one byte".to_string(),
            ));
        }
        let size = self.artifact_size(artifact)?;
        if offset > size {
            return Err(StorageError::InvalidState(
                "artifact range starts beyond the artifact".to_string(),
            ));
        }
        let length = (size.saturating_sub(offset) as usize).min(max_bytes);
        let mut file = File::open(self.artifact_path(artifact))?;
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = vec![0u8; length];
        file.read_exact(&mut bytes)?;
        Ok((bytes, size))
    }

    pub fn partial_artifact_size(&self, artifact: ArtifactId) -> Result<u64, StorageError> {
        match fs::metadata(self.partial_artifact_path(artifact)) {
            Ok(metadata) => {
                if metadata.len() > self.max_artifact_bytes {
                    return Err(StorageError::ArtifactTooLarge {
                        artifact,
                        size: metadata.len(),
                        max: self.max_artifact_bytes,
                    });
                }
                Ok(metadata.len())
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(StorageError::Io(error)),
        }
    }

    pub fn write_partial_artifact(
        &self,
        artifact: ArtifactId,
        expected_size: u64,
        offset: u64,
        bytes: &[u8],
    ) -> Result<u64, StorageError> {
        if expected_size == 0 || expected_size > self.max_artifact_bytes {
            return Err(StorageError::ArtifactTooLarge {
                artifact,
                size: expected_size,
                max: self.max_artifact_bytes,
            });
        }
        let current = self.partial_artifact_size(artifact)?;
        if bytes.is_empty()
            || offset != current
            || offset.saturating_add(bytes.len() as u64) > expected_size
        {
            return Err(StorageError::InvalidState(
                "partial artifact chunk is not the next contiguous range".to_string(),
            ));
        }
        let used = self.used_bytes()?;
        // The partial file is already included in `used`. Reserve the complete
        // final artifact against the remaining store rather than charging the
        // existing partial bytes a second time on every chunk.
        let base_used = used.saturating_sub(current);
        if base_used.saturating_add(expected_size) > self.quota_bytes {
            return Err(StorageError::QuotaExceeded {
                used: base_used,
                requested: expected_size,
                quota: self.quota_bytes,
            });
        }
        let path = self.partial_artifact_path(artifact);
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        file.write_all(bytes)?;
        file.sync_data()?;
        Ok(offset.saturating_add(bytes.len() as u64))
    }

    pub fn finalize_partial_artifact(
        &self,
        artifact: ArtifactId,
        expected_size: u64,
        expected_hash: ArtifactId,
    ) -> Result<(), StorageError> {
        let partial = self.partial_artifact_path(artifact);
        let metadata = fs::metadata(&partial)?;
        if metadata.len() != expected_size || expected_hash != artifact {
            return Err(StorageError::Integrity { artifact });
        }
        if hash_file(&partial)? != expected_hash {
            self.quarantine_partial(artifact, &partial)?;
            return Err(StorageError::Integrity { artifact });
        }
        let destination = self.artifact_path(artifact);
        if destination.exists() {
            fs::remove_file(partial)?;
            return self.verify_artifact(artifact);
        }
        fs::rename(partial, destination)?;
        Ok(())
    }

    pub fn discard_partial_artifact(&self, artifact: ArtifactId) -> Result<(), StorageError> {
        match fs::remove_file(self.partial_artifact_path(artifact)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(StorageError::Io(error)),
        }
    }

    pub fn verify_artifact(&self, artifact: ArtifactId) -> Result<(), StorageError> {
        let path = self.artifact_path(artifact);
        let metadata = fs::metadata(&path)?;
        if metadata.len() > self.max_artifact_bytes {
            self.quarantine(artifact, &path)?;
            return Err(StorageError::ArtifactTooLarge {
                artifact,
                size: metadata.len(),
                max: self.max_artifact_bytes,
            });
        }
        if hash_file(&path)? != artifact {
            self.quarantine(artifact, &path)?;
            return Err(StorageError::Integrity { artifact });
        }
        Ok(())
    }

    pub fn garbage_collect(&self) -> Result<u64, StorageError> {
        let quarantine = self.root.join("quarantine");
        let mut entries = Vec::new();
        let mut total = 0u64;
        for entry in fs::read_dir(&quarantine)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if !metadata.is_file() {
                continue;
            }
            total = total.saturating_add(metadata.len());
            entries.push((
                metadata.modified().unwrap_or(UNIX_EPOCH),
                entry.path(),
                metadata.len(),
            ));
        }
        let limit = self
            .quota_bytes
            .checked_div(MAX_QUARANTINE_FRACTION)
            .unwrap_or(0);
        entries.sort_by_key(|(modified, _, _)| *modified);
        let mut reclaimed = 0u64;
        for (_, path, size) in entries {
            if total <= limit {
                break;
            }
            fs::remove_file(path)?;
            total = total.saturating_sub(size);
            reclaimed = reclaimed.saturating_add(size);
        }
        Ok(reclaimed)
    }

    pub fn save_peer_records(&self, peers: &[PeerRecord]) -> Result<(), StorageError> {
        if peers.len() > MAX_PERSISTED_PEERS {
            return Err(StorageError::InvalidState(
                "peer state exceeds local bound".to_string(),
            ));
        }
        for peer in peers {
            peer.validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        self.write_json(
            "peers.json",
            &PeerState {
                version: STATE_FORMAT_VERSION,
                peers: peers.to_vec(),
            },
        )
    }

    pub fn load_peer_records(&self) -> Result<Vec<PeerRecord>, StorageError> {
        let state: PeerState = self.read_json_or_default(
            "peers.json",
            PeerState {
                version: STATE_FORMAT_VERSION,
                peers: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.peers.len() > MAX_PERSISTED_PEERS {
            return Err(StorageError::InvalidState(
                "peer state exceeds local bound".to_string(),
            ));
        }
        for peer in &state.peers {
            peer.validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        Ok(state.peers)
    }

    pub fn save_key_rotations(&self, rotations: &[KeyRotation]) -> Result<(), StorageError> {
        if rotations.len() > MAX_PERSISTED_ROTATIONS {
            return Err(StorageError::InvalidState(
                "key rotation state exceeds local bound".to_string(),
            ));
        }
        for rotation in rotations {
            rotation
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        self.write_json(
            "rotations.json",
            &RotationState {
                version: STATE_FORMAT_VERSION,
                rotations: rotations.to_vec(),
            },
        )
    }

    pub fn load_key_rotations(&self) -> Result<Vec<KeyRotation>, StorageError> {
        let state: RotationState = self.read_json_or_default(
            "rotations.json",
            RotationState {
                version: STATE_FORMAT_VERSION,
                rotations: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.rotations.len() > MAX_PERSISTED_ROTATIONS {
            return Err(StorageError::InvalidState(
                "key rotation state exceeds local bound".to_string(),
            ));
        }
        for rotation in &state.rotations {
            rotation
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        Ok(state.rotations)
    }

    pub fn save_dht_records(&self, records: &[DhtRecord]) -> Result<(), StorageError> {
        if records.len() > MAX_PERSISTED_DHT_RECORDS {
            return Err(StorageError::InvalidState(
                "DHT record state exceeds local bound".to_string(),
            ));
        }
        for record in records {
            record
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        self.write_json(
            "dht.json",
            &DhtState {
                version: STATE_FORMAT_VERSION,
                records: records.to_vec(),
            },
        )
    }

    pub fn load_dht_records(&self) -> Result<Vec<DhtRecord>, StorageError> {
        let state: DhtState = self.read_json_or_default(
            "dht.json",
            DhtState {
                version: STATE_FORMAT_VERSION,
                records: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.records.len() > MAX_PERSISTED_DHT_RECORDS {
            return Err(StorageError::InvalidState(
                "DHT record state exceeds local bound".to_string(),
            ));
        }
        for record in &state.records {
            record
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        Ok(state.records)
    }

    pub fn save_dht_contacts(&self, contacts: &[SignedAnnouncement]) -> Result<(), StorageError> {
        if contacts.len() > MAX_PERSISTED_DHT_CONTACTS {
            return Err(StorageError::InvalidState(
                "DHT contact state exceeds local bound".to_string(),
            ));
        }
        for contact in contacts {
            contact
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        self.write_json(
            "dht-contacts.json",
            &DhtContactState {
                version: STATE_FORMAT_VERSION,
                contacts: contacts.to_vec(),
            },
        )
    }

    pub fn load_dht_contacts(&self) -> Result<Vec<SignedAnnouncement>, StorageError> {
        let state: DhtContactState = self.read_json_or_default(
            "dht-contacts.json",
            DhtContactState {
                version: STATE_FORMAT_VERSION,
                contacts: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.contacts.len() > MAX_PERSISTED_DHT_CONTACTS {
            return Err(StorageError::InvalidState(
                "DHT contact state exceeds local bound".to_string(),
            ));
        }
        for contact in &state.contacts {
            contact
                .validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        Ok(state.contacts)
    }

    pub fn save_evidence(&self, evidence: &[SignedEvidence]) -> Result<(), StorageError> {
        if evidence.len() > MAX_PERSISTED_EVIDENCE {
            return Err(StorageError::InvalidState(
                "evidence state exceeds local bound".to_string(),
            ));
        }
        for item in evidence {
            item.validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        self.write_json(
            "evidence.json",
            &EvidenceState {
                version: STATE_FORMAT_VERSION,
                evidence: evidence.to_vec(),
            },
        )
    }

    pub fn load_evidence(&self) -> Result<Vec<SignedEvidence>, StorageError> {
        let state: EvidenceState = self.read_json_or_default(
            "evidence.json",
            EvidenceState {
                version: STATE_FORMAT_VERSION,
                evidence: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.evidence.len() > MAX_PERSISTED_EVIDENCE {
            return Err(StorageError::InvalidState(
                "evidence state exceeds local bound".to_string(),
            ));
        }
        for item in &state.evidence {
            item.validate()
                .map_err(|error| StorageError::InvalidState(error.to_string()))?;
        }
        Ok(state.evidence)
    }

    pub fn save_jobs(&self, jobs: &[PersistedJob]) -> Result<(), StorageError> {
        if jobs.len() > MAX_PERSISTED_JOBS {
            return Err(StorageError::InvalidState(
                "job state exceeds local bound".to_string(),
            ));
        }
        for job in jobs {
            if job
                .output
                .as_ref()
                .is_some_and(|output| output.len() > MAX_JOB_OUTPUT)
            {
                return Err(StorageError::InvalidState(
                    "persisted job output exceeds bound".to_string(),
                ));
            }
        }
        self.write_json(
            "jobs.json",
            &JobStateFile {
                version: STATE_FORMAT_VERSION,
                jobs: jobs.to_vec(),
            },
        )
    }

    pub fn load_jobs(&self) -> Result<Vec<PersistedJob>, StorageError> {
        let state: JobStateFile = self.read_json_or_default(
            "jobs.json",
            JobStateFile {
                version: STATE_FORMAT_VERSION,
                jobs: Vec::new(),
            },
        )?;
        validate_state_version(state.version)?;
        if state.jobs.len() > MAX_PERSISTED_JOBS {
            return Err(StorageError::InvalidState(
                "job state exceeds local bound".to_string(),
            ));
        }
        for job in &state.jobs {
            if job
                .output
                .as_ref()
                .is_some_and(|output| output.len() > MAX_JOB_OUTPUT)
            {
                return Err(StorageError::InvalidState(
                    "persisted job output exceeds bound".to_string(),
                ));
            }
        }
        Ok(state.jobs)
    }

    pub fn write_json<T: Serialize>(&self, name: &str, value: &T) -> Result<(), StorageError> {
        if name.contains('/') || name.contains('\\') || name.is_empty() {
            return Err(StorageError::InvalidState(
                "state file name is not local".to_string(),
            ));
        }
        let bytes = serde_json::to_vec(value)?;
        if bytes.len() as u64 > MAX_STATE_BYTES {
            return Err(StorageError::InvalidState(
                "state file exceeds bound".to_string(),
            ));
        }
        let path = self.root.join("state").join(name);
        let previous_size = fs::metadata(&path)
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        let used = self.used_bytes()?.saturating_sub(previous_size);
        if used.saturating_add(bytes.len() as u64) > self.quota_bytes {
            return Err(StorageError::QuotaExceeded {
                used,
                requested: bytes.len() as u64,
                quota: self.quota_bytes,
            });
        }
        let temp = self.temp_path("state");
        {
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(temp, path)?;
        Ok(())
    }

    /// Read one bounded state file by its local filename.
    ///
    /// Callers still own the schema validation. The filename guard keeps this
    /// helper from becoming a path traversal surface for peer-controlled data.
    pub fn read_json<T: DeserializeOwned>(&self, name: &str) -> Result<Option<T>, StorageError> {
        if name.contains('/') || name.contains('\\') || name.is_empty() {
            return Err(StorageError::InvalidState(
                "state file name is not local".to_string(),
            ));
        }
        let path = self.root.join("state").join(name);
        match fs::metadata(&path) {
            Ok(metadata) => {
                if metadata.len() > MAX_STATE_BYTES {
                    return Err(StorageError::InvalidState(
                        "state file exceeds bound".to_string(),
                    ));
                }
                Ok(Some(serde_json::from_slice(&fs::read(path)?)?))
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(StorageError::Io(error)),
        }
    }

    fn read_json_or_default<T: DeserializeOwned>(
        &self,
        name: &str,
        default: T,
    ) -> Result<T, StorageError> {
        Ok(self.read_json(name)?.unwrap_or(default))
    }

    fn artifact_path(&self, artifact: ArtifactId) -> PathBuf {
        self.root.join("artifacts").join(artifact.to_string())
    }

    fn partial_artifact_path(&self, artifact: ArtifactId) -> PathBuf {
        self.root
            .join("state")
            .join("transfers")
            .join(format!("{artifact}.part"))
    }

    fn quarantine_partial(&self, artifact: ArtifactId, path: &Path) -> Result<(), StorageError> {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let target = self.root.join("quarantine").join(format!(
            "partial-{}.{}.{}",
            artifact,
            now_millis(),
            sequence
        ));
        fs::rename(path, target)?;
        let _ = self.garbage_collect()?;
        Ok(())
    }

    fn quarantine(&self, artifact: ArtifactId, path: &Path) -> Result<(), StorageError> {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let target = self.root.join("quarantine").join(format!(
            "{}.{}.{}",
            artifact,
            now_millis(),
            sequence
        ));
        fs::rename(path, target)?;
        let _ = self.garbage_collect()?;
        Ok(())
    }

    fn temp_path(&self, prefix: &str) -> PathBuf {
        let sequence = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        self.root.join("state").join(format!(
            ".{prefix}.{}.{}.{}",
            std::process::id(),
            now_millis(),
            sequence
        ))
    }
}

fn validate_state_version(version: u16) -> Result<(), StorageError> {
    if version > STATE_FORMAT_VERSION {
        return Err(StorageError::InvalidState(format!(
            "state format version {version} is newer than supported version {STATE_FORMAT_VERSION}"
        )));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<ArtifactId, StorageError> {
    let mut file = File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(ArtifactId::from_bytes(*hasher.finalize().as_bytes()))
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "intelligence-storage-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn artifacts_are_content_addressed_and_verified() {
        let root = temp_root();
        let store = LocalStore::open(&root, 1024 * 1024, 1024).unwrap();
        let id = store.put_artifact(b"fixture").unwrap();
        assert_eq!(store.get_artifact(id).unwrap(), b"fixture");
        fs::write(store.artifact_path(id), b"corrupt").unwrap();
        assert!(matches!(
            store.get_artifact(id),
            Err(StorageError::Integrity { .. })
        ));
        assert!(!store.has_artifact(id));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn quota_is_enforced_before_write() {
        let root = temp_root();
        let store = LocalStore::open(&root, 4, 4).unwrap();
        assert!(matches!(
            store.put_artifact(b"12345"),
            Err(StorageError::ArtifactTooLarge { .. })
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn supplied_artifact_ids_must_match_content() {
        let root = temp_root();
        let store = LocalStore::open(&root, 1024, 1024).unwrap();
        let wrong = ArtifactId::from_bytes([9; 32]);
        assert!(matches!(
            store.put_artifact_with_id(wrong, b"fixture"),
            Err(StorageError::Integrity { artifact }) if artifact == wrong
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn partial_artifact_resume_reserves_final_size_and_verifies_before_publish() {
        let root = temp_root();
        let store = LocalStore::open(&root, 16, 16).unwrap();
        let bytes = b"0123456789abcdef";
        let artifact = ArtifactId::from_bytes_hashed(bytes);
        assert_eq!(
            store
                .write_partial_artifact(artifact, 16, 0, &bytes[..7])
                .unwrap(),
            7
        );
        assert_eq!(store.partial_artifact_size(artifact).unwrap(), 7);
        assert_eq!(
            store
                .write_partial_artifact(artifact, 16, 7, &bytes[7..])
                .unwrap(),
            16
        );
        store
            .finalize_partial_artifact(artifact, 16, artifact)
            .unwrap();
        assert_eq!(store.get_artifact(artifact).unwrap(), bytes);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn quarantine_is_garbage_collected_against_a_bounded_budget() {
        let root = temp_root();
        let store = LocalStore::open(&root, 64, 64).unwrap();
        let id = store.put_artifact(&[1u8; 16]).unwrap();
        fs::write(store.artifact_path(id), [2u8; 16]).unwrap();
        assert!(matches!(
            store.get_artifact(id),
            Err(StorageError::Integrity { .. })
        ));
        assert_eq!(fs::read_dir(root.join("quarantine")).unwrap().count(), 0);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dht_contacts_and_signed_evidence_survive_restart() {
        let root = temp_root();
        let store = LocalStore::open(&root, 1024 * 1024, 1024).unwrap();
        let public_key = [7u8; 32];
        let node = NodeId::from_public_key(&public_key);
        let record = DhtRecord {
            namespace: intelligence_protocol::DhtNamespace::Artifact,
            key: intelligence_protocol::DhtKey::for_name(
                intelligence_protocol::DhtNamespace::Artifact,
                "fixture",
            ),
            owner: node,
            owner_public_key: public_key,
            sequence: 1,
            expires_at: 2,
            value: b"provider".to_vec(),
            signature: vec![0; 64],
        };
        let contact = SignedAnnouncement {
            record: PeerRecord {
                node_id: node,
                public_key,
                addresses: vec!["10.0.0.2:40000".to_string()],
                capabilities: Vec::new(),
                announced_at: 1,
                expires_at: 2,
                observed_latency_ms: None,
            },
            signature: vec![0; 64],
        };
        let evidence = SignedEvidence {
            issuer: node,
            issuer_public_key: public_key,
            subject: node,
            kind: intelligence_protocol::EvidenceKind::ObservedOnline,
            sequence: 1,
            observed_at: 1,
            expires_at: 2,
            payload: b"online".to_vec(),
            signature: vec![0; 64],
        };
        store
            .save_dht_records(std::slice::from_ref(&record))
            .unwrap();
        store
            .save_dht_contacts(std::slice::from_ref(&contact))
            .unwrap();
        store
            .save_evidence(std::slice::from_ref(&evidence))
            .unwrap();
        assert_eq!(store.load_dht_records().unwrap(), vec![record]);
        assert_eq!(store.load_dht_contacts().unwrap(), vec![contact]);
        assert_eq!(store.load_evidence().unwrap(), vec![evidence]);
        let _ = fs::remove_dir_all(root);
    }
}
