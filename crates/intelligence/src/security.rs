//! V6 local adversarial-trust primitives.
//!
//! The types in this module are deliberately local and bounded.  They are
//! used by the node and by the deterministic lab, but they do not form a
//! network-wide reputation service.  A signature authenticates the issuer of
//! a claim; it does not turn an indirect report into an observation.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use intelligence_protocol::{
    ArtifactId, BackendKind, JobId, MAX_V4_VECTOR, NodeId, TRAINING_SCALE, V4ByzantinePolicy,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
use thiserror::Error;

use crate::trust::IdentityMaturity;

const MAX_SECURITY_DETAIL: usize = 256;
const MAX_SECURITY_SCOPE: usize = 128;
const MAX_CLAIM_KEY: usize = 128;

fn default_direct_observation_ttl() -> u64 {
    900
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityProfile {
    Open,
    Balanced,
    Strict,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SecurityEventKind {
    ReplayRejected,
    StaleStateRejected,
    DuplicateContributionRejected,
    EquivocationDetected,
    InvalidStateRejected,
    ArtifactCorrupt,
    CheckpointCorrupt,
    OptimizerStateRejected,
    CapabilityFailed,
    TrainingUpdateRejected,
    EvaluatorDisagreement,
    EvaluatorSelection,
    DhtPoisoningRejected,
    ResourceRateLimited,
    PeerQuarantined,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CriticalRole {
    CheckpointProvider,
    OptimizerReplica,
    TensorRecoveryProvider,
    Aggregator,
    Evaluator,
    PlanParticipant,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SecurityPolicy {
    pub profile: SecurityProfile,
    pub max_events: usize,
    pub max_replay_entries: usize,
    pub max_claims: usize,
    pub quarantine_seconds: u64,
    pub min_direct_successes_for_critical_role: usize,
    pub max_direct_failures_for_critical_role: usize,
    pub max_training_value: i64,
    pub max_challenge_per_peer_window: u16,
    pub challenge_window_seconds: u64,
    pub max_evaluators_per_source_group: usize,
    /// Successful local behavior is useful evidence, not a permanent
    /// immunity grant. Expired observations become `New` until observed again.
    #[serde(default = "default_direct_observation_ttl")]
    pub direct_observation_ttl_seconds: u64,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            profile: SecurityProfile::Balanced,
            max_events: 2_048,
            max_replay_entries: 4_096,
            max_claims: 2_048,
            quarantine_seconds: 300,
            min_direct_successes_for_critical_role: 1,
            max_direct_failures_for_critical_role: 2,
            max_training_value: 100 * TRAINING_SCALE,
            max_challenge_per_peer_window: 8,
            challenge_window_seconds: 60,
            max_evaluators_per_source_group: 1,
            direct_observation_ttl_seconds: default_direct_observation_ttl(),
        }
    }
}

impl SecurityPolicy {
    pub fn for_profile(profile: SecurityProfile) -> Self {
        let mut policy = Self {
            profile,
            ..Self::default()
        };
        match profile {
            SecurityProfile::Open => {
                policy.min_direct_successes_for_critical_role = 0;
                policy.max_direct_failures_for_critical_role = 3;
                policy.max_challenge_per_peer_window = 4;
            }
            SecurityProfile::Balanced => {}
            SecurityProfile::Strict => {
                policy.min_direct_successes_for_critical_role = 2;
                policy.max_direct_failures_for_critical_role = 1;
                policy.max_challenge_per_peer_window = 4;
                policy.quarantine_seconds = 600;
            }
        }
        policy
    }

    fn validate(&self) -> Result<(), SecurityError> {
        if self.max_events == 0
            || self.max_replay_entries == 0
            || self.max_claims == 0
            || self.quarantine_seconds == 0
            || self.challenge_window_seconds == 0
            || self.max_challenge_per_peer_window == 0
            || self.max_evaluators_per_source_group == 0
            || self.max_training_value <= 0
            || self.direct_observation_ttl_seconds == 0
        {
            return Err(SecurityError::InvalidPolicy(
                "V6 security bounds must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub sequence: u64,
    pub observed_at: u64,
    pub subject: Option<NodeId>,
    pub kind: SecurityEventKind,
    pub scope: String,
    pub detail: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingContribution {
    pub job_id: JobId,
    pub worker: NodeId,
    pub branch: ArtifactId,
    pub plan_generation: u64,
    pub generation: u64,
    pub sequence: u64,
    pub values: Vec<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrainingContext {
    pub job_id: JobId,
    pub branch: ArtifactId,
    pub plan_generation: u64,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingAdmission {
    Accepted,
    Duplicate,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityConfidence {
    pub successful_challenges: u32,
    pub failed_challenges: u32,
    pub successful_tasks: u32,
    pub failed_tasks: u32,
    pub last_observed_at: u64,
}

impl CapabilityConfidence {
    pub fn confidence_permille(&self) -> u16 {
        let successes = self
            .successful_challenges
            .saturating_add(self.successful_tasks);
        let failures = self.failed_challenges.saturating_add(self.failed_tasks);
        let total = successes.saturating_add(failures);
        if total == 0 {
            return 0;
        }
        (successes.saturating_mul(1000) / total).min(1000) as u16
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PeerObservation {
    pub direct_successes: u32,
    pub direct_failures: u32,
    pub last_observed_at: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SignedStateClaim {
    pub signer: NodeId,
    pub signer_public_key: [u8; 32],
    pub object: String,
    pub job_id: Option<JobId>,
    pub branch: Option<ArtifactId>,
    pub generation: u64,
    pub sequence: u64,
    pub digest: ArtifactId,
    pub expires_at: u64,
    pub signature: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EquivocationProof {
    pub first: SignedStateClaim,
    pub second: SignedStateClaim,
}

#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum SecurityError {
    #[error("invalid V6 security policy: {0}")]
    InvalidPolicy(String),
    #[error("security record is malformed: {0}")]
    InvalidRecord(String),
    #[error("training contributor is not authorized for this group")]
    UnauthorizedTraining,
    #[error("training contribution is stale")]
    StaleTraining,
    #[error("training contribution is a duplicate")]
    DuplicateTraining,
    #[error("training contributor equivocated")]
    TrainingEquivocation,
    #[error("peer is locally quarantined")]
    Quarantined,
    #[error("training contribution exceeds the local safety bound")]
    TrainingBounds,
    #[error("equivocation proof is invalid")]
    InvalidEquivocationProof,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
struct ReplayKey {
    job_id: JobId,
    worker: NodeId,
    branch: ArtifactId,
    generation: u64,
    sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
struct StreamKey {
    job_id: JobId,
    worker: NodeId,
    branch: ArtifactId,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
struct ClaimKey {
    signer: NodeId,
    object: String,
    job_id: Option<JobId>,
    branch: Option<ArtifactId>,
    generation: u64,
    sequence: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
struct CapabilityKey {
    peer: NodeId,
    backend: BackendKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TrustAssessment {
    pub peer: NodeId,
    pub score: i16,
    pub maturity: IdentityMaturity,
    pub direct_successes: u32,
    pub direct_failures: u32,
    pub quarantined: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocalSecurityState {
    pub local_node: NodeId,
    pub policy: SecurityPolicy,
    #[serde(default)]
    events: VecDeque<SecurityEvent>,
    #[serde(default)]
    replay: BTreeMap<ReplayKey, ArtifactId>,
    #[serde(default)]
    latest: BTreeMap<StreamKey, (u64, u64)>,
    #[serde(default)]
    contexts: BTreeMap<JobId, TrainingContext>,
    #[serde(default)]
    claims: BTreeMap<ClaimKey, SignedStateClaim>,
    /// Claims observed over an authenticated peer session. These are kept
    /// separate from `claims` because the transport identity is authentic but
    /// the state payload is not a detached Ed25519 proof object.
    #[serde(default)]
    authenticated_claims: BTreeMap<ClaimKey, ArtifactId>,
    #[serde(default)]
    quarantined: BTreeMap<NodeId, u64>,
    #[serde(default)]
    capabilities: BTreeMap<CapabilityKey, CapabilityConfidence>,
    #[serde(default)]
    observations: BTreeMap<NodeId, PeerObservation>,
    #[serde(default)]
    next_event_sequence: u64,
}

impl LocalSecurityState {
    pub fn new(local_node: NodeId, policy: SecurityPolicy) -> Result<Self, SecurityError> {
        policy.validate()?;
        Ok(Self {
            local_node,
            policy,
            events: VecDeque::new(),
            replay: BTreeMap::new(),
            latest: BTreeMap::new(),
            contexts: BTreeMap::new(),
            claims: BTreeMap::new(),
            authenticated_claims: BTreeMap::new(),
            quarantined: BTreeMap::new(),
            capabilities: BTreeMap::new(),
            observations: BTreeMap::new(),
            next_event_sequence: 1,
        })
    }

    pub fn policy(&self) -> &SecurityPolicy {
        &self.policy
    }

    /// Validate a persisted state before it can influence a local decision.
    /// Deserialization alone is intentionally not treated as validation.
    pub fn validate(&self) -> Result<(), SecurityError> {
        self.policy.validate()?;
        if self.local_node == NodeId::default()
            || self.events.len() > self.policy.max_events
            || self.replay.len() > self.policy.max_replay_entries
            || self.claims.len() > self.policy.max_claims
            || self.authenticated_claims.len() > self.policy.max_claims
        {
            return Err(SecurityError::InvalidRecord(
                "persisted V6 security state exceeds local bounds".to_string(),
            ));
        }
        if self.events.iter().any(|event| {
            event.scope.len() > MAX_SECURITY_SCOPE || event.detail.len() > MAX_SECURITY_DETAIL
        }) {
            return Err(SecurityError::InvalidRecord(
                "persisted security event exceeds local bounds".to_string(),
            ));
        }
        Ok(())
    }

    pub fn events(&self) -> impl Iterator<Item = &SecurityEvent> {
        self.events.iter()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Add a bounded local event without exposing the internal event store or
    /// turning it into a remotely writable log.
    pub fn record_event(
        &mut self,
        observed_at: u64,
        subject: Option<NodeId>,
        kind: SecurityEventKind,
        scope: &str,
        detail: &str,
    ) {
        self.event(observed_at, subject, kind, scope, detail);
    }

    pub fn is_quarantined(&mut self, peer: NodeId, now: u64) -> bool {
        self.quarantined.retain(|_, until| *until >= now);
        self.quarantined
            .get(&peer)
            .is_some_and(|until| *until >= now)
    }

    pub fn observe_context(
        &mut self,
        context: TrainingContext,
        now: u64,
    ) -> Result<(), SecurityError> {
        if context.job_id == JobId::default()
            || context.branch == ArtifactId::default()
            || context.plan_generation == 0
            || context.generation == 0
        {
            self.event(
                now,
                None,
                SecurityEventKind::InvalidStateRejected,
                "training_context",
                "zero or missing lineage field",
            );
            return Err(SecurityError::InvalidRecord(
                "training context lineage is incomplete".to_string(),
            ));
        }
        if let Some(previous) = self.contexts.get(&context.job_id) {
            if context.plan_generation < previous.plan_generation
                || context.generation < previous.generation
            {
                self.event(
                    now,
                    None,
                    SecurityEventKind::StaleStateRejected,
                    "training_context",
                    "older plan or model generation",
                );
                return Err(SecurityError::StaleTraining);
            }
            if context.plan_generation == previous.plan_generation
                && context.generation == previous.generation
                && (context.branch != previous.branch)
            {
                self.event(
                    now,
                    None,
                    SecurityEventKind::EquivocationDetected,
                    "training_context",
                    "conflicting branch at the same generation",
                );
                return Err(SecurityError::TrainingEquivocation);
            }
        }
        self.contexts.insert(context.job_id, context);
        Ok(())
    }

    pub fn admit_training_update(
        &mut self,
        contribution: &TrainingContribution,
        members: &[NodeId],
        now: u64,
    ) -> Result<TrainingAdmission, SecurityError> {
        if !members.contains(&contribution.worker) {
            self.event(
                now,
                Some(contribution.worker),
                SecurityEventKind::TrainingUpdateRejected,
                "training_update",
                "worker is outside the authenticated job membership",
            );
            return Err(SecurityError::UnauthorizedTraining);
        }
        if self.is_quarantined(contribution.worker, now) {
            self.event(
                now,
                Some(contribution.worker),
                SecurityEventKind::TrainingUpdateRejected,
                "training_update",
                "worker is locally quarantined",
            );
            return Err(SecurityError::Quarantined);
        }
        if contribution.job_id == JobId::default()
            || contribution.branch == ArtifactId::default()
            || contribution.plan_generation == 0
            || contribution.generation == 0
            || contribution.sequence == 0
            || contribution.values.is_empty()
            || contribution.values.len() > MAX_V4_VECTOR
            || contribution
                .values
                .iter()
                .any(|value| value.unsigned_abs() > self.policy.max_training_value.unsigned_abs())
        {
            self.event(
                now,
                Some(contribution.worker),
                SecurityEventKind::TrainingUpdateRejected,
                "training_update",
                "lineage, vector, or magnitude bound failed",
            );
            return Err(SecurityError::TrainingBounds);
        }
        self.observe_context(
            TrainingContext {
                job_id: contribution.job_id,
                branch: contribution.branch,
                plan_generation: contribution.plan_generation,
                generation: contribution.generation,
            },
            now,
        )?;
        let stream = StreamKey {
            job_id: contribution.job_id,
            worker: contribution.worker,
            branch: contribution.branch,
        };
        if let Some((generation, sequence)) = self.latest.get(&stream).copied()
            && (contribution.generation < generation
                || (contribution.generation == generation && contribution.sequence < sequence))
        {
            self.event(
                now,
                Some(contribution.worker),
                SecurityEventKind::StaleStateRejected,
                "training_update",
                "older generation or sequence",
            );
            return Err(SecurityError::StaleTraining);
        }
        let digest = contribution_digest(contribution);
        let key = ReplayKey {
            job_id: contribution.job_id,
            worker: contribution.worker,
            branch: contribution.branch,
            generation: contribution.generation,
            sequence: contribution.sequence,
        };
        if let Some(previous) = self.replay.get(&key).copied() {
            if previous == digest {
                self.event(
                    now,
                    Some(contribution.worker),
                    SecurityEventKind::DuplicateContributionRejected,
                    "training_update",
                    "same signed contribution was submitted twice",
                );
                return Err(SecurityError::DuplicateTraining);
            }
            self.quarantine(
                contribution.worker,
                now,
                SecurityEventKind::EquivocationDetected,
                "same contribution sequence carried different values",
            );
            return Err(SecurityError::TrainingEquivocation);
        }
        self.bound_replay();
        self.replay.insert(key, digest);
        self.latest
            .insert(stream, (contribution.generation, contribution.sequence));
        Ok(TrainingAdmission::Accepted)
    }

    pub fn record_direct_observation(&mut self, peer: NodeId, success: bool, now: u64) {
        {
            let observation = self.observations.entry(peer).or_insert(PeerObservation {
                direct_successes: 0,
                direct_failures: 0,
                last_observed_at: now,
            });
            if success {
                observation.direct_successes = observation.direct_successes.saturating_add(1);
            } else {
                observation.direct_failures = observation.direct_failures.saturating_add(1);
            }
            observation.last_observed_at = now;
        }
        if !success {
            self.event(
                now,
                Some(peer),
                SecurityEventKind::TrainingUpdateRejected,
                "direct_observation",
                "direct job observation failed",
            );
        }
    }

    pub fn record_capability_challenge(
        &mut self,
        peer: NodeId,
        backend: BackendKind,
        success: bool,
        now: u64,
    ) {
        {
            let confidence = self
                .capabilities
                .entry(CapabilityKey { peer, backend })
                .or_insert(CapabilityConfidence {
                    successful_challenges: 0,
                    failed_challenges: 0,
                    successful_tasks: 0,
                    failed_tasks: 0,
                    last_observed_at: now,
                });
            if success {
                confidence.successful_challenges =
                    confidence.successful_challenges.saturating_add(1);
            } else {
                confidence.failed_challenges = confidence.failed_challenges.saturating_add(1);
            }
            confidence.last_observed_at = now;
        }
        if !success {
            self.event(
                now,
                Some(peer),
                SecurityEventKind::CapabilityFailed,
                "capability_challenge",
                "challenge result did not verify the advertised backend",
            );
        }
    }

    pub fn record_capability_task(
        &mut self,
        peer: NodeId,
        backend: BackendKind,
        success: bool,
        now: u64,
    ) {
        let confidence = self
            .capabilities
            .entry(CapabilityKey { peer, backend })
            .or_insert(CapabilityConfidence {
                successful_challenges: 0,
                failed_challenges: 0,
                successful_tasks: 0,
                failed_tasks: 0,
                last_observed_at: now,
            });
        if success {
            confidence.successful_tasks = confidence.successful_tasks.saturating_add(1);
        } else {
            confidence.failed_tasks = confidence.failed_tasks.saturating_add(1);
        }
        confidence.last_observed_at = now;
    }

    pub fn capability_confidence(
        &self,
        peer: NodeId,
        backend: BackendKind,
    ) -> Option<CapabilityConfidence> {
        self.capabilities
            .get(&CapabilityKey { peer, backend })
            .cloned()
    }

    pub fn assessment(&mut self, peer: NodeId, now: u64) -> TrustAssessment {
        let observation = self
            .observations
            .get(&peer)
            .cloned()
            .unwrap_or(PeerObservation {
                direct_successes: 0,
                direct_failures: 0,
                last_observed_at: 0,
            });
        let observation = if observation.last_observed_at == 0
            || now.saturating_sub(observation.last_observed_at)
                <= self.policy.direct_observation_ttl_seconds
        {
            observation
        } else {
            PeerObservation {
                direct_successes: 0,
                direct_failures: 0,
                last_observed_at: observation.last_observed_at,
            }
        };
        let quarantined = self.is_quarantined(peer, now);
        let positive = (observation.direct_successes.min(20) as i16).saturating_mul(5);
        let negative = (observation.direct_failures.min(20) as i16).saturating_mul(12);
        let score = positive.saturating_sub(negative).clamp(-100, 100);
        let maturity = if quarantined {
            IdentityMaturity::Quarantined
        } else if score < 0 {
            IdentityMaturity::Degraded
        } else if observation.direct_successes >= 3 {
            IdentityMaturity::Established
        } else if observation.direct_successes > 0 {
            IdentityMaturity::Observed
        } else {
            IdentityMaturity::New
        };
        let reason = match maturity {
            IdentityMaturity::New => "no direct local observation".to_string(),
            IdentityMaturity::Observed => "bounded direct observation exists".to_string(),
            IdentityMaturity::Established => "repeated direct observation exists".to_string(),
            IdentityMaturity::TrustedLocally => "local trust threshold satisfied".to_string(),
            IdentityMaturity::Degraded => "recent direct failures outweigh successes".to_string(),
            IdentityMaturity::Quarantined => "temporary local quarantine is active".to_string(),
        };
        TrustAssessment {
            peer,
            score,
            maturity,
            direct_successes: observation.direct_successes,
            direct_failures: observation.direct_failures,
            quarantined,
            reason,
        }
    }

    pub fn eligible_for_critical_role(
        &mut self,
        peer: NodeId,
        role: CriticalRole,
        now: u64,
    ) -> bool {
        if self.is_quarantined(peer, now) {
            return false;
        }
        let assessment = self.assessment(peer, now);
        let requires_observation = matches!(
            role,
            CriticalRole::CheckpointProvider
                | CriticalRole::OptimizerReplica
                | CriticalRole::TensorRecoveryProvider
                | CriticalRole::Aggregator
                | CriticalRole::Evaluator
        ) && !matches!(self.policy.profile, SecurityProfile::Open);
        (!requires_observation
            || assessment.direct_successes
                >= self.policy.min_direct_successes_for_critical_role as u32)
            && assessment.direct_failures
                <= self.policy.max_direct_failures_for_critical_role as u32
            && assessment.maturity != IdentityMaturity::Degraded
            && assessment.maturity != IdentityMaturity::Quarantined
    }

    pub fn observe_claim(
        &mut self,
        claim: SignedStateClaim,
        now: u64,
    ) -> Result<Option<EquivocationProof>, SecurityError> {
        verify_claim(&claim, now)?;
        if claim.object.len() > MAX_CLAIM_KEY
            || claim.signer == NodeId::default()
            || claim.sequence == 0
            || claim.generation == 0
            || claim.digest == ArtifactId::default()
        {
            return Err(SecurityError::InvalidRecord(
                "state claim bounds are invalid".to_string(),
            ));
        }
        let key = ClaimKey {
            signer: claim.signer,
            object: claim.object.clone(),
            job_id: claim.job_id,
            branch: claim.branch,
            generation: claim.generation,
            sequence: claim.sequence,
        };
        if let Some(previous) = self.claims.get(&key) {
            if previous.digest == claim.digest {
                return Ok(None);
            }
            let proof = EquivocationProof {
                first: previous.clone(),
                second: claim.clone(),
            };
            if !proof.verify(now) {
                return Err(SecurityError::InvalidEquivocationProof);
            }
            self.quarantine(
                claim.signer,
                now,
                SecurityEventKind::EquivocationDetected,
                "two valid signed claims conflict at one generation and sequence",
            );
            return Ok(Some(proof));
        }
        if self.claims.len() >= self.policy.max_claims {
            if let Some(key) = self.claims.keys().next().cloned() {
                self.claims.remove(&key);
            }
        }
        self.claims.insert(key, claim);
        Ok(None)
    }

    /// Observe a state claim bound to the authenticated transport peer. A
    /// conflicting value at one generation/sequence is local equivocation
    /// evidence. It is intentionally not represented as a signed detached
    /// proof; callers must use `observe_claim` when the protocol carries such
    /// a proof object.
    #[allow(clippy::too_many_arguments)]
    pub fn observe_authenticated_claim(
        &mut self,
        signer: NodeId,
        object: &str,
        job_id: Option<JobId>,
        branch: Option<ArtifactId>,
        generation: u64,
        sequence: u64,
        digest: ArtifactId,
        now: u64,
    ) -> Result<bool, SecurityError> {
        if signer == NodeId::default()
            || object.is_empty()
            || object.len() > MAX_CLAIM_KEY
            || generation == 0
            || sequence == 0
            || digest == ArtifactId::default()
        {
            self.event(
                now,
                Some(signer),
                SecurityEventKind::InvalidStateRejected,
                "authenticated_claim",
                "state claim bounds are invalid",
            );
            return Err(SecurityError::InvalidRecord(
                "authenticated state claim bounds are invalid".to_string(),
            ));
        }
        let key = ClaimKey {
            signer,
            object: object.to_string(),
            job_id,
            branch,
            generation,
            sequence,
        };
        if let Some(previous) = self.authenticated_claims.get(&key).copied() {
            if previous == digest {
                return Ok(false);
            }
            self.quarantine(
                signer,
                now,
                SecurityEventKind::EquivocationDetected,
                "authenticated peer sent conflicting state for one generation and sequence",
            );
            return Err(SecurityError::TrainingEquivocation);
        }
        if self.authenticated_claims.len() >= self.policy.max_claims {
            if let Some(key) = self.authenticated_claims.keys().next().cloned() {
                self.authenticated_claims.remove(&key);
            }
        }
        self.authenticated_claims.insert(key, digest);
        Ok(true)
    }

    fn quarantine(&mut self, peer: NodeId, now: u64, kind: SecurityEventKind, detail: &str) {
        let until = now.saturating_add(self.policy.quarantine_seconds);
        self.quarantined
            .entry(peer)
            .and_modify(|value| *value = (*value).max(until))
            .or_insert(until);
        self.event(now, Some(peer), kind, "quarantine", detail);
        self.event(
            now,
            Some(peer),
            SecurityEventKind::PeerQuarantined,
            "quarantine",
            "local quarantine applied; no network-wide ban was published",
        );
    }

    fn bound_replay(&mut self) {
        while self.replay.len() >= self.policy.max_replay_entries {
            let Some(key) = self.replay.keys().next().cloned() else {
                break;
            };
            self.replay.remove(&key);
        }
    }

    fn event(
        &mut self,
        observed_at: u64,
        subject: Option<NodeId>,
        kind: SecurityEventKind,
        scope: &str,
        detail: &str,
    ) {
        let scope = truncate(scope, MAX_SECURITY_SCOPE);
        let detail = truncate(detail, MAX_SECURITY_DETAIL);
        if self.events.iter().any(|event| {
            event.subject == subject
                && event.kind == kind
                && event.scope == scope
                && event.detail == detail
        }) {
            return;
        }
        if self.events.len() >= self.policy.max_events {
            self.events.pop_front();
        }
        let sequence = self.next_event_sequence;
        self.next_event_sequence = self.next_event_sequence.saturating_add(1);
        self.events.push_back(SecurityEvent {
            sequence,
            observed_at,
            subject,
            kind,
            scope,
            detail,
        });
    }
}

impl EquivocationProof {
    pub fn verify(&self, now: u64) -> bool {
        self.first.signer == self.second.signer
            && self.first.signer_public_key == self.second.signer_public_key
            && self.first.object == self.second.object
            && self.first.job_id == self.second.job_id
            && self.first.branch == self.second.branch
            && self.first.generation == self.second.generation
            && self.first.sequence == self.second.sequence
            && self.first.digest != self.second.digest
            && verify_claim(&self.first, now).is_ok()
            && verify_claim(&self.second, now).is_ok()
    }
}

pub fn claim_signing_bytes(claim: &SignedStateClaim) -> Result<Vec<u8>, SecurityError> {
    postcard::to_allocvec(&(
        claim.signer,
        claim.signer_public_key,
        &claim.object,
        claim.job_id,
        claim.branch,
        claim.generation,
        claim.sequence,
        claim.digest,
        claim.expires_at,
    ))
    .map_err(|error| SecurityError::InvalidRecord(error.to_string()))
}

pub fn verify_claim(claim: &SignedStateClaim, now: u64) -> Result<(), SecurityError> {
    if claim.signer != NodeId::from_public_key(&claim.signer_public_key)
        || claim.expires_at < now
        || claim.object.is_empty()
        || claim.object.len() > MAX_CLAIM_KEY
    {
        return Err(SecurityError::InvalidRecord(
            "claim identity or expiry is invalid".to_string(),
        ));
    }
    let bytes = claim_signing_bytes(claim)?;
    let key = VerifyingKey::from_bytes(&claim.signer_public_key)
        .map_err(|_| SecurityError::InvalidEquivocationProof)?;
    let signature = Signature::from_slice(&claim.signature)
        .map_err(|_| SecurityError::InvalidEquivocationProof)?;
    key.verify(&bytes, &signature)
        .map_err(|_| SecurityError::InvalidEquivocationProof)
}

fn contribution_digest(contribution: &TrainingContribution) -> ArtifactId {
    let bytes = postcard::to_allocvec(&(
        contribution.job_id,
        contribution.worker,
        contribution.branch,
        contribution.plan_generation,
        contribution.generation,
        contribution.sequence,
        &contribution.values,
    ))
    .unwrap_or_default();
    ArtifactId::from_bytes_hashed(&bytes)
}

fn truncate(value: &str, max: usize) -> String {
    value.chars().take(max).collect()
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorCandidate {
    pub node: NodeId,
    pub maturity: IdentityMaturity,
    pub direct_successes: u32,
    pub source_group: u16,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvaluatorSelection {
    pub selected: Vec<NodeId>,
    pub source_groups: usize,
    pub rejected: Vec<NodeId>,
}

/// Deterministic evaluator selection with bounded repeated pairing and a
/// source-group diversity heuristic.  The group is an input supplied by local
/// observation; it is not treated as proof of human independence.
pub fn select_evaluators(
    candidates: &[EvaluatorCandidate],
    count: usize,
    seed: u64,
    max_per_source_group: usize,
) -> EvaluatorSelection {
    let mut ordered = candidates
        .iter()
        .filter(|candidate| {
            !matches!(
                candidate.maturity,
                IdentityMaturity::Degraded | IdentityMaturity::Quarantined
            )
        })
        .cloned()
        .collect::<Vec<_>>();
    ordered.sort_by_key(|candidate| {
        (
            std::cmp::Reverse(candidate.maturity),
            deterministic_rank(seed, candidate.node),
            std::cmp::Reverse(candidate.direct_successes.min(3)),
            candidate.node,
        )
    });
    let mut selected = Vec::new();
    let mut group_counts = BTreeMap::<u16, usize>::new();
    for candidate in &ordered {
        if selected.len() >= count {
            break;
        }
        let count_for_group = group_counts
            .get(&candidate.source_group)
            .copied()
            .unwrap_or(0);
        if count_for_group >= max_per_source_group.max(1) {
            continue;
        }
        group_counts
            .entry(candidate.source_group)
            .and_modify(|value| *value += 1)
            .or_insert(1);
        selected.push(candidate.node);
    }
    // If the local population is smaller than the diversity policy allows,
    // fill from the remaining candidates rather than turning the policy into
    // an availability outage.
    for candidate in &ordered {
        if selected.len() >= count {
            break;
        }
        if !selected.contains(&candidate.node) {
            selected.push(candidate.node);
        }
    }
    let mut rejected = candidates
        .iter()
        .filter(|candidate| {
            matches!(
                candidate.maturity,
                IdentityMaturity::Degraded | IdentityMaturity::Quarantined
            )
        })
        .map(|candidate| candidate.node)
        .collect::<Vec<_>>();
    rejected.extend(
        ordered
            .iter()
            .filter(|candidate| !selected.contains(&candidate.node))
            .map(|candidate| candidate.node)
            .collect::<Vec<_>>(),
    );
    EvaluatorSelection {
        selected,
        source_groups: group_counts.len(),
        rejected,
    }
}

fn deterministic_rank(seed: u64, node: NodeId) -> u64 {
    let mut input = Vec::with_capacity(40);
    input.extend_from_slice(&seed.to_le_bytes());
    input.extend_from_slice(node.as_bytes());
    let hash = blake3::hash(&input);
    u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap_or([0; 8]))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RobustAggregationReport {
    pub policy: V4ByzantinePolicy,
    pub aggregate: Vec<i64>,
    pub accepted_updates: usize,
    pub rejected_updates: usize,
    pub communication_fanin: usize,
}

pub fn aggregate_v6_updates(
    policy: V4ByzantinePolicy,
    updates: &[Vec<i64>],
    clip_abs: i64,
) -> Result<RobustAggregationReport, SecurityError> {
    if updates.is_empty()
        || updates.len() > 64
        || updates.iter().any(|update| {
            update.is_empty()
                || update.len() > MAX_V4_VECTOR
                || update.len() != updates[0].len()
                || update
                    .iter()
                    .any(|value| value.unsigned_abs() > clip_abs.unsigned_abs())
        })
        || clip_abs <= 0
    {
        return Err(SecurityError::TrainingBounds);
    }
    let report = crate::training_fabric::aggregate_updates(policy, updates, clip_abs)
        .map_err(|error| SecurityError::InvalidRecord(error.to_string()))?;
    Ok(RobustAggregationReport {
        policy,
        aggregate: report.aggregate,
        accepted_updates: report.accepted_updates,
        rejected_updates: report.rejected_updates,
        // This is the local group fan-in, not a global coordinator fan-in.
        communication_fanin: updates.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::{SeedableRng, rngs::StdRng};

    fn contribution(worker: u8, sequence: u64, value: i64) -> TrainingContribution {
        TrainingContribution {
            job_id: JobId::from_bytes([1; 16]),
            worker: NodeId::from_bytes([worker; 32]),
            branch: ArtifactId::from_bytes([2; 32]),
            plan_generation: 1,
            generation: 1,
            sequence,
            values: vec![value, value],
        }
    }

    #[test]
    fn duplicate_and_stale_updates_are_rejected_without_weight_multiplication() {
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        let first = contribution(1, 1, 10);
        assert_eq!(
            state.admit_training_update(&first, &[first.worker], 10),
            Ok(TrainingAdmission::Accepted)
        );
        assert_eq!(
            state.admit_training_update(&first, &[first.worker], 10),
            Err(SecurityError::DuplicateTraining)
        );
        let stale = contribution(1, 0, 10);
        assert_eq!(
            state.admit_training_update(&stale, &[stale.worker], 10),
            Err(SecurityError::TrainingBounds)
        );
        assert_eq!(state.event_count(), 2);
    }

    #[test]
    fn conflicting_updates_quarantine_only_the_local_view() {
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        let first = contribution(1, 1, 10);
        state
            .admit_training_update(&first, &[first.worker], 10)
            .unwrap();
        let mut conflicting = first.clone();
        conflicting.values = vec![11, 10];
        assert_eq!(
            state.admit_training_update(&conflicting, &[first.worker], 10),
            Err(SecurityError::TrainingEquivocation)
        );
        assert!(state.is_quarantined(first.worker, 10));
        assert!(!state.is_quarantined(NodeId::from_bytes([2; 32]), 10));
    }

    #[test]
    fn signed_equivocation_proof_requires_two_valid_conflicting_claims() {
        let mut rng = StdRng::seed_from_u64(7);
        let key = SigningKey::generate(&mut rng);
        let public_key = key.verifying_key().to_bytes();
        let signer = NodeId::from_public_key(&public_key);
        let make = |digest| {
            let mut claim = SignedStateClaim {
                signer,
                signer_public_key: public_key,
                object: "plan".to_string(),
                job_id: Some(JobId::from_bytes([1; 16])),
                branch: Some(ArtifactId::from_bytes([2; 32])),
                generation: 1,
                sequence: 1,
                digest: ArtifactId::from_bytes([digest; 32]),
                expires_at: 100,
                signature: Vec::new(),
            };
            claim.signature = key
                .sign(&claim_signing_bytes(&claim).unwrap())
                .to_bytes()
                .to_vec();
            claim
        };
        let first = make(3);
        let second = make(4);
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        assert!(state.observe_claim(first, 10).unwrap().is_none());
        let proof = state.observe_claim(second, 10).unwrap().unwrap();
        assert!(proof.verify(10));
    }

    #[test]
    fn authenticated_conflict_is_local_equivocation_evidence() {
        let peer = NodeId::from_bytes([3; 32]);
        let job = JobId::from_bytes([1; 16]);
        let branch = ArtifactId::from_bytes([2; 32]);
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        assert!(
            state
                .observe_authenticated_claim(
                    peer,
                    "plan",
                    Some(job),
                    Some(branch),
                    1,
                    1,
                    ArtifactId::from_bytes([4; 32]),
                    10,
                )
                .unwrap()
        );
        assert_eq!(
            state.observe_authenticated_claim(
                peer,
                "plan",
                Some(job),
                Some(branch),
                1,
                1,
                ArtifactId::from_bytes([5; 32]),
                10,
            ),
            Err(SecurityError::TrainingEquivocation)
        );
        assert!(state.is_quarantined(peer, 10));
        assert!(
            state
                .events()
                .any(|event| event.kind == SecurityEventKind::EquivocationDetected)
        );
    }

    #[test]
    fn evaluator_selection_caps_correlated_groups_but_keeps_new_peers_eligible() {
        let candidates = (1..=6)
            .map(|value| EvaluatorCandidate {
                node: NodeId::from_bytes([value; 32]),
                maturity: if value < 3 {
                    IdentityMaturity::Established
                } else {
                    IdentityMaturity::New
                },
                direct_successes: u32::from(value < 3),
                source_group: if value < 5 { 1 } else { value as u16 },
            })
            .collect::<Vec<_>>();
        let selection = select_evaluators(&candidates, 3, 11, 1);
        assert_eq!(selection.selected.len(), 3);
        assert!(selection.source_groups >= 2);
        assert!(selection.selected.contains(&NodeId::from_bytes([5; 32])));
    }

    #[test]
    fn robust_aggregation_is_bounded_and_not_a_global_fanin() {
        let report = aggregate_v6_updates(
            V4ByzantinePolicy::CoordinateMedian,
            &[vec![10, 10], vec![11, 9], vec![-100, -100]],
            100,
        )
        .unwrap();
        assert_eq!(report.aggregate, vec![10, 9]);
        assert_eq!(report.communication_fanin, 3);
    }

    #[test]
    fn policy_profiles_are_bounded_and_deterministic() {
        let balanced = SecurityPolicy::for_profile(SecurityProfile::Balanced);
        let strict = SecurityPolicy::for_profile(SecurityProfile::Strict);
        assert!(
            strict.min_direct_successes_for_critical_role
                > balanced.min_direct_successes_for_critical_role
        );
        assert_eq!(balanced.max_events, SecurityPolicy::default().max_events);
    }

    #[test]
    fn persisted_state_validation_rejects_corrupt_bounds_and_observations_decay() {
        let peer = NodeId::from_bytes([4; 32]);
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        state.record_direct_observation(peer, true, 10);
        assert_eq!(
            state.assessment(peer, 10).maturity,
            IdentityMaturity::Observed
        );
        assert_eq!(
            state.assessment(peer, 10_000).maturity,
            IdentityMaturity::New
        );
        // BTreeMap keys are intentionally structured rather than JSON object
        // names. Validate the deserialized-equivalent state directly instead
        // of introducing a lossy JSON representation for the local cache.
        state.policy.max_events = 0;
        assert!(state.validate().is_err());
    }

    #[test]
    fn farm_then_attack_adapts_local_trust() {
        let peer = NodeId::from_bytes([7; 32]);
        let mut state =
            LocalSecurityState::new(NodeId::from_bytes([9; 32]), SecurityPolicy::default())
                .unwrap();
        for now in 1..=3 {
            state.record_direct_observation(peer, true, now);
        }
        assert_eq!(
            state.assessment(peer, 3).maturity,
            IdentityMaturity::Established
        );
        assert!(state.eligible_for_critical_role(peer, CriticalRole::Evaluator, 3));

        state.record_direct_observation(peer, false, 4);
        state.record_direct_observation(peer, false, 5);
        let degraded = state.assessment(peer, 5);
        assert_eq!(degraded.maturity, IdentityMaturity::Degraded);
        assert!(!state.eligible_for_critical_role(peer, CriticalRole::Evaluator, 5));

        // Positive history is not permanent immunity: after the bounded TTL
        // the peer returns to the new/unknown state in this local view.
        assert_eq!(
            state
                .assessment(
                    peer,
                    5 + SecurityPolicy::default().direct_observation_ttl_seconds + 1
                )
                .maturity,
            IdentityMaturity::New
        );
    }

    #[test]
    fn malformed_v6_records_never_panic() {
        let mut state = 0x6a31_9e42u64;
        let rounds = std::env::var("INTELLIGENCE_FUZZ_ROUNDS")
            .ok()
            .and_then(|value| value.parse::<u32>().ok())
            .unwrap_or(20_000)
            .min(1_000_000);
        for round in 0..rounds {
            let length = (next_fuzz_word(&mut state) as usize % 32_768)
                .saturating_add((round % 17) as usize);
            let mut bytes = vec![0u8; length];
            for byte in &mut bytes {
                *byte = next_fuzz_word(&mut state) as u8;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = postcard::from_bytes::<SecurityPolicy>(&bytes);
                let _ = postcard::from_bytes::<SecurityEvent>(&bytes);
                let _ = postcard::from_bytes::<TrainingContribution>(&bytes);
                let _ = postcard::from_bytes::<TrainingContext>(&bytes);
                let _ = postcard::from_bytes::<CapabilityConfidence>(&bytes);
                let _ = postcard::from_bytes::<PeerObservation>(&bytes);
                let _ = postcard::from_bytes::<SignedStateClaim>(&bytes);
                let _ = postcard::from_bytes::<EquivocationProof>(&bytes);
                let _ = postcard::from_bytes::<TrustAssessment>(&bytes);
                if let Ok(value) = postcard::from_bytes::<LocalSecurityState>(&bytes) {
                    let _ = value.validate();
                }
            }));
            assert!(result.is_ok(), "V6 decoder panicked for {length} bytes");
        }
    }

    fn next_fuzz_word(state: &mut u64) -> u64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *state
    }
}
