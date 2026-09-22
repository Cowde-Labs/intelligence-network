//! Local, bounded evidence evaluation.
//!
//! This is deliberately not a global reputation system.  Each node evaluates
//! the evidence it has received under its own policy.  Endorsements are
//! depth-limited and never multiply merely because many identities repeat the
//! same claim.

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use intelligence_protocol::{EvidenceKind, NodeId, SignedEvidence};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SybilClaimLevel {
    Unmitigated,
    BasicDefenses,
    CaptureResistantUnderTestedModel,
    GlobalSolved,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum IdentityMaturity {
    New,
    Observed,
    Established,
    TrustedLocally,
    Degraded,
    Quarantined,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrustPolicy {
    pub max_evidence: usize,
    pub max_clock_skew_seconds: u64,
    pub max_endorsements_per_issuer: usize,
    pub max_endorsement_depth: u8,
    pub established_direct_successes: usize,
    pub established_independent_issuers: usize,
    pub trusted_direct_successes: usize,
    pub trusted_independent_issuers: usize,
    #[serde(default = "default_negative_threshold")]
    pub quarantine_negative_evidence: usize,
    #[serde(default = "default_indirect_negative_limit")]
    pub max_indirect_negative_influence: usize,
    #[serde(default = "default_direct_observation_ttl")]
    pub direct_observation_ttl_seconds: u64,
}

fn default_negative_threshold() -> usize {
    3
}

fn default_indirect_negative_limit() -> usize {
    8
}

fn default_direct_observation_ttl() -> u64 {
    900
}

impl Default for TrustPolicy {
    fn default() -> Self {
        Self {
            max_evidence: 4096,
            max_clock_skew_seconds: 300,
            max_endorsements_per_issuer: 32,
            max_endorsement_depth: 1,
            established_direct_successes: 3,
            established_independent_issuers: 2,
            trusted_direct_successes: 5,
            trusted_independent_issuers: 3,
            quarantine_negative_evidence: default_negative_threshold(),
            max_indirect_negative_influence: default_indirect_negative_limit(),
            direct_observation_ttl_seconds: default_direct_observation_ttl(),
        }
    }
}

#[derive(Debug, Error)]
pub enum TrustError {
    #[error("invalid signed evidence: {0}")]
    Invalid(String),
    #[error("evidence signature is invalid")]
    InvalidSignature,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrustDecision {
    pub subject: NodeId,
    pub score: u8,
    pub maturity: IdentityMaturity,
    pub direct_successes: usize,
    pub direct_failures: usize,
    pub local_direct_successes: usize,
    pub local_direct_failures: usize,
    pub independent_issuers: usize,
    pub indirect_negative_reports: usize,
    pub endorsement_influence: usize,
    pub quarantined: bool,
    pub accepted_evidence: usize,
    pub rejected_evidence: usize,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct EvidenceGraph {
    policy: TrustPolicy,
    entries: Vec<SignedEvidence>,
    rejected: usize,
    local_issuer: Option<NodeId>,
}

impl EvidenceGraph {
    pub fn new(policy: TrustPolicy) -> Result<Self, TrustError> {
        if policy.max_evidence == 0
            || policy.max_endorsements_per_issuer == 0
            || policy.max_endorsement_depth == 0
            || policy.direct_observation_ttl_seconds == 0
        {
            return Err(TrustError::Invalid(
                "trust bounds must be positive".to_string(),
            ));
        }
        Ok(Self {
            policy,
            entries: Vec::new(),
            rejected: 0,
            local_issuer: None,
        })
    }

    pub fn with_local_issuer(mut self, issuer: NodeId) -> Self {
        self.local_issuer = Some(issuer);
        self
    }

    pub fn policy(&self) -> &TrustPolicy {
        &self.policy
    }

    pub fn entries(&self) -> &[SignedEvidence] {
        &self.entries
    }

    pub fn rejected_count(&self) -> usize {
        self.rejected
    }

    pub fn insert(&mut self, evidence: SignedEvidence, now: u64) -> Result<bool, TrustError> {
        if evidence.validate().is_err()
            || evidence.expires_at < now
            || evidence.observed_at > now.saturating_add(self.policy.max_clock_skew_seconds)
            || !verify_signed_evidence(&evidence)
        {
            self.rejected = self.rejected.saturating_add(1);
            return Err(TrustError::InvalidSignature);
        }

        let duplicate = self.entries.iter().position(|existing| {
            existing.issuer == evidence.issuer
                && existing.subject == evidence.subject
                && existing.kind == evidence.kind
        });
        if let Some(index) = duplicate {
            if self.entries[index].sequence >= evidence.sequence {
                self.rejected = self.rejected.saturating_add(1);
                return Ok(false);
            }
            self.entries[index] = evidence;
            return Ok(true);
        }

        if self.entries.len() >= self.policy.max_evidence {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, item)| (item.expires_at, item.observed_at, item.sequence))
                .map(|(index, _)| index)
                .unwrap_or(0);
            self.entries.remove(oldest);
        }
        self.entries.push(evidence);
        Ok(true)
    }

    pub fn expire(&mut self, now: u64) {
        self.entries.retain(|item| item.expires_at >= now);
    }

    pub fn decision(&self, subject: NodeId, now: u64) -> TrustDecision {
        let live = self
            .entries
            .iter()
            .filter(|item| item.subject == subject && item.expires_at >= now)
            .collect::<Vec<_>>();
        let direct = live
            .iter()
            .filter(|item| {
                !matches!(
                    item.kind,
                    EvidenceKind::Endorsement | EvidenceKind::ViolationReport
                )
            })
            .collect::<Vec<_>>();
        let direct_successes = direct
            .iter()
            .filter(|item| {
                matches!(
                    item.kind,
                    EvidenceKind::JobCompleted
                        | EvidenceKind::Evaluation
                        | EvidenceKind::ArtifactVerified
                )
            })
            .count();
        let direct_failures = direct.iter().filter(|item| is_negative(item.kind)).count();
        let is_local =
            |item: &&SignedEvidence| self.local_issuer.is_none_or(|issuer| item.issuer == issuer);
        let local_direct_successes = direct
            .iter()
            .filter(|item| {
                is_local(item)
                    && matches!(
                        item.kind,
                        EvidenceKind::JobCompleted
                            | EvidenceKind::Evaluation
                            | EvidenceKind::ArtifactVerified
                    )
            })
            .count();
        let local_direct_failures = direct
            .iter()
            .filter(|item| is_local(item) && is_negative(item.kind))
            .count();
        let indirect_negative_reports = live
            .iter()
            .filter(|item| matches!(item.kind, EvidenceKind::ViolationReport))
            .count()
            .min(self.policy.max_indirect_negative_influence);
        let independent_issuers = direct
            .iter()
            .map(|item| item.issuer)
            .filter(|issuer| *issuer != subject)
            .collect::<HashSet<_>>();

        let mut accepted_endorsements = 0usize;
        let mut endorsers = HashSet::new();
        for item in live
            .iter()
            .filter(|item| matches!(item.kind, EvidenceKind::Endorsement))
        {
            if item.issuer == subject
                || endorsers.contains(&item.issuer)
                || endorsers.len() >= self.policy.max_endorsements_per_issuer
            {
                continue;
            }
            // An endorsement contributes only when the endorser has its own
            // direct evidence. This prevents a mutually endorsing fresh
            // cluster from bootstrapping influence from zero evidence.
            let endorser_has_direct = self.entries.iter().any(|candidate| {
                candidate.subject == item.issuer
                    && candidate.issuer != item.issuer
                    && candidate.expires_at >= now
                    && is_positive(candidate.kind)
            });
            if endorser_has_direct {
                endorsers.insert(item.issuer);
                accepted_endorsements = accepted_endorsements.saturating_add(1);
            }
        }

        let remote_direct_successes = direct_successes.saturating_sub(local_direct_successes);
        let remote_direct_failures = direct_failures.saturating_sub(local_direct_failures);
        let positive_score = local_direct_successes
            .saturating_mul(15)
            .saturating_add(remote_direct_successes.saturating_mul(5))
            .saturating_add(independent_issuers.len().saturating_mul(10))
            .saturating_add(accepted_endorsements.saturating_mul(3));
        let negative_score = local_direct_failures
            .saturating_mul(20)
            .saturating_add(remote_direct_failures.saturating_mul(5))
            .saturating_add(indirect_negative_reports.saturating_mul(2));
        let score = positive_score.saturating_sub(negative_score).min(100) as u8;
        let quarantined = direct_failures >= self.policy.quarantine_negative_evidence;
        let maturity = if quarantined {
            IdentityMaturity::Quarantined
        } else if negative_score > positive_score && (direct_failures > 0 || direct_successes > 0) {
            IdentityMaturity::Degraded
        } else if direct_successes >= self.policy.trusted_direct_successes
            && independent_issuers.len() >= self.policy.trusted_independent_issuers
        {
            IdentityMaturity::TrustedLocally
        } else if direct_successes >= self.policy.established_direct_successes
            && independent_issuers.len() >= self.policy.established_independent_issuers
        {
            IdentityMaturity::Established
        } else if !direct.is_empty() {
            IdentityMaturity::Observed
        } else {
            IdentityMaturity::New
        };
        let reason = match maturity {
            IdentityMaturity::New => "no independent direct evidence".to_string(),
            IdentityMaturity::Observed => "direct evidence exists but has not matured".to_string(),
            IdentityMaturity::Established => {
                "bounded direct evidence from independent issuers".to_string()
            }
            IdentityMaturity::TrustedLocally => "local policy threshold satisfied".to_string(),
            IdentityMaturity::Degraded => {
                "negative evidence outweighs positive evidence".to_string()
            }
            IdentityMaturity::Quarantined => "local quarantine policy is active".to_string(),
        };
        TrustDecision {
            subject,
            score,
            maturity,
            direct_successes,
            direct_failures,
            local_direct_successes,
            local_direct_failures,
            independent_issuers: independent_issuers.len(),
            indirect_negative_reports,
            endorsement_influence: accepted_endorsements,
            quarantined,
            accepted_evidence: direct.len().saturating_add(accepted_endorsements),
            rejected_evidence: self.rejected,
            reason,
        }
    }

    pub fn sybil_claim_level(&self) -> SybilClaimLevel {
        SybilClaimLevel::BasicDefenses
    }
}

fn is_negative(kind: EvidenceKind) -> bool {
    matches!(
        kind,
        EvidenceKind::ProtocolViolation
            | EvidenceKind::InvalidSignature
            | EvidenceKind::ReplayAttempt
            | EvidenceKind::Equivocation
            | EvidenceKind::CorruptArtifact
            | EvidenceKind::CorruptCheckpoint
            | EvidenceKind::CapabilityFailure
            | EvidenceKind::Timeout
            | EvidenceKind::DhtPoisoning
            | EvidenceKind::TrainingUpdateRejected
    )
}

fn is_positive(kind: EvidenceKind) -> bool {
    matches!(
        kind,
        EvidenceKind::JobCompleted
            | EvidenceKind::Evaluation
            | EvidenceKind::ObservedOnline
            | EvidenceKind::ArtifactVerified
    )
}

pub fn evidence_signing_bytes(evidence: &SignedEvidence) -> Result<Vec<u8>, TrustError> {
    postcard::to_allocvec(&(
        evidence.issuer,
        evidence.issuer_public_key,
        evidence.subject,
        evidence.kind,
        evidence.sequence,
        evidence.observed_at,
        evidence.expires_at,
        &evidence.payload,
    ))
    .map_err(|error| TrustError::Invalid(error.to_string()))
}

pub fn verify_signed_evidence(evidence: &SignedEvidence) -> bool {
    if evidence.validate().is_err()
        || evidence.issuer != NodeId::from_public_key(&evidence.issuer_public_key)
    {
        return false;
    }
    let Ok(bytes) = evidence_signing_bytes(evidence) else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(&evidence.issuer_public_key) else {
        return false;
    };
    let Ok(signature) = Signature::from_slice(&evidence.signature) else {
        return false;
    };
    key.verify(&bytes, &signature).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;

    fn evidence(
        issuer: &SigningKey,
        subject: NodeId,
        kind: EvidenceKind,
        sequence: u64,
    ) -> SignedEvidence {
        let public_key = issuer.verifying_key().to_bytes();
        let mut value = SignedEvidence {
            issuer: NodeId::from_public_key(&public_key),
            issuer_public_key: public_key,
            subject,
            kind,
            sequence,
            observed_at: 10,
            expires_at: 100,
            payload: b"evidence".to_vec(),
            signature: Vec::new(),
        };
        value.signature = issuer
            .sign(&evidence_signing_bytes(&value).unwrap())
            .to_bytes()
            .to_vec();
        value
    }

    #[test]
    fn endorsement_farm_without_direct_evidence_has_no_influence() {
        let mut rng = OsRng;
        let issuer = SigningKey::generate(&mut rng);
        let attacker = SigningKey::generate(&mut rng);
        let subject = NodeId::from_bytes([9; 32]);
        let mut graph = EvidenceGraph::new(TrustPolicy::default()).unwrap();
        let mut endorsement = evidence(&attacker, subject, EvidenceKind::Endorsement, 1);
        endorsement.observed_at = 10;
        endorsement.expires_at = 100;
        endorsement.signature = attacker
            .sign(&evidence_signing_bytes(&endorsement).unwrap())
            .to_bytes()
            .to_vec();
        assert!(graph.insert(endorsement, 10).is_ok());
        assert_eq!(graph.decision(subject, 10).maturity, IdentityMaturity::New);
        assert_eq!(graph.decision(subject, 10).score, 0);
        assert!(
            graph
                .insert(
                    evidence(&issuer, subject, EvidenceKind::JobCompleted, 1),
                    10
                )
                .is_ok()
        );
        assert_eq!(
            graph.decision(subject, 10).maturity,
            IdentityMaturity::Observed
        );
    }

    #[test]
    fn stale_sequence_and_forged_signature_are_rejected() {
        let mut rng = OsRng;
        let issuer = SigningKey::generate(&mut rng);
        let subject = NodeId::from_bytes([7; 32]);
        let mut graph = EvidenceGraph::new(TrustPolicy::default()).unwrap();
        let first = evidence(&issuer, subject, EvidenceKind::Evaluation, 2);
        assert!(graph.insert(first.clone(), 10).unwrap());
        assert!(
            !graph
                .insert(evidence(&issuer, subject, EvidenceKind::Evaluation, 1), 10)
                .unwrap()
        );
        let mut forged = first;
        forged.payload = b"forged".to_vec();
        assert!(graph.insert(forged, 10).is_err());
        assert_eq!(graph.rejected_count(), 2);
    }

    #[test]
    fn one_indirect_accusation_cannot_quarantine_a_peer() {
        let mut rng = OsRng;
        let issuer = SigningKey::generate(&mut rng);
        let subject = NodeId::from_bytes([8; 32]);
        let mut graph = EvidenceGraph::new(TrustPolicy::default()).unwrap();
        assert!(
            graph
                .insert(
                    evidence(&issuer, subject, EvidenceKind::ViolationReport, 1),
                    10
                )
                .is_ok()
        );
        let decision = graph.decision(subject, 10);
        assert_eq!(decision.indirect_negative_reports, 1);
        assert_eq!(decision.maturity, IdentityMaturity::New);
        assert!(!decision.quarantined);
    }

    #[test]
    fn local_observations_outweigh_remote_reports_and_expire() {
        let mut rng = OsRng;
        let remote_issuer = SigningKey::generate(&mut rng);
        let local_issuer = SigningKey::generate(&mut rng);
        let subject = NodeId::from_bytes([6; 32]);
        let local = NodeId::from_public_key(&local_issuer.verifying_key().to_bytes());
        let mut graph = EvidenceGraph::new(TrustPolicy::default())
            .unwrap()
            .with_local_issuer(local);
        let remote = evidence(&remote_issuer, subject, EvidenceKind::JobCompleted, 1);
        assert!(graph.insert(remote, 10).is_ok());
        let remote_only = graph.decision(subject, 10);
        assert_eq!(remote_only.local_direct_successes, 0);
        assert_eq!(remote_only.direct_successes, 1);
        let local_evidence = evidence(&local_issuer, subject, EvidenceKind::Evaluation, 2);
        assert!(graph.insert(local_evidence, 10).is_ok());
        let with_local = graph.decision(subject, 10);
        assert_eq!(with_local.local_direct_successes, 1);
        assert_eq!(with_local.direct_successes, 2);
        assert_eq!(graph.decision(subject, 1_000).direct_successes, 0);
    }
}
