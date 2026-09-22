//! AI-specific manifests, inference, evaluation, and evidence-aware routing.

pub mod compute;
pub mod security;
mod training;
mod training_fabric;
mod trust;

pub use compute::{
    BackendRegistry, ComputeBackend, ComputeError, ComputeInput, ComputeOperation, ComputeOutput,
    ComputeTask, PreparedTask, backend_can_run, backend_satisfies_requirements,
    default_requirements, new_challenge, reference_backend_capability,
};
pub use security::{
    CapabilityConfidence, CriticalRole, EquivocationProof, EvaluatorCandidate, EvaluatorSelection,
    LocalSecurityState, PeerObservation, RobustAggregationReport, SecurityError, SecurityEvent,
    SecurityEventKind, SecurityPolicy, SecurityProfile, SignedStateClaim, TrainingAdmission,
    TrainingContext, TrainingContribution, TrustAssessment, aggregate_v6_updates,
    claim_signing_bytes, select_evaluators, verify_claim,
};
pub use training::{
    HardwareKind, PlannerError, StrategySupport, TrainingPlanDecision, TrainingStrategy,
    WorkerProfile, plan_training,
};
pub use training_fabric::{
    V4BranchDecision, V4ByzantineReport, V4CommunicationEstimate, V4PlanDecision, V4PlanRequest,
    V4PlannerError, V4PlannerObjective, V4RebalanceDecision, V4TopologyLink, aggregate_updates,
    communication_estimate, hash_plan, plan_v4, rebalance_shard, reconcile_branches,
};
pub use trust::{
    EvidenceGraph, IdentityMaturity, SybilClaimLevel, TrustDecision, TrustError, TrustPolicy,
    evidence_signing_bytes, verify_signed_evidence,
};

use intelligence_protocol::{
    ArtifactId, Capability, CapabilityEvidence, Evidence, JobId, JobKind, JobRequest,
    ModelManifest, NodeId, PeerRecord, PrivacyPolicy, ResourceLimits,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum IntelligenceError {
    #[error("inference input is not valid UTF-8 or JSON text")]
    InvalidInput,
    #[error("no peer advertises required capability")]
    NoRoute,
    #[error("evaluation output is not a built-in classifier result")]
    InvalidEvaluationOutput,
    #[error("intelligence value is invalid: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityObservation {
    pub node: NodeId,
    pub capability: String,
    pub success: bool,
    pub latency_ms: u32,
    pub observed_at: u64,
    pub result_hash: Option<ArtifactId>,
}

#[derive(Clone, Debug)]
pub struct CapabilityGraph {
    observations: Vec<CapabilityObservation>,
    max_observations: usize,
}

impl CapabilityGraph {
    pub fn new(max_observations: usize) -> Result<Self, IntelligenceError> {
        if max_observations == 0 {
            return Err(IntelligenceError::Invalid(
                "capability observation bound must be positive".to_string(),
            ));
        }
        Ok(Self {
            observations: Vec::new(),
            max_observations,
        })
    }

    pub fn record(&mut self, observation: CapabilityObservation) {
        if self.observations.len() == self.max_observations {
            self.observations.remove(0);
        }
        self.observations.push(observation);
    }

    pub fn observations(&self) -> &[CapabilityObservation] {
        &self.observations
    }

    pub fn success_rate(&self, node: NodeId, capability: &str) -> Option<f32> {
        let mut successes = 0u32;
        let mut total = 0u32;
        for observation in self
            .observations
            .iter()
            .filter(|value| value.node == node && value.capability == capability)
        {
            total = total.saturating_add(1);
            successes = successes.saturating_add(u32::from(observation.success));
        }
        (total > 0).then_some(successes as f32 / total as f32)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteChoice {
    pub peer: NodeId,
    pub address: String,
    pub score: i64,
}

pub fn choose_peer(
    peers: &[PeerRecord],
    capability: &str,
    now: u64,
    avoid: Option<NodeId>,
) -> Option<RouteChoice> {
    peers
        .iter()
        .filter(|peer| Some(peer.node_id) != avoid && peer.expires_at >= now)
        .filter_map(|peer| {
            let advertised = peer
                .capabilities
                .iter()
                .find(|item| item.name == capability)?;
            let evidence_score = match advertised.evidence {
                CapabilityEvidence::Claimed => 0,
                CapabilityEvidence::Observed {
                    successful_jobs,
                    failed_jobs,
                    ..
                } => {
                    let total = successful_jobs.saturating_add(failed_jobs).max(1);
                    (successful_jobs as i64 * 100) / total as i64
                }
                CapabilityEvidence::Verified { .. } => 200,
            };
            let latency_penalty = peer.observed_latency_ms.unwrap_or(5000).min(5000) as i64;
            let address = peer.addresses.first()?.clone();
            Some(RouteChoice {
                peer: peer.node_id,
                address,
                score: evidence_score * 10 - latency_penalty,
            })
        })
        .max_by_key(|choice| (choice.score, choice.peer))
}

pub fn builtin_model_manifest() -> ModelManifest {
    let identity = "builtin.tiny-sentiment.v1".to_string();
    let format = "sparse-linear-text-v1".to_string();
    let artifact = ArtifactId::from_bytes_hashed(b"intelligence-network/builtin.tiny-sentiment.v1");
    ModelManifest {
        artifact,
        identity,
        format,
        size: 512,
        weights: Vec::new(),
        capabilities: vec!["inference.text".to_string(), "evaluation.text".to_string()],
        runtime_requirements: vec![intelligence_protocol::MetadataEntry {
            key: "backend".to_string(),
            value: "native-cpu".to_string(),
        }],
        adapters: Vec::new(),
        local_path: None,
        local_only: false,
    }
}

pub fn builtin_capability(expires_at: u64) -> Capability {
    Capability {
        name: "inference.text".to_string(),
        version: 1,
        model: Some("builtin.tiny-sentiment.v1".to_string()),
        resources: ResourceLimits {
            max_input_bytes: 64 * 1024,
            max_output_bytes: 16 * 1024,
            memory_bytes: 16 * 1024 * 1024,
            cpu_millis: 1000,
        },
        evidence: CapabilityEvidence::Claimed,
        expires_at,
        metadata: vec![intelligence_protocol::MetadataEntry {
            key: "model_format".to_string(),
            value: "sparse-linear-text-v1".to_string(),
        }],
        compute_backends: Vec::new(),
    }
}

pub fn make_inference_job(
    job_id: JobId,
    origin: NodeId,
    capability: impl Into<String>,
    input: impl Into<Vec<u8>>,
    deadline_ms: u64,
    max_output_bytes: u32,
    privacy: PrivacyPolicy,
) -> Result<JobRequest, IntelligenceError> {
    let request = JobRequest {
        job_id,
        origin,
        kind: JobKind::Inference,
        capability: capability.into(),
        model: None,
        input: input.into(),
        deadline_ms,
        max_output_bytes,
        privacy,
    };
    request
        .validate()
        .map_err(|error| IntelligenceError::Invalid(error.to_string()))?;
    Ok(request)
}

pub fn make_evaluation_job(
    job_id: JobId,
    origin: NodeId,
    text: &str,
    expected_label: &str,
    deadline_ms: u64,
) -> Result<JobRequest, IntelligenceError> {
    if text.is_empty() || expected_label.is_empty() {
        return Err(IntelligenceError::InvalidInput);
    }
    let input = serde_json::to_vec(&serde_json::json!({
        "text": text,
        "expected_label": expected_label,
    }))
    .map_err(|error| IntelligenceError::Invalid(error.to_string()))?;
    let request = JobRequest {
        job_id,
        origin,
        kind: JobKind::Evaluation,
        capability: "evaluation.text".to_string(),
        model: Some(builtin_model_manifest().artifact),
        input,
        deadline_ms,
        max_output_bytes: 16 * 1024,
        privacy: PrivacyPolicy::default(),
    };
    request
        .validate()
        .map_err(|error| IntelligenceError::Invalid(error.to_string()))?;
    Ok(request)
}

pub fn score_builtin_output(
    evaluator: NodeId,
    output: &[u8],
    expected_label: &str,
    observed_at: u64,
) -> Result<Evidence, IntelligenceError> {
    let value: serde_json::Value =
        serde_json::from_slice(output).map_err(|_| IntelligenceError::InvalidEvaluationOutput)?;
    let label = value
        .get("label")
        .and_then(serde_json::Value::as_str)
        .ok_or(IntelligenceError::InvalidEvaluationOutput)?;
    let score = if label == expected_label { 100 } else { 0 };
    let evaluation_id =
        ArtifactId::from_bytes_hashed(&[expected_label.as_bytes(), output].concat());
    Ok(Evidence {
        evaluator,
        evaluation_id,
        score,
        score_scale: "percent-correct".to_string(),
        result_hash: ArtifactId::from_bytes_hashed(output),
        observed_at,
        verified: true,
    })
}

pub fn public_capability_map(capabilities: &[Capability]) -> BTreeMap<String, Capability> {
    capabilities
        .iter()
        .cloned()
        .map(|capability| (capability.name.clone(), capability))
        .collect()
}

pub fn hash_manifest(manifest: &ModelManifest) -> Result<ArtifactId, IntelligenceError> {
    let bytes = postcard::to_allocvec(manifest)
        .map_err(|error| IntelligenceError::Invalid(error.to_string()))?;
    Ok(ArtifactId::from_bytes_hashed(&bytes))
}

pub fn now_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            duration.as_millis().min(u64::MAX as u128) as u64
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use intelligence_protocol::CapabilityEvidence;

    #[test]
    fn route_score_prefers_verified_capability_over_claim() {
        let mut claimed = PeerRecord {
            node_id: NodeId::from_bytes([1; 32]),
            public_key: [1; 32],
            addresses: vec!["127.0.0.1:1".to_string()],
            capabilities: vec![builtin_capability(100)],
            announced_at: 1,
            expires_at: 100,
            observed_latency_ms: Some(1),
        };
        claimed.capabilities[0].evidence = CapabilityEvidence::Claimed;
        let mut verified = claimed.clone();
        verified.node_id = NodeId::from_bytes([2; 32]);
        verified.capabilities[0].evidence = CapabilityEvidence::Verified {
            evaluation_id: ArtifactId::from_bytes([9; 32]),
            score_basis: "test".to_string(),
            observed_at: 2,
        };
        let selected = choose_peer(&[claimed, verified], "inference.text", 2, None).unwrap();
        assert_eq!(selected.peer, NodeId::from_bytes([2; 32]));
    }

    #[test]
    fn evaluation_is_content_addressed() {
        let evidence = score_builtin_output(
            NodeId::from_bytes([1; 32]),
            br#"{"label":"positive"}"#,
            "positive",
            1,
        )
        .unwrap();
        assert_eq!(evidence.score, 100);
        assert_eq!(
            evidence.result_hash,
            ArtifactId::from_bytes_hashed(br#"{"label":"positive"}"#)
        );
    }
}
