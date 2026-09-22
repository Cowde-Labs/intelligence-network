//! Training-fabric planning and small, deterministic reference algorithms.
//!
//! These functions deliberately operate on bounded metadata and integer
//! reference tensors.  They are useful for exercising the distributed
//! protocol without pulling an ML framework into the network core.  A plan is
//! explicit about what is supported; it is never inferred from a strategy
//! name alone. Public `V4*` types remain versioned because they are part of the
//! protocol and persistent-state compatibility surface.

use crate::compute::{backend_satisfies_requirements, reference_backend_capability};
use crate::training::{HardwareKind, WorkerProfile};
use intelligence_protocol::{
    ArtifactId, BackendAssignment, BackendKind, ComputeRequirements, DataLocality, JobId, NodeId,
    V4AcceleratorCapability, V4AcceleratorFamily, V4AggregationGroup, V4ByzantinePolicy,
    V4NumericalFormat, V4ParallelismStrategy, V4ShardLifecycle, V4ShardOwnership, V4SupportLevel,
    V4TrainingBranch, V4TrainingPlan, V4WorkerCapability,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const MAX_PLANNER_WORKERS: usize = 64;
const MAX_GROUP_SIZE: usize = 4;
const MAX_TENSOR_DEGREE: u16 = 8;
const MAX_PIPELINE_STAGES: u16 = 16;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct V4PlannerObjective {
    pub throughput_weight: u16,
    pub bandwidth_weight: u16,
    pub fault_tolerance_weight: u16,
    pub locality_weight: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4TopologyLink {
    pub from: NodeId,
    pub to: NodeId,
    pub rtt_ms: u32,
    pub bandwidth_mbps: u32,
    pub reliability: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4PlanRequest {
    pub job_id: JobId,
    pub proposer: NodeId,
    pub model_bytes: u64,
    pub requested_workers: usize,
    pub strategy: V4ParallelismStrategy,
    pub tensor_degree: u16,
    pub pipeline_stages: u16,
    pub local_steps: u16,
    pub max_staleness: u16,
    pub checkpoint_replication: u16,
    pub data_locality: DataLocality,
    pub workers: Vec<WorkerProfile>,
    pub links: Vec<V4TopologyLink>,
    pub objective: V4PlannerObjective,
    #[serde(default)]
    pub backend_requirements: Vec<ComputeRequirements>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4PlanDecision {
    pub plan: V4TrainingPlan,
    pub selected_workers: Vec<NodeId>,
    pub rejected_workers: Vec<NodeId>,
    pub explanations: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4RebalanceDecision {
    pub shard_id: u16,
    pub from: NodeId,
    pub to: NodeId,
    pub accepted: bool,
    pub reason: String,
    pub next_ownership_generation: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4BranchDecision {
    pub accepted: bool,
    pub branch: Option<ArtifactId>,
    pub value: Option<i64>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4ByzantineReport {
    pub policy: V4ByzantinePolicy,
    pub aggregate: Vec<i64>,
    pub accepted_updates: usize,
    pub rejected_updates: usize,
    pub coordinate_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct V4CommunicationEstimate {
    pub parameter_count: u64,
    pub workers: u64,
    pub groups: u64,
    pub precision_bytes: u8,
    pub tensor_degree: u16,
    pub pipeline_stages: u16,
    pub model_bytes: u64,
    pub optimizer_bytes: u64,
    pub activation_bytes_per_microbatch: u64,
    pub local_update_bytes_per_window: u64,
    pub cross_group_bytes_per_window: u64,
    pub checkpoint_bytes: u64,
    pub notes: String,
}

#[derive(Debug, Error)]
pub enum V4PlannerError {
    #[error("V4 model size must be positive")]
    InvalidModelSize,
    #[error("V4 plan requires between two and {0} workers")]
    InvalidWorkerCount(usize),
    #[error("V4 plan requires at least two eligible workers")]
    InsufficientWorkers,
    #[error("V4 tensor degree is unsupported")]
    InvalidTensorDegree,
    #[error("V4 pipeline stage count is unsupported")]
    InvalidPipelineStages,
    #[error("V4 strategy parameters are not supported by the reference runtime")]
    UnsupportedStrategy,
    #[error("V4 topology link is invalid or references an unknown worker")]
    InvalidTopology,
    #[error("V4 local steps must be positive")]
    InvalidLocalSteps,
    #[error("V4 checkpoint replication is invalid")]
    InvalidReplication,
    #[error("V4 worker profile contains invalid reliability")]
    InvalidWorkerProfile,
    #[error("V4 update vectors must be non-empty, equal length, and bounded")]
    InvalidUpdates,
    #[error("V4 branch histories are not compatible")]
    IncompatibleBranches,
    #[error("unsupported backend assignment: {0}")]
    UnsupportedBackendAssignment(String),
}

pub fn plan_v4(request: &V4PlanRequest) -> Result<V4PlanDecision, V4PlannerError> {
    if request.model_bytes == 0 {
        return Err(V4PlannerError::InvalidModelSize);
    }
    if !(2..=MAX_PLANNER_WORKERS).contains(&request.requested_workers) {
        return Err(V4PlannerError::InvalidWorkerCount(MAX_PLANNER_WORKERS));
    }
    if request.tensor_degree > MAX_TENSOR_DEGREE {
        return Err(V4PlannerError::InvalidTensorDegree);
    }
    if request.pipeline_stages > MAX_PIPELINE_STAGES {
        return Err(V4PlannerError::InvalidPipelineStages);
    }
    if request.local_steps == 0 {
        return Err(V4PlannerError::InvalidLocalSteps);
    }
    if request.checkpoint_replication == 0 || request.checkpoint_replication > 16 {
        return Err(V4PlannerError::InvalidReplication);
    }
    if request.workers.is_empty() {
        return Err(V4PlannerError::InsufficientWorkers);
    }
    if request.backend_requirements.len() > intelligence_protocol::MAX_V5_REQUIREMENTS {
        return Err(V4PlannerError::UnsupportedBackendAssignment(
            "backend requirement list exceeds the bounded planner limit".to_string(),
        ));
    }
    let worker_ids = request
        .workers
        .iter()
        .map(|worker| worker.node)
        .collect::<std::collections::HashSet<_>>();
    if worker_ids.len() != request.workers.len()
        || request.workers.iter().any(|worker| {
            !worker.reliability.is_finite()
                || !(0.0..=1.0).contains(&worker.reliability)
                || worker.memory_bytes == 0
                || worker.compute_units == 0
        })
    {
        return Err(V4PlannerError::InvalidWorkerProfile);
    }
    if request.links.len() > MAX_PLANNER_WORKERS.saturating_mul(MAX_PLANNER_WORKERS)
        || request.links.iter().any(|link| {
            link.from == link.to
                || !worker_ids.contains(&link.from)
                || !worker_ids.contains(&link.to)
                || link.rtt_ms == 0
                || link.bandwidth_mbps == 0
                || !link.reliability.is_finite()
                || !(0.0..=1.0).contains(&link.reliability)
        })
    {
        return Err(V4PlannerError::InvalidTopology);
    }

    let worker_capabilities = request
        .workers
        .iter()
        .map(worker_capability)
        .collect::<Vec<_>>();
    for requirement in &request.backend_requirements {
        let candidates = worker_capabilities
            .iter()
            .filter(|worker| {
                worker
                    .backends
                    .iter()
                    .any(|backend| backend_satisfies_requirements(backend, requirement))
            })
            .count();
        if candidates == 0 {
            return Err(V4PlannerError::UnsupportedBackendAssignment(format!(
                "no advertised worker satisfies {:?} / kernel {}",
                requirement.task_kind, requirement.kernel_id
            )));
        }
    }
    let requested = request.requested_workers.min(request.workers.len());
    let shard_memory = request
        .model_bytes
        .div_ceil(requested.max(2) as u64)
        .saturating_mul(2);
    let mut eligible = request
        .workers
        .iter()
        .filter(|worker| {
            worker.memory_bytes >= shard_memory
                && worker.compute_units > 0
                && worker.reliability.is_finite()
                && worker.reliability >= 0.5
                && (!matches!(&request.data_locality, DataLocality::LocalOnly)
                    || worker.dataset_available)
        })
        .cloned()
        .collect::<Vec<_>>();
    if eligible.len() < 2 {
        return Err(V4PlannerError::InsufficientWorkers);
    }
    eligible.sort_by(|left, right| {
        let left_score = worker_score(left, request.objective);
        let right_score = worker_score(right, request.objective);
        right_score
            .partial_cmp(&left_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                topology_score(right.node, &request.links)
                    .cmp(&topology_score(left.node, &request.links))
            })
            .then_with(|| left.node.cmp(&right.node))
    });
    eligible.truncate(request.requested_workers);
    if eligible.len() < 2 {
        return Err(V4PlannerError::InsufficientWorkers);
    }
    for requirement in &request.backend_requirements {
        if !eligible.iter().any(|worker| {
            worker_capability(worker)
                .backends
                .iter()
                .any(|backend| backend_satisfies_requirements(backend, requirement))
        }) {
            return Err(V4PlannerError::UnsupportedBackendAssignment(format!(
                "selected workers cannot execute kernel {}",
                requirement.kernel_id
            )));
        }
    }
    let selected_workers = eligible
        .iter()
        .map(|worker| worker.node)
        .collect::<Vec<_>>();
    let selected_set = selected_workers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let rejected_workers = request
        .workers
        .iter()
        .filter(|worker| !selected_set.contains(&worker.node))
        .map(|worker| worker.node)
        .collect::<Vec<_>>();

    let groups = topology_groups(&selected_workers, &request.links)
        .into_iter()
        .enumerate()
        .map(|(group_index, members)| {
            let aggregator = members[0];
            let parent = (group_index > 0).then_some(selected_workers[0]);
            V4AggregationGroup {
                group_id: group_index as u16,
                members,
                aggregator,
                parent,
            }
        })
        .collect::<Vec<_>>();

    let mut shards = Vec::with_capacity(selected_workers.len());
    for (index, owner) in selected_workers.iter().copied().enumerate() {
        let replica = selected_workers
            .iter()
            .copied()
            .find(|candidate| *candidate != owner);
        let state_bytes = request.model_bytes.div_ceil(selected_workers.len() as u64);
        let content_hash = ArtifactId::from_bytes_hashed(
            format!("v4:{}:{}:{}", request.job_id, index, request.model_bytes).as_bytes(),
        );
        shards.push(V4ShardOwnership {
            shard_id: index as u16,
            model_generation: 1,
            owners: vec![owner],
            replicas: replica.into_iter().collect(),
            ownership_generation: 1,
            content_hash,
            state_bytes: state_bytes.max(1),
            memory_bytes: state_bytes.saturating_mul(2).max(1),
            runtime_requirement: "reference.cpu.i64".to_string(),
            lifecycle: V4ShardLifecycle::Active,
        });
    }

    let worker_capabilities = eligible.iter().map(worker_capability).collect::<Vec<_>>();
    let backend_assignments = request
        .backend_requirements
        .iter()
        .map(|requirement| {
            let candidate = eligible
                .iter()
                .filter_map(|worker| {
                    let capability = worker_capability(worker);
                    let backend = capability
                        .backends
                        .iter()
                        .find(|backend| backend_satisfies_requirements(backend, requirement))?;
                    Some((worker, backend.kind))
                })
                .min_by_key(|(worker, _)| (worker.rtt_ms, worker.node))
                .ok_or_else(|| {
                    V4PlannerError::UnsupportedBackendAssignment(format!(
                        "selected workers cannot execute kernel {}",
                        requirement.kernel_id
                    ))
                })?;
            Ok(BackendAssignment {
                task_kind: requirement.task_kind,
                worker: candidate.0.node,
                backend: candidate.1,
                device_index: 0,
                requirements: requirement.clone(),
            })
        })
        .collect::<Result<Vec<_>, V4PlannerError>>()?;
    let support = strategy_support(
        request.strategy,
        request.tensor_degree,
        request.pipeline_stages,
    );
    if support == V4SupportLevel::Unsupported {
        return Err(V4PlannerError::UnsupportedStrategy);
    }
    let rationale = format!(
        "selected {} workers with shard memory >= {} bytes; groups use <= {} members; local SGD and bounded generation transitions avoid a global per-step barrier",
        selected_workers.len(),
        shard_memory,
        MAX_GROUP_SIZE
    );
    let mut plan = V4TrainingPlan {
        job_id: request.job_id,
        proposer: request.proposer,
        plan_generation: 1,
        model_generation: 1,
        training_epoch: 1,
        membership_epoch: 1,
        coordination_term: 1,
        branch: ArtifactId::from_bytes_hashed(format!("v4-branch:{}:1", request.job_id).as_bytes()),
        strategy: request.strategy,
        support,
        workers: selected_workers.clone(),
        groups,
        shards,
        tensor_degree: request.tensor_degree,
        pipeline_stages: request.pipeline_stages,
        local_steps: request.local_steps,
        max_staleness: request.max_staleness,
        checkpoint_replication: request
            .checkpoint_replication
            .min(selected_workers.len() as u16),
        data_locality: request.data_locality.clone(),
        accelerator: None,
        worker_capabilities,
        rationale,
        plan_hash: ArtifactId::from_bytes([0; 32]),
        parent_plan_hash: None,
        compute_requirements: request.backend_requirements.clone(),
        backend_assignments,
    };
    plan.plan_hash = hash_plan(&plan);

    let explanations = eligible
        .iter()
        .map(|worker| {
            format!(
                "{} selected: memory={} >= shard_memory={}, hardware={:?}, RTT={}ms, bandwidth={}Mbps, reliability={:.2}, topology_score={}",
                worker.node,
                worker.memory_bytes,
                shard_memory,
                worker.hardware,
                worker.rtt_ms,
                worker.throughput_mbps,
                worker.reliability,
                topology_score(worker.node, &request.links),
            )
        })
        .collect();
    Ok(V4PlanDecision {
        plan,
        selected_workers,
        rejected_workers,
        explanations,
    })
}

pub fn hash_plan(plan: &V4TrainingPlan) -> ArtifactId {
    let mut unsigned = plan.clone();
    unsigned.plan_hash = ArtifactId::from_bytes([0; 32]);
    let encoded = postcard::to_allocvec(&unsigned)
        .unwrap_or_else(|_| b"intelligence-network:v4-plan:serialization-error".to_vec());
    ArtifactId::from_bytes_hashed(&encoded)
}

#[allow(clippy::too_many_arguments)]
pub fn rebalance_shard(
    shard: &V4ShardOwnership,
    candidate: &WorkerProfile,
    now_ms: u64,
    last_migration_ms: Option<u64>,
    cooldown_ms: u64,
    minimum_gain: f32,
    source_score: f32,
    candidate_score: f32,
) -> V4RebalanceDecision {
    let reason = if shard.lifecycle != V4ShardLifecycle::Active {
        "shard is not active".to_string()
    } else if shard.owners.is_empty() {
        "active shard has no current owner".to_string()
    } else if shard.owners.contains(&candidate.node) || shard.replicas.contains(&candidate.node) {
        "candidate already stores the shard".to_string()
    } else if candidate.memory_bytes < shard.memory_bytes {
        "candidate cannot fit shard memory requirement".to_string()
    } else if last_migration_ms.is_some_and(|last| now_ms.saturating_sub(last) < cooldown_ms) {
        "migration cooldown has not elapsed".to_string()
    } else if source_score - candidate_score < minimum_gain {
        "candidate does not provide the configured hysteresis gain".to_string()
    } else {
        String::new()
    };
    V4RebalanceDecision {
        shard_id: shard.shard_id,
        from: shard
            .owners
            .first()
            .copied()
            .unwrap_or_else(|| NodeId::from_bytes([0; 32])),
        to: candidate.node,
        accepted: reason.is_empty(),
        reason: if reason.is_empty() {
            "candidate improves placement under current topology and policy".to_string()
        } else {
            reason
        },
        next_ownership_generation: shard.ownership_generation.saturating_add(1),
    }
}

pub fn reconcile_branches(
    left: &V4TrainingBranch,
    right: &V4TrainingBranch,
    policy: intelligence_protocol::V4ReconciliationPolicy,
    left_value: i64,
    right_value: i64,
) -> Result<V4BranchDecision, V4PlannerError> {
    let compatible = left.job_id == right.job_id
        && left.parent_generation == right.parent_generation
        && left.model_generation == right.model_generation
        && left.optimizer_generation == right.optimizer_generation
        && left.dataset_progress == right.dataset_progress
        && left.plan_generation == right.plan_generation
        && left.branch != right.branch;
    if !compatible {
        return match policy {
            intelligence_protocol::V4ReconciliationPolicy::AbortBranch
            | intelligence_protocol::V4ReconciliationPolicy::ManualOperatorRequired
            | intelligence_protocol::V4ReconciliationPolicy::SelectBranch => Ok(V4BranchDecision {
                accepted: false,
                branch: None,
                value: None,
                reason: "incompatible branch lineage requires explicit operator policy".to_string(),
            }),
            _ => Err(V4PlannerError::IncompatibleBranches),
        };
    }
    match policy {
        intelligence_protocol::V4ReconciliationPolicy::AverageCompatibleState
        | intelligence_protocol::V4ReconciliationPolicy::MergeLocalSgdState => {
            let merged = ((left_value as i128 + right_value as i128) / 2) as i64;
            let branch = ArtifactId::from_bytes_hashed(
                format!(
                    "v4-merge:{}:{}:{}",
                    left.branch, right.branch, left.parent_generation
                )
                .as_bytes(),
            );
            Ok(V4BranchDecision {
                accepted: true,
                branch: Some(branch),
                value: Some(merged),
                reason: "compatible local-SGD branches with equal data progress merged from a common generation".to_string(),
            })
        }
        intelligence_protocol::V4ReconciliationPolicy::SelectBranch => Ok(V4BranchDecision {
            accepted: true,
            branch: Some(left.branch),
            value: Some(left_value),
            reason: "operator-selected branch retained; the alternate branch was not merged"
                .to_string(),
        }),
        intelligence_protocol::V4ReconciliationPolicy::AbortBranch
        | intelligence_protocol::V4ReconciliationPolicy::ManualOperatorRequired => {
            Ok(V4BranchDecision {
                accepted: false,
                branch: None,
                value: None,
                reason: "policy does not permit automatic branch reconciliation".to_string(),
            })
        }
    }
}

pub fn aggregate_updates(
    policy: V4ByzantinePolicy,
    updates: &[Vec<i64>],
    clip_abs: i64,
) -> Result<V4ByzantineReport, V4PlannerError> {
    if updates.is_empty()
        || updates.len() > 64
        || updates[0].is_empty()
        || updates[0].len() > 128
        || clip_abs <= 0
        || updates
            .iter()
            .any(|update| update.len() != updates[0].len())
    {
        return Err(V4PlannerError::InvalidUpdates);
    }
    let rejected = updates
        .iter()
        .filter(|update| {
            update
                .iter()
                .any(|value| value.unsigned_abs() > clip_abs as u64)
        })
        .count();
    let accepted_updates = updates
        .iter()
        .filter(|update| {
            update
                .iter()
                .all(|value| value.unsigned_abs() <= clip_abs as u64)
        })
        .collect::<Vec<_>>();
    if accepted_updates.is_empty() {
        return Err(V4PlannerError::InvalidUpdates);
    }
    let width = updates[0].len();
    let mut aggregate = Vec::with_capacity(width);
    for coordinate in 0..width {
        let mut values = accepted_updates
            .iter()
            .map(|update| update[coordinate].clamp(-clip_abs, clip_abs))
            .collect::<Vec<_>>();
        let value = match policy {
            V4ByzantinePolicy::Mean | V4ByzantinePolicy::ClippedMean => {
                let total = values.iter().map(|value| *value as i128).sum::<i128>();
                (total / values.len() as i128) as i64
            }
            V4ByzantinePolicy::TrimmedMean => {
                if values.len() >= 3 {
                    values.sort_unstable();
                    let total = values[1..values.len() - 1]
                        .iter()
                        .map(|value| *value as i128)
                        .sum::<i128>();
                    (total / (values.len() - 2) as i128) as i64
                } else {
                    let total = values.iter().map(|value| *value as i128).sum::<i128>();
                    (total / values.len() as i128) as i64
                }
            }
            V4ByzantinePolicy::CoordinateMedian => {
                values.sort_unstable();
                if values.len() % 2 == 0 {
                    let middle = values.len() / 2;
                    (i128::from(values[middle - 1]) + i128::from(values[middle]))
                        .checked_div(2)
                        .unwrap_or_default() as i64
                } else {
                    values[values.len() / 2]
                }
            }
        };
        aggregate.push(value.clamp(-clip_abs, clip_abs));
    }
    Ok(V4ByzantineReport {
        policy,
        aggregate,
        accepted_updates: accepted_updates.len(),
        rejected_updates: rejected,
        coordinate_count: width,
    })
}

pub fn communication_estimate(
    parameter_count: u64,
    workers: u64,
    groups: u64,
    precision_bytes: u8,
    tensor_degree: u16,
    pipeline_stages: u16,
) -> V4CommunicationEstimate {
    let workers = workers.max(1);
    let groups = groups.clamp(1, workers);
    let precision = u64::from(precision_bytes.max(1));
    let model_bytes = parameter_count.saturating_mul(precision);
    let optimizer_bytes = model_bytes.saturating_mul(2);
    let activation_bytes_per_microbatch = parameter_count
        .saturating_div(u64::from(pipeline_stages.max(1)))
        .saturating_mul(precision);
    let local_update_bytes_per_window = model_bytes.saturating_div(workers).max(1);
    let cross_group_bytes_per_window = model_bytes
        .saturating_mul(groups.saturating_sub(1))
        .saturating_div(groups)
        .max(1);
    V4CommunicationEstimate {
        parameter_count,
        workers,
        groups,
        precision_bytes,
        tensor_degree,
        pipeline_stages,
        model_bytes,
        optimizer_bytes,
        activation_bytes_per_microbatch,
        local_update_bytes_per_window,
        cross_group_bytes_per_window,
        checkpoint_bytes: model_bytes.saturating_add(optimizer_bytes),
        notes: "modeled byte counts; no tensors are allocated".to_string(),
    }
}

fn worker_score(worker: &WorkerProfile, objective: V4PlannerObjective) -> f32 {
    let throughput = worker.throughput_mbps.max(1) as f32;
    let latency = 1.0 / worker.rtt_ms.max(1) as f32;
    throughput * f32::from(objective.throughput_weight.max(1))
        + latency * f32::from(objective.bandwidth_weight.max(1)) * 100.0
        + worker.reliability * f32::from(objective.fault_tolerance_weight.max(1))
        + f32::from(u8::from(worker.dataset_available))
            * f32::from(objective.locality_weight.max(1))
}

fn topology_score(node: NodeId, links: &[V4TopologyLink]) -> u64 {
    links
        .iter()
        .filter(|link| link.from == node || link.to == node)
        .map(|link| {
            let reliability_permille = (link.reliability * 1000.0).round() as u64;
            u64::from(link.bandwidth_mbps)
                .saturating_mul(reliability_permille)
                .saturating_div(u64::from(link.rtt_ms.max(1)))
        })
        .sum()
}

fn link_quality(left: NodeId, right: NodeId, links: &[V4TopologyLink]) -> u64 {
    links
        .iter()
        .find(|link| {
            (link.from == left && link.to == right) || (link.from == right && link.to == left)
        })
        .map_or(0, |link| {
            let reliability_permille = (link.reliability * 1000.0).round() as u64;
            u64::from(link.bandwidth_mbps)
                .saturating_mul(reliability_permille)
                .saturating_div(u64::from(link.rtt_ms.max(1)))
        })
}

fn topology_groups(selected: &[NodeId], links: &[V4TopologyLink]) -> Vec<Vec<NodeId>> {
    let mut remaining = selected.to_vec();
    let mut groups = Vec::with_capacity(selected.len().div_ceil(MAX_GROUP_SIZE));
    while !remaining.is_empty() {
        let first = remaining.remove(0);
        let mut members = vec![first];
        while members.len() < MAX_GROUP_SIZE && !remaining.is_empty() {
            let candidate_index = remaining
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    let left_quality = members
                        .iter()
                        .map(|member| link_quality(*member, **left, links))
                        .sum::<u64>();
                    let right_quality = members
                        .iter()
                        .map(|member| link_quality(*member, **right, links))
                        .sum::<u64>();
                    left_quality
                        .cmp(&right_quality)
                        .then_with(|| right.cmp(left))
                })
                .map(|(index, _)| index)
                .unwrap_or(0);
            members.push(remaining.remove(candidate_index));
        }
        groups.push(members);
    }
    groups
}

fn strategy_support(
    strategy: V4ParallelismStrategy,
    tensor_degree: u16,
    pipeline_stages: u16,
) -> V4SupportLevel {
    match strategy {
        V4ParallelismStrategy::LocalSgd => V4SupportLevel::Supported,
        V4ParallelismStrategy::TensorParallel if tensor_degree == 2 => V4SupportLevel::Supported,
        // The node currently ships a two-stage pipeline reference.  Do not
        // advertise an arbitrary stage count as executable until the wire
        // path carries and validates a complete stage route.
        V4ParallelismStrategy::PipelineParallel if pipeline_stages == 2 => {
            V4SupportLevel::Supported
        }
        V4ParallelismStrategy::Hybrid if tensor_degree == 2 && pipeline_stages == 2 => {
            V4SupportLevel::Experimental
        }
        _ => V4SupportLevel::Unsupported,
    }
}

fn worker_capability(worker: &WorkerProfile) -> V4WorkerCapability {
    let family = match worker.hardware {
        HardwareKind::Cpu => V4AcceleratorFamily::Cpu,
        HardwareKind::AppleSilicon => V4AcceleratorFamily::Metal,
        HardwareKind::AmdGpu => V4AcceleratorFamily::Rocm,
        HardwareKind::NvidiaConsumer
        | HardwareKind::A100
        | HardwareKind::H100
        | HardwareKind::B200 => V4AcceleratorFamily::Cuda,
        HardwareKind::Unknown => V4AcceleratorFamily::Cpu,
    };
    V4WorkerCapability {
        node: worker.node,
        accelerator: V4AcceleratorCapability {
            family,
            device_model: format!("{:?}", worker.hardware),
            device_count: 1,
            memory_bytes: worker.memory_bytes,
            formats: vec![V4NumericalFormat::I8, V4NumericalFormat::F32],
            runtime: "reference".to_string(),
            runtime_version: "v4".to_string(),
            physical_verified: matches!(worker.hardware, HardwareKind::Cpu),
        },
        memory_bytes: worker.memory_bytes,
        compute_units: worker.compute_units,
        rtt_ms: worker.rtt_ms,
        bandwidth_mbps: worker.throughput_mbps,
        reliability_permille: (worker.reliability.clamp(0.0, 1.0) * 1000.0).round() as u16,
        backends: if worker.backends.is_empty() {
            // Legacy V4 hardware labels are not V5 capability evidence.  A
            // peer must explicitly advertise a typed backend before the V5
            // planner may bind accelerator work to it.
            vec![reference_backend_capability(
                BackendKind::Cpu,
                worker.memory_bytes,
                true,
            )]
        } else {
            worker.backends.clone()
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(value: u8, memory_bytes: u64) -> WorkerProfile {
        WorkerProfile {
            node: NodeId::from_bytes([value; 32]),
            hardware: HardwareKind::Cpu,
            memory_bytes,
            compute_units: 2,
            rtt_ms: 10 + u32::from(value),
            throughput_mbps: 100,
            reliability: 0.99,
            dataset_available: true,
            backends: Vec::new(),
        }
    }

    #[test]
    fn plan_is_sharded_and_explainable() {
        let request = V4PlanRequest {
            job_id: JobId::from_bytes([1; 16]),
            proposer: NodeId::from_bytes([9; 32]),
            model_bytes: 2048,
            requested_workers: 4,
            strategy: V4ParallelismStrategy::Hybrid,
            tensor_degree: 2,
            pipeline_stages: 2,
            local_steps: 2,
            max_staleness: 2,
            checkpoint_replication: 2,
            data_locality: DataLocality::Selective,
            workers: (1..=4).map(|id| worker(id, 2048)).collect(),
            links: Vec::new(),
            objective: V4PlannerObjective {
                throughput_weight: 2,
                bandwidth_weight: 2,
                fault_tolerance_weight: 2,
                locality_weight: 1,
            },
            backend_requirements: Vec::new(),
        };
        let decision = plan_v4(&request).unwrap();
        assert_eq!(decision.plan.shards.len(), 4);
        assert_eq!(decision.plan.groups.len(), 1);
        assert_eq!(decision.plan.support, V4SupportLevel::Experimental);
        assert_eq!(decision.plan.plan_hash, hash_plan(&decision.plan));
        assert!(!decision.explanations.is_empty());
    }

    #[test]
    fn hysteresis_blocks_noisy_migration() {
        let shard = V4ShardOwnership {
            shard_id: 1,
            model_generation: 1,
            owners: vec![NodeId::from_bytes([1; 32])],
            replicas: Vec::new(),
            ownership_generation: 3,
            content_hash: ArtifactId::from_bytes([2; 32]),
            state_bytes: 10,
            memory_bytes: 20,
            runtime_requirement: "reference.cpu.i64".to_string(),
            lifecycle: V4ShardLifecycle::Active,
        };
        let candidate = worker(3, 100);
        let decision = rebalance_shard(&shard, &candidate, 100, Some(90), 20, 0.2, 0.5, 0.4);
        assert!(!decision.accepted);
        assert!(decision.reason.contains("cooldown"));
    }

    #[test]
    fn empty_owner_is_rejected_without_panicking() {
        let shard = V4ShardOwnership {
            shard_id: 9,
            model_generation: 1,
            owners: Vec::new(),
            replicas: Vec::new(),
            ownership_generation: 1,
            content_hash: ArtifactId::from_bytes([7; 32]),
            state_bytes: 10,
            memory_bytes: 20,
            runtime_requirement: "reference.cpu.i64".to_string(),
            lifecycle: V4ShardLifecycle::Active,
        };
        let decision = rebalance_shard(&shard, &worker(3, 100), 100, None, 0, 0.1, 1.0, 0.1);
        assert!(!decision.accepted);
        assert!(decision.reason.contains("no current owner"));
    }

    #[test]
    fn topology_links_influence_groups_and_reject_duplicate_profiles() {
        let workers = (1..=4).map(|id| worker(id, 2048)).collect::<Vec<_>>();
        let mut links = Vec::new();
        for (left, right) in [(1_u8, 2_u8), (3, 4)] {
            links.push(V4TopologyLink {
                from: NodeId::from_bytes([left; 32]),
                to: NodeId::from_bytes([right; 32]),
                rtt_ms: 2,
                bandwidth_mbps: 1_000,
                reliability: 0.99,
            });
        }
        let request = V4PlanRequest {
            job_id: JobId::from_bytes([7; 16]),
            proposer: NodeId::from_bytes([9; 32]),
            model_bytes: 2048,
            requested_workers: 4,
            strategy: V4ParallelismStrategy::LocalSgd,
            tensor_degree: 0,
            pipeline_stages: 0,
            local_steps: 1,
            max_staleness: 1,
            checkpoint_replication: 2,
            data_locality: DataLocality::Selective,
            workers: workers.clone(),
            links,
            objective: V4PlannerObjective::default(),
            backend_requirements: Vec::new(),
        };
        let decision = plan_v4(&request).unwrap();
        let group_sets = decision
            .plan
            .groups
            .iter()
            .map(|group| {
                group
                    .members
                    .iter()
                    .map(|node| node.0[0])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(
            group_sets
                .iter()
                .any(|group| group.contains(&1) && group.contains(&2))
        );
        assert!(
            group_sets
                .iter()
                .any(|group| group.contains(&3) && group.contains(&4))
        );

        let mut duplicate = workers;
        duplicate.push(worker(1, 2048));
        let duplicate_request = V4PlanRequest {
            workers: duplicate,
            ..request
        };
        assert!(matches!(
            plan_v4(&duplicate_request),
            Err(V4PlannerError::InvalidWorkerProfile)
        ));
    }

    #[test]
    fn planner_does_not_advertise_unimplemented_parallel_shapes() {
        let workers = (1..=4).map(|id| worker(id, 2048)).collect::<Vec<_>>();
        let pipeline = V4PlanRequest {
            job_id: JobId::from_bytes([8; 16]),
            proposer: NodeId::from_bytes([9; 32]),
            model_bytes: 2048,
            requested_workers: 4,
            strategy: V4ParallelismStrategy::PipelineParallel,
            tensor_degree: 0,
            pipeline_stages: 3,
            local_steps: 1,
            max_staleness: 1,
            checkpoint_replication: 2,
            data_locality: DataLocality::Selective,
            workers: workers.clone(),
            links: Vec::new(),
            objective: V4PlannerObjective::default(),
            backend_requirements: Vec::new(),
        };
        assert!(matches!(
            plan_v4(&pipeline),
            Err(V4PlannerError::UnsupportedStrategy)
        ));

        let hybrid = V4PlanRequest {
            strategy: V4ParallelismStrategy::Hybrid,
            tensor_degree: 2,
            pipeline_stages: 4,
            ..pipeline
        };
        assert!(matches!(
            plan_v4(&hybrid),
            Err(V4PlannerError::UnsupportedStrategy)
        ));
    }

    #[test]
    fn median_limits_sign_flip() {
        let report = aggregate_updates(
            V4ByzantinePolicy::CoordinateMedian,
            &[vec![10, 10], vec![11, 9], vec![-1_000, -1_000]],
            100,
        )
        .unwrap();
        assert_eq!(report.aggregate, vec![10, 9]);
        assert_eq!(report.rejected_updates, 1);
    }

    #[test]
    fn compatible_local_sgd_branches_merge() {
        let left = V4TrainingBranch {
            job_id: JobId::from_bytes([1; 16]),
            branch: ArtifactId::from_bytes([2; 32]),
            parent_generation: 4,
            model_generation: 4,
            optimizer_generation: 4,
            dataset_progress: 8,
            plan_generation: 2,
            created_by: NodeId::from_bytes([3; 32]),
        };
        let mut right = left.clone();
        right.branch = ArtifactId::from_bytes([5; 32]);
        let result = reconcile_branches(
            &left,
            &right,
            intelligence_protocol::V4ReconciliationPolicy::MergeLocalSgdState,
            10,
            14,
        )
        .unwrap();
        assert!(result.accepted);
        assert_eq!(result.value, Some(12));

        let mut divergent = right;
        divergent.dataset_progress += 1;
        assert!(matches!(
            reconcile_branches(
                &left,
                &divergent,
                intelligence_protocol::V4ReconciliationPolicy::MergeLocalSgdState,
                10,
                14,
            ),
            Err(V4PlannerError::IncompatibleBranches)
        ));
    }

    #[test]
    fn a_plan_cannot_claim_itself_as_its_parent() {
        let request = V4PlanRequest {
            job_id: JobId::from_bytes([11; 16]),
            proposer: NodeId::from_bytes([12; 32]),
            model_bytes: 2048,
            requested_workers: 2,
            strategy: V4ParallelismStrategy::LocalSgd,
            tensor_degree: 0,
            pipeline_stages: 0,
            local_steps: 1,
            max_staleness: 1,
            checkpoint_replication: 2,
            data_locality: DataLocality::Selective,
            workers: (1..=2).map(|id| worker(id, 2048)).collect(),
            links: Vec::new(),
            objective: V4PlannerObjective::default(),
            backend_requirements: Vec::new(),
        };
        let mut plan = plan_v4(&request).unwrap().plan;
        plan.parent_plan_hash = Some(plan.plan_hash);
        assert!(plan.validate().is_err());
    }

    #[test]
    fn heterogeneous_backend_requirements_are_bound_and_false_evidence_is_rejected() {
        use intelligence_protocol::{ComputeFeature, ComputeTaskKind, NumericFormat};

        let mut workers = (1..=4).map(|id| worker(id, 4096)).collect::<Vec<_>>();
        for (profile, backend) in workers.iter_mut().zip([
            BackendKind::Cpu,
            BackendKind::Cuda,
            BackendKind::Rocm,
            BackendKind::Metal,
        ]) {
            profile.backends = vec![reference_backend_capability(backend, 4096, false)];
        }
        let requirement = ComputeRequirements {
            task_kind: ComputeTaskKind::TensorForward,
            required_backend: Some(BackendKind::Cuda),
            allowed_backends: vec![BackendKind::Cuda],
            required_formats: vec![NumericFormat::F32],
            required_memory_bytes: 128,
            max_tensor_elements: 16,
            required_features: vec![ComputeFeature::DeterministicKernel],
            kernel_id: "test.cuda.forward.v1".to_string(),
            kernel_version: 1,
            fallback_backends: Vec::new(),
            fallback_allowed: false,
        };
        let request = V4PlanRequest {
            job_id: JobId::from_bytes([7; 16]),
            proposer: NodeId::from_bytes([9; 32]),
            model_bytes: 2048,
            requested_workers: 4,
            strategy: V4ParallelismStrategy::LocalSgd,
            tensor_degree: 0,
            pipeline_stages: 0,
            local_steps: 1,
            max_staleness: 1,
            checkpoint_replication: 2,
            data_locality: DataLocality::Selective,
            workers: workers.clone(),
            links: Vec::new(),
            objective: V4PlannerObjective::default(),
            backend_requirements: vec![requirement],
        };
        let decision = plan_v4(&request).expect("CUDA-capable worker should be eligible");
        assert_eq!(decision.plan.backend_assignments.len(), 1);
        assert_eq!(
            decision.plan.backend_assignments[0].backend,
            BackendKind::Cuda
        );

        let mut false_capability = reference_backend_capability(BackendKind::Cuda, 4096, false);
        false_capability.observed_failures = 1;
        false_capability.observed_successes = 0;
        let mut false_workers = workers;
        false_workers[1].backends = vec![false_capability];
        let false_request = V4PlanRequest {
            workers: false_workers,
            ..request
        };
        assert!(matches!(
            plan_v4(&false_request),
            Err(V4PlannerError::UnsupportedBackendAssignment(_))
        ));
    }
}
