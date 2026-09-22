//! Inspectable training placement and strategy selection.
//!
//! The planner deliberately returns a small supported set.  It does not claim
//! that a strategy is executable merely because it has a name in a document.

use intelligence_protocol::{BackendCapabilities, DataLocality, NodeId};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum HardwareKind {
    Cpu,
    AppleSilicon,
    AmdGpu,
    NvidiaConsumer,
    A100,
    H100,
    B200,
    Unknown,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerProfile {
    pub node: NodeId,
    pub hardware: HardwareKind,
    pub memory_bytes: u64,
    pub compute_units: u32,
    pub rtt_ms: u32,
    pub throughput_mbps: u32,
    pub reliability: f32,
    pub dataset_available: bool,
    #[serde(default)]
    pub backends: Vec<BackendCapabilities>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum TrainingStrategy {
    SynchronousDataParallel,
    BoundedStaleness,
    LocalSgd,
    Unsupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum StrategySupport {
    Supported,
    Experimental,
    Unsupported,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingPlanDecision {
    pub strategy: TrainingStrategy,
    pub support: StrategySupport,
    pub selected_workers: Vec<NodeId>,
    pub rejected_workers: Vec<NodeId>,
    pub synchronization_timeout_ms: u64,
    pub checkpoint_replicas: usize,
    pub data_locality: DataLocality,
    pub rationale: String,
}

#[derive(Debug, Error)]
pub enum PlannerError {
    #[error("training requires at least two compatible workers")]
    InsufficientWorkers,
    #[error("training model size must be positive")]
    InvalidModelSize,
    #[error("local-only data requires every selected worker to have a local copy")]
    LocalityViolation,
}

pub fn plan_training(
    model_bytes: u64,
    data_locality: DataLocality,
    requested_workers: usize,
    workers: &[WorkerProfile],
) -> Result<TrainingPlanDecision, PlannerError> {
    if model_bytes == 0 {
        return Err(PlannerError::InvalidModelSize);
    }
    let required_memory = model_bytes.saturating_mul(2);
    let mut eligible = workers
        .iter()
        .filter(|worker| {
            worker.memory_bytes >= required_memory
                && worker.compute_units > 0
                && worker.reliability.is_finite()
                && worker.reliability >= 0.5
                && (!matches!(data_locality, DataLocality::LocalOnly) || worker.dataset_available)
        })
        .cloned()
        .collect::<Vec<_>>();
    eligible.sort_by_key(|worker| {
        (
            worker.rtt_ms,
            std::cmp::Reverse(worker.reliability.to_bits()),
        )
    });
    eligible.truncate(requested_workers.max(2));
    if eligible.len() < 2 {
        if matches!(data_locality, DataLocality::LocalOnly)
            && workers
                .iter()
                .filter(|worker| worker.memory_bytes >= required_memory)
                .count()
                >= 2
        {
            return Err(PlannerError::LocalityViolation);
        }
        return Err(PlannerError::InsufficientWorkers);
    }
    let selected_workers = eligible
        .iter()
        .map(|worker| worker.node)
        .collect::<Vec<_>>();
    let selected_set = selected_workers
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>();
    let rejected_workers = workers
        .iter()
        .filter(|worker| !selected_set.contains(&worker.node))
        .map(|worker| worker.node)
        .collect::<Vec<_>>();
    let max_rtt = eligible
        .iter()
        .map(|worker| worker.rtt_ms)
        .max()
        .unwrap_or(0);
    let min_reliability = eligible
        .iter()
        .map(|worker| worker.reliability)
        .fold(1.0_f32, f32::min);
    let (strategy, support, timeout, rationale) = if max_rtt <= 200 && min_reliability >= 0.8 {
        (
            TrainingStrategy::SynchronousDataParallel,
            StrategySupport::Supported,
            (max_rtt as u64).saturating_mul(4).max(1_000),
            "bounded synchronous data parallelism fits measured RTT and reliability".to_string(),
        )
    } else {
        (
            TrainingStrategy::BoundedStaleness,
            StrategySupport::Experimental,
            (max_rtt as u64).saturating_mul(6).max(2_000),
            "slow or unreliable workers require timeout and bounded staleness; replanning remains enabled".to_string(),
        )
    };
    Ok(TrainingPlanDecision {
        strategy,
        support,
        selected_workers,
        rejected_workers,
        synchronization_timeout_ms: timeout,
        checkpoint_replicas: eligible.len().min(3),
        data_locality,
        rationale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(value: u8, rtt_ms: u32, reliability: f32, dataset_available: bool) -> WorkerProfile {
        WorkerProfile {
            node: NodeId::from_bytes([value; 32]),
            hardware: HardwareKind::Cpu,
            memory_bytes: 1024,
            compute_units: 1,
            rtt_ms,
            throughput_mbps: 10,
            reliability,
            dataset_available,
            backends: Vec::new(),
        }
    }

    #[test]
    fn planner_selects_supported_strategy_and_reports_rejections() {
        let plan = plan_training(
            256,
            DataLocality::Selective,
            2,
            &[
                worker(1, 10, 0.99, true),
                worker(2, 20, 0.95, true),
                worker(3, 500, 0.4, true),
            ],
        )
        .unwrap();
        assert_eq!(plan.strategy, TrainingStrategy::SynchronousDataParallel);
        assert_eq!(plan.support, StrategySupport::Supported);
        assert_eq!(plan.selected_workers.len(), 2);
        assert_eq!(plan.rejected_workers.len(), 1);
    }

    #[test]
    fn planner_rejects_locality_violation() {
        let error = plan_training(
            256,
            DataLocality::LocalOnly,
            2,
            &[worker(1, 10, 0.99, true), worker(2, 20, 0.99, false)],
        )
        .unwrap_err();
        assert!(matches!(error, PlannerError::LocalityViolation));
    }
}
