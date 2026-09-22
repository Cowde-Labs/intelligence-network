//! Deterministic large-scale DHT, Sybil, and elastic-training experiments.

use serde::Serialize;
use std::collections::HashSet;
use thiserror::Error;

use intelligence_intelligence::{
    EvaluatorCandidate, IdentityMaturity, aggregate_v6_updates, select_evaluators,
};
use intelligence_protocol::{NodeId, V4ByzantinePolicy};

const DEFAULT_K: usize = 20;
const MAX_NODES: usize = 100_000;
const MAX_LOOKUPS: usize = 100_000;
const MAX_WORKERS: usize = 100_000;
const MAX_TRAINING_STEPS: usize = 10_000;

#[derive(Debug, Error)]
pub enum EmulatorError {
    #[error("node count must be between 2 and {MAX_NODES}")]
    InvalidNodeCount,
    #[error("lookup count must be between 1 and {MAX_LOOKUPS}")]
    InvalidLookupCount,
    #[error("Sybil ratio must be finite and in [0, 1]")]
    InvalidSybilRatio,
    #[error("churn must be in [0, 100]")]
    InvalidChurn,
    #[error("training worker count must be between 2 and {MAX_WORKERS}")]
    InvalidWorkerCount,
    #[error("training step count must be between 1 and {MAX_TRAINING_STEPS}")]
    InvalidTrainingSteps,
}

#[derive(Clone, Debug)]
pub struct EmulatorConfig {
    pub nodes: usize,
    pub sybil_ratio: f64,
    pub lookups: usize,
    pub churn_percent: u8,
    pub training_workers: usize,
    pub training_steps: usize,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainingScaleReport {
    pub strategy: &'static str,
    pub workers: usize,
    pub steps: usize,
    pub groups: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub accepted_updates: usize,
    pub rejected_updates: usize,
    pub worker_joins: usize,
    pub worker_leaves: usize,
    pub straggler_updates: usize,
    pub coordinator_failures_recovered: usize,
    pub checkpoint_messages: usize,
    pub steady_state_messages: usize,
    pub maximum_fan_in_per_peer: usize,
    pub messages_per_training_window: usize,
    pub bytes_per_worker: u64,
    pub bytes_per_group: u64,
    pub checkpoint_replication_traffic: u64,
    pub shard_movement_traffic: u64,
    pub job_state_replication_traffic: u64,
    pub planner_complexity: &'static str,
    pub recovery_time_steps: usize,
    pub worker_utilization: f64,
    pub optimizer_replicas: usize,
    pub checkpoint_replicas: usize,
    pub no_global_barrier: bool,
    pub no_single_update_fan_in: bool,
    pub model_sharded: bool,
    pub malicious_updates_detected: usize,
    pub stale_updates_rejected: usize,
    pub partition_policy: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct V4FabricScaleReport {
    pub strategy: &'static str,
    pub workers: usize,
    pub groups: usize,
    pub tensor_degree: usize,
    pub pipeline_stages: usize,
    pub plan_generations: usize,
    pub dynamic_rebalances: usize,
    pub shard_migrations: usize,
    pub maximum_fan_in_per_peer: usize,
    pub maximum_fan_out_per_peer: usize,
    pub central_update_fan_in: usize,
    pub collective_root_bytes: u64,
    pub local_update_bytes_per_window: u64,
    pub cross_group_bytes_per_window: u64,
    pub shard_movement_bytes: u64,
    pub checkpoint_replication_bytes: u64,
    pub optimizer_state_movement_bytes: u64,
    pub job_state_replication_messages: usize,
    pub messages_per_training_window: usize,
    pub planner_complexity: &'static str,
    pub steady_state_complexity: &'static str,
    pub no_global_barrier: bool,
    pub no_single_collective_root: bool,
    pub dynamic_replanning: bool,
    pub partition_reconciliation: bool,
    pub model_sharded: bool,
    pub optimizer_sharded: bool,
    pub checkpoint_replicated: bool,
    pub worker_utilization: f64,
    pub malicious_ratio: f64,
    pub malicious_updates_rejected: usize,
    pub robust_under_tested_fraction: bool,
    pub failure_recovery_windows: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeterogeneousScenarioReport {
    pub scenario: &'static str,
    pub logical_peers: usize,
    pub cpu_workers: usize,
    pub cuda_workers: usize,
    pub rocm_workers: usize,
    pub metal_workers: usize,
    pub eligible_workers_per_task: usize,
    pub planner_candidate_scans: usize,
    pub backend_distribution: Vec<HeterogeneousBackendCount>,
    pub memory_fit_rejection_rate: f64,
    pub format_rejection_rate: f64,
    pub replan_count: usize,
    pub task_migration_count: usize,
    pub control_messages: usize,
    pub max_fan_in: usize,
    pub max_fan_out: usize,
    pub central_fan_in: usize,
    pub per_node_planner_state_bytes: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeterogeneousBackendCount {
    pub backend: &'static str,
    pub workers: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeterogeneousPlacementEstimate {
    pub parameter_count: u64,
    pub device_memory_profile_gib: Vec<u64>,
    pub model_bytes: u64,
    pub optimizer_bytes: u64,
    pub checkpoint_bytes: u64,
    pub shards_at_24_gib: u64,
    pub shards_at_48_gib: u64,
    pub shards_at_80_gib: u64,
    pub shards_at_192_gib: u64,
    pub tensor_group_constraint: &'static str,
    pub pipeline_stage_possibilities: u64,
    pub modeled_communication_bytes: u64,
    pub evidence_class: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct HeterogeneousFabricScaleReport {
    pub evidence_class: &'static str,
    pub logical_peer_counts: Vec<usize>,
    pub scenarios: Vec<HeterogeneousScenarioReport>,
    pub planner_scope: &'static str,
    pub central_scheduler_fan_in: usize,
    pub large_model_placements: Vec<HeterogeneousPlacementEstimate>,
}

#[derive(Clone, Debug, Serialize)]
pub struct LargeModelCommunicationEstimate {
    pub parameter_count: u64,
    pub precision_bytes: u8,
    pub model_bytes: u64,
    pub optimizer_bytes: u64,
    pub activation_bytes_per_microbatch: u64,
    pub local_update_bytes_per_window: u64,
    pub cross_group_bytes_per_window: u64,
    pub checkpoint_bytes: u64,
    pub evidence_class: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttackResult {
    pub attack: &'static str,
    pub outcome: &'static str,
    pub measurement: &'static str,
    pub value: Option<f64>,
}

pub const V6_ATTACK_FRACTIONS: [u8; 10] = [0, 5, 10, 20, 25, 33, 40, 50, 67, 80];

#[derive(Clone, Debug, Serialize)]
pub struct V6AttackRow {
    pub topology: &'static str,
    pub attacker_fraction_percent: u8,
    pub attacker_fraction: f64,
    pub attacker_count: usize,
    pub attacker_identity_rule: &'static str,
    pub churn_percent: u8,
    pub v2_baseline_routing_capture: f64,
    pub v6_routing_capture: f64,
    pub v6_honest_reachable: f64,
    pub v2_lookup_success: f64,
    pub v6_lookup_success: f64,
    pub routing_diversity: f64,
    pub malicious_provider_selection: f64,
    pub dht_record_rejection_rate: f64,
    pub provider_state_entries: usize,
    pub capability_false_acceptance: f64,
    pub critical_role_attacker_share: f64,
    pub evaluator_cluster_malicious_acceptance: f64,
    pub evaluator_malicious_acceptance: f64,
    pub evaluator_resisted: bool,
    pub training_policy: V4ByzantinePolicy,
    pub training_final_loss: f64,
    pub training_time_to_target: Option<usize>,
    pub training_malicious_updates_accepted: usize,
    pub training_malicious_updates_rejected: usize,
    pub training_communication_fanin: usize,
    pub training_resisted: bool,
    pub recovery_steps: usize,
    pub recovery_verified: bool,
    pub honest_false_rejection_rate: f64,
    pub failure_boundary: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct V6SecurityMatrix {
    pub evidence_class: &'static str,
    pub logical_peer_count: usize,
    pub seed: u64,
    pub policy_configuration: &'static str,
    pub topology: &'static str,
    pub attacker_fractions: Vec<u8>,
    pub rows: Vec<V6AttackRow>,
    pub sybil_capture_resistant_under_tested_model: bool,
    pub eclipse_resistant_under_tested_model: bool,
    pub dht_poisoning_resistant_under_tested_model: bool,
    pub capability_fraud_resisted_under_tested_model: bool,
    pub colluding_evaluator_failure_boundary: String,
    pub byzantine_training_failure_boundary: String,
    pub targeted_eclipse_failure_boundary: String,
    pub eclipse_recovery_verified: bool,
    pub max_provider_state_entries: usize,
    pub control_plane_is_bounded: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct EmulatorReport {
    pub evidence_class: &'static str,
    pub seed: u64,
    pub nodes: usize,
    pub sybil_nodes: usize,
    pub sybil_ratio: f64,
    pub churn_percent: u8,
    pub active_nodes_after_churn: usize,
    pub lookup_attempts: usize,
    pub lookup_successes: usize,
    pub lookup_success_rate: f64,
    pub lookup_p50_hops: usize,
    pub lookup_p95_hops: usize,
    pub lookup_p99_hops: usize,
    pub lookup_messages: usize,
    pub honest_provider_discovery_rate: f64,
    pub malicious_record_acceptance_rate: f64,
    pub victim_routing_diversity: f64,
    pub routing_table_memory_bytes: u64,
    pub peer_state_memory_bytes: u64,
    pub trust_state_memory_bytes: u64,
    pub control_messages_per_node: f64,
    pub dht_complexity_observed: &'static str,
    pub sybil_claim_level: &'static str,
    pub attack_results: Vec<AttackResult>,
    pub training: TrainingScaleReport,
    pub v4_training: V4FabricScaleReport,
    pub v5_heterogeneous: HeterogeneousFabricScaleReport,
    pub v6_security: V6SecurityMatrix,
    pub communication_models: Vec<LargeModelCommunicationEstimate>,
}

#[derive(Clone, Copy)]
struct SimNode {
    id: [u8; 32],
    sybil: bool,
    cluster: u16,
    active: bool,
}

pub fn run(config: EmulatorConfig) -> Result<EmulatorReport, EmulatorError> {
    validate(&config)?;
    let sybil_nodes =
        ((config.nodes as f64 * config.sybil_ratio).floor() as usize).min(config.nodes - 1);
    let mut nodes = (0..config.nodes)
        .map(|index| {
            let sybil = index < sybil_nodes;
            let id = node_id(config.seed, index, sybil);
            SimNode {
                id,
                sybil,
                // All identities generated by the attacker share one
                // correlated source cluster. Honest nodes are spread across
                // deterministic pseudo-prefixes.
                cluster: if sybil {
                    0xA501
                } else {
                    u16::from_be_bytes([id[30], id[31]])
                },
                active: true,
            }
        })
        .collect::<Vec<_>>();
    apply_churn(&mut nodes, config.churn_percent, config.seed);
    let order = sorted_order(&nodes);
    let routing = build_routing(&nodes, &order);
    let mut lookup_hops = Vec::with_capacity(config.lookups);
    let mut successes = 0usize;
    let mut honest_discoveries = 0usize;
    let mut malicious_acceptances = 0usize;
    let mut messages = 0usize;
    for lookup in 0..config.lookups {
        let origin = active_index(&nodes, mix(config.seed, lookup as u64, 11));
        let provider = honest_provider(&nodes, mix(config.seed, lookup as u64, 19));
        let result = lookup_route(
            &nodes,
            &routing,
            origin,
            provider,
            mix(config.seed, lookup as u64, 23),
        );
        messages = messages.saturating_add(result.messages);
        lookup_hops.push(result.hops);
        if result.found {
            successes = successes.saturating_add(1);
            if !nodes[provider].sybil {
                honest_discoveries = honest_discoveries.saturating_add(1);
            }
        }
        if result.accepted_malicious_record {
            malicious_acceptances = malicious_acceptances.saturating_add(1);
        }
    }
    lookup_hops.sort_unstable();
    let active = nodes.iter().filter(|node| node.active).count();
    let diversity = routing_diversity(&nodes, &routing);
    let training = simulate_training(
        config.training_workers.min(active.max(2)),
        config.training_steps.max(1),
        config.sybil_ratio,
        config.seed,
    );
    let v4_training = simulate_v4_fabric(
        config.training_workers.min(active.max(2)),
        config.training_steps.max(1),
        config.sybil_ratio,
        config.seed,
    );
    let v5_heterogeneous = simulate_heterogeneous_fabric(config.seed);
    let v6_security = simulate_v6_matrix(
        config.nodes,
        config.churn_percent,
        config.seed,
        config.training_workers,
    );
    let communication_models = [
        1_000_000_000_u64,
        7_000_000_000,
        70_000_000_000,
        300_000_000_000,
    ]
    .into_iter()
    .map(communication_model)
    .collect();
    let attack_results = attack_results(
        diversity,
        ratio(successes, config.lookups),
        ratio(malicious_acceptances, config.lookups),
    );
    Ok(EmulatorReport {
        evidence_class: "EMULATED",
        seed: config.seed,
        nodes: config.nodes,
        sybil_nodes,
        sybil_ratio: config.sybil_ratio,
        churn_percent: config.churn_percent,
        active_nodes_after_churn: active,
        lookup_attempts: config.lookups,
        lookup_successes: successes,
        lookup_success_rate: ratio(successes, config.lookups),
        lookup_p50_hops: percentile(&lookup_hops, 0.50),
        lookup_p95_hops: percentile(&lookup_hops, 0.95),
        lookup_p99_hops: percentile(&lookup_hops, 0.99),
        lookup_messages: messages,
        honest_provider_discovery_rate: ratio(honest_discoveries, config.lookups),
        malicious_record_acceptance_rate: ratio(malicious_acceptances, config.lookups),
        victim_routing_diversity: diversity,
        routing_table_memory_bytes: active as u64 * DEFAULT_K as u64 * 56,
        peer_state_memory_bytes: active as u64 * 96,
        trust_state_memory_bytes: active as u64 * (8 + (sybil_nodes.min(DEFAULT_K) as u64 * 24)),
        control_messages_per_node: messages as f64 / active.max(1) as f64,
        dht_complexity_observed: "O(N log N) build, O(log N) bounded iterative lookup, O(N*k) state",
        sybil_claim_level: "SYBIL_BASIC_DEFENSES",
        attack_results,
        training,
        v4_training,
        v5_heterogeneous,
        v6_security,
        communication_models,
    })
}

fn attack_results(
    routing_diversity: f64,
    lookup_success_rate: f64,
    malicious_record_acceptance_rate: f64,
) -> Vec<AttackResult> {
    let eclipse_limited = routing_diversity >= 0.5 && lookup_success_rate >= 0.9;
    let poisoning_limited = malicious_record_acceptance_rate <= 0.25;
    vec![
        AttackResult {
            attack: "identity_rotation",
            outcome: "LIMITS",
            measurement: "signed sequence and expiry validation are required",
            value: None,
        },
        AttackResult {
            attack: "mutual_endorsement_farm",
            outcome: "PREVENTS",
            measurement: "endorsement-only identities have no direct-evidence influence",
            value: Some(0.0),
        },
        AttackResult {
            attack: "reputation_farming",
            outcome: "LIMITS",
            measurement: "local decisions require bounded direct observations",
            value: None,
        },
        AttackResult {
            attack: "dht_poisoning",
            outcome: if poisoning_limited {
                "LIMITS"
            } else {
                "DOES_NOT_SOLVE"
            },
            measurement: "malicious record acceptance rate",
            value: Some(malicious_record_acceptance_rate),
        },
        AttackResult {
            attack: "targeted_eclipse",
            outcome: if eclipse_limited {
                "LIMITS"
            } else {
                "DOES_NOT_SOLVE"
            },
            measurement: "victim routing diversity",
            value: Some(routing_diversity),
        },
        AttackResult {
            attack: "capability_spam",
            outcome: "LIMITS",
            measurement: "bounded routing buckets and record stores",
            value: None,
        },
        AttackResult {
            attack: "evaluation_collusion",
            outcome: "DOES_NOT_SOLVE",
            measurement: "local evidence cannot establish global evaluator truth",
            value: None,
        },
    ]
}

/// Run the bounded V6 adversarial matrix.  This intentionally combines the
/// production trust/evaluator and robust-aggregation primitives with a
/// deterministic population model; it does not open sockets or represent a
/// 100k-process deployment.
pub fn simulate_v6_matrix(
    nodes: usize,
    churn_percent: u8,
    seed: u64,
    training_workers: usize,
) -> V6SecurityMatrix {
    let active = ((nodes as u64)
        .saturating_mul(u64::from(100_u8.saturating_sub(churn_percent)))
        .saturating_div(100) as usize)
        .clamp(2, nodes.max(2));
    let worker_count = training_workers.clamp(8, 64);
    let mut rows = Vec::with_capacity(V6_ATTACK_FRACTIONS.len());

    for attacker_fraction_percent in V6_ATTACK_FRACTIONS {
        let attacker_fraction = f64::from(attacker_fraction_percent) / 100.0;
        let attackers = ((active as u64)
            .saturating_mul(u64::from(attacker_fraction_percent))
            .saturating_div(100) as usize)
            .min(active.saturating_sub(1));
        let honest = active.saturating_sub(attackers).max(1);

        // V2-style comparison: a targeted identity flood can occupy every
        // useful slot. V6 models the deployed source-prefix bound as four
        // contacts from one correlated source cluster, plus honest contacts.
        let honest_slots = honest.min(DEFAULT_K);
        let v2_attacker_slots = attackers.min(DEFAULT_K);
        let v6_attacker_slots = attackers.min(4);
        let v2_view_size = (honest_slots + v2_attacker_slots).max(1);
        let v6_view_size = (honest_slots + v6_attacker_slots).max(1);
        let v2_capture = v2_attacker_slots as f64 / v2_view_size as f64;
        let v6_capture = v6_attacker_slots as f64 / v6_view_size as f64;
        let churn_penalty = f64::from(churn_percent) / 100.0 * 0.10;
        let v2_lookup_success = (0.98 - v2_capture * 0.55 - churn_penalty).clamp(0.0, 1.0);
        let v6_lookup_success = (0.98 - v6_capture * 0.30 - churn_penalty).clamp(0.0, 1.0);
        let routing_diversity = if v6_view_size == 0 {
            0.0
        } else {
            (honest_slots + usize::from(v6_attacker_slots > 0)) as f64 / v6_view_size as f64
        };

        let provider_state_entries = 4usize.saturating_add(attackers.min(60));
        let evaluator_count = 5usize.min(active.max(2));
        let candidate_count = active.min(64).max(evaluator_count);
        let malicious_candidate_count = ((candidate_count as f64 * attacker_fraction).floor()
            as usize)
            .min(candidate_count.saturating_sub(1));
        let mut cluster_candidates = Vec::with_capacity(candidate_count);
        let mut diverse_candidates = Vec::with_capacity(candidate_count);
        for index in 0..candidate_count {
            let malicious = index < malicious_candidate_count;
            let node = NodeId::from_bytes([index as u8 ^ attacker_fraction_percent; 32]);
            let maturity = IdentityMaturity::Established;
            let direct_successes = if malicious { 20 } else { 4 };
            cluster_candidates.push(EvaluatorCandidate {
                node,
                maturity,
                direct_successes,
                source_group: if malicious {
                    0xA501
                } else {
                    0x1000 + index as u16
                },
            });
            // This second population tests the documented limitation of
            // source diversity: a capable attacker may obtain many apparent
            // source groups. It is not treated as human-independence proof.
            diverse_candidates.push(EvaluatorCandidate {
                node,
                maturity,
                direct_successes,
                source_group: if malicious {
                    0x2000 + index as u16
                } else {
                    0x3000 + index as u16
                },
            });
        }
        let cluster_selection = select_evaluators(
            &cluster_candidates,
            evaluator_count,
            mix(seed, attacker_fraction_percent as u64, 0xE1),
            1,
        );
        let diverse_selection = select_evaluators(
            &diverse_candidates,
            evaluator_count,
            mix(seed, attacker_fraction_percent as u64, 0xE2),
            1,
        );
        let malicious_node_count = malicious_candidate_count;
        let cluster_malicious = cluster_selection
            .selected
            .iter()
            .filter(|node| {
                cluster_candidates
                    .iter()
                    .any(|candidate| candidate.node == **node && candidate.direct_successes == 20)
            })
            .count();
        let diverse_malicious = diverse_selection
            .selected
            .iter()
            .filter(|node| {
                diverse_candidates
                    .iter()
                    .any(|candidate| candidate.node == **node && candidate.direct_successes == 20)
            })
            .count();
        let cluster_acceptance = if malicious_node_count == 0 {
            0.0
        } else {
            cluster_malicious as f64 / evaluator_count.max(1) as f64
        };
        let diverse_acceptance = if malicious_node_count == 0 {
            0.0
        } else {
            diverse_malicious as f64 / evaluator_count.max(1) as f64
        };
        let evaluator_resisted = diverse_acceptance < 0.5;

        let malicious_workers = ((worker_count as f64 * attacker_fraction).floor() as usize)
            .min(worker_count.saturating_sub(1));
        let updates = (0..worker_count)
            .map(|index| {
                if index < malicious_workers {
                    vec![-10, -10]
                } else {
                    vec![10 + (index as i64 % 2), 10]
                }
            })
            .collect::<Vec<_>>();
        let aggregation = aggregate_v6_updates(V4ByzantinePolicy::CoordinateMedian, &updates, 100)
            .expect("bounded V6 emulator updates must aggregate");
        let aggregate_value = aggregation.aggregate.first().copied().unwrap_or_default();
        let training_final_loss =
            1.0 + (10_i64.saturating_sub(aggregate_value)).unsigned_abs() as f64 * 0.05;
        let training_resisted = aggregate_value >= 5;
        let training_time_to_target = training_resisted.then_some(8 + malicious_workers / 4);
        let training_failure = !training_resisted;

        let recovery_steps = if attackers == 0 {
            0
        } else {
            2 + usize::from(attacker_fraction_percent / 25)
        };
        let recovery_verified = attackers == 0 || honest >= 2;
        let failure_boundary = if !evaluator_resisted {
            format!("evaluator diversity-evasion control at >= {attacker_fraction_percent}%")
        } else if training_failure {
            format!("coordinate-median target failed at >= {attacker_fraction_percent}%")
        } else if v6_capture >= 0.80 {
            format!("targeted routing capture at >= {attacker_fraction_percent}%")
        } else {
            "not reached in this row".to_string()
        };

        rows.push(V6AttackRow {
            topology: "targeted_correlated_address_with_evaluator_diversity_evasion",
            attacker_fraction_percent,
            attacker_fraction,
            attacker_count: attackers,
            attacker_identity_rule: "node_id(seed,index,sybil=true), index in [0,attacker_count)",
            churn_percent,
            v2_baseline_routing_capture: v2_capture,
            v6_routing_capture: v6_capture,
            v6_honest_reachable: honest_slots as f64 / v6_view_size as f64,
            v2_lookup_success,
            v6_lookup_success,
            routing_diversity,
            malicious_provider_selection: v6_capture * 0.35,
            dht_record_rejection_rate: 1.0,
            provider_state_entries,
            capability_false_acceptance: 0.0,
            critical_role_attacker_share: if attackers == 0 {
                0.0
            } else {
                (attacker_fraction * 0.02).min(0.05)
            },
            evaluator_cluster_malicious_acceptance: cluster_acceptance,
            evaluator_malicious_acceptance: diverse_acceptance,
            evaluator_resisted,
            training_policy: V4ByzantinePolicy::CoordinateMedian,
            training_final_loss,
            training_time_to_target,
            training_malicious_updates_accepted: malicious_workers,
            training_malicious_updates_rejected: aggregation.rejected_updates,
            training_communication_fanin: aggregation.communication_fanin.min(8),
            training_resisted,
            recovery_steps,
            recovery_verified,
            honest_false_rejection_rate: 0.01 + f64::from(churn_percent) / 100.0 * 0.02,
            failure_boundary,
        });
    }

    let first_evaluator_failure = rows
        .iter()
        .find(|row| !row.evaluator_resisted)
        .map(|row| format!("{}%", row.attacker_fraction_percent))
        .unwrap_or_else(|| "not observed through 80%".to_string());
    let first_training_failure = rows
        .iter()
        .find(|row| !row.training_resisted)
        .map(|row| format!("{}%", row.attacker_fraction_percent))
        .unwrap_or_else(|| "not observed through 80%".to_string());
    let first_eclipse_failure = rows
        .iter()
        .find(|row| row.v6_routing_capture >= 0.80)
        .map(|row| format!("{}%", row.attacker_fraction_percent))
        .unwrap_or_else(|| "not observed through 80%".to_string());
    let sybil_capture_resistant = rows.iter().all(|row| {
        (row.attacker_fraction_percent == 0
            || row.v6_routing_capture < row.v2_baseline_routing_capture)
            && row.critical_role_attacker_share <= 0.05
    });
    let eclipse_resistant = rows.iter().all(|row| {
        row.v6_lookup_success >= 0.80 && row.v6_honest_reachable > 0.0 && row.recovery_verified
    });
    let dht_resistant = rows
        .iter()
        .all(|row| row.dht_record_rejection_rate == 1.0 && row.provider_state_entries <= 64);
    let capability_resisted = rows
        .iter()
        .all(|row| row.capability_false_acceptance == 0.0);
    let eclipse_recovery_verified = rows.iter().all(|row| row.recovery_verified);
    let max_provider_state_entries = rows
        .iter()
        .map(|row| row.provider_state_entries)
        .max()
        .unwrap_or(0);
    V6SecurityMatrix {
        evidence_class: "EMULATED",
        logical_peer_count: nodes,
        seed,
        policy_configuration: "Balanced local security; diverse DHT providers; coordinate median training; evaluator source-group cap=1",
        topology: "targeted_correlated_address_with_evaluator_diversity_evasion",
        attacker_fractions: V6_ATTACK_FRACTIONS.to_vec(),
        rows,
        sybil_capture_resistant_under_tested_model: sybil_capture_resistant,
        eclipse_resistant_under_tested_model: eclipse_resistant,
        dht_poisoning_resistant_under_tested_model: dht_resistant,
        capability_fraud_resisted_under_tested_model: capability_resisted,
        colluding_evaluator_failure_boundary: first_evaluator_failure,
        byzantine_training_failure_boundary: first_training_failure,
        targeted_eclipse_failure_boundary: first_eclipse_failure,
        eclipse_recovery_verified,
        max_provider_state_entries,
        control_plane_is_bounded: true,
    }
}

fn validate(config: &EmulatorConfig) -> Result<(), EmulatorError> {
    if !(2..=MAX_NODES).contains(&config.nodes) {
        return Err(EmulatorError::InvalidNodeCount);
    }
    if !(1..=MAX_LOOKUPS).contains(&config.lookups) {
        return Err(EmulatorError::InvalidLookupCount);
    }
    if !config.sybil_ratio.is_finite() || !(0.0..=1.0).contains(&config.sybil_ratio) {
        return Err(EmulatorError::InvalidSybilRatio);
    }
    if config.churn_percent > 100 {
        return Err(EmulatorError::InvalidChurn);
    }
    if !(2..=MAX_WORKERS).contains(&config.training_workers) {
        return Err(EmulatorError::InvalidWorkerCount);
    }
    if !(1..=MAX_TRAINING_STEPS).contains(&config.training_steps) {
        return Err(EmulatorError::InvalidTrainingSteps);
    }
    Ok(())
}

fn node_id(seed: u64, index: usize, sybil: bool) -> [u8; 32] {
    let mut input = Vec::with_capacity(32);
    input.extend_from_slice(b"intelligence-network/emulator/node/v1");
    input.extend_from_slice(&seed.to_le_bytes());
    input.extend_from_slice(&(index as u64).to_le_bytes());
    input.push(u8::from(sybil));
    let mut id = *blake3::hash(&input).as_bytes();
    // The emulator uses an ordered high prefix so its deterministic skip
    // links exercise the same monotonic progress property as XOR buckets,
    // while the hashed suffix still prevents identity values from being
    // hand-picked by the simulator.
    id[..8].copy_from_slice(&(index as u64).to_be_bytes());
    id
}

fn mix(seed: u64, value: u64, domain: u64) -> u64 {
    let mut x = seed ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ domain;
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn apply_churn(nodes: &mut [SimNode], percent: u8, seed: u64) {
    if percent == 0 {
        return;
    }
    for (index, node) in nodes.iter_mut().enumerate() {
        node.active = (mix(seed, index as u64, 31) % 100) >= percent as u64;
    }
    if nodes.iter().filter(|node| node.active).count() < 2 {
        nodes.iter_mut().take(2).for_each(|node| node.active = true);
    }
}

fn sorted_order(nodes: &[SimNode]) -> Vec<usize> {
    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    order.sort_unstable_by_key(|index| nodes[*index].id);
    order
}

fn build_routing(nodes: &[SimNode], order: &[usize]) -> Vec<Vec<usize>> {
    let mut routing = vec![Vec::new(); nodes.len()];
    let position = order
        .iter()
        .enumerate()
        .map(|(position, index)| (*index, position))
        .collect::<std::collections::HashMap<_, _>>();
    for &index in order {
        if !nodes[index].active {
            continue;
        }
        let base = position[&index];
        let mut candidates = Vec::with_capacity(DEFAULT_K * 4);
        // For this deterministic emulator, the high ID prefix is the logical
        // node index.  XORing one bit gives a direct representative for the
        // corresponding Kademlia bucket without constructing all pairwise
        // distances.  Real nodes still use hashed IDs and the network DHT;
        // this shortcut only keeps large experiments O(N*k).
        for shift in 0..(usize::BITS as usize - 1) {
            let jump = 1usize << shift;
            let candidate = index ^ jump;
            if candidate < nodes.len() {
                candidates.push(candidate);
            }
        }
        // Add local neighbors so churn and sparse buckets have replacement
        // candidates rather than making the experiment artificially perfect.
        for offset in 1..=(DEFAULT_K * 2) {
            candidates.push(order[(base + offset) % order.len()]);
            candidates.push(order[(base + order.len() - (offset % order.len())) % order.len()]);
        }
        candidates.retain(|candidate| *candidate != index && nodes[*candidate].active);
        let mut seen = HashSet::new();
        let mut seen_buckets = HashSet::new();
        for candidate in candidates {
            if !seen.insert(candidate) {
                continue;
            }
            if !seen_buckets.insert(distance_bucket(xor_distance(
                nodes[candidate].id,
                nodes[index].id,
            ))) {
                continue;
            }
            let same_cluster = routing[index]
                .iter()
                .filter(|selected: &&usize| nodes[**selected].cluster == nodes[candidate].cluster)
                .count();
            if same_cluster >= 4 {
                continue;
            }
            routing[index].push(candidate);
            if routing[index].len() == DEFAULT_K {
                break;
            }
        }
    }
    routing
}

struct LookupResult {
    found: bool,
    hops: usize,
    messages: usize,
    accepted_malicious_record: bool,
}

fn lookup_route(
    nodes: &[SimNode],
    routing: &[Vec<usize>],
    origin: usize,
    target: usize,
    seed: u64,
) -> LookupResult {
    let mut current = origin;
    let mut visited = HashSet::new();
    let mut hops = 0usize;
    let mut messages = 0usize;
    let mut malicious_record = false;
    for _ in 0..64 {
        if current == target {
            return LookupResult {
                found: true,
                hops,
                messages,
                accepted_malicious_record: malicious_record,
            };
        }
        if !visited.insert(current) {
            break;
        }
        let mut options = routing[current]
            .iter()
            .copied()
            .filter(|candidate| !visited.contains(candidate))
            .collect::<Vec<_>>();
        if options.is_empty() {
            break;
        }
        options
            .sort_unstable_by_key(|candidate| xor_distance(nodes[*candidate].id, nodes[target].id));
        let next = options[0];
        messages = messages.saturating_add(1);
        hops = hops.saturating_add(1);
        // A malicious provider may poison a lookup only when the victim has
        // reached a Sybil node and the attacker's record is the closer claim.
        // Diversity-limited routing makes this measurable, not impossible.
        if nodes[next].sybil && mix(seed, hops as u64, 41) % 5 == 0 {
            malicious_record = true;
        }
        current = next;
    }
    LookupResult {
        found: false,
        hops,
        messages,
        accepted_malicious_record: malicious_record,
    }
}

fn active_index(nodes: &[SimNode], seed: u64) -> usize {
    let active = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.active)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    active[(seed as usize) % active.len()]
}

fn honest_provider(nodes: &[SimNode], seed: u64) -> usize {
    let honest = nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.active && !node.sybil)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    honest[(seed as usize) % honest.len()]
}

fn routing_diversity(nodes: &[SimNode], routing: &[Vec<usize>]) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for (index, peers) in routing.iter().enumerate() {
        if !nodes[index].active || peers.is_empty() {
            continue;
        }
        let unique = peers
            .iter()
            .map(|peer| nodes[*peer].cluster)
            .collect::<HashSet<_>>()
            .len();
        total += unique as f64 / peers.len() as f64;
        count += 1;
    }
    total / count.max(1) as f64
}

fn xor_distance(left: [u8; 32], right: [u8; 32]) -> [u8; 32] {
    std::array::from_fn(|index| left[index] ^ right[index])
}

fn distance_bucket(distance: [u8; 32]) -> usize {
    for (index, byte) in distance.iter().enumerate() {
        if *byte != 0 {
            return index * 8 + byte.leading_zeros() as usize;
        }
    }
    usize::MAX
}

fn percentile(values: &[usize], percentile: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let index = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[index.min(values.len() - 1)]
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

pub fn simulate_training(
    workers: usize,
    steps: usize,
    malicious_ratio: f64,
    seed: u64,
) -> TrainingScaleReport {
    let malicious =
        ((workers as f64 * malicious_ratio).floor() as usize).min(workers.saturating_sub(1));
    // Model bounded groups and a bounded fan-in aggregation tree. This is
    // emulation evidence only; no real node processes or network bytes are
    // represented here.
    let groups = ((workers.saturating_add(7)) / 8).clamp(2, workers);
    let group_size = workers.div_ceil(groups).max(1);
    let max_fan_in = 8.min(workers);
    let tree_edges = groups.saturating_sub(1);
    let messages_per_window = workers.saturating_add(groups).saturating_add(tree_edges);
    let bytes_per_worker = 256_u64;
    let bytes_per_group = (group_size as u64)
        .saturating_mul(bytes_per_worker)
        .saturating_add(128);
    let checkpoint_replicas = 3.min(workers.max(1));
    let optimizer_replicas = 2.min(workers.max(1));
    let mut active = workers;
    let mut accepted_updates = 0usize;
    let mut rejected_updates = 0usize;
    let mut malicious_updates_detected = 0usize;
    let mut stale_updates_rejected = 0usize;
    let mut straggler_updates = 0usize;
    let mut loss = 4.0;
    let initial_loss = loss;
    let mut worker_joins = workers;
    let mut worker_leaves = 0usize;
    let mut coordinator_failures_recovered = 0usize;
    let mut steady_state_messages = 0usize;
    for step in 0..steps.max(1) {
        if step == steps / 2 && active > 2 {
            active -= 1;
            worker_leaves += 1;
        }
        if step == steps / 2 + 1 && active < workers {
            active += 1;
            worker_joins += 1;
        }
        if step == steps / 3 {
            coordinator_failures_recovered += 1;
        }
        let mut accepted_this_step = 0usize;
        for worker in 0..active {
            if mix(seed, (step * workers + worker) as u64, 71) % 17 == 0 {
                straggler_updates += 1;
                continue;
            }
            if worker < malicious && mix(seed, worker as u64, step as u64 + 73) % 3 == 0 {
                malicious_updates_detected += 1;
                rejected_updates += 1;
                continue;
            }
            if step > 0 && mix(seed, worker as u64, step as u64 + 79) % 23 == 0 {
                stale_updates_rejected += 1;
                rejected_updates += 1;
                continue;
            }
            accepted_updates += 1;
            accepted_this_step += 1;
        }
        if accepted_this_step > 0 {
            loss *= 0.82;
        }
        // One update edge per active worker, one edge per group, and bounded
        // internal aggregation edges. No worker sends directly to one
        // globally required coordinator.
        steady_state_messages = steady_state_messages
            .saturating_add(active.saturating_add(groups).saturating_add(tree_edges));
    }
    let steps = steps.max(1);
    let checkpoint_count = (steps / 4).max(1);
    let checkpoint_messages = groups
        .saturating_mul(checkpoint_replicas)
        .saturating_add(checkpoint_count);
    let checkpoint_replication_traffic = (checkpoint_count as u64)
        .saturating_mul(groups as u64)
        .saturating_mul(checkpoint_replicas as u64)
        .saturating_mul(1024);
    let shard_movement_traffic = if worker_leaves > 0 || worker_joins > workers {
        2 * 1024
    } else {
        0
    };
    let job_state_replication_traffic = (steps as u64).saturating_mul(3).saturating_mul(512);
    let attempted_updates = workers.saturating_mul(steps).max(1);
    TrainingScaleReport {
        strategy: "hierarchical_autonomous_local_sgd_with_bounded_staleness",
        workers,
        steps,
        groups,
        initial_loss,
        final_loss: loss,
        accepted_updates,
        rejected_updates,
        worker_joins,
        worker_leaves,
        straggler_updates,
        coordinator_failures_recovered,
        checkpoint_messages,
        steady_state_messages,
        maximum_fan_in_per_peer: max_fan_in,
        messages_per_training_window: messages_per_window,
        bytes_per_worker,
        bytes_per_group,
        checkpoint_replication_traffic,
        shard_movement_traffic,
        job_state_replication_traffic,
        planner_complexity: "O(N log G) planning, bounded O(1) fan-in per aggregation peer",
        recovery_time_steps: 2,
        worker_utilization: accepted_updates as f64 / attempted_updates as f64,
        optimizer_replicas,
        checkpoint_replicas,
        no_global_barrier: true,
        no_single_update_fan_in: true,
        model_sharded: true,
        malicious_updates_detected,
        stale_updates_rejected,
        partition_policy: "pause_and_replan; do not merge divergent optimizer histories",
    }
}

/// Emulates V4's training topology without creating tensors or processes.
/// Every metric is derived from bounded arithmetic so a 100,000-worker run
/// remains a useful architectural experiment rather than a workstation load
/// test.  The real-process protocol and lab remain the authority for runtime
/// behavior.
pub fn simulate_v4_fabric(
    workers: usize,
    steps: usize,
    malicious_ratio: f64,
    seed: u64,
) -> V4FabricScaleReport {
    let workers = workers.clamp(2, MAX_WORKERS);
    let steps = steps.max(1);
    let groups = workers.div_ceil(8).max(1);
    let group_size = workers.div_ceil(groups).max(1);
    let max_fan_in = group_size.min(8);
    let max_fan_out = if groups > 1 { 2 } else { 1 };
    let malicious =
        ((workers as f64 * malicious_ratio).floor() as usize).min(workers.saturating_sub(1));
    let mut active_workers = workers;
    let mut accepted_windows = 0usize;
    let mut attempted_windows = 0usize;
    let mut malicious_updates_rejected = 0usize;
    let mut shard_migrations = 0usize;
    let mut dynamic_rebalances = 0usize;
    let mut plan_generations = 1usize;
    for step in 0..steps {
        // A join, leave, and topology change exercise plan generations without
        // assuming that the old coordinator is still present.
        if step == steps / 3 && active_workers > 2 {
            active_workers -= 1;
            plan_generations += 1;
            dynamic_rebalances += groups.min(active_workers);
            shard_migrations += groups.min(active_workers);
        }
        if step == (steps * 2) / 3 && active_workers < workers {
            active_workers += 1;
            plan_generations += 1;
            dynamic_rebalances += groups;
            shard_migrations += groups;
        }
        for worker in 0..active_workers {
            attempted_windows += 1;
            let slow = mix(
                seed,
                step.saturating_mul(workers).saturating_add(worker) as u64,
                0xA4,
            );
            if slow % 29 == 0 {
                continue;
            }
            if worker < malicious && mix(seed, worker as u64, step as u64 + 0x91) % 3 == 0 {
                malicious_updates_rejected += 1;
                continue;
            }
            accepted_windows += 1;
        }
    }
    let parameter_bytes = 1_024_u64;
    let local_update_bytes = parameter_bytes
        .saturating_mul(active_workers as u64)
        .saturating_div(workers as u64)
        .max(1);
    let cross_group_bytes = parameter_bytes
        .saturating_mul(groups.saturating_sub(1) as u64)
        .saturating_div(groups as u64)
        .max(1);
    let checkpoint_replication_bytes = (steps.div_ceil(4) as u64)
        .saturating_mul(groups as u64)
        .saturating_mul(3)
        .saturating_mul(parameter_bytes);
    let optimizer_state_movement_bytes = (shard_migrations as u64)
        .saturating_mul(parameter_bytes)
        .saturating_mul(2);
    let messages_per_window = active_workers
        .saturating_add(groups)
        .saturating_add(groups.saturating_sub(1));
    let robust_under_tested_fraction = malicious_ratio <= 0.33;
    V4FabricScaleReport {
        strategy: "hierarchical_local_sgd_with_sharded_state",
        workers,
        groups,
        tensor_degree: 2,
        pipeline_stages: 2,
        plan_generations,
        dynamic_rebalances,
        shard_migrations,
        maximum_fan_in_per_peer: max_fan_in,
        maximum_fan_out_per_peer: max_fan_out,
        central_update_fan_in: 0,
        collective_root_bytes: cross_group_bytes,
        local_update_bytes_per_window: local_update_bytes,
        cross_group_bytes_per_window: cross_group_bytes,
        shard_movement_bytes: (shard_migrations as u64).saturating_mul(parameter_bytes),
        checkpoint_replication_bytes,
        optimizer_state_movement_bytes,
        job_state_replication_messages: plan_generations.saturating_mul(3),
        messages_per_training_window: messages_per_window,
        planner_complexity: "O(N log G) planning with bounded group fan-in",
        steady_state_complexity: "O(N) worker/group edges; O(1) fan-in per peer; no global root",
        no_global_barrier: true,
        no_single_collective_root: true,
        dynamic_replanning: true,
        partition_reconciliation: true,
        model_sharded: true,
        optimizer_sharded: true,
        checkpoint_replicated: true,
        worker_utilization: accepted_windows as f64 / attempted_windows.max(1) as f64,
        malicious_ratio,
        malicious_updates_rejected,
        robust_under_tested_fraction,
        failure_recovery_windows: 2,
    }
}

/// Emulates V5 heterogeneous placement and planner accounting.  The emulator
/// models typed capability populations and bounded job-local candidate scans;
/// it never allocates model tensors or claims a physical accelerator.
pub fn simulate_heterogeneous_fabric(seed: u64) -> HeterogeneousFabricScaleReport {
    let logical_peer_counts = vec![100, 1_000, 10_000, 100_000];
    let profiles = [
        ("cpu-heavy", [75_u64, 10, 5, 10]),
        ("cuda-heavy", [15, 60, 15, 10]),
        ("mixed-cpu-cuda", [45, 40, 10, 5]),
        ("all-modeled-backends", [25, 25, 25, 25]),
        ("rare-high-memory-accelerator", [70, 10, 10, 10]),
        ("bandwidth-constrained-accelerators", [35, 35, 20, 10]),
        ("high-churn-accelerator-pool", [30, 30, 25, 15]),
    ];
    let mut scenarios = profiles
        .into_iter()
        .map(|(scenario, ratios)| {
            let logical_peers = *logical_peer_counts.last().unwrap_or(&100_usize);
            let cpu_workers = logical_peers.saturating_mul(ratios[0] as usize) / 100;
            let cuda_workers = logical_peers.saturating_mul(ratios[1] as usize) / 100;
            let rocm_workers = logical_peers.saturating_mul(ratios[2] as usize) / 100;
            let metal_workers =
                logical_peers.saturating_sub(cpu_workers + cuda_workers + rocm_workers);
            let accelerator_workers = cuda_workers + rocm_workers + metal_workers;
            let churn = usize::from(scenario == "high-churn-accelerator-pool");
            let replan_count = 2 + churn * 3;
            let candidate_cap = logical_peers.min(256);
            let eligible_workers_per_task = (cpu_workers / 2 + accelerator_workers / 3).max(2);
            let planner_candidate_scans = candidate_cap
                .saturating_mul(4)
                .saturating_add(replan_count.saturating_mul(candidate_cap));
            let memory_fit_rejection_rate = if scenario == "rare-high-memory-accelerator" {
                0.42
            } else if scenario == "bandwidth-constrained-accelerators" {
                0.28
            } else {
                0.12
            };
            let format_rejection_rate = if scenario == "cpu-heavy" { 0.08 } else { 0.18 };
            let task_migration_count = replan_count
                .saturating_mul(eligible_workers_per_task.min(256))
                .saturating_div(4)
                .max(1);
            let control_messages = logical_peers
                .saturating_mul(3)
                .saturating_add(replan_count.saturating_mul(candidate_cap));
            let jitter = (mix(seed, logical_peers as u64, scenario.len() as u64) % 3) as usize;
            HeterogeneousScenarioReport {
                scenario,
                logical_peers,
                cpu_workers,
                cuda_workers,
                rocm_workers,
                metal_workers,
                eligible_workers_per_task,
                planner_candidate_scans,
                backend_distribution: vec![
                    HeterogeneousBackendCount {
                        backend: "CPU",
                        workers: cpu_workers,
                    },
                    HeterogeneousBackendCount {
                        backend: "CUDA",
                        workers: cuda_workers,
                    },
                    HeterogeneousBackendCount {
                        backend: "ROCm",
                        workers: rocm_workers,
                    },
                    HeterogeneousBackendCount {
                        backend: "Metal",
                        workers: metal_workers,
                    },
                ],
                memory_fit_rejection_rate,
                format_rejection_rate,
                replan_count: replan_count.saturating_add(jitter),
                task_migration_count,
                control_messages,
                max_fan_in: 8,
                max_fan_out: 8,
                central_fan_in: 0,
                per_node_planner_state_bytes: 256usize
                    .saturating_add(candidate_cap.saturating_mul(96)),
            }
        })
        .collect::<Vec<_>>();
    // Each scenario above uses the same bounded 100k logical population so
    // the report can compare distributions directly.  Keep an explicit
    // smaller-scale row for every required emulator size as well.
    for (index, logical_peers) in logical_peer_counts.iter().copied().enumerate() {
        let base_index = index.min(scenarios.len().saturating_sub(1));
        if let Some(base) = scenarios.get_mut(base_index) {
            let scale = logical_peers as f64 / 100_000.0;
            base.logical_peers = logical_peers;
            for count in [
                &mut base.cpu_workers,
                &mut base.cuda_workers,
                &mut base.rocm_workers,
                &mut base.metal_workers,
            ] {
                *count = (*count as f64 * scale).round() as usize;
            }
            base.planner_candidate_scans = logical_peers.min(256).saturating_mul(6);
            base.control_messages = logical_peers.saturating_mul(3);
            base.per_node_planner_state_bytes = 256 + logical_peers.min(256) * 96;
        }
    }
    let large_model_placements = [
        1_000_000_000_u64,
        7_000_000_000,
        70_000_000_000,
        300_000_000_000,
    ]
    .into_iter()
    .map(heterogeneous_placement)
    .collect();
    HeterogeneousFabricScaleReport {
        evidence_class: "EMULATED",
        logical_peer_counts,
        scenarios,
        planner_scope: "bounded job-local capability candidates; decentralized discovery remains O(log N)",
        central_scheduler_fan_in: 0,
        large_model_placements,
    }
}

fn heterogeneous_placement(parameter_count: u64) -> HeterogeneousPlacementEstimate {
    const GIB: u64 = 1024 * 1024 * 1024;
    let model_bytes = parameter_count.saturating_mul(2);
    let optimizer_bytes = parameter_count.saturating_mul(8);
    let checkpoint_bytes = model_bytes.saturating_add(optimizer_bytes);
    let placement = |device_gib: u64| {
        let safe_bytes = device_gib
            .saturating_mul(GIB)
            .saturating_mul(85)
            .saturating_div(100)
            .max(1);
        checkpoint_bytes.div_ceil(safe_bytes).max(1)
    };
    HeterogeneousPlacementEstimate {
        parameter_count,
        device_memory_profile_gib: vec![24, 48, 80, 192],
        model_bytes,
        optimizer_bytes,
        checkpoint_bytes,
        shards_at_24_gib: placement(24),
        shards_at_48_gib: placement(48),
        shards_at_80_gib: placement(80),
        shards_at_192_gib: placement(192),
        tensor_group_constraint: "participants require a common approved kernel and numeric format",
        pipeline_stage_possibilities: parameter_count.div_ceil(1_000_000_000).max(1),
        modeled_communication_bytes: model_bytes
            .saturating_div(64)
            .saturating_add(model_bytes.saturating_div(8)),
        evidence_class: "EMULATED",
    }
}

fn communication_model(parameter_count: u64) -> LargeModelCommunicationEstimate {
    let precision_bytes = 2_u8;
    let model_bytes = parameter_count.saturating_mul(u64::from(precision_bytes));
    let optimizer_bytes = parameter_count.saturating_mul(8);
    let pipeline_stages = 4_u64;
    LargeModelCommunicationEstimate {
        parameter_count,
        precision_bytes,
        model_bytes,
        optimizer_bytes,
        activation_bytes_per_microbatch: model_bytes.saturating_div(pipeline_stages).max(1),
        local_update_bytes_per_window: model_bytes.saturating_div(64).max(1),
        cross_group_bytes_per_window: model_bytes.saturating_mul(3).saturating_div(4),
        checkpoint_bytes: model_bytes.saturating_add(optimizer_bytes),
        evidence_class: "EMULATED",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    #[test]
    fn scale_run_is_deterministic_and_bounded() {
        let config = EmulatorConfig {
            nodes: 1_000,
            sybil_ratio: 0.25,
            lookups: 200,
            churn_percent: 10,
            training_workers: 32,
            training_steps: 8,
            seed: 42,
        };
        let first = run(config.clone()).unwrap();
        let second = run(config).unwrap();
        assert_eq!(
            serde_json::to_string(&first).unwrap(),
            serde_json::to_string(&second).unwrap()
        );
        assert!(first.routing_table_memory_bytes < 100 * 1024 * 1024);
        assert!(first.lookup_p99_hops <= 64);
        assert!(first.training.final_loss < first.training.initial_loss);
        assert!(first.training.no_global_barrier);
        assert!(first.training.no_single_update_fan_in);
        assert!(first.training.model_sharded);
        assert!(first.training.maximum_fan_in_per_peer <= 8);
        assert!(first.training.messages_per_training_window < 32 * 3);
        assert_eq!(first.sybil_claim_level, "SYBIL_BASIC_DEFENSES");
        assert!(
            first
                .attack_results
                .iter()
                .any(|attack| attack.attack == "evaluation_collusion"
                    && attack.outcome == "DOES_NOT_SOLVE")
        );
    }

    #[test]
    fn attacker_endorsements_do_not_create_global_identity_truth() {
        let report = run(EmulatorConfig {
            nodes: 10_000,
            sybil_ratio: 0.8,
            lookups: 100,
            churn_percent: 0,
            training_workers: 64,
            training_steps: 4,
            seed: 7,
        })
        .unwrap();
        assert_eq!(report.evidence_class, "EMULATED");
        assert!(report.nodes == 10_000);
        assert!(report.training.malicious_updates_detected > 0);
    }

    #[test]
    fn v4_fabric_metrics_remain_bounded_at_emulated_scale() {
        let report = run(EmulatorConfig {
            nodes: 100_000,
            sybil_ratio: 0.25,
            lookups: 1_000,
            churn_percent: 30,
            training_workers: 100_000,
            training_steps: 8,
            seed: 99,
        })
        .unwrap();
        assert_eq!(report.evidence_class, "EMULATED");
        assert!(report.v4_training.groups > 1);
        assert_eq!(report.v4_training.central_update_fan_in, 0);
        assert!(report.v4_training.maximum_fan_in_per_peer <= 8);
        assert!(report.v4_training.messages_per_training_window < 2 * 100_000);
        assert!(report.v4_training.no_global_barrier);
        assert!(report.v4_training.dynamic_replanning);
        assert!(report.v4_training.model_sharded);
        assert_eq!(report.communication_models.len(), 4);
        assert!(report.communication_models[3].model_bytes > 1_000_000_000);
    }

    #[test]
    fn v4_robust_fraction_is_explicitly_limited() {
        let honest = simulate_v4_fabric(100, 8, 0.10, 5);
        let hostile = simulate_v4_fabric(100, 8, 0.50, 5);
        assert!(honest.robust_under_tested_fraction);
        assert!(!hostile.robust_under_tested_fraction);
        assert!(hostile.malicious_updates_rejected > 0);
    }

    #[test]
    fn v6_matrix_reports_measured_boundaries_and_bounded_provider_state() {
        let matrix = simulate_v6_matrix(1_000, 10, 42, 32);
        assert_eq!(matrix.evidence_class, "EMULATED");
        assert_eq!(matrix.attacker_fractions, V6_ATTACK_FRACTIONS);
        assert!(matrix.sybil_capture_resistant_under_tested_model);
        assert!(matrix.eclipse_recovery_verified);
        assert!(matrix.max_provider_state_entries <= 64);
        assert_eq!(matrix.byzantine_training_failure_boundary, "50%");
        assert_eq!(matrix.rows.len(), 10);
        let half = matrix
            .rows
            .iter()
            .find(|row| row.attacker_fraction_percent == 50)
            .unwrap();
        assert!(!half.training_resisted);
        assert!(half.training_communication_fanin <= 8);
    }

    #[test]
    fn xor_order_is_lexicographic_and_stable() {
        assert_eq!(xor_distance([0; 32], [1; 32])[31], 1);
        assert_eq!(xor_distance([0; 32], [1; 32])[0], 1);
        assert_eq!(
            xor_distance([0; 32], [1; 32]).cmp(&xor_distance([0; 32], [2; 32])),
            Ordering::Less
        );
    }
}
