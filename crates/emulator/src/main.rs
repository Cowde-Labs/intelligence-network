//! Deterministic, in-process scale experiments.
//!
//! This executable deliberately does not open sockets or start node processes.
//! Its output is an emulation evidence class and must not be read as a
//! deployment measurement.

use clap::Parser;
use intelligence_emulator::{EmulatorConfig, run};

#[derive(Debug, Parser)]
#[command(
    name = "intelligence-emulator",
    about = "Run bounded deterministic V2/V3/V4/V5/V6 scale experiments"
)]
struct Cli {
    #[arg(long, default_value_t = 1_000)]
    nodes: usize,
    #[arg(long, default_value_t = 0.25)]
    sybil_ratio: f64,
    #[arg(long, default_value_t = 100)]
    lookups: usize,
    #[arg(long, default_value_t = 0)]
    churn_percent: u8,
    #[arg(long, default_value_t = 4)]
    workers: usize,
    #[arg(long, default_value_t = 16)]
    training_steps: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let report = run(EmulatorConfig {
        nodes: cli.nodes,
        sybil_ratio: cli.sybil_ratio,
        lookups: cli.lookups,
        churn_percent: cli.churn_percent,
        training_workers: cli.workers,
        training_steps: cli.training_steps,
        seed: cli.seed,
    })?;
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("INTELLIGENCE NETWORK — EMULATED SCALE REPORT");
        println!("evidence_class: {}", report.evidence_class);
        println!("nodes: {} (sybil: {})", report.nodes, report.sybil_nodes);
        println!(
            "lookup success: {:.2}% | p50/p95/p99 hops: {}/{}/{} | messages: {}",
            report.lookup_success_rate * 100.0,
            report.lookup_p50_hops,
            report.lookup_p95_hops,
            report.lookup_p99_hops,
            report.lookup_messages
        );
        println!(
            "honest provider discovery: {:.2}% | malicious acceptance: {:.2}% | routing diversity: {:.2}",
            report.honest_provider_discovery_rate * 100.0,
            report.malicious_record_acceptance_rate * 100.0,
            report.victim_routing_diversity
        );
        println!(
            "routing memory: {} bytes | peer state: {} bytes | trust state: {} bytes",
            report.routing_table_memory_bytes,
            report.peer_state_memory_bytes,
            report.trust_state_memory_bytes
        );
        println!("Sybil claim: {}", report.sybil_claim_level);
        for attack in &report.attack_results {
            println!(
                "attack {}: {} ({})",
                attack.attack, attack.outcome, attack.measurement
            );
        }
        println!(
            "training: {} workers, loss {:.4} -> {:.4}, updates {}, rejected {}",
            report.training.workers,
            report.training.initial_loss,
            report.training.final_loss,
            report.training.accepted_updates,
            report.training.rejected_updates
        );
        println!(
            "training topology: {} groups | max fan-in {} | messages/window {} | no global barrier {} | model sharded {}",
            report.training.groups,
            report.training.maximum_fan_in_per_peer,
            report.training.messages_per_training_window,
            report.training.no_global_barrier,
            report.training.model_sharded
        );
        println!(
            "V4 fabric: {} workers, {} groups | max fan-in {} | plan generations {} | shard migrations {} | central fan-in {} | no barrier {} | dynamic replan {}",
            report.v4_training.workers,
            report.v4_training.groups,
            report.v4_training.maximum_fan_in_per_peer,
            report.v4_training.plan_generations,
            report.v4_training.shard_migrations,
            report.v4_training.central_update_fan_in,
            report.v4_training.no_global_barrier,
            report.v4_training.dynamic_replanning
        );
        println!(
            "V4 scale: steady state {} | optimizer sharded {} | checkpoints replicated {} | robust <=33% {}",
            report.v4_training.steady_state_complexity,
            report.v4_training.optimizer_sharded,
            report.v4_training.checkpoint_replicated,
            report.v4_training.robust_under_tested_fraction
        );
        println!(
            "V5 heterogeneous: {} scenarios | peer sizes {:?} | planner scope: {} | central scheduler fan-in {}",
            report.v5_heterogeneous.scenarios.len(),
            report.v5_heterogeneous.logical_peer_counts,
            report.v5_heterogeneous.planner_scope,
            report.v5_heterogeneous.central_scheduler_fan_in
        );
        println!(
            "V6 adversarial matrix: {} rows | Sybil capture resistant {} | eclipse recovery {} | evaluator failure boundary {} | training failure boundary {}",
            report.v6_security.rows.len(),
            report
                .v6_security
                .sybil_capture_resistant_under_tested_model,
            report.v6_security.eclipse_recovery_verified,
            report.v6_security.colluding_evaluator_failure_boundary,
            report.v6_security.byzantine_training_failure_boundary
        );
        for scenario in &report.v5_heterogeneous.scenarios {
            println!(
                "V5 {}: peers {} | CPU/CUDA/ROCm/Metal {}/{}/{}/{} | candidate scans {} | replans {} | memory reject {:.2}% | format reject {:.2}%",
                scenario.scenario,
                scenario.logical_peers,
                scenario.cpu_workers,
                scenario.cuda_workers,
                scenario.rocm_workers,
                scenario.metal_workers,
                scenario.planner_candidate_scans,
                scenario.replan_count,
                scenario.memory_fit_rejection_rate * 100.0,
                scenario.format_rejection_rate * 100.0
            );
        }
        for model in &report.communication_models {
            println!(
                "modeled {} parameters: model {} B, optimizer {} B, checkpoint {} B",
                model.parameter_count,
                model.model_bytes,
                model.optimizer_bytes,
                model.checkpoint_bytes
            );
        }
    }
    Ok(())
}
