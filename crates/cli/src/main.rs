use clap::{Parser, Subcommand};
use intelligence_node::{AdminRequest, Identity, Node, NodeConfig, admin_call};
use std::{
    fs,
    path::{Path, PathBuf},
};
use tokio::signal;

#[derive(Debug, Parser)]
#[command(
    name = "intelligence",
    version,
    about = "Operate an Intelligence Network node",
    long_about = "Operate a locally sovereign Intelligence Network node.\n\nRun a node, inspect local state, route bounded work, and manage distributed training through the authenticated peer protocol."
)]
struct Cli {
    #[arg(
        long,
        global = true,
        default_value = "intelligence.toml",
        env = "INTELLIGENCE_CONFIG"
    )]
    config: PathBuf,
    #[arg(
        long,
        global = true,
        help = "Emit machine-readable JSON where applicable"
    )]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Start the node process")]
    Run,
    #[command(about = "Create a node configuration and identity")]
    Init,
    #[command(about = "Print software and protocol versions")]
    Version,
    #[command(about = "Print the local node identity")]
    Identity,
    #[command(about = "Rotate the node identity using a signed transition")]
    Rotate {
        #[arg(long)]
        new_path: PathBuf,
        #[arg(long, default_value_t = 1)]
        sequence: u64,
        #[arg(long)]
        valid_until: Option<u64>,
    },
    #[command(about = "Show node health, peers, jobs, and resource counters")]
    Status,
    #[command(about = "List currently known peers")]
    Peers,
    #[command(about = "List locally advertised capabilities")]
    Capabilities,
    #[command(about = "List persisted and active jobs")]
    Jobs,
    #[command(about = "Print the effective node configuration")]
    Config,
    #[command(about = "Show bounded DHT routing statistics")]
    DhtStats,
    #[command(about = "Publish a signed local DHT record")]
    DhtPublish {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        value: String,
        #[arg(long)]
        ttl_seconds: Option<u64>,
        #[arg(long)]
        sequence: Option<u64>,
    },
    #[command(about = "Look up a DHT record through the authenticated network")]
    DhtLookup {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        name: String,
    },
    #[command(about = "Find peers near a routing key")]
    DhtFindNode {
        #[arg(long)]
        key: String,
    },
    #[command(about = "Inspect the local evidence-based trust decision for a peer")]
    Trust {
        #[arg(long)]
        subject: String,
    },
    #[command(about = "Plan a bounded distributed training job")]
    PlanTraining {
        #[arg(long)]
        model_bytes: u64,
        #[arg(long, default_value = "selective")]
        data_locality: String,
        #[arg(long)]
        workers: Option<u16>,
    },
    #[command(about = "Submit an inference job")]
    Infer {
        #[arg(long)]
        capability: String,
        #[arg(long)]
        input: String,
        #[arg(long)]
        deadline_ms: Option<u64>,
        #[arg(long)]
        max_output_bytes: Option<u32>,
        #[arg(long)]
        local_only: bool,
        #[arg(
            long,
            help = "Use a caller-selected 16-byte hex job ID so it can be cancelled while running"
        )]
        job_id: Option<String>,
    },
    #[command(about = "Register a local model artifact")]
    RegisterModel {
        #[arg(long)]
        path: String,
        #[arg(long)]
        identity: String,
        #[arg(long, default_value = "opaque")]
        format: String,
        #[arg(long)]
        local_only: bool,
    },
    #[command(about = "Inspect a locally stored artifact")]
    Inspect {
        #[arg(long)]
        artifact: String,
    },
    #[command(about = "Fetch and verify an artifact from a peer")]
    FetchArtifact {
        #[arg(long)]
        peer: String,
        #[arg(long)]
        artifact: String,
    },
    #[command(about = "Evaluate a text sample through the network")]
    Evaluate {
        #[arg(long)]
        text: String,
        #[arg(long)]
        expected_label: String,
        #[arg(long)]
        deadline_ms: Option<u64>,
    },
    #[command(about = "Run the small distributed reference-training job")]
    TrainReference {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long, help = "Resume from a locally verified checkpoint artifact")]
        resume_checkpoint: Option<String>,
    },
    #[command(about = "Run the distributed training fabric")]
    TrainV3 {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 3)]
        local_steps: u16,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(about = "Start distributed training and return immediately")]
    TrainV3Start {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 3)]
        local_steps: u16,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(
        about = "Run the frontier training fabric",
        visible_alias = "train-frontier"
    )]
    TrainV4 {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(
        about = "Start frontier training and return immediately",
        visible_alias = "train-frontier-start"
    )]
    TrainV4Start {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(
        about = "Plan a frontier training graph",
        visible_alias = "plan-frontier-training"
    )]
    PlanTrainingV4 {
        #[arg(long)]
        model_bytes: u64,
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long, default_value = "local_sgd")]
        strategy: String,
        #[arg(long)]
        tensor_degree: Option<u16>,
        #[arg(long)]
        pipeline_stages: Option<u16>,
    },
    #[command(about = "Replan a frontier training graph")]
    ReplanTrainingV4 {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        model_bytes: u64,
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long, default_value = "local_sgd")]
        strategy: String,
        #[arg(long)]
        tensor_degree: Option<u16>,
        #[arg(long)]
        pipeline_stages: Option<u16>,
    },
    #[command(about = "Activate an approved frontier training graph")]
    ActivateTrainingV4 {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Seed a frontier training shard for a reference operation")]
    V4SeedShard {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        state: String,
    },
    #[command(about = "Migrate a frontier training shard")]
    V4MigrateShard {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        target: String,
    },
    #[command(about = "Run the bounded tensor-parallel reference operation")]
    V4TensorDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Run the bounded pipeline-parallel reference operation")]
    V4PipelineDemo {
        #[arg(long, value_delimiter = ',')]
        stages: Vec<String>,
        #[arg(long)]
        microbatches: Option<u16>,
    },
    #[command(about = "Run the bounded decentralized collective reference operation")]
    V4CollectiveDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Reconcile two compatible training branches")]
    V4ReconcileDemo {
        #[arg(long)]
        worker: String,
        #[arg(long)]
        left_value: i64,
        #[arg(long)]
        right_value: i64,
        #[arg(long, default_value = "local_sgd")]
        policy: String,
    },
    #[command(about = "Run the bounded robust-aggregation reference operation")]
    V4ByzantineDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
        #[arg(long)]
        malicious: Option<u16>,
        #[arg(long, default_value = "median")]
        policy: String,
    },
    #[command(about = "Replicate a training state record to selected peers")]
    V4ReplicateState {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Show the state of a distributed training job")]
    TrainingStatus {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Cancel a running job")]
    Cancel {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Request a graceful node shutdown")]
    Shutdown,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.command {
        Command::Run => run_node(&cli.config).await?,
        Command::Init => init_config(&cli.config)?,
        Command::Version => print_json(
            serde_json::json!({
                "software_version": env!("CARGO_PKG_VERSION"),
                "protocol_version": format!(
                    "{}.{}",
                    intelligence_protocol::PROTOCOL_MAJOR,
                    intelligence_protocol::PROTOCOL_MINOR
                ),
                "commit": option_env!("INTELLIGENCE_COMMIT").unwrap_or("unknown"),
                "target_arch": std::env::consts::ARCH,
                "target_os": std::env::consts::OS,
            }),
            cli.json,
        )?,
        Command::Identity => {
            let config = load_config(&cli.config)?;
            let identity_path = config
                .identity_path
                .clone()
                .unwrap_or_else(|| config.data_dir.join("identity.key"));
            let identity = Identity::load_or_generate(identity_path)?;
            print_json(
                serde_json::json!({
                    "node_id": identity.node_id().to_string(),
                    "public_key": hex::encode(identity.public_key()),
                }),
                cli.json,
            )?;
        }
        Command::Rotate {
            new_path,
            sequence,
            valid_until,
        } => {
            let config = load_config(&cli.config)?;
            let old_path = config
                .identity_path
                .clone()
                .unwrap_or_else(|| config.data_dir.join("identity.key"));
            let expiry = valid_until.unwrap_or_else(|| {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |duration| duration.as_secs())
                    .saturating_add(365 * 24 * 60 * 60)
            });
            let rotation = Identity::rotate(old_path, new_path, sequence, expiry)?;
            print_json(serde_json::to_value(rotation)?, cli.json)?;
        }
        Command::Status => call_and_print(&cli.config, AdminRequest::Status, cli.json).await?,
        Command::Peers => call_and_print(&cli.config, AdminRequest::Peers, cli.json).await?,
        Command::Capabilities => {
            call_and_print(&cli.config, AdminRequest::Capabilities, cli.json).await?
        }
        Command::Jobs => call_and_print(&cli.config, AdminRequest::Jobs, cli.json).await?,
        Command::Config => call_and_print(&cli.config, AdminRequest::Config, cli.json).await?,
        Command::DhtStats => call_and_print(&cli.config, AdminRequest::DhtStats, cli.json).await?,
        Command::DhtPublish {
            namespace,
            name,
            value,
            ttl_seconds,
            sequence,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::DhtPublish {
                    namespace,
                    name,
                    value,
                    ttl_seconds,
                    sequence,
                },
                cli.json,
            )
            .await?
        }
        Command::DhtLookup { namespace, name } => {
            call_and_print(
                &cli.config,
                AdminRequest::DhtLookup { namespace, name },
                cli.json,
            )
            .await?
        }
        Command::DhtFindNode { key } => {
            call_and_print(&cli.config, AdminRequest::DhtFindNode { key }, cli.json).await?
        }
        Command::Trust { subject } => {
            call_and_print(&cli.config, AdminRequest::Trust { subject }, cli.json).await?
        }
        Command::PlanTraining {
            model_bytes,
            data_locality,
            workers,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::PlanTraining {
                    model_bytes,
                    data_locality,
                    workers,
                },
                cli.json,
            )
            .await?
        }
        Command::Infer {
            capability,
            input,
            deadline_ms,
            max_output_bytes,
            local_only,
            job_id,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::Infer {
                    capability,
                    input,
                    deadline_ms,
                    max_output_bytes,
                    allow_input_transfer: Some(!local_only),
                    job_id,
                },
                cli.json,
            )
            .await?
        }
        Command::RegisterModel {
            path,
            identity,
            format,
            local_only,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::RegisterModel {
                    path,
                    identity,
                    format,
                    local_only,
                },
                cli.json,
            )
            .await?
        }
        Command::Inspect { artifact } => {
            call_and_print(&cli.config, AdminRequest::Inspect { artifact }, cli.json).await?
        }
        Command::FetchArtifact { peer, artifact } => {
            call_and_print(
                &cli.config,
                AdminRequest::FetchArtifact { peer, artifact },
                cli.json,
            )
            .await?
        }
        Command::Evaluate {
            text,
            expected_label,
            deadline_ms,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::Evaluate {
                    text,
                    expected_label,
                    deadline_ms,
                },
                cli.json,
            )
            .await?
        }
        Command::TrainReference {
            workers,
            steps,
            resume_checkpoint,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainReference {
                    workers,
                    steps,
                    resume_checkpoint,
                },
                cli.json,
            )
            .await?
        }
        Command::TrainV3 {
            workers,
            windows,
            local_steps,
            checkpoint_every,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainV3 {
                    workers,
                    windows,
                    local_steps: Some(local_steps),
                    checkpoint_every: Some(checkpoint_every),
                },
                cli.json,
            )
            .await?
        }
        Command::TrainV3Start {
            workers,
            windows,
            local_steps,
            checkpoint_every,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainV3Start {
                    workers,
                    windows,
                    local_steps: Some(local_steps),
                    checkpoint_every: Some(checkpoint_every),
                },
                cli.json,
            )
            .await?
        }
        Command::TrainV4 {
            workers,
            windows,
            checkpoint_every,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainV4 {
                    workers,
                    windows,
                    checkpoint_every: Some(checkpoint_every),
                },
                cli.json,
            )
            .await?
        }
        Command::TrainV4Start {
            workers,
            windows,
            checkpoint_every,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainV4Start {
                    workers,
                    windows,
                    checkpoint_every: Some(checkpoint_every),
                },
                cli.json,
            )
            .await?
        }
        Command::TrainingStatus { job_id } => {
            call_and_print(
                &cli.config,
                AdminRequest::TrainingStatus { job_id },
                cli.json,
            )
            .await?
        }
        Command::PlanTrainingV4 {
            model_bytes,
            workers,
            strategy,
            tensor_degree,
            pipeline_stages,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::PlanTrainingV4 {
                    model_bytes,
                    workers,
                    strategy,
                    tensor_degree,
                    pipeline_stages,
                },
                cli.json,
            )
            .await?
        }
        Command::ReplanTrainingV4 {
            job_id,
            model_bytes,
            workers,
            strategy,
            tensor_degree,
            pipeline_stages,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::ReplanTrainingV4 {
                    job_id,
                    model_bytes,
                    workers,
                    strategy,
                    tensor_degree,
                    pipeline_stages,
                },
                cli.json,
            )
            .await?
        }
        Command::ActivateTrainingV4 { job_id } => {
            call_and_print(
                &cli.config,
                AdminRequest::ActivateTrainingV4 { job_id },
                cli.json,
            )
            .await?
        }
        Command::V4SeedShard {
            job_id,
            shard_id,
            state,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4SeedShard {
                    job_id,
                    shard_id,
                    state,
                },
                cli.json,
            )
            .await?
        }
        Command::V4MigrateShard {
            job_id,
            shard_id,
            target,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4MigrateShard {
                    job_id,
                    shard_id,
                    target,
                },
                cli.json,
            )
            .await?
        }
        Command::V4TensorDemo { workers } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4TensorDemo { workers },
                cli.json,
            )
            .await?
        }
        Command::V4PipelineDemo {
            stages,
            microbatches,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4PipelineDemo {
                    stages,
                    microbatches,
                },
                cli.json,
            )
            .await?
        }
        Command::V4CollectiveDemo { workers } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4CollectiveDemo { workers },
                cli.json,
            )
            .await?
        }
        Command::V4ReconcileDemo {
            worker,
            left_value,
            right_value,
            policy,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4ReconcileDemo {
                    worker,
                    left_value,
                    right_value,
                    policy,
                },
                cli.json,
            )
            .await?
        }
        Command::V4ByzantineDemo {
            workers,
            malicious,
            policy,
        } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4ByzantineDemo {
                    workers,
                    malicious,
                    policy,
                },
                cli.json,
            )
            .await?
        }
        Command::V4ReplicateState { workers } => {
            call_and_print(
                &cli.config,
                AdminRequest::V4ReplicateState { workers },
                cli.json,
            )
            .await?
        }
        Command::Cancel { job_id } => {
            call_and_print(&cli.config, AdminRequest::Cancel { job_id }, cli.json).await?
        }
        Command::Shutdown => call_and_print(&cli.config, AdminRequest::Shutdown, cli.json).await?,
    }
    Ok(())
}

async fn run_node(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let node = Node::start(load_config(path)?).await?;
    tokio::select! {
        _ = node.wait_for_shutdown() => {},
        signal_result = signal::ctrl_c() => {
            signal_result?;
            node.shutdown().await;
        }
    }
    Ok(())
}

async fn call_and_print(
    path: &PathBuf,
    request: AdminRequest,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(path)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    let value = admin_call(socket, &request).await?;
    print_json(value, json)?;
    Ok(())
}

fn load_config(path: &PathBuf) -> Result<NodeConfig, Box<dyn std::error::Error>> {
    if path.exists() {
        Ok(NodeConfig::load(path)?)
    } else {
        Ok(NodeConfig::from_environment()?)
    }
}

fn init_config(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("configuration already exists: {}", path.display()).into());
    }
    let mut config = NodeConfig::default();
    if config.data_dir.as_path() == Path::new("state") {
        config.data_dir = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("state");
    }
    let builtin = intelligence_node::CapabilityConfig {
        name: "inference.text".to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "builtin_text".to_string(),
        model: Some("builtin.tiny-sentiment.v1".to_string()),
        model_path: None,
        program: None,
        args: Vec::new(),
        env: std::collections::BTreeMap::new(),
        metadata: std::collections::BTreeMap::new(),
        sandbox: "trusted_local".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 1000,
    };
    config.capabilities.push(builtin.clone());
    let mut evaluation = builtin;
    evaluation.name = "evaluation.text".to_string();
    config.capabilities.push(evaluation);
    let training = intelligence_node::CapabilityConfig {
        name: "training.reference".to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "builtin_training".to_string(),
        model: None,
        model_path: None,
        program: None,
        args: Vec::new(),
        env: std::collections::BTreeMap::new(),
        metadata: std::collections::BTreeMap::new(),
        sandbox: "trusted_local".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 1000,
    };
    config.capabilities.push(training);
    config.normalize();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, config.to_toml()?)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn print_json(value: serde_json::Value, _json: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .json()
        .try_init();
}
