use clap::{Args, Parser, Subcommand, ValueEnum};
use intelligence_node::{AdminRequest, Identity, Node, NodeConfig, admin_call};
use std::{
    collections::BTreeSet,
    env,
    fs::{self, OpenOptions},
    io::Write,
    net::{SocketAddr, ToSocketAddrs, UdpSocket},
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::{signal, time::sleep};

#[derive(Debug, Parser)]
#[command(
    name = "intelligence",
    version,
    about = "Run AI across independent machines without one central server."
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
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    #[command(flatten)]
    Current(Command),
    #[command(flatten)]
    Legacy(LegacyCommand),
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(
        about = "Start a node in the background (creates config and identity on first run)",
        display_order = 1
    )]
    Up {
        #[arg(
            long = "peer",
            visible_alias = "peers",
            value_delimiter = ',',
            help = "Add a bootstrap peer (repeat or separate with commas)"
        )]
        peers: Vec<String>,
        #[arg(long, help = "Do not add the replaceable default bootstrap seeds")]
        no_default_seeds: bool,
        #[arg(
            long = "capability",
            value_delimiter = ',',
            help = "Enable a safe built-in capability (repeat or separate with commas)"
        )]
        capabilities: Vec<String>,
        #[arg(long, help = "Bind the node to a specific local address")]
        listen_addr: Option<String>,
    },
    #[command(about = "Stop the local node", display_order = 2)]
    Down,
    #[command(
        about = "Node health, peers, jobs and resource counters",
        display_order = 3
    )]
    Status,
    #[command(
        about = "Check networking, storage, hardware, models and node health",
        display_order = 4
    )]
    Doctor,
    #[command(about = "Run an inference job", display_order = 5)]
    Infer {
        #[arg(value_name = "TEXT")]
        text: Option<String>,
        #[arg(long, default_value = "inference.text")]
        capability: String,
        #[arg(long)]
        input: Option<String>,
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
    #[command(
        about = "Evaluate a text sample through the network and record evidence",
        display_order = 6
    )]
    Evaluate {
        #[arg(long)]
        text: String,
        #[arg(long)]
        expected_label: String,
        #[arg(long)]
        deadline_ms: Option<u64>,
    },
    #[command(about = "Manage local models", display_order = 7)]
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    #[command(
        about = "Inspect and fetch content-addressed artifacts",
        display_order = 8
    )]
    Artifact {
        #[command(subcommand)]
        command: ArtifactCommand,
    },
    #[command(
        about = "Distributed training",
        display_order = 9,
        args_conflicts_with_subcommands = true
    )]
    Train {
        #[command(subcommand)]
        command: Option<TrainCommand>,
        #[command(flatten)]
        args: TrainArgs,
    },
    #[command(about = "Peers, capabilities and routing", display_order = 10)]
    Network {
        #[command(subcommand)]
        command: NetworkCommand,
    },
    #[command(
        about = "Local trust decisions about peers",
        display_order = 11,
        args_conflicts_with_subcommands = true
    )]
    Trust {
        #[command(subcommand)]
        command: Option<TrustCommand>,
        #[arg(long, help = "Peer node ID (equivalent to `trust inspect --subject`)")]
        subject: Option<String>,
    },
    #[command(about = "Show or rotate the node identity", display_order = 12)]
    Identity {
        #[command(subcommand)]
        command: Option<IdentityCommand>,
    },
    #[command(about = "List or cancel jobs", display_order = 13)]
    Jobs {
        #[command(subcommand)]
        command: Option<JobsCommand>,
    },
    #[command(about = "Print the effective configuration", display_order = 14)]
    Config,
    #[command(about = "Optional systemd user service", display_order = 15)]
    Service {
        #[command(subcommand)]
        command: ServiceCommand,
    },
    #[command(about = "Run the node in the foreground", display_order = 16)]
    Run,
    #[command(
        about = "Create config and identity without starting the node",
        display_order = 17
    )]
    Init,
    #[command(about = "Print software and protocol versions", display_order = 18)]
    Version,
    #[command(
        hide = true,
        about = "Internal reference operations and debugging (unstable)"
    )]
    Dev {
        #[command(subcommand)]
        command: DevCommand,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TrainMode {
    /// Shard the model graph across workers
    Sharded,
    /// Synchronous local-SGD rounds across workers
    LocalSgd,
}

#[derive(Debug, Args)]
struct TrainArgs {
    #[arg(long, value_enum, default_value = "sharded")]
    mode: TrainMode,
    #[arg(long, help = "Number of worker peers to use")]
    workers: Option<u16>,
    #[arg(long, help = "Number of training windows to run")]
    windows: Option<u64>,
    #[arg(long, default_value_t = 2, help = "Write a checkpoint every N windows")]
    checkpoint_every: u64,
    #[arg(long, help = "Local steps per window (only with --mode local-sgd)")]
    local_steps: Option<u16>,
}

impl Default for TrainArgs {
    fn default() -> Self {
        Self {
            mode: TrainMode::Sharded,
            workers: None,
            windows: None,
            checkpoint_every: 2,
            local_steps: None,
        }
    }
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    #[command(about = "Import and hash a local model without downloading anything")]
    Add {
        path: PathBuf,
        #[arg(long, help = "Stable local model identity (defaults to the file name)")]
        identity: Option<String>,
        #[arg(long, help = "Model format (defaults to the file extension)")]
        format: Option<String>,
    },
    #[command(
        about = "List model files present in the local model directories",
        alias = "scan"
    )]
    List,
    #[command(about = "Register a local model artifact")]
    Register {
        #[arg(long)]
        path: String,
        #[arg(long)]
        identity: String,
        #[arg(long, default_value = "opaque")]
        format: String,
        #[arg(long)]
        local_only: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ArtifactCommand {
    #[command(about = "Inspect a locally stored artifact")]
    Inspect {
        #[arg(long)]
        artifact: String,
    },
    #[command(about = "Fetch and verify an artifact from a peer")]
    Fetch {
        #[arg(long)]
        peer: String,
        #[arg(long)]
        artifact: String,
    },
}

#[derive(Debug, Subcommand)]
enum NetworkCommand {
    #[command(about = "List currently known peers")]
    Peers,
    #[command(about = "List locally advertised capabilities")]
    Capabilities,
    #[command(about = "Show bounded DHT routing statistics")]
    Stats,
    #[command(about = "Publish a signed local DHT record")]
    Publish {
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
    Lookup {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        name: String,
    },
    #[command(about = "Find peers near a routing key")]
    Find {
        #[arg(long)]
        key: String,
    },
}

#[derive(Debug, Subcommand)]
enum TrustCommand {
    #[command(about = "Inspect the local evidence-based trust decision for a peer")]
    Inspect {
        #[arg(long)]
        subject: String,
    },
}

#[derive(Debug, Subcommand)]
enum IdentityCommand {
    #[command(about = "Rotate the node identity using a signed transition")]
    Rotate {
        #[arg(long)]
        new_path: PathBuf,
        #[arg(long, default_value_t = 1)]
        sequence: u64,
        #[arg(long)]
        valid_until: Option<u64>,
    },
}

#[derive(Debug, Subcommand)]
enum JobsCommand {
    #[command(about = "Cancel a running job")]
    Cancel {
        #[arg(long)]
        job_id: String,
    },
}

#[derive(Debug, Subcommand)]
enum ServiceCommand {
    #[command(about = "Install a per-user service without root or a system daemon")]
    Install,
}

#[derive(Debug, Subcommand)]
enum TrainCommand {
    #[command(about = "Start distributed training and return immediately")]
    Start {
        #[command(flatten)]
        args: TrainArgs,
    },
    #[command(about = "Plan a distributed training job")]
    Plan {
        #[arg(long)]
        model_bytes: u64,
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long, value_enum, default_value = "sharded")]
        mode: TrainMode,
        #[arg(long, default_value = "local_sgd")]
        strategy: String,
        #[arg(long)]
        tensor_degree: Option<u16>,
        #[arg(long)]
        pipeline_stages: Option<u16>,
        #[arg(long)]
        data_locality: Option<String>,
    },
    #[command(about = "Replan a distributed training graph")]
    Replan {
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
    #[command(about = "Activate an approved training graph")]
    Activate {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Show the state of a distributed training job")]
    Status {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Cancel a running job")]
    Cancel {
        #[arg(long)]
        job_id: String,
    },
    #[command(about = "Migrate a training shard to another peer")]
    Migrate {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        target: String,
    },
    #[command(about = "Replicate a training state record to selected peers")]
    Replicate {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Reconcile two compatible training branches")]
    Reconcile {
        #[arg(long)]
        worker: String,
        #[arg(long)]
        left_value: i64,
        #[arg(long)]
        right_value: i64,
        #[arg(long, default_value = "local_sgd")]
        policy: String,
    },
    #[command(about = "Run the small synchronous reference training job")]
    Reference {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long, help = "Resume from a locally verified checkpoint artifact")]
        resume_checkpoint: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum DevCommand {
    #[command(about = "Seed a training shard for a reference operation")]
    SeedShard {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        state: String,
    },
    #[command(about = "Run the bounded tensor-parallel reference operation")]
    TensorDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Run the bounded pipeline-parallel reference operation")]
    PipelineDemo {
        #[arg(long, value_delimiter = ',')]
        stages: Vec<String>,
        #[arg(long)]
        microbatches: Option<u16>,
    },
    #[command(about = "Run the bounded decentralized collective reference operation")]
    CollectiveDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(about = "Run the bounded robust-aggregation reference operation")]
    ByzantineDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
        #[arg(long)]
        malicious: Option<u16>,
        #[arg(long, default_value = "median")]
        policy: String,
    },
    #[command(about = "Request a graceful node shutdown")]
    Shutdown,
}

#[derive(Debug, Subcommand)]
enum LegacyCommand {
    #[command(hide = true)]
    Peers,
    #[command(hide = true)]
    Capabilities,
    #[command(hide = true)]
    DhtStats,
    #[command(hide = true)]
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
    #[command(hide = true)]
    DhtLookup {
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        name: String,
    },
    #[command(hide = true)]
    DhtFindNode {
        #[arg(long)]
        key: String,
    },
    #[command(hide = true)]
    Rotate {
        #[arg(long)]
        new_path: PathBuf,
        #[arg(long, default_value_t = 1)]
        sequence: u64,
        #[arg(long)]
        valid_until: Option<u64>,
    },
    #[command(hide = true)]
    Cancel {
        #[arg(long)]
        job_id: String,
    },
    #[command(hide = true)]
    Shutdown,
    #[command(hide = true)]
    Inspect {
        #[arg(long)]
        artifact: String,
    },
    #[command(hide = true)]
    FetchArtifact {
        #[arg(long)]
        peer: String,
        #[arg(long)]
        artifact: String,
    },
    #[command(hide = true)]
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
    #[command(hide = true)]
    PlanTraining {
        #[arg(long)]
        model_bytes: u64,
        #[arg(long, default_value = "selective")]
        data_locality: String,
        #[arg(long)]
        workers: Option<u16>,
    },
    #[command(hide = true)]
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
    #[command(hide = true)]
    PlanFrontierTraining {
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
    #[command(hide = true)]
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
    #[command(hide = true)]
    ActivateTrainingV4 {
        #[arg(long)]
        job_id: String,
    },
    #[command(hide = true)]
    TrainReference {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        steps: Option<u64>,
        #[arg(long, help = "Resume from a locally verified checkpoint artifact")]
        resume_checkpoint: Option<String>,
    },
    #[command(hide = true)]
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
    #[command(hide = true)]
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
    #[command(hide = true)]
    TrainV4 {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(hide = true)]
    TrainFrontier {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(hide = true)]
    TrainV4Start {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(hide = true)]
    TrainFrontierStart {
        #[arg(long)]
        workers: Option<u16>,
        #[arg(long)]
        windows: Option<u64>,
        #[arg(long, default_value_t = 2)]
        checkpoint_every: u64,
    },
    #[command(hide = true)]
    TrainingStatus {
        #[arg(long)]
        job_id: String,
    },
    #[command(hide = true)]
    V4SeedShard {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        state: String,
    },
    #[command(hide = true)]
    V4MigrateShard {
        #[arg(long)]
        job_id: String,
        #[arg(long)]
        shard_id: u16,
        #[arg(long)]
        target: String,
    },
    #[command(hide = true)]
    V4ReplicateState {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(hide = true)]
    V4TensorDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(hide = true)]
    V4PipelineDemo {
        #[arg(long, value_delimiter = ',')]
        stages: Vec<String>,
        #[arg(long)]
        microbatches: Option<u16>,
    },
    #[command(hide = true)]
    V4CollectiveDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
    },
    #[command(hide = true)]
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
    #[command(hide = true)]
    V4ByzantineDemo {
        #[arg(long, value_delimiter = ',')]
        workers: Vec<String>,
        #[arg(long)]
        malicious: Option<u16>,
        #[arg(long, default_value = "median")]
        policy: String,
    },
}

impl LegacyCommand {
    fn into_current(self) -> Command {
        match self {
            Self::Peers => Command::Network {
                command: NetworkCommand::Peers,
            },
            Self::Capabilities => Command::Network {
                command: NetworkCommand::Capabilities,
            },
            Self::DhtStats => Command::Network {
                command: NetworkCommand::Stats,
            },
            Self::DhtPublish {
                namespace,
                name,
                value,
                ttl_seconds,
                sequence,
            } => Command::Network {
                command: NetworkCommand::Publish {
                    namespace,
                    name,
                    value,
                    ttl_seconds,
                    sequence,
                },
            },
            Self::DhtLookup { namespace, name } => Command::Network {
                command: NetworkCommand::Lookup { namespace, name },
            },
            Self::DhtFindNode { key } => Command::Network {
                command: NetworkCommand::Find { key },
            },
            Self::Rotate {
                new_path,
                sequence,
                valid_until,
            } => Command::Identity {
                command: Some(IdentityCommand::Rotate {
                    new_path,
                    sequence,
                    valid_until,
                }),
            },
            Self::Cancel { job_id } => Command::Jobs {
                command: Some(JobsCommand::Cancel { job_id }),
            },
            Self::Shutdown => Command::Dev {
                command: DevCommand::Shutdown,
            },
            Self::Inspect { artifact } => Command::Artifact {
                command: ArtifactCommand::Inspect { artifact },
            },
            Self::FetchArtifact { peer, artifact } => Command::Artifact {
                command: ArtifactCommand::Fetch { peer, artifact },
            },
            Self::RegisterModel {
                path,
                identity,
                format,
                local_only,
            } => Command::Model {
                command: ModelCommand::Register {
                    path,
                    identity,
                    format,
                    local_only,
                },
            },
            Self::PlanTraining {
                model_bytes,
                data_locality,
                workers,
            } => Command::Train {
                command: Some(TrainCommand::Plan {
                    model_bytes,
                    workers,
                    mode: TrainMode::LocalSgd,
                    strategy: "local_sgd".to_string(),
                    tensor_degree: None,
                    pipeline_stages: None,
                    data_locality: Some(data_locality),
                }),
                args: TrainArgs::default(),
            },
            Self::PlanTrainingV4 {
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            }
            | Self::PlanFrontierTraining {
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            } => Command::Train {
                command: Some(TrainCommand::Plan {
                    model_bytes,
                    workers,
                    mode: TrainMode::Sharded,
                    strategy,
                    tensor_degree,
                    pipeline_stages,
                    data_locality: None,
                }),
                args: TrainArgs::default(),
            },
            Self::ReplanTrainingV4 {
                job_id,
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            } => Command::Train {
                command: Some(TrainCommand::Replan {
                    job_id,
                    model_bytes,
                    workers,
                    strategy,
                    tensor_degree,
                    pipeline_stages,
                }),
                args: TrainArgs::default(),
            },
            Self::ActivateTrainingV4 { job_id } => Command::Train {
                command: Some(TrainCommand::Activate { job_id }),
                args: TrainArgs::default(),
            },
            Self::TrainReference {
                workers,
                steps,
                resume_checkpoint,
            } => Command::Train {
                command: Some(TrainCommand::Reference {
                    workers,
                    steps,
                    resume_checkpoint,
                }),
                args: TrainArgs::default(),
            },
            Self::TrainV3 {
                workers,
                windows,
                local_steps,
                checkpoint_every,
            } => Command::Train {
                command: None,
                args: TrainArgs {
                    mode: TrainMode::LocalSgd,
                    workers,
                    windows,
                    checkpoint_every,
                    local_steps: Some(local_steps),
                },
            },
            Self::TrainV3Start {
                workers,
                windows,
                local_steps,
                checkpoint_every,
            } => Command::Train {
                command: Some(TrainCommand::Start {
                    args: TrainArgs {
                        mode: TrainMode::LocalSgd,
                        workers,
                        windows,
                        checkpoint_every,
                        local_steps: Some(local_steps),
                    },
                }),
                args: TrainArgs::default(),
            },
            Self::TrainV4 {
                workers,
                windows,
                checkpoint_every,
            }
            | Self::TrainFrontier {
                workers,
                windows,
                checkpoint_every,
            } => Command::Train {
                command: None,
                args: TrainArgs {
                    mode: TrainMode::Sharded,
                    workers,
                    windows,
                    checkpoint_every,
                    local_steps: None,
                },
            },
            Self::TrainV4Start {
                workers,
                windows,
                checkpoint_every,
            }
            | Self::TrainFrontierStart {
                workers,
                windows,
                checkpoint_every,
            } => Command::Train {
                command: Some(TrainCommand::Start {
                    args: TrainArgs {
                        mode: TrainMode::Sharded,
                        workers,
                        windows,
                        checkpoint_every,
                        local_steps: None,
                    },
                }),
                args: TrainArgs::default(),
            },
            Self::TrainingStatus { job_id } => Command::Train {
                command: Some(TrainCommand::Status { job_id }),
                args: TrainArgs::default(),
            },
            Self::V4SeedShard {
                job_id,
                shard_id,
                state,
            } => Command::Dev {
                command: DevCommand::SeedShard {
                    job_id,
                    shard_id,
                    state,
                },
            },
            Self::V4MigrateShard {
                job_id,
                shard_id,
                target,
            } => Command::Train {
                command: Some(TrainCommand::Migrate {
                    job_id,
                    shard_id,
                    target,
                }),
                args: TrainArgs::default(),
            },
            Self::V4ReplicateState { workers } => Command::Train {
                command: Some(TrainCommand::Replicate { workers }),
                args: TrainArgs::default(),
            },
            Self::V4TensorDemo { workers } => Command::Dev {
                command: DevCommand::TensorDemo { workers },
            },
            Self::V4PipelineDemo {
                stages,
                microbatches,
            } => Command::Dev {
                command: DevCommand::PipelineDemo {
                    stages,
                    microbatches,
                },
            },
            Self::V4CollectiveDemo { workers } => Command::Dev {
                command: DevCommand::CollectiveDemo { workers },
            },
            Self::V4ReconcileDemo {
                worker,
                left_value,
                right_value,
                policy,
            } => Command::Train {
                command: Some(TrainCommand::Reconcile {
                    worker,
                    left_value,
                    right_value,
                    policy,
                }),
                args: TrainArgs::default(),
            },
            Self::V4ByzantineDemo {
                workers,
                malicious,
                policy,
            } => Command::Dev {
                command: DevCommand::ByzantineDemo {
                    workers,
                    malicious,
                    policy,
                },
            },
        }
    }
}
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut cli = Cli::parse();
    cli.config = resolve_config_path(cli.config)?;
    let Cli {
        config,
        json,
        command,
    } = cli;
    let command = match command {
        Commands::Current(command) => command,
        Commands::Legacy(command) => command.into_current(),
    };
    execute(&config, json, command).await
}

fn train_request(start: bool, args: TrainArgs) -> Result<AdminRequest, Box<dyn std::error::Error>> {
    match args.mode {
        TrainMode::Sharded => {
            if args.local_steps.is_some() {
                return Err("--local-steps is only valid with --mode local-sgd".into());
            }
            let workers = args.workers;
            let windows = args.windows;
            let checkpoint_every = Some(args.checkpoint_every);
            Ok(if start {
                AdminRequest::TrainV4Start {
                    workers,
                    windows,
                    checkpoint_every,
                }
            } else {
                AdminRequest::TrainV4 {
                    workers,
                    windows,
                    checkpoint_every,
                }
            })
        }
        TrainMode::LocalSgd => {
            let workers = args.workers;
            let windows = args.windows;
            let local_steps = Some(args.local_steps.unwrap_or(3));
            let checkpoint_every = Some(args.checkpoint_every);
            Ok(if start {
                AdminRequest::TrainV3Start {
                    workers,
                    windows,
                    local_steps,
                    checkpoint_every,
                }
            } else {
                AdminRequest::TrainV3 {
                    workers,
                    windows,
                    local_steps,
                    checkpoint_every,
                }
            })
        }
    }
}

async fn execute(
    config: &PathBuf,
    json: bool,
    command: Command,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Command::Run => run_node(config).await?,
        Command::Up {
            peers,
            no_default_seeds,
            capabilities,
            listen_addr,
        } => {
            up_node(
                config,
                peers,
                no_default_seeds,
                capabilities,
                listen_addr,
                json,
            )
            .await?
        }
        Command::Down => down_node(config, json).await?,
        Command::Init => init_config(config)?,
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
            json,
        )?,
        Command::Identity { command } => match command {
            None => {
                let config = load_config(config)?;
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
                    json,
                )?;
            }
            Some(IdentityCommand::Rotate {
                new_path,
                sequence,
                valid_until,
            }) => {
                let config = load_config(config)?;
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
                print_json(serde_json::to_value(rotation)?, json)?;
            }
        },
        Command::Status => status_command(config, json).await?,
        Command::Doctor => doctor_command(config, json).await?,
        Command::Service { command } => match command {
            ServiceCommand::Install => service_install(config, json)?,
        },
        Command::Jobs { command } => match command {
            None => call_and_print(config, AdminRequest::Jobs, json).await?,
            Some(JobsCommand::Cancel { job_id }) => {
                call_and_print(config, AdminRequest::Cancel { job_id }, json).await?
            }
        },
        Command::Config => call_and_print(config, AdminRequest::Config, json).await?,
        Command::Network { command } => match command {
            NetworkCommand::Peers => call_and_print(config, AdminRequest::Peers, json).await?,
            NetworkCommand::Capabilities => {
                call_and_print(config, AdminRequest::Capabilities, json).await?
            }
            NetworkCommand::Stats => call_and_print(config, AdminRequest::DhtStats, json).await?,
            NetworkCommand::Publish {
                namespace,
                name,
                value,
                ttl_seconds,
                sequence,
            } => {
                call_and_print(
                    config,
                    AdminRequest::DhtPublish {
                        namespace,
                        name,
                        value,
                        ttl_seconds,
                        sequence,
                    },
                    json,
                )
                .await?
            }
            NetworkCommand::Lookup { namespace, name } => {
                call_and_print(config, AdminRequest::DhtLookup { namespace, name }, json).await?
            }
            NetworkCommand::Find { key } => {
                call_and_print(config, AdminRequest::DhtFindNode { key }, json).await?
            }
        },
        Command::Trust { command, subject } => {
            let subject = match command {
                Some(TrustCommand::Inspect { subject }) => subject,
                None => subject.ok_or(
                    "a peer subject is required: intelligence trust inspect --subject <node-id>",
                )?,
            };
            call_and_print(config, AdminRequest::Trust { subject }, json).await?
        }
        Command::Infer {
            text,
            capability,
            input,
            deadline_ms,
            max_output_bytes,
            local_only,
            job_id,
        } => {
            let positional_input = text.is_some();
            let input = match (text, input) {
                (Some(text), None) | (None, Some(text)) => text,
                (Some(_), Some(_)) => {
                    return Err(
                        "provide inference text either positionally or with --input, not both"
                            .into(),
                    );
                }
                (None, None) => {
                    return Err("inference text is required: intelligence infer \"hello\"".into());
                }
            };
            infer_command(
                config,
                AdminRequest::Infer {
                    capability,
                    input,
                    deadline_ms,
                    max_output_bytes: max_output_bytes.or(Some(16 * 1024)),
                    allow_input_transfer: Some(!local_only),
                    job_id,
                },
                json,
                positional_input,
            )
            .await?
        }
        Command::Model { command } => match command {
            ModelCommand::Add {
                path,
                identity,
                format,
            } => {
                let format = format.unwrap_or_else(|| model_format(&path));
                model_add(config, path, identity, format, json).await?
            }
            ModelCommand::List => model_scan(config, json)?,
            ModelCommand::Register {
                path,
                identity,
                format,
                local_only,
            } => {
                call_and_print(
                    config,
                    AdminRequest::RegisterModel {
                        path,
                        identity,
                        format,
                        local_only,
                    },
                    json,
                )
                .await?
            }
        },
        Command::Artifact { command } => match command {
            ArtifactCommand::Inspect { artifact } => {
                call_and_print(config, AdminRequest::Inspect { artifact }, json).await?
            }
            ArtifactCommand::Fetch { peer, artifact } => {
                call_and_print(config, AdminRequest::FetchArtifact { peer, artifact }, json).await?
            }
        },
        Command::Evaluate {
            text,
            expected_label,
            deadline_ms,
        } => {
            call_and_print(
                config,
                AdminRequest::Evaluate {
                    text,
                    expected_label,
                    deadline_ms,
                },
                json,
            )
            .await?
        }
        Command::Train { command, args } => match command {
            None => {
                let request = train_request(false, args)?;
                call_and_print(config, request, json).await?
            }
            Some(TrainCommand::Start { args }) => {
                let request = train_request(true, args)?;
                call_and_print(config, request, json).await?
            }
            Some(TrainCommand::Plan {
                model_bytes,
                workers,
                mode,
                strategy,
                tensor_degree,
                pipeline_stages,
                data_locality,
            }) => {
                let request = match mode {
                    TrainMode::Sharded => {
                        if data_locality.is_some() {
                            return Err(
                                "--data-locality is only valid with --mode local-sgd".into()
                            );
                        }
                        AdminRequest::PlanTrainingV4 {
                            model_bytes,
                            workers,
                            strategy,
                            tensor_degree,
                            pipeline_stages,
                        }
                    }
                    TrainMode::LocalSgd => {
                        if tensor_degree.is_some() || pipeline_stages.is_some() {
                            return Err(
                                "--tensor-degree and --pipeline-stages are only valid with --mode sharded"
                                    .into(),
                            );
                        }
                        AdminRequest::PlanTraining {
                            model_bytes,
                            data_locality: data_locality.unwrap_or_else(|| "selective".to_string()),
                            workers,
                        }
                    }
                };
                call_and_print(config, request, json).await?
            }
            Some(TrainCommand::Replan {
                job_id,
                model_bytes,
                workers,
                strategy,
                tensor_degree,
                pipeline_stages,
            }) => {
                call_and_print(
                    config,
                    AdminRequest::ReplanTrainingV4 {
                        job_id,
                        model_bytes,
                        workers,
                        strategy,
                        tensor_degree,
                        pipeline_stages,
                    },
                    json,
                )
                .await?
            }
            Some(TrainCommand::Activate { job_id }) => {
                call_and_print(config, AdminRequest::ActivateTrainingV4 { job_id }, json).await?
            }
            Some(TrainCommand::Status { job_id }) => {
                call_and_print(config, AdminRequest::TrainingStatus { job_id }, json).await?
            }
            Some(TrainCommand::Cancel { job_id }) => {
                call_and_print(config, AdminRequest::Cancel { job_id }, json).await?
            }
            Some(TrainCommand::Migrate {
                job_id,
                shard_id,
                target,
            }) => {
                call_and_print(
                    config,
                    AdminRequest::V4MigrateShard {
                        job_id,
                        shard_id,
                        target,
                    },
                    json,
                )
                .await?
            }
            Some(TrainCommand::Replicate { workers }) => {
                call_and_print(config, AdminRequest::V4ReplicateState { workers }, json).await?
            }
            Some(TrainCommand::Reconcile {
                worker,
                left_value,
                right_value,
                policy,
            }) => {
                call_and_print(
                    config,
                    AdminRequest::V4ReconcileDemo {
                        worker,
                        left_value,
                        right_value,
                        policy,
                    },
                    json,
                )
                .await?
            }
            Some(TrainCommand::Reference {
                workers,
                steps,
                resume_checkpoint,
            }) => {
                call_and_print(
                    config,
                    AdminRequest::TrainReference {
                        workers,
                        steps,
                        resume_checkpoint,
                    },
                    json,
                )
                .await?
            }
        },
        Command::Dev { command } => match command {
            DevCommand::SeedShard {
                job_id,
                shard_id,
                state,
            } => {
                call_and_print(
                    config,
                    AdminRequest::V4SeedShard {
                        job_id,
                        shard_id,
                        state,
                    },
                    json,
                )
                .await?
            }
            DevCommand::TensorDemo { workers } => {
                call_and_print(config, AdminRequest::V4TensorDemo { workers }, json).await?
            }
            DevCommand::PipelineDemo {
                stages,
                microbatches,
            } => {
                call_and_print(
                    config,
                    AdminRequest::V4PipelineDemo {
                        stages,
                        microbatches,
                    },
                    json,
                )
                .await?
            }
            DevCommand::CollectiveDemo { workers } => {
                call_and_print(config, AdminRequest::V4CollectiveDemo { workers }, json).await?
            }
            DevCommand::ByzantineDemo {
                workers,
                malicious,
                policy,
            } => {
                call_and_print(
                    config,
                    AdminRequest::V4ByzantineDemo {
                        workers,
                        malicious,
                        policy,
                    },
                    json,
                )
                .await?
            }
            DevCommand::Shutdown => call_and_print(config, AdminRequest::Shutdown, json).await?,
        },
    }
    Ok(())
}
const DEFAULT_BOOTSTRAP_SEEDS: &[&str] = &[
    "bootstrap-1.intelligence.network:4000",
    "bootstrap-2.intelligence.network:4000",
];
const MAX_SEED_ADDRESSES: usize = 16;
const MAX_MODEL_SCAN_ENTRIES: usize = 512;

struct UserPaths {
    config_file: PathBuf,
    state_dir: PathBuf,
    models_dir: PathBuf,
}

fn user_paths() -> Result<UserPaths, Box<dyn std::error::Error>> {
    let home = env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or("cannot determine the user home directory; set HOME or USERPROFILE")?;
    let config_root = if cfg!(windows) {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Roaming"))
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".config"))
    };
    let state_root = if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Local"))
            .join("state")
    } else {
        env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local").join("state"))
    };
    let data_root = if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Local"))
    } else {
        env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".local").join("share"))
    };
    Ok(UserPaths {
        config_file: config_root.join("intelligence").join("config.toml"),
        state_dir: state_root.join("intelligence"),
        models_dir: data_root.join("intelligence").join("models"),
    })
}

fn resolve_config_path(path: PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if path != Path::new("intelligence.toml")
        || env::var_os("INTELLIGENCE_CONFIG").is_some()
        || path.exists()
    {
        return Ok(path);
    }
    Ok(user_paths()?.config_file)
}

fn is_standard_config(path: &Path) -> bool {
    user_paths()
        .map(|paths| paths.config_file == path)
        .unwrap_or(false)
}

fn standard_state_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(user_paths()?.state_dir)
}

fn management_config(path: &Path) -> Result<NodeConfig, Box<dyn std::error::Error>> {
    if path.exists() {
        return Ok(NodeConfig::load(path)?);
    }
    let mut config = NodeConfig::from_environment()?;
    if is_standard_config(path) && env::var_os("INTELLIGENCE_DATA_DIR").is_none() {
        config.data_dir = standard_state_dir()?;
        if env::var_os("INTELLIGENCE_IDENTITY_PATH").is_none() {
            config.identity_path = None;
        }
        if env::var_os("INTELLIGENCE_ADMIN_SOCKET").is_none() {
            config.admin_socket = None;
        }
        config.runtime.work_dir = PathBuf::from("runtime-work");
        config.normalize();
    }
    Ok(config)
}

fn load_config(path: &Path) -> Result<NodeConfig, Box<dyn std::error::Error>> {
    management_config(path)
}

async fn run_node(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let config = load_config(path)?;
    let pid_path = config.data_dir.join("node.pid");
    let node = Node::start(config).await?;
    write_pid_file(&pid_path, std::process::id())?;
    tokio::select! {
        _ = node.wait_for_shutdown() => {},
        signal_result = signal::ctrl_c() => {
            signal_result?;
            node.shutdown().await;
        }
    }
    remove_pid_if_owned(&pid_path, std::process::id());
    Ok(())
}

async fn call_and_print(
    path: &Path,
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

async fn status_command(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(path)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    let value = admin_call(socket, &AdminRequest::Status).await?;
    if json {
        print_json(value, true)?;
    } else {
        print_status_summary(&value);
    }
    Ok(())
}

async fn infer_command(
    path: &Path,
    request: AdminRequest,
    json: bool,
    human: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(path)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    let value = admin_call(socket, &request).await?;
    if json || !human {
        print_json(value, true)?;
        return Ok(());
    }
    println!("state: {}", value["state"].as_str().unwrap_or("unknown"));
    if let Some(error) = value["error"].as_str() {
        println!("error: {error}");
    }
    if let Some(output) = value["output"].as_array() {
        let bytes = output
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .filter_map(|byte| u8::try_from(byte).ok())
            .collect::<Vec<_>>();
        if let Ok(decoded) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            println!("output: {}", serde_json::to_string_pretty(&decoded)?);
        } else if let Ok(decoded) = String::from_utf8(bytes) {
            println!("output: {decoded}");
        }
    }
    Ok(())
}

async fn up_node(
    path: &Path,
    peers: Vec<String>,
    no_default_seeds: bool,
    capabilities: Vec<String>,
    listen_addr: Option<String>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (config, warnings) =
        prepare_up_config(path, peers, no_default_seeds, capabilities, listen_addr)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    if let Ok(value) = admin_call(socket.clone(), &AdminRequest::Status).await {
        if json {
            print_json(value, true)?;
        } else {
            println!("Intelligence Network is already running");
            print_status_summary(&value);
        }
        return Ok(());
    }

    fs::create_dir_all(&config.data_dir)?;
    let log_path = config.data_dir.join("node.log");
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let stderr = stdout.try_clone()?;
    let executable = env::current_exe()?;
    let mut command = ProcessCommand::new(executable);
    command
        .arg("--config")
        .arg(path)
        .arg("run")
        .env("INTELLIGENCE_DAEMON", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP so the node survives the
        // console window closing.
        command.creation_flags(0x00000008 | 0x00000200);
    }
    let mut child = command.spawn()?;

    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if let Ok(value) = tokio::time::timeout(
            Duration::from_millis(300),
            admin_call(socket.clone(), &AdminRequest::Status),
        )
        .await
        .unwrap_or_else(|_| {
            Err(intelligence_node::NodeError::InvalidConfig(
                "status probe timed out".to_string(),
            ))
        }) {
            if json {
                print_json(
                    serde_json::json!({
                        "config": path,
                        "state_dir": config.data_dir,
                    "models_dir": user_paths().ok().map(|paths| paths.models_dir),
                        "host": host_snapshot(),
                        "warnings": warnings,
                        "status": value,
                    }),
                    true,
                )?;
            } else {
                for warning in &warnings {
                    eprintln!("warning: {warning}");
                }
                println!("Intelligence Network is up");
                println!("config: {}", path.display());
                println!("state:  {}", config.data_dir.display());
                println!(
                    "models: {} (local files only)",
                    discover_models(path, &config).len()
                );
                print_host_summary(&host_snapshot());
                print_status_summary(&value);
                println!("Try: intelligence status  |  intelligence infer \"hello\"");
            }
            return Ok(());
        }
        if let Some(status) = child.try_wait()? {
            let log_tail = read_log_tail(&log_path);
            return Err(format!(
                "node exited during startup with {status}; inspect {}{}",
                log_path.display(),
                if log_tail.is_empty() {
                    String::new()
                } else {
                    format!("\n{log_tail}")
                }
            )
            .into());
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "node did not become healthy within 20 seconds; inspect {}",
                log_path.display()
            )
            .into());
        }
        sleep(Duration::from_millis(100)).await;
    }
}

async fn down_node(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(path)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    let pid_path = config.data_dir.join("node.pid");
    if let Ok(Ok(value)) = tokio::time::timeout(
        Duration::from_secs(10),
        admin_call(socket.clone(), &AdminRequest::Shutdown),
    )
    .await
    {
        let deadline = Instant::now() + Duration::from_secs(5);
        // The node may bind a /tmp fallback when the configured socket path
        // exceeds the unix length limit; watch the resolved path.
        #[cfg(unix)]
        let socket_file = intelligence_node::admin_socket_path(&socket);
        #[cfg(unix)]
        while socket_file.exists() && Instant::now() < deadline {
            sleep(Duration::from_millis(100)).await;
        }
        // Named pipes leave no filesystem entry; poll the admin channel until
        // the node stops answering instead.
        #[cfg(windows)]
        while Instant::now() < deadline {
            if admin_call(socket.clone(), &AdminRequest::Status)
                .await
                .is_err()
            {
                break;
            }
            sleep(Duration::from_millis(100)).await;
        }
        remove_pid_if_owned(&pid_path, read_pid(&pid_path).unwrap_or(0));
        if json {
            print_json(value, true)?;
        } else {
            println!("Intelligence Network stopped");
        }
        return Ok(());
    }

    if let Some(pid) = read_pid(&pid_path)
        && pid != std::process::id()
        && pid_is_node(pid)
    {
        terminate_pid(pid)?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while pid_is_alive(pid) && Instant::now() < deadline {
            sleep(Duration::from_millis(100)).await;
        }
        if pid_is_alive(pid) {
            return Err(format!("node process {pid} did not stop after SIGTERM").into());
        }
        remove_pid_if_owned(&pid_path, pid);
        if json {
            print_json(serde_json::json!({"stopped": true, "pid": pid}), true)?;
        } else {
            println!("Intelligence Network stopped");
        }
    } else if json {
        print_json(
            serde_json::json!({"stopped": false, "running": false}),
            true,
        )?;
    } else {
        println!("Intelligence Network is not running");
    }
    Ok(())
}

fn prepare_up_config(
    path: &Path,
    peers: Vec<String>,
    no_default_seeds: bool,
    capabilities: Vec<String>,
    listen_addr: Option<String>,
) -> Result<(NodeConfig, Vec<String>), Box<dyn std::error::Error>> {
    let existed = path.exists();
    let standard = is_standard_config(path);
    let manual_listen = listen_addr.is_some();
    let mut changed = !existed;
    let mut warnings = Vec::new();
    let mut config = if existed {
        NodeConfig::load(path)?
    } else {
        let mut config = NodeConfig::default();
        if config.data_dir == Path::new("state") {
            config.data_dir = if standard {
                standard_state_dir()?
            } else {
                path.parent()
                    .filter(|parent| !parent.as_os_str().is_empty())
                    .unwrap_or_else(|| Path::new("."))
                    .join("state")
            };
        }
        config
    };

    if let Some(listen_addr) = listen_addr {
        config.listen_addr = listen_addr
            .parse::<SocketAddr>()
            .map_err(|error| format!("invalid --listen-addr: {error}"))?;
        config.advertise_addr = None;
        changed = true;
    }

    if !existed
        && !manual_listen
        && config.listen_addr == SocketAddr::from(([127, 0, 0, 1], 4000))
        && UdpSocket::bind(config.listen_addr).is_err()
    {
        let socket = UdpSocket::bind(("127.0.0.1", 0))?;
        config.listen_addr = socket.local_addr()?;
        config.advertise_addr = None;
        warnings.push(format!(
            "127.0.0.1:4000 was busy; selected {} for this local node",
            config.listen_addr
        ));
        changed = true;
    }

    if !peers.is_empty() {
        config.bootstrap.extend(resolve_peer_hints(&peers)?);
        changed = true;
    }

    if !no_default_seeds && config.bootstrap.is_empty() {
        let (seeds, failures) = resolve_peer_hints_with_failures(&default_seed_hints());
        config.bootstrap.extend(seeds);
        if !config.bootstrap.is_empty() {
            changed = true;
        }
        warnings.extend(failures.into_iter().map(|failure| {
            format!("default bootstrap seed {failure} is unavailable; the node can run without it")
        }));
    }

    if config.capabilities.is_empty() {
        config.capabilities = if capabilities.is_empty() {
            safe_default_capabilities()
        } else {
            capabilities_from_names(&capabilities)?
        };
        changed = true;
    } else if !capabilities.is_empty() {
        let existing = config
            .capabilities
            .iter()
            .map(|capability| capability.name.clone())
            .collect::<BTreeSet<_>>();
        for capability in capabilities_from_names(&capabilities)? {
            if !existing.contains(capability.name.as_str()) {
                config.capabilities.push(capability);
                changed = true;
            }
        }
    }

    config.normalize();
    config.validate()?;
    fs::create_dir_all(&config.data_dir)?;
    if let Ok(paths) = user_paths() {
        fs::create_dir_all(paths.models_dir)?;
    }
    if changed {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, config.to_toml()?)?;
    }
    Ok((config, warnings))
}

fn default_seed_hints() -> Vec<String> {
    if let Ok(value) = env::var("INTELLIGENCE_DEFAULT_SEEDS") {
        return value
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect();
    }
    DEFAULT_BOOTSTRAP_SEEDS
        .iter()
        .map(|seed| (*seed).to_string())
        .collect()
}

fn resolve_peer_hints(hints: &[String]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let (addresses, failures) = resolve_peer_hints_with_failures(hints);
    if let Some(failure) = failures.first() {
        return Err(format!("cannot resolve bootstrap peer {failure}").into());
    }
    Ok(addresses)
}

fn resolve_peer_hints_with_failures(hints: &[String]) -> (Vec<String>, Vec<String>) {
    let mut addresses = Vec::new();
    let mut failures = Vec::new();
    for hint in hints {
        if hint.parse::<SocketAddr>().is_ok() {
            addresses.push(hint.clone());
            continue;
        }
        match hint.to_socket_addrs() {
            Ok(resolved) => {
                for address in resolved.take(MAX_SEED_ADDRESSES.saturating_sub(addresses.len())) {
                    addresses.push(address.to_string());
                }
            }
            Err(_) => failures.push(hint.clone()),
        }
        if addresses.len() >= MAX_SEED_ADDRESSES {
            break;
        }
    }
    addresses.sort();
    addresses.dedup();
    (addresses, failures)
}

fn capabilities_from_names(
    names: &[String],
) -> Result<Vec<intelligence_node::CapabilityConfig>, Box<dyn std::error::Error>> {
    let mut result = Vec::new();
    let mut seen = BTreeSet::new();
    for name in names {
        let name = name.trim();
        if name.is_empty() || !seen.insert(name.to_string()) {
            continue;
        }
        let capability = safe_capability(name).ok_or_else(|| {
            format!(
                "unsupported safe capability {name:?}; choose inference.text, evaluation.text, or training.reference"
            )
        })?;
        result.push(capability);
    }
    if result.is_empty() {
        return Err("at least one capability name is required".into());
    }
    Ok(result)
}

fn safe_default_capabilities() -> Vec<intelligence_node::CapabilityConfig> {
    ["inference.text", "evaluation.text", "training.reference"]
        .iter()
        .filter_map(|name| safe_capability(name))
        .collect()
}

fn safe_capability(name: &str) -> Option<intelligence_node::CapabilityConfig> {
    let mut capability = intelligence_node::CapabilityConfig {
        name: name.to_string(),
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
    match name {
        "inference.text" | "evaluation.text" => Some(capability),
        "training.reference" => {
            capability.kind = "builtin_training".to_string();
            capability.model = None;
            Some(capability)
        }
        _ => None,
    }
}

async fn model_add(
    path: &Path,
    model_path: PathBuf,
    identity: Option<String>,
    format: String,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !model_path.is_file() {
        return Err(format!("model path is not a regular file: {}", model_path.display()).into());
    }
    let identity = identity.unwrap_or_else(|| model_identity(&model_path));
    let config = load_config(path)?;
    let socket = config
        .admin_socket
        .clone()
        .unwrap_or_else(|| config.data_dir.join("node.sock"));
    let value = admin_call(
        socket,
        &AdminRequest::RegisterModel {
            path: model_path.to_string_lossy().into_owned(),
            identity,
            format,
            local_only: true,
        },
    )
    .await
    .map_err(|error| format!("model import needs a running node: {error}"))?;
    print_json(value, json)?;
    Ok(())
}

fn model_identity(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("model");
    let mut identity = String::from("local.");
    for character in stem.chars().take(80) {
        identity.push(
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '-'
            },
        );
    }
    identity
}

fn model_scan(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let config = load_config(path)?;
    let entries = discover_models(path, &config);
    let value = serde_json::Value::Array(entries.clone());
    if json {
        print_json(value, true)?;
    } else if entries.is_empty() {
        println!("No local model files found. Models are never downloaded automatically.");
    } else {
        println!("Local models ({}):", entries.len());
        for entry in entries {
            println!(
                "  {}  {} bytes  {}",
                entry["path"].as_str().unwrap_or("<unknown>"),
                entry["size"].as_u64().unwrap_or(0),
                entry["format"].as_str().unwrap_or("unknown")
            );
        }
    }
    Ok(())
}

fn model_scan_dirs(path: &Path, config: &NodeConfig) -> Vec<PathBuf> {
    let mut dirs = BTreeSet::new();
    if let Ok(paths) = user_paths() {
        dirs.insert(paths.models_dir);
    }
    dirs.insert(config.data_dir.join("models"));
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        dirs.insert(parent.join("models"));
    }
    dirs.into_iter().collect()
}

fn scan_model_files(directories: &[PathBuf]) -> Vec<serde_json::Value> {
    let mut result = Vec::new();
    for directory in directories {
        scan_model_directory(directory, 0, &mut result);
        if result.len() >= MAX_MODEL_SCAN_ENTRIES {
            break;
        }
    }
    result
}

fn discover_models(path: &Path, config: &NodeConfig) -> Vec<serde_json::Value> {
    let mut result = scan_model_files(&model_scan_dirs(path, config));
    let state_dir = config.data_dir.join("state");
    if let Ok(entries) = fs::read_dir(state_dir) {
        for entry in entries.flatten() {
            if result.len() >= MAX_MODEL_SCAN_ENTRIES {
                break;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !name.starts_with("model-") || !name.ends_with(".json") {
                continue;
            }
            let Ok(contents) = fs::read_to_string(entry.path()) else {
                continue;
            };
            let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&contents) else {
                continue;
            };
            result.push(serde_json::json!({
                "path": manifest["local_path"].as_str().unwrap_or("<content-addressed artifact>"),
                "size": manifest["size"].as_u64().unwrap_or(0),
                "format": manifest["format"].as_str().unwrap_or("opaque"),
                "identity": manifest["identity"],
                "artifact": manifest["artifact"],
                "stored": true,
            }));
        }
    }
    result
}

fn scan_model_directory(directory: &Path, depth: usize, result: &mut Vec<serde_json::Value>) {
    if depth > 2 || result.len() >= MAX_MODEL_SCAN_ENTRIES {
        return;
    }
    let Ok(entries) = fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        if result.len() >= MAX_MODEL_SCAN_ENTRIES {
            return;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            scan_model_directory(&path, depth + 1, result);
        } else if file_type.is_file()
            && let Ok(metadata) = entry.metadata()
            && is_model_file(&path, &metadata)
        {
            result.push(serde_json::json!({
                "path": path,
                "size": metadata.len(),
                "format": model_format(&path),
            }));
        }
    }
}

fn is_model_file(path: &Path, metadata: &fs::Metadata) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    // Skip hidden files and the JSON manifests the node writes alongside
    // registered artifacts; every other regular file counts as a model file.
    !name.starts_with('.') && !name.ends_with(".json") && metadata.len() >= 1
}

fn model_format(path: &Path) -> String {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| extension.to_ascii_lowercase())
        .unwrap_or_else(|| "opaque".to_string())
}

#[derive(Clone, Debug)]
struct DoctorCheck {
    name: &'static str,
    status: &'static str,
    detail: String,
    fix: Option<String>,
}

async fn doctor_command(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let mut checks = Vec::new();
    let config = match load_config(path) {
        Ok(config) => {
            checks.push(DoctorCheck {
                name: "config",
                status: if path.exists() { "pass" } else { "warn" },
                detail: if path.exists() {
                    format!("valid configuration at {}", path.display())
                } else {
                    format!("{} will be created by intelligence up", path.display())
                },
                fix: (!path.exists()).then(|| "run intelligence up".to_string()),
            });
            config
        }
        Err(error) => {
            checks.push(DoctorCheck {
                name: "config",
                status: "fail",
                detail: error.to_string(),
                fix: Some("repair the configuration or use --config with a valid file".to_string()),
            });
            NodeConfig::default()
        }
    };

    if config.data_dir.is_dir() {
        match writable_directory(&config.data_dir) {
            Ok(()) => checks.push(DoctorCheck {
                name: "storage",
                status: "pass",
                detail: format!("state directory is writable: {}", config.data_dir.display()),
                fix: None,
            }),
            Err(error) => checks.push(DoctorCheck {
                name: "storage",
                status: "fail",
                detail: error.to_string(),
                fix: Some(
                    "choose a writable state directory with INTELLIGENCE_DATA_DIR".to_string(),
                ),
            }),
        }
    } else {
        checks.push(DoctorCheck {
            name: "storage",
            status: "warn",
            detail: format!(
                "state directory does not exist yet: {}",
                config.data_dir.display()
            ),
            fix: Some("run intelligence up".to_string()),
        });
    }

    let identity_path = config
        .identity_path
        .clone()
        .unwrap_or_else(|| config.data_dir.join("identity.key"));
    if identity_path.is_file() {
        match Identity::load_or_generate(&identity_path) {
            Ok(identity) => checks.push(DoctorCheck {
                name: "identity",
                status: "pass",
                detail: format!("persistent node identity {}", identity.node_id()),
                fix: None,
            }),
            Err(error) => checks.push(DoctorCheck {
                name: "identity",
                status: "fail",
                detail: error.to_string(),
                fix: Some(
                    "restore the identity backup or choose a new state directory".to_string(),
                ),
            }),
        }
    } else {
        checks.push(DoctorCheck {
            name: "identity",
            status: "warn",
            detail: "no identity exists yet; up will create one without replacing future state"
                .to_string(),
            fix: Some("run intelligence up".to_string()),
        });
    }

    let node_status = if config.admin_socket.is_some() {
        let socket = config
            .admin_socket
            .clone()
            .unwrap_or_else(|| config.data_dir.join("node.sock"));
        tokio::time::timeout(
            Duration::from_secs(1),
            admin_call(socket, &AdminRequest::Status),
        )
        .await
        .ok()
        .and_then(Result::ok)
    } else {
        None
    };
    if let Some(status) = &node_status {
        checks.push(DoctorCheck {
            name: "node",
            status: "pass",
            detail: format!(
                "healthy node {} with {} connected peer(s)",
                status["node_id"].as_str().unwrap_or("unknown"),
                status["connected_peers"].as_u64().unwrap_or(0)
            ),
            fix: None,
        });
        checks.push(DoctorCheck {
            name: "quic",
            status: "pass",
            detail: format!(
                "authenticated protocol {} is responding",
                status["protocol"].as_str().unwrap_or("1.6")
            ),
            fix: None,
        });
    } else {
        checks.push(DoctorCheck {
            name: "node",
            status: "warn",
            detail: "node is not running; runtime health will be checked after up".to_string(),
            fix: Some("run intelligence up".to_string()),
        });
        checks.push(DoctorCheck {
            name: "quic",
            status: "warn",
            detail: "QUIC can be checked once the node is running".to_string(),
            fix: Some("run intelligence up, then intelligence doctor".to_string()),
        });
    }

    if node_status.is_none() {
        match UdpSocket::bind(config.listen_addr) {
            Ok(_) => checks.push(DoctorCheck {
                name: "network",
                status: "pass",
                detail: format!("{} is available for the node", config.listen_addr),
                fix: None,
            }),
            Err(error) => checks.push(DoctorCheck {
                name: "network",
                status: "fail",
                detail: format!("cannot bind {}: {error}", config.listen_addr),
                fix: Some(
                    "choose another --listen-addr or stop the process using the port".to_string(),
                ),
            }),
        }
    } else {
        checks.push(DoctorCheck {
            name: "network",
            status: "pass",
            detail: format!("node is listening at {}", config.listen_addr),
            fix: None,
        });
    }

    checks.push(
        if config.relay_enabled && !config.relay_addresses.is_empty() {
            DoctorCheck {
                name: "nat-relay",
                status: "pass",
                detail: format!(
                    "{} operator-supplied relay hint(s) configured",
                    config.relay_addresses.len()
                ),
                fix: None,
            }
        } else if config.hole_punch_enabled {
            DoctorCheck {
                name: "nat-relay",
                status: "warn",
                detail:
                    "direct discovery and bounded hole punching are enabled; no relay is mandatory"
                        .to_string(),
                fix: None,
            }
        } else {
            DoctorCheck {
                name: "nat-relay",
                status: "warn",
                detail: "no relay or hole-punch path is configured".to_string(),
                fix: Some(
                    "add operator-run relay_addresses if this node is behind restrictive NAT"
                        .to_string(),
                ),
            }
        },
    );

    let host = host_snapshot();
    checks.push(DoctorCheck {
        name: "hardware",
        status: "pass",
        detail: format!(
            "{} CPU thread(s), {} bytes host memory, GPU hints: {}",
            host["cpu_threads"].as_u64().unwrap_or(1),
            host["memory_bytes"].as_u64().unwrap_or(0),
            host["gpu_hints"].as_array().map_or(0, Vec::len)
        ),
        fix: None,
    });
    let models = discover_models(path, &config);
    checks.push(DoctorCheck {
        name: "models",
        status: "pass",
        detail: format!(
            "{} local model file(s) found; no model or large dependency will be downloaded",
            models.len()
        ),
        fix: None,
    });

    let overall = if checks.iter().any(|check| check.status == "fail") {
        "fail"
    } else if checks.iter().any(|check| check.status == "warn") {
        "warn"
    } else {
        "pass"
    };
    let report = serde_json::json!({
        "status": overall,
        "config": path,
        "checks": checks.iter().map(|check| serde_json::json!({
            "name": check.name,
            "status": check.status,
            "detail": check.detail,
            "fix": check.fix,
        })).collect::<Vec<_>>(),
        "host": host,
        "protocol": format!("{}.{}", intelligence_protocol::PROTOCOL_MAJOR, intelligence_protocol::PROTOCOL_MINOR),
    });
    if json {
        print_json(report, true)?;
    } else {
        println!("Intelligence Network doctor: {overall}");
        for check in checks {
            println!(
                "{} {:<10} {}",
                status_marker(check.status),
                check.name,
                check.detail
            );
            if let Some(fix) = check.fix {
                println!("  hint: {fix}");
            }
        }
    }
    Ok(())
}

fn writable_directory(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let probe = path.join(format!(
        ".doctor-write-{}-{}",
        std::process::id(),
        now_millis()
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&probe)?;
    file.write_all(b"ok")?;
    file.sync_all()?;
    drop(file);
    fs::remove_file(probe)?;
    Ok(())
}

fn status_marker(status: &str) -> &'static str {
    match status {
        "pass" => "PASS",
        "warn" => "WARN",
        _ => "FAIL",
    }
}

fn host_snapshot() -> serde_json::Value {
    let cpu_threads = std::thread::available_parallelism()
        .map(|value| value.get() as u64)
        .unwrap_or(1);
    let memory_bytes = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let mut fields = line.split_whitespace();
                if fields.next() == Some("MemTotal:") {
                    fields
                        .next()
                        .and_then(|value| value.parse::<u64>().ok())
                        .map(|value| value.saturating_mul(1024))
                } else {
                    None
                }
            })
        })
        .unwrap_or(0);
    let mut gpu_hints = Vec::new();
    if Path::new("/dev/nvidiactl").exists() || Path::new("/dev/nvidia0").exists() {
        gpu_hints.push("cuda-device".to_string());
    }
    if Path::new("/dev/kfd").exists() {
        gpu_hints.push("rocm-device".to_string());
    }
    #[cfg(windows)]
    if env::var_os("PATH")
        .map(|paths| env::split_paths(&paths).any(|dir| dir.join("nvidia-smi.exe").is_file()))
        .unwrap_or(false)
    {
        gpu_hints.push("cuda-device".to_string());
    }
    if cfg!(target_os = "macos") {
        gpu_hints.push("metal-platform".to_string());
    }
    serde_json::json!({
        "cpu_threads": cpu_threads,
        "memory_bytes": memory_bytes,
        "gpu_hints": gpu_hints,
        "backend_detection": "performed by the node's backend registry at startup",
        "os": std::env::consts::OS,
        "architecture": std::env::consts::ARCH,
    })
}

fn service_install(path: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        let _ = prepare_up_config(path, Vec::new(), false, Vec::new(), None)?;
    }
    let _ = NodeConfig::load(path)?;
    #[cfg(target_os = "linux")]
    {
        let home = env::var_os("HOME").ok_or("HOME is required for a user service")?;
        let unit_dir = PathBuf::from(home).join(".config/systemd/user");
        fs::create_dir_all(&unit_dir)?;
        let unit_path = unit_dir.join("intelligence.service");
        let executable = env::current_exe()?;
        let config_path = fs::canonicalize(path)?;
        let unit = format!(
            "[Unit]\nDescription=Intelligence Network user node\nAfter=network-online.target\n\n[Service]\nExecStart={} --config {} run\nRestart=on-failure\nRestartSec=2\n\n[Install]\nWantedBy=default.target\n",
            systemd_quote(&executable),
            systemd_quote(&config_path),
        );
        fs::write(&unit_path, unit)?;
        let reload = ProcessCommand::new("systemctl")
            .args(["--user", "daemon-reload"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let enable = ProcessCommand::new("systemctl")
            .args(["--user", "enable", "--now", "intelligence.service"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let active = reload.as_ref().is_ok_and(|status| status.success())
            && enable.as_ref().is_ok_and(|status| status.success());
        let value = serde_json::json!({
            "unit": unit_path,
            "enabled": active,
            "scope": "per-user",
            "root_required": false,
        });
        if json {
            print_json(value, true)?;
        } else if active {
            println!(
                "Installed and started per-user service: {}",
                unit_path.display()
            );
        } else {
            println!("Wrote per-user service: {}", unit_path.display());
            println!(
                "systemd --user was unavailable or refused activation; run systemctl --user enable --now intelligence.service when ready"
            );
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        let home = env::var_os("HOME").ok_or("HOME is required for a user agent")?;
        let agent_dir = PathBuf::from(&home).join("Library/LaunchAgents");
        fs::create_dir_all(&agent_dir)?;
        let plist_path = agent_dir.join("network.intelligence.node.plist");
        let executable = env::current_exe()?;
        let config_path = fs::canonicalize(path)?;
        let config = NodeConfig::load(path)?;
        let log_path = config.data_dir.join("node.log");
        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let plist = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n<plist version=\"1.0\">\n<dict>\n    <key>Label</key>\n    <string>network.intelligence.node</string>\n    <key>ProgramArguments</key>\n    <array>\n        <string>{}</string>\n        <string>--config</string>\n        <string>{}</string>\n        <string>run</string>\n    </array>\n    <key>RunAtLoad</key>\n    <true/>\n    <key>KeepAlive</key>\n    <true/>\n    <key>StandardOutPath</key>\n    <string>{}</string>\n    <key>StandardErrorPath</key>\n    <string>{}</string>\n</dict>\n</plist>\n",
            executable.display(),
            config_path.display(),
            log_path.display(),
            log_path.display(),
        );
        fs::write(&plist_path, plist)?;
        let domain = format!("gui/{}", unsafe { libc::getuid() });
        let _ = ProcessCommand::new("launchctl")
            .args(["bootout", &domain])
            .arg(&plist_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let bootstrap = ProcessCommand::new("launchctl")
            .args(["bootstrap", &domain])
            .arg(&plist_path)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let active = bootstrap.as_ref().is_ok_and(|status| status.success());
        let value = serde_json::json!({
            "plist": plist_path,
            "enabled": active,
            "scope": "per-user",
            "root_required": false,
        });
        if json {
            print_json(value, true)?;
        } else if active {
            println!(
                "Installed and started per-user agent: {}",
                plist_path.display()
            );
        } else {
            println!("Wrote per-user agent: {}", plist_path.display());
            println!(
                "launchctl bootstrap failed; run launchctl bootstrap {domain} {} when ready",
                plist_path.display()
            );
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        let executable = env::current_exe()?;
        let config_path = fs::canonicalize(path)?;
        let strip_unc = |path: &Path| {
            let display = path.to_string_lossy();
            display
                .strip_prefix(r"\\?\")
                .unwrap_or(&display)
                .to_string()
        };
        let run_line = format!(
            "\"{}\" --config \"{}\" run",
            strip_unc(&executable),
            strip_unc(&config_path)
        );
        let create = ProcessCommand::new("schtasks")
            .args([
                "/Create",
                "/F",
                "/SC",
                "ONLOGON",
                "/TN",
                "Intelligence Network",
                "/TR",
                &run_line,
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let run = ProcessCommand::new("schtasks")
            .args(["/Run", "/TN", "Intelligence Network"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let active = create.as_ref().is_ok_and(|status| status.success())
            && run.as_ref().is_ok_and(|status| status.success());
        let value = serde_json::json!({
            "task": "Intelligence Network",
            "enabled": active,
            "scope": "per-user",
            "root_required": false,
        });
        if json {
            print_json(value, true)?;
        } else if active {
            println!("Installed and started scheduled task: Intelligence Network");
        } else {
            println!("Registered scheduled task: Intelligence Network");
            println!(
                "Task Scheduler refused activation; run schtasks /Run /TN \"Intelligence Network\" when ready"
            );
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = (path, json);
        Err("service install currently supports Linux systemd, macOS launchd, or Windows Task Scheduler; intelligence up/down remain available".into())
    }
}

#[cfg(target_os = "linux")]
fn systemd_quote(path: &Path) -> String {
    format!(
        "\"{}\"",
        path.to_string_lossy()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    )
}

fn init_config(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if path.exists() {
        return Err(format!("configuration already exists: {}", path.display()).into());
    }
    let mut config = NodeConfig::default();
    if config.data_dir == Path::new("state") {
        config.data_dir = if is_standard_config(path) {
            standard_state_dir()?
        } else {
            path.parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .join("state")
        };
    }
    config.capabilities = safe_default_capabilities();
    config.normalize();
    config.validate()?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, config.to_toml()?)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn write_pid_file(path: &Path, pid: u32) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)?;
    writeln!(file, "{pid}")?;
    file.sync_all()?;
    Ok(())
}

fn read_pid(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn remove_pid_if_owned(path: &Path, pid: u32) {
    if read_pid(path) == Some(pid) {
        let _ = fs::remove_file(path);
    }
}

fn pid_is_node(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        let command_line = fs::read(format!("/proc/{pid}/cmdline")).unwrap_or_default();
        command_line.split(|byte| *byte == 0).any(|argument| {
            argument
                .windows("intelligence".len())
                .any(|window| window == b"intelligence")
        })
    }
    #[cfg(target_os = "macos")]
    {
        ProcessCommand::new("ps")
            .args(["-o", "command=", "-p", &pid.to_string()])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains("intelligence"))
    }
    #[cfg(windows)]
    {
        ProcessCommand::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains("intelligence"))
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", windows)))]
    {
        let _ = pid;
        false
    }
}

fn pid_is_alive(pid: u32) -> bool {
    #[cfg(unix)]
    {
        ProcessCommand::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }
    #[cfg(windows)]
    {
        ProcessCommand::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}")])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
    }
}

fn terminate_pid(pid: u32) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(unix)]
    {
        let status = ProcessCommand::new("kill")
            .args(["-TERM", &pid.to_string()])
            .stderr(Stdio::null())
            .status()?;
        if !status.success() && pid_is_alive(pid) {
            return Err(format!("failed to signal node process {pid}").into());
        }
        Ok(())
    }
    #[cfg(windows)]
    {
        let status = ProcessCommand::new("taskkill")
            .args(["/PID", &pid.to_string()])
            .status()?;
        if !status.success() {
            return Err(format!("failed to stop node process {pid}").into());
        }
        Ok(())
    }
}

fn read_log_tail(path: &Path) -> String {
    fs::read_to_string(path)
        .ok()
        .map(|contents| {
            contents
                .lines()
                .rev()
                .take(12)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn print_status_summary(value: &serde_json::Value) {
    println!(
        "node      {}",
        value["node_id"].as_str().unwrap_or("unknown")
    );
    println!(
        "protocol  {}",
        value["protocol"].as_str().unwrap_or("unknown")
    );
    println!(
        "listen    {}",
        value["listen_addr"].as_str().unwrap_or("unknown")
    );
    println!(
        "peers     {} connected / {} known",
        value["connected_peers"].as_u64().unwrap_or(0),
        value["known_peers"].as_u64().unwrap_or(0)
    );
    println!(
        "jobs      {} running / {} total",
        value["running_jobs"].as_u64().unwrap_or(0),
        value["job_count"].as_u64().unwrap_or(0)
    );
    println!(
        "capability {} advertised",
        value["capability_count"].as_u64().unwrap_or(0)
    );
    if let Some(backends) = value["compute_backends"].as_array() {
        let names = backends
            .iter()
            .filter_map(|backend| backend["kind"].as_str())
            .collect::<Vec<_>>();
        if !names.is_empty() {
            println!("compute   {}", names.join(", "));
        }
    }
}

fn print_host_summary(value: &serde_json::Value) {
    let gpu_hints = value["gpu_hints"]
        .as_array()
        .map(|hints| {
            hints
                .iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|hints| !hints.is_empty())
        .unwrap_or_else(|| "none detected".to_string());
    println!(
        "host      {} CPU thread(s), {} bytes RAM, GPU hints: {}",
        value["cpu_threads"].as_u64().unwrap_or(1),
        value["memory_bytes"].as_u64().unwrap_or(0),
        gpu_hints
    );
}

fn print_json(value: serde_json::Value, _json: bool) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(&value)?);
    Ok(())
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
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

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn cli_structure_is_valid() {
        Cli::command().debug_assert();
    }
}
