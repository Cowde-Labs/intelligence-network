//! Distributed training runtime.
//!
//! This module intentionally implements one small, inspectable reference
//! strategy instead of pretending to solve every distributed-training
//! algorithm.  The strategy is periodic local SGD over two parameter shards:
//! workers update only their assigned shard, group aggregators combine updates
//! inside their group, and a temporary coordinator commits bounded metadata.
//! Values and optimizer state live with shard owners and replicas; the
//! coordinator sees aggregates and hashes, not every worker update or a full
//! model. `V3*` protocol and state types remain versioned for compatibility.

use super::{Node, NodeError, now_secs};
use intelligence_intelligence::TrainingContribution;
use intelligence_protocol::{
    ArtifactId, Message, NodeId, TRAINING_SCALE, TrainingAck, TrainingAckKind, TrainingAggregate,
    TrainingCheckpointCommit, TrainingCheckpointManifest, TrainingCheckpointOffer,
    TrainingCheckpointShard, TrainingElection, TrainingExecution, TrainingGroup, TrainingMessage,
    TrainingSample, TrainingShardAssignment, TrainingShardState, TrainingShardSummary,
    TrainingStart, TrainingState, TrainingUpdate, TrainingWindow,
};
use serde_json::Value;
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, oneshot},
    time::{Instant, sleep, timeout},
};

const V3_CAPABILITY: &str = "training.reference";
const V3_MIN_WORKERS: usize = 4;
const V3_MAX_WORKERS: usize = 16;
const V3_MESSAGE_TIMEOUT: Duration = Duration::from_secs(3);
const V3_UPDATE_WINDOW_TIMEOUT: Duration = Duration::from_millis(750);
const V3_REPLICATION_TIMEOUT: Duration = Duration::from_secs(4);
const V3_ELECTION_TIMEOUT: Duration = Duration::from_secs(2);
const V3_ELECTION_ATTEMPTS: usize = 3;
const V3_ELECTION_SEND_TIMEOUT: Duration = Duration::from_millis(300);
const V3_ELECTION_RETRY_DELAY: Duration = Duration::from_millis(100);
const V3_STARTUP_COORDINATOR_RECHECKS: usize = 4;
const V3_STARTUP_COORDINATOR_RECHECK_DELAY: Duration = Duration::from_millis(500);
const V3_MAX_DELTA: i64 = 10 * TRAINING_SCALE;

fn log_election_event(message: std::fmt::Arguments<'_>) {
    tracing::debug!(message = %message, "distributed training election event");
}

#[derive(Clone)]
pub(crate) struct V3JobHandle {
    pub(crate) sender: mpsc::Sender<V3Inbound>,
}

// The transport has already bounded and decoded the frame before it enters
// this bounded per-job mailbox. Keep the message inline to avoid another
// allocation on every training event.
#[allow(clippy::large_enum_variant)]
pub(crate) enum V3Inbound {
    Message {
        peer: NodeId,
        message: TrainingMessage,
    },
    PeerDisconnected(NodeId),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Role {
    Worker,
    Coordinator,
}

struct Context {
    start: TrainingStart,
    role: Role,
    coordinator: NodeId,
    term: u64,
    window: u64,
    checkpoint_generation: u64,
    checkpoint: Option<ArtifactId>,
    final_checkpoint_committed: bool,
    assignment: TrainingShardAssignment,
    value: i64,
    optimizer: i64,
    generation: u64,
    samples: Vec<TrainingSample>,
    branch: ArtifactId,
    failed: HashSet<NodeId>,
    voted_term: u64,
    voted_candidate: Option<NodeId>,
    shard_summaries: BTreeMap<u16, TrainingShardSummary>,
    checkpoints: Vec<ArtifactId>,
    checkpoint_providers: BTreeMap<String, Vec<NodeId>>,
    accepted_updates: u64,
    rejected_updates: u64,
    stragglers: u64,
    group_progress: BTreeMap<u16, u64>,
    group_completion_order: Vec<u16>,
    non_barrier_progress_observed: bool,
    shard_values: BTreeMap<u16, (i64, i64)>,
    initial_loss: i64,
    started_at: Instant,
    result: Option<oneshot::Sender<Result<Value, String>>>,
    receiver: mpsc::Receiver<V3Inbound>,
}

pub(crate) async fn start_training(
    node: &Arc<Node>,
    requested_workers: u16,
    max_windows: u64,
    local_steps: u16,
    checkpoint_every: u64,
) -> Result<Value, NodeError> {
    start_training_with_mode(
        node,
        requested_workers,
        max_windows,
        local_steps,
        checkpoint_every,
        true,
    )
    .await
}

pub(crate) async fn start_training_background(
    node: &Arc<Node>,
    requested_workers: u16,
    max_windows: u64,
    local_steps: u16,
    checkpoint_every: u64,
) -> Result<Value, NodeError> {
    start_training_with_mode(
        node,
        requested_workers,
        max_windows,
        local_steps,
        checkpoint_every,
        false,
    )
    .await
}

async fn start_training_with_mode(
    node: &Arc<Node>,
    requested_workers: u16,
    max_windows: u64,
    local_steps: u16,
    checkpoint_every: u64,
    wait_for_result: bool,
) -> Result<Value, NodeError> {
    let requested_workers = requested_workers as usize;
    if !(V3_MIN_WORKERS..=V3_MAX_WORKERS).contains(&requested_workers)
        || !(1..=256).contains(&max_windows)
        || !(1..=64).contains(&local_steps)
        || checkpoint_every == 0
    {
        return Err(NodeError::InvalidConfig(
            "V3 training requires 4-16 workers, 1-256 windows, 1-64 local steps, and a positive checkpoint interval"
                .to_string(),
        ));
    }

    let mut workers = node
        .network
        .peer_records()
        .await
        .into_iter()
        .filter(|record| {
            record.node_id != node.node_id()
                && record.expires_at >= now_secs()
                && record.capabilities.iter().any(|capability| {
                    (capability.name == V3_CAPABILITY || capability.name == "training.v3")
                        // A signed capability is not, by itself, proof that
                        // the peer can host this model shard.  The lab uses
                        // deliberately under-capacity peers for V4 join and
                        // replacement tests; do not let ordinary V3
                        // admission select those records and then fail at
                        // the authenticated start acknowledgement.
                        && capability.resources.memory_bytes >= 64 * 1024 * 1024
                })
        })
        .map(|record| record.node_id)
        .collect::<Vec<_>>();
    workers.sort_unstable();
    workers.dedup();
    workers.truncate(requested_workers);
    if workers.len() < V3_MIN_WORKERS {
        return Err(NodeError::InvalidConfig(format!(
            "V3 training needs at least {V3_MIN_WORKERS} live training workers; discovered {}",
            workers.len()
        )));
    }

    let job_id = super::random_job_id();
    let branch = ArtifactId::from_bytes_hashed(&serde_json::to_vec(&(
        "intelligence-network/v3/branch",
        job_id,
    ))?);
    let groups = make_groups(&workers);
    let mut shards = Vec::with_capacity(groups.len());
    for group in &groups {
        let state_hash = state_hash(job_id, group.shard_id, 0, 0, 0);
        shards.push(TrainingShardAssignment {
            shard_id: group.shard_id,
            group_id: group.group_id,
            owners: group.members.clone(),
            replicas: group.members.clone(),
            generation: 0,
            state_hash,
        });
    }
    let plan_hash = ArtifactId::from_bytes_hashed(&serde_json::to_vec(&(
        "v3-local-sgd",
        &workers,
        &groups,
        &shards,
        max_windows,
        local_steps,
        checkpoint_every,
    ))?);

    let mut worker_starts = Vec::with_capacity(workers.len());
    for (index, worker) in workers.iter().copied().enumerate() {
        let assignment = shards[index % shards.len()].clone();
        let samples = samples_for_shard(assignment.shard_id);
        let start = TrainingStart {
            job_id,
            coordinator: node.node_id(),
            term: 1,
            membership_epoch: 1,
            branch,
            plan_hash,
            max_windows,
            local_steps,
            max_staleness: local_steps,
            checkpoint_every,
            execution: TrainingExecution::AutonomousLocalSgd,
            model_state_bytes: 2 * 1024,
            shard_state_bytes: 1024,
            participants: workers.clone(),
            groups: groups.clone(),
            shards: shards.clone(),
            assignment,
            initial_value: 0,
            initial_optimizer: 0,
            samples,
        };
        Message::Training(TrainingMessage::Start(start.clone()))
            .validate()
            .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
        worker_starts.push((worker, start));
    }

    let coordinator_start = worker_starts
        .first()
        .map(|(_, start)| start.clone())
        .ok_or_else(|| NodeError::InvalidConfig("no V3 workers selected".to_string()))?;
    start_coordinator(
        node.clone(),
        coordinator_start,
        worker_starts,
        wait_for_result,
    )
    .await
}

fn make_groups(workers: &[NodeId]) -> Vec<TrainingGroup> {
    (0..2)
        .filter_map(|shard_id| {
            let members = workers
                .iter()
                .enumerate()
                .filter_map(|(index, worker)| (index % 2 == shard_id).then_some(*worker))
                .collect::<Vec<_>>();
            members.first().copied().map(|aggregator| TrainingGroup {
                group_id: shard_id as u16,
                shard_id: shard_id as u16,
                members,
                aggregator,
            })
        })
        .collect()
}

fn samples_for_shard(shard_id: u16) -> Vec<TrainingSample> {
    let target = if shard_id == 0 { 2 } else { 3 };
    (1..=8)
        .map(|feature| TrainingSample {
            feature,
            target: target * feature,
        })
        .collect()
}

fn state_hash(
    job_id: intelligence_protocol::JobId,
    shard_id: u16,
    generation: u64,
    value: i64,
    optimizer: i64,
) -> ArtifactId {
    let bytes =
        serde_json::to_vec(&(job_id, shard_id, generation, value, optimizer)).unwrap_or_default();
    ArtifactId::from_bytes_hashed(&bytes)
}

fn checkpoint_hash(manifest: &TrainingCheckpointManifest) -> ArtifactId {
    let mut copy = manifest.clone();
    copy.hash = ArtifactId::default();
    ArtifactId::from_bytes_hashed(&serde_json::to_vec(&copy).unwrap_or_default())
}

fn quorum(size: usize) -> usize {
    size / 2 + 1
}

fn final_checkpoint_generation(start: &TrainingStart) -> u64 {
    start
        .max_windows
        .saturating_add(start.checkpoint_every.saturating_sub(1))
        / start.checkpoint_every
}

fn state_covers_final_model(start: &TrainingStart, state: &TrainingState) -> bool {
    state.window >= start.max_windows
        && state.checkpoint_generation >= final_checkpoint_generation(start)
        && start.shards.iter().all(|expected| {
            state
                .shards
                .iter()
                .find(|actual| actual.shard_id == expected.shard_id)
                .is_some_and(|actual| actual.generation >= start.max_windows)
        })
}

fn context_covers_final_model(context: &Context) -> bool {
    context.final_checkpoint_committed
        && context.window >= context.start.max_windows
        && context.checkpoint_generation >= final_checkpoint_generation(&context.start)
        && context.start.shards.iter().all(|expected| {
            context
                .shard_summaries
                .get(&expected.shard_id)
                .is_some_and(|actual| actual.generation >= context.start.max_windows)
        })
}

fn group_for(start: &TrainingStart, group_id: u16) -> Option<&TrainingGroup> {
    start.groups.iter().find(|group| group.group_id == group_id)
}

async fn start_coordinator(
    node: Arc<Node>,
    start: TrainingStart,
    worker_starts: Vec<(NodeId, TrainingStart)>,
    wait_for_result: bool,
) -> Result<Value, NodeError> {
    let (sender, mut receiver) = mpsc::channel(512);
    let (result_sender, result_receiver) = oneshot::channel();
    node.v3_jobs.lock().await.insert(
        start.job_id,
        V3JobHandle {
            sender: sender.clone(),
        },
    );
    persist_start(&node, &start)?;
    let expected_workers = worker_starts
        .iter()
        .map(|(worker, _)| *worker)
        .collect::<HashSet<_>>();

    // A successful send only proves that the message entered the local QUIC
    // writer queue.  It does not prove that the worker admitted the job and
    // installed its durable training state.  Admission is therefore an
    // explicit handshake.  This is deliberately a one-time job-start
    // barrier; autonomous training never uses a global per-window barrier.
    for (worker, worker_start) in &worker_starts {
        let mut sent = false;
        let mut last_error = None;
        for attempt in 0..3 {
            if attempt > 0 {
                let _ = node.network.reconnect_peer(*worker).await;
            }
            match node
                .network
                .send_to(
                    *worker,
                    Message::Training(TrainingMessage::Start(worker_start.clone())),
                )
                .await
            {
                Ok(()) => {
                    sent = true;
                    break;
                }
                Err(error) => {
                    last_error = Some(error.to_string());
                    sleep(Duration::from_millis(100)).await;
                }
            }
        }
        if !sent {
            node.v3_jobs.lock().await.remove(&start.job_id);
            return Err(NodeError::InvalidConfig(format!(
                "V3 worker {worker} did not accept a start message: {}",
                last_error.unwrap_or_else(|| "not connected".to_string())
            )));
        }
    }

    let mut acknowledged = HashSet::new();
    let admission_deadline = Instant::now() + Duration::from_secs(10);
    while acknowledged.len() < expected_workers.len() {
        let remaining = admission_deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            let missing = expected_workers
                .difference(&acknowledged)
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            node.v3_jobs.lock().await.remove(&start.job_id);
            return Err(NodeError::InvalidConfig(format!(
                "V3 workers did not acknowledge durable start: {}",
                missing.join(", ")
            )));
        }
        let Some(event) = timeout(remaining, receiver.recv()).await.ok().flatten() else {
            continue;
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Ack(ack),
            } if ack.job_id == start.job_id
                && ack.term == start.term
                && ack.peer == peer
                && expected_workers.contains(&peer)
                && matches!(ack.kind, TrainingAckKind::Start) =>
            {
                acknowledged.insert(peer);
            }
            V3Inbound::PeerDisconnected(peer) if expected_workers.contains(&peer) => {
                // The worker may have admitted the job just before its
                // connection changed.  Re-sending Start is idempotent and
                // causes an already-admitted worker to send the ACK again.
                if let Some((_, worker_start)) =
                    worker_starts.iter().find(|(worker, _)| *worker == peer)
                {
                    let _ = node.network.reconnect_peer(peer).await;
                    let _ = node
                        .network
                        .send_to(
                            peer,
                            Message::Training(TrainingMessage::Start(worker_start.clone())),
                        )
                        .await;
                }
            }
            _ => {}
        }
    }

    // Every worker has now admitted the job before the coordinator can
    // dispatch its first autonomous window.
    let context = Context::new(
        start.clone(),
        Role::Coordinator,
        receiver,
        wait_for_result.then_some(result_sender),
    );
    tokio::spawn(run_context(node.clone(), context));

    if !wait_for_result {
        return Ok(serde_json::json!({
            "kind": "v3_training_started",
            "evidence_class": "REAL_PROCESS_LOCAL",
            "job_id": start.job_id.to_string(),
            "coordinator": start.coordinator.to_string(),
            "term": start.term,
            "groups": start.groups,
            "shards": start.shards,
            "model_state_bytes": start.model_state_bytes,
            "shard_state_bytes": start.shard_state_bytes,
        }));
    }
    let result = timeout(
        Duration::from_secs(start.max_windows.saturating_mul(8).saturating_add(15)),
        result_receiver,
    )
    .await;
    match result {
        Ok(Ok(Ok(value))) => Ok(value),
        Ok(Ok(Err(error))) => Err(NodeError::InvalidConfig(error)),
        Ok(Err(_)) => {
            node.v3_jobs.lock().await.remove(&start.job_id);
            Err(NodeError::InvalidConfig(
                "V3 coordinator task stopped".to_string(),
            ))
        }
        Err(_) => {
            // Dropping the coordinator mailbox is the local cancellation
            // signal. The run loop observes closure and removes its handle;
            // workers have a bounded context lifetime as well.
            node.v3_jobs.lock().await.remove(&start.job_id);
            Err(NodeError::InvalidConfig(
                "V3 training timed out".to_string(),
            ))
        }
    }
}

impl Context {
    fn new(
        start: TrainingStart,
        role: Role,
        receiver: mpsc::Receiver<V3Inbound>,
        result: Option<oneshot::Sender<Result<Value, String>>>,
    ) -> Self {
        let assignment = start.assignment.clone();
        let mut shard_summaries = BTreeMap::new();
        for shard in &start.shards {
            shard_summaries.insert(
                shard.shard_id,
                TrainingShardSummary {
                    shard_id: shard.shard_id,
                    generation: shard.generation,
                    state_hash: shard.state_hash,
                    providers: shard.replicas.clone(),
                },
            );
        }
        // Workers materialize only their assigned parameter/optimizer shard.
        // The coordinator keeps the small reference values needed to assemble
        // a result, but a worker never receives a complete model state map.
        let shard_values = match role {
            Role::Coordinator => start
                .shards
                .iter()
                .map(|shard| {
                    (
                        shard.shard_id,
                        (start.initial_value, start.initial_optimizer),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
            Role::Worker => BTreeMap::from([(
                assignment.shard_id,
                (start.initial_value, start.initial_optimizer),
            )]),
        };
        let initial_loss = start
            .shards
            .iter()
            .map(|shard| samples_loss(0, &samples_for_shard(shard.shard_id)))
            .sum();
        Self {
            coordinator: start.coordinator,
            term: start.term,
            window: 0,
            checkpoint_generation: 0,
            checkpoint: None,
            final_checkpoint_committed: false,
            value: start.initial_value,
            optimizer: start.initial_optimizer,
            generation: assignment.generation,
            samples: start.samples.clone(),
            branch: start.branch,
            assignment,
            failed: HashSet::new(),
            voted_term: start.term,
            voted_candidate: None,
            shard_summaries,
            checkpoints: Vec::new(),
            checkpoint_providers: BTreeMap::new(),
            accepted_updates: 0,
            rejected_updates: 0,
            stragglers: 0,
            group_progress: start
                .groups
                .iter()
                .map(|group| (group.group_id, 0))
                .collect(),
            group_completion_order: Vec::new(),
            non_barrier_progress_observed: false,
            shard_values,
            initial_loss,
            started_at: Instant::now(),
            result,
            start,
            role,
            receiver,
        }
    }

    fn state(&self) -> TrainingState {
        TrainingState {
            job_id: self.start.job_id,
            coordinator: self.coordinator,
            term: self.term,
            membership_epoch: self.start.membership_epoch,
            branch: self.branch,
            window: self.window,
            checkpoint_generation: self.checkpoint_generation,
            checkpoint: self.checkpoint,
            shards: self.shard_summaries.values().cloned().collect(),
        }
    }
}

pub(crate) async fn accept_start(
    node: Arc<Node>,
    peer: NodeId,
    start: TrainingStart,
) -> Result<(), NodeError> {
    if start.coordinator != peer
        || !start.assignment.owners.contains(&node.node_id())
        || !start.participants.contains(&node.node_id())
    {
        return Err(NodeError::InvalidConfig(
            "V3 start is not authorized for this worker".to_string(),
        ));
    }
    if start.shard_state_bytes > node.config.training_memory_bytes {
        return Err(NodeError::InvalidConfig(format!(
            "assigned V3 shard requires {} bytes but worker budget is {} bytes",
            start.shard_state_bytes, node.config.training_memory_bytes
        )));
    }
    Message::Training(TrainingMessage::Start(start.clone()))
        .validate()
        .map_err(|error| NodeError::InvalidConfig(error.to_string()))?;
    persist_start(&node, &start)?;
    if node.v3_jobs.lock().await.contains_key(&start.job_id) {
        send_ack(
            &node,
            peer,
            start.job_id,
            start.term,
            TrainingAckKind::Start,
        )
        .await;
        return Ok(());
    }
    let (sender, receiver) = mpsc::channel(512);
    node.v3_jobs.lock().await.insert(
        start.job_id,
        V3JobHandle {
            sender: sender.clone(),
        },
    );
    let context = Context::new(start.clone(), Role::Worker, receiver, None);
    log_election_event(format_args!(
        "job={} node={} accepted worker start coordinator={} participants={}",
        start.job_id,
        node.node_id(),
        start.coordinator,
        start.participants.len()
    ));
    let restore_sender = sender.clone();
    let restore_coordinator = start.coordinator;
    let restore_node = node.clone();
    tokio::spawn(async move {
        sleep(Duration::from_secs(1)).await;
        let mut connected = false;
        for attempt in 0..V3_STARTUP_COORDINATOR_RECHECKS {
            if restore_node.network.is_connected(restore_coordinator).await {
                connected = true;
                break;
            }
            if attempt + 1 < V3_STARTUP_COORDINATOR_RECHECKS {
                let _ = timeout(
                    V3_ELECTION_SEND_TIMEOUT,
                    restore_node.network.reconnect_peer(restore_coordinator),
                )
                .await;
                sleep(V3_STARTUP_COORDINATOR_RECHECK_DELAY).await;
            }
        }
        log_election_event(format_args!(
            "job={} node={} startup coordinator check peer={} connected={}",
            start.job_id,
            restore_node.node_id(),
            restore_coordinator,
            connected
        ));
        if !connected {
            log_election_event(format_args!(
                "job={} node={} enqueue startup coordinator disconnect peer={}",
                start.job_id,
                restore_node.node_id(),
                restore_coordinator
            ));
            let _ = restore_sender
                .send(V3Inbound::PeerDisconnected(restore_coordinator))
                .await;
        }
    });
    tokio::spawn(run_context(node.clone(), context));
    send_ack(
        &node,
        peer,
        start.job_id,
        start.term,
        TrainingAckKind::Start,
    )
    .await;
    Ok(())
}

async fn run_context(node: Arc<Node>, mut context: Context) {
    let deadline = context.started_at
        + Duration::from_secs(
            context
                .start
                .max_windows
                .saturating_mul(8)
                .saturating_add(30),
        );
    let result = loop {
        if context.role == Role::Coordinator {
            match drive_coordinator(&node, &mut context).await {
                Ok(DriveResult::Finished(value)) => break Ok(value),
                Ok(DriveResult::Continue) => continue,
                Err(error) => {
                    tracing::warn!(
                        job_id = %context.start.job_id,
                        window = context.window,
                        term = context.term,
                        error = %error,
                        "V3 coordinator stopped"
                    );
                    break Err(error.to_string());
                }
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break Err("V3 training context timed out".to_string());
        }
        match timeout(remaining, context.receiver.recv()).await {
            Err(_) => break Err("V3 training context timed out".to_string()),
            Ok(Some(V3Inbound::Message { peer, message })) => {
                if let Err(error) = handle_worker_message(&node, &mut context, peer, message).await
                {
                    tracing::warn!(job_id = %context.start.job_id, error = %error, "V3 worker message rejected");
                }
                // The final committed checkpoint is the durable terminal
                // record for an autonomous job.  Workers must retire their
                // in-memory job handle after observing it; otherwise a
                // completed job remains indistinguishable from a live one
                // and consumes a bounded job slot forever.  The generation
                // check is safe because every V3 start has a fresh JobId and
                // the commit handler rejects older coordination terms.
                if context.start.execution == TrainingExecution::AutonomousLocalSgd
                    && context_covers_final_model(&context)
                {
                    break Ok(serde_json::Value::Null);
                }
                if context.role == Role::Coordinator {
                    continue;
                }
            }
            Ok(Some(V3Inbound::PeerDisconnected(peer))) => {
                log_election_event(format_args!(
                    "job={} node={} observed disconnect peer={} coordinator={} term={} role={:?}",
                    context.start.job_id,
                    node.node_id(),
                    peer,
                    context.coordinator,
                    context.term,
                    context.role
                ));
                tracing::debug!(
                    job_id = %context.start.job_id,
                    node = %node.node_id(),
                    peer = %peer,
                    coordinator = %context.coordinator,
                    "V3 training peer disconnected"
                );
                if peer == context.coordinator {
                    let restored = restore_training_peer(&node, &mut context, peer).await;
                    log_election_event(format_args!(
                        "job={} node={} restore coordinator={} result={}",
                        context.start.job_id,
                        node.node_id(),
                        peer,
                        restored
                    ));
                    if !restored {
                        if let Err(error) = maybe_elect(&node, &mut context, peer).await {
                            log_election_event(format_args!(
                                "job={} node={} election error={error}",
                                context.start.job_id,
                                node.node_id()
                            ));
                            tracing::warn!(job_id = %context.start.job_id, error = %error, "V3 coordinator election failed");
                        }
                    }
                } else if !restore_training_peer(&node, &mut context, peer).await {
                    context.failed.insert(peer);
                }
            }
            Ok(None) => break Err("V3 training mailbox closed".to_string()),
        }
    };
    if let Some(sender) = context.result.take() {
        let _ = sender.send(result);
    }
    let mut jobs = node.v3_jobs.lock().await;
    jobs.remove(&context.start.job_id);
}

enum DriveResult {
    Finished(Value),
    Continue,
}

async fn drive_coordinator(
    node: &Arc<Node>,
    context: &mut Context,
) -> Result<DriveResult, NodeError> {
    if context.start.execution == TrainingExecution::AutonomousLocalSgd {
        return drive_autonomous_coordinator(node, context).await;
    }
    let state = context.state();
    persist_state(node, &state)?;
    node.v3_states
        .lock()
        .await
        .insert(state.job_id, state.clone());
    broadcast_state(node, context, &state).await;

    for window in (context.window + 1)..=context.start.max_windows {
        let mut aggregates = Vec::new();
        for group in context.start.groups.clone() {
            let Some(aggregate) = run_group_window(node, context, &group, window).await? else {
                context.stragglers = context.stragglers.saturating_add(1);
                context.failed.insert(group.aggregator);
                let Some(replacement) = group
                    .members
                    .iter()
                    .copied()
                    .find(|member| !context.failed.contains(member))
                else {
                    return Err(NodeError::InvalidConfig(format!(
                        "no live aggregator remains for V3 group {}",
                        group.group_id
                    )));
                };
                let retry = TrainingGroup {
                    aggregator: replacement,
                    members: group
                        .members
                        .iter()
                        .copied()
                        .filter(|member| !context.failed.contains(member))
                        .collect(),
                    ..group.clone()
                };
                tracing::warn!(
                    job_id = %context.start.job_id,
                    group = group.group_id,
                    failed_aggregator = %group.aggregator,
                    replacement = %replacement,
                    window,
                    "V3 replacing failed group aggregator"
                );
                let Some(aggregate) = run_group_window(node, context, &retry, window).await? else {
                    return Err(NodeError::InvalidConfig(format!(
                        "V3 group {} did not produce an aggregate",
                        group.group_id
                    )));
                };
                if let Some(active_group) = context
                    .start
                    .groups
                    .iter_mut()
                    .find(|active| active.group_id == group.group_id)
                {
                    *active_group = retry.clone();
                }
                aggregates.push(aggregate);
                continue;
            };
            aggregates.push(aggregate);
        }

        context.window = window;
        context.accepted_updates = context.accepted_updates.saturating_add(
            aggregates
                .iter()
                .map(|aggregate| aggregate.contributors.len() as u64)
                .sum::<u64>(),
        );
        for aggregate in &aggregates {
            context
                .shard_values
                .insert(aggregate.shard_id, (aggregate.value, aggregate.optimizer));
            context.shard_summaries.insert(
                aggregate.shard_id,
                TrainingShardSummary {
                    shard_id: aggregate.shard_id,
                    generation: aggregate.generation,
                    state_hash: aggregate.state_artifact,
                    providers: providers(aggregate),
                },
            );
        }

        let state = context.state();
        persist_state(node, &state)?;
        node.v3_states
            .lock()
            .await
            .insert(state.job_id, state.clone());
        if !broadcast_state_and_wait(node, context, &state).await {
            return Err(NodeError::InvalidConfig(
                "V3 state did not reach a quorum; no canonical window committed".to_string(),
            ));
        }

        if window % context.start.checkpoint_every == 0 || window == context.start.max_windows {
            let manifest = make_checkpoint(context, &aggregates)?;
            let commit = TrainingCheckpointCommit {
                job_id: context.start.job_id,
                coordinator: context.coordinator,
                term: context.term,
                membership_epoch: context.start.membership_epoch,
                branch: context.branch,
                manifest: manifest.clone(),
                quorum: context.start.participants.clone(),
            };
            persist_checkpoint(node, &commit)?;
            context.checkpoint_generation = manifest.checkpoint_generation;
            context.checkpoint = Some(manifest.hash);
            context.checkpoints.push(manifest.hash);
            for shard in &manifest.shards {
                context
                    .checkpoint_providers
                    .insert(shard.artifact.to_string(), shard.providers.clone());
            }
            if !broadcast_checkpoint_and_wait(node, context, &commit).await {
                return Err(NodeError::InvalidConfig(
                    "V3 checkpoint commit did not reach a quorum".to_string(),
                ));
            }
            // The checkpoint commit advances durable training state after the
            // window state was committed. Persist that transition as well so
            // a worker restarted after the commit restores the committed
            // checkpoint generation rather than the preceding window's
            // metadata.
            let committed_state = context.state();
            persist_state(node, &committed_state)?;
            node.v3_states
                .lock()
                .await
                .insert(committed_state.job_id, committed_state);
        }
    }

    let final_loss = context
        .shard_values
        .iter()
        .map(|(shard_id, (value, _))| samples_loss(*value, &samples_for_shard(*shard_id)))
        .sum::<i64>();
    let result = serde_json::json!({
        "kind": "v3_distributed_training_reference",
        "evidence_class": "REAL_PROCESS_LOCAL",
        "job_id": context.start.job_id.to_string(),
        "coordinator": context.coordinator.to_string(),
        "term": context.term,
        "windows": context.window,
        "local_steps": context.start.local_steps,
        "sync_mode": "periodic_local_sgd",
        "global_step_barrier": false,
        "groups": context.start.groups,
        "shards": context.start.shards,
        "accepted_updates": context.accepted_updates,
        "rejected_updates": context.rejected_updates,
        "coordinator_update_fanin": 0,
        "maximum_group_fan_in": context
            .start
            .groups
            .iter()
            .map(|group| group.members.len())
            .max()
            .unwrap_or(0),
        "optimizer_state_replication_factor": context
            .start
            .shards
            .iter()
            .map(|shard| shard.replicas.len())
            .max()
            .unwrap_or(0),
        "straggler_windows": context.stragglers,
        "checkpoints": context.checkpoints,
        "checkpoint_providers": context.checkpoint_providers,
        "initial_loss": context.initial_loss,
        "final_loss": final_loss,
        "improved": final_loss < context.initial_loss,
        "all_updates_to_one_coordinator": false,
        "single_optimizer_authority": false,
        "single_checkpoint_authority": false,
        "model_must_fit_one_worker": false,
        "full_model_materialized_on_worker": false,
        "elapsed_ms": context.started_at.elapsed().as_millis(),
    });
    Ok(DriveResult::Finished(result))
}

/// Drive the V3 autonomous local-SGD mode.  Groups advance independently and
/// publish aggregates as they finish a local window.  The coordinator stores
/// bounded job metadata and checkpoint manifests, but it does not dispatch or
/// wait for every group at each window.
async fn drive_autonomous_coordinator(
    node: &Arc<Node>,
    context: &mut Context,
) -> Result<DriveResult, NodeError> {
    tracing::debug!(
        job_id = %context.start.job_id,
        groups = context.start.groups.len(),
        "V3 autonomous coordinator started"
    );
    let state = context.state();
    persist_state(node, &state)?;
    node.v3_states
        .lock()
        .await
        .insert(state.job_id, state.clone());
    broadcast_state(node, context, &state).await;

    for group in context.start.groups.clone() {
        let next_window = context
            .shard_summaries
            .get(&group.shard_id)
            .map(|summary| summary.generation.saturating_add(1))
            .unwrap_or(1);
        if next_window <= context.start.max_windows {
            dispatch_autonomous_window(node, context, &group, next_window).await;
        }
    }

    // A replacement coordinator may recover after one group has already
    // committed its final shard generation.  That group will not emit another
    // aggregate, so completion must be reconstructed from durable shard
    // state rather than inferred only from messages received by this process.
    let mut completed = HashSet::new();
    for group in &context.start.groups {
        let generation = context
            .shard_summaries
            .get(&group.shard_id)
            .map(|summary| summary.generation)
            .unwrap_or(0);
        context.group_progress.insert(group.group_id, generation);
        if generation >= context.start.max_windows {
            completed.insert(group.group_id);
        }
    }
    let deadline = Instant::now()
        + Duration::from_secs(
            context
                .start
                .max_windows
                .saturating_mul(8)
                .saturating_add(30),
        );
    while completed.len() < context.start.groups.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(NodeError::InvalidConfig(
                "autonomous V3 training timed out waiting for group progress".to_string(),
            ));
        }
        let Some(event) = timeout(remaining, context.receiver.recv())
            .await
            .ok()
            .flatten()
        else {
            return Err(NodeError::InvalidConfig(
                "autonomous V3 training mailbox closed".to_string(),
            ));
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Aggregate(aggregate),
            } => {
                let Some(group) = context
                    .start
                    .groups
                    .iter()
                    .find(|group| group.group_id == aggregate.group_id)
                    .cloned()
                else {
                    continue;
                };
                if peer != aggregate.aggregator
                    || !group.members.contains(&peer)
                    || aggregate.job_id != context.start.job_id
                    || aggregate.term != context.term
                    || aggregate.branch != context.branch
                    || aggregate.shard_id != group.shard_id
                    || aggregate.window == 0
                    || aggregate.window > context.start.max_windows
                    || context
                        .shard_summaries
                        .get(&aggregate.shard_id)
                        .is_some_and(|summary| aggregate.generation <= summary.generation)
                {
                    context.rejected_updates = context.rejected_updates.saturating_add(1);
                    continue;
                }
                context.window = context.window.max(aggregate.window);
                context
                    .group_progress
                    .insert(aggregate.group_id, aggregate.window);
                context.accepted_updates = context
                    .accepted_updates
                    .saturating_add(aggregate.contributors.len() as u64);
                context
                    .shard_values
                    .insert(aggregate.shard_id, (aggregate.value, aggregate.optimizer));
                context.shard_summaries.insert(
                    aggregate.shard_id,
                    TrainingShardSummary {
                        shard_id: aggregate.shard_id,
                        generation: aggregate.generation,
                        state_hash: aggregate.state_artifact,
                        providers: providers(&aggregate),
                    },
                );
                let state = context.state();
                persist_state(node, &state)?;
                node.v3_states
                    .lock()
                    .await
                    .insert(state.job_id, state.clone());
                broadcast_state(node, context, &state).await;

                let all_shards_have_state = context
                    .shard_summaries
                    .values()
                    .all(|summary| summary.generation > 0);
                if all_shards_have_state && aggregate.window % context.start.checkpoint_every == 0 {
                    commit_context_checkpoint(node, context).await?;
                }
                if aggregate.window >= context.start.max_windows {
                    if context.group_progress.iter().any(|(group_id, progress)| {
                        *group_id != aggregate.group_id && *progress < context.start.max_windows
                    }) {
                        context.non_barrier_progress_observed = true;
                    }
                    if !completed.contains(&group.group_id) {
                        context.group_completion_order.push(group.group_id);
                    }
                    completed.insert(group.group_id);
                }
            }
            V3Inbound::PeerDisconnected(peer) => {
                // QUIC connection churn is not equivalent to worker death.
                // Give a known group member one bounded reconnect attempt
                // before changing membership or replacing its role.  A
                // permanently stopped process still fails this attempt and
                // follows the normal replacement path below.
                if restore_training_peer(node, context, peer).await {
                    let groups = context
                        .start
                        .groups
                        .iter()
                        .filter(|group| group.aggregator == peer)
                        .cloned()
                        .collect::<Vec<_>>();
                    for group in groups {
                        let next_window = context
                            .shard_summaries
                            .get(&group.shard_id)
                            .map(|summary| summary.generation.saturating_add(1))
                            .unwrap_or(1);
                        if next_window <= context.start.max_windows {
                            dispatch_autonomous_window(node, context, &group, next_window).await;
                        }
                    }
                    continue;
                }
                context.failed.insert(peer);
                let groups = context
                    .start
                    .groups
                    .iter()
                    .filter(|group| group.aggregator == peer)
                    .cloned()
                    .collect::<Vec<_>>();
                for group in groups {
                    for member in &group.members {
                        // The peer that generated this disconnect is already
                        // known to be unavailable.  Retrying it here would
                        // spend another full reconnect timeout before the
                        // surviving group member can take over.  Other
                        // members may only have a stale connection event, so
                        // they still receive one bounded recovery attempt.
                        if *member != peer
                            && context.failed.contains(member)
                            && restore_training_peer(node, context, *member).await
                        {
                            context.failed.remove(member);
                        }
                    }
                    let Some(replacement) = group
                        .members
                        .iter()
                        .copied()
                        .find(|member| !context.failed.contains(member))
                    else {
                        return Err(NodeError::InvalidConfig(format!(
                            "no live autonomous aggregator remains for V3 group {}",
                            group.group_id
                        )));
                    };
                    let retry = TrainingGroup {
                        aggregator: replacement,
                        members: group
                            .members
                            .iter()
                            .copied()
                            .filter(|member| !context.failed.contains(member))
                            .collect(),
                        ..group.clone()
                    };
                    if let Some(active) = context
                        .start
                        .groups
                        .iter_mut()
                        .find(|active| active.group_id == group.group_id)
                    {
                        *active = retry.clone();
                    }
                    let next_window = context
                        .shard_summaries
                        .get(&group.shard_id)
                        .map(|summary| summary.generation.saturating_add(1))
                        .unwrap_or(1);
                    if next_window <= context.start.max_windows {
                        dispatch_autonomous_window(node, context, &retry, next_window).await;
                    }
                }
            }
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Election(election),
            } if election.term > context.term => {
                handle_election(node, context, peer, election).await?;
                // An election request is a pre-vote.  The current
                // coordinator remains authoritative until a quorum-backed
                // replacement publishes a higher-term state record.
            }
            V3Inbound::Message {
                peer,
                message: TrainingMessage::State(state),
            } if state.term > context.term => {
                let previous_coordinator = context.coordinator;
                handle_state(node, context, peer, state).await?;
                if context.coordinator != previous_coordinator {
                    context.role = Role::Worker;
                    return Ok(DriveResult::Continue);
                }
            }
            V3Inbound::Message { .. } => {}
        }
    }

    let all_shards_reached_final_generation = context.start.shards.iter().all(|expected| {
        context
            .shard_summaries
            .get(&expected.shard_id)
            .is_some_and(|actual| actual.generation >= context.start.max_windows)
    });
    if all_shards_reached_final_generation && !context.final_checkpoint_committed {
        // A checkpoint interval is an intermediate durability policy, not a
        // terminal-state policy. Autonomous groups may finish at different
        // times, so the last aggregate can complete the model without
        // landing on a checkpoint interval. Never report success until that
        // complete model has a committed final manifest.
        commit_context_checkpoint(node, context).await?;
    }
    let final_loss = context
        .shard_values
        .iter()
        .map(|(shard_id, (value, _))| samples_loss(*value, &samples_for_shard(*shard_id)))
        .sum::<i64>();
    let result = serde_json::json!({
        "kind": "v3_distributed_training_reference",
        "evidence_class": "REAL_PROCESS_LOCAL",
        "job_id": context.start.job_id.to_string(),
        "coordinator": context.coordinator.to_string(),
        "term": context.term,
        "windows": context.window,
        "local_steps": context.start.local_steps,
        "sync_mode": "autonomous_local_sgd",
        "barrier_scope": "none",
        "global_step_barrier": false,
        "groups": context.start.groups,
        "shards": context.start.shards,
        "accepted_updates": context.accepted_updates,
        "rejected_updates": context.rejected_updates,
        "coordinator_update_fanin": 0,
        "maximum_group_fan_in": context
            .start
            .groups
            .iter()
            .map(|group| group.members.len())
            .max()
            .unwrap_or(0),
        "optimizer_state_replication_factor": context
            .start
            .shards
            .iter()
            .map(|shard| shard.replicas.len())
            .max()
            .unwrap_or(0),
        "straggler_windows": context.stragglers,
        "group_progress": context.group_progress,
        "group_completion_order": context.group_completion_order,
        "non_barrier_progress_observed": context.non_barrier_progress_observed,
        "checkpoints": context.checkpoints,
        "checkpoint_generation": context.checkpoint_generation,
        "final_checkpoint_committed": context.final_checkpoint_committed,
        "checkpoint_providers": context.checkpoint_providers,
        "initial_loss": context.initial_loss,
        "final_loss": final_loss,
        "improved": final_loss < context.initial_loss,
        "all_updates_to_one_coordinator": false,
        "single_optimizer_authority": false,
        "single_checkpoint_authority": false,
        "model_must_fit_one_worker": false,
        "full_model_materialized_on_worker": false,
        "model_state_bytes": context.start.model_state_bytes,
        "shard_state_bytes": context.start.shard_state_bytes,
        "elapsed_ms": context.started_at.elapsed().as_millis(),
    });
    Ok(DriveResult::Finished(result))
}

async fn restore_training_peer(node: &Arc<Node>, context: &mut Context, peer: NodeId) -> bool {
    if peer != context.start.coordinator && !context.start.participants.contains(&peer) {
        return false;
    }
    // A training peer can have several authenticated QUIC sessions while the
    // network replaces a duplicate connection.  For workers and aggregators,
    // an event for one session is not evidence that the peer is unavailable
    // when another sender is still installed.  The coordinator is handled
    // differently: its disappearance is the trigger for term replacement,
    // so that path deliberately forces a fresh bounded handshake below.
    if peer != context.coordinator && node.network.is_connected(peer).await {
        log_election_event(format_args!(
            "job={} node={} peer={} considered restored by existing connection",
            context.start.job_id,
            node.node_id(),
            peer
        ));
        context.failed.remove(&peer);
        return true;
    }
    // A disconnect event may race with a second QUIC session in the network
    // table.  Do not treat that stale sender bit as proof that the peer is
    // alive: force one bounded replacement handshake.  A live peer recovers
    // through its signed address record; a permanently stopped peer fails
    // this attempt and follows the replacement path.
    let reconnect_result = timeout(Duration::from_secs(3), node.network.reconnect_peer(peer))
        .await
        .is_ok_and(|result| result.is_ok());
    log_election_event(format_args!(
        "job={} node={} reconnect peer={} result={} connected={}",
        context.start.job_id,
        node.node_id(),
        peer,
        reconnect_result,
        node.network.is_connected(peer).await
    ));
    if reconnect_result {
        context.failed.remove(&peer);
        return true;
    }
    false
}

async fn dispatch_autonomous_window(
    node: &Arc<Node>,
    context: &mut Context,
    group: &TrainingGroup,
    window: u64,
) {
    tracing::debug!(
        job_id = %context.start.job_id,
        group = group.group_id,
        window,
        aggregator = %group.aggregator,
        "V3 autonomous window dispatch"
    );
    let window_message = TrainingWindow {
        job_id: context.start.job_id,
        coordinator: context.coordinator,
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        window,
        group_id: group.group_id,
        shard_id: group.shard_id,
        aggregator: group.aggregator,
        members: group.members.clone(),
        base_generation: context
            .shard_summaries
            .get(&group.shard_id)
            .map(|summary| summary.generation)
            .unwrap_or(0),
        local_steps: context.start.local_steps,
        max_staleness: context.start.max_staleness,
    };
    for member in &group.members {
        if context.failed.contains(member) {
            continue;
        }
        if *member == node.node_id() {
            if *member == group.aggregator {
                let _ = collect_group_updates(node, context, group, &window_message).await;
            } else if context.assignment.shard_id == group.shard_id {
                let _ =
                    compute_and_send_update(node, context, &window_message, group.aggregator).await;
            }
        } else {
            if let Err(error) = node
                .network
                .send_to(
                    *member,
                    Message::Training(TrainingMessage::Window(window_message.clone())),
                )
                .await
            {
                tracing::warn!(
                    job_id = %context.start.job_id,
                    group = group.group_id,
                    window,
                    member = %member,
                    error = %error,
                    "V3 autonomous window dispatch failed"
                );
            }
        }
    }
}

async fn commit_context_checkpoint(
    node: &Arc<Node>,
    context: &mut Context,
) -> Result<(), NodeError> {
    let manifest = make_checkpoint_from_context(context)?;
    let commit = TrainingCheckpointCommit {
        job_id: context.start.job_id,
        coordinator: context.coordinator,
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        manifest: manifest.clone(),
        quorum: context.start.participants.clone(),
    };
    persist_checkpoint(node, &commit)?;
    context.checkpoint_generation = manifest.checkpoint_generation;
    context.checkpoint = Some(manifest.hash);
    context.checkpoints.push(manifest.hash);
    for shard in &manifest.shards {
        context
            .checkpoint_providers
            .insert(shard.artifact.to_string(), shard.providers.clone());
    }
    let final_checkpoint = manifest.checkpoint_generation
        >= final_checkpoint_generation(&context.start)
        && manifest
            .shards
            .iter()
            .all(|shard| shard.generation >= context.start.max_windows);
    if final_checkpoint {
        // A terminal result is also a restart/recovery boundary.  Ensure the
        // final manifest has been durably observed by the live membership
        // before reporting success.  Intermediate autonomous checkpoints stay
        // non-blocking, so this does not reintroduce a per-window barrier.
        if !broadcast_checkpoint_and_wait(node, context, &commit).await {
            return Err(NodeError::InvalidConfig(
                "V3 final checkpoint did not reach the live membership".to_string(),
            ));
        }
        context.final_checkpoint_committed = true;
    } else {
        broadcast_checkpoint(node, context, &commit).await;
    }
    let state = context.state();
    persist_state(node, &state)?;
    node.v3_states.lock().await.insert(state.job_id, state);
    Ok(())
}

async fn run_group_window(
    node: &Arc<Node>,
    context: &mut Context,
    group: &TrainingGroup,
    window: u64,
) -> Result<Option<TrainingAggregate>, NodeError> {
    let aggregator = group.aggregator;
    tracing::debug!(job_id = %context.start.job_id, node = %node.node_id(), group = group.group_id, window, aggregator = %aggregator, "V3 coordinator dispatching group window");
    let window_message = TrainingWindow {
        job_id: context.start.job_id,
        coordinator: context.coordinator,
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        window,
        group_id: group.group_id,
        shard_id: group.shard_id,
        aggregator,
        members: group.members.clone(),
        base_generation: context
            .shard_summaries
            .get(&group.shard_id)
            .map(|summary| summary.generation)
            .unwrap_or(0),
        local_steps: context.start.local_steps,
        max_staleness: context.start.max_staleness,
    };
    for member in &group.members {
        if context.failed.contains(member) {
            continue;
        }
        if *member == node.node_id() {
            if *member != aggregator {
                if let Err(error) =
                    compute_and_send_update(node, context, &window_message, aggregator).await
                {
                    tracing::debug!(job_id = %context.start.job_id, error = %error, "V3 coordinator could not send its local update");
                }
            }
        } else {
            if let Err(error) = node
                .network
                .send_to(
                    *member,
                    Message::Training(TrainingMessage::Window(window_message.clone())),
                )
                .await
            {
                tracing::debug!(job_id = %context.start.job_id, member = %member, error = %error, "V3 coordinator could not dispatch group window");
            }
        }
    }

    if aggregator == node.node_id() {
        collect_group_updates(node, context, group, &window_message)
            .await
            .map(Some)
    } else {
        wait_for_aggregate(node, context, group.group_id, window).await
    }
}

async fn wait_for_aggregate(
    node: &Arc<Node>,
    context: &mut Context,
    group_id: u16,
    window: u64,
) -> Result<Option<TrainingAggregate>, NodeError> {
    let deadline = Instant::now() + V3_MESSAGE_TIMEOUT;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Ok(None);
        }
        let Some(event) = timeout(remaining, context.receiver.recv())
            .await
            .ok()
            .flatten()
        else {
            return Ok(None);
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Aggregate(aggregate),
            } if peer == aggregate.aggregator
                && aggregate.group_id == group_id
                && aggregate.window == window
                && aggregate.term == context.term
                && aggregate.job_id == context.start.job_id =>
            {
                if !aggregate.contributors.is_empty() {
                    tracing::debug!(
                        job_id = %context.start.job_id,
                        group = group_id,
                        window,
                        aggregator = %aggregate.aggregator,
                        "V3 coordinator accepted group aggregate"
                    );
                    return Ok(Some(aggregate));
                }
            }
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Election(election),
            } if election.term > context.term => {
                let _ = handle_election(node, context, peer, election).await;
                return Err(NodeError::InvalidConfig(
                    "coordinator term changed while waiting for aggregate".to_string(),
                ));
            }
            V3Inbound::PeerDisconnected(peer) => {
                context.failed.insert(peer);
                if peer == context.coordinator {
                    return Ok(None);
                }
            }
            _ => {}
        }
    }
}

async fn collect_group_updates(
    node: &Arc<Node>,
    context: &mut Context,
    group: &TrainingGroup,
    window: &TrainingWindow,
) -> Result<TrainingAggregate, NodeError> {
    tracing::debug!(job_id = %context.start.job_id, node = %node.node_id(), group = group.group_id, window = window.window, "V3 aggregator collecting updates");
    let mut updates = Vec::new();
    let own = compute_update(context, window, node.node_id());
    updates.push(own);
    let expected = group
        .members
        .iter()
        .filter(|member| !context.failed.contains(member))
        .copied()
        .collect::<HashSet<_>>();
    let deadline = Instant::now() + V3_UPDATE_WINDOW_TIMEOUT;
    while updates.len() < expected.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            context.stragglers = context.stragglers.saturating_add(1);
            break;
        }
        let Some(event) = timeout(remaining, context.receiver.recv())
            .await
            .ok()
            .flatten()
        else {
            context.stragglers = context.stragglers.saturating_add(1);
            break;
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Update(update),
            } => {
                if peer != update.worker
                    || update.job_id != context.start.job_id
                    || update.term != context.term
                    || update.membership_epoch != context.start.membership_epoch
                    || update.branch != context.branch
                    || update.window != window.window
                    || update.shard_id != window.shard_id
                    || update.base_generation != window.base_generation
                    || !expected.contains(&update.worker)
                    || updates
                        .iter()
                        .any(|item: &TrainingUpdate| item.worker == update.worker)
                {
                    context.rejected_updates = context.rejected_updates.saturating_add(1);
                    continue;
                }
                let contribution = TrainingContribution {
                    job_id: update.job_id,
                    worker: update.worker,
                    branch: update.branch,
                    plan_generation: update.term.max(1),
                    generation: update.base_generation.max(1),
                    sequence: update.update_sequence.max(1),
                    values: vec![update.value, update.optimizer],
                };
                if node
                    .security
                    .lock()
                    .await
                    .admit_training_update(
                        &contribution,
                        &expected.iter().copied().collect::<Vec<_>>(),
                        now_secs(),
                    )
                    .is_err()
                {
                    context.rejected_updates = context.rejected_updates.saturating_add(1);
                    node.persist_security_state().await;
                    continue;
                }
                node.persist_security_state().await;
                updates.push(update);
            }
            V3Inbound::PeerDisconnected(peer) => {
                if expected.contains(&peer) && restore_training_peer(node, context, peer).await {
                    context.failed.remove(&peer);
                } else {
                    context.failed.insert(peer);
                }
                if expected.contains(&peer) && context.failed.contains(&peer) {
                    context.stragglers = context.stragglers.saturating_add(1);
                }
            }
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Election(election),
            } => {
                let previous_term = context.term;
                handle_election(node, context, peer, election).await?;
                if context.term > previous_term {
                    return Err(NodeError::InvalidConfig(
                        "V3 group window superseded by a newer coordination term".to_string(),
                    ));
                }
            }
            _ => {}
        }
    }
    if updates.is_empty() {
        return Err(NodeError::InvalidConfig(
            "V3 group produced no updates".to_string(),
        ));
    }
    node.metrics
        .training_max_update_fanin
        .fetch_max(updates.len() as u64, std::sync::atomic::Ordering::Relaxed);
    let value = median(updates.iter().map(|update| update.value).collect());
    let optimizer = median(updates.iter().map(|update| update.optimizer).collect());
    if value.saturating_abs() > V3_MAX_DELTA {
        return Err(NodeError::InvalidConfig(
            "V3 aggregate exceeded the update bound".to_string(),
        ));
    }
    let generation = window.base_generation.saturating_add(1);
    let new_state = TrainingShardState {
        job_id: context.start.job_id,
        owner: node.node_id(),
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        shard_id: window.shard_id,
        generation,
        value,
        optimizer,
        state_hash: state_hash(
            context.start.job_id,
            window.shard_id,
            generation,
            value,
            optimizer,
        ),
    };
    // The immutable shard artifact is the recovery source for replicas.  It
    // contains the complete small reference shard state, while the training
    // protocol carries only bounded metadata and updates.  A future runtime
    // can replace this encoding with a tensor chunk manifest without changing
    // ownership or verification semantics.
    let state_artifact = node.store.put_artifact(&serde_json::to_vec(&new_state)?)?;
    persist_shard_state(node, &new_state)?;
    for member in &group.members {
        if *member != node.node_id() && !context.failed.contains(member) {
            let _ = node
                .network
                .send_to(
                    *member,
                    Message::Training(TrainingMessage::ShardState(new_state.clone())),
                )
                .await;
        }
    }
    let mut replicas = vec![node.node_id()];
    for member in &group.members {
        if *member == node.node_id() || context.failed.contains(member) {
            continue;
        }
        let offer = TrainingCheckpointOffer {
            job_id: context.start.job_id,
            creator: node.node_id(),
            term: context.term,
            checkpoint_generation: generation,
            shard_id: window.shard_id,
            artifact: state_artifact,
        };
        let mut offer_sent = false;
        for attempt in 0..3 {
            // A successful enqueue does not prove that the peer's current
            // QUIC writer still owns a live stream.  After an offer has
            // timed out, replace that authenticated session once before
            // retrying.  The reconnect is bounded and the offer remains
            // content-addressed/idempotent, so this repairs transport churn
            // without weakening the two-provider requirement.
            if attempt > 0 {
                let _ = timeout(V3_REPLICATION_TIMEOUT, node.network.reconnect_peer(*member)).await;
            }
            if timeout(
                V3_REPLICATION_TIMEOUT,
                node.network.send_to(
                    *member,
                    Message::Training(TrainingMessage::CheckpointOffer(offer.clone())),
                ),
            )
            .await
            .is_ok_and(|result| result.is_ok())
            {
                offer_sent = true;
                break;
            }
        }
        if !offer_sent {
            // A provider that cannot acknowledge a verified artifact after
            // bounded session replacement is unavailable for this group
            // window. Keep the job live with the remaining owner instead of
            // leaving the worker mailbox wedged forever; the aggregate and
            // checkpoint metadata expose the reduced provider set.
            context.failed.insert(*member);
            context.stragglers = context.stragglers.saturating_add(1);
            continue;
        }
        let deadline = Instant::now() + V3_REPLICATION_TIMEOUT;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Some(event) = timeout(remaining, context.receiver.recv())
                .await
                .ok()
                .flatten()
            else {
                break;
            };
            match event {
                V3Inbound::Message {
                    peer,
                    message: TrainingMessage::Ack(ack),
                } if peer == *member
                    && ack.peer == *member
                    && ack.job_id == context.start.job_id
                    && matches!(ack.kind, TrainingAckKind::Checkpoint { artifact } if artifact == state_artifact) =>
                {
                    replicas.push(*member);
                    break;
                }
                V3Inbound::Message {
                    peer,
                    message: TrainingMessage::Update(update),
                } => {
                    context.rejected_updates = context.rejected_updates.saturating_add(1);
                    let _ = (peer, update);
                }
                V3Inbound::Message {
                    peer,
                    message: TrainingMessage::Election(election),
                } => {
                    let previous_term = context.term;
                    handle_election(node, context, peer, election).await?;
                    if context.term > previous_term {
                        return Err(NodeError::InvalidConfig(
                            "V3 replication superseded by a newer coordination term".to_string(),
                        ));
                    }
                }
                V3Inbound::PeerDisconnected(disconnected) if disconnected == *member => {
                    context.failed.insert(*member);
                    context.stragglers = context.stragglers.saturating_add(1);
                    break;
                }
                _ => {}
            }
        }
        if !replicas.contains(member) {
            context.failed.insert(*member);
            context.stragglers = context.stragglers.saturating_add(1);
        }
    }
    let required_replicas = group
        .members
        .iter()
        .filter(|member| !context.failed.contains(member))
        .count()
        .clamp(1, 3);
    if replicas.len() < required_replicas {
        return Err(NodeError::InvalidConfig(format!(
            "V3 shard {} reached only {} of {} required providers",
            window.shard_id,
            replicas.len(),
            required_replicas
        )));
    }
    context.value = value;
    context.optimizer = optimizer;
    context.generation = generation;
    let aggregate = TrainingAggregate {
        job_id: context.start.job_id,
        aggregator: node.node_id(),
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        window: window.window,
        group_id: group.group_id,
        shard_id: window.shard_id,
        generation,
        value,
        optimizer,
        loss: updates.iter().map(|update| update.loss).sum::<i64>() / updates.len() as i64,
        contributors: updates.iter().map(|update| update.worker).collect(),
        state_artifact,
        replicas,
    };
    tracing::debug!(job_id = %context.start.job_id, node = %node.node_id(), group = group.group_id, window = window.window, contributors = aggregate.contributors.len(), replicas = aggregate.replicas.len(), "V3 aggregator produced aggregate");
    if let Err(error) = node
        .network
        .send_to(
            context.coordinator,
            Message::Training(TrainingMessage::Aggregate(aggregate.clone())),
        )
        .await
    {
        tracing::debug!(job_id = %context.start.job_id, coordinator = %context.coordinator, error = %error, "V3 aggregator could not send aggregate");
        if context.start.execution != TrainingExecution::AutonomousLocalSgd {
            return Err(error.into());
        }
    }
    node.metrics
        .training_aggregates_sent
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(aggregate)
}

async fn compute_and_send_update(
    node: &Arc<Node>,
    context: &Context,
    window: &TrainingWindow,
    aggregator: NodeId,
) -> Result<(), NodeError> {
    let update = compute_update(context, window, node.node_id());
    node.network
        .send_to(
            aggregator,
            Message::Training(TrainingMessage::Update(update)),
        )
        .await?;
    node.metrics
        .training_updates_sent
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

fn compute_update(context: &Context, window: &TrainingWindow, worker: NodeId) -> TrainingUpdate {
    let mut value = context.value;
    let mut optimizer = context.optimizer;
    for _ in 0..window.local_steps {
        let mut gradient_sum = 0_i64;
        for sample in &context.samples {
            let prediction = value.saturating_mul(sample.feature);
            let error = prediction.saturating_sub(sample.target);
            gradient_sum = gradient_sum.saturating_add(error.saturating_mul(sample.feature));
        }
        let gradient = gradient_sum / context.samples.len().max(1) as i64;
        optimizer = gradient;
        // The fixed-point reference model intentionally uses a conservative
        // learning rate so integer updates converge without overflowing the
        // protocol's bounded update domain.
        let delta = (gradient as i128 * 50_000_i128 / TRAINING_SCALE as i128) as i64;
        value = value.saturating_sub(delta);
    }
    TrainingUpdate {
        job_id: context.start.job_id,
        worker,
        term: context.term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        window: window.window,
        shard_id: window.shard_id,
        base_generation: window.base_generation,
        update_sequence: window.window.saturating_add(1),
        value,
        optimizer,
        loss: samples_loss(value, &context.samples),
        samples: context.samples.len() as u32,
    }
}

async fn handle_worker_message(
    node: &Arc<Node>,
    context: &mut Context,
    peer: NodeId,
    message: TrainingMessage,
) -> Result<(), NodeError> {
    match message {
        TrainingMessage::Window(window) => {
            tracing::debug!(job_id = %context.start.job_id, node = %node.node_id(), peer = %peer, group = window.group_id, window = window.window, aggregator = %window.aggregator, "V3 worker received window");
            let sender_is_coordinator = peer == window.coordinator;
            let sender_is_group_aggregator =
                peer == window.aggregator && window.members.contains(&peer);
            let window_rejected = context.role != Role::Worker
                || (!sender_is_coordinator && !sender_is_group_aggregator)
                || (window.coordinator != context.start.coordinator
                    && !context.start.participants.contains(&window.coordinator))
                || !window
                    .members
                    .iter()
                    .all(|member| context.start.participants.contains(member))
                || window.term < context.term
                || window.membership_epoch != context.start.membership_epoch
                || window.branch != context.branch
                || window.members.is_empty()
                || !window.members.contains(&node.node_id())
                || context.assignment.shard_id != window.shard_id;
            if window_rejected {
                context.rejected_updates = context.rejected_updates.saturating_add(1);
                return Ok(());
            }
            if window.term > context.term {
                // Election traffic can be lost while a QUIC session is
                // being replaced.  A window is an authenticated, bounded
                // command from a participant in the persisted membership;
                // adopting its newer term lets a live worker rejoin the
                // current branch without accepting stale commands.  The
                // participant, branch, membership and monotonic-term checks
                // above are deliberately performed before this transition.
                context.term = window.term;
                context.coordinator = window.coordinator;
                context.failed.clear();
                let state = context.state();
                persist_state(node, &state)?;
                node.v3_states.lock().await.insert(state.job_id, state);
            }
            context.window = context.window.max(window.window);
            context
                .group_progress
                .insert(window.group_id, window.window);
            if node.config.training_window_delay_ms > 0 {
                sleep(Duration::from_millis(node.config.training_window_delay_ms)).await;
            }
            if node.node_id() == window.aggregator {
                let Some(base_group) = group_for(&context.start, window.group_id) else {
                    return Ok(());
                };
                let group = TrainingGroup {
                    group_id: base_group.group_id,
                    shard_id: base_group.shard_id,
                    members: window.members.clone(),
                    aggregator: window.aggregator,
                };
                let mut current_window = window;
                loop {
                    let aggregate =
                        collect_group_updates(node, context, &group, &current_window).await?;
                    context.window = context.window.max(current_window.window);
                    context.generation = aggregate.generation;
                    context.value = aggregate.value;
                    context.optimizer = aggregate.optimizer;
                    context.shard_summaries.insert(
                        aggregate.shard_id,
                        TrainingShardSummary {
                            shard_id: aggregate.shard_id,
                            generation: aggregate.generation,
                            state_hash: aggregate.state_artifact,
                            providers: providers(&aggregate),
                        },
                    );
                    if context.start.execution != TrainingExecution::AutonomousLocalSgd
                        || current_window.window >= context.start.max_windows
                    {
                        break;
                    }
                    let next_window = TrainingWindow {
                        job_id: context.start.job_id,
                        coordinator: context.coordinator,
                        term: context.term,
                        membership_epoch: context.start.membership_epoch,
                        branch: context.branch,
                        window: current_window.window.saturating_add(1),
                        group_id: group.group_id,
                        shard_id: group.shard_id,
                        aggregator: node.node_id(),
                        members: group.members.clone(),
                        base_generation: aggregate.generation,
                        local_steps: context.start.local_steps,
                        max_staleness: context.start.max_staleness,
                    };
                    for member in &group.members {
                        if *member != node.node_id() && !context.failed.contains(member) {
                            let _ = node
                                .network
                                .send_to(
                                    *member,
                                    Message::Training(TrainingMessage::Window(next_window.clone())),
                                )
                                .await;
                        }
                    }
                    current_window = next_window;
                }
            } else {
                compute_and_send_update(node, context, &window, window.aggregator).await?;
            }
        }
        TrainingMessage::ShardState(state) => {
            if peer != state.owner
                || state.job_id != context.start.job_id
                || state.term < context.term
                || state.branch != context.branch
                || state.shard_id != context.assignment.shard_id
                || state.generation < context.generation
                || state_hash(
                    state.job_id,
                    state.shard_id,
                    state.generation,
                    state.value,
                    state.optimizer,
                ) != state.state_hash
            {
                return Ok(());
            }
            context.value = state.value;
            context.optimizer = state.optimizer;
            context.generation = state.generation;
            persist_shard_state(node, &state)?;
        }
        TrainingMessage::State(state) => {
            handle_state(node, context, peer, state).await?;
        }
        TrainingMessage::Election(election) => {
            handle_election(node, context, peer, election).await?;
        }
        TrainingMessage::CheckpointOffer(offer) => {
            handle_checkpoint_offer(node, context, peer, offer).await?;
        }
        TrainingMessage::CheckpointCommit(commit) => {
            handle_checkpoint_commit(node, context, peer, commit).await?;
        }
        TrainingMessage::Ack(ack) => {
            if let TrainingAckKind::Vote { .. } = ack.kind {
                // Election code consumes votes directly while it is active.
                context.voted_term = context.voted_term.max(ack.term);
            }
        }
        TrainingMessage::Start(_) | TrainingMessage::Update(_) | TrainingMessage::Aggregate(_) => {}
    }
    Ok(())
}

async fn handle_state(
    node: &Arc<Node>,
    context: &mut Context,
    peer: NodeId,
    state: TrainingState,
) -> Result<(), NodeError> {
    if peer != state.coordinator
        || state.job_id != context.start.job_id
        || state.branch != context.branch
        || state.term < context.term
        || (state.term == context.term && state.window < context.window)
        || (state.term == context.term
            && state.checkpoint_generation == context.checkpoint_generation
            && state.checkpoint.is_some()
            && context.checkpoint.is_some()
            && state.checkpoint != context.checkpoint)
    {
        return Ok(());
    }
    context.coordinator = state.coordinator;
    context.term = state.term;
    context.window = state.window;
    if state.checkpoint_generation > context.checkpoint_generation {
        context.checkpoint_generation = state.checkpoint_generation;
        context.checkpoint = state.checkpoint;
    } else if context.checkpoint.is_none()
        && state.checkpoint_generation == context.checkpoint_generation
    {
        context.checkpoint = state.checkpoint;
    }
    for shard in &state.shards {
        let should_advance = context
            .shard_summaries
            .get(&shard.shard_id)
            .is_none_or(|current| shard.generation > current.generation);
        if should_advance {
            context
                .shard_summaries
                .insert(shard.shard_id, shard.clone());
        }
    }
    let merged_state = context.state();
    let state_hash = merged_state
        .shards
        .first()
        .map(|shard| shard.state_hash)
        .unwrap_or_default();
    persist_state(node, &merged_state)?;
    node.v3_states
        .lock()
        .await
        .insert(merged_state.job_id, merged_state.clone());
    send_ack(
        node,
        peer,
        merged_state.job_id,
        merged_state.term,
        TrainingAckKind::State {
            window: merged_state.window,
            state_hash,
        },
    )
    .await;
    Ok(())
}

async fn handle_election(
    node: &Arc<Node>,
    context: &mut Context,
    peer: NodeId,
    election: TrainingElection,
) -> Result<(), NodeError> {
    log_election_event(format_args!(
        "job={} node={} received election candidate={} peer={} election_term={} local_term={}",
        context.start.job_id,
        node.node_id(),
        election.candidate,
        peer,
        election.term,
        context.term
    ));
    if peer != election.candidate
        || election.job_id != context.start.job_id
        || election.membership_epoch != context.start.membership_epoch
        || election.branch != context.branch
        || !context.start.participants.contains(&election.candidate)
    {
        return Ok(());
    }
    // Election changes coordination, not the committed training state. Send
    // the candidate this voter's bounded state metadata, but do not advance
    // the active term yet.  This is a pre-vote: changing the active term
    // before the candidate has a quorum would make a failed election reject
    // valid commands from the current coordinator and create split brain.
    let recovery_state = context.state();
    let granted = if election.term > context.term {
        context.voted_term = election.term;
        context.voted_candidate = Some(election.candidate);
        true
    } else {
        election.term == context.voted_term && context.voted_candidate == Some(election.candidate)
    };
    send_ack(
        node,
        peer,
        election.job_id,
        election.term,
        TrainingAckKind::Vote {
            candidate: election.candidate,
            term: election.term,
            last_window: context.window,
            granted,
        },
    )
    .await;
    log_election_event(format_args!(
        "job={} node={} vote candidate={} term={} granted={}",
        context.start.job_id,
        node.node_id(),
        election.candidate,
        election.term,
        granted
    ));
    if granted {
        let _ = node
            .network
            .send_to(
                peer,
                Message::Training(TrainingMessage::State(recovery_state)),
            )
            .await;
    }
    Ok(())
}

async fn maybe_elect(
    node: &Arc<Node>,
    context: &mut Context,
    failed_coordinator: NodeId,
) -> Result<(), NodeError> {
    context.failed.insert(failed_coordinator);
    let previous_term = context.term;
    let previous_coordinator = context.coordinator;
    let aggregators = context
        .start
        .groups
        .iter()
        .map(|group| group.aggregator)
        .collect::<HashSet<_>>();
    let mut candidates = context
        .start
        .participants
        .iter()
        .copied()
        .filter(|peer| !context.failed.contains(peer) && !aggregators.contains(peer))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates = context
            .start
            .participants
            .iter()
            .copied()
            .filter(|peer| !context.failed.contains(peer))
            .collect();
    }
    candidates.sort_unstable();
    let Some(candidate) = candidates.first().copied() else {
        return Err(NodeError::InvalidConfig(
            "no V3 election candidate remains".to_string(),
        ));
    };
    log_election_event(format_args!(
        "job={} node={} election candidate={} failed={} participants={} local_term={}",
        context.start.job_id,
        node.node_id(),
        candidate,
        failed_coordinator,
        context.start.participants.len(),
        context.term
    ));
    if candidate != node.node_id() {
        log_election_event(format_args!(
            "job={} node={} will not start election; candidate={} owns term={}",
            context.start.job_id,
            node.node_id(),
            candidate,
            context.term.saturating_add(1)
        ));
        return Ok(());
    }
    let term = context.term.saturating_add(1);
    let election = TrainingElection {
        job_id: context.start.job_id,
        candidate,
        term,
        membership_epoch: context.start.membership_epoch,
        branch: context.branch,
        last_window: context.window,
        last_state_hash: context
            .shard_summaries
            .values()
            .next()
            .map(|summary| summary.state_hash)
            .unwrap_or_default(),
    };
    // A PeerDisconnected event is generated for individual QUIC sessions, not
    // for a durable participant membership change.  A replacement election
    // therefore retries the same term and full membership set.  This keeps a
    // transient connection rotation from permanently removing a vote while
    // retaining the normal majority quorum below.
    let quorum_size = quorum(context.start.participants.len());
    let mut votes = HashSet::from([node.node_id()]);
    log_election_event(format_args!(
        "job={} node={} requesting term={} quorum={}",
        context.start.job_id,
        node.node_id(),
        term,
        quorum_size
    ));
    for attempt in 0..V3_ELECTION_ATTEMPTS {
        let mut sends = Vec::new();
        for peer in &context.start.participants {
            if *peer == node.node_id() || *peer == failed_coordinator {
                continue;
            }
            let network = node.network.clone();
            let message = Message::Training(TrainingMessage::Election(election.clone()));
            let peer = *peer;
            sends.push(tokio::spawn(async move {
                // A vote request is control traffic and must not be lost just
                // because the peer is between authenticated QUIC sessions.
                // Reconnect only when needed on the first attempt, then
                // force a bounded session replacement on later attempts.
                if attempt > 0 || !network.is_connected(peer).await {
                    let _ = timeout(Duration::from_secs(3), network.reconnect_peer(peer)).await;
                }
                timeout(V3_ELECTION_SEND_TIMEOUT, network.send_to(peer, message))
                    .await
                    .is_ok_and(|result| result.is_ok())
            }));
        }
        for send in sends {
            if !send.await.unwrap_or(false) {
                tracing::debug!(
                    job_id = %context.start.job_id,
                    candidate = %candidate,
                    attempt,
                    "V3 election send did not complete within the bounded retry window"
                );
            }
        }

        let deadline = Instant::now() + V3_ELECTION_TIMEOUT;
        while votes.len() < quorum_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Some(event) = timeout(remaining, context.receiver.recv())
                .await
                .ok()
                .flatten()
            else {
                break;
            };
            match event {
                V3Inbound::Message {
                    peer,
                    message:
                        TrainingMessage::Ack(TrainingAck {
                            job_id,
                            peer: ack_peer,
                            term: ack_term,
                            kind:
                                TrainingAckKind::Vote {
                                    candidate: ack_candidate,
                                    term: vote_term,
                                    granted,
                                    ..
                                },
                            ..
                        }),
                } if peer != node.node_id()
                    && peer == ack_peer
                    && context.start.participants.contains(&peer)
                    && job_id == context.start.job_id
                    && ack_term == term
                    && vote_term == term
                    && ack_candidate == candidate
                    && granted =>
                {
                    votes.insert(peer);
                    log_election_event(format_args!(
                        "job={} node={} received vote from={} votes={}/{} term={}",
                        context.start.job_id,
                        node.node_id(),
                        peer,
                        votes.len(),
                        quorum_size,
                        term
                    ));
                }
                V3Inbound::Message {
                    peer,
                    message: TrainingMessage::Election(other),
                } if other.term > term => {
                    handle_election(node, context, peer, other).await?;
                    return Ok(());
                }
                V3Inbound::Message {
                    peer,
                    message: TrainingMessage::State(state),
                } if peer != node.node_id()
                    && context.start.participants.contains(&peer)
                    && state.job_id == context.start.job_id
                    && state.branch == context.branch
                    && state.term == previous_term
                    && state.coordinator == previous_coordinator
                    && (state.window > context.window
                        || state.checkpoint_generation > context.checkpoint_generation
                        || state.shards.iter().any(|shard| {
                            context
                                .shard_summaries
                                .get(&shard.shard_id)
                                .is_none_or(|current| shard.generation > current.generation)
                        })) =>
                {
                    // This is recovery metadata from the term that just
                    // ended. It cannot move the candidate backwards or
                    // change the job branch; the election quorum still
                    // decides whether the new term is valid.
                    context.window = context.window.max(state.window);
                    context.checkpoint_generation = context
                        .checkpoint_generation
                        .max(state.checkpoint_generation);
                    context.checkpoint = state.checkpoint;
                    for shard in state.shards {
                        let replace = context
                            .shard_summaries
                            .get(&shard.shard_id)
                            .is_none_or(|current| shard.generation > current.generation);
                        if replace {
                            context.shard_summaries.insert(shard.shard_id, shard);
                        }
                    }
                }
                // Session churn must not turn a healthy participant into an
                // election failure.  The next attempt resends the vote
                // request and the majority requirement remains unchanged.
                V3Inbound::PeerDisconnected(_) => {}
                _ => {}
            }
        }
        if votes.len() >= quorum_size {
            break;
        }
        if attempt + 1 < V3_ELECTION_ATTEMPTS {
            sleep(V3_ELECTION_RETRY_DELAY).await;
        }
    }
    if votes.len() < quorum_size {
        log_election_event(format_args!(
            "job={} node={} quorum failed votes={}/{} term={}",
            context.start.job_id,
            node.node_id(),
            votes.len(),
            quorum_size,
            term
        ));
        return Err(NodeError::InvalidConfig(format!(
            "V3 election could not obtain a quorum: received {} of {} votes",
            votes.len(),
            quorum_size
        )));
    }
    context.role = Role::Coordinator;
    context.term = term;
    context.coordinator = node.node_id();
    log_election_event(format_args!(
        "job={} node={} became coordinator term={} votes={}/{}",
        context.start.job_id,
        node.node_id(),
        term,
        votes.len(),
        quorum_size
    ));
    context.voted_term = term;
    context.voted_candidate = Some(node.node_id());
    // Connection-level failure observations are local and may describe a
    // superseded QUIC session rather than a dead participant.  A new term
    // must retry the full membership set; only the process that triggered
    // this election is known to be unavailable.  Any other genuinely dead
    // peer will be rejected again by the bounded dispatch/replacement path.
    context.failed.clear();
    context.failed.insert(failed_coordinator);
    let state = context.state();
    persist_state(node, &state)?;
    node.v3_states
        .lock()
        .await
        .insert(state.job_id, state.clone());
    broadcast_state(node, context, &state).await;
    Ok(())
}

async fn handle_checkpoint_offer(
    node: &Arc<Node>,
    context: &Context,
    peer: NodeId,
    offer: TrainingCheckpointOffer,
) -> Result<(), NodeError> {
    if peer != offer.creator || offer.job_id != context.start.job_id || offer.term < context.term {
        return Ok(());
    }
    let verified = if node.store.has_artifact(offer.artifact) {
        node.store.verify_artifact(offer.artifact).is_ok()
    } else {
        node.fetch_artifact(&peer.to_string(), &offer.artifact.to_string())
            .await
            .is_ok()
    };
    if verified {
        if let Ok((bytes, size)) =
            node.store
                .read_artifact_range(offer.artifact, 0, 8 * 1024 * 1024)
            && size <= 8 * 1024 * 1024
            && let Ok(state) = serde_json::from_slice::<TrainingShardState>(&bytes)
            && state.job_id == context.start.job_id
            && state.branch == context.branch
            && state.shard_id == context.assignment.shard_id
            && state.generation >= context.generation
            && state_hash(
                state.job_id,
                state.shard_id,
                state.generation,
                state.value,
                state.optimizer,
            ) == state.state_hash
        {
            persist_shard_state(node, &state)?;
        }
        send_ack(
            node,
            peer,
            offer.job_id,
            offer.term,
            TrainingAckKind::Checkpoint {
                artifact: offer.artifact,
            },
        )
        .await;
    }
    Ok(())
}

async fn handle_checkpoint_commit(
    node: &Arc<Node>,
    context: &mut Context,
    peer: NodeId,
    commit: TrainingCheckpointCommit,
) -> Result<(), NodeError> {
    if peer != commit.coordinator
        || commit.job_id != context.start.job_id
        || commit.branch != context.branch
        || commit.term < context.term
        || commit.manifest.checkpoint_generation < context.checkpoint_generation
        || checkpoint_hash(&commit.manifest) != commit.manifest.hash
    {
        return Ok(());
    }
    persist_checkpoint(node, &commit)?;
    context.term = commit.term;
    context.coordinator = commit.coordinator;
    context.window = context.window.max(commit.manifest.model_generation);
    context.checkpoint_generation = commit.manifest.checkpoint_generation;
    context.checkpoint = Some(commit.manifest.hash);
    for shard in &commit.manifest.shards {
        context.shard_summaries.insert(
            shard.shard_id,
            TrainingShardSummary {
                shard_id: shard.shard_id,
                generation: shard.generation,
                state_hash: shard.artifact,
                providers: shard.providers.clone(),
            },
        );
    }
    context.final_checkpoint_committed = commit.manifest.checkpoint_generation
        >= final_checkpoint_generation(&context.start)
        && commit
            .manifest
            .shards
            .iter()
            .all(|shard| shard.generation >= context.start.max_windows);
    let committed_state = context.state();
    persist_state(node, &committed_state)?;
    node.v3_states
        .lock()
        .await
        .insert(committed_state.job_id, committed_state);
    send_ack(
        node,
        peer,
        commit.job_id,
        commit.term,
        TrainingAckKind::Checkpoint {
            artifact: commit.manifest.hash,
        },
    )
    .await;
    Ok(())
}

async fn send_ack(
    node: &Arc<Node>,
    peer: NodeId,
    job_id: intelligence_protocol::JobId,
    term: u64,
    kind: TrainingAckKind,
) {
    let _ = node
        .network
        .send_to(
            peer,
            Message::Training(TrainingMessage::Ack(TrainingAck {
                job_id,
                peer: node.node_id(),
                term,
                kind,
            })),
        )
        .await;
}

async fn broadcast_state(node: &Arc<Node>, context: &Context, state: &TrainingState) {
    for peer in &context.start.participants {
        // A node already has its own state. Sending a self-message through
        // normal routing can select a relay and create an invalid envelope
        // whose origin and target are the same identity.
        if *peer == node.node_id() || context.failed.contains(peer) {
            continue;
        }
        let _ = timeout(
            V3_MESSAGE_TIMEOUT,
            node.network.send_to(
                *peer,
                Message::Training(TrainingMessage::State(state.clone())),
            ),
        )
        .await;
    }
}

async fn broadcast_state_and_wait(
    node: &Arc<Node>,
    context: &mut Context,
    state: &TrainingState,
) -> bool {
    broadcast_state(node, context, state).await;
    let expected_acks = context
        .start
        .participants
        .iter()
        .copied()
        .filter(|peer| !context.failed.contains(peer))
        .collect::<HashSet<_>>();
    let mut acks = HashSet::from([node.node_id()]);
    let deadline = Instant::now() + V3_MESSAGE_TIMEOUT;
    while acks.len() < expected_acks.len() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return false;
        }
        let Some(event) = timeout(remaining, context.receiver.recv())
            .await
            .ok()
            .flatten()
        else {
            return false;
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Ack(ack),
            } if ack.job_id == state.job_id
                && ack.term == state.term
                && matches!(ack.kind, TrainingAckKind::State { window, .. } if window == state.window) =>
            {
                acks.insert(peer);
            }
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Election(election),
            } if election.term > context.term => {
                let _ = handle_election(node, context, peer, election).await;
                return false;
            }
            V3Inbound::PeerDisconnected(peer) => {
                context.failed.insert(peer);
            }
            _ => {}
        }
    }
    true
}

async fn broadcast_checkpoint_and_wait(
    node: &Arc<Node>,
    context: &mut Context,
    commit: &TrainingCheckpointCommit,
) -> bool {
    log_election_event(format_args!(
        "job={} node={} final checkpoint={} generation={} sending to {} participants failed={}",
        commit.job_id,
        node.node_id(),
        commit.manifest.hash,
        commit.manifest.checkpoint_generation,
        context.start.participants.len(),
        context.failed.len()
    ));
    for peer in &context.start.participants {
        if *peer == node.node_id() || context.failed.contains(peer) {
            continue;
        }
        let _ = timeout(
            V3_MESSAGE_TIMEOUT,
            node.network.send_to(
                *peer,
                Message::Training(TrainingMessage::CheckpointCommit(commit.clone())),
            ),
        )
        .await;
    }
    let mut expected_acks = context
        .start
        .participants
        .iter()
        .copied()
        .filter(|peer| !context.failed.contains(peer))
        .collect::<HashSet<_>>();
    let mut acks = HashSet::new();
    if expected_acks.contains(&node.node_id()) {
        acks.insert(node.node_id());
    }
    // The coordinator's own verified manifest is one durable copy.  A final
    // checkpoint therefore needs a majority of the currently live
    // participant set, not an acknowledgment from every transient QUIC
    // session.  Waiting for every participant would turn a connection churn
    // event into a global training barrier and would make a healthy replicated
    // checkpoint impossible to commit while one peer is reconnecting.
    let required_acks =
        |participants: usize| quorum(participants.saturating_add(1)).saturating_sub(1);
    let deadline = Instant::now() + V3_MESSAGE_TIMEOUT;
    while acks.len() < required_acks(expected_acks.len()) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            log_election_event(format_args!(
                "job={} node={} checkpoint={} ack timeout received={:?} expected={:?}",
                commit.job_id,
                node.node_id(),
                commit.manifest.hash,
                acks,
                expected_acks
            ));
            return false;
        }
        let Some(event) = timeout(remaining, context.receiver.recv())
            .await
            .ok()
            .flatten()
        else {
            log_election_event(format_args!(
                "job={} node={} checkpoint={} receiver closed received={:?} expected={:?}",
                commit.job_id,
                node.node_id(),
                commit.manifest.hash,
                acks,
                expected_acks
            ));
            return false;
        };
        match event {
            V3Inbound::Message {
                peer,
                message: TrainingMessage::Ack(ack),
            } if peer == ack.peer
                && expected_acks.contains(&peer)
                && ack.job_id == commit.job_id
                && ack.term == commit.term
                && matches!(ack.kind, TrainingAckKind::Checkpoint { artifact } if artifact == commit.manifest.hash) =>
            {
                acks.insert(peer);
                log_election_event(format_args!(
                    "job={} node={} checkpoint={} ack from={} received={:?} expected={:?}",
                    commit.job_id,
                    node.node_id(),
                    commit.manifest.hash,
                    peer,
                    acks,
                    expected_acks
                ));
            }
            V3Inbound::PeerDisconnected(peer) => {
                // A peer that disappears while a checkpoint is being
                // committed is no longer part of the live acknowledgment
                // set. It remains an artifact provider only if its
                // already-verified shard exists; the next training window
                // will use the normal bounded recovery path for its role.
                expected_acks.remove(&peer);
                context.failed.insert(peer);
            }
            _ => {}
        }
    }
    true
}

async fn broadcast_checkpoint(
    node: &Arc<Node>,
    context: &Context,
    commit: &TrainingCheckpointCommit,
) {
    for peer in &context.start.participants {
        if *peer == node.node_id() || context.failed.contains(peer) {
            continue;
        }
        let _ = timeout(
            V3_MESSAGE_TIMEOUT,
            node.network.send_to(
                *peer,
                Message::Training(TrainingMessage::CheckpointCommit(commit.clone())),
            ),
        )
        .await;
    }
}

fn make_checkpoint_from_context(
    context: &Context,
) -> Result<TrainingCheckpointManifest, NodeError> {
    let shards = context
        .shard_summaries
        .values()
        .map(|summary| TrainingCheckpointShard {
            shard_id: summary.shard_id,
            artifact: summary.state_hash,
            generation: summary.generation,
            providers: summary.providers.clone(),
        })
        .collect::<Vec<_>>();
    if shards.is_empty() {
        return Err(NodeError::InvalidConfig(
            "cannot commit an autonomous checkpoint without shard state".to_string(),
        ));
    }
    let generation = shards
        .iter()
        .map(|shard| shard.generation)
        .max()
        .unwrap_or(0);
    let mut manifest = TrainingCheckpointManifest {
        job_id: context.start.job_id,
        model_generation: generation,
        optimizer_generation: generation,
        membership_epoch: context.start.membership_epoch,
        term: context.term,
        checkpoint_generation: context.checkpoint_generation.saturating_add(1),
        parent: context.checkpoint,
        dataset_progress: shards
            .iter()
            .map(|shard| (shard.shard_id, shard.generation))
            .collect(),
        shards,
        hash: ArtifactId::default(),
    };
    manifest.hash = checkpoint_hash(&manifest);
    Ok(manifest)
}

fn make_checkpoint(
    context: &Context,
    aggregates: &[TrainingAggregate],
) -> Result<TrainingCheckpointManifest, NodeError> {
    let shards = aggregates
        .iter()
        .map(|aggregate| TrainingCheckpointShard {
            shard_id: aggregate.shard_id,
            artifact: aggregate.state_artifact,
            generation: aggregate.generation,
            providers: providers(aggregate),
        })
        .collect::<Vec<_>>();
    let mut manifest = TrainingCheckpointManifest {
        job_id: context.start.job_id,
        model_generation: aggregates
            .iter()
            .map(|aggregate| aggregate.generation)
            .max()
            .unwrap_or(0),
        optimizer_generation: aggregates
            .iter()
            .map(|aggregate| aggregate.generation)
            .max()
            .unwrap_or(0),
        membership_epoch: context.start.membership_epoch,
        term: context.term,
        checkpoint_generation: context.checkpoint_generation.saturating_add(1),
        parent: context.checkpoint,
        shards,
        dataset_progress: aggregates
            .iter()
            .map(|aggregate| (aggregate.shard_id, aggregate.window))
            .collect(),
        hash: ArtifactId::default(),
    };
    manifest.hash = checkpoint_hash(&manifest);
    Ok(manifest)
}

fn providers(aggregate: &TrainingAggregate) -> Vec<NodeId> {
    let mut providers = vec![aggregate.aggregator];
    for provider in &aggregate.replicas {
        if !providers.contains(provider) {
            providers.push(*provider);
        }
    }
    providers
}

fn persist_state(node: &Arc<Node>, state: &TrainingState) -> Result<(), NodeError> {
    node.store
        .write_json(&format!("training-v3-{}.json", state.job_id), state)?;
    Ok(())
}

fn persist_start(node: &Arc<Node>, start: &TrainingStart) -> Result<(), NodeError> {
    node.store
        .write_json(&format!("training-v3-start-{}.json", start.job_id), start)?;
    Ok(())
}

fn persist_shard_state(node: &Arc<Node>, state: &TrainingShardState) -> Result<(), NodeError> {
    node.store.write_json(
        &format!("training-v3-shard-{}-{}.json", state.job_id, state.shard_id),
        state,
    )?;
    Ok(())
}

pub(crate) async fn restore_workers(
    node: Arc<Node>,
    starts: Vec<TrainingStart>,
) -> Result<(), NodeError> {
    for start in starts {
        if start.coordinator == node.node_id()
            || !start.participants.contains(&node.node_id())
            || !start.assignment.owners.contains(&node.node_id())
        {
            continue;
        }
        if Message::Training(TrainingMessage::Start(start.clone()))
            .validate()
            .is_err()
        {
            tracing::warn!(job_id = %start.job_id, "ignoring invalid persisted V3 training start");
            continue;
        }
        if node.v3_jobs.lock().await.contains_key(&start.job_id) {
            continue;
        }
        let persisted_state = node.v3_states.lock().await.get(&start.job_id).cloned();
        let persisted_final_checkpoint = persisted_state.as_ref().and_then(|state| {
            state.checkpoint.and_then(|checkpoint| {
                let name = format!("training-v3-checkpoint-{checkpoint}.json");
                node.store
                    .read_json::<TrainingCheckpointManifest>(&name)
                    .ok()
                    .flatten()
                    .filter(|manifest| {
                        manifest.hash == checkpoint && checkpoint_hash(manifest) == checkpoint
                    })
            })
        });
        if start.execution == TrainingExecution::AutonomousLocalSgd
            && persisted_state
                .as_ref()
                .is_some_and(|state| state_covers_final_model(&start, state))
            && persisted_final_checkpoint.is_some()
        {
            // A cleanly completed autonomous job remains on disk for status
            // and recovery inspection, but must not be revived as a live
            // mailbox when the worker process restarts later.
            continue;
        }
        let (sender, receiver) = mpsc::channel(512);
        node.v3_jobs.lock().await.insert(
            start.job_id,
            V3JobHandle {
                sender: sender.clone(),
            },
        );
        let mut context = Context::new(start.clone(), Role::Worker, receiver, None);
        if let Some(state) = persisted_state {
            context.coordinator = state.coordinator;
            context.term = state.term;
            context.window = state.window;
            context.checkpoint_generation = state.checkpoint_generation;
            context.checkpoint = state.checkpoint;
            for shard in state.shards {
                context.shard_summaries.insert(shard.shard_id, shard);
            }
        }
        let shard_name = format!(
            "training-v3-shard-{}-{}.json",
            start.job_id, start.assignment.shard_id
        );
        if let Some(shard) = node.store.read_json::<TrainingShardState>(&shard_name)?
            && shard.job_id == start.job_id
            && shard.shard_id == start.assignment.shard_id
            && shard.branch == start.branch
            && state_hash(
                shard.job_id,
                shard.shard_id,
                shard.generation,
                shard.value,
                shard.optimizer,
            ) == shard.state_hash
        {
            context.value = shard.value;
            context.optimizer = shard.optimizer;
            context.generation = shard.generation;
        }
        let restore_sender = sender.clone();
        let restore_coordinator = context.coordinator;
        let restore_node = node.clone();
        tokio::spawn(async move {
            sleep(Duration::from_secs(1)).await;
            if !restore_node.network.is_connected(restore_coordinator).await {
                let _ = restore_sender
                    .send(V3Inbound::PeerDisconnected(restore_coordinator))
                    .await;
            }
        });
        tokio::spawn(run_context(node.clone(), context));
    }
    Ok(())
}

fn persist_checkpoint(
    node: &Arc<Node>,
    commit: &TrainingCheckpointCommit,
) -> Result<(), NodeError> {
    node.store.write_json(
        &format!("training-v3-checkpoint-{}.json", commit.manifest.hash),
        &commit.manifest,
    )?;
    Ok(())
}

fn samples_loss(value: i64, samples: &[TrainingSample]) -> i64 {
    if samples.is_empty() {
        return 0;
    }
    samples
        .iter()
        .map(|sample| {
            let error = value
                .saturating_mul(sample.feature)
                .saturating_sub(sample.target);
            error.saturating_mul(error)
        })
        .sum::<i64>()
        / samples.len() as i64
}

fn median(mut values: Vec<i64>) -> i64 {
    values.sort_unstable();
    values[values.len() / 2]
}
