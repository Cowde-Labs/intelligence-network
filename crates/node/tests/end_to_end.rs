use intelligence_node::{
    AdminRequest, CapabilityConfig, Node, NodeConfig, StorageConfig, admin_call,
};
use intelligence_protocol::{JobId, JobState, NodeId, TrainingStart};
use intelligence_storage::{LocalStore, PersistedJob};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    net::{SocketAddr, UdpSocket},
    path::PathBuf,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::time::{Duration, Instant, sleep, timeout};

#[cfg(unix)]
fn require_bubblewrap() {
    // Mirror the runtime's namespace and mount table so a host that cannot
    // run the real sandbox fails here with a direct message instead of as
    // a job that never starts.
    let mut command = std::process::Command::new("bwrap");
    command.args([
        "--die-with-parent",
        "--unshare-all",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
    ]);
    for library_root in ["/lib", "/lib64"] {
        if std::path::Path::new(library_root).exists() {
            command.args(["--ro-bind", library_root, library_root]);
        }
    }
    match command.args(["--", "/bin/sh", "-c", "true"]).output() {
        Ok(output) if output.status.success() => {}
        outcome => panic!(
            "these tests execute jobs under bubblewrap; `bwrap` must be installed and permitted to create user namespaces: {outcome:?}"
        ),
    }
}

fn free_addr() -> SocketAddr {
    static USED_PORTS: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    let used_ports = USED_PORTS.get_or_init(|| Mutex::new(HashSet::new()));
    for _ in 0..1_000 {
        // Nodes bind QUIC over UDP; probe the same protocol so a TCP-free
        // port that is still held by another UDP socket is never handed out.
        let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
        let address = socket.local_addr().unwrap();
        if used_ports.lock().unwrap().insert(address.port()) {
            return address;
        }
    }
    panic!("unable to allocate an isolated test port")
}

fn root(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "intelligence-node-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn builtin_capability() -> CapabilityConfig {
    builtin_capability_named("inference.text")
}

fn builtin_capability_named(name: &str) -> CapabilityConfig {
    CapabilityConfig {
        name: name.to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "builtin_text".to_string(),
        model: Some("builtin.tiny-sentiment.v1".to_string()),
        model_path: None,
        program: None,
        args: Vec::new(),
        env: BTreeMap::new(),
        metadata: BTreeMap::new(),
        sandbox: "trusted_local".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 1000,
    }
}

fn training_capability() -> CapabilityConfig {
    CapabilityConfig {
        name: "training.reference".to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "builtin_training".to_string(),
        model: None,
        model_path: None,
        program: None,
        args: Vec::new(),
        env: BTreeMap::new(),
        metadata: BTreeMap::new(),
        sandbox: "trusted_local".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 1000,
    }
}

#[cfg(unix)]
fn slow_process_capability() -> CapabilityConfig {
    CapabilityConfig {
        name: "inference.text".to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "process".to_string(),
        model: None,
        model_path: None,
        program: Some(PathBuf::from("/bin/sh")),
        args: vec!["-c".to_string(), "sleep 5".to_string()],
        env: BTreeMap::new(),
        metadata: BTreeMap::new(),
        sandbox: "bubblewrap".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 1000,
    }
}

#[cfg(unix)]
fn counted_process_capability() -> CapabilityConfig {
    CapabilityConfig {
        name: "inference.text".to_string(),
        version: 1,
        public: true,
        accept_remote_jobs: true,
        kind: "process".to_string(),
        model: None,
        model_path: None,
        program: Some(PathBuf::from("/bin/sh")),
        args: vec!["-c".to_string(), "sleep 1; printf '{}'".to_string()],
        env: BTreeMap::new(),
        metadata: BTreeMap::new(),
        sandbox: "bubblewrap".to_string(),
        max_input_bytes: 64 * 1024,
        max_output_bytes: 16 * 1024,
        memory_bytes: 64 * 1024 * 1024,
        cpu_millis: 5000,
    }
}

fn config(
    root: PathBuf,
    address: SocketAddr,
    bootstrap: Vec<String>,
    capabilities: Vec<CapabilityConfig>,
) -> NodeConfig {
    NodeConfig {
        data_dir: root,
        identity_path: None,
        admin_socket: None,
        listen_addr: address,
        advertise_addr: Some(address.to_string()),
        bootstrap,
        relay_addresses: Vec::new(),
        relay_enabled: false,
        relay_max_sessions: 64,
        relay_max_bytes: 64 * 1024 * 1024,
        max_connections: 8,
        max_frame_size: intelligence_protocol::MAX_FRAME_SIZE,
        peer_ttl_seconds: 300,
        allow_private_addresses: true,
        prefer_relay: false,
        hole_punch_enabled: true,
        hole_punch_max_attempts: 4,
        dht_enabled: true,
        dht_k: 20,
        dht_alpha: 3,
        dht_max_records: 2_048,
        // The V3 fixture advertises a 2 KiB complete model and 1 KiB
        // assigned shard.  Keeping the worker budget at 1.5 KiB makes the
        // model-fit acceptance condition observable in every real-process
        // training test without allocating a large model.
        training_memory_bytes: 1536,
        training_window_delay_ms: 0,
        storage: StorageConfig {
            quota_bytes: 32 * 1024 * 1024,
            max_artifact_bytes: 4 * 1024 * 1024,
        },
        runtime: intelligence_runtime::RuntimeConfig::default(),
        capabilities,
    }
}

async fn wait_for_connected(node: &Node, expected: usize) {
    for _ in 0..800 {
        if node
            .status()
            .await
            .ok()
            .and_then(|status| {
                status
                    .get("connected_peers")
                    .and_then(serde_json::Value::as_u64)
            })
            .is_some_and(|count| count >= expected as u64)
        {
            return;
        }
        sleep(Duration::from_millis(25)).await;
    }
    panic!("node did not reach {expected} connected peers");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn routes_real_inference_over_the_peer_protocol() {
    let root_a = root("a");
    let root_b = root("b");
    let address_a = free_addr();
    let address_b = free_addr();
    let node_a = Node::start(config(
        root_a.clone(),
        address_a,
        vec![address_b.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    let node_b = Node::start(config(
        root_b.clone(),
        address_b,
        Vec::new(),
        vec![builtin_capability()],
    ))
    .await
    .unwrap();

    let mut status = serde_json::json!({});
    for _ in 0..100 {
        if let Ok(value) = admin_call(node_a.admin_socket(), &AdminRequest::Status).await {
            status = value;
            if status["connected_peers"] == 1 {
                break;
            }
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(status["connected_peers"], 1);
    let result = admin_call(
        node_a.admin_socket(),
        &AdminRequest::Infer {
            capability: "inference.text".to_string(),
            input: "This is good and useful".to_string(),
            deadline_ms: Some(5000),
            max_output_bytes: Some(16 * 1024),
            allow_input_transfer: Some(true),
            job_id: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(result["state"], "Succeeded");
    assert!(result["output"].is_array());
    let trust = admin_call(
        node_a.admin_socket(),
        &AdminRequest::Trust {
            subject: node_b.node_id().to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(trust["decision"]["direct_successes"], 1);
    assert_eq!(trust["decision"]["maturity"], "Observed");

    node_a.shutdown().await;
    node_b.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_a);
    let _ = fs::remove_dir_all(root_b);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dht_provider_lookup_survives_bootstrap_shutdown() {
    let root_bootstrap = root("dht-bootstrap");
    let root_provider = root("dht-provider");
    let root_requester = root("dht-requester");
    let address_bootstrap = free_addr();
    let address_provider = free_addr();
    let address_requester = free_addr();

    let bootstrap = Node::start(config(
        root_bootstrap.clone(),
        address_bootstrap,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let provider = Node::start(config(
        root_provider.clone(),
        address_provider,
        vec![address_bootstrap.to_string()],
        vec![builtin_capability()],
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_bootstrap.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();

    let mut provider_status = serde_json::json!({});
    for _ in 0..100 {
        provider_status = admin_call(provider.admin_socket(), &AdminRequest::Status)
            .await
            .unwrap_or_default();
        if provider_status["connected_peers"] == 1 {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(provider_status["connected_peers"], 1);

    let published = admin_call(
        provider.admin_socket(),
        &AdminRequest::DhtPublish {
            namespace: "capability".to_string(),
            name: "inference.text".to_string(),
            value: serde_json::to_string(&serde_json::json!({
                "provider": provider.node_id().to_string(),
                "evidence": "claimed"
            }))
            .unwrap(),
            ttl_seconds: Some(300),
            sequence: Some(1),
        },
    )
    .await
    .unwrap();
    let provider_id = serde_json::to_value(provider.node_id()).unwrap();
    assert_eq!(published["owner"], provider_id);

    let mut records = Vec::new();
    for _ in 0..120 {
        records = admin_call(
            requester.admin_socket(),
            &AdminRequest::DhtLookup {
                namespace: "capability".to_string(),
                name: "inference.text".to_string(),
            },
        )
        .await
        .unwrap_or_default()
        .as_array()
        .cloned()
        .unwrap_or_default();
        if records.iter().any(|record| record["owner"] == provider_id) {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert!(records.iter().any(|record| record["owner"] == provider_id));

    bootstrap.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let after_bootstrap = admin_call(
        requester.admin_socket(),
        &AdminRequest::DhtLookup {
            namespace: "capability".to_string(),
            name: "inference.text".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(
        after_bootstrap
            .as_array()
            .is_some_and(|items| { items.iter().any(|record| record["owner"] == provider_id) })
    );
    let status = admin_call(requester.admin_socket(), &AdminRequest::DhtStats)
        .await
        .unwrap();
    assert!(status["lookup_successes"].as_u64().unwrap_or(0) > 0);

    provider.shutdown().await;
    requester.shutdown().await;
    let _ = fs::remove_dir_all(root_bootstrap);
    let _ = fs::remove_dir_all(root_provider);
    let _ = fs::remove_dir_all(root_requester);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn learned_peer_continues_work_after_bootstrap_shutdown() {
    let root_bootstrap = root("bootstrap");
    let root_requester = root("requester");
    let root_worker = root("worker");
    let address_bootstrap = free_addr();
    let address_requester = free_addr();
    let address_worker = free_addr();
    let bootstrap = Node::start(config(
        root_bootstrap.clone(),
        address_bootstrap,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let worker = Node::start(config(
        root_worker.clone(),
        address_worker,
        vec![address_bootstrap.to_string()],
        vec![builtin_capability()],
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_bootstrap.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();

    let requester_id = serde_json::to_value(requester.node_id()).unwrap();
    let worker_id = serde_json::to_value(worker.node_id()).unwrap();
    let mut direct_requester_worker = false;
    for _ in 0..160 {
        let requester_peers = admin_call(requester.admin_socket(), &AdminRequest::Peers).await;
        let worker_peers = admin_call(worker.admin_socket(), &AdminRequest::Peers).await;
        if requester_peers
            .as_ref()
            .ok()
            .and_then(|value| value.as_array())
            .is_some_and(|peers| peers.iter().any(|peer| peer["node_id"] == worker_id))
            && worker_peers
                .as_ref()
                .ok()
                .and_then(|value| value.as_array())
                .is_some_and(|peers| peers.iter().any(|peer| peer["node_id"] == requester_id))
        {
            direct_requester_worker = true;
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert!(
        direct_requester_worker,
        "learned peer records were not exchanged"
    );
    bootstrap.shutdown().await;
    sleep(Duration::from_millis(500)).await;

    let result = admin_call(
        requester.admin_socket(),
        &AdminRequest::Infer {
            capability: "inference.text".to_string(),
            input: "good after bootstrap outage".to_string(),
            deadline_ms: Some(5000),
            max_output_bytes: Some(16 * 1024),
            allow_input_transfer: Some(true),
            job_id: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(result["state"], "Succeeded");

    requester.shutdown().await;
    worker.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_bootstrap);
    let _ = fs::remove_dir_all(root_requester);
    let _ = fs::remove_dir_all(root_worker);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn remote_evaluation_returns_local_evidence() {
    let root_requester = root("evaluation-requester");
    let root_worker = root("evaluation-worker");
    let address_requester = free_addr();
    let address_worker = free_addr();
    let worker = Node::start(config(
        root_worker.clone(),
        address_worker,
        Vec::new(),
        vec![builtin_capability_named("evaluation.text")],
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_worker.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    wait_for_connected(&requester, 1).await;
    let result = admin_call(
        requester.admin_socket(),
        &AdminRequest::Evaluate {
            text: "good and useful".to_string(),
            expected_label: "positive".to_string(),
            deadline_ms: Some(5000),
        },
    )
    .await
    .unwrap();
    assert_eq!(result["state"], "Succeeded");
    assert_eq!(result["evidence"]["score"], 100);

    requester.shutdown().await;
    worker.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_requester);
    let _ = fs::remove_dir_all(root_worker);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn remote_job_can_be_cancelled_by_its_known_id() {
    require_bubblewrap();
    let root_requester = root("cancel-requester");
    let root_worker = root("cancel-worker");
    let address_requester = free_addr();
    let address_worker = free_addr();
    let worker = Node::start(config(
        root_worker.clone(),
        address_worker,
        Vec::new(),
        vec![slow_process_capability()],
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_worker.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    let job_id = "06060606060606060606060606060606".to_string();
    wait_for_connected(&requester, 1).await;
    let requester_socket = requester.admin_socket().to_path_buf();
    let request = AdminRequest::Infer {
        capability: "inference.text".to_string(),
        input: "cancel me".to_string(),
        deadline_ms: Some(5000),
        max_output_bytes: Some(16 * 1024),
        allow_input_transfer: Some(true),
        job_id: Some(job_id.clone()),
    };
    let pending = tokio::spawn(async move { admin_call(requester_socket, &request).await });
    sleep(Duration::from_millis(1_000)).await;
    let cancelled = admin_call(requester.admin_socket(), &AdminRequest::Cancel { job_id })
        .await
        .unwrap();
    assert_eq!(cancelled["cancelled"], true, "cancel response: {cancelled}");
    let result = timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result["state"], "Cancelled");

    requester.shutdown().await;
    worker.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_requester);
    let _ = fs::remove_dir_all(root_worker);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn duplicate_inflight_job_is_executed_at_most_once() {
    require_bubblewrap();
    let root_requester = root("duplicate-requester");
    let root_worker = root("duplicate-worker");
    let address_requester = free_addr();
    let address_worker = free_addr();
    let mut worker_config = config(
        root_worker.clone(),
        address_worker,
        Vec::new(),
        vec![counted_process_capability()],
    );
    worker_config.runtime.max_concurrent_jobs = 1;
    worker_config.runtime.max_queued_jobs = 2;
    let worker = Node::start(worker_config).await.unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_worker.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    let job_id = "07070707070707070707070707070707".to_string();
    wait_for_connected(&requester, 1).await;
    let request = AdminRequest::Infer {
        capability: "inference.text".to_string(),
        input: "execute once".to_string(),
        deadline_ms: Some(5000),
        max_output_bytes: Some(16 * 1024),
        allow_input_transfer: Some(true),
        job_id: Some(job_id),
    };
    let first_socket = requester.admin_socket().to_path_buf();
    let second_socket = requester.admin_socket().to_path_buf();
    let started_at = Instant::now();
    let first = tokio::spawn(async move { admin_call(first_socket, &request).await });
    sleep(Duration::from_millis(100)).await;
    let second_request = AdminRequest::Infer {
        capability: "inference.text".to_string(),
        input: "execute once".to_string(),
        deadline_ms: Some(5000),
        max_output_bytes: Some(16 * 1024),
        allow_input_transfer: Some(true),
        job_id: Some("07070707070707070707070707070707".to_string()),
    };
    let second = tokio::spawn(async move { admin_call(second_socket, &second_request).await });
    let first_result = timeout(Duration::from_secs(8), first)
        .await
        .unwrap()
        .unwrap();
    let second_result = timeout(Duration::from_secs(8), second)
        .await
        .unwrap()
        .unwrap();
    assert!(
        first_result
            .as_ref()
            .is_ok_and(|value| value["state"] == "Succeeded")
            || second_result
                .as_ref()
                .is_ok_and(|value| value["state"] == "Succeeded"),
        "neither duplicate submission succeeded: first={first_result:?} second={second_result:?}"
    );
    assert!(started_at.elapsed() < Duration::from_millis(1800));

    requester.shutdown().await;
    worker.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_requester);
    let _ = fs::remove_dir_all(root_worker);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn reference_training_uses_two_remote_workers_and_improves_loss() {
    let root_a = root("training-a");
    let root_b = root("training-b");
    let root_c = root("training-c");
    let address_a = free_addr();
    let address_b = free_addr();
    let address_c = free_addr();
    let node_a = Node::start(config(root_a.clone(), address_a, Vec::new(), Vec::new()))
        .await
        .unwrap();
    let node_b = Node::start(config(
        root_b.clone(),
        address_b,
        vec![address_a.to_string()],
        vec![training_capability()],
    ))
    .await
    .unwrap();
    let node_c = Node::start(config(
        root_c.clone(),
        address_c,
        vec![address_a.to_string()],
        vec![training_capability()],
    ))
    .await
    .unwrap();
    for _ in 0..100 {
        if node_a
            .status()
            .await
            .unwrap()
            .get("known_peers")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count >= 2)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let value = admin_call(
        node_a.admin_socket(),
        &AdminRequest::PlanTraining {
            model_bytes: 256,
            data_locality: "selective".to_string(),
            workers: Some(2),
        },
    )
    .await
    .unwrap();
    assert_eq!(value["evidence_class"], "REAL_PROCESS_LOCAL");
    assert_eq!(
        value["decision"]["selected_workers"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let value = admin_call(
        node_a.admin_socket(),
        &AdminRequest::TrainReference {
            workers: Some(2),
            steps: Some(8),
            resume_checkpoint: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(value["kind"], "v1_distributed_training_reference");
    assert_eq!(value["improved"], true);
    assert_eq!(value["checkpoints"].as_array().unwrap().len(), 8);
    let checkpoint = value["checkpoints"].as_array().unwrap().last().unwrap();
    let checkpoint = checkpoint.as_str().unwrap().to_string();
    // The coordinator role is recoverable across a process restart. This is
    // intentionally narrower than an automatic election: the operator (or a
    // higher-level scheduler) resumes from the verified checkpoint.
    node_a.shutdown().await;
    drop(node_a);
    sleep(Duration::from_millis(100)).await;
    let node_a = Node::start(config(root_a.clone(), address_a, Vec::new(), Vec::new()))
        .await
        .unwrap();
    for _ in 0..100 {
        if node_a
            .status()
            .await
            .unwrap()
            .get("connected_peers")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count >= 2)
        {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    let resumed = admin_call(
        node_a.admin_socket(),
        &AdminRequest::TrainReference {
            workers: Some(2),
            steps: Some(2),
            resume_checkpoint: Some(checkpoint),
        },
    )
    .await
    .unwrap();
    assert_eq!(resumed["improved"], true);
    assert!(resumed["resumed_from"].as_str().is_some());
    assert_eq!(resumed["start_step"], 9);
    node_a.shutdown().await;
    node_b.shutdown().await;
    node_c.shutdown().await;
    let _ = fs::remove_dir_all(root_a);
    let _ = fs::remove_dir_all(root_b);
    let _ = fs::remove_dir_all(root_c);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v3_training_uses_group_updates_and_replicated_shards() {
    let coordinator_root = root("v3-coordinator");
    let worker_roots = (0..4)
        .map(|index| root(&format!("v3-worker-{index}")))
        .collect::<Vec<_>>();
    let coordinator_address = free_addr();
    let worker_addresses = (0..4).map(|_| free_addr()).collect::<Vec<_>>();
    let all_addresses = worker_addresses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let coordinator = Node::start(config(
        coordinator_root.clone(),
        coordinator_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let mut workers = Vec::new();
    for (index, worker_root) in worker_roots.iter().enumerate() {
        let mut bootstrap = vec![coordinator_address.to_string()];
        bootstrap.extend(
            all_addresses
                .iter()
                .filter(|address| **address != worker_addresses[index].to_string())
                .cloned(),
        );
        workers.push(
            Node::start(config(
                worker_root.clone(),
                worker_addresses[index],
                bootstrap,
                vec![training_capability()],
            ))
            .await
            .unwrap(),
        );
        assert!(index < 4);
    }
    for _ in 0..200 {
        let status = coordinator.status().await.unwrap();
        if status["known_peers"].as_u64().unwrap_or_default() >= 4 {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }
    sleep(Duration::from_secs(2)).await;
    let result = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::TrainV3 {
            workers: Some(4),
            windows: Some(4),
            local_steps: Some(3),
            checkpoint_every: Some(2),
        },
    )
    .await
    .unwrap();
    assert_eq!(result["kind"], "v3_distributed_training_reference");
    assert_eq!(result["global_step_barrier"], false);
    assert_eq!(result["all_updates_to_one_coordinator"], false);
    assert_eq!(result["single_optimizer_authority"], false);
    assert_eq!(result["single_checkpoint_authority"], false);
    assert_eq!(result["model_must_fit_one_worker"], false);
    assert_eq!(result["full_model_materialized_on_worker"], false);
    assert_eq!(result["coordinator_update_fanin"], 0);
    assert!(result["maximum_group_fan_in"].as_u64().unwrap_or_default() <= 2);
    assert!(
        result["optimizer_state_replication_factor"]
            .as_u64()
            .unwrap_or_default()
            >= 2
    );
    assert_eq!(result["windows"], 4);
    assert!(result["checkpoint_generation"].as_u64().unwrap_or_default() >= 2);
    assert_eq!(result["final_checkpoint_committed"], true);
    assert!(
        result["initial_loss"].as_i64().unwrap_or_default()
            > result["final_loss"].as_i64().unwrap_or(i64::MAX)
    );
    assert_eq!(result["final_loss"], 0);
    assert!(
        result["checkpoints"]
            .as_array()
            .is_some_and(|items| !items.is_empty())
    );
    let coordinator_status = coordinator.status().await.unwrap();
    assert_eq!(coordinator_status["training_updates_received"], 0);
    assert!(
        coordinator_status["training_aggregates_received"]
            .as_u64()
            .unwrap_or_default()
            >= 8
    );
    assert!(
        coordinator_status["training_max_update_fanin"]
            .as_u64()
            .unwrap_or_default()
            == 0
    );

    let restarted_root = worker_roots[0].clone();
    let restarted_address = worker_addresses[0];
    let stopped_worker = workers.remove(0);
    stopped_worker.shutdown().await;
    drop(stopped_worker);
    sleep(Duration::from_millis(100)).await;
    let restarted_worker = Node::start(config(
        restarted_root,
        restarted_address,
        vec![coordinator_address.to_string()],
        vec![training_capability()],
    ))
    .await
    .unwrap();
    let restored = admin_call(
        restarted_worker.admin_socket(),
        &AdminRequest::TrainingStatus {
            job_id: result["job_id"].as_str().unwrap().to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(restored["window"], 4);
    assert_eq!(restored["role"], "worker");
    assert_eq!(restored["model_shard_count"], 2);
    assert_eq!(restored["materialized_shard_count"], 1);
    assert_eq!(restored["full_model_materialized_on_worker"], false);
    assert_eq!(restored["model_state_bytes"], 2048);
    assert_eq!(restored["assigned_shard_state_bytes"], 1024);
    assert_eq!(restored["training_memory_bytes"], 1536);
    assert_eq!(restored["model_requires_sharding"], true);
    assert!(
        restored["checkpoint_generation"]
            .as_u64()
            .unwrap_or_default()
            >= 2
    );
    restarted_worker.shutdown().await;

    coordinator.shutdown().await;
    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(coordinator_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v3_training_is_not_globally_barriered_by_a_slow_worker() {
    let coordinator_root = root("v3-straggler-coordinator");
    let worker_roots = (0..4)
        .map(|index| root(&format!("v3-straggler-worker-{index}")))
        .collect::<Vec<_>>();
    let coordinator_address = free_addr();
    let worker_addresses = (0..4).map(|_| free_addr()).collect::<Vec<_>>();
    let all_addresses = worker_addresses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let coordinator = Node::start(config(
        coordinator_root.clone(),
        coordinator_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let mut workers = Vec::new();
    for (index, worker_root) in worker_roots.iter().enumerate() {
        let mut bootstrap = vec![coordinator_address.to_string()];
        bootstrap.extend(
            all_addresses
                .iter()
                .filter(|address| **address != worker_addresses[index].to_string())
                .cloned(),
        );
        let mut worker_config = config(
            worker_root.clone(),
            worker_addresses[index],
            bootstrap,
            vec![training_capability()],
        );
        if index == 0 {
            // V3's 750 ms group collection timeout deliberately bounds this
            // straggler. The other group must advance without waiting for it.
            worker_config.training_window_delay_ms = 1_200;
        }
        workers.push(Node::start(worker_config).await.unwrap());
    }
    for _ in 0..200 {
        if coordinator.status().await.unwrap()["known_peers"]
            .as_u64()
            .unwrap_or_default()
            >= 4
        {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }
    sleep(Duration::from_millis(200)).await;

    let result = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::TrainV3 {
            workers: Some(4),
            windows: Some(3),
            local_steps: Some(2),
            checkpoint_every: Some(3),
        },
    )
    .await
    .unwrap();
    assert_eq!(result["global_step_barrier"], false);
    assert_eq!(result["non_barrier_progress_observed"], true);
    assert_eq!(
        result["group_progress"]
            .as_object()
            .map(|value| value.len()),
        Some(2)
    );
    assert_eq!(
        result["group_completion_order"]
            .as_array()
            .map(|value| value.len()),
        Some(2)
    );
    assert!(
        result["final_loss"].as_i64().unwrap_or(i64::MAX)
            < result["initial_loss"].as_i64().unwrap_or(i64::MIN)
    );

    coordinator.shutdown().await;
    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(coordinator_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v3_training_replaces_a_permanently_stopped_coordinator() {
    let coordinator_root = root("v3-election-coordinator");
    let worker_roots = (0..4)
        .map(|index| root(&format!("v3-election-worker-{index}")))
        .collect::<Vec<_>>();
    let coordinator_address = free_addr();
    let worker_addresses = (0..4).map(|_| free_addr()).collect::<Vec<_>>();
    let all_addresses = worker_addresses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let coordinator = Node::start(config(
        coordinator_root.clone(),
        coordinator_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let mut workers = Vec::new();
    for (index, worker_root) in worker_roots.iter().enumerate() {
        let mut bootstrap = vec![coordinator_address.to_string()];
        bootstrap.extend(
            all_addresses
                .iter()
                .filter(|address| **address != worker_addresses[index].to_string())
                .cloned(),
        );
        workers.push(
            Node::start(config(
                worker_root.clone(),
                worker_addresses[index],
                bootstrap,
                vec![training_capability()],
            ))
            .await
            .unwrap(),
        );
    }
    for _ in 0..200 {
        if coordinator.status().await.unwrap()["known_peers"]
            .as_u64()
            .unwrap_or_default()
            >= 4
        {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }
    sleep(Duration::from_millis(200)).await;
    let started = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::TrainV3Start {
            workers: Some(4),
            windows: Some(256),
            local_steps: Some(16),
            checkpoint_every: Some(16),
        },
    )
    .await
    .unwrap();
    let job_id = started["job_id"].as_str().unwrap().to_string();
    let original = coordinator.node_id().to_string();
    let mut worker_started = false;
    for _ in 0..100 {
        for worker in &workers {
            if admin_call(
                worker.admin_socket(),
                &AdminRequest::TrainingStatus {
                    job_id: job_id.clone(),
                },
            )
            .await
            .is_ok()
            {
                worker_started = true;
                break;
            }
        }
        if worker_started {
            break;
        }
        sleep(Duration::from_millis(10)).await;
    }
    assert!(
        worker_started,
        "at least one worker must accept the training start"
    );
    sleep(Duration::from_millis(100)).await;
    coordinator.shutdown().await;
    // Wait until the surviving workers have observed the transport failure.
    // The election is driven by an authenticated PeerDisconnected event; the
    // assertion must not race endpoint shutdown with delivery of that event.
    for _ in 0..100 {
        let mut observed = false;
        for worker in &workers {
            if worker
                .status()
                .await
                .ok()
                .and_then(|status| status["connected_peers"].as_u64())
                .is_some_and(|peers| peers < 4)
            {
                observed = true;
                break;
            }
        }
        if observed {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }

    let mut replacement = None;
    for _ in 0..240 {
        for worker in &workers {
            if let Ok(state) = admin_call(
                worker.admin_socket(),
                &AdminRequest::TrainingStatus {
                    job_id: job_id.clone(),
                },
            )
            .await
                && state["term"].as_u64().unwrap_or_default() >= 2
                && state["coordinator"].as_str() != Some(original.as_str())
                && state["window"].as_u64().unwrap_or_default() > 0
            {
                replacement = Some(state);
                break;
            }
        }
        if replacement.is_some() {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    let replacement = match replacement {
        Some(state) => state,
        None => {
            for worker in &workers {
                let status = worker.status().await.unwrap_or_default();
                let training = admin_call(
                    worker.admin_socket(),
                    &AdminRequest::TrainingStatus {
                        job_id: job_id.clone(),
                    },
                )
                .await
                .unwrap_or_else(|error| serde_json::json!({"error": error.to_string()}));
                eprintln!("replacement diagnostic status={status} training={training}");
            }
            panic!("a surviving worker must replace the stopped coordinator");
        }
    };
    assert!(replacement["term"].as_u64().unwrap_or_default() >= 2);

    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(coordinator_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v3_training_recovers_from_optimizer_owner_failure() {
    let coordinator_root = root("v3-optimizer-owner-coordinator");
    let worker_roots = (0..4)
        .map(|index| root(&format!("v3-optimizer-owner-worker-{index}")))
        .collect::<Vec<_>>();
    let coordinator_address = free_addr();
    let worker_addresses = (0..4).map(|_| free_addr()).collect::<Vec<_>>();
    let all_addresses = worker_addresses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let coordinator = Node::start(config(
        coordinator_root.clone(),
        coordinator_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let mut workers = Vec::new();
    for (index, worker_root) in worker_roots.iter().enumerate() {
        let mut bootstrap = vec![coordinator_address.to_string()];
        bootstrap.extend(
            all_addresses
                .iter()
                .filter(|address| **address != worker_addresses[index].to_string())
                .cloned(),
        );
        workers.push(
            Node::start(config(
                worker_root.clone(),
                worker_addresses[index],
                bootstrap,
                vec![training_capability()],
            ))
            .await
            .unwrap(),
        );
    }
    for _ in 0..200 {
        if coordinator.status().await.unwrap()["known_peers"]
            .as_u64()
            .unwrap_or_default()
            >= 4
        {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }

    let started = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::TrainV3Start {
            workers: Some(4),
            windows: Some(8),
            local_steps: Some(2),
            checkpoint_every: Some(2),
        },
    )
    .await
    .unwrap();
    let job_id = started["job_id"].as_str().unwrap().to_string();
    let start_path = coordinator_root
        .join("state")
        .join(format!("training-v3-start-{job_id}.json"));
    let mut training_start = None;
    for _ in 0..100 {
        if let Ok(bytes) = fs::read(&start_path) {
            training_start = Some(serde_json::from_slice::<TrainingStart>(&bytes).unwrap());
            break;
        }
        sleep(Duration::from_millis(10)).await;
    }
    let training_start =
        training_start.expect("training plan must be durable before failure injection");
    let owner_index = workers
        .iter()
        .position(|worker| worker.node_id() == training_start.groups[0].aggregator)
        .expect("optimizer owner must be one of the training workers");
    let mut owner_started = false;
    for _ in 0..250 {
        if admin_call(
            workers[owner_index].admin_socket(),
            &AdminRequest::TrainingStatus {
                job_id: job_id.clone(),
            },
        )
        .await
        .ok()
        .and_then(|state| state["window"].as_u64())
        .is_some_and(|window| window >= 1)
        {
            owner_started = true;
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(
        owner_started,
        "optimizer owner must process a window before failure injection"
    );
    let stopped_owner = workers.remove(owner_index);
    stopped_owner.shutdown().await;
    drop(stopped_owner);
    for _ in 0..100 {
        if coordinator
            .status()
            .await
            .ok()
            .and_then(|status| status["connected_peers"].as_u64())
            .is_some_and(|peers| peers < 4)
        {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }

    let mut recovered = None;
    for _ in 0..300 {
        if let Ok(state) = admin_call(
            coordinator.admin_socket(),
            &AdminRequest::TrainingStatus {
                job_id: job_id.clone(),
            },
        )
        .await
            && state["window"].as_u64().unwrap_or_default() >= 8
            && state["checkpoint_generation"].as_u64().unwrap_or_default() >= 1
        {
            recovered = Some(state);
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    let _recovered = recovered.expect("training must recover after optimizer owner loss");

    coordinator.shutdown().await;
    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(coordinator_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v3_checkpoint_replica_survives_creator_loss_and_rejects_corruption() {
    let coordinator_root = root("v3-checkpoint-replica-coordinator");
    let worker_roots = (0..6)
        .map(|index| root(&format!("v3-checkpoint-replica-worker-{index}")))
        .collect::<Vec<_>>();
    let coordinator_address = free_addr();
    let worker_addresses = (0..6).map(|_| free_addr()).collect::<Vec<_>>();
    let all_addresses = worker_addresses
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let coordinator = Node::start(config(
        coordinator_root.clone(),
        coordinator_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let mut workers = Vec::new();
    for (index, worker_root) in worker_roots.iter().enumerate() {
        let mut bootstrap = vec![coordinator_address.to_string()];
        bootstrap.extend(
            all_addresses
                .iter()
                .filter(|address| **address != worker_addresses[index].to_string())
                .cloned(),
        );
        workers.push(
            Node::start(config(
                worker_root.clone(),
                worker_addresses[index],
                bootstrap,
                vec![training_capability()],
            ))
            .await
            .unwrap(),
        );
    }
    let worker_root_by_id = workers
        .iter()
        .zip(worker_roots.iter())
        .map(|(worker, path)| (worker.node_id().to_string(), path.clone()))
        .collect::<BTreeMap<_, _>>();
    wait_for_connected(&coordinator, 6).await;
    sleep(Duration::from_millis(500)).await;

    let result = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::TrainV3 {
            workers: Some(6),
            windows: Some(2),
            local_steps: Some(2),
            checkpoint_every: Some(1),
        },
    )
    .await
    .unwrap();
    let (artifact, provider_values) = result["checkpoint_providers"]
        .as_object()
        .and_then(|entries| {
            entries.iter().find_map(|(artifact, providers)| {
                let providers = providers.as_array()?.clone();
                (providers.len() >= 3).then_some((artifact.clone(), providers))
            })
        })
        .expect("a checkpoint shard must have three-member replication evidence");
    let provider_names = provider_values
        .iter()
        .map(|provider| {
            serde_json::from_value::<NodeId>(provider.clone())
                .unwrap()
                .to_string()
        })
        .collect::<Vec<_>>();
    let creator = provider_names[0].clone();
    let surviving_replica = provider_names[1].clone();
    let alternate_replica = provider_names[2].clone();
    let creator_index = workers
        .iter()
        .position(|worker| worker.node_id().to_string() == creator)
        .expect("checkpoint creator must be a worker");
    let stopped_creator = workers.remove(creator_index);
    stopped_creator.shutdown().await;
    drop(stopped_creator);

    let fetched = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::FetchArtifact {
            peer: surviving_replica.clone(),
            artifact: artifact.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(fetched["verified"], true);
    assert_eq!(fetched["source"], surviving_replica);

    let replica_path = worker_root_by_id
        .get(&surviving_replica)
        .expect("surviving replica must be a worker")
        .join("artifacts")
        .join(&artifact);
    fs::write(&replica_path, b"corrupted-checkpoint-shard").unwrap();
    fs::remove_file(coordinator_root.join("artifacts").join(&artifact)).unwrap();
    let rejected = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::FetchArtifact {
            peer: surviving_replica,
            artifact: artifact.clone(),
        },
    )
    .await;
    assert!(
        rejected.is_err(),
        "corrupt checkpoint replica must fail hash verification"
    );

    let recovered = admin_call(
        coordinator.admin_socket(),
        &AdminRequest::FetchArtifact {
            peer: alternate_replica.clone(),
            artifact,
        },
    )
    .await
    .unwrap();
    assert_eq!(recovered["verified"], true);
    assert_eq!(recovered["source"], alternate_replica);

    coordinator.shutdown().await;
    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(coordinator_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn optional_relay_routes_authenticated_inference_when_direct_is_disabled() {
    let relay_root = root("relay");
    let worker_root = root("relay-worker");
    let requester_root = root("relay-requester");
    let relay_address = free_addr();
    let worker_address = free_addr();
    let requester_address = free_addr();
    let mut relay_config = config(relay_root.clone(), relay_address, Vec::new(), Vec::new());
    relay_config.relay_enabled = true;
    relay_config.relay_max_sessions = 8;
    relay_config.relay_max_bytes = 4 * 1024 * 1024;
    let relay = Node::start(relay_config).await.unwrap();
    let mut worker_config = config(
        worker_root.clone(),
        worker_address,
        vec![relay_address.to_string()],
        vec![builtin_capability()],
    );
    worker_config.relay_addresses = vec![relay_address.to_string()];
    let worker = Node::start(worker_config).await.unwrap();
    for _ in 0..100 {
        if relay
            .status()
            .await
            .unwrap()
            .get("known_peers")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count >= 1)
        {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    let mut requester_config = config(
        requester_root.clone(),
        requester_address,
        vec![relay_address.to_string()],
        Vec::new(),
    );
    requester_config.relay_addresses = vec![relay_address.to_string()];
    requester_config.prefer_relay = true;
    let requester = Node::start(requester_config).await.unwrap();
    for _ in 0..150 {
        if requester
            .status()
            .await
            .unwrap()
            .get("known_peers")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count >= 2)
        {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    let result = admin_call(
        requester.admin_socket(),
        &AdminRequest::Infer {
            capability: "inference.text".to_string(),
            input: "relay path is useful".to_string(),
            deadline_ms: Some(5_000),
            max_output_bytes: Some(16 * 1024),
            allow_input_transfer: Some(true),
            job_id: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(result["state"], "Succeeded");
    let output = result["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap_or_default() as u8)
        .collect::<Vec<_>>();
    assert!(String::from_utf8(output).unwrap().contains("positive"));
    requester.shutdown().await;
    worker.shutdown().await;
    relay.shutdown().await;
    let _ = fs::remove_dir_all(relay_root);
    let _ = fs::remove_dir_all(worker_root);
    let _ = fs::remove_dir_all(requester_root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn artifact_transfer_streams_chunks_and_verifies_the_content_hash() {
    let provider_root = root("artifact-provider");
    let requester_root = root("artifact-requester");
    let source = provider_root.join("operator-model.bin");
    fs::create_dir_all(&provider_root).unwrap();
    let fixture = (0..600_000u32)
        .map(|value| (value.wrapping_mul(31) & 0xff) as u8)
        .collect::<Vec<_>>();
    fs::write(&source, &fixture).unwrap();
    let provider_address = free_addr();
    let requester_address = free_addr();
    let provider = Node::start(config(
        provider_root.clone(),
        provider_address,
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        requester_root.clone(),
        requester_address,
        vec![provider_address.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    let registered = admin_call(
        provider.admin_socket(),
        &AdminRequest::RegisterModel {
            path: source.to_string_lossy().into_owned(),
            identity: "fixture.transfer.model".to_string(),
            format: "opaque".to_string(),
            local_only: false,
        },
    )
    .await
    .unwrap();
    let artifact = registered["artifact"].as_str().unwrap().to_string();
    assert_eq!(registered["dht"]["model_published"], true);
    assert_eq!(registered["dht"]["artifact_published"], true);
    for _ in 0..100 {
        if requester
            .status()
            .await
            .unwrap()
            .get("connected_peers")
            .and_then(serde_json::Value::as_u64)
            .is_some_and(|count| count >= 1)
        {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    let mut model_records = serde_json::json!([]);
    for _ in 0..120 {
        model_records = admin_call(
            requester.admin_socket(),
            &AdminRequest::DhtLookup {
                namespace: "model".to_string(),
                name: "fixture.transfer.model".to_string(),
            },
        )
        .await
        .unwrap_or_else(|_| serde_json::json!([]));
        if model_records.as_array().is_some_and(|records| {
            records
                .iter()
                .any(|record| record["owner"] == serde_json::to_value(provider.node_id()).unwrap())
        }) {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    assert!(model_records.as_array().is_some_and(|records| {
        records
            .iter()
            .any(|record| record["owner"] == serde_json::to_value(provider.node_id()).unwrap())
    }));
    let fetched = admin_call(
        requester.admin_socket(),
        &AdminRequest::FetchArtifact {
            peer: provider.node_id().to_string(),
            artifact: artifact.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(fetched["verified"], true);
    assert_eq!(fetched["size"], fixture.len());
    let inspected = admin_call(
        requester.admin_socket(),
        &AdminRequest::Inspect { artifact },
    )
    .await
    .unwrap();
    assert_eq!(inspected["verified"], true);
    requester.shutdown().await;
    provider.shutdown().await;
    let _ = fs::remove_dir_all(provider_root);
    let _ = fs::remove_dir_all(requester_root);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn restart_marks_inflight_jobs_failed_before_accepting_work() {
    let data_dir = root("restart-recovery");
    let job_id = JobId::from_bytes([8; 16]);
    let store = LocalStore::open(&data_dir, 32 * 1024 * 1024, 4 * 1024 * 1024).unwrap();
    store
        .save_jobs(&[PersistedJob {
            job_id,
            origin: NodeId::from_bytes([9; 32]),
            state: JobState::Running,
            updated_at: 1,
            output_hash: None,
            output: None,
            error: None,
        }])
        .unwrap();

    let node = Node::start(config(
        data_dir.clone(),
        free_addr(),
        Vec::new(),
        Vec::new(),
    ))
    .await
    .unwrap();
    let jobs = admin_call(node.admin_socket(), &AdminRequest::Jobs)
        .await
        .unwrap();
    let recovered = jobs
        .as_array()
        .unwrap()
        .iter()
        .find(|job| job["job_id"] == serde_json::to_value(job_id).unwrap())
        .unwrap();
    assert_eq!(recovered["state"], "Failed");
    assert_eq!(
        recovered["error"],
        "node restarted before the job reached a terminal state"
    );

    node.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(data_dir);
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 6)]
async fn disconnect_during_remote_job_returns_failure() {
    let root_requester = root("disconnect-requester");
    let root_worker = root("disconnect-worker");
    let address_requester = free_addr();
    let address_worker = free_addr();
    let worker = Node::start(config(
        root_worker.clone(),
        address_worker,
        Vec::new(),
        vec![slow_process_capability()],
    ))
    .await
    .unwrap();
    let requester = Node::start(config(
        root_requester.clone(),
        address_requester,
        vec![address_worker.to_string()],
        Vec::new(),
    ))
    .await
    .unwrap();
    for _ in 0..100 {
        if let Ok(status) = admin_call(requester.admin_socket(), &AdminRequest::Status).await
            && status["connected_peers"] == 1
        {
            break;
        }
        sleep(Duration::from_millis(50)).await;
    }
    let requester_socket = requester.admin_socket().to_path_buf();
    let request = AdminRequest::Infer {
        capability: "inference.text".to_string(),
        input: "disconnect me".to_string(),
        deadline_ms: Some(5000),
        max_output_bytes: Some(16 * 1024),
        allow_input_transfer: Some(true),
        job_id: Some("05050505050505050505050505050505".to_string()),
    };
    let pending = tokio::spawn(async move { admin_call(requester_socket, &request).await });
    sleep(Duration::from_millis(250)).await;
    worker.shutdown().await;
    let result = timeout(Duration::from_secs(5), pending)
        .await
        .unwrap()
        .unwrap();
    if let Ok(value) = result {
        assert_ne!(value["state"], "Succeeded");
    }

    requester.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_requester);
    let _ = fs::remove_dir_all(root_worker);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn v4_reference_fabric_uses_real_peer_protocol_for_parallelism_and_state_movement() {
    let worker_roots = (0..4)
        .map(|index| root(&format!("v4-worker-{index}")))
        .collect::<Vec<_>>();
    let addresses = (0..4).map(|_| free_addr()).collect::<Vec<_>>();
    let mut workers = Vec::new();
    for index in 0..4 {
        let bootstrap = addresses
            .iter()
            .enumerate()
            .filter(|(peer_index, _)| *peer_index != index)
            .map(|(_, address)| address.to_string())
            .collect::<Vec<_>>();
        workers.push(
            Node::start(config(
                worker_roots[index].clone(),
                addresses[index],
                bootstrap,
                vec![training_capability()],
            ))
            .await
            .unwrap(),
        );
    }
    for worker in &workers {
        wait_for_connected(worker, 1).await;
    }
    let requester_root = root("v4-requester");
    let requester = Node::start(config(
        requester_root.clone(),
        free_addr(),
        addresses.iter().map(ToString::to_string).collect(),
        Vec::new(),
    ))
    .await
    .unwrap();
    wait_for_connected(&requester, 4).await;
    for worker in &workers {
        wait_for_connected(worker, 3).await;
    }
    let worker_ids = workers
        .iter()
        .map(|worker| worker.node_id())
        .collect::<Vec<_>>();
    let worker_args = worker_ids
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let plan = admin_call(
        requester.admin_socket(),
        &AdminRequest::PlanTrainingV4 {
            model_bytes: 2_048,
            workers: Some(4),
            strategy: "hybrid".to_string(),
            tensor_degree: Some(2),
            pipeline_stages: Some(2),
        },
    )
    .await
    .unwrap();
    assert_eq!(plan["plan"]["support"], "Experimental");
    assert_eq!(plan["plan"]["shards"].as_array().unwrap().len(), 4);
    assert!(!plan["explanations"].as_array().unwrap().is_empty());

    let plan_job: JobId = serde_json::from_value(plan["plan"]["job_id"].clone()).unwrap();
    let plan_owner: NodeId =
        serde_json::from_value(plan["plan"]["shards"][0]["owners"][0].clone()).unwrap();
    let plan_target = plan["plan"]["workers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| serde_json::from_value::<NodeId>(value.clone()).unwrap())
        .find(|worker| *worker != plan_owner)
        .unwrap();
    let plan_owner_node = workers
        .iter()
        .find(|worker| worker.node_id() == plan_owner)
        .unwrap();
    assert_eq!(plan["plan_notifications_acknowledged"], 4);
    let replan = admin_call(
        requester.admin_socket(),
        &AdminRequest::ReplanTrainingV4 {
            job_id: plan_job.to_string(),
            model_bytes: 2_048,
            workers: Some(4),
            strategy: "local_sgd".to_string(),
            tensor_degree: Some(0),
            pipeline_stages: Some(0),
        },
    )
    .await
    .unwrap();
    assert_eq!(replan["plan_active"], false);
    assert_eq!(replan["plan_activation_requires_migration"], true);
    assert_eq!(replan["topology_change_requires_restart"], false);
    assert!(replan["previous_plan_hash"] != replan["plan"]["plan_hash"]);
    let activated = admin_call(
        requester.admin_socket(),
        &AdminRequest::ActivateTrainingV4 {
            job_id: plan_job.to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(activated["plan_active"], true);
    assert_eq!(activated["migrated_shards"], 0);
    assert_eq!(activated["topology_change_requires_restart"], false);
    // Activation sends the committed plan over the authenticated transport;
    // let the owner persist that generation before seeding its live state.
    sleep(Duration::from_millis(250)).await;

    let plan_seeded = admin_call(
        plan_owner_node.admin_socket(),
        &AdminRequest::V4SeedShard {
            job_id: plan_job.to_string(),
            shard_id: 0,
            state: "plan-bound-live-shard".to_string(),
        },
    )
    .await
    .unwrap();
    assert!(
        plan_seeded["plan_update"]["plan_generation"]
            .as_u64()
            .is_some()
    );
    wait_for_connected(plan_owner_node, 3).await;
    let plan_target_node = workers
        .iter()
        .find(|worker| worker.node_id() == plan_target)
        .unwrap();
    wait_for_connected(plan_target_node, 3).await;
    let plan_migrated = admin_call(
        plan_owner_node.admin_socket(),
        &AdminRequest::V4MigrateShard {
            job_id: plan_job.to_string(),
            shard_id: 0,
            target: plan_target.to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(plan_migrated["verified"], true);
    assert_eq!(plan_migrated["source_retired"], true);
    assert_eq!(plan_migrated["plan_update"]["applied"], true);
    let expected_generation = plan_migrated["plan_update"]["plan_generation"]
        .as_u64()
        .unwrap();
    for _ in 0..200 {
        if requester
            .status()
            .await
            .ok()
            .and_then(|status| {
                status["training_v4_plan_generations"][plan_job.to_string()].as_u64()
            })
            .is_some_and(|generation| generation >= expected_generation)
        {
            break;
        }
        sleep(Duration::from_millis(25)).await;
    }
    for worker in &workers {
        let mut synchronized = false;
        for _ in 0..200 {
            if worker
                .status()
                .await
                .ok()
                .and_then(|status| {
                    status["training_v4_plan_generations"][plan_job.to_string()].as_u64()
                })
                .is_some_and(|generation| generation >= expected_generation)
            {
                synchronized = true;
                break;
            }
            sleep(Duration::from_millis(25)).await;
        }
        assert!(
            synchronized,
            "worker {} did not converge on migrated plan generation {}",
            worker.node_id(),
            expected_generation
        );
    }

    // Reverse the just-migrated shard through the proposal protocol.  This
    // exercises a real proposer -> old-owner request -> target transfer
    // without allowing the proposal to become active before the transfer.
    let reverse_replan = admin_call(
        requester.admin_socket(),
        &AdminRequest::ReplanTrainingV4 {
            job_id: plan_job.to_string(),
            model_bytes: 2_048,
            workers: Some(4),
            strategy: "local_sgd".to_string(),
            tensor_degree: Some(0),
            pipeline_stages: Some(0),
        },
    )
    .await
    .unwrap();
    assert_eq!(reverse_replan["plan_active"], false);
    let reverse_activation = admin_call(
        requester.admin_socket(),
        &AdminRequest::ActivateTrainingV4 {
            job_id: plan_job.to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(reverse_activation["plan_active"], true);
    assert_eq!(reverse_activation["migrated_shards"], 1);
    assert_eq!(reverse_activation["notifications_failed"], 0);

    let replicated = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4ReplicateState {
            workers: worker_args.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(replicated["single_durable_authority"], false);
    assert_eq!(replicated["state_replicas"], 4);
    assert_eq!(replicated["checkpoint_replica_factor"], 4);
    assert_eq!(replicated["acknowledged_copies"], 16);
    assert_eq!(replicated["optimizer_replica_factor"], 2);

    let tensor = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4TensorDemo {
            workers: worker_args[..2].to_vec(),
        },
    )
    .await
    .unwrap();
    assert_eq!(tensor["partitioned_operation"], true);
    assert_eq!(tensor["full_model_materialized_on_worker"], false);
    assert_eq!(tensor["backward_shards"], 2);
    assert_eq!(tensor["forward_outputs"].as_array().unwrap().len(), 4);

    let pipeline = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4PipelineDemo {
            stages: worker_args[..2].to_vec(),
            microbatches: Some(3),
        },
    )
    .await
    .unwrap();
    assert_eq!(pipeline["central_pipeline_controller"], false);
    assert_eq!(pipeline["microbatches"].as_array().unwrap().len(), 3);
    assert!(pipeline["backward"]["gradient"].is_array());

    let collective = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4CollectiveDemo {
            workers: worker_args.clone(),
        },
    )
    .await
    .unwrap();
    assert_eq!(collective["single_collective_root"], false);
    assert_eq!(collective["independent_results"], 2);
    assert_eq!(collective["collective_roots"].as_array().unwrap().len(), 2);
    assert_eq!(collective["maximum_fan_in"], 2);
    assert!(collective["result"]["values"].is_array());
    assert!(
        collective["result"]["root_received_bytes"]
            .as_u64()
            .unwrap()
            < collective["result"]["contributor_bytes"].as_u64().unwrap()
    );

    let byzantine = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4ByzantineDemo {
            workers: worker_args[..3].to_vec(),
            malicious: Some(1),
            policy: "median".to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(byzantine["response"]["robust"], true);
    assert_eq!(
        byzantine["response"]["rejected_workers"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    let mut replay_observed = false;
    let mut equivocation_observed = false;
    for _ in 0..500 {
        if workers[0]
            .status()
            .await
            .ok()
            .and_then(|status| status["v6_security"]["events"].as_array().cloned())
            .is_some_and(|events| {
                replay_observed = events
                    .iter()
                    .any(|event| event["kind"] == "duplicate_contribution_rejected");
                equivocation_observed = events
                    .iter()
                    .any(|event| event["kind"] == "equivocation_detected");
                replay_observed && equivocation_observed
            })
        {
            replay_observed = true;
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert!(
        replay_observed,
        "real V4 path must record a replay rejection"
    );
    assert!(
        equivocation_observed,
        "real V4 path must record conflicting signed-state evidence"
    );

    let reconcile = admin_call(
        requester.admin_socket(),
        &AdminRequest::V4ReconcileDemo {
            worker: worker_ids[1].to_string(),
            left_value: 10,
            right_value: 14,
            policy: "local_sgd".to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(reconcile["accepted"], true);
    assert_eq!(reconcile["merged_value"], 12);

    let migration_job = "a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1a1".to_string();
    let seeded = admin_call(
        workers[0].admin_socket(),
        &AdminRequest::V4SeedShard {
            job_id: migration_job.clone(),
            shard_id: 7,
            state: "network-migrated-v4-state".to_string(),
        },
    )
    .await
    .unwrap();
    let migrated = admin_call(
        workers[0].admin_socket(),
        &AdminRequest::V4MigrateShard {
            job_id: migration_job,
            shard_id: 7,
            target: worker_ids[1].to_string(),
        },
    )
    .await
    .unwrap();
    assert_eq!(seeded["lifecycle"], "active");
    assert_eq!(migrated["verified"], true);
    assert_eq!(migrated["source_retired"], true);
    let source_status = workers[0].status().await.unwrap();
    let target_status = workers[1].status().await.unwrap();
    let reverse_shard = &reverse_replan["plan"]["shards"][0];
    let expected_plan_target_shards = usize::from(
        reverse_shard["owners"]
            .as_array()
            .unwrap()
            .iter()
            .chain(reverse_shard["replicas"].as_array().unwrap())
            .any(|worker| worker == &serde_json::to_value(worker_ids[0]).unwrap()),
    );
    assert_eq!(
        source_status["training_v4_data_shards"],
        expected_plan_target_shards
    );
    assert!(
        target_status["training_v4_data_shards"]
            .as_u64()
            .unwrap_or(0)
            >= 1
    );

    requester.shutdown().await;
    for worker in &workers {
        worker.shutdown().await;
    }
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(requester_root);
    for worker_root in worker_roots {
        let _ = fs::remove_dir_all(worker_root);
    }
}
