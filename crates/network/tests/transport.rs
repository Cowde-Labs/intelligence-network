use intelligence_network::{Identity, NetworkConfig, NetworkEvent, start};
use intelligence_storage::LocalStore;
use std::{
    collections::HashSet,
    fs,
    net::{SocketAddr, TcpListener},
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::mpsc,
    time::{Duration, sleep, timeout},
};

fn free_addr() -> SocketAddr {
    static USED_PORTS: OnceLock<Mutex<HashSet<u16>>> = OnceLock::new();
    let used_ports = USED_PORTS.get_or_init(|| Mutex::new(HashSet::new()));
    for _ in 0..1_000 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("free port");
        let address = listener.local_addr().expect("address");
        if used_ports.lock().unwrap().insert(address.port()) {
            return address;
        }
    }
    panic!("unable to allocate an isolated test port")
}

fn root(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "intelligence-network-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

async fn wait_connected(receiver: &mut mpsc::Receiver<NetworkEvent>) {
    timeout(Duration::from_secs(10), async {
        loop {
            if matches!(receiver.recv().await, Some(NetworkEvent::PeerConnected(_))) {
                break;
            }
        }
    })
    .await
    .expect("peer connection");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn restart_on_same_address_succeeds_repeatedly() {
    let root_dir = root("restart");
    for _ in 0..5 {
        let address = free_addr();
        let (events_tx, _events) = mpsc::channel(64);
        let config = NetworkConfig {
            listen_addr: address,
            advertise_addr: address.to_string(),
            ..NetworkConfig::default()
        };
        let identity = Identity::load_or_generate(root_dir.join("identity.key")).unwrap();
        let node = start(
            config.clone(),
            identity,
            LocalStore::open(root_dir.join("state"), 16 * 1024 * 1024, 1024 * 1024).unwrap(),
            events_tx,
        )
        .await
        .unwrap();
        node.shutdown().await;
        let (events_tx, _events) = mpsc::channel(64);
        let identity = Identity::load_or_generate(root_dir.join("identity.key")).unwrap();
        start(
            config,
            identity,
            LocalStore::open(root_dir.join("state"), 16 * 1024 * 1024, 1024 * 1024).unwrap(),
            events_tx,
        )
        .await
        .expect("same-address restart")
        .shutdown()
        .await;
    }
    let _ = fs::remove_dir_all(root_dir);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_nodes_bind_identity_over_encrypted_transport() {
    let address_a = free_addr();
    let address_b = free_addr();
    let root_a = root("a");
    let root_b = root("b");
    let (events_a_tx, mut events_a) = mpsc::channel(64);
    let (events_b_tx, mut events_b) = mpsc::channel(64);
    let config_a = NetworkConfig {
        listen_addr: address_a,
        advertise_addr: address_a.to_string(),
        ..NetworkConfig::default()
    };
    let config_b = NetworkConfig {
        listen_addr: address_b,
        advertise_addr: address_b.to_string(),
        bootstrap: vec![address_a.to_string()],
        ..NetworkConfig::default()
    };
    let identity_a = Identity::load_or_generate(root_a.join("identity.key")).unwrap();
    let identity_b = Identity::load_or_generate(root_b.join("identity.key")).unwrap();
    let node_a = start(
        config_a,
        identity_a,
        LocalStore::open(root_a.join("state"), 16 * 1024 * 1024, 1024 * 1024).unwrap(),
        events_a_tx,
    )
    .await
    .unwrap();
    let node_b = start(
        config_b,
        identity_b,
        LocalStore::open(root_b.join("state"), 16 * 1024 * 1024, 1024 * 1024).unwrap(),
        events_b_tx,
    )
    .await
    .unwrap();
    wait_connected(&mut events_a).await;
    wait_connected(&mut events_b).await;
    assert_eq!(node_a.connected_peers().await, 1);
    assert_eq!(node_b.connected_peers().await, 1);
    assert_eq!(node_a.peer_records().await.len(), 1);
    assert_eq!(node_b.peer_records().await.len(), 1);
    node_a.shutdown().await;
    node_b.shutdown().await;
    sleep(Duration::from_millis(100)).await;
    let _ = fs::remove_dir_all(root_a);
    let _ = fs::remove_dir_all(root_b);
}
