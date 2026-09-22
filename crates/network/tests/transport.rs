use intelligence_network::{Identity, NetworkConfig, NetworkEvent, start};
use intelligence_storage::LocalStore;
use std::{
    fs,
    net::{SocketAddr, TcpListener},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    sync::mpsc,
    time::{Duration, sleep, timeout},
};

fn free_addr() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").expect("free port");
    listener.local_addr().expect("address")
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
