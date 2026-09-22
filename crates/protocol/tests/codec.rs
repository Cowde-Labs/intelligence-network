use intelligence_protocol::{
    ArtifactId, BackendKind, Capability, CapabilityChallenge, CapabilityEvidence, ComputeTaskKind,
    DataLocality, DhtKey, DhtNamespace, DhtRecord, DhtRequest, DhtRequestKind, DhtResponse,
    DhtResponseKind, FRAME_HEADER_SIZE, FRAME_MAGIC, FrameHeader, FrameKind, Hello, JobId, JobKind,
    JobRequest, JobState, MAX_DHT_RECORD_VALUE, MAX_FRAME_SIZE, Message, NodeId, NumericFormat,
    PeerRecord, PrivacyPolicy, RelayEnvelope, RequestId, ResourceLimits, TrainingExecution,
    TrainingGroup, TrainingMessage, TrainingSample, TrainingShardAssignment, TrainingStart,
    TrainingV4Message, TrainingV5Message, V4AcceleratorCapability, V4AcceleratorFamily,
    V4AggregationGroup, V4NumericalFormat, V4ParallelismStrategy, V4ShardLifecycle,
    V4ShardOwnership, V4SupportLevel, V4TensorForward, V4TrainingPlan, V4WorkerCapability,
    VersionRange, decode_frame, encode_message, encode_message_at_version,
};

fn record() -> PeerRecord {
    PeerRecord {
        node_id: NodeId::from_bytes([7; 32]),
        public_key: [3; 32],
        addresses: vec!["127.0.0.1:4000".to_string()],
        capabilities: vec![Capability {
            name: "inference.text".to_string(),
            version: 1,
            model: Some("builtin.tiny-sentiment.v1".to_string()),
            resources: ResourceLimits {
                max_input_bytes: 1024,
                max_output_bytes: 4096,
                memory_bytes: 64 * 1024 * 1024,
                cpu_millis: 1000,
            },
            evidence: CapabilityEvidence::Claimed,
            expires_at: 2,
            metadata: Vec::new(),
            compute_backends: Vec::new(),
        }],
        announced_at: 1,
        expires_at: 2,
        observed_latency_ms: None,
    }
}

#[test]
fn round_trips_a_bounded_message() {
    let message = Message::Hello(Hello {
        nonce: [4; 16],
        supported: VersionRange::current(),
        record: record(),
        signature: vec![0; 64],
    });
    let frame = encode_message(&message, MAX_FRAME_SIZE).expect("valid frame");
    assert_eq!(
        decode_frame(&frame, MAX_FRAME_SIZE).expect("decode"),
        message
    );
}

#[test]
fn rejects_oversized_job_input_before_encoding() {
    let message = Message::JobRequest(JobRequest {
        job_id: JobId::from_bytes([1; 16]),
        origin: NodeId::from_bytes([2; 32]),
        kind: JobKind::Inference,
        capability: "inference.text".to_string(),
        model: None,
        input: vec![0; 64 * 1024 + 1],
        deadline_ms: 1000,
        max_output_bytes: 100,
        privacy: PrivacyPolicy::default(),
    });
    assert!(encode_message(&message, MAX_FRAME_SIZE).is_err());
}

#[test]
fn rejects_frame_that_claims_a_large_payload() {
    let header = FrameHeader {
        major: 1,
        minor: 0,
        kind: FrameKind::Ping,
        payload_len: (MAX_FRAME_SIZE as u32) + 1,
    }
    .encode();
    let mut frame = Vec::with_capacity(FRAME_HEADER_SIZE);
    frame.extend_from_slice(&header);
    assert!(decode_frame(&frame, MAX_FRAME_SIZE).is_err());
}

#[test]
fn frame_limit_includes_the_wire_header() {
    let message = Message::Ping(intelligence_protocol::Ping { nonce: [8; 16] });
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid frame");
    assert!(encode_message(&message, encoded.len() - 1).is_err());
    assert!(decode_frame(&encoded, encoded.len() - 1).is_err());
}

#[test]
fn rejects_wrong_magic_and_version() {
    let mut bytes = [0u8; FRAME_HEADER_SIZE];
    bytes[..4].copy_from_slice(&FRAME_MAGIC);
    bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
    bytes[8..10].copy_from_slice(&(FrameKind::Ping as u16).to_le_bytes());
    assert!(decode_frame(&bytes, MAX_FRAME_SIZE).is_err());

    bytes[..4].copy_from_slice(b"NOPE");
    assert!(decode_frame(&bytes, MAX_FRAME_SIZE).is_err());
}

#[test]
fn state_machine_has_no_terminal_resurrection() {
    assert!(JobState::Received.can_transition(JobState::Validated));
    assert!(JobState::Running.can_transition(JobState::Succeeded));
    assert!(!JobState::Succeeded.can_transition(JobState::Running));
    assert!(!JobState::Cancelled.can_transition(JobState::Succeeded));
}

#[test]
fn v1_messages_round_trip_and_legacy_minor_can_still_be_encoded() {
    let origin = NodeId::from_public_key(&[9; 32]);
    let target = NodeId::from_public_key(&[10; 32]);
    let message = Message::RelayEnvelope(RelayEnvelope {
        session_id: RequestId::from_bytes([1; 16]),
        origin,
        origin_public_key: [9; 32],
        target,
        payload: vec![1, 2, 3],
        signature: vec![0; 64],
    });
    let frame = encode_message(&message, MAX_FRAME_SIZE).expect("v1.1 frame");
    assert_eq!(decode_frame(&frame, MAX_FRAME_SIZE).unwrap(), message);

    let legacy = encode_message_at_version(
        &Message::Ping(intelligence_protocol::Ping { nonce: [2; 16] }),
        MAX_FRAME_SIZE,
        1,
        0,
    )
    .expect("legacy frame");
    assert_eq!(legacy[6], 0);
    assert!(matches!(
        decode_frame(&legacy, MAX_FRAME_SIZE),
        Ok(Message::Ping(_))
    ));
    assert!(encode_message_at_version(&message, MAX_FRAME_SIZE, 1, 0).is_err());
}

#[test]
fn dht_messages_round_trip_only_on_protocol_minor_two() {
    let owner_key = [3; 32];
    let dht = Message::DhtRequest(DhtRequest {
        request_id: RequestId::from_bytes([4; 16]),
        origin: NodeId::from_public_key(&owner_key),
        request: DhtRequestKind::Put {
            record: DhtRecord {
                namespace: DhtNamespace::Capability,
                key: DhtKey::for_name(DhtNamespace::Capability, "inference.text"),
                owner: NodeId::from_public_key(&owner_key),
                owner_public_key: owner_key,
                sequence: 1,
                expires_at: 100,
                value: b"provider".to_vec(),
                signature: vec![0; 64],
            },
        },
    });
    let encoded = encode_message(&dht, MAX_FRAME_SIZE).expect("DHT frame");
    assert_eq!(decode_frame(&encoded, MAX_FRAME_SIZE).unwrap(), dht);
    assert!(encode_message_at_version(&dht, MAX_FRAME_SIZE, 1, 1).is_err());

    let response = Message::DhtResponse(DhtResponse {
        request_id: RequestId::from_bytes([5; 16]),
        responder: NodeId::from_bytes([6; 32]),
        response: DhtResponseKind::NotFound,
    });
    assert_eq!(
        decode_frame(
            &encode_message(&response, MAX_FRAME_SIZE).unwrap(),
            MAX_FRAME_SIZE
        )
        .unwrap(),
        response
    );
}

#[test]
fn dht_record_value_is_bounded_before_encoding() {
    let public_key = [8; 32];
    let message = Message::DhtRequest(DhtRequest {
        request_id: RequestId::from_bytes([1; 16]),
        origin: NodeId::from_public_key(&public_key),
        request: DhtRequestKind::Put {
            record: DhtRecord {
                namespace: DhtNamespace::Artifact,
                key: DhtKey::from_bytes([2; 32]),
                owner: NodeId::from_public_key(&public_key),
                owner_public_key: public_key,
                sequence: 1,
                expires_at: 10,
                value: vec![0; MAX_DHT_RECORD_VALUE + 1],
                signature: vec![0; 64],
            },
        },
    });
    assert!(encode_message(&message, MAX_FRAME_SIZE).is_err());
}

#[test]
fn v3_training_messages_round_trip_and_are_rejected_by_minor_two() {
    let node = NodeId::from_bytes([11; 32]);
    let job_id = JobId::from_bytes([12; 16]);
    let branch = ArtifactId::from_bytes([13; 32]);
    let state_hash = ArtifactId::from_bytes([14; 32]);
    let assignment = TrainingShardAssignment {
        shard_id: 0,
        group_id: 0,
        owners: vec![node],
        replicas: vec![node],
        generation: 0,
        state_hash,
    };
    let message = Message::Training(TrainingMessage::Start(TrainingStart {
        job_id,
        coordinator: node,
        term: 1,
        membership_epoch: 1,
        branch,
        plan_hash: ArtifactId::from_bytes([15; 32]),
        max_windows: 2,
        local_steps: 1,
        max_staleness: 1,
        checkpoint_every: 1,
        execution: TrainingExecution::AutonomousLocalSgd,
        model_state_bytes: 2048,
        shard_state_bytes: 1024,
        participants: vec![node],
        groups: vec![TrainingGroup {
            group_id: 0,
            shard_id: 0,
            members: vec![node],
            aggregator: node,
        }],
        shards: vec![assignment.clone()],
        assignment,
        initial_value: 0,
        initial_optimizer: 0,
        samples: vec![TrainingSample {
            feature: 1,
            target: 2,
        }],
    }));
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid V3 frame");
    assert_eq!(decode_frame(&encoded, MAX_FRAME_SIZE).unwrap(), message);
    assert!(encode_message_at_version(&message, MAX_FRAME_SIZE, 1, 2).is_err());
}

#[test]
fn v4_tensor_messages_are_bounded_and_require_protocol_minor_four() {
    let job_id = JobId::from_bytes([21; 16]);
    let message = Message::TrainingV4(TrainingV4Message::TensorForward(V4TensorForward {
        request_id: job_id,
        job_id,
        plan_generation: 1,
        model_generation: 1,
        shard_id: 0,
        sequence: 1,
        input: vec![1, 2, 3, 4],
    }));
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid V4 frame");
    assert_eq!(decode_frame(&encoded, MAX_FRAME_SIZE).unwrap(), message);
    assert!(encode_message_at_version(&message, MAX_FRAME_SIZE, 1, 3).is_err());

    let oversized = Message::TrainingV4(TrainingV4Message::TensorForward(V4TensorForward {
        request_id: job_id,
        job_id,
        plan_generation: 1,
        model_generation: 1,
        shard_id: 0,
        sequence: 1,
        input: vec![0; intelligence_protocol::MAX_V4_VECTOR + 1],
    }));
    assert!(oversized.validate().is_err());
}

#[test]
fn v4_plan_activation_messages_round_trip_and_bind_lineage() {
    let job_id = JobId::from_bytes([31; 16]);
    let proposer = NodeId::from_bytes([32; 32]);
    let source = NodeId::from_bytes([33; 32]);
    let target = NodeId::from_bytes([34; 32]);
    let message = Message::TrainingV4(TrainingV4Message::ShardMigrationRequest(
        intelligence_protocol::V4ShardMigrationRequest {
            request_id: JobId::from_bytes([35; 16]),
            job_id,
            proposer,
            proposal_hash: ArtifactId::from_bytes([36; 32]),
            plan_generation: 2,
            shard_id: 1,
            from: source,
            to: target,
            ownership_generation: 2,
            content_hash: ArtifactId::from_bytes([37; 32]),
            reply_to: proposer,
        },
    ));
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid activation frame");
    assert_eq!(decode_frame(&encoded, MAX_FRAME_SIZE).unwrap(), message);

    let ack = Message::TrainingV4(TrainingV4Message::PlanProposalAck(
        intelligence_protocol::V4PlanProposalAck {
            job_id,
            plan_generation: 2,
            proposal_hash: ArtifactId::from_bytes([36; 32]),
            accepted: true,
            reason: "proposal persisted for activation".to_string(),
        },
    ));
    assert_eq!(
        decode_frame(
            &encode_message(&ack, MAX_FRAME_SIZE).expect("valid proposal ack"),
            MAX_FRAME_SIZE,
        )
        .unwrap(),
        ack
    );
}

#[test]
fn v5_compute_messages_round_trip_and_require_protocol_minor_six() {
    let message = Message::TrainingV5(TrainingV5Message::CapabilityChallenge(
        CapabilityChallenge {
            challenge_id: RequestId::from_bytes([51; 16]),
            job_id: JobId::from_bytes([52; 16]),
            worker: NodeId::from_bytes([53; 32]),
            backend: BackendKind::Cuda,
            task_kind: ComputeTaskKind::CapabilityChallenge,
            format: NumericFormat::F32,
            input_elements: 16,
            seed: 7,
            deadline_ms: 5_000,
        },
    ));
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid V5 frame");
    assert_eq!(decode_frame(&encoded, MAX_FRAME_SIZE).unwrap(), message);
    assert!(encode_message_at_version(&message, MAX_FRAME_SIZE, 1, 5).is_err());

    let mut malformed = match message {
        Message::TrainingV5(TrainingV5Message::CapabilityChallenge(value)) => value,
        _ => unreachable!(),
    };
    malformed.task_kind = ComputeTaskKind::TensorForward;
    assert!(
        Message::TrainingV5(TrainingV5Message::CapabilityChallenge(malformed))
            .validate()
            .is_err()
    );
}

#[test]
fn initial_v4_plan_keeps_absent_parent_on_the_wire() {
    let workers = [
        NodeId::from_public_key(&[41; 32]),
        NodeId::from_public_key(&[42; 32]),
    ];
    let branch = ArtifactId::from_bytes_hashed(b"v4-codec-branch");
    let shard_hashes = [
        ArtifactId::from_bytes_hashed(b"v4-codec-shard-0"),
        ArtifactId::from_bytes_hashed(b"v4-codec-shard-1"),
    ];
    let accelerator = V4AcceleratorCapability {
        family: V4AcceleratorFamily::Cpu,
        device_model: "reference-cpu".to_string(),
        device_count: 1,
        memory_bytes: 1024,
        formats: vec![V4NumericalFormat::I8],
        runtime: "reference".to_string(),
        runtime_version: "v4".to_string(),
        physical_verified: false,
    };
    let worker_capabilities = workers
        .iter()
        .copied()
        .map(|node| V4WorkerCapability {
            node,
            accelerator: accelerator.clone(),
            memory_bytes: 1024,
            compute_units: 1,
            rtt_ms: 1,
            bandwidth_mbps: 1,
            reliability_permille: 1000,
            backends: Vec::new(),
        })
        .collect::<Vec<_>>();
    let mut plan = V4TrainingPlan {
        job_id: JobId::from_bytes([43; 16]),
        proposer: workers[0],
        plan_generation: 1,
        model_generation: 1,
        training_epoch: 1,
        membership_epoch: 1,
        coordination_term: 1,
        branch,
        strategy: V4ParallelismStrategy::LocalSgd,
        support: V4SupportLevel::Supported,
        workers: workers.to_vec(),
        groups: vec![V4AggregationGroup {
            group_id: 0,
            members: workers.to_vec(),
            aggregator: workers[0],
            parent: None,
        }],
        shards: workers
            .iter()
            .enumerate()
            .map(|(index, owner)| V4ShardOwnership {
                shard_id: index as u16,
                model_generation: 1,
                owners: vec![*owner],
                replicas: vec![workers[1 - index]],
                ownership_generation: 1,
                content_hash: shard_hashes[index],
                state_bytes: 1,
                memory_bytes: 1,
                runtime_requirement: "reference.cpu.i64".to_string(),
                lifecycle: V4ShardLifecycle::Active,
            })
            .collect(),
        tensor_degree: 0,
        pipeline_stages: 0,
        local_steps: 1,
        max_staleness: 0,
        checkpoint_replication: 2,
        data_locality: DataLocality::Selective,
        accelerator: None,
        worker_capabilities,
        rationale: "codec compatibility fixture".to_string(),
        plan_hash: ArtifactId::from_bytes([0; 32]),
        parent_plan_hash: None,
        compute_requirements: Vec::new(),
        backend_assignments: Vec::new(),
    };
    let mut unsigned = plan.clone();
    unsigned.plan_hash = ArtifactId::from_bytes([0; 32]);
    plan.plan_hash = ArtifactId::from_bytes_hashed(&postcard::to_allocvec(&unsigned).unwrap());
    plan.validate().expect("valid initial plan");

    let message = Message::TrainingV4(TrainingV4Message::Plan(plan.clone()));
    let encoded = encode_message(&message, MAX_FRAME_SIZE).expect("valid initial plan frame");
    let decoded = decode_frame(&encoded, MAX_FRAME_SIZE).expect("decode initial plan frame");
    let Message::TrainingV4(TrainingV4Message::Plan(decoded_plan)) = decoded else {
        panic!("wrong decoded message kind");
    };
    assert_eq!(decoded_plan, plan);
    assert_eq!(decoded_plan.parent_plan_hash, None);
}
