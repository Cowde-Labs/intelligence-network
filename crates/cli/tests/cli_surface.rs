use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_intelligence");

const VISIBLE_COMMANDS: &[&str] = &[
    "up", "down", "status", "doctor", "infer", "evaluate", "model", "artifact", "train", "network",
    "trust", "identity", "jobs", "config", "service", "run", "init", "version",
];

const RETIRED_NAMES: &[&str] = &[
    "v3",
    "v4",
    "frontier",
    "demo",
    "dht-",
    "register-model",
    "fetch-artifact",
    "training-status",
    "dev",
];

const LEGACY_COMMANDS: &[&str] = &[
    "peers",
    "capabilities",
    "dht-stats",
    "dht-publish",
    "dht-lookup",
    "dht-find-node",
    "rotate",
    "cancel",
    "shutdown",
    "inspect",
    "fetch-artifact",
    "register-model",
    "plan-training",
    "plan-training-v4",
    "plan-frontier-training",
    "replan-training-v4",
    "activate-training-v4",
    "train-reference",
    "train-v3",
    "train-v3-start",
    "train-v4",
    "train-frontier",
    "train-v4-start",
    "train-frontier-start",
    "training-status",
    "v4-seed-shard",
    "v4-migrate-shard",
    "v4-replicate-state",
    "v4-tensor-demo",
    "v4-pipeline-demo",
    "v4-collective-demo",
    "v4-reconcile-demo",
    "v4-byzantine-demo",
];

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("failed to spawn intelligence")
}

#[test]
fn top_level_help_lists_only_current_commands() {
    let output = run(&["--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in VISIBLE_COMMANDS {
        assert!(stdout.contains(name), "--help missing `{name}`:\n{stdout}");
    }
    for name in RETIRED_NAMES {
        assert!(
            !stdout.contains(name),
            "--help leaks retired name `{name}`:\n{stdout}"
        );
    }
}

#[test]
fn train_help_lists_current_subcommands() {
    let output = run(&["train", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in [
        "start",
        "plan",
        "replan",
        "activate",
        "status",
        "cancel",
        "migrate",
        "replicate",
        "reconcile",
        "reference",
    ] {
        assert!(
            stdout.contains(name),
            "train --help missing `{name}`:\n{stdout}"
        );
    }
    for name in ["v3", "v4"] {
        assert!(
            !stdout.contains(name),
            "train --help leaks retired name `{name}`:\n{stdout}"
        );
    }
}

#[test]
fn legacy_commands_still_parse() {
    for name in LEGACY_COMMANDS {
        let output = run(&[name, "--help"]);
        assert!(
            output.status.success(),
            "legacy `{name} --help` failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn dev_group_is_available_but_hidden() {
    let output = run(&["dev", "--help"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    for name in [
        "seed-shard",
        "tensor-demo",
        "pipeline-demo",
        "collective-demo",
        "byzantine-demo",
        "shutdown",
    ] {
        assert!(
            stdout.contains(name),
            "dev --help missing `{name}`:\n{stdout}"
        );
    }
}
