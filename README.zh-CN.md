# Intelligence Network

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | **简体中文** | [Español](README.es.md)

在互相独立的计算机上运行 AI，无需由中央服务器控制网络。

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
intelligence up
```

Intelligence Network 是一个 Rust 单文件二进制程序，可将计算机变为网络节点。节点通过经过身份验证的 QUIC 相互发现，公布可执行的任务，将推理和训练任务路由给合适的 peer，在多台计算机间分片保存状态，并在 peer 离线时继续运行。无需账户或云服务；除二进制程序外不会下载其他内容。只有在你主动放入模型时，模型才会传输。

这不是加密货币项目：没有代币、区块链、质押或市场。节点因其运营者配置的连接而协作。

## 快速开始

支持 Linux、macOS 和 Windows（x86_64 与 arm64；Windows arm64 需从源码构建）。安装程序从 GitHub Releases 下载发布二进制及校验和，验证后将一个文件安装到 `~/.local/bin`，不安装其他组件。

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Windows PowerShell：

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

```bash
intelligence up              # 创建身份和配置，并在后台启动节点
intelligence status           # 健康状态、peers、任务和资源计数
intelligence infer "hello"    # 通过节点路由任务
intelligence down             # 停止节点
```

`up` 可重复执行：如果节点已运行，会报告其状态。配置位于 `~/.config/intelligence/config.toml`，状态和日志位于 `~/.local/state/intelligence`，模型默认从 `~/.local/share/intelligence/models` 查找；也会尊重 `XDG_*_HOME` 环境变量。

默认情况下，节点提供三个可在任意 CPU 上运行的小型内置能力：`inference.text`、`evaluation.text` 和 `training.reference`。首次运行 `infer` 会使用内置的小型情感分类器，返回标签而不是聊天回复。这是有意设计的：默认安装应能在没有模型文件的 Raspberry Pi 上工作，并在接入真实模型前验证路由、身份和任务流程。llama.cpp 用法见[模型](#模型)。

在一台笔记本上测试 P2P：启动第二个节点，使用独立配置并连接第一个节点。

```bash
intelligence up
intelligence --config /tmp/node-b.toml up --peer 127.0.0.1:4000 --no-default-seeds
intelligence --config /tmp/node-b.toml network peers
intelligence --config /tmp/node-b.toml infer "this is great"
```

节点 B 会发现节点 A、学习其能力，并根据本地证据将任务路由给最合适的 peer。节点 A 停止后，节点 B 会继续提供本地可用的服务。

遇到问题？`intelligence doctor` 会检查配置、身份、存储权限、QUIC endpoint、NAT 可达性、检测到的硬件、本地模型文件和节点健康状态，并给出修复建议。

## 功能

- **无需目录服务即可发现 peers。** Bootstrap 地址只是提示，并非权威来源。节点发现一个 peer 后，可通过签名 peer 交换和有界 Kademlia 风格 DHT 学习其他节点，并在重启后保留记录。
- **跨机器推理。** 在本地提交任务，节点可自行执行，也可将任务路由到公布对应能力且通过本地信任策略的 peer。结果以流式返回，并受大小和期限限制。
- **向网络提供真实模型。** 将已有的 `llama-cli` 二进制和 `.gguf` 文件配置为 `llama_cpp` capability。远程任务在 bubblewrap 中运行。
- **跨机器训练。** 参考训练通过真实 worker 进程运行，包含参数分片、优化器状态复制、内容寻址 checkpoint、分层聚合，以及 tensor/pipeline 并行参考操作。单个协调器不会看到所有更新，worker 也不需要完整模型。
- **应对故障。** 测试覆盖 bootstrap peer、协调器、shard 所有者、relay 或 pipeline 阶段中途退出，以及重连、退避、按 term 替换协调器、提升副本和恢复 checkpoint。
- **混合硬件。** 节点公布 compute backend（CPU；启用对应 feature 后可检测 CUDA、ROCm 和 Metal）、数据格式与内存，planner 据此放置 shards 并选择策略。
- **安全传输 artifacts。** 模型、checkpoint 和数据集使用 BLAKE3 内容寻址，通过可续传分块传输，到达后校验；完整性或 quota 检查失败的内容会被隔离。
- **由操作者掌控。** 每项 capability 都是可选且有界的，可仅本地使用或公开。不会隐式下载内容。节点无需 peers 也能运行。

## 工作原理

每个节点都有持久的 Ed25519 身份。Peers 通过绑定该身份的 QUIC 通信，因此收到的地址、capability 公告、DHT 记录、训练更新和 checkpoint 均由发送 peer 签名。

Capabilities 是声明。信任是本地且基于证据的：节点分别记录直接观察和第三方报告，并仅将任务路由给符合本地策略的 peers。系统没有可供操纵的全局信誉分数。

任务明确限制输入、输出、内存、CPU 和期限。Runtime 将任务加入有界队列，支持取消，并持久化状态以便节点重启后恢复。

分布式训练将模型状态划分为由 workers 持有的 shards，并将 workers 分组交给聚合器，使协调器只看到聚合结果；优化器状态和 checkpoint 会在 peers 间复制。角色持有者离线时，使用单调递增的 term 轮换协调角色。

```text
所有更新 -> 单一协调器        否
单一优化器状态权威            否
单一 checkpoint 权威          否
全局同步屏障                  否
模型必须完整放入一个 worker    否
```

## 安装

### 发布二进制

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
```

Windows PowerShell：

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

使用 `INTELLIGENCE_VERSION=v1.0.0` 固定版本，或镜像发布文件并设置 `INTELLIGENCE_RELEASE_BASE_URL`。如果校验和不匹配或压缩包包含不安全路径，安装程序会拒绝安装。

### 从源码构建

需要 Rust 1.85 或更新版本。

```bash
git clone https://github.com/Cowde-Labs/intelligence-network
cd intelligence-network
cargo build --release -p intelligence-cli
./target/release/intelligence up
```

使用 `--features cuda`、`--features rocm` 或 `--features metal` 启用加速器检测。节点会探测驱动并公布 backend，但不会安装厂商 SDK。

### 作为服务运行

```bash
intelligence service install    # 安装用户级服务：Linux systemd、macOS launchd 或 Windows 计划任务；无需 root
```

系统范围内以受限服务账号部署，请参阅 [`infra/systemd`](infra/systemd)。

## 运行

`intelligence up` 在后台启动节点，并在健康检查通过后返回。`intelligence run` 在前台运行节点，适用于 supervisor 或调试场景。

```bash
intelligence up --peer 203.0.113.10:4000 --peer 198.51.100.20:4000
intelligence up --no-default-seeds                 # 仅使用指定 peers
intelligence up --listen-addr 0.0.0.0:4000         # 接受其他计算机的连接
intelligence up --capability inference.text        # 启用部分内置能力
intelligence --config ./node.toml up               # 启动第二个节点或使用其他配置路径
```

默认监听地址为 `127.0.0.1:4000`。若要接受其他计算机连接，需绑定可路由地址；若要向 peers 公布该地址，还需设置 `advertise_addr`。绑定 loopback 的节点会自动接受私有 peer 地址；其他节点默认会过滤此类地址，LAN 场景可设置 `allow_private_addresses = true`。

编译时默认 seeds 为 `bootstrap-1.intelligence.network:4000` 和 `bootstrap-2.intelligence.network:4000`。它们只是可替换提示；不可用时节点会报告并继续运行。使用 `--no-default-seeds` 或 `INTELLIGENCE_DEFAULT_SEEDS=host:port,host:port` 替换它们。

## 模型

节点不会下载模型权重。你提供文件（接受任意文件类型），节点计算其 hash 并将其作为内容寻址 artifact 管理。

```bash
intelligence model list
intelligence model add ~/models/llama-3.2-1b-q4.gguf
intelligence model add ./model.bin --identity local.mymodel.v1 --format opaque
intelligence artifact inspect --artifact <hash>
intelligence artifact fetch --peer <node-id> --artifact <hash>
```

要运行模型，请在配置中添加 `llama_cpp` capability。你提供 `llama-cli` 二进制和模型路径；节点提供 sandbox、资源边界和路由：

```toml
[[capabilities]]
name = "inference.llama-cpp"
version = 1
public = true
accept_remote_jobs = true
kind = "llama_cpp"
program = "/usr/local/bin/llama-cli"
model_path = "/srv/models/llama-3.2-1b-q4.gguf"
args = ["--ctx-size", "2048"]
sandbox = "bubblewrap"
max_input_bytes = 65536
max_output_bytes = 65536
memory_bytes = 4294967296
cpu_millis = 30000
```

```bash
intelligence infer --capability inference.llama-cpp "用一段话解释 QUIC"
intelligence infer --capability inference.llama-cpp --local-only "..."
```

公开且接受远程任务的 `llama_cpp` 或 `process` capability 必须使用 `sandbox = "bubblewrap"`，否则节点会拒绝配置。仅供本地使用的模型应设为 `public = false`。llama.cpp adapter 标记为实验性：进程隔离、限制和配置验证已有测试，但尚未覆盖大量模型和构建版本。

## CLI 命令

全局参数：`--config <path>`（或 `INTELLIGENCE_CONFIG`）以及在支持的命令中输出机器可读结果的 `--json`。

| 类别 | 常用命令与说明 |
|---|---|
| 生命周期 | `up` 启动节点；`down` 停止；`run` 前台启动；`init` 创建配置和身份；`service install` 安装用户服务；`version` 显示版本。 |
| 检查 | `status` 查看健康、peers、jobs 和资源；`doctor` 检查配置和硬件；`config` 查看生效配置；`jobs` 查看任务；`identity` 查看身份；`network peers/capabilities/stats` 查看网络信息；`trust inspect --subject <node-id>` 查看本地信任决策。 |
| 工作 | `infer` 提交推理；`evaluate` 评估样本并记录证据；`jobs cancel --job-id <id>` 取消任务。 |
| 模型和 artifacts | `model list/add/register` 管理本地模型；`artifact inspect/fetch` 检查或从 peer 获取 artifact。 |
| 身份和 DHT | `identity rotate` 轮换身份；`network publish/lookup/find` 发布、查询记录或发现邻近 peers。 |
| 训练 | `train` 运行训练；`train start/plan/replan/activate/status/cancel` 管理计划与任务；`train migrate/replicate/reconcile` 管理 shards、状态和分支；`train reference` 运行小型参考训练。 |

运行 `intelligence <command> --help` 查看完整参数。1.0 之前的命令名称仍作为隐藏别名可用。

## 高级配置

`intelligence config` 显示应用环境变量和默认值后的配置。每个字段都可在 TOML 中设置，也可通过相应的 `INTELLIGENCE_*` 环境变量覆盖，例如 `INTELLIGENCE_LISTEN_ADDR`、`INTELLIGENCE_BOOTSTRAP`、`INTELLIGENCE_STORAGE_QUOTA_BYTES` 和 `INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS`。完整注释示例见 [`config/node.toml.example`](config/node.toml.example)。

**网络：** `listen_addr`、`advertise_addr`、`bootstrap`、`allow_private_addresses`、`max_connections`、`peer_ttl_seconds`。Hole punching 默认启用且尝试次数有限；并不保证能穿透恶意或特殊配置的 NAT。

**Relays：** 任意节点都可成为 relay。设置 `relay_enabled`、`relay_max_sessions` 和 `relay_max_bytes`；客户端可配置 `relay_addresses` 和 `prefer_relay`。Relays 仅转发经过认证的加密 envelope，无法读取任务或成为权威节点。

**DHT：** `dht_enabled`、`dht_k`、`dht_alpha`、`dht_max_records`。DHT 仅使用直接认证的 peers，不经过 relays。

**存储：** `[storage] quota_bytes` 与 `max_artifact_bytes`。完整性检查失败的 artifacts 会被隔离，而不会删除。

**Runtime：** `[runtime] max_queued_jobs`、`max_concurrent_jobs`、`max_input_bytes`、`max_output_bytes`、`default_timeout_ms`、`process_memory_bytes`、`process_cpu_seconds` 和 `work_dir`。

**Capabilities：** 每个 `[[capabilities]]` 的类型为 `builtin_text`、`builtin_training`、`process` 或 `llama_cpp`，sandbox 为 `trusted_local` 或 `bubblewrap`，并有独立资源限制和 `public`/`accept_remote_jobs` 开关。元数据会被复制到签名公告中用于规划；它们是声明，peer 会结合观察证据评估。

**训练：** `training_memory_bytes` 是 worker 执行所分配 shard 时可用的本地状态预算。`training_window_delay_ms` 用于实验性地模拟 straggler，正常运行请设为零。

## 架构

七个主要 Rust crate，依赖方向单一：

```text
protocol      版本化 wire 类型、有界验证、manifests 和训练消息
storage       本地状态、内容寻址 artifacts、隔离和 quota
runtime       admission、deadline、取消、进程限制和 bubblewrap
network       Ed25519 身份、QUIC、签名发现、gossip、DHT 和 relays
intelligence  能力证据、本地信任、compute 规划和训练算法
node          组合上述组件、持久化任务并提供本地管理 socket
cli           intelligence 命令行二进制
```

Wire 协议当前为 1.6；每个面向 peers 的 parser 都有边界并经过 fuzz 测试。CLI 通过本地管理通道与节点通信：Linux/macOS 使用 Unix socket，Windows 使用 named pipe；没有 TCP 监听端口。

设计原则：没有中央控制面、强制依赖的第一方基础设施、经济层，或会因某个协议依赖消失而使网络瘫痪的组件。独立的确定性 emulator（`crates/emulator`）模拟 100 到 100,000 个逻辑 peers 的发现和训练拓扑，不会启动同等数量的进程。其结果标记为 `EMULATED`。

详见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 安全与限制

所有远程 peer 都被视为不可信输入。身份和记录经过签名，artifacts 经过 hash 校验，replay 和 equivocation 会被拒绝；远程工作受消息大小、队列、并发、内存、速率和期限限制。远程外部进程只能在 bubblewrap 中执行。完整安全说明见 [docs/SECURITY.md](docs/SECURITY.md)。

- **网络验证：** 已在单机和受控实验室验证，尚未在公共互联网验证。跨独立运营者部署、真实恶意 NAT 和公共规模行为仍需测量。
- **训练规模：** 分片、复制、协调器替换、tensor/pipeline 并行和分支协调已在真实进程中以有界 CPU 确定性负载运行；这不是前沿模型训练。
- **信任：** 信任是本地的，不是全局的。系统无法证明一个身份对应一个自然人，也不承诺超出已测试场景的 Sybil 抵抗能力。
- **拜占庭容错：** 能力有限。中位数聚合和更新验证会提高投毒成本，但不构成 BFT。
- **加速器：** backend 可被检测并公布用于规划，但 node 不提供通用 GPU 推理；当前 GPU 推理由操作者配置的 llama.cpp 进程执行。
- **平台：** 发布版支持 Linux、macOS 和 Windows。bubblewrap 远程隔离执行及 systemd 服务仅支持 Linux。macOS/Windows 通过 CI 验证，尚无长期部署验证。

## 需要解决的问题

以下是根据当前仓库识别出的工作项，并非已实现的功能：

- 建立可信的本地资源 profile 与自适应预算，考虑容器限制、系统负载、可用内存、能源和温度。
- 让 scheduler 综合比较计算、每个 shard 的内存、延迟、带宽、数据传输成本和观察到的可靠性。
- 用可更新、可解释的观测替代不足的硬件声明，同时避免公开敏感本地信息。
- 扩大加速器的真实执行能力，明确区分 backend 检测/公告与生产级 kernel 支持。
- 在更多操作系统、网络和异构硬件上验证安全、连接性与故障行为。
- 为移动设备准备机会式参与模式，不假设应用能持续在后台运行。

## 开发

```bash
export CARGO_TARGET_DIR=.cache/rust-target
cargo build -p intelligence-cli
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

启动真实 node 进程的集成场景：

```bash
./scripts/compatibility-smoke.sh
./scripts/local-testnet.sh
```

受控网络实验室、规模 emulator、fuzz 和 release 流程见 [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)。

贡献应让协议更清晰、更互操作或更能承受故障。面向 peer 的 parser 需要边界和畸形输入测试；长时间运行的任务路径需要取消与恢复；用于路由的 capability 声明需要证据支持。不要在没有改变项目基本主张的 ADR 前提下引入代币、区块链、市场或中央服务。请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可

仅采用 GNU Affero General Public License v3.0。详见 [LICENSE](LICENSE)。

---

此翻译由社区维护，可能落后于英文原文。如有差异，以 [README.md](README.md) 为准。
