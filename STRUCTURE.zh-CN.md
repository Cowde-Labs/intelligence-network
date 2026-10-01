# 仓库结构

[English](STRUCTURE.md) | [Português (Brasil)](STRUCTURE.pt-BR.md) | **简体中文** | [Español](STRUCTURE.es.md)

本仓库包含 Rust 发布二进制以及公开的架构和安全文档。开发与研究笔记保存在本地 `internal/` 目录中，不包含在 Git 公开内容中。

## Rust crates

- `crates/node`：进程启动、配置、生命周期、本地管理 socket、任务持久化、分布式训练和 frontier 训练 runtime。
- `crates/protocol`：版本化 wire 类型、capability 描述、任务、artifacts、验证和 codecs。
- `crates/network`：身份、QUIC 传输、签名发现、有界 gossip、peer 表和重连。
- `crates/runtime`：有界执行、准入、取消、进程限制和可选 bubblewrap。
- `crates/intelligence`：推理/评估契约、基于证据的路由、manifests、capability 图、计算规划和训练 fabric 算法。
- `crates/storage`：本地状态、内容寻址 blobs、完整性隔离和 quota 管理。
- `crates/cli`：面向操作者的命令和可复现的本地操作。
- `crates/emulator`：DHT、信任和大规模训练拓扑的确定性实验；模拟结果不代表真实网络执行。

## 公开文档

- `README.md`：安装、操作、架构边界和 release 检查。
- `docs/ARCHITECTURE.md`：crate 职责和去中心化边界。
- `docs/SECURITY.md`：威胁假设、防御措施和操作者限制。

## 规范

内部 Markdown 目录是长期设计的规范来源。`internal/IMPLEMENTATION_MATRIX.md` 记录本版本中各项要求的状态：已实现、实验性、延期或研究中。

## 待解决的架构工作

- 统一本地硬件检测和贡献预算，同时避免泄露敏感信息。
- 将自适应预算接入 runtime 限制和任务调度。
- 在保持 wire 格式版本兼容的前提下，让 planner 考虑通信成本和观测到的可靠性。
- 扩大真实 backend、异构硬件和移动平台的验证。

---

社区翻译。如有差异，以 [STRUCTURE.md](STRUCTURE.md) 为准。
