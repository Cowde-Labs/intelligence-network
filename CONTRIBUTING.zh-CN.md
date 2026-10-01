# 参与贡献

[English](CONTRIBUTING.md) | [Português (Brasil)](CONTRIBUTING.pt-BR.md) | **简体中文** | [Español](CONTRIBUTING.es.md)

贡献应优先提升协议清晰度、互操作性、可衡量的性能以及故障下的生存能力。

提出新抽象前，请说明它具体消除了哪些重复或不兼容。增加中央服务前，请解释为何该行为不能采用 P2P 或重建方式实现。向协议路径添加依赖前，请说明该依赖消失后网络如何运行。

每个面向 peer 的 parser 都必须有边界和畸形输入测试。每个长期运行的任务路径都必须支持取消和恢复。用于路由的每项 capability 声明都必须有获取证据的途径。每条 AI 更新路径都应具备与影响相称的评估和回滚机制。

除非通过明确的 ADR 修改项目根本原则，否则不要添加代币、区块链要求、marketplace 逻辑、Clean Architecture 分层、DI 容器、repository/service/controller 模式或框架式抽象。

## 欢迎解决的问题

- 可移植的资源检测与预算，尊重容器限制、可用内存、系统负载和能源情况。
- 可解释的异构调度，综合考虑数据传输成本、延迟和观察到的可靠性。
- 超越能力公告、真正安全执行加速器 workload 的实现。
- 在公共网络、异构硬件和移动系统上的验证，同时不假设应用可以持续后台运行。

---

社区翻译。如有差异，以 [CONTRIBUTING.md](CONTRIBUTING.md) 为准。
