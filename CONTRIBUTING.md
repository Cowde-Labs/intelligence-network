# Contributing

[English](CONTRIBUTING.md) | [Português (Brasil)](CONTRIBUTING.pt-BR.md) | [简体中文](CONTRIBUTING.zh-CN.md) | [Español](CONTRIBUTING.es.md)

Contributions should optimize for protocol clarity, interoperability, measurable performance, and survivability under failure.

Before proposing a new abstraction, show the concrete duplication or incompatibility it removes. Before adding a central service, explain why the behavior cannot be peer-to-peer or reconstructable. Before adding a new dependency to the protocol path, document how the network behaves if that dependency disappears.

Every peer-facing parser needs bounds and malformed-input tests. Every long-running job path needs cancellation and recovery behavior. Every capability claim used for routing needs a path to evidence. Every AI update path needs evaluation and rollback appropriate to its impact.

Do not add tokens, blockchain requirements, marketplace logic, Clean Architecture layers, DI containers, repository/service/controller patterns, or framework-shaped abstractions without changing the foundational thesis through an explicit ADR.

## Problems needing contributions

- Portable resource detection and budgets that respect container limits, available memory, system load, and energy.
- Explainable heterogeneous scheduling that accounts for transfer cost, latency, and observed reliability.
- Real and safe accelerator workload execution beyond capability advertising.
- Validation on public networks, heterogeneous hardware, and mobile systems without assuming continuous background execution.
