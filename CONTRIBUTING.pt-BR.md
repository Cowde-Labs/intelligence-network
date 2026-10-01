# Contribuindo

[English](CONTRIBUTING.md) | **Português (Brasil)** | [简体中文](CONTRIBUTING.zh-CN.md) | [Español](CONTRIBUTING.es.md)

As contribuições devem priorizar clareza do protocolo, interoperabilidade, desempenho mensurável e resiliência a falhas.

Antes de propor uma abstração, mostre a duplicação ou incompatibilidade concreta que ela elimina. Antes de adicionar um serviço central, explique por que o comportamento não pode ser P2P ou reconstruído. Antes de adicionar uma dependência ao caminho do protocolo, documente como a rede se comporta se essa dependência desaparecer.

Todo parser exposto a peers precisa de limites e testes com entradas malformadas. Todo caminho de job de longa duração precisa de cancelamento e recuperação. Toda declaração de capability usada no roteamento precisa de um caminho para obter evidências. Todo caminho de atualização de IA precisa de avaliação e rollback proporcionais ao impacto.

Não adicione tokens, requisitos de blockchain, lógica de marketplace, camadas de Clean Architecture, containers DI, padrões repository/service/controller ou abstrações moldadas por frameworks sem alterar a tese fundamental por meio de um ADR explícito.

## Problemas que precisam de contribuição

- Detecção portátil de recursos e orçamentos que respeitem limites de containers, memória disponível, carga e energia.
- Escalonamento heterogêneo explicável, considerando custo de transferência, latência e confiabilidade observada.
- Execução real e segura de workloads em aceleradores, além de anúncios de capacidade.
- Validação em redes públicas, hardware heterogêneo e sistemas móveis, sem prometer execução contínua em segundo plano.

---

Tradução comunitária. Em caso de divergência, prevalece [CONTRIBUTING.md](CONTRIBUTING.md).
