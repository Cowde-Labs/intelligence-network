# Estrutura do repositório

[English](STRUCTURE.md) | **Português (Brasil)** | [简体中文](STRUCTURE.zh-CN.md) | [Español](STRUCTURE.es.md)

Este repositório contém o binário Rust de release e a documentação pública de arquitetura e segurança. Notas de desenvolvimento e pesquisa ficam localmente em `internal/` e não fazem parte da superfície pública do Git.

## Crates Rust

- `crates/node`: inicialização do processo, configuração, ciclo de vida, socket administrativo local, persistência de jobs, treinamento distribuído e runtime de treinamento frontier.
- `crates/protocol`: tipos wire versionados, descrições de capabilities, jobs, artefatos, validação e codecs.
- `crates/network`: identidade, transporte QUIC, descoberta assinada, gossip limitado, tabelas de peers e reconexão.
- `crates/runtime`: execução limitada, admissão, cancelamento, limites de processos e bubblewrap opcional.
- `crates/intelligence`: contratos de inferência/avaliação, roteamento baseado em evidências, manifests, grafo de capabilities, planejamento de compute e algoritmos do fabric de treinamento.
- `crates/storage`: estado local, blobs content-addressed, quarentena de integridade e aplicação de quotas.
- `crates/cli`: comandos do operador e operação local reproduzível.
- `crates/emulator`: experimentos determinísticos de topologias DHT, confiança e treinamento em escala; resultados simulados não representam execuções reais da rede.

## Documentação pública

- `README.md`: instalação, operação, limites de arquitetura e verificações de release.
- `docs/ARCHITECTURE.md`: responsabilidades dos crates e limites de descentralização.
- `docs/SECURITY.md`: pressupostos de ameaça, defesas e limites operacionais.

## Especificações

A árvore Markdown interna é a fonte normativa de projeto de longo prazo. `internal/IMPLEMENTATION_MATRIX.md` registra quais requisitos estão implementados, experimentais, adiados ou em pesquisa nesta versão.

## Trabalho arquitetural pendente

- Unificar detecção local de hardware e orçamento de contribuição sem expor dados sensíveis.
- Integrar o orçamento adaptativo aos limites do runtime e ao escalonamento de jobs.
- Evoluir os planners para custo de comunicação e confiabilidade observada, preservando os formatos wire versionados.
- Ampliar validação real de backends, hardware heterogêneo e plataformas móveis.

---

Tradução comunitária. Em caso de divergência, prevalece [STRUCTURE.md](STRUCTURE.md).
