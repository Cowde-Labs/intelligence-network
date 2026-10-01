# Estructura del repositorio

[English](STRUCTURE.md) | [Português (Brasil)](STRUCTURE.pt-BR.md) | [简体中文](STRUCTURE.zh-CN.md) | **Español**

Este repositorio contiene el binario Rust de release y la documentación pública de arquitectura y seguridad. Las notas de desarrollo e investigación se conservan localmente en `internal/` y no forman parte de la superficie pública de Git.

## Crates Rust

- `crates/node`: inicio del proceso, configuración, ciclo de vida, socket administrativo local, persistencia de trabajos, entrenamiento distribuido y runtime de entrenamiento frontier.
- `crates/protocol`: tipos wire versionados, descripciones de capabilities, trabajos, artifacts, validación y codecs.
- `crates/network`: identidad, transporte QUIC, descubrimiento firmado, gossip acotado, tablas de peers y reconexión.
- `crates/runtime`: ejecución acotada, admisión, cancelación, límites de procesos y bubblewrap opcional.
- `crates/intelligence`: contratos de inferencia/evaluación, enrutamiento basado en evidencias, manifests, grafo de capabilities, planificación de cómputo y algoritmos del fabric de entrenamiento.
- `crates/storage`: estado local, blobs content-addressed, cuarentena de integridad y cuotas.
- `crates/cli`: comandos para operadores y operación local reproducible.
- `crates/emulator`: experimentos deterministas de topologías DHT, confianza y entrenamiento a escala; los resultados simulados no representan ejecuciones reales de la red.

## Documentación pública

- `README.md`: instalación, operación, límites de arquitectura y verificaciones de release.
- `docs/ARCHITECTURE.md`: responsabilidades de los crates y límites de descentralización.
- `docs/SECURITY.md`: supuestos de amenazas, defensas y límites operativos.

## Especificaciones

El árbol Markdown interno es la fuente normativa del diseño a largo plazo. `internal/IMPLEMENTATION_MATRIX.md` registra qué requisitos están implementados, son experimentales, están aplazados o siguen en investigación en esta versión.

## Trabajo arquitectónico pendiente

- Unificar la detección local de hardware y el presupuesto de contribución sin exponer datos sensibles.
- Integrar el presupuesto adaptativo con los límites del runtime y la planificación de trabajos.
- Evolucionar los planners para considerar el coste de comunicación y la fiabilidad observada, preservando la compatibilidad de los formatos wire versionados.
- Ampliar la validación real de backends, hardware heterogéneo y plataformas móviles.

---

Traducción comunitaria. En caso de discrepancia, prevalece [STRUCTURE.md](STRUCTURE.md).
