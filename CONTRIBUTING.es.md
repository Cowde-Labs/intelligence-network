# Contribuir

[English](CONTRIBUTING.md) | [Português (Brasil)](CONTRIBUTING.pt-BR.md) | [简体中文](CONTRIBUTING.zh-CN.md) | **Español**

Las contribuciones deben priorizar la claridad del protocolo, la interoperabilidad, el rendimiento medible y la resiliencia ante fallos.

Antes de proponer una abstracción, muestra la duplicación o incompatibilidad concreta que elimina. Antes de añadir un servicio central, explica por qué ese comportamiento no puede ser P2P o reconstruible. Antes de añadir una dependencia a la ruta del protocolo, documenta cómo se comportaría la red si esa dependencia desaparece.

Todo parser expuesto a peers necesita límites y pruebas con entradas malformadas. Toda ruta de trabajos de larga duración necesita cancelación y recuperación. Toda declaración de capability usada para enrutar debe tener una vía para obtener evidencias. Toda ruta de actualización de IA debe contar con evaluación y rollback proporcionales a su impacto.

No añadas tokens, requisitos de blockchain, lógica de marketplace, capas de Clean Architecture, contenedores DI, patrones repository/service/controller ni abstracciones propias de frameworks sin cambiar la tesis fundamental mediante un ADR explícito.

## Problemas que necesitan contribuciones

- Detección portátil de recursos y presupuestos que respeten límites de contenedores, memoria disponible, carga y energía.
- Planificación heterogénea explicable que considere el coste de transferencia, la latencia y la fiabilidad observada.
- Ejecución real y segura de cargas en aceleradores, más allá de anunciar capacidades.
- Validación en redes públicas, hardware heterogéneo y sistemas móviles, sin asumir ejecución continua en segundo plano.

---

Traducción comunitaria. En caso de discrepancia, prevalece [CONTRIBUTING.md](CONTRIBUTING.md).
