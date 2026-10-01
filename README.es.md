# Intelligence Network

[English](README.md) | [Português (Brasil)](README.pt-BR.md) | [简体中文](README.zh-CN.md) | **Español**

Ejecuta IA en máquinas independientes sin un servidor central que controle la red.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
intelligence up
```

Intelligence Network es un único binario Rust que convierte una máquina en un nodo. Los nodos se descubren mediante QUIC autenticado, anuncian lo que pueden ejecutar, enrutan trabajos de inferencia y entrenamiento a los peers adecuados, distribuyen el estado entre máquinas y siguen funcionando cuando los peers desaparecen. No hace falta crear una cuenta ni acceder a una nube, y no se descarga nada salvo el binario. Los modelos solo se transfieren si tú los incorporas.

No es un proyecto de criptomonedas: no hay token, blockchain, staking ni marketplace. Los nodos cooperan porque sus operadores los conectan.

## Inicio rápido

Linux, macOS y Windows (x86_64 y arm64; Windows arm64 se compila desde el código fuente). El instalador descarga el binario de release y su checksum de GitHub Releases, lo verifica e instala un único archivo en `~/.local/bin`. No instala nada más.

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

En Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

```bash
intelligence up              # crea identidad y configuración e inicia el nodo en segundo plano
intelligence status           # salud, peers, trabajos y contadores de recursos
intelligence infer "hello"    # enruta un trabajo a través del nodo
intelligence down             # detiene el nodo
```

`up` es idempotente: si se ejecuta otra vez, informa que el nodo ya está activo. La configuración se guarda en `~/.config/intelligence/config.toml`, el estado y los logs en `~/.local/state/intelligence`, y los modelos se buscan en `~/.local/share/intelligence/models` (se respetan las variables `XDG_*_HOME`).

De forma predeterminada, el nodo ofrece tres capacidades pequeñas que funcionan en cualquier CPU: `inference.text`, `evaluation.text` y `training.reference`. El primer comando `infer` usa un pequeño clasificador de sentimientos integrado y devuelve una etiqueta, no una respuesta de chat. Es intencional: la instalación predeterminada debe funcionar en una Raspberry Pi sin archivos de modelo y demostrar el flujo de enrutamiento, identidad y trabajos antes de configurar un modelo real. Consulta [Modelos](#modelos) para la ruta de llama.cpp.

Para probar P2P en una laptop, inicia un segundo nodo con su propia configuración y conéctalo al primero:

```bash
intelligence up
intelligence --config /tmp/node-b.toml up --peer 127.0.0.1:4000 --no-default-seeds
intelligence --config /tmp/node-b.toml network peers
intelligence --config /tmp/node-b.toml infer "this is great"
```

El nodo B descubre el nodo A, aprende sus capacidades y enruta el trabajo al peer con las mejores evidencias locales. Si el nodo A se detiene, el nodo B sigue ofreciendo lo que puede por sí mismo.

¿Algo falla? `intelligence doctor` comprueba la configuración, identidad, permisos de almacenamiento, endpoint QUIC, alcance a través de NAT, hardware detectado, modelos locales y salud del nodo, y ofrece indicaciones para corregirlo.

## Qué puedes hacer

- **Descubrir peers sin un servicio de directorio.** Las direcciones bootstrap son pistas, no autoridades. Tras conocer un peer, el nodo aprende sobre otros mediante intercambio firmado y una DHT acotada al estilo Kademlia, y los recuerda después de reiniciar.
- **Ejecutar inferencia entre máquinas.** Envía un trabajo local; el nodo puede ejecutarlo o enviarlo a un peer que anuncie la capacidad y haya obtenido confianza local. Los resultados vuelven en streaming, con límites de tamaño y plazo.
- **Ofrecer un modelo real a la red.** Configura una capability `llama_cpp` con un binario `llama-cli` y un archivo `.gguf` que ya tengas. Los trabajos remotos se ejecutan dentro de bubblewrap.
- **Entrenar entre máquinas.** Los trabajos de referencia usan procesos worker reales, sharding de parámetros, estado del optimizador replicado, checkpoints content-addressed, agregación jerárquica y operaciones de referencia tensor/pipeline-parallel. Ningún coordinador único ve todas las actualizaciones y ningún worker necesita el modelo completo.
- **Sobrevivir a fallos.** Las pruebas interrumpen peers bootstrap, coordinadores, propietarios de shards, relays o etapas de pipeline. También ejercitan reconexión, backoff, sustitución del coordinador por term, promoción de réplicas y recuperación de checkpoints.
- **Combinar hardware.** Los nodos anuncian backends de cómputo (CPU; CUDA, ROCm y Metal al compilar con esas features), formatos y memoria. El planner usa estos datos para ubicar shards y elegir estrategias.
- **Mover artifacts de forma segura.** Modelos, checkpoints y datasets se identifican mediante BLAKE3, se transfieren en bloques reanudables, se verifican al llegar y se ponen en cuarentena si fallan la integridad o la cuota.
- **Mantener el control.** Cada capability es opcional, acotada y local o pública. No se descarga nada de forma implícita. Un nodo funciona aunque no tenga peers.

## Cómo funciona

Cada nodo tiene una identidad Ed25519 persistente. Los peers se comunican por QUIC asociado a esa identidad. Por eso, cada registro recibido está firmado por el peer que lo produjo: direcciones, anuncios de capabilities, registros DHT, actualizaciones de entrenamiento y checkpoints.

Las capabilities son declaraciones. La confianza es local y basada en evidencias: cada nodo distingue lo que observó directamente de lo que le contaron otros, y solo enruta trabajos a peers que cumplen su política. No existe una puntuación global de reputación.

Cada trabajo tiene límites explícitos de entrada, salida, memoria, CPU y plazo. El runtime lo admite en una cola acotada, permite cancelarlo y persiste su estado para recuperarlo tras un reinicio.

El entrenamiento distribuido divide el estado del modelo en shards asignados a workers y los agrupa bajo agregadores para que el coordinador solo vea agregados. El estado del optimizador y los checkpoints se replican entre peers, y las funciones de coordinación rotan con terms monotónicos cuando desaparece quien las ocupa.

```text
todas las actualizaciones -> un coordinador      NO
una autoridad del estado del optimizador         NO
una autoridad de checkpoints                     NO
barrera síncrona global                          NO
el modelo debe caber en un worker                 NO
```

## Instalación

### Binario de release

```bash
curl -fsSL https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.sh | sh
```

En Windows (PowerShell):

```powershell
irm https://raw.githubusercontent.com/Cowde-Labs/intelligence-network/main/install.ps1 | iex
```

Fija una versión con `INTELLIGENCE_VERSION=v1.0.0`, o replica los artifacts y configura `INTELLIGENCE_RELEASE_BASE_URL`. El instalador rechaza checksums incorrectos y archivos comprimidos con rutas inseguras.

### Desde el código fuente

Rust 1.85 o posterior.

```bash
git clone https://github.com/Cowde-Labs/intelligence-network
cd intelligence-network
cargo build --release -p intelligence-cli
./target/release/intelligence up
```

Activa la detección de aceleradores con `--features cuda`, `--features rocm` o `--features metal`. El nodo comprueba el driver y anuncia el backend; no instala ningún SDK del proveedor.

### Como servicio

```bash
intelligence service install    # servicio de usuario: systemd en Linux, launchd en macOS o tarea programada en Windows; sin root
```

Para una instalación de sistema con una cuenta de servicio restringida, consulta [`infra/systemd`](infra/systemd).

## Ejecución

`intelligence up` inicia el nodo en segundo plano y retorna cuando está saludable. `intelligence run` ejecuta el mismo nodo en primer plano, útil bajo un supervisor o durante la depuración.

```bash
intelligence up --peer 203.0.113.10:4000 --peer 198.51.100.20:4000
intelligence up --no-default-seeds                 # usa únicamente los peers indicados
intelligence up --listen-addr 0.0.0.0:4000         # acepta conexiones de otras máquinas
intelligence up --capability inference.text        # habilita algunas capacidades integradas
intelligence --config ./node.toml up               # inicia otro nodo o usa otra configuración
```

La dirección de escucha predeterminada es `127.0.0.1:4000`. Para recibir conexiones desde otra máquina, vincula el nodo a una dirección enrutable y configura `advertise_addr` si quieres anunciarla a los peers. Un nodo vinculado a loopback acepta automáticamente direcciones privadas; los demás las filtran salvo que se configure `allow_private_addresses = true`, apropiado para una LAN.

Las seeds predeterminadas compiladas son `bootstrap-1.intelligence.network:4000` y `bootstrap-2.intelligence.network:4000`. Son pistas reemplazables: si no están disponibles, el nodo lo indica y continúa. Usa `--no-default-seeds` o `INTELLIGENCE_DEFAULT_SEEDS=host:port,host:port` para reemplazarlas.

## Modelos

El nodo nunca descarga pesos. Tú proporcionas el archivo (se acepta cualquier tipo), el nodo calcula su hash y lo registra como artifact content-addressed.

```bash
intelligence model list
intelligence model add ~/models/llama-3.2-1b-q4.gguf
intelligence model add ./model.bin --identity local.mymodel.v1 --format opaque
intelligence artifact inspect --artifact <hash>
intelligence artifact fetch --peer <node-id> --artifact <hash>
```

Para ejecutar un modelo, añade una capability `llama_cpp` a la configuración. Tú proporcionas el binario `llama-cli` y la ruta del modelo; el nodo aporta sandbox, límites y enrutamiento:

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
intelligence infer --capability inference.llama-cpp "Explica QUIC en un párrafo"
intelligence infer --capability inference.llama-cpp --local-only "..."
```

Una capability `llama_cpp` o `process` pública que acepte trabajos remotos debe usar `sandbox = "bubblewrap"`; de lo contrario, el nodo rechaza la configuración. Usa `public = false` para mantener un modelo solo local. El adaptador llama.cpp es experimental: se prueban el límite de proceso, los límites de recursos y la validación, pero aún no se ha probado con una amplia variedad de modelos y compilaciones.

## Referencia de la CLI

Opciones globales: `--config <path>` (o `INTELLIGENCE_CONFIG`) y `--json` para salida legible por máquina en los comandos compatibles.

| Categoría | Comandos y función |
|---|---|
| Ciclo de vida | `up` inicia el nodo; `down` lo detiene; `run` lo ejecuta en primer plano; `init` crea configuración e identidad; `service install` instala un servicio de usuario; `version` muestra versiones. |
| Inspección | `status` muestra salud, peers, trabajos y recursos; `doctor` comprueba configuración y hardware; `config`, `jobs` e `identity` muestran configuración, trabajos e identidad; `network peers/capabilities/stats` muestra datos de red; `trust inspect --subject <node-id>` muestra confianza local. |
| Trabajo | `infer` envía inferencia; `evaluate` evalúa una muestra y registra evidencia; `jobs cancel --job-id <id>` cancela un trabajo. |
| Modelos y artifacts | `model list/add/register` gestiona modelos; `artifact inspect/fetch` inspecciona o descarga artifacts desde un peer. |
| Identidad y DHT | `identity rotate` rota la identidad; `network publish/lookup/find` publica, consulta o busca peers cercanos. |
| Entrenamiento | `train` ejecuta entrenamiento; `train start/plan/replan/activate/status/cancel` administra trabajos y planes; `train migrate/replicate/reconcile` gestiona shards, estado y ramas; `train reference` ejecuta el pequeño entrenamiento de referencia. |

Ejecuta `intelligence <comando> --help` para ver todas las opciones. Los nombres de comandos anteriores a 1.0 siguen disponibles como aliases ocultos.

## Configuración avanzada

`intelligence config` muestra la configuración efectiva. Cada campo puede definirse en TOML o sobrescribirse mediante una variable `INTELLIGENCE_*`, como `INTELLIGENCE_LISTEN_ADDR`, `INTELLIGENCE_BOOTSTRAP`, `INTELLIGENCE_STORAGE_QUOTA_BYTES` e `INTELLIGENCE_RUNTIME_MAX_CONCURRENT_JOBS`. Hay un ejemplo comentado en [`config/node.toml.example`](config/node.toml.example).

**Red:** `listen_addr`, `advertise_addr`, `bootstrap`, `allow_private_addresses`, `max_connections` y `peer_ttl_seconds`. Hole punching está habilitado por defecto y tiene intentos acotados; no se garantiza frente a NAT hostiles.

**Relays:** cualquier nodo puede funcionar como relay. Configura `relay_enabled`, `relay_max_sessions` y `relay_max_bytes`; los clientes pueden declarar `relay_addresses` y activar `prefer_relay`. Los relays reenvían envelopes autenticados opacos, no leen trabajos ni se convierten en autoridades.

**DHT:** `dht_enabled`, `dht_k`, `dht_alpha` y `dht_max_records`. La DHT utiliza solo peers directos autenticados, no relays.

**Almacenamiento:** `[storage] quota_bytes` y `max_artifact_bytes`. Los artifacts que no superan la verificación de integridad se ponen en cuarentena, no se eliminan.

**Runtime:** `[runtime] max_queued_jobs`, `max_concurrent_jobs`, `max_input_bytes`, `max_output_bytes`, `default_timeout_ms`, `process_memory_bytes`, `process_cpu_seconds` y `work_dir`.

**Capabilities:** cada `[[capabilities]]` usa `builtin_text`, `builtin_training`, `process` o `llama_cpp`, sandbox `trusted_local` o `bubblewrap`, límites propios y flags `public`/`accept_remote_jobs`. Los metadatos se copian en anuncios firmados y se usan para planificar; son declaraciones que se contrastan con evidencias observadas.

**Entrenamiento:** `training_memory_bytes` es el presupuesto local de estado que debe respetar el shard asignado. `training_window_delay_ms` inyecta retrasos de worker para experimentos; déjalo en cero en producción normal.

## Arquitectura

Siete crates Rust principales con dependencias unidireccionales:

```text
protocol      tipos wire versionados, validación acotada, manifests y mensajes de entrenamiento
storage       estado local, artifacts content-addressed, cuarentena y cuotas
runtime       admisión, plazos, cancelación, límites de procesos y bubblewrap
network       identidad Ed25519, QUIC, descubrimiento firmado, gossip, DHT y relays
intelligence  evidencias, confianza local, planificación de cómputo y algoritmos de entrenamiento
node          compone los crates, persiste trabajos y ofrece el socket administrativo local
cli           binario de línea de comandos intelligence
```

El protocolo wire está versionado (actualmente 1.6) y cada parser expuesto a peers tiene límites y pruebas de fuzz. La CLI se comunica con el node mediante un canal administrativo local: Unix socket en Linux/macOS y named pipe en Windows; no hay listener TCP.

Reglas del proyecto: sin plano de control central, infraestructura propia obligatoria, capa económica ni dependencias de protocolo cuya desaparición derribe la red. Un emulator determinista separado (`crates/emulator`) modela topologías de descubrimiento y entrenamiento con 100 a 100.000 peers lógicos sin iniciar tantos procesos. Sus resultados se etiquetan como `EMULATED`.

Más detalles en [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Seguridad y limitaciones

Todo peer remoto se trata como entrada no confiable. Identidades y registros se firman, los artifacts se verifican por hash, y se rechazan replay y equivocation. Los trabajos remotos tienen límites de tamaño, cola, concurrencia, memoria, tasa y plazo. La ejecución remota de procesos externos solo se permite dentro de bubblewrap. Consulta [docs/SECURITY.md](docs/SECURITY.md).

- **Validación de red:** validado en una máquina y en un laboratorio controlado, no en Internet pública. Faltan mediciones entre operadores independientes, con NAT hostiles reales y a escala pública.
- **Escala del entrenamiento:** sharding, replicación, sustitución de coordinador, paralelismo tensor/pipeline y reconciliación de ramas funcionan en procesos reales con cargas CPU deterministas y acotadas. No es entrenamiento de modelos de frontera.
- **Confianza:** es local, no global. El modelo no prueba una persona por identidad ni garantiza resistencia Sybil fuera de los escenarios probados.
- **Tolerancia bizantina:** limitada. La agregación por mediana y validación de actualizaciones elevan el coste del envenenamiento, pero no son BFT.
- **Aceleradores:** se anuncian para planificación, pero el node no ejecuta inferencia GPU general. La inferencia GPU actual ocurre en el proceso llama.cpp del operador.
- **Plataformas:** hay binarios para Linux, macOS y Windows. La ejecución remota aislada con bubblewrap y el servicio systemd son exclusivos de Linux. macOS/Windows se verifican en CI, pero todavía no en despliegues prolongados.

## Problemas pendientes

Estos puntos de trabajo se identifican a partir del estado actual del repositorio; no son funcionalidades ya implementadas:

- Crear un perfil local fiable de recursos y un presupuesto adaptativo que considere límites de contenedores, carga, memoria disponible, energía y temperatura.
- Hacer que el scheduler compare coste de cómputo, memoria por shard, latencia, ancho de banda, transferencia de datos y fiabilidad observada.
- Sustituir declaraciones de hardware insuficientes por observaciones actualizadas y explicables, sin publicar información local sensible.
- Ampliar la ejecución real de aceleradores y distinguir claramente la detección/anuncio de backends de los kernels de producción.
- Validar seguridad, conectividad y comportamiento ante fallos en más sistemas, redes y hardware heterogéneo.
- Preparar participación móvil oportunista sin suponer ejecución continua en segundo plano.

## Desarrollo

```bash
export CARGO_TARGET_DIR=.cache/rust-target
cargo build -p intelligence-cli
cargo test --workspace -- --test-threads=1
cargo clippy --workspace --all-targets -- -D warnings
```

Escenarios de integración que inician procesos node reales:

```bash
./scripts/compatibility-smoke.sh
./scripts/local-testnet.sh
```

El laboratorio de red controlado, los emuladores de escala, fuzzing y el pipeline de release se documentan en [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

Las contribuciones deben hacer que el protocolo sea más claro, interoperable o resistente a fallos. Los parsers expuestos a peers necesitan límites y pruebas con entradas malformadas; las rutas de trabajos largos necesitan cancelación y recuperación; las declaraciones de capacidades usadas para enrutar necesitan un camino hacia evidencias. No añadas tokens, blockchains, marketplaces ni servicios centrales sin un ADR que cambie la tesis del proyecto. Consulta [CONTRIBUTING.md](CONTRIBUTING.md).

## Licencia

Únicamente GNU Affero General Public License v3.0. Consulta [LICENSE](LICENSE).

---

Esta traducción la mantiene la comunidad y puede quedar desactualizada. En caso de discrepancia, prevalece [README.md](README.md).
